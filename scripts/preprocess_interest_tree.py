"""scripts/preprocess_interest_tree.py -- P6.4 offline interest-graph cache.
=============================================================================

Builds the item-item co-occurrence graph S^c (TAMER Eq. 3) from the training
interactions and writes a compact .npz cache consumed by
``run_p6_4_tamer.py``.  Because S^c depends only on the train split (not on
the modality features), we can pay the cost once per dataset and reuse the
cache across every P6.4 grid cell.

P6.4a speedups (Windows-native, RTX 5090):
    * ``--cooc_method`` selects the co-occurrence backend
      (``auto`` picks torch (CUDA) > sparse > roaring).  The scipy CSR
      ``M^T @ M`` path is 10-30x faster than roaring on Clothing;
      the torch CUDA path adds another 3-5x on top.
    * Vectorised ``_load_train_pairs`` -- numpy fromstring instead of
      per-line Python parsing.
    * ``--precompute_tree`` (default: on) also runs the BFS Interest Tree
      once during preprocessing and stores the flat traversal in the .npz.
      This moves the (formerly ~30-60 s) BFS + LIL random-access cost off
      the training loop -- every grid cell then only pays a single
      ``sp.multiply`` on the coefficient matrix (O(nnz)).
    * ``--tree_workers N`` fans the BFS across N Windows spawn processes.

Usage
-----
::

    # From MMHCL_DAMPS_Project/ — data lives at the repo-root sibling:
    python scripts/preprocess_interest_tree.py \\
        --dataset Clothing \\
        --data_dir ../data \\
        --core 5 \\
        --output   ./results/interest_tree_clothing.npz \\
        --knn_k_cooc 20 \\
        --knn_k_mod  10 \\
        --n_order    3 \\
        --gamma      1.0 \\
        --tau        1.0 \\
        --cooc_method auto \\
        --precompute_tree \\
        --tree_workers 4

Inputs
------
* Preferred (this repo / MMHCL):
  ``--data_dir/<dataset>/<core>-core/train.json`` — dict
  ``{uid_str: [item_id, ...]}`` as used by ``utility/load_data.py``.
* Fallback (Original-MMHCL):
  ``--data_dir/<dataset>/train.txt`` — one training user per line,
  whitespace-separated ``user_id item_id [item_id ...]``.

Outputs
-------
* .npz with fields ``{cooc_rows, cooc_cols, cooc_vals, knn_k_cooc, knn_k_mod,
  n_order, gamma, tau, n_items}`` plus (when ``--precompute_tree``) the flat
  BFS interest tree ``{tree_anchors, tree_neighbours, tree_orders,
  tree_weights}`` -- see ``codes/damps_tamer.py``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent / "src"
sys.path.insert(0, str(_ROOT))

from codes.damps_tamer import save_interest_cache  # noqa: E402
from codes.interest_tree import (  # noqa: E402
    build_weighted_binary_relations,
    precompute_interest_tree_flat,
    precompute_interest_tree_flat_parallel,
)
from codes.roaring_cooc import timed_cooc  # noqa: E402


# ---------------------------------------------------------------------------
# Train split loaders (MMHCL train.json  OR  Original-MMHCL train.txt)
# ---------------------------------------------------------------------------
def _resolve_train_file(
    data_dir: Path, dataset: str, core: int
) -> Path:
    """Locate the train split under this repo's data layout.

    Preference order:
      1. ``<data_dir>/<dataset>/<core>-core/train.json``  (MMHCL / this repo)
      2. ``<data_dir>/<dataset>/train.json``
      3. ``<data_dir>/<dataset>/train.txt``               (Original-MMHCL)
      4. ``<data_dir>/train.txt``                         (already inside dataset)
    """
    candidates = [
        data_dir / dataset / f"{core}-core" / "train.json",
        data_dir / dataset / "train.json",
        data_dir / dataset / "train.txt",
        data_dir / "train.txt",
        data_dir / f"{core}-core" / "train.json",
    ]
    for path in candidates:
        if path.is_file():
            return path
    tried = "\n  - ".join(str(c) for c in candidates)
    raise SystemExit(
        f"train split not found under data_dir={data_dir} "
        f"dataset={dataset} core={core}. Tried:\n  - {tried}"
    )


def _pairs_from_ragged(
    users: list[int], items: list[np.ndarray]
) -> tuple[np.ndarray, int]:
    """Stack ragged per-user item lists into an (E, 2) int64 array."""
    if not users:
        return np.empty((0, 2), dtype=np.int64), 0
    counts = np.fromiter(
        (t.size for t in items), dtype=np.int64, count=len(items)
    )
    us_arr = np.repeat(np.asarray(users, dtype=np.int64), counts)
    it_arr = np.concatenate(items).astype(np.int64, copy=False)
    pairs = np.stack([us_arr, it_arr], axis=1)  # (E, 2)
    n_items = int(it_arr.max()) + 1
    return pairs, n_items


def _load_train_pairs_json(path: Path) -> tuple[np.ndarray, int]:
    """Return (pairs, n_items) from MMHCL ``{uid: [item, ...]}`` JSON."""
    with path.open("r", encoding="utf-8") as fh:
        train = json.load(fh)
    if not isinstance(train, dict):
        raise SystemExit(
            f"Expected dict in {path}, got {type(train).__name__}"
        )
    users: list[int] = []
    items: list[np.ndarray] = []
    for uid_str, item_list in train.items():
        if not item_list:
            continue
        toks = np.asarray(item_list, dtype=np.int64)
        if toks.size == 0:
            continue
        users.append(int(uid_str))
        items.append(toks)
    return _pairs_from_ragged(users, items)


def _load_train_pairs_txt(path: Path) -> tuple[np.ndarray, int]:
    """Return (pairs, n_items) from Original-MMHCL whitespace train.txt.

    Vectorised: reads the whole file, ``str.split`` per line, then
    numpy-parses the ragged token lists into an (E, 2) int64 array.
    """
    with path.open("r", encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    users: list[int] = []
    items: list[np.ndarray] = []
    for ln in lines:
        parts = ln.split()
        if len(parts) < 2:
            continue
        u = int(parts[0])
        toks = np.fromstring(" ".join(parts[1:]), sep=" ", dtype=np.int64)
        if toks.size == 0:
            continue
        users.append(u)
        items.append(toks)
    return _pairs_from_ragged(users, items)


def _load_train_pairs(path: Path) -> tuple[np.ndarray, int]:
    """Dispatch to JSON or TXT loader based on file suffix."""
    if path.suffix.lower() == ".json":
        return _load_train_pairs_json(path)
    return _load_train_pairs_txt(path)


def _infer_n_items(
    data_dir: Path, dataset: str, n_items_from_train: int
) -> int:
    """Bump n_items from modality feature rows when available."""
    for name in ("image_feat.npy", "text_feat.npy"):
        feat = data_dir / dataset / name
        if feat.is_file():
            n_feat = int(np.load(feat, mmap_mode="r").shape[0])
            return max(n_items_from_train, n_feat)
    return n_items_from_train


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument(
        "--data_dir",
        default="./data",
        help="Parent of <dataset>/ (repo-root data/ for this project).",
    )
    p.add_argument(
        "--core",
        type=int,
        default=5,
        help="MMHCL core split folder (<dataset>/<core>-core/train.json). "
             "Ignored when a train.txt is found instead.",
    )
    p.add_argument("--output", required=True)
    p.add_argument("--knn_k_cooc", type=int, default=20)
    p.add_argument("--knn_k_mod", type=int, default=10)
    p.add_argument("--n_order", type=int, default=3)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--min_shared", type=int, default=2)
    # P6.4a knobs
    p.add_argument(
        "--cooc_method",
        default="auto",
        choices=["auto", "torch", "sparse", "roaring", "sets"],
        help="Co-occurrence backend. 'auto' picks torch (CUDA) > sparse > roaring.",
    )
    p.add_argument(
        "--block_size",
        type=int,
        default=1024,
        help="Row-block width for the sparse/torch co-occurrence paths.",
    )
    p.add_argument(
        "--device",
        default=None,
        help="torch device for the 'torch' cooc backend (default: cuda:0 if available).",
    )
    p.add_argument(
        "--precompute_tree",
        dest="precompute_tree",
        action="store_true",
        default=True,
        help="Run the BFS Interest Tree once and cache the flat traversal (default).",
    )
    p.add_argument(
        "--no_precompute_tree",
        dest="precompute_tree",
        action="store_false",
        help="Skip the flat BFS tree cache (legacy: BFS at train time).",
    )
    p.add_argument(
        "--tree_workers",
        type=int,
        default=0,
        help="ProcessPool workers for the BFS Interest Tree (0/1 = sequential).",
    )
    args = p.parse_args()

    # 1) Load train split (MMHCL train.json or Original-MMHCL train.txt).
    data_dir = Path(args.data_dir)
    # Common notebook misconfig: cwd=MMHCL_DAMPS_Project and
    # ``--data_dir data`` (local empty folder). Prefer the repo-root
    # sibling ``../data`` that actually holds Clothing/5-core/.
    if not (data_dir / args.dataset).is_dir():
        alt = (_ROOT.parent / "data").resolve()
        if (alt / args.dataset).is_dir():
            print(
                f"[P6.4-preprocess] data_dir={data_dir} has no "
                f"{args.dataset}/; falling back to {alt}"
            )
            data_dir = alt
    train_path = _resolve_train_file(data_dir, args.dataset, int(args.core))
    print(f"[P6.4-preprocess] loading {train_path} ...")
    t0 = time.perf_counter()
    pairs_arr, n_items_from_file = _load_train_pairs(train_path)
    n_items_from_file = _infer_n_items(
        data_dir, args.dataset, n_items_from_file
    )
    wall_load = time.perf_counter() - t0
    n_pairs = int(pairs_arr.shape[0])
    print(
        f"    n_pairs={n_pairs}  n_items={n_items_from_file}  "
        f"({wall_load:.2f}s)"
    )

    # timed_cooc expects a list of (u, i) tuples for the roaring path, but
    # the sparse/torch paths take any (E, 2)-indexable structure. Convert
    # once for compatibility.
    pairs_list = pairs_arr.tolist()

    # 2) Co-occurrence top-k.
    print(
        f"[P6.4-preprocess] building co-occurrence top-{args.knn_k_cooc} "
        f"(method={args.cooc_method}, block={args.block_size}) ..."
    )
    wall_cooc, (rows, cols, vals) = timed_cooc(
        pairs_list,
        args.knn_k_cooc,
        method=args.cooc_method,
        min_shared=args.min_shared,
        block_size=args.block_size,
        device=args.device,
    )
    n_items = max(n_items_from_file, int(rows.max()) + 1 if rows.size else 0,
                  int(cols.max()) + 1 if cols.size else 0)
    print(f"    nnz={rows.shape[0]}  ({wall_cooc:.2f}s)")

    # 3) Precompute the flat BFS Interest Tree (optional).
    tree_anchors = tree_neighbours = tree_orders = tree_weights = None
    if args.precompute_tree:
        print(
            f"[P6.4-preprocess] precomputing BFS interest tree "
            f"(n_order={args.n_order}, workers={args.tree_workers}) ..."
        )
        t_bfs = time.perf_counter()
        graph = build_weighted_binary_relations(rows, cols, vals, args.knn_k_cooc)
        if args.tree_workers and args.tree_workers > 1:
            (tree_anchors, tree_neighbours, tree_orders,
             tree_weights) = precompute_interest_tree_flat_parallel(
                graph, args.n_order, num_workers=args.tree_workers
            )
        else:
            (tree_anchors, tree_neighbours, tree_orders,
             tree_weights) = precompute_interest_tree_flat(graph, args.n_order)
        wall_bfs = time.perf_counter() - t_bfs
        print(
            f"    tree_nnz={tree_anchors.shape[0]}  ({wall_bfs:.2f}s)"
        )

    # 4) Persist.
    out_path = Path(args.output)
    save_interest_cache(
        out_path,
        cooc_rows=rows,
        cooc_cols=cols,
        cooc_vals=vals,
        knn_k_cooc=args.knn_k_cooc,
        knn_k_mod=args.knn_k_mod,
        n_order=args.n_order,
        gamma=args.gamma,
        tau=args.tau,
        n_items=n_items,
        tree_anchors=tree_anchors,
        tree_neighbours=tree_neighbours,
        tree_orders=tree_orders,
        tree_weights=tree_weights,
    )
    total = wall_load + wall_cooc + (wall_bfs if args.precompute_tree else 0.0)
    print(
        f"[P6.4-preprocess] wrote {out_path}  "
        f"(load={wall_load:.2f}s + cooc={wall_cooc:.2f}s"
        + (f" + bfs={wall_bfs:.2f}s" if args.precompute_tree else "")
        + f" = {total:.2f}s)"
    )


if __name__ == "__main__":
    main()

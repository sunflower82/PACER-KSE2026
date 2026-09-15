"""scripts/build_rsfp_interest_tree.py -- P6.5 RSFPGrowth-augmented cache.
=============================================================================

Builds RSFP-augmented ``interest_tree_clothing_rsfp_a{alpha}.npz`` variants by
blending the pre-existing P6.4 interest cache with 2-itemset edges mined by
PAMI ``RSFPGrowth`` from the training user-item transactions.

Zero-touch to model code: output is just another interest_tree .npz that
``run_p6_4_tamer.py`` consumes via ``--tamer_interest_cache``.

Pipeline
--------
1. Load train pairs (``data_dir/<dataset>/<core>-core/train.json`` or
   fallback ``train.txt``).
2. Emit tab-separated transactional DB (one line per user, tab-separated
   item ids) to ``<work_dir>/rsfp_txn_<dataset>.tsv``.
3. Run ``PAMI.relativeFrequentPattern.RSFPGrowth`` with the given
   ``--min_sup`` / ``--min_ratio``.
4. Filter mined patterns to 2-itemsets; build symmetric COO edges
   ``(i, j, support)`` -> ``M_rsfp``.  Normalise per-row to unit max, matching
   the P6.4 cache convention.
5. Blend with the P6.4 cache edges:
   ``M_blend = (1 - alpha) * M_orig + alpha * M_rsfp``.
6. Re-top-k per row at ``knn_k_cooc`` (from the base cache) and save.

Tree handling (rev57 patch — P8.2 RSFP-tree fix)
-----------------------------------------------
By default (``--rebuild_tree 0`` == legacy P6.5 behaviour) the tree_*
fields from the base cache are copied through UNCHANGED, so the interest-
tree BFS bonus in ``codes/damps_tamer.build_augmented_modality_graph``
coincides byte-for-byte with the P6.4 baseline. This gave a clean
(and cheap) A1 vs A0 ablation stub, but the KSE-final 5-seed benchmark
revealed that the resulting RSFP influence is limited to the direct
``s_c`` branch (``alpha_interest * s_c``) — the interest-tree bonus
(``coef_csr = f(tree_anchors, tree_neighbours, tree_orders, tree_weights)``
applied per modality view) stays identical between the RSFP cache and
the pure-cooc base cache. Effective RSFP share of the fused graph is
therefore only ~2%, and the A1 ablation reduces to a no-op.

With ``--rebuild_tree 1`` (recommended for P8.2+), after M_blend has
been row-normalised, symmetrised, and top-k pruned, we re-run
``codes.interest_tree.precompute_interest_tree_flat_parallel`` on the
blended graph, so the new ``tree_*`` fields inherit the RSFP-mined
2-itemset edges. The RSFP alpha then controls BOTH branches of Eq. 9
(direct ``s_c`` AND the per-view ``coef_csr`` bonus), matching the
KSE-final paper claim ("RSFPGrowth-augmented Interest-Tree Cache").

Usage
-----
::

    python scripts/build_rsfp_interest_tree.py \\
        --base_cache results/interest_tree_clothing.npz \\
        --dataset Clothing \\
        --data_dir ../data \\
        --core 5 \\
        --alphas 0.10 0.20 0.40 \\
        --min_sup 20 \\
        --min_ratio 0.4 \\
        --output_prefix results/interest_tree_clothing_rsfp \\
        --work_dir ./results/_rsfp_work

Outputs one .npz per alpha at ``<output_prefix>_a{alpha_pct}.npz``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent / "src"
sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Train-pair loader (mirror of preprocess_interest_tree.py)
# ---------------------------------------------------------------------------
def _resolve_train_path(data_dir: Path, dataset: str, core: int) -> Path:
    """Locate train.json/txt, with fallbacks for common notebook path bugs.

    Notebook cells sometimes pass ``REPO.parent / "data"`` instead of
    ``REPO / "data"``.  When the requested dir is empty we also probe the
    repo-root ``data/`` sibling of ``MMHCL_DAMPS_Project``.
    """
    data_dir = Path(data_dir)
    roots = [
        data_dir,
        _ROOT.parent / "data",  # <repo>/data
        Path.cwd().resolve().parent / "data",  # ../data from project cwd
        Path.cwd().resolve() / "data",
    ]

    searched: list[str] = []
    seen: set[str] = set()
    for root in roots:
        try:
            root_key = str(root.resolve())
        except OSError:
            root_key = str(root)
        if root_key in seen:
            continue
        seen.add(root_key)
        candidates = [
            root / dataset / f"{core}-core" / "train.json",
            root / dataset / "train.json",
            root / dataset / "train.txt",
            root / "train.txt",
        ]
        for path in candidates:
            searched.append(str(path))
            if path.is_file():
                try:
                    requested = data_dir.resolve()
                except OSError:
                    requested = data_dir
                if path.resolve().parents[2] != requested and path.resolve().parent != requested:
                    print(
                        f"[rsfp] data_dir={data_dir} missed; "
                        f"using fallback train file: {path}"
                    )
                return path
    raise FileNotFoundError(
        f"No train file found. Searched: {searched}"
    )


def _load_transactions(path: Path) -> tuple[list[list[int]], int]:
    """Return (list-of-item-lists per user, n_items)."""
    txns: list[list[int]] = []
    max_iid = -1
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as fh:
            train = json.load(fh)
        for _uid, items in train.items():
            if not items:
                continue
            row = [int(x) for x in items]
            if row:
                txns.append(row)
                max_iid = max(max_iid, max(row))
    else:
        with path.open("r", encoding="utf-8") as fh:
            for ln in fh:
                parts = ln.split()
                if len(parts) < 2:
                    continue
                row = [int(x) for x in parts[1:]]
                if row:
                    txns.append(row)
                    max_iid = max(max_iid, max(row))
    return txns, (max_iid + 1)


# ---------------------------------------------------------------------------
# RSFPGrowth mining via PAMI
# ---------------------------------------------------------------------------
def _write_pami_input(txns: list[list[int]], path: Path, sep: str = "\t") -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in txns:
            fh.write(sep.join(str(x) for x in row))
            fh.write("\n")


def _pami_absolute_min_sup(min_sup: float | int | str) -> int:
    """Convert CLI min_sup into an *absolute count* for PAMI RSFPGrowth.

    PAMI's ``RSFPGrowth.__convert`` treats ``float`` (and numeric strings
    containing ``'.'``) as a *fraction of |Database|*:

        float 20.0  ->  20.0 * N_txns   (e.g. 787740 on Clothing)
        int   20    ->  20             (absolute count, intended)
        str  "20"   ->  20
        str  "0.01" ->  0.01 * N_txns

    Our CLI documents ``--min_sup`` as an absolute item-count threshold, so
    we always coerce to ``int`` before constructing the miner.
    """
    if isinstance(min_sup, bool):
        raise ValueError(f"min_sup must be a positive count, got {min_sup!r}")
    if isinstance(min_sup, int):
        value = min_sup
    elif isinstance(min_sup, float):
        if not min_sup.is_integer():
            raise ValueError(
                f"--min_sup={min_sup!r} is a non-integer float. Pass an "
                "absolute integer count (e.g. 20). PAMI would otherwise "
                "treat floats as a fraction of |Database|."
            )
        value = int(min_sup)
    else:
        text = str(min_sup).strip()
        if not text:
            raise ValueError("--min_sup is empty")
        if "." in text:
            raise ValueError(
                f"--min_sup={min_sup!r} looks fractional. Pass an absolute "
                "integer count (e.g. 20)."
            )
        value = int(text)
    if value <= 0:
        raise ValueError(f"--min_sup must be > 0, got {value}")
    return value


def _mine_rsfp(
    input_tsv: Path,
    min_sup: float | int | str,
    min_ratio: float,
    sep: str = "\t",
) -> dict:
    """Run PAMI RSFPGrowth. Returns dict {pattern_str: support}."""
    from PAMI.relativeFrequentPattern.basic import RSFPGrowth as alg

    abs_min_sup = _pami_absolute_min_sup(min_sup)
    # Pass int (NOT float): see _pami_absolute_min_sup docstring.
    obj = alg.RSFPGrowth(str(input_tsv), abs_min_sup, float(min_ratio), sep=sep)
    obj.mine()
    patterns = obj.getPatterns()
    print(
        f"[rsfp] PAMI converted minSup={getattr(obj, '_minSup', abs_min_sup)} "
        f"(requested absolute count={abs_min_sup})"
    )
    return patterns


# ---------------------------------------------------------------------------
# Pattern -> edge triples
# ---------------------------------------------------------------------------
def _parse_pattern_tokens(pat: object) -> list[int]:
    """Normalise a PAMI pattern key into a list of item ids.

    ``getPatterns()`` joins items with tabs and often leaves a trailing tab
    (e.g. ``'123\\t456\\t'``).  ``__finalPatterns`` may also use tuples of
    string item ids.
    """
    if isinstance(pat, (list, tuple)):
        raw = [str(t).strip() for t in pat]
    else:
        raw = [t for t in str(pat).replace(",", " ").split() if t.strip()]
        # split() already collapses tabs/spaces; keep an explicit tab path
        # for odd encodings that survive as single tokens.
        if len(raw) == 1 and "\t" in raw[0]:
            raw = [t for t in raw[0].split("\t") if t.strip()]
    toks: list[int] = []
    for token in raw:
        token = token.strip()
        if not token:
            continue
        toks.append(int(token))
    return toks


def _parse_support(sup: object) -> float:
    """Parse PAMI support values such as ``'35 : 1.0'`` or bare numbers."""
    if isinstance(sup, (int, float)):
        return float(sup)
    text = str(sup).strip()
    if ":" in text:
        text = text.split(":", maxsplit=1)[0].strip()
    return float(text.split()[0])


def _patterns_to_2itemset_edges(
    patterns: dict, n_items: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Filter to 2-itemsets and return symmetric COO (rows, cols, vals).

    PAMI returns ``{pattern_str: support_str_or_int}``.  Pattern_str is
    typically tab/space-separated tokens; support is often
    ``'{count} : {ratio}'``.
    """
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    dropped = 0
    n_one = 0
    n_longer = 0
    for pat, sup in patterns.items():
        toks = _parse_pattern_tokens(pat)
        if len(toks) == 1:
            n_one += 1
            continue
        if len(toks) != 2:
            n_longer += 1
            continue
        i, j = toks
        if i == j or i < 0 or j < 0 or i >= n_items or j >= n_items:
            dropped += 1
            continue
        w = _parse_support(sup)
        rows.extend([i, j])
        cols.extend([j, i])
        vals.extend([w, w])
    print(
        f"[rsfp] pattern sizes: 1-item={n_one} 2-item={len(rows) // 2} "
        f">2-item={n_longer} dropped_oor={dropped}"
    )
    if not rows:
        return (
            np.zeros(0, np.int64),
            np.zeros(0, np.int64),
            np.zeros(0, np.float32),
        )
    return (
        np.asarray(rows, np.int64),
        np.asarray(cols, np.int64),
        np.asarray(vals, np.float32),
    )


# ---------------------------------------------------------------------------
# Blend + top-k
# ---------------------------------------------------------------------------
def _row_max_normalise(M: sp.csr_matrix) -> sp.csr_matrix:
    """Divide each row by its max non-zero entry (in-place friendly)."""
    if M.nnz == 0:
        return M
    row_max = np.asarray(M.max(axis=1).todense()).ravel()
    row_max[row_max == 0] = 1.0
    inv = 1.0 / row_max
    D = sp.diags(inv)
    return (D @ M).tocsr()


def _topk_per_row(M: sp.csr_matrix, k: int) -> sp.csr_matrix:
    """Keep top-k entries per row of a CSR matrix (by value)."""
    if k <= 0 or M.nnz == 0:
        return M
    M = M.tocsr()
    new_rows, new_cols, new_vals = [], [], []
    for r in range(M.shape[0]):
        s, e = M.indptr[r], M.indptr[r + 1]
        if e - s <= k:
            new_rows.extend([r] * (e - s))
            new_cols.extend(M.indices[s:e].tolist())
            new_vals.extend(M.data[s:e].tolist())
            continue
        idx = np.argpartition(-M.data[s:e], k)[:k]
        new_rows.extend([r] * k)
        new_cols.extend(M.indices[s:e][idx].tolist())
        new_vals.extend(M.data[s:e][idx].tolist())
    return sp.csr_matrix(
        (np.asarray(new_vals, np.float32),
         (np.asarray(new_rows, np.int64), np.asarray(new_cols, np.int64))),
        shape=M.shape,
    )


def _symmetrise(M: sp.csr_matrix) -> sp.csr_matrix:
    return ((M + M.T) * 0.5).tocsr()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_cache", type=Path, required=True,
                    help="Path to the P6.4 interest_tree_<dataset>.npz.")
    ap.add_argument("--dataset", type=str, default="Clothing")
    ap.add_argument("--data_dir", type=Path, default=Path("../data"))
    ap.add_argument("--core", type=int, default=5)
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.10, 0.20, 0.40])
    ap.add_argument(
        "--min_sup",
        type=int,
        default=20,
        help=(
            "PAMI RSFPGrowth minSup as an absolute transaction count. "
            "Must be int: PAMI treats floats as a fraction of |Database|."
        ),
    )
    ap.add_argument("--min_ratio", type=float, default=0.4,
                    help="PAMI RSFPGrowth minRatio (relative frequent).")
    ap.add_argument("--output_prefix", type=Path,
                    default=Path("results/interest_tree_clothing_rsfp"))
    ap.add_argument("--work_dir", type=Path,
                    default=Path("./results/_rsfp_work"))
    ap.add_argument("--skip_mining_if_cached", type=int, default=1,
                    help="If a non-empty patterns.pkl already exists, reuse it.")
    ap.add_argument(
        "--force_remine",
        type=int,
        default=0,
        help="Ignore cached patterns.pkl and re-run RSFPGrowth.",
    )
    # ---- P8.2 RSFP-tree fix ------------------------------------------------
    ap.add_argument(
        "--rebuild_tree",
        type=int,
        default=0,
        help=(
            "If 1, re-run codes.interest_tree.precompute_interest_tree_flat_* "
            "on the RSFP-blended co-occurrence graph so tree_* fields carry "
            "the RSFP signal. If 0 (legacy P6.5), copy tree_* from base_cache. "
            "When enabled, output filenames get an extra `_tree` infix to "
            "avoid overwriting P6.5 caches: `<prefix>_tree_a{alpha}.npz`."
        ),
    )
    ap.add_argument(
        "--tree_workers",
        type=int,
        default=0,
        help=(
            "Parallel workers for tree BFS when --rebuild_tree=1. 0 or 1 "
            "means single-thread. Matches preprocess_interest_tree.py."
        ),
    )
    args = ap.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)

    # 1) Load base cache.
    print(f"[rsfp] loading base cache: {args.base_cache}")
    base = dict(np.load(args.base_cache, allow_pickle=False))
    n_items = int(base["n_items"])
    knn_k = int(base["knn_k_cooc"])
    print(f"[rsfp] base: n_items={n_items} knn_k_cooc={knn_k} nnz={len(base['cooc_vals'])}")

    # 2) Build M_orig (row-max-normalised for scale-matching with M_rsfp).
    M_orig = sp.coo_matrix(
        (base["cooc_vals"].astype(np.float32),
         (base["cooc_rows"].astype(np.int64), base["cooc_cols"].astype(np.int64))),
        shape=(n_items, n_items),
    ).tocsr()
    M_orig = _row_max_normalise(M_orig)

    # 3) Load transactions + write PAMI input.
    train_path = _resolve_train_path(args.data_dir, args.dataset, args.core)
    print(f"[rsfp] loading transactions: {train_path}")
    txns, n_items_data = _load_transactions(train_path)
    if n_items_data > n_items:
        raise SystemExit(
            f"[rsfp] data has n_items={n_items_data} > cache n_items={n_items}"
        )
    print(f"[rsfp] loaded {len(txns)} transactions, n_items_data={n_items_data}")

    pami_input = args.work_dir / f"rsfp_txn_{args.dataset}.tsv"
    abs_min_sup = _pami_absolute_min_sup(args.min_sup)
    patterns_pkl = (
        args.work_dir
        / f"rsfp_patterns_{args.dataset}_ms{abs_min_sup}_mr{args.min_ratio}.pkl"
    )

    import pickle

    patterns: dict | None = None
    reuse_ok = (
        (not args.force_remine)
        and bool(args.skip_mining_if_cached)
        and patterns_pkl.exists()
    )
    if reuse_ok:
        print(f"[rsfp] loading cached patterns: {patterns_pkl}")
        with patterns_pkl.open("rb") as fh:
            patterns = pickle.load(fh)
        if not isinstance(patterns, dict) or not patterns:
            print(
                "[rsfp] cached patterns are empty/invalid "
                "(likely mined with float min_sup). Re-mining..."
            )
            patterns = None

    if patterns is None:
        _write_pami_input(txns, pami_input, sep="\t")
        print(f"[rsfp] wrote PAMI input: {pami_input}")
        t0 = time.time()
        print(
            f"[rsfp] mining RSFPGrowth minSup={abs_min_sup} (absolute) "
            f"minRatio={args.min_ratio} ..."
        )
        patterns = _mine_rsfp(
            pami_input, abs_min_sup, args.min_ratio, sep="\t"
        )
        print(f"[rsfp] mined {len(patterns)} patterns in {time.time() - t0:.1f}s")
        if not patterns:
            raise SystemExit(
                "[rsfp] RSFPGrowth returned 0 patterns. "
                "Lower --min_sup / --min_ratio, and ensure --min_sup is an "
                "integer absolute count (PAMI treats floats as |DB| fractions)."
            )
        with patterns_pkl.open("wb") as fh:
            pickle.dump(patterns, fh)
        print(f"[rsfp] cached patterns -> {patterns_pkl}")

    # 4) Build M_rsfp (2-itemset edges).
    rows, cols, vals = _patterns_to_2itemset_edges(patterns, n_items)
    print(
        f"[rsfp] 2-itemset edges: {len(rows) // 2} unique pairs "
        f"(symmetric COO nnz={len(rows)})"
    )
    if len(rows) == 0:
        raise SystemExit(
            "[rsfp] no 2-itemset patterns found. Lower --min_sup or "
            "--min_ratio (and delete empty patterns.pkl if present)."
        )
    M_rsfp = sp.coo_matrix(
        (vals, (rows, cols)), shape=(n_items, n_items)
    ).tocsr()
    M_rsfp = _row_max_normalise(M_rsfp)

    # 5) Blend + top-k per alpha; save.
    if bool(args.rebuild_tree):
        # Lazy import so the legacy path has no extra module load.
        from codes.interest_tree import (  # noqa: E402
            build_weighted_binary_relations,
            precompute_interest_tree_flat,
            precompute_interest_tree_flat_parallel,
        )

    for alpha in args.alphas:
        M_blend = ((1.0 - alpha) * M_orig + alpha * M_rsfp).tocsr()
        M_blend = _symmetrise(M_blend)
        M_blend = _topk_per_row(M_blend, knn_k)
        coo = M_blend.tocoo()
        alpha_tag = f"a{int(round(alpha * 100)):03d}"
        if bool(args.rebuild_tree):
            out_path = Path(f"{args.output_prefix}_tree_{alpha_tag}.npz")
        else:
            out_path = Path(f"{args.output_prefix}_{alpha_tag}.npz")

        kwargs = {
            "cooc_rows": coo.row.astype(np.int64),
            "cooc_cols": coo.col.astype(np.int64),
            "cooc_vals": coo.data.astype(np.float32),
            "knn_k_cooc": np.int32(knn_k),
            "knn_k_mod": base["knn_k_mod"],
            "n_order": base["n_order"],
            "gamma": base["gamma"],
            "tau": base["tau"],
            "n_items": np.int32(n_items),
        }

        if bool(args.rebuild_tree):
            # P8.2 fix: rebuild tree_* from the blended graph so RSFP
            # signal reaches Eq. 7 (interest-tree bonus).
            t_bfs = time.time()
            graph = build_weighted_binary_relations(
                coo.row.astype(np.int64),
                coo.col.astype(np.int64),
                coo.data.astype(np.float32),
                knn_k,
            )
            n_order = int(base["n_order"])
            if args.tree_workers and args.tree_workers > 1:
                (t_a, t_n, t_o, t_w) = precompute_interest_tree_flat_parallel(
                    graph, n_order, num_workers=int(args.tree_workers)
                )
            else:
                (t_a, t_n, t_o, t_w) = precompute_interest_tree_flat(
                    graph, n_order
                )
            wall_bfs = time.time() - t_bfs
            kwargs["tree_anchors"] = t_a
            kwargs["tree_neighbours"] = t_n
            kwargs["tree_orders"] = t_o
            kwargs["tree_weights"] = t_w
            print(
                f"[rsfp] rebuilt tree for alpha={alpha:.2f}: "
                f"tree_nnz={t_a.shape[0]} ({wall_bfs:.2f}s)"
            )
        else:
            # Legacy P6.5: copy tree_* fields verbatim from base cache.
            for k in ("tree_anchors", "tree_neighbours", "tree_orders", "tree_weights"):
                if k in base:
                    kwargs[k] = base[k]

        np.savez_compressed(out_path, **kwargs)
        print(f"[rsfp] wrote {out_path}  (blend alpha={alpha:.2f}, nnz={coo.nnz})")

    print("[rsfp] done.")


if __name__ == "__main__":
    main()

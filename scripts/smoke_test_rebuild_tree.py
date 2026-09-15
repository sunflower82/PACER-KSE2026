"""scripts/smoke_test_rebuild_tree.py -- P8.2 RSFP-tree fix smoke test.
======================================================================

Verifies the ``--rebuild_tree 1`` path of ``build_rsfp_interest_tree.py``.

Two valid cache kinds exist:

1. **Legacy P6.5** (``interest_tree_clothing_rsfp_a010.npz``):
   ``cooc_*`` is RSFP-blended, but ``tree_*`` was copied verbatim from the
   pure-cooc base cache.  Rebuilding the tree from ``cooc_*`` MUST differ
   from the stored ``tree_*``.

2. **P8.2 tree cache** (``interest_tree_clothing_rsfp_tree_a010.npz``):
   ``tree_*`` was rebuilt from the blended graph.  Rebuilding from
   ``cooc_*`` MUST match the stored ``tree_*``, and (when ``--base_cache``
   is given) MUST differ from the pure-cooc base tree.

Usage
-----
::

    # Legacy detection (expect DIFF):
    python scripts/smoke_test_rebuild_tree.py \\
        --rsfp_cache results/interest_tree_clothing_rsfp_a010.npz

    # P8.2 consistency (expect IDENTICAL to rebuild, DIFF vs base):
    python scripts/smoke_test_rebuild_tree.py \\
        --rsfp_cache results/interest_tree_clothing_rsfp_tree_a010.npz \\
        --base_cache results/interest_tree_clothing.npz

Exit code 0 = pass. Exit code 1 = fail.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent / "src"
sys.path.insert(0, str(_ROOT))

from codes.interest_tree import (  # noqa: E402
    build_weighted_binary_relations,
    precompute_interest_tree_flat,
)


def _fmt_diff(a: np.ndarray, b: np.ndarray, label: str) -> str:
    if a.shape != b.shape:
        return f"{label}: shape {a.shape} vs {b.shape} DIFF"
    if a.dtype.kind in "if":
        eq = bool(np.allclose(a, b, equal_nan=True))
    else:
        eq = bool(np.array_equal(a, b))
    if eq:
        return f"{label}: IDENTICAL (shape={a.shape})"
    if a.dtype.kind in "if":
        d = np.abs(a.astype(np.float64) - b.astype(np.float64))
        return (
            f"{label}: DIFF  shape={a.shape}  "
            f"max={float(d.max()):.4g}  mean={float(d.mean()):.4g}"
        )
    n_diff = int((a != b).sum())
    return f"{label}: DIFF  shape={a.shape}  n_diff={n_diff}"


def _any_diff(diffs: list[str]) -> bool:
    return any("DIFF" in d for d in diffs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--rsfp_cache",
        type=Path,
        required=True,
        help="RSFP cache (.npz), legacy or *_tree_* P8.2.",
    )
    ap.add_argument(
        "--base_cache",
        type=Path,
        default=None,
        help=(
            "Optional pure-cooc base cache. Required for a strong PASS on "
            "P8.2 *_tree_* caches (must DIFF vs base tree_*)."
        ),
    )
    args = ap.parse_args()

    print(f"[smoke] loading cache: {args.rsfp_cache}")
    cache = dict(np.load(args.rsfp_cache, allow_pickle=False))
    n_items = int(cache["n_items"])
    knn_k = int(cache["knn_k_cooc"])
    n_order = int(cache["n_order"])
    print(
        f"[smoke]   n_items={n_items} knn_k_cooc={knn_k} "
        f"n_order={n_order} cooc_nnz={cache['cooc_vals'].shape[0]}"
    )
    have_tree = all(
        k in cache
        for k in (
            "tree_anchors",
            "tree_neighbours",
            "tree_orders",
            "tree_weights",
        )
    )
    if not have_tree:
        print("[smoke] FAIL: cache lacks tree_* fields; cannot compare.")
        return 1
    print(
        f"[smoke]   stored tree_anchors nnz={cache['tree_anchors'].shape[0]}"
    )

    print("[smoke] rebuilding graph from cache cooc_* ...")
    graph = build_weighted_binary_relations(
        cache["cooc_rows"].astype(np.int64),
        cache["cooc_cols"].astype(np.int64),
        cache["cooc_vals"].astype(np.float32),
        knn_k,
    )
    print(f"[smoke]   graph anchors={len(graph)}")

    print(f"[smoke] running BFS interest tree (n_order={n_order}) ...")
    t0 = time.time()
    (new_a, new_n, new_o, new_w) = precompute_interest_tree_flat(
        graph, n_order
    )
    wall = time.time() - t0
    print(f"[smoke]   rebuilt tree nnz={new_a.shape[0]}  ({wall:.2f}s)")

    print("\n=== tree_* diff  (stored vs rebuild-from-cooc) ===")
    diffs = [
        _fmt_diff(cache["tree_anchors"], new_a, "tree_anchors"),
        _fmt_diff(cache["tree_neighbours"], new_n, "tree_neighbours"),
        _fmt_diff(cache["tree_orders"], new_o, "tree_orders"),
        _fmt_diff(cache["tree_weights"], new_w, "tree_weights"),
    ]
    for line in diffs:
        print(" ", line)

    looks_like_tree_cache = "rsfp_tree_" in args.rsfp_cache.name
    rebuilt_matches_stored = not _any_diff(diffs)

    if looks_like_tree_cache or rebuilt_matches_stored:
        # P8.2 path: stored tree must match rebuild-from-cooc.
        if not rebuilt_matches_stored:
            print(
                "\n[smoke] FAIL: *_tree_* cache is inconsistent with its "
                "own cooc_* (rebuild diverged)."
            )
            return 1
        print(
            "\n[smoke] stored tree_* matches rebuild-from-cooc "
            "(P8.2-consistent)."
        )
        if args.base_cache is None:
            default_base = (
                args.rsfp_cache.parent / "interest_tree_clothing.npz"
            )
            if default_base.is_file():
                args.base_cache = default_base
        if args.base_cache is not None and args.base_cache.is_file():
            base = dict(np.load(args.base_cache, allow_pickle=False))
            print(
                f"\n=== tree_* diff  (P8.2 cache vs base {args.base_cache.name}) ==="
            )
            base_diffs = [
                _fmt_diff(
                    cache["tree_anchors"], base["tree_anchors"], "tree_anchors"
                ),
                _fmt_diff(
                    cache["tree_neighbours"],
                    base["tree_neighbours"],
                    "tree_neighbours",
                ),
                _fmt_diff(
                    cache["tree_orders"], base["tree_orders"], "tree_orders"
                ),
                _fmt_diff(
                    cache["tree_weights"],
                    base["tree_weights"],
                    "tree_weights",
                ),
            ]
            for line in base_diffs:
                print(" ", line)
            if not _any_diff(base_diffs):
                print(
                    "\n[smoke] FAIL: P8.2 tree_* is identical to the pure-"
                    "cooc base tree — rebuild did not carry RSFP signal."
                )
                return 1
            print(
                "\n[smoke] PASS: P8.2 tree_* is consistent with cooc_* and "
                "differs from the pure-cooc base tree."
            )
            return 0
        print(
            "\n[smoke] PASS (weak): P8.2 tree_* is consistent with cooc_*. "
            "Pass --base_cache for a stronger check vs pure-cooc tree."
        )
        return 0

    # Legacy P6.5 path: stored tree was copied from base → must DIFF.
    if not _any_diff(diffs):
        print(
            "\n[smoke] FAIL: rebuild produced identical tree_* on a legacy "
            "P6.5 cache. Expected DIFF (copied base tree vs blended cooc)."
        )
        return 1
    print(
        "\n[smoke] PASS: rebuild produces tree_* that differ from the "
        "legacy base-cache copy. The --rebuild_tree=1 fix is effective."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

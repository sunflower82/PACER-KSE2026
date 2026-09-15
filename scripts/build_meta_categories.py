"""
build_meta_categories.py
========================

Generate a high-quality ``meta_categories.npy`` for Amazon-Clothing
(or any dataset shipped via the MMHCL release).

Why this script exists
----------------------

The HuggingFace release ``Xu-SII-BNU/MMHCL`` ships ONLY:

    Clothing/
        image_feat.npy        (755 MB, shape (n_items, d_img))
        text_feat.npy         ( 94 MB, shape (n_items, d_txt))
        5-core/{train,val,test}.json

There is **no** ``meta_categories.npy``.  ``MMHCL_DAMPS_Project/utility/
load_data.py`` is explicit (lines 215--239): the file is *optional* and, when
absent, the loader silently falls back to ``(ids * 2654435761) % n_cats``
-- a deterministic Knuth-hash that has **no semantic content**.  The Day-1
diagnostic sprint just confirmed this fallback is **actively harmful**
(combined+APC=1 hash R@20=0.0851 vs combined+APC=0 R@20=0.0920, +8.1% rel).

To use APC as the paper INTENDED, ``meta_categories.npy`` must be a
semantically meaningful clustering of items.  The most principled and
reproducible way -- given that the paper does not ship a class file -- is to
run **K-Means on the concatenated, L2-normalised image+text embeddings**.

This is also what the APC module does internally for its prototype init, so
using K-Means clusters as the supervised metadata target is consistent with
the architecture's inductive bias.

Usage
-----

From any directory (paths are resolved against ``--data-path``)::

    python build_meta_categories.py \\
        --data-path "C:\\Users\\Anh Khoi\\Desktop\\MyCode\\Saved research codes\\5090_Professional_Original_paper_MMHCL_randoms\\data" \\
        --dataset Clothing \\
        --num-categories 10 \\
        --modality both \\
        --output "C:\\Users\\Anh Khoi\\Desktop\\MyCode\\My research progress\\DAMPS_upgrade_for_MMHCL_randoms_Amazon_Clothing\\data\\Clothing\\meta_categories.npy"

The script will:
    1. Load image_feat.npy and text_feat.npy from ``data/<dataset>/``
    2. Verify ``len == n_items`` against the 5-core JSON splits
    3. L2-normalise and concatenate (or use either modality alone)
    4. Run K-Means (sklearn MiniBatchKMeans -- 23 k items is small)
    5. Save the (n_items,) int64 array to ``--output``
    6. Print a histogram, silhouette score, and a sanity check

Reproducibility: K-Means seed defaults to 42 and is logged.

Dependencies: numpy, scikit-learn, scipy (already required by the project).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate meta_categories.npy from MMHCL modality features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-path", type=str, required=True,
                   help="Root data directory (contains <dataset>/image_feat.npy).")
    p.add_argument("--dataset", type=str, default="Clothing",
                   help="Subdirectory name under --data-path.")
    p.add_argument("--core", type=int, default=5,
                   help="Core filtering used to infer n_items.")
    p.add_argument("--num-categories", type=int, default=10,
                   help="Number of K-Means clusters (matches --damps_num_categories).")
    p.add_argument("--modality", choices=("image", "text", "both"), default="both",
                   help="Which modality features to cluster.")
    p.add_argument("--output", type=str, default=None,
                   help="Output .npy path. Defaults to <data-path>/<dataset>/meta_categories.npy.")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for K-Means.")
    p.add_argument("--max-iter", type=int, default=500,
                   help="K-Means max iterations.")
    p.add_argument("--batch-size", type=int, default=4096,
                   help="MiniBatchKMeans batch size (irrelevant if --use-full-kmeans).")
    p.add_argument("--use-full-kmeans", action="store_true",
                   help="Use sklearn.cluster.KMeans (n_init=10) instead of MiniBatchKMeans.")
    p.add_argument("--skip-silhouette", action="store_true",
                   help="Skip silhouette score (can be slow for n>20k).")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite the output file if it already exists.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _infer_n_items(data_path: str, dataset: str, core: int) -> tuple[int, int]:
    """Mirror Data.__init__: max(item_id) + 1 across train+val+test."""
    base = os.path.join(data_path, dataset, f"{core}-core")
    n_items, n_users = 0, 0
    for fname in ("train.json", "test.json", "val.json"):
        fp = os.path.join(base, fname)
        if not os.path.isfile(fp):
            continue
        with open(fp, "r", encoding="utf-8") as f:
            blob: dict[str, list[int]] = json.load(f)
        for uid_str, items in blob.items():
            if not items:
                continue
            n_users = max(n_users, int(uid_str))
            n_items = max(n_items, max(items))
    return n_items + 1, n_users + 1


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, eps, None)


def _load_modality(data_path: str, dataset: str, name: str) -> Optional[np.ndarray]:
    fp = os.path.join(data_path, dataset, f"{name}_feat.npy")
    if not os.path.isfile(fp):
        return None
    print(f"  loading {name:5s} from {fp}  ({os.path.getsize(fp) / 1e6:.1f} MB)")
    arr = np.load(fp).astype(np.float32, copy=False)
    return arr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    args = _parse_args()

    print("=" * 72)
    print("build_meta_categories.py -- semantic clusters for MMHCL APC")
    print("=" * 72)
    print(f"  dataset         : {args.dataset}")
    print(f"  data_path       : {args.data_path}")
    print(f"  num_categories  : {args.num_categories}")
    print(f"  modality        : {args.modality}")
    print(f"  seed            : {args.seed}")
    print()

    # -------------------------------------------------------------------
    # 1) Infer n_items from the JSON splits (same logic as Data.__init__)
    # -------------------------------------------------------------------
    print("[1/5] inferring n_items from splits ...")
    n_items, n_users = _infer_n_items(args.data_path, args.dataset, args.core)
    print(f"      n_items={n_items:,}  n_users={n_users:,}")

    # -------------------------------------------------------------------
    # 2) Load modality features
    # -------------------------------------------------------------------
    print("\n[2/5] loading modality features ...")
    img = txt = None
    if args.modality in ("image", "both"):
        img = _load_modality(args.data_path, args.dataset, "image")
    if args.modality in ("text", "both"):
        txt = _load_modality(args.data_path, args.dataset, "text")

    if img is None and txt is None:
        print(f"[FATAL] no modality features found under "
              f"{os.path.join(args.data_path, args.dataset)}", file=sys.stderr)
        return 2

    for name, arr in (("image", img), ("text", txt)):
        if arr is None:
            continue
        if arr.shape[0] != n_items:
            print(f"[FATAL] {name}_feat.npy has shape {arr.shape} but "
                  f"n_items={n_items}.  Aborting.", file=sys.stderr)
            return 3
        print(f"      {name:5s}: shape={arr.shape}  dtype={arr.dtype}  "
              f"contains_nan={bool(np.isnan(arr).any())}")

    # -------------------------------------------------------------------
    # 3) L2-normalise + concat (cosine-friendly K-Means)
    # -------------------------------------------------------------------
    print("\n[3/5] L2-normalising features ...")
    parts: list[np.ndarray] = []
    if img is not None:
        parts.append(_l2_normalize(img))
    if txt is not None:
        parts.append(_l2_normalize(txt))

    X = np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]
    # Re-normalise the concatenation so neither modality dominates.
    if len(parts) > 1:
        X = _l2_normalize(X)
    print(f"      final feature matrix: shape={X.shape}  "
          f"size={X.nbytes / 1e6:.1f} MB")

    # -------------------------------------------------------------------
    # 4) K-Means
    # -------------------------------------------------------------------
    print(f"\n[4/5] K-Means (k={args.num_categories}) ...")
    t0 = time.time()
    try:
        if args.use_full_kmeans:
            from sklearn.cluster import KMeans
            km = KMeans(
                n_clusters=args.num_categories,
                n_init=10,
                max_iter=args.max_iter,
                random_state=args.seed,
                verbose=0,
            )
        else:
            from sklearn.cluster import MiniBatchKMeans
            km = MiniBatchKMeans(
                n_clusters=args.num_categories,
                batch_size=args.batch_size,
                max_iter=args.max_iter,
                random_state=args.seed,
                n_init=10,
                verbose=0,
            )
    except ImportError as e:
        print(f"[FATAL] scikit-learn required: {e}", file=sys.stderr)
        print("        pip install scikit-learn", file=sys.stderr)
        return 4

    labels = km.fit_predict(X).astype(np.int64)
    print(f"      fitted in {time.time() - t0:.1f}s   inertia={km.inertia_:.4f}")

    # -------------------------------------------------------------------
    # 5) Save + sanity report
    # -------------------------------------------------------------------
    out = args.output or os.path.join(
        args.data_path, args.dataset, "meta_categories.npy"
    )
    print(f"\n[5/5] writing output to {out}")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.isfile(out) and not args.overwrite:
        print(f"[FATAL] {out} already exists.  Pass --overwrite to replace.",
              file=sys.stderr)
        return 5
    np.save(out, labels)

    # Sanity report
    hist = Counter(labels.tolist())
    sorted_hist = sorted(hist.items())
    print()
    print("=" * 72)
    print("RESULT  --  meta_categories.npy")
    print("=" * 72)
    print(f"  path      : {out}")
    print(f"  shape     : {labels.shape}")
    print(f"  dtype     : {labels.dtype}")
    print(f"  n_unique  : {len(hist)}  (target {args.num_categories})")
    print(f"  histogram :")
    total = labels.shape[0]
    for k, v in sorted_hist:
        bar = "#" * int(40 * v / max(hist.values()))
        print(f"     cluster {k:2d}: {v:6d}  ({100 * v / total:5.1f}%)  {bar}")
    minc = min(hist.values())
    maxc = max(hist.values())
    print(f"  cluster size range : {minc} ... {maxc}  "
          f"(imbalance = {maxc / minc:.2f}x)")

    # Optional silhouette (slow but informative on first run)
    if not args.skip_silhouette and X.shape[0] <= 30_000:
        try:
            from sklearn.metrics import silhouette_score
            print("\n  computing silhouette score (this may take ~1 min) ...")
            sub = np.random.default_rng(args.seed).choice(
                X.shape[0], size=min(8_000, X.shape[0]), replace=False
            )
            sil = silhouette_score(X[sub], labels[sub], metric="cosine")
            print(f"  silhouette (cosine, n=8000): {sil:.4f}   "
                  "(>0.05 = some structure; >0.15 = strong)")
        except Exception as e:
            print(f"  silhouette skipped: {e}")

    # H2 verdict line that mirrors what day1_diagnostic_sprint.py reports
    print()
    if len(hist) >= 5 and labels.shape[0] == n_items:
        print("  [OK] file passes the day1_diagnostic_sprint.py D1 audit:")
        print("       (exists + shape == n_items + unique >= 5)")
    else:
        print("  [WARN] file may still trip the D1 audit  "
              f"(unique={len(hist)}, len={labels.shape[0]}, n_items={n_items}).")

    print("\nDone.  Next: re-run day1_diagnostic_sprint.py to confirm D1 PASS, "
          "then re-execute the Phase-1 sweep with the real metadata.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""codes/knn_hnsw.py -- P6.6 speedup #1: HNSW-backed top-k for modality S^m.
==========================================================================

Drop-in replacement for the exact cosine k-NN used to build the modality
adjacency A^m in MACP/TAMER-style pipelines.  For n_items in the 30-100k
range the exact path (torch.mm on the full n x n similarity, top-k) is
O(n^2) memory and O(n^2 * d) FLOPs; HNSW brings it down to O(n * log n)
with recall@10 typically > 0.98 at M=16, ef_construction=200.

Wire-up
-------
* ``build_hnsw_index(feats, ...)`` returns an ``hnswlib.Index`` over
  L2-normalised rows (cosine similarity ~= inner product).
* ``knn_topk_hnsw(index, feats, k)`` returns ``(rows, cols, vals)``
  suitable for ``scipy.sparse.csr_matrix`` construction.  Self-loops
  are removed.
* When ``hnswlib`` is unavailable we fall back to the exact torch path
  and print a one-line warning so pipelines never break.

Notes
-----
* Cosine similarity is computed as inner product on L2-normalised rows,
  matching TAMER (Meng et al. MM'25, Eq. 2) and the DAMPS-MACP pipeline.
* Determinism: HNSW is non-deterministic across ``num_threads`` values;
  the driver pins ``num_threads=1`` when reproducibility is needed.
"""
from __future__ import annotations

import time
import warnings
from typing import Optional, Tuple

import numpy as np
import torch

try:  # optional dep -- required for the fast path
    import hnswlib  # type: ignore

    _HAVE_HNSW = True
except ImportError:  # pragma: no cover
    hnswlib = None
    _HAVE_HNSW = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _l2_normalise(feats: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (feats / norms).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# HNSW build + query
# ---------------------------------------------------------------------------
def build_hnsw_index(
    feats: np.ndarray,
    *,
    M: int = 16,
    ef_construction: int = 200,
    num_threads: int = 4,
    seed: int = 100,
):
    """Return an ``hnswlib.Index`` over L2-normalised ``feats``.

    Args:
        feats           : (n_items, d) float32 array of modality embeddings.
        M               : HNSW graph out-degree (16 is the recall/RAM sweet spot).
        ef_construction : Beam width during build (200 gives recall@10 > 0.98).
        num_threads     : # OpenMP threads for build.  1 = deterministic.
        seed            : RNG seed for HNSW.

    Raises:
        RuntimeError if hnswlib is not installed.
    """
    if not _HAVE_HNSW:
        raise RuntimeError(
            "hnswlib not installed -- `pip install hnswlib>=0.7.0` or use "
            "knn_topk_exact() for the fallback path."
        )
    feats = _l2_normalise(np.ascontiguousarray(feats, dtype=np.float32))
    n, dim = feats.shape
    idx = hnswlib.Index(space="ip", dim=dim)
    idx.init_index(max_elements=n, ef_construction=ef_construction, M=M, random_seed=seed)
    idx.set_num_threads(num_threads)
    idx.add_items(feats, np.arange(n, dtype=np.int64))
    return idx


def knn_topk_hnsw(
    index,
    feats: np.ndarray,
    k: int,
    *,
    ef_search: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Query ``index`` for each row in ``feats`` and return COO triplets.

    Self-loops are removed (top-``k+1`` queried, first hit dropped when it
    matches the row index).

    Returns:
        rows : (nnz,) int64
        cols : (nnz,) int64
        vals : (nnz,) float32 -- cosine similarities in [-1, 1].
    """
    if ef_search is not None:
        index.set_ef(ef_search)
    feats = _l2_normalise(np.ascontiguousarray(feats, dtype=np.float32))
    labels, distances = index.knn_query(feats, k=k + 1)
    # hnswlib returns 1 - inner_product for 'ip' space -> convert back to sim
    sims = 1.0 - distances
    n = feats.shape[0]
    rows_out, cols_out, vals_out = [], [], []
    for i in range(n):
        picked = 0
        for j in range(k + 1):
            neigh = int(labels[i, j])
            if neigh == i:  # drop self-loop
                continue
            rows_out.append(i)
            cols_out.append(neigh)
            vals_out.append(float(sims[i, j]))
            picked += 1
            if picked == k:
                break
    return (
        np.asarray(rows_out, dtype=np.int64),
        np.asarray(cols_out, dtype=np.int64),
        np.asarray(vals_out, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Exact fallback (torch) -- kept for benchmarking and unit tests
# ---------------------------------------------------------------------------
def knn_topk_exact(
    feats: np.ndarray,
    k: int,
    *,
    device: str = "cpu",
    chunk: int = 4096,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact cosine top-k via chunked torch.mm.  Reference implementation."""
    feats_n = _l2_normalise(np.ascontiguousarray(feats, dtype=np.float32))
    x = torch.from_numpy(feats_n).to(device)
    n = x.shape[0]
    rows_all, cols_all, vals_all = [], [], []
    for start in range(0, n, chunk):
        end = min(n, start + chunk)
        sim = x[start:end] @ x.T  # (chunk, n)
        # mask self-loops
        for i, row in enumerate(range(start, end)):
            sim[i, row] = -float("inf")
        top_vals, top_idx = torch.topk(sim, k=k, dim=1)
        top_vals = top_vals.detach().cpu().numpy()
        top_idx = top_idx.detach().cpu().numpy()
        for i, row in enumerate(range(start, end)):
            for jj in range(k):
                rows_all.append(row)
                cols_all.append(int(top_idx[i, jj]))
                vals_all.append(float(top_vals[i, jj]))
    return (
        np.asarray(rows_all, dtype=np.int64),
        np.asarray(cols_all, dtype=np.int64),
        np.asarray(vals_all, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Convenience: benchmark a single (n, d, k) configuration
# ---------------------------------------------------------------------------
def timed_knn(
    feats: np.ndarray,
    k: int,
    *,
    method: str = "hnsw",
    **kwargs,
) -> Tuple[float, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return ``(wall_seconds, (rows, cols, vals))`` for the chosen method."""
    t0 = time.perf_counter()
    if method == "hnsw":
        if not _HAVE_HNSW:
            warnings.warn("hnswlib missing -- falling back to exact path.")
            out = knn_topk_exact(feats, k, **{k_: v for k_, v in kwargs.items() if k_ in {"device", "chunk"}})
        else:
            idx = build_hnsw_index(
                feats,
                M=kwargs.get("M", 16),
                ef_construction=kwargs.get("ef_construction", 200),
                num_threads=kwargs.get("num_threads", 4),
                seed=kwargs.get("seed", 100),
            )
            out = knn_topk_hnsw(idx, feats, k, ef_search=kwargs.get("ef_search", 64))
    elif method == "exact":
        out = knn_topk_exact(feats, k, **{k_: v for k_, v in kwargs.items() if k_ in {"device", "chunk"}})
    else:
        raise ValueError(f"unknown method={method!r} (expected 'hnsw'|'exact')")
    return time.perf_counter() - t0, out


__all__ = [
    "build_hnsw_index",
    "knn_topk_hnsw",
    "knn_topk_exact",
    "timed_knn",
]

"""
utility/metrics.py — Recommendation Evaluation Metrics
========================================================

Implements top-K evaluation metrics used in recommendation system research:

    Precision@K  — fraction of recommended items that are relevant
    Recall@K     — fraction of relevant items that are recommended
    NDCG@K       — normalised discounted cumulative gain
    Hit@K        — whether at least one relevant item appears in top-K
    AUC          — area under the ROC curve

All functions expect a binary relevance vector ``r`` where
``r[i] = 1`` if the item at rank i is relevant (ground truth), else 0.
"""

from __future__ import annotations

from typing import Sequence, Union

import numpy as np
import numpy.typing as npt

try:
    from sklearn.metrics import roc_auc_score                     # type: ignore
    _SKLEARN_OK: bool = True
except ImportError:                                                # pragma: no cover
    _SKLEARN_OK = False


def precision_at_k(r: Sequence[int], k: int) -> float:
    """Precision@K — fraction of top-K items that are relevant."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    r_arr = np.asarray(r)[:k]
    return float(np.mean(r_arr)) if r_arr.size else 0.0


def recall_at_k(r: Sequence[Union[int, float]], k: int, all_pos_num: int) -> float:
    """Recall@K — fraction of all positive items found in the top-K ranking."""
    if all_pos_num == 0:
        return 0.0
    r_arr = np.asarray(r, dtype=np.float32)[:k]
    return float(np.sum(r_arr) / float(all_pos_num))


def dcg_at_k(r: Sequence[Union[int, float]], k: int, method: int = 1) -> float:
    """Discounted Cumulative Gain (DCG@K)."""
    r_arr = np.asarray(r, dtype=np.float32)[:k]
    if r_arr.size == 0:
        return 0.0
    if method == 0:
        return float(r_arr[0] + np.sum(r_arr[1:] / np.log2(np.arange(2, r_arr.size + 1))))
    if method == 1:
        return float(np.sum(r_arr / np.log2(np.arange(2, r_arr.size + 2))))
    raise ValueError("method must be 0 or 1")


def ndcg_at_k(r: Sequence[Union[int, float]], k: int, method: int = 1) -> float:
    """NDCG@K — normalised DCG."""
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.0
    return dcg_at_k(r, k, method) / dcg_max


def hit_at_k(r: Sequence[Union[int, float]], k: int) -> float:
    """Hit@K — 1 if at least one relevant item appears in top-K, else 0."""
    return 1.0 if np.sum(np.asarray(r)[:k]) > 0 else 0.0


def auc(ground_truth: list[int], prediction: list[float]) -> float:
    """AUC (Area Under ROC Curve) — uses sklearn; returns 0 on failure."""
    if not _SKLEARN_OK:
        return 0.0
    try:
        return float(roc_auc_score(y_true=ground_truth, y_score=prediction))
    except Exception:
        return 0.0


def gini_coefficient(counts: npt.NDArray[np.floating]) -> float:
    """
    Compute the Gini coefficient of an array of recommendation counts.

    Used by the Coverage@20 / Gini@20 fairness diagnostics from spec
    Section 4 ("Fairness & Robustness Metrics").

    Args:
        counts : (n_items,) — how often each item is recommended in top-K.

    Returns:
        float in [0, 1]: 0 = perfect equality, 1 = perfect inequality.
    """
    arr = np.asarray(counts, dtype=np.float64)
    if arr.size == 0 or arr.sum() == 0:
        return 0.0
    arr = np.sort(arr)
    n = arr.size
    cumvals = np.cumsum(arr)
    # Standard Gini formula
    return float((n + 1 - 2 * np.sum(cumvals) / cumvals[-1]) / n)


def coverage_at_k(rec_lists: list[list[int]], n_items: int) -> float:
    """
    Coverage@K — fraction of catalogue items recommended at least once.

    Args:
        rec_lists : list of length-K item-id lists, one per evaluated user.
        n_items   : total catalogue size.

    Returns:
        float in [0, 1].
    """
    if n_items <= 0:
        return 0.0
    seen: set[int] = set()
    for lst in rec_lists:
        seen.update(lst)
    return float(len(seen) / n_items)

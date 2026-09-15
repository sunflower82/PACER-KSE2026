"""
damps/popularity_prior.py — Long-tail Popularity Prior for LogQ Correction
==========================================================================

Implements the **q(i) estimator** used by the LogQ correction (variant "h"
of T5, rev53 §3.1, eq. (1)). This module is the **1B prerequisite** of
the M1.5 milestone — it computes and caches the per-item log-probability
vector ``log_q ∈ R^{n_items}`` that is later subtracted from contrastive
logits inside ``batched_contrastive_loss``.

Why a separate module?
----------------------
The rev53 review board (§3.1, line 104) flagged that "Special attention
must be paid to the estimation method of q(i) (raw frequency vs. Laplace
smoothing) to avoid falsely penalizing genuinely high-quality popular
items." Mixing the prior estimator with the InfoNCE patch in a single
commit makes it impossible to attribute a Recall@20 regression to either
(a) a wrong smoothing coefficient β or (b) a wrong sign / scaling inside
the loss. Keeping this module loss-free makes the 1B PR auditable in
isolation by unit tests, and lets the 1A PR (model.py patch) treat the
log_q tensor as a frozen input.

Mathematical formulation
------------------------
Given the binary train interaction matrix ``R ∈ {0,1}^{n_users × n_items}``,
let ``n_i = Σ_u R[u, i]`` be the raw interaction count for item i. The
Laplace-smoothed marginal is

    q(i) = (n_i + β) / (Σ_j n_j + n_items · β)              (1)

The vector ``log_q(i) = log q(i)`` is then subtracted from per-column
logits inside InfoNCE:

    L_NCEQ = -log[ exp((sim(u,i⁺) - log_q(i⁺))/τ) /
                   Σ_j exp((sim(u,i⁻ⱼ) - log_q(i⁻ⱼ))/τ) ]   (2)

For an item with zero interactions, q(i) = β / (N_train + n_items·β),
which is **strictly positive** thanks to Laplace smoothing — preventing
``log 0 = -inf`` blow-ups. β = 1.0 is the canonical "add-one" choice.

Three estimator modes are supported (selected by ``mode``):
    "laplace" — eq. (1) above; safe default.
    "raw"     — q(i) ∝ n_i (no smoothing); only valid if every item has
                ≥ 1 interaction. Useful for ablation studies.
    "sqrt"    — q(i) ∝ sqrt(n_i + β); the less-aggressive correction
                proposed in DGRec (Wang et al., WWW 2024) for
                cosine-normalised embeddings.

Public API
----------
``compute_item_counts(train_items, n_items) -> Tensor[Long]``
    (n_items,) raw interaction count from the train.json dict.

``compute_log_q(counts, beta=1.0, mode="laplace") -> Tensor[Float]``
    (n_items,) log q(i) vector. Validated for finiteness.

``load_or_build_log_q(cache_dir, n_items, train_items, beta=1.0,
                      mode="laplace", force_rebuild=False) -> Tensor[Float]``
    Cached wrapper. The cache key includes (mode, β, n_items, sum(counts))
    so changing any of them invalidates the cache.

Code-line evidence
------------------
* Interaction matrix construction:
  ``utility/load_data.py:153-160``  (``self.R = sp.dok_matrix(...)``)
* InfoNCE call sites that will consume ``log_q``:
  ``train.py:575-581``  (``bcl_item = self.model.batched_contrastive_loss(...)``)
* Existing cache pattern this module mirrors:
  ``utility/load_data.py:314``  (``cache = ...f"UI_mat_{norm_type}.pth"``)

References
----------
* Yi, X. et al. "Sampling-Bias-Corrected Neural Modeling for Large
  Corpus Item Recommendations." RecSys 2019.
* Wang, J. et al. "Distributionally Robust Graph-based Recommendation
  System." WWW 2024 (DGRec).
"""

from __future__ import annotations

import hashlib
import os
import warnings
from typing import Dict, List, Optional

import torch


# ---------------------------------------------------------------------------
#  1. Raw popularity counts
# ---------------------------------------------------------------------------
def compute_item_counts(
    train_items: Dict[int, List[int]],
    n_items: int,
) -> torch.Tensor:
    """
    Count raw interactions per item from the train split dict.

    Args:
        train_items : ``{user_id: [item_id, ...]}`` as built in
                      ``utility/load_data.py:136-141``.
        n_items     : total number of distinct items in the catalog
                      (``self.n_items`` after the +1 adjustment at
                      ``utility/load_data.py:142``).

    Returns:
        (n_items,) Long tensor where ``counts[i] = |{u : i ∈ train_items[u]}|``.
        Items absent from the train split receive count 0; the downstream
        ``compute_log_q`` is responsible for safe smoothing.
    """
    if n_items <= 0:
        raise ValueError(f"n_items must be positive; got {n_items}")

    counts = torch.zeros(n_items, dtype=torch.long)
    for _uid, items in train_items.items():
        if not items:
            continue
        idx = torch.as_tensor(items, dtype=torch.long)
        if idx.numel() == 0:
            continue
        if int(idx.max()) >= n_items:
            raise ValueError(
                f"train_items contains item id {int(idx.max())} "
                f">= n_items={n_items}. Check load_data.py:103-118 split scan."
            )
        counts.scatter_add_(0, idx, torch.ones_like(idx, dtype=torch.long))
    return counts


# ---------------------------------------------------------------------------
#  2. log q(i) estimators
# ---------------------------------------------------------------------------
_VALID_MODES = ("laplace", "raw", "sqrt")


def compute_log_q(
    counts: torch.Tensor,
    beta: float = 1.0,
    mode: str = "laplace",
) -> torch.Tensor:
    """
    Convert raw counts into ``log q(i)``.

    Args:
        counts : (n_items,) Long/Float tensor — raw interaction counts.
        beta   : Laplace smoothing coefficient. Must be > 0 in "laplace"
                 and "sqrt" modes. Ignored in "raw" mode.
        mode   : one of {"laplace", "raw", "sqrt"}.

    Returns:
        (n_items,) Float tensor of ``log q(i)``. Guaranteed finite.

    Raises:
        ValueError on invalid mode, non-positive β, or zero-count items
        in "raw" mode (which would produce ``log 0 = -inf``).
    """
    if mode not in _VALID_MODES:
        raise ValueError(f"mode must be in {_VALID_MODES}; got {mode!r}")
    if mode in ("laplace", "sqrt") and beta <= 0.0:
        raise ValueError(
            f"beta must be > 0 in mode={mode!r}; got beta={beta}. "
            "Use mode='raw' if you intentionally want unsmoothed counts."
        )

    n_items = counts.shape[0]
    c = counts.detach().to(torch.float64)   # float64 for stable normalisation

    if mode == "laplace":
        # q(i) = (n_i + β) / (Σn_j + n_items · β)
        num = c + beta
        denom = c.sum() + float(n_items) * beta
    elif mode == "sqrt":
        # q(i) ∝ sqrt(n_i + β); normalise to sum to 1.
        num = torch.sqrt(c + beta)
        denom = num.sum()
    else:  # mode == "raw"
        if (c <= 0).any():
            n_zero = int((c <= 0).sum())
            raise ValueError(
                f"mode='raw' requires every item to have ≥ 1 interaction, "
                f"but {n_zero} item(s) have zero count. "
                "Use mode='laplace' (Laplace smoothing) for production runs."
            )
        num = c
        denom = c.sum()

    q = num / denom                              # (n_items,) float64
    log_q = torch.log(q).to(torch.float32)       # downcast for buffer storage

    if not torch.isfinite(log_q).all():
        # Should be impossible with the guards above; defensive check.
        bad = int((~torch.isfinite(log_q)).sum())
        raise ValueError(
            f"compute_log_q produced {bad} non-finite value(s). "
            f"Inputs: counts.sum()={int(c.sum())}, n_items={n_items}, "
            f"beta={beta}, mode={mode!r}."
        )
    return log_q


# ---------------------------------------------------------------------------
#  3. Cached loader
# ---------------------------------------------------------------------------
def _cache_filename(mode: str, beta: float) -> str:
    """Cache filename — mirrors the ``UI_mat_{norm}.pth`` convention."""
    # Normalise β to a stable short string (β=1.0 → "1p0", β=0.1 → "0p1").
    beta_str = f"{beta:.4f}".replace(".", "p").rstrip("0").rstrip("p") or "0"
    return f"log_q_{mode}_b{beta_str}.pth"


def _signature(
    n_items: int,
    counts_sum: int,
    beta: float,
    mode: str,
) -> str:
    """Short SHA1 of (mode, β, n_items, Σcounts) for cache validation."""
    msg = f"{mode}|{beta:.6f}|{n_items}|{counts_sum}".encode("utf-8")
    return hashlib.sha1(msg).hexdigest()[:16]


def load_or_build_log_q(
    cache_dir: str,
    n_items: int,
    train_items: Dict[int, List[int]],
    beta: float = 1.0,
    mode: str = "laplace",
    force_rebuild: bool = False,
) -> torch.Tensor:
    """
    Load ``log_q`` from cache or recompute from scratch.

    The cache file is keyed by (mode, β) and additionally verified by a
    SHA1 signature of (mode, β, n_items, Σcounts). If the dataset split
    changes (different train.json), Σcounts changes, the signature
    mismatches, and the cache is silently rebuilt.

    Args:
        cache_dir     : usually ``data_generator.path`` (the same directory
                        that holds ``UI_mat_sym.pth``).
        n_items       : item-catalog size.
        train_items   : ``{user_id: [item_id, ...]}``.
        beta          : Laplace smoothing coefficient.
        mode          : one of {"laplace", "raw", "sqrt"}.
        force_rebuild : if True, ignore an existing cache and recompute.

    Returns:
        (n_items,) Float tensor of ``log q(i)``.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, _cache_filename(mode, beta))

    counts = compute_item_counts(train_items, n_items)
    sig_now = _signature(n_items, int(counts.sum()), beta, mode)

    if (not force_rebuild) and os.path.exists(cache_path):
        try:
            blob = torch.load(cache_path, map_location="cpu")
            if isinstance(blob, dict) and blob.get("signature") == sig_now:
                log_q = blob["log_q"]
                if log_q.shape == (n_items,) and torch.isfinite(log_q).all():
                    return log_q
                warnings.warn(
                    f"log_q cache at {cache_path} has wrong shape/non-finite "
                    "values — rebuilding."
                )
            else:
                warnings.warn(
                    f"log_q cache at {cache_path} has stale signature — "
                    "rebuilding (dataset split or β/mode likely changed)."
                )
        except Exception as exc:                                # noqa: BLE001
            warnings.warn(f"Failed to load log_q cache ({exc}); rebuilding.")

    log_q = compute_log_q(counts, beta=beta, mode=mode)
    torch.save(
        {
            "signature": sig_now,
            "log_q": log_q,
            "meta": {
                "mode": mode,
                "beta": beta,
                "n_items": n_items,
                "counts_sum": int(counts.sum()),
                "min_count": int(counts.min()),
                "max_count": int(counts.max()),
                "n_zero_items": int((counts == 0).sum()),
            },
        },
        cache_path,
    )
    return log_q


# ---------------------------------------------------------------------------
#  4. Sanity report (called once at train startup; not on hot path)
# ---------------------------------------------------------------------------
def describe_log_q(
    log_q: torch.Tensor,
    counts: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Return a small dict of diagnostics for logging.

    Useful at train startup to verify that log_q is on a sane scale before
    the first contrastive batch — protects against silently passing
    ``log_q = 0`` (uninitialised buffer).
    """
    finite = torch.isfinite(log_q)
    report: Dict[str, float] = {
        "log_q_min": float(log_q[finite].min()) if finite.any() else float("nan"),
        "log_q_max": float(log_q[finite].max()) if finite.any() else float("nan"),
        "log_q_mean": float(log_q[finite].mean()) if finite.any() else float("nan"),
        "log_q_std": float(log_q[finite].std()) if finite.any() else float("nan"),
        "n_items": int(log_q.shape[0]),
        "n_finite": int(finite.sum()),
    }
    if counts is not None:
        report["count_min"] = int(counts.min())
        report["count_max"] = int(counts.max())
        report["count_sum"] = int(counts.sum())
        report["n_zero_items"] = int((counts == 0).sum())
    return report

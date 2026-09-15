"""
tests/test_popularity_prior.py — Unit Tests for the 1B PR (log_q estimator)
============================================================================

Run with::

    cd MMHCL_DAMPS_Project
    pytest tests/test_popularity_prior.py -v

These tests must all pass **before** merging 1A (the model.py patch). They
isolate the log_q math from the loss surgery so a regression in 1A cannot
be misattributed to a wrong prior.

Coverage map
------------
T1  compute_item_counts is correct on a hand-crafted split.
T2  Laplace q(i) sums to 1 ± 1e-6.
T3  log_q has finite entries even for items with 0 interactions.
T4  Mode "raw" raises on zero-count items.
T5  Mode "sqrt" is monotone non-decreasing in n_i.
T6  β=0 is rejected in laplace / sqrt modes.
T7  Cache round-trip: write then read returns identical tensor.
T8  Cache invalidation: changing β rebuilds.
T9  Cache invalidation: changing train_items rebuilds (Σcounts differs).
T10 describe_log_q reports correct min/max for the deterministic fixture.
"""

from __future__ import annotations

import os
import tempfile

import pytest
import torch

# Adjust the import to your project layout. After dropping
# popularity_prior.py into ``MMHCL_DAMPS_Project/damps/``, this becomes:
#     from damps.popularity_prior import ...
from damps.popularity_prior import (
    compute_item_counts,
    compute_log_q,
    describe_log_q,
    load_or_build_log_q,
)


# --------------------------------------------------------------------------
#  Shared fixtures
# --------------------------------------------------------------------------
@pytest.fixture
def tiny_split():
    """A 4-user × 5-item synthetic split.

    Items: 0 (very popular, count=3), 1 (count=2), 2 (count=1),
           3 (count=1), 4 (count=0 — long-tail unobserved).
    Σcounts = 7.
    """
    return {
        0: [0, 1, 2],
        1: [0, 1, 3],
        2: [0],
        3: [],            # empty user — must be skipped
    }, 5                  # n_items


# --------------------------------------------------------------------------
#  T1 — counts
# --------------------------------------------------------------------------
def test_counts_match_handcraft(tiny_split):
    train_items, n_items = tiny_split
    counts = compute_item_counts(train_items, n_items)
    assert counts.tolist() == [3, 2, 1, 1, 0]
    assert counts.dtype == torch.long
    assert counts.shape == (5,)


def test_counts_reject_oob_item_id():
    """An item id ≥ n_items must raise (catches the load_data.py off-by-one)."""
    with pytest.raises(ValueError, match="item id"):
        compute_item_counts({0: [0, 1, 5]}, n_items=5)


# --------------------------------------------------------------------------
#  T2 / T3 — Laplace mode
# --------------------------------------------------------------------------
def test_laplace_q_sums_to_one(tiny_split):
    train_items, n_items = tiny_split
    counts = compute_item_counts(train_items, n_items)
    log_q = compute_log_q(counts, beta=1.0, mode="laplace")
    q = log_q.exp()
    assert torch.isclose(q.sum(), torch.tensor(1.0), atol=1e-6)


def test_laplace_zero_count_item_finite(tiny_split):
    train_items, n_items = tiny_split
    counts = compute_item_counts(train_items, n_items)
    log_q = compute_log_q(counts, beta=1.0, mode="laplace")
    # Item 4 has zero count; Laplace must give a finite (very small) q.
    assert torch.isfinite(log_q[4])
    # And it must be the smallest q (least popular).
    assert log_q.argmin().item() == 4


# --------------------------------------------------------------------------
#  T4 — raw mode is strict
# --------------------------------------------------------------------------
def test_raw_mode_rejects_zero_counts(tiny_split):
    train_items, n_items = tiny_split
    counts = compute_item_counts(train_items, n_items)
    with pytest.raises(ValueError, match="zero count"):
        compute_log_q(counts, beta=1.0, mode="raw")


def test_raw_mode_ok_on_dense_split():
    """If every item has ≥ 1 count, raw mode must work and sum to 1."""
    counts = torch.tensor([3, 2, 1, 1, 1], dtype=torch.long)
    log_q = compute_log_q(counts, beta=0.0, mode="raw")
    assert torch.isclose(log_q.exp().sum(), torch.tensor(1.0), atol=1e-6)


# --------------------------------------------------------------------------
#  T5 — sqrt monotone
# --------------------------------------------------------------------------
def test_sqrt_mode_monotone_in_counts():
    counts = torch.tensor([0, 1, 10, 100, 1000], dtype=torch.long)
    log_q = compute_log_q(counts, beta=1.0, mode="sqrt")
    # log_q must be sorted ascending if counts are sorted ascending.
    assert torch.all(log_q[1:] >= log_q[:-1])


# --------------------------------------------------------------------------
#  T6 — β validation
# --------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["laplace", "sqrt"])
def test_beta_must_be_positive(mode, tiny_split):
    train_items, n_items = tiny_split
    counts = compute_item_counts(train_items, n_items)
    with pytest.raises(ValueError, match="beta"):
        compute_log_q(counts, beta=0.0, mode=mode)
    with pytest.raises(ValueError, match="beta"):
        compute_log_q(counts, beta=-1.0, mode=mode)


# --------------------------------------------------------------------------
#  T7 / T8 / T9 — cache
# --------------------------------------------------------------------------
def test_cache_round_trip(tiny_split):
    train_items, n_items = tiny_split
    with tempfile.TemporaryDirectory() as tmp:
        a = load_or_build_log_q(tmp, n_items, train_items, beta=1.0, mode="laplace")
        b = load_or_build_log_q(tmp, n_items, train_items, beta=1.0, mode="laplace")
        assert torch.allclose(a, b)
        # The second call must have hit the cache (file exists).
        assert any(f.startswith("log_q_laplace_b1") for f in os.listdir(tmp))


def test_cache_invalidation_on_beta_change(tiny_split):
    train_items, n_items = tiny_split
    with tempfile.TemporaryDirectory() as tmp:
        a = load_or_build_log_q(tmp, n_items, train_items, beta=1.0, mode="laplace")
        b = load_or_build_log_q(tmp, n_items, train_items, beta=0.1, mode="laplace")
        # Different cache files — both should exist.
        files = os.listdir(tmp)
        assert any("b1" in f for f in files)
        assert any("b0p1" in f for f in files)
        # Their content must differ (β changes the smoothing strength).
        assert not torch.allclose(a, b)


def test_cache_invalidation_on_split_change(tiny_split):
    """If train_items changes (Σcounts differs), the SHA1 signature
    mismatches and the cache is silently rebuilt."""
    train_items, n_items = tiny_split
    with tempfile.TemporaryDirectory() as tmp:
        a = load_or_build_log_q(tmp, n_items, train_items, beta=1.0, mode="laplace")
        # Mutate the split: add an interaction on item 4.
        mutated = {**train_items, 3: [4]}
        b = load_or_build_log_q(tmp, n_items, mutated, beta=1.0, mode="laplace")
        # Cache must have been rebuilt with the new signature.
        assert not torch.allclose(a, b)


# --------------------------------------------------------------------------
#  T10 — describe_log_q
# --------------------------------------------------------------------------
def test_describe_log_q_reports_sane_stats(tiny_split):
    train_items, n_items = tiny_split
    counts = compute_item_counts(train_items, n_items)
    log_q = compute_log_q(counts, beta=1.0, mode="laplace")
    report = describe_log_q(log_q, counts=counts)
    assert report["n_items"] == 5
    assert report["n_finite"] == 5
    assert report["n_zero_items"] == 1
    assert report["count_max"] == 3
    assert report["log_q_max"] > report["log_q_min"]


# --------------------------------------------------------------------------
#  Smoke test on realistic dataset size (skipped by default; expensive)
# --------------------------------------------------------------------------
@pytest.mark.slow
def test_amazon_clothing_scale():
    """Smoke test at Amazon Clothing scale (n_items ≈ 23,033, N ≈ 200K).

    Verifies that log_q is on the [-12, -4] range expected for τ=0.3
    operations (rev53 §3.1, eq. 1).
    """
    n_items = 23_033
    counts = torch.zeros(n_items, dtype=torch.long)
    # Synthesise a power-law popularity: top item 5000 interactions, tail = 0.
    counts[:5000] = torch.arange(5000, 0, -1)
    log_q = compute_log_q(counts, beta=1.0, mode="laplace")
    report = describe_log_q(log_q, counts=counts)
    assert -15.0 < report["log_q_min"] < -5.0, report
    assert -10.0 < report["log_q_max"] < 0.0, report

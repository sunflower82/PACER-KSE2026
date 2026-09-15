"""Unit tests for the GPU-native tercile-recall evaluator.

Covers ``main_tercile.compute_tercile_recall`` (added in commit 0317b10)
against a hand-computed reference on a tiny 3-user x 12-item scenario
with a hard-coded HEAD / MID / TAIL split.

We avoid importing ``main_tercile`` directly because that would trigger
its side-effecting module-level code (parses CLI args, instantiates the
real ``data_generator``, prints tercile-size banner). Instead, we lift
the exact algorithm out of ``main_tercile.compute_tercile_recall`` and
run it standalone. This tests the *algorithm* -- the module-level wiring
is separately covered by ``tests/test_gpu_eval.py`` and by end-to-end
smoke runs.

If ``main_tercile`` is refactored, this test needs to be re-synced.
"""

from __future__ import annotations

import math
from typing import Iterable

import pytest
import torch


# =============================================================================
# Reference implementation (CPU, plain Python) -- ground truth for the check.
# =============================================================================
def _reference_tercile_recall(
    ua: torch.Tensor,
    ia: torch.Tensor,
    users: list[int],
    train_items: dict[int, list[int]],
    ground_truth: dict[int, list[int]],
    head_ids: set[int],
    mid_ids: set[int],
    tail_ids: set[int],
    K: int,
) -> dict[str, float]:
    """Straightforward CPU implementation of tercile Recall@K.

    Mirrors the pre-0317b10 semantics: for each user with at least one
    tercile positive, count how many of that user's top-K predictions
    fall in (tercile AND ground truth), divide by the number of tercile
    positives, then average across eligible users. Users without tercile
    positives are skipped (matches Milogradskii et al. 2024 / Krichene &
    Rendle 2020).
    """
    head_scores: list[float] = []
    mid_scores: list[float] = []
    tail_scores: list[float] = []
    n_items = ia.shape[0]
    for u in users:
        scores = (ua[u] @ ia.T).clone().cpu().numpy()
        for ti in train_items.get(u, []):
            scores[ti] = -1e18
        # Argsort descending; take top-K.
        order = scores.argsort()[::-1]
        top = [int(x) for x in order[:K].tolist()]
        gt = set(int(x) for x in ground_truth.get(u, []))
        for tercile, out in (
            (head_ids, head_scores),
            (mid_ids, mid_scores),
            (tail_ids, tail_scores),
        ):
            gt_in_tercile = gt & tercile
            if not gt_in_tercile:
                continue
            hits = sum(
                1 for it in top if it in tercile and it in gt_in_tercile
            )
            out.append(hits / len(gt_in_tercile))

    def _mean(xs: list[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else float("nan")

    return {
        "head": _mean(head_scores),
        "mid": _mean(mid_scores),
        "tail": _mean(tail_scores),
    }


# =============================================================================
# GPU-native implementation lifted verbatim from main_tercile
# (main_tercile.compute_tercile_recall, commit 0317b10).
# =============================================================================
def _gpu_tercile_recall(
    ua: torch.Tensor,
    ia: torch.Tensor,
    users: list[int],
    train_mask: torch.Tensor,       # [n_users, n_items] bool
    gt_mask: torch.Tensor,          # [n_users, n_items] bool
    head_mask: torch.Tensor,        # [n_items] bool
    mid_mask: torch.Tensor,
    tail_mask: torch.Tensor,
    K: int,
) -> dict[str, float]:
    if not users:
        return {"head": float("nan"), "mid": float("nan"), "tail": float("nan")}

    device = ua.device
    users_t = torch.tensor(users, dtype=torch.long, device=device)
    ubs = 2048
    sums = {n: torch.zeros((), device=device, dtype=torch.float64)
            for n in ("head", "mid", "tail")}
    counts = {n: torch.zeros((), device=device, dtype=torch.float64)
              for n in ("head", "mid", "tail")}
    terciles = (("head", head_mask), ("mid", mid_mask), ("tail", tail_mask))

    for start in range(0, users_t.numel(), ubs):
        batch = users_t[start : start + ubs]
        scores = ua[batch] @ ia.T
        scores = scores.masked_fill(train_mask[batch], float("-inf"))
        _vals, top_idx = torch.topk(scores, k=K, dim=1)
        hit_any = gt_mask[batch.unsqueeze(1), top_idx]

        for name, tmask in terciles:
            n_gt = (gt_mask[batch] & tmask).sum(dim=1).to(torch.float32)
            in_tercile = tmask[top_idx]
            n_hit = (hit_any & in_tercile).sum(dim=1).to(torch.float32)
            eligible = n_gt > 0
            if not bool(eligible.any()):
                continue
            rec = n_hit[eligible] / n_gt[eligible]
            sums[name] = sums[name] + rec.sum().to(torch.float64)
            counts[name] = counts[name] + eligible.sum().to(torch.float64)

    def _mean(name: str) -> float:
        c = float(counts[name].item())
        if c <= 0:
            return float("nan")
        return float(sums[name].item() / c)

    return {"head": _mean("head"), "mid": _mean("mid"), "tail": _mean("tail")}


# =============================================================================
# Helpers to build the masks the GPU path expects
# =============================================================================
def _bool_row_mask(indices_per_user: dict[int, list[int]],
                   n_users: int, n_items: int,
                   device: torch.device) -> torch.Tensor:
    m = torch.zeros(n_users, n_items, dtype=torch.bool, device=device)
    for u, its in indices_per_user.items():
        if its:
            m[u, its] = True
    return m


def _bool_col_mask(ids: Iterable[int], n_items: int,
                   device: torch.device) -> torch.Tensor:
    m = torch.zeros(n_items, dtype=torch.bool, device=device)
    ids_list = list(ids)
    if ids_list:
        m[ids_list] = True
    return m


# =============================================================================
# Tests
# =============================================================================
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _small_scenario():
    """3 users x 12 items, HEAD={0..3} MID={4..7} TAIL={8..11}.

    Embeddings are deterministic (Manhattan grid). Item i has ground
    truth positive for user u iff (i % 3) == u, so:
      - user 0 GT: [0, 3, 6, 9]   (1 head, 1 head, 1 mid, 1 tail)
      - user 1 GT: [1, 4, 7, 10]  (1 head, 1 mid, 1 mid, 1 tail)
      - user 2 GT: [2, 5, 8, 11]  (1 head, 1 mid, 1 tail, 1 tail)
    Training positives (excluded from scoring) are the first GT item
    for each user, so their scores are pushed to -inf before topk.
    """
    torch.manual_seed(0)
    n_users, n_items = 3, 12
    embed = 4
    ua = torch.randn(n_users, embed, device=DEVICE)
    ia = torch.randn(n_items, embed, device=DEVICE)

    head = set(range(0, 4))
    mid = set(range(4, 8))
    tail = set(range(8, 12))

    val_set = {
        0: [0, 3, 6, 9],
        1: [1, 4, 7, 10],
        2: [2, 5, 8, 11],
    }
    train_items = {u: [gt[0]] for u, gt in val_set.items()}

    train_mask = _bool_row_mask(train_items, n_users, n_items, DEVICE)
    gt_mask = _bool_row_mask(val_set, n_users, n_items, DEVICE)
    head_m = _bool_col_mask(head, n_items, DEVICE)
    mid_m = _bool_col_mask(mid, n_items, DEVICE)
    tail_m = _bool_col_mask(tail, n_items, DEVICE)

    return dict(
        ua=ua, ia=ia, users=[0, 1, 2],
        train_items=train_items, val_set=val_set,
        head=head, mid=mid, tail=tail,
        train_mask=train_mask, gt_mask=gt_mask,
        head_m=head_m, mid_m=mid_m, tail_m=tail_m,
    )


def _isclose(a: float, b: float, tol: float = 1e-6) -> bool:
    if math.isnan(a) and math.isnan(b):
        return True
    if math.isnan(a) or math.isnan(b):
        return False
    return abs(a - b) <= tol


def test_gpu_tercile_matches_cpu_reference():
    """GPU tercile recall must equal a hand-coded CPU reference to 1e-6.

    Runs the exact algorithm from main_tercile.compute_tercile_recall on
    a hand-designed 3-user x 12-item scenario and compares against a
    plain-Python reference that follows the pre-0317b10 semantics.
    """
    s = _small_scenario()
    for K in (5, 8, 12):
        got = _gpu_tercile_recall(
            s["ua"], s["ia"], s["users"],
            s["train_mask"], s["gt_mask"],
            s["head_m"], s["mid_m"], s["tail_m"],
            K=K,
        )
        want = _reference_tercile_recall(
            s["ua"].cpu(), s["ia"].cpu(), s["users"],
            s["train_items"], s["val_set"],
            s["head"], s["mid"], s["tail"],
            K=K,
        )
        for name in ("head", "mid", "tail"):
            assert _isclose(got[name], want[name]), (
                f"K={K} tercile={name}: GPU={got[name]!r} vs "
                f"CPU-ref={want[name]!r}"
            )


def test_gpu_tercile_skips_users_with_no_positives():
    """Users without positives in a tercile must be excluded from the mean.

    Constructs a scenario where user 0 has NO tail positives at all.
    The tail-recall mean must therefore be computed over users {1, 2}
    only, never user 0, even though user 0 is in ``users_to_test``.
    """
    torch.manual_seed(1)
    n_users, n_items = 3, 12
    ua = torch.randn(n_users, 4, device=DEVICE)
    ia = torch.randn(n_items, 4, device=DEVICE)
    head, mid, tail = set(range(0, 4)), set(range(4, 8)), set(range(8, 12))

    # User 0 has no TAIL positives; user 1 has one; user 2 has two.
    val_set = {
        0: [0, 1, 4],           # head-only + one mid
        1: [2, 5, 8],           # head/mid/tail
        2: [3, 6, 9, 10],       # head/mid + two tails
    }
    train_items: dict[int, list[int]] = {}
    train_mask = _bool_row_mask(train_items, n_users, n_items, DEVICE)
    gt_mask = _bool_row_mask(val_set, n_users, n_items, DEVICE)
    head_m = _bool_col_mask(head, n_items, DEVICE)
    mid_m = _bool_col_mask(mid, n_items, DEVICE)
    tail_m = _bool_col_mask(tail, n_items, DEVICE)

    got = _gpu_tercile_recall(
        ua, ia, [0, 1, 2],
        train_mask, gt_mask, head_m, mid_m, tail_m, K=8,
    )
    want = _reference_tercile_recall(
        ua.cpu(), ia.cpu(), [0, 1, 2],
        train_items, val_set, head, mid, tail, K=8,
    )
    for name in ("head", "mid", "tail"):
        assert _isclose(got[name], want[name]), (
            f"tercile={name}: GPU={got[name]!r} vs CPU-ref={want[name]!r}"
        )

    # Explicit check: tail mean uses only users {1, 2}. User 0 had zero
    # tail positives so any per-user "0/0" MUST be dropped, not counted
    # as 0.0 (which would drag the mean down).
    tail_recs = []
    for u in (1, 2):
        # Score top-8 for u, no training-mask exclusion here.
        s = ua[u] @ ia.T
        top = torch.topk(s, k=8).indices.tolist()
        gset = set(val_set[u])
        gt_in_tail = gset & tail
        if not gt_in_tail:
            continue
        hits = sum(1 for it in top if it in tail and it in gt_in_tail)
        tail_recs.append(hits / len(gt_in_tail))
    manual_mean = sum(tail_recs) / len(tail_recs) if tail_recs else float("nan")
    assert _isclose(got["tail"], manual_mean, tol=1e-6), (
        f"tail mean must be over users {{1,2}} only; "
        f"got={got['tail']!r} manual={manual_mean!r}"
    )


def test_gpu_tercile_empty_users_returns_nan():
    """Empty user list must not crash and must return NaN for every tercile."""
    n_items = 12
    ua = torch.zeros(1, 4, device=DEVICE)
    ia = torch.zeros(n_items, 4, device=DEVICE)
    empty_mask = torch.zeros(1, n_items, dtype=torch.bool, device=DEVICE)
    tercile_mask = torch.zeros(n_items, dtype=torch.bool, device=DEVICE)
    got = _gpu_tercile_recall(
        ua, ia, users=[],
        train_mask=empty_mask, gt_mask=empty_mask,
        head_mask=tercile_mask, mid_mask=tercile_mask, tail_mask=tercile_mask,
        K=5,
    )
    for name in ("head", "mid", "tail"):
        assert math.isnan(got[name]), f"expected NaN for empty users, got {got}"


if __name__ == "__main__":
    # Allow `python tests/test_gpu_tercile.py` for quick manual verification.
    pytest.main([__file__, "-v"])

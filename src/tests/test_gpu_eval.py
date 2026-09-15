"""Equivalence smoke tests for GPU-native eval helpers."""

from __future__ import annotations

import torch

from utility import metrics as mtr


def _ndcg_from_hits(hit: torch.Tensor, k: int) -> torch.Tensor:
    """Mirror of ``batch_test._ndcg_from_hits`` (kept local to avoid Data load)."""
    r = hit[:, :k].to(dtype=torch.float32)
    device = r.device
    discounts = (
        1.0 / torch.log2(torch.arange(2, k + 2, device=device, dtype=torch.float32))
    )
    dcg = (r * discounts).sum(dim=1)
    ideal, _ = torch.sort(r, dim=1, descending=True)
    idcg = (ideal * discounts).sum(dim=1)
    return torch.where(idcg > 0, dcg / idcg, torch.zeros_like(dcg))


def test_ndcg_matches_metrics_helper() -> None:
    """GPU NDCG reduction must match ``metrics.ndcg_at_k`` (method=1)."""
    hit = torch.tensor(
        [
            [1, 0, 1, 0],
            [0, 0, 0, 0],
            [1, 1, 1, 0],
        ],
        dtype=torch.bool,
    )
    for k in (1, 2, 3, 4):
        gpu = _ndcg_from_hits(hit, k).numpy()
        for i in range(hit.size(0)):
            r = hit[i, :k].int().tolist()
            cpu = mtr.ndcg_at_k(r, k)
            assert abs(float(gpu[i]) - cpu) < 1e-6, (k, i, gpu[i], cpu)


def test_gpu_topk_mask_semantics() -> None:
    """Masked train items must not enter top-K."""
    torch.manual_seed(0)
    n_users, n_items, d = 4, 10, 8
    ua = torch.nn.functional.normalize(torch.randn(n_users, d), dim=-1)
    ia = torch.nn.functional.normalize(torch.randn(n_items, d), dim=-1)

    train_mask = torch.zeros(n_users, n_items, dtype=torch.bool)
    train_mask[0, [0, 1]] = True

    scores = ua[0:1] @ ia.T
    scores = scores.masked_fill(train_mask[0:1], float("-inf"))
    _vals, top_idx = torch.topk(scores, k=5, dim=1)
    assert 0 not in top_idx[0].tolist()
    assert 1 not in top_idx[0].tolist()


if __name__ == "__main__":
    test_ndcg_matches_metrics_helper()
    test_gpu_topk_mask_semantics()
    print("test_gpu_eval: OK")

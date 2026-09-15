"""
utility/batch_test.py — Evaluation Pipeline
=============================================

Evaluates trained DAMPS-MMHCL models on validation/test splits.

Default path (``--use_gpu_eval 1``) is a fully GPU-native scorer that
keeps embeddings, train/GT masks, top-K, and metric reductions on-device
(PACER_NRDMC_lite_eval_bottleneck_EN §4). The legacy CPU multiprocessing
path remains available behind ``--use_gpu_eval 0`` for bit-level audits.

Public API
----------
*   ``data_generator`` — global ``Data`` instance, shared with ``train.py``.
*   ``test_torch(ua_emb, ia_emb, users_to_test, is_val)`` — evaluation.
"""

from __future__ import annotations

import atexit
import heapq
import multiprocessing
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from utility import metrics as mtr
from utility.load_data import Data
from utility.parser import parse_args


args = parse_args()
Ks: list[int] = eval(args.Ks)

# ---------------------------------------------------------------------------
#  Multiprocessing pool size (capped to avoid host CPU starvation)
# ---------------------------------------------------------------------------
_cpu_total: int = multiprocessing.cpu_count()
cores: int = max(1, min(_cpu_total // 5 if _cpu_total > 5 else 1, 8))

# Persistent pool for the CPU fallback (created lazily, closed at exit).
_eval_pool: multiprocessing.Pool | None = None


def _get_eval_pool() -> multiprocessing.Pool:
    """Return a process-wide persistent ``Pool`` (speedup guide §5.1)."""
    global _eval_pool
    if _eval_pool is None:
        _eval_pool = multiprocessing.Pool(cores)

        def _close_pool() -> None:
            global _eval_pool
            if _eval_pool is not None:
                _eval_pool.close()
                _eval_pool.join()
                _eval_pool = None

        atexit.register(_close_pool)
    return _eval_pool


# ---------------------------------------------------------------------------
#  Module-level data generator
# ---------------------------------------------------------------------------
data_generator: Data = Data(
    path=args.data_path + args.dataset, batch_size=args.batch_size
)
USR_NUM: int = data_generator.n_users
ITEM_NUM: int = data_generator.n_items
N_TRAIN: int = data_generator.n_train
N_TEST: int = data_generator.n_test
BATCH_SIZE: int = args.batch_size


MetricsDict = dict[str, Any]


# ---------------------------------------------------------------------------
#  Per-user ranking helpers (CPU fallback)
# ---------------------------------------------------------------------------
def ranklist_by_heapq(
    user_pos_test: list[int],
    test_items: list[int],
    rating: npt.NDArray[np.floating],
    Ks_local: list[int],
) -> tuple[list[int], float]:
    """Top-K ranking via heapq (faster, no AUC)."""
    item_score = {i: rating[i] for i in test_items}
    K_max = max(Ks_local)
    top_items = heapq.nlargest(K_max, item_score, key=item_score.get)
    r = [1 if i in user_pos_test else 0 for i in top_items]
    return r, 0.0


def ranklist_by_sorted(
    user_pos_test: list[int],
    test_items: list[int],
    rating: npt.NDArray[np.floating],
    Ks_local: list[int],
) -> tuple[list[int], float]:
    """Full sort ranking + AUC."""
    item_score = {i: rating[i] for i in test_items}
    K_max = max(Ks_local)
    top_items = heapq.nlargest(K_max, item_score, key=item_score.get)
    r = [1 if i in user_pos_test else 0 for i in top_items]
    sorted_items = sorted(item_score.items(), key=lambda kv: kv[1], reverse=True)
    posterior = [v for _, v in sorted_items]
    gt = [1 if i in user_pos_test else 0 for i, _ in sorted_items]
    return r, mtr.auc(ground_truth=gt, prediction=posterior)


def get_performance(
    user_pos_test: list[int],
    r: list[int],
    auc_val: float,
    Ks_local: list[int],
) -> MetricsDict:
    """Return per-K metric arrays for one user."""
    precision: list[float] = []
    recall: list[float] = []
    ndcg: list[float] = []
    hit_ratio: list[float] = []
    for K in Ks_local:
        precision.append(mtr.precision_at_k(r, K))
        recall.append(mtr.recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(mtr.ndcg_at_k(r, K))
        hit_ratio.append(mtr.hit_at_k(r, K))
    return {
        "recall": np.array(recall),
        "precision": np.array(precision),
        "ndcg": np.array(ndcg),
        "hit_ratio": np.array(hit_ratio),
        "auc": auc_val,
    }


def test_one_user(x: tuple[npt.NDArray[np.floating], int, bool]) -> MetricsDict:
    """Evaluate a single user — designed for multiprocessing.Pool.map."""
    rating: npt.NDArray[np.floating] = x[0]
    u: int = x[1]
    is_val: bool = x[2]

    try:
        training_items = data_generator.train_items[u]
    except Exception:
        training_items = []

    user_pos_test = (
        data_generator.val_set[u] if is_val else data_generator.test_set[u]
    )
    candidates = list(set(range(ITEM_NUM)) - set(training_items))

    if args.test_flag == "part":
        r, auc_val = ranklist_by_heapq(user_pos_test, candidates, rating, Ks)
    else:
        r, auc_val = ranklist_by_sorted(user_pos_test, candidates, rating, Ks)
    return get_performance(user_pos_test, r, auc_val, Ks)


# ---------------------------------------------------------------------------
#  GPU-native evaluation (primary path)
# ---------------------------------------------------------------------------
def _ndcg_from_hits(hit: torch.Tensor, k: int) -> torch.Tensor:
    """Per-user NDCG@k matching ``metrics.ndcg_at_k`` (method=1).

    Ideal DCG is computed from the sorted top-k binary relevance vector
    (not from the global positive count), identical to the CPU helper.
    """
    r = hit[:, :k].to(dtype=torch.float32)  # [B, k]
    device = r.device
    discounts = (
        1.0 / torch.log2(torch.arange(2, k + 2, device=device, dtype=torch.float32))
    )
    dcg = (r * discounts).sum(dim=1)
    ideal, _ = torch.sort(r, dim=1, descending=True)
    idcg = (ideal * discounts).sum(dim=1)
    return torch.where(idcg > 0, dcg / idcg, torch.zeros_like(dcg))


@torch.inference_mode()
def test_torch_gpu(
    ua_embeddings: torch.Tensor,
    ia_embeddings: torch.Tensor,
    users_to_test: list[int],
    is_val: bool,
    u_batch_size: int | None = None,
) -> MetricsDict:
    """GPU-native Recall / NDCG / Precision / Hit@K (no D2H, no Pool).

    Args:
        ua_embeddings: ``[n_users, d]`` user embeddings on device.
        ia_embeddings: ``[n_items, d]`` item embeddings on device.
        users_to_test: User IDs to evaluate.
        is_val: ``True`` → val ground-truth; ``False`` → test.
        u_batch_size: Users per scoring matmul. Defaults to ``2 * BATCH_SIZE``.

    Returns:
        Metric dict with the same keys / shapes as the CPU ``test_torch``.
    """
    if not users_to_test:
        return {
            "precision": np.zeros(len(Ks)),
            "recall": np.zeros(len(Ks)),
            "ndcg": np.zeros(len(Ks)),
            "hit_ratio": np.zeros(len(Ks)),
            "auc": 0.0,
        }

    device = ua_embeddings.device
    n_test_users = len(users_to_test)
    K_max = max(Ks)
    if u_batch_size is None:
        u_batch_size = max(1, BATCH_SIZE * 2)

    train_mask = data_generator.get_train_mask_gpu(device)
    gt_mask = data_generator.get_gt_mask_gpu(is_val, device)

    users_t = torch.tensor(users_to_test, dtype=torch.long, device=device)
    n_pos = gt_mask[users_t].sum(dim=1).to(dtype=torch.float32)  # [N]

    sum_precision = torch.zeros(len(Ks), device=device, dtype=torch.float64)
    sum_recall = torch.zeros(len(Ks), device=device, dtype=torch.float64)
    sum_ndcg = torch.zeros(len(Ks), device=device, dtype=torch.float64)
    sum_hit = torch.zeros(len(Ks), device=device, dtype=torch.float64)

    for start in range(0, n_test_users, u_batch_size):
        end = min(start + u_batch_size, n_test_users)
        batch_users = users_t[start:end]
        u_emb = ua_embeddings[batch_users]
        scores = u_emb @ ia_embeddings.T
        scores = scores.masked_fill(train_mask[batch_users], float("-inf"))

        _top_scores, top_idx = torch.topk(scores, k=K_max, dim=1)
        hit = gt_mask[batch_users.unsqueeze(1), top_idx]  # [B, K_max]
        batch_n_pos = n_pos[start:end]

        for ki, K in enumerate(Ks):
            r_k = hit[:, :K].to(dtype=torch.float32)
            sum_precision[ki] += r_k.mean(dim=1).sum()
            # Match metrics.recall_at_k: 0 when all_pos_num == 0.
            rec = torch.where(
                batch_n_pos > 0,
                r_k.sum(dim=1) / batch_n_pos,
                torch.zeros_like(batch_n_pos),
            )
            sum_recall[ki] += rec.sum()
            sum_ndcg[ki] += _ndcg_from_hits(hit, K).sum()
            sum_hit[ki] += (r_k.sum(dim=1) > 0).to(dtype=torch.float32).sum()

    denom = float(n_test_users)
    return {
        "precision": (sum_precision / denom).detach().cpu().numpy(),
        "recall": (sum_recall / denom).detach().cpu().numpy(),
        "ndcg": (sum_ndcg / denom).detach().cpu().numpy(),
        "hit_ratio": (sum_hit / denom).detach().cpu().numpy(),
        "auc": 0.0,
    }


def test_torch_cpu(
    ua_embeddings: torch.Tensor,
    ia_embeddings: torch.Tensor,
    users_to_test: list[int],
    is_val: bool,
    batch_test_flag: bool = False,
) -> MetricsDict:
    """Legacy CPU multiprocessing eval with a persistent ``Pool``."""
    del batch_test_flag  # retained for API compatibility
    result: MetricsDict = {
        "precision": np.zeros(len(Ks)),
        "recall": np.zeros(len(Ks)),
        "ndcg": np.zeros(len(Ks)),
        "hit_ratio": np.zeros(len(Ks)),
        "auc": 0.0,
    }
    pool = _get_eval_pool()

    u_batch_size = BATCH_SIZE * 2
    n_test_users = len(users_to_test)
    n_user_batchs = (n_test_users + u_batch_size - 1) // u_batch_size

    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = min((u_batch_id + 1) * u_batch_size, n_test_users)
        user_batch = users_to_test[start:end]
        if not user_batch:
            continue

        u_emb = ua_embeddings[user_batch]
        # non_blocking D2H + pin via .cpu(); numpy() still syncs but avoids
        # an extra pageable staging copy on some drivers.
        rate_batch = (
            torch.matmul(u_emb, ia_embeddings.transpose(0, 1))
            .detach()
            .cpu()
            .numpy()
        )

        payload = list(zip(rate_batch, user_batch, [is_val] * len(user_batch)))
        batch_result = pool.map(test_one_user, payload)

        for re in batch_result:
            result["precision"] += re["precision"] / n_test_users
            result["recall"] += re["recall"] / n_test_users
            result["ndcg"] += re["ndcg"] / n_test_users
            result["hit_ratio"] += re["hit_ratio"] / n_test_users
            result["auc"] += re["auc"] / n_test_users
    return result


# ---------------------------------------------------------------------------
#  Main evaluation entry point
# ---------------------------------------------------------------------------
def test_torch(
    ua_embeddings: torch.Tensor,
    ia_embeddings: torch.Tensor,
    users_to_test: list[int],
    is_val: bool,
    batch_test_flag: bool = False,
) -> MetricsDict:
    """
    Evaluate the model on a list of users.

    Args:
        ua_embeddings   : (n_users, d) all user embeddings (GPU).
        ia_embeddings   : (n_items, d) all item embeddings (GPU).
        users_to_test   : list of user IDs to evaluate.
        is_val          : True for validation, False for test.
        batch_test_flag : retained for API compatibility (unused on GPU path).

    Returns:
        dict with keys: 'precision', 'recall', 'ndcg', 'hit_ratio', 'auc'.
    """
    use_gpu = bool(getattr(args, "use_gpu_eval", 1))
    if use_gpu and ua_embeddings.is_cuda:
        return test_torch_gpu(
            ua_embeddings, ia_embeddings, users_to_test, is_val
        )
    return test_torch_cpu(
        ua_embeddings,
        ia_embeddings,
        users_to_test,
        is_val,
        batch_test_flag=batch_test_flag,
    )

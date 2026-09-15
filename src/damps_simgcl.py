"""
damps_simgcl.py -- Branch A shim over batch-N SimGCL helpers.
=============================================================

Re-exports the Branch A batch-N primitives from ``branchA_simgcl_batchN``
while preserving the rev54 public API expected by ``tests/test_simgcl.py``.

``compute_simgcl_view_loss`` returns a scalar (loss only) for backward
compatibility; the full Branch A path in ``model.simgcl_view_forward``
imports ``branchA_simgcl_batchN`` directly and uses the 4-tuple view cache.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from branchA_simgcl_batchN import (
    batched_contrastive_loss_batchN,
    inject_uniform_noise,
    simgcl_view_invariance_loss,
)
from branchA_simgcl_batchN import (
    compute_simgcl_view_loss as _compute_simgcl_view_loss_full,
)


def compute_simgcl_view_loss(
    *,
    propagate_fn,
    ego_user: torch.Tensor,
    ego_item: torch.Tensor,
    eps: float,
    tau: torch.Tensor | float,
    batch_size_user: int = 2048,
    batch_size_item: int = 2048,
    views_cached: Optional[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ] = None,
) -> torch.Tensor:
    """Backward-compatible wrapper: return scalar L_view only."""
    loss, _ = _compute_simgcl_view_loss_full(
        propagate_fn=propagate_fn,
        ego_user=ego_user,
        ego_item=ego_item,
        eps=eps,
        tau=tau,
        batch_size_user=batch_size_user,
        batch_size_item=batch_size_item,
        views_cached=views_cached,
    )
    return loss


__all__ = [
    "batched_contrastive_loss_batchN",
    "compute_simgcl_view_loss",
    "inject_uniform_noise",
    "simgcl_view_invariance_loss",
]

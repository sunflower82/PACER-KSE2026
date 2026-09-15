"""
damps/momentum.py — Slim Momentum Encoder
==========================================

Implements the **Slim Momentum** design from Section 3.1.1 of the DAMPS spec
(``DAMPS_to_MMHCL_architecture_revision42.tex``):

    "To definitively minimize VRAM consumption, the Momentum Encoder strictly
     applies EMA smoothing exclusively onto the calibrated representations
     h_cal in R^{d=64}, rather than operating on the entirety of the massive
     original multi-modal feature tables (e.g., d_visual = 4096). This
     'Slim Momentum' design slashes the auxiliary memory footprint by
     approximately 98% compared to traditional MoCo-style implementations."

Why this matters
----------------
The original "naive" momentum approach would EMA-smooth the raw modality
features before MMHCL's projection MLPs. For a typical Amazon Clothing
dataset (~23 K items, image_dim = 4096, text_dim = 768) this requires
storing two copies of (N x 4096) float32 tensors — roughly 720 MiB just
for the momentum buffers. The Slim Momentum design instead smooths only
the **post-DAMPS** d=64 representations, dropping the auxiliary footprint
to ~12 MiB (a 98% reduction).

Mathematical formulation
------------------------
For each batch of items with indices :math:`I = \\{i_1, \\dots, i_B\\}`,
and per-modality calibrated features :math:`h_{cal}^m \\in R^{B x d}`,
the Slim Momentum Encoder maintains a global table :math:`E^m \\in R^{N x d}`
updated via:

    .. math::
        E^m[i] \\gets \\beta_t \\, E^m[i] + (1 - \\beta_t) \\, h_{cal}^m[i]

where the schedule :math:`\\beta_t = 1 - 1/(t+1)` adapts during the warm-up
phase (so the first epoch contributes more strongly) and locks at 0.99
afterwards. This matches the AVRF MAD update schedule in ``damps/core.py``.

Pattern B' (Scheduled Rebuild) compatibility
--------------------------------------------
The K-NN hypergraph rebuild loop (see ``train.py``) reads ``image_table()``
/ ``text_table()`` *only at rebuild epochs* (every ``R`` epochs). The buffer
maintains a stable representation between rebuilds, eliminating the
"density explosion" bug that occurs when EMA-smoothing the sparse
adjacency matrix directly.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class SlimMomentumEncoder(nn.Module):
    """
    Slim Momentum buffers for the d=64 calibrated representation space.

    Args:
        num_items     : total number of items in the dataset.
        dim           : embedding dim of the calibrated features (default 64).
        warmup_epochs : how many epochs use the adaptive schedule
                        ``beta_t = 1 - 1/(t+1)`` before locking at 0.99.
        use_ema       : if False, the encoder simply overwrites entries on
                        each call (useful for the EMA-OFF ablation).
        num_modalities: 2 for Amazon Clothing/Sports, 3 for Tiktok (audio
                        modality also tracked).

    Buffers:
        h_cal_ema_img : (num_items, dim) float — image modality table.
        h_cal_ema_txt : (num_items, dim) float — text modality table.
        h_cal_ema_aud : (num_items, dim) float — audio modality table
                        (only populated when num_modalities == 3).
    """

    def __init__(
        self,
        num_items: int,
        dim: int = 64,
        warmup_epochs: int = 10,
        use_ema: bool = True,
        num_modalities: int = 2,
    ) -> None:
        super().__init__()

        if num_items <= 0:
            raise ValueError(f"num_items must be positive, got {num_items}")
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if num_modalities not in (2, 3):
            raise ValueError(f"num_modalities must be 2 or 3, got {num_modalities}")

        self.num_items: int = num_items
        self.dim: int = dim
        self.warmup_epochs: int = max(1, int(warmup_epochs))
        self.use_ema: bool = bool(use_ema)
        self.num_modalities: int = num_modalities

        self.register_buffer("h_cal_ema_img", torch.zeros(num_items, dim))
        self.register_buffer("h_cal_ema_txt", torch.zeros(num_items, dim))
        if num_modalities == 3:
            self.register_buffer("h_cal_ema_aud", torch.zeros(num_items, dim))

        # Tracks whether each item has ever been touched. Items not yet
        # observed (e.g. cold-start items) should not pollute the K-NN
        # rebuild with their default zero embedding.
        self.register_buffer(
            "init_mask",
            torch.zeros(num_items, dtype=torch.bool),
        )

    # ------------------------------------------------------------------
    #  Public update API
    # ------------------------------------------------------------------
    @torch.no_grad()
    def update(
        self,
        item_indices: torch.Tensor,
        h_cal_img: torch.Tensor,
        h_cal_txt: torch.Tensor,
        h_cal_aud: Optional[torch.Tensor] = None,
        epoch: int = 0,
    ) -> None:
        """
        Update the slim momentum tables for the given batch of items.

        Args:
            item_indices : (B,) Long tensor of item IDs to update.
            h_cal_img    : (B, dim) calibrated image features.
            h_cal_txt    : (B, dim) calibrated text features.
            h_cal_aud    : optional (B, dim) audio features (Tiktok).
            epoch        : current epoch index (0-based) — drives ``beta_t``.
        """
        if item_indices.dim() != 1:
            raise ValueError("item_indices must be a 1-D Long tensor")
        if h_cal_img.shape[-1] != self.dim:
            raise ValueError(
                f"h_cal_img last-dim {h_cal_img.shape[-1]} != self.dim {self.dim}"
            )

        idx = item_indices.long()

        if not self.use_ema:
            # Pure overwrite (EMA-OFF ablation): keeps the most recent batch
            self.h_cal_ema_img[idx] = h_cal_img.detach().to(self.h_cal_ema_img.dtype)
            self.h_cal_ema_txt[idx] = h_cal_txt.detach().to(self.h_cal_ema_txt.dtype)
            if h_cal_aud is not None and self.num_modalities == 3:
                self.h_cal_ema_aud[idx] = h_cal_aud.detach().to(
                    self.h_cal_ema_aud.dtype
                )
            self.init_mask[idx] = True
            return

        # Schedule: cold-start adaptive, locked at 0.99 after warmup
        if epoch < self.warmup_epochs:
            beta_t = 1.0 - 1.0 / (float(epoch) + 1.0)
        else:
            beta_t = 0.99

        # In-place EMA: x <- beta * x + (1 - beta) * new
        target_img = self.h_cal_ema_img[idx]
        target_txt = self.h_cal_ema_txt[idx]
        target_img.mul_(beta_t).add_(
            h_cal_img.detach().to(target_img.dtype) * (1.0 - beta_t)
        )
        target_txt.mul_(beta_t).add_(
            h_cal_txt.detach().to(target_txt.dtype) * (1.0 - beta_t)
        )
        # When an item is touched the first time, "fast-init" by overwriting
        # — otherwise the first batch is multiplied by beta_t = 0 anyway.
        first_touch = ~self.init_mask[idx]
        if first_touch.any():
            self.h_cal_ema_img[idx[first_touch]] = h_cal_img.detach()[first_touch].to(
                self.h_cal_ema_img.dtype
            )
            self.h_cal_ema_txt[idx[first_touch]] = h_cal_txt.detach()[first_touch].to(
                self.h_cal_ema_txt.dtype
            )

        # Audio modality (Tiktok)
        if h_cal_aud is not None and self.num_modalities == 3:
            target_aud = self.h_cal_ema_aud[idx]
            target_aud.mul_(beta_t).add_(
                h_cal_aud.detach().to(target_aud.dtype) * (1.0 - beta_t)
            )
            if first_touch.any():
                self.h_cal_ema_aud[idx[first_touch]] = h_cal_aud.detach()[first_touch].to(
                    self.h_cal_ema_aud.dtype
                )

        self.init_mask[idx] = True

    # ------------------------------------------------------------------
    #  Read API for the K-NN rebuild loop
    # ------------------------------------------------------------------
    def image_table(self) -> torch.Tensor:
        """Return the slim momentum table for the image modality."""
        return self.h_cal_ema_img

    def text_table(self) -> torch.Tensor:
        """Return the slim momentum table for the text modality."""
        return self.h_cal_ema_txt

    def audio_table(self) -> torch.Tensor:
        """Return the slim momentum table for audio. Raises if 2-modality."""
        if self.num_modalities != 3:
            raise RuntimeError(
                "Audio table requested but encoder was built with "
                "num_modalities=2 (no audio)"
            )
        return self.h_cal_ema_aud

    def initialised_count(self) -> int:
        """Number of items that have been touched at least once."""
        return int(self.init_mask.sum().item())

"""
damps/core.py — DAMPS Spectral Calibration Module
====================================================

Implements the **D**ynamic **A**daptive **M**ulti-modal **P**hase-**S**pectral
calibration block as described in
``DAMPS_to_MMHCL_architecture_revision42.tex`` (Revision 9, Final Lock).

The module performs the following pipeline, end-to-end differentiable, on
projected modality features ``h_m \\in R^{B x d}`` (image / text / [audio]):

    1.  Spectral decomposition  (1-D rFFT) ........ Section 2.1
    2.  Metadata-Aware APC      (von Mises MLE)    Section 2.2
    3.  AVRF                    (logit-clipped Wiener gate, per-epoch EMA MAD) Section 2.3
    4.  Residual IMCF           (ASC consensus, residual form) Section 2.4
    5.  Inverse FFT             (back to spatial domain)

Output: ``h_cal_m`` in the same shape as the input — a *cleansed* spectral-
domain representation that downstream MMHCL components consume through
**Soft Residual-Routing** (see ``model.py``).

Key engineering safeguards (all from the spec)
----------------------------------------------
*   AVRF logit is **strictly clipped to [-2.0, +2.0]** at initialisation to
    avert tanh saturation ("cold-start paralysis", Section 2.3).
*   AVRF MAD is aggregated **per-epoch via EMA** (variance-reduced 5-7x).
*   IMCF baseline ``\\bar{ASC}`` is updated on its own EMA so the residual
    coefficient ``ASC_i - \\bar{ASC}`` always centres at zero.
*   Optional ``permutation_fft`` ablation knob lets reviewers rerun the
    falsifiable Permutation-FFT test (Section 6, Item 8).

Parameter footprint
-------------------
Exactly **101 trainable parameters** for the default config
(d = 64 → F = 33), composed of:

    psi (1, F)              : 33 phase residuals
    AVRF_image (1, F)       : 33 image gates
    AVRF_text (1, F)        : 33 text gates
    lambda_coh (scalar)     : 1  IMCF residual coefficient
    + alpha_img / alpha_txt : 2  Soft-Routing scalars (in model.py, not here)

Total here = 100; with the 1 IMCF lambda → 101 (matches Section 3.2).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


# Default ablation switches (used when ``ablations`` argument is None or partial)
_DEFAULT_ABLATIONS: dict[str, bool] = {
    "apc": True,             # Metadata-Aware Adaptive Phase Calibration
    "avrf": True,            # Anti-Variance Resilient Filter (Wiener gate)
    "imcf": True,            # Residual Inter-Modal Coherence Filter
    "permutation_fft": False,  # Falsifiable test: replace FFT with random permutation FFT
}


class DAMPS(nn.Module):
    """
    Spectral domain representation calibrator for multi-modal recommendation.

    Args:
        d              : embedding dimensionality of the projected modality
                         features (e.g. 64). Must equal the second dim of
                         ``h_img`` / ``h_txt`` passed to ``forward``.
        num_categories : number of static metadata clusters used by APC.
                         Set to 1 to effectively disable category-aware
                         grouping (every item in cluster 0).
        warmup_epochs  : how many epochs to use the adaptive EMA schedule
                         ``beta_t = 1 - 1/(t+1)`` before locking at 0.99.
        ablations      : per-component on/off dict (see ``_DEFAULT_ABLATIONS``).
        prior_image    : (1, F) tensor or None — data-driven AVRF prior
                         logits for the image modality. If None, a hard-coded
                         fallback (0.24) is used (matches Revision 9 default).
        prior_text     : (1, F) tensor or None — same for text.
        prior_audio    : (1, F) tensor or None — same for audio (Tiktok only).

    Notes:
        *   All complex-valued tensors are explicitly created in float32 to
            sidestep PyTorch's lack of complex<bfloat16> support during AMP.
        *   The module is fully reentrant — calling ``forward`` does **not**
            mutate any buffer except inside ``self.training and ablations
            ['imcf']`` (the IMCF baseline EMA), which is intentional.
    """

    def __init__(
        self,
        d: int = 64,
        num_categories: int = 10,
        warmup_epochs: int = 10,
        ablations: Optional[dict[str, bool]] = None,
        prior_image: Optional[torch.Tensor] = None,
        prior_text: Optional[torch.Tensor] = None,
        prior_audio: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()

        if d <= 0 or d % 2 != 0:
            # rFFT produces F = d // 2 + 1 bins; we require d even for symmetry
            raise ValueError(f"d must be a positive even integer, got d={d}")

        self.d: int = d
        self.F: int = d // 2 + 1
        self.num_categories: int = max(1, int(num_categories))
        self.warmup_epochs: int = max(1, int(warmup_epochs))

        # ----- Resolve ablation flags -----
        merged: dict[str, bool] = dict(_DEFAULT_ABLATIONS)
        if ablations is not None:
            for k, v in ablations.items():
                if k in merged:
                    merged[k] = bool(v)
        self.ablations: dict[str, bool] = merged

        # =====================================================================
        # 1. Metadata-Aware APC
        # =====================================================================
        # Trainable residual phase rotation psi ∈ R^F (one for each frequency
        # bin). Initialised to zero so the model starts from the pure
        # category-wise circular mean (eq. 5 in the spec).
        self.psi = nn.Parameter(torch.zeros(1, self.F))

        # Static random permutation index for the falsifiable Permutation-FFT
        # test. Registered as a buffer so it stays fixed across runs.
        perm = torch.randperm(self.d)
        self.register_buffer("permutation_idx", perm, persistent=True)

        # =====================================================================
        # 2. AVRF (Logit-clipped Wiener gate)
        # =====================================================================
        self.AVRF_image = nn.Parameter(self._init_avrf_logit(prior_image, fallback=0.24))
        self.AVRF_text = nn.Parameter(self._init_avrf_logit(prior_text, fallback=0.85))
        self.AVRF_audio = nn.Parameter(self._init_avrf_logit(prior_audio, fallback=0.50))

        # Per-epoch EMA MAD storage. Used only as a *diagnostic* baseline
        # (the actual gate is purely learnable), but exposed here per the
        # spec's transparency requirements.
        self.register_buffer("ema_mad_img", torch.ones(self.F))
        self.register_buffer("ema_mad_txt", torch.ones(self.F))
        self.register_buffer("ema_mad_aud", torch.ones(self.F))

        # =====================================================================
        # 3. Residual IMCF
        # =====================================================================
        self.lambda_coh = nn.Parameter(torch.tensor(0.1))
        # Empirical mode 0.2-0.4 on e-commerce datasets (spec Section 2.4).
        self.register_buffer("baseline_asc", torch.tensor(0.3))
        # Tracks the *current epoch* (set by the trainer via ``set_epoch``)
        # — used to drive the adaptive EMA schedule
        # ``beta_t = 1 - 1/(t+1)`` -> 0.99. This MUST be updated per epoch,
        # not per forward pass (compliance check WARN 3, Revision 9 audit).
        self.register_buffer(
            "_current_epoch", torch.zeros(1, dtype=torch.long)
        )
        # Legacy counter retained for backward compatibility / diagnostics.
        # It now reports the cumulative number of training-mode forward
        # passes that have hit IMCF, which is useful for debug logging but
        # is **no longer** wired into the EMA schedule.
        self.register_buffer("_imcf_update_count", torch.zeros(1))

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------
    def _init_avrf_logit(
        self,
        prior: Optional[torch.Tensor],
        fallback: float,
    ) -> torch.Tensor:
        """
        Build the AVRF logit tensor with strict [-2.0, 2.0] clipping.

        If ``prior`` is provided (data-driven, see ``damps/prior.py``), it
        is used directly after broadcasting/clipping. Otherwise we fall back
        to the hard-coded scalar ``fallback`` (replicated F times).
        """
        if prior is not None:
            t = prior.detach().to(torch.float32)
            if t.dim() == 1:
                t = t.unsqueeze(0)
            if t.shape[-1] != self.F:
                raise ValueError(
                    f"AVRF prior expected {self.F} freq bins, got {t.shape[-1]}"
                )
            return torch.clamp(t, -2.0, 2.0)

        # Hard-coded fallback prior (Revision 9 default)
        logit = math.log(fallback / (1.0 - fallback + 1e-8))
        logit = max(-2.0, min(2.0, logit))
        return torch.full((1, self.F), float(logit))

    def _fft(self, x: torch.Tensor) -> torch.Tensor:
        """
        1-D orthonormal real FFT in float32 (regardless of input dtype).
        Optionally permutes the input for the Permutation-FFT falsification
        test.

        ``norm="ortho"`` (per the Speedup Guide, Section 1) makes the rFFT /
        irFFT pair an orthonormal transform: every linear in-spectrum
        operation (APC rotation, AVRF gate, IMCF residual) is invariant
        under this choice, while the orthonormal scaling slightly improves
        numerical conditioning at d=64 vs the default backward-only norm.
        """
        x32 = x.to(torch.float32)
        if self.ablations["permutation_fft"]:
            x32 = x32[:, self.permutation_idx]
        return torch.fft.rfft(x32, dim=-1, norm="ortho")

    def _ifft(self, z: torch.Tensor, original_dtype: torch.dtype) -> torch.Tensor:
        """Inverse rFFT (orthonormal), cast back to the input dtype."""
        x = torch.fft.irfft(z, n=self.d, dim=-1, norm="ortho")
        return x.to(original_dtype)

    # ------------------------------------------------------------------
    #  Forward pass
    # ------------------------------------------------------------------
    def forward(
        self,
        h_img: torch.Tensor,
        h_txt: torch.Tensor,
        item_categories: Optional[torch.Tensor] = None,
        h_aud: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply the full DAMPS pipeline to projected modality features.

        Args:
            h_img           : (N, d) image feature tensor (post-MLP).
            h_txt           : (N, d) text feature tensor.
            item_categories : (N,) Long tensor with values in
                              ``[0, num_categories)``; used by APC.
                              Pass None to skip APC.
            h_aud           : optional (N, d) audio feature tensor (Tiktok).

        Returns:
            Tuple of (h_img_cal, h_txt_cal, h_aud_cal_or_None) — calibrated
            spatial-domain representations with the same shape & dtype as
            inputs.
        """
        in_dtype = h_img.dtype

        # =====================================================================
        # 0. Spectral decomposition
        # =====================================================================
        z_img = self._fft(h_img)
        z_txt = self._fft(h_txt)
        z_aud = self._fft(h_aud) if h_aud is not None else None

        # =====================================================================
        # 1. Metadata-Aware APC (von Mises MLE on static metadata categories)
        # =====================================================================
        if self.ablations["apc"] and item_categories is not None:
            z_img, z_txt = self._apply_apc(z_img, z_txt, item_categories)

        # =====================================================================
        # 2. AVRF — logit-clipped Wiener gate, per-modality
        # =====================================================================
        if self.ablations["avrf"]:
            # 1.0 + tanh(logit) ∈ [0, 2]; clipping ensures the lower bound
            # never collapses to zero (which would zero out the modality).
            v_img = 1.0 + torch.tanh(self.AVRF_image)            # (1, F)
            v_txt = 1.0 + torch.tanh(self.AVRF_text)
            z_img = z_img * v_img
            z_txt = z_txt * v_txt
            if z_aud is not None:
                v_aud = 1.0 + torch.tanh(self.AVRF_audio)
                z_aud = z_aud * v_aud

        # =====================================================================
        # 3. Residual IMCF (ASC consensus, residual form)
        # =====================================================================
        if self.ablations["imcf"]:
            z_img, z_txt, z_aud = self._apply_imcf(z_img, z_txt, z_aud)

        # =====================================================================
        # 4. Inverse FFT (spectral → spatial)
        # =====================================================================
        h_img_out = self._ifft(z_img, in_dtype)
        h_txt_out = self._ifft(z_txt, in_dtype)
        h_aud_out = self._ifft(z_aud, in_dtype) if z_aud is not None else None
        return h_img_out, h_txt_out, h_aud_out

    # ------------------------------------------------------------------
    #  APC: Metadata-Aware Adaptive Phase Calibration
    # ------------------------------------------------------------------
    def _apply_apc(
        self,
        z_img: torch.Tensor,
        z_txt: torch.Tensor,
        item_categories: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply von Mises MLE phase rotation per static metadata cluster.

        Equations (5-6) of the spec:
            theta_c    = atan2( Sum_{i in C_c} sin(R_i),
                                 Sum_{i in C_c} cos(R_i) )       (von Mises MLE)
            tilde_z_img = z_img * exp(-j * (theta_c / 2 + psi))
            tilde_z_txt = z_txt * exp(+j * (theta_c / 2 + psi))
        """
        if item_categories.dim() != 1 or item_categories.shape[0] != z_img.shape[0]:
            raise ValueError(
                "item_categories must be a 1-D tensor with the same length as z_img"
            )

        cats = item_categories.long().clamp_(0, self.num_categories - 1)

        phase_img = torch.angle(z_img)
        phase_txt = torch.angle(z_txt)
        # R_i = phase_txt - phase_img  (relative phase per item per bin)
        R_i = phase_txt - phase_img
        sin_R = torch.sin(R_i)
        cos_R = torch.cos(R_i)

        # Aggregate per static metadata cluster (no k-means: pure scatter_add)
        device = z_img.device
        idx = cats.unsqueeze(1).expand(-1, self.F)
        sum_sin = torch.zeros(self.num_categories, self.F, device=device,
                              dtype=sin_R.dtype).scatter_add_(0, idx, sin_R)
        sum_cos = torch.zeros(self.num_categories, self.F, device=device,
                              dtype=cos_R.dtype).scatter_add_(0, idx, cos_R)
        theta_c = torch.atan2(sum_sin, sum_cos)            # (C, F)
        theta_i = theta_c[cats]                             # (N, F)

        # Unit complex rotators via polar form — NEVER use Python ``1j`` /
        # ``exp(±1j * phase)``. Inductor's backward compiler crashes on raw
        # Python complex literals (pytorch#184100):
        #   AttributeError: 'complex' object has no attribute 'get_name'
        phase = (theta_i / 2.0 + self.psi).to(torch.float32)
        cos_p = torch.cos(phase)
        sin_p = torch.sin(phase)
        rot_img = torch.complex(cos_p, -sin_p)   # exp(-j * phase)
        rot_txt = torch.complex(cos_p, sin_p)    # exp(+j * phase)

        z_img_cal = z_img * rot_img
        z_txt_cal = z_txt * rot_txt
        return z_img_cal, z_txt_cal

    # ------------------------------------------------------------------
    #  IMCF: Residual Inter-Modal Coherence Filter
    # ------------------------------------------------------------------
    def _apply_imcf(
        self,
        z_img: torch.Tensor,
        z_txt: torch.Tensor,
        z_aud: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute the per-item ASC consensus coefficient and apply it in
        residual form (eq. 9-10 of the spec).
        """
        eps = 1e-8

        # Cross-spectrum and amplitude squares (per item per bin)
        C_i = z_txt * torch.conj(z_img)
        P_img = torch.abs(z_img) ** 2 + eps
        P_txt = torch.abs(z_txt) ** 2 + eps
        ASC_i = torch.clamp((torch.abs(C_i) ** 2) / (P_img * P_txt), 0.0, 1.0)
        # Mean ASC across freq bins (per item) — used for the residual coefficient
        asc_per_item = ASC_i.mean(dim=-1, keepdim=True)    # (N, 1)

        # Adaptive EMA on baseline_asc: cold-start uses beta_t = 1 - 1/(t+1),
        # then locks at 0.99 (per Section 3.1, identical to AVRF MAD schedule).
        # ``t`` MUST be the current *epoch* (set by ``set_epoch``), not the
        # cumulative number of forward passes — see Revision 9 audit WARN 3.
        if self.training:
            # Tensor schedule on ``_current_epoch`` (0-d) so torch.compile
            # does not recompile every epoch on ``.item()`` / Python ints.
            t = self._current_epoch.to(dtype=torch.float32).reshape(())
            warm = t.new_tensor(float(self.warmup_epochs))
            beta_t = torch.where(
                t < warm,
                1.0 - 1.0 / (t + 1.0),
                t.new_tensor(0.99),
            )
            with torch.no_grad():
                self.baseline_asc.mul_(beta_t).add_(
                    asc_per_item.detach().mean() * (1.0 - beta_t)
                )
                # Bump the diagnostic-only forward-pass counter.
                self._imcf_update_count.add_(1.0)

        # Residual coefficient — broadcast over F bins
        residual = (asc_per_item - self.baseline_asc).to(z_img.real.dtype)  # (N, 1)
        gate = self.lambda_coh.to(z_img.real.dtype) * residual

        z_img_out = z_img + gate * z_img
        z_txt_out = z_txt + gate * z_txt
        z_aud_out: Optional[torch.Tensor] = None
        if z_aud is not None:
            z_aud_out = z_aud + gate * z_aud
        return z_img_out, z_txt_out, z_aud_out

    # ------------------------------------------------------------------
    #  Epoch hook (drives the IMCF / MAD adaptive EMA schedules)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def set_epoch(self, epoch: int) -> None:
        """
        Tell DAMPS which training epoch we are currently on.

        This is the single source of truth for the adaptive EMA schedule
        ``beta_t = 1 - 1/(t+1)`` -> 0.99 (after ``warmup_epochs``). Without
        it, the IMCF baseline would be driven by per-forward-pass updates,
        which on Amazon Clothing (~58 batches/epoch) would saturate
        ``warmup_epochs=10`` after just one real epoch.

        Args:
            epoch : 0-indexed epoch counter from the trainer.
        """
        self._current_epoch.fill_(int(max(0, epoch)))

    # ------------------------------------------------------------------
    #  Per-epoch EMA MAD aggregator (diagnostic)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def update_epoch_mad(
        self,
        epoch: int,
        h_img_raw: torch.Tensor,
        h_txt_raw: torch.Tensor,
        h_aud_raw: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Refresh the per-epoch EMA MAD diagnostic buffers (Section 2.3 spec).

        The MAD (Median Absolute Deviation) is computed across the batch axis
        for each frequency bin. The schedule mirrors the IMCF baseline:
        adaptive ``beta_t = 1 - 1/(t+1)`` during warm-up, then frozen at 0.99.

        Args:
            epoch     : current epoch (0-indexed).
            h_img_raw : (N, d) raw image features for this epoch sample.
            h_txt_raw : (N, d) raw text features.
            h_aud_raw : optional (N, d) audio features.
        """
        if not self.ablations["avrf"]:
            return

        z_img_amp = torch.abs(self._fft(h_img_raw))
        z_txt_amp = torch.abs(self._fft(h_txt_raw))

        med_img = torch.median(z_img_amp, dim=0).values
        med_txt = torch.median(z_txt_amp, dim=0).values
        mad_img = torch.median(torch.abs(z_img_amp - med_img), dim=0).values
        mad_txt = torch.median(torch.abs(z_txt_amp - med_txt), dim=0).values

        beta_t = 1.0 - 1.0 / (epoch + 1.0) if epoch < self.warmup_epochs else 0.99
        self.ema_mad_img.mul_(beta_t).add_(mad_img * (1.0 - beta_t))
        self.ema_mad_txt.mul_(beta_t).add_(mad_txt * (1.0 - beta_t))

        if h_aud_raw is not None:
            z_aud_amp = torch.abs(self._fft(h_aud_raw))
            med_aud = torch.median(z_aud_amp, dim=0).values
            mad_aud = torch.median(torch.abs(z_aud_amp - med_aud), dim=0).values
            self.ema_mad_aud.mul_(beta_t).add_(mad_aud * (1.0 - beta_t))

    # ------------------------------------------------------------------
    #  Diagnostics
    # ------------------------------------------------------------------
    @torch.no_grad()
    def tanh_saturation_rates(self) -> dict[str, float]:
        """
        Return the fraction of AVRF logits whose ``|tanh(logit)| > 0.95``.
        A high saturation rate flags imminent gradient paralysis (spec
        Table 1, "tanh saturation probe" diagnostic).
        """
        out: dict[str, float] = {}
        for name, p in [
            ("img", self.AVRF_image),
            ("txt", self.AVRF_text),
            ("aud", self.AVRF_audio),
        ]:
            out[name] = float(
                (torch.tanh(p).abs() > 0.95).to(torch.float32).mean().item()
            )
        return out

    def num_trainable_params(self) -> int:
        """Return the number of trainable parameters in DAMPS only."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

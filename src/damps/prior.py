"""
damps/prior.py — Data-Driven AVRF Prior Derivation
====================================================

Implements the **automatic SNR-based prior** referenced in Section 2.3 of the
DAMPS revision (``DAMPS_to_MMHCL_architecture_revision42.tex``):

    "Initialization Safeguard: At frequency bins with high SNR, native logit
     initialization can spike and saturate the tanh function. To rectify this,
     the parameter is strictly clipped within the [-2.0, 2.0] range, integrated
     with a data-driven automatic prior derivation algorithm based on the
     signal-to-noise ratio, replacing hard-coded values."

Mathematical formulation
------------------------
Given raw multi-modal feature matrices ``X_m \\in R^{N x d}``  for each
modality ``m`` (image / text / audio), we estimate a per-frequency-bin signal
ratio:

    SNR_f^m  =  Var_inter[|FFT(X_m)|_f] / (Var_intra[|FFT(X_m)|_f] + eps)

The Wiener-style shrinkage prior is then

    p_f^m  =  SNR_f^m / (SNR_f^m + 1)         in (0, 1)

This prior is what the AVRF logit parameter is initialized to track
``logit(p) = log(p / (1 - p))`` and then **strictly clipped to [-2.0, 2.0]**
to avert tanh saturation at warm-up.

Why this matters
----------------
Without data-driven priors, the AVRF logits are hard-coded to e.g. ``0.24``
for image and ``0.85`` for text — values that were tuned for one specific
benchmark. On a new dataset (e.g. Tiktok with audio, or a new Amazon vertical),
those constants either over-attenuate the signal or fail to denoise at all.
This module computes the prior **once per dataset** at model construction
time, with O(N * d) cost (a single FFT pass).

Public API
----------
``compute_avrf_prior(features: torch.Tensor, eps: float = 1e-8) -> torch.Tensor``
    Returns a ``(F,)`` tensor of priors in ``(0, 1)`` where ``F = d // 2 + 1``.

``compute_avrf_logit(features: torch.Tensor, clip: float = 2.0) -> torch.Tensor``
    Returns the **clipped logit** ready to be wrapped in ``nn.Parameter``.
"""

from __future__ import annotations

import torch


def compute_avrf_prior(
    features: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Estimate the per-frequency-bin Wiener prior p_f from raw modality features.

    Args:
        features : (N, d) float tensor — raw feature vectors of N items.
                   Pass image_feat, text_feat, or audio_feat *before* the
                   MLP projection.
        eps      : numerical stabilisation floor (default 1e-8).

    Returns:
        (F,) float tensor in (0, 1) where ``F = d // 2 + 1``.
        Each value is the Wiener shrinkage prior at frequency bin f.
    """
    if features.dim() != 2:
        raise ValueError(
            f"features must be 2-D (N, d); got shape {tuple(features.shape)}"
        )

    # Cast to float32 to avoid FFT precision loss in bfloat16
    x = features.detach().to(torch.float32)

    # Spectral magnitude per item per frequency bin: |FFT(x_i)|_f.
    # ``norm="ortho"`` matches the orthonormal rFFT used in damps/core.py
    # (Speedup Guide Section 1). The Wiener prior below is a ratio of
    # variances, so the result is invariant to the choice of norm; we use
    # the orthonormal form for code consistency with the calibrator.
    z_amp = torch.abs(torch.fft.rfft(x, dim=-1, norm="ortho"))   # (N, F)

    # Inter-item variance at each frequency bin (signal energy)
    var_inter = z_amp.var(dim=0, unbiased=False)   # (F,)

    # Intra-item variance: how spread out a single item's spectrum is
    # (proxy for noise power that is uncorrelated across frequency bins)
    var_intra = z_amp.var(dim=1, unbiased=False).mean()  # scalar

    snr = var_inter / (var_intra + eps)            # (F,)
    prior = snr / (snr + 1.0)                      # (F,) in (0, 1)
    return prior


def compute_avrf_logit(
    features: torch.Tensor,
    clip: float = 2.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Convert the data-driven prior into a **clipped logit** suitable for
    ``nn.Parameter`` initialisation of the AVRF gate.

    Mathematical pipeline:
        prior      = SNR / (SNR + 1)              in (0, 1)
        logit_raw  = log( prior / (1 - prior) )   in R
        logit_clip = clip(logit_raw, -clip, +clip)

    The strict clipping (Section 2.3 of the DAMPS spec) prevents the tanh
    activation in AVRF from saturating immediately at warm-up, which would
    otherwise paralyse gradient flow ("cold-start phenomenon").

    Args:
        features : (N, d) raw modality feature tensor.
        clip     : symmetric clipping threshold (default 2.0).
        eps      : numerical stabilisation floor.

    Returns:
        (1, F) float tensor — ready to wrap with ``nn.Parameter``.
    """
    prior = compute_avrf_prior(features, eps=eps)
    logit = torch.log((prior + eps) / (1.0 - prior + eps))
    logit = torch.clamp(logit, -clip, +clip)
    return logit.unsqueeze(0)                      # (1, F)

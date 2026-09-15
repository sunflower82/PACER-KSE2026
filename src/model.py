"""
model.py — DAMPS-MMHCL Neural Network
========================================

Implements the full **DAMPS-MMHCL** model that integrates the spectral
calibration block (``damps/core.py``) into the original MMHCL backbone via
**Soft Residual-Routing** (eq. 3 of the spec).

Forward pipeline (per epoch)
----------------------------
::

    raw_image, raw_text                 # (N, image_dim), (N, text_dim)
        │
        ▼  per-modality MLP projection
    h_img_raw, h_txt_raw                # (N, d=64)
        │
        ▼  DAMPS spectral calibration (FFT → APC → AVRF → IMCF → IFFT)
    h_img_cal, h_txt_cal                # (N, d)
        │
        ▼  Soft Residual-Routing (eq. 3)
    h_img_input  = h_img_raw + α_img · LayerNorm(h_img_cal)
    h_txt_input  = h_txt_raw + α_txt · LayerNorm(h_txt_cal)
        │
        ▼  hypergraph convolution (uses I2I_mat from Pattern B' rebuild)
    ii_emb                              # (N, d)
        │
        ▼  fuse with CF view (LightGCN/NGCF/MF on UI graph) + UU view
    final_user_emb, final_item_emb      # (n_users, d), (n_items, d)

Compared with the original MMHCL the only architectural deltas are:

1.  An **additional DAMPS pre-pass** before hypergraph propagation.
2.  A **Soft Residual-Routing** branch that mixes raw + calibrated features
    to evade the double over-smoothing of GCN stacks.
3.  An InfoNCE temperature ``τ`` that can be either:
    * **Learnable** (Revision 9 / rev42 baseline; ``τ`` registered as
      ``nn.Parameter``, initialised at 0.1).
    * **Static** (Revision 11 / rev44 Phase 1 Quick Win; ``τ`` registered as
      a non-trainable buffer, anchor value 0.3, sweep set
      ``{0.2, 0.3, 0.5}``). Empirically the learnable τ's gradient vanishes
      and gets stuck at ~0.0909 across all 10 seeds, triggering an embedding
      collapse and a 10.7 % Recall@20 deficit relative to the MMHCL paper.
      Phase 1 fixes this by pinning τ to 0.3.
4.  A **Slim Momentum Encoder** that lives outside the autograd graph and
    feeds the Pattern B' rebuild (see ``train.py``).

Everything else — UI bipartite graph, U2U co-interaction graph, BPR loss,
contrastive loss — is identical to the original MMHCL.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from damps import DAMPS, SlimMomentumEncoder, compute_avrf_logit


# ===========================================================================
#  AMP-safe sparse matmul
# ===========================================================================
def _safe_sparse_mm(sparse_mat: torch.Tensor, dense: torch.Tensor) -> torch.Tensor:
    """
    Autocast-safe wrapper around ``torch.sparse.mm``.

    Background
    ----------
    PyTorch's CUDA sparse matmul kernel (``addmm_sparse_cuda``) is currently
    only implemented for ``float32`` / ``float64``; calling it with a
    ``BFloat16`` dense operand raises::

        NotImplementedError: "addmm_sparse_cuda" not implemented for 'BFloat16'

    With ``--use_amp 1`` (Speedup Guide Section 3) the surrounding
    ``torch.autocast(dtype=torch.bfloat16)`` region casts the LightGCN /
    hypergraph embeddings to bfloat16, which then crashes inside
    ``torch.sparse.mm``. This helper:

    1. Disables autocast for the matmul (so the result is not re-autocast).
    2. Promotes both operands to ``float32`` if the dense input is bf16/fp16.
    3. Runs the sparse mm.
    4. Casts the result back to the dense input's original dtype, so the
       caller's pipeline (e.g. residual adds, ``F.normalize``) continues
       to operate in the autocast dtype as intended.

    The sparse-matrix operands here (``UI_mat``, ``U2U_mat``,
    ``Item_mat`` / ``I2I_mat``) are tiny relative to the dense embeddings,
    so the temporary fp32 promotion has negligible memory cost.
    """
    orig_dtype = dense.dtype
    needs_promote = orig_dtype in (torch.bfloat16, torch.float16)
    # Always disable autocast inside this region so the cast we do is
    # respected and the result dtype is deterministic.
    with torch.amp.autocast(device_type=dense.device.type, enabled=False):
        d = dense.float() if needs_promote else dense
        s = sparse_mat
        # Pattern B' rebuild / eval may hand us inference-mode sparse
        # mats; autograd rejects those in torch.sparse.mm.
        if torch.is_inference(s):
            s = s.clone()
        if needs_promote and s.dtype in (torch.bfloat16, torch.float16):
            s = s.float()
        out = torch.sparse.mm(s, d)
    return out.to(orig_dtype) if needs_promote else out


# ===========================================================================
#  Modality MLP projection head
# ===========================================================================
class ModalityProjection(nn.Module):
    """
    Single-layer projection ``h_m = W_m x_m + b_m``  (eq. 4 of the spec).

    Mirrors the original MMHCL projection layer.
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=True)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


# ===========================================================================
#  DAMPS_MMHCL — main model
# ===========================================================================
class DAMPS_MMHCL(nn.Module):
    """
    Full DAMPS-MMHCL recommender.

    Args:
        n_users               : number of users.
        n_items               : number of items.
        embedding_dim         : embedding dim ``d`` (default 64).
        image_feats           : (n_items, image_dim) tensor or None.
        text_feats            : (n_items, text_dim) tensor or None.
        audio_feats           : optional (n_items, audio_dim) tensor (Tiktok).
        ablations             : per-component on/off dict
                                (apc, avrf, imcf, soft_routing, momentum,
                                permutation_fft).
        cf_model              : 'LightGCN' (default), 'NGCF', or 'MF'.
        ui_layers, user_layers, item_layers : GNN depths.
        weight_size           : NGCF per-layer sizes (only used when
                                ``cf_model == 'NGCF'``).
        item_loss_ratio       : weight for item-side contrastive loss.
        user_loss_ratio       : weight for user-side contrastive loss.
        temperature_init      : value for τ. When ``learnable_tau=True`` this
                                is the initialisation of ``nn.Parameter``; when
                                ``learnable_tau=False`` it is the fixed anchor
                                value used throughout training. Phase 1 (rev44)
                                anchor: 0.3.
        learnable_tau         : if True, register τ as ``nn.Parameter``
                                (Revision 9 / rev42 behaviour, default 0.1).
                                If False, register τ as a non-trainable buffer
                                (Revision 11 / rev44 Phase 1, anchor 0.3).
                                The default below is False to match the
                                rev44 Phase 1 recommended config.
        warmup_epochs         : warm-up window for the EMA schedules.
        damps_num_categories  : number of static metadata clusters.
        data_driven_prior     : if True, derive AVRF priors from raw features.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 64,
        image_feats: Optional[torch.Tensor] = None,
        text_feats: Optional[torch.Tensor] = None,
        audio_feats: Optional[torch.Tensor] = None,
        ablations: Optional[dict[str, bool]] = None,
        cf_model: str = "LightGCN",
        ui_layers: int = 2,
        user_layers: int = 3,
        item_layers: int = 2,
        weight_size: Optional[list[int]] = None,
        item_loss_ratio: float = 0.07,
        user_loss_ratio: float = 0.03,
        temperature_init: float = 0.3,
        learnable_tau: bool = False,
        warmup_epochs: int = 10,
        damps_num_categories: int = 10,
        data_driven_prior: bool = True,
        enable_logq: bool = False,            # rev53 §3.1 — variant "h"
        logq_scale: float = 1.0,              # multiplier on log_q before subtraction
        logq_clip: float = 5.0,               # symmetric clip on scale*log_q
        # --- Wave 2 / M1 -- SimGCL view-invariance (Yu et al. SIGIR 2022) ---
        enable_simgcl: bool = False,          # toggles the third contrastive term
        simgcl_eps: float = 0.1,              # noise magnitude (rev54 default 0.1)
        simgcl_batch_size_user: int = 4096,   # row-chunk for user-branch L_view
        simgcl_batch_size_item: int = 4096,   # row-chunk for item-branch L_view
        # ---- Branch A (rev55 §8.1) -- speedup levers ----
        branchA_view_every_k: int = 2,
        branchA_bcl_batchn: bool = True,
        branchA_view_bsz: int = 2048,
        branchA_bcl_bsz: int = 2048,
        # ---- Branch A' / NRDMC-lite (rev55 §8.2) -- learnable view generators ----
        enable_nrdmc_lite: bool = False,
        nrdmc_lite_layers: int = 2,
        # ---- Branch A' / P3 (rev56) -- Prototype-Aware View (PTV) ----
        enable_ptv: bool = False,
        n_prototypes: int = 32,
        lambda_ptv: float = 1.0,
        # ---- Branch A' / P4 (rev57) -- ASC gate reparameterization ----
        # ``asc_gate_mode``: {"raw", "sigmoid", "tanh_signed", "tanh01"}.
        # "raw" reproduces the original rev55/rev56 behaviour (alpha = theta,
        # unconstrained). The other modes constrain the effective gate and
        # are the P4 fix for the alpha_img collapse observed in the P3 PTV
        # sweep logs (alpha_img: +0.09 -> -0.68 across 75 epochs).
        asc_gate_mode: str = "raw",
    ) -> None:
        super().__init__()

        # ------------------------------------------------------------------
        # Config
        # ------------------------------------------------------------------
        self.n_users: int = n_users
        self.n_items: int = n_items
        self.embedding_dim: int = embedding_dim
        self.cf_model: str = cf_model
        self.ui_layers: int = ui_layers
        self.user_layers: int = user_layers
        self.item_layers: int = item_layers
        self.weight_size: list[int] = weight_size or [embedding_dim] * (ui_layers + 1)
        self.item_loss_ratio: float = item_loss_ratio
        self.user_loss_ratio: float = user_loss_ratio

        self.has_audio: bool = audio_feats is not None

        # Default ablation switches (mirrors the spec's full lock-in defaults)
        defaults: dict[str, bool] = {
            "apc": True,
            "avrf": True,
            "imcf": True,
            "permutation_fft": False,
            "soft_routing": True,
            "momentum": True,
        }
        if ablations:
            defaults.update({k: bool(v) for k, v in ablations.items()})
        self.ablations: dict[str, bool] = defaults

        # ------------------------------------------------------------------
        # 1. Learnable embedding tables (CF + hypergraph branches)
        # ------------------------------------------------------------------
        self.user_ui_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_ui_embedding = nn.Embedding(n_items, embedding_dim)
        self.uu_embedding = nn.Embedding(n_users, embedding_dim)
        self.ii_embedding = nn.Embedding(n_items, embedding_dim)
        for emb in [
            self.user_ui_embedding,
            self.item_ui_embedding,
            self.uu_embedding,
            self.ii_embedding,
        ]:
            nn.init.xavier_uniform_(emb.weight)

        # ------------------------------------------------------------------
        # 2. Optional NGCF transformation layers
        # ------------------------------------------------------------------
        self.GC_Linear_list = nn.ModuleList()
        self.Bi_Linear_list = nn.ModuleList()
        self.dropout_list = nn.ModuleList()
        if cf_model == "NGCF":
            for i in range(ui_layers):
                self.GC_Linear_list.append(
                    nn.Linear(self.weight_size[i], self.weight_size[i + 1])
                )
                self.Bi_Linear_list.append(
                    nn.Linear(self.weight_size[i], self.weight_size[i + 1])
                )
                self.dropout_list.append(nn.Dropout(0.1))

        # ------------------------------------------------------------------
        # 3. Modality projection MLPs (raw → d=64)
        # ------------------------------------------------------------------
        self.image_dim: Optional[int] = (
            image_feats.shape[-1] if image_feats is not None else None
        )
        self.text_dim: Optional[int] = (
            text_feats.shape[-1] if text_feats is not None else None
        )
        self.audio_dim: Optional[int] = (
            audio_feats.shape[-1] if audio_feats is not None else None
        )

        if self.image_dim is None or self.text_dim is None:
            raise RuntimeError(
                "DAMPS-MMHCL requires both image and text modalities. "
                "Please make sure image_feat.npy / text_feat.npy are present."
            )

        self.image_proj = ModalityProjection(self.image_dim, embedding_dim)
        self.text_proj = ModalityProjection(self.text_dim, embedding_dim)
        self.audio_proj: Optional[ModalityProjection] = (
            ModalityProjection(self.audio_dim, embedding_dim)
            if self.has_audio else None
        )

        # ------------------------------------------------------------------
        # 4. Cache raw feature buffers (read-only, for DAMPS forward pass)
        # ------------------------------------------------------------------
        self.register_buffer("raw_image", image_feats.float(), persistent=False)
        self.register_buffer("raw_text", text_feats.float(), persistent=False)
        if self.has_audio:
            self.register_buffer("raw_audio", audio_feats.float(), persistent=False)

        # ------------------------------------------------------------------
        # 5. Static metadata categories for APC (registered later via setter)
        # ------------------------------------------------------------------
        self.register_buffer(
            "meta_categories",
            torch.zeros(n_items, dtype=torch.long),
            persistent=False,
        )

        # ------------------------------------------------------------------
        # 6. DAMPS spectral calibrator
        # ------------------------------------------------------------------
        prior_image: Optional[torch.Tensor] = None
        prior_text: Optional[torch.Tensor] = None
        prior_audio: Optional[torch.Tensor] = None
        if data_driven_prior:
            # Project raw features to d=64 before computing the prior so the
            # spectral statistics match the actual DAMPS input distribution.
            #
            # Device alignment: ``image_feats`` / ``text_feats`` may already be
            # on a CUDA device because the trainer pre-moves them BEFORE
            # constructing the model (see train.py: feats.to(self.device) ->
            # DAMPS_MMHCL(...).to(self.device)). The projection MLPs above
            # were just instantiated with ``nn.Linear`` and live on CPU, so
            # we must move them to the inputs' device before calling them.
            # The trainer's outer ``.to(self.device)`` after __init__ is then
            # a no-op for these submodules.
            target_device = image_feats.device
            self.image_proj.to(target_device)
            self.text_proj.to(target_device)
            if self.has_audio and self.audio_proj is not None:
                self.audio_proj.to(target_device)
            with torch.no_grad():
                proj_img = self.image_proj(image_feats.float())
                proj_txt = self.text_proj(text_feats.float())
                prior_image = compute_avrf_logit(proj_img)
                prior_text = compute_avrf_logit(proj_txt)
                if self.has_audio and audio_feats is not None:
                    proj_aud = self.audio_proj(audio_feats.float())   # type: ignore[union-attr]
                    prior_audio = compute_avrf_logit(proj_aud)

        damps_ablations = {
            "apc": self.ablations["apc"],
            "avrf": self.ablations["avrf"],
            "imcf": self.ablations["imcf"],
            "permutation_fft": self.ablations["permutation_fft"],
        }
        self.damps = DAMPS(
            d=embedding_dim,
            num_categories=damps_num_categories,
            warmup_epochs=warmup_epochs,
            ablations=damps_ablations,
            prior_image=prior_image,
            prior_text=prior_text,
            prior_audio=prior_audio,
        )

        # ------------------------------------------------------------------
        # 7. Soft Residual-Routing (eq. 3 of the spec)
        # ------------------------------------------------------------------
        # rev57 / P4 -- ASC gate reparameterization.
        #
        # The scalar residual gate ``alpha_v`` (v in {img, txt, aud}) was
        # originally a raw ``nn.Parameter(torch.tensor(0.1))`` used directly
        # in ``h_raw + alpha * ln(h_cal)`` (see ``_soft_route``). Diagnostic
        # logs from the P3 PTV sweep (2 seeds x 100 epochs, Amazon Clothing)
        # show alpha_img drifting from +0.09 (epoch 0) to -0.68 (epoch 75),
        # meaning the model learned to *subtract* the image branch. That
        # collapses multimodal fusion well before the val-recall peak.
        #
        # rev57 P4 introduces four ``asc_gate_mode`` variants; the raw one is
        # the identity function so the flag is fully backward-compatible.
        # ``theta`` is the underlying ``nn.Parameter`` and ``alpha`` denotes
        # the *effective* gate consumed by ``_soft_route``.
        #
        #   "raw"          alpha = theta                  in (-inf, +inf)
        #   "sigmoid"      alpha = sigmoid(theta)         in (0, 1)
        #   "tanh_signed"  alpha = tanh(theta)            in (-1, 1)
        #   "tanh01"       alpha = 0.5*(tanh(theta)+1)    in (0, 1)
        #
        # ``theta`` is initialised so that ``alpha(theta_init) == 0.1`` in
        # every mode -- so the very first forward pass reproduces the rev55
        # residual-routing magnitude regardless of mode.
        self.asc_gate_mode: str = str(asc_gate_mode).lower()
        _valid_modes = {"raw", "sigmoid", "tanh_signed", "tanh01"}
        if self.asc_gate_mode not in _valid_modes:
            raise ValueError(
                f"asc_gate_mode={asc_gate_mode!r} not in {sorted(_valid_modes)}"
            )
        _target_alpha_init: float = 0.1
        if self.asc_gate_mode == "raw":
            _theta_init = _target_alpha_init
        elif self.asc_gate_mode == "sigmoid":
            # sigmoid(theta_init) == 0.1  =>  theta_init = logit(0.1)
            _theta_init = math.log(
                _target_alpha_init / (1.0 - _target_alpha_init)
            )
        elif self.asc_gate_mode == "tanh_signed":
            # tanh(theta_init) == 0.1
            _theta_init = math.atanh(_target_alpha_init)
        else:  # "tanh01"
            # 0.5*(tanh(theta_init)+1) == 0.1  =>  tanh(theta_init) == -0.8
            _theta_init = math.atanh(2.0 * _target_alpha_init - 1.0)
        self.alpha_img = nn.Parameter(torch.tensor(float(_theta_init)))
        self.alpha_txt = nn.Parameter(torch.tensor(float(_theta_init)))
        self.alpha_aud = nn.Parameter(torch.tensor(float(_theta_init)))
        self.ln_img = nn.LayerNorm(embedding_dim)
        self.ln_txt = nn.LayerNorm(embedding_dim)
        self.ln_aud = nn.LayerNorm(embedding_dim) if self.has_audio else None

        # ------------------------------------------------------------------
        # 8. Slim Momentum Encoder (eats h_cal_*, drives Pattern B' rebuild)
        # ------------------------------------------------------------------
        self.momentum = SlimMomentumEncoder(
            num_items=n_items,
            dim=embedding_dim,
            warmup_epochs=warmup_epochs,
            use_ema=self.ablations["momentum"],
            num_modalities=3 if self.has_audio else 2,
        )

        # ------------------------------------------------------------------
        # 9. InfoNCE temperature τ
        # ------------------------------------------------------------------
        # Revision 9 (rev42) made τ a learnable ``nn.Parameter`` initialised
        # at 0.1. Empirical analysis on Amazon Clothing across 10 seeds shows
        # the gradient of τ vanishes and the value gets pinned to ~0.0909,
        # which collapses the embedding space and causes a 10.7 % Recall@20
        # deficit. Revision 11 (rev44) Phase 1 -- Quick Win -- pivots to a
        # **static τ sweep** anchored at τ = 0.3 (sweep set {0.2, 0.3, 0.5})
        # to break the collapse.
        #
        # ``learnable_tau`` toggles between the two regimes:
        #   * True  -> nn.Parameter (rev42 baseline reproduction).
        #   * False -> register_buffer (rev44 Phase 1 default).
        #
        # ``batched_contrastive_loss`` clamps τ to >= 0.01 from below in both
        # cases to prevent division blow-ups.
        # ------------------------------------------------------------------
        self.learnable_tau: bool = bool(learnable_tau)
        tau_tensor = torch.tensor(float(temperature_init))
        if self.learnable_tau:
            self.tau = nn.Parameter(tau_tensor)
        else:
            self.register_buffer("tau", tau_tensor)

        # ------------------------------------------------------------------
        # 10. LogQ correction state (rev53 §3.1, eq. 1 — variant "h")
        # ------------------------------------------------------------------
        # Toggles the popularity-corrected InfoNCE proposed by Yi et al.
        # (RecSys 2019). When enable_logq=True, batched_contrastive_loss
        # subtracts ``logq_scale * clip(log_q[j], -logq_clip, +logq_clip)``
        # from per-column logits BEFORE dividing by τ and taking exp.
        #
        # The log_q buffer must be populated via set_log_q(...) before the
        # first training step; an uninitialised (all-zero) buffer triggers
        # a fail-fast inside batched_contrastive_loss to avoid silently
        # disabling the correction (rev53 §3.1, line 104).
        #
        # logq_scale and logq_clip default to (1.0, 5.0) which matches the
        # rev53 spec literally. For τ=0.3 and cosine sim ∈ [-1,1], expect
        # log_q ∈ [-12, -4] at Amazon Clothing scale, so the unscaled
        # subtraction will dominate the τ-normalised logits. The first
        # sanity sweep MUST cover logq_scale ∈ {0.05, 0.1, 0.3, 1.0} on a
        # subset of seeds before locking the spec default — see the M1.5
        # protocol in the LogQ README.
        # ------------------------------------------------------------------
        self.enable_logq: bool = bool(enable_logq)
        self.logq_scale: float = float(logq_scale)
        self.logq_clip: float = float(logq_clip)
        self.register_buffer("log_q", torch.zeros(n_items, dtype=torch.float32))

        # ------------------------------------------------------------------
        # 11. Wave 2 / M1 -- SimGCL view-invariance (Yu et al. SIGIR 2022)
        # ------------------------------------------------------------------
        # When enable_simgcl=True, a third contrastive term L_SimGCL is
        # added to the total loss. Two perturbed LightGCN propagations are
        # run per step; simgcl_view_forward() delegates to damps_simgcl.py.
        # _ui_mat is a transient reference to the UI_mat passed in forward();
        # it is cached so _lightgcn_propagate / simgcl_view_forward can
        # be invoked within the same training step without re-receiving it.
        # ------------------------------------------------------------------
        self.enable_simgcl: bool = bool(enable_simgcl)
        self.simgcl_eps: float = float(simgcl_eps)
        self.simgcl_batch_size_user: int = int(simgcl_batch_size_user)
        self.simgcl_batch_size_item: int = int(simgcl_batch_size_item)
        # ---- Branch A (rev55 §8.1) -- speedup levers ----
        self.branchA_view_every_k: int = int(branchA_view_every_k)
        self.branchA_bcl_batchn: bool = bool(branchA_bcl_batchn)
        self.branchA_view_bsz: int = int(branchA_view_bsz)
        self.branchA_bcl_bsz: int = int(branchA_bcl_bsz)
        self._simgcl_view_cache: Optional[tuple] = None
        self._simgcl_view_epoch: int = -1
        self._ui_mat: Optional[torch.Tensor] = None

        # ---- Branch A' / NRDMC-lite (rev55 §8.2) ----
        # Learnable SAV + IAV view generators + adaptive fusion + view GCN.
        # See ``damps/nrdmc_lite.py`` for the math + design notes.
        self.enable_nrdmc_lite: bool = bool(enable_nrdmc_lite)
        self.nrdmc_lite_layers: int = int(nrdmc_lite_layers)
        # ---- P3 (rev56) Prototype-Aware View ----
        self.enable_ptv: bool = bool(enable_ptv)
        self.n_prototypes: int = int(n_prototypes)
        self.lambda_ptv: float = float(lambda_ptv)
        if self.enable_nrdmc_lite:
            # Lazy import so the SimGCL-only path has zero import cost.
            from damps.nrdmc_lite import NRDMCLiteView  # pylint: disable=C0415
            self.nrdmc_lite_view = NRDMCLiteView(
                n_users=self.n_users,
                n_items=self.n_items,
                embed_dim=self.embedding_dim,
                n_layers=self.nrdmc_lite_layers,
                enable_ptv=self.enable_ptv,
                n_prototypes=self.n_prototypes,
                lambda_ptv=self.lambda_ptv,
            )
        else:
            self.nrdmc_lite_view = None  # type: ignore[assignment]

    # ------------------------------------------------------------------
    #  Setters & accessors
    # ------------------------------------------------------------------
    def set_meta_categories(self, cats: torch.Tensor) -> None:
        """Register the static metadata category vector (n_items,) for APC."""
        if cats.shape[0] != self.n_items:
            raise ValueError(
                f"meta_categories length {cats.shape[0]} != n_items {self.n_items}"
            )
        self.meta_categories.copy_(cats.long())

    def set_log_q(self, log_q: torch.Tensor) -> None:
        """Register the per-item log-popularity vector (n_items,).

        Must be called once after model construction, before the first
        training step, when ``enable_logq=True``. The tensor is copied
        into the registered buffer so it follows the model on .to(device)
        and is persisted by state_dict.

        Source of truth: ``damps.popularity_prior.load_or_build_log_q``.
        """
        if log_q.shape != (self.n_items,):
            raise ValueError(
                f"log_q shape {tuple(log_q.shape)} != ({self.n_items},)"
            )
        if not torch.isfinite(log_q).all():
            n_bad = int((~torch.isfinite(log_q)).sum())
            raise ValueError(
                f"log_q contains {n_bad} non-finite value(s); "
                "rebuild with mode='laplace' and beta > 0."
            )
        self.log_q.copy_(log_q.detach().to(self.log_q.dtype))

    @property
    def has_item_branch(self) -> bool:
        return self.item_loss_ratio != 0.0

    @property
    def has_user_branch(self) -> bool:
        return self.user_loss_ratio != 0.0

    # ------------------------------------------------------------------
    #  DAMPS pre-pass: raw modality → calibrated d=64 representation
    # ------------------------------------------------------------------
    def _damps_calibration(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor],
               torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Project raw modality features to d=64, then run DAMPS calibration.

        Returns:
            (h_img_raw, h_txt_raw, h_aud_raw, h_img_cal, h_txt_cal, h_aud_cal)
        """
        h_img_raw = self.image_proj(self.raw_image)
        h_txt_raw = self.text_proj(self.raw_text)
        h_aud_raw: Optional[torch.Tensor] = None
        if self.has_audio:
            h_aud_raw = self.audio_proj(self.raw_audio)            # type: ignore[union-attr]

        cats = self.meta_categories if self.ablations["apc"] else None
        h_img_cal, h_txt_cal, h_aud_cal = self.damps(
            h_img_raw, h_txt_raw, item_categories=cats, h_aud=h_aud_raw
        )
        return h_img_raw, h_txt_raw, h_aud_raw, h_img_cal, h_txt_cal, h_aud_cal

    # ------------------------------------------------------------------
    #  Soft Residual-Routing (eq. 3 of the spec)
    # ------------------------------------------------------------------
    def _alpha_effective(self, theta: torch.Tensor) -> torch.Tensor:
        """Map raw ``theta`` (``nn.Parameter``) to the effective gate value.

        rev57 P4 -- see ``__init__`` doc-block for the four modes.
        "raw" is the identity (backward-compat with rev55/rev56).
        """
        mode = self.asc_gate_mode
        if mode == "sigmoid":
            return torch.sigmoid(theta)
        if mode == "tanh_signed":
            return torch.tanh(theta)
        if mode == "tanh01":
            return 0.5 * (torch.tanh(theta) + 1.0)
        return theta  # "raw"

    def _soft_route(
        self,
        h_raw: torch.Tensor,
        h_cal: torch.Tensor,
        ln: nn.LayerNorm,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        if not self.ablations["soft_routing"]:
            return h_cal
        alpha_eff = self._alpha_effective(alpha)
        return h_raw + alpha_eff * ln(h_cal)

    # ------------------------------------------------------------------
    #  Forward pass
    # ------------------------------------------------------------------
    def forward(
        self,
        UI_mat: torch.Tensor,
        I2I_mat: torch.Tensor,
        U2U_mat: torch.Tensor,
        item_indices: Optional[torch.Tensor] = None,
        epoch: int = 0,
        update_momentum: bool = True,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass that produces both the CF view and the hypergraph view.

        Args:
            UI_mat          : (n_users + n_items)² sparse — bipartite graph.
            I2I_mat         : (n_items, n_items) sparse — multi-modal hypergraph.
            U2U_mat         : (n_users, n_users) sparse — co-interaction graph.
            item_indices    : optional (B,) — items covered by this batch
                              (used by the Slim Momentum updater to limit
                              the buffer write to relevant rows).
            epoch           : current epoch (drives the EMA schedule).
            update_momentum : if False, the Slim Momentum buffers are not
                              touched (e.g. during evaluation).

        Returns:
            Dict with keys:
                u_ui_emb, i_ui_emb     : final user/item embeddings.
                ii_emb, uu_emb         : hypergraph-only embeddings.
                h_img_cal, h_txt_cal   : calibrated modality features.
                h_aud_cal              : audio (Tiktok), else None.
                damps_node_input       : the actual node features fed into HGCN.
        """
        # Cache UI_mat so _lightgcn_propagate / simgcl_view_forward can
        # reuse it within this training step without receiving it as a
        # parameter. This is the Block 2 Wave 2 structural prerequisite.
        self._ui_mat = UI_mat

        # =====================================================================
        # (A) DAMPS calibration pass (cheap: ~5 ms for N=23k on RTX 5090)
        # =====================================================================
        # Drive the IMCF / MAD adaptive EMA schedule via the explicit epoch
        # index from the trainer (Revision 9 audit WARN 3). Without this,
        # ``_apply_imcf`` would tick its counter once per forward pass and
        # exhaust ``warmup_epochs`` inside the first real epoch.
        if self.training:
            self.damps.set_epoch(epoch)
        (
            h_img_raw,
            h_txt_raw,
            h_aud_raw,
            h_img_cal,
            h_txt_cal,
            h_aud_cal,
        ) = self._damps_calibration()

        # =====================================================================
        # (B) Slim Momentum update — only on the items covered by this batch
        # =====================================================================
        if self.training and update_momentum and item_indices is not None:
            with torch.no_grad():
                self.momentum.update(
                    item_indices=item_indices.to(h_img_cal.device),
                    h_cal_img=h_img_cal[item_indices],
                    h_cal_txt=h_txt_cal[item_indices],
                    h_cal_aud=(
                        h_aud_cal[item_indices] if h_aud_cal is not None else None
                    ),
                    epoch=epoch,
                )

        # =====================================================================
        # (C) Soft Residual-Routing → produce HGCN node features
        # =====================================================================
        node_img = self._soft_route(h_img_raw, h_img_cal, self.ln_img, self.alpha_img)
        node_txt = self._soft_route(h_txt_raw, h_txt_cal, self.ln_txt, self.alpha_txt)
        node_aud: Optional[torch.Tensor] = None
        if h_aud_cal is not None and self.ln_aud is not None:
            node_aud = self._soft_route(
                h_aud_raw,                                          # type: ignore[arg-type]
                h_aud_cal,
                self.ln_aud,
                self.alpha_aud,
            )

        # Average the per-modality node features into a single (n_items, d)
        # signal that drives the hypergraph propagation. (We *also* keep the
        # original ``ii_embedding`` table — see eq. (4)/(11) of the spec.)
        if node_aud is not None:
            damps_node_signal = (node_img + node_txt + node_aud) / 3.0
        else:
            damps_node_signal = (node_img + node_txt) / 2.0

        # =====================================================================
        # (D) Hypergraph branches (item-side and user-side)
        # =====================================================================
        # The DAMPS-fed node signal is *added* to the learnable ii embedding
        # so the hypergraph propagation has both a learnable bias and a
        # spectral-cleansed multi-modal signal (Section 1.3 of the spec).
        ii_emb = self.ii_embedding.weight + damps_node_signal
        uu_emb = self.uu_embedding.weight

        if self.has_item_branch:
            for _ in range(self.item_layers):
                ii_emb = _safe_sparse_mm(I2I_mat, ii_emb)

        if self.has_user_branch:
            for _ in range(self.user_layers):
                uu_emb = _safe_sparse_mm(U2U_mat, uu_emb)

        # =====================================================================
        # (E) CF branch — LightGCN / NGCF / MF
        # =====================================================================
        if self.cf_model == "LightGCN":
            # Delegate to the extracted method so SimGCL can reuse the
            # same propagation path with perturbed egos (Wave 2 Block 2).
            u_ui_emb, i_ui_emb = self._lightgcn_propagate(
                self.user_ui_embedding.weight,
                self.item_ui_embedding.weight,
            )
        elif self.cf_model == "NGCF":
            ego = torch.cat(
                [self.user_ui_embedding.weight, self.item_ui_embedding.weight], dim=0
            )
            stack = [ego]
            for i in range(self.ui_layers):
                side = _safe_sparse_mm(UI_mat, ego)
                sum_e = F.leaky_relu(self.GC_Linear_list[i](side))
                bi_e = F.leaky_relu(
                    self.Bi_Linear_list[i](torch.mul(ego, side))
                )
                ego = self.dropout_list[i](sum_e + bi_e)
                stack.append(F.normalize(ego, p=2, dim=1))
            mean = torch.stack(stack, dim=1).mean(dim=1)
            u_ui_emb, i_ui_emb = torch.split(mean, [self.n_users, self.n_items], dim=0)
        else:                                                       # 'MF'
            u_ui_emb = self.user_ui_embedding.weight
            i_ui_emb = self.item_ui_embedding.weight

        # =====================================================================
        # (F) Fuse hypergraph view into CF view (mirrors original MMHCL)
        # =====================================================================
        if self.has_item_branch:
            i_ui_emb = i_ui_emb + F.normalize(ii_emb, p=2, dim=1)
        if self.has_user_branch:
            u_ui_emb = u_ui_emb + F.normalize(uu_emb, p=2, dim=1)

        return {
            "u_ui_emb": u_ui_emb,
            "i_ui_emb": i_ui_emb,
            "ii_emb": ii_emb,
            "uu_emb": uu_emb,
            "h_img_cal": h_img_cal,
            "h_txt_cal": h_txt_cal,
            "h_aud_cal": h_aud_cal,
            "damps_node_signal": damps_node_signal,
        }

    # ------------------------------------------------------------------
    #  LightGCN propagation (extracted for SimGCL Wave 2 reuse)
    # ------------------------------------------------------------------
    def _lightgcn_propagate(
        self,
        ego_user: torch.Tensor,
        ego_item: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """LightGCN propagation loop, extracted from forward() block E.

        Concatenates user/item ego embeddings, runs ``self.ui_layers``
        rounds of sparse propagation on the normalised bipartite adjacency
        ``self._ui_mat`` (cached at the start of every forward() call),
        layer-averages the trace, and splits the result.

        This method is invoked exactly once per forward() call on the
        anchor ego, and twice more from simgcl_view_forward() on perturbed
        egos when SimGCL is enabled. The default forward path is bit-for-bit
        identical to the pre-Wave-2 inline loop.

        Args:
            ego_user : (n_users, d) user ego embeddings.
            ego_item : (n_items, d) item ego embeddings.

        Returns:
            Tuple (u_ui_emb, i_ui_emb) after LightGCN propagation and
            layer averaging.

        Raises:
            RuntimeError : if called before forward() has set self._ui_mat.
        """
        if self._ui_mat is None:
            raise RuntimeError(
                "_lightgcn_propagate called before forward() cached _ui_mat. "
                "This should never happen during normal training."
            )
        ego = torch.cat([ego_user, ego_item], dim=0)         # (n_u + n_i, d)
        all_embs = [ego]
        for _ in range(self.ui_layers):
            ego = _safe_sparse_mm(self._ui_mat, ego)
            all_embs.append(ego)
        mean = torch.stack(all_embs, dim=1).mean(dim=1)      # LightGCN avg
        return mean[:self.n_users], mean[self.n_users:]

    # ------------------------------------------------------------------
    #  SimGCL view-invariance forward (Wave 2 / M1)
    # ------------------------------------------------------------------
    def simgcl_view_forward(
        self,
        epoch: int = 0,
        *,
        return_diag: bool = False,
    ) -> torch.Tensor:
        """Compute L_SimGCL = 0.5 * (L_user + L_item) with view-cache reuse.

        Branch A (rev55 §8.1) augments the rev54 helper with batch-N InfoNCE
        and epoch-aware view caching via ``branchA_view_every_k``.

        Args:
            epoch: Current training epoch index, supplied by train.py.
            return_diag: If True (NRDMC-lite only), also cache PTV/SAV/IAV
                scalar diagnostics on ``self._last_nrdmc_diag``. Default
                False avoids per-step CUDA syncs (P3 perf guide).

        Returns:
            Scalar loss tensor with gradient flow into ego embeddings.
        """
        # ------------------------------------------------------------------
        # Branch A' (rev55 §8.2) -- NRDMC-lite short-circuit.
        # When enabled, this method returns L_mv computed against learnable
        # SAV + IAV view generators instead of the SimGCL noise views. The
        # SimGCL branch below is UNCHANGED and remains bit-for-bit identical
        # when enable_nrdmc_lite=False.
        # ------------------------------------------------------------------
        if self.enable_nrdmc_lite:
            from damps.nrdmc_lite import (  # pylint: disable=import-outside-toplevel
                compute_nrdmc_view_loss,
            )
            ego_u = self.user_ui_embedding.weight
            ego_i = self.item_ui_embedding.weight
            # One extra LightGCN pass to get post-GCN E_hat (ê) for SAV/IAV.
            u_hat, i_hat = self._lightgcn_propagate(ego_u, ego_i)
            if self._ui_mat is None:
                raise RuntimeError(
                    "nrdmc_lite view called before forward() cached _ui_mat."
                )
            out = compute_nrdmc_view_loss(
                view_module=self.nrdmc_lite_view,
                e_u_hat=u_hat,
                e_i_hat=i_hat,
                e_u_ego=ego_u,
                e_i_ego=ego_i,
                ui_mat=self._ui_mat,
                tau=self.tau,
                batch_size=self.branchA_view_bsz,
                return_diag=return_diag,
            )
            if return_diag:
                loss_t, diag = out  # type: ignore[misc]
                self._last_nrdmc_diag = diag
                return loss_t
            return out  # type: ignore[return-value]

        if not self.enable_simgcl:
            return torch.zeros(
                (), device=self.user_ui_embedding.weight.device
            )

        from branchA_simgcl_batchN import (  # pylint: disable=import-outside-toplevel
            compute_simgcl_view_loss,
            inject_uniform_noise,
        )

        refresh = (
            self._simgcl_view_cache is None
            or (
                epoch != self._simgcl_view_epoch
                and (epoch % self.branchA_view_every_k == 0)
            )
        )

        if refresh or self._simgcl_view_cache is None:
            loss, views = compute_simgcl_view_loss(
                propagate_fn=self._lightgcn_propagate,
                ego_user=self.user_ui_embedding.weight,
                ego_item=self.item_ui_embedding.weight,
                eps=self.simgcl_eps,
                tau=self.tau,
                batch_size_user=self.branchA_view_bsz,
                batch_size_item=self.branchA_view_bsz,
                views_cached=None,
            )
            self._simgcl_view_cache = tuple(v.detach() for v in views)
            self._simgcl_view_epoch = epoch
        else:
            ego_u = self.user_ui_embedding.weight
            ego_i = self.item_ui_embedding.weight
            u_pert = inject_uniform_noise(ego_u, self.simgcl_eps)
            i_pert = inject_uniform_noise(ego_i, self.simgcl_eps)
            u_now, i_now = self._lightgcn_propagate(u_pert, i_pert)
            u_now = F.normalize(u_now, dim=-1)
            i_now = F.normalize(i_now, dim=-1)
            u_cached, _, i_cached, _ = self._simgcl_view_cache
            views_paired = (u_now, u_cached, i_now, i_cached)
            loss, _ = compute_simgcl_view_loss(
                propagate_fn=self._lightgcn_propagate,
                ego_user=ego_u,
                ego_item=ego_i,
                eps=self.simgcl_eps,
                tau=self.tau,
                batch_size_user=self.branchA_view_bsz,
                batch_size_item=self.branchA_view_bsz,
                views_cached=views_paired,
            )
        return loss

    # ------------------------------------------------------------------
    #  P5.1 (rev58) — Cross-modal Alignment Loss L_align
    #  NRDMC IPM 2026 Section 5.4.1 Eq. 21 (item-side, symmetric).
    # ------------------------------------------------------------------
    def align_loss_forward(
        self,
        item_indices: torch.Tensor | None = None,
        tau0: float = 0.2,
    ) -> torch.Tensor:
        """Symmetric item-side cross-modal InfoNCE (L_align, Eq. 21).

        Aligns the raw projected image and text embeddings of each item
        in the current BPR mini-batch. Positives are the diagonal of
        ``sim(h_v[B], h_t[B])`` and negatives are all other items in
        the same batch.

        Args:
            item_indices : (B,) LongTensor of item ids for the current
                           mini-batch. If ``None``, the full item set is
                           used (only recommended for very small n_items;
                           full-batch is O(n_items^2) memory).
            tau0         : InfoNCE temperature; clamped at 0.01 to avoid
                           exp overflow. Paper default: 0.2 on Clothing.

        Returns:
            scalar 0-d tensor (fp32). Zero-tensor short-circuit is
            handled by the caller (train.py) when lambda_align == 0.

        Notes:
            * Runs in fp32 with autocast disabled so cross-modal
              gradients are not clipped by bf16 rounding.
            * Uses the *same* ``image_proj`` / ``text_proj`` layers as
              the main forward pass, so L_align directly steers the
              raw modality projections (which is exactly the pathway
              the frozen-cl diagnostic in P4 flagged as dormant).
        """
        device = self.raw_image.device
        if item_indices is None:
            v_feat = self.raw_image
            t_feat = self.raw_text
        else:
            v_feat = self.raw_image[item_indices]
            t_feat = self.raw_text[item_indices]
        # Force fp32 for numerical stability; L_align is a pure loss term
        # (never enters the CUDAGraph capture in P5.1) so extra fp32 work
        # is a per-batch O(B^2) hit that is negligible at B=1024.
        with torch.amp.autocast(device_type=device.type, enabled=False):
            h_v = self.image_proj(v_feat.float())
            h_t = self.text_proj(t_feat.float())
            h_v = F.normalize(h_v, p=2, dim=1)
            h_t = F.normalize(h_t, p=2, dim=1)
            tau0_eff = max(float(tau0), 0.01)
            logits_t2v = (h_t @ h_v.transpose(0, 1)) / tau0_eff
            logits_v2t = (h_v @ h_t.transpose(0, 1)) / tau0_eff
            n = h_v.size(0)
            labels = torch.arange(n, device=h_v.device)
            l_t2v = F.cross_entropy(logits_t2v, labels)
            l_v2t = F.cross_entropy(logits_v2t, labels)
            l_align = 0.5 * (l_t2v + l_v2t)
        return l_align

    # ------------------------------------------------------------------
    #  Contrastive Loss (Learnable τ — Section 3.1 of the spec)
    # ------------------------------------------------------------------
    def batched_contrastive_loss(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        batch_size: int = 4096,
        apply_logq: bool = False,
    ) -> torch.Tensor:
        """
        InfoNCE contrastive loss with optional LogQ popularity correction.

        Math (rev53 §3.1, eq. 1; variant "h"):
            L_NCEQ = - log[ exp((sim(u,i+) - s·clip(log_q(i+))) / τ) /
                            Σ_j exp((sim(u,i-) - s·clip(log_q(j))) / τ) ]
        where s = logq_scale and clip(·) = clamp(·, -logq_clip, +logq_clip).

        Args:
            z1, z2     : (N, d) row-normalised embeddings. Positive pairs
                         are the diagonal of sim(z1, z2). Rows index either
                         items (for ``bcl_item``) or users (for ``bcl_user``).
            batch_size : column-chunk size for the row-wise InfoNCE
                         (unchanged from rev45).
            apply_logq : when True AND ``self.enable_logq=True``, subtract
                         the scaled+clipped log_q from each column logit.
                         When False, the loss reduces to the original
                         rev45 baseline (bit-for-bit identical).

        Backward compatibility:
            * Existing call ``self.model.batched_contrastive_loss(z1, z2)``
              uses ``apply_logq=False`` — no behaviour change.
            * The user-branch call MUST stay at ``apply_logq=False`` because
              log_q is an item-popularity prior; applying it on users would
              double-count user activity bias.
        """
        # ---- Branch A (rev55 §8.1) -- batch-N variant for speed ----
        if getattr(self, "branchA_bcl_batchn", False):
            from branchA_simgcl_batchN import (  # pylint: disable=import-outside-toplevel
                batched_contrastive_loss_batchN,
            )
            log_q_arg = self.log_q if (self.enable_logq and apply_logq) else None
            return batched_contrastive_loss_batchN(
                z1=z1,
                z2=z2,
                tau=self.tau,
                batch_size=self.branchA_bcl_bsz,
                apply_logq=bool(self.enable_logq and apply_logq),
                log_q=log_q_arg,
                logq_scale=float(self.logq_scale),
                logq_clip=float(self.logq_clip),
            )
        # ---- (else: fall through to the rev54 all-rank implementation) ----

        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        tau = torch.clamp(self.tau, min=0.01)

        # Decide once per call whether we are in LogQ mode.
        use_logq = bool(self.enable_logq and apply_logq)
        if use_logq:
            if self.log_q.shape[0] != num_nodes:
                raise ValueError(
                    f"LogQ correction requires log_q.shape[0] == z1.shape[0]; "
                    f"got log_q={self.log_q.shape[0]}, z1={num_nodes}. "
                    "Pass apply_logq=True ONLY on the item branch."
                )
            if float(self.log_q.abs().sum()) == 0.0:
                # Fail-fast (rev53 §3.1, line 104): an uninitialised log_q
                # buffer would silently disable the correction.
                raise RuntimeError(
                    "enable_logq=True but log_q is zero. "
                    "Call model.set_log_q(...) once after construction."
                )
            log_q_term = torch.clamp(
                self.logq_scale * self.log_q.to(device),
                min=-self.logq_clip,
                max=+self.logq_clip,
            )                                                 # (num_nodes,)

            def f_logq(
                sim_block: torch.Tensor,
                col_slice: torch.Tensor,
            ) -> torch.Tensor:
                return torch.exp(
                    (sim_block - log_q_term[col_slice][None, :]) / tau
                )
        else:
            def f_simple(sim_block: torch.Tensor) -> torch.Tensor:  # type: ignore[misc]
                return torch.exp(sim_block / tau)

        indices = torch.arange(0, num_nodes, device=device)
        losses: list[torch.Tensor] = []

        for i in range(num_batches):
            mask = indices[i * batch_size : (i + 1) * batch_size]
            if use_logq:
                # All columns participate, so col_slice is the full indices.
                refl_sim = f_logq(self._sim(z1[mask], z1), indices)
                between_sim = f_logq(self._sim(z1[mask], z2), indices)
            else:
                refl_sim = f_simple(self._sim(z1[mask], z1))
                between_sim = f_simple(self._sim(z1[mask], z2))

            losses.append(
                -torch.log(
                    between_sim[:, i * batch_size : (i + 1) * batch_size].diag()
                    / (
                        refl_sim.sum(1)
                        + between_sim.sum(1)
                        - refl_sim[:, i * batch_size : (i + 1) * batch_size].diag()
                    )
                )
            )
        return torch.cat(losses).mean()

    @staticmethod
    def _sim(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Cosine similarity between two row-normalised tensors."""
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    # ------------------------------------------------------------------
    #  Diagnostics
    # ------------------------------------------------------------------
    @torch.no_grad()
    def diagnostics(self) -> dict[str, Any]:
        """
        Return a summary of the model's current state (used by train.py for
        per-epoch logs):

            damps_params         : trainable params inside DAMPS only.
            tau                  : current learnable temperature.
            tau_clamped          : effective temperature after clamping.
            alpha_img / alpha_txt: Soft-Routing scalars.
            tanh_saturation      : per-modality saturation rates.
            momentum_init        : how many items have been touched.
        """
        sat = self.damps.tanh_saturation_rates()
        return {
            "damps_params": self.damps.num_trainable_params(),
            "tau": float(self.tau.item()),
            "tau_clamped": float(torch.clamp(self.tau, min=0.01).item()),
            "tau_mode": "learnable" if self.learnable_tau else "static",
            # rev57 P4: report *effective* alpha (post-transform) so the
            # train.py diagnostic log traces the value the model is actually
            # multiplying by, plus the raw theta for reproducibility.
            "alpha_img": float(self._alpha_effective(self.alpha_img).item()),
            "alpha_txt": float(self._alpha_effective(self.alpha_txt).item()),
            "alpha_aud": (
                float(self._alpha_effective(self.alpha_aud).item())
                if self.has_audio else None
            ),
            "alpha_img_theta": float(self.alpha_img.item()),
            "alpha_txt_theta": float(self.alpha_txt.item()),
            "alpha_aud_theta": (
                float(self.alpha_aud.item()) if self.has_audio else None
            ),
            "asc_gate_mode": self.asc_gate_mode,
            "tanh_sat_img": sat["img"],
            "tanh_sat_txt": sat["txt"],
            "tanh_sat_aud": sat["aud"] if self.has_audio else None,
            "lambda_coh": float(self.damps.lambda_coh.item()),
            "baseline_asc": float(self.damps.baseline_asc.item()),
            "imcf_epoch": int(self.damps._current_epoch.item()),
            "imcf_forward_passes": int(self.damps._imcf_update_count.item()),
            "momentum_init": self.momentum.initialised_count(),
        }

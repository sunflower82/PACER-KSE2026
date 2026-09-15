"""
utility/parser.py — DAMPS-MMHCL command-line argument parser
==============================================================

Centralises every hyperparameter and configuration flag used by the
DAMPS-MMHCL training script (``train.py``).

Compatible with the original MMHCL CLI surface (every flag from
``codes/utility/parser.py`` is preserved) plus the new DAMPS-specific
options listed in Section 3 of the DAMPS spec.

Phase-1 / rev44 / Revision 11 defaults (Quick Win)
--------------------------------------------------
* ``--temperature 0.3`` (was 0.1)
* ``--learnable_tau 0`` (new flag; rev42 used a learnable nn.Parameter, which
  empirically saturates at ~0.0909 and triggers an embedding collapse).
* ``--damps_avrf 0`` (was 1; rev44 disables AVRF for Phase 1 to recover
  Recall@20 coverage on the sparse Amazon Clothing dataset).

Usage examples
--------------
::

    # Default invocation == rev44 Phase 1 recommended config (d):
    python train.py --dataset Clothing --seed 42

    # Reproduce the rev42 / Revision 9 baseline anchor (a):
    python train.py --dataset Clothing --seed 42 \\
        --temperature 0.1 --learnable_tau 1 --damps_avrf 1

    # Phase 1 variant (b) -- only the static-tau fix:
    python train.py --dataset Clothing --seed 42 \\
        --temperature 0.3 --learnable_tau 0 --damps_avrf 1

    # Phase 1 variant (c) -- only the AVRF-off fix:
    python train.py --dataset Clothing --seed 42 \\
        --temperature 0.1 --learnable_tau 1 --damps_avrf 0
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    """Parse and return all DAMPS-MMHCL CLI arguments."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description=(
            "DAMPS-MMHCL: Multi-Modal Hypergraph Contrastive Learning "
            "with Spectral Domain Representation Calibration"
        )
    )

    # =====================================================================
    #  General / Data
    # =====================================================================
    parser.add_argument("--data_path", type=str, default="../data/",
                        help="Root path to all dataset folders.")
    parser.add_argument("--seed", type=int, default=2024,
                        help="Random seed for reproducibility.")
    parser.add_argument("--dataset", type=str, default="Clothing",
                        help="Dataset name: {Tiktok, Sports, Clothing}.")
    parser.add_argument("--core", type=int, default=5,
                        help="K-core filtering threshold.")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU index to use.")
    parser.add_argument("--debug", default="True",
                        help='If "True", enable per-run file logging.')

    # =====================================================================
    #  Training Schedule
    # =====================================================================
    parser.add_argument("--verbose", type=int, default=5,
                        help="Run validation every N epochs.")
    parser.add_argument("--epoch", type=int, default=250,
                        help="Maximum number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=1024,
                        help="Number of BPR triplets per mini-batch.")
    parser.add_argument("--regs", type=float, default=1e-3,
                        help="L2 regularisation coefficient (BPR).")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Initial learning rate for Adam.")
    parser.add_argument("--clip_grad_norm", type=float, default=1.0,
                        help="Max L2 norm for gradient clipping (0 = disabled).")

    # =====================================================================
    #  Model Architecture (MMHCL backbone)
    # =====================================================================
    parser.add_argument("--embed_size", type=int, default=64,
                        help="User/item embedding dimensionality.")
    parser.add_argument("--weight_size", type=str, default="[64,64,64]",
                        help="Per-layer GNN sizes (NGCF only).")
    parser.add_argument("--topk", type=int, default=5,
                        help="K for the K-NN sparsification.")
    parser.add_argument("--cf_model", type=str, default="LightGCN",
                        help="CF backbone: {MF, NGCF, LightGCN}.")
    parser.add_argument("--norm_type", type=str, default="sym",
                        help="Adjacency normalisation: {sym, rw, origin}.")
    parser.add_argument("--sparse", type=int, default=0,
                        help="1 = sparse UI graph; 0 = dense.")
    parser.add_argument("--UI_layers", type=int, default=2,
                        help="GNN layers on user-item bipartite graph.")
    parser.add_argument("--User_layers", type=int, default=3,
                        help="GNN layers on user-user co-interaction graph.")
    parser.add_argument("--Item_layers", type=int, default=2,
                        help="GNN layers on item-item multi-modal hypergraph.")
    parser.add_argument("--user_loss_ratio", type=float, default=0.03,
                        help="Weight for user-side contrastive loss.")
    parser.add_argument("--item_loss_ratio", type=float, default=0.07,
                        help="Weight for item-side contrastive loss.")
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="InfoNCE temperature tau (shared by trunk BCL "
                             "and NRDMC-lite / SimGCL view InfoNCE). "
                             "Default 0.3 matches rev44 Phase 1. "
                             "Branch A' upgrade P2 probes {0.30, 0.40} "
                             "(softer view gradients vs clothing τ=0.251). "
                             "Learnable tau: --learnable_tau 1 "
                             "--temperature 0.1.")
    parser.add_argument("--learnable_tau", type=int, default=0,
                        help="0 = static tau (rev44 Phase 1 default; tau "
                             "is a non-trainable buffer fixed at "
                             "--temperature throughout training); "
                             "1 = learnable tau (rev42 baseline; tau is an "
                             "nn.Parameter initialised at --temperature, "
                             "clamped to >= 0.01). Set to 0 to break the "
                             "tau-saturation embedding-collapse failure mode "
                             "documented in rev44 Section 3.")

    # =====================================================================
    #  Evaluation
    # =====================================================================
    parser.add_argument("--Ks", type=str, default="[10,20]",
                        help="Cut-offs for Recall/Precision/NDCG/Hit@K.")
    parser.add_argument("--test_flag", type=str, default="part",
                        help="{part, full}. 'part' = heapq top-K (faster).")
    parser.add_argument(
        "--use_gpu_eval",
        type=int,
        default=1,
        help="1 = GPU-native test_torch (eval bottleneck guide §4; "
             "~5-8x faster eval). 0 = legacy CPU multiprocessing Pool "
             "path (kept for metric-equivalence audits).",
    )
    parser.add_argument(
        "--eval_every",
        type=int,
        default=0,
        help="Run validation every N epochs. 0 = fall back to --verbose "
             "(legacy). Eval-bottleneck guide recommends 5. Always eval "
             "in the last --eval_last_epochs epochs regardless.",
    )
    parser.add_argument(
        "--eval_last_epochs",
        type=int,
        default=20,
        help="Always run eval during the final N epochs of --epoch "
             "(used with --eval_every > 0). Default 20.",
    )

    # =====================================================================
    #  Early Stopping
    # =====================================================================
    parser.add_argument("--early_stopping_patience", type=int, default=5,
                        help="Non-improving evaluation cycles before stop.")
    parser.add_argument("--early_stopping_min_epochs", type=int, default=0,
                        help="Min epochs before early stopping can trigger.")
    parser.add_argument("--early_stopping_min_delta", type=float, default=1e-4,
                        help="Min metric improvement to count as progress.")
    parser.add_argument("--early_stopping_mode", type=str, default="max",
                        help="{max, min}.")
    parser.add_argument("--early_stopping_restore_best", type=int, default=1,
                        help="1 = restore best weights on stop.")
    parser.add_argument(
        "--early_stopping_monitor",
        type=str,
        default="val_recall@20",
        help="Validation metric that drives the early-stopping patience "
             "counter. Accepted forms: 'val_recall@K', 'val_ndcg@K', "
             "'val_precision@K' (K must appear in --Ks), or the aliases "
             "'recall' / 'ndcg' / 'precision' (uses the last K in --Ks). "
             "Peak-snapshot bookkeeping for recall AND ndcg is unchanged; "
             "only the patience reset rule is gated by this flag. "
             "Default 'val_recall@20' matches the PACER / smoke protocol.",
    )
    parser.add_argument(
        "--adaptive_patience",
        type=int,
        default=0,
        help="0 = fixed --early_stopping_patience (default; smoke / "
             "PACER tercile protocol). "
             "1 = grow effective patience by +1 every 50 epochs after "
             "--early_stopping_min_epochs (mild schedule; keeps long "
             "runs from stopping too aggressively on noisy plateaus).",
    )

    # =====================================================================
    #  ReduceLROnPlateau
    # =====================================================================
    parser.add_argument("--use_reduce_lr", type=int, default=0)
    parser.add_argument("--reduce_lr_factor", type=float, default=0.5)
    parser.add_argument("--reduce_lr_patience", type=int, default=3)
    parser.add_argument("--reduce_lr_min", type=float, default=1e-6)

    # =====================================================================
    #  W&B Tracking
    # =====================================================================
    # ``--wandb`` is an exact alias for ``--use_wandb``. Without it, bare
    # ``--wandb`` is an ambiguous prefix of ``--wandb_project`` / ``_entity``
    # / … and argparse exits 2 (silent NaN aggregates in notebook drivers).
    parser.add_argument(
        "--use_wandb",
        "--wandb",
        type=int,
        default=0,
        dest="use_wandb",
        help="1 = enable Weights & Biases logging (alias: --wandb).",
    )
    parser.add_argument("--wandb_project", type=str, default="damps-mmhcl")
    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument("--wandb_run_name", type=str, default="")
    parser.add_argument(
        "--wandb_group", type=str, default="",
        help="W&B run group; groups related runs together in the W&B UI "
             "(e.g. 'wave2_branchA'). Empty string = no group.",
    )
    parser.add_argument(
        "--wandb_tags", type=str, default="",
        help="Comma-separated W&B tags attached to the run "
             "(e.g. 'branchA,batchN,rev55'). Empty string = no tags.",
    )
    parser.add_argument(
        "--wandb_job_type", type=str, default="",
        help="W&B job_type label (e.g. 'train', 'sweep_seed'). "
             "Empty string = W&B default.",
    )
    parser.add_argument(
        "--wandb_name", type=str, default="",
        help="W&B run name (alias for --wandb_run_name; takes precedence "
             "when both are set). Accepted by the RQ2 ablation runner and "
             "any script that follows the W&B CLI convention.",
    )

    # =====================================================================
    #  DAMPS-Specific (Revision 9)
    # =====================================================================
    parser.add_argument("--damps_apc", type=int, default=1,
                        help="1 = enable Metadata-Aware Adaptive Phase Calibration.")
    parser.add_argument("--damps_avrf", type=int, default=0,
                        help="0 = AVRF off (rev44 Phase 1 default; preserves "
                             "sparse signals on Amazon Clothing where AVRF "
                             "tends to over-attenuate). "
                             "1 = AVRF on (rev42 baseline; logit-clipped "
                             "Wiener gate). Phase 1 ablation explicitly sets "
                             "this to 0 to recover Recall@20 coverage.")
    parser.add_argument("--damps_imcf", type=int, default=1,
                        help="1 = enable Residual Inter-Modal Coherence Filter.")
    parser.add_argument("--damps_permutation_fft", type=int, default=0,
                        help="1 = run the falsifiable Permutation-FFT ablation.")
    parser.add_argument("--damps_soft_routing", type=int, default=1,
                        help="1 = use Soft Residual-Routing into HGCN; "
                             "0 = forward h_cal directly (will trigger over-smoothing).")
    parser.add_argument("--damps_momentum", type=int, default=1,
                        help="1 = enable Slim Momentum Encoder.")
    parser.add_argument("--damps_data_driven_prior", type=int, default=1,
                        help="1 = compute AVRF priors from raw modality features; "
                             "0 = use the hard-coded fallback (0.24/0.85).")
    parser.add_argument("--damps_num_categories", type=int, default=10,
                        help="Number of static metadata clusters for APC.")
    parser.add_argument("--damps_warmup_epochs", type=int, default=10,
                        help="Warm-up window for adaptive EMA schedules.")

    # =====================================================================
    #  rev53 §3.1 — LogQ popularity correction (variant "h")
    # =====================================================================
    parser.add_argument("--enable_logq", type=int, default=0,
                        help="1 enables the LogQ popularity correction in "
                             "the item-branch InfoNCE (rev53 §3.1 eq. 1). "
                             "Requires --logq_mode and --logq_beta to be set; "
                             "log_q is rebuilt and cached under the dataset "
                             "directory on first use.")
    parser.add_argument("--logq_mode", type=str, default="laplace",
                        choices=["laplace", "raw", "sqrt"],
                        help="q(i) estimator. 'laplace' = (n+β)/(N+|I|β); "
                             "'sqrt' = the DGRec WWW2024 less-aggressive "
                             "variant; 'raw' requires every item >= 1 "
                             "interaction (rare).")
    parser.add_argument("--logq_beta", type=float, default=1.0,
                        help="Laplace smoothing coefficient β > 0 for "
                             "logq_mode in {laplace, sqrt}. Ignored for raw.")
    parser.add_argument("--logq_scale", type=float, default=1.0,
                        help="Multiplier on log_q before subtraction. With "
                             "cosine-normalised sim and τ=0.3, the spec "
                             "default 1.0 may dominate the logits; sweep "
                             "{0.05, 0.1, 0.3, 1.0} at M1.5 before locking.")
    parser.add_argument("--logq_clip", type=float, default=5.0,
                        help="Symmetric clip on logq_scale*log_q to keep "
                             "exp(./τ) numerically safe.")

    # =====================================================================
    #  Wave 2 / M1 -- SimGCL view-invariance (Yu et al. SIGIR 2022)
    # =====================================================================
    parser.add_argument(
        "--enable_simgcl", type=int, default=0,
        help="Master switch for the SimGCL view-invariance term. "
             "0 = Wave 1 LogQ-only baseline (default); 1 = Wave 2 M1.",
    )
    parser.add_argument(
        "--simgcl_eps", type=float, default=0.1,
        help="Magnitude of the uniform-noise perturbation injected into "
             "ego embeddings before LightGCN propagation. "
             "Yu et al. (SIGIR 2022) recommend 0.1; the rev54 Optuna "
             "search range is [0.05, 0.2].",
    )
    parser.add_argument(
        "--lambda_view", type=float, default=0.05,
        help="Weight of the view-invariance loss (SimGCL or NRDMC-lite). "
             "Branch A' upgrade P1 lowers the aggressive λ=0.2 baseline "
             "to {0.05, 0.10} so BPR is not drowned by view InfoNCE. "
             "M1 ablation historically covered {0.01, 0.05, 0.1}.",
    )
    parser.add_argument(
        "--simgcl_batch_size_user", type=int, default=4096,
        help="Row-chunk size for the user-branch view-invariance loss.",
    )
    parser.add_argument(
        "--simgcl_batch_size_item", type=int, default=4096,
        help="Row-chunk size for the item-branch view-invariance loss.",
    )

    # =====================================================================
    #  Branch A -- speedup overlays for Wave 2 SimGCL (rev55 §8.1)
    # =====================================================================
    parser.add_argument(
        "--branchA_view_every_k", type=int, default=2,
        help="Compute the SimGCL view-invariance loss every K epochs and "
             "reuse the cached perturbed views on the off-epochs. "
             "K=1 reproduces the dense Wave 2 schedule; K=2 halves the "
             "number of perturbed LightGCN propagations and is the "
             "Branch A default. Set K=1 for the S2 bit-for-bit smoke "
             "test against Wave 1.",
    )
    parser.add_argument(
        "--branchA_bcl_batchn", type=int, default=1,
        help="1 = replace the (B, N) chunked InfoNCE in "
             "batched_contrastive_loss with a batch-N InfoNCE that "
             "compares each anchor against the (B-1) other rows of the "
             "mini-batch (Branch A default; ~22x FLOPs reduction on "
             "Amazon-Clothing). 0 = keep the legacy (B, N) path used in "
             "Wave 1 / Wave 2 audit runs.",
    )
    parser.add_argument(
        "--branchA_view_bsz", type=int, default=2048,
        help="Row-chunk size used by the Branch A batch-N SimGCL view "
             "loss. Must be <= simgcl_batch_size_user / "
             "simgcl_batch_size_item; the 2048 default keeps the per-chunk "
             "(B, B) Gram matrix under 16 MB FP32.",
    )
    parser.add_argument(
        "--branchA_bcl_bsz", type=int, default=2048,
        help="Row-chunk size used by the Branch A batch-N bcl_item "
             "contrastive loss when --branchA_bcl_batchn=1. Trades VRAM "
             "for throughput; 2048 matches the SimGCL chunk for cache "
             "reuse on a single A100 / RTX 4090.",
    )

    # =====================================================================
    #  Branch A' -- NRDMC-lite learnable view generators (rev55 §8.2)
    #  Replaces the SimGCL noise-based views with SAV + IAV + adaptive
    #  fusion (NRDMC IPM 2026 Eq. 14, 16, 17-19). PTV is dropped per §8.2.
    # =====================================================================
    parser.add_argument(
        "--enable_nrdmc_lite", type=int, default=0,
        help="1 = replace the SimGCL view path with the NRDMC-lite "
             "learnable SAV+IAV+adaptive-fusion view generators from "
             "rev55 §8.2 (Branch A'). Mutually exclusive with "
             "--enable_simgcl (train.py refuses both being on).",
    )
    parser.add_argument(
        "--nrdmc_lite_layers", type=int, default=2,
        help="Number of LightGCN steps to propagate over the learned "
             "contrastive graph inside the NRDMC-lite view module. "
             "Default 2 keeps the extra compute below 1%% of an epoch.",
    )

    # =====================================================================
    #  Branch A' -- P3 (rev56) Prototype-Aware View (PTV)
    #  Re-instates the third NRDMC view (Eq. 15, 20-22) that was dropped
    #  in rev55 §8.2. See PACER_NRDMC_lite_upgrade_analysis_EN §6 (rev56).
    # =====================================================================
    parser.add_argument(
        "--enable_ptv", type=int, default=0,
        help="1 = enable the Prototype-Aware View (PTV) in NRDMC-lite. "
             "Requires --enable_nrdmc_lite=1. Extends the K=2 fusion "
             "(SAV+IAV) to K=3 (SAV+IAV+PTV) via NRDMC IPM 2026 Eq. 20-22.",
    )
    parser.add_argument(
        "--n_prototypes", type=int, default=32,
        help="Number of learnable prototypes K in PTV. Only used when "
             "--enable_ptv=1. NRDMC paper defaults K in {16, 32, 64}; "
             "K=32 is the P3 default (Table 4 ablation optimum).",
    )
    parser.add_argument(
        "--lambda_ptv", type=float, default=1.0,
        help="PTV mixing coefficient inside adaptive fusion (Eq. 19 K=3). "
             "lambda_ptv=0 recovers the exact K=2 baseline; lambda_ptv=1 "
             "is the paper default (equal PTV weight with SAV/IAV).",
    )

    # =====================================================================
    #  Branch A' -- P5.1 (rev58) Cross-modal alignment loss L_align
    #  Implements NRDMC IPM 2026 Section 5.4.1 Eq. 21 as a separate loss
    #  term. Prior revs (rev55-57) only exposed L_mv (multi-view CL); the
    #  paper's L_align (item text<->image InfoNCE) was missing. Enabling
    #  L_align injects a fresh, non-saturated gradient into the raw
    #  modality projections, addressing the frozen-cl / dormant-CL
    #  diagnostic recorded in P4 (see analysis doc Section 7).
    # =====================================================================
    parser.add_argument(
        "--enable_align", type=int, default=0,
        help="1 = enable the cross-modal alignment loss L_align "
             "(NRDMC IPM 2026 Eq. 21). Symmetric item-side InfoNCE "
             "between raw projected image and text embeddings, added "
             "to the total loss with weight --lambda_align. Requires "
             "both image_feats and text_feats to be present.",
    )
    parser.add_argument(
        "--lambda_align", type=float, default=0.0,
        help="Weight of the cross-modal alignment loss L_align in the "
             "total objective (Eq. 25). NRDMC paper uses values in "
             "{0.001, 0.01, 0.1, 1.0}; P5.1 pilot defaults to 0.1. "
             "lambda_align=0 turns the loss into a zero tensor "
             "(also skips its forward pass for wall-time neutrality).",
    )
    parser.add_argument(
        "--align_temperature", type=float, default=0.2,
        help="Temperature tau_0 for the L_align InfoNCE softmax "
             "(NRDMC IPM 2026 Eq. 21). Paper reports tau_0=0.2 as the "
             "Clothing sweet spot; must be > 0.01 (clamped at read "
             "time to avoid exp overflow).",
    )

    # =====================================================================
    #  P6.0 -- MACP (Multi-Aspect Content Preprocessing, TAMER MM'25)
    #
    #  Offline text-only whitening. Two pre-computed streams
    #    ``text_feat_pca_ica.npy`` (PCA->ICA rotation)
    #    ``text_feat_zca.npy``     (ZCA whitening)
    #  live next to ``text_feat.npy`` and are produced by
    #  ``scripts/preprocess_macp.py``. Flags below decide how (or
    #  whether) to inject them into ``text_feats`` at load time. Image
    #  features are deliberately left raw -- P5.0 confirmed the model
    #  adaptively drives alpha_img to negative values on Clothing.
    # =====================================================================
    parser.add_argument(
        "--use_macp", type=int, default=0,
        help="1 = enable MACP whitening (P6.0 text and/or P6.1 image). "
             "0 (default) = leave modality feats untouched, matches the "
             "P5.1 baseline bit-for-bit. When 1, per-modality mode flags "
             "(--macp_mode, --macp_image_mode) decide which streams "
             "actually replace the raw features.",
    )
    parser.add_argument(
        "--macp_mode", type=str, default="residual",
        choices=("raw", "replace_pca", "replace_zca", "residual"),
        help="[TEXT modality] How to combine the raw and whitened text "
             "streams. 'raw' short-circuits to no-op. 'replace_*' swaps "
             "text_feats with the corresponding stream. 'residual' adds "
             "a scaled auxiliary signal on top of the raw features.",
    )
    parser.add_argument(
        "--macp_alpha_p", type=float, default=0.10,
        help="[TEXT] Residual weight for the PCA->ICA stream in "
             "'residual' mode. Tuned in {0.05, 0.10, 0.20} for the "
             "P6.0 grid.",
    )
    parser.add_argument(
        "--macp_alpha_z", type=float, default=0.10,
        help="[TEXT] Residual weight for the ZCA stream in 'residual' mode.",
    )
    # ---- P6.1: symmetric image whitening --------------------------------
    parser.add_argument(
        "--macp_image_mode", type=str, default="raw",
        choices=("raw", "replace_pca", "replace_zca", "residual"),
        help="[IMAGE modality] How to combine raw and whitened image "
             "streams. Default 'raw' preserves the P6.0 behaviour "
             "(image left untouched). 'replace_*' swaps image_feats "
             "with the corresponding stream. Prerequisite: run "
             "scripts/preprocess_macp.py --modality image first.",
    )
    parser.add_argument(
        "--macp_image_alpha_p", type=float, default=0.10,
        help="[IMAGE] Residual weight for the PCA->ICA stream in "
             "'residual' image mode.",
    )
    parser.add_argument(
        "--macp_image_alpha_z", type=float, default=0.10,
        help="[IMAGE] Residual weight for the ZCA stream in 'residual' "
             "image mode.",
    )
    parser.add_argument(
        "--macp_verbose", type=int, default=1,
        help="1 = print one-line MACP diagnostic during load "
             "(one per modality when both are active).",
    )

    # =====================================================================
    #  P6.4 -- TAMER Interest Tree augmentation
    # =====================================================================
    parser.add_argument(
        "--enable_tamer",
        type=int,
        default=0,
        help="1 = replace Item_mat with the TAMER interest-augmented "
             "modality graph from codes.damps_tamer (requires "
             "--tamer_interest_cache). 0 (default) = P6.3 modality-only.",
    )
    parser.add_argument(
        "--tamer_interest_cache",
        type=str,
        default="",
        help="Path to .npz written by "
             "scripts/preprocess_interest_tree.py "
             "(cooc_* + optional tree_* flat arrays).",
    )
    parser.add_argument(
        "--alpha_interest",
        type=float,
        default=0.25,
        help="TAMER Eq. 9 mixing weight for the interest (S^c) branch. "
             "0.0 recovers the modality-only graph.",
    )

    # =====================================================================
    #  P6.5' Bucket-aware training (popularity-inverse edge reweighting +
    #  bucket-geo-mean early-stopping monitor).
    # =====================================================================
    parser.add_argument(
        "--pop_inverse_eta",
        type=float,
        default=0.0,
        help="P6.5' popularity-inverse exponent applied to the interest "
             "branch (s_c AND coef_csr tree bonus) as "
             "M[i,j] *= (pop_i * pop_j) ** (-eta). 0.0 (default) = P6.4 "
             "bit-for-bit compatibility; typical grid {0.5, 1.0}. Requires "
             "--enable_tamer 1 and a valid --tamer_interest_cache.",
    )

    # =====================================================================
    #  Pattern B' (Scheduled Rebuild)
    # =====================================================================
    parser.add_argument("--rebuild_R", type=int, default=5,
                        help="Rebuild K-NN hypergraph every R epochs.")
    parser.add_argument("--faiss_threshold", type=int, default=60_000,
                        help="N >= threshold -> switch to FAISS HNSW path.")
    parser.add_argument("--knn_chunk_size", type=int, default=4_096,
                        help="Row-chunk size for the chunked PyTorch K-NN path.")
    parser.add_argument("--faiss_use_gpu", type=int, default=1,
                        help="1 = use FAISS GPU resources when available "
                             "(StandardGpuResources + index_cpu_to_gpu); "
                             "0 = stay on CPU FAISS. Speedup guide Section 2.")
    parser.add_argument("--knn_efsearch", type=int, default=64,
                        help="HNSW efSearch parameter (controls recall/speed "
                             "trade-off). Higher = better recall, slower.")

    # =====================================================================
    #  Mixed Precision (bfloat16)
    # =====================================================================
    parser.add_argument("--use_amp", type=int, default=1,
                        help="1 = enable bfloat16 mixed precision training.")

    # =====================================================================
    #  Training Speedup Flags (Speedup Guide, Sections 1-10)
    # =====================================================================
    parser.add_argument("--use_torch_compile", type=int, default=1,
                        help="1 = wrap the DAMPS spectral block in "
                             "torch.compile (PyTorch >= 2.0). Default ON "
                             "for Branch A' speedups; gated by the Inductor "
                             "complex-FFT backward probe. Reported "
                             "+20-40%% training speedup with "
                             "mode=reduce-overhead.")
    parser.add_argument(
        "--torch_compile_mode",
        type=str,
        default="default",
        help="torch.compile mode: {default, reduce-overhead, max-autotune}. "
             "Default 'default' — DAMPS complex FFT + IMCF EMA is unsafe under "
             "reduce-overhead CUDAGraphs (multi-step overwrite). Trainer "
             "auto-falls back if a requested mode fails the multi-step probe.",
    )
    parser.add_argument("--torch_compile_dynamic", type=int, default=0,
                        help="1 = compile with dynamic=True. Default 0 for "
                             "fixed-shape DAMPS inputs. Set 1 only if you must "
                             "tolerate shape changes.")
    parser.add_argument(
        "--use_gpu_sample",
        type=int,
        default=1,
        help="1 = sample BPR triplets on GPU (speedup guide Section C / "
             "GPU-side negative sampling; ~3-5x vs CPU on batch 1024). "
             "0 = legacy CPU rejection sampling in Data.sample().",
    )
    parser.add_argument(
        "--use_cuda_graph",
        type=int,
        default=0,
        help="1 = capture a persistent CUDAGraph of the train step "
             "(requires fixed batch shapes; incompatible with "
             "--torch_compile_dynamic 1 and with NRDMC-lite / SimGCL "
             "view paths that rebuild sparse graphs every step). "
             "Default 0; enable only for LogQ-only fixed-shape runs.",
    )

    # =====================================================================
    #  rev57 / P4 -- ASC gate reparameterization (fix alpha_img collapse)
    # =====================================================================
    # Diagnostic logs from the P3 PTV sweep show alpha_img drifting from
    # +0.09 (ep 0) to -0.68 (ep 75) -- the model learns to subtract the
    # image branch, collapsing multimodal fusion. These flags introduce
    # three complementary knobs to guard against the collapse. All default
    # to a no-op so pre-P4 runs reproduce exactly.
    parser.add_argument(
        "--asc_gate_mode", type=str, default="raw",
        choices=["raw", "sigmoid", "tanh_signed", "tanh01"],
        help=(
            "Reparameterization of the Soft-Routing residual gate alpha_v "
            "(v in {img, txt, aud}). 'raw' (default) reproduces rev55/rev56 "
            "exactly (alpha = theta, unconstrained). 'sigmoid' constrains "
            "alpha in (0, 1); 'tanh_signed' constrains alpha in (-1, 1); "
            "'tanh01' constrains alpha in (0, 1). In every non-raw mode the "
            "underlying theta is initialised so that alpha == 0.1 at step 0."
        ),
    )
    parser.add_argument(
        "--asc_warmup_epochs", type=int, default=0,
        help=(
            "Freeze alpha_img / alpha_txt / alpha_aud at their init value "
            "for the first N epochs (requires_grad=False, so their .grad "
            "stays None). After epoch N the gates are unfrozen and can "
            "train. Default 0 = no freeze (rev55/rev56 behaviour)."
        ),
    )
    parser.add_argument(
        "--asc_reg_l2", type=float, default=0.0,
        help=(
            "Coefficient for the ASC gate L2 pull-to-target regularizer "
            "L_reg = asc_reg_l2 * sum_v (alpha_v_eff - asc_reg_target)^2 "
            "added to the training loss. Default 0.0 = disabled. "
            "Suggested starting point: 0.01 (per-batch, per-modality)."
        ),
    )
    parser.add_argument(
        "--asc_reg_target", type=float, default=0.3,
        help=(
            "Target value that asc_reg_l2 pulls the effective gate toward. "
            "0.3 mirrors the empirical mid-range that the raw gate transits "
            "before drifting negative in the P3 logs."
        ),
    )

    # =====================================================================
    #  Ablation Tag
    # =====================================================================
    parser.add_argument("--ablation_target", type=str, default="",
                        help="Tag for the experiment directory.")

    return parser.parse_args()

#!/usr/bin/env bash
# =============================================================================
#  Branch A — single-seed runner (rev55 §8.1, Wave 2 Phase 2)
# =============================================================================
#  Target runtime: ~6 h / seed on a single A100-40GB or RTX 4090
#  (down from >40 h / seed for the dense Wave 2 audit run).
#
#  Frozen Wave-1 constraints (DO NOT TOUCH):
#    --enable_logq 1 --logq_scale 1.0 --logq_clip 5.0  (laplace mode default,
#                                                       beta=1.0 inside model)
#    --patience 20
#    Backbone:  apc_off_combined  (set in your config / model defaults)
#    Dataset:   Amazon-Clothing  5-core  8:1:1
#
#  Branch A overlays (rev55 §8.1):
#    --enable_simgcl 1 --simgcl_eps 0.1 --lambda_view 0.05
#    --branchA_view_every_k 2     (L_view every 2 epochs, reuse cached views)
#    --branchA_bcl_batchn 1       (batch-N InfoNCE for bcl_item, K = B-1)
#    --branchA_view_bsz 2048      (SimGCL chunk)
#    --branchA_bcl_bsz  2048      (bcl_item chunk)
#    --use_amp 1                  (bfloat16, already wired in train.py:566)
#
#  Acceptance window (rev55 §8.1): R@20 in [0.0900, 0.0945].
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
#  Configurable knobs (CLI overrides via env vars)
# -----------------------------------------------------------------------------
SEED="${SEED:-0}"
DATASET="${DATASET:-clothing}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
EPOCH="${EPOCH:-500}"
PATIENCE="${PATIENCE:-20}"
LR="${LR:-1e-3}"
LAMBDA_VIEW="${LAMBDA_VIEW:-0.05}"
SIMGCL_EPS="${SIMGCL_EPS:-0.1}"
VIEW_EVERY_K="${VIEW_EVERY_K:-2}"
USE_BCL_BATCHN="${USE_BCL_BATCHN:-1}"
LOG_DIR="${LOG_DIR:-logs/branchA}"

# -----------------------------------------------------------------------------
#  Pre-flight checks
# -----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/../src"
if [[ -f "${SRC_DIR}/train.py" ]]; then
    cd "${SRC_DIR}"
elif [[ ! -f "train.py" ]]; then
    echo "[run_branchA.sh] ERROR: cannot find src/train.py"
    exit 1
fi

mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/seed${SEED}_$(date +%Y%m%d_%H%M%S).log"

echo "[run_branchA.sh] Starting Branch A single-seed run"
echo "                seed=${SEED}  dataset=${DATASET}  batch_size=${BATCH_SIZE}"
echo "                lambda_view=${LAMBDA_VIEW}  simgcl_eps=${SIMGCL_EPS}"
echo "                view_every_k=${VIEW_EVERY_K}  bcl_batchn=${USE_BCL_BATCHN}"
echo "                log -> ${LOG_FILE}"
echo

# -----------------------------------------------------------------------------
#  Launch
# -----------------------------------------------------------------------------
python -u train.py \
    --dataset            "${DATASET}" \
    --seed               "${SEED}" \
    --batch_size         "${BATCH_SIZE}" \
    --epoch              "${EPOCH}" \
    --patience           "${PATIENCE}" \
    --lr                 "${LR}" \
    --use_amp            1 \
    --enable_logq        1 \
    --logq_scale         1.0 \
    --logq_clip          5.0 \
    --enable_simgcl      1 \
    --simgcl_eps         "${SIMGCL_EPS}" \
    --lambda_view        "${LAMBDA_VIEW}" \
    --simgcl_batch_size_user 2048 \
    --simgcl_batch_size_item 2048 \
    --branchA_view_every_k   "${VIEW_EVERY_K}" \
    --branchA_bcl_batchn     "${USE_BCL_BATCHN}" \
    --branchA_view_bsz       2048 \
    --branchA_bcl_bsz        2048 \
    2>&1 | tee "${LOG_FILE}"

echo
echo "[run_branchA.sh] Finished seed=${SEED}.  Log: ${LOG_FILE}"
echo "[run_branchA.sh] To extract R@20 / NDCG@20 best epoch:"
echo "                 grep -E 'best|Recall@20' \"${LOG_FILE}\" | tail -20"

# =============================================================================
#  5-SEED SWEEP (only after S1/S2/S3 smoke tests pass — see branchA_README.md)
# =============================================================================
#  for s in 0 1 2 3 4; do
#      SEED=$s bash run_branchA.sh
#  done
# =============================================================================

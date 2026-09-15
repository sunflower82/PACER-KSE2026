#!/usr/bin/env bash
# scripts/run_ablations.sh — Eight defensive ablations from §6 of the spec
#
# Iterates the canonical seven-row ablation grid plus the falsifiable
# Permutation-FFT test. All runs share identical training schedules so
# you can directly compare summary lines under
#     ../<dataset>/MM/sum_<ablation_target>.txt
set -euo pipefail
cd "$(dirname "$0")/../src"

DATASET="${DATASET:-Clothing}"
SEED="${SEED:-42}"
EPOCH="${EPOCH:-250}"

# ---------------------------------------------------------------------------
#  Helper
# ---------------------------------------------------------------------------
run () {
    local tag="$1"; shift
    echo "============================================================"
    echo "[ablation] ${tag}"
    echo "============================================================"
    python train.py --dataset "${DATASET}" --seed "${SEED}" \
        --epoch "${EPOCH}" --ablation_target "${tag}" "$@"
}

# 1. MMHCL Baseline (everything DAMPS off)
run mmhcl_baseline \
    --damps_apc 0 --damps_avrf 0 --damps_imcf 0 \
    --damps_soft_routing 0 --damps_momentum 0

# 2. Rebuild frequency sweep
for R in 1 5 10 20; do
    run "rebuild_R${R}" --rebuild_R "${R}"
done

# 3. Momentum encoder ON/OFF
run momentum_off --damps_momentum 0
run momentum_on  --damps_momentum 1

# 4. + APC only
run apc_only --damps_apc 1 --damps_avrf 0 --damps_imcf 0

# 5. + AVRF only
run avrf_only --damps_apc 0 --damps_avrf 1 --damps_imcf 0

# 6. + Residual IMCF only
run imcf_only --damps_apc 0 --damps_avrf 0 --damps_imcf 1

# 7. Full DAMPS-MMHCL
run damps_full

# 8. Permutation-FFT falsifiability test
run permutation_fft --damps_permutation_fft 1

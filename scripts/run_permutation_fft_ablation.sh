#!/usr/bin/env bash
# scripts/run_permutation_fft_ablation.sh
# ---------------------------------------------------------------------------
# Falsifiability ablation (Specification Section 6, Item 8 / compliance check
# INFO 4): re-run the canonical DAMPS-MMHCL pipeline with the FFT replaced
# by a fixed random permutation FFT. Required for the paper's robustness
# section. The hypothesis under test is:
#
#     H0: Random-permutation FFT performs within 1 % of the standard 1-D FFT.
#
# If H0 holds we should switch the spectral basis to DCT-II as a fallback
# (per the spec). Otherwise, the standard 1-D FFT path is validated.
#
# This script runs N_SEEDS paired comparisons (same seed for both arms)
# and writes results into ../<DATASET>/MM/sum_<tag>.txt where each row is
# parsed by ``scripts/_aggregate_permutation_fft.py``.
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/../src"

DATASET="${DATASET:-Clothing}"
EPOCH="${EPOCH:-250}"
N_SEEDS="${N_SEEDS:-3}"
SEED_BASE="${SEED_BASE:-42}"

run () {
    local tag="$1"; shift
    local seed="$1"; shift
    echo "============================================================"
    echo "[ablation] ${tag}  seed=${seed}"
    echo "============================================================"
    python train.py \
        --dataset "${DATASET}" \
        --seed "${seed}" \
        --epoch "${EPOCH}" \
        --ablation_target "${tag}" \
        --damps_apc 1 --damps_avrf 1 --damps_imcf 1 \
        --damps_soft_routing 1 --damps_momentum 1 \
        --damps_data_driven_prior 1 --use_amp 1 \
        "$@"
}

for offset in $(seq 0 $((N_SEEDS - 1))); do
    seed=$((SEED_BASE + offset))
    run "perm_fft_off_seed${seed}" "${seed}" --damps_permutation_fft 0
    run "perm_fft_on_seed${seed}"  "${seed}" --damps_permutation_fft 1
done

echo
echo "Now run scripts/_aggregate_permutation_fft.py to compute the paired"
echo "t-test and tabulate Recall@20 / NDCG@20 for both arms."

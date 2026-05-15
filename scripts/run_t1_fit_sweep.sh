#!/usr/bin/env bash
# Run budget × noise combinations and generate plots for each.
# Noise is expressed as SNR (signal-to-noise ratio); sigma is auto-computed
# from the first block's k-space RMS so levels are physically meaningful.
#
# Usage:
#   bash scripts/run_t1_fit_sweep.sh
#
# Outputs (one subdir per run under scripts/):
#   b160s_nonoise/   b160s_snr50/   b160s_snr20/
#   b250s_nonoise/   b250s_snr50/   b250s_snr20/

set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

BUDGETS=(120) # (120 160 250)
# "0" = no noise; otherwise target SNR value
SNRS=(0 20)

for BUDGET in "${BUDGETS[@]}"; do
    for SNR in "${SNRS[@]}"; do
        if [ "$SNR" -eq 0 ]; then
            LABEL="b${BUDGET}s_nonoise"
            NOISE_FLAG=""
            NOISE_LABEL_ARG=""
        else
            LABEL="b${BUDGET}s_snr${SNR}"
            NOISE_FLAG="--snr $SNR"
            NOISE_LABEL_ARG="--noise-label SNR=${SNR}"
        fi

        echo ""
        echo "══════════════════════════════════════════════"
        echo "  Running: budget=${BUDGET}s  SNR=${SNR:-none}"
        echo "══════════════════════════════════════════════"

        julia --project=. scripts/t1_fit_vs_true.jl --budget $BUDGET $NOISE_FLAG

        python scripts/t1_fit_vs_true.py \
            --subdir "$LABEL" \
            $NOISE_LABEL_ARG

        echo "  → scripts/${LABEL}/t1_fit_vs_true.png"
    done
done

echo ""
echo "All runs complete. Figures:"
for BUDGET in "${BUDGETS[@]}"; do
    for SNR in "${SNRS[@]}"; do
        if [ "$SNR" -eq 0 ]; then
            echo "  scripts/b${BUDGET}s_nonoise/t1_fit_vs_true.png"
        else
            echo "  scripts/b${BUDGET}s_snr${SNR}/t1_fit_vs_true.png"
        fi
    done
done

#!/usr/bin/env bash
# Re-evaluate every memory-ablation global-best policy on the strict held-out
# seed (600000), CPU, logging per-episode (realised T1, abs-pct-error) pairs for
# the error-vs-T1 breakdown. Writes eval_t1err_heldout.json into each run's
# global_best/ dir. The LSTM arm is auto-detected as recurrent from its config.
#
# Usage:  bash scripts/eval_memory_arms_t1err.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source .venv/bin/activate

# Memory arms feeding fig:run-b-memory (560 s, five-sphere, held-out).
RUNS=(
  "mf_runB_5sphere_hist_560s_gpu"    # R1: TI-coverage histogram
  "mf_runB_5sphere_lstm_560s_gpu"    # R2: LSTM / RecurrentPPO
  "mf_runB_5sphere_sigma_560s_gpu"   # sigma-channel
  "mf_runB_5sphere_560s_gpu"         # no-memory control
)

for name in "${RUNS[@]}"; do
  R="runs/e2/$name"
  echo "==================================================================="
  echo "[eval] $name"
  echo "==================================================================="
  PYTHON_JULIAPKG_OFFLINE=yes \
  PYTHON_JULIACALL_THREADS=5 \
  PYTHON_JULIACALL_HANDLE_SIGNALS=yes \
  JULIA_NUM_THREADS=5 \
  python python/eval_e2.py \
    --from-run "$R" \
    --policy "$R/global_best/best_policy.zip" \
    --vecnorm "$R/global_best/best_vecnorm.pkl" \
    --seed 600000 --episodes 24 \
    --roi-radius 1 \
    --cpu \
    --skip-fixed-baseline \
    --summary-name eval_t1err_heldout.json
done

echo "All memory arms done. Per-episode T1/error pairs in each run's"
echo "global_best/eval_t1err_heldout.json (key: per_episode_t1_err)."

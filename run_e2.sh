#!/usr/bin/env bash
# Run an E2 training experiment on this machine and (optionally) commit the
# results back to the repo. Forwards any extra args to train_e2.py.
#
# Usage examples:
#     bash run_e2.sh                              # 200k steps, default config
#     bash run_e2.sh --timesteps 100000           # shorter smoke
#     bash run_e2.sh --reward-mode delta_mape \
#                    --simplified-action \
#                    --mape-alpha 0.5 \
#                    --out runs/e2/optA_unc_100k  # E2.2 candidate
#
# Set PUSH_RESULTS=1 to also git add/commit/push the run directory after
# training. The default RUN_BRANCH is the *current* branch.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# Activate venv + load PYTHON_JULIAPKG_* env vars written by setup.sh
# shellcheck disable=SC1091
source .venv/bin/activate
# shellcheck disable=SC1091
source .envrc.local

# Default output path; overridable by passing --out … through to train_e2.py.
DEFAULT_OUT="runs/e2/$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$DEFAULT_OUT"
ARGS=()
saw_out=0
for a in "$@"; do
    ARGS+=("$a")
    if [ "$saw_out" -eq 1 ]; then OUT_DIR="$a"; saw_out=0; continue; fi
    if [ "$a" = "--out" ]; then saw_out=1; fi
done
if [ "${ARGS[*]:-}" = "" ] || ! printf '%s\n' "${ARGS[@]}" | grep -qx -- '--out'; then
    ARGS+=(--out "$OUT_DIR")
fi

echo "[run_e2] OUT_DIR = $OUT_DIR"
echo "[run_e2] CMD    = python python/train_e2.py ${ARGS[*]}"
python python/train_e2.py "${ARGS[@]}"

# ── Optional: commit + push the run directory ───────────────────────────────
if [ "${PUSH_RESULTS:-0}" = "1" ]; then
    BRANCH="${RUN_BRANCH:-$(git rev-parse --abbrev-ref HEAD)}"
    HOST="$(hostname -s 2>/dev/null || echo remote)"
    echo "[run_e2] Committing results to branch $BRANCH from $HOST…"
    git add "$OUT_DIR"
    git commit -m "E2 results from $HOST ($(basename "$OUT_DIR"))" || {
        echo "[run_e2] Nothing to commit."
        exit 0
    }
    git push origin "$BRANCH"
    echo "[run_e2] Pushed."
fi

#!/usr/bin/env bash
# Full one-shot setup: installs juliaup + Julia 1.11, creates venv, instantiates
# Julia projects, writes .envrc.local, and smoke-tests the Python↔Julia bridge.
#
# Safe to re-run (idempotent). First run: ~5–15 min (Julia precompile).
#
# Usage:
#     bash setup_full.sh
#     source .venv/bin/activate
#     bash run_e2.sh

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# ── 1. juliaup ────────────────────────────────────────────────────────────────
# KEY: non-interactive shells don't source ~/.bashrc, so we must export PATH
# here ourselves after the installer writes to it.
if ! command -v juliaup &>/dev/null && [ ! -x "$HOME/.juliaup/bin/juliaup" ]; then
    echo "[setup] Installing juliaup…"
    curl -fsSL https://install.julialang.org | sh -s -- --yes --default-channel 1.11
fi
# Always set PATH explicitly — don't rely on ~/.bashrc being sourced
export PATH="$HOME/.juliaup/bin:$PATH"
command -v juliaup &>/dev/null || { echo "juliaup install failed." >&2; exit 1; }

# ── 2. Julia 1.11 ─────────────────────────────────────────────────────────────
if ! ls -d "$HOME"/.juliaup/julia-1.11.*/bin/julia &>/dev/null; then
    echo "[setup] Adding Julia 1.11…"
    juliaup add 1.11
fi

JULIA_EXE="$(ls -d "$HOME"/.juliaup/julia-1.11.*/bin/julia 2>/dev/null | head -1)"
[ -x "$JULIA_EXE" ] || { echo "Julia 1.11 not found after install." >&2; exit 1; }
echo "[setup] Julia: $JULIA_EXE"

# ── 3. Python venv + deps ─────────────────────────────────────────────────────
python3 -m venv .venv
source .venv/bin/activate
pip install --quiet --upgrade pip wheel
pip install --quiet -r python/requirements.txt

# ── 4. Julia projects ─────────────────────────────────────────────────────────
"$JULIA_EXE" --project=. -e 'using Pkg; Pkg.instantiate()'
"$JULIA_EXE" --project=python/julia_runtime -e 'using Pkg; Pkg.instantiate()'

# ── 5. Env file ───────────────────────────────────────────────────────────────
cat > .envrc.local <<EOF
export PYTHON_JULIAPKG_OFFLINE=yes
export PYTHON_JULIAPKG_EXE="$JULIA_EXE"
EOF

# ── 6. Smoke-test the bridge ──────────────────────────────────────────────────
echo "[setup] Smoke-testing Python↔Julia bridge…"
cd python
python -c "
from qalibremd_gym.env_e2 import QalibreMDE2Env
env = QalibreMDE2Env(rng_seed=0, simplified_action=True)
obs, _ = env.reset(seed=42)
print('obs_dim =', obs.shape[0], '(expect 159)')
assert obs.shape[0] == 159, f'unexpected obs_dim {obs.shape[0]}'
"
cd ..

echo ""
echo "Done. source .venv/bin/activate && bash run_e2.sh"

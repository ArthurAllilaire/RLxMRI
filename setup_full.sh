#!/usr/bin/env bash
# Full one-shot setup: installs juliaup + Julia 1.11 + 1.12, creates venv,
# instantiates Julia projects, writes .envrc.local, and smoke-tests the
# Python↔Julia bridge.
#
# Julia 1.12 — main project (development, tests)
# Julia 1.11 — python/julia_runtime (juliacall requires ≤ 1.11)
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
    curl -fsSL https://install.julialang.org | sh -s -- --yes --default-channel 1.12
fi
export PATH="$HOME/.juliaup/bin:$PATH"
command -v juliaup &>/dev/null || { echo "juliaup install failed." >&2; exit 1; }

# ── 2. Julia versions ─────────────────────────────────────────────────────────
# juliaup stores Julia in ~/.juliaup/ (newer) or ~/.julia/juliaup/ (older).
_find_julia() {
    find "$HOME/.juliaup" "$HOME/.julia/juliaup" \
         -name julia -path "*/julia-${1}*/bin/julia" 2>/dev/null | head -1
}

for VER in 1.11 1.12; do
    if [ -z "$(_find_julia "$VER")" ]; then
        echo "[setup] Adding Julia $VER…"
        juliaup add "$VER"
    fi
done

JULIA11="$(_find_julia 1.11)"
JULIA12="$(_find_julia 1.12)"
[ -x "$JULIA11" ] || { echo "Julia 1.11 not found after install." >&2; exit 1; }
[ -x "$JULIA12" ] || { echo "Julia 1.12 not found after install." >&2; exit 1; }
echo "[setup] Julia 1.11: $JULIA11"
echo "[setup] Julia 1.12: $JULIA12"

# ── 3. Python venv + deps ─────────────────────────────────────────────────────
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel
pip install -r python/requirements.txt

# ── 4. Julia projects ─────────────────────────────────────────────────────────
# Main project: use 1.12 (matches committed Manifest.toml)
"$JULIA12" --project=. -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()'
# Python bridge runtime: must use 1.11 (juliacall requires ≤ 1.11)
"$JULIA11" --project=python/julia_runtime -e 'using Pkg; Pkg.instantiate()'

# ── 5. Env file ───────────────────────────────────────────────────────────────
cat > .envrc.local <<EOF
export PYTHON_JULIAPKG_OFFLINE=yes
export PYTHON_JULIAPKG_EXE="$JULIA11"
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
echo "To run Julia tests: $JULIA12 --project=. test/runtests.jl"

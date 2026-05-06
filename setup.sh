#!/usr/bin/env bash
# One-shot setup for E2 on a fresh clone.
#
# Prereqs (do these once by hand):
#     curl -fsSL https://install.julialang.org | sh -s -- --yes --default-channel 1.11
#     export PATH="$HOME/.juliaup/bin:$PATH"        # add to ~/.bashrc too
#     juliaup add 1.11
#
# Then from the repo root:
#     bash setup.sh
#     source .venv/bin/activate
#     bash run_e2.sh

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

JULIA_EXE="$(find "$HOME/.juliaup" "$HOME/.julia/juliaup" -name julia -path "*/julia-1.11*/bin/julia" 2>/dev/null | head -1)"
[ -x "$JULIA_EXE" ] || { echo "Julia 1.11 not found — see prereqs at top of this script." >&2; exit 1; }
echo "Julia: $JULIA_EXE"

# Python venv + deps
python3 -m venv .venv
source .venv/bin/activate
pip install --quiet --upgrade pip wheel
pip install --quiet -r python/requirements.txt

# Julia projects
"$JULIA_EXE" --project=. -e 'using Pkg; Pkg.instantiate()'
"$JULIA_EXE" --project=python/julia_runtime -e 'using Pkg; Pkg.instantiate()'

# Env file for run_e2.sh
cat > .envrc.local <<EOF
export PYTHON_JULIAPKG_OFFLINE=yes
export PYTHON_JULIAPKG_EXE="$JULIA_EXE"
EOF

echo "Done. source .venv/bin/activate && bash run_e2.sh"

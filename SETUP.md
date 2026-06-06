# Setup (macOS / Ubuntu)

Manual, step-by-step setup for this repo. Run every step from the repo root.
Differences between macOS (Apple Silicon) and Ubuntu/Linux are called out inline.

The Julia digital-twin package lives in the sibling repo
`../MRISystemPhantom.jl` and is dev-depended via the committed
`Manifest.toml`s — clone it next to this repo before instantiating:

```
y3/
├── RLxMRI/                ← this repo
└── MRISystemPhantom.jl/   ← sibling package (clone it here)
```

Two Julia versions are needed: **1.12** (default, for the main project) and
**1.11** (for the Python bridge — `juliacall` is pinned to Julia ≤ 1.11).

## 0. Prerequisites

- `git`, `curl`, `bash`
- Python ≥ 3.10

**Ubuntu** also needs the venv/pip packages (macOS gets these with its system /
Homebrew Python):

```bash
sudo apt update && sudo apt install -y python3-venv python3-pip
```

## 1. Install Julia 1.12 + 1.11

Install via [`juliaup`](https://github.com/JuliaLang/juliaup) and make 1.12 the
default. The installer edits your shell rc — **macOS** uses zsh (`~/.zshrc`),
**Ubuntu** uses bash (`~/.bashrc`):

```bash
curl -fsSL https://install.julialang.org | sh
source ~/.zshrc            # macOS;  Ubuntu: source ~/.bashrc
juliaup add 1.12
juliaup add 1.11
juliaup default 1.12

juliaup status            # should list 1.11 and 1.12, default = 1.12
julia --version           # 1.12.x
julia +1.11 --version     # 1.11.x
```

## 2. Instantiate both Julia projects

The default `julia` is 1.12; use `julia +1.11` for the bridge runtime so each
`Manifest.toml` resolves under the Julia that actually runs it. The
`KomaMRIBase` fork (`fix/grad-fp`) is pinned in the committed Manifests, so
`instantiate` picks it up automatically (see §2b — no extra `Pkg.add` needed on
a fresh clone).

```bash
julia        --project=.                    -e 'using Pkg; Pkg.instantiate()'
julia +1.11  --project=python/julia_runtime -e 'using Pkg; Pkg.instantiate()'
```

First run takes ~5–15 min (KomaMRI precompile); later runs are seconds.

### 2b. KomaMRIBase fork (temporary — `fix/grad-fp`)

We depend on a fork of `KomaMRIBase` carrying a gradient fix not yet in the
registered release. The fork is pinned by git-tree-sha1 in **both** committed
`Manifest.toml`s, so fresh clones that run the `instantiate` above pick it up
and **do not** need the `add` below. Only run it if you're (re)setting the
branch/rev or the pin is missing.

Run each project under its own Julia version so resolved stdlib versions match
the Julia that runs it. Use the API form (not the `pkg>` string form — the REPL
parser mangles `:subdir#rev`):

```bash
# Main project — default Julia (1.12)
julia --project=. -e '
  using Pkg
  Pkg.add(url="https://github.com/ArthurAllilaire/KomaMRI.jl.git",
          subdir="KomaMRIBase", rev="fix/grad-fp")'

# Python-bridge runtime — Julia 1.11
julia +1.11 --project=python/julia_runtime -e '
  using Pkg
  Pkg.add(url="https://github.com/ArthurAllilaire/KomaMRI.jl.git",
          subdir="KomaMRIBase", rev="fix/grad-fp")'
```

Each `[[deps.KomaMRIBase]]` block should then show (identical `git-tree-sha1`
across both manifests):

```
repo-rev = "fix/grad-fp"
repo-subdir = "KomaMRIBase"
repo-url = "https://github.com/ArthurAllilaire/KomaMRI.jl.git"
```

To pull newer commits on the branch later:

```bash
julia        --project=.                    -e 'using Pkg; Pkg.update("KomaMRIBase")'
julia +1.11  --project=python/julia_runtime -e 'using Pkg; Pkg.update("KomaMRIBase")'
```

**Reverting once the fix lands upstream:** when a fixed `KomaMRIBase` is
registered, detach the fork in both projects, bump the `[compat]` floor for
`KomaMRIBase` in `Project.toml` and `python/julia_runtime/Project.toml` to the
release containing the fix, and commit the updated manifests:

```bash
julia        --project=.                    -e 'using Pkg; Pkg.free("KomaMRIBase")'
julia +1.11  --project=python/julia_runtime -e 'using Pkg; Pkg.free("KomaMRIBase")'
```

## 3. Python venv

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel
pip install -r python/requirements.txt
```

## 4. Write `.envrc.local`

The training scripts read `PYTHON_JULIAPKG_EXE` to find the Julia 1.11 binary
and `PYTHON_JULIAPKG_OFFLINE=yes` to skip re-resolving Julia packages on every
launch. `run_e2.sh` sources this file automatically. Resolve the real binary
path with `Sys.BINDIR` — this is correct on both platforms (on macOS juliaup
ships Julia inside a `.app` bundle, so a hand-written path is easy to get wrong;
on Ubuntu it lands under `~/.julia/juliaup/julia-1.11.x+0.x64.linux.gnu/bin/`):

```bash
JULIA11="$(julia +1.11 -e 'print(joinpath(Sys.BINDIR, "julia"))')"
cat > .envrc.local <<EOF
export PYTHON_JULIAPKG_OFFLINE=yes
export PYTHON_JULIAPKG_EXE="$JULIA11"
EOF
source .envrc.local
```

## 5. Smoke test (optional)

```bash
cd python && python -c "
from qalibremd_gym.env_e2 import QalibreMDE2Env
env = QalibreMDE2Env(rng_seed=0, simplified_action=True)
obs, _ = env.reset(seed=42)
print('obs_dim =', obs.shape[0], '(expect 159 with default Nfe=16, Npe=8)')
" && cd ..
```

`obs_dim = 159` means the Python ↔ Julia bridge is working. First run hangs
~30 s while Julia JIT-precompiles KomaMRI inside the Python process; later
runs are < 2 s.

You're set up. To train: `bash run_e2.sh` (see the script for flags).

## Troubleshooting

| Symptom | Fix |
|---|---|
| `juliaup: command not found` after install | `source ~/.zshrc` (macOS) / `~/.bashrc` (Ubuntu), or `export PATH=$HOME/.juliaup/bin:$PATH` |
| `Missing source file … MRISystemPhantom` | Sibling package not cloned at `../MRISystemPhantom.jl`, or a `Manifest.toml` has a stale path — it should read `path = "../MRISystemPhantom.jl"` (root) / `"../../../MRISystemPhantom.jl"` (runtime). |
| `juliacall` can't find Julia | `source .envrc.local` |
| `juliacall.JuliaError: package KomaMRI not found` from Python | The `python/julia_runtime` instantiate (step 2) was skipped — re-run it. |
| `Pkg.instantiate()` hangs > 5 min on a JLL | Network timeout downloading artifacts — Ctrl-C and retry. |
| `python3 -m venv` fails on Ubuntu | Install `python3-venv` (see §0). |
| `ImportError: No module named qalibremd_gym` | You forgot to `cd python` first — the package is at `python/qalibremd_gym/`. |

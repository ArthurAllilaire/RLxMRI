# Remote-GPU setup for E2

End-to-end recipe for cloning this repo onto a fresh box (e.g. a remote GPU
node), training E2, and pushing the results back to GitHub. For local /
development setup with more context, see `docs/GETTING_STARTED.md`.

## Prerequisites on the remote box

- `git`, `curl`, `bash`
- `python3` (≥ 3.10) with `python3-venv`
- A working internet connection (juliaup downloads Julia; pip pulls the
  Python deps)

GPU is **optional**. The KomaMRI Bloch simulator runs on CPU by default and
this is fine for E2's plate-sized phantoms. CUDA is *not* required — if you
do have it, KomaMRI will pick it up automatically.

## 1. Clone

```bash
git clone https://github.com/ArthurAllilaire/RLxMRI.git icr
cd icr
```

If you intend to push results back from this machine, configure git auth
first (SSH key, `gh auth login`, or an HTTPS PAT):

```bash
gh auth login                  # easiest for HTTPS clones
# or:
git remote set-url origin git@github.com:ArthurAllilaire/RLxMRI.git
```

## 2. One-shot setup

There are two setup scripts depending on what's already installed:

| Script | Use when |
|---|---|
| `setup_full.sh` | Fresh box — installs juliaup + Julia 1.11/1.12, then sets everything up |
| `setup.sh` | juliaup + Julia 1.11 already present; just creates venv + instantiates |

For a fresh remote box, use the full script:

```bash
bash setup_full.sh
```

Both scripts are **idempotent** — safe to re-run. `setup_full.sh` will:

1. Install `juliaup` if not already present, then `export PATH` in-process
   (non-interactive shells don't source `~/.bashrc`, so this is required).
2. Install Julia 1.11 and 1.12 via `juliaup add` if not already present.
3. Create `.venv/` and `pip install -r python/requirements.txt`.
4. `Pkg.instantiate()` the main Julia project with 1.12 and the Python-bridge
   runtime at `python/julia_runtime/` with 1.11.
5. Write `.envrc.local` with the env vars `juliacall` needs:
   `PYTHON_JULIAPKG_OFFLINE=yes` and `PYTHON_JULIAPKG_EXE=<julia-1.11 path>`.
6. Smoke-test the Python ↔ Julia bridge (`obs_dim = 159`).

First run takes ~5–15 min (Julia precompile dominates). Subsequent runs are
seconds.

> **Why two scripts?** `bash script.sh` runs in a non-interactive child
> process that never sources `~/.bashrc`. After the juliaup installer writes
> to `~/.bashrc`, the running script can't see the new PATH — unless you
> explicitly `export PATH="$HOME/.juliaup/bin:$PATH"` in the script itself.
> `setup_full.sh` does this; `setup.sh` assumes juliaup is already on PATH.

## 2b. KomaMRIBase fork (temporary — `fix/grad-fp`)

We currently depend on a fork of `KomaMRIBase` that carries a gradient
fix not yet in the registered release. This must be applied to **both**
Julia projects (the main project *and* the Python-bridge runtime), or the
two `Manifest.toml`s will resolve to different `KomaMRIBase` versions.

Only `KomaMRIBase` comes from the fork; `KomaMRI`, `KomaMRICore`, etc. stay
on the registry. That's fine as long as the fork keeps its version in the
`0.11.x` range the registry packages expect.

Run each project **under its own Julia version** so the resolved stdlib
versions in each `Manifest.toml` match the Julia that actually runs the
project (main = 1.12, Python-bridge runtime = 1.11 — juliacall is pinned to
≤ 1.11). Mixing them works but produces noisy stdlib churn in the diff.

Use the API form (not the `pkg>` REPL string form) — the REPL parser does
not split `:subdir#rev` cleanly and produces a malformed `repo-rev` entry.

From the repo root:

```bash
# Main project — use the default Julia (1.12)
julia --project=. -e '
  using Pkg
  Pkg.add(url="https://github.com/ArthurAllilaire/KomaMRI.jl.git",
          subdir="KomaMRIBase",
          rev="fix/grad-fp")'

# Python-bridge runtime — must use Julia 1.11 (juliacall constraint)
julia +1.11 --project=python/julia_runtime -e '
  using Pkg
  Pkg.add(url="https://github.com/ArthurAllilaire/KomaMRI.jl.git",
          subdir="KomaMRIBase",
          rev="fix/grad-fp")'
```

Verify both `Manifest.toml`s show three separate fields for `KomaMRIBase`
(and identical `git-tree-sha1`):

```
repo-rev = "fix/grad-fp"
repo-subdir = "KomaMRIBase"
repo-url = "https://github.com/ArthurAllilaire/KomaMRI.jl.git"
```

This pins each `[[deps.KomaMRIBase]]` block to a fork git-tree-sha1, so the
override travels with the committed `Manifest.toml` — fresh clones that run
`Pkg.instantiate()` pick it up automatically and **do not** need to re-run
the `add`. Only re-run it if you're changing the branch/rev.

To pull newer commits on the branch later, run in each project (mind the
Julia channel):

```bash
julia        --project=.                     -e 'using Pkg; Pkg.update("KomaMRIBase")'
julia +1.11  --project=python/julia_runtime  -e 'using Pkg; Pkg.update("KomaMRIBase")'
```

### Reverting once the fix lands in `main`

When the gradient fix is merged upstream and a new `KomaMRIBase` is
registered, drop the fork and return to the registered release in **both**
projects:

```bash
julia        --project=.                     -e 'using Pkg; Pkg.free("KomaMRIBase")'
julia +1.11  --project=python/julia_runtime  -e 'using Pkg; Pkg.free("KomaMRIBase")'
```

`free` detaches `KomaMRIBase` from the git branch and resolves it back to the
latest registered version. Then bump the `[compat]` floor for `KomaMRIBase`
in `Project.toml` and `python/julia_runtime/Project.toml` to the release that
contains the fix, and commit the updated `Manifest.toml`s.

## 3. Train

```bash
source .venv/bin/activate     # if you opened a new shell
bash run_e2.sh                 # 200k-step default
```

`run_e2.sh` wraps `python python/train_e2.py` with the right env vars and
forwards any extra flags. Examples for the §16.4 candidates:

```bash
# Option A only (worst-case-weighted MAPE)
bash run_e2.sh --timesteps 100000 \
               --reward-mode delta_mape \
               --simplified-action \
               --mape-alpha 0.5 \
               --out runs/e2/optA_100k

# Option A + Option C (uncertainty channel — already in the obs by default)
# i.e. same command as above; the σ-channel is wired into _e2_observation
# unconditionally as of EXPERT_REPORT §16.4 implementation.
```

Every run drops `policy.zip`, `vecnorm.pkl`, `eval_history.json`, and a
`tb/` tensorboard log under the `--out` directory. Checkpoints (`ckpt_N.zip`
+ `vecnorm_ckpt_N.pkl`) are written every 50k steps by default so a killed
run can be resumed:

```bash
# resume an interrupted run — same flags as the original, plus --resume
bash run_e2.sh --timesteps 200000 --out runs/e2/ppo_200k --resume
```

`--checkpoint-interval N` changes the cadence; `0` disables checkpointing.

## 4. Commit results back

Two options:

### Inline (one-shot training + push)

```bash
PUSH_RESULTS=1 bash run_e2.sh --timesteps 100000 \
                              --out runs/e2/optA_100k
```

This runs training, then `git add runs/e2/optA_100k`, commits with a message
tagged by hostname, and pushes to the *current* branch (override with
`RUN_BRANCH=<name>`).

### Manually (after inspecting the run)

```bash
git add runs/e2/optA_100k
git commit -m "E2 Option A 100k smoke (remote-GPU)"
git push
```

If you're running multiple experiments in parallel and don't want to step on
`main`, push to a per-experiment branch instead:

```bash
git checkout -b e2/optA-100k
PUSH_RESULTS=1 RUN_BRANCH=e2/optA-100k bash run_e2.sh ...
```

## 5. Pull results locally

On your dev machine:

```bash
git pull                                        # fetches the runs/ dir
python python/eval_e2.py --policy runs/e2/optA_100k/policy.zip --episodes 30
python python/diagnose_e2.py --policy runs/e2/optA_100k/policy.zip
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `juliaup: command not found` after install | `source ~/.bashrc`, or `export PATH=$HOME/.juliaup/bin:$PATH` |
| `juliacall` can't find Julia | `source .envrc.local` (or re-run `setup_full.sh`) |
| `Pkg.instantiate` fails on a JLL | First `instantiate` needs network; retry once. |
| Out of disk on `runs/` | `runs/*/tb/` (tensorboard) is the heaviest; delete it before pushing if you don't need it on the dev box. |
| `git push` rejected | Another machine pushed first — `git pull --rebase` and try again. |

---

## Manual fallback (skip `setup.sh`)

If `setup.sh` keeps failing and you want to drive each step by hand, run
these in order from the repo root. Start with the juliaup path; if juliaup
leaves a broken 1.11 install, use the direct release tarball fallback below.

### 1. Install Julia 1.11 + 1.12

```bash
curl -fsSL https://install.julialang.org | sh -s -- --yes --default-channel 1.11
export PATH="$HOME/.juliaup/bin:$PATH"        # add to ~/.bashrc as well
juliaup add 1.11
juliaup add 1.12
```

Resolve the absolute Julia binary paths once (avoids juliaup launcher quirks
in non-interactive shells). The main project uses 1.12; the Python bridge
runtime uses 1.11 because `juliacall` is pinned to Julia <= 1.11:

```bash
JULIA11="$(find "$HOME/.juliaup" "$HOME/.julia/juliaup" -name julia -path "*/julia-1.11*/bin/julia" 2>/dev/null | head -1)"
JULIA12="$(find "$HOME/.juliaup" "$HOME/.julia/juliaup" -name julia -path "*/julia-1.12*/bin/julia" 2>/dev/null | head -1)"
echo "$JULIA11"
echo "$JULIA12"
$JULIA11 --version                             # should print 1.11.x
$JULIA12 --version                             # should print 1.12.x
```

If the juliaup-managed 1.11 install is corrupt and `using Pkg` fails with a
`MbedTLS_jll`-style "not installed" error, use the official Julia release
tarball directly instead of the juliaup launcher:

```bash
REPO_ROOT="$PWD"
mkdir -p "$HOME/julia-releases"
cd "$HOME/julia-releases"
curl -fL -o julia-1.11-linux-x86_64.tar.gz https://julialang-s3.julialang.org/bin/linux/x64/1.11/julia-1.11-latest-linux-x86_64.tar.gz
tar -xzf julia-1.11-linux-x86_64.tar.gz
JULIA11="$(find "$HOME/julia-releases" -maxdepth 3 -path "*/julia-1.11*/bin/julia" -type f | sort | tail -1)"
cd "$REPO_ROOT"
```

Verify both `Pkg` stdlibs load:

```bash
$JULIA11 -e 'using Pkg'                        # should print nothing
$JULIA12 -e 'using Pkg'                        # should print nothing
```

### 2. Python venv + deps

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel
pip install -r python/requirements.txt
```

### 3. Instantiate Julia projects

```bash
$JULIA12 --project=. -e 'using Pkg; Pkg.instantiate()'
$JULIA11 --project=python/julia_runtime -e 'using Pkg; Pkg.instantiate()'
```

First run takes 5–15 min (KomaMRI precompile). Subsequent runs are seconds.

### 4. Write `.envrc.local`

The training scripts read `PYTHON_JULIAPKG_EXE` to find the Julia 1.11
binary, and `PYTHON_JULIAPKG_OFFLINE=yes` to skip re-resolving Julia
packages on every Python launch. `run_e2.sh` sources this file
automatically.

```bash
cat > .envrc.local <<EOF
export PYTHON_JULIAPKG_OFFLINE=yes
export PYTHON_JULIAPKG_EXE="$JULIA11"
EOF
source .envrc.local
```

### 5. Smoke-test the bridge

```bash
cd python && python -c "
from qalibremd_gym.env_e2 import QalibreMDE2Env
env = QalibreMDE2Env(rng_seed=0, simplified_action=True)
obs, _ = env.reset(seed=42)
print('obs_dim =', obs.shape[0], '(expect 159 with default Nfe=16, Npe=8)')
" && cd ..
```

If that prints `obs_dim = 159` you're fully set up — proceed to step 3
("Train") at the top of this file.

### Common manual-step failures

| Symptom | Likely cause + fix |
|---|---|
| `Pkg.instantiate()` hangs > 5 min on a JLL | Network timeout. Ctrl-C, retry. JLL artifacts are downloaded one at a time. |
| `MbedTLS_jll … is required but does not seem to be installed` | Julia install is corrupt. Try `juliaup remove 1.11 && rm -rf ~/.julia/juliaup/julia-1.11.* ~/.juliaup/julia-1.11.* ~/.julia/compiled/v1.11 && juliaup add 1.11`; if it still fails, use the official release tarball path above and set `JULIA11` to that binary. |
| `ImportError: No module named qalibremd_gym` in step 5 | You forgot to `cd python` first — the package is at `python/qalibremd_gym/`. |
| `juliacall.JuliaError: package KomaMRI not found` from Python | Step 3b skipped — re-run the `--project=python/julia_runtime` instantiate. |
| `pip install` fails on `gymnasium[box2d]` or similar | Not needed for E2 — `python/requirements.txt` only pulls the core SB3 + gymnasium + juliacall. If you see this, you're using a different requirements file. |
| Smoke test hangs ~30 s on first run | Normal — Julia is JIT-precompiling KomaMRI inside the Python process. Subsequent runs are < 2 s. |

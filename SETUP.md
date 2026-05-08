# Remote-GPU setup for E2

End-to-end recipe for cloning this repo onto a fresh box (e.g. a remote GPU
node), training E2, and pushing the results back to GitHub. For local /
development setup with more context, see `docs/GETTING_STARTED.md`.

## Prerequisites on the remote box

- `git`, `curl`, `bash`
- `python3` (≥ 3.10) with `python3-venv`
- A working internet connection (juliaup downloads Julia 1.11; pip pulls the
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
| `setup_full.sh` | Fresh box — installs juliaup + Julia 1.11, then sets everything up |
| `setup.sh` | juliaup + Julia 1.11 already present; just creates venv + instantiates |

For a fresh remote box, use the full script:

```bash
bash setup_full.sh
```

Both scripts are **idempotent** — safe to re-run. `setup_full.sh` will:

1. Install `juliaup` if not already present, then `export PATH` in-process
   (non-interactive shells don't source `~/.bashrc`, so this is required).
2. Install Julia 1.11 via `juliaup add 1.11` if not already present.
3. Create `.venv/` and `pip install -r python/requirements.txt`.
4. `Pkg.instantiate()` the main Julia project and the Python-bridge runtime
   at `python/julia_runtime/`.
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
these in order from the repo root. They are exactly what the script does,
just without the conditionals — so you'll see immediately which step is
the real problem.

### 1. Install juliaup + Julia 1.11

```bash
curl -fsSL https://install.julialang.org | sh -s -- --yes --default-channel 1.11
export PATH="$HOME/.juliaup/bin:$PATH"        # add to ~/.bashrc as well
juliaup add 1.11
```

Resolve the absolute Julia binary path once (avoids juliaup launcher quirks
in non-interactive shells):

```bash
JULIA_EXE="$(ls -d $HOME/.juliaup/julia-1.11.*/bin/julia | head -1)"
echo "$JULIA_EXE"
$JULIA_EXE --version                          # should print 1.11.x
```

Verify Pkg loads. If it fails with a `MbedTLS_jll`-style "not installed"
error, the Julia install is corrupt — see the troubleshooting section
above for the hard-reinstall recipe.

```bash
$JULIA_EXE -e 'using Pkg'                     # should print nothing
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
$JULIA_EXE --project=. -e 'using Pkg; Pkg.instantiate()'
$JULIA_EXE --project=python/julia_runtime -e 'using Pkg; Pkg.instantiate()'
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
export PYTHON_JULIAPKG_EXE="$JULIA_EXE"
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
| `MbedTLS_jll … is required but does not seem to be installed` | Julia install is corrupt. `juliaup remove 1.11 && rm -rf ~/.julia/juliaup/julia-1.11.* ~/.julia/compiled && juliaup add 1.11`. |
| `ImportError: No module named qalibremd_gym` in step 5 | You forgot to `cd python` first — the package is at `python/qalibremd_gym/`. |
| `juliacall.JuliaError: package KomaMRI not found` from Python | Step 3b skipped — re-run the `--project=python/julia_runtime` instantiate. |
| `pip install` fails on `gymnasium[box2d]` or similar | Not needed for E2 — `python/requirements.txt` only pulls the core SB3 + gymnasium + juliacall. If you see this, you're using a different requirements file. |
| Smoke test hangs ~30 s on first run | Normal — Julia is JIT-precompiling KomaMRI inside the Python process. Subsequent runs are < 2 s. |

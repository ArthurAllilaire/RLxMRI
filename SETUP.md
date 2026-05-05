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

```bash
bash setup.sh
```

This is **idempotent** — safe to re-run. It will:

1. Install `juliaup` and Julia 1.11 (skipped if already present).
2. Create `.venv/` and `pip install -r python/requirements.txt`.
3. `Pkg.instantiate()` the main Julia project and the Python-bridge runtime
   at `python/julia_runtime/`.
4. Write `.envrc.local` with the env vars `juliacall` needs:
   `PYTHON_JULIAPKG_OFFLINE=yes` and `PYTHON_JULIAPKG_EXE=<julia-1.11 path>`.
5. Smoke-test the Python ↔ Julia bridge by constructing one `E2Env` and
   running a reset.

First run takes ~5–15 min (Julia precompile dominates). Subsequent runs are
seconds.

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
`tb/` tensorboard log under the `--out` directory.

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
| `juliacall` can't find Julia | `source .envrc.local` (or re-run `setup.sh`) |
| `Pkg.instantiate` fails on a JLL | First `instantiate` needs network; retry once. |
| Out of disk on `runs/` | `runs/*/tb/` (tensorboard) is the heaviest; delete it before pushing if you don't need it on the dev box. |
| `git push` rejected | Another machine pushed first — `git pull --rebase` and try again. |

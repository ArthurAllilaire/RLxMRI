# Getting Started

Quick setup guide for the QalibreMD RL experiments. The stack is:
**Julia** (KomaMRI digital twin) + **Python** (Gymnasium / Stable-Baselines3),
bridged via [juliacall](https://juliapy.github.io/PythonCall.jl/).

---

## Prerequisites

| Tool | Minimum version | Install |
|------|-----------------|---------|
| Julia | 1.11 (see note) | [juliaup](https://julialang.org/downloads/) |
| Python | 3.10+ | system / pyenv |
| git | any | system |

> **Julia version note:** juliacall 0.9.31 requires Julia ≤ 1.11. The repo's
> main Julia project can run on 1.12+, but the Python bridge uses a separate
> `python/julia_runtime/` project pinned to 1.11. Install 1.11 via juliaup:
> ```bash
> juliaup add 1.11
> ```

---

## 1 — Clone and set up Julia

```bash
git clone <repo-url> && cd icr

# Instantiate the main Julia project (tests, E0 examples, figure scripts)
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Instantiate the Python-bridge Julia sub-project
julia +1.11 --project=python/julia_runtime -e 'using Pkg; Pkg.instantiate()'
```

---

## 2 — Create a Python virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install juliacall gymnasium 'stable-baselines3[extra]'
```

juliacall will detect the Julia 1.11 binary automatically (it looks for the
`PYTHON_JULIAPKG_EXE` env var, or the `julia_runtime/` juliapkg manifest).

---

## 3 — Smoke-test the bridge

```bash
python python/test_wrapper.py
```

This boots Julia, constructs one `E1Env`, runs a single reset+step, and
prints the observation shape and reward. Expect a Julia precompilation
pause (~30 s) on first run; subsequent runs are fast.

---

## 4 — Run the experiments

### E1 — single-sphere T1 estimation (RL)

**Fixed-grid baseline** (no learning, ~1 min):
```bash
python python/baseline_e1.py --episodes 200
```

**Train PPO agent** (~10 min on CPU for 50 k steps):
```bash
python python/train_e1.py --timesteps 50000 --out runs/e1/ppo
```

Writes `runs/e1/ppo/policy.zip` and `eval_history.json`.

**Evaluate trained policy vs baseline** (held-out seeds):
```bash
python python/eval_e1.py --policy runs/e1/ppo/policy.zip --episodes 500
```

### E0 — conventional sequence baseline (Julia only)

```bash
julia --project=. examples/conventional_baseline.jl
```

---

## 5 — Run Julia tests

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `juliacall` can't find Julia 1.11 | `export PYTHON_JULIAPKG_EXE=$(julia +1.11 -e 'println(Sys.BINDIR * "/julia")')` |
| `PackageCompilationError` on first import | Run `julia +1.11 --project=python/julia_runtime -e 'using Pkg; Pkg.instantiate()'` |
| Slow episodes during training | `--backend simulate` is off by default; the default `:analytical` backend is fast. Check `E1Env` kwargs. |
| `CUDA not found` warning | Harmless; training runs on CPU by default. |

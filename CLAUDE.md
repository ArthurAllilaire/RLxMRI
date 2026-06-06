# CLAUDE.md — Research Assistant Context

This is an Imperial College London BEng final-year project (FYP). The goal is to apply reinforcement learning to adaptive quantitative MRI sequence design, using a physics simulator and a digital phantom. **The primary objective is to maximise the final report mark while doing interesting, publishable work.**

---

## The project in one paragraph

We built a digital twin of the Calibur QalibreMD Model 130 MRI phantom in KomaMRI (a Julia-based Bloch-equation MRI simulator). We are training RL agents to design adaptive pulse sequences that efficiently estimate quantitative tissue parameters (T1, T2) from that phantom — beating the fixed conventional sequences used in clinical practice. The agent runs inside a Gymnasium environment (Python) and calls KomaMRI via juliacall at every step.

---

## People

| Person | Role | Cadence |
|---|---|---|
| **Andreas Wetscherek** (awetscherek) | ICR researcher, domain supervisor — MRI/qMRI expert, drives the technical direction | Monday 1-on-1 + Thursday standup |
| **Wayne Luk** | Imperial professor, academic supervisor — not MRI-specific, cares about report structure, weekly progress updates, grade | Weekly email update |

Wayne's key asks (from `project_context/wayne_emails/interim_report_feedback.md`):
- Challenges C1–C3, achievements A1–A3, three technical chapters
- Weekly progress emails with: (a) done this week, (b) planned next week, (c) deviations from timetable
- Basic report draft with bullet points for all chapters by **1 June 2026**
- Final submission: **12 June 2026**

---

## Key documents to read first

| File | What it is |
|---|---|
| `PLAN.md` | Main technical plan: MDP formulation, experiment ladder E0–E5, reward design, Julia↔Python architecture, fidelity tradeoffs. **Single source of truth for the experiment design.** |
| `E2_PLAN.md` | Detailed weekly execution plan for the current experiment (E2). Read this before writing any E2 code. |
| `docs/E1_RESULTS.md` | E1 post-mortem — the agent collapsed to a degenerate fixed policy. Critical context for E2 reward design. |
| `project_context/PROJECT.md` | Submission timeline, chapter structure, C1–C3/A1–A3 definitions, weekly milestones with concrete dates. |
| `project_context/wayne_emails/` | Full email history with Wayne. Read before drafting any email to him. |
| `project_context/meeting_notes/M1_polished.md` | Expanded notes from the first supervisor meeting (M1). |
| `docs/MRI.md` | MRI physics background (T1, T2, inversion recovery, etc.). |
| `docs/RL_LEARNING.md` | RL concepts relevant to the project. |

---

## Repo structure

**Two repos.** The digital-twin Julia package (`MRISystemPhantom`) lives in a
**sibling directory** `../MRISystemPhantom.jl`. This repo (`RLxMRI`) is the RL +
driver + report side, and dev-depends on that package via the `Manifest.toml`
(path `../MRISystemPhantom.jl`; the runtime manifest uses `../../../MRISystemPhantom.jl`).

```
RLxMRI/                          ← THIS repo (RL, training, report)
├── python/                      RL training/eval (Gymnasium + Stable-Baselines3)
│   ├── qalibremd_gym/
│   │   ├── env.py               E1 Gymnasium wrapper over juliacall
│   │   ├── env_e2.py            E2 Gymnasium wrapper
│   │   ├── schedules.py         Curriculum / noise schedules
│   │   └── juliapkg.json        Julia dep spec for the Python side
│   ├── train_e1.py / train_e2.py / train_e2_mf.py   PPO training scripts
│   ├── eval_e1.py / eval_e2.py                       Evaluation + metrics
│   ├── baseline_e1.py / baseline_e2.py              Fixed-grid baselines
│   ├── diagnose_*.py, e2_config.py                  Diagnostics + E2 config
│   └── julia_runtime/           Julia 1.11 project for juliacall (its own Manifest)
│
├── julia/                       RL Julia glue (loaded into Main for the gym)
│   ├── rl_boot.jl               `using MRISystemPhantom` + symbol aliases
│   └── rl/e2.jl                 E2 RL env (e2_reset!/e2_step!)
│
├── test/                        Julia test suite (runtests.jl + test_*.jl)
├── scripts/                     Diagnostic / Koma-investigation Julia scripts
├── runs/                        Trained policies + eval history + tensorboard logs
├── examples/                    Julia demo scripts (baselines, plotting)
├── report_latex/ report/ report_plots/   Final report sources + figures
├── docs/                        Background reading + experiment notes (E1_RESULTS.md…)
├── project_context/             Supervisor emails, meeting notes, marking docs, timeline
├── PLAN.md / E2_PLAN.md         Master + current-experiment plans
├── run_e2.sh                    E2 launcher (sources .envrc.local)
├── .envrc.local                 Machine-local env vars (gitignored)
└── Project.toml / Manifest.toml Julia env that dev-depends on ../MRISystemPhantom.jl

../MRISystemPhantom.jl/          ← SIBLING repo: the digital-twin Julia package
└── src/
    ├── MRISystemPhantom.jl      Package root — exports everything
    ├── config.jl                PhantomConfig, AugmentConfig structs
    ├── builder.jl               build_phantom(), augment()
    ├── augment.jl               Pose randomisation + material jitter
    ├── forward_model.jl, imaging.jl, water_cache.jl, sphere_descriptor.jl
    ├── geometry/                sphere.jl, plane.jl, plate_layouts.jl, projection.jl
    ├── materials/               t1_array.jl, t2_array.jl, pd_array.jl,
    │                            background.jl, fiducial.jl (PLACEHOLDER — not real yet)
    ├── sequences/blocks.jl      ir_sequence(), se_sequence(), generalized_ir_signal()…
    ├── fitting/fits.jl          fit_t1_generalized_ir() — Levenberg–Marquardt T1 fitter
    ├── baselines/               conventional.jl, cr_optimal.jl, cr_optimal_alpha.jl
    └── diagnostics/snr.jl       SNR / Rayleigh helpers
```

---

## Technology stack

**Julia side (simulation):**
- `MRISystemPhantom` — the digital-twin package (in sibling repo `../MRISystemPhantom.jl`, dev-depended via Manifest)
- `KomaMRI.jl` — Bloch-equation MRI simulator; `simulate(phantom, seq, scanner)` is the core call
- `PythonCall.jl` / juliacall — the bridge; Julia stays in-process across all env steps (amortises JIT cost)
- Julia 1.11 for the Python-facing runtime (juliacall 0.9.x constraint); Julia 1.12 (default) for direct Julia use. Both installed via `juliaup` on macOS (Apple Silicon).

**Python side (RL):**
- `gymnasium` — Env API
- `stable-baselines3` — PPO (currently); SAC available
- `juliacall` — calls Julia from Python
- venv at `.venv/`; activate with `source .venv/bin/activate`
- Julia runtime for Python at `python/julia_runtime/` (separate project that dev-depends on this repo)

**Important env vars for Python scripts** (set in `.envrc.local`, gitignored, sourced by `run_e2.sh`):
```bash
PYTHON_JULIAPKG_OFFLINE=yes   # don't try to re-resolve Julia packages at runtime
# macOS / Apple Silicon: juliaup installs Julia as a .app bundle, so the 1.11
# binary lives inside Contents/Resources/julia/bin/julia
PYTHON_JULIAPKG_EXE="$HOME/.julia/juliaup/julia-1.11.9+0.aarch64.apple.darwin14/Julia-1.11.app/Contents/Resources/julia/bin/julia"
```

**Running things:**
```bash
# Activate Python venv
source .venv/bin/activate

# E1 baseline
PYTHON_JULIAPKG_OFFLINE=yes python python/baseline_e1.py --episodes 200

# E1 training
PYTHON_JULIAPKG_OFFLINE=yes python python/train_e1.py --timesteps 50000 --out runs/e1/ppo

# Julia tests
julia --project=. test/runtests.jl
```

---

## Experiment status

| ID | Description | Status | Key file |
|---|---|---|---|
| E0 | Conventional IR-TSE + multi-TE SE baseline | **Done** | `../MRISystemPhantom.jl/src/baselines/conventional.jl` |
| E1 | Single-voxel RL (PPO, discrete TI×α actions) | **Done — degenerate policy** | `julia/rl/e1.jl`, `python/qalibremd_gym/env.py` |
| E2 | Full T1-plate RL with gradients, 2D imaging, noise, localisation | **In progress** | `E2_PLAN.md` |
| E3 | MRF-style fingerprinting (learned FA/TR) | Planned | `PLAN.md §4 E3` |
| E4 | Adaptive k-space trajectories (radial spoke) | Stretch | `PLAN.md §4 E4` |
| E5 | Full 6-DoF pose estimation + parameter mapping | Stretch | `PLAN.md §4 E5` |

---

## Critical E1 finding — read before touching the reward function

The E1 agent achieved ~0.55% MAPE but for the **wrong reason**: it collapsed to a degenerate fixed policy (same 12 actions every episode regardless of T1_true). The T1 fitter (`fit_t1_generalized_ir`) is powerful enough to recover T1 from a single informative measurement, so the agent learned to exploit the fitter rather than design adaptive sequences.

Root causes (from `docs/E1_RESULTS.md`):
1. The `terminal_bonus = +1.0` dominated the per-step error penalty, so the agent only needed to collect the bonus — not optimise the sequence.
2. Without noise, the fitter works perfectly from almost any single measurement.

Fixes being applied in E2:
- Complex Gaussian noise on simulator output (makes redundant repeated actions useless)
- Harder task (14 spheres, spatial encoding required) so the fitter can't trivially solve from one measurement
- Consider reducing or removing the terminal bonus in favour of purely dense rewards

---

## The C1–C3 / A1–A3 structure (for report and Wayne emails)

**Challenges:**
- **C1** — Adaptive pulse sequence design: no fixed-protocol optimum; the optimal sequence depends on unknown tissue properties
- **C2** — Scalable simulation-in-the-loop RL: live Bloch solver is expensive; need to train feasibly while keeping fidelity
- **C3** — Spatial localisation under pose uncertainty: phantom/patient position unknown; agent must localise and map jointly

**Achievements → chapters:**
- **A1 / Ch3** — Digital twin + conventional baseline (E0): validates simulator, establishes quantitative yardstick
- **A2 / Ch4** — RL agent for adaptive T1/T2 mapping (E1 → E2): addresses C1, C2
- **A3 / Ch5** — Spatial localisation and domain generalisation (E2 localisation, E3): addresses C3

---

## Known issues / gotchas

- **Fiducial calibration values** in `../MRISystemPhantom.jl/src/materials/fiducial.jl` are placeholders. Not blocking for E1–E3, but must be replaced before any sim-to-real comparison.
- **PDG not EPG**: the interim report incorrectly calls MRzero an EPG simulator. It uses Phase Distribution Graphs (PDG). Fix this everywhere before submission.
- **Julia 1.11 vs 1.12 split**: juliacall requires Julia ≤ 1.11. The Python-facing runtime at `python/julia_runtime/` pins 1.11; the main project can use 1.12. Do not break this separation.
- **`simulate()` must include all phantom spins**: when using a slice-selective pulse, do not subset the phantom by slice before calling `simulate()` — off-slice spins affect steady-state magnetisation.
- **E1 policy is degenerate**: the saved policy at `runs/e1/ppo/policy.zip` is not an adaptive policy and should not be cited as a positive result. It is documented as a failure mode in `docs/E1_RESULTS.md`.
- **`action` indexing**: Python 0-based actions are converted to Julia 1-based in `env.py:step()` with `a_julia = int(action) + 1`. Don't double-offset.

---

## Report writing guidance (from Wayne)

Every technical chapter must explicitly state:
1. Which challenge (C1/C2/C3) this chapter addresses
2. What is **novel** about this approach vs prior work
3. The **quantified benefit** (Pareto curve, MAPE table, robustness plot) vs a published or conventional baseline

The `project_context/marking_docs/fyp26assess-info.pdf` contains the official assessment criteria — read it before drafting the report.

## `report_latex` guidance

The final report source lives in `report_latex/`. Compile it from that
directory with:

```bash
latexmk -lualatex main.tex
```

The report is space-constrained: the final submission can contain at most
60 pages of content. Prefer compact layouts in `report_latex` edits: concise captions, compact
tables, and no unnecessary whitespace.

Wayne email template for weekly updates:
- Subject: `Weekly Progress Update — Arthur Allilaire FYP`
- Bullet: tasks completed this week
- Bullet: tasks planned next week  
- Bullet: anything different from the timetable
- Keep it concise; Wayne is not an MRI expert

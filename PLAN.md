# Initial RL experiments on the QalibreMD digital twin

Working plan for the first round of reinforcement-learning experiments
using `QalibreMDPhantom` + KomaMRI as the simulator. Explicitly written
against the supervisor comments on the interim report (`awetscherek`
annotations on `BEng_Interim_Report_aw.pdf`, pp. 19–22), which push backon several of the interim's framings and are summarised inline below.

---

## 1. Goal of this phase

Produce the **simplest possible RL loop** that learns something useful on
the digital twin, end-to-end, then scale the task ladder. Concretely, by
the end of this phase we want:

1. A **Gymnasium environment** that wraps KomaMRI + the twin, exposes a
   parameterised pulse-sequence action space, and returns signal
   observations and a scalar reward.
2. A **conventional-sequence baseline** (IR-TSE for T1, multi-TE SE for T2)
   run on the twin whose fits match the manual values to within a few
   percent — this is both a simulator sanity check and the yardstick the
   RL agent has to beat (Wetscherek: *"probably would try to start with
   that and … test whether the implementation works as expected"*).
3. **One working trained agent** on the simplest task (single-sphere T1
   estimation with domain randomisation) before scaling up.

Non-goals for this phase: MR-LINAC deployment, novel k-space trajectories,
anything with anatomy. Keep the physics surface area small so failures are
interpretable.

---

## 2. Where the real challenge sits

Interim report framed Julia↔Python as "the key challenge". Supervisor
disagrees: *"calling Python from Julia (or vice versa) can be achieved
with packages such as PyCall or PythonCall… probably the bigger challenge
/ tradeoff might be between realistic simulation (the more spins the
more realistic) and the total run time"*.

So the binding constraint we design around is **simulator wallclock per
step**. RL algorithms like PPO/SAC want 10⁵–10⁶ env steps; KomaMRI on a
200k-spin phantom is not that fast. This drives several decisions below
(coarse voxels for training, single-sphere simulations for episodes,
caching descriptors, GPU backend, dictionary pre-computation).

A second important supervisor point: **do not "port" the agent across
simulators** — *"in the end I would think of the agent as an independent
program - as long as there is a way for the agent to interact with the
Gym, there are no real language constraints"*. So the agent lives in
Python (Stable-Baselines3 / CleanRL), the simulator lives in Julia
(KomaMRI), and the only contract between them is a Gym `step()` call.

---

## 3. MDP formulation

We define one shared MDP template; each experiment instantiates it with
different action spaces, observation shapes, and reward terms.

**State (internal, not all observable):** the simulated spin ensemble for
the current episode — a `Phantom` built from a fresh `PhantomConfig` with
randomised relaxation properties and pose. The agent never sees the
phantom state directly; it only sees the complex signal it has
accumulated so far plus the blocks it has already scheduled.

**Action:** parameters of the next pulse-sequence block. Following
Wetscherek's hint — *"[the action] could e.g. one from a list of
predefined blocks or created from a function with a couple of parameters
(e.g. flip angle, TR and radial k-space angle)"* — we use a small
parameterised block. v1 action space (single-sphere IR-SE):

| Parameter | Range | Type |
|---|---|---|
| TI (inversion time) | 10–3000 ms | continuous |
| TE (echo time) | 5–200 ms | continuous |
| TR (repetition time) | 50–5000 ms | continuous |
| flip angle α | 5–180° | continuous |
| do-inversion | {0, 1} | discrete |

Later experiments extend with a radial k-space angle and a slice
selection gradient.

**Observation:** the complex signal samples returned from the last block
(downsampled to a fixed length, split into real/imag channels) plus a
running vector of `(cumulative_time_used, blocks_played_so_far,
running_estimate)`.

**Transition:** call `simulate(phantom, seq_block, scanner)`, append the
returned signal to the episode buffer, advance the simulated clock.

**Episode termination:** hit the scan-time budget (e.g. 60 s simulated
scan time) OR a fixed number of blocks (e.g. 32) OR agent emits a special
"done" action that exposes its final parameter estimate.

**Reward:** dense + terminal (see §5).

---

## 4. Experiment ladder

Five experiments of increasing difficulty. Start at E1 only; do not move
on until each is reproducible and the previous rung is holding.

### E0 — Non-RL baseline (week 1)

Not an RL experiment; the yardstick. Wetscherek explicitly suggests
starting here.

* **Sequence:** conventional IR-TSE (for T1) and multi-TE single-echo SE
  (for T2) written in Pulseq.
* **Targets:** recover manual T1/T2 values of all 14 T1-array spheres and
  all 14 T2-array spheres, MAPE < 3 %.
* **Purpose:** validates the simulator + twin + fitting pipeline before
  any learning is added. Fits also become the baseline the RL agent has
  to match (initially) and beat (on scan time).

### E1 — Single-sphere T1 estimation, discrete action set (weeks 2–3)

The simplest thing that is still meaningful RL.

* **Env:** each episode samples one sphere. Use `PhantomConfig(
  include_plates = [:T1])` and pick a descriptor whose `T1` is drawn
  uniformly from a *jittered* version of `T1_ARRAY[cfg.field]` — this
  kills the "agent just memorises the 14 manual values" failure mode
  (Wetscherek: *"How do you avoid that the agent simply 'learns' the
  manual values - do you plan to vary the T1 and T2 values and the
  position of the vials during the simulations?"*).
* **Action space:** discrete — 16 predefined IR-SE blocks covering TI ∈
  {10, 30, 100, 300, 1000, 3000} ms × α ∈ {10°, 90°, 180°}.
* **Observation:** the magnitude of the first 64 ADC samples from the
  last block, concatenated with the running Levenberg–Marquardt T1
  estimate and number of blocks used.
* **Reward:** −|T̂₁ − T₁_true| / T₁_true per step + terminal bonus if
  within 3 % at scan end. Scan time enters via the cost
  `λ · block_time / budget` per step.
* **Algorithm:** PPO with a small MLP policy.
* **Success:** converges to a policy whose terminal MAPE on held-out T1
  values is better than a fixed 8-block IR grid within the same scan
  time budget.

### E2 — Single-plate multi-sphere T1 & T2 mapping (weeks 3–4)

Scale the action space and observation, keep only one plate so voxel
count stays low.

* **Env:** `PhantomConfig(include_plates = [:T1])` at coarse voxel size
  (3–4 mm) for training, fine (1 mm) for evaluation. Agent sees 14
  spheres simultaneously (signal becomes a sum).
* **Action space:** continuous IR-SE parameters (TI, TE, TR, α, inversion
  flag).
* **Reward:** mean over spheres of −|T̂ᵢ − Tᵢ_true|/Tᵢ_true, with a
  per-step time cost. Separate channels for T1-plate (NiCl₂) and T2-plate
  (MnCl₂) versions of the experiment.
* **Domain randomisation via the existing `AugmentConfig`:**
  * `rotation` uniformly over SO(3) — kills position memorisation.
  * `translation_mm` ~ 𝒩(0, 5 mm).
  * `T1_sigma_rel = 0.05`, `T2_sigma_rel = 0.05` — jitter values per
    episode so the agent can't look up.
  * `B0_sigma_Hz` ~ 𝒰(0, 10) — off-resonance, as Wetscherek notes both
    KomaMRI and MRzero handle this natively.
  * Optional: `drop_sphere_p = 0.05` — robustness to missing data.
* **Success:** MAPE < 5 % across spheres at 3× speedup vs the E0 grid
  baseline.

### E3 — MRF-style fingerprinting via learned FA/TR schedules

Once E2 works, swap the action space for the canonical MRF parameters
(variable flip angle + TR per TR-block) and replace the
parameter-accuracy reward with a **dictionary-discriminability reward**:
how well the acquired signal evolution projects onto a pre-computed
Bloch dictionary. This is the most direct tie-in to the qMRI literature
cited in the interim (Jordan et al., 2021 [12]).

### E4 — Adaptive k-space trajectories (stretch)

Single 2D slice through Plate T1, action = next radial spoke angle +
α + TR (Wetscherek's exact suggestion of *"flip angle, TR and radial
k-space angle"*). Reward combines image-space reconstruction error with
scan-time penalty. This is the experiment that most closely mirrors the
Walker-Samuel "autonomous sensing" paradigm.

### E5 — Parameter-map + pose estimation (stretch)

Agent must simultaneously localise the sphere centroids (x, y) and
estimate each sphere's T1/T2. Bridges toward the "autonomous qMRI
mapping" extension in the interim §3.1.4.

---

## 5. Reward function design

Supervisor was sharp about this: *"The metrics in the table are very
different, i.e. some are can be used directly for optimisation (parameter
accuracy and spatial localization), but how would the agent learn that
scan time matters? I'm not sure how one would define the SNR metric."*

Concrete split for E1–E3:

* **Optimisation reward (dense, per step):** smooth function of running
  parameter error.
  $$r_t = -\text{MAPE}(\hat T_t, T_{\text{true}}) \cdot \Delta t_{\text{block}} \cdot w_t$$
  weighted so the agent is pushed to reduce error early.
* **Terminal shaping:** +$B$ if MAPE < 3 %, else 0. Encourages
  commitment, avoids over-budgeting.
* **Scan-time cost (hard budget):** episode truncates when
  cumulative `block_time ≥ T_budget`. No soft time penalty — makes the
  tradeoff explicit and easier to tune.
* **Safety penalty:** any block whose SAR or dB/dt estimate exceeds the
  MR-LINAC limits (Elekta Unity 1.5 T specs) gets −1 and episode
  terminates. This is also a legal / ethical requirement from §3.3.

**SNR is not a direct reward term.** Treat it as a monitoring metric;
during training the agent pays for SNR implicitly because poor-SNR
signals produce bad parameter estimates, which hurts the MAPE reward.

---

## 6. Julia ↔ Python plumbing

Supervisor explicitly sanctioned the two canonical options — `PyCall` /
`PythonCall` for the Julia side, or their Python mirrors. We use
**PythonCall.jl + juliacall** (the modern pair; PyCall is legacy and has
a GIL footgun with worker-based RL trainers).

### Chosen architecture

```
┌─ Python process (the agent) ─────────────────────────────┐
│                                                          │
│  Stable-Baselines3 PPO / SAC                             │
│         │                                                │
│         ▼                                                │
│  gymnasium.Env wrapper  (qalibremd_gym.py)               │
│         │  step()                                        │
│         ▼                                                │
│  juliacall — holds a long-lived Julia subinterpreter     │
│         │                                                │
│         ▼                                                │
│  Julia: QalibreMDPhantomGym.jl (new tiny package)        │
│         │                                                │
│         ▼                                                │
│  KomaMRI.simulate(obj, seq, sys)                         │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

Why this direction (Python → Julia via juliacall):

* The RL ecosystem is Python (SB3, CleanRL, torch, wandb).
* Julia stays as the simulator; we already have `QalibreMDPhantom` and
  the Bloch solver.
* juliacall re-uses a single Julia instance across `env.step` calls, so
  JIT warmup amortises across millions of steps.
* Zero-copy NumPy ↔ Julia arrays for observations.
* Works with vectorised envs as long as each worker process owns its
  own Julia runtime (measure: if that's too expensive, switch to one
  Julia runtime with thread-parallel `simulate`).

### What we have to build

* **Julia-side**: thin `Gym.jl`-like module `QalibreMDPhantomGym.jl` that
  exposes `reset(cfg_sampler, rng_seed)`, `step(action)`, `render()`. It
  caches a voxelised `Phantom` per episode so each step only constructs
  a sequence block and runs `simulate` on the cached phantom.
* **Python-side**: `gymnasium.Env` subclass that proxies through
  juliacall; handles observation normalisation, action rescaling,
  episode-time accounting.
* **Pulseq block library** (Julia): a small catalogue of parameterised
  `Sequence` templates (IR, SE, SR-SE, variable-FA block) that the agent
  parameterises.

### What we explicitly do NOT do

* No "porting" the agent to Julia after training (supervisor: agent is
  an independent program).
* No Python-hosted Bloch solver — keep simulation in Julia where we've
  already validated the twin. (MRzero remains a separate comparison
  track, not a simulator to migrate into; supervisor also corrected the
  interim's "EPG" claim — MRzero uses phase distribution graphs.)

---

## 7. Fidelity-vs-wallclock tradeoffs

Supervisor's flagged main challenge. Concrete decisions:

| Axis | Training | Evaluation |
|------|----------|------------|
| Voxel size | 3–4 mm | 1 mm |
| Plates included | minimum needed (often single sphere) | all plates |
| Background water | excluded | included |
| Fiducials | excluded | included |
| Field strength | fixed per experiment | swept |
| Solver | BlochSimple (CPU) | Bloch with GPU if available |
| Scan-time budget | short (10–30 s sim time) | per-experiment protocol |

Instrumentation: log wallclock per `simulate` call; if it exceeds ~50 ms
on the training config, we drop voxel count further before blaming the
agent.

Two optimisations worth trying early:

1. **Single-sphere phantom = single voxel** at the limit — for E1 the
   "phantom" can literally be `Phantom(x=[0.], T1=[T1_sample],
   T2=[T2_sample])` (see `01-FID.jl` for the pattern). Orders of
   magnitude faster than the voxelised sphere; valid because
   single-sphere MRF/IR signals don't depend on spatial extent when
   there's no gradient encoding.
2. **Dictionary caching** for E3 — pre-compute Bloch responses for a
   grid of (T1, T2, FA, TR) and interpolate during training; fall back
   to live `simulate` only at evaluation.

---

## 8. Dealing with the "agent memorises the phantom" failure mode

The single most important item from the supervisor's review:

> "How do you avoid that the agent simply 'learns' the manual values —
> do you plan to vary the T1 and T2 values and the position of the vials
> during the simulations?"

Mitigations, all already supported by `PhantomConfig` / `AugmentConfig`:

1. **Per-episode T1/T2 sampling.** Draw `T1` from a log-uniform
   distribution spanning the manual range (20 ms – 2 s at 3 T) rather
   than picking discrete manual values. Implement as a custom
   `PhantomConfig.custom_sphere_map`.
2. **Pose randomisation.** Rotation over SO(3) + translation ~ 𝒩(0, 5 mm).
3. **Field randomisation.** 50/50 between `:T15` and `:T3` per episode.
4. **Background noise.** Complex Gaussian on the raw signal (Wetscherek:
   *"adding complex Gaussian noise to the simulated raw data"*).
5. **Held-out evaluation set.** Fix a seed-based test split of phantom
   configs that the agent never sees during training; report metrics
   only on that set.

---

## 9. Evaluation protocol

Three metrics, all computed on the held-out config set:

* **Parameter MAPE (T1, T2, ρ)** — the primary optimisation target.
* **Relative scan time** — agent's wallclock vs E0 baseline's for the
  same MAPE level. The Pareto curve of (MAPE, scan time) is the
  headline plot.
* **Robustness curves** — MAPE vs rotation angle, vs translation
  magnitude, vs B0 offset, vs SNR. Tests the domain-randomisation
  claim.

One non-optimised but monitored metric: SNR of the acquired signal.
Supervisor correctly flagged this isn't directly trainable; report it
for interpretability only.

---

## 10. Milestones (8-week budget)

| Week | Deliverable |
|-----:|---|
| 1 | E0 running: IR-TSE + multi-TE SE on the full twin, fits within 3 % of manual. Pulseq block library v1. |
| 2 | `QalibreMDPhantomGym.jl` + juliacall Python wrapper. Random-policy episodes roll in a Jupyter notebook. |
| 3 | E1 trained: PPO on single-sphere T1, beats fixed IR grid on held-out configs. |
| 4 | E1 ablations: no-randomisation vs full-randomisation; report memorisation failure explicitly. |
| 5–6 | E2 trained on the T1 plate with full `AugmentConfig` randomisation. |
| 7 | E3 pilot (MRF-style FA/TR schedules with dictionary matching). |
| 8 | Pareto curves, write-up. If time permits, start E4 as stretch. |

---

## 11. Open questions / known risks

* **"Dwell" terminology** — supervisor queried what "dwell longer on
  specific k-space coordinates" meant in the interim. Clarify before
  promising it as a mechanism; in E3/E4 this probably translates to
  longer ADC or repeated acquisition at the same k-space location.
* **PDG vs EPG** — interim said MRzero is EPG; supervisor corrected
  that it uses phase distribution graphs. Update the write-up and make
  sure any comparison plots reflect this.
* **Fixed vs adaptive sequences** — supervisor's single strongest point:
  *"one of the core ideas of this project is that there might be no
  such thing as a 'final sequence' … the optimal sequence might differ
  between patients"*. Experiment reporting should emphasise the
  *policy*, not the average sequence length — e.g. histograms of TI/TR
  choices as a function of the current running estimate.
* **Single-Julia-runtime vs multi-process vectorised envs** — will the
  memory footprint of N Julia subinterpreters for N parallel workers be
  prohibitive? Measure on the box we're training on; fall back to
  single-process async collection if needed.
* **Fiducial calibration values** — still placeholder in
  `src/materials/fiducial.jl`. Not blocking for E1–E3, but replace
  before any sim-to-real comparison.
* **Transfer to real hardware** is out of scope for this plan; PLAN
  explicitly stops at Phase II (sim-to-sim). Anything MR-LINAC related
  is a follow-on document.

---

## 12. What exists today

* `QalibreMDPhantom` package (this repo, `src/`): parameterised builder,
  material tables, geometry, augmentations, determinism — 373-test
  suite green. Ready to be wrapped by the Gym env above.
* Visualisation scripts (`examples/plot_phantom.jl`) — useful for
  debugging whatever sequence the agent proposes.
* Supervisor's interim-report annotations — these drive the design
  choices above; revisit the PDF annotations directly if anything in
  this plan looks off.

---

## 13. References (in addition to `README.md`)

* KomaMRI.jl — <https://github.com/JuliaHealth/KomaMRI.jl>
* juliacall / PythonCall.jl — <https://juliapy.github.io/PythonCall.jl/stable/>
* Gymnasium — <https://gymnasium.farama.org/>
* Stable-Baselines3 — <https://stable-baselines3.readthedocs.io/>
* Pulseq — <https://pulseq.github.io/>
* MRzero (Loktyushin et al., 2021) — cited [10] in the interim.
* AUTOSEQ (Zhu et al., 2018) — cited [16] in the interim; inspiration
  for E3.
* Walker-Samuel (2019) — cited [18] in the interim; inspiration for E4.
* Jordan et al., PNAS 2021 — MRF sequence optimisation, referenced as
  the qMRI precedent for E3.

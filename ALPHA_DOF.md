# α-DOF — explainer for the α-aware CR-opt baseline and Run A

**Status:** built · **Owner:** Arthur · **Updated:** 2026-05-26

This document explains what the α degree-of-freedom work actually does, now that it
is built. Run A frees the excitation flip angle α as a learned RL action; the
companion CR-opt machinery (`src/baselines/cr_optimal_alpha.jl`) provides the fixed
schedules we measure RL against. The original build plan has been replaced by this
explainer — the code is the source of truth now.

---

## The three reference points

"Just set α to the Ernst angle" is not a clean fixed baseline, because the Ernst
angle `cos α* = exp(−TR/T1)` is **T1-dependent** (14 different fleet T1s → no single
α) and maximises **SNR**, not **Fisher information on T1** (the actual estimation
objective). So we have three fixed reference points of increasing strength, and RL's
α-freedom is measured against them:

- **(a) α=90° CR-opt anchor** — the existing `cr_optimal.jl` solver, α hard-coded to
  π/2. This is the headline yardstick: RL's α-freedom is the advantage being measured.
- **(b) Ernst-fixed-α** — reuses the CR-opt `(TI, TR)` timing but swaps α=90° for the
  Ernst angle at the fleet-median T1 (see below). A cheap "smart SNR heuristic" middle
  point.
- **(c) CR-opt-with-α** — re-optimises α per block under the *estimation* objective
  (`cr_optimal_alpha.jl`). The strongest fixed control: "could an estimation-aware
  fixed α have closed the gap?"

The gap between (b) and (c) is itself interesting: it isolates SNR-optimal α vs
Fisher-optimal α.

---

## What `cr_optimal_alpha.jl` computes (reference point c)

It's the Cramér–Rao-optimal *fixed* IR-SE schedule when α is a free **per-block
design variable**, on top of the (TI, TR) timing the α=90° solver already optimises.
It lives in a separate file from `cr_optimal.jl` on purpose, so the α=90° anchor (a)
stays byte-for-byte intact and the two paths can be diffed.

### Key modelling decision — the Fisher stays 2×2

α is a **commanded knob, not an estimated parameter**. `fit_t1_generalized_ir` takes
α as a *known input* and estimates only (T1, A). So:

- The Fisher information is **2×2 over (T1, A)**, exactly as in `cr_optimal.jl`.
  Optimising α just changes *where* that 2×2 Jacobian is evaluated
  (`jacobian_row_alpha` threads α into `f_signal`) — it does **not** add a third
  estimated parameter.
- At α=π/2 every function reduces exactly to its `cr_optimal.jl` counterpart.
- This keeps the CR bound consistent with the estimator the fleet actually uses.

> Note: an earlier draft of this plan proposed an **n×3 Jacobian / 3×3 Fisher** with
> α as a third estimated parameter. That was wrong for this estimator and was not
> built — the header comment in `cr_optimal_alpha.jl` records the reasoning.

### How the optimisation works

Same two-stage structure as the α=90° solver, carrying a per-block α:

1. **Objective** (`cr_fleet_objective_alpha`): A-optimality — `Σ_j w_j · Var(T1_j)/T1_j²`
   over the fleet, each `Var(T1_j)` from the 2×2 CR bound at that sphere's T1.
2. **Search** (`cr_optimize_alpha`): multi-start (default 1000 random budget-feasible
   schedules) → keep top N by objective → coordinate-descent refine each
   (`refine_coordinate_descent_alpha`). TI/TR perturbed multiplicatively, **α
   perturbed additively** in ±{4°,10°,25°} steps, clamped to [5°,90°]. Accept any
   budget-feasible move that lowers L.
3. **Sweep** (`cr_optimize_sweep_alpha`): runs the above for each
   `n_blocks ∈ {4,6,8,12,16}` and returns the global best.

---

## The Ernst-fixed-α baseline (reference point b)

`ernst_angle(TR, T1) = acos(exp(−TR/T1))` is the SNR-optimal flip angle for a spoiled
steady-state acquisition. It is T1-dependent, so for a multi-T1 fleet it can't be
evaluated once — `ernst_fixed_schedule(TIs, TRs, T1_ref)` evaluates it at a single
**reference T1** (the fleet median) for each block's TR, giving one α per block,
clamped to [5°,90°].

Crucially, this baseline **borrows the CR-opt `(TI, TR)` timing unchanged** and only
overwrites the flip angle. So it answers a narrow, clean question: holding the optimal
timing fixed, is the SNR-heuristic flip angle enough, or does estimation-aware α
(point c) matter? Physics lives in the Julia package; `python/baseline_e2.py`'s
`_ernst_fixed_factory` round-trips through `ernst_fixed_schedule` so there's no
duplicated physics on the Python side.

---

## Assumptions baked into the CR-opt solver

- **TR is free in [0.5, 5.0]** — `TR_lo_floor=0.5`, `TR_hi=5.0`. The actual budget
  constraint uses `schedule_time_s(TRs, Npe)` with the *real* per-block TRs.
- **The `0.5` in the n_blocks sweep gate is not a TR assumption.** Line ~276,
  `block_time_s(0.5, Npe) * nb > budget_s && continue`, is a cheap feasibility
  pre-filter: `0.5` is the TR *floor*, so this is the cheapest possible block. It
  skips an `n_blocks` only if it can't fit even at minimum TR; the optimiser then
  finds the real TRs. (Latent inconsistency: this gate hardcodes `0.5` instead of
  referencing the `TR_lo_floor` kwarg — if the floor ever changes, update the gate.)
- **Timing cost model** (`block_time_s`): one block ≈ `Npe · TR + 0.05s` overhead;
  Npe defaults to 8. TI is consumed *inside* the TR window, so it doesn't add to block
  time.
- **Perfect transverse spoiling** between TRs (inherited from
  `transient_mz_at_excite_npe`).
- **θ_inv = π** (ideal 180° inversion), **σ_obs=1, A=1** for the bound.
- **TR > TI always** — enforced via `max(TIs[k]+0.05, TR_lo_floor)`.

---

## Action space — the with/without-α ablation

To isolate α as the **single** new DOF, both runs fix TE=20 ms and differ only in
whether α is learned:

| Run | Action dims | Physical mapping | α |
|---|---|---|---|
| **A0 — without α** | 2: `[TI, TR]` | TE=0.020, **α=90°**, slice_z=0 | fixed |
| **A — with α** | 3: `[TI, TR, α]` | TE=0.020, α∈[5°,90°], slice_z=0 | learned |

The flags `fix_te` and `learn_alpha` on `QalibreMDE2Env` select these modes;
`learn_alpha` requires `fix_te` (fail-fast guard). The α upper bound is **90°, not
180°**, so the Ernst window sits mid-range and the policy can't waste mass on inverting
excitations. No Julia changes were needed — the env, sequence builder, and fitter were
already α-aware (`cos(α)` in the Mz recurrence; the env divides each measured magnitude
by `sin(α)` before fitting).

---

## Run sequencing & commands

```bash
# Step 0 — re-baseline at σ* (per E2_RERUN_PLAN §3); produces α=90° CR-opt anchor (a)
python python/baseline_e2.py --episodes 50 --noise σ* --time-budget 160 \
   --max-blocks 30 --cr-optimal --out runs/e2/rerun_baselines

# (b) Ernst-fixed-α baseline (reuses the CR-opt (TI,TR) timing)
python python/baseline_e2.py --episodes 50 --noise σ* --time-budget 160 \
   --max-blocks 30 --cr-optimal --ernst-baseline --out runs/e2/rerun_baselines

# (c) CR-opt-with-α control (stretch)
python python/baseline_e2.py --episodes 50 --noise σ* --time-budget 160 \
   --max-blocks 30 --cr-optimal --cr-optimize-alpha --out runs/e2/rerun_baselines

# Run A0 — WITHOUT α (clean 2-dim [TI,TR] ablation)
PYTHON_JULIAPKG_OFFLINE=yes python python/train_e2.py \
   --fix-te \
   --reward-mode delta_mape --terminal-bonus 0.0 --mape-alpha 1.0 \
   --noise σ* --time-budget 160 --max-blocks 30 \
   --timesteps 300000 --out runs/e2/rerun_A0_noalpha

# Run A — WITH α (3-dim [TI,TR,α])
PYTHON_JULIAPKG_OFFLINE=yes python python/train_e2.py \
   --fix-te --learn-alpha \
   --reward-mode delta_mape --terminal-bonus 0.0 --mape-alpha 1.0 \
   --noise σ* --time-budget 160 --max-blocks 30 \
   --timesteps 300000 --out runs/e2/rerun_A_alpha

# Eval both (paired seeds 500000+) vs all baselines; diagnose Ernst behaviour
python python/eval_e2.py  --run runs/e2/rerun_A0_noalpha --baselines runs/e2/rerun_baselines
python python/eval_e2.py  --run runs/e2/rerun_A_alpha    --baselines runs/e2/rerun_baselines
python python/diagnose_e2.py --run runs/e2/rerun_A_alpha   # α-vs-TR-vs-T1 plot
```

The headline number is **A vs A0 MAPE** (the α gain), reported against the **(a)**
CR-opt anchor, with **(b)** and **(c)** as the "could a fixed α have closed the gap?"
controls. `diagnose_e2.py` overlays the learned `(TR, α)` pairs against the Ernst
curve to test for emergent Ernst-angle behaviour — the citable Run A signal.

---

## Tests

Three suites cover the α path; each `@testset` / test function carries a comment
describing what it pins.

- `test/test_fit_alpha.jl` — the fitter at non-90° α (the `cos(α)` recurrence and the
  env's `sin(α)` magnitude correction).
- `test/test_cr_optimal_alpha.jl` — the α-aware CR-opt solver and Ernst helpers.
- `python/tests/test_alpha_action_modes.py` — the `fix_te` / `learn_alpha` action-mode
  mappings.

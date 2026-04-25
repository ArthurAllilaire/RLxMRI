# TO READ

Need to refresh on the different MRI sequences and pulses.

Need to learn RL algorithms.

---

## Inversion Recovery signal — closed-form derivation

The `generalized_ir_signal` function computes the MRI signal analytically
without calling the simulator. Here is what each step does physically.

### Setup

A single spin starts at equilibrium: Mz = 1 (along +z). The sequence is:

1. Prep pulse — tip Mz by angle α
2. Wait TI — magnetisation recovers toward equilibrium
3. 90° excite — tip recovered Mz into transverse plane
4. ADC — read out signal as transverse magnetisation decays

### Step 1 — prep pulse

A hard pulse of flip angle α rotates Mz instantly:

    Mz_after_prep = cos(α)

- α = π  (180°): Mz = −1  — full inversion, classic IR
- α = π/2 (90°): Mz =  0  — saturation recovery
- α = small:     Mz ≈  1  — barely perturbed

"Perfect spoiling" means any transverse component created by this pulse is
immediately destroyed — only Mz survives to the next step.

### Step 2 — T1 recovery during TI

The Bloch equation solution for Mz recovery:

    Mz(TI) = 1 − (1 − Mz_after_prep) · exp(−TI / T1)

At TI → ∞ this reaches 1 (full recovery). At TI = 0 it equals Mz_after_prep.

### Step 3 — 90° excite

Tips the recovered Mz entirely into the transverse plane.
Signal amplitude = |Mz(TI)|. Absolute value because magnitude readout
cannot distinguish +Mz from −Mz.

### Step 4 — ADC readout

Signal decays with T2 across the readout window:

    S(t) = |Mz(TI)| · exp(−t / T2),   t ∈ [0, dur_adc]

Sampled at n_adc = 64 evenly-spaced points.

### Why ~10,000× faster than simulate()

simulate() numerically integrates the Bloch equations across every spin,
every time step, every gradient, every RF point. This function is 3
arithmetic operations and an array fill. It is only valid under three
idealisations:

1. Single spin — no spatial averaging across a voxel
2. Hard pulse  — instantaneous RF (no pulse shape to simulate)
3. Perfect spoiling — transverse magnetisation destroyed between prep and excite

All three hold well enough for RL training, where fast approximate signal
is needed rather than ground-truth simulation.

---

## Material number discrepancies — PLAN.md vs source

### T1 array spheres — point differences

| Field | Sphere | Property | PLAN.md | Source | Diff |
|-------|--------|----------|---------|--------|------|
| 3T    | 2      | T1       | 1398 ms | 1362 ms | +36 ms — likely transcription error |
| 3T    | 2      | T2       | 1035 ms | 1039 ms | −4 ms |
| 3T    | 3      | T2       | 728.3 ms | 718.3 ms | +10 ms |

1.5T T1 array matches exactly.

### T2 array spheres — structural mismatch (bigger issue)

PLAN.md has two columns for T2-array spheres: T1_ms and T2_ms (the T1 and T2
relaxation times of those spheres). The source only stores one set of values
in `T2_ARRAY`, plus a single constant `T1_OF_T2_ARRAY_DEFAULT = 3000 ms`.

The values in source `T2_ARRAY` match PLAN.md's **T1_ms column**, not T2_ms.
This means:

- The actual T2 values of T2 spheres (e.g. 1044 ms → 7.8 ms at 1.5T;
  646 ms → 5.4 ms at 3T) are **not encoded in the source at all**
- All T2 spheres get the same T1 = 3000 ms instead of their true T1s
  (which range from ~2640 ms down to ~80 ms)

Within the T1-of-T2-spheres values that are stored, there are also
discrepancies between PLAN.md and source:

| Field | Sphere | PLAN.md | Source | Diff |
|-------|--------|---------|--------|------|
| 1.5T  | 4      | 1489 ms | 1609 ms | −120 ms |
| 1.5T  | 7      | 733.9 ms | 782.9 ms | −49 ms |
| 3T    | 10     | 299.8 ms | 314.8 ms | −15 ms |

The 1.5T sphere 4 and 7 differences are large enough to be transcription
errors — the manual should be checked to confirm which is correct.

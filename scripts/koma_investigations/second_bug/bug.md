# Residual time-driven jump in KomaMRI simulator

Single self-contained brief on a deterministic jump in the per-shot signal of
multi-shot Koma simulations once cumulative simulated time exceeds a
sequence-dependent threshold. Same family as the previously fixed bug
documented in `scripts/koma_bug_minimal.jl`; the prior fix raised the
threshold but did not eliminate the underlying mechanism.

## TL;DR

- **Symptom**: per-shot |signal| is constant for the first N stable shots, then
  jumps by +30–50 % at a specific shot, then plateaus at a new wrong value
  (no gradients) or oscillates chaotically (with per-shot-varying gradients).
- **Threshold**: sim_time ≈ 270 s for the simplest pattern; drops to ≈ 70 s as
  per-shot arithmetic grows (full IR-SE imaging).
- **Deterministic**: identical jump shot and identical post-jump value across
  re-runs and across `precision="f32"` vs `precision="f64"` (so it is **not**
  a magnetisation-precision roundoff bug).
- **Likely cause**: a hardcoded absolute epsilon in a time comparison
  somewhere in `KomaMRICore` — once accumulated F64 noise on the time
  variable exceeds that epsilon, the comparison flips and the simulator
  enters a different code path for every subsequent block. Same mechanism
  family as the previously fixed bug (which was also "ε too small").

## How to reproduce

The reproducer lives at `scripts/koma_investigation/koma_bug_residual.jl`.
It is self-contained — only depends on `KomaMRI`, `Suppressor`, `Printf`.

```bash
julia --project=. scripts/koma_investigation/koma_bug_residual.jl
```

Sequence per shot, repeated 24 times at TR = 15 s (sim_time = 360 s):

```
180° inversion (1 ms hard pulse, no gradients)
delay TI = 3 s
90° excitation (1 ms hard pulse, no gradients)
ADC, 1 sample, 1 ms window, no gradients
delay TR − TI − 2 ms = 11.998 s
```

Phantom: a single spin at the origin with T1 = T2 = 1 s.

Expected output: 24 identical |signal| values (every shot is the same
physical event on the same steady-state spin).

Actual output:

```
shot  1..17  (sim_time ≤ 255 s):  |signal| = 0.476267   ← stable, all identical
shot 18      (sim_time = 270 s):  |signal| = 0.722479   ← +52 % jump
shot 19..24  (sim_time ≥ 285 s):  |signal| = 0.677159   ← new wrong plateau (+42 %)
```

The jump is reproducible bit-for-bit across runs and across f32/f64 precision
modes (only the 7th decimal differs between f32 and f64; the jump is at the
same shot with the same magnitude either way).

## Context: previously fixed bug

The previously fixed bug had the same character — discrete jump in per-shot
|signal| after a threshold in cumulative sim_time. Documented in
`scripts/koma_bug_minimal.jl`, which uses the same sequence pattern as this
reproducer but stops at N = 16 (sim_time = 240 s, before the new threshold).
That script returns 16 identical values after the fix, so the prior fix is
genuinely doing something — it just doesn't cover all the comparisons that
suffer from the same root cause.

Before fix: jump at sim_time ≈ 70–150 s, +22 %.
After fix: jump at sim_time ≈ 270 s for the same pattern, +52 %.

The fix shifted the threshold ~4×; finding what else uses an analogous
tight ε would likely close the remaining failure mode.

## What we have ruled out

| Hypothesis | Disproved by |
|---|---|
| Float32 magnetisation roundoff | `precision="f64"` gives the same jump at the same shot. The drift is deterministic at the bit-flip level, not a stochastic precision walk. |
| Multi-shot coherence pathways | Drift threshold depends on `sim_time`, not on shot count. 8 shots at TR=10 s and 40 shots at TR=2 s both jump near 70 s. |
| Steady-state Mz transient | Steady state is reached by shot 2 (the shot-1 → shot-2 jump is the recovery transient, well below the threshold). Shots 2..N-1 are bit-identical until the threshold. |
| Gradient-specific (RF, gradient, ADC, etc.) | All four gradient configurations tested (none, readout-only, prewinder+readout, full IR-SE) show the jump. Gradient activity only affects *when* the jump happens (proportional to per-shot arithmetic density) and what the *post-jump* trajectory looks like (stable new plateau vs chaotic). |
| Total sim time alone (≥ some absolute) | Threshold varies with per-shot complexity. Simple pattern: ~270 s. Full IR-SE: ~70 s. So it is not "Koma is unreliable past X seconds of sim_time" — it is something that scales with the *work* done per unit sim_time. |

## Hypothesis: tight absolute epsilon in a time comparison

Float64 has ~15-digit relative precision. At t = 270 s, accumulated rounding
noise on a `t += dt` accumulator is roughly t × eps(F64) ≈ 6e-14 s; with
extra arithmetic per step it grows faster. If somewhere Koma compares time
quantities with an *absolute* epsilon that's too small (e.g. `< 1e-12` or
`< 1e-13`), that comparison flips deterministically at a specific accumulated
sim_time. From that point onward the simulator takes a consistently different
code path → discrete jump to a new attractor.

Why this fits every observation:

- **Determinism**. The comparison flips at a precise accumulated-noise
  threshold, not stochastically.
- **f32 vs f64 indistinguishable.** The variable being compared is already
  Float64 (time is); promoting magnetisation precision doesn't change time
  arithmetic.
- **Threshold scales with arithmetic density.** More per-shot operations →
  accumulated noise crosses ε in fewer wall-time seconds of simulation.
- **Plateau, not gradient.** Once t is past the ε threshold, the comparison
  is consistently on the wrong side for every subsequent block. So you see
  one transition and then a stable wrong state.
- **Prior fix raised threshold but did not eliminate.** The fix likely
  loosened a single ε somewhere; the remaining failure path uses a different
  small ε that's still in the source.

## Where to look in `KomaMRICore` source

High-priority candidates — comparisons involving an accumulated time variable
and a hardcoded small absolute number. Likely shape of the offender:

```julia
# any of:
if abs(t - t_target) < SOMETHING_SMALL
if t - t_block_end < SOMETHING_SMALL
if t >= t_event - SOMETHING_SMALL
isapprox(t, t_target; atol = SOMETHING_SMALL)
```

where `SOMETHING_SMALL` is a literal like `1e-12`, `1e-13`, `1e-14`, `eps()`,
or a small named constant. The thing on the left is a sim-time variable that
accumulates over many block boundaries.

Specific places worth grepping:

- **Block boundary advancement** in the simulation main loop — where Koma
  decides "this block ends here, move to the next one". Off-by-one at a
  block boundary at large t would shift every subsequent event by one
  integration step.
- **ADC sample placement** — picking which simulator timesteps to record
  as ADC samples. If sample timestamps are checked against block start
  times with a too-tight ε, samples could land in the wrong block.
- **RF envelope evaluation** — picking which RF samples cover each
  integration step.
- **Delay block handling** — the prior fix was in delay-related code; the
  remaining failure may be in a sibling check (e.g. zero-amplitude gradient
  blocks during a "delay" interval).
- **Sequence iterator** — anything that walks through `seq.DUR` summing
  durations and comparing the running sum to an event time.

A good ripgrep pass (run from a clean checkout of KomaMRICore):

```bash
rg '<\s*1\.?0?e-?1[0-5]'                       # absolute ε in [1e-10, 1e-15]
rg 'atol\s*=\s*1\.?0?e-?1[0-5]'                # isapprox with small atol
rg 'eps\(\s*\)|eps\(Float64\)'                 # eps() literals
rg -i '\bdelay\b'                              # delay-handling sites
rg 'cumsum.*DUR|sum.*DUR'                      # running time accumulators
```

Then check whether any of the matches gates on a time variable that is
accumulated across blocks. Compare git blame against the previous fix
commit — the fix probably touched a similar comparison, and the bug may
live in a nearby branch.

## How to validate a candidate fix

1. Run `julia --project=. scripts/koma_investigation/koma_bug_residual.jl`
   and confirm all 24 shots return the same value (to ~7 digits in f32 or
   ~14 digits in f64).

2. Run the prior reproducer `julia --project=. scripts/koma_bug_minimal.jl`
   to confirm the existing fix still passes.

3. If both pass, increase `N_SHOTS` in `koma_bug_residual.jl` to e.g. 100
   (1500 s sim_time) and confirm the trace stays flat. The fix should
   eliminate the threshold entirely, not just push it further out.

4. Re-run the full project's tests: `julia --project=. test/runtests.jl` —
   852 tests, all expected to pass.

5. Optionally re-run `scripts/pixel_grid_overlay.jl --npe 32 --nfe 64 --voxel-mm 1.0 --ti 0.1 --tr 20.0 --spoil`
   and confirm the unspoiled image's monotonic-with-shot-index contamination
   in `pixel_grid_overlay_kspace_diff.png` is gone or much reduced. (That
   contamination is the gradient-amplified manifestation of this same bug
   in our IR-SE imaging.)

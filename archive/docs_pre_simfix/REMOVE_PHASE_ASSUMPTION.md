# Phase-sensitive recon: add global-phase correction

## Context

In the bMANUAL phase-sensitive no-noise run we just produced
(`scripts/runs/t1_fit_vs_true/bMANUAL_nonoise_npe32fe128_ps`) the
Koma-observed `real(IFFT(ksp))` values are inconsistent with the IR
forward model: for long-T1 spheres at sub-null TIs the values are
**sometimes the right sign, sometimes flipped**, with magnitudes ~25×
too small. The fitter therefore sees noise-like signed data and lands
~40% MAPE no matter the schedule.

User's diagnosis is correct: the recon currently assumes the
M_z-bearing complex image is purely real in the rotating frame — i.e.
that the global complex phase of `IFFT(ksp)` is 0. In KomaMRI it is
not. There is an unknown global rotation $\phi$ applied by the
RF/encoding chain (deterministic, but block-dependent in principle).
`real(img · e^{i\phi})` then yields `cos(\phi) · signed_signal`, which
can be either sign.

Confirmed by reading the code: `src/rl/e2.jl::kspace_to_image`
(lines 120–123) does `fftshift(ifft(ifftshift(ksp)))` then takes
`real()` or `abs()` with **no phase correction whatsoever**. The
clean-recon variant `scripts/t1_fit_vs_true.jl::clean_kspace_to_image`
(lines 217–240) does the same after Hamming+zero-pad. The whole
`--phase-sensitive` code path silently assumes $\phi=0$.

## Fix: per-sweep reference-block phase calibration

**Hypothesis** (cheap to verify, see step 3): in our simulator the
global phase $\phi$ is **constant across blocks** because every block
uses identical pulse-sequence structure (same FOV, Npe, Nfe, TE,
gradient waveforms, RF phases) — only TI and TR differ, and those
affect signal *amplitude*, not the rotating-frame *phase* of M_z. If
that holds, one reference acquisition fixes every block.

If the hypothesis fails (φ varies block-to-block), fall back to per-block
estimation of $\phi$ from a phantom-mask weighted mean of the complex
image — but this has a π-ambiguity for blocks whose mean signal is
negative (sub-null), so it needs the reference block anyway to break
the ambiguity. Either way, step 1 below is the load-bearing change.

### Step 1 — Add a reference block

Augment `scripts/t1_fit_vs_true.jl` to run, before the main schedule
loop, a single "reference" IR-SE block with:

- `α_inv = 0` (no inversion) so all spins recover to M0 = 1 by the
  90° excite, **or** TI = TR = very long (≥5×T1_max) so M_z = +1
  everywhere regardless of inversion.
- TR ≥ 5×T1_max — same.
- Same Npe, Nfe, FOV, TE as the rest of the sweep.

After simulating + reconstructing the reference block as a **complex**
image (skip the `real()`/`abs()` step), compute

```
mask = phantom-occupancy > 0  (see phantom_occupancy in src/rl/e2.jl:126)
φ_ref = angle(sum(img_cplx[mask]))
```

The mask ensures we only weigh phantom voxels (not background phase
noise). One scalar per sweep, called `φ_ref`.

### Step 2 — Rotate every block's image by `exp(-i·φ_ref)` before `real()`

Modify `kspace_to_image` (`src/rl/e2.jl:120-123`) to accept an
optional `phase_correction::Float64 = 0.0` kwarg:

```julia
function kspace_to_image(ksp::Matrix{ComplexF32};
                         phase_sensitive::Bool = false,
                         phase_correction::Float64 = 0.0)
    img = fftshift(ifft(ifftshift(ksp, (1, 2)), (1, 2)), (1, 2))
    if phase_correction != 0.0
        img = img .* exp(-1im * Float32(phase_correction))
    end
    return phase_sensitive ? Float32.(real.(img)) : Float32.(abs.(img))
end
```

Same kwarg added to `clean_kspace_to_image` in `scripts/t1_fit_vs_true.jl`
(line 217). Caller passes `phase_correction = φ_ref` when
`phase_sensitive` is true.

### Step 3 — Verification (does φ really stay constant across blocks?)

Before threading `φ_ref` through everywhere, run a one-off diagnostic
inside `scripts/t1_fit_vs_true.jl`: for each block, also compute
`angle(sum(img_cplx[mask]))` *without* applying any correction, and
print the value. If all blocks return the same φ (modulo π for sub-null
blocks), the constant-φ hypothesis holds and step 2's single-scalar
correction is sufficient. If φ drifts, switch to per-block estimation
using the brightest sphere's ROI as a known-sign anchor (the longest-T1
sphere's null is at the largest TI in the schedule, so its sign is
predictable from T1·ln2).

### Step 4 — Re-run + re-plot

```
julia --project=. scripts/t1_fit_vs_true.jl --manual --phase-sensitive --snr 0 --npe 32 --nfe 128
```

Expected: PS no-noise mean MAPE drops from ~41% to roughly the
phase-sensitive-with-fix figure the doc §6.1 cites (≤15% on long-T1
spheres). Markers in `recovery_curves_koma.png` should line up tightly
with the signed M_z curve (blue solid) instead of scattering both above
and below.

## Files to touch

- `src/rl/e2.jl` — `kspace_to_image` (lines 120–123); add `phase_correction` kwarg.
- `scripts/t1_fit_vs_true.jl` — `clean_kspace_to_image` (217–240); same kwarg.
  Add reference-block simulation + φ_ref computation near the top of the main
  block loop (lines 257–343). Thread `φ_ref` into the recon calls (line 285–287).
- (Out of scope) `scripts/t1_fit_vs_true_MReco.jl` — deprecated; skip.

## Files to reuse, do not duplicate

- `phantom_occupancy(phantom, Npe, Nfe, FOV)` in `src/rl/e2.jl:126` — already
  builds the phantom-pixel mask. Use it for the mean-image phase estimate.
- `ir_se_2d_sequence` from `src/sequences/blocks.jl` — already used for the
  schedule blocks; reuse with `α_inv = 0` (or TI=TR=5s and ignore inversion)
  for the reference block.
- `kspace_to_image` itself — extend, don't fork.

## Effort estimate

- Diagnostic (step 3): ~15 lines, 30 minutes including one Koma rerun.
- Implementation (steps 1+2): ~25 lines across two files, 1 hour.
- Verification (step 4): one Koma rerun + replot, 30 minutes.
- **Total: ~2 hours**, conditional on the constant-φ hypothesis holding.

If φ drifts block-to-block (step 3 fails), add 1–2 hours for per-block
phase estimation with the brightest-sphere-ROI sign anchor.

## Gotchas

1. **π-ambiguity of `angle(mean(img))`**: only safe if the reference block has
   *known-positive* M_z everywhere. That is why step 1 uses `α_inv = 0` or
   `TI = TR ≥ 5×T1_max`. Do not estimate φ from a sub-null block.
2. **PSF leak doesn't disappear** — even with perfect phase correction, markers
   will still scatter around the curve due to neighbour-sphere bleed (see
   the no-noise clean-recon analysis in the conversation). Phase correction
   fixes the *sign* of each marker, not its amplitude.
3. **E2 trainer compatibility**: `e2.jl::kspace_to_image` is also called from
   the RL env (lines 397, 560, 791). Adding an optional kwarg defaults to 0 →
   zero behaviour change for E2 until the RL env is taught to pass
   `phase_correction`. That's a follow-up, not part of this change.
4. **Real-hardware caveat**: on real scanners the phase is per-pixel (B0
   inhomogeneity, coil sensitivities) and the calibration block becomes a
   per-pixel phase map, not a scalar. Out of scope for this simulator-only fix
   but worth noting in the report.

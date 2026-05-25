# Why the T1-fit MAPE blows up at SNR=2.5: schedule, prefix-flip, and an assumed-zero phase

**Date started:** 2026-05-15
**Last revised:** 2026-05-16
**Driver scripts:** `scripts/run_t1_fit_sweep.py` → `scripts/t1_fit_vs_true.jl`
**Plot script:** `scripts/plot_recovery_curves.py` (new)
**Sibling document:** `scripts/REMOVE_PHASE_ASSUMPTION.md` — the proposed fix
for the phase-sensitive recon bug uncovered here.

---

## 1. Original observation

Mean per-sphere MAPE on the T1 fleet *increased* with the time budget
at SNR=2.5, Npe=32, Nfe=128 — counter-intuitive, since more scan time
should give more accurate fits.

| Budget | Mean MAPE (magnitude recon, original sweep) |
|---|---|
| 80 s  | ~ 20 % |
| 120 s | ~ 25 % |
| 160 s | ~ 30 % |
| 240 s | ~ 40 % |

## 2. First hypothesis (overturned): Rician bias

> The CR optimiser assumes Gaussian noise, but magnitude reconstruction
> gives Rician noise. The CR optimiser schedules TIs near the null
> because they are theoretically informative; those measurements collect
> a magnitude-floor bias the fitter can't distinguish from real signal.
> Larger budgets give the optimiser more freedom to schedule additional
> near-null TIs, amplifying the bias.

The hypothesis predicts more near-null TIs at higher budgets. Reading
the saved schedules:

| Budget | TIs (s) | TRs (s) | σ_image |
|---|---|---|---|
| 80  | 0.047, 0.331, 0.313, 0.390 | 0.51, 0.56, 0.89, 0.51 | 0.104 |
| 120 | 0.580, 0.286, 0.029, 0.062 | 1.92, 0.60, 0.54, 0.67 | 0.142 |
| 160 | 0.032, 0.033, 0.143, 0.644 | 0.50, 0.52, 1.04, 2.90 | 0.113 |
| 240 | 0.036, 0.148, 0.031, 0.587 | 0.82, 1.26, 0.63, 4.77 | 0.125 |

All budgets schedule the same 4 acquisitions. TIs do **not** concentrate
near nulls at higher budgets. What scales with budget is per-block TR
(some reach 4.8 s), and consequently the image-domain noise σ from the
SNR=2.5-on-peak calibration drifts up at the longer budgets.

## 3. Synthetic isolation test (Approach C)

To bypass Koma, the recon, and any pipeline confounders, run only the
forward model + fitter under controlled noise.

Script: `scripts/synthetic_rician_test.py`
Output: `scripts/runs/synthetic_rician/results.json`

Method: pull each sweep's saved TIs, TRs, Npe and image σ, compute
signed signal via `transient_mz_at_excite_npe`, draw 100 Monte-Carlo
trials per sphere under either Rician magnitude or signed-Gaussian
noise, fit with the production `fit_t1_generalized_ir`.

| Budget | MAPE (Rician) | MAPE (Gaussian/signed) | Bias (Rician) | Bias (Gaussian) |
|---|---|---|---|---|
| 80 s  | 31.1 % | 14.6 % | +6.4 % | +2.8 % |
| 120 s | 17.8 % |  8.5 % | −2.4 % | +0.5 % |
| 160 s | 31.9 % | 12.1 % | +10.6 % | −2.6 % |
| 240 s | **55.4 %** | 14.5 % | **+31.0 %** | −2.4 % |

The synthetic test reproduces the budget trend under Rician noise but
**not** under Gaussian/signed noise. The signed-bias direction is
positive and grows with budget (+6 → +31 %) — classic Rician
magnitude-floor fingerprint. Phase-sensitive MAPE stays roughly flat.

Rician bias is at least *capable* of producing the observed budget
trend at this SNR/schedule regime.

## 4. Confirming in the production pipeline

If Rician bias is the cause, rerunning the 240 s config with
`--phase-sensitive` should drop the mean MAPE close to the synthetic
prediction (~15 %). A 2×2 with clean signal separates schedule quality
from the recon mechanism.

| Config | Mean MAPE | Median MAPE | Max MAPE |
|---|---|---|---|
| 240 s, magnitude, SNR=2.5    | 41 %  | 36 % |   168 % |
| 240 s, phase-sensitive, SNR=2.5 | **585 %** | 33 % | **6132 %** |
| 240 s, magnitude, no noise      | **194 %** | 34 % |  2364 % |
| 240 s, phase-sensitive, no noise |  41 % | 34 % |   235 % |

This is the result that overturns the headline hypothesis.

- **All four medians are 33–36 %.** That is the floor — set by the
  *schedule itself*, not by noise model or recon choice.
- **Mean is dominated by which spheres rail-pin** in each condition.
  Noise and recon mode shuffle which sphere blows up; they do not
  change the schedule's information content.
- **Phase-sensitive, no noise (41 %)** is the cleanest condition
  possible — perfect signed signal, real fitter, no noise. The fact
  that this is *still* ~41 % mean / 34 % median says the schedule
  produces fundamentally inaccurate T1 estimates regardless of
  everything downstream.
- **Magnitude, no noise (194 %)** is a magnitude-IR pathology at
  infinite SNR: near-null samples deterministically snap to the wrong
  sign in the prefix-flip search, the fit lands in a wrong basin, and
  cannot escape. Noise smears those samples and *helps* by breaking
  the degeneracy — which is why magnitude+noise (41 %) looks better
  than magnitude+no-noise (194 %).
- **Phase-sensitive + noise (585 %)** trades Rician bias for a
  different failure: phase recovery is unreliable at low |Mz|, so
  short-T1 spheres whose null sits inside the TI grid (T1_12: T1=0.048,
  TIs include 0.031 and 0.036 — straddling null at 0.033) suffer
  noise-induced sign flips and rail-pin to the grid edge.

## 5. Visualising it: per-sphere recovery curves

The above is correct but unintuitive when read off a MAPE table. We
built `scripts/plot_recovery_curves.py` to show, per sphere:

- the signed M_z curve at T1_true (blue solid),
- the |M_z| curve (blue dotted) — the magnitude-recon view,
- the |M_z| curve at T1_fit (red dashed) — what the fitter believes,
- the **actual Koma-observed magnitudes** (black dots) when
  `--source koma` (reads `block_signals.csv`, a new per-block per-sphere
  CSV emitted by `t1_fit_vs_true.jl`).

### 5.1 b240s magnitude, SNR=2.5

![b240s magnitude, SNR=2.5](runs/t1_fit_vs_true/b240s_snr2p5_npe32fe128/recovery_curves_koma.png)

What it shows: long-T1 spheres (T1_1=1.879s, T1_2=1.432s, T1_3, T1_4)
have **one** observed marker past TI=0.6s. The recovery tail is
under-constrained → fitted (red dashed) curve can sit far from the
true (blue) curve while still matching every dot. Mid-T1 spheres
sit on top of the curve. Short-T1 spheres are fine.

### 5.2 b240s magnitude, no noise — the prefix-flip pathology

![b240s magnitude, no noise](runs/t1_fit_vs_true/b240s_nonoise_npe32fe128/recovery_curves_koma.png)

Mean MAPE **194 %** (matches §4). Notable rail-pins: T1_10 fits to
2.333s (true 0.095s, MAPE 2364 %). The markers themselves trace the
true V-shape of |M_z| fine; the fitter just lands in a wrong basin
because the abs() prefix-flip search has a deeper SSE minimum at the
wrong T1.

### 5.3 b240s magnitude, no noise, clean-recon (Hamming + zero-pad + 3×3 ROI)

![b240s magnitude, no noise, clean](runs/t1_fit_vs_true/b240s_nonoise_npe32fe128_clean/recovery_curves_koma.png)

Mean MAPE **452 %** — *worse* than without clean-recon. The markers
visibly snap closer to the |M_z| curves (PSF leak from neighbour
spheres is suppressed, confirming PSF leak was the source of the
marker scatter in §5.2). But removing that "smear" makes near-null
samples land deterministically at zero, deepening the prefix-flip
basin. T1_14 now rail-pins to 1.494 s (true 0.024 s, 6083 % MAPE).

**Counter-intuitive lesson:** PSF leak was paradoxically *helping*
the magnitude fitter by breaking the abs() degeneracy. Strip it
away and the fit gets worse, not better.

### 5.4 Manual schedule (14 log-spaced TIs, TR=5 s everywhere), no noise

![bMANUAL no noise magnitude](runs/t1_fit_vs_true/bMANUAL_nonoise_npe32fe128/recovery_curves_koma.png)

This schedule was hand-crafted to span every sphere's null with
anchor TIs at 2.0 and 2.8 s. Mean MAPE **42.7 %**, basically
identical to `--noise 0.12` (42.8 %) and the SNR=2.5 case (64 %
because of a separate calibration bug, §6.1). Long-T1 spheres still
rail-pin to 0.01 — the abs() prefix-flip basin is so deep that even
a perfectly-designed schedule cannot escape it without phase-sensitive
recon or Rician-aware fitting.

### 5.5 Manual schedule, no noise, phase-sensitive

![bMANUAL no noise phase-sensitive](runs/t1_fit_vs_true/bMANUAL_nonoise_npe32fe128_ps/recovery_curves_koma.png)

Mean MAPE **41.1 %** — only marginally better than magnitude. The
markers are scattered, with no clear sign structure, and the global
ρ refit collapsed to 0.05 (vs 0.94 in magnitude mode). The recon's
`real(IFFT(ksp))` produces near-random signs at low |M_z| because
the recon **assumes the global complex phase φ is zero** and KomaMRI's
phase is not zero. This is the new bug — see §6.4 and
`REMOVE_PHASE_ASSUMPTION.md`.

## 6. New bugs uncovered in this session

### 6.1 `--snr` defaults to 2.5; `--noise` is silently overridden

`scripts/t1_fit_vs_true.jl:46` sets `global target_snr = 2.5` instead
of `nothing`. Passing `--noise X` does not disable the SNR auto-calib,
so `target_snr=2.5` wins and the resulting σ is whatever the
first-block ksp_rms / 2.5 happens to be.

For the manual schedule this is catastrophic. Its first block is
TI=0.015 s — near the null for almost every sphere — so ksp_rms is
tiny and σ gets scaled up ~6× to compensate (image σ ≈ 0.79 vs
≈ 0.12 on the CR runs). The "manual run is broken" hypothesis we
chased early in the session was largely an artefact of this.

**Workaround until fixed:** pass `--snr 0 --noise X` to actually use
a fixed noise level. Or set `target_snr = nothing` as default.

### 6.2 `--noise` is k-space σ, not image σ

The flag is documented as "absolute k-space sigma" in the script
header, but the project's other SNR metrics (NEMA, background_std,
sphere SNR) are all image-domain. There is no constant FFT-scaling
ratio between the two: on b240s `(13.09, 0.125)` and on manual no
noise `(0.0, 0.787)`. The mapping isn't even monotone — see §6.3.

### 6.3 `background_std` is contaminated by sphere signal

Image-domain background ROI is supposed to measure noise. On the
manual schedule (every block TR=5 s → every sphere near peak signal),
`background_std = 0.787` even when `noise_sigma_abs = 0.0`. That is
PSF leak from bright spheres bleeding into the "background" region,
not noise. The whole `snr_report` table is unreliable on bright
schedules; treat `background_std` as an upper bound on noise + leak,
not a noise estimate.

### 6.4 `kspace_to_image` assumes the rotating-frame phase is zero

`src/rl/e2.jl:120-123`:

```julia
function kspace_to_image(ksp::Matrix{ComplexF32}; phase_sensitive::Bool=false)
    img = fftshift(ifft(ifftshift(ksp, (1, 2)), (1, 2)), (1, 2))
    phase_sensitive ? Float32.(real.(img)) : Float32.(abs.(img))
end
```

Taking `real()` of the image is correct only if the image is purely
real in the rotating frame, i.e. the global complex phase φ = 0.
KomaMRI's RF/gradient chain imposes a deterministic but non-zero φ,
so `real(img · e^{iφ}) = cos(φ) · signed_signal`. The sign of cos(φ)
varies, producing the scattered markers in §5.5.

The `clean_kspace_to_image` variant in
`scripts/t1_fit_vs_true.jl:217-240` has the same bug.

**Does this affect magnitude recon too?** A global φ_block does *not*
affect abs readings — `|signal · e^{iφ}| = |signal|`, so the magnitude
path silently eats the bug. But two related phase mechanisms *do*
distort magnitude readings:

- **Coherent complex PSF leak.** Neighbour-sphere bleed into a target
  pixel adds as complex numbers, not magnitudes. Within a block all
  spheres share the same global φ_block, so their *relative* phases
  come from the sign of M_z and the (real, for symmetric trajectories)
  PSF coefficient. Leak from a sub-null neighbour therefore *subtracts*
  coherently from the target's complex signal before abs() is taken —
  exactly what we saw in §5.3 (clean-recon = less leak = markers snap
  closer to curve = more deterministic prefix-flip = worse fit).
- **Rician magnitude-floor bias.** `|signal + n_R + i·n_I|` has a
  positive bias near zero signal because complex Gaussian noise has
  random phase that prevents cancellation under abs(). This is the
  original Rician hypothesis (§2–§3) and is independent of the global
  φ_block bug — it persists at any φ.

So magnitude recon is *robust to the global-phase bug* but still
suffers from coherent PSF leak and the Rician floor. Neither is fixed
by `--phase-sensitive` alone (the latter is killed by phase-sensitive
recon; the former is not). The diagnostic in
`REMOVE_PHASE_ASSUMPTION.md` step 3 should also check whether φ varies
*spatially* across the phantom mask — if so, that would imply a
B0-like inhomogeneity that *does* affect magnitude recon and would
need separate handling.

Full fix proposal: **`scripts/REMOVE_PHASE_ASSUMPTION.md`**
(~25 lines across two files, ~2 hours).

## 7. Revised conclusion

The original Rician-bias hypothesis is **partially correct** but not
the dominant mechanism.

1. **Rician bias is real** and the synthetic test (§3) isolates it
   cleanly. For medium/long-T1 spheres with well-determined phase,
   switching to phase-sensitive recon should substantially improve
   fits — but only after the φ=0 assumption (§6.4) is fixed.

2. **The dominant cause of the 240 s mean-MAPE blow-up is the
   schedule itself.** The CR-optimal solver picks a 4-block
   schedule that covers almost no information in T1 ∈ [0.7, 2.0].
   Long-T1 spheres can't be fit accurately from this schedule even
   with infinite SNR and ideal phase recovery.

3. **The naïve "mean MAPE" metric is misleading at this budget.** A
   handful of rail-pinning spheres dominate. The median is a more
   honest summary — and the medians are essentially flat across
   noise/recon conditions, confirming the schedule is the bottleneck.

4. **Magnitude+no-noise is paradoxically the worst case.** Noise and
   PSF leak both *help* the magnitude fitter by smearing near-null
   samples enough to break the abs() prefix-flip degeneracy. Removing
   them (zero noise + clean-recon) makes more spheres rail-pin.

5. **Phase-sensitive recon, as currently implemented, is not a fix.**
   `kspace_to_image` assumes φ=0 (§6.4); the resulting signed values
   are near-randomly flipped at low |M_z|. Mean MAPE on the manual
   schedule comes out ~41 % no-noise, hardly different from
   magnitude.

## 8. Recommendations / next steps

Ordered by cost / value:

1. **Fix the phase-sensitive recon (assumed-zero φ).** Single
   reference-block calibration, ~25 lines, ~2 hours.
   See `REMOVE_PHASE_ASSUMPTION.md`.

2. **Fix the `--snr` default and the `--noise` semantics.** Default
   `target_snr = nothing` so `--noise X` actually wins. Rename
   `--noise` to `--noise-ksp` (or add a separate `--noise-img` that
   converts via the analytic IFFT scaling). 30 minutes.

3. **Replot the original sweep using median MAPE** instead of mean.
   The budget trend likely flattens, and that's a more accurate
   characterisation of "how good is the fit on a typical sphere".
   30 minutes.

4. **Investigate the CR optimiser objective and `n_block_grid` at
   240 s.** The default scans `[4, 6, 8, 10, 14, 18]` but 240 s
   consistently lands on `n_blocks=4`. Either the objective
   under-weights long-T1 coverage, or `n_blocks=4` genuinely is the
   CR minimum for the assumed Gaussian noise. In either case, a
   schedule with more TIs covering [0.2, 1.5] would help.

5. **Implement bias-corrected squared-magnitude fitting** (~30 lines,
   half a day). Use the identity `E[m²] = ν² + 2σ²` to subtract the
   Rician floor before fitting. Same closed-form A scan, just
   substitute the data and model. Avoids the phase-recovery pathology
   of `--phase-sensitive` while removing first-order Rician bias.

6. **Full Rician MLE fit** (~80 lines, 1–2 days). Replace SSE with
   the Rician negative log-likelihood
   `−log L = Σ [(m² + ν²)/(2σ²) − log I₀(mν/σ²) − log(m/σ²)]`, add
   a 1D inner Brent minimisation for A at each T1 grid point. Use
   `besselix` from SpecialFunctions.jl for numerical stability. Most
   rigorous; provides a textbook three-way report figure
   (Gaussian / Rician-corrected / phase-sensitive).

7. **Fix `background_std`** so it doesn't include sphere-signal leak
   — either by tightening the background ROI to a sphere-free corner,
   or by computing the analytic noise σ from `noise_sigma_abs`
   and the IFFT normalisation. 1 hour.

## 9. Artefacts produced this session

- New per-sphere recovery-curve plotter:
  `scripts/plot_recovery_curves.py`.
- New per-block per-sphere signal dump:
  `block_signals.csv` (now emitted by `t1_fit_vs_true.jl`).
- Real-pipeline runs:
  - `runs/t1_fit_vs_true/b240s_snr2p5_npe32fe128/`           (mag, SNR=2.5)
  - `runs/t1_fit_vs_true/b240s_snr2p5_npe32fe128_ps/`        (PS,  SNR=2.5)
  - `runs/t1_fit_vs_true/b240s_nonoise_npe32fe128/`          (mag, no noise) ★
  - `runs/t1_fit_vs_true/b240s_nonoise_npe32fe128_clean/`    (mag, no noise, Hamming + zero-pad) ★
  - `runs/t1_fit_vs_true/bMANUAL_snr2p5_npe32fe128/`         (mag, σ inflated by §6.1 bug) ★
  - `runs/t1_fit_vs_true/bMANUAL_nonoise_npe32fe128/`        (mag, no noise) ★
  - `runs/t1_fit_vs_true/bMANUAL_nonoise_npe32fe128_clean/`  (mag, no noise, clean-recon) ★
  - `runs/t1_fit_vs_true/bMANUAL_nonoise_npe32fe128_ps/`     (PS, no noise, broken-φ recon) ★

  ★ produced this session, includes `recovery_curves_koma.png` and
  `t1_fit_vs_true.png`.
- Synthetic test script + JSON (§3):
  - `scripts/synthetic_rician_test.py`
  - `scripts/runs/synthetic_rician/results.json`
- Companion plan document for the highest-value next step:
  `scripts/REMOVE_PHASE_ASSUMPTION.md`.

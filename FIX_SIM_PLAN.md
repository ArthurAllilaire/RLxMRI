# FIX_SIM_PLAN.md — post-M3 simulation correctness pass

Author: Arthur Allilaire · Date: 2026-05-11 · Trigger: M3 supervisor feedback
(`project_context/meeting_notes/M3.md` §9).

Andreas flagged four issues. The headline one — missing `fftshift` around the
2D IFFT in `src/rl/e2.jl:400` — invalidates the per-sphere ROI extraction that
every downstream result rests on (V5, V12, CR-optimal comparison, the §20
SSE-landscape kill of the C1 claim). This document fixes the simulator, fixes
the noise model, bumps imaging resolution, and — most importantly — backfills
the end-to-end tests that should have caught the FFT-shift bug on day one.

---

## 1. Why we need the FFT shift

### 1.1 What KomaMRI returns

`simulate(phantom, seq, Scanner())` returns ADC profiles in **acquisition
order**. For a Cartesian readout, that is one row of k-space per PE shot, with
samples ordered from `-k_max` to `+k_max` along the readout direction. After
stacking PE rows in shot order, the resulting `ksp` matrix has DC (k=0) at the
**centre** of the array — index `(Npe÷2+1, Nfe÷2+1)` — not at index `(1,1)`.

### 1.2 What `ifft` expects

Julia's `FFTW.ifft` (and NumPy's, and MATLAB's) uses the standard DFT
convention: input index `(1,1)` is DC; the corners of the array are low
frequency; the centre of the array is the Nyquist frequency. To reconstruct
an image correctly from acquired k-space, the pipeline must be:

```
ksp_acquired                       (DC at centre)
        │ ifftshift               ← move DC from centre to (1,1)
        ▼
ksp_dftorder                       (DC at (1,1))
        │ ifft                    ← reconstruct image, DC at (1,1)
        ▼
img_dftorder                       (image centre at (1,1), corners at edges)
        │ fftshift                ← move image centre from (1,1) to centre
        ▼
img_centred                        (image centred — phantom-aligned)
```

The current code is `img_complex = ifft(ksp, (1, 2))` with no shifts at all.
That is equivalent to feeding DC-at-centre data into a DFT routine that
believes DC is at `(1,1)`. The result is the true image multiplied pointwise
by `(-1)^(i+j)` (a chequerboard) **and** wrapped half-FOV in both directions.

### 1.3 The visible consequence in E2

`_e2_build_episode_phantom` (`e2.jl:313–320`) computes ROI pixels from sphere
positions in metres using `mod(round(cx · Nfe / FOV), Nfe) + 1`. That formula
assumes the **phantom centre lands at image pixel `(1,1)`** — i.e. an
unshifted reconstruction. Today, the phantom is at the origin of physical
space (`x=y=0` after the random translation), so `round(0 · Nfe / FOV) + 1 = 1`,
and the ROI for a centre-of-FOV sphere is pixel `(1,1)`.

Two facts then collide:

1. With the buggy recon (no shifts), pixel `(1,1)` happens to receive the
   half-FOV-wrapped, chequerboard-modulated DC signal. For an isotropic
   phantom this is **roughly the right amplitude** for the centre sphere —
   which is exactly why a smoke-test "the agent learns something" never
   tripped on it.
2. For **off-centre spheres**, the wrap collapses the alias: a sphere at
   `(+FOV/4, 0)` lands at image-domain pixel `(Nfe÷4+Nfe÷2, ·)` after the
   correct shift, but at `(Nfe÷4, ·)` under the current ROI formula — i.e.
   half-FOV away from where the magnitude actually peaks. The ROI samples
   either zero, noise, or another sphere's tail.

The agent has been training against a corrupted measurement function where
the per-sphere signal is partly the right sphere, partly the diagonally-opposite
sphere, and partly background. That this still produced a plausible-looking
TI-vs-T1 correlation (§2.4 of M3.md) is a (small) miracle and probably a
testament to the chequerboard-modulated centre pixel still being roughly
correct for the highest-T1 sphere on the plate.

### 1.4 The fix

Replace `e2.jl:399–400` with:

```julia
img_complex = fftshift(ifft(ifftshift(ksp, (1, 2)), (1, 2)), (1, 2))
```

And change the ROI mapping in `_e2_build_episode_phantom` to centred indexing:

```julia
ife = mod(round(Int, cx * env.Nfe / env.FOV) + env.Nfe ÷ 2, env.Nfe) + 1
ipe = mod(round(Int, cy * env.Npe / env.FOV) + env.Npe ÷ 2, env.Npe) + 1
```

`ifftshift` and `fftshift` differ only for odd `N`; since defaults are even
(`Nfe=64`, `Npe=32` after item §3), either is fine in the most common path,
but `ifftshift`-then-`fftshift` is the safe pairing for arbitrary `N`.

### 1.6 The new test surfaced a *second* imaging-pipeline bug

While implementing **T8** (phantom-aligned recon test, §5.1), the recon failed
to resemble the T1-plate phantom under **every** combination of `fftshift` /
`ifftshift` around the IFFT — Pearson correlation between `|image|` and the
phantom occupancy stayed at ≈ 0; all 14 brightest pixels missed every real
sphere. Inspection of `|ksp|` per PE row shows a strongly asymmetric
distribution (smooth rise to row 16, then a flat plateau at rows 17–32) which
is **not** what a correctly-encoded k-space of a multi-sphere phantom should
look like — a normal k-space would have a roughly symmetric roll-off from DC.

This points to a second bug somewhere in `ir_se_2d_sequence` or in how the
env stacks `raw.profiles` into `ksp`. Hypotheses to chase, ordered by
likelihood:

1. **Cross-shot magnetisation state.** Each shot's residual transverse / Mz
   state may be leaking into the next, biasing the back half of the PE
   schedule. KomaMRI is stateful across `Sequence` blocks; if the TR delay
   isn't long enough or the spoiler/recovery is incomplete, the second half
   of the PE encoding sees a different effective phantom than the first.
2. **PE ordering / sign mismatch.** Stacking `raw.profiles[k]` into row `k`
   assumes that profile-k corresponds to ky_steps[k] in the order the
   sequence builder declared. If KomaMRI internally reorders ADC blocks (or
   if the prewinder sign convention disagrees with what the stacker
   assumes), every PE row goes to the wrong array index → the image
   collapses to a smear.
3. **Half-pixel ky offset.** `ky_steps[k] = (k - (Npe+1)/2) · Δky` puts DC
   at *index 16.5* between two array rows. After `fftshift` this manifests
   as a half-pixel y-shift but should not destroy structure — likely not
   the dominant issue but worth fixing to `ky_steps[k] = (k - Npe÷2 - 1) ·
   Δky` so DC sits exactly on a row.

The §1.4 FFT-shift fix is **necessary but not sufficient**. The work order
in §6 below now also needs a "diagnose why the T1-plate slice does not
resolve" step before T8 can flip green.

### 1.5 Why this also affects the §20 SSE diagnostic

`python/diagnose_sse_landscape.py` reuses `env.block_mags` from training runs.
Those magnitudes were sampled at the wrong pixel, so the "data" the SSE is
evaluated against is partly cross-sphere contamination. The "truth is at the
76th percentile" finding in M3.md §2.6 can change qualitatively after the
fix — possibly making the C1 claim recoverable. **Re-running V12 and §20
after this fix is non-optional before drafting Ch4.**

---

### 2.3 Documentation cost

After `abs()`, residuals are Rician, not Gaussian. The fitter's Gaussian
likelihood is therefore misspecified at low SNR — exactly the short-T1 regime
that fails in §20. We document this in a header comment but **do not** swap
to a Rician likelihood in this pass; the right fix is phase-sensitive recon
(M3.md §4 Option A), which collapses Rician → Gaussian at source. Note it in
Ch4 limitations.

---

## 3. Imaging resolution bump

Andreas: *"increase the resolution to 64 × 32 even if that increases training
time."*

Change defaults at `e2.jl:168–169` from `Nfe=16, Npe=8` → `Nfe=64, Npe=32`.
Mirror in `python/qalibremd_gym/env_e2.py`. Compute cost: per-block simulator
work scales roughly with `Npe` (more shots) and weakly with `Nfe` (longer ADC
window). Expect ~4× slower training. Validate end-to-end runtime on a 5-block
canary episode before kicking off a full V12 retrain.

Knock-on: observation dim becomes `64 × 32 + 2 · n_spheres + 3 = 2059` for
`n_spheres = 14` (was `131`). PPO policy network is currently a default MLP;
moving the image into a small CNN feature extractor is probably worth it but
out of scope for this fix — flag for the follow-up V13 run.

---

## 4. Survey of current tests, and gaps

Existing tests fall into three layers. None of them assert that the image
produced by `_e2_simulate_step` is aligned with the phantom — which is why
the FFT-shift bug survived.

### 4.1 What `test/` actually covers

| File | Layer | What it tests | Coverage of the recon pipeline |
|---|---|---|---|
| `test_materials.jl` | data tables | T1/T2/PD/fiducial arrays have right length, ordering, field-dependence | none |
| `test_geometry.jl` | geometry | sphere voxelisation volume converges; plate has 14 spheres at correct z; fiducial grid count and spacing | none |
| `test_builder.jl` | phantom assembly | descriptors → Phantom; subset selection; field selection; background-water exclusion | none |
| `test_augment.jl` | augmentation | rotation is orthogonal & det 1; translation shifts means; T1/T2 jitter has expected σ; PD clipped to [0,1]; B0 jitter sets Δw | none |
| `test_determinism.jl` | reproducibility | seeded phantom builds bit-identical; different seed produces different T1s | none |
| `test_simulation.jl` | KomaMRI smoke | single-sphere SE: ADC returns nADC samples; magnitudes decay; T2 fit recovers truth to 10 % | indirectly tests `simulate()`, but no image-domain assertions |
| `test_baseline.jl` | analytical fits + E0 | closed-form IR/SE fits unbiased; sequence-builder block counts; `run_conventional_baseline` T1/T2 MAPE < 3 % | exercises the full E0 pipeline end-to-end on a built phantom **without imaging** — uses single-spin signals, not images |
| `test_e1.jl` | E1 env | analytic IR signal identities; fitter recovery from clean data; env construction; reset determinism; step termination; backend-consistency analytical vs simulator on single voxel | single-voxel only, no 2D recon |
| `test_e2.jl` | E2 sequence + fitter | `ir_se_2d_sequence` is Cartesian (no Gy during ADC); PE prewinder signs symmetric around 0; F1+ closed-form vs KomaMRI agrees to 5 %; fitter unbiased on adaptive schedules; steady-state fitter is provably biased on Npe-shot data | k-space *structure* is checked (gradient signs and PE counts), but the **forward path k-space → image → ROI value → fitter** is never tested as a whole |

### 4.2 The hole that let the FFT-shift bug through

`test_e2.jl` does precisely the right thing one layer below the image:
`F1+ matches KomaMRI on Npe-shot IR-SE single-spin` (line 335) confirms the
**signal at the sphere** matches the forward model. But it pulls that signal
from `simulate(single_spin_phantom, …)` directly, bypassing
`_e2_simulate_step` and its IFFT entirely. So the simulator is verified, the
fitter is verified against the simulator, but the **glue that turns k-space
into per-sphere magnitudes** has never been tested.

Equivalently: every test that ever passed a "magnitude" to the fitter
computed that magnitude analytically or from a phantom-of-one. Multi-sphere
imaging is the part that breaks under a bad recon, and we don't test it.

### 4.3 Other gaps worth flagging (beyond M3 feedback)

1. **No test that ROI pixels actually point at the right sphere.** Independent
   of FFT shift, the centred-indexing formula (§1.4) needs its own test.
2. **No noise-level integration test.** Once §2 lands we have an absolute σ;
   we should have a "fit MAPE < X% at SNR = Y" test that pins the noise-to-
   accuracy relationship and would catch a regression in any of {seq builder,
   simulate, recon, ROI mapping, fitter}.
3. **No test of the magnitude-vs-phase-sensitive recon switch.** The
   `env.phase_sensitive` branch at `e2.jl:401–410` exists but has no unit
   test asserting that the phase-sensitive image is signed and the magnitude
   image is non-negative.
4. **No test of the α-scaling correction at `e2.jl:423–426`.** A regression
   that drops the `sin_α` division would silently bias every fit.
5. **No test that pose augmentation moves sphere ROIs correctly.** The
   rotation/translation in `_e2_build_episode_phantom` and the ROI pixel
   computation downstream both implement the same coordinate change; a unit
   test should pin them together.
6. **No test that `info["T1_est"]` ever converges**. Currently the env's
   integration test is implicit ("training works"). A 5-block-with-fixed-seed
   test asserting per-sphere fit converges below e.g. 30% MAPE under the
   noiseless setting would catch broken-end-to-end regressions in CI.
7. **No fft-shift test on the python `env.py` wrapper.** The juliacall bridge
   reshapes `obs` and could in principle introduce its own ordering bug. A
   Python-side test should reshape and assert image-peak position too.
8. **No `KomaMRI.simulate` version test.** A KomaMRI bump that changes ADC
   ordering would silently corrupt our recon — pin the version in
   `Manifest.toml` and add a regression test that single-spike phantom →
   single peak at expected pixel.

---

## 5. New tests to add (the M3 deliverable)

Two test files. Each test is named, scoped, and lists the exact regression it
would catch. Wherever practical the test should be cheap (< 1 s) so it runs
in `julia --project=. test/runtests.jl` by default.

### 5.1 `test/test_e2_imaging.jl`

**T1. `image_peak_at_centre_sphere`** *(catches the FFT-shift bug)*
- Build a phantom containing one sphere at the FOV centre `(0, 0)`.
- Run `_e2_simulate_step` once at long TI (`TI = T1·ln2`), zero noise.
- Assert `argmax(image_mag) == (Npe÷2+1, Nfe÷2+1)` (within ±1 pixel).
- **Catches:** any combination of missing `fftshift` / `ifftshift` /
  swapped axes / off-by-one in ROI indexing. Bare-`ifft` recon will put the
  peak at `(1,1)` and fail loudly.

**T2. `image_peak_at_offcentre_sphere`** *(catches half-FOV aliasing)*
- Sphere at `(+FOV/4, +FOV/4)`.
- Assert peak at the corresponding centred-indexing pixel
  `(Npe÷2+1 + Npe÷4, Nfe÷2+1 + Nfe÷4)` (mod N, ±1 pixel).
- **Catches:** the specific failure mode where the centre-sphere accidentally
  works because of the chequerboard symmetry, but off-centre spheres land at
  the wrong pixel. This is the test that makes T1 strict.

**T3. `image_symmetry_two_spheres`** *(catches phase mishandling)*
- Two equal-amplitude spheres at `(+x0, 0)` and `(-x0, 0)`.
- Assert `image_mag[:, j_left] ≈ image_mag[:, j_right]` to ≤ 1 % under
  zero noise.
- **Catches:** asymmetric phase ramps in `ir_se_2d_sequence` and any
  half-pixel offset in the recon.

**T4. `image_value_matches_closed_form_centre`** *(the end-to-end accuracy
test Andreas asked for)*
- Single sphere at FOV centre, long T2 so finite ADC decay is negligible.
- Run a 5-TI schedule.
- Compute `S_pred = sin(α_exc) · (1 − 2·exp(−TI/T1)) · exp(−TE/T2)`.
- Assert `image_mag[centre] / image_mag[centre, first_block] ≈
   |S_pred| / |S_pred_first|` per block to ≤ 5 % (zero noise, ≥ 64×32 grid).
- **Catches:** any forward-path error from seq generation through to image
  amplitude. This is the test we should always have had.

**T5. `pose_augmentation_moves_sphere`** *(catches ROI-mapping vs phantom-
transform drift)*
- Same single-sphere phantom, but enable a known rotation `(rx, ry, rz)` and
  translation `(tx, ty, tz)` via `env.episode_rotation` /
  `episode_translation_m`.
- Assert `argmax(image_mag)` lands at the pixel predicted by applying the
  same `R · c + t` transform and then the centred-indexing formula.
- **Catches:** any divergence between the phantom-side and ROI-side
  application of the pose transform. Closes gap §4.3 (5).

**T6. `phase_sensitive_image_is_signed`** *(catches recon-mode switch)*
- Build single sphere, set `env.phase_sensitive = true`.
- Run a TI in the inverted regime (`TI < T1·ln2`).
- Assert `image_mag[centre] < 0`.
- Flip to `phase_sensitive = false`, assert `image_mag[centre] > 0` and
  `abs.(image_mag) ≈ image_mag` element-wise.
- **Catches:** regressions in the `if env.phase_sensitive ... else ...`
  branch at `e2.jl:401–410`. Closes gap §4.3 (3).

**T7. `alpha_scaling_recovers_amplitude`** *(catches the α correction)*
- Same single sphere, fixed TI, two different `α_exc` values.
- After dividing each block's recorded magnitude by `sin(α_exc)`, the two
  values must agree to ≤ 2 %.
- **Catches:** removal or sign-flip of the `sin_α` division at
  `e2.jl:423–426`. Closes gap §4.3 (4).

**T8. `noise_is_complex_gaussian_independent`** *(catches the §2 fix)*
- Snapshot 1000 noise realisations at fixed σ on a zero-phantom k-space.
- Assert `mean(real(noise)) ≈ 0`, `mean(imag(noise)) ≈ 0`, `std(real) ≈ σ`,
  `std(imag) ≈ σ`, `cor(real, imag) ≈ 0`.
- **Catches:** regression to relative-RMS noise or accidental coupling
  between real/imag channels.

**T9. `multi_sphere_fit_converges_noiseless`** *(end-to-end success test)*
- 3-sphere phantom (one short, one mid, one long T1), noiseless.
- Run a 6-block CR-optimal schedule (hard-coded).
- Assert per-sphere fit `MAPE < 5 %` after final block.
- **Catches:** any regression in the full chain seq → simulate → ksp →
  ifftshift → ifft → fftshift → abs → ROI sample → α-correct → fitter.
  This is the integration test, the slowest in the file (~5–10 s), but the
  one that gives a single signal for "did we break the pipeline".

**T10. `fit_mape_at_known_snr`** *(noise calibration anchor)*
- Same 3-sphere phantom as T9, set `noise_sigma_abs` such that SNR at peak
  ≈ 30 (single-shot k-space sample).
- 50 seeds, assert `median MAPE < 10 %`, `p90 MAPE < 25 %`.
- **Catches:** mis-scaled noise (would shift MAPE by an order of magnitude),
  fit-σ regressions. Closes gap §4.3 (2).

### 5.2 `python/tests/test_e2_imaging.py`

**P1. `env_obs_image_is_centred`** *(catches juliacall-side reshape bugs)*
- `env = QalibreMDE2Env(seed=0, n_spheres=1, centre_only=True, noise=0)`.
- `obs, _ = env.reset()`; `obs, _, _, _, info = env.step(action)`.
- Reshape the image part of `obs` to `(Npe, Nfe)`; assert peak at centre.
- **Catches:** the Python-side reshape order — Julia and NumPy disagree on
  column-vs-row major; a `.reshape(Npe, Nfe)` vs `.reshape(Nfe, Npe).T` mix-up
  here would re-introduce the bug only on the Python side.

**P2. `env_t1_est_converges_seeded`** *(Python integration)*
- 5-block fixed schedule on the centre sphere with `noise = 0`.
- Assert `info["T1_est"][0]` within 5 % of `info["T1_true"][0]` after block 5.
- **Catches:** anything broken in the juliacall bridge that doesn't show up
  in the Julia tests.

### 5.3 Visual diagnostic (not a unit test, but a checkpoint)

Extend `python/diagnose_e2.py` to dump, for one fixed seed and the canonical
clinical IR-TSE schedule:

- `report_plots/sim_fix/phantom_slice.png` — true T1 map at the imaging slice
- `report_plots/sim_fix/recon_pre_fix.png` — recon before §1.4 fix
- `report_plots/sim_fix/recon_post_fix.png` — recon after §1.4 fix
- `report_plots/sim_fix/baseline_e0_vs_e2.png` — side-by-side at matched
  `(TI, TR)`

These get committed and become the human-eyeball reference for any future
recon change. They are the "make sure they look the same with the baseline"
deliverable Andreas asked for.

---

## 6. Execution order (TDD-style)

1. **Write T1, T2, T4, T9 first** — they will fail on `main`. Commit the
   red tests so the history records the bug we found.
2. Apply §1.4 (FFT-shift fix + centred ROI indexing) → T1, T2, T4 pass.
3. Apply §2 (noise model) → write T8, T10; pass.
4. Add T3, T5, T6, T7 → pass.
5. Add P1, P2 (Python side) → pass.
6. Apply §3 (resolution bump). Re-run all of the above. Adjust ε in T4 if
   needed (higher resolution actually tightens the closed-form comparison).
7. Add §5.3 visual diagnostic. Commit reference PNGs.
8. **Re-run V12 + the §20 SSE-landscape diagnostic** on the fixed env. Do
   not draft Ch4 with the broken-env results — they may change qualitatively.

---

## 7. What this does NOT cover (deliberate)

- **MRzero/PDG vs KomaMRI agreement** beyond the existing single-spin gates
  in `python/gate/`. Out of scope for the fix; tracked separately.
- **Rician likelihood** in the fitter. Documented in §2.3; the real fix is
  phase-sensitive recon (M3.md §4 Option A), pursued in the next experiment.
- **Multi-coil simulation.** Andreas noted KomaMRI is single-coil and that
  this is idealised. Acknowledged as a Ch4 limitation; not addressed here.
- **CR-optimal recomputation** with the fixed env. Out of scope for this
  document but is a non-optional follow-up before Ch4.

---

## 8. Definition of done

- [ ] All tests in §5.1, §5.2 added and green on `julia --project=. test/runtests.jl`
      and `pytest python/tests/test_e2_imaging.py`.
- [ ] `e2.jl:400` uses `fftshift(ifft(ifftshift(...), ...), ...)`.
- [ ] `_e2_build_episode_phantom` uses centred ROI indexing.
- [ ] `noise_sigma_rel` renamed `noise_sigma_abs` everywhere; noise added as
      complex Gaussian.
- [ ] Defaults: `Nfe = 64`, `Npe = 32` in both `e2.jl` and `env_e2.py`.
- [ ] `report_plots/sim_fix/` reference PNGs committed.
- [ ] V12 retrained on the fixed env, §20 SSE-landscape diagnostic re-run,
      results appended to `M3_gate_log.md` under a "Post-fix" section.

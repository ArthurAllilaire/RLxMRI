<!--
  Polished KomaMRI chapter draft.

  Narrative arc kept deliberately simple:
  1. KomaMRI is the forward simulator, so its time discretisation is load-bearing.
  2. A noise-free T1 sanity check failed badly.
  3. I ruled out plausible MR explanations: spoiling, Gibbs ringing, voxelisation,
     phase handling, and k-space indexing.
  4. The pixel-grid overlay showed a time cliff, not a physical artefact.
  5. Minimal reproducers isolated two related floating-point timing bugs.
  6. Bug #1 was accepted upstream as PR #780; Bug #2 is PR #789 and remains the
     local fork dependency at the time of writing.

  Cut but potentially useful:
  - The earliest local hypothesis was "B1 leakage / one extra RF step". That was
    close in spirit but not the final upstream mechanism for Bug #1. Issue #779
    narrows Bug #1 to a rise-edge marker collapse that *removes* one RF dt at
    large absolute time. Bug #2 is the later closing-knot/rebasing collapse.
  - The first_bug scripts show the route through several wrong-but-useful
    hypotheses: Float32 state drift, RF thresholding, near-duplicate time knots,
    and B1 boundary leakage. I have kept the conclusion rather than every detour.
-->

# KomaMRI Simulator Bugs

Before training an RL agent to design MRI sequences, the simulation
environment itself had to be trusted. KomaMRI is the source of every image
and fitted T1 value used by the agent, so a simulator artefact would be
indistinguishable from a failure of the policy or reconstruction pipeline.
This chapter explains how two timing bugs in KomaMRI were found, reduced to
minimal reproducers, and fixed upstream.

## Simulator Context

KomaMRI is the simulation environment used throughout this project to train
and evaluate RL agents. It simulates the magnetisation of each spin in a
`Phantom` under the variable magnetic fields defined by a `Sequence`. At the
user level, the simulation is called as:

```julia
raw = simulate(phantom, sequence, scanner; sim_params)
```

A KomaMRI `Phantom` is represented as a cloud of independent spin
isochromats: groups of nearby nuclei that are assumed to share the same
position $(x,y,z)$, tissue parameters
$(T_1, T_2, T_2^\ast, \rho, \Delta\omega)$, and local resonance frequency.
Here $\rho$ is proton density and $\Delta\omega$ is off-resonance. In this
project these phantoms were built using the MRISystemPhantom package
described in the previous section. The `Sequence` is the scanner timeline:
RF pulses, gradient waveforms, delays, and analogue-to-digital (ADC)
sampling windows. The `Scanner` stores field strength and hardware settings.

KomaMRI integrates each isochromat's magnetisation vector
$\mathbf{M}=(M_x,M_y,M_z)$ forward through the sequence. RF blocks rotate
the magnetisation vector, precession blocks apply $T_1/T_2$ relaxation and
gradient-induced phase, and ADC blocks record the transverse magnetisation
$M_{xy}=M_x+iM_y$ summed over all isochromats. At each ADC time point $t$,
this sum gives one complex measurement of the object under the gradients
being applied at that instant. In a Cartesian acquisition, those measurements
correspond to samples at specific k-space locations; after arranging them
onto the k-space grid, an inverse FFT gives the reconstructed image.

Internally, KomaMRI converts the continuous sequence description into a
discrete time grid. RF pulses are sampled at time intervals of `Δt_rf`, while
gradient and free-precession periods are sampled at time intervals of `Δt`.
KomaMRI then applies the Bloch equations step-by-step on this grid for every
isochromat. The timeline is split into simulation blocks to reduce memory
use. Blocks are classified into two regimes: excitation, where RF fields are
active and the full RF rotation must be simulated; and precession, where the
RF field is zero and the update reduces to relaxation plus gradient-induced
phase evolution.

This combination of faithful Bloch simulation and computational efficiency is why KomaMRI was a good fit for this project. The key simplifying assumption is that spin isochromats do not interact with one another, a standard assumption for this type of MRI simulator. This allows the phantom to be divided and parallelised across CPU threads or GPU kernels, with the per-thread signal contributions summed at ADC samples.

The larger modelling assumptions are elsewhere. Tissue properties are fixed
for each isochromat, motion and diffusion are not included unless explicitly
modelled, and relaxation is represented through the bulk parameters $T_1$,
$T_2$, and $T_2^\ast$ rather than by simulating microscopic spin interactions
directly. The hardware is also idealised: RF pulses and gradients are applied
as specified in the sequence. Real scanners have effects such as eddy
currents, gradient delays, RF inhomogeneity, and imperfect calibration, so a
nominal $180^\circ$ inversion or refocusing pulse is never exactly perfect in
practice. These assumptions are acceptable for the controlled digital-phantom
experiments in this project, but they matter when interpreting simulator
results as scanner-realistic predictions.

This context matters because the bugs in this chapter were not in the Bloch
equations themselves, but in the discretised time axis used to apply them.
Event-boundary markers that should have remained distinct were rounded onto each other after enough cumulative simulated time.

<!--
  Figure choice:
  - Removed the voxelised T1 slice after review; it distracted more than it
    clarified in this chapter.
  - The KomaMRI documentation call graph is useful but probably too detailed
    for this chapter. It would fit better in an appendix or footnote if you
    want to discuss Koma's implementation extensibility.
  - Cut detail from the KomaMRI explainer: `return_type` can be raw/mat/state;
    `sim_method` defaults to Bloch() but can be changed to BlochDict() or
    BlochMagnus variants; custom methods can specialise initialize_spin_state,
    run_spin_excitation!, and run_spin_precession! via Julia multiple dispatch.
-->

## Initial Symptom: Noisy Recovery Curves

The first failure appeared in a simple validation experiment: a noise-free
inversion-recovery $T_1$ fit. An inversion-recovery experiment measures
$T_1$ by sampling the longitudinal magnetisation ($M_z$) at a ladder of
inversion times $T_I$
and fitting the recovery curve
$M_z(T_I)=M_0(1-2e^{-T_I/T_1})$. With no noise added, the fitted $T_1$ for
each phantom sphere should be close to the known ground-truth value, and each
recovery curve should be a smooth exponential. Instead, a tested schedule
with 14 inversion times and TR = 5 s produced a mean absolute percentage
error of 35.5 %. When plotted against inversion time, the sampled points
jumped between values that no smooth exponential recovery curve could
connect. Because this was not a noisy experiment, either the simulator,
reconstruction, or fitting pipeline had to be wrong, or there was a physical
phenomenon I had not yet accounted for.

![Noise-free inversion-recovery validation before the KomaMRI timing fixes. The schedule used 14 inversion times with TR = 5 s and no added noise. Fitted $T_1$ values should lie close to the diagonal, but the mean MAPE is 35.5 %, with several spheres failing catastrophically. Generated by `scripts/t1_fit_vs_true.jl` and rendered with `scripts/t1_fit_vs_true.py`.](/scripts/runs/with_fp_bugs/t1_fit_vs_true/bMANUAL_nonoise/t1_fit_vs_true.png)

![Per-sphere recovery curves from the same noise-free validation run. Black points are KomaMRI samples at the tested inversion times; the fitted curves show the expected smooth inversion-recovery shape. Several sampled points jump discontinuously away from any plausible exponential, indicating a deterministic error rather than measurement noise. Generated by `scripts/t1_fit_vs_true.jl` and rendered from its recovery-curve output with `scripts/plot_recovery_curves_koma.py`.](/scripts/runs/with_fp_bugs/t1_fit_vs_true/bMANUAL_nonoise/recovery_curves.png)

## Ruling Out MRI Explanations

My first assumption was that the bug was in my sequence or reconstruction,
not in KomaMRI. I investigated and ruled out the most likely MRI failure modes.

Leftover transverse magnetisation was the most plausible explanation. In a
multi-shot sequence, residual transverse magnetisation ($M_{xy}$) can survive
from one shot and refocus into a later ADC sample, producing shot-to-shot
signal variation that the simple inversion-recovery fit does not model. The
standard fix is a gradient spoiler, which I added around the refocusing pulse
and at the end of TR. This helped: the same noise-free run improved from
35.5 % mean MAPE to 23.1 %. However, that was still far too large an error.

A second way to reduce cross-shot contamination is to increase TR, giving
longitudinal magnetisation ($M_z$) more time to recover and transverse
magnetisation ($M_{xy}$) more time to decay. Confusingly, this increased the
MAPE. In hindsight, longer TR also meant more cumulative simulated time,
which was exactly the variable that triggered the real bug.

<!--
  Quantitative note:
  - with_fp_bugs/bMANUAL_nonoise:       mean MAPE 35.54 %, median 37.44 %
  - with_fp_bugs/bMANUAL_nonoise_spoil: mean MAPE 23.07 %, median 19.59 %
  This is useful because it shows why spoiling was a reasonable hypothesis,
  but also why it could not be the whole explanation.
-->

I also checked Gibbs ringing and voxelisation by plotting the reconstructed
images. This was why I added the Hamming-windowed reconstruction: if the
large errors were mainly caused by ringing from the sharp phantom boundaries,
the windowed image should have reduced them substantially. As the figure below shows, it did not. I also
investigated whether magnitude fitting was losing phase information, whether
the starting phase was being mishandled, and whether the 64 x 32 sampling
grid was offset from the intended pixel centres. None of these explanations
matched the observed failure.

## The Diagnostic That Exposed It

The decisive diagnostic was plotting the reconstructed image together with
the acquired k-space. In this sequence, phase-encode lines are acquired one
after another, so the vertical axis of the k-space panel is also cumulative
simulator time. A physical artefact such as ringing, voxelisation, or
residual coherence should vary with spatial frequency, object edges, phase,
or sequence state. Instead, the signal was healthy and then suddenly
corrupted after enough simulated time had elapsed. That pointed to KomaMRI's
event timing.

![Pixel-grid overlay before the fixes. For each inversion time, the top row shows the normal reconstructed image, the middle row shows the Hamming-windowed reconstruction used to test whether Gibbs ringing was the dominant problem, and the bottom row shows k-space magnitude. The Hamming window changes the ringing pattern but does not remove the corruption. The k-space panels are initially structured and then abruptly collapse into horizontal banding partway through the phase-encode direction. Since phase encode is acquired sequentially, this is a time cliff, not a spatial artefact.](/scripts/runs/with_fp_bugs/pixel_grid_overlay/stitched_npe64_nfe128_fov0p2_vox1p0mm.png)

## Minimal Reproducer

I reduced the problem to a single spin at the origin with no gradients:

```text
(180 degree inversion -> TI delay -> 90 degree excitation -> one ADC sample -> TR delay) x N
```

Every shot is physically identical, so the signal magnitude should be
constant after the first transient. Before the first fix, the minimal script
`scripts/koma_bug_minimal.jl` showed a deterministic jump:

```text
shot 1..9   (<= 135 s): |signal| = 0.47627
shot 10     (150 s):    |signal| = 0.58490   (+22.8 %)
shot 11..16:            |signal| = 0.58490
```

Changing TR moved the shot number but not the cumulative time band: the jump
appeared once total simulated time reached roughly 70-150 s. Running with
`precision="f64"` did not remove it, so this was not ordinary Float32
magnetisation drift.

## Bug #1: Rise-Edge Collapse

KomaMRI builds a discrete time vector around RF and gradient events. To mark
event edges it used a fixed separation,
`MIN_RISE_TIME = 1e-14` s, for example placing an RF rise marker at
`t1 + MIN_RISE_TIME`. At large absolute times the spacing between adjacent
Float64 numbers grows. Once the absolute time is large enough,
`t1 + 1e-14` rounds back to `t1`, so `unique!` removes the marker.

For Bug #1 this collapsed the leading RF edge. The RF interpolation then lost
one `Delta t_rf` of pulse area. With a 1 ms hard pulse and the default RF
time step, that is a non-negligible fraction of the pulse, which explains the
large signal jump in a supposedly identical-shot sequence.

I reported this as GitHub issue
[#779](https://github.com/JuliaHealth/KomaMRI.jl/issues/779), "Inaccurate RF
pulse duration causing 25% amplitude drift on sequences", and submitted
[#780](https://github.com/JuliaHealth/KomaMRI.jl/pull/780). The accepted fix
made event-edge markers adaptive to floating-point spacing using
`nextfloat`/`prevfloat`-style guards, so edge markers remain distinct even at
large absolute times. PR #780 was merged on 14 May 2026.

<!--
  Issue/PR status checked on 2026-06-05:
  - Issue #779 closed by PR #780.
  - PR #780 merged into JuliaHealth/KomaMRI.jl master on 2026-05-14.
  - PR text reports KomaMRIBase tests passing locally, 537/537.
-->

## Bug #2: Rebased Closing-Knot Collapse

After Bug #1 was fixed, a residual version of the same class of problem
remained. The minimal reproducer was stable for the original 16-shot case,
but extending the same sequence to 24 shots produced another deterministic
jump:

```text
shot 1..17  (<= 255 s): |signal| = 0.476267
shot 18     (270 s):    |signal| = 0.722479   (+52 %)
shot 19..24 (>= 285 s): |signal| = 0.677159
```

This second failure was not caused by the original rise-edge marker. KomaMRI
also builds block-relative event times and then rebases them to absolute time
with `T0 .+ t`. The closing knot had already been separated by the fixed
`1e-14` s gap in block-relative coordinates, but after rebasing to large
absolute time that gap could again collapse. The bug therefore survived at
the rebasing sites in `get_samples` and `get_variable_times`.

The second investigation ruled out three alternatives:

- It was not transverse coherence: a no-gradient, single-spin sequence still
  failed.
- It was not shot count: TR = 10 s with 8 shots and TR = 2 s with 40 shots
  both failed near the same cumulative time in the gradient-heavy tests.
- It was not a steady-state transient: the early shots were stable, then the
  signal jumped discretely.

The supporting scripts in `scripts/koma_investigations/second_bug` show the
pattern. `01_time_vs_shotcount.jl` found drift onset at 66-80 s for several
TR/shot-count combinations in full IR-SE imaging. `02_periodic_or_one_off.jl`
showed that long gradient-heavy runs became chaotic after the first time
cliff rather than showing a smooth physical transient. `koma_bug_residual.jl`
then reduced the issue back down to identical no-gradient shots, where the
threshold was around 270 s.

I reported this as issue
[#788](https://github.com/JuliaHealth/KomaMRI.jl/issues/788) and submitted
[#789](https://github.com/JuliaHealth/KomaMRI.jl/pull/789) from
`ArthurAllilaire/KomaMRI.jl:fix/grad-fp`. The fix adds an idempotent
post-rebasing pass:

```julia
function _reseparate_closing_knot!(t)
    length(t) >= 2 || return t
    gap = max(MIN_RISE_TIME, eps(t[end]))
    t[end - 1] = min(t[end - 1], t[end] - gap)
    return t
end
```

Near zero this preserves the old `1e-14` s gap. At large absolute time the
gap grows to at least one unit in the last place, so the closing knot survives
rebasing. The PR applies the helper at the rebasing sites for RF and gradient
time vectors. At the time of writing, PR #789 is still open, so this project
depends on the fork at a pinned commit; that is the only deviation from the
registered KomaMRI packages.

<!--
  Issue/PR status checked on 2026-06-05:
  - Issue #788 is open.
  - PR #789 is open, one commit from ArthurAllilaire:fix/grad-fp into
    JuliaHealth:master.
  - PR #789 reports `koma_bug_residual.jl` fixed and KomaMRIBase tests passing,
    544/544.
-->

## After The Fixes

With the fixed KomaMRI fork, the same no-noise manual T1 experiment drops
from 35.5 % mean MAPE to about 0.5 %. The recovery curves become smooth
inversion-recovery curves, and the k-space time cliff disappears.

![Noise-free T1 fit after the fixes. The same manual schedule now gives mean MAPE 0.54 %, with all spheres close to the diagonal.](/scripts/runs/t1_fit_vs_true/bMANUAL_nonoise_npe32fe128/t1_fit_vs_true.png)

![Recovery curves after the fixes. The discontinuous jumps are gone and the simulated points lie on smooth recovery curves.](/scripts/runs/t1_fit_vs_true/bMANUAL_nonoise_npe32fe128/recovery_curves_koma.png)

![Pixel-grid overlay after the fixes. The k-space panels remain structured from top to bottom, and the remaining image differences are ordinary reconstruction effects rather than a simulator time cliff.](/scripts/runs/pixel_grid_overlay/npe32_nfe64_fov0p2_vox1p0mm_TI0p1_spoil/pixel_grid_overlay_variants.png)

## Transverse Spoiling Note

This section is written so it can be moved into the evaluation chapter.

Spoiling was still a useful investigation because it separated two effects
that otherwise looked similar. True transverse coherence can matter in short
TR multi-shot imaging: if residual `Mxy` survives into the next shot, the
simple inversion-recovery T1 model is no longer the correct forward model.
The spoiler gradients dephase that residual transverse component, at the cost
of extra sequence time.

However, the simulator bug was not a spoiling problem. The strongest evidence
is that the failure survived in a no-gradient single-spin sequence, where
there is no spatially varying phase history for a spoiler to correct. The
second investigation also found that drift onset tracked cumulative simulated
time rather than shot count or TR. That is the opposite of the expected
signature for multi-shot coherence.

The quantitative runs make the distinction clearer:

| Run | Mean effect |
|---|---:|
| Buggy manual T1, no spoiling | 35.5 % mean MAPE |
| Buggy manual T1, with spoiling | 23.1 % mean MAPE |
| Fixed manual T1, no spoiling | 0.5 % mean MAPE |
| Fixed dry pixel overlay, normal vs spoiled | 1.1 % relative k-space L2 difference |
| Fixed water-background pixel overlay, normal vs spoiled | 6.7 % relative k-space L2 difference |

So spoiling was not irrelevant: it changed the buggy images substantially and
still has a small real effect after the fixes, especially with background
water. But it could not explain a 20-35 % noise-free T1 error, and it could
not explain a deterministic signal cliff at a fixed absolute simulation time.
For later evaluation, the safe framing is: spoiling is a real modelling
choice for short-TR sequence design, while the KomaMRI bug was a separate
floating-point timing artefact that had to be removed before evaluating that
choice.

<!--
  Extra numbers from local artifacts:
  - With FP bugs, dry TI=0.1, TR=5, Npe=32/Nfe=64:
    normal-vs-spoiled k-space relative L2 = 0.843, image mean diff = 0.709
    relative to mean image magnitude. This is too contaminated by the timing
    bug to use as a clean physics number.
  - With FP bugs, TR=20, same setup:
    relative k-space L2 = 0.977, again dominated by the time bug.
  - After fixes, dry TI=0.1, TR=5:
    relative k-space L2 = 0.011, image mean diff = 0.008.
  - After fixes, water-background TI=0.1, TR=5:
    relative k-space L2 = 0.067, image mean diff = 0.026.
  These are useful if you want a fuller evaluation subsection later.
-->

<!--
=====================================================================
KomaMRI chapter — report-ready draft (fp_bugs_final.md)
Reconstructed from rough notes in fp_bugs.md + investigation scripts in
scripts/koma_investigations/{first_bug,second_bug} + GitHub issues/PRs.

Ground-truth sources used:
  - Issue #779 / PR #780 (bug 1, MERGED) "Fix large-time event edge collapse due to FP errors"
  - Issue #788 / PR #789 (bug 2, OPEN)   "Fix FP knot collapse at large absolute time"
  - Real PR #789 diff verified via `gh`/curl: touches Sequence.jl (get_samples),
    KeyValuesCalculation.jl (new _reseparate_closing_knot!), TimeStepCalculation.jl
    (get_variable_times), Grad.jl (_strictly_increasing_knots! made adaptive).
  - koma_bug_isolate.jl   -> the eps()-vs-MIN_RISE_TIME smoking gun (re-run, numbers below are real)
  - second_bug/00_NOTES.md + bug.md -> the H1/H2/H3 elimination and the residual-bug characterisation
  - t1_fit_vs_true.csv / recovery / pixel_grid_overlay figures (eae656a buggy vs 7ceced7 fixed)

Figure commits:
  eae656a_current-koma = registered KomaMRI (PR #780 merged, residual bug #788 still present) -> 39% MAPE
  d0d5c5f_both-bugs    = before EITHER fix -> catastrophic 924% MAPE (kept for a comment only)
  7ceced7_fixed        = both fixes applied -> 0.5% MAPE

HTML comments below flag material cut for length but worth reinstating if room allows.
The final "Transverse spoiling" section is written to be relocated to the Evaluation chapter.
=====================================================================
-->

# Validating the Simulator: Two Floating-Point Bugs in KomaMRI

Every reward the RL agent receives is ultimately a number returned by a KomaMRI
`simulate` call, and a digital twin is only as trustworthy as the simulator beneath
it. Before any RL result can be believed, the simulator itself must be validated.
This chapter documents that validation. A no-noise sanity check — fit a $T_1$ from a
clean inversion-recovery curve and confirm it matches ground truth — revealed two
floating-point bugs in KomaMRI's time-discretisation that corrupted long sequences
without raising any error. I diagnosed the cause, reduced it to a minimal reproducer,
and contributed two fixes upstream (issues #779 and #788; PR #780 merged, PR #789
proposed).

This work underpins **A1 (Ch. 3 — digital twin and baseline):** a quantitative
baseline is meaningful only if the forward model is correct. It also serves **C2
(simulation-in-the-loop RL):** training issues thousands of long multi-shot
sequences — the regime in which these bugs occur — so an undetected error here would
propagate to every downstream experiment.

## How KomaMRI works

> *KomaMRI simulates the magnetization of each spin of a `Phantom` for variable
> magnetic fields given by a `Sequence`.*

A `Phantom` is a cloud of spin isochromats; ours come from our own digital-twin
package, `MRISystemPhantom.jl`. Each isochromat lumps together nearby nuclei sharing a
position $(x,y,z)$, tissue parameters $(T_1, T_2, T_2^*, \rho, \Delta\omega)$, and
hence one magnetisation vector evolving under the Bloch equations in the rotating
frame. The user drives everything through one call:

```julia
raw = simulate(phantom, sequence, scanner; sim_params)
```

Internally, `simulate` first **discretises** the sequence onto a time grid — a coarse
raster $\Delta t$ for gradients, a finer $\Delta t_\text{rf}$ for RF — then walks it
in blocks, classifying each as either *excitation* (RF on: the full $3\times3$
magnetisation rotation is integrated) or *precession* (RF nulled: the physics
collapses to cheap analytic relaxation and phase accrual). This split is what makes
Koma efficient — the expensive rotation integrator runs only where needed, and memory
stays bounded by processing one block at a time. At each ADC sample the simulator sums
the transverse magnetisation over all spins to form the signal $\text{sig}[t]$.

One assumption makes this tractable: **each spin evolves independently.** Standard for
MRI simulators, it is exactly what lets Koma parallelise — the phantom is partitioned
across `Nthreads` CPU threads (or GPU kernels) and the per-thread signals summed at
the end. That scalability is the main reason we chose Koma for a sim-in-the-loop RL
project (C2). The same assumption brings the usual simplifications, all acceptable
here: $T_2^*$ is a bulk dephasing rate rather than microscopic spin–spin coupling,
and hardware non-idealities (eddy currents, $B_0/B_1$ inhomogeneity, concomitant
fields, off-resonance beyond the per-spin $\Delta\omega$) are not modelled. For a
calibration phantom under clean inversion-recovery and spin-echo sequences, none are
first-order effects.

<!-- CUT (move to a methods footnote if room): gyromagnetic-ratio convention.
KomaMRI uses gamma in cycles/s/T, gamma = 42.577e6 Hz/T, so k-space is
k(t) = gamma * integral G dtau with gamma in Hz/T (NOT 2*pi*gamma). Matters when
hand-deriving the prewinder/readout gradient areas in pixel_grid_overlay. Src: explainer.md -->

## Initial symptom: recovery curves that would not fit

The validation experiment is the simplest quantitative task in the project. With **no
noise added**, image each phantom sphere over a sweep of inversion times $\text{TI}$
and fit the monoexponential inversion-recovery model to the per-sphere signal:

$$
M_z(\text{TI}) = M_0\left(1 - 2e^{-\text{TI}/T_1}\right).
$$

Noise-free, this fit should be near-exact — sub-percent error on every sphere.
Instead the recovered $T_1$ values were substantially in error (mean absolute
percentage error, MAPE, of **39 %**; some spheres above 99 %), and the sampled points
did not lie on any monoexponential (Figs. 1–2).

![Buggy T1 fit](figs/eae656a_current-koma/t1_fit_vs_true/bMANUAL_nonoise/t1_fit_vs_true.png)

**Figure 1. Noise-free $T_1$ recovery on the buggy simulator (mean MAPE 39.4 %).**
Fitted vs. true $T_1$ (left, log–log) falls outside the $\pm10\%$ band rather than
tracking the diagonal; the fitted spatial map (right) departs from ground truth;
per-sphere MAPE (bottom) averages 39.4 %, with several spheres at the fitter's bounds
(≈99 %). With no noise added, errors of this size indicate a deterministic fault
rather than statistical variation.

![Buggy recovery curves](figs/eae656a_current-koma/t1_fit_vs_true/bMANUAL_nonoise/recovery_curves_koma.png)

**Figure 2. Simulated recovery points do not lie on any monoexponential.** Per-sphere
$|M_z(\text{TI})|$ (black) against the best-fit curves (blue/red). With no noise
present, the departure from the curve is a structured, reproducible deviation rather
than scatter, indicating a deterministic fault in the simulator, sequence, or
reconstruction.

With no noise, the error must originate in one of three places — the simulator, the
sequence, or the fit/reconstruction pipeline — unless it reflects a physical effect I
had not modelled. I examined the physical explanations first.

### Ruling out the physics

**Imperfect spoiling.** The monoexponential model assumes each shot starts from clean
longitudinal magnetisation; surviving transverse magnetisation refocused into a later
ADC would drift the signal. Adding gradient spoilers and lengthening $\text{TR}$
should fix that — but spoiling moved the MAPE *up* (39 % → 44 %; full analysis at the
end of this chapter) and longer $\text{TR}$ increased the error. This was in fact the
relevant clue, initially misread: longer $\text{TR}$ means longer total simulation
time, and the bug is triggered by *cumulative simulated time*, not by any coherence
pathway.

**Gibbs ringing / voxelisation.** I added a Hamming window and ROI mask in
`MRISystemPhantom`, tried phase-sensitive instead of magnitude reconstruction, and
questioned the assumed initial phase $\phi=0$. None resolved it — and the artefact was
far too large and too streaky to be Gibbs ringing, a few-percent oscillation confined
to sharp edges.

<!-- CUT (colour, from fp_bugs.md notes): much time on phase hypotheses — "are
magnitudes mis-represented?", "is phi actually 0, and can it drift shot-to-shot?",
"is the 64x32 grid landing on the right k-space coords?". All dead ends, but they
show the breadth of the physical search before I accepted it was numerical. -->

### Direct inspection of the image and k-space

The diagnostic that localised the fault was to set aside the fitted summary and
inspect the full 2D acquisition — the reconstructed image and its k-space — directly.

![Buggy pixel grid overlay](figs/eae656a_current-koma/pixel_grid_overlay/stitched_npe64_nfe128_fov0p2_vox1p0mm.png)

**Figure 3. The error is time-driven rather than physical.** Reconstructed image (top
rows) and k-space (bottom) for three TIs on the buggy simulator. The phase-encode axis
(vertical) is acquired shot-by-shot, so reading downward corresponds to advancing in
simulated time: spheres are well-resolved for the first several phase-encode lines,
then smear into horizontal streaks once cumulative simulation time exceeds a threshold.
The streaking is too large and too structured to be truncation ringing; the
reconstruction is accurate up to a fixed simulated time and degrades beyond it.

This reframed the question from which physical effect was missing to what the
simulator does differently after $N$ seconds of simulated time — answerable by
reducing the sequence to its simplest form.

## Bug #1: edge-marker collapse at large absolute time (issue #779, PR #780)

The minimal reproducer (`koma_bug_isolate.jl`) drops the phantom, gradients, and
reconstruction and asks the discretiser one question: *does a 1 ms RF pulse discretise
into the same number of time points regardless of its absolute start time?* It should.

When Koma discretises a sequence it brackets every RF and gradient edge with marker
times nudged by a constant `MIN_RISE_TIME = 1e-14 s`, so rising and falling edges
stay distinct knots; a later `unique!` pass deduplicates the time vector. The flaw:
`MIN_RISE_TIME` is *absolute*, but Float64 spacing is *relative*. Adjacent doubles
near $t$ differ by $\texttt{eps}(t)\approx t\cdot2^{-52}$, so once
$\texttt{eps}(t)/2 > 10^{-14}$ (round-to-nearest, i.e. $t \gtrsim 128\,$s) the nudge
rounds straight back onto $t$. The two markers become bit-identical, `unique!`
removes one, and the pulse's leading edge loses exactly one $\Delta t_\text{rf}$
of integration support.

The mechanism linking a $10^{-14}$ second rounding error to a 39 % $T_1$ error is as
follows. With the default $\Delta t_\text{rf}=5\times10^{-5}\,$s and a 1 ms pulse, the
lost step is **5 % of the $B_1$ area**, i.e. a 5 % flip-angle error (≈ 7.7° on a
90°/180° pair). Because the IR signal scales with $\cos(\alpha_{180})\sin(\alpha_{90})$,
this produces a ≈ 25 % change in $|M_{xy}|$ per pulse (the ≈ 22 % per-shot shift first
reported). The error is $\sim10^{-14}\,$s in time but $\sim25\%$ in signal, and
produces no warning or `NaN`.

Re-running the isolated discretiser makes it quantitative:

```
t0 [s]   nraw  nunique   eps(t0)
0.0       24      23     5.0e-324
3.0       24      23     4.44e-16
67.0      24      23     1.42e-14     <- eps(t0) overtakes MIN_RISE_TIME (1e-14)...
99.0      24      23     1.42e-14     <- ...but markers still distinct (needs eps/2 > 1e-14)
200.0     26      21     2.84e-14     <- unique markers collapse 23 -> 21: discretisation breaks
1000.0    26      21     1.14e-13
```

The collapse appears between $t_0=99\,$s and $200\,$s — the $t\gtrsim128\,$s point
where round-to-nearest first absorbs the nudge — consistent with both the upstream
issue and the streak onset in Fig. 3.

**The fix** (PR #780, in `KomaMRIBase/src/timing/TimeStepCalculation.jl`,
`get_variable_times`) replaces the absolute nudge — used for RF edges, gradient edges,
and the ADC bookends — with one guaranteed to land on a *different* float:

```julia
next_time(t) = max(t + MIN_RISE_TIME, nextfloat(t))
prev_time(t) = min(t - MIN_RISE_TIME, prevfloat(t))
```

`nextfloat`/`prevfloat` step to the adjacent representable double, so every marker
stays at least 1 ULP from its neighbour: the nudge keeps its intended behaviour at
small $t$ and degrades gracefully to a 1-ULP separation at large $t$, where it can no
longer collapse. This was accepted upstream.

## Bug #2: residual knot collapse after time-rebasing (issue #788, PR #789)

Fixing bug #1 pushed the threshold out by roughly $4\times$ but did not eliminate it:
the same root cause remained at a second site. The reproducer (`koma_bug_residual.jl`)
is a gradient-free IR sequence — `180° → TI → 90° → 1-sample ADC → TR pad` — on a
single spin, 24 times at $\text{TR}=15\,$s. Every shot is physically identical, so all
24 signals should match. Instead:

```
shot  1..17  (sim_time <= 255 s):  |signal| = 0.476267   <- stable, all identical
shot 18      (sim_time  = 270 s):  |signal| = 0.722479   <- +52 % jump
shot 19..24  (sim_time >= 285 s):  |signal| = 0.677159   <- new wrong plateau (+42 %)
```

Before calling it numerical I ruled out the physical and implementation alternatives
(full table in `second_bug/00_NOTES.md`):

| Hypothesis | Ruled out by |
|---|---|
| Float32 magnetisation roundoff | `precision="f64"` gives the *same* jump at the *same* shot — deterministic, not a precision walk. |
| Multi-shot coherence pathways | Threshold tracks cumulative `sim_time`, not shot count: 8 shots × TR=10 s and 40 shots × TR=2 s both jump near the same time. |
| Steady-state $M_z$ transient | Steady state is reached by shot 2; shots 2–17 are bit-identical until the threshold. |
| Gradient-specific | Every configuration (none → full IR-SE) jumps; gradients only set *when* (∝ per-shot arithmetic) and the post-jump character (stable plateau vs. chaotic). |

The `f32`/`f64` determinism is the key evidence: the corrupted quantity is the *time
vector* (always Float64), not the magnetisation. The cause is
`_separate_closing_knot!`, which separates a block's closing knot by a fixed $10^{-14}$
in **block-relative** coordinates — safe locally, but `get_samples` and
`get_variable_times` then rebase to absolute time via `T0 .+ t`. Once
$\texttt{eps}(T_0+t_\text{end}) > 10^{-14}$ ($T_0 \gtrsim 270\,$s for ms-scale events)
the rebased gap collapses, the closing knot equals the block-end knot, and `unique!`
deletes it — bug #1's mechanism, one accumulation site downstream.

**The fix** (PR #789) leaves `_separate_closing_knot!` untouched (making it adaptive
broke gradient-linearity tests) and instead re-separates the closing knot *after*
rebasing, on the absolute-time vector, with an adaptive gap:

```julia
_reseparate_closing_knot!(t) =
    t[end-1] = min(t[end-1], t[end] - max(MIN_RISE_TIME, eps(t[end])))
```

The gap widens to ≥ 1 ULP at large absolute time; the function is idempotent (the
rebased vector has already passed through `_separate_closing_knot!` once) and leaves
separated knots untouched. It is applied at every rebasing site — both RF channels in
`get_samples` and the gradient path in `get_variable_times`, the latter also
re-applying an adaptive `_strictly_increasing_knots!`. With it, all 24 shots match and
`KomaMRIBase` passes 544/544, including the bug #1 regression, gradient-addition
linearity, and k-space rotation tests.

> **Dependency note.** The project dev-depends on a fork of KomaMRI pinned to the
> PR #789 commit: bug #1 is merged upstream, but the residual fix is still awaiting
> merge. This pinned fork is the only deviation from the registered Julia package set.

## Result: a validated simulator

With both fixes applied, the same no-noise validation passes: $T_1$ fits on the
diagonal, monoexponential recovery curves, and a well-resolved phantom at every
phase-encode line.

![Fixed pixel grid overlay](figs/7ceced7_fixed/pixel_grid_overlay/stitched_npe64_nfe128_fov0p2_vox1p0mm.png)

**Figure 4. Streaking is eliminated after the fix (cf. Fig. 3).** Same acquisition on
the fixed simulator: all 14 spheres render as well-resolved circles on the expected
grid across every TI, with smooth k-space and no time-dependent contamination.

![Fixed T1 fit](figs/7ceced7_fixed/t1_fit_vs_true/bMANUAL_nonoise/t1_fit_vs_true.png)

**Figure 5. Mean $T_1$ MAPE decreases from 39.4 % to 0.48 % (cf. Fig. 1).** Every
sphere now lies on the diagonal, within $\pm1.3\%$ — the sub-percent accuracy expected
of a noise-free fit, establishing the simulator as a reliable reference for the RL
experiments.

![Fixed recovery curves](figs/7ceced7_fixed/t1_fit_vs_true/bMANUAL_nonoise/recovery_curves_koma.png)

**Figure 6. Recovery points lie on the fitted curves after the fix (cf. Fig. 2).**
Per-sphere $|M_z(\text{TI})|$ against the best-fit monoexponentials, all 14 spheres.

<!-- CUT (full dynamic range, one sentence if wanted): a third commit
d0d5c5f_both-bugs, captured before EITHER fix, has a catastrophic 924% mean MAPE
(one sphere 12,317%), vs 39% with only #780 fixed (eae656a) and 0.5% with both fixed.
The bugs compound: #780 carries the bulk of the collapse, #788 the residual long-seq drift. -->

<!-- CUT (limitations beat, from 00_NOTES.md): even fixed, the safe envelope is
finite — keep TR x Npe within the validated range. The fixes push the threshold far
past anything our experiments use, but bounding cumulative sim-time is itself a
reportable methodology point. -->

---

<!-- CUT (bug-interaction detail): on the worst commit (d0d5c5f, both bugs) spoiling
moved mean MAPE 924% -> 93% — partially masking the bug by dephasing its signature,
which is exactly why imperfect spoiling was such a seductive false lead. The
gradient-amplification mechanism is in 03_minimum_gradient_trigger.jl: per-shot-varying
Gy phase encodes turn the post-threshold bug from a stable wrong plateau into chaotic
per-shot drift — the source of the streaks in Fig 3. -->

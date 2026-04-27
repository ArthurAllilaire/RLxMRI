# MRI Concepts — refresher

Physics-first reference for the concepts used in this repo. Assumes A-level physics; aimed at someone who did a deep dive once and needs a reminder.

---

## The physical foundation

**Nuclear spin and precession.** Hydrogen nuclei (protons) have spin-½, so in a static magnetic field B₀ they precess like gyroscopes at the **Larmor frequency** ω₀ = γB₀, where γ ≈ 267 Mrad/s/T is the proton gyromagnetic ratio. At 3 T this is ~127 MHz (RF, not audio). The code uses `:T3` and `:T15` as tags for 3 T and 1.5 T scanners — different field strengths give different Larmor frequencies and subtly different T1/T2 values for the same tissue.

**Net magnetisation.** At equilibrium, slightly more spins align with B₀ than against it (Boltzmann), creating a net magnetisation **M** pointing along z (the field axis). This is the signal you're measuring — it's tiny but detectable.

**RF excitation.** A short radiofrequency pulse at exactly the Larmor frequency tips **M** away from z by a **flip angle α** (in radians). A 90° pulse tips it fully into the xy-plane. A 180° pulse inverts it to −z. The flip angle is controlled by `α = 2πγB₁t` — longer pulse or stronger B₁ field → bigger tip. B₁ is the RF field amplitude in tesla.

---

## The two relaxation times

Once **M** is tipped, it relaxes back to equilibrium via two independent mechanisms.

### T1 — longitudinal relaxation (spin-lattice)

After excitation, Mz (the z-component) recovers back to M₀ exponentially:

```
Mz(t) = M₀(1 − e^{−t/T1})
```

The physics: spins exchange energy with their surroundings (the "lattice") at a rate set by how well molecular motion couples to the Larmor frequency. Slower, larger molecules → longer T1.

Typical values at 3 T: water ~4 s, brain white matter ~1 s, the phantom spheres range 23 ms–1.84 s (the whole point of the calibration phantom).

**Why it matters**: T1 limits how quickly you can repeat a measurement. You have to wait ~5×T1 for full recovery between shots.

### T2 — transverse relaxation (spin-spin)

After a 90° pulse all spins precess in-phase in the xy-plane. They immediately start dephasing because neighbouring spins slightly perturb each other's local field. The transverse magnetisation decays:

```
Mxy(t) = M₀ · e^{−t/T2}
```

There are two distinct contributions to T2, which add as rates:

```
1/T2 = 1/(2T1) + 1/T2_pure_dephasing
```

- **Pure dephasing** (`1/T2_pure`): random spin-spin interactions scramble phases irreversibly without any spin flipping. This is the classic "spin-spin" mechanism.
- **Spin-flip contribution** (`1/(2T1)`): a T1 event is a spin *flipping* between |↑⟩ and |↓⟩. When a spin that was contributing to Mxy flips, it loses its phase relationship with the others — that is also a decoherence event and reduces Mxy. T1 relaxation therefore directly erodes the transverse signal.

This gives the hard bound **T2 ≤ 2T1** — even in a perfectly homogeneous field with no spin-spin interactions, spin flips alone set a floor: T2 cannot exceed 2T1.

For most biological tissues T2 ≪ T1 (e.g. brain: T1 ≈ 1000 ms, T2 ≈ 80 ms), so pure dephasing dominates and the `1/(2T1)` correction is small. For pure water T1 ≈ T2 ≈ 3–4 s, so the spin-flip contribution is significant. The phantom sphere values in `src/materials/` are independently NIST-measured and already reflect this relationship — the code doesn't need to model it explicitly.

### T2\* — free-decay variant

T2\* < T2 because in reality B₀ is not perfectly uniform. Static field inhomogeneities add an extra, *reversible* dephasing on top of T2. T2\* is what you observe from a free induction decay (FID). A spin-echo refocuses the reversible part, recovering T2. The code stores `T2s` (T2\*) on each `SphereDescriptor` but the E0/E1 experiments focus on T1 and T2.

### PD — proton density (ρ in the code)

The overall signal amplitude scales with how many protons are in the voxel. ρ is a pure density measure, not a relaxation time. It's the `ρ` field on `SphereDescriptor`; `AugmentConfig.PD_sigma_abs` lets you jitter it to simulate density variation.

---

## The timing parameters: TE, TI, TR

| Parameter | Full name | Meaning |
|---|---|---|
| **TR** | Repetition time | Time between successive excitation pulses. Controls how much T1 recovery happens before the next pulse; sets T1 weighting. |
| **TE** | Echo time | Time from excitation to signal readout. Controls how much T2 decay happens before sampling; sets T2 weighting. |
| **TI** | Inversion time | Time between the 180° inversion pulse and the 90° excitation. The key knob for T1 measurement — where on the T1-recovery curve you sample. |

---

## Pulse sequences

A pulse sequence is a recipe of RF pulses and timing that determines what physical quantity you measure.

### IR — Inversion Recovery (T1 measurement)

```
180° pulse → wait TI → 90° pulse → read signal
```

1. A 180° pulse inverts Mz to −M₀.
2. Mz recovers: `Mz(TI) = M₀(1 − 2·e^{−TI/T1})`.
3. The 90° tips whatever Mz has recovered into the xy-plane; you measure its magnitude.
4. The signal is `|M₀(1 − 2·e^{−TI/T1})|` — this has a null (zero crossing) at `TI = T1·ln 2`. The magnitude sign-flip is why `fit_t1_ir` has the "flip the first k points" logic: magnitude-only data destroys the sign, so the fit tries every possible sign pattern and picks the physically consistent one.

The generalised version allows α ≠ 180°:

```
S = |A · (1 − (1 − cos α) · e^{−TI/T1})|
```

- α = 180° → canonical inversion recovery.
- α = 90° → saturation recovery (no null crossing, no sign ambiguity).
- α = 10° → small-tip prep (barely perturbs Mz; low sensitivity but safe for very short T1).

The E1 RL agent picks (TI, α) pairs from a discrete grid — that is its action space.

### SE — Spin Echo (T2 measurement)

```
90° pulse → wait TE/2 → 180° refocus pulse → wait TE/2 → read echo
```

1. 90° tips M into the xy-plane.
2. Spins dephase due to T2 and field inhomogeneities (T2\*).
3. The 180° refocusing pulse — applied as an RF pulse about a **transverse axis** (say x), not z — mirrors every spin's phase in the xy-plane.
4. Each spin continues precessing in the same direction. Fast spins catch up to where slow spins now are; at time TE they all meet — the echo.
5. Only the irreversible T2 decay is unaffected by the refocusing; the reversible T2\* dephasing cancels.
6. Signal: `S = S₀ · e^{−TE/T2}`.

**Why a transverse-axis 180°, not a z-axis 180°?** A 180° pulse about z would just relabel all phases (φ → −φ in azimuth) — that's a mirror, and subsequent precession would re-diverge them identically. The refocusing works because the pulse is about x (or y): the transformation is (x, y, z) → (x, −y, −z), which reverses the *sense* of the accumulated phase without reversing the direction of precession, so the spins then converge rather than diverge.

In the rotating frame concretely: a fast spin that was at +φ from x is now at −φ. Over the next TE/2 it accumulates another +φ and arrives at 0. A slow spin that was at −φ is now at +φ, accumulates −φ, and also arrives at 0. All spins refocus at x simultaneously regardless of their individual field offset.

**Does T1 recovery during TE reduce the signal?** Yes — T1 spin-flip events destroy transverse coherence (see the `1/(2T1)` term in the T2 section above). This is already captured in the T2 you measure. What does *not* happen is Mz recovery "stealing energy" from Mxy in a conserved-energy sense: the energy for Mz recovery comes from the lattice (heat bath), not from Mxy. The two processes drain energy from the spin system independently. The practical rule is: use TR ≫ T1 so Mz is fully recovered before each shot; then vary TE only to fit T2 from the slope of log S vs TE.

Varying TE and measuring the signal, you fit T2 by log-linearising: `log S = log S₀ − TE/T2` — a straight line. That is exactly what `fit_t2_se` does.

### FID — Free Induction Decay

The bare signal immediately after a single RF excitation with no refocusing pulse. Decays as `e^{−t/T2*}` because both T2 and static inhomogeneities dephase the spins. `test_simulation.jl` checks that a KomaMRI FID simulation yields a T2 fit within 10%.

### ADC — Analog-to-Digital Converter (the readout window)

In MRI, "the ADC" refers not to the chip itself but to the **readout window** — the period during which the scanner's receiver coil samples the precessing transverse magnetisation. It is not a pulse; it is a timed gate: "record signal for `dur_adc` seconds, taking `n_adc` evenly-spaced samples."

The signal during the ADC window is the FID (or echo) decaying as `e^{−t/T2}`. In a single-sphere, no-gradient experiment like E0 and E1:
- There is no spatial encoding (no gradients), so all samples within one ADC window are points on the same exponential decay curve.
- The **first sample** (t = 0 of the window) gives the signal amplitude right at the start of readout — that is what E0's fit uses as the "signal at this TI/TE".
- The **middle sample** of an SE window approximates the echo peak.

In a real imaging experiment the ADC window is filled with gradient echoes that encode spatial frequency (k-space), but that is irrelevant here — E0/E1 are purely quantitative single-spin measurements.

### SR — Saturation Recovery

Equivalent to IR with a 90° prep instead of 180°. Mz starts at 0 (saturated), recovers: `Mz(TI) = M₀(1 − e^{−TI/T1})`. No null crossing, no sign ambiguity, faster recovery but less dynamic range. Covered by α = 90° in the generalised IR model.

---

## The signal model used in E1 training

The closed-form `generalized_ir_signal` that the RL training hot path calls (no KomaMRI simulation):

```
S(TI, α) = |A · (1 − (1 − cos α) · e^{−TI/T1})| · e^{−TE/T2}
```

The first factor is the T1-weighted prep (inversion/saturation/small-tip); the second is T2 decay during readout. Because TE is fixed and short, the T2 factor mostly scales amplitude — T1 is what the agent is estimating. Under single-spin + hard-pulse + perfect-spoiling assumptions (valid for calibration phantoms) this is exact, not an approximation. It runs in ~μs vs ~ms for a full KomaMRI simulation.

---

## Code walkthrough — sequences and signal

### `rf_duration` (`src/sequences/blocks.jl:15`)

```julia
rf_duration(α; amp_T = 2e-6) = α / (2π * γ * amp_T)
```

Rearrangement of `α = 2π·γ·B1·t`. Given a target flip angle α (radians) and a B1 amplitude `amp_T` (tesla), it returns the pulse duration in seconds. `amp_T = 2e-6` T = 2 μT is a soft pulse — long duration, gentle. `se_sequence` uses 20 μT instead so the 180° pulse is short enough to fit inside TE/2 without the duration itself eating the available time.

### `ir_sequence` (`src/sequences/blocks.jl:26`)

```julia
seq += RF(amp_T, d180)   # 180° inversion pulse
seq += Delay(TI)          # wait TI — Mz recovers during this
seq += RF(amp_T, d90)    # 90° excitation tips recovered Mz into xy
seq += ADC(n_adc, dur_adc, adc_delay)  # sample the resulting FID
```

This is a literal transcription of the IR pulse diagram. Each `+=` concatenates a block onto the sequence timeline. `Delay(TI)` is pure dead time — no RF, no gradients, just T1 recovery happening. The ADC then samples the FID that results from the 90° excitation. The first ADC sample's magnitude is `|1 − 2·exp(−TI/T1)|` — the IR signal formula.

### `se_sequence` (`src/sequences/blocks.jl:48`)

```julia
seq += RF(amp_T, d90)    # 90° excitation
seq += Delay(pre)         # pre = TE/2 − d90/2 − d180/2
seq += RF(amp_T, d180)   # 180° refocusing pulse
seq += Delay(post)        # post = TE/2 − d180/2 − dur_adc/2
seq += ADC(n_adc, dur_adc, 0.0)
```

The delays are carefully computed so the ADC window is centred on the echo peak at time TE. `pre` accounts for half the 90° pulse duration (the pulse centre is what sets t = 0); `post` accounts for half the 180° duration and half the ADC window. If TE is too short for the pulse durations to fit, the function raises an error rather than silently producing a broken sequence — that is the `pre <= 0 && error(...)` guard.

`n_adc = 33` (odd) so there is exactly one sample at the centre of the window — the echo peak. The middle sample's magnitude is `S₀·exp(−TE/T2)`.

`amp_T = 20e-6` T (20 μT) for SE vs 2 μT for IR — harder pulses → shorter duration → fits inside the shortest TE the sweep covers.

### `generalized_ir_signal` (`src/sequences/blocks.jl:87`)

```julia
Mz_after_prep = cos(α)
Mz_at_excite  = 1 - (1 - Mz_after_prep) * exp(-TI / T1)
amp           = abs(Mz_at_excite)
ts            = range(0, dur_adc; length = n_adc)
Float64[amp * exp(-t / T2) for t in ts]
```

This computes the signal analytically — no KomaMRI simulation.

- Line 1: after a hard prep pulse of flip angle α, the z-magnetisation is `Mz = cos(α)`. At α = π (180°) this gives −1 (full inversion). At α = π/2 (90°) it gives 0 (saturation). At α = 10° it gives ≈ 0.98 (barely touched).
- Line 2: standard T1 recovery formula from that starting point, evaluated at time TI. `1 − (1 − Mz_after_prep)·exp(−TI/T1)` is the general solution to `dMz/dt = (1 − Mz)/T1` with initial condition `Mz(0) = cos(α)`.
- Line 3: magnitude — because the receiver measures |signal|, not signed Mz.
- Lines 4–5: the 90° excitation tips `amp` into the xy-plane, which then decays as `e^{−t/T2}` during the ADC window. Returns a vector of `n_adc` samples.

This function is the E1 training hot path (~μs). The full KomaMRI `simulate` call does the same physics but integrates the Bloch equations numerically over all spins and gradient waveforms (~ms). For single-spin, no-gradient experiments they agree — verified in `test_e1.jl`.

---

## Δw — off-resonance

`Δw` (delta-omega) is the offset in rad/s between a spin's actual precession frequency and the nominal Larmor frequency ω₀. Caused by local field variations (susceptibility, chemical shift). Off-resonance spins accumulate extra phase during TR, causing signal modulation and image artefacts. In the phantom it is small; `AugmentConfig.B0_sigma_Hz` jitters it to simulate field imperfections during RL training.

---

## How the phantom is a calibration tool

The QalibreMD Model 130 is a sphere filled with spheres — 14 T1-reference spheres, 14 T2-reference spheres, 57 fiducials. Each contrast sphere has a known, NIST-traceable T1 or T2 at 1.5 T and 3 T (stored in `src/materials/`). You scan it, fit your sequence's signal to T1/T2, and compare to the table values. If your scanner + sequence recovers the right numbers, you trust it for tissue measurements.

The RL project's goal is to find sequences — adaptive choices of (TI, α) — that recover T1 with fewer measurements or less scan time than the standard fixed protocol.

---

## Summary of what the RL agent is doing physically

It is designing an MRI protocol: choosing which (TI, flip-angle) pair to play next, one "shot" at a time, to maximally inform a T1 estimate while spending as little total scan time as possible. Each step is one IR-prep block; the episode ends when the scan-time budget or block-count limit is hit.

# Meeting 1 — Polished Notes

**Date:** 2026-04-27  
**Attendees:** Arthur Allilaire, Supervisor (Wetscherek)  
**Context:** First progress review after completing E0 (non-RL baseline) and E1 (single-voxel RL).

---

## Progress Summary

### What was demonstrated

| Experiment | Status | Notes |
|---|---|---|
| E0 — Conventional baseline | Done | IR-TSE and multi-TE SE on QalibreMD twin; fits recover T1/T2 to within a few percent |
| E1 — Single-voxel RL | Done | PPO agent learns TI and flip angle for T1 estimation on one randomised voxel |

**QalibreMD twin features confirmed working:**
- Full 3D phantom rebuild matching the Calibur MD-130 physical dimensions
- T1-array, T2-array, and PD-sphere plates with correct material properties
- Fiducial spheres placed at approximate manual positions
- Voxelisation with configurable spatial resolution (voxels/mm³)
- Live connection to KomaMRI via juliacall

### Why E1 converges so fast

The single-voxel problem is too easy — the agent is only fitting one tissue type at a time and the signal model reduces to a scalar exponential with one unknown. It hits near-zero MAPE in very few episodes. The domain randomisation (T1 drawn uniformly within a physiological range per episode) prevents explicit memorisation, but the task itself doesn't require spatial reasoning or the ability to localise signal sources. This is the key motivation for scaling up to E2.

---

## Core Ideas Discussed for Next Steps

### 1. What "full phantom" means in practice

The natural next scale-up is from a single voxel to the **full T1-array plate** — 14 NiCl₂ spheres at different concentrations. Each sphere has a distinct T1 (and a known T2/PD), so the agent must now acquire enough signal to distinguish between spheres and estimate the T1 of each one, rather than just one.

Key differences from E1:
- Signal observed by the agent is the **sum of contributions from all spheres** plus background water (after slice selection, it may be dominated by one 2D plane, but all spins in-plane contribute).
- The agent needs to **build a spatial image**, not just a time-series at a single point.
- Sequence design now has to balance: inversion recovery contrast (to resolve different T1 values) vs. spatial encoding (to separate signal from different spheres).

### 2. The 2D imaging pipeline

The supervisor sketched the simplest end-to-end pipeline that bridges RL to real MR acquisition:

1. **Acquire a full 2D image at several inversion times** — each acquisition is one complete k-space fill with inversion pulses interleaved. The inversion pulse takes most of the time budget; choosing which TI values to acquire at is the core decision.
2. **Reconstruct an image for each TI** — KomaMRI can produce k-space raw data; Fourier-transform gives a 2D magnitude image.
3. **Pixel-wise T1 fit** — fit a two- or three-parameter exponential to the per-pixel TI-signal curve to get a T1 map.

The RL agent's job: **choose which TI (and flip angle) to use for each acquisition block**, given the signal it has seen so far, to maximise T1 map accuracy within a scan-time budget.

An important optimisation: you do **not** need to fill the full k-space for every TI. Interleaving k-space lines across different TIs — acquiring some lines at TI=300 ms, others at TI=1000 ms, etc. — can in principle recover the T1 map from less total data. KomaMRI has reconstruction scripts that handle this. This is the compressed-sensing/MRF regime; start with the simple "one full image per TI" approach and consider this as a stretch goal.

### 3. Adding gradients — why and how

E1 used no gradient encoding: the simulator returned a single complex number (FID from a single spin). Gradients are needed to:

- **Slice selection** (Gz): excite only a 2D slab of the phantom. This is critical for spatial resolution — without it, signal from all slices superimposes. KomaMRI's `Sequence` object supports slice-selective RF + gradient; we add a sinc-shaped RF pulse paired with a Gz ramp.
- **Frequency encoding** (Gx readout gradient): spatially encodes the echo along one axis. Required for 2D image formation.
- **Phase encoding** (Gy): steps through k-space lines to fill a 2D Fourier space. This is where scan time goes — each phase encode step is one TR.

In the action space, the agent may control:
- Which **TI** to use before the echo (inversion-recovery contrast)
- Which **slice** to excite (slice centre position, thickness) — relevant once localisation is introduced
- Flip angle α

The frequency/phase encoding gradients are largely fixed by the target FOV and matrix size; the agent doesn't need to choose them directly (though in E4 the radial spoke angle becomes an action).

**Important simulator note from supervisor:** when you apply a slice-selective pulse, you must still simulate the **entire phantom**, including spins outside the slice — they experience the off-resonant RF and contribute to the steady-state magnetisation, which affects subsequent echoes. KomaMRI handles this correctly as long as the phantom includes those spins. Do not subset the phantom spatially before calling `simulate`.

### 4. Localisation

The supervisor highlighted this as a genuinely interesting sub-problem: the phantom's **exact position and orientation inside the scanner is unknown** before acquisition begins. In clinical practice, a fast localiser scan runs first to find the patient/phantom, then the diagnostic sequence is planned on that.

**The 6-DoF localisation problem:**
- 3 translational degrees of freedom: (x₀, y₀, z₀) — phantom centre relative to scanner isocentre
- 3 rotational degrees of freedom: (θ_x, θ_y, θ_z) — phantom orientation

**Tissue parameters per sphere:** T1, T2, PD — bringing the total unknowns to ~6 + 3×14 for a full T1 plate.

The agent can localise by acquiring **low-resolution localiser images** (large voxels, thick slice, fast TR) at several orthogonal orientations, then fitting sphere centroids. This is a realistic 10-second localiser. An RL agent that does localisation + parameter mapping in sequence — or jointly — would be a novel contribution.

A simplified version for E2: use `AugmentConfig` rotation and translation randomisation (already in the codebase) but have the agent observe which slice gave the strongest signal (a proxy for phantom centre) before committing to diagnostic TI choices.

### 5. Noise

Adding complex Gaussian noise to the simulator output is essential for realistic training and mandatory for evaluation. The model: after `KomaMRI.simulate()` returns the complex signal vector `s`, add:

```
s_noisy = s + σ * (randn(n) + im * randn(n))
```

Same `σ` for both channels; `σ` is a training hyperparameter. More noise = harder problem. The supervisor noted that noise may not be necessary for initial convergence training (it slows learning) but is essential to evaluate whether the policy is actually robust. Recommend: train with moderate noise (`σ` ~ 0.05× mean signal magnitude), evaluate across a sweep of `σ` values.

### 6. Model-based reconstruction as an alternative to pixel-wise fitting

The supervisor also described a **gradient-based reconstruction** approach that bypasses explicit pixel-wise fitting:

1. Initialise a guess for the T1/T2/PD map.
2. For each acquired k-space line, **forward-simulate the signal** expected under that guess using KomaMRI (or an analytical approximation).
3. Compute the residual between simulated and acquired k-space, **backpropagate through the signal model** to update the map guess.

This is slower than a direct fit but works from **far fewer k-space lines** — potentially very few acquisitions if the forward model is accurate. It is conceptually similar to compressed-sensing MRI with a physics-informed prior. This is not the focus of E2 but motivates why having differentiable signal models (or at least differentiable approximations via MRzero's PDG framework) would be valuable later.

---

## Clarifications and Corrections

- **PDG vs EPG**: the interim report incorrectly called MRzero an EPG simulator. It uses Phase Distribution Graphs (PDG). Update the write-up before the final report.
- **"Dwell at k-space coordinates"**: this phrase in the interim is vague. In the context of E2/E3 it likely means: repeated acquisition at the same k-space location to average down noise, or intentionally over-sampling low-frequency k-space. Clarify before using it in writing.
- **Julia↔Python boundary**: the agent is an independent program; no need to "port" it to Julia. The Gym interface is the sole contract.

---

## Action Points

- [ ] Scale to full T1-array plate: 14-sphere phantom, coarse voxels (3–4 mm) for training
- [ ] Add live `KomaMRI.simulate()` call per RL step (no caching/assumptions)
- [ ] Add slice-selection and readout gradients to the sequence blocks
- [ ] Implement 2D image reconstruction (FFT of k-space) and pixel-wise T1 fit as the observation/reward pathway
- [ ] Add complex Gaussian noise to simulator output
- [ ] Enable domain randomisation: rotation (SO(3)), translation (~N(0, 5 mm)), T1/T2 jitter (±5%)
- [ ] Sketch a localisation sub-task: randomise phantom pose and have the agent run a brief localiser before the diagnostic sequence
- [ ] Benchmark per-step wallclock at 3–4 mm voxel size; confirm < 100 ms before scaling

# E2 — Full-Phantom Multi-Sphere T1 Mapping with Gradients and Localisation

**Week target:** 2026-04-28 – 2026-05-04  
**Status:** In progress  
**Depends on:** E0 (baseline, done), E1 (single-voxel RL, done)

---

## Goal

Extend the RL loop from a single voxel to the full T1-array plate of the QalibreMD phantom. The agent must now:

1. Use live KomaMRI simulations (with gradient encoding) at every step — no analytical signal models or cached assumptions.
2. Acquire 2D images at chosen inversion times, reconstruct them, and fit a pixel-wise T1 map as the reward signal.
3. Operate under domain randomisation including phantom pose uncertainty — seeding the localisation sub-problem.

Success: the agent's T1-map MAPE across all 14 spheres, measured on held-out phantom configs, is within 5% and better than a fixed-TI grid baseline of the same total scan time.

---

## Key Differences from E1

| Dimension | E1 | E2 |
|---|---|---|
| Phantom | Single voxel (one spin) | Full T1-array plate, 14 spheres, 3–4 mm voxels |
| Gradients | None | Slice-select (Gz) + readout (Gx) + phase encode (Gy) |
| Simulator call | One `simulate()` per episode | One `simulate()` per RL step (live, per block) |
| Observation | Scalar complex FID | 2D reconstructed image or k-space signal vector |
| Reward | Single-sphere T1 error | Mean MAPE across all 14 spheres |
| Noise | Not yet added | Gaussian noise on real + imaginary channels |
| Pose | Fixed | Randomised: rotation (SO(3)) + translation ~N(0, 5 mm) |
| Localisation | N/A | Optional sub-task (see §6) |

---

## 1. MDP Formulation

### State (internal)
A voxelised `Phantom` built from `PhantomConfig(include_plates = [:T1])` at 3–4 mm resolution, with material properties jittered per episode:
- T1 drawn from log-uniform distribution spanning the plate range
- T2, PD scaled by same relative jitter factor
- Pose drawn from `AugmentConfig` (rotation + translation)

The agent never directly observes the phantom state.

### Action
Continuous 5-parameter vector defining the next acquisition block:

| Parameter | Range | Meaning |
|---|---|---|
| TI | 10–3000 ms | Inversion recovery time |
| TE | 5–80 ms | Echo time |
| TR | 500–5000 ms | Repetition time |
| α (flip angle) | 5–180° | Excitation flip angle |
| slice_z | −60 to +60 mm | Slice centre position along z |

The readout bandwidth, FOV, and matrix size are fixed (32×32 at 3–4 mm resolution during training; 128×128 at 1 mm for evaluation).

### Observation
After each simulated acquisition block, the agent receives:

1. **Image vector**: magnitude of the 2D reconstructed slice (FFT of k-space, flattened, normalised) — or a downsampled version (e.g. 16×16 tokens) to keep observation size tractable.
2. **Running T1 estimate vector** (length 14): current per-sphere T1 estimate from a pixel-wise exponential fit on accumulated images. Updated after each block.
3. **Scan budget state**: `(cumulative_time_used, blocks_remaining, max_budget)`.

Concatenate all three into a flat observation vector. Normalise by the first-episode baseline.

### Transition
```
signal_kspace = KomaMRI.simulate(phantom_episode, seq_block, scanner)
image = fft2d(reshape(signal_kspace, Nfe, Npe))
obs = [vec(abs.(image)) / norm_factor; running_t1_estimate; time_state]
```

The phantom is **cached for the episode** (no re-voxelisation per step); only `simulate` is called per step on the existing spin ensemble.

**Critical:** include all phantom spins in the `simulate` call — spins outside the excited slice still affect steady-state magnetisation and should not be stripped out.

### Episode termination
- Cumulative `block_time ≥ T_budget` (e.g. 120 s simulated), or
- Fixed maximum number of blocks (e.g. 20 blocks), or
- Agent emits a terminal "submit map" action.

### Reward
Dense per-step + terminal:

```
r_t  = −mean_i( |T̂1_i − T1_i| / T1_i )     # negative MAPE across 14 spheres
r_T  = +B   if  MAPE < 0.05 at termination    # terminal bonus
```

Episode truncates (no reward) if any block exceeds SAR limits. No soft time penalty — scan time is controlled by the hard budget truncation.

---

## 2. Sequence Block Structure

Each action maps to an IR-SE block in KomaMRI. The block has:

```
[180° inversion pulse (slice-selective, slab z ± thickness/2)]
[TI delay]
[90° excitation pulse (slice-selective, same slice)]
[TE/2 delay]
[180° refocusing pulse]
[TE/2 delay + ADC: Nfe samples at readout gradient Gx]
[TR − TI − TE delay]
```

Phase encoding: for a full 2D image, the block must be played `Npe` times with different Gy amplitudes. For fast RL steps, use a short phase-encode table (Npe = 8–16 for training; 128 for evaluation). Optionally, encode multiple phase steps per block to reduce total block count.

The slice-select gradient uses a sinc RF pulse; thickness is fixed at 5 mm during training (thin enough to isolate one plate, thick enough for signal). The `slice_z` action shifts the slice centre.

---

## 3. Image Reconstruction and T1 Fitting

At each step, after `simulate()` returns k-space data:

```julia
# Julia side, inside the Gym step function
image = abs.(ifftshift(ifft2(fftshift(reshape(kspace, Nfe, Npe)))))
```

To get per-sphere T1 estimates, segment the image by sphere ROIs (using known sphere positions after applying the episode's pose transform) and fit:

```
S(TI) = S0 * |1 − 2 * exp(−TI / T1)|
```

using a simple Levenberg–Marquardt fit on the accumulated `(TI, mean_signal_in_ROI)` pairs so far. Update the running estimate vector after each block.

For the first 1–2 blocks (insufficient TIs for a fit), return a zero-initialised estimate vector.

---

## 4. Noise Model

Apply complex Gaussian noise after `simulate()`:

```julia
function add_noise(signal::Vector{ComplexF32}, σ::Float32)
    return signal .+ σ .* (randn(Float32, length(signal)) .+ im .* randn(Float32, length(signal)))
end
```

`σ` is set relative to the noiseless signal RMS:
- Training: `σ = 0.05 × rms(signal)` (moderate)
- Evaluation sweep: `σ ∈ {0.0, 0.02, 0.05, 0.10, 0.20}`

Independent per sample, same `σ` for real and imaginary channels.

---

## 5. Domain Randomisation

Every episode draws fresh phantom parameters using `AugmentConfig`:

```julia
aug = AugmentConfig(
    rotation = true,             # uniform over SO(3)
    translation_mm = 5.0,        # ~N(0, 5mm) per axis
    T1_sigma_rel = 0.05,         # ±5% T1 jitter per sphere
    T2_sigma_rel = 0.05,
    B0_sigma_Hz = 10.0,          # off-resonance jitter
)
phantom_episode = augment(build_phantom(cfg), aug, rng)
```

This means the agent cannot memorise sphere positions or T1 values — it must estimate them from the signal it observes. This addresses the supervisor's key concern from the interim review.

---

## 6. Localisation Sub-Task (this week: simplified version)

Full 6-DoF pose estimation is a stretch goal, but we seed it this week with:

1. **Pose randomisation**: the episode phantom is translated up to ±20 mm from isocentre and rotated up to 15° around each axis.
2. **Localiser observation**: the first step of every episode is a fixed "scout" block — a fast, low-resolution 3-plane acquisition (three orthogonal thick slices at TR=300 ms, no inversion). This returns three low-res images that reveal the rough phantom centre. The agent observes these as the initial observation.
3. **Agent task**: use the scout images to infer where to centre the diagnostic slice (via the `slice_z` action) before spending scan budget on high-quality TI acquisitions.

This is not joint pose estimation (the agent is not asked to output a pose estimate) but forces the policy to use spatial information to make good slice-placement decisions.

Full localisation (output (x₀, y₀, z₀, θ_x, θ_y, θ_z) as part of the terminal action) is an E5 goal.

---

## 7. Fidelity vs Wallclock

Training configuration (must achieve < 100 ms per `simulate()` call):

| Setting | Training | Evaluation |
|---|---|---|
| Voxel resolution | 3–4 mm | 1 mm |
| Matrix size (Nfe × Npe) | 16 × 8 | 128 × 128 |
| Plates included | T1 only (14 spheres) | T1 only |
| Background water | Excluded | Included |
| Fiducials | Excluded | Included |
| Noise | Moderate (σ = 0.05) | Sweep |

**Benchmarking task (day 1):** before writing the RL loop, measure `simulate()` wallclock for:
- 1 block, 3–4 mm voxels, T1 plate only, 16×8 matrix → target < 100 ms
- If > 100 ms: reduce matrix to 8×4 or drop voxel count further until satisfied

---

## 8. Algorithm

PPO with a small MLP policy (same as E1, widened slightly):
- Actor: 3-layer MLP, 256 units, tanh activations
- Critic: same architecture, separate weights
- Observation normalisation: running mean/std via `VecNormalize`
- Action: continuous, tanh-squashed Gaussian, clipped to parameter bounds

Hyperparameters (starting point):
- `n_steps = 512` per rollout
- `batch_size = 64`
- `learning_rate = 3e-4`
- `gamma = 0.99`
- `ent_coef = 0.01` (encourage exploration of TI choices)
- `n_envs = 4` (4 parallel Julia subinterpreters, 4 × 8 GB RAM budget — measure before committing)

If memory from parallel Julia runtimes is prohibitive, fall back to 1 env with longer rollouts.

---

## 9. Evaluation Protocol

Held-out test set: 50 phantom configs sampled from `AugmentConfig` with fixed seeds (never seen during training).

Primary metrics:
1. **T1 MAPE across all 14 spheres** (target: < 5%)
2. **Relative scan time** at matched MAPE vs the E0 fixed-TI grid baseline
3. **Robustness to noise**: MAPE vs σ curve

Monitoring-only (not reward):
- SNR of acquired signal
- Distribution of chosen TI values (histogram) — should not be uniform; agent should favour informationally rich TIs for the current running estimate
- Wallclock per step

---

## 10. Weekly Task Breakdown

### Day 1 (Monday)
- [ ] Benchmark `simulate()` wallclock at training config; confirm < 100 ms
- [ ] Implement gradient-encoded sequence block (IR-SE with Gz, Gx, Gy) in the Julia block library
- [ ] Verify that FFT of the returned k-space produces a recognisable phantom image (visual sanity check)

### Day 2 (Tuesday)
- [ ] Implement `add_noise()` function and unit test against expected SNR
- [ ] Implement per-sphere ROI segmentation on reconstructed images (using augmented sphere positions)
- [ ] Implement running T1 estimator (Levenberg–Marquardt on accumulated TI-signal pairs)

### Day 3 (Wednesday)
- [ ] Update `QalibreMDPhantomGym.jl` to support multi-sphere episodes (full T1 plate, coarse voxels)
- [ ] Update observation and action spaces for E2
- [ ] Implement the scout/localiser first step and integrate its output into the initial observation

### Day 4 (Thursday)
- [ ] Run random-policy episode rollouts; verify reward signal is non-degenerate and observations are finite
- [ ] Begin PPO training run; log to wandb; monitor per-step wallclock
- [ ] If training is unstable: check reward scale, observation normalisation, gradient clipping

### Day 5 (Friday)
- [ ] Evaluate trained agent on held-out configs; compute MAPE across 14 spheres
- [ ] Compare against fixed-TI grid (E0 baseline at same scan time budget)
- [ ] Plot TI-choice histogram for trained agent — does it show a non-trivial adaptive policy?
- [ ] Write up results as a Jupyter notebook with key plots

---

## 11. Relation to PLAN.md

This plan executes the **E2** rung of the experiment ladder defined in `PLAN.md §4`, with the following additions from the M1 supervisor meeting:

| PLAN.md E2 spec | Status in this plan |
|---|---|
| Full T1 plate, 14 spheres | Included |
| Continuous IR-SE action space | Included |
| Live `simulate()` per step | **Added (was not explicit in PLAN.md)** |
| Gradient encoding (Gz, Gx, Gy) | **Added** |
| 2D image reconstruction | **Added** |
| Pixel-wise T1 fit as observation | **Added** |
| Domain randomisation (AugmentConfig) | Included |
| Gaussian noise on real/imag | **Added** |
| Localisation sub-task | **Added (simplified version)** |

Items deferred to later experiments:
- Model-based backprop reconstruction (discussed in M1, deferred to E3/E4)
- K-space line interleaving across TIs (MRF-style, E3)
- Full 6-DoF pose estimation as an explicit output (E5)
- Radial k-space spoke angle in action space (E4)

---

## 12. Open Questions

- **Memory footprint of 4 parallel Julia runtimes**: measure peak RAM before committing to `n_envs = 4`. If > 16 GB total, use `n_envs = 1` with async collection.
- **ROI segmentation robustness**: the per-sphere T1 fit assumes we can correctly segment sphere pixels in the noisy, low-resolution training image. Consider using a fixed-radius circle mask centred on the expected (augmented) sphere position — simpler and more robust than adaptive thresholding at 3–4 mm resolution.
- **Scout block cost**: the fixed localiser step uses a small fraction of scan budget (< 5 s simulated). Verify it doesn't dominate the episode; if so, replace it with a free prior observation (inject augmented pose parameters directly into the observation as a noisy estimate).
- **Convergence vs scan-time tradeoff tuning**: the hard budget truncation makes the time/accuracy Pareto curve implicit. After initial convergence, sweep `T_budget` to produce the Pareto curve plot for the write-up.

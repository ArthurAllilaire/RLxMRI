# E2 Code Explainer

A thorough walkthrough of every component added for the E2 experiment. Read
this alongside `E2_PLAN.md` (the design document) and the source files
themselves.

---

## Table of contents

1. [What E2 adds vs E1](#1-what-e2-adds-vs-e1)
2. [Sequence building: `ir_se_2d_sequence`](#2-sequence-building-ir_se_2d_sequence)
3. [k-space physics and the sign of the prewinder](#3-k-space-physics-and-the-sign-of-the-prewinder)
4. [Image reconstruction: `abs.(ifft(ksp))`](#4-image-reconstruction-absifft ksp)
5. [Sphere → pixel coordinate mapping](#5-sphere--pixel-coordinate-mapping)
6. [E2Env struct overview](#6-e2env-struct-overview)
7. [Episode initialisation: `_e2_build_episode_phantom`](#7-episode-initialisation-_e2_build_episode_phantom)
8. [Per-step simulation: `_e2_simulate_step`](#8-per-step-simulation-_e2_simulate_step)
9. [T1 fitting per sphere: `_e2_update_t1_estimates!`](#9-t1-fitting-per-sphere-_e2_update_t1_estimates)
10. [Reward and episode termination](#10-reward-and-episode-termination)
11. [Observation vector layout](#11-observation-vector-layout)
12. [Action space and Python-side rescaling](#12-action-space-and-python-side-rescaling)
13. [Domain randomisation](#13-domain-randomisation)
14. [Python wrapper: `QalibreMDE2Env`](#14-python-wrapper-qalibremd2env)
15. [Training script: `train_e2.py`](#15-training-script-train_e2py)
16. [Evaluation script: `eval_e2.py`](#16-evaluation-script-eval_e2py)
17. [Known limitations and next steps](#17-known-limitations-and-next-steps)

---

## 1. What E2 adds vs E1

| Dimension | E1 | E2 |
|---|---|---|
| Phantom | Single voxel (1 spin) | Full T1 plate (14 spheres, 3–4 mm voxels) |
| Gradients | None | Gx (readout), Gy (phase-encode), no Gz |
| Simulator call | Analytical model or single FID | Full `KomaMRI.simulate()` every step |
| Observation | Scalar FID magnitude | 16×8 magnitude image + T1 estimates |
| Reward | Single-sphere T1 error | Mean MAPE across 14 spheres |
| Noise | None | Complex Gaussian (σ = 5 % of signal RMS) |
| Pose | Fixed | Random rotation (σ ≈ 8.6°) + translation (σ = 5 mm) |
| Action | Discrete (18 options) | Continuous 5-D box |

The central challenge E2 introduces is that the agent must allocate N steps (each choosing a TI) to efficiently cover the full T1 range [23 ms, 1838 ms] of the 14 spheres—while observing a coarse image rather than a scalar FID.

---

## 2. Sequence building: `ir_se_2d_sequence`

**File:** `src/sequences/blocks.jl`  
**Signature:**
```julia
ir_se_2d_sequence(TI, TE, TR;
    α_exc = π/2, FOV = 0.2, Nfe = 16, Npe = 8, amp_T = 20e-6)
    → Sequence
```

### What it builds

For each of the `Npe` phase-encode (PE) steps, the function appends **9 sub-blocks** to a single KomaMRI `Sequence`:

```
For k = 1 .. Npe:
  1. 180° inversion RF (non-selective)
  2. TI delay
  3. Excitation pulse (α_exc)
  4. Gx prewinder + Gy_k phase encode     ← applied after excitation
  5. TE/2 delay to the 180° refocus
  6. 180° refocus RF
  7. TE/2 delay to the echo
  8. ADC readout (Nfe samples) + Gx readout + Gy_k rewind
  9. TR recovery delay
```

After `KomaMRI.simulate(phantom, seq, Scanner())`, the result has exactly `Npe` profiles (one per ADC block). Profile `k` contains the `Nfe` complex k-space samples for PE step `k`.

### Gradient amplitudes

Using γ = 2π × 42.577 MHz/T (proton gyromagnetic ratio):

```
kmax_x = Nfe / (2 × FOV)           # e.g. 16/(2×0.2) = 40 m⁻¹
Gx_ro  = 2·kmax_x / (γ·dur_adc)   # ≈ 3 × 10⁻⁴ T/m  (well within Gmax = 60 mT/m)
Gx_pre = kmax_x  / (γ·dur_pe)     # = Gx_ro (since dur_pe = dur_adc/2)

Δky = 1/FOV = 5 m⁻¹
ky[k] = (k − (Npe+1)/2)·Δky       # ranges from −3.5·Δky to +3.5·Δky
Gy[k] = ky[k] / (γ·dur_pe)        # ≈ ±1.3 × 10⁻⁴ T/m
```

### Timing arithmetic

The delays are computed once (not per PE step) since TI, TE, TR are fixed for the whole block:

```
ti_d  = max(TI  − d_inv/2 − d_exc/2,  0)   # from inversion centre to excitation centre
te1_d = max(TE/2 − d_exc/2 − dur_pe − d_ref/2, 0)  # prewinder end → refocus centre
te2_d = max(TE/2 − d_ref/2 − dur_adc/2,      0)    # refocus centre → echo centre
tr_d  = max(TR  − shot_time,                  0)    # recovery
```

All RF durations are ≈ 0.3–0.6 ms at `amp_T = 20e-6 T`, so the delays are dominated by TI and TE.

---

## 3. k-space physics and the sign of the prewinder

This is the trickiest part of the sequence. **The prewinder for a spin echo must be positive (same sign as the readout).**

Here is why. Define the effective k-space position as the phase accumulated by a spin at position x:

```
φ(t) = ∫₀ᵗ γ Gx(τ) dτ · x
```

The 180° refocus pulse **negates all accumulated phase**: effectively kx_eff → −kx_current at the moment of the refocus.

| Event | kx before refocus | After refocus |
|---|---|---|
| Prewinder (positive, Gx_pre > 0) | +kmax_x | **−kmax_x** |
| Readout (positive, Gx_ro > 0) sweeps from −kmax_x to +kmax_x | echo at midpoint ✓ |

With a **negative** prewinder: kx before refocus = −kmax_x → after refocus = +kmax_x → readout goes from +kmax_x further positive → echo never crosses zero. **Wrong.**

The prewinder must be placed **after excitation** (not before). Gradients applied before the 90° pulse dephase spins that are still along z and do not affect the transverse k-space trajectory.

---

## 4. Image reconstruction: `abs.(ifft(ksp))`

```julia
ksp = zeros(ComplexF32, Npe, Nfe)
for k in 1:Npe
    ksp[k, :] = raw.profiles[k].data[:, 1]
end
image_mag = abs.(ifft(ksp, (1, 2)))
```

**Why no fftshift?**  
The k-space is acquired from kx = −kmax_x to +kmax_x (sample 1 at −kmax_x). By the Fourier shift theorem, this is equivalent to the standard k-space (starting at 0) multiplied by a linear phase ramp exp(i·kmax_x·x). Taking the magnitude eliminates this phase ramp:

```
|ifft(ksp)| = |image_true · exp(i·kmax_x·x)| = |image_true|
```

So `abs.(ifft(ksp))` gives the correct magnitude image. The pixel x-coordinate is:

```
x(i_fe) = (i_fe − 1) · FOV/Nfe     (0-indexed from 0 to FOV·(1−1/Nfe))
```

Negative-x spheres appear at pixels near index `Nfe` due to the periodic (wrap-around) nature of the DFT.

---

## 5. Sphere → pixel coordinate mapping

The T1-plate spheres have known base positions. After applying the episode rotation/translation:

```julia
R = rotation_matrix(rx, ry, rz)       # 3×3 rotation matrix
cx, cy = (R * collect(centre) + t_m)[1:2]

i_fe = mod(round(Int, cx * Nfe / FOV), Nfe) + 1   # 1-based
i_pe = mod(round(Int, cy * Npe / FOV), Npe) + 1
```

**The `mod` handles the DFT's periodic boundary**: a sphere at x = −65 mm with FOV = 200 mm wraps to pixel `round(−65/12.5) mod 16 + 1 = 12`. 

**Do all 14 spheres map to distinct pixels?**  
Yes, for the default config (Nfe=16, Npe=8, FOV=0.2 m, T1-plate arrangement). With domain randomisation (±5 mm translation, ±8.6° rotation), a sphere can shift by ≤ 2 pixels. The mapping is recomputed each episode from the episode's rotation/translation to stay correct.

---

## 6. E2Env struct overview

Defined in `src/rl/e2.jl`.

**Static fields** (set at construction, never mutated):
- `cfg_field`, `voxel_size_mm`, `FOV`, `Nfe`, `Npe` — simulator/image config
- `max_blocks`, `time_budget_s` — episode length limits
- `terminal_bonus`, `success_tol` — reward shaping
- `noise_sigma_rel` — noise level
- `T1_sigma_rel`, `translation_sigma_mm`, `rotation_sigma_rad` — domain randomisation
- `sphere_centres_base` — 14 nominal sphere centres from `contrast_plate_centres(PLATE_Z_MM.T1)`
- `T1_base`, `T2_ratio` — nominal tissue values

**Episode state** (mutated by `e2_reset!` / `e2_step!`):
- `phantom` — cached KomaMRI Phantom (not rebuilt per step)
- `T1_true[14]` — per-sphere ground truth for this episode
- `sphere_px[14]` — `(i_pe, i_fe)` pixel location per sphere
- `block_TIs[14]`, `block_mags[14]` — accumulated measurements per sphere
- `T1_est[14]` — running T1 estimates
- `last_image_mag` — flattened magnitude image from the last step

---

## 7. Episode initialisation: `_e2_build_episode_phantom`

Called from `e2_reset!`. Steps:

1. **T1 jitter**: sample `T1_ep[i] = T1_base[i] · exp(σ_T1 · N(0,1))` per sphere.
2. **T2 jitter**: preserve the T2/T1 ratio: `T2_ep[i] = T1_ep[i] · T2_ratio[i]`.
3. **custom_sphere_map**: override each sphere's descriptor with the jittered values so `build_phantom` uses them.
4. **Rotation/translation**: sample `rx, ry, rz ~ N(0, σ_rot)` and `tx, ty, tz ~ N(0, σ_trans)`.
5. **Build phantom**: `build_phantom(PhantomConfig(...))` — the rotation/translation is baked into spin positions.
6. **Transformed centres**: apply the same R, t to each `sphere_centres_base[i]` to get `sphere_px[i]`.

Why not use `AugmentConfig.T1_sigma_rel` for the jitter? Because that applies *per-spin* (Gaussian on each voxel), making it impossible to know the exact per-sphere `T1_true`. The custom_sphere_map approach gives uniform T1 per sphere and an exact `T1_true[i]` for the reward.

---

## 8. Per-step simulation: `_e2_simulate_step`

```julia
function _e2_simulate_step(env, TI, TE, TR, α_exc_deg)
    seq = ir_se_2d_sequence(TI, TE, TR; α_exc=deg2rad(α_exc_deg), ...)
    raw = simulate(env.phantom, seq, Scanner())
    
    # Assemble k-space
    ksp = zeros(ComplexF32, Npe, Nfe)
    for k in 1:Npe
        ksp[k, :] = raw.profiles[k].data[:, 1]
    end
    
    # Complex Gaussian noise
    σ = noise_sigma_rel · rms(ksp)
    ksp .+= σ · (randn(Npe,Nfe) + im·randn(Npe,Nfe))
    
    # Magnitude image
    image_mag = abs.(ifft(ksp, (1,2)))
    return image_mag, ksp
end
```

**Phantom caching**: the phantom is built once per episode in `e2_reset!`. `simulate()` is called with the same phantom object every step (no re-voxelisation).

**IMPORTANT from CLAUDE.md**: the phantom includes ALL spins (all 14 spheres). Do not subset by z before calling `simulate()` — spins outside the excited region still affect the steady-state magnetisation.

**Wallclock**: ≈ 36 ms per step after JIT warmup (3 mm voxels, Nfe=16, Npe=8). This is within the 100 ms budget from E2_PLAN.md §7.

---

## 9. T1 fitting per sphere: `_e2_update_t1_estimates!`

After `_e2_simulate_step` returns `image_mag`:

```julia
for i in 1:14
    (i_pe, i_fe) = sphere_px[i]
    mag_i = image_mag[i_pe, i_fe]
    push!(block_TIs[i], TI)
    push!(block_mags[i], mag_i)
    
    if length(block_TIs[i]) >= 2
        T1_est[i] = fit_t1_generalized_ir(block_TIs[i], fill(π, n), block_mags[i]).T1
    else
        T1_est[i] = sqrt(T1_range[1] * T1_range[2])   # neutral prior
    end
end
```

**Why α = π (radians)?**  
`fit_t1_generalized_ir` takes the *prep pulse* angle (the inversion pulse angle), not the excitation flip angle. Since the inversion is always 180° = π, we pass `fill(π, n)`. The excitation flip angle α_exc affects the signal amplitude (via `sin(α_exc)`) but not the T1 dependence — the fitting function absorbs this into the amplitude parameter A.

**Signal model used by the fitter:**
```
|S(TI)| = A · |1 − 2·exp(−TI/T1)|
```
The fitter scans a log-spaced T1 grid [T1_lo, T1_hi] and for each candidate T1 finds the optimal A analytically. Requires ≥ 2 TI samples.

**After 1 block**: T1_est[i] = geometric mean of the search range ≈ 0.17 s (neutral prior). Reward is 0 for the first block (see §10).

---

## 10. Reward and episode termination

```julia
mape = n_blocks >= 2 ? mean(|T1_est[i] − T1_true[i]| / T1_true[i]) : 0.0
reward = −mape

if done && mape < success_tol
    reward += terminal_bonus
end
```

**Dense reward**: the agent gets −MAPE after every block (except the first). This provides a gradient signal at every step, unlike E1 where the terminal bonus dominated.

**Why zero reward for block 1?** With only 1 TI, we can't fit T1. Using the neutral prior gives MAPE ≈ 100% which would discourage all action choices equally — not useful.

**Episode ends when** `n_blocks ≥ max_blocks` OR `time_used_s ≥ time_budget_s`.

**Scan time per block**: `block_time = Npe × TR`. With Npe=8 and TR=2 s, one block costs 16 s of simulated scan time. With `time_budget_s=120 s` and `max_blocks=15`, the agent gets up to 15 steps.

---

## 11. Observation vector layout

Total length: `Nfe × Npe + n_spheres + 3 = 128 + 14 + 3 = 145`

| Slice | Content | Notes |
|---|---|---|
| `[0:128]` | Magnitude image / image_max | Normalised to [0, 1] |
| `[128:142]` | log10(T1_est[i]) for i=1..14 | Clipped to [log10(1e-4), log10(10)] |
| `[142]` | time_used / time_budget | ∈ [0, 1] |
| `[143]` | n_blocks / max_blocks | ∈ [0, 1] |
| `[144]` | 1.0 (constant bias) | |

**Image normalisation**: divided by the max pixel value within the episode step. If the image is all zeros (very short TI → complete nulling), the image component is all zeros.

**T1 estimate encoding**: `log10(clamp(T1_est_i, 1e-4, 10.0))`. NaN estimates (first block) → 0.0.

---

## 12. Action space and Python-side rescaling

The Julia E2Env accepts a **physical** 5-element action vector:

| Index | Parameter | Range |
|---|---|---|
| 0 | TI | [0.01, 3.0] s |
| 1 | TE | [0.005, 0.08] s |
| 2 | TR | [0.5, 5.0] s |
| 3 | α_exc | [5, 180] ° |
| 4 | slice_z | [−60, 60] mm (stored, not yet used in sequence) |

The Python `QalibreMDE2Env` exposes a **normalised** Box(5) action space ∈ [−1, 1] to SB3. On `step()`:

```python
phys = ACT_LO + (action + 1) / 2 * (ACT_HI - ACT_LO)
```

SB3's PPO policy outputs tanh-squashed values. The normalisation avoids scale mismatches between TI (0–3 s) and TE (0–80 ms).

**Two clamps applied inside `e2_step!`**:
```julia
TI = min(TI, TR * 0.90)   # ensure ≥10% TR for recovery
TE = min(TE, TR * 0.30)   # prevent TE from exceeding TR
```
These handle edge cases when the policy hasn't learned the physical constraint.

---

## 13. Domain randomisation

Each episode independently samples:

1. **T1 jitter** (per sphere, log-normal): `T1_ep[i] = T1_base[i] · exp(σ_T1 · N(0,1))`  
   Default σ_T1 = 0.05 → ±5% variation. Prevents memorising the 14 nominal T1 values.

2. **Rotation** (Euler XYZ angles): `rx, ry, rz ~ N(0, σ_rot)` with σ_rot ≈ 0.15 rad (≈ 8.6°).  
   Applied via `PhantomConfig.rotation`, which calls `apply_transform!` inside `build_phantom`.

3. **Translation**: `tx, ty, tz ~ N(0, σ_trans)` with σ_trans = 5 mm.  
   Applied via `PhantomConfig.translation_mm`.

4. **B0 inhomogeneity**: `AugmentConfig(B0_sigma_Hz = 5.0)` — per-spin off-resonance noise from `apply_per_spin_noise!`.

The agent observes **only the image** — it cannot see the rotation/translation directly. It must learn to use the spatial structure of the image to figure out where spheres are.

---

## 14. Python wrapper: `QalibreMDE2Env`

**File:** `python/qalibremd_gym/env_e2.py`

Key design decisions:
- Re-uses `_ensure_julia()` from `env.py` (the E1 wrapper), so both envs share the same Julia boot code and one Julia runtime per process.
- `_denorm_action()` maps [-1, 1] → physical ranges (linear rescale).
- `reset()` and `step()` proxy to `qmd.e2_reset_b` / `qmd.e2_step_b` (the `_b` suffix aliases bypass Julia's `!`-ending restriction via juliacall).
- Info dict: complex Julia values (arrays of T1_true, T1_est) are converted to numpy arrays.

**VecNormalize**: the training script wraps the env in `VecNormalize` to normalise observations and returns. This is important because the observation contains a mix of scales (image values ∈ [0,1] and log T1 estimates ∈ [−4, 1]).

---

## 15. Training script: `train_e2.py`

```
python python/train_e2.py \
    --timesteps 200000 \
    --out runs/e2/ppo \
    --field T3 \
    --max-blocks 15 \
    --noise 0.05
```

PPO hyperparameters (from E2_PLAN.md §8):
- `n_steps=512` — rollout length per update
- `batch_size=64`
- `learning_rate=3e-4`
- `ent_coef=0.01` — exploration bonus
- `net_arch=[256, 256]` — 3-layer MLP (input + 2 hidden)

The `E2EvalCallback` runs `n_eval_episodes` episodes every `eval_interval` steps, logs MAPE and saves `eval_history.json`.

**Memory note**: each Julia process that boots KomaMRI uses ~2–4 GB RAM. With `n_envs=1` (the safe default), memory usage is ~3 GB total.

---

## 16. Evaluation script: `eval_e2.py`

```
python python/eval_e2.py \
    --policy runs/e2/ppo/policy.zip \
    --vecnorm runs/e2/ppo/vecnorm.pkl \
    --episodes 50
```

Reports:
- Per-sphere MAPE table (14 spheres)
- TI-choice histogram — a non-uniform distribution means the agent is adaptive
- Comparison against a fixed 7-point log-spaced TI grid baseline (same scan budget)
- Optional noise robustness sweep (`--noise-sweep`)

The `evaluate_fixed_grid` function cycles through TIs [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0] s using TE=20 ms, TR=4 s. This is the equivalent of the E0 conventional approach adapted to the new imaging framework.

---

## 17. Known limitations and next steps

### Current limitations

1. **No slice selection (Gz)**: the sequence is non-selective — all spins in the phantom contribute. The `slice_z` action is recorded but not used in the sequence. This means the agent can't focus on specific z-planes.

2. **Coarse resolution (16×8)**: with 12.5 mm × 25 mm pixels, sphere overlap is minimal (all 14 spheres map to distinct pixels for default config), but domain randomisation can occasionally cause two spheres to share a pixel.

3. **No Gy_rewind direction check**: the Gy_k gradient in the prewinder and the −Gy_k in the rewind use `ky_steps[k] / (γ·dur_pe)`. For half-integer centred steps (k − (Npe+1)/2), this is never zero, which is correct.

4. **T1 fitting assumes 180° inversion**: the per-sphere fitting uses `α = π` (inversion angle). If in a future version the agent controls the inversion flip angle (instead of fixing it at 180°), the fitting code must pass the actual inversion angle.

5. **`slice_z` not wired up**: to implement slice selection, add a slice-selective Gz sinc pulse around the inversion and excitation RF blocks, and shift the excitation frequency by `γ · Gz · slice_z` to excite the desired z-position.

### Suggested next steps

- **Benchmark first step in isolation**: run `_e2_simulate_step` alone with `@btime` to confirm the ~36 ms timing persists across episodes (memory grows slightly as the Julia JIT caches more code paths).
- **Sanity-check image**: visualise `env.last_image_mag` (reshape to [Npe, Nfe]) at several TIs to verify spheres are visible and move with rotation/translation.
- **Increase max_blocks to 15** and run a few random-policy rollouts to confirm the reward signal is non-degenerate (varies per episode, improves with more informative TI choices).
- **Start PPO training** with `--timesteps 50000` first to check convergence speed.
- **Add Gz slice selection** once the 2D imaging is working well — this is the localisation sub-task from E2_PLAN.md §6.

---

## Quick reference: running E2

```bash
source .venv/bin/activate

# Check one random-policy episode
python - <<'EOF'
import sys, os
sys.path.insert(0, 'python')
os.environ['PYTHON_JULIAPKG_OFFLINE'] = 'yes'
from qalibremd_gym.env_e2 import QalibreMDE2Env
env = QalibreMDE2Env(rng_seed=42, max_blocks=3)
obs, info = env.reset(seed=1)
for _ in range(3):
    a = env.action_space.sample()
    obs, r, done, trunc, info = env.step(a)
    print(f"reward={r:.3f}  mape={info.get('mape',0)*100:.1f}%  done={done}")
EOF

# Train
PYTHON_JULIAPKG_OFFLINE=yes python python/train_e2.py \
    --timesteps 200000 --out runs/e2/ppo

# Evaluate
PYTHON_JULIAPKG_OFFLINE=yes python python/eval_e2.py \
    --policy runs/e2/ppo/policy.zip \
    --vecnorm runs/e2/ppo/vecnorm.pkl \
    --episodes 50 --noise-sweep
```

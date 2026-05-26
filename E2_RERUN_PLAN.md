# E2 Rerun Plan — adaptive 14-sphere T1 mapping on the fixed simulator

**Status:** planning · **Owner:** Arthur · **Date:** 2026-05-24
**Supersedes:** `archive/docs_pre_simfix/E2_TRACTABILITY_PLAN.md` and all V5/V9–V12 results.
**Depends on:** `FIX_SIM_PLAN.md` (the bug fix this plan re-baselines against).

---

## 0. Why this document exists

Two simulator bugs (see `FIX_SIM_PLAN.md`) corrupted the per-sphere measurement
function in every E2 run to date:

1. **Missing `fftshift`/`ifftshift`** around the 2D IFFT → chequerboard + half-FOV
   wrap → ROI samples were a mix of the target sphere, the diagonally-opposite
   sphere, and background.
2. **Gradient amplitude built with γ_rad instead of γ_Hz** (`src/sequences/blocks.jl`)
   → gradients 2π× too small → collapsed image FOV.

Both are now fixed. **Consequence: every prior number is invalid** — V5/V9–V12
MAPE (220–1267%), the CR-opt baselines, the SSE-landscape "wrong-basin" story, the
Rician/phase-sensitivity diagnoses, and the entire `CH4_DRAFT.md`. Those docs are
archived under `archive/docs_pre_simfix/`. The phase-sensitivity and fitter-σ
machinery they motivated are red herrings of the bugs and are **not** carried
forward.

This plan re-baselines the fixed simulator and launches a fresh, well-controlled
RL run with an expanded, physically-meaningful action space.

---

## 1. What already exists (reuse, don't rebuild)

| Piece | Location | State |
|---|---|---|
| E2 env (Julia) | `src/rl/e2.jl` | Working; FFT-shift + γ_Hz fixes landed. |
| E2 env (Python) | `python/qalibremd_gym/env_e2.py` | Working; 5-dim or 3-dim (simplified) action. |
| Sequence builders | `src/sequences/blocks.jl` | `ir_se_2d_sequence` (+`SpoilerConfig`), `gre_2d_sequence`. |
| T1 fitter | `src/fitting/fits.jl` | F1+ transient closed form. |
| CR-opt solver | `cr_optimize_sweep` (src/baselines) | Working; used by `baseline_e2.py`. |
| Fixed-grid + CR baselines | `python/baseline_e2.py` | Working; `--cr-optimal`, `--subset-size`. |
| Trainer / eval / diagnostics | `python/{train,eval,diagnose}_e2.py` | Working; all accept `--time-budget`, `--subset-size`. |
| SNR sweep | `scripts/snr_sweep.jl` + `.py` | Working; σ-input, reports NEMA dual-acq SNR. |

We are re-running and lightly extending, not rewriting.

---

## 2. Experimental target

**Fleet:** full **14-sphere** T1 plate (T3 field), per-episode T1 jitter + pose
randomisation (already in env). This matches the "RL agent with 14 spheres" ask.

**Claim we want to be able to make:** on the fixed simulator at a realistic noise
level, the RL policy's 14-sphere T1 MAPE beats the Cramér–Rao-optimal *fixed*
schedule of the same scan-time budget, and is demonstrably observation-conditional
(adaptive). Whether it holds is now an open question again — the fixed sim may make
the 14-sphere problem cleanly solvable, which would re-open a real win/adaptivity
story that the buggy runs could never show.

**Secondary fleet (adaptivity stress, optional):** 5-of-14 random subset per
episode (the old tractability framing). With 14 fixed spheres, the CR-opt fixed
schedule is already near-optimal because every episode looks the same (archived
`CH4_DRAFT.md` §4.6); the 5-of-14 subset forces the optimum to depend on the
episode, which is where adaptivity has the most headroom. Run this only if the
full-14 result is ambiguous.

---

## 3. Step 0 — re-baseline the fixed simulator (BEFORE any RL)

Nothing downstream is meaningful until these re-run on the fixed sim.

### 3.1 Re-run the SNR sweep → choose the noise level

```bash
julia --project=. scripts/snr_sweep.jl --voxel-mm 1.0 --npe 32 --nfe 64 --budget 160
python scripts/snr_sweep.py            # report-ready σ vs (NEMA dual, MAPE) curves
```

Read off the σ at which **NEMA MS-1 dual-acquisition SNR ≈ 25** (the clinical
anchor; `snr_dual_peak` column). Per your decision (**realism first**), that σ is
the training noise level — we report whatever CR-opt and RL MAPE result, rather
than tuning σ to manufacture RL headroom.

**Recorded (2026-05-25, 25 seeds, 64×32, 1 mm voxel, 1 mm axial slab through the
T1 plate, no water, field :T15):**

| quantity | value |
|---|---|
| **σ_abs for NEMA_dual ≈ 25** | **50** (snr_dual_peak = 24.9) |
| NEMA single-image SNR at that σ | 25.2 |
| snr_ksp at that σ | 3.40 (phantom-corrected 10.6) |
| CR-opt 14-sphere MAPE at that σ | **mean 10.3 %, median 3.7 %, max 63.4 %** |

Collapse cliff: mean MAPE explodes (catastrophic short-T1 rail-pins) past σ ≈ 85
(NEMA_dual ≈ 17); median stays ~5–6 % through there. Use **median** as the headline.

**T3-vs-T15 comparison (2026-05-25, `--field T3`, same 64×32 / 1 mm / 15-seed grid):**
the σ↔SNR map and CR-opt MAPE are **field-independent to within seed noise** — the
σ* = 50 number above is authoritative for both fields. At σ = 50:

| field | snr_dual | MAPE median | MAPE mean | MAPE max |
|---|---|---|---|---|
| :T15 | 24.9 | 3.72 % | 10.3 % | 63.4 % |
| :T3  | 24.1 | 3.86 % | 10.1 % | 62.8 % |

(Makes sense: σ is absolute/hardware-set and the fleet T1s differ only modestly
between fields, so per-sphere SNR — and therefore the fit — barely move.) Results in
`scripts/runs/snr_sweep_voxel_1mm_T3/`. The mean column is noisier than T15 at a few
σ (rare short-T1 rail-pins, e.g. σ=70/100 spike to ~200 %); median is smooth and
matched — another reason to headline median.

**CRITICAL — geometry must match between sweep and training, or σ*=50 is invalid:**

- **Slicing/voxel:** the sweep slices a 1 mm axial slab and uses 1 mm voxels. The E2
  env was non-slice-selective full-3D at a 3 mm voxel default — now made to match:
  `voxelise_sphere`/`build_sphere` gained a `z_range` arg, and
  `_e2_build_episode_phantom` voxelises **only the 1 mm slab** (centred on the T1
  plate) so out-of-slab voxels are never materialised (low memory). Pose is now
  **in-plane only** (rz + tx,ty; rx=ry=tz=0) so the constructed slab is exactly the
  imaged slab and in-slice signal is constant every episode. Default voxel → 1 mm
  (`e2.jl` + `env_e2.py`). Verified: env episode phantom = 4816 spins = sweep; e2 /
  e2_imaging / geometry / builder tests pass.
- **Resolution:** ~~the env default is 16×8~~ **Resolved (2026-05-25):** env default
  bumped to **Nfe=64, Npe=32** in both `src/rl/e2.jl` and
  `python/qalibremd_gym/env_e2.py`, so training/baseline now match the sweep by
  default (no per-call override needed). 16×8 collided ROIs (§6.1).
- **Field:** ~~this sweep used :T15; plan §2 targets :T3~~ **Resolved (2026-05-25):**
  standardised on **:T15** (matches the 1.5 T scanner default, `t1_fit_vs_true`, and
  the existing sweep). Env default flipped to `:T15` in `e2.jl` + `env_e2.py`, and
  `scripts/snr_sweep.jl` gained a `--field {T15,T3}` flag (default :T15; T3 runs land
  in a `_T3`-suffixed outdir). The T3-vs-T15 comparison sweep is recorded below.
  `python/baseline_e2.py` reads its 14-sphere fleet field-parametrically from Julia's
  `T1_ARRAY` (`nominal_fleet_t1s(field)`), so `--field T15` yields the matching T15
  fleet for the yardstick.

**Water:** not in the env (training has none). A water sweep was run for realism but
water is a large long-T1 volume that partially cancels the spheres at low spatial
frequency and leaks into the coarse-grid ROIs (noise-free median MAPE 1.9 %→4.3 %);
`--clean-recon` (hamming+pad+3×3 ROI) did **not** fix it (median worsened to ~11 %).
Keep water out unless a custom eroded-interior ROI is added later.

**T1_1 caveat:** sphere T1_1 (T1_true = 1.879 s) fits to ~0.68 s even at σ=0 (64 %
MAPE, noiseless) — the CR-opt 4-block schedule (max TI 0.64 s, TR 2.9 s) can't
constrain a 1.88 s T1. This single sphere sets `max` MAPE across all low σ; report
median + per-decade, not mean.

> Old memory note (SNR target 2.5 -> NEMA≈30) is from the buggy/unsliced sim — do not
> trust it. The corrected σ↔NEMA map (1 mm slice, 1 mm voxel) gives **σ* ≈ 50**.

### 3.2 Re-run the fixed-schedule + CR-optimal baselines at that σ

```bash
PYTHON_JULIAPKG_OFFLINE=yes python python/baseline_e2.py \
  --episodes 50 --seed 500000 \
  --out runs/e2/rerun_baselines \
  --max-blocks 30 --time-budget 160.0 \
  --noise <σ_from_3.1> \
  --sigma-method asymptotic \
  --cr-optimal --cr-block-grid 6 10 14 18 --cr-starts 1000 --cr-refine 10
```

This produces the **new yardstick**: the CR-opt fixed-schedule MAPE the RL agent
must beat. `log_grid` / `clinical_irse` are the conventional-protocol references
for the Ch4 quantified-benefit table.

---

## 4. Noise — decision and rationale

- **Model:** absolute complex-Gaussian σ on k-space (hardware-determined,
  scene-independent) — already what `add_noise!` does. Keep.
- **Level:** σ fixed so NEMA dual-acq SNR ≈ 25 at the reference block (§3.1).
  Realism-first; MAPE falls where it falls.
- **Eval robustness sweep:** at eval time, sweep σ for NEMA_dual ∈ {40, 25, 15, 8}
  to produce the MAPE-vs-SNR robustness curve (monitoring, not training).

Rationale: a realistic, defensible noise level is worth more in the report than a
cherry-picked one, and the supervisor (Andreas) cares about realism. If the
realism-first σ leaves no RL headroom over CR-opt, that is itself a clean,
honest result (and we then fall back to the 5-of-14 fleet for the adaptivity story).

---

## 5. Action space — tradeoffs and the phased plan

You're undecided here, so this section lays out the full tradeoff. **Recommended:
phased** (validate each DOF before adding the next).

### 5.1 The candidate degrees of freedom

| DOF | Physics it buys | Cost / risk |
|---|---|---|
| **TI** (have it) | The core T1 contrast knob. | — |
| **TR** (have it) | Recovery vs scan-time tradeoff; partial-recovery efficiency. | — |
| **TE** (have it, simplified) | T2 weighting; mostly wants to be small. Low value as a *learned* dim. | Wastes a dim. |
| **α (flip angle)** | **Ernst angle** `cos α* = exp(−TR/T1)` maximises SNR per unit time. The cheapest physically-meaningful DOF; Andreas (M2) flagged Ernst-angle emergence as worth looking for. Was removed during reward-collapse debugging. | One extra continuous dim; well-behaved. |
| **Spoiler / crusher toggle ("spoiling"/"cleaning")** | At short TR the prior shot's residual Mxy contaminates the next excitation. Crushers around the 180° + a TR spoiler clean it, at the cost of sequence time. The F1+ fitter *assumes* perfect spoiling, so if the agent doesn't spoil at short TR the data violates the model → fit degrades. Learning **when spoiling is worth the time** is a genuinely novel adaptive-design result. | Discrete-ish dim; interacts with the forward-model assumption (interesting, but a confound to control). |
| **FOV** | Bigger FOV → bigger voxels → more spins/voxel → more signal (SNR↑) at the cost of spatial resolution / risk of sphere-ROI aliasing. A real SNR-vs-resolution lever. | Changes the recon grid mapping; ROI pixel formula must track FOV per step; more plumbing. |
| **Sequence type (SE vs GRE)** | GRE (`gre_2d_sequence`) is T2*-weighted, faster, no 180°; SE removes T2*. A "pick your physics" choice. | Two forward models; doubles validation surface; defer. |

### 5.2 Why phased beats "full menu at once"

- **Attribution.** If we dump α + spoiler + FOV + seq-type into one 7-dim action
  and MAPE moves, we can't say *which* DOF did it. Phased runs give a clean
  ablation ladder for Ch4 ("adding α bought X%; adding spoiler bought Y%").
- **Trainability.** PPO on a small MLP handles 3–4 continuous dims comfortably;
  6–7 mixed continuous+discrete dims with sparse-ish reward is where training gets
  fragile (we saw clip_fraction blow-ups historically).
- **The spoiler dim changes the physics the fitter assumes.** Introducing it
  alongside everything else muddies whether a regression is "bad policy" or
  "fitter-model violation". Isolate it.

### 5.3 The phased plan

- **Run A — `[TI, TR, α]`** (TE fixed at 20 ms). 3-dim, same dimensionality as the
  current simplified action, so PPO hyperparameters transfer directly. α mapped to
  **[5°, 90°]** (the Ernst window for our T1 range and learned TRs sits ~30–70°,
  well inside this). This is the primary scientific run: does freeing the flip
  angle let RL find Ernst-optimal, SNR-efficient schedules that beat CR-opt?
  - *Note:* CR-opt's Jacobian currently assumes α=90°. To keep the comparison
    apples-to-apples, either (a) compare Run A against the α=90° CR-opt and let RL's
    α-freedom be its *advantage*, or (b) extend the CR solver to also optimise α.
    Recommend (a) for the headline (RL's extra DOF is the point), and report (b) as
    a "does CR-opt-with-α close the gap?" control if time permits.

- **Run B — `[TI, TR, α, spoil]`** where `spoil` is a continuous action thresholded
  to {off, on}. Adds the spoiling/cleaning decision on top of Run A. The story:
  the agent learns to spend time on crushers only when its chosen TR is short
  enough that contamination would otherwise wreck the fit.

- **(Deferred) FOV and SE/GRE** — only if A and B land cleanly and there's report
  room. FOV is the more interesting of the two (SNR-vs-resolution Pareto).

### 5.4 If you'd rather not phase

The mid-scope `[TI, TR, α, FOV, spoil]` (5-dim) is defensible if you want a single
"design the whole acquisition" headline and accept weaker per-DOF attribution.
Skip SE/GRE either way for now. Flag this and I'll adjust the code plan.

---

## 6. Observation & resolution — tradeoffs and recommendation

You guessed "keep image + coarse grid"; here's the full tradeoff so we decide
deliberately.

### 6.1 Imaging resolution (Nfe × Npe)

| Option | Pro | Con |
|---|---|---|
| **16 × 8 (current)** | Fastest sim; smallest obs. | **14 spheres over a 0.2 m FOV at 8 PE rows ≈ 25 mm/row — sphere ROIs collide/alias.** Post-fix, this likely caps achievable MAPE regardless of the policy. |
| **32 × 16** | ROIs mostly separable; ~modest slowdown. | Still tight for the closest spheres. |
| **64 × 32 (Andreas's ask, FIX_SIM_PLAN §3)** | Clean ROI separation; matches supervisor request; honest imaging. | ~4× slower per block; large flat obs if image is kept (2048 px). |

**Recommendation:** **64 × 32.** With 14 spheres the ROI-separability problem is
real and binds the result; sim cost is acceptable (per-block ~tens of ms). This is
independent of whether the image stays in the observation.

### 6.2 Does the image belong in the observation?

The ROI extraction + T1 fit happen **inside** the env; the policy is handed the
running per-sphere `T1_est`. The raw flattened image is therefore largely redundant
to the policy's decision.

| Option | Pro | Con |
|---|---|---|
| **Keep image (flat MLP)** | Status quo; no plumbing; lets the policy in principle see SNR/structure the fit summary drops. | At 64×32 the obs is ~2048-dim of mostly-uninformative pixels → slower, noisier value learning, VecNormalize on 2048 dims. |
| **Keep image + small CNN extractor** | Image used properly (spatial features). | New `features_extractor_class` plumbing; more compute; probably overkill for what the policy needs. |
| **Drop image; obs = [T1_est(14), budget(3)]** | Tiny obs; fast, stable PPO; the policy gets exactly the decision-relevant summary. | Loses any raw-image signal (but that signal is already distilled into T1_est/σ). |

**Recommendation:** I'd lean **drop the image** (obs = running T1_est + budget),
because it's the cleanest and the image is redundant given in-env fitting — but
your "keep image + coarse grid" instinct is reasonable if you want to preserve the
option of the policy reacting to raw SNR. **Tentative default for the plan: keep
the image but at 32×16** (compromise: ROIs more separable than 16×8, obs not huge),
and treat "drop image + go 64×32" as the fast-follow if training is slow or noisy.
Easy to flip — it's a few lines in `e2_obs_dim` / `_e2_observation`.

### 6.3 Drop the σ observation channel

The per-sphere fitter-σ channel (`2*n_spheres` → the `sig_obs` half) is part of the
fitter-uncertainty machinery you flagged as a red herring. Recommend **removing it
from the observation** (obs T1-part becomes `n_spheres`, not `2*n_spheres`) and
using `sigma_method=:asymptotic` (or none) internally. Simpler obs, no
misspecified-σ confound.

---

## 7. Reward, fitter, randomisation, eval

- **Reward:** keep the configuration that worked structurally — `delta_mape` with a
  small absolute-MAPE anchor and **`terminal_bonus = 0.0`** (the terminal bonus is
  the documented E1/E2.0 degenerate-policy driver). `mape_alpha = 1.0` (plain mean)
  to start; consider `0.5` (mean+max mix) only if the policy ignores one decade.
- **Fitter:** F1+ transient model (keep). `sigma_method = :asymptotic`. Drop
  profile-likelihood/bootstrap from the hot path (red herrings).
- **Domain randomisation (already in env):** per-sphere T1 log-jitter (5%), pose
  rotation (~8.6° σ) + translation (5 mm σ), B0 per-spin (5 Hz). Keep — this is the
  "agent can't memorise the phantom" guard Andreas asked for.
- **Eval protocol:** 50 held-out seeds (500000+i), paired across RL and all
  baselines (same phantom realisations). Metrics: mean/median/p90 MAPE (report
  median too — it's more honest than mean when a short-T1 sphere rail-pins),
  per-decade (long/mid/short) MAPE, scan-time-matched comparison, MAPE-vs-SNR curve.
- **Adaptivity diagnostics (`diagnose_e2.py`):** TI-vs-running-T1_est correlation
  and (for the 5-of-14 secondary) the subset-bucket TI-distribution KS test. Add an
  **α-vs-TR-vs-T1 plot** to check for Ernst-angle behaviour (the new Run A signal).

---

## 8. Concrete code changes (file by file)

### 8.1 Run A — learn the flip angle `[TI, TR, α]`

**`python/qalibremd_gym/env_e2.py`**
- Add a 3-dim action mode that maps `[TI, TR, α]` (TE fixed 20 ms). Cleanest: add
  `learn_alpha: bool = False`; when `simplified_action and learn_alpha`, action is
  3-dim and `_denorm_action` fills `full[0]=TI(u0)`, `full[2]=TR(u1)`,
  `full[3]=α∈[5,90](u2)`, `full[1]=0.020`, `full[4]=0`. (Keeps the existing
  `[TI,TE,TR]` simplified mode intact for back-compat.)
- α upper bound: use **90°** (not 180°) in this mode so the Ernst window is the
  middle of the range and the policy can't waste mass on inverting excitations.

**`python/train_e2.py`, `eval_e2.py`, `baseline_e2.py`, `diagnose_e2.py`**
- Add `--learn-alpha` flag, thread into `env_kwargs`.

**`src/rl/e2.jl`** — no change needed: `e2_step!` already reads `α_exc_deg =
action_vec[4]` and passes it through to the sequence + fitter. (Confirm the fixed
schedules in `baseline_e2.py` still send α=90° so the CR-opt comparison is the
α-fixed anchor — they do.)

### 8.2 Run B — spoiler/crusher action `[TI, TR, α, spoil]`

**`src/rl/e2.jl`**
- `_e2_simulate_step`: accept `spoil::Bool=false`; build
  `SpoilerConfig(enabled=spoil, amp_T=30e-3, dur=5e-3, axis=:z)` and pass
  `spoiler=` to `ir_se_2d_sequence`.
- `e2_step!`: read an extra action element (e.g. `action_vec[6]`) as the spoil
  flag (`> 0.5`), pass to `_e2_simulate_step`. Account for the added crusher time in
  `block_time` (the sequence builder already absorbs `d_crush` into the shot; just
  make sure `block_time = env.Npe * TR` still reflects the realised shot time, or
  compute shot time from the sequence).

**`python/qalibremd_gym/env_e2.py`**
- Add `spoiler_action: bool = False`; when on, append a 4th normalised dim,
  threshold `u>0.5 → spoil=True`, and pass through (extend the physical action
  vector sent to `e2_step_b` with the spoil flag).

**`python/train_e2.py` etc.** — add `--spoiler-action` flag.

**`src/sequences/blocks.jl`** — no change (spoiler path already implemented).

### 8.3 Observation / resolution

**`src/rl/e2.jl`**
- Defaults: bump `Nfe`, `Npe` to the chosen resolution (32×16 tentative; 64×32 if
  we drop the image). Mirror in `env_e2.py`.
- **Drop the σ channel:** change `e2_obs_dim` to `Nfe*Npe + n_spheres + 3` and
  remove `sig_obs` from `_e2_observation`. (If we also drop the image:
  `e2_obs_dim = n_spheres + 3` and `_e2_observation` returns `[t1_obs; bgt]`.)

**`python/qalibremd_gym/env_e2.py`** — mirror the resolution defaults; obs_dim is
read from Julia so no other change.

### 8.4 Re-baseline config

**`python/baseline_e2.py`** — no code change; just run §3.2 with the new
`--noise`, `--max-blocks 30`, `--time-budget`, resolution defaults.

### 8.5 Tests (don't skip — these are why the bugs survived)

`FIX_SIM_PLAN.md` §5 specifies the imaging-pipeline tests (T1–T10, P1–P2). Confirm
they're in `test/test_e2_imaging.jl` / `python/tests/`; if not, add at least:
- ROI peak lands at the correct centred pixel for an off-centre sphere (catches any
  recon regression).
- α-scaling recovery test (Run A relies on the `sin α` correction).
- 3-sphere noiseless fit MAPE < 5% (end-to-end gate).

---

## 9. Run sequencing & commands

σ* = **50** (§3.1), field **:T15**, 64×32, run-once CR baselines, budget-guarded
env (realised scan time ≤ budget). Observation defaults to `[T1_est, budget]`
(image + σ channels off — §6.2/6.3); add `--include-image`/`--include-sigma` only
to ablate. `--noise` now defaults to 50, but the commands pass it explicitly.

```bash
# Step 0 — re-baseline (must precede RL). σ* already chosen (§3.1); re-run the
# sweep only if geometry changes.
julia --project=. scripts/snr_sweep.jl --voxel-mm 1.0 --npe 32 --nfe 64 --budget 160
python scripts/snr_sweep.py                       # pick σ at NEMA_dual≈25  → σ*=50
PYTHON_JULIAPKG_OFFLINE=yes python python/baseline_e2.py \
   --episodes 50 --seed 500000 --out runs/e2/baselines \
   --field T15 --max-blocks 30 --time-budget 160.0 --noise 50 \
   --sigma-method asymptotic \
   --cr-optimal --cr-block-grid 6 10 14 18 --cr-starts 1000 --cr-refine 10

# Run A0 — WITHOUT α (clean 2-dim [TI, TR] ablation; TE fixed at 20 ms)
PYTHON_JULIAPKG_OFFLINE=yes python python/train_e2.py \
   --fix-te --field T15 \
   --reward-mode delta_mape --terminal-bonus 0.0 --mape-alpha 1.0 \
   --sigma-method asymptotic --noise 50 --time-budget 160 --max-blocks 30 \
   --timesteps 300000 --out runs/e2/rerun_A0_noalpha

# Run A — WITH α (3-dim [TI, TR, α]; the primary scientific run)
PYTHON_JULIAPKG_OFFLINE=yes python python/train_e2.py \
   --fix-te --learn-alpha --field T15 \
   --reward-mode delta_mape --terminal-bonus 0.0 --mape-alpha 1.0 \
   --sigma-method asymptotic --noise 50 --time-budget 160 --max-blocks 30 \
   --timesteps 300000 --out runs/e2/rerun_A_alpha

# Run B — add spoiler (after A is understood)
PYTHON_JULIAPKG_OFFLINE=yes python python/train_e2.py \
   --fix-te --learn-alpha --spoiler-action --field T15 \
   --reward-mode delta_mape --terminal-bonus 0.0 --mape-alpha 1.0 \
   --sigma-method asymptotic --noise 50 --time-budget 160 --max-blocks 30 \
   --timesteps 300000 --out runs/e2/rerun_B_spoiler
```

A0 and A are independent processes, so run them concurrently to use separate
CPU cores (the env can't spread one run's rollouts across cores — juliacall
keeps Julia in-process). Eval each run with `eval_e2.py` (paired seeds 500000+,
matching `--fix-te`/`--learn-alpha`) against `runs/e2/baselines`, and
`diagnose_e2.py` for adaptivity + Ernst-angle plots. The headline is **A vs A0
MAPE** (the α gain) against the CR-opt anchor.

---

## 10. Interesting findings from the investigations (what survives the bugs)

Things worth carrying into the report / keeping in mind, separated from the
bug-artifact noise:

1. **The bugs themselves are the headline methods story.** "We built imaging
   end-to-end, got plausible-looking training curves, and *still* had a corrupt
   measurement function because no test asserted the recon was phantom-aligned" is a
   genuine, instructive Ch4/Ch5 narrative — and the new imaging tests are the fix.
3. **Ernst angle is unexplored headroom.** Fixing α=90° during debugging removed the
   single most physically-meaningful DOF. Run A directly tests whether RL discovers
   `cos α* = exp(−TR/T1)`-like behaviour — a clean, citable emergent result if it does.
4. **The T1 fleet is heavily short-skewed** (4/14 spheres < 50 ms; T1 ∈ [23 ms, 1.84 s]).
   The short-T1 tail dominates mean MAPE; report **median + per-decade** MAPE, not
   just mean, or one rail-pinning sphere distorts the headline.
5. **Spoiling has a real time cost and the fitter assumes it's done** — so "when is
   it worth spoiling?" is a legitimate adaptive-design question, not just a knob.
6. **CR-optimal is the right anchor, and on a fixed-fleet problem a good fixed
   schedule is hard to beat** (archived `CH4_DRAFT.md` §4.6). If the full-14 result is
   a tie, that's an honest finding — and the 5-of-14 fleet is where adaptivity can
   actually pay, so keep it ready.
7. **NEMA dual-acquisition SNR is the clinical metric**; `snr_ksp` understates
   in-tissue SNR (Parseval over mostly-empty background) and single-image NEMA reads
   low on coarse grids (Gibbs). Use `snr_dual_peak`.

---

## 11. Risks

| Risk | Mitigation |
|---|---|
| Realism-first σ leaves no RL headroom over CR-opt on the 14-sphere fleet | Pre-committed: report honestly; pivot adaptivity claim to the 5-of-14 secondary fleet. |
| α-freedom makes the CR-opt comparison unfair (RL has a DOF CR-opt lacks) | Headline: that DOF *is* the point. Control: extend CR solver to optimise α and report whether it closes the gap. |
| Spoiler dim destabilises training (mixed continuous+discrete) | Phased — Run B only after Run A is understood; threshold a continuous dim rather than a true discrete action. |
| 64×32 + image-in-obs slows training / 2048-dim VecNormalize is noisy | Drop image from obs (obs = T1_est + budget); or keep image at 32×16. Decision deferred to §6, easy to flip. |
| Coarse grid + 14 spheres → ROI collision caps MAPE | Bump resolution (§6.1); the imaging tests (§8.5) assert ROI separation. |
| KomaMRI multi-shot drift past ~60 s/block | Constrain TR×Npe per block < 60 s; finding #2. |

---

## 12. Open decisions for Arthur

1. **Action space:** confirm phased `[TI,TR,α]` → `[TI,TR,α,spoil]` (§5.3), or pick
   the mid-scope 5-dim (§5.4).
2. **Observation/resolution:** confirm tentative "keep image @ 32×16" vs the
   recommended "drop image @ 64×32" (§6.2).
3. **Fleet:** confirm full-14 primary (per your "14 spheres" ask); 5-of-14 as
   secondary adaptivity stress (§2).

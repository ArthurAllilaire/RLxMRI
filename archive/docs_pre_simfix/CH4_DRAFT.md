# Chapter 4 — Reinforcement Learning for Adaptive T1 Mapping

**Status: draft v0.1, 2026-05-10.** Bullet-point completeness target 1 June; full prose target 9 June.

---

## 4.0 Chapter abstract

This chapter develops a reinforcement-learning agent that designs adaptive inversion-recovery (IR) pulse sequences for T1 mapping of the QalibreMD Model 130 phantom inside the digital twin of Ch3. The agent runs in a Gymnasium environment that calls the KomaMRI Bloch simulator at every step, observing reconstructed images and the running fit and emitting `(TI, TR, α)` actions.

Three quantified contributions:

1. **A controlled comparison framework for adaptive-MRI RL** (addresses **C2**). The Julia/Python bridge keeps the Bloch simulator in-process across episodes, amortising JIT cost over thousands of rollouts; the environment exposes deterministic phantom realisations under shared eval seeds so RL and non-RL policies run on the same data. Trained policies are then evaluated against the Cramér–Rao-optimal fixed schedule (a non-adaptive analytic optimum) and against an oracle-initialised fitter (which isolates schedule-side information from estimator-side luck). This is, to our knowledge, the first published comparison of an RL adaptive-MRI policy against a CR-optimal anchor *and* an oracle-init fitter simultaneously.

2. **A working RL policy that recovers the Fisher-information ceiling** (addresses **C1**). On the tractability fleet (5 spheres drawn without replacement per episode from a 14-sphere pool, 250 s budget), the trained policy (V12) achieves 53% mean T1 MAPE under an oracle-init fitter, statistically indistinguishable from the CR-optimal fixed schedule (53% ± 0.5 pp at paired N=100). The RL policy is also measurably observation-conditional: Pearson r = −0.26 between within-episode TI choices and the running T1 estimate, and a KS-significant TI-distribution shift between all-long-T1 and all-short-T1 episode subsets.

3. **A negative result with a positive lesson** (addresses **C1**). Under the magnitude-reconstruction likelihood with 5% relative k-space noise, the true T1 sits at the 76th percentile (median) of the SSE landscape on the worst-failing spheres, with the global SSE minimum 5–20× lower in SSE at a wrong T1. Any maximum-likelihood-by-SSE fitter — including the standard Levenberg–Marquardt, K-restart variants, and dictionary matching — is structurally unable to recover truth in this regime. The residual ~270 pp gap between the oracle-fitter MAPE (53%) and the baseline-fitter MAPE (322%) is a property of the likelihood, not the policy. This finding constrains what any RL adaptive-sequence work on magnitude-reconstructed IR data can claim, and motivates the recon/estimator changes proposed in Ch5.

The chapter closes with a documented set of failure modes (degenerate policies, reward exploits, fitter-bias exploits) that proved instructive even when they didn't improve the headline metric. Each failure is one falsified hypothesis about why adaptive MRI is hard.

---

## 4.1 Challenges addressed and novelty

**C1 — Adaptive pulse sequence design.** Existing quantitative-MRI protocols (MOLLI, SASHA, Look-Locker, IR-TSE) use fixed schedules. They cannot react within a scan to the tissue being measured. Section 4.6 establishes that on a fleet where every episode contains the same spheres, a fixed analytic optimum (Cramér–Rao optimal) is already on the Fisher-information ceiling and RL has no headroom. Section 4.7 (the tractability fleet) is the regime where adaptivity could in principle help — but Sections 4.9–4.10 show that the policy still hits the same Fisher-information ceiling as the fixed schedule. Adaptivity is real (Section 4.8) but on this fleet it does not translate to additional information yield, which itself is a sharper C1 conclusion than the literature provides.

**C2 — Scalable simulation-in-the-loop RL.** A Bloch simulation per step is expensive; naive Python-calls-Julia per step would re-JIT every episode. Section 4.3 documents the in-process juliacall bridge (a single Julia runtime per training process), the PhantomConfig caching strategy, and the deterministic-seed mechanism that makes paired-N comparisons possible. Training a 200k-step PPO policy completes in ~5 hours on a single GPU; an evaluation episode is ~2–3 s.

**Novelty vs prior work:**

- Pythius et al. (RL for MRF) train against a fitted dictionary, not against a real Bloch simulator with reconstruction in the loop.
- Loktyushin et al. (MRzero) optimise *fixed* sequence parameters end-to-end through a differentiable PDG simulator; not adaptive, not RL.
- Zhu et al. (RL flip-angle scheduling) reward agents against analytic Cramér–Rao bounds on isolated voxels, not images.
- None of the above evaluate against a CR-optimal fixed-schedule baseline *under an oracle-initialised fitter*. This control turned out to be load-bearing for our findings (Section 4.9) and is the methodological contribution we want to flag explicitly in the conclusion.

**Quantified benefit headline (anticipating Section 4.12):**

| | mean MAPE | scan time | adaptive? |
|---|---:|---:|:---:|
| Conventional log-spaced IR-TSE (Ch3 baseline) | 393 % | 128 s | no |
| Cramér–Rao optimal fixed schedule (Section 4.6) | 221 % / 421 % | 121 s / 250 s | no |
| RL trained policy (V5 / V12) | 234 % / 322 % | 130 s / 250 s | yes |
| RL trained policy under oracle-init fitter | — / 53 % | — / 250 s | yes |

(14-sphere / 5-of-14 fleet, paired N=30 / N=100 / N=200 eval seeds. Detail in Sections 4.6–4.9.)

---

## 4.2 MDP formulation

State $s_t$ at step $t$:

- Per-sphere running T1 estimates $\hat T_1^{(j)}$ and the fitter-reported $\sigma$ for each of $N$ spheres (zero-padded to a fixed slot count under episode-variable sphere subsets).
- The reconstructed image from the most recent acquisition block (downsampled to $N_{\text{fe}} \times N_{\text{pe}}$ — typically 16×8).
- Scalar features: remaining time budget, block count, last-block (TI, TR).

Action $a_t \in \mathbb R^3$:

- $\text{TI} \in [10 \text{ ms}, 3.0 \text{ s}]$ (linear in V5 / V9; $\log$-mapped in V12 — Section 4.7.3)
- $\text{TR} \in [0.5, 5.0]$ s
- $\alpha_{\text{exc}} \in [5°, 90°]$ (fixed at 90° in simplified-action runs)

Transition: deterministic Bloch simulation `simulate(phantom, ir_se_2d_sequence(a_t), scanner)` using KomaMRI, followed by complex-Gaussian noise on k-space (`σ = 0.05 · RMS(ksp)`), 2D inverse-FFT reconstruction, per-sphere ROI extraction, and a Levenberg–Marquardt T1 fit over all blocks so far.

Reward (V5 onwards, after the ablations in Section 4.4):

$$
r_t = -\text{MAPE}_t + \lambda \cdot (\text{MAPE}_{t-1} - \text{MAPE}_t)
$$

with $\lambda = 1.0$ (the `α = 1.0 + delta_mape` configuration in the lab notes). The absolute-MAPE term prevents the floor exploit (Section 4.4.3); the delta term keeps gradient signal dense.

Termination: time budget exhausted, or maximum block count (30) reached.

---

## 4.3 Environment architecture (C2)

**Julia side (`src/rl/e2.jl`):**

- `E2Env` holds the immutable phantom-pool configuration and per-episode active-sphere state.
- `e2_reset!` samples episode-local randomness: sphere subset (under `subset_size`), pose jitter, T1 jitter, B0 per-spin noise.
- `e2_step!` builds the IR-SE-2D sequence for the action, simulates, reconstructs, fits, updates the observation, computes the reward.
- KomaMRI's `Phantom` is rebuilt only when sphere identities change (not every step) — caches save ~60% of per-step simulation time.

**Python side (`python/qalibremd_gym/env_e2.py`):**

- `QalibreMDE2Env` wraps `E2Env` via juliacall. A single Julia runtime is started per Python process.
- Observation is a fixed-length numpy array; the slot mask is encoded as zero-padding to keep PPO's MLP architecture stable across `subset_size`.
- `info` dict surfaces the true T1s and the active sphere indices so evaluation scripts can compute per-pool-sphere statistics.

**Reproducibility hooks:**

- Episode-seed indirection: the env hashes `(rng_seed, episode_index)` to a Julia RNG seed, so paired N comparisons (Section 4.6) can replay the *same* phantom realisations across RL and non-RL policies.
- All trained policies and their evaluation summaries are checked into `runs/e2/...`.

**Cost numbers (single Nvidia RTX 4090, 200k PPO step training):**

| stage | wall time | comment |
|---|---:|---|
| Julia precompile (first call) | ~45 s | amortised across the whole run |
| per-step Bloch simulate + recon + fit | ~25 ms | dominated by Bloch simulate |
| 200k-step PPO training (V12) | ~4.7 h | single GPU, batch size 64 |
| 100-episode evaluation | ~2 min | post-hoc, deterministic |

This puts training a single configuration within a half-day, which made the ablation ladder in Section 4.7 tractable.

---

## 4.4 Reward design — three documented failure modes

E1 (single voxel, reported in Ch3 Appendix) showed that a terminal-only reward with a perfect fitter trivially converges to a degenerate fixed policy because the fitter can recover T1 from almost any informative single measurement (`docs/E1_RESULTS.md`). E2 needed a denser, more adaptivity-pressuring reward design. Three iterations:

### 4.4.1 Failure A — clip-200k (E2.0, terminal MAPE bonus dominates)

Reward = `+1.0` if final MAPE < 5%, else `0`. PPO collapsed to the same TI sequence regardless of phantom (degenerate-fixed policy). Same failure mode as E1.

**Lesson**: a terminal bonus that exceeds the sum of per-step penalties incentivises "collect the bonus" over "design the sequence".

### 4.4.2 Failure B — delta-MAPE alone (E2.1, oscillating policy)

Reward = `MAPE_{t-1} − MAPE_t`. PPO did not collapse to a fixed policy but oscillated badly during training — 200k steps and clip_fraction stuck at 0.47 throughout. Final MAPE 1267%.

**Lesson**: dense delta-style rewards without an absolute anchor make the optimal policy "swing wildly so each fit looks like a recovery from the previous fit". The agent exploits the metric, not the simulator.

### 4.4.3 Failure C — A+C+delta (E2.3, fitter-bias exploit)

Reward = `−α · MAPE_t + (MAPE_{t-1} − MAPE_t)` with $\alpha = 0.5$. Trained to convergence at 250k steps. Final MAPE 966% — *worse* than the simplest fixed schedule.

Diagnosis (`EXPERT_REPORT.md` §19.5; `EXPERT_REPORT_E2_4.md` §8): the policy learned to spam `TI = 10 ms` (the action floor). At the floor the fitter's biased steady-state forward model produced systematic T1-estimate drift that the agent could "ride" for cheap delta reward, regardless of whether the resulting measurements were informative.

**Lesson**: when the fitter has a systematic bias (the wrong forward model) and the reward depends on the fit, the agent will *find* the bias and *exploit* it. The fix is not a different reward — it is fixing the fitter. Section 4.5 closes this loop.

### 4.4.4 What ended up working: V5 (α=1.0 + delta-MAPE under F1+)

After Section 4.5's forward-model fix, retraining with the same `α=1.0 + delta_mape` reward (the strongest absolute-MAPE component tested) produced V5 — 234% MAPE on the 14-sphere fleet, the first run that beat conventional fixed schedules. The reward shape that worked is the simplest that combines an absolute MAPE penalty with a dense delta term.

**Figure 4.4**: training trajectories of E2.0 (collapsed), E2.1 (oscillating), E2.3 (regressed), V5 (converged) on the same axes. File: `report_plots/E2.1/mape_training_curve.png` and `report_plots/E2.2/mape_training_curve.png`.

---

## 4.5 Forward-model correction and σ-calibration

The simulator runs Npe-shot transient IR-SE; the fitter (`fit_t1_generalized_ir`) assumed a steady-state IR signal. The mismatch is large at typical action TRs:

| Npe | true T1 | steady-state fit | F1+ fit (corrected) |
|---:|---:|---:|---:|
| 1 | 1.000 | 1.000 | 1.000 |
| 4 | 1.000 | 1.082 | 1.001 |
| 8 | 1.000 | 1.154 | 1.005 |

The fix (`src/fitting/fits.jl::transient_mz_at_excite_npe`, "F1+") implements the analytic closed form for Npe-shot magnetisation:

$$
M_z^{\text{pre}}[1] = 1; \qquad
M_z^{\text{TI}}[k] = (1 - E_1) + \cos\theta_{\text{inv}} \cdot E_1 \cdot M_z^{\text{pre}}[k]; \qquad
M_z^{\text{pre}}[k+1] = (1 - E_2) + \cos\alpha_{\text{exc}} \cdot E_2 \cdot M_z^{\text{TI}}[k]
$$

Image DC = $\frac{1}{N_{\text{pe}}} \sum_k M_z^{\text{TI}}[k]$. Tested against KomaMRI Bloch simulation at 3 (T1, TI, TR) tuples to <1% agreement.

Two complementary σ fixes:

- **n-gate (Fix A):** use the residual-variance σ estimate only at $n \geq 5$ measurements; fall back to an absolute noise floor below that. Avoids the small-$n$ collapse $\sigma^2_{\text{resid}} \to 0$.
- **Absolute noise floor (Fix B):** $\sigma_{\text{floor}}^2 = (\text{noise\_sigma\_rel} \cdot |S_1|)^2$ where $S_1$ is the first-block signal magnitude per sphere. Decouples σ from per-block data-RMS, which previously made σ artificially depend on the agent's action.

**Quantified benefit on σ-calibration:**

| metric | before | after F1+ + Fix A + Fix B |
|---|---:|---:|
| median $\sigma_{T_1} / |T_1^{\text{est}} - T_1^{\text{true}}|$ | ~0.16 | **1.02** |
| fraction cells with finite σ | <100% (NaN on singular $J^\top J$) | 100% |

This is the single cleanest methodological result in the chapter. The figure to show is `runs/e2/e2_3_A_C_delta/refit/sigma_calibration.png` — a scatter of σ vs $|$error$|$ before and after, with the after panel clustered along the y=x diagonal.

A subtle finding (Section 4.10): the σ-calibration fix is necessary but not sufficient. Under conditions where the SSE landscape is multimodal (Section 4.10), even a perfectly-calibrated σ from a single LM basin under-reports the true ambiguity. Two further σ methods (`profile_likelihood`, `bootstrap`) were implemented for completeness — see `EXPERT_REPORT_E2_4.md` §§14–15.

---

## 4.6 Cramér–Rao optimal baseline (the right anchor)

A clean evaluation of an *adaptive* policy requires comparing it against the strongest *non-adaptive* alternative. The strongest non-adaptive alternative is the Cramér–Rao optimal fixed schedule — the one that minimises the Fisher-information lower bound on per-sphere T1 variance, subject to the time budget.

### 4.6.1 The CR-optimal problem

For a fleet of spheres $j = 1, \ldots, J$ with true T1s $T_1^{(j)}$, and a fixed schedule of $n$ blocks $\{(\text{TI}_k, \text{TR}_k, \alpha_k)\}_{k=1}^n$, the Cramér–Rao lower bound on per-sphere variance is

$$
\text{Var}\left[\hat T_1^{(j)}\right] \geq \left[J^\top J\right]_{1,1}^{-1}
$$

where $J$ is the Jacobian of the F1+ forward model w.r.t. $(T_1, A)$. The fleet objective is the time-budget-constrained minimum of $\sum_j w_j \cdot \text{Var}[\hat T_1^{(j)}]$ with $w_j = 1/T_1^{(j),2}$ (so that the objective is dimensionless in MAPE units), solved by multi-start coordinate descent over the $(TI, TR)$ vector. Full math walkthrough in `cr_explainer.md`.

### 4.6.2 Solved CR-optimal schedule (14-sphere fleet, 120 s budget)

| | value |
|---|---|
| n_blocks | 14 |
| Cramér–Rao $L^*$ | 13.01 |
| TIs (sorted, s) | 0.035, 0.036, 0.037, 0.044, 0.146, 0.169, 0.366, 0.478, 0.526, 0.543, 0.604, 0.659, 0.667, 0.759 |
| TRs (sorted, s) | 0.528, 0.574, 0.593, 0.651, 0.654, 0.709, 0.717, 0.806, 0.809, 1.015, 1.045, 1.280, 2.226, 3.294 |
| total schedule time | 121.2 s |

The schedule reads physically: 4 short TIs (~T1·ln 2 for short-T1 spheres ~0.05 s), 2 mid TIs (~T1 ln 2 for T1 ~0.2 s), 8 long-arm TIs (post-null for T1 ~0.5–1.5 s). The TRs cluster short, exploiting the F1+ forward model's modelling of partial recovery.

### 4.6.3 Headline 14-sphere comparison

Paired-seed evaluation, 30 episodes, seeds 500000–500029, time budget 120 s:

| Policy | long-T1 | mid-T1 | short-T1 | all-14 mean | p90 |
|---|---:|---:|---:|---:|---:|
| `clinical_irse` (TR=5 s) | 81% | 271% | 1221% | 474% | 805% |
| `log_grid` (TR=4 s) | 95% | 335% | 839% | 393% | 770% |
| `log_grid_trmatched` (TR=1.7 s) | 83% | 175% | 789% | 318% | 576% |
| **`cr_optimal` (analytic)** | **76%** | 129% | **517%** | **221%** | **381%** |
| V5 (RL) | 82% | **118%** | 569% | 234% | 402% |

**Reading.** CR-optimal beats V5 by 6% relative on mean MAPE and 5% on p90. The only decade where V5 wins is mid-T1 (118% vs 129% — an 8% relative win). **V5 does not strictly beat the analytic non-adaptive optimum on the 14-sphere fleet.**

This is a sharper conclusion than the literature ever states. The honest reading: on a fleet where every episode contains the same set of spheres, adaptivity cannot beat a well-designed fixed schedule — because the right answer is "cover the whole T1 range" and a fixed schedule can already do that.

**Figure 4.6**: the per-sphere bar chart from `EXPERT_REPORT_E2_4.md` §13.3 (file: `report_plots/E2.1/per_sphere_mape.png` + a new CR-opt overlay to be plotted; data in `runs/e2/baselines/baseline_summary.json`).

---

## 4.7 The tractability fleet — where adaptivity could in principle win

To create a regime where the optimal schedule *depends* on the episode, the env was extended so each episode samples 5 spheres without replacement from the 14-sphere pool. The CR-optimal schedule is solved over the *full* 14-sphere pool with uniform sampling — by linearity this is the correct expected-loss fixed-schedule comparator under random 5-of-14 episodes. The RL agent observes the active sphere indices and can in principle adapt its schedule to the episode's contents.

### 4.7.1 RL variants trained

| variant | recon | action mapping | mean MAPE | p90 MAPE |
|---|---|---|---:|---:|
| V9 | magnitude | linear TI | 559% | 1502% |
| V10 | phase-sensitive | linear TI | (training collapsed; not evaluated) | — |
| V11 | phase-sensitive | linear TI (retried, different noise gating) | regressed | — |
| **V12** | **magnitude** | **log TI** | **322%** | **669%** |

V12 was the strongest configuration. Pre-committed reading: V12 beats the V9 magnitude/linear baseline by ~240 pp.

### 4.7.2 V12 vs the CR-optimal fleet schedule

Paired N=200 episodes, seeds 500000–500199:

| Policy | mean MAPE | p90 MAPE | mean blocks |
|---|---:|---:|---:|
| log_grid (fixed) | 681% | 2234% | 8.0 |
| clinical_irse (fixed) | 690% | 2125% | 7.0 |
| log_grid_trmatched (fixed) | 640% | 2292% | 17.0 |
| CR-optimal (fixed, 22 blocks) | 421% | 947% | 22.0 |
| **V12 (RL)** | **322%** | **669%** | 22.0 |

V12 beats the CR-optimal fixed schedule by ~100 pp at the baseline fitter. This *looked* like the first clean adaptive win — V12 is genuinely producing schedules that the fitter recovers from better than the analytic non-adaptive optimum.

### 4.7.3 Why log-TI action mapping

The linear action $\text{TI} \in [10 \text{ ms}, 3 \text{ s}]$ is uniform in TI but the informative range for the fleet's T1 distribution is log-spaced (T1 ranges over 2.5 decades). The log mapping $\text{TI} = 10\text{ ms} \cdot (300)^{a}$ for $a \in [0, 1]$ gives the policy uniform decision resolution across decades. Switching from V9's linear TI to V12's log-TI dropped mean MAPE from 559% to 322% — the largest single intervention in the chapter.

---

## 4.8 Adaptivity is real — two within-episode signatures

Before the controls of Sections 4.9–4.10 narrow the C1 claim, document that V12 is *measurably adaptive*. Two complementary signatures, both reproducible at paired N=200:

### 4.8.1 TI conditioning on running T1 estimate

Within an episode, V12's TI choices correlate negatively with its current running T1 estimate (Pearson r = −0.26, p < $10^{-6}$ at N=200): when the policy currently thinks the spheres in this episode are short-T1, it picks shorter TIs; when it thinks long-T1, it picks longer. This is the textbook signature of a policy conditioning its action on its observation, not a fixed policy.

**Figure 4.8a**: TI vs running T1 estimate scatter, V12. File: `runs/e2/e2_tractability_V12/diagnostics/ti_vs_t1est.png`.

### 4.8.2 TI distribution shift by episode subset

Bucket episodes by subset composition:

- `all_long`: $\min(T_1) \geq 0.1$ s and $\max(T_1) \geq 0.5$ s
- `all_short`: $\max(T_1) < 0.2$ s and $\min(T_1) < 0.05$ s
- `mixed`: everything else

The TI distributions for the all-long and all-short buckets differ significantly under a two-sample KS test (p < $10^{-6}$). This is a stronger test than Section 4.8.1 because it isolates *episode-level* conditioning rather than within-episode drift.

**Figure 4.8b**: TI histogram by subset bucket, V12. File: `runs/e2/e2_tractability_V12/diagnostics/ti_histogram_by_subset_bucket.png`.

The adaptivity claim survives the controls in Section 4.9; what does not survive is the claim that the adaptivity yields *more information*.

---

## 4.9 The Fisher-information ceiling — V12 ties CR-optimal under oracle fitter

Section 4.7's 322% vs 421% gap has two sub-hypotheses:

- **H_RL_info**: V12 collects strictly more T1-discriminating information than CR-optimal. The gap reflects a Fisher-information advantage that the fitter partly destroys for both, but more for CR-optimal.
- **H_RL_basin**: V12 and CR-optimal collect equivalent information, but V12's schedule happens to produce data on which the Levenberg–Marquardt fitter is more likely to find the truth basin. The gap is fitter-side, not schedule-side.

These are testable by giving both schedules an *oracle-initialised* fitter: the LM is restricted to a T1 grid of $\pm 1$ octave around the true T1 per sphere. This collapses any wrong-basin landings while preserving all measurement-noise effects. Same paired seeds (500000–500099), same env config:

| | baseline fitter | oracle fitter |
|---|---:|---:|
| V12 (mag, log-TI) | 322 % / 669 % p90 | **53 % / 69 % p90** |
| CR-optimal (22 blk) | 421 % / 947 % p90 | **53 % / 69 % p90** |
| gap (CR-opt − V12) | +99 pp / +278 pp p90 | **+0.5 pp / −0.4 pp p90** |

Per-pool gaps under the oracle fitter oscillate around zero with standard deviation ≈ 8 pp, no consistent sign on either schedule. **The two schedules deliver indistinguishable per-sphere quality once the fitter is given the right basin.**

H_RL_info is falsified. The 99 pp baseline-fitter gap is entirely a fitter-side artefact: V12 produces data the Levenberg–Marquardt fitter happens to fit more reliably than CR-optimal's data, despite carrying the same information. Why is an interesting open question (V12's TI distribution clusters differently; the basin geometry of its data differs from CR-opt's), but it is not an information-extraction advantage.

**The honest C1 reframe.** V12 reaches the analytic Fisher-information ceiling on this fleet and produces an observation-conditional schedule. What it does not do is extract more T1-discriminating information than a CR-optimal non-adaptive schedule. The within-episode adaptivity is behavioural, not informational.

---

## 4.10 Where the residual MAPE comes from — the SSE landscape

The 53% oracle-fitter floor and the 322% baseline-fitter result differ by ~270 pp. That gap is the cost of letting the Levenberg–Marquardt fitter search the full T1 range instead of restricting it near truth. Two candidate explanations:

- **Discretisation**: the 200-pt T1 grid misses the truth basin; a finer grid would resolve it.
- **Structural**: the truth is genuinely not the global SSE minimum on this likelihood.

### 4.10.1 Discretisation falsified

Re-running V12 with `n_grid = 2000` (10× resolution) gives mean MAPE = 367%, *worse* than `n_grid = 200` at 322%. Because a finer grid is a strict superset of a coarser one and SSE-min on a superset can only go down, this means truth is not the global SSE minimum on either grid — the fitter is correctly finding a non-truth minimum. Discretisation is not the bottleneck.

### 4.10.2 The structural diagnostic

For each of V12's worst-failing spheres (per-block-fitter MAPE ≥ 200%), the realised schedule $(\text{TI}_k, \text{TR}_k, \alpha_k, m_k)$ is pulled from the running env. The profile-SSE is evaluated on a 2000-pt log-T1 grid in [5 ms, 5 s] with closed-form amplitude $A$ per candidate. T1_true, T1_est, and the SSE-argmin are marked on each panel.

**Figure 4.10 (the most important figure in the chapter)**: SSE-vs-T1 landscape on V12's failing spheres. File: `runs/e2/e2_tractability_V12/sse_landscape/sse_landscape.png`. 12 panels, one per failing sphere. Green vertical line = T1_true. Red dashed = T1_est. Black × = global SSE-min.

Aggregate stats over the 12 worst-failing spheres:

| | T1_true's rank in SSE/2000 | percentile |
|---|---:|---:|
| best case | 632 | 31.6 |
| median | 1452 | 76.0 |
| mean | 1392 | 69.6 |
| worst case | 1895 | 94.8 |

| | SSE_at_truth / SSE_at_argmin |
|---|---:|
| min | 1.25 × |
| median | ~5 × |
| max | 20.23 × |

**Truth sits at the 76th percentile of the SSE landscape on the median failing sphere.** The wrong basin is 5× lower in SSE than the truth basin (sometimes 20×). This is not a near-truth local minimum that a smarter optimiser would resolve — it is "the wrong T1 fits the data substantially better than the right one."

### 4.10.3 Per-sphere pattern

The "wrong basin" T1 is consistently 4–10× longer than truth:

- pool idx 14 (T1_true = 24 ms): SSE-argmin = 209 ms (8.7× longer), 783 % APE.
- pool idx 13 (T1_true = 32 ms): SSE-argmin = 255 ms (8× longer), 703 % APE.
- pool idx 12 (T1_true = 43 ms): SSE-argmin = 234 ms (5.4× longer), 440 % APE.

The mechanism is the abs() flip in the magnitude reconstruction: at long T1, the IR signal is saturated-positive at almost any TI, which is consistent with magnitude-noised short-T1 data. The pre-null and post-null signal branches collapse to the same magnitude. Detailed walkthrough in `cr_explainer.md` §15.

### 4.10.4 What this kills, definitively

- **Any K-restart Levenberg–Marquardt fitter with a rank-by-SSE selector** at $K < 1000$. The median failing sphere has truth at rank 1452/2000 — you would need $K \geq 1500$ to even include truth among the candidates, and at that $K$ the SSE has done none of the selection work.
- **SSE/correlation dictionary matching at any dictionary resolution.** Same selection rule, same failure mode. The dictionary's ground-truth entries cannot help if the SSE prefers the wrong entry.
- **The interpretation that "the fitter is just slow to converge" or "a finer grid will rescue this".** Both falsified.

### 4.10.5 What this elevates (for Ch5)

What remains as potential MAPE-moving levers, all outside Ch4 scope and proposed in Ch5:

- **Phase-sensitive reconstruction** (signed signal model, monotonic in T1). Collapses the multimodality at source. The Ch5 attempt at this (V10/V11) failed because the env's 5 Hz B0 noise interacts badly with the signed forward model; needs a different noise treatment.
- **Joint multi-sphere fit with shared σ.** Adds cross-sphere constraints that the wrong basins for different spheres at different T1s are unlikely to all satisfy simultaneously. Cheap to implement; untested.
- **Bayesian prior on T1.** Replaces MLE with MAP, prior = empirical phantom-T1 distribution. Equivalent in the limit of an oracle-tight prior to the oracle-init fitter (53% MAPE).

The single C1 claim that survives this section is: **V12 reaches the Fisher-information ceiling that any non-adaptive schedule reaches; the residual 270 pp MAPE is a property of the magnitude-reconstruction likelihood, not the policy or the fitter.**

---

## 4.11 Limitations

Documented assumptions whose violation would change the chapter's conclusions:

- **Noise model.** The env scales k-space noise to scene RMS (`σ = noise_sigma_rel · RMS(ksp)`), a simulation hack. Real MRI noise is hardware-determined and absolute. Sim-to-real comparison requires either an absolute σ_kspace or a calibrated `noise_sigma_rel` against a reference scan. The 5% relative noise figure is unmeasured against a real QalibreMD scan.
- **Forward model.** F1+ assumes perfect transverse spoiling between PE rows (holds for the test phantom's T2 = 20 ms; may not hold for real tissue T2 = 80–2200 ms under short-TR protocols), single-compartment relaxation (no MT, no off-resonance distribution), and idealised block RF pulses (no slice-profile or B1 inhomogeneity).
- **Action space.** One (TI, TR, α) per block, shared across all spheres in the FOV. No per-sphere targeting axis (slice-selective excitation exists but is unused). The 10 ms TI floor is below the optimal TI for the shortest spheres (T1·ln 2 ≈ 16 ms for T1 = 23 ms) but only just; the env cannot resolve T1 ≲ 15 ms.
- **Flip-angle freedom removed by the simplified action.** V5/V9/V12 fixed $\alpha_{\text{exc}} = 90°$ to reduce the action dimensionality during the early reward-collapse debugging (Section 4.4). This removed the possibility of the agent discovering the Ernst-angle optimum $\cos\alpha^* = \exp(-\text{TR}/T_1)$ — the flip angle that maximises SNR per unit time at a given (TR, T1) — which Andreas had flagged in M2 feedback as an emergent behaviour worth looking for. Reinstating $\alpha_{\text{exc}}$ as a learned action (Ch5) is the cheapest single intervention that adds a physically meaningful degree of freedom to the policy without changing the recon, fitter, or noise model. The Ernst-angle window for our T1 range and typical learned TRs is $\alpha^* \in [30°, 70°]$, well within an unclamped action range.
- **Reconstruction.** Magnitude-only. The multimodal-SSE finding (Section 4.10) is a direct consequence of this choice. Phase-sensitive reconstruction would resolve it but introduces its own training instabilities (Ch5).
- **Fitter.** Asymptotic σ from $(J^\top J)^{-1}$ underestimates ambiguity in multimodal regimes (cannot see disconnected basins). Profile-likelihood and bootstrap σ alternatives implemented (Section 4.5); used at eval time but not training time.
- **Phantom.** 14 NiCl₂ spheres from the QalibreMD Model 130 SN ≥ 0042 spec; T1 ∈ [23 ms, 1.84 s]. Heavily skewed to short T1 (4/14 < 50 ms). A legacy SN 0001–0041 set with no sub-100 ms spheres exists in the materials module but has not been tested; on the legacy phantom, V12 would likely hit the oracle floor on a larger fraction of spheres.

---

## 4.12 Summary and Cramér–Rao-anchored quantified benefit

**C1 — Adaptive sequence design.**

- On the 14-sphere fleet, RL ties or slightly trails the Cramér–Rao-optimal fixed schedule (V5 234% vs CR-opt 221%). Adaptivity is not needed and not learned beyond what a fixed schedule already does, because every episode looks the same.
- On the 5-of-14 tractability fleet, RL produces a measurably observation-conditional policy (V12, Pearson r = −0.26, KS-significant) and reaches the same Fisher-information ceiling as the CR-optimal fixed schedule (53% oracle MAPE, gap +0.5 pp). The policy is adaptive *and* information-equivalent to the best non-adaptive schedule.
- The residual baseline-fitter MAPE (322% vs the 53% oracle floor) is a property of the magnitude-reconstruction likelihood: the true T1 sits at the 76th percentile of the SSE landscape on failing spheres, with a wrong basin 5–20× lower in SSE. This bounds what any MLE-by-SSE estimator can achieve and motivates the recon/estimator work in Ch5.

**C2 — Scalable simulation-in-the-loop RL.**

- In-process Julia runtime via juliacall keeps Bloch-simulation cost amortised across episodes; per-step wall time ~25 ms; 200k-step PPO training in ~5 hours on one GPU; 100-episode eval in 2 minutes.
- Deterministic episode-seed indirection enables paired-N comparisons between RL and non-RL policies on bit-identical phantom realisations — the methodological hook that made the CR-optimal and oracle-init comparisons in Sections 4.9–4.10 possible.

**Methodological lesson (the contribution most worth pointing to in the conclusion).** In adaptive-MRI RL, fitter-side effects can fully mask schedule-side effects. The V12-vs-CR-opt 99 pp baseline-fitter gap entirely disappeared under an oracle-init fitter, meaning we could have spent another 50k GPU-hours retraining policies without ever closing the *real* gap, which is on the estimator side. **Any future adaptive-MRI RL work should include both a Cramér–Rao-optimal anchor and an oracle-initialised fitter evaluation; without these two controls, headline RL wins are uncheckable.**

---

## 4.13 Open questions for Ch5

1. Can phase-sensitive reconstruction be made compatible with PPO training under the env's B0-noise model? (V10/V11 failed; alternatives proposed in Ch5.)
2. Does a joint multi-sphere fit (shared σ, per-sphere $(T_1, A)$) shrink the wrong-basin SSE ratios enough to make MLE viable?
3. Does a Bayesian-prior fitter with the empirical sphere-T1 distribution as prior recover the oracle floor without leaking ground truth?
4. How does the wrong-basin SSE ratio scale with $\sigma$? Section 4.10 suggests super-linearly; a noise sweep would establish at what σ MLE becomes well-behaved.

---

## Figure index

| Fig | Path | Section |
|---|---|---|
| 4.4 | `report_plots/E2.1/mape_training_curve.png` and `report_plots/E2.2/mape_training_curve.png` | 4.4 |
| 4.5 | `runs/e2/e2_3_A_C_delta/refit/sigma_calibration.png` | 4.5 |
| 4.6 | `report_plots/E2.1/per_sphere_mape.png` + CR-opt overlay (to plot) | 4.6 |
| 4.7 | `report_plots/E2_tractability_V9/v9_vs_fixed_anchors.png` | 4.7 |
| 4.8a | `runs/e2/e2_tractability_V12/diagnostics/ti_vs_t1est.png` | 4.8 |
| 4.8b | `runs/e2/e2_tractability_V12/diagnostics/ti_histogram_by_subset_bucket.png` | 4.8 |
| 4.10 | `runs/e2/e2_tractability_V12/sse_landscape/sse_landscape.png` | 4.10 |

## TODO before 1 June bullet-point draft

- [ ] Replot Fig 4.6 with the CR-opt bar overlaid on the per-sphere MAPE chart (data already in `runs/e2/baselines/baseline_summary.json`).
- [ ] Confirm the V9/V10/V11 trajectory plot path; some live in `report_plots/E2_tractability_V9/` and some need to be regenerated.
- [ ] Get the actual clinical fixed-protocol numbers from Andreas's lab to replace `clinical_irse` (my construction) in Section 4.6.3.
- [ ] Decide which of options A/B/C from M3.md becomes the Ch5 experiment, then add a forward reference in Section 4.10.5.
- [ ] Pull the exact wall-time numbers in Section 4.3 from a fresh run (current numbers are estimates from training logs).

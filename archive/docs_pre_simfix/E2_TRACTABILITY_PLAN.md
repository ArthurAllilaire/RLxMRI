# E2-tractability — Adaptivity test (5 random spheres per episode, 250 s budget, vs CR-optimal)

**Status:** planning · **Owner:** Arthur · **Date written:** 2026-05-08 · **Predecessors:** `EXPERT_REPORT_E2_4.md` §§10–12, `E2_5_PLAN.md`. **Naming:** distinct from `PLAN.md` §4's E3 (MR-fingerprinting); this is a controlled-adaptivity sub-experiment within the E2 family.

---

## 0. One-paragraph summary

E2.4 V5 beats fixed schedules by 1.36× on a 14-sphere phantom with a 120 s budget, but **the result is contaminated by joint-estimation pressure**: 14 spheres spanning ~80× T1 range under one-TI-per-block forces every action to be a compromise, and ~half the budget is consumed before any policy can plausibly produce informative TIs across all decades. This makes "is V5 *truly* adaptive?" hard to answer cleanly from §§10–11 — the win is real but mixed with budget effects. **E2-tractability strips the env to a tractable regime** — 5 spheres per episode (drawn at random from the 14-sphere pool, so the policy sees a *different* T1 distribution every episode), 250 s budget, everything else identical to V5 — and compares an RL policy directly against the Cramér–Rao optimal fixed schedule for the *same distribution of sphere subsets*. **The question this experiment answers**: when the problem is tractable and the policy must generalise across sphere distributions (so it cannot memorise a single schedule), does PPO recover or beat the analytic fixed-schedule optimum?

> **Note on "structural unreachability".** Earlier framing in `EXPERT_REPORT_E2_4.md` §11.4 / §10.4 called short-T1 spheres "structurally unreachable". That overstated the action-set limitation. The action range TI ∈ [0.01, 3.0] s covers the optimal TI for *every* phantom sphere (T1 = 0.023 s → opt TI = 16 ms, well within the 10 ms floor). The short-T1 failure mode is **joint-estimation compromise + multimodal SSE**, not a missing TI. This experiment specifically tests whether RL can resolve the compromise when given more budget per sphere and a varying sphere-subset to generalise across.

---

## 1. Hypothesis and acceptance criteria

### 1.1 Hypotheses

**H1 (tractability).** With 5 randomly-selected spheres per episode and a 250 s budget, the long+mid-T1 region (T1 ≥ 0.1 s) is achievable to < 30 % MAPE by *some* policy. This sets the scale for what counts as "the agent is adapting".

**H2 (RL adaptivity claim).** A retrained PPO policy on the 5-random-sphere env (V9) achieves mean MAPE strictly lower than the Cramér–Rao optimal fixed-block schedule, where CR-opt is solved on the *expected* fleet objective `E[L]` over the random-subset distribution (so it must also generalise, not over-fit one subset). Predicted gap: 10–40 % relative improvement (V9 ≤ 0.85 × CR-opt MAPE).

**H3 (within-episode adaptivity).** V9's TI choices in late blocks correlate with running mean(T1_est) — `|Pearson r| > 0.3` on a TI-vs-T1_est-at-decision scatter. This is the diagnose_e2 test that V5 essentially failed (mean log–log r near zero past block 1).

**H4 (cross-episode generalisation).** V9 picks measurably different action distributions on episodes drawn from disjoint sphere subsets (e.g., "all-long" vs "all-short" subsets), with the difference predictable from the running T1_est obs. If V9 picks the *same* TI distribution regardless of which spheres are present, it has memorised a global schedule and is not truly conditioning on observations — equivalent to the fixed-schedule baseline.

### 1.2 Acceptance criteria

| Criterion | Target | Measured by |
|---|---|---|
| C1 — tractable regime exists | CR-optimal achieves < 50 % MAPE on long+mid spheres in random-5 env | new `baseline_e2.py` run with random-subset sampling |
| C2 — V9 beats CR-optimal | V9 mean MAPE < CR-opt mean MAPE by ≥ 10 % relative across the random-subset distribution | new run `runs/e2/e2_tractability_V9` |
| C3 — V9 is *behaviourally* adaptive | |Pearson r| in TI-vs-T1_est scatter > 0.3 past block 2 | `python/diagnose_e2.py --policy V9` |
| C4 — V9 conditions on the sphere subset | V9's TI distribution on "all-long" subsets differs measurably from "all-short" subsets (KS-test p < 0.05 on TI distributions) | new diagnostic in `python/diagnose_e2.py` |

C2 is the C1 (adaptive sequence design) headline. C3 + C4 are the supporting claims.

### 1.3 What each outcome means

| Outcome | Interpretation |
|---|---|
| C2 + C3 + C4 all pass | RL is genuinely adaptive in the tractable regime. The E2.4 result was tractability-limited, not adaptivity-limited. Strong Ch4 claim. |
| C2 passes but C3/C4 don't | RL beats CR-opt by exploiting a non-adaptivity axis (e.g., TR efficiency). Adaptivity claim weakens. |
| C2 fails by < 5 % | RL ties CR-opt → "PPO recovers the analytic optimum without solving it" — clean methods claim, weaker C1 win. |
| C2 fails by > 5 % | RL trails the fixed-schedule optimum → adaptivity *isn't* the right axis to optimise; the §10 win was tractability/TR efficiency only. Honest negative result. |
| C1 fails (CR-opt > 50 % MAPE) | Even the tractable regime is hard. The phantom + action-set is the binding constraint, not RL. Pivot to E3 (fingerprinting) or E5 (localisation). |

---

## 2. Experimental setup

### 2.1 Phantom — 5 spheres drawn at random per episode

**Each episode reset draws 5 sphere indices without replacement** from the 14-sphere pool of `T1_ARRAY[:T3]`. Sometimes uniformly spread, sometimes clustered (e.g. all-long, all-short, or clumped in mid range) — this is by design. The policy must handle whatever fleet T1 distribution it gets, which is the cross-episode generalisation test (H4).

**Why random rather than fixed:**
- A fixed subset rewards memorisation of one schedule. The policy could learn "TIs at 0.7, 0.2, 0.04, ..., s" and apply it every episode without conditioning on observations — the same failure mode as V5's fixed cold-start TI = 0.629 s in block 1.
- A random subset *forces* the policy to read the running T1_est obs and condition its TI choices on the actual fleet present. This is what "truly adaptive" means.
- It also gives a stronger CR-optimal comparison: the CR-opt schedule is solved over the *distribution* of subsets (i.e. minimises `E[L]`), not over a single subset. So the analytic optimum is also a generalising schedule, and V9 vs CR-opt is apples-to-apples.

**Sphere pool**: full 14 from `T1_ARRAY[:T3]` (T1 ∈ [0.023, 1.838] s).

**Sampling distribution**: uniform without replacement over `binom(14, 5) = 2002` possible subsets per episode. Seed-deterministic given the episode's `rng_seed`.

**No new `cfg_field` needed.** Implementation is a small edit to `E2Env`:
- Add kwarg `subset_size::Union{Nothing, Int} = nothing` — when `nothing`, use all 14 (V5 behaviour, full back-compat); when an integer `k`, draw `k` spheres at reset.
- At `e2_reset!` time: sample subset indices via the per-episode RNG, build phantom from those sphere centres + T1 values, populate per-sphere arrays of size `k`, set `env.n_spheres = k` for the duration of the episode.
- `env.T1_base[i]` becomes `env.T1_base_pool[idx_subset[i]]` after subset selection. The lognormal jitter (`T1_sigma_rel`) applies on top, as today.

The phantom builder already accepts arbitrary sphere lists (`build_phantom` takes a centres + materials vector), so no change there. Per-episode rebuild adds ~10 ms to reset — negligible vs the 16 s simulator-per-block. **~2 hours code + test.**

### 2.2 Time budget — 250 s

Doubled from 120 s. Rationale:
- V5 averaged ~16 s/block × 8 blocks = 128 s. 250 s budget at the same per-block cost gives **~16 blocks per episode**.
- With 5 spheres, that's **~3.2 informative TIs per sphere on average** — enough for a 2-parameter (T1, A) fit with measurable σ on each sphere.
- Compared to V5's ~0.6 informative TIs per sphere (8 informative blocks ÷ 14 spheres) — 5× more per-sphere data density.

This is the single biggest tractability lever; pairing it with sphere reduction is the controlled simplification.

### 2.3 Other env parameters — *unchanged from V5*

To keep the comparison clean:
- Action space: 3-dim simplified (TI, TE, TR) ∈ same bounds as V5
- Forward model: F1+ (no EPG)
- Noise: `noise_sigma_rel = 0.05`
- Fitter: profile-likelihood σ if E2.5 §3 has landed by the time this experiment runs; otherwise asymptotic σ as in V5
- Pose / T1 jitter: same as V5
- Reward: `delta_mape`, `α = 1.0`, `terminal_bonus = 0.0` (V5 settings)

The single difference vs V5 is `(n_spheres, time_budget_s) = (5, 250)` instead of `(14, 120)`.

### 2.4 Eval seeds

Same convention as `train_e2.py`: eval seeds = 500 000 + i for i ∈ [0, 30). New env, new phantom config → fresh seeds give fresh phantom realisations; per-episode determinism preserved.

---

## 3. Cramér–Rao optimal baseline (5-sphere version)

### 3.1 Adapt `E2_5_PLAN.md §4` to the random-subset distribution

The CR-opt schedule must also generalise across the random subsets — otherwise V9 vs CR-opt is unfair. Two formulations:

**Formulation A — expected-loss CR-opt (preferred for V9 comparison).** Minimise the *expected* fleet objective over the 2002 possible subsets, weighted by their probability under uniform sampling:
```
L(schedule) = E_{S ⊂ pool, |S|=5} [ Σ_{j ∈ S} σ²_T1_j(schedule) / T1_j² ]
            = (1/binom(14,5)) Σ_{S} Σ_{j ∈ S} σ²_T1_j / T1_j²
            = (1/14) Σ_{j ∈ pool} σ²_T1_j / T1_j²    # by linearity (each j in C(13,4)/C(14,5) = 5/14 of subsets)
```

So Formulation A reduces to **CR-opt over all 14 spheres simultaneously** — i.e. a single global schedule that tries to minimise variance across the whole pool. This is the analytic counterpart of "policy that doesn't know which subset it'll see". Direct comparison to V9.

**Formulation B — oracle CR-opt.** Solve a different schedule for each sampled subset — gives the agent the unfair advantage of seeing the subset before acting. Reports `E_S [L(schedule_S* | S)]`. Useful as a *lower bound* on what any non-adaptive schedule can achieve, but not an apples-to-apples comparison to V9 (which doesn't see the subset upfront either, only learns it from data).

Run both. Report Formulation A as the primary V9 comparison. Formulation B as a sanity bound — V9 should *not* beat oracle CR-opt (that would be impossible), and the gap V9 ↔ oracle bounds how much information V9 is recovering from observations.

Sweep `n_blocks ∈ {6, 10, 14, 18, 22}`, multi-start optimiser per `n_blocks`, pick global minimum over all configs. Same code as E2_5_PLAN.md §4.3; only the fleet T1 list and the budget change.

### 3.2 Predicted CR-optimal behaviour (Formulation A)

With 14 spheres in the pool and ~16 blocks, the global-optimal schedule is likely to:
- Place TIs at roughly each sphere's T1·ln 2, possibly with some spheres skipped if their CR contribution is dwarfed by neighbours
- Pick TRs near 1.5–2 s (V5's empirical optimum)

A non-adaptive schedule cannot beat the same schedule run subset-by-subset — but a well-chosen 16-block schedule covering the full T1 range can hit *most* of any 5-sphere subset's informative TIs. V9 needs to do better by *not* spending blocks on spheres absent from the current episode.

---

## 4. V9 — RL training

### 4.1 Command

```bash
PYTHON_JULIAPKG_OFFLINE=yes python python/train_e2.py \
    --reward-mode delta_mape --simplified-action \
    --terminal-bonus 0.0 --mape-alpha 1.0 \
    --max-blocks 30 \
    --time-budget 250.0 \
    --subset-size 5 \
    --timesteps 200000 \
    --out runs/e2/e2_tractability_V9
```

(Three flags wire through: `--time-budget` and `--subset-size` are new; `--max-blocks` already exists. The env kwargs `time_budget_s` and `subset_size` are added per §2.1. ~1.5 hours code, including the per-reset random-subset draw and per-episode array sizing.)

### 4.2 Expected training dynamics

- Episodes will be longer (16 blocks vs 8) → wall-clock per timestep up ~2×.
- 200k timesteps total → ~10 h compute (vs V5's ~5 h).
- ep_rew_mean scale will differ (5 spheres instead of 14), so direct comparison to V5 isn't meaningful — use eval MAPE instead.

### 4.3 Eval cadence

`E2EvalCallback` already runs every 10k steps. Keep that; reduce eval episodes to 20 (vs 30) to save eval time given longer episodes.

---

## 5. Diagnostics — three plots, three claims

### 5.1 TI-vs-T1_est-at-decision scatter (C3)

`python/diagnose_e2.py --policy V9 --vecnorm V9_vn --episodes 30 --simplified-action`. Look at `ti_vs_t1est.png`'s log–log Pearson r. **Pass: |r| > 0.3**.

### 5.2 Per-subset-bucket TI distribution (C4)

Run V9 on 100 eps with random subsets recorded. Bucket episodes by their fleet T1 distribution:
- **all-long**: max(T1) ≥ 0.5 s, min(T1) ≥ 0.1 s (no short-T1 sphere present)
- **all-short**: max(T1) < 0.2 s, min(T1) < 0.05 s (no long-T1 sphere present)
- **mixed**: anything else

For each bucket, plot the aggregate TI histogram (across all blocks of all episodes in that bucket). **Pass: KS-test p < 0.05 between all-long and all-short buckets' TI distributions** — the policy demonstrably picks different TIs when given different sphere distributions.

If the histograms are statistically indistinguishable, V9 has memorised a global schedule and is not conditioning on observations — equivalent to a fixed schedule and falsifies H4 even if H2 passes. Honest negative result.

### 5.3 V9 vs CR-opt per-sphere bar chart

Same as §11.2 of `EXPERT_REPORT_E2_4.md`, but 5-bar version. **Pass: V9 ≤ CR-opt on ≥ 4 of 5 spheres, mean MAPE V9 < CR-opt by ≥ 10 % relative.**

---

## 6. What this experiment closes off

- **If H2 + H3 hold (V9 beats CR-opt by ≥ 10 % and is behaviourally adaptive)**: the C1 claim graduates from "RL beats fixed grids on the 14-sphere problem" to "RL beats *the theoretical fixed-schedule optimum* in a tractable regime — a within-episode adaptivity that no fixed schedule can express". That's the strongest version of C1 available without hardware.
- **If H2 fails (V9 ≤ CR-opt)**: PPO has not captured the within-episode information available in the F1+ model. The 14-sphere V5 win was TR efficiency + tractability artefacts. This is a clean negative result and points the next experiment at *why* — likely the σ-channel encoding or the policy-network capacity / observation set.
- **If H1 fails (CR-opt > 50 % MAPE in the tractable regime)**: even the simplified problem is too hard for fixed schedules — meaning the env's structural limits (action floor, magnitude-only recon, single-TI-per-block) bind regardless of T1 sample density. Pivot earlier than expected to the structural fixes (slice-selective excitation, log-spaced action mapping, EPG).

In all three branches, this experiment produces a publishable methods result. The H2-passing branch is the strongest C1 narrative; the H2-failing branch is a useful "where RL leaves structure on the table" result; the H1-failing branch is the structural-limitations finding.

---

## 7. Effort and timeline

| Task | Effort | Compute |
|---|---:|---:|
| Wire `--subset-size` and `--time-budget` flags + per-reset random-subset draw + per-episode array sizing | 1.5 h | — |
| Add random-subset support to `baseline_e2.py` (sample subset per eval episode, run schedule, aggregate) | 1 h | — |
| Adapt CR-opt solver from `E2_5_PLAN.md §4` to expected-loss formulation over 14-sphere pool | 30 min (after E2.5 §4 lands) | — |
| Add subset-bucket KS-test diagnostic to `python/diagnose_e2.py` | 45 min | — |
| Run CR-optimal solver | — | ~30 min CPU |
| Run CR-optimal schedule + log_grid through 5-sphere env (30 eps each) | — | ~30 min |
| Train V9 (200k timesteps) | — | ~10 h |
| Eval V9 (30 eps) + diagnostics | 30 min | ~30 min |
| Write up §13 of `EXPERT_REPORT_E2_4.md` (or new results doc) | 2 h | — |
| **Total hands-on** | **~6 h** | — |
| **Total compute** | — | **~12 h** |

Single working day of attention; one overnight RL run. Fits comfortably inside one Wayne update cycle.

**Dependency:** §3 (CR-optimal solver) reuses `E2_5_PLAN.md §4` infrastructure. E2.5 should land first or in parallel.

---

## 8. Risks

| Risk | Likelihood | Diagnostic | Mitigation |
|---|---|---|---|
| **R1.** V9 still gets stuck on the 0.01 s floor exploit | medium | TI histogram modal-bin share > 30 % at floor | Run V7 (`TI_min = 0.05 s`) version of V9 in parallel — same env otherwise |
| **R2.** 250 s budget is gameable too — agent finds a longer-duration analogue of the floor exploit | low | unexpected MAPE plateau early in training | Inspect ep_rew_mean curve; if it plateaus before timestep 50k, diagnose action distribution at that point |
| **R3.** CR-optimal solver doesn't converge well (multimodal, gets stuck in local min) | medium | randomised restart spread > 2× in objective | Multi-start (5+ random seeds), pick best; report seed-spread as honesty |
| **R4.** V9's first-block TI is again a fixed cold-start picker (not adaptive in block 1) | high | block-1 TIs identical across episodes | Expected — block-1 has no T1_est obs, can't adapt. Caveat in §5.1 ("adaptivity past block 1"); not a fail. |
| **R5.** The `n_blocks ≈ 16` regime exposes a different failure mode (e.g., σ-channel saturates differently, value-function fitting struggles with longer episodes) | low | training metrics outside V5's healthy ranges | Treat as new finding; document but don't retry blindly. |

R1 and R3 are the real risks. R4 is anticipated and not a real failure. R2/R5 are low-probability surprises.

---

## 9. Why this is the right next experiment

E2.4 produced a real C1 win but with caveats (§§11–12). E2.5 (σ correctness + CR-opt baseline + per-decade reporting) closes the *measurement* and *reporting* gaps. **E2-tractability closes the controlled-comparison gap**: when the problem is tractable, does the agent demonstrate within-episode adaptivity beyond what any fixed schedule can?

Three properties make this the right test, in order:

1. **It's the cheapest single experiment that can promote the C1 claim.** ~12 h compute, ~5 h hands-on, single overnight. Compare to retraining on a richer action space (E3 fingerprinting or slice-selective targeting) which is multiple days of code + days of compute.
2. **It uses only existing infrastructure** (F1+ env, simplified-action mode, baseline_e2.py framework, CR-optimal solver from E2.5) plus three small flag additions. No new physics, no new MDP semantics.
3. **The negative result is informative.** If V9 doesn't beat CR-opt, we have evidence that PPO is leaving structure on the table even in tractable conditions — a precise diagnosis pointing the next-but-one experiment at the right axis (obs encoding, policy capacity, or reward shaping) rather than at the env.

The dissertation Ch4 is best served by either an H2-passing E2-tractability run (strongest possible C1 sentence) or an H2-failing run with a follow-up (precise diagnostic-driven experiment). Either way it's net-positive narrative material.

---

## 10. What this experiment doesn't address (and why that's fine)

- **Real-data validation.** Still simulator-only. Sim-to-real is E5 territory.
- **Per-sphere targeting.** Still one TI per block, no slice-selective RF. The structural limit on short-T1 spheres is unchanged. (E3 fingerprinting / E5 territory.)
- **Multi-compartment / EPG / off-resonance.** Forward model is still F1+. (E2.4 §2.5 if needed.)

These are all "next-after" axes. E2-tractability is specifically the controlled adaptivity test.
# E2.5 — Honest σ + theoretical baseline + per-decade reporting

**Status:** planning · **Owner:** Arthur · **Date written:** 2026-05-08 · **Predecessors:** `EXPERT_REPORT_E2_4.md` §§8–10, `E2_4_PLAN.md`, `docs/T1_FIT_AND_KOMA_TESTS.md` §7.5.

---

## 0. One-paragraph summary

E2.4 V5 is a clean win (234 % all-14 MAPE / ~78 % long+mid MAPE; 2.0× over `log_grid` baseline; 1.36× over TR-matched fixed grid). But three issues surfaced in §§9.9–10.9 that block a clean Ch4 chapter and a fair sim-to-real story: (1) the fitter's σ formula is calibrated to the env's specific noise model (per-sphere relative), so it under-estimates σ on short-T1 spheres by 5–30× — fine on average, **wrong on the cells the policy most needs to flag**; (2) we have no theoretical lower bound for what *any* fixed schedule can achieve, so the 1.36× residual win over TR-matched is unanchored; (3) the headline metric averages over structurally-unreachable short-T1 spheres, hiding both the win and the failure. **E2.5 is therefore a measurement experiment, not an RL experiment** — replace asymptotic σ with profile-likelihood σ (sim-agnostic), add a Cramér–Rao optimal fixed-grid baseline (theoretical lower bound), report MAPE per T1 decade, and document the simulation assumptions that make our results contingent on the noise/spoiling/single-compartment models. **No retraining required for the headline; an optional V8 retrain is gated on whether the corrected σ-channel changes policy obs distribution meaningfully.**

---

## 1. Hypothesis and acceptance criteria

### 1.1 Hypothesis

**H1 (σ correctness, *not* root cause).** Profile-likelihood σ widens σ_T1 by ≥3× on the multimodal-SSE failures (short-T1 spheres in wrong basins) without materially changing σ on well-determined cells. **The point-estimate T1 is unchanged** (LM minimum is unaffected; only the confidence interval changes). Median σ/|err| stays near 1.0; the *tail* of σ/|err| (currently dominated by over-confident short-T1 cells) tightens.

**H2 (theoretical anchor).** A Cramér–Rao optimal fixed-block schedule (`argmin_{TIs, TRs}` of `trace((J^T J)⁻¹)` weighted by 1/T1²) achieves a measurable MAPE on `QalibreMDE2Env`. V5 either matches/beats this (= RL has captured the fixed-schedule optimum without solving the optimisation problem analytically) or trails it (= RL has not yet matched the analytic optimum).

**H3 (per-decade reporting reframe).** Splitting MAPE by T1 decade (long ≥ 0.5 s, mid 0.05–0.5 s, short < 0.05 s) reveals the V5 wheelhouse (mid-T1) and isolates the structurally-unreachable tail (short-T1) without dropping it from the report.

### 1.2 Acceptance criteria

| Criterion | Target | Measured by |
|---|---|---|
| C1 — profile-likelihood σ widens correctly | σ_T1 ≥ 3× larger on cells where \|T1_est − T1_true\|/T1_est > 1.0; ≤ 1.5× on cells where |err| < 0.1 | new test in `test/test_e2.jl` + V5 re-eval scatter |
| C2 — point estimate unchanged | per-cell \|ΔT1_est\| < 1 % between asymptotic-σ and profile-likelihood-σ runs | direct comparison on V5 rollouts |
| C3 — CR-optimal grid runs cleanly | reports per-sphere MAPE, scan time, and grid TIs | new `python/baseline_e2.py:_cr_optimal_grid` |
| C4 — per-decade table is actionable | three rows (long/mid/short), V5 vs each baseline per row, gap clearly attributable to TR efficiency vs TI adaptivity vs unreachability | new §11 of `EXPERT_REPORT_E2_4.md` |

C1–C3 are engineering deliverables; C4 is the report-writing deliverable.

---

## 2. Why my original σ-floor "fix" was wrong (and what we're doing instead)

In conversation we proposed a 1-line σ-floor fix that replaced per-sphere `noise_sigma_rel × |first-block magnitude|` with scene-level `noise_sigma_rel × scene_RMS`. **That was overfitting to the env's specific noise model** and we discarded it. Three reasons:

1. **The env's noise scaling is a simulation hack, not physical.** `src/rl/e2.jl:260` adds σ_kspace = `noise_sigma_rel × RMS(full ksp)`. Real MRI noise σ_kspace is hardware-determined (thermal, ∝ √(k_B T BW R)/G), independent of signal scene. The env's "5 % of scene RMS" formulation is a normalisation hack to keep noise meaningful across phantom configs.
2. **Hardcoding the env model into the fitter breaks sim-to-real.** A fitter that assumes `σ ∝ scene RMS` works on the env but is misspecified on real scanner data. The whole point of having a fitter that operates on data, not on simulator parameters, is that it transfers.
3. **It doesn't fix the actual failure mode.** The §9.9.4 short-T1 failures are **multimodal SSE** — abs() in `S = A·|1 − 2·exp(−TI/T1)|` creates two T1 basins for the same data. Asymptotic σ from local J^T J is a quadratic approximation around *one* basin and cannot see the other. Even with a perfect noise floor, asymptotic σ would still be over-confident on these cells.

**Profile-likelihood σ addresses the root** by not assuming a noise model *or* a quadratic SSE landscape — it walks the SSE surface and reports the actual T1 range consistent with the data.

---

## 3. Profile-likelihood σ — implementation

### 3.1 Method (per `docs/T1_FIT_AND_KOMA_TESTS.md` §7.5 fix 3)

For each sphere with n ≥ 2 measurements:

1. Run the existing LM fit → `(T1*, A*, SSE_min)`.
2. Build a candidate-T1 grid: `T1_c ∈ exp(linspace(log(0.5·T1*), log(2·T1*), 31))` (log-spaced, ±factor-of-2 around the best). 31 points balances resolution vs cost.
3. For each `T1_c`: re-fit *only* A (closed form for the IR magnitude model — no LM needed) → `SSE(T1_c)`.
4. The 1σ T1 range is `{T1_c : SSE(T1_c) ≤ SSE_min · (1 + F(1, n−2; 0.683)/(n−2))}` — F-test threshold for one nuisance parameter (A) at 68.3 % confidence.
5. `T1_sigma = max(T1* − T1_lo, T1_hi − T1*)` — the half-width of the 1σ interval (asymmetric handled conservatively by max).
6. **Multimodal handling:** if the 1σ set is disconnected (two basins both pass the SSE threshold), report `T1_sigma = T1_hi − T1_lo` over the *full extent* (lo = leftmost-basin lo, hi = rightmost-basin hi). This is the conservative wide-σ behaviour we want for cells in wrong basins.

### 3.2 Closed-form A given T1

The model is `mag_k = A · |f_k(T1)|` where `f_k(T1) = transient_mz_at_excite_npe(T1, TI_k, TR_k, π, α_exc_k; Npe)`. With T1 fixed, `A* = sum(mag_k · |f_k|) / sum(|f_k|^2)` and `SSE = sum((mag_k − A* · |f_k|)^2)`. No iteration needed — one matvec per `T1_c`. Cost ~30 floating-point ops × 31 grid × 14 spheres = ~13 k ops per env step. Negligible vs the ~16 s KomaMRI sim.

### 3.3 Code change — `src/fitting/fits.jl`

Add `sigma_method = :asymptotic | :profile_likelihood` keyword to `fit_t1_generalized_ir`. Default `:asymptotic` for back-compat. New helper:

```julia
function _profile_likelihood_sigma(T1_best, A_best, SSE_min,
                                    TIs, mags, TRs, α_excs;
                                    Npe, n_grid = 31, conf = 0.683)
    n = length(TIs)
    n ≥ 3 || return NaN          # need ≥3 points for an F-test with 1 free A
    # Threshold: SSE_min · (1 + F_{1, n-2}(conf) / (n - 2))
    F_crit  = quantile(FDist(1, n - 2), conf)
    SSE_thr = SSE_min * (1 + F_crit / (n - 2))

    T1_grid = exp.(range(log(0.5 * T1_best), log(2.0 * T1_best); length = n_grid))
    sse_at  = similar(T1_grid)
    @inbounds for (i, T1c) in enumerate(T1_grid)
        f = [transient_mz_at_excite_npe(T1c, TIs[k], TRs[k], π, α_excs[k];
                                          Npe = Npe) for k in 1:n]
        af = abs.(f)
        A_c = sum(mags .* af) / max(sum(af .* af), 1e-30)
        sse_at[i] = sum((mags .- A_c .* af) .^ 2)
    end
    # Find the 1σ band: contiguous-or-disconnected set where sse ≤ SSE_thr
    mask = sse_at .≤ SSE_thr
    any(mask) || return NaN
    lo = T1_grid[findfirst(mask)]
    hi = T1_grid[findlast(mask)]
    return (hi - lo) / 2          # conservative half-width, includes any
                                   # disconnected basins by spanning lo..hi
end
```

Wire `sigma_method = :profile_likelihood` through `_e2_update_t1_estimates!` (`src/rl/e2.jl:302`). Keep the asymptotic path as a back-compat option for the existing tests.

### 3.4 Tests

In `test/test_e2.jl`, three new testsets:

1. **Profile-likelihood σ on well-determined fit.** T1_true = 0.5 s, 8 TIs in [0.05, 2.0] s, no noise → SSE_min ≈ 0; asymptotic σ ≈ 0; profile-likelihood σ ≈ 0 (band collapses). `@test profile_σ < 0.05 · T1_true`.
2. **Profile-likelihood σ on multimodal SSE.** Synthetic short-T1 case (T1_true = 0.02 s) with TIs only in [0.5, 3.0] s (saturated regime). LM lands in wrong basin (T1_est ≈ 0.05–0.5 s). Asymptotic σ tiny; profile-likelihood σ ≥ T1_est (i.e., should span the wrong basin and the right one).
3. **Profile-likelihood σ regression — point estimate unchanged.** Run both methods on a held-out V5 rollout (recorded `(TI, TR, mag)` tuples). `@test all(abs.(T1_asymp .- T1_profile) ./ T1_asymp .< 0.001)`.

### 3.5 What this *does not* fix

To set expectations explicitly:

- **Point estimate T1 is unchanged** (LM minimum is the same; only the confidence interval moves). So **MAPE will not change** under this fix. Headline numbers stay 234 % / 78 % long+mid.
- **Multimodal SSE basins are still chosen by LM init** — the wrong basin is still picked when the data is degenerate. We just *report* the ambiguity now via wide σ.
- **Policy obs σ-channel** sees a different distribution, but only retraining (V8) tests whether the policy can use the new signal.

So this is a **correctness fix for honest reporting**, not a root-cause fix for short-T1 failures. The root cause (action set × phantom range × multimodal abs() landscape) is structural and addressed in E3+, not E2.5.

---

## 4. Cramér–Rao optimal fixed-grid baseline — implementation

### 4.1 Why this is the right anchor

`§10` baselines are my construction (`log_grid`, `clinical_irse`) — sensible but arbitrary. The 1.36× residual win over TR-matched fixed grid only shows V5 beats *that particular* fixed schedule. The defensible Ch4 claim wants the form **"V5 beats the theoretical optimum for any fixed schedule under the F1+ forward model"** — which requires solving the fixed-schedule optimisation analytically and running the result.

### 4.2 The optimisation problem

Fix `n_blocks` and a scan-time budget `T_budget = 120 s`. For each sphere `j` with nominal `T1_j` and noise model σ_j (we'll use σ = constant absolute value matching the env's effective per-pixel σ — *not* the per-sphere relative model), the Cramér–Rao bound on T1_j given a schedule `(TI_k, TR_k)_{k=1..n}` is:

```
σ²_T1_j ≥ σ²_j · [(J_j^T J_j)⁻¹]_{T1, T1}
```

where `J_j` is the n×2 Jacobian of `S(TI_k; T1_j, A_j) = A_j · transient_mz_at_excite_npe(T1_j, TI_k, TR_k, π, π/2; Npe=8)` evaluated at the truth `(T1_j, A_j=1)`. The *fleet-level* objective is the total weighted T1 variance across the 14 spheres:

```
L(schedule) = Σ_{j=1..14} σ²_T1_j / T1_j²        # MAPE-like weighting
s.t.  Σ_k block_time(TI_k, TR_k) ≤ T_budget
       n_k = 8 shots per block (Npe)
       0.005 ≤ TI_k ≤ 3.0
       max(TI_k + 0.05, 0.5) ≤ TR_k ≤ 5.0
```

`block_time(TI, TR) = Npe · TR + small overhead`.

### 4.3 Solver

The objective is non-convex (multimodal in T1 → multimodal in J^T J), so use a global optimiser. Two-stage:

1. **Coarse grid scan**: enumerate `n_blocks ∈ {4, 6, 8, 10, 12}` and for each, sample 1000 random schedules respecting the constraints; pick the best 10.
2. **Local refine**: warm-start a constrained nonlinear optimiser (`Optim.jl` `LBFGS` with box constraints) from each of the top 10. Pick the global minimum.

Output: best fixed schedule for each `n_blocks` value plus the `n_blocks` that minimises `L`. Run that schedule through `QalibreMDE2Env` (deterministically — same eval seeds as V5) → report MAPE / p90 / per-sphere / scan time. Compare to V5.

### 4.4 Code

New file `python/baseline_e2_cr.py` — Julia call (the optimiser) + Python eval. Or pure Julia in `src/baselines/cr_optimal.jl` and a Python wrapper that reads the resulting schedule and runs it via `baseline_e2.py`. Estimated ~150 LOC + ~30 LOC test.

### 4.5 Test

Synthetic 2-sphere problem (T1 ∈ {0.1, 1.0} s) with 4-block budget. CR-optimal schedule should pick TIs near `{T1·ln 2}` for each sphere → TIs ≈ {0.07, 0.7} s with two repeats for averaging. `@test |TI_grid_optimal - {0.07, 0.07, 0.7, 0.7}| < 0.05`.

### 4.6 Effort

- Optimiser code: ~3 h
- Test: ~30 min
- Run CR-optimal schedule through env: ~30 min compute (30 eps)
- Per-sphere comparison plot: ~30 min

---

## 5. Per-decade MAPE reporting — schema and implementation

### 5.1 Schema

Three T1 decades:

| Region | T1 range | Spheres (idx) | n |
|---|---|---|---|
| **long-T1** | T1 ≥ 0.5 s | 0–4 | 5 |
| **mid-T1** | 0.05 ≤ T1 < 0.5 s | 5–10 | 6 |
| **short-T1** | T1 < 0.05 s | 11–13 | 3 |

The boundaries are physically motivated:
- 0.5 s: roughly 1 s / 2, the geometric centre of the long-T1 cluster
- 0.05 s: the action-set lower bound (TI_min for non-floor choices)

### 5.2 Reported metrics per region

For each (policy, region):
- `mean_MAPE` — mean across spheres in the region, mean across episodes
- `p90_MAPE` — 90th percentile across (sphere × episode) cells
- `mean σ_T1` — average reported confidence
- `σ-calibration ratio` — median σ/|err| within the region

### 5.3 Where to add it

- New section `§11` in `EXPERT_REPORT_E2_4.md` with three tables (one per region) comparing V4 / V5 / log_grid / log_grid_TRmatched / clinical_irse / CR-optimal.
- New flag `--per-decade` in `python/eval_e2.py` and `python/baseline_e2.py` that writes a `per_decade_summary.json` alongside the existing summary.
- New plotting routine in `python/plots_for_report.py` for the three-panel per-decade comparison plot (one bar chart per region, policies side-by-side).

### 5.4 Effort

- JSON output: ~30 min (eval scripts + baselines)
- Three-panel plot: ~45 min
- Section 11 writeup: ~1 h

---

## 6. The σ question — incremental correctness or root cause?

**It's incremental correctness, not root cause. Here's why.**

What σ affects:
- (a) The fitter's reported confidence interval (an *output* to the policy obs and to the report).
- (b) The fitter's noise floor `σ²_eff = max(σ²_resid, σ²_floor)` — used to scale termination criteria but *not* the LM minimum itself.
- (c) The policy's σ-obs channel (`log10(σ_T1 / T1_est)` clamped to [−3, 0]) — informs adaptive behaviour.

What σ does *not* affect:
- The point estimate T1 (LM finds the SSE minimum regardless of σ).
- MAPE = `|T1_est − T1_true| / T1_true` — depends only on the point estimate.

So:
- **Headline MAPE is unchanged by any σ fix.** V5 stays at 234 % / 78 % long+mid regardless.
- The σ fix changes (a) — honest σ for the reader; (b) — slightly better LM behaviour at the margins; (c) — a new signal the policy *could* learn to use, but only if retrained.

**Was σ already accurate?** On average yes — V1 reported median σ/|err| = 1.02. But per-cell:
- Median ≈ accurate
- Tail (~5–10 % of cells, concentrated on multimodal-SSE short-T1 failures) is over-confident by 5–30×
- A median statistic hides the cells the policy most needs to flag

The fix raises the floor of σ-honesty; it doesn't change what the policy or the env produces.

**Does this affect the C1 narrative?** Indirectly — yes:
- Without the σ fix, claiming "the σ channel calibrates well" is misleading because of the per-cell tail.
- With the σ fix, σ is honestly large on the cells that fail, and the σ channel becomes a real "this sphere needs more attention" signal.
- A retrain (V8) under honest σ might find a policy that revisits short-T1 spheres more — but might not, because the structural unreachability still applies.

**Verdict.** σ correctness is a Ch4 reporting requirement (don't claim calibrated σ when the tail is over-confident); it's not the lever that closes the headline MAPE gap. Document it as such — fix it, report on the change, but don't expect the headline to move.

---

## 7. Documented assumptions — what we now know is contingent

A side-effect of this conversation is that several simulator/forward-model assumptions are now visible. They should be documented in the report's limitations section (proposed §12 of `EXPERT_REPORT_E2_4.md`):

1. **Noise model is a simulation hack.** `σ_kspace = noise_sigma_rel × RMS(full ksp)` is calibrated to scene loudness, not hardware. Real MRI noise is absolute. Effect on results: roughly absolute-noise-like in practice (because RMS is dominated by long-T1 spheres), but coupled to scene composition — change the phantom and the effective per-sphere SNR shifts.
2. **Forward model assumes perfect spoiling.** F1+ is correct for `T2 ≪ TR − TI`, which our `T2 = 20 ms` test phantom satisfies. On real-T2 tissue this assumption breaks → EPG (`E2_4_PLAN.md` §2.5).
3. **Single-compartment relaxation.** No MT, no off-resonance distribution, one (T1, T2) per voxel. Real tissue has multi-compartment effects (myelin water, etc.).
4. **Idealised RF pulses.** Block pulses, no slice profile, no B1 inhomogeneity. At amp_T = 20 μT default, d180 ≈ 0.59 ms / d90 ≈ 0.29 ms — much shorter than `TI_min = 10 ms`, so pulse-overlap isn't binding, but slice profile effects are unmodelled.
5. **Sequential PE ordering.** F1+'s "average over Npe shots" assumes uniform PE encoding. Centric or other orderings change the per-shot weighting.
6. **One TI per block, shared across all 14 spheres.** No per-sphere targeting in the action space. Structural — only addressable via slice-selective excitation (`slice_z` axis exists but is unused) or a multi-action-per-block MDP.
7. **abs() in the signal magnitude → multimodal SSE.** Phase-sensitive reconstruction would resolve this but isn't in our env.
8. **Fitter assumes per-sphere noise scaling.** Replacing this with profile-likelihood σ (this plan) removes the assumption — but the LM init still picks the wrong basin on degenerate data; only σ widens.
9. **Phantom T1 distribution is heavily short-T1 weighted.** 4/14 spheres have T1 < 0.05 s — outside the action set's effective resolution. Reporting all-14 mean MAPE drags the headline 3× higher than the solvable subset.

---

## 8. Order of operations (what to do, in order)

### 8.1 Phase 1 — σ fix (1 day)

| Task | Effort | Output |
|---|---|---|
| Implement profile-likelihood σ in `fits.jl` | 1.5 h | `_profile_likelihood_sigma`, `sigma_method` kwarg |
| Three new tests (well-determined / multimodal / regression) | 1 h | green test suite |
| Wire through `_e2_update_t1_estimates!` | 15 min | env uses profile-likelihood σ |
| Re-eval V5 under new σ | 30 min compute | per-cell σ scatter, σ-calibration plot |
| Update §9.9.4 / §10 of EXPERT_REPORT with corrected σ-tail | 30 min | honest σ-calibration claim |

**Decision gate**: if profile-likelihood σ works (passes C1 + C2), proceed to Phase 2. If not, debug.

### 8.2 Phase 2 — CR-optimal baseline (1 day)

| Task | Effort | Output |
|---|---|---|
| Implement CR objective + optimiser (`src/baselines/cr_optimal.jl`) | 3 h | optimal schedule per `n_blocks` |
| Test on synthetic 2-sphere problem | 30 min | green test |
| Run CR-optimal schedule through env (30 eps) | 30 min compute | per-sphere MAPE comparison |
| Update §10 of EXPERT_REPORT with the CR-optimal row | 30 min | "V5 vs theoretical fixed-schedule optimum" |

**Decision gate**: if V5 ≥ CR-optimal, the C1 claim becomes "V5 matches theoretical fixed-schedule optimum" — strong. If V5 < CR-optimal, "V5 trails the fixed-schedule optimum by X %, suggesting RL has not yet captured all the structure available in the F1+ model" — also a clean (negative) claim.

### 8.3 Phase 3 — per-decade reporting + assumptions writeup (0.5 day)

| Task | Effort | Output |
|---|---|---|
| Add `--per-decade` flag to eval scripts | 30 min | `per_decade_summary.json` |
| Three-panel comparison plot in `plots_for_report.py` | 45 min | `per_decade_comparison.png` |
| Write §11 (per-decade tables) and §12 (assumptions) | 1.5 h | `EXPERT_REPORT_E2_4.md` complete |

### 8.4 Phase 4 — optional V8 retrain (~5 h compute, only if Phase 1 changes σ-channel meaningfully)

If V5 re-eval shows σ-channel obs distribution shifted significantly under profile-likelihood σ (e.g. mean σ-obs moves from −2.8 to −1.5 on short-T1 spheres), retrain a V8 with the same V5 settings (α=1.0 + delta_mape) under the corrected fitter. Hypothesis: V8 might revisit short-T1 spheres. Don't expect headline movement; this is exploratory.

If σ-channel obs distribution is roughly unchanged, skip V8 — the policy is observably indifferent to the σ-channel anyway (§9.7) and a retrain wouldn't move things.

---

## 9. What this experiment closes off (for the dissertation)

After E2.5 the C1 claim reads (proposed):

> **"V5 closes a 318 % → 234 % MAPE gap (1.36×) over a TR-matched fixed grid through learned per-sphere TI targeting concentrated on mid-T1 spheres (idx 5–10). It [matches/trails] the Cramér–Rao optimal fixed-block schedule by [X %], demonstrating that PPO recovers [most of / a portion of] the analytic optimum for the F1+ forward model without solving the optimisation problem analytically. The reported confidence intervals are profile-likelihood σ values that honestly span multimodal-SSE basins; the σ channel is well-calibrated per cell, not just on average. The all-14 mean MAPE of 234 % is dragged up by the short-T1 tail (T1 < 0.05 s), where the action set's TI ≥ 0.01 s combined with the 14-sphere joint estimation problem makes the fits structurally degenerate; on the solvable subset (T1 ≥ 0.05 s, 11/14 spheres), V5 achieves [Y %] mean MAPE."**

Each `[bracket]` is a number that comes out of E2.5. After this, Ch4 has:
- Forward-model fix (E2.4 §1) — methods contribution
- σ-channel honesty (E2.5 §3) — methods contribution
- Theoretical baseline (E2.5 §4) — anchor
- Per-decade reporting (E2.5 §5) — interpretation
- Documented assumptions (E2.5 §7) — limitations
- The single quantified C1 win above

That is a complete chapter. After E2.5 the work shifts to E3 (per-sphere targeting, slice-selective RF, EPG forward model) which is a different chapter.

---

## 10. Effort and timeline

| Phase | Hands-on | Compute |
|---|---:|---:|
| 1 — profile-likelihood σ | 4 h | 30 min |
| 2 — CR-optimal baseline | 4 h | 30 min |
| 3 — per-decade reporting + assumptions | 3 h | 0 |
| 4 (optional) — V8 retrain | 30 min | 5 h |
| **Total (without V8)** | **~11 h** | **1 h** |
| Total (with V8) | ~11.5 h | 6 h |

Fits inside one Wayne-update cycle. After E2.5, draft Ch4 from EXPERT_REPORT §§1–12.

---

## 11. Risks

| Risk | Likelihood | Diagnostic | Mitigation |
|---|---|---|---|
| **R1.** Profile-likelihood σ doesn't widen on multimodal cases (band stays narrow because grid is too coarse) | low | C1 test fails | Increase `n_grid` from 31 to 81, or refine grid around T1* with 2-stage zoom |
| **R2.** CR-optimal solver converges to a non-global minimum | medium | run sensitivity: 5 random restarts, check L-spread | Multi-start (already in §4.3); compare against the trivial log-grid as sanity floor |
| **R3.** CR-optimal schedule beats V5 by a lot (e.g. 100 %) | medium | new §10 row shows V5 trails by ≥ 50 pp | Honest negative result — write up as "RL has not yet matched theoretical fixed-schedule optimum"; suggests E3+ should target the gap |
| **R4.** Per-decade boundaries don't carve cleanly (e.g. a sphere on the boundary swings region between policies) | low | one sphere wanders idx → region mapping | Pin region by sphere idx (above), not by T1_est; report T1_true range per region for transparency |

R3 is the biggest report risk and also the most informative outcome: it means RL is leaving structure on the table, pointing E3 at a clear target.
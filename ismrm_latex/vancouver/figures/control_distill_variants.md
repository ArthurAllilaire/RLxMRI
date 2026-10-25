# Control: distilled protocols of the no-memory and fitter-uncertainty policies

Question: the no-memory policy scores 4.16% on the held-out phantoms against 2.93% for the TI-histogram policy.
Is its schedule shape worse, or does it lose accuracy by adapting each schedule too strongly?
Method: the validation-derived fixed median recipe of `control_summary.md`, applied to each policy: per block index, the median log TI and median TR of its `global_best` schedules on its 24 validation scans (seed 500000, `global_best/diagnostics/episodes.json` without the budget-rejected final proposals), truncated so that 32 x sum(TR) <= 560 s, then replayed with no policy in the loop.
Phantoms: the 24 held-out scans, seeds 600000-600023.
Every replay uses the TI-histogram run's E2Env (`runs/e2/mf_runB_5sphere_hist_560s_gpu/run_config.json`), so all arms share phantoms, pose, per-seed noise stream, full-Bloch forward model (CPU) and generalised-IR T1 fitter; only the (TI, TR) schedule differs.
Adaptive rows are each policy's own held-out evaluation (`global_best/eval_t1err_heldout.json`), paired per phantom.

## Checks

- **Env settings**: `base_env_kwargs` match the TI-histogram run in every forward-model setting (budget, Npe, noise, fidelity, spheres, T1 sampler, TE, flip angle, pose); the only differences are observation channels, which `julia/rl/e2.jl` reads only to build the observation:
  - No memory (`runs/e2/mf_runB_5sphere_560s_gpu`): `include_ti_history` null (TI histogram true), `ti_hist_bins` null (TI histogram 12).
  - Fitter uncertainty (`runs/e2/mf_runB_5sphere_sigma_560s_gpu`): `include_sigma` true (TI histogram false), `include_ti_history` null (TI histogram true), `ti_hist_bins` null (TI histogram 12).
  `null` means unset (E2Env default, off). The fitter computes its bootstrap sigma in every run with its own fixed seed, so `include_sigma` does not touch the noise stream.
- **Training config** (outside `base_env_kwargs`, no effect on the replay): `mf.mf_full_early_stop_delta`: TI histogram 0.1, No memory unset, Fitter uncertainty unset; `mf.mf_full_early_stop_min_evals`: TI histogram 5, No memory unset, Fitter uncertainty unset; `mf.mf_full_early_stop_min_steps`: TI histogram 0, No memory unset, Fitter uncertainty unset; `mf.mf_full_early_stop_patience`: TI histogram 0, No memory unset, Fitter uncertainty unset.
- **Sanity check**: rerunning the TI-histogram run's validation-derived protocol reproduces `control_results.json` (`arms.fixed_median_validation`) exactly: protocol identical, every phantom's T1 estimates and MAPE identical (max |dT1_est| 0.0e+00 s, max |dMAPE| 0.0e+00).
- **Pairing**: each run's `eval_t1err_heldout.json` was scored on the replayed phantoms (max |dT1_true| 0.0e+00 s for all three), and the bootstrap CIs of its per-scan MAPEs match the recorded `agent_mape_ci95_pct`.

## Results (24 held-out phantoms)

| Policy | Arm | Mean MAPE % [95% CI] | Scans < 5% | Worst scan MAPE % | Blocks | Scan time (s) |
|---|---|---|---|---|---|---|
| TI histogram | Adaptive (policy in the loop) | 2.93 [2.58, 3.30] | 95.8% (23/24) | 5.13 | - | - |
| TI histogram | Distilled (fixed, validation-derived) | 3.34 [2.30, 4.86] | 91.7% (22/24) | 17.87 | 13 | 549.2 |
| No memory | Adaptive (policy in the loop) | 4.16 [3.41, 4.99] | 66.7% (16/24) | 9.50 | - | - |
| No memory | Distilled (fixed, validation-derived) | 3.63 [3.13, 4.16] | 87.5% (21/24) | 7.26 | 15 | 549.5 |
| Fitter uncertainty | Adaptive (policy in the loop) | 5.25 [4.32, 6.22] | 58.3% (14/24) | 10.40 | - | - |
| Fitter uncertainty | Distilled (fixed, validation-derived) | 9.38 [7.44, 11.57] | 16.7% (4/24) | 22.56 | 16 | 541.4 |
| Reference | Fixed log TI grid, TR 1.7 s | 6.04 [5.36, 6.70] | 25.0% (6/24) | - | 9 | 503.6 |
| Reference | Cramer-Rao optimal | 7.74 [7.08, 8.42] | 8.3% (2/24) | - | 10 | 557.5 |

MAPE: mean over the five spheres of |T1 estimate - T1 true| / T1 true, averaged over scans.
CI: percentile bootstrap, 5000 resamples over the 24 phantoms, numpy `default_rng(0)` (as `python/baseline_e2.py:41-58`).
`eval_e2.py` saved no per-scan block counts, so adaptive rows have none; each distilled protocol runs all its blocks on every phantom.
Reference rows are the held-out baseline run of `control_summary.md`; per-scan values were not saved, so they have no worst scan.

## Paired differences

| Comparison | Mean difference [95% CI] | Median per-phantom difference (pp) | Phantoms where the first is worse | Without phantoms 7 and 17 (post hoc) |
|---|---|---|---|---|
| TI histogram distilled - TI histogram adaptive | +0.41 pp [-0.69, +1.92] | -0.26 | 12/24 | -0.43 pp [-0.97, +0.12] |
| No memory distilled - No memory adaptive | -0.53 pp [-1.45, +0.37] | -0.56 | 10/24 | -0.57 pp [-1.54, +0.39] |
| Fitter uncertainty distilled - Fitter uncertainty adaptive | +4.13 pp [+2.08, +6.38] | +2.79 | 19/24 | +4.51 pp [+2.34, +6.97] |
| No memory adaptive - TI histogram adaptive | +1.23 pp [+0.43, +2.10] | +1.20 | 15/24 | +1.18 pp [+0.28, +2.14] |
| No memory distilled - TI histogram distilled | +0.29 pp [-1.36, +1.43] | +0.62 | 16/24 | +1.04 pp [+0.42, +1.65] |
| No memory distilled - TI histogram adaptive | +0.70 pp [-0.01, +1.39] | +1.23 | 17/24 | +0.61 pp [-0.09, +1.23] |
| Fitter uncertainty adaptive - TI histogram adaptive | +2.32 pp [+1.28, +3.41] | +1.71 | 18/24 | +2.11 pp [+1.11, +3.22] |
| Fitter uncertainty distilled - TI histogram distilled | +6.03 pp [+3.50, +8.61] | +6.18 | 22/24 | +7.04 pp [+4.86, +9.49] |
| Fitter uncertainty distilled - TI histogram adaptive | +6.45 pp [+4.44, +8.76] | +5.03 | 23/24 | +6.62 pp [+4.37, +9.11] |

Paired bootstrap over the phantoms (5000 resamples, `default_rng(0)`).
The last column drops the 2 phantoms with the shortest sphere T1, as in `control_summary.md`.

## Short-T1 phantoms (post hoc)

The distilled protocols' shortest TIs are 243 ms (TI histogram), 175 ms (No memory), 167 ms (Fitter uncertainty).

| Phantom | Shortest T1 (ms) | Shortest-sphere APE %, TI histogram: adaptive / distilled | Shortest-sphere APE %, No memory: adaptive / distilled | Shortest-sphere APE %, Fitter uncertainty: adaptive / distilled | Scan MAPE %, TI histogram: adaptive / distilled | Scan MAPE %, No memory: adaptive / distilled | Scan MAPE %, Fitter uncertainty: adaptive / distilled |
|---|---|---|---|---|---|---|---|
| 7 | 41 | 0.3 / 32.8 | 5.6 / 18.4 | 8.7 / 5.8 | 2.85 / 7.66 | 5.28 / 7.26 | 10.40 / 7.10 |
| 17 | 46 | 1.2 / 74.3 | 10.8 / 4.6 | 7.7 / 6.7 | 3.40 / 17.87 | 4.47 / 2.39 | 5.13 / 8.28 |

## Interpretation

Decision rule: a distilled no-memory protocol near the TI-histogram run's distilled 3.34% would mean both policies found an equally good schedule and no-memory loses accuracy through its adaptation; near its own adaptive 4.16% or worse would mean its schedule shape is worse.
Result: 3.63% [3.13, 4.16], in between (36% of the way from 3.34% to 4.16%), and the intervals are too wide to call it either way at n = 24 with one training seed per run.
Paired, it is +0.29 pp [-1.36, +1.43] against the TI-histogram distilled protocol and -0.53 pp [-1.45, +0.37] against the no-memory adaptive policy; both intervals include zero.
The paired gap between the two adaptive policies, +1.23 pp [+0.43, +2.10], splits exactly into three paired terms: the no-memory policy's adaptation cost (its adaptive - distilled) +0.53 pp [-0.37, +1.45], the shape difference (no-memory distilled - TI-histogram distilled) +0.29 pp [-1.36, +1.43], and the TI-histogram policy's adaptation gain (its distilled - adaptive) +0.41 pp [-0.69, +1.92].
On all 24 phantoms the point estimates put most of the gap on adaptation and little on shape, but no term is resolved, and the shape term is held down by the TI-histogram protocol's failures on phantoms 7 and 17.
Without those 2 phantoms (post hoc), the terms become +0.57 pp [-0.39, +1.54], +1.04 pp [+0.42, +1.65] and -0.43 pp [-0.97, +0.12]: on typical phantoms the no-memory schedule shape is worse by about 1 pp (3.52% against 2.49%), the only resolved term.
Removing its adaptation does make the no-memory policy more reliable: scans below 5% rise from 16/24 to 21/24, the worst scan falls from 9.50% to 7.26%, and the distilled protocol is better on 14/24 phantoms.
Its shorter distilled TIs (down to 175 ms against 243 ms) also make it more robust than the TI-histogram distilled protocol on phantoms 7 and 17 (shortest-sphere APE 18.4% and 4.6% against 32.8% and 74.3%).
Under the decision rule the answer is in between: the no-memory policy loses some accuracy through over-adaptation (about half a point, not significant) and, on typical phantoms, its schedule shape is also worse (about 1 pp, post hoc); this control cannot assign the gap to one cause.
The fitter-uncertainty policy does not distil: its fixed protocol reaches 9.38% [7.44, 11.57], +4.13 pp [+2.08, +6.38] against its own adaptive policy (worse on 19/24 phantoms) and above the fixed log TI grid (6.04%).
Its mean APE by sphere T1 rank, shortest to longest, adaptive / distilled: 5.0 / 8.1%, 5.8 / 7.0%, 4.9 / 14.4%, 5.0 / 10.7%, 5.6 / 6.6%.
Either this policy relies on per-scan adaptation, or per-block medians do not summarise its schedules: 9 of its 16 median blocks sit at the 0.5 s TR floor, and its last three blocks are medians over only 12, 9 and 6 of the 24 validation scans; this control cannot tell which.
For the abstract: the schedule-shape claim holds for the TI-histogram policy and is consistent with the no-memory policy (its distilled protocol, 3.63%, also stays far below the fixed log TI grid and the Cramer-Rao schedule), but do not attribute the no-memory deficit to over-adaptation alone, and do not extend the claim to the fitter-uncertainty policy.
Caveats: 24 phantoms and one training seed per run; the cut of phantoms 7 and 17 was chosen in `control_summary.md` after seeing results; per-block medians are one crude distillation among many; the no-memory and fitter-uncertainty adaptive references are their recorded GPU evaluations, not local rollouts (for the TI-histogram run the local CPU rollout reproduced its GPU evaluation exactly).

## Distilled protocols

| Block | TI histogram TI (s) | TR (s) | Scans reaching block | No memory TI (s) | TR (s) | Scans reaching block | Fitter uncertainty TI (s) | TR (s) | Scans reaching block |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1.0621 | 2.3051 | 24 | 0.6332 | 3.0463 | 24 | 0.6278 | 2.8501 | 24 |
| 2 | 0.4948 | 1.8687 | 24 | 0.3072 | 2.1188 | 24 | 0.8416 | 3.3185 | 24 |
| 3 | 0.3966 | 1.8806 | 24 | 0.7517 | 2.6182 | 24 | 0.5188 | 3.0139 | 24 |
| 4 | 0.6372 | 2.1856 | 24 | 0.6667 | 1.5412 | 24 | 0.5307 | 1.3991 | 24 |
| 5 | 0.4823 | 1.7774 | 24 | 0.5293 | 1.3718 | 24 | 0.3873 | 0.5682 | 24 |
| 6 | 0.3456 | 1.4737 | 24 | 0.3931 | 0.9813 | 24 | 0.2498 | 0.5000 | 24 |
| 7 | 0.2672 | 0.8798 | 24 | 0.3125 | 0.5670 | 24 | 0.2341 | 0.5000 | 24 |
| 8 | 0.2470 | 0.6642 | 24 | 0.2604 | 0.5230 | 24 | 0.2189 | 0.5000 | 24 |
| 9 | 0.2578 | 0.6647 | 24 | 0.2297 | 0.5000 | 21 | 0.2391 | 0.5000 | 24 |
| 10 | 0.2643 | 0.7492 | 24 | 0.2023 | 0.5000 | 20 | 0.2326 | 0.5000 | 24 |
| 11 | 0.2432 | 0.8684 | 24 | 0.1915 | 0.5058 | 20 | 0.2196 | 0.5000 | 20 |
| 12 | 0.2773 | 0.9186 | 20 | 0.1914 | 0.6230 | 20 | 0.1959 | 0.5854 | 19 |
| 13 | 0.2598 | 0.9275 | 14 | 0.1833 | 0.6877 | 15 | 0.1902 | 0.6836 | 16 |
| 14 |  | |  | 0.1795 | 0.7580 | 14 | 0.1806 | 0.5000 | 12 |
| 15 |  | |  | 0.1754 | 0.8306 | 11 | 0.1766 | 0.5000 | 9 |
| 16 |  | |  |  | |  | 0.1671 | 0.5000 | 6 |
| Total | 13 blocks, 549.2 s | |  | 15 blocks, 549.5 s | |  | 16 blocks, 541.4 s | |  |

## Runtime

Local CPU, Julia 1.12.6, 8 threads: 3.8 min for 1056 full-Bloch blocks (0.22 s per block including the fit).

## Regenerate

From the RLxMRI repo root (needs `figures/control_data/control_results.json` from `control_schedule_swap.jl` for the sanity check):

```bash
julia -t 8 --project=. ismrm_latex/vancouver/scripts/control_distill_variants.jl
.venv/bin/python ismrm_latex/vancouver/scripts/control_distill_summary.py
```

Outputs: `figures/control_data/control_distill_variants.json` (protocols, per-phantom results and adaptive references of every policy) and this file.

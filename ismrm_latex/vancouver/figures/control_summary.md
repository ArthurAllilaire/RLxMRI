# Control: adaptation versus schedule shape

Question: does the TI-histogram policy's held-out T1 accuracy come from adapting the schedule to each scan, or only from the shape of its schedules (short TIs paired with short TRs, about 13 blocks in 560 s)?
Policy: `runs/e2/mf_runB_5sphere_hist_560s_gpu`, checkpoint `global_best`.
Phantoms: the 24 held-out scans, seeds 600000-600023, as in `global_best/eval_t1err_heldout.json`.
Every arm runs on the same phantoms (same T1s and pose) with the same per-seed noise stream, the same E2Env forward model (full Bloch, CPU) and the same generalised-IR T1 fitter; only the (TI, TR) schedule differs.

## Step 1: reproduction of the held-out evaluation

The pure-Julia rollout of `global_best` on seeds 600000-600023 reproduces `eval_t1err_heldout.json` exactly.
Largest per-sphere APE difference: 0.0e+00; largest true-T1 difference: 0.0e+00 s.
Mean MAPE 2.9310% (recorded 2.9310%), and the bootstrap CI below matches `eval_summary.json`.
Replaying each scan's own executed schedule without the policy (identity replay) reproduces every scan's final T1 estimates bit for bit, so the replay path is the same pipeline the policy was scored on.

## Arms

- **Adaptive**: each phantom gets the schedule the policy chose for it online.
- **Swapped**: phantom k gets the executed schedule of scan (k + s) mod 24, for s = 1, 8, 16 (72 runs, no policy in the loop).
  Each shift is a derangement, so every schedule is used once per shift and never on its own phantom; block counts and scan times are therefore the same as Adaptive.
- **Fixed median protocol**: per block index, the median log TI and median TR over the 24 adaptive held-out schedules (over the scans that reached that block), truncated so that 32 x sum(TR) <= 560 s; the same schedule on all 24 phantoms.
- **Fixed median protocol, validation-derived** (added robustness check): the same recipe applied to the policy's executed schedules on the 24 validation scans (seed 500000, `global_best/diagnostics/episodes.json` without the budget-rejected final proposals), so no held-out information shapes the protocol.
- **References**: the held-out baseline run (`runs/e2/baselines_heldout_560s/baseline_summary.json`); per-scan values were not saved, so they have no worst scan or paired difference.

## Results (24 held-out phantoms)

| Arm | Schedule | Mean MAPE % [95% CI] | Scans < 5% | Worst scan MAPE % | Mean blocks | Mean scan time (s) |
|---|---|---|---|---|---|---|
| Adaptive (policy in the loop) | own, chosen online | 2.93 [2.58, 3.30] | 95.8% (23/24) | 5.13 | 12.54 | 541 |
| Swapped, shift 1 | scan (k + 1) mod 24 | 3.59 [2.86, 4.44] | 83.3% (20/24) | 10.37 | 12.54 | 541 |
| Swapped, shift 8 | scan (k + 8) mod 24 | 3.67 [2.88, 4.70] | 83.3% (20/24) | 13.43 | 12.54 | 541 |
| Swapped, shift 16 | scan (k + 16) mod 24 | 3.56 [2.84, 4.40] | 87.5% (21/24) | 10.66 | 12.54 | 541 |
| Swapped, mean of 3 shifts | another scan's | 3.61 [3.01, 4.45] | 84.7% (61/72) | 13.43 | 12.54 | 541 |
| Fixed median protocol | medians of held-out schedules | 2.99 [2.50, 3.59] | 95.8% (23/24) | 8.49 | 13.00 | 550 |
| Fixed median protocol, validation-derived | medians of validation schedules | 3.34 [2.30, 4.86] | 91.7% (22/24) | 17.87 | 13.00 | 549 |
| Fixed log TI grid, TR 1.7 s (reference) | fixed | 6.04 [5.36, 6.70] | 25.0% (6/24) | - | 9.00 | 504 |
| Cramer-Rao optimal (reference) | fixed | 7.74 [7.08, 8.42] | 8.3% (2/24) | - | 10.00 | 558 |

MAPE: mean over the five spheres of |T1 estimate - T1 true| / T1 true, averaged over scans.
CI: percentile bootstrap, 5000 resamples over the 24 phantoms, numpy `default_rng(0)` (as `python/baseline_e2.py:41-58`).
For the swap mean, each phantom's 3 swapped MAPEs are averaged before resampling; its success share and worst scan are over all 72 runs.

## Paired differences to Adaptive

| Arm | Mean difference, arm - Adaptive [95% CI] | Median per-phantom difference (pp) | Phantoms where the arm is worse | Without phantoms 7 and 17 (post hoc) |
|---|---|---|---|---|
| Swapped, shift 1 | +0.66 pp [-0.15, +1.58] | +0.12 | 12/24 | +0.20 pp [-0.41, +0.90] |
| Swapped, shift 8 | +0.74 pp [-0.11, +1.82] | +0.46 | 15/24 | +0.27 pp [-0.36, +0.93] |
| Swapped, shift 16 | +0.63 pp [-0.13, +1.49] | +0.40 | 13/24 | +0.35 pp [-0.27, +0.93] |
| Swapped, mean of 3 shifts | +0.67 pp [-0.01, +1.55] | +0.38 | 16/24 | +0.28 pp [-0.21, +0.73] |
| Fixed median protocol | +0.06 pp [-0.43, +0.67] | -0.03 | 12/24 | -0.22 pp [-0.57, +0.11] |
| Fixed median protocol, validation-derived | +0.41 pp [-0.69, +1.92] | -0.26 | 12/24 | -0.43 pp [-0.97, +0.12] |

Paired bootstrap over the phantoms (5000 resamples, `default_rng(0)`); for the swap mean, the 3 shifts are averaged per phantom first.
The last column drops the 2 phantoms with the shortest sphere T1 (chosen after seeing the results; see below).

## Where adaptation does work (post hoc)

The 5 phantoms with the shortest sphere T1; APE is that sphere's error, MAPE the scan's.
Across all 24 held-out scans, the policy's shortest TI tracks the phantom's shortest T1 (Pearson r = +0.57 on log scales).
The fixed median protocols' shortest TIs are 234 ms (held-out-derived) and 243 ms (validation-derived).

| Phantom | Shortest T1 (s) | Policy's shortest TI (s) | Shortest-sphere APE %: adaptive / swapped mean / fixed median / fixed median (val.) | Scan MAPE %: adaptive / swapped mean / fixed median / fixed median (val.) |
|---|---|---|---|---|
| 7 | 0.041 | 0.113 | 0.3 / 46.2 / 36.7 / 32.8 | 2.85 / 11.49 / 8.49 / 7.66 |
| 17 | 0.046 | 0.132 | 1.2 / 14.3 / 4.0 / 74.3 | 3.40 / 4.89 / 4.13 / 17.87 |
| 13 | 0.055 | 0.184 | 3.6 / 5.2 / 0.7 / 2.2 | 1.92 / 3.50 / 2.63 / 2.95 |
| 8 | 0.096 | 0.209 | 6.0 / 2.0 / 3.0 / 0.1 | 2.98 / 2.84 / 2.26 / 1.38 |
| 15 | 0.098 | 0.207 | 7.5 / 2.4 / 1.4 / 1.4 | 4.58 / 1.75 / 2.45 / 1.39 |

## Interpretation

The mean accuracy gain over the fixed comparators comes mostly from the shape of the policy's schedules, not from tailoring each schedule to its scan.
A single fixed protocol built from the per-block medians of the policy's held-out schedules reaches 2.99% [2.50, 3.59] against 2.93% [2.58, 3.30] for the adaptive policy (paired +0.06 pp [-0.43, +0.67]); built only from validation scans it reaches 3.34% [2.30, 4.86] (paired +0.41 pp [-0.69, +1.92]).
Both are well below the fixed log TI grid (6.04%) and the Cramer-Rao schedule (7.74%), and neither mean differs significantly from the adaptive policy at n = 24.
Giving each phantom another scan's schedule costs +0.67 pp [-0.01, +1.55] (3.61% [3.01, 4.45], median per phantom +0.38 pp): a borderline penalty, 53% of it from phantom 7.
Per-scan adaptation shows in the tail, not the mean: phantoms 7 and 17 have spheres of 41 and 46 ms, the policy placed TIs down to 113 and 132 ms and measured those spheres to 0.3% and 1.2%, while the fixed median protocols (held-out-derived / validation-derived) got 36.7% / 32.8% on phantom 7's and 4.0% / 74.3% on phantom 17's.
As a result, the worst scan is 5.13% for the adaptive policy against 8.49% and 17.87% for the fixed median protocols and 13.43% for swapped schedules.
Without those 2 phantoms (a post hoc cut), the fixed median protocols match or slightly beat the adaptive policy (-0.22 pp [-0.57, +0.11] and -0.43 pp [-0.97, +0.12]).
With only 2 such phantoms, this robustness effect is an observation, not a measured rate.
For the abstract: claim that RL found a schedule shape (short TIs paired with short TRs, about 13 blocks in 560 s) that cuts the error of the best fixed comparator from 6.04% to 2.93%, and that a fixed protocol distilled from it keeps most of that gain; claim per-scan adaptation only as a robustness effect seen on rare very short T1s, not as the source of the mean gain.

## Fixed median protocols

| Block | Held-out-derived TI (s) | TR (s) | Scans reaching block | Validation-derived TI (s) | TR (s) | Scans reaching block |
|---|---|---|---|---|---|---|
| 1 | 1.0621 | 2.3051 | 24 | 1.0621 | 2.3051 | 24 |
| 2 | 0.4948 | 1.8687 | 24 | 0.4948 | 1.8687 | 24 |
| 3 | 0.4283 | 1.5062 | 24 | 0.3966 | 1.8806 | 24 |
| 4 | 0.5262 | 2.0271 | 24 | 0.6372 | 2.1856 | 24 |
| 5 | 0.4258 | 1.7319 | 24 | 0.4823 | 1.7774 | 24 |
| 6 | 0.3342 | 1.5233 | 24 | 0.3456 | 1.4737 | 24 |
| 7 | 0.2660 | 1.0518 | 24 | 0.2672 | 0.8798 | 24 |
| 8 | 0.2341 | 0.7697 | 23 | 0.2470 | 0.6642 | 24 |
| 9 | 0.2583 | 0.7284 | 23 | 0.2578 | 0.6647 | 24 |
| 10 | 0.2704 | 0.8210 | 22 | 0.2643 | 0.7492 | 24 |
| 11 | 0.2590 | 0.8918 | 20 | 0.2432 | 0.8684 | 24 |
| 12 | 0.2464 | 0.9273 | 16 | 0.2773 | 0.9186 | 20 |
| 13 | 0.2620 | 1.0255 | 13 | 0.2598 | 0.9275 | 14 |
| Total | 13 blocks, 549.7 s | | | 13 blocks, 549.2 s | | |

## Runtime

Local CPU, Julia 1.12.6, 8 threads: 10.4 min for 2129 full-Bloch blocks (0.29 s per block including the fit).

| Arm | Simulated blocks | Wall time (s) |
|---|---|---|
| adaptive | 301 | 74 |
| identity | 301 | 71 |
| swap_1 | 301 | 83 |
| swap_8 | 301 | 98 |
| swap_16 | 301 | 98 |
| fixed_median | 312 | 100 |
| fixed_median_validation | 312 | 98 |

## Regenerate

From the RLxMRI repo root (the policy arrays come from `fig3evo_export_policies.py`, already in `figures/`):

```bash
julia -t 8 --project=. ismrm_latex/vancouver/scripts/control_schedule_swap.jl
.venv/bin/python ismrm_latex/vancouver/scripts/control_summary.py
```

Outputs: `figures/control_data/control_adaptive_heldout.json` (full adaptive rollouts), `figures/control_data/control_results.json` (per-phantom results of every arm), and this file.

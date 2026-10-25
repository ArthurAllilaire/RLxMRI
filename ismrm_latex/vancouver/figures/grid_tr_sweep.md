# Control: TR of the fixed log TI grid comparator

Question: the abstract's main fixed comparator, a log-spaced TI grid at TR 1.7 s (`log_grid_trmatched`), took its TR from an earlier agent's average rather than from a sweep. Is 1.7 s close to the best TR for this grid, or does another TR give a clearly lower MAPE?
Phantoms: the 24 held-out scans, seeds 600000-600023, with the env config of `runs/e2/mf_runB_5sphere_hist_560s_gpu` (`run_config.json` -> `base_env_kwargs`), as the recorded comparator used.
Every TR runs on the same phantoms (same T1s and pose) with the same per-seed noise stream, the same E2Env forward model (full Bloch, CPU), the same generalised-IR T1 fitter and the same Python action round trip (numpy2 promotion rules) as the recorded comparators; only the grid's TR and TIs change.
Budget 560 s, 32 phase-encode shots per block, TE 20 ms, flip angle 90 deg, at most 20 blocks.

## Step 1: reproduction of the recorded comparator

The as-built grid at TR 1.7 s reproduces the recorded per-scan results (`ismrm_latex/vancouver/figures/fig5_v2_data/fixed_comparators_heldout.json`) bit for bit on all 24 scans: T1 true, T1 estimates, MAPE, block count, scan time and executed (TI, TR) blocks.
Its mean MAPE 6.0367% and bootstrap CI [5.3622, 6.6986] match `runs/e2/baselines_heldout_560s/baseline_summary.json` to 1e-9.
The RL references below are read from each policy's `global_best/eval_t1err_heldout.json`; their phantoms' true T1s equal the sweep's exactly and their recomputed CIs match the recorded ones, so the paired differences are over the same phantoms.

## TRs and feasibility rule

Grid TRs: 0.8, 1, 1.2, 1.35, 1.5, 1.7, 2, 2.5, 3, 4 s.
Policy-matched TRs, from the TI-histogram policy's executed TRs on its 24 validation scans (`global_best/diagnostics/episodes.json` without each episode's budget-rejected final proposal, 316 blocks), so no held-out information shapes a comparator: block-weighted mean 1.2950 s (policy mean TR, equalises block count with the policy) and median 1.0772 s (policy median TR).
For reference only, the same statistics on the held-out schedules are 1.3487 s and 1.1145 s.

`make_log_grid_trmatched` (`python/baseline_e2.py:86-101`) builds floor(560 / (32 x TR)) blocks at one TR, with TIs log-spaced over 0.05-3 s whatever the TR; it does not cap TI by TR.
The env lifts a block's TR to (TI + TE) / 0.9 when TI + TE exceeds 0.9 TR, and its budget guard rejects the first block that would overrun 560 s, which ends the episode.

- **As built** (primary): the construction exactly as the reported comparator ran, lifts and rejections included; the TR 1.7 s row is the abstract's comparator. Lifted blocks, grid blocks never run and the reason each episode ended are listed below.
- **TI capped** (secondary): top TI = min(3 s, 0.9 TR - TE - 1 ms) and at most 20 blocks, so every block runs at the nominal TR and every grid TI is executed (the script stops otherwise). The 1 ms keeps the top TI clear of the env's lift test after the float32 action round trip. The block cap binds at TR 0.8 s (21 blocks fit the budget).

## Results, as built (primary)

| TR (s) | TI range run (s) | Blocks run of grid | Scan time (s) | Mean MAPE % [95% CI] | Scans < 5% | Worst scan % | Paired vs as-built TR 1.7 s (pp) [95% CI] | Paired vs RL 240 s (pp) [95% CI] |
|---|---|---|---|---|---|---|---|---|
| 0.8 | 0.050-1.623 | 18 of 21 (5 lifted, 3 not run) | 536 | 17.69 [15.18, 20.22] | 0/24 | 29.82 | +11.65 pp [+9.12, +14.23] | +13.07 pp [+10.65, +15.53] |
| 1 | 0.050-1.798 | 15 of 17 (3 lifted, 2 not run) | 538 | 12.45 [10.66, 14.26] | 0/24 | 21.56 | +6.41 pp [+4.62, +8.19] | +7.83 pp [+6.13, +9.51] |
| 1.077 (policy median TR) | 0.050-1.738 | 14 of 16 (3 lifted, 2 not run) | 526 | 11.99 [10.06, 13.94] | 3/24 | 22.75 | +5.95 pp [+4.09, +7.87] | +7.36 pp [+5.53, +9.22] |
| 1.2 | 0.050-1.598 | 12 of 14 (2 lifted, 2 not run) | 484 | 9.34 [8.09, 10.59] | 1/24 | 15.47 | +3.31 pp [+2.13, +4.48] | +4.72 pp [+3.62, +5.84] |
| 1.295 (policy mean TR) | 0.050-2.133 | 12 of 13 (2 lifted, 1 not run) | 546 | 9.10 [7.73, 10.49] | 3/24 | 16.46 | +3.06 pp [+1.79, +4.31] | +4.47 pp [+3.26, +5.69] |
| 1.35 | 0.050-2.068 | 11 of 12 (2 lifted, 1 not run) | 514 | 7.51 [6.45, 8.56] | 3/24 | 13.16 | +1.47 pp [+0.52, +2.39] | +2.88 pp [+1.98, +3.78] |
| 1.5 | 0.050-1.992 | 10 of 11 (1 lifted, 1 not run) | 504 | 7.31 [6.49, 8.10] | 2/24 | 12.16 | +1.27 pp [+0.38, +2.20] | +2.68 pp [+1.90, +3.49] |
| 1.7 | 0.050-1.903 | 9 of 10 (1 lifted, 1 not run) | 504 | 6.04 [5.36, 6.70] | 6/24 | 10.29 | reference | +1.41 pp [+0.80, +2.06] |
| 2 | 0.050-3.000 | 8 of 8 (1 lifted) | 555 | 5.83 [4.96, 6.73] | 11/24 | 11.96 | -0.21 pp [-0.99, +0.62] | +1.21 pp [+0.24, +2.26] |
| **2.5** | 0.050-1.516 | 6 of 7 (1 not run) | 480 | **5.60 [4.92, 6.32]** | 8/24 | 9.59 | -0.44 pp [-1.48, +0.53] | +0.97 pp [+0.07, +1.91] |
| 3 | 0.050-3.000 | 5 of 5 (1 lifted) | 491 | 7.41 [6.15, 8.82] | 7/24 | 16.05 | +1.37 pp [-0.01, +2.81] | +2.78 pp [+1.27, +4.39] |
| 4 | 0.050-3.000 | 4 of 4 | 512 | 11.81 [8.84, 15.39] | 4/24 | 40.53 | +5.78 pp [+2.76, +9.38] | +7.19 pp [+4.12, +10.81] |

## Results, TI capped (secondary)

| TR (s) | TI range run (s) | Blocks run of grid | Scan time (s) | Mean MAPE % [95% CI] | Scans < 5% | Worst scan % | Paired vs as-built TR 1.7 s (pp) [95% CI] | Paired vs RL 240 s (pp) [95% CI] |
|---|---|---|---|---|---|---|---|---|
| 0.8 | 0.050-0.699 | 20 of 20 (max_blocks) | 512 | 18.22 [15.38, 21.06] | 0/24 | 31.61 | +12.18 pp [+9.33, +15.03] | +13.59 pp [+10.85, +16.37] |
| 1 | 0.050-0.879 | 17 of 17 | 544 | 12.67 [10.60, 14.74] | 1/24 | 24.41 | +6.63 pp [+4.59, +8.67] | +8.04 pp [+6.10, +10.04] |
| 1.077 (policy median TR) | 0.050-0.948 | 16 of 16 | 552 | 11.13 [9.51, 12.79] | 1/24 | 19.85 | +5.10 pp [+3.48, +6.76] | +6.51 pp [+5.00, +8.09] |
| 1.2 | 0.050-1.059 | 14 of 14 | 538 | 9.28 [7.91, 10.71] | 1/24 | 17.35 | +3.24 pp [+1.93, +4.56] | +4.65 pp [+3.34, +5.98] |
| 1.295 (policy mean TR) | 0.050-1.144 | 13 of 13 | 539 | 8.52 [7.13, 9.95] | 4/24 | 17.21 | +2.48 pp [+1.15, +3.84] | +3.89 pp [+2.59, +5.19] |
| 1.35 | 0.050-1.194 | 12 of 12 | 518 | 7.33 [6.24, 8.45] | 5/24 | 13.65 | +1.29 pp [+0.28, +2.27] | +2.70 pp [+1.71, +3.68] |
| 1.5 | 0.050-1.329 | 11 of 11 | 528 | 6.49 [5.65, 7.35] | 6/24 | 11.21 | +0.46 pp [-0.24, +1.11] | +1.87 pp [+1.19, +2.54] |
| 1.7 | 0.050-1.509 | 10 of 10 | 544 | 5.82 [5.12, 6.49] | 7/24 | 9.79 | -0.22 pp [-0.65, +0.18] | +1.19 pp [+0.63, +1.81] |
| **2** | 0.050-1.779 | 8 of 8 | 512 | **5.12 [4.52, 5.71]** | 11/24 | 8.67 | -0.91 pp [-1.47, -0.33] | +0.50 pp [-0.19, +1.19] |
| 2.5 | 0.050-2.229 | 7 of 7 | 560 | 6.00 [5.02, 7.03] | 10/24 | 12.67 | -0.04 pp [-1.24, +1.10] | +1.38 pp [+0.14, +2.60] |
| 3 | 0.050-2.679 | 5 of 5 | 480 | 7.45 [6.30, 8.69] | 8/24 | 15.40 | +1.41 pp [+0.21, +2.71] | +2.82 pp [+1.45, +4.29] |
| 4 | 0.050-3.000 | 4 of 4 | 512 | 11.81 [8.84, 15.39] | 4/24 | 40.53 | +5.78 pp [+2.76, +9.38] | +7.19 pp [+4.12, +10.81] |

RL references on the same phantoms: TI-histogram policy, 560 s (`mf_runB_5sphere_hist_560s_gpu`) 2.93 [2.58, 3.30]%, 23/24 scans < 5%, worst 5.13%; RL policy, 240 s (`mf_runB_cached3_cached_full3_full_gpu`) 4.62 [4.06, 5.19]%, 15/24 scans < 5%, worst 7.74%.
MAPE: mean over the five spheres of |T1 estimate - T1 true| / T1 true, averaged over scans. Bold: the lowest mean MAPE of the series.
CI: percentile bootstrap, 5000 resamples over the 24 phantoms, numpy `default_rng(0)` (as `python/baseline_e2.py:41-58`). Paired: grid minus the other arm per phantom, bootstrapped the same way; negative means the grid row is more accurate.
Blocks run of grid: executed blocks out of the construction's TIs; lifted blocks ran at a longer TR than the grid's, and blocks not run did not fit the remaining budget.
Both series run the same schedule at TR 4 s, where the cap does not bind.

## Lifted and rejected blocks

Every scan executes the same blocks (lifts and the budget guard depend only on the timings; the script checks it).

As built (primary):

| TR (s) | Grid TIs | Run | Lifted blocks (TR requested -> executed) | Grid blocks not run | Episode ended by | Scan time (s) |
|---|---|---|---|---|---|---|
| 0.8 | 21 | 18 | block 14 TI 0.716 s, TR 0.800 -> 0.818 s; block 15 TI 0.878 s, TR 0.800 -> 0.998 s; block 16 TI 1.078 s, TR 0.800 -> 1.220 s; block 17 TI 1.323 s, TR 0.800 -> 1.492 s; block 18 TI 1.623 s, TR 0.800 -> 1.826 s | 3 | budget guard rejected schedule block 19 (TI 1.992 s, needs TR 2.236 s) | 536.1 |
| 1 | 17 | 15 | block 13 TI 1.078 s, TR 1.000 -> 1.220 s; block 14 TI 1.392 s, TR 1.000 -> 1.569 s; block 15 TI 1.798 s, TR 1.000 -> 2.020 s | 2 | budget guard rejected schedule block 16 (TI 2.323 s, needs TR 2.603 s) | 537.9 |
| 1.077 (policy median TR) | 16 | 14 | block 12 TI 1.007 s, TR 1.077 -> 1.141 s; block 13 TI 1.323 s, TR 1.077 -> 1.492 s; block 14 TI 1.738 s, TR 1.077 -> 1.953 s | 2 | budget guard rejected schedule block 15 (TI 2.283 s, needs TR 2.559 s) | 525.9 |
| 1.2 | 14 | 12 | block 11 TI 1.166 s, TR 1.200 -> 1.318 s; block 12 TI 1.598 s, TR 1.200 -> 1.798 s | 2 | budget guard rejected schedule block 13 (TI 2.189 s, needs TR 2.455 s) | 483.7 |
| 1.295 (policy mean TR) | 13 | 12 | block 11 TI 1.516 s, TR 1.295 -> 1.707 s; block 12 TI 2.133 s, TR 1.295 -> 2.392 s | 1 | no time left for a block at the TR floor | 545.6 |
| 1.35 | 12 | 11 | block 10 TI 1.425 s, TR 1.350 -> 1.606 s; block 11 TI 2.068 s, TR 1.350 -> 2.320 s | 1 | budget guard rejected schedule block 12 (TI 3.000 s, needs TR 3.356 s) | 514.4 |
| 1.5 | 11 | 10 | block 10 TI 1.992 s, TR 1.500 -> 2.236 s | 1 | budget guard rejected schedule block 11 (TI 3.000 s, needs TR 3.356 s) | 503.5 |
| 1.7 | 10 | 9 | block 9 TI 1.903 s, TR 1.700 -> 2.137 s | 1 | budget guard rejected schedule block 10 (TI 3.000 s, needs TR 3.356 s) | 503.6 |
| 2 | 8 | 8 | block 8 TI 3.000 s, TR 2.000 -> 3.356 s | 0 | no time left for a block at the TR floor | 555.4 |
| 2.5 | 7 | 6 | none | 1 | budget guard rejected schedule block 7 (TI 3.000 s, needs TR 3.356 s) | 480.0 |
| 3 | 5 | 5 | block 5 TI 3.000 s, TR 3.000 -> 3.356 s | 0 | budget guard rejected the repeat of block 1 | 491.4 |
| 4 | 4 | 4 | none | 0 | budget guard rejected the repeat of block 1 | 512.0 |

TI capped (secondary):

| TR (s) | Grid TIs | Run | Lifted blocks (TR requested -> executed) | Grid blocks not run | Episode ended by | Scan time (s) |
|---|---|---|---|---|---|---|
| 0.8 | 20 | 20 | none | 0 | max_blocks | 512.0 |
| 1 | 17 | 17 | none | 0 | budget guard rejected the repeat of block 1 | 544.0 |
| 1.077 (policy median TR) | 16 | 16 | none | 0 | no time left for a block at the TR floor | 551.5 |
| 1.2 | 14 | 14 | none | 0 | budget guard rejected the repeat of block 1 | 537.6 |
| 1.295 (policy mean TR) | 13 | 13 | none | 0 | budget guard rejected the repeat of block 1 | 538.7 |
| 1.35 | 12 | 12 | none | 0 | budget guard rejected the repeat of block 1 | 518.4 |
| 1.5 | 11 | 11 | none | 0 | budget guard rejected the repeat of block 1 | 528.0 |
| 1.7 | 10 | 10 | none | 0 | no time left for a block at the TR floor | 544.0 |
| 2 | 8 | 8 | none | 0 | budget guard rejected the repeat of block 1 | 512.0 |
| 2.5 | 7 | 7 | none | 0 | no time left for a block at the TR floor | 560.0 |
| 3 | 5 | 5 | none | 0 | budget guard rejected the repeat of block 1 | 480.0 |
| 4 | 4 | 4 | none | 0 | budget guard rejected the repeat of block 1 | 512.0 |

## Interpretation

- **As built (primary)**: the best TR is 2.5 s at 5.60 [4.92, 6.32]% (6 blocks, 480 s), -0.44 pp [-1.48, +0.53] against TR 1.7 s, which ranks 3 of 12. No as-built TR is significantly better than 1.7 s, so the reported comparator is close to the best TR of its own construction.
- **TI capped (secondary)**: the best TR is 2 s at 5.12 [4.52, 5.71]% (8 blocks, 512 s), -0.91 pp [-1.47, -0.33] against the reported comparator. Rows significantly better than the reported comparator: 2 s capped. Capping alone at TR 1.7 s gives 5.82 [5.12, 6.49]% (10 blocks, 544 s), -0.22 pp [-0.65, +0.18].
- **Policy-matched TRs**: at the policy mean TR (1.295 s) the grid reaches 9.10% as built and 8.52% capped; at the policy median TR (1.077 s) the grid reaches 11.99% as built and 11.13% capped. Short TRs make this grid much worse, so matching the policy's TR does not bring it near the policy; the TI placement matters as well, consistent with the schedule-shape control (`control_summary.md`).
- **The 240 s claim (4.62%)**: rows below it in mean MAPE: none; the lowest is 5.12% (2 s capped). The 240 s policy is significantly better than every as-built row (closest: 2.5 s, grid - RL +0.97 pp [+0.07, +1.91]), but not than 2 s capped (grid - RL +0.50 pp [-0.19, +1.19]).
- **The main result (2.93%)**: the closest row is 2 s capped at 5.12%, grid - RL +2.19 pp [+1.48, +2.86]; every row is significantly worse than the 560 s policy.
- The best TRs are picked after seeing 24 rows on the same 24 phantoms, so their MAPEs and paired gains are optimistic.
- For the abstract: the sentence that the 240 s policy beat both fixed designs at 560 s holds against the reported comparator (grid - RL +1.41 pp [+0.80, +2.06]) and in mean against every grid here, but not significantly against 2 s capped; if that grid is cited, say the 240 s policy matched it rather than beat it.

## Runtime

Local CPU, Julia 1.12.6, 8 threads: 69 min for 6264 full-Bloch blocks (0.66 s per block including the fit).

## Regenerate

From the RLxMRI repo root:

```bash
julia -t 8 --project=. ismrm_latex/vancouver/scripts/grid_tr_sweep.jl
.venv/bin/python ismrm_latex/vancouver/scripts/grid_tr_sweep_summary.py
```

Outputs: `figures/control_data/grid_tr_sweep.json` (per-scan results, schedules and feasibility of every row) and this file.

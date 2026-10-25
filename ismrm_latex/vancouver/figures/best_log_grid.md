# Fixed log TI grid comparator: TR 2.0 s

The abstract's fixed log TI grid is `best_log_grid_schedule` (`scripts/fixed_schedules.jl`): the best row of the TR sweep (`grid_tr_sweep.md`, TI-capped series), replacing `log_grid_trmatched` (TR 1.7 s).
Schedule: 8 blocks at TR 2.0 s, 32 shots each (512 s of the 560 s budget), TIs log-spaced over 0.05-1.779 s (0.050, 0.083, 0.139, 0.231, 0.385, 0.641, 1.068, 1.779 s); the top TI is capped at 0.9 TR - TE - 1 ms, so no block's TR is lifted.
Replayed by `scripts/best_log_grid.jl` as the fixed comparators were (same E2Env, generalised-IR T1 fitter, per-seed noise and Python action round trip, numpy2 promotion rules).

## Reproduction checks

- Held-out: the replay reproduces the sweep's capped TR 2.0 s row (`ismrm_latex/vancouver/figures/control_data/grid_tr_sweep.json`) bit for bit on all 24 scans (every field: T1 true and estimates, MAPE after each block, blocks, scan time, executed timings).
- Clustered-T1: before the grid, `log_grid_trmatched` replayed on the 96 phantoms reproduces its arm of `ismrm_latex/vancouver/figures/control_data/control_narrow_band.json` bit for bit, so the grid ran on the same phantoms and noise as that file's arms.

## Held-out (24 phantoms, seeds 600000-600023)

| Arm | Mean MAPE % [95% CI] | Scans < 5% | Worst scan MAPE % | Mean blocks | Mean scan time (s) |
|---|---|---|---|---|---|
| Fixed log TI grid, TR 2.0 s | 5.12 [4.52, 5.71] | 45.8% (11/24) | 8.67 | 8.00 | 512 |
| Cramer-Rao optimal | 7.74 [7.08, 8.42] | 8.3% (2/24) | 11.72 | 10.00 | 558 |
| Distilled fixed protocol | 3.34 [2.30, 4.86] | 91.7% (22/24) | 17.87 | 13.00 | 549 |
| Fixed log TI grid, TR 1.7 s (replaced) | 6.04 [5.36, 6.70] | 25.0% (6/24) | 10.29 | 9.00 | 504 |
| TI-histogram policy, 560 s | 2.93 [2.58, 3.30] | 95.8% (23/24) | 5.13 | - | - |
| RL policy, 240 s | 4.62 [4.06, 5.19] | 62.5% (15/24) | 7.74 | - | - |

| Comparison | Grid - arm, mean [95% CI] | Median per-phantom difference (pp) | Phantoms where the grid is worse |
|---|---|---|---|
| Grid - TI-histogram policy, 560 s | +2.19 pp [+1.48, +2.86] | +2.38 | 21/24 |
| Grid - Distilled fixed protocol | +1.78 pp [+0.27, +3.10] | +1.78 | 20/24 |
| Grid - RL policy, 240 s | +0.50 pp [-0.19, +1.19] | +0.69 | 16/24 |
| Grid - Fixed log TI grid, TR 1.7 s (replaced) | -0.91 pp [-1.47, -0.33] | -0.97 | 6/24 |

## Clustered-T1 (96 phantoms, seeds 600000-600095, T1s within +-15% of a random centre)

| Arm | Mean MAPE % [95% CI] | Scans < 5% | Worst scan MAPE % | Mean blocks | Mean scan time (s) |
|---|---|---|---|---|---|
| Fixed log TI grid, TR 2.0 s | 8.77 [8.02, 9.54] | 18.8% (18/96) | 21.61 | 8.00 | 512 |
| Cramer-Rao optimal | 8.53 [7.95, 9.16] | 10.4% (10/96) | 18.56 | 10.00 | 558 |
| Distilled fixed protocol | 12.20 [8.02, 16.94] | 78.1% (75/96) | 91.47 | 13.00 | 549 |
| Fixed log TI grid, TR 1.7 s (replaced) | 8.31 [7.55, 9.10] | 18.8% (18/96) | 22.68 | 9.00 | 504 |
| Adaptive (policy in the loop) | 2.76 [2.58, 2.95] | 96.9% (93/96) | 5.35 | 16.71 | 545 |

| Comparison | Grid - arm, mean [95% CI] | Median per-phantom difference (pp) | Phantoms where the grid is worse |
|---|---|---|---|
| Grid - Adaptive (policy in the loop) | +6.01 pp [+5.26, +6.80] | +6.12 | 88/96 |

By band centre c (bins of `control_narrow_band.md`, edges 0.1 and 0.5 s); CIs and paired differences resample the phantoms within each bin.

| Centre bin | Grid MAPE % [95% CI] | Scans < 5% | Worst scan MAPE % | Grid - Adaptive [95% CI] |
|---|---|---|---|---|
| c < 0.1 s (n = 26) | 10.30 [8.72, 12.00] | 7.7% (2/26) | 21.61 | +7.85 pp [+6.40, +9.37] |
| 0.1 <= c < 0.5 s (n = 49) | 10.02 [9.41, 10.64] | 0.0% (0/49) | 15.70 | +7.15 pp [+6.51, +7.80] |
| c >= 0.5 s (n = 21) | 3.97 [3.21, 4.76] | 76.2% (16/21) | 7.55 | +1.07 pp [+0.18, +1.99] |

## For the text

- Held-out, grid: 5.12 [4.52, 5.71]%, 11/24 scans < 5%, worst 8.67%, 8 blocks, 512 s.
- Held-out, grid - TI-histogram policy, 560 s: +2.19 pp [+1.48, +2.86].
- Held-out, grid - Distilled fixed protocol: +1.78 pp [+0.27, +3.10].
- Held-out, grid - RL policy, 240 s: +0.50 pp [-0.19, +1.19].
- Clustered-T1, grid: 8.77 [8.02, 9.54]%, 18/96 scans < 5%, worst 21.61%; grid - adaptive +6.01 pp [+5.26, +6.80].
- Worst-scan MAPE of the three fixed schedules on held-out (grid, Cramer-Rao, distilled): 8.67-17.87%.
- Mean MAPE of the three fixed schedules on clustered-T1 (grid, Cramer-Rao, distilled): 8.53-12.20%.

MAPE: mean over the five spheres of |T1 estimate - T1 true| / T1 true, averaged over scans.
CI: percentile bootstrap, 5000 resamples over the phantoms, numpy `default_rng(0)` (as `python/baseline_e2.py:41-58`). Paired: grid minus the other arm per phantom, bootstrapped the same way; positive means the grid is less accurate.
The TR 2.0 s grid was picked as the best of the sweep's rows on these same 24 held-out phantoms, so its held-out MAPE is optimistic; the clustered-T1 phantoms played no part in the choice.

## Regenerate

From the RLxMRI repo root:

```bash
julia -t 6 --project=. ismrm_latex/vancouver/scripts/best_log_grid.jl
.venv/bin/python ismrm_latex/vancouver/scripts/best_log_grid_summary.py
```

Outputs: `figures/control_data/best_log_grid_heldout.json`, `figures/control_data/best_log_grid_narrow_band.json` (per-scan results) and this file.

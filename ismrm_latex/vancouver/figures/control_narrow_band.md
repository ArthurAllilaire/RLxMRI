# Control: per-scan adaptation on narrow-band phantoms

Question: when each scan's five sphere T1s sit in a narrow band at an unknown position, does the TI-histogram policy's per-scan adaptation pay off against fixed schedules?
Training and the held-out evaluation draw each sphere T1 independently and uniformly over the T1-plate range, so every scan spans most of it and needs similar TIs (`control_summary.md`: adaptation helped only on two phantoms with a very short T1 sphere).
Policy: `runs/e2/mf_runB_5sphere_hist_560s_gpu`, checkpoint `global_best`, evaluated out of its training distribution; no retraining.
Phantoms: 96 scans, seeds 600000-600095, `t1_sampler = :narrow_band` (`julia/rl/e2.jl`): a centre c drawn log-uniformly over the T1-plate range 0.024-1.879 s, then each sphere's T1 uniformly over c(1 +- 0.15), with T2 and T2* keeping each sphere's nominal T2/T1 ratio.
Every other env setting is the histogram run's `base_env_kwargs`.
The sampler draws the centre from its own stream, so phantom k keeps held-out scan k's pose and per-seed noise stream and differs from it only in its T1s.
Every arm runs on the same phantoms with the same noise stream, the same E2Env forward model (full Bloch, CPU) and the same generalised-IR T1 fitter; only the (TI, TR) schedule differs.

## Step 1: regression check on the default sampler

After the sampler change, the pure-Julia rollout of `global_best` on the default held-out seeds still reproduces `eval_t1err_heldout.json` exactly: largest per-sphere APE difference 0.0e+00, largest true-T1 difference 0.0e+00 s, mean MAPE 2.9310% (recorded 2.9310%).
Replaying the distilled protocol and the two fixed comparators on those phantoms reproduces their recorded per-scan T1 estimates, MAPEs and block counts bit for bit (`control_results.json` `arms.fixed_median_validation`; `fig5_v2_data/fixed_comparators_heldout.json`).

## Arms

- **Adaptive**: the policy in the loop, rolled out as in `control_schedule_swap.jl`.
- **Swapped**: phantom k gets the executed schedule of scan (k + s) mod 96, for s = 1, 32, 64 (288 runs, no policy in the loop); the direct test of per-scan tailoring.
- **Distilled**: the fixed protocol `median_protocols.fixed_median_validation` of `control_results.json` (13 blocks, 549.2 s, shortest TI 243 ms), as recorded, not re-derived.
- **Fixed log TI grid** and **Cramer-Rao optimal**: the 560 s schedules of the abstract's comparisons, replayed as `fixed_comparators_heldout.jl` replays them (`fixed_schedules.jl`).
  The Cramer-Rao schedule is not re-optimised: the band's position is unknown, so the full-range design is the fair fixed comparator.

## Results (96 narrow-band phantoms)

| Arm | Schedule | Mean MAPE % [95% CI] | Scans < 5% | Worst scan MAPE % | Mean blocks | Mean scan time (s) | Wide-range held-out MAPE % |
|---|---|---|---|---|---|---|---|
| Adaptive (policy in the loop) | own, chosen online | 2.76 [2.58, 2.95] | 96.9% (93/96) | 5.35 | 16.71 | 545 | 2.93 |
| Swapped, shift 1 | scan (k + 1) mod 96 | 6.15 [3.73, 9.48] | 77.1% (74/96) | 100.00 | 16.71 | 545 | - |
| Swapped, shift 32 | scan (k + 32) mod 96 | 4.94 [3.40, 7.32] | 79.2% (76/96) | 100.00 | 16.71 | 545 | - |
| Swapped, shift 64 | scan (k + 64) mod 96 | 4.91 [3.53, 6.68] | 80.2% (77/96) | 49.74 | 16.71 | 545 | - |
| Swapped, mean of 3 shifts | another scan's | 5.33 [3.98, 7.00] | 78.8% (227/288) | 100.00 | 16.71 | 545 | 3.61 |
| Distilled fixed protocol | medians of validation schedules | 12.20 [8.02, 16.94] | 78.1% (75/96) | 91.47 | 13.00 | 549 | 3.34 |
| Fixed log TI grid, TR 1.7 s | fixed | 8.31 [7.55, 9.10] | 18.8% (18/96) | 22.68 | 9.00 | 504 | 6.04 |
| Cramer-Rao optimal | fixed | 8.53 [7.95, 9.16] | 10.4% (10/96) | 18.56 | 10.00 | 558 | 7.74 |

MAPE: mean over the five spheres of |T1 estimate - T1 true| / T1 true, averaged over scans.
CI: percentile bootstrap, 5000 resamples over the 96 phantoms, numpy `default_rng(0)` (as `python/baseline_e2.py:41-58`).
For the swap mean, each phantom's 3 swapped MAPEs are averaged before resampling; its success share and worst scan are over all 288 runs.
Wide-range held-out: the same arm on the 24 default held-out phantoms (`control_summary.md`; shifts 1, 8, 16 there).

## Paired differences to Adaptive

| Arm | Mean difference, arm - Adaptive [95% CI] | Median per-phantom difference (pp) | Phantoms where the arm is worse |
|---|---|---|---|
| Swapped, shift 1 | +3.38 pp [+0.99, +6.71] | +0.00 | 47/96 |
| Swapped, shift 32 | +2.18 pp [+0.61, +4.56] | +0.44 | 57/96 |
| Swapped, shift 64 | +2.15 pp [+0.75, +3.93] | +0.02 | 50/96 |
| Swapped, mean of 3 shifts | +2.57 pp [+1.21, +4.24] | +0.50 | 67/96 |
| Distilled fixed protocol | +9.43 pp [+5.24, +14.17] | -0.24 | 43/96 |
| Fixed log TI grid, TR 1.7 s | +5.54 pp [+4.80, +6.33] | +5.58 | 91/96 |
| Cramer-Rao optimal | +5.77 pp [+5.09, +6.47] | +5.58 | 94/96 |

Paired bootstrap over the phantoms (5000 resamples, `default_rng(0)`); for the swap mean, the 3 shifts are averaged per phantom first.
On the wide-range held-out phantoms the same comparisons were +0.67 pp [-0.01, +1.55] (swapped mean) and +0.41 pp [-0.69, +1.92] (distilled).

## By band centre

Bins of the band centre c with edges 0.1 and 0.5 s (fixed before the run); CIs and paired differences resample the phantoms within each bin.

| Centre bin | Arm | Mean MAPE % [95% CI] | Scans < 5% | Worst scan MAPE % | Paired difference, arm - Adaptive [95% CI] |
|---|---|---|---|---|---|
| c < 0.1 s (n = 26) | Adaptive | 2.45 [2.09, 2.83] | 96.2% (25/26) | 5.09 | - |
| c < 0.1 s (n = 26) | Swapped, mean of 3 shifts | 10.43 [5.93, 16.03] | 70.5% (55/78) | 100.00 | +7.97 pp [+3.56, +13.53] |
| c < 0.1 s (n = 26) | Distilled fixed protocol | 39.45 [29.13, 49.64] | 19.2% (5/26) | 91.47 | +36.99 pp [+26.80, +47.05] |
| c < 0.1 s (n = 26) | Fixed log TI grid, TR 1.7 s | 10.38 [8.50, 12.41] | 11.5% (3/26) | 22.68 | +7.93 pp [+6.23, +9.75] |
| c < 0.1 s (n = 26) | Cramer-Rao optimal | 8.84 [7.51, 10.15] | 15.4% (4/26) | 16.66 | +6.39 pp [+4.82, +7.89] |
| 0.1 <= c < 0.5 s (n = 49) | Adaptive | 2.87 [2.63, 3.12] | 95.9% (47/49) | 5.35 | - |
| 0.1 <= c < 0.5 s (n = 49) | Swapped, mean of 3 shifts | 2.90 [2.64, 3.19] | 91.8% (135/147) | 9.18 | +0.03 pp [-0.30, +0.38] |
| 0.1 <= c < 0.5 s (n = 49) | Distilled fixed protocol | 1.75 [1.59, 1.90] | 100.0% (49/49) | 2.84 | -1.13 pp [-1.46, -0.82] |
| 0.1 <= c < 0.5 s (n = 49) | Fixed log TI grid, TR 1.7 s | 8.84 [8.29, 9.39] | 2.0% (1/49) | 13.32 | +5.97 pp [+5.34, +6.57] |
| 0.1 <= c < 0.5 s (n = 49) | Cramer-Rao optimal | 8.87 [8.11, 9.66] | 6.1% (3/49) | 18.56 | +6.00 pp [+5.18, +6.87] |
| c >= 0.5 s (n = 21) | Adaptive | 2.89 [2.51, 3.29] | 100.0% (21/21) | 4.78 | - |
| c >= 0.5 s (n = 21) | Swapped, mean of 3 shifts | 4.71 [4.21, 5.22] | 58.7% (37/63) | 8.34 | +1.82 pp [+1.18, +2.46] |
| c >= 0.5 s (n = 21) | Distilled fixed protocol | 2.84 [2.44, 3.24] | 100.0% (21/21) | 4.57 | -0.05 pp [-0.51, +0.40] |
| c >= 0.5 s (n = 21) | Fixed log TI grid, TR 1.7 s | 4.48 [3.62, 5.48] | 66.7% (14/21) | 10.06 | +1.58 pp [+0.74, +2.54] |
| c >= 0.5 s (n = 21) | Cramer-Rao optimal | 7.34 [6.22, 8.46] | 14.3% (3/21) | 11.96 | +4.45 pp [+3.21, +5.71] |

What the policy did in each bin (medians over scans of each scan's shortest and median executed TI):

| Centre bin | Phantoms | Median centre (ms) | Mean blocks | Shortest TI (ms) | Median TI (ms) |
|---|---|---|---|---|---|
| c < 0.1 s | 26 | 45 | 18.3 | 53 | 120 |
| 0.1 <= c < 0.5 s | 49 | 243 | 18.1 | 63 | 100 |
| c >= 0.5 s | 21 | 948 | 11.5 | 220 | 446 |
| Distilled protocol | - | - | 13 | 243 | 277 |

## Does the policy track the band?

Pearson r on log scales over the 96 narrow-band scans, with percentile bootstrap CIs (5000 resamples, `default_rng(0)`); the slope is the least-squares slope of log TI on log c (1 = TIs scale with the band).

| Pair | Narrow band: r [95% CI] | Slope | Wide-range held-out (24 scans): r [95% CI] |
|---|---|---|---|
| log c vs log median TI | +0.71 [+0.60, +0.80] | 0.37 | - |
| log c vs log shortest TI | +0.86 [+0.82, +0.90] | 0.45 | - |
| log shortest T1 vs log shortest TI | +0.87 [+0.82, +0.90] | - | +0.57 [+0.28, +0.83] |

## Interpretation

Decision rule (set before the run): adaptation pays off if the adaptive policy clearly beats both the swapped schedules and the distilled protocol (paired 95% CIs exclude zero); if it is no better than either, the policy does not exploit narrow bands and the idea needs a policy trained on them.
Result: it beats both, so by the rule the policy's per-scan adaptation pays off when scans are atypical: another scan's schedule costs +2.57 pp [+1.21, +4.24] and the distilled protocol +9.43 pp [+5.24, +14.17], against +0.67 pp [-0.01, +1.55] and +0.41 pp [-0.69, +1.92] on the wide-range held-out phantoms.
Out of its training distribution and without retraining, the policy stays as accurate as on wide-range scans (2.76% [2.58, 2.95] against 2.93%), with 93/96 scans below 5% and a worst scan of 5.35%; the fixed log TI grid (8.31%) and the Cramer-Rao schedule (8.53%) trail it by +5.54 pp [+4.80, +6.33] and +5.77 pp [+5.09, +6.47].
The result is mixed across the centre range, and the pooled gain over the distilled protocol (12.20%) is a tail effect: that protocol is better than the policy on the median phantom (-0.24 pp, worse on only 43/96) but fails (scan MAPE above 10%) on 20 phantoms, all with c below 63 ms.
Short bands, c < 0.1 s (26 phantoms): every alternative fails; swapped +7.97 pp [+3.56, +13.53], distilled +36.99 pp [+26.80, +47.05], log TI grid +7.93 pp [+6.23, +9.75], Cramer-Rao +6.39 pp [+4.82, +7.89].
All 13 catastrophic swapped runs (scan MAPE above 20%, 3 capped at 100%) are such phantoms given a schedule whose shortest TI is 204 ms or more, while the policy chose about 18 blocks with TIs down to about 53 ms.
Long bands, c >= 0.5 s (21 phantoms): the policy switches to fewer, longer blocks (11.5 blocks, median TI 446 ms); swapped schedules still cost +1.82 pp [+1.18, +2.46], but the distilled protocol matches the policy (-0.05 pp [-0.51, +0.40]).
Mid bands, 0.1 <= c < 0.5 s (49 phantoms): adaptation buys nothing; swapped schedules match the policy (+0.03 pp [-0.30, +0.38]) and the distilled protocol beats it (-1.13 pp [-1.46, -0.82]; 1.75% against 2.87%).
On 34/49 of these phantoms, up to c = 338 ms, the policy keeps a short-T1 schedule (shortest TI below 100 ms), where the distilled protocol's longer TIs (shortest 243 ms) do better.
Without the 26 phantoms with c < 0.1 s (a post hoc cut), the distilled protocol is better than the policy (paired -0.80 pp [-1.09, -0.51]), while another scan's schedule still costs +0.56 pp [+0.21, +0.92]: outside short bands the policy's tailoring is worth more than another scan's schedule but less than a well-placed fixed protocol.
The policy's TIs do track the band, more tightly than on wide-range scans: r = +0.86 [+0.82, +0.90] between log c and the log of its shortest TI, and +0.87 [+0.82, +0.90] between log shortest T1 and log shortest TI, against +0.57 [+0.28, +0.83] on the wide-range held-out scans.
The slopes (0.45 for the shortest TI, 0.37 for the median TI) show its TIs moving less than half as far as the band in log units (mean blocks 18.3 / 18.1 / 11.5 from short to long bands).
For the abstract: on narrow-band scans the TI-histogram policy, without retraining, tailors its schedule to the scan; another scan's schedule costs +2.6 pp and the distilled fixed protocol +9.4 pp, almost all of it in short-T1 bands, while a fixed protocol is better for mid-range bands; claim adaptation as robustness to atypical scans, not as a uniform accuracy gain.
Caveats: one training seed; out-of-distribution evaluation of a policy trained on wide-range scans; one band width (w = 0.15); the centre bins hold 26, 49 and 21 phantoms and the c >= 0.1 s cut is post hoc; E2Env caps a scan's MAPE at 100%, which can only understate the failing arms' means; the distilled protocol and the Cramer-Rao schedule were designed for wide-range scans, and a fixed protocol tuned to narrow bands at unknown centres (for example distilled from the policy's narrow-band schedules on separate seeds) was not tested and is the stronger fixed comparator for this question.

## Runtime

Local CPU, Julia 1.12.6, 8 threads: 35.8 min for the 9488 full-Bloch blocks of the narrow-band arms (0.23 s per block including the fit), plus 4.0 min for the regression check.

| Arm | Simulated blocks | Wall time (s) |
|---|---|---|
| adaptive | 1604 | 362 |
| swap_1 | 1604 | 363 |
| swap_32 | 1604 | 364 |
| swap_64 | 1604 | 361 |
| distilled | 1248 | 282 |
| log_grid_trmatched | 864 | 199 |
| cr_optimal | 960 | 216 |

## Regenerate

From the RLxMRI repo root (needs `figures/control_data/control_results.json` from `control_schedule_swap.jl` and `figures/fig5_v2_data/fixed_comparators_heldout.json` from `fixed_comparators_heldout.jl`):

```bash
julia -t 8 --project=. ismrm_latex/vancouver/scripts/control_narrow_band.jl 96
.venv/bin/python ismrm_latex/vancouver/scripts/control_narrow_band_summary.py
```

Outputs: `figures/control_data/control_narrow_band.json` (band centres, executed adaptive schedules and per-phantom results of every arm) and this file.

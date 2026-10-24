| Checkpoint | Rung | Saved as | Step | Training time (h) | Blocks/scan | MAPE % [95% CI] |
|---|---|---|---|---|---|---|
| stage0_analytic_end | analytic | end of rung | 4096 | 0.034 | 6.54 | 14.78 [11.77, 18.02] |
| stage1_cached3_best | cached 3 mm | best in rung | 10000 | 0.158 | 6.96 | 5.80 [5.11, 6.51] |
| stage1_cached3_end | cached 3 mm | end of rung | 12288 | 0.218 | 7.42 | 4.67 [3.91, 5.49] |
| stage2_cached_best (same weights as stage1_cached3_end) | cached 1 mm | best in rung | 12289 | 0.225 | 7.42 | 4.57 [3.85, 5.37] |
| stage2_cached_end | cached 1 mm | end of rung | 20480 | 0.435 | 6.08 | 5.50 [4.81, 6.27] |
| stage3_full3_best (same weights as stage2_cached_end) | full 3 mm | best in rung | 20481 | 0.438 | 6.08 | 5.50 [4.81, 6.27] |
| stage3_full3_end | full 3 mm | end of rung | 32768 | 0.763 | 7.46 | 7.20 [6.09, 8.37] |
| global_best | full 1 mm | selected | 242769 | 7.002 | 13.17 | 3.00 [2.52, 3.50] |
| stage4_full_best | full 1 mm | best in rung | 282769 | 8.232 | 14.29 | 2.84 [2.49, 3.22] |
| stage4_full_end | full 1 mm | end of rung | 307305 | 8.993 | 15.54 | 3.16 [2.50, 3.92] |

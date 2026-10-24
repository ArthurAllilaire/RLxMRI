# ISMRM 2027 Standard Abstract - working draft

Deadline: 28 Oct 2026, 23:59 UTC (ECHO).
Meeting: 8-13 May 2027, Vancouver; presenter from ICR attends in person.

## Decisions so far

- Type: ISMRM Standard Abstract.
- Headline: what the policy remembers matters.
  A policy with an explicit memory of the inversion times it has already sampled reaches T1 precision that fixed protocols could not reach with more scan time (nominal ~4x acceleration under the convention of Beracha et al.).
- Supporting story: RL learned both a fixed component (an opening probe shared by every scan, then coverage of TI space like a log grid) and per-scan adaptation.
- Open-source contribution: MRISystemPhantom.jl (twin, random phantoms, sequences, fitters, Cramer-Rao solver, and the forward models for every fidelity rung).
  The bias-aware promotion rule is described as a method, not a library.
  RLxMRI is linked in the Acknowledgements only, as code to reproduce the results.
- Comparators: as in the final report, the fixed log-spaced TI grid (TR 1.7 s) and the Cramer-Rao-optimal schedule.
  The clinical IR-SE schedule is dropped: at 560 s only 3 of its 6 TIs (0.05-0.4 s) fit, so it is a straw man.
- KomaMRI bugs: at most one clause ("simulator validated by T1 recovery").
- Existing results only; no new runs before the deadline.
- 4 figures (5 allowed, not required).

## Title (max 125 characters)

Adaptive T1 mapping with reinforcement learning: acquisition memory buys precision that extra scan time cannot

## Authors

Arthur Allilaire (1), Andreas Wetscherek (2), Wayne Luk (1).
(1) Department of Computing, Imperial College London, London, UK.
(2) Joint Department of Physics, The Institute of Cancer Research and The Royal Marsden NHS Foundation Trust, London, UK.
Presenting author: TBC (ICR).

## Categories and keywords

- Primary: Acquisition & Reconstruction / AI methods (101).
- Secondary: Acquisition & Reconstruction / Quantitative Imaging: Relaxometry, Multi-Parametric, MR Fingerprinting, & Synthetic MRI (114).
- Alternative secondary: Acquisition & Reconstruction / Open Source Software, Sequences, & Reconstruction Algorithms (119), or Physics & Engineering / In Silico (1305).
- General keyword: Simulations.
- Extra keywords: Quantitative, Relaxation, Phantom, Precision & Accuracy.
- Option: tick "Deep Dive into New Dimensions" (quantitative biomarker dimension).

## Impact (max 40 words, text only) - DRAFT

Adaptive acquisition could shorten quantitative MRI: a learned policy that remembers which inversion times it has sampled reached T1 precision that extra scan time could not give fixed protocols. Our open-source phantom twin lets others develop such policies.

## Synopsis (max 100 words across the four parts, text only) - DRAFT

**Motivation:** Quantitative MRI protocols fix their timings before the scan, spending time on measurements that are uninformative for the tissue actually present.

**Goal(s):** To test whether a reinforcement-learning policy that chooses each inversion time from the data acquired so far can estimate T1 more efficiently than fixed protocols.

**Approach:** We built an open-source digital twin of the ISMRM/NIST system phantom with multi-fidelity forward models, and trained policies on noisy Bloch-simulated inversion-recovery imaging.

**Results:** Remembering its sampled inversion times, the policy halved T1 error versus the best fixed schedule (2.9% vs 6.0%), a nominal fourfold acceleration; tripling fixed-schedule scan time barely helped.

## Body outline (750 words max, references excluded)

### Introduction (~110 words)

- T1 mapping protocols fix their inversion times (TIs) in advance so they stay informative over the whole range of plausible tissue, which spends scan time on measurements that carry little information for the tissue actually present.
- Adaptive MR updates a Bayesian posterior during the scan and picks the next timing from it (1); simulated T1 accelerations of 1.7-4x, but for a single voxel and one tissue, via an analytic lookup.
- Reinforcement learning (RL) could learn such rules directly from simulation, including multiple tissues in one image and reconstruction effects, without a hand-built posterior.
- RL in MRI so far either optimises a fixed protocol offline (2,3) or targets image-level goals such as shape classification or k-space sampling (4,5).
  To our knowledge, no closed-loop policy has been trained and evaluated on fitted quantitative error.
- Aim: (i) an open-source digital twin with multi-fidelity forward models that makes RL with a Bloch simulator in the loop affordable; (ii) test whether an adaptive policy, and what it remembers, beats fixed T1 protocols.

### Methods (~300 words)

- **Digital twin.** MRISystemPhantom.jl models the ISMRM/NIST system phantom (6): housing, T1/T2/PD plates, fiducials, reference relaxation values at 1.5/3 T.
  It returns KomaMRI (7) phantoms and samples random phantoms (materials, pose) with hidden ground truth.
  It also ships 2-D IR-SE sequences, reconstruction and ROI extraction, a generalised IR T1 fitter that models partial recovery with per-block TR, and a Cramer-Rao schedule solver.
  Simulator validated by recovering the phantom's reference T1 without noise (one clause only).
- **Task.** Five T1-plate positions; each scan draws T1 uniformly over 0.024-1.88 s (T2/T1 ratio preserved) with in-plane pose jitter.
  1.5 T, 64x32 matrix, 1 mm voxels and slab, complex Gaussian k-space noise (NEMA SNR ~17).
  Each action is one IR-SE block, TI in [0.01, 3] s (log scale) and TR in [0.5, 5] s; TE 20 ms, 90 deg; a block costs 32 TR; the scan ends at a 240 s or 560 s budget.
- **Agent.** PPO (8) with an MLP; observation = log running T1 estimates plus budget used; reward = per-block reduction in log MAPE.
  Memory arms: none; a 12-bin histogram of executed log-TIs; the fitter's uncertainty; an LSTM.
- **Multi-fidelity training.** Ladder from an analytic model, to cached water, to coarse water, to full Bloch (~30x range in cost per step).
  Cached water exploits linearity: the water k-space is a per-shot scalar times a fixed template, so only sphere spins are simulated (matches full-Bloch T1 fits within 0.12%).
  Promotion between rungs and checkpoint selection are scored only on held-out full-Bloch probes; 9 h GPU per arm.
- **Comparators** (as in the final report). A fixed log-spaced TI grid at TR 1.7 s with as many TIs as fit the budget, and a Cramer-Rao-optimal (TI, TR) schedule (9) solved for the nominal plate under the same budget.
  All methods use the same fitter.
- **Evaluation.** 24 held-out phantoms whose seeds were never used for training or checkpoint selection, identical for every method; mean absolute percentage error (MAPE) of fitted T1, bootstrap 95% CI.

### Results (~170 words)

- Memory ablation at 560 s (Fig 2): TI histogram 2.93% [2.58-3.30] vs fixed grid 6.04% [5.36-6.70] and Cramer-Rao 7.74%.
  95.8% vs 25% of scans under 5% error; confidence intervals do not overlap.
  No memory 4.16%, fitter uncertainty 5.25%, LSTM 7.31%.
- Precision ratio 6.04/2.93 = 2.06, so a nominal ~4x acceleration under the precision-squared-to-time convention of (1).
- More time did not buy fixed protocols this precision (Fig 4).
  Fixed grid: 6.86% at 163 s, 6.04% at 504 s (1/sqrt(t) scaling predicts 3.9%).
  Best fixed schedule at any tested budget: 6.04%.
  Even the memoryless policy at 240 s (4.62%) beat the fixed grid at 504 s.
- Behaviour (Fig 3): the learned opening probe (TI 1.06 s then 0.49 s) is identical in every scan; from block 3 the TIs diverge per scan (SD of log10 TI across scans 0.15-0.19).
  The histogram policy fills unsampled TI bins and is only weakly coupled to the current estimate (r = -0.12); the LSTM tracks the estimate (r = +0.78) and does worse.
- Error is even across the T1 range: 2.6% (short) to 3.2% (long).

### Discussion (~120 words)

- The running T1 estimate is a lossy summary of what has been measured; the value of the next TI depends on which TIs were already sampled.
  The order-invariant histogram matches what the fitter consumes (a set of TI/signal pairs) and beat both learned recurrent memory and the fitter's uncertainty.
- The policy combined what fixed protocols do well (a fixed opening probe and coverage of TI space) with per-scan adaptation, and reached precision more scan time did not give.
  Caveat: the first two blocks are identical by construction, since no T1 fit exists before two measurements; the policy chose their values but not that they are fixed.
- Limits: simulation only (ideal RF, no B1/eddy currents/motion), five tissues, one training seed per arm, 24 held-out scans.
  The LSTM received ~7x fewer full-fidelity steps under the shared wall-clock budget.
  On the full 14-sphere plate (a fixed set of tissues) the Cramer-Rao fixed schedule still won (4.70% vs 9.44%): adaptivity pays when tissue varies from scan to scan.
  The ~4x is nominal, since MAPE also includes bias.

### Conclusion (~50 words)

- Adaptive acquisition with explicit memory of past measurements reached T1 precision fixed protocols could not reach with more time.
- The open-source twin makes adaptive qMRI methods testable against known ground truth before scanner time.
- Next: run learned protocols on the physical phantom on the MR-Linac, and widen the action space (T2, other sequence families).

## Figures (4; captions max 500 characters)

### Fig 1 - Digital twin, closed loop, fidelity ladder

Panels: (A) twin of the phantom (plates, housing); (B) T1 plate with the five active positions; (C) noisy simulated IR-SE image; (D) closed loop: sample phantom, simulate block, reconstruct and fit, update TI histogram, policy picks next TI/TR; (E) fidelity ladder strip with cost per step and cached-water agreement.
Source: refresh ismrm_latex/british/figures/method_overview plus Table e2-fidelity-ladder from the report.

Caption draft: Open-source digital twin and closed-loop experiment. (A) Twin of the ISMRM/NIST system phantom. (B) T1 plate; outlined positions are resampled each scan. (C) Simulated noisy IR-SE image. (D) After each block the image is reconstructed, T1 refitted and the TI histogram updated; the policy then chooses the next TI and TR. (E) Training climbs a ladder of forward models spanning ~30x in cost; cached water matches full Bloch within 0.12%.

### Fig 2 - Main result: memory ablation

Panels: (A) mean T1 MAPE with 95% CI for histogram, no memory, fitter uncertainty, LSTM, fixed grid, Cramer-Rao, annotated with % of scans under 5%; (B) per-sphere error vs realised T1 for the histogram policy with band means.
Source: ismrm_latex/british/figures/heldout_results (extend with sigma and LSTM arms); runs/e2/mf_runB_5sphere_*_560s_gpu/global_best/eval_summary.json; runs/e2/baselines_heldout_560s/baseline_summary.json.

Caption draft: T1 error on 24 held-out phantoms at a 560 s budget. (A) Mean MAPE with bootstrap 95% CIs; percentages give scans under 5% error. Only explicit memory of sampled TIs (histogram) separates clearly from all others; a learned LSTM memory and the fitter's uncertainty do not help. (B) Error of every sphere for the histogram policy against its realised T1, with means over short, mid and long T1.

### Fig 3 - The policy learns a fixed opening, then adapts

Panels: TI vs block index, one line per held-out scan, for the histogram policy (left) and the LSTM (right); shade blocks 1-2.
Source: runs/e2/mf_runB_5sphere_{hist,lstm}_560s_gpu/global_best/diagnostics/episodes.json (plot locally).

Caption draft: Inversion time chosen at each block, one line per held-out scan. Both policies open with the same two-block probe in every scan (shaded), then diverge per scan. The histogram policy (left) fills unsampled TI ranges and is nearly uncorrelated with the current T1 estimate (r = -0.12); the LSTM (right) tracks the estimate (r = +0.78) and is less accurate.

### Fig 4 - Precision that more time could not buy

Panels: T1 MAPE vs actual scan time: fixed grid at 163 s and 504 s, Cramer-Rao at 240 s and 558 s, memoryless RL at 235 s and 556 s, histogram RL at 560 s; dashed 1/sqrt(t) curve from the fixed grid's short-budget point; annotate the time the grid would need under 1/sqrt(t) to reach 2.93% (~2,100 s, ~4x).
Source: the baseline_summary.json files above plus runs/e2/mf_runB_cached3_cached_full3_full_gpu (240 s run).
Also the preview figure in simplified form (3:2, large text, no caption).

Caption draft: Error versus scan time on the same 24 held-out phantoms. Tripling the fixed grid's scan time reduced its error only from 6.9% to 6.0%, far less than 1/sqrt(t) scaling predicts (dashed). The adaptive policy beat the fixed grid at half the time, and with TI memory reached 2.9%. Matching that precision by repetition alone would need ~4x the time.

## Key numbers (all on held-out seeds 600000-600023)

| Method | Budget | Actual time | Blocks | MAPE [95% CI] | p90 | <5% | Source |
|---|---|---|---|---|---|---|---|
| RL + TI histogram | 560 s | 560 s | 13.6 | 2.93 [2.58, 3.30] | 4.19 | 95.8% | mf_runB_5sphere_hist_560s_gpu |
| RL, no memory | 560 s | 556 s | 14.0 | 4.16 [3.41, 4.99] | 6.64 | 66.7% | mf_runB_5sphere_560s_gpu |
| RL + fitter uncertainty | 560 s | - | 14.2 | 5.25 [4.32, 6.22] | 9.34 | 58.3% | mf_runB_5sphere_sigma_560s_gpu |
| RL, LSTM | 560 s | 577 s | 11.0 | 7.31 [6.10, 8.65] | 12.21 | 25.0% | mf_runB_5sphere_lstm_560s_gpu |
| RL, no memory | 240 s | 235 s | 5.9 | 4.62 [4.06, 5.19] | 6.24 | 62.5% | mf_runB_cached3_cached_full3_full_gpu |
| Fixed log grid (TR 1.7 s) | 560 s | 504 s | 10 | 6.04 [5.36, 6.70] | 7.81 | 25.0% | baselines_heldout_560s |
| Fixed log grid (TR 1.7 s) | 240 s | 163 s | 4 | 6.86 [5.83, 7.90] | 10.11 | 20.8% | mf_runB_cached3_..._gpu/baselines_heldout |
| Cramer-Rao optimal | 560 s | 558 s | 10 | 7.74 [7.08, 8.42] | 9.78 | 8.3% | baselines_heldout_560s |
| Cramer-Rao optimal | 240 s | 240 s | 4 | 15.96 [12.26, 19.81] | 29.29 | 0.0% | mf_runB_cached3_..._gpu/baselines_heldout |

Derived:
- Precision ratio 6.04 / 2.93 = 2.06; squared = 4.25, so "nominal ~4x".
- 1/sqrt(t) from the grid's 240 s point: 6.86 x sqrt(163/504) = 3.9% expected at 504 s, 6.04% observed.
- Time for the grid to reach 2.93% under 1/sqrt(t) from 504 s: 504 x 4.25 = ~2,140 s.
- Behaviour (histogram arm): blocks 1-2 identical in all scans (TI 1.06 s, 0.49 s); SD of log10 TI across scans 0.16, 0.19, 0.15, 0.15 at blocks 3-6.
- Correlation of TI with running estimate: histogram r = -0.12, LSTM r = +0.78 (report, section on memory ablation).
- Error by realised T1 band (histogram): 2.6% short, 3.2% long (report).
- Twin/ladder: 0.034 s/step analytic to 1.03 s/step full Bloch (CPU, 14 spheres); cached rungs 8x faster than matched full Bloch; cached water vs full Bloch T1 0.12% mean, 1.15% worst.
- 14-sphere plate (negative result): Cramer-Rao fixed 4.70% vs RL cached-best 9.44% (report, Run A).

## Novelty check (20 Oct 2026)

The report's claim stands with one wording fix.
Report wording: first RL agent for adaptive qMRI, a policy that conditions each acquisition on the current fitted tissue parameters and is trained and evaluated on fitted-T1 error.
No closed-loop RL policy for quantitative MRI was found.
But the report's clause "prior RL in MRI targets non-quantitative objectives" is no longer accurate.
Two RL papers optimise fixed quantitative protocols offline: SCREENER (2) and RL flip-angle schedules for MRF (3).
Neither adapts during the scan.
Abstract wording: "to our knowledge, the first closed-loop RL policy for quantitative MRI", and cite (2,3) as offline RL protocol design.
Also adjacent, not MRI-applied: policy-based Bayesian experimental design (Deep Adaptive Design, RL-DAD); not needed in 750 words.

## References (Vancouver; order provisional until the body is written)

1. Beracha I, Seginer A, Tal A. Adaptive model-based magnetic resonance. Magn Reson Med. 2023;90(3):839-851. doi:10.1002/mrm.29688
2. Zheng T, Wang Z, Bray T, Alexander DC, Wu D, Zhang H. SCREENER: a general framework for task-specific experiment design in quantitative MRI. arXiv. 2024;2408.11834. doi:10.48550/arXiv.2408.11834
3. Zhong S, Chen Z, Chen Z. Optimize flip angle schedules in MR fingerprinting using reinforcement learning. arXiv. 2025;2511.19941. doi:10.48550/arXiv.2511.19941
4. Walker-Samuel S. Using deep reinforcement learning to actively, adaptively and autonomously control a simulated MRI scanner. Proc Intl Soc Mag Reson Med. 2019;27:0473.
5. Pineda L, Basu S, Romero A, Calandra R, Drozdzal M. Active MR k-space sampling with reinforcement learning. In: MICCAI 2020. Springer; 2020:23-33. doi:10.1007/978-3-030-59713-9_3
6. Russek SE, Boss MA, Charles HC, et al. MRI scanner characterization with the ISMRM/NIST system phantom. Proc Intl Soc Mag Reson Med. 2020;28:3384.
7. Castillo-Passi C, Coronado R, Varela-Mattatall G, Alberola-Lopez C, Botnar R, Irarrazaval P. KomaMRI.jl: an open-source framework for general MRI simulations with GPU acceleration. Magn Reson Med. 2023;90(1):329-342. doi:10.1002/mrm.29635
8. Schulman J, Wolski F, Dhariwal P, Radford A, Klimov O. Proximal policy optimization algorithms. arXiv. 2017;1707.06347. doi:10.48550/arXiv.1707.06347
9. Jones JA, Hodgkinson P, Barker AL, Hore PJ. Optimal sampling strategies for the measurement of spin-spin relaxation times. J Magn Reson B. 1996;113(1):25-34. doi:10.1006/jmrb.1996.0151

## Acknowledgements (not shown to reviewers; put code links here)

- MRISystemPhantom.jl: https://github.com/ArthurAllilaire/MRISystemPhantom.jl (docs: https://arthurallilaire.github.io/MRISystemPhantom.jl/dev/).
- Code to reproduce all results: https://github.com/ArthurAllilaire/RLxMRI.
- Funding: TBC.

## Open items

- [x] Choose a title.
- [ ] Confirm presenting author (ICR) and get approval from all co-authors.
- [ ] Funding line for Acknowledgements.
- [ ] Write body prose from the outline (target 750 words).
- [ ] Build Figs 1-4 and the preview figure from saved results (all local plotting).
- [ ] Final word and character counts in ECHO.

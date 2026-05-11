Subject: Weekly Progress Update — Arthur Allilaire FYP

Hi Wayne,

A more substantial update this week — E2 has produced clear (mostly negative) results, and I want to flag a timetable change before drafting Ch4.

Tasks completed this week:

- Closed out the 14-sphere E2 experiment with controlled fixed-schedule baselines. The best RL policy (V5) reaches 234% mean T1 MAPE at iso-budget vs the textbook log-spaced grid at 393% — a 1.7× headline win. About half of that gap is TR efficiency (matching a TR-shortened fixed grid closes most of it).
- Implemented a Cramér–Rao optimal fixed schedule as a theoretical anchor. CR-optimal achieves 221% MAPE — i.e. **the analytic non-adaptive optimum beats RL on the 14-sphere problem.** This was the right baseline to add; it dissolves the original "RL beats fixed schedule" claim.
- Built E2-tractability (sample 5 of 14 spheres per episode) to make adaptivity informative. Trained four RL variants (V9–V12). The strongest (V12) reaches 322% mean MAPE, beating CR-optimal at 421% — a 100 pp gap. The policy is also observation-conditional (TI choice correlates with running T1 estimate, Pearson r = −0.26).
- The decisive control: re-ran both V12 and CR-optimal through an oracle-initialised fitter (T1 grid constrained to ±1 octave around truth). Both land at 53% MAPE, gap +0.4 pp — within paired-seed noise. **V12's apparent advantage was a fitter-side artefact, not extra information extraction.**
- Diagnosed the residual MAPE floor. On the worst-failing spheres I plotted the SSE-vs-T1 landscape directly: the true T1 sits at the **76th percentile** of the SSE landscape with a wrong basin 5–20× lower in SSE. Any maximum-likelihood-by-SSE estimator (LM, dictionary matching, K-restart) is structurally unable to recover truth on this likelihood. The bottleneck is the magnitude-reconstruction + Gaussian-noise likelihood, not the policy.

Key figures (in repo):

- `runs/e2/e2_tractability_V12/sse_landscape/sse_landscape.png` — the SSE landscape plot. This is the most important figure of the week; it's what should have been checked before training the RL agents.
- `report_plots/E2_tractability_V9/v9_vs_fixed_anchors.png` — RL vs fixed schedules.
- `report_plots/E2.1/per_sphere_mape.png` — per-sphere T1 MAPE breakdown.

Honest reading of the C1 (adaptive sequence design) claim:

- V12 achieves the analytic Fisher-information ceiling that any non-adaptive schedule can deliver on this fleet.
- V12 produces observation-conditional schedules — measurable within-episode adaptivity (KS-significant TI distribution shift between all-long-T1 and all-short-T1 subsets).
- The residual ~270 pp MAPE is a property of the likelihood (magnitude reconstruction + 5% noise), not of the policy or the fitter.
- The defensible narrative is "RL recovers the Fisher-information optimum with an observation-conditional schedule; further headline gains require a different estimator class (Bayesian prior, joint multi-sphere fit, or phase-sensitive reconstruction) rather than more RL training."

Tasks planned this week:

- Decide between three options for the one remaining experimental lever: (a) phase-sensitive reconstruction with a noise model that doesn't destabilise PPO (an earlier attempt collapsed training); (b) Bayesian-prior fitter using the known sphere-distribution; (c) joint multi-sphere fit with shared noise σ. Discuss with Andreas at the Monday meeting.
- Start the Ch3 (digital twin + E0 baseline) draft — this chapter is unaffected by the E2 findings.
- Outline Ch4 around the C1 reframe above. The honest narrative is publishable and defensible even if the headline MAPE doesn't move.

Deviations from the timetable:

- E3 (MRF-style fingerprinting) was scheduled for weeks 6–7 (12–25 May). Given the E2 findings, MRF on this likelihood would inherit the same wrong-basin problem and produce no new insight. I am proposing to **replace E3 with one focused estimator-side experiment** (option a/b/c above) so Ch5 still has a concrete contribution. This is the only material slip vs the timetable in `progress_update_1.md`; the 1 June bullet-point draft and 12 June submission dates are unchanged.

Two things I'd value your steer on (no rush — can wait for next week's reply):

1. Is the C1 reframe ("recovers Fisher-information ceiling, observation-conditional, residual MAPE is likelihood-side") acceptable as a Ch4 contribution, or do you think the report needs a clean headline MAPE win to be safe?
2. For the assessment criteria, would you class the SSE-landscape diagnostic as a Ch4 result or a Ch5 limitation? I'm currently planning to put it in Ch4 as the central "why the apparent RL win wasn't real" figure.

Many thanks,
Arthur

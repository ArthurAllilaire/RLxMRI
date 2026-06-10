# E2 history ablation — action-history observation vs recurrent policy (560 s)

Status: planned 10 June 2026. Goal: the lowest-error 560 s result before the
12 June submission, and a clean three-way *memory mechanism* ablation for the
report (Ch4/A2, addresses C1).

Companion to `section_multi_fidelity.md` (§"560 s GPU runs — eval & diagnose"
has the eval protocol and the controls these runs compare against).

---

## 1. Motivation

The E2 environment is a POMDP. The current observation
(`_e2_observation`, `julia/rl/e2.jl:492`) is

```
[ log10(T1_est) per sphere ; (optional σ-channel) ; t_frac, n_frac, 1 ]
```

The running T1 estimates are a **lossy summary of the acquisition history**:
two very different sampled-TI sets can produce similar T1_est vectors, but the
information value of the *next* TI depends on which TIs the agent has already
spent budget on. The policy currently cannot see that.

The σ-channel (per-sphere `log10(σ_T1/T1_est)` from the LM fit) was the first
attempt at exposing it — a *learned-estimator summary* of history. The 560 s
result says it did not help:

| 560 s run | global best (12-ep confirm, seed 510000) |
|---|---|
| no-σ control `mf_runB_5sphere_560s_gpu` | **3.45 %** MAPE / p90 6.03 % / 75 % |
| σ treatment `mf_runB_5sphere_sigma_560s_gpu` | 4.04 % / p90 6.94 % / 75 % |

A plausible reading: the LM-fit σ is noisy/degenerate early in the episode
(< 2 measurements → no fit at all; sentinel 0 = "fully uncertain"), so the
channel adds noise where it should add information. This plan tests the two
remaining memory mechanisms:

- **A. State augmentation** — put the raw action history in the observation
  (here: a log-TI coverage histogram) and keep plain PPO. Makes the POMDP
  approximately Markov by construction.
- **B. Recurrence** — keep the base observation and let an LSTM hidden state
  learn its own summary (RecurrentPPO).

Report framing: *three ways to give a sequential-design agent memory —
estimator-derived summary (σ), task-informed sufficient statistic (TI
coverage), learned summary (LSTM) — compared at a fixed 560 s scan budget.*
That is a defensible ablation table regardless of which one wins.

---

## 2. Option A — log-TI coverage histogram in the observation

### Design

- The fitter (`fit_t1_generalized_ir`) consumes the **set** of (TI, mag) pairs
  per sphere — order-invariant — so a permutation-invariant encoding matches
  the true sufficient statistic better than an ordered action list, and is
  far more compact (12 dims vs 20×2 padded).
- Bins: `ti_hist_bins = 12`, uniform in **log TI** between the live action
  bounds `TI ∈ [0.010, 3.0] s` (`e2_action_lo/hi`) — i.e. uniform in the
  agent's own log-TI action coordinate `u = log(TI/lo)/log(hi/lo)`
  (2.48 decades → ~0.21 decades/bin).
- Bin value: count of **executed** blocks whose TI falls in the bin,
  normalised by `max_blocks` (20) → each bin in [0, 1]. Executed (post-repair)
  TIs come from `env.block_TIs` — already stored per sphere, identical across
  spheres, read sphere 1. Using executed rather than requested TIs means the
  channel stays truthful under TR-lift/TE-clamp repairs.
- TR history is **not** encoded in v1: `t_frac` already carries spent scan
  time, and TR's effect on information is secondary to TI coverage. Revisit
  only if R1 stalls.

### Code changes (all small, no new deps)

1. `julia/rl/e2.jl` — `include_ti_history::Bool` + `ti_hist_bins::Int` fields
   (constructor kwargs, defaults `false`/`12`); `e2_obs_dim` `+ ti_hist_bins`
   when enabled; histogram block in `_e2_observation`. Reset already clears
   `block_TIs` → histogram starts at zeros.
2. `python/qalibremd_gym/env_e2.py` — pass-through kwargs
   `include_ti_history`, `ti_hist_bins` to the Julia ctor (mirror
   `include_sigma`).
3. `python/e2_config.py` — `--include-ti-history` (and `--ti-hist-bins`,
   default 12) → `env_kwargs` → saved in `run_config.json`, so
   `eval_e2/diagnose_e2 --from-run` inherit it with zero further changes.

VecNormalize handles the scaling (channel already ≈[0, 1]).

### Diagnostics bonus

`diagnose_e2.py` already records per-block TI choices. With this channel the
adaptivity question becomes directly checkable: *does the policy avoid
re-sampling already-covered bins?* Plot TI choice vs current bin occupancy.

---

## 3. Option B — RecurrentPPO (LSTM)

### Design

- `sb3-contrib==2.8.0` (pin to match `stable_baselines3 2.8.0`; **not
  currently installed**).
- `RecurrentPPO("MlpLstmPolicy", ...)`, base observation (no σ, no histogram)
  — keeps the ablation clean: memory via recurrence *instead of* state
  augmentation, not on top of it.
- Same PPO hyperparameters as the 560 s control where they transfer
  (`n_steps 512, batch 64`); leave LSTM size at the sb3-contrib default
  (256) for v1.

### Code changes (IMPLEMENTED 10 Jun)

As built: `sb3-contrib 2.8.0` installed and pinned in
`python/requirements.txt`. `build_model(..., recurrent=)` and a class-aware
`load_policy()` live in `e2_train_common.py`; `rollout_eval` threads LSTM
`state`/`episode_start` through `model.predict` (plain PPO accepts and
ignores both, so one loop serves both classes — this covers every trainer
eval site: screening, global-best confirmation, stage probes, lookahead).
`train_e2_mf.py --recurrent` switches all three `build_model` sites (cold
start, optimizer-reset stage switch, lookahead clone) and records
`"recurrent": true` in `run_config.json`; `eval_e2.py`/`diagnose_e2.py`
infer it via `--from-run` (or take an explicit `--recurrent`) and run the
same stateful predict loop. Verified by unit test (build → learn →
weights-only clone incl. 8 LSTM tensors → save/load → stateful rollout for
both classes) plus the S2 trainer smoke below.

### Original change list (for reference — budget half a day)

1. `python/train_e2_mf.py` — `--recurrent` flag → model class switch; **two
   known risk spots**:
   - the multi-fidelity stage-switch logic rebuilds PPO with fresh Adam and
     copies policy weights — verify the copy covers LSTM weights and shapes
     match across stages (same obs/action dims, so it should);
   - every eval helper (`screening eval`, global-best confirmation,
     lookahead/decision rollouts) calls `model.predict(obs, deterministic=…)`
     — RecurrentPPO needs `state` + `episode_start` threading. Factor one
     stateful `predict_episode` helper and use it everywhere.
2. `python/eval_e2.py`, `python/diagnose_e2.py` — same stateful predict loop;
   pick the model class from a `recurrent` flag in `run_config.json` so
   `--from-run` keeps working.

### Abort criterion

If the RecurrentPPO smoke (S2 below) is not training cleanly (reward moving,
no shape/state errors, stage 0→1 switch survives) by **~3 h after starting
the integration**, drop option B and reallocate its overnight slot to a
second option-A variant (see run matrix). The deadline does not allow
debugging LSTM training dynamics.

---

## 4. Run matrix and schedule

All runs: 560 s budget, GPU box, same flags as `mf_runB_5sphere_560s_gpu`
(the 3.45 % control) except where noted — `--fix-te --log-ti-action`, no
`--learn-alpha`, T15, σ=50, spheres 1,3,6,8,14, `roi_radius 1`, seed 0 /
eval-seed 500000.

| Slot | When (10–11 Jun) | Run | Out dir |
|---|---|---|---|
| S1 | ~~skipped~~ — option A verified by direct env smoke (obs dim, bin placement, reset clearing, back-compat); R1 launched directly and monitored live | — | — |
| S2 | ~~done 10 Jun~~ — tiny `--recurrent` MF run (analytic→cached3, 16×8 grid) crossed a real fidelity switch with global-best saves and exit 0; `eval_e2 --from-run` then loaded and rolled the recurrent policy with no flags needed | — | — |
| R1 | tonight, 9 h | **A full**: histogram | `runs/e2/mf_runB_5sphere_hist_560s_gpu` |
| R2 | tonight, 9 h | **B full**: LSTM (or fallback: A with `--ti-hist-bins 8` + `--include-sigma` combo) | `runs/e2/mf_runB_5sphere_lstm_560s_gpu` |
| C1/C2 | tomorrow day (optional) | second seed of the winner, or eval-only | — |

Smoke pass criteria (S1/S2): obs dim correct, run survives ≥1 fidelity
switch, episode reward not flat at stage 0, `run_config.json` round-trips
through `eval_e2 --from-run` on the smoke checkpoint.

**Hard cutoffs:** overnight runs must launch by ~22:00 on 10 Jun (done
~07:00); any tomorrow-day run must end by ~18:00 on 11 Jun to leave eval +
write-up time before the 12 Jun submission. If only one thing fits, it is R1.

### R1 command (A, full)

```bash
PYTHON_JULIAPKG_PROJECT="$PWD/python/julia_runtime_gpu" \
PYTHON_JULIAPKG_OFFLINE=yes PYTHON_JULIACALL_HANDLE_SIGNALS=yes \
PYTHON_JULIACALL_THREADS=3 JULIA_NUM_THREADS=3 \
PYTHONUNBUFFERED=1 python -u python/train_e2_mf.py \
  --out runs/e2/mf_runB_5sphere_hist_560s_gpu \
  --multi-fidelity --mf-plan analytic,cached3,cached,full3,full \
  --reward-mode delta_log_mape --mape-alpha 1.0 \
  --fix-te --log-ti-action --include-ti-history \
  --n-envs 1 --field T15 --time-budget 560 --max-blocks 20 \
  --subset-size 5 --forced-sphere-indices 1,3,6,8,14 \
  --t1-sampler linear_uniform_range \
  --pose-mode inplane_jitter --translation-sigma-mm 2.0 \
  --rotation-sigma-rad 0.05 --roi-radius 1 \
  --use-gpu \
  --train-seed 0 --eval-seed 500000 \
  --mf-budget-hours 9 --mf-full-reserve-frac 0.20 \
  --mf-min-steps 4096,8192,8192,8192,0 \
  --mf-max-steps 20000,160000,160000,80000,300000 \
  --n-steps 512 --batch-size 64 \
  --eval-interval 10000 --eval-episodes 20 \
  --mf-decision-rollouts 4 --mf-probe-episodes-full 4 \
  --mf-global-best-episodes 12 \
  --mf-use-lookahead --mf-lookahead-rollouts 1 \
  --mf-lookahead-margin 1.15 --mf-slope-collapse-frac 0.25 \
  2>&1 | tee runs/e2/mf_runB_5sphere_hist_560s_gpu/run.log
```

R2 is identical with `--include-ti-history` replaced by `--recurrent` and the
out dir swapped. Smokes S1/S2 use the same commands with
`--mf-budget-hours 1.5` and `_smoke` out dirs.

---

## 5. Eval protocol and success criteria

Identical to the 560 s block in `section_multi_fidelity.md`: for each finished
run, `eval_e2 --from-run` on `global_best/best_policy.zip`, 24 episodes,
`--roi-radius 1`, at **seed 500000** (baseline-comparable; overlaps the
training screening seed — same honesty caveat as the existing runs) **and
seed 600000** (strictly held-out), plus `diagnose_e2` at 500000.

Comparison set (all 560 s, 5-sphere):

| Policy | MAPE | p90 | success |
|---|---|---|---|
| no-memory control (global best) | 3.45 % | 6.03 % | 75 % |
| σ-channel | 4.04 % | 6.94 % | 75 % |
| `log_grid_trmatched` baseline | 5.71 % | — | 33.3 % |
| `cr_optimal` baseline | 7.34 % | — | 12.5 % |
| **A: TI-coverage histogram** | ? | ? | ? |
| **B: LSTM** | ? | ? | ? |

- **Primary success:** beat the 3.45 % control on the 24-episode evals (both
  seeds pointing the same way).
- **Secondary (report-worthy even if MAPE ties):** diagnose shows the
  histogram policy actively avoids covered TI bins — direct evidence of
  history-conditioned behaviour, which the σ-run never showed.
- **Null result is still a result:** if neither beats 3.45 %, the table reads
  "T1_est + budget is already a sufficient statistic at this task scale" —
  one paragraph, still publishable as an ablation.

---

## 6. Risks

- **LSTM × multi-fidelity interaction** (stage-switch weight cloning, eval
  state threading) is the big unknown — hence the smoke + 3 h abort rule.
- LSTM sample efficiency: 9 h may simply be too little; flag in the report if
  the learning curve is still rising at cutoff.
- Histogram channel + VecNormalize: zeros-heavy early bins are fine (running
  stats), but check the smoke's normalised obs aren't saturating.
- Seed-500000 overlap with training screening: mitigated by the seed-600000
  held-out eval, as in the main runbook.

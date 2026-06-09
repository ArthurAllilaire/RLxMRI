# Multi-Fidelity Curriculum Training

Todo:
1. How often do we check the MAPE on full Bloch? Could we probe the layer below
   full instead of full itself to cut probe cost?

Note: `fidelity_history.json["wall_s"]` is an absolute Unix epoch timestamp,
not a relative duration. `python/plot_mf_curriculum.py` subtracts the first
recorded event timestamp, i.e. the stage-0 wallclock origin, before plotting.

> **Challenge addressed:** C2 — *scalable simulation-in-the-loop RL*.
> **Novelty:** a literature-grounded, bias-aware rule for *when* to switch
> simulator fidelity during RL training, specialised to adaptive qMRI sequence
> design. **Quantified benefit:** final full-Bloch MAPE reached at a fraction of
> the single-fidelity wallclock (the money-plot, §5).

## 1. Motivation — the cost wall

E2 trains a PPO agent with the KomaMRI Bloch solver *in the loop*: every action
(a choice of TI/TE/TR/α) triggers a full multi-shot simulation and 2-D
reconstruction. The per-step cost scales as `Npe · TR · n_spins`, which at the
1 mm / 64×32 evaluation configuration is **~2.3–2.5 s/step on CPU — roughly 8
days for a 300k-step run**. A conventional single-fidelity curriculum (train
everything on the full simulator) is therefore infeasible at the compute budget
available for this project.

The simulator, however, is not one thing. We expose a **ladder of fidelities** of
the *same* forward model, trading bias against cost:

| Fidelity | What it computes | Per-step cost | Bias vs full Bloch |
|---|---|---|---|
| `analytic` | per-sphere signal from the closed form the fitter inverts; no Koma | ~µs–0.4 ms | noise-limited only; **no recon, no water, no B0** crosstalk |
| `dry` *(optional)* | full Bloch on the spheres only (no background water) | ≈ sphere-only Bloch | real recon/B0/noise, **zero water crosstalk** (≈ +2.4% MAPE) |
| `cached` | Bloch on the spheres + cached-Koma water template (see `water_cache.md`) | ~8× faster than full | water approximated to the **T1-grid floor (0.24% T1)** |
| `full` | full Bloch on spheres + water — the target | full | ground truth |

Each fidelity has a physical **accuracy floor**: the analytic model bottoms out
at its noise floor, the cached model at the T1-grid floor, the full model at the
true noise. The curriculum idea is to *learn cheaply at low fidelity, then warm-
start up the ladder*, spending expensive full-Bloch steps only where they buy
accuracy a cheaper model cannot.

## 2. The hard part — *when* to switch

Warm-starting between fidelities is mechanically trivial (carry the policy
weights). The research question is **when** to promote. The naive answer —
"train N steps per stage" — wastes budget and is impossible to justify.

The key correctness insight (and the central design decision) is:

> **The signal that triggers a switch must be improvement in held-out
> *full-simulator* validation, *not* improvement in the current cheap
> simulator.**

A biased cheap model can keep improving its *own* score long after the real
(full-sim) score has plateaued — indeed, it can improve its own score by
*exploiting its bias*. This is precisely the failure mode observed in E1, where
the agent collapsed to a degenerate fixed policy that exploited the noiseless
fitter rather than designing informative sequences (see `docs/E1_RESULTS.md`).
We therefore probe the **full-Bloch** environment periodically *even during the
cheap stages* — bounded to a few episodes at a coarse cadence, this is the honest
price of a bias-aware switch.

## 3. Literature basis

The switch rule adapts two ideas:

- **Sifaou & Simeone (2025), *Multi-Fidelity Hybrid RL via Information Gain
  Maximization*** (arXiv:2509.14848). Their principle: select the fidelity with
  the best *expected information gain about the target policy per unit simulation
  cost*, gated by a budget-adaptive threshold that pushes toward high fidelity as
  cheap-sim information stops being worth its bias. Their full machinery (a
  CQL/bootstrap-ensemble posterior over policies) is an offline-online algorithm
  too heavy for our on-policy PPO setting, so we borrow the *principle*, not the
  estimator.
- **Cutler et al. (2015), *RL with Multi-Fidelity Simulators*.** The classic
  switch-up/switch-down rule: train at low fidelity until the policy is good
  enough; if the low-fidelity optimum is *invalid at higher fidelity*, that
  fidelity is exhausted.

**Honesty note (important for the assessment):** we do **not** implement the
paper's posterior information gain. The quantity our code computes is named
`target_slope_per_cost` — the improvement in the full-sim score per second of
wallclock at the current fidelity. We describe it as an *IG-per-cost-inspired
heuristic*, never as "the information gain."

## 4. The switch rule (as implemented)

At each decision point (every `--mf-decision-rollouts` PPO rollouts) we probe the
policy on the current-fidelity env and the full-Bloch env (each capped at
`--mf-probe-episodes-full`, default 4). Define the full-sim score
`P_H = −log₁₀(clamp(MAPE_full, floor, 1))` (higher is better; the clamp models
the floor and gives a usable sub-1% gradient). We **promote** out of the current
cheap fidelity when **any** of the following holds (after a `min_steps` gate):

1. **Target plateau** *(primary).* `P_H` has stopped improving over the last
   `plateau_window` decision points — the cheap sim has given what it can.
2. **Ranking breakdown** *(Cutler switch-down).* The Spearman correlation between
   the cheap-sim MAPE and the full-sim MAPE across recent checkpoints falls below
   a guardrail (0.7 for `analytic`, 0.8 for `cached`) — the cheap sim no longer
   *ranks* policies like the target. This is only evaluated once ≥4 decision
   checkpoints exist (below that, rank correlation is statistical noise).
3. **Bias intolerance.** The mean (or p90) MAPE gap between cheap and full sim
   exceeds a tolerance (1% / 2% absolute).
4. **Budget reserve.** Remaining wallclock has fallen to the fraction reserved
   for the final full-Bloch stage (`--mf-full-reserve-frac`, default 30%), which
   overrides everything to guarantee the target fidelity gets enough compute.

Safety caps `min_steps`/`max_steps` bound each stage. On every promotion we
record the **fidelity gap** — the full-sim MAPE of the warm-started policy
*before* any training at the new stage, minus the previous stage's final full-sim
MAPE — as the transfer diagnostic. The decision logic is isolated in
`python/mf_switch.py` (pure, unit-tested in `python/tests/test_mf_switch.py`); the
PPO/Julia orchestration is in `python/train_e2_mf.py`.

A note on what is **deferred to v2**: the truly faithful version of the paper's
rule would compare improvement-per-cost *across* fidelities, but the next stage's
learning slope is not observable without briefly training there. v1 uses only the
current fidelity's slope against the full-sim target; the cross-fidelity
"info-per-cost crossover" needs a short probe fine-tune and is future work.

## 5. Evaluation (money-plot)

We compare three runs at the **same total wallclock budget Γ**:

1. **Criterion curriculum** — the proposed switch rule (`--schedule criterion`).
2. **Fixed-budget curriculum** — equal wallclock thirds (`--schedule fixed`),
   ablating the rule while keeping the curriculum.
3. **Full-Bloch only** — the conventional single-fidelity baseline.

`python/plot_mf_curriculum.py` plots held-out full-Bloch MAPE against **cumulative
wallclock seconds** (not steps — the whole point is that fidelities cost
differently). Reported numbers: final MAPE, wallclock-to-reach-5%-MAPE, and the
measured fidelity gap / rank-correlation at each switch as evidence that the rule
promotes at the right time. An optional fourth run inserts the `dry` stage to test
whether a real-recon-but-no-water bridge between `analytic` and `cached` helps.

*(Figure and table to be filled from the run outputs.)*

## 6. Reproduce

```bash
# criterion curriculum
python python/train_e2_mf.py --out runs/e2/mf_criterion --multi-fidelity \
  --mf-plan analytic,cached,full --reward-mode delta_log_mape \
  --fix-te --learn-alpha --n-envs 1 \
  --field T15 --mf-budget-hours 24 --schedule criterion

# fixed-schedule ablation (same budget)
python python/train_e2_mf.py --out runs/e2/mf_fixed --multi-fidelity \
  --mf-plan analytic,cached,full --reward-mode delta_log_mape \
  --fix-te --learn-alpha --n-envs 1 \
  --field T15 --mf-budget-hours 24 --schedule fixed

# full-Bloch-only baseline (existing single-fidelity trainer)
python python/train_e2.py --out runs/e2/full_only \
  --forward-model bloch --water-model bloch --reward-mode delta_log_mape \
  --fix-te --learn-alpha --field T15 --timesteps <budget-matched>

# overlay
python python/plot_mf_curriculum.py \
  runs/e2/mf_criterion:Criterion runs/e2/mf_fixed:Fixed runs/e2/full_only:Full-only \
  --out runs/e2/mf_moneyplot.png
```

CPU threaded V2 example (2 subprocess envs × 6 Julia/Koma threads each):

```bash
PYTHON_JULIAPKG_OFFLINE=yes PYTHON_JULIACALL_HANDLE_SIGNALS=yes \
PYTHON_JULIACALL_THREADS=6 JULIA_NUM_THREADS=6 \
PYTHON_JULIAPKG_EXE=~/.julia/juliaup/julia-1.11.9+0.x64.linux.gnu/bin/julia \
  PYTHONUNBUFFERED=1 python -u python/train_e2_mf.py \
    --out runs/e2/mf_v2_runA_cpu \
    --multi-fidelity --mf-plan analytic,cached3,full3,full \
    --reward-mode delta_log_mape --mape-alpha 1.0 \
    --fix-te --learn-alpha \
    --n-envs 2 \
    --field T15 --time-budget 240 --max-blocks 20 \
    --mf-budget-hours 24 --mf-min-steps 0 --mf-max-steps 200000 \
    --n-steps 512 --batch-size 64 \
    --mf-use-lookahead --mf-lookahead-rollouts 1 \
    --mf-lookahead-margin 1.15 --mf-slope-collapse-frac 0.25 \
    2>&1 | tee runs/e2/mf_v2_runA_cpu/run.log
```

## 7. Three-phase extension plan

The current implementation is intentionally conservative: it probes the
held-out full-Bloch target while training at cheap fidelity and promotes when
the full-sim score plateaus, when cheap/full ranking breaks down, when the
cheap/full MAPE gap becomes too large, or when the wallclock reserve for the
final full stage is reached. This already captures the most important safety
principle from multi-fidelity RL: the cheap simulator must not be allowed to
validate itself. The weakness is that the logged
`target_slope_per_cost = ΔP_H / wallclock` is not yet part of the switch
decision. V2 should make that cost-sensitive quantity operational while keeping
the current plateau rule as a robust fallback.

### Phase A — lookahead slope per cost

**Goal.** Replace the current "plateau only" promotion trigger with an
observed improvement-per-cost comparison, activated only when there is evidence
that the current fidelity is becoming unproductive. This is the direct practical
V2.

The rule should be:

1. Continue to train normally at the current fidelity.
2. At each decision point, compute the recent current-fidelity target slope:
   `s_current = ΔP_H / seconds_current`, where `P_H` is the held-out full-Bloch
   score.
3. If the slope has collapsed relative to its own earlier slope, or if the
   existing relative plateau trigger is near firing, clone the policy and run a
   short lookahead at the next fidelity.
4. Evaluate the lookahead clone on the same held-out full-Bloch probe seeds and
   compute `s_next = ΔP_H / seconds_next`.
5. Promote only when the next fidelity is now more cost-effective:
   `s_next > margin * s_current`, with a small margin to avoid switching on
   noise. The old relative `P_H` plateau remains a fallback promotion trigger.

This grounds the decision in Sifaou and Simeone's information-gain-per-cost
principle without importing their full hybrid offline-online machinery. Their
MF-HRL-IGM algorithm selects a simulator level by maximizing information gain
per unit simulation cost, using bootstrapped policies to approximate uncertainty
over the optimal policy. For E2/PPO, we do not have a posterior over policies,
so `ΔP_H / second` is an empirical proxy: it measures how much a training chunk
at a given fidelity improves the target high-fidelity metric per unit wallclock.

The trigger should be deliberately sparse. Running a next-fidelity lookahead at
every decision point would erase the compute advantage of the curriculum. The
lookahead is only justified once current-fidelity evidence says "this simulator
may be exhausted." This matches the spirit of Cutler et al.'s switch-up logic:
cheap models are useful until they stop producing target-fidelity progress or
their induced policy becomes invalid at higher fidelity.

**Why keep the old plateau.** The slope-per-cost estimate is noisy because
full-Bloch probes are intentionally small. The existing relative plateau test is
dimensionless and hardware-insensitive; it should remain as a fallback with a
reason such as `target_plateau`. The new trigger should produce distinct
reasons such as `slope_collapse` and `lookahead_better`, so the run history
shows whether a switch was caused by the V2 criterion or by the original guard.

**Evaluation.** Phase A should be compared against the current criterion run,
the fixed-budget curriculum, and full-only training. The key diagnostics are:
final full-Bloch MAPE, wallclock-to-threshold MAPE, number of lookaheads, cost
spent on lookaheads, switch reasons, and the full-sim MAPE gap at each switch.

### Phase B — pooled multi-fidelity samples

**Goal.** Evaluate a qualitatively different approach: rather than committing to
one fidelity at a time and warm-starting between stages, collect samples from
multiple fidelities and use them together during learning.

This phase is motivated by the multi-fidelity RL literature cited by Sifaou and
Simeone: multi-fidelity policy-gradient methods, control-variate approaches,
and GP-based model/model-free algorithms all treat cheap simulations as useful
but biased evidence, not merely as pretraining stages. In E2 terms, analytic,
cached-water, coarse-water, and full-Bloch rollouts could all contribute to a
single learner, provided their bias is corrected or bounded.

There are three plausible variants:

1. **Replay-buffer/off-policy variant.** Move away from pure PPO for this
   ablation and use an off-policy actor-critic or CQL-style learner with a
   replay buffer labelled by fidelity. Full-Bloch transitions receive the
   highest trust; lower-fidelity transitions are downweighted or corrected.
2. **Control-variate variant.** Estimate high-fidelity return or advantage as a
   cheap-fidelity estimate plus a correction learned from paired high-fidelity
   probes. This is statistically attractive because the cheap simulator can
   reduce variance while sparse full-Bloch samples control bias.
3. **PPO-compatible weighted-batch variant.** Keep PPO, but build training
   batches that include rollouts from multiple envs and attach fidelity-specific
   advantage weights or penalties. This is the least invasive but also the
   least theoretically clean, because PPO assumes on-policy data from the
   current environment.

This phase is not a replacement for Phase A; it is a comparison arm. The
question is whether E2 benefits more from **stagewise curriculum selection** or
from **joint biased-sample reuse**. The correct baseline is therefore the Phase
A lookahead curriculum, not the current V1 only.

**Evaluation.** Report the same wallclock-normalised full-Bloch MAPE curves, but
also include the fraction of updates contributed by each fidelity, the measured
cheap/full bias over time, and whether pooled learning destabilises PPO. A
negative result would still be useful: it would justify the simpler curriculum
if pooling introduces policy-gradient bias that is not worth the sample reuse.

### Phase C — BO-informed V2+

**Goal.** Use multi-fidelity Bayesian optimization to improve the fidelity
selection rule and reduce ad hoc thresholds.

Song, Chen, and Yue's MF-MI-Greedy framework is not an RL algorithm, but it is
highly relevant to the fidelity-selection subproblem. They model several
information sources with different costs as correlated outputs, use shared
latent Gaussian-process structure to let low-fidelity observations reduce
uncertainty about the target fidelity, and choose low-fidelity queries by
cost-sensitive mutual information. Two ideas transfer cleanly to E2:

1. **Model cheap/full coupling explicitly.** Instead of using fixed
   rank-correlation and bias thresholds, fit a lightweight surrogate over
   recent decision records:
   `policy_state_features, fidelity -> predicted full-Bloch MAPE change`.
   At minimum this could be a GP over scalar summaries such as current MAPE,
   recent slope, bias, p90 gap, stage index, and fidelity cost. The surrogate
   should predict target-fidelity improvement and uncertainty after another
   chunk at each candidate fidelity.
2. **Choose lookaheads by value of information per cost.** Phase A runs a
   next-fidelity lookahead only near collapse. Phase C can ask a richer
   question: which fidelity-query would most reduce uncertainty about the best
   next training stage per second? This converts the lookahead from a hand-coded
   "next level only" probe into a BO-style acquisition decision.

The practical V2+ algorithm would maintain a small decision-level dataset:

```text
stage, fidelity, steps, wall_s, P_H_before, P_H_after,
MAPE_f, MAPE_H, p90_f, p90_H, rank_corr, sec_per_step
```

From this, fit a surrogate for `ΔP_H` and uncertainty by fidelity. At decision
time, evaluate an acquisition such as:

```text
expected_target_gain(fidelity) / expected_cost(fidelity)
+ uncertainty_bonus(fidelity) / expected_cost(fidelity)
```

This is not full MF-MI-Greedy and should not be presented as such. It is a
BO-inspired controller for the simulator-selection layer, using the E2
full-Bloch probes as the target-output observations. The expected advantage is
that it can distinguish two cases that the Phase A rule may conflate:

- the current fidelity has low slope because the policy is genuinely near its
  target optimum;
- the current fidelity has low slope because another fidelity would now provide
  more informative gradients.

**Evaluation.** Phase C should be treated as an ablation after Phase A is
stable. It adds modelling assumptions, so it must beat the simpler lookahead
rule on wallclock-to-MAPE and not merely produce more sophisticated logs.

### Proposed progression

Implement Phase A first. It is closest to the current code and directly fixes
the documentation/code mismatch around `target_slope_per_cost`. Run it as the
main V2. Phase B should be a separate comparison because it changes the learning
problem from stagewise curriculum to multi-source data fusion. Phase C should
only follow once enough Phase A runs exist to provide a meaningful
decision-level dataset for a BO-style controller.

### References for the extension

- Sifaou and Simeone, **Multi-Fidelity Hybrid Reinforcement Learning via
  Information Gain Maximization**, arXiv:2509.14848, 2025. Relevant because it
  frames fidelity selection as information gain per unit simulation cost under
  a fixed budget, using bootstrapped hybrid RL to estimate uncertainty over the
  best policy.
- Song, Chen, and Yue, **A General Framework for Multi-fidelity Bayesian
  Optimization with Gaussian Processes**, AISTATS 2019. Relevant because it
  models multiple information sources as correlated outputs and chooses
  low-fidelity queries by cost-sensitive mutual information about the target
  fidelity optimum.
- Cutler, Walsh, and How, **Reinforcement Learning with Multi-Fidelity
  Simulators**, ICRA/TRO-era multi-fidelity RL work. Relevant because it
  motivates switch-up/switch-down logic: cheap simulators should be used while
  informative, but invalid cheap-fidelity optima must be detected against
  higher fidelity.
- Khairy and Balaprakash, **Multi-fidelity reinforcement learning with control
  variates**, Neurocomputing 2024; Liu et al., **Multi-fidelity policy gradient
  algorithms**, arXiv:2503.05696. Relevant to Phase B because they point toward
  using samples from multiple simulators jointly rather than only as sequential
  curriculum stages.

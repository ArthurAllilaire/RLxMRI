# Multi-Fidelity V2 Implementation Plan

This document converts the literature-facing three-phase plan in
`multi_fidelity.md` into concrete code changes. Phase A is the implementation
target for V2; Phases B and C are comparison/future-work branches.

Literature anchors:

- Sifaou and Simeone 2025: fidelity selection by information gain per unit cost.
- Song, Chen, and Yue 2019: cost-sensitive mutual information for
  multi-fidelity Bayesian optimization.
- Cutler, Walsh, and How: switch-up/switch-down multi-fidelity RL.
- Khairy and Balaprakash 2024 / Liu et al. 2025: pooled multi-fidelity samples,
  control variates, and policy-gradient variants.

## Phase A — lookahead slope per cost

### Objective

Use the already logged target slope per cost in the switch decision. Keep the
current relative full-sim plateau, rank breakdown, bias intolerance, and budget
reserve as fallbacks.

The new decision path is:

```text
full-sim probe on current policy
update decision history
compute current slope per cost
if slope collapsed or plateau is near:
    clone policy
    train clone briefly at next fidelity
    evaluate clone on held-out full sim
    compute lookahead slope per cost
    promote if lookahead slope beats current slope by a margin
else:
    use existing fallbacks
```

### New CLI flags

Add to `python/train_e2_mf.py`:

```text
--mf-use-lookahead
--n-envs INT                         default 1
--mf-lookahead-rollouts INT          default 1
--mf-lookahead-probe-episodes INT    default same as --mf-probe-episodes-full
--mf-lookahead-margin FLOAT          default 1.15
--mf-slope-collapse-frac FLOAT       default 0.25
--mf-near-plateau-frac FLOAT         default 2.0
--mf-min-lookahead-points INT        default plateau_window + 1
--mf-lookahead-max-frac FLOAT        default 0.10
```

Meanings:

- `mf-use-lookahead`: enable Phase A behaviour.
- `n-envs`: number of SB3 vectorized training workers. `n_envs=1` keeps one
  Julia runtime warm; `n_envs>1` uses one subprocess and one Julia runtime per
  worker. PPO rollout size is `n_steps × n_envs`.
- `mf-lookahead-rollouts`: short train budget for the cloned next-fidelity
  policy, measured in PPO rollouts.
- `mf-lookahead-margin`: require next slope to exceed current slope by this
  factor.
- `mf-slope-collapse-frac`: trigger lookahead when recent slope is below this
  fraction of the stage's earlier best/median slope.
- `mf-near-plateau-frac`: trigger lookahead when the relative plateau metric is
  within this multiple of `mf_plateau_delta`.
- `mf-lookahead-max-frac`: cap total lookahead wallclock as a fraction of the
  global budget.

### Pure switch-logic changes

Update `python/mf_switch.py`.

Add fields to `SwitchThresholds`:

```python
slope_collapse_frac: float = 0.25
near_plateau_frac: float = 2.0
min_lookahead_points: int = 4
```

Add helper functions:

```python
def relative_target_gain(p_hist: list[float], window: int) -> float:
    ...

def slope_collapse(
    slope_hist: list[float],
    collapse_frac: float,
    min_points: int,
) -> tuple[bool, str]:
    ...

def should_run_lookahead(
    state: SwitchState,
    th: SwitchThresholds,
    slope_hist: list[float],
) -> tuple[bool, str]:
    ...
```

Expected behaviour:

- If there are not enough decision points, do not look ahead.
- If recent slope is non-finite or no previous finite slope exists, do not look
  ahead.
- Trigger `slope_collapse` when recent finite slope is below
  `collapse_frac * max(previous finite slopes)`.
- Trigger `near_plateau` when relative target gain is positive but less than
  `near_plateau_frac * plateau_delta`.
- Existing `decide_switch` remains available and still provides the fallback
  promotion reason.

Add unit tests to `python/tests/test_mf_switch.py`:

- slope collapse triggers only after minimum points.
- near plateau triggers before hard plateau.
- no lookahead on insufficient history.
- fallback plateau still promotes without lookahead.

### Trainer changes

Update `FidelitySwitchCallback` in `python/train_e2_mf.py`.

Add constructor inputs:

```python
next_spec: FidelitySpec | None
base_env_kwargs: dict
n_steps: int
n_envs: int
batch_size: int
lookahead_enabled: bool
lookahead_rollouts: int
lookahead_probe_episodes: int
lookahead_margin: float
lookahead_budget_s: float
```

Track:

```python
self.slope_hist: list[float]
self.lookahead_wall_s: float
```

At each decision:

1. Run current and full probes as V1 does.
2. Append decision point.
3. Compute and append `target_slope_per_cost`.
4. Call `should_run_lookahead`.
5. If true and `next_spec is not None`, call `_run_lookahead`.
6. Promote with reason `lookahead_better:<trigger>` if:

```python
lookahead_slope_per_cost > lookahead_margin * current_slope_per_cost
```

7. Otherwise call the existing `decide_switch` fallback.

### Lookahead implementation details

Implement `_run_lookahead` conservatively:

```python
def _run_lookahead(self, vec_norm) -> dict:
    next_kwargs = {**base_env_kwargs, **next_spec.env_overrides}
    next_vec_env = build_vec_env(next_kwargs, n_envs=n_envs, train_seed=...)
    next_vec_env.obs_rms = copy/deepcopy(current_vec_env.obs_rms)
    next_vec_env.ret_rms = RunningMeanStd(shape=())

    clone = build_model(next_vec_env, n_steps=n_steps, batch_size=batch_size)
    clone.policy.load_state_dict(self.model.policy.state_dict())

    eval full before or reuse current full probe as P_H_before
    clone.learn(total_timesteps=lookahead_rollouts * n_steps, ...)
    evaluate clone on gt_env
    return metrics
```

Important constraints:

- Do not mutate the real training model.
- Do not mutate the real training `VecNormalize` stats.
- Reset reward normalization for the lookahead clone.
- Use the same full-sim probe seeds for before/after where possible.
- Include a wall-budget callback so lookahead cannot consume the final full
  reserve.
- Save no PPO artifact from lookahead by default; only log metrics.

If copying `obs_rms` by assignment risks shared mutation, use `copy.deepcopy`.

### Logging schema

Extend each decision entry in `fidelity_history.json`:

```json
{
  "target_slope_per_cost": 0.00012,
  "relative_target_gain": 0.018,
  "lookahead_trigger": "slope_collapse",
  "lookahead_ran": true,
  "lookahead_stage": "full3",
  "lookahead_wall_s": 54.2,
  "lookahead_mape_pre_pct": 8.1,
  "lookahead_mape_post_pct": 7.4,
  "lookahead_slope_per_cost": 0.00019,
  "lookahead_promote": true,
  "switch_reason": "lookahead_better:slope_collapse"
}
```

Also include cumulative lookahead cost in `run_config.json` or a final history
entry.

### Callback construction

In the main stage loop:

```python
next_spec = specs[i + 1] if i + 1 < len(specs) else None
```

Pass `next_spec` into `FidelitySwitchCallback` only for non-final stages.

For `--schedule fixed`, do not run lookahead. Fixed schedule should remain an
ablation of the rule.

### Recommended first V2 command

```bash
PYTHON_JULIAPKG_OFFLINE=yes PYTHON_JULIACALL_HANDLE_SIGNALS=yes \
PYTHON_JULIACALL_THREADS=6 JULIA_NUM_THREADS=6 \
PYTHON_JULIAPKG_EXE=~/.julia/juliaup/julia-1.11.9+0.x64.linux.gnu/bin/julia \
  PYTHONUNBUFFERED=1 python -u python/train_e2_mf.py \
  --out runs/e2/mf_v2_lookahead --multi-fidelity \
  --mf-plan analytic,cached3,full3,full \
  --reward-mode delta_log_mape --fix-te --learn-alpha --field T15 \
  --n-envs 2 \
  --mf-budget-hours 24 --schedule criterion \
  --mf-use-lookahead \
  --mf-lookahead-rollouts 1 \
  --mf-lookahead-margin 1.15 \
  --mf-slope-collapse-frac 0.25 \
  2>&1 | tee runs/e2/mf_v2_lookahead/run.log
```

Run A-equivalent CPU command:

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

### Validation

Run before long experiments:

```bash
python -m py_compile python/train_e2_mf.py python/mf_switch.py
python -m pytest python/tests/test_mf_switch.py
PYTHON_JULIAPKG_OFFLINE=yes python python/train_e2_mf.py \
  --out runs/e2/mf_v2_smoke --multi-fidelity \
  --mf-plan analytic,cached3,full3,full \
  --reward-mode delta_log_mape --fix-te --learn-alpha \
  --field T15 --Nfe 16 --Npe 8 --voxel-mm 3 --n-envs 2 \
  --mf-budget-hours 0.15 --mf-min-steps 256 \
  --mf-max-steps 512 --mf-decision-rollouts 1 \
  --mf-probe-episodes-full 2 --mf-use-lookahead \
  --mf-lookahead-rollouts 1
```

Check:

- `fidelity_history.json` contains lookahead entries.
- Switch reasons distinguish `lookahead_better`, `target_plateau`,
  `rank_breakdown`, `bias_intolerance`, and `budget_reserve`.
- Lookahead wallclock stays under `mf_lookahead_max_frac`.
- Final full stage still receives its reserve.

## Phase B — pooled multi-fidelity samples

This is a separate comparison branch, not a small V2 patch.

Implementation options:

1. Add a replay-buffer learner labelled by fidelity.
2. Test a control-variate return/advantage correction:

```text
target_estimate = cheap_estimate + learned(full - cheap)
```

3. Prototype a PPO-compatible weighted batch only if the off-policy route is too
   large.

Expected code impact:

- New training script, likely `python/train_e2_mf_pool.py`.
- Fidelity-labelled transition storage.
- Bias/correction model or sample weights.
- New plots reporting contribution by fidelity.

Risk:

- PPO on mixed-fidelity samples is no longer cleanly on-policy with respect to
  the target environment.
- A negative result is acceptable and should be reported as evidence that the
  stagewise curriculum is the more stable approach for E2.

## Phase C — BO-informed controller

This should follow after Phase A produces enough decision records.

Implementation outline:

1. Build a decision-level dataset from `fidelity_history.json`.
2. Fit a surrogate for `ΔP_H` by fidelity using features such as:

```text
stage index, fidelity, current P_H, recent slope, cheap/full bias,
p90 gap, rank_corr, sec_per_step, remaining budget
```

3. Use an acquisition score:

```text
expected ΔP_H / expected cost + uncertainty bonus / expected cost
```

4. Let the acquisition choose whether to:

- continue current fidelity;
- probe next fidelity;
- skip to a later fidelity;
- allocate a small lookahead to the most informative candidate.

Suggested first implementation:

- Use scikit-learn GaussianProcessRegressor or a small Bayesian linear model.
- Keep the controller offline initially: replay old history files and compare
  what decisions it would have made.
- Only move online after the offline counterfactual looks sane.

Success criterion:

- Beats Phase A on wallclock-to-MAPE with similar or lower lookahead overhead.
- Produces interpretable acquisition logs showing why it chose each fidelity.

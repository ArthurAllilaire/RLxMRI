# E1 Results — Single-Sphere T1 Estimation

## Summary

The PPO agent trained on E1 achieves the same MAPE as the fixed-grid baseline
(~0.55%, 100% success rate under 3% tolerance) but for the wrong reason: it
collapsed to a **degenerate fixed policy** that never adapts to the unknown T1.
The apparent performance is explained by the fitting algorithm, not learned
sequence design.

---

## Reproducing the results

All commands assume the venv is activated (`source .venv/bin/activate`) and are
run from the repo root.

### 1 — Fixed-grid baseline

```bash
PYTHON_JULIAPKG_OFFLINE=yes python python/baseline_e1.py --episodes 200
```

Output (reproduced):
```
Fixed 8-block IR grid baseline
  n_episodes           = 200
  mape_pct             = 0.5904
  median_err_pct       = 0.5831
  p90_err_pct          = 1.0559
  mean_time_s          = 5.6200
  mean_reward          = -2.2465
  success_rate_3pct    = 1.0000
```

The baseline plays 8 IR blocks at TI ∈ {10, 30, 100, 300, 600, 1000, 1800,
3000} ms (all α=180°) — a deliberate log-spaced sweep of the full T1 range.

### 2 — Train PPO

```bash
PYTHON_JULIAPKG_OFFLINE=yes python python/train_e1.py --timesteps 50000 --out runs/e1/ppo
```

Eval checkpoints logged every 5 000 steps (50 episodes, held-out seeds):

| Step | MAPE | p90 | Success (<3%) |
|------|------|-----|---------------|
| 5 000 | 0.554% | 1.016% | 100% |
| 15 000 | 0.855% | 1.056% | 98% |
| 25 000 | 0.554% | 1.016% | 100% |
| 50 000 | 0.554% | 1.016% | 100% |

Training time: ~160 s on CPU. Policy saved to `runs/e1/ppo/policy.zip`.

### 3 — Inspect what the agent actually does

```python
# python python/inspect_e1_policy.py  (or paste into a REPL)
import sys; sys.path.insert(0, "python")
import numpy as np
from stable_baselines3 import PPO
from qalibremd_gym.env import QalibreMDE1Env

model = PPO.load("runs/e1/ppo/policy")
env = QalibreMDE1Env(rng_seed=999)
action_table = env.action_table

for ep_seed in [200001, 200002, 200003, 200004, 200005]:
    obs, info = env.reset(seed=ep_seed)
    T1_true = info["T1_true"]
    actions_taken = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        actions_taken.append(int(action))
        obs, r, done, _, info = env.step(int(action))
    tis = [f"{action_table[a][0]*1000:.0f}ms/α={np.degrees(action_table[a][1]):.0f}°"
           for a in actions_taken]
    print(f"T1_true={T1_true*1000:6.1f}ms  err={info['err']*100:.2f}%  "
          f"n={int(info['n_blocks'])}  actions={actions_taken}")
    print(f"  {' → '.join(tis)}")
```

Output:
```
T1_true=  50.3ms  err=0.34%  n=12  actions=[13, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  1000ms/α=90° → 10ms/α=90° × 11
T1_true= 672.8ms  err=0.19%  n=12  actions=[13, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  1000ms/α=90° → 10ms/α=90° × 11
T1_true= 914.4ms  err=0.41%  n=12  actions=[13, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  1000ms/α=90° → 10ms/α=90° × 11
T1_true=1411.3ms  err=0.15%  n=12  actions=[13, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  1000ms/α=90° → 10ms/α=90° × 11
T1_true= 843.5ms  err=0.71%  n=12  actions=[13, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  1000ms/α=90° → 10ms/α=90° × 11
```

---

## Conclusion

The agent plays the **same 12 actions on every episode** regardless of T1_true:
one saturation-recovery prep (TI=1000ms, α=90°), then eleven repeats of the
shortest block (TI=10ms, α=90°). This is a degenerate fixed policy, not an
adaptive one.

It achieves ~0.55% MAPE because `fit_t1_generalized_ir` is powerful enough to
get a good T1 estimate from a single informative measurement (the first
saturation-recovery block already constrains the fit well). The subsequent
11 short-TI repeats add redundant constraints that the fitter absorbs without
complaint. The observation vector contains the running `T1_est_log`, which the
agent is supposed to use to adapt its next action — it never learned to do so.

**Why the reward did not prevent this:**

The episode reward has two terms:

```
reward_step  = −|T̂₁ − T₁_true| / T₁_true  −  λ · block_time / budget
terminal     = +1.0  if final err < success_tol (3%)
```

Once the agent discovers it can almost always collect the +1.0 terminal bonus,
the per-step error penalty is too weak to push it toward a policy that actually
explores the TI range. The terminal bonus dominates and the agent learns to
exploit the fitter rather than to design informative sequences.

---

## Next steps

### 1 — Verify the fitter is causing the collapse, not a reward bug

Run a sanity check: replace the agent with a random policy and measure MAPE.
If it is also low, the fitter is doing all the work irrespective of actions.

```bash
PYTHON_JULIAPKG_OFFLINE=yes python - <<'EOF'
import sys; sys.path.insert(0, "python")
import numpy as np, statistics
from qalibremd_gym.env import QalibreMDE1Env

env = QalibreMDE1Env(rng_seed=0)
errs = []
for ep in range(200):
    obs, _ = env.reset(seed=100_000 + ep)
    done = False
    info = {}
    while not done:
        obs, _, done, _, info = env.step(env.action_space.sample())
    errs.append(info["err"])
print(f"random policy MAPE = {100*statistics.mean(errs):.3f}%")
print(f"random success(<3%) = {sum(e<0.03 for e in errs)/len(errs):.2%}")
EOF
```

If random MAPE is also ~0.5%, the task is trivially solvable by the fitter
and the E1 RL problem is not well-posed.

### 2 — Make the fitter harder to exploit

Options in increasing order of invasiveness:

- **Limit fitter data:** only pass the last N measurements to the fitter
  (forces the agent to keep choosing informative TIs rather than repeating
  short ones that add little).
- **Add measurement noise:** inject Gaussian noise into the simulated signal
  (makes redundant repeats of the same action useless and forces diversity).
- **Remove the terminal bonus** (or reduce it substantially, e.g. 0.1 instead
  of 1.0). The +1 bonus is disproportionate to the per-step penalties and
  creates the exploit.
- **Penalise action repetition** in the reward: subtract a small penalty when
  `action_counts[a] > 1`. Already tracked in `env.action_counts`; just needs
  wiring into `e1_step!`.

### 3 — Add an adaptive-policy diagnostic to the eval loop

The current `eval_e1.py` only reports aggregate MAPE. Extend it to log
per-episode action sequences and check whether the same sequence appears for
T1 < 100 ms vs T1 > 1 000 ms. A genuinely adaptive policy must pick different
actions in those two regimes.

### 4 — Increase training scale cautiously

50 k timesteps is the minimum viable run. If reward shaping is fixed first,
re-run at 200 k–500 k timesteps before concluding the agent cannot learn.
The current ~265 steps/s means 500 k steps ≈ 30 min on CPU, which is
acceptable.

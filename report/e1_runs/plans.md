Given a phantom guess Ts and PD

Currently:

- discrete action space

Fixed policy 8 block 

| E1 | Single-voxel RL (PPO, discrete TI×α actions) | **Done — degenerate policy** | `src/rl/e1.jl`, `python/qalibremd_gym/env.py` |

The env:

- 

Mistakes:

The E1 agent achieved ~0.55% MAPE but for the **wrong reason**: it collapsed to a degenerate fixed policy (same 12 actions every episode regardless of T1_true). The T1 fitter (`fit_t1_generalized_ir`) is powerful enough to recover T1 from a single informative measurement, so the agent learned to exploit the fitter rather than design adaptive sequences.

Root causes (from `docs/E1_RESULTS.md`):
1. The `terminal_bonus = +1.0` dominated the per-step error penalty, so the agent only needed to collect the bonus — not optimise the sequence.
2. Without noise, the fitter works perfectly from almost any single measurement.
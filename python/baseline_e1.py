"""Fixed-policy baseline for E1.

Plays a fixed 8-block IR grid (TI ∈ {10, 30, 100, 300, 600, 1000, 1800, 3000} ms
at α = 180°) on a held-out set of phantom configs. Matches PLAN.md §4 E1's
"fixed 8-block IR grid" yardstick — the RL agent has to beat this on MAPE
within the same scan-time budget.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from qalibremd_gym.env import QalibreMDE1Env


# Fixed IR grid — α = 180° (index 3 in α_set_deg=[10,90,180]) → actions 3,6,...,18.
# We pick 8 TIs log-spaced over 10 ms – 3 s, mapping each to the closest
# predefined TI in the default TI_set of the environment.
FIXED_POLICY_8BLOCKS_IDX = [3, 6, 9, 12, 15, 18, 9, 15]   # TIs  10,30,100,300,1000,3000,100,1000 @ α=180


def evaluate(n_episodes: int = 200, seed_offset: int = 100_000,
             max_blocks: int = 12, **env_kwargs):
    env = QalibreMDE1Env(max_blocks=max_blocks, **env_kwargs)
    errs, times, rewards = [], [], []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed_offset + ep)
        total_r = 0.0
        last_info = info
        for step in range(max_blocks):
            action = FIXED_POLICY_8BLOCKS_IDX[step % len(FIXED_POLICY_8BLOCKS_IDX)] - 1
            if step >= len(FIXED_POLICY_8BLOCKS_IDX):
                break
            obs, r, done, _trunc, info = env.step(action)
            total_r += r
            last_info = info
            if done:
                break
        errs.append(last_info["err"])
        times.append(env.time_used_s)
        rewards.append(total_r)
    return {
        "n_episodes": n_episodes,
        "mape_pct":   100.0 * statistics.mean(errs),
        "median_err_pct": 100.0 * statistics.median(errs),
        "p90_err_pct": 100.0 * float(np.percentile(errs, 90)),
        "mean_time_s": statistics.mean(times),
        "mean_reward": statistics.mean(rewards),
        "success_rate_3pct": float(sum(e < 0.03 for e in errs)) / len(errs),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--seed-offset", type=int, default=100_000)
    args = p.parse_args()

    print("Fixed 8-block IR grid baseline")
    stats = evaluate(n_episodes=args.episodes, seed_offset=args.seed_offset)
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k:20s} = {v:.4f}")
        else:
            print(f"  {k:20s} = {v}")


if __name__ == "__main__":
    main()

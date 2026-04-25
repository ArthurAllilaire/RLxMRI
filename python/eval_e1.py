"""Evaluate a trained PPO agent against the fixed-grid baseline on a
held-out seed split. Prints a side-by-side MAPE table.

Usage:
    python python/eval_e1.py --policy runs/e1/ppo/policy.zip --episodes 500
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from stable_baselines3 import PPO

from qalibremd_gym.env import QalibreMDE1Env
from baseline_e1 import evaluate as baseline_evaluate


def evaluate_policy(policy_path: Path, n_episodes: int, seed_offset: int,
                    **env_kwargs):
    env = QalibreMDE1Env(**env_kwargs)
    model = PPO.load(str(policy_path))
    errs, times, rewards = [], [], []
    for ep in range(n_episodes):
        obs, _info = env.reset(seed=seed_offset + ep)
        total_r = 0.0
        done = False
        info = {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, _trunc, info = env.step(int(action))
            total_r += r
        errs.append(info["err"])
        times.append(env.time_used_s)
        rewards.append(total_r)
    return {
        "mape_pct": 100.0 * statistics.mean(errs),
        "median_err_pct": 100.0 * statistics.median(errs),
        "p90_err_pct": 100.0 * float(np.percentile(errs, 90)),
        "mean_time_s": statistics.mean(times),
        "mean_reward": statistics.mean(rewards),
        "success_rate_3pct": float(sum(e < 0.03 for e in errs)) / len(errs),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--seed-offset", type=int, default=100_000)
    args = p.parse_args()

    print("running fixed-grid baseline …")
    base = baseline_evaluate(n_episodes=args.episodes,
                             seed_offset=args.seed_offset)

    print("running PPO policy …")
    agent = evaluate_policy(args.policy, args.episodes, args.seed_offset)

    print()
    print(f"{'metric':<24} {'baseline':>14} {'agent':>14}")
    print("-" * 56)
    for key in ("mape_pct", "median_err_pct", "p90_err_pct",
                "mean_time_s", "mean_reward", "success_rate_3pct"):
        bv = base[key]
        av = agent[key]
        print(f"{key:<24} {bv:>14.4f} {av:>14.4f}")


if __name__ == "__main__":
    main()

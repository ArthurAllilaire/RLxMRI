"""Train a PPO agent on the E1 single-sphere T1 environment.

Usage:
    python python/train_e1.py --timesteps 50000 --out runs/e1/ppo

Every `--eval-interval` timesteps we evaluate on a fixed held-out seed
offset and log MAPE against the fixed-grid baseline (see
`baseline_e1.evaluate`).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from qalibremd_gym.env import QalibreMDE1Env
from baseline_e1 import evaluate as baseline_evaluate, FIXED_POLICY_8BLOCKS_IDX


def make_env(rank: int, train_seed: int, **env_kwargs):
    def _init():
        env = QalibreMDE1Env(rng_seed=train_seed + rank, **env_kwargs)
        return Monitor(env)
    return _init


class EvalAgainstBaselineCallback(BaseCallback):
    """Periodically roll the current policy on held-out seeds and print MAPE."""

    def __init__(self, eval_env: QalibreMDE1Env, every_n_steps: int,
                 n_eval_episodes: int, seed_offset: int, log_path: Path,
                 verbose: int = 1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.every_n_steps = every_n_steps
        self.n_eval_episodes = n_eval_episodes
        self.seed_offset = seed_offset
        self.log_path = log_path
        self._last_eval = 0
        self.history = []

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval < self.every_n_steps:
            return True
        self._last_eval = self.num_timesteps

        errs = []
        for ep in range(self.n_eval_episodes):
            obs, _info = self.eval_env.reset(seed=self.seed_offset + ep)
            done = False
            info = {}
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, r, done, trunc, info = self.eval_env.step(int(action))
            errs.append(info["err"])

        mape = 100.0 * float(np.mean(errs))
        p90  = 100.0 * float(np.percentile(errs, 90))
        succ = float(sum(e < 0.03 for e in errs)) / len(errs)
        if self.verbose:
            print(f"[eval @ step {self.num_timesteps}] "
                  f"MAPE={mape:.3f}%  p90={p90:.3f}%  success(<3%)={succ:.2%}")
        self.history.append({
            "step": self.num_timesteps,
            "mape_pct": mape, "p90_pct": p90, "success_rate": succ,
        })
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("w") as f:
            import json
            json.dump(self.history, f, indent=2)
        return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--timesteps", type=int, default=50_000)
    p.add_argument("--n-envs", type=int, default=1,
                   help="Parallel env workers. Note each worker boots its own "
                        "Julia runtime; memory scales accordingly.")
    p.add_argument("--eval-episodes", type=int, default=50)
    p.add_argument("--eval-interval", type=int, default=5_000)
    p.add_argument("--train-seed", type=int, default=0)
    p.add_argument("--eval-seed-offset", type=int, default=100_000)
    p.add_argument("--out", type=Path, default=Path("runs/e1/ppo"))
    p.add_argument("--skip-baseline", action="store_true")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    if not args.skip_baseline:
        print("[pre] fixed-grid baseline on eval split …")
        t0 = time.time()
        base = baseline_evaluate(n_episodes=args.eval_episodes,
                                 seed_offset=args.eval_seed_offset)
        print(f"  baseline MAPE = {base['mape_pct']:.3f}%  "
              f"success(<3%) = {base['success_rate_3pct']:.2%}  "
              f"({time.time()-t0:.1f}s)")

    env = DummyVecEnv([make_env(i, args.train_seed) for i in range(args.n_envs)])
    eval_env = QalibreMDE1Env(rng_seed=args.eval_seed_offset)

    model = PPO(
        "MlpPolicy", env,
        n_steps=256, batch_size=64, learning_rate=3e-4,
        gamma=0.99, gae_lambda=0.95,
        policy_kwargs=dict(net_arch=[128, 128]),
        verbose=1, tensorboard_log=str(args.out / "tb"),
    )

    callback = EvalAgainstBaselineCallback(
        eval_env=eval_env, every_n_steps=args.eval_interval,
        n_eval_episodes=args.eval_episodes,
        seed_offset=args.eval_seed_offset,
        log_path=args.out / "eval_history.json",
    )

    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, callback=callback)
    print(f"training finished in {time.time()-t0:.1f}s")
    model.save(args.out / "policy")
    print(f"policy saved to {args.out / 'policy.zip'}")


if __name__ == "__main__":
    main()

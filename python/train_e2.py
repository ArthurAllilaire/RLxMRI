"""Train a PPO agent on the E2 multi-sphere T1 mapping environment.

Usage:
    python python/train_e2.py --timesteps 200000 --out runs/e2/ppo

The agent uses a continuous action space (5 parameters) normalised to [-1,1].
Observations are VecNormalise-scaled. Eval MAPE is logged every
--eval-interval steps.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv, SubprocVecEnv, VecNormalize,
)

from qalibremd_gym.env_e2 import QalibreMDE2Env
from e2_config import add_e2_env_args, e2_env_kwargs


def make_env(rank: int, train_seed: int, **env_kwargs):
    def _init():
        env = QalibreMDE2Env(rng_seed=train_seed + rank, **env_kwargs)
        return Monitor(env)
    return _init


class E2CheckpointCallback(BaseCallback):
    """Save model + VecNormalize stats every save_freq timesteps."""

    def __init__(self, save_freq: int, out_dir: Path, vec_env: VecNormalize,
                 verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.out_dir   = out_dir
        self.vec_env   = vec_env

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            step = self.num_timesteps
            self.model.save(self.out_dir / f"ckpt_{step}")
            self.vec_env.save(self.out_dir / f"vecnorm_ckpt_{step}.pkl")
            meta = {"timesteps_done": step, "wall_time": time.time()}
            (self.out_dir / "checkpoint_meta.json").write_text(json.dumps(meta))
            if self.verbose:
                print(f"[ckpt] Saved checkpoint at step {step}")
        return True


class E2EvalCallback(BaseCallback):
    """Periodically evaluate the policy and log per-sphere MAPE."""

    def __init__(self, eval_env: QalibreMDE2Env, every_n_steps: int,
                 n_eval_episodes: int, seed_offset: int,
                 log_path: Path, best_dir: Path | None = None,
                 env_kwargs: dict | None = None,
                 verbose: int = 1):
        super().__init__(verbose)
        self.eval_env         = eval_env
        self.every_n_steps    = every_n_steps
        self.n_eval_episodes  = n_eval_episodes
        self.seed_offset      = seed_offset
        self.log_path         = log_path
        self.best_dir         = best_dir
        self.env_kwargs       = env_kwargs or {}
        self._last_eval       = 0
        self.history          = (json.loads(log_path.read_text())
                                 if log_path.exists() else [])
        self.best_mape        = min((h["mape_pct"] for h in self.history),
                                    default=float("inf"))

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval < self.every_n_steps:
            return True
        self._last_eval = self.num_timesteps

        vec_norm = self.model.get_vec_normalize_env()

        mapes, times = [], []
        for ep in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset(seed=self.seed_offset + ep)
            done = False
            info = {}
            while not done:
                obs_in = vec_norm.normalize_obs(obs) if vec_norm is not None else obs
                action, _ = self.model.predict(obs_in, deterministic=True)
                obs, _r, done, _trunc, info = self.eval_env.step(action)
            mapes.append(float(info.get("mape", np.nan)))
            times.append(self.eval_env.time_used_s)

        mape_mean = float(np.nanmean(mapes)) * 100
        mape_p90  = float(np.nanpercentile(mapes, 90)) * 100
        succ      = float(np.mean([m < 0.05 for m in mapes if not np.isnan(m)]))
        mean_time = float(np.mean(times))

        if self.verbose:
            print(f"[E2 eval @ step {self.num_timesteps}]  "
                  f"MAPE={mape_mean:.2f}%  p90={mape_p90:.2f}%  "
                  f"success(<5%)={succ:.1%}  mean_scan_time={mean_time:.1f}s")

        self.history.append({
            "step":         self.num_timesteps,
            "mape_pct":     mape_mean,
            "p90_pct":      mape_p90,
            "success_rate": succ,
            "mean_time_s":  mean_time,
        })
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("w") as f:
            json.dump(self.history, f, indent=2)

        # Save best-eval policy/vecnorm so resumed/long runs don't lose the
        # best checkpoint to overtraining drift (see EXPERT_REPORT_TRAC §10.2).
        if self.best_dir is not None and mape_mean < self.best_mape:
            self.best_mape = mape_mean
            self.best_dir.mkdir(parents=True, exist_ok=True)
            self.model.save(str(self.best_dir / "best_policy"))
            vec_norm = self.model.get_vec_normalize_env()
            if vec_norm is not None:
                vec_norm.save(str(self.best_dir / "best_vecnorm.pkl"))
            with (self.best_dir / "best_meta.json").open("w") as f:
                json.dump({
                    "step": self.num_timesteps,
                    "mape_pct": mape_mean,
                    "p90_pct": mape_p90,
                    "mean_time_s": mean_time,
                    "env_kwargs": self.env_kwargs,
                }, f, indent=2)
            if self.verbose:
                print(f"[E2 eval]  → new best MAPE {mape_mean:.2f}% saved to "
                      f"{self.best_dir}")
        return True


class ProgressETACallback(BaseCallback):
    """Log a wall-clock ETA to completion at the end of each PPO rollout.

    Uses an EMA of seconds-per-timestep (the per-step cost drifts as the policy
    concentrates its TR distribution, since sim cost ∝ Npe·TR), and records
    `time/eta_hours` so it shows up next to `time/fps` in the SB3 table.
    """

    def __init__(self, total_timesteps: int, ema_alpha: float = 0.3,
                 verbose: int = 1):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.ema_alpha       = ema_alpha
        self._last_t         = None
        self._last_steps     = None
        self._ema_spp        = None   # EMA seconds-per-timestep

    def _on_training_start(self) -> None:
        self._last_t     = time.time()
        self._last_steps = self.num_timesteps

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        now   = time.time()
        steps = self.num_timesteps
        d_steps = steps - (self._last_steps or steps)
        d_time  = now - (self._last_t or now)
        self._last_t, self._last_steps = now, steps
        if d_steps <= 0 or d_time <= 0:
            return
        spp = d_time / d_steps
        self._ema_spp = spp if self._ema_spp is None else (
            self.ema_alpha * spp + (1 - self.ema_alpha) * self._ema_spp)
        remaining = max(0, self.total_timesteps - steps)
        eta_s = remaining * self._ema_spp
        self.logger.record("time/eta_hours", eta_s / 3600.0)
        self.logger.record("time/sec_per_step", self._ema_spp)
        if self.verbose:
            print(f"[E2] ~{eta_s/3600.0:.1f}h remaining "
                  f"({remaining} steps left @ {self._ema_spp:.2f}s/step)")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--timesteps",      type=int,   default=200_000)
    p.add_argument("--n-envs",         type=int,   default=1)
    p.add_argument("--n-steps",        type=int,   default=2048,
                   help="PPO rollout length per env. The ETA line prints once "
                        "per rollout (every n_steps×n_envs timesteps), so lower "
                        "it for a quick watch run.")
    p.add_argument("--eval-episodes",  type=int,   default=30)
    p.add_argument("--eval-interval",  type=int,   default=10_000)
    p.add_argument("--train-seed",     type=int,   default=0)
    p.add_argument("--eval-seed",      type=int,   default=500_000)
    p.add_argument("--out",            type=Path,  default=Path("runs/e2/ppo"))
    p.add_argument("--checkpoint-interval", type=int, default=50_000,
                   help="Save a checkpoint every N timesteps (0 = disabled)")
    p.add_argument("--resume", action="store_true",
                   help="Resume from latest checkpoint in --out directory")
    add_e2_env_args(p)   # shared env flags (incl. --Nfe/--Npe/--voxel-mm/--use-gpu)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    env_kwargs = e2_env_kwargs(args)

    # Dump the full env config (Npe, voxel_size_mm, noise, field, …) + key
    # training hyperparams alongside the policy, so a saved run is reproducible
    # and self-describing without re-deriving the command from shell history.
    run_config = {
        "env_kwargs": env_kwargs,
        "train": {
            "timesteps":     args.timesteps,
            "n_envs":        args.n_envs,
            "train_seed":    args.train_seed,
            "eval_seed":     args.eval_seed,
            "eval_episodes": args.eval_episodes,
            "eval_interval": args.eval_interval,
        },
    }
    (args.out / "run_config.json").write_text(json.dumps(run_config, indent=2))

    # SubprocVecEnv runs each env in its own OS process with its own in-process
    # Julia, so rollouts parallelise across cores (DummyVecEnv is serial — one
    # Julia, one core). Use it only when there's >1 env to avoid the subprocess
    # + per-worker JIT-warmup overhead for the n_envs=1 case.
    print(f"[E2] Building train env (n_envs={args.n_envs}) …")
    env_fns = [make_env(i, args.train_seed, **env_kwargs)
               for i in range(args.n_envs)]
    vec_env = (SubprocVecEnv(env_fns, start_method="spawn")
               if args.n_envs > 1 else DummyVecEnv(env_fns))
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True,
                           clip_obs=10.0, clip_reward=10.0)

    print("[E2] Building eval env …")
    eval_env = QalibreMDE2Env(rng_seed=args.eval_seed, **env_kwargs)

    if args.resume:
        ckpts = sorted(args.out.glob("ckpt_*.zip"),
                       key=lambda p: int(p.stem.split("_")[1]))
        if not ckpts:
            raise FileNotFoundError(f"No checkpoints found in {args.out}")
        latest = ckpts[-1]
        step_done = int(latest.stem.split("_")[1])
        remaining = args.timesteps - step_done
        if remaining <= 0:
            raise ValueError(
                f"--timesteps={args.timesteps} is not above latest checkpoint "
                f"{latest.name} ({step_done} steps). Increase --timesteps or "
                "start a fresh output directory."
            )
        vec_env = VecNormalize.load(
            args.out / f"vecnorm_ckpt_{step_done}.pkl", vec_env)
        vec_env.training = True
        model = PPO.load(latest, env=vec_env,
                         tensorboard_log=str(args.out / "tb"))
        print(f"[E2] Resumed from {latest.name} "
              f"({step_done} steps done, {remaining} remaining)")
    else:
        remaining = args.timesteps
        model = PPO(
            "MlpPolicy", vec_env,
            n_steps       = args.n_steps,  # longer rollouts → better advantage estimates
            batch_size    = 64,
            learning_rate = 1e-4,    # smaller steps → tame clip_fraction (was 0.5 at 3e-4)
            gamma         = 0.99,
            gae_lambda    = 0.95,
            ent_coef      = 0.005,   # let policy concentrate sooner
            max_grad_norm = 0.5,
            policy_kwargs = dict(net_arch=[256, 256]),
            device        = "cpu",   # MLP policy is faster on CPU; GPU is for KomaMRI sim
            verbose       = 1,
            tensorboard_log = str(args.out / "tb"),
        )

    callbacks: list[BaseCallback] = [
        E2EvalCallback(
            eval_env        = eval_env,
            every_n_steps   = args.eval_interval,
            n_eval_episodes = args.eval_episodes,
            seed_offset     = args.eval_seed,
            log_path        = args.out / "eval_history.json",
            best_dir        = args.out / "best",
            env_kwargs      = env_kwargs,
        ),
    ]
    if args.checkpoint_interval > 0:
        callbacks.append(E2CheckpointCallback(
            save_freq = args.checkpoint_interval,
            out_dir   = args.out,
            vec_env   = vec_env,
        ))
    callbacks.append(ProgressETACallback(total_timesteps=remaining))

    t0 = time.time()
    model.learn(
        total_timesteps=remaining,
        callback=callbacks,
        reset_num_timesteps=not args.resume,
    )
    elapsed = time.time() - t0
    print(f"Training done in {elapsed:.1f}s")

    model.save(args.out / "policy")
    vec_env.save(args.out / "vecnorm.pkl")
    print(f"Policy → {args.out / 'policy.zip'}")
    print(f"VecNormalize → {args.out / 'vecnorm.pkl'}")


if __name__ == "__main__":
    main()

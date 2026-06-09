"""Shared E2 training building blocks.

Both `train_e2.py` (single-fidelity) and `train_e2_mf.py` (multi-fidelity
curriculum) import the env/model/eval helpers and callbacks from here, so the
PPO hyperparameters, eval rollout, and checkpoint/ETA logic live in exactly one
place. Keeping them here is what lets the multi-fidelity orchestrator reuse the
*same* evaluation as the single-fidelity script — switch decisions and the
single-fidelity eval curve are then directly comparable.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv, SubprocVecEnv, VecNormalize,
)

from qalibremd_gym.env_e2 import QalibreMDE2Env


# ── env / model construction ────────────────────────────────────────────────

def make_env(rank: int, train_seed: int, **env_kwargs):
    def _init():
        env = QalibreMDE2Env(rng_seed=train_seed + rank, **env_kwargs)
        return Monitor(env)
    return _init


def build_vec_env(env_kwargs: dict, *, n_envs: int, train_seed: int) -> VecNormalize:
    """Build the (optionally subprocess-parallel) VecNormalize-wrapped train env.

    SubprocVecEnv runs each env in its own OS process with its own in-process
    Julia, so rollouts parallelise across cores (DummyVecEnv is serial — one
    Julia, one core). Use Subproc only when there's >1 env to avoid the
    subprocess + per-worker JIT-warmup overhead for the n_envs=1 case (the
    multi-fidelity orchestrator always runs n_envs=1 to keep one Julia alive).
    """
    env_fns = [make_env(i, train_seed, **env_kwargs) for i in range(n_envs)]
    vec_env = (SubprocVecEnv(env_fns, start_method="spawn")
               if n_envs > 1 else DummyVecEnv(env_fns))
    return VecNormalize(vec_env, norm_obs=True, norm_reward=True,
                        clip_obs=10.0, clip_reward=10.0)


def build_model(vec_env: VecNormalize, *, n_steps: int, batch_size: int,
                tensorboard_log: str | None = None) -> PPO:
    """Construct a fresh PPO with the canonical E2 hyperparameters.

    Used both for cold starts and for the optimizer-reset-on-switch path in the
    multi-fidelity orchestrator (build fresh → copy policy weights), so the
    Adam state and LR schedule are always recreated from the same settings.
    """
    return PPO(
        "MlpPolicy", vec_env,
        n_steps       = n_steps,    # longer rollouts → better advantage estimates
        batch_size    = batch_size,
        learning_rate = 1e-4,       # smaller steps → tame clip_fraction (was 0.5 at 3e-4)
        gamma         = 0.99,
        gae_lambda    = 0.95,
        ent_coef      = 0.005,      # let policy concentrate sooner
        max_grad_norm = 0.5,
        policy_kwargs = dict(net_arch=[256, 256]),
        device        = "cpu",      # MLP policy is faster on CPU; GPU is for KomaMRI sim
        verbose       = 1,
        tensorboard_log = tensorboard_log,
    )


# ── evaluation ──────────────────────────────────────────────────────────────

def rollout_eval(model, eval_env: QalibreMDE2Env, n_episodes: int,
                 seed_offset: int, vec_norm: VecNormalize | None = None):
    """Roll the deterministic policy on `eval_env` for `n_episodes` paired seeds.

    Returns (mapes, times) as float arrays. `vec_norm` (if given) supplies the
    obs-normalisation stats the policy was trained under — pass the training
    VecNormalize so eval matches training-time observation scaling.
    """
    mapes, times = [], []
    for ep in range(n_episodes):
        obs, _ = eval_env.reset(seed=seed_offset + ep)
        done = False
        info: dict = {}
        while not done:
            obs_in = vec_norm.normalize_obs(obs) if vec_norm is not None else obs
            action, _ = model.predict(obs_in, deterministic=True)
            obs, _r, done, _trunc, info = eval_env.step(action)
        mapes.append(float(info.get("mape", np.nan)))
        times.append(float(eval_env.time_used_s))
    return np.asarray(mapes, dtype=float), np.asarray(times, dtype=float)


def summarise_eval(mapes: np.ndarray, times: np.ndarray) -> dict:
    """Reduce per-episode (mapes, times) to the scalar metrics we log."""
    return {
        "mape_pct":     float(np.nanmean(mapes)) * 100,
        "p90_pct":      float(np.nanpercentile(mapes, 90)) * 100,
        "success_rate": float(np.mean([m < 0.05 for m in mapes if not np.isnan(m)])),
        "mean_time_s":  float(np.mean(times)),
    }


# ── callbacks ───────────────────────────────────────────────────────────────

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
                 global_best=None,
                 global_best_stage: str | None = None,
                 global_best_source: str = "eval_callback",
                 verbose: int = 1):
        super().__init__(verbose)
        self.eval_env         = eval_env
        self.every_n_steps    = every_n_steps
        self.n_eval_episodes  = n_eval_episodes
        self.seed_offset      = seed_offset
        self.log_path         = log_path
        self.best_dir         = best_dir
        self.env_kwargs       = env_kwargs or {}
        self.global_best      = global_best
        self.global_best_stage = global_best_stage
        self.global_best_source = global_best_source
        self._last_eval       = 0
        self._t0             = time.time()
        self.history          = (json.loads(log_path.read_text())
                                 if log_path.exists() else [])
        self.best_mape        = min((h["mape_pct"] for h in self.history),
                                    default=float("inf"))

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval < self.every_n_steps:
            return True
        self._last_eval = self.num_timesteps

        vec_norm = self.model.get_vec_normalize_env()
        mapes, times = rollout_eval(self.model, self.eval_env,
                                    self.n_eval_episodes, self.seed_offset,
                                    vec_norm=vec_norm)
        m = summarise_eval(mapes, times)
        mape_mean, mape_p90 = m["mape_pct"], m["p90_pct"]
        succ, mean_time = m["success_rate"], m["mean_time_s"]

        if self.verbose:
            print(f"[E2 eval @ step {self.num_timesteps}]  "
                  f"MAPE={mape_mean:.2f}%  p90={mape_p90:.2f}%  "
                  f"success(<5%)={succ:.1%}  mean_scan_time={mean_time:.1f}s")

        self.history.append({
            "step":         self.num_timesteps,
            "wall_s":       time.time() - self._t0,   # MAPE-vs-wallclock plotting
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
        if self.global_best is not None:
            self.global_best.maybe_save(
                model=self.model,
                vec_norm=vec_norm,
                metrics=m,
                stage=self.global_best_stage or "unknown",
                source=self.global_best_source,
                step=int(self.num_timesteps),
                extra={
                    "eval_log_path": str(self.log_path),
                    "seed_offset": self.seed_offset,
                    "n_eval_episodes": self.n_eval_episodes,
                },
            )
        return True


class ProgressETACallback(BaseCallback):
    """Log a wall-clock ETA to completion at the end of each PPO rollout.

    Uses an EMA of seconds-per-timestep (the per-step cost drifts as the policy
    concentrates its TR distribution, since sim cost ∝ Npe·TR), and records
    `time/eta_hours` so it shows up next to `time/fps` in the SB3 table. The
    multi-fidelity orchestrator also reads `sec_per_step` as the per-fidelity
    cost λ_k in its switch criterion.
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

    @property
    def sec_per_step(self) -> float | None:
        return self._ema_spp

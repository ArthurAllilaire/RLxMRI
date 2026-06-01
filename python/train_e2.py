"""Train a PPO agent on the E2 multi-sphere T1 mapping environment.

Usage:
    python python/train_e2.py --timesteps 200000 --out runs/e2/ppo

The agent uses a continuous action space (5 parameters) normalised to [-1,1].
Observations are VecNormalise-scaled. Eval MAPE is logged every
--eval-interval steps.

PYTHON_JULIAPKG_OFFLINE=yes PYTHON_JULIAPKG_EXE=~/.julia/juliaup/julia-1.11.9+0.x64.linux.gnu/bin/julia \
  PYTHONUNBUFFERED=1 python -u python/train_e2.py \
    --out runs/e2/stop_check \
    --forward-model analytic --fix-te --allow-stop --include-sigma \
    --reward-mode delta_log_mape --time-penalty 0.3 \
    --field T15 --time-budget 240 --max-blocks 15 \
    --timesteps 20000 --n-steps 2048 --batch-size 256 \
    --eval-interval 5000 --eval-episodes 16 --checkpoint-interval 0 \
    2>&1 | tee runs/e2/stop_check/run.lo

Run A — delta_log_mape, cached_perline water, α-DOF, T1.5, 200k steps:
PYTHON_JULIAPKG_OFFLINE=yes PYTHON_JULIAPKG_EXE=~/.julia/juliaup/julia-1.11.9+0.x64.linux.gnu/bin/julia \
  PYTHONUNBUFFERED=1 python -u python/train_e2.py \
    --out runs/e2/ppo_runA_deltamape \
    --reward-mode delta_log_mape --mape-alpha 1.0 \
    --water-model cached_perline \
    --fix-te --learn-alpha \
    --field T15 --time-budget 240 --max-blocks 20 \
    --timesteps 200000 --n-steps 512 --batch-size 64 \
    2>&1 | tee runs/e2/ppo_runA_deltamape/run.log

Run A on GPU (T4) — ~3.7h vs ~12.5h for 4 CPU envs (benched). --use-gpu loads
CUDA so the Bloch sim runs on the device; keep --n-envs 1 (one env saturates the
GPU; multiple envs would just contend over the single device). Don't combine
--use-gpu with --n-envs>1.
PYTHON_JULIAPKG_OFFLINE=yes PYTHON_JULIAPKG_EXE=~/.julia/juliaup/julia-1.11.9+0.x64.linux.gnu/bin/julia \
  PYTHONUNBUFFERED=1 python -u python/train_e2.py \
    --out runs/e2/ppo_runA_deltamape \
    --reward-mode delta_log_mape --mape-alpha 1.0 \
    --water-model cached_perline \
    --fix-te --learn-alpha \
    --use-gpu --n-envs 1 \
    --field T15 --time-budget 240 --max-blocks 20 \
    --timesteps 200000 --n-steps 512 --batch-size 64 \
    2>&1 | tee runs/e2/ppo_runA_deltamape/run.log
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize

from qalibremd_gym.env_e2 import QalibreMDE2Env
from e2_config import add_e2_env_args, e2_env_kwargs
from e2_train_common import (
    build_model, build_vec_env,
    E2CheckpointCallback, E2EvalCallback, ProgressETACallback,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--timesteps",      type=int,   default=200_000)
    p.add_argument("--n-envs",         type=int,   default=1)
    p.add_argument("--n-steps",        type=int,   default=512,
                   help="PPO rollout length PER ENV. Each env collects n_steps "
                        "transitions before an update, so the rollout buffer the "
                        "optimiser sees = n_steps×n_envs. The ETA line prints once "
                        "per rollout (every n_steps×n_envs timesteps).")
    p.add_argument("--batch-size",     type=int,   default=64,
                   help="PPO minibatch size for the SGD epochs. The rollout "
                        "(n_steps×n_envs) should be divisible by it, else SB3 "
                        "warns about a ragged final minibatch.")
    p.add_argument("--eval-episodes",  type=int,   default=30)
    p.add_argument("--eval-interval",  type=int,   default=10_000)
    p.add_argument("--train-seed",     type=int,   default=0)
    p.add_argument("--eval-seed",      type=int,   default=500_000)
    p.add_argument("--out",            type=Path,  default=Path("runs/e2/ppo"))
    p.add_argument("--checkpoint-interval", type=int, default=50_000,
                   help="Save a checkpoint every N timesteps (0 = disabled)")
    p.add_argument("--resume", action="store_true",
                   help="Resume from latest checkpoint in --out directory")
    p.add_argument("--init-from", type=Path, default=None,
                   help="Warm-start from another run's saved policy: loads "
                        "policy.zip + vecnorm.pkl from this directory onto a "
                        "FRESH env (e.g. train on --forward-model analytic then "
                        "continue on bloch). Timestep count restarts from 0. The "
                        "source run's obs/action layout must match (same "
                        "--include-sigma/--fix-te/--learn-alpha/--allow-stop/"
                        "--subset-size). Mutually exclusive with --resume.")
    add_e2_env_args(p)   # shared env flags (incl. --Nfe/--Npe/--voxel-mm/--use-gpu)
    args = p.parse_args()

    if args.resume and args.init_from is not None:
        raise ValueError("--resume and --init-from are mutually exclusive")

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
            "n_steps":       args.n_steps,
            "batch_size":    args.batch_size,
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
    vec_env = build_vec_env(env_kwargs, n_envs=args.n_envs,
                            train_seed=args.train_seed)

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
    elif args.init_from is not None:
        # Warm-start: load weights + obs-normalisation stats from another run
        # and keep training on THIS (freshly built) env. Used for the
        # analytic→bloch curriculum. Timestep count restarts from 0.
        remaining = args.timesteps
        src = args.init_from
        vec_env = VecNormalize.load(str(src / "vecnorm.pkl"), vec_env)
        vec_env.training = True
        model = PPO.load(str(src / "policy"), env=vec_env,
                         tensorboard_log=str(args.out / "tb"))
        print(f"[E2] Warm-started from {src} "
              f"({args.timesteps} fresh steps on this env)")
    else:
        remaining = args.timesteps
        model = build_model(vec_env, n_steps=args.n_steps,
                            batch_size=args.batch_size,
                            tensorboard_log=str(args.out / "tb"))

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

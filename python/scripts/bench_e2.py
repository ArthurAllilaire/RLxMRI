"""Estimate E2 training wall-time from a short mini-run.

Mirrors the train_e2.py flags (via the shared e2_config module), so you point it
at the *same* command, swap the script name, and get a projected total run time
before committing days of compute. Per-step sim cost ≈ Npe·TR·n_spins (+ Nfe ADC
samples/shot), so changing --Npe/--Nfe/--voxel-mm/--use-gpu/--n-envs here lets you
price each speedup lever (run-time plan §4).

Usage (price the Run A config):
    PYTHON_JULIAPKG_OFFLINE=yes python python/scripts/bench_e2.py \
        --fix-te --learn-alpha --field T15 --noise 50 \
        --time-budget 160 --max-blocks 30 --timesteps 300000

    # price the old fast config / a speedup:
    python python/scripts/bench_e2.py ... --Npe 8 --Nfe 16 --voxel-mm 3.0
    python python/scripts/bench_e2.py ... --n-envs 8     # process-level parallelism
    python python/scripts/bench_e2.py ... --nthreads 8   # spin-loop threads (RAM-cheap)

    PYTHON_JULIAPKG_OFFLINE=yes python python/scripts/bench_e2.py \
        --fix-te --learn-alpha --field T15 --noise 50 \
        --time-budget 160 --max-blocks 30 --timesteps 300000 --n-envs 2 --nthreads 4

All comes from the below run:
PYTHON_JULIAPKG_OFFLINE=yes python python/train_e2.py \
   --fix-te --learn-alpha --field T15 \
   --reward-mode delta_mape --terminal-bonus 0.0 --mape-alpha 1.0 \
   --sigma-method asymptotic --noise 50 --time-budget 160 --max-blocks 30 \
   --timesteps 300000 --out runs/e2/rerun_A_alpha

Using cpu device
Logging to runs/e2/rerun_A_alpha/tb/PPO_1
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 1.86     |
|    ep_rew_mean     | -0.79    |
| time/              |          |
|    fps             | 0        |
|    iterations      | 1        |
|    time_elapsed    | 5214     |
|    total_timesteps | 2048     |
---------------------------------

5214/2048 * 200_000 = 509_200 seconds = 6 days: OUCH! especially with ablations
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from qalibremd_gym.env_e2 import QalibreMDE2Env
from e2_config import add_e2_env_args, e2_env_kwargs
from _bench_io import git_meta, append_result, RUNS_DIR


def _fmt(seconds: float) -> str:
    h = seconds / 3600.0
    return f"{seconds:,.0f}s  ({h:.1f}h | {h/24:.2f}d)"


def _make_env_fn(seed: int, env_kwargs: dict):
    def _init():
        return QalibreMDE2Env(rng_seed=seed, **env_kwargs)
    return _init


def _time_single(env_kwargs: dict, seed: int, warmup: int, steps: int):
    """Time `steps` env-steps on one env. Returns (per_step_times, ep_lens,
    n_no_sim, n_spins)."""
    env = QalibreMDE2Env(rng_seed=seed, **env_kwargs)
    rng = np.random.default_rng(seed)

    def rand_action():
        return rng.uniform(-1.0, 1.0,
                           size=env.action_space.shape).astype(np.float32)

    # Warm-up (discarded): the first simulate() triggers tens of seconds of
    # Julia JIT compilation that must not pollute the timing.
    env.reset(seed=seed)
    n_spins = int(len(env._env.phantom.x))   # phantom built at reset
    for _ in range(warmup):
        _o, _r, done, _t, _i = env.step(rand_action())
        if done:
            env.reset(seed=seed)

    per_step, ep_lens, n_no_sim = [], [], 0
    cur_len = 0
    for _ in range(steps):
        t0 = time.perf_counter()
        _o, _r, done, _t, info = env.step(rand_action())
        per_step.append(time.perf_counter() - t0)
        cur_len += 1
        # Budget-exceeded terminal steps don't call simulate() (block_time==0).
        if info.get("budget_exceeded") or info.get("block_time", 1.0) == 0.0:
            n_no_sim += 1
        if done:
            ep_lens.append(cur_len)
            cur_len = 0
            env.reset(seed=seed)
    return np.array(per_step), ep_lens, n_no_sim, n_spins


def _time_vec(env_kwargs: dict, seed: int, n_envs: int, warmup: int, steps: int):
    """Time `steps` vectorised steps across n_envs subprocesses. Returns the
    effective wall-seconds per single env-step (total_wall / (steps*n_envs))."""
    from stable_baselines3.common.vec_env import SubprocVecEnv
    venv = SubprocVecEnv(
        [_make_env_fn(seed + i, env_kwargs) for i in range(n_envs)],
        start_method="spawn",
    )
    act_dim = venv.action_space.shape[0]
    rng = np.random.default_rng(seed)
    venv.reset()
    for _ in range(warmup):
        venv.step(rng.uniform(-1, 1, size=(n_envs, act_dim)).astype(np.float32))
    t0 = time.perf_counter()
    for _ in range(steps):
        venv.step(rng.uniform(-1, 1, size=(n_envs, act_dim)).astype(np.float32))
    wall = time.perf_counter() - t0
    venv.close()
    return wall / (steps * n_envs)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # Projection / training-shape knobs (kept consistent with train_e2.py).
    p.add_argument("--timesteps",     type=int, default=300_000,
                   help="Training timesteps to project to.")
    p.add_argument("--n-envs",        type=int, default=1,
                   help="If >1, also measure real SubprocVecEnv throughput across "
                        "this many worker processes (multi-core lever).")
    p.add_argument("--nthreads",      type=int, default=1,
                   help="Julia/Koma threads per env: parallelises the Bloch "
                        "spin-loop *inside* one simulate(), via "
                        "PYTHON_JULIACALL_THREADS. Orthogonal to --n-envs "
                        "(process-level) and RAM-cheap (shares one runtime per "
                        "worker). Koma's default Nthreads tracks nthreads(), so "
                        "setting this before boot is sufficient — no Julia "
                        "change. Total cores used ≈ n_envs × nthreads.")
    p.add_argument("--eval-episodes", type=int, default=30)
    p.add_argument("--eval-interval", type=int, default=10_000)
    # Bench controls.
    p.add_argument("--warmup", type=int, default=3,
                   help="Discarded warm-up steps to absorb Julia JIT compilation.")
    p.add_argument("--steps",  type=int, default=30,
                   help="Timed steps for the per-step estimate.")
    p.add_argument("--seed",   type=int, default=0)
    p.add_argument("--out", type=str, default=str(RUNS_DIR / "bench_results.json"),
                   help="Append a reproducible result record (config + git + "
                        "metrics) to this JSON list. Set to '' to disable.")
    add_e2_env_args(p)
    args = p.parse_args()

    # Must be set BEFORE the Julia runtime boots (first env construction below).
    # juliacall reads PYTHON_JULIACALL_THREADS at init; spawn workers inherit
    # the parent env, so this propagates to SubprocVecEnv children too.
    # HANDLE_SIGNALS=yes is REQUIRED with >1 thread: without it juliacall's
    # signal handling interferes with Julia threading — the multithreaded
    # Bloch sim either runs with no speedup or segfaults outright (measured).
    # Trade-off: it disables Python's Ctrl-C (KeyboardInterrupt) handling.
    if args.nthreads > 1:
        os.environ["PYTHON_JULIACALL_THREADS"] = str(args.nthreads)
        os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"
        os.environ.setdefault("JULIA_NUM_THREADS", str(args.nthreads))

    env_kwargs = e2_env_kwargs(args)
    print(f"[bench] env: Nfe={args.Nfe} Npe={args.Npe} voxel={args.voxel_mm}mm "
          f"use_gpu={args.use_gpu} budget={args.time_budget}s "
          f"max_blocks={args.max_blocks} noise={args.noise} "
          f"nthreads={args.nthreads}")
    if args.n_envs > 1 and args.nthreads > 1:
        print(f"[bench] cores in use ≈ n_envs×nthreads = "
              f"{args.n_envs}×{args.nthreads} = {args.n_envs * args.nthreads} "
              f"(of {os.cpu_count()} available)")
    print(f"[bench] warming up ({args.warmup} steps) + timing "
          f"{args.steps} steps on 1 env …")

    per_step, ep_lens, n_no_sim, n_spins = _time_single(
        env_kwargs, args.seed, args.warmup, args.steps)
    print(f"[bench] phantom spins: {n_spins:,}")

    mean_step = float(per_step.mean())
    med_step  = float(np.median(per_step))
    p90_step  = float(np.percentile(per_step, 90))
    n_sim     = args.steps - n_no_sim
    mean_sim  = float(per_step.sum() / max(n_sim, 1)) if n_sim else float("nan")
    mean_ep   = float(np.mean(ep_lens)) if ep_lens else float(args.steps)

    print("\n── per-step wall time (1 env) ─────────────────────────────")
    print(f"  mean   {mean_step:.3f}s   median {med_step:.3f}s   "
          f"p90 {p90_step:.3f}s")
    print(f"  no-sim (budget-exceeded) steps: {n_no_sim}/{args.steps}")
    print(f"  mean per simulating step: {mean_sim:.3f}s")
    print(f"  mean episode length: {mean_ep:.2f} steps")

    # Effective per-step used for the rollout projection.
    eff_step = mean_step
    if args.n_envs > 1:
        print(f"\n[bench] measuring SubprocVecEnv throughput "
              f"(n_envs={args.n_envs}) …")
        eff_step = _time_vec(env_kwargs, args.seed, args.n_envs,
                             args.warmup, max(5, args.steps // 2))
        speedup = mean_step / eff_step if eff_step > 0 else float("nan")
        print(f"  effective per single-step wall: {eff_step:.3f}s "
              f"→ {speedup:.1f}× vs 1 env")

    # ── projection ─────────────────────────────────────────────────────────
    rollout_s = args.timesteps * eff_step
    n_evals   = args.timesteps // max(args.eval_interval, 1)
    eval_steps = n_evals * args.eval_episodes * mean_ep
    eval_s    = eval_steps * mean_step       # eval runs serially on one env
    total_s   = rollout_s + eval_s

    print("\n── projected wall time for --timesteps "
          f"{args.timesteps:,} ──────────")
    if args.n_envs > 1:
        print(f"  rollout (n_envs={args.n_envs}): {_fmt(rollout_s)}")
    else:
        print(f"  rollout:                {_fmt(rollout_s)}")
    print(f"  eval ({n_evals}×{args.eval_episodes} eps, serial): {_fmt(eval_s)}")
    print(f"  TOTAL:                  {_fmt(total_s)}")
    print("\n  caveat: per-step cost ∝ Npe·TR; a random policy's TR distribution "
          "differs\n  from a trained one, so treat this as a planning estimate.")

    if args.out:
        record = {
            **git_meta(),
            "args": vars(args),
            "metrics": {
                "n_spins": n_spins,
                "mean_step_s": mean_step,
                "median_step_s": med_step,
                "p90_step_s": p90_step,
                "mean_sim_s": mean_sim,
                "mean_ep_len": mean_ep,
                "n_no_sim": n_no_sim,
                "eff_step_s": eff_step,
                "rollout_s": rollout_s,
                "eval_s": eval_s,
                "total_s": total_s,
            },
        }
        out = append_result(args.out, record)
        print(f"\n[bench] appended result → {out}")


if __name__ == "__main__":
    main()

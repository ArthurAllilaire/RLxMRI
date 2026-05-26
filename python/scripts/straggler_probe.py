"""Measure the SubprocVecEnv straggler tax for the E2 env.

Each macro-step, SubprocVecEnv blocks until ALL workers finish their step(), so
the parent pays the *slowest* worker, not the average. On a variable-TR workload
(sim cost ∝ Npe·TR, TR ∈ [0.5, 5]s) fast workers sit idle at the barrier. This
script logs each worker's own step() wall time — smuggled home in the info dict,
which SubprocVecEnv already returns to the parent for free — and reports the
idle fraction per macro-step:

    idle_frac = 1 - mean(worker_times) / max(worker_times)

averaged over the run = the fraction of worker compute wasted waiting. That's the
quantified cost of the --n-envs lever; --nthreads avoids it entirely because a
single sim's spin-work is uniform (see thread_probe.py).

Measured so far (RANDOM actions, n_envs=2, 16 steps — worst-case upper bound):
    idle fraction ............ 32.6% ± 20.3%   (waste at the barrier)
    parallel efficiency ...... 67.4% of an ideal 2× rollout
    IPC/parent overhead ...... ~180 ms/step
    effective rollout speedup  1.32× (vs nominal 2×)
  i.e. random-policy n_envs=2 buys ~1.3×, not 2×. A trained policy's narrower TR
  distribution should lower this; re-run with --policy for the report figure.
  (16 steps is noisy — use --steps 60+ to tighten the mean.)

Usage:
    # random-action UPPER BOUND (no checkpoint needed):
    PYTHON_JULIAPKG_OFFLINE=yes python python/scripts/straggler_probe.py \
        --fix-te --learn-alpha --field T15 --noise 50 \
        --time-budget 160 --max-blocks 30 --n-envs 4 --steps 60

    # REALISTIC number from a trained policy (narrower TR distribution):
    PYTHON_JULIAPKG_OFFLINE=yes python python/scripts/straggler_probe.py \
        --fix-te --learn-alpha --field T15 --noise 50 \
        --time-budget 160 --max-blocks 30 --n-envs 4 --steps 60 \
        --policy runs/e2/ppo/policy.zip --vecnorm runs/e2/ppo/vecnorm.pkl
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# python/ on the path so qalibremd_gym + e2_config import regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import gymnasium as gym

from qalibremd_gym.env_e2 import QalibreMDE2Env
from e2_config import add_e2_env_args, e2_env_kwargs
from _bench_io import git_meta, append_result, RUNS_DIR


class StepTimer(gym.Wrapper):
    """Record this worker's own step() wall time into info['worker_step_s'].

    Defined at module top-level so SubprocVecEnv's spawned workers can rebuild it.
    The duration rides the existing info-dict IPC back to the parent — no extra
    pipes or shared memory needed.
    """

    def step(self, action):
        t0 = time.perf_counter()
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["worker_step_s"] = time.perf_counter() - t0
        return obs, reward, terminated, truncated, info


def _make_env_fn(seed: int, env_kwargs: dict):
    def _init():
        return StepTimer(QalibreMDE2Env(rng_seed=seed, **env_kwargs))
    return _init


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-envs", type=int, default=4,
                   help="Worker processes (≥2 — need a barrier to measure).")
    p.add_argument("--steps", type=int, default=60,
                   help="Timed macro-steps after warmup.")
    p.add_argument("--warmup", type=int, default=3,
                   help="Discarded steps to absorb Julia JIT compilation.")
    p.add_argument("--seed", type=int, default=0)
    # --policy branch: use a trained checkpoint's action distribution instead of
    # uniform-random. --vecnorm feeds the policy the obs normalisation it trained
    # with; without it the policy sees raw obs and may act off-distribution.
    p.add_argument("--policy", type=str, default=None,
                   help="Path to a saved PPO policy (.zip). Omit for random "
                        "actions (worst-case upper bound on the tax).")
    p.add_argument("--vecnorm", type=str, default=None,
                   help="Path to the matching VecNormalize stats (.pkl). "
                        "Recommended whenever --policy is set.")
    p.add_argument("--deterministic", action="store_true",
                   help="Use the policy's mean action. Default samples, matching "
                        "PPO's stochastic rollout collection (more realistic).")
    p.add_argument("--out", type=str,
                   default=str(RUNS_DIR / "straggler_results.json"),
                   help="Append a reproducible result record (config + git + "
                        "metrics) to this JSON list. Set to '' to disable.")
    add_e2_env_args(p)
    args = p.parse_args()

    if args.n_envs < 2:
        p.error("--n-envs must be ≥ 2 — there's no barrier to measure with 1 env.")

    env_kwargs = e2_env_kwargs(args)
    mode = f"policy={args.policy}" if args.policy else "random actions"
    print(f"[straggler] n_envs={args.n_envs} steps={args.steps} "
          f"Nfe={args.Nfe} Npe={args.Npe} noise={args.noise} | {mode}")

    from stable_baselines3.common.vec_env import SubprocVecEnv
    venv = SubprocVecEnv(
        [_make_env_fn(args.seed + i, env_kwargs) for i in range(args.n_envs)],
        start_method="spawn",
    )

    model = None
    if args.policy:
        from stable_baselines3 import PPO
        model = PPO.load(args.policy)
        if args.vecnorm:
            from stable_baselines3.common.vec_env import VecNormalize
            venv = VecNormalize.load(args.vecnorm, venv)
            venv.training = False        # freeze running stats
            venv.norm_reward = False     # we only need obs normalisation here
        else:
            print("[straggler] WARNING: --policy without --vecnorm — the policy "
                  "sees un-normalised obs and may act off-distribution.")

    act_dim = venv.action_space.shape[0]
    rng = np.random.default_rng(args.seed)

    def get_actions(obs):
        if model is not None:
            actions, _ = model.predict(obs, deterministic=args.deterministic)
            return actions
        return rng.uniform(-1.0, 1.0,
                           size=(args.n_envs, act_dim)).astype(np.float32)

    obs = venv.reset()
    for _ in range(args.warmup):
        obs, _r, _d, _i = venv.step(get_actions(obs))

    idle_fracs, barrier_walls, max_w, mean_w = [], [], [], []
    n_idle_skipped = 0
    for _ in range(args.steps):
        t0 = time.perf_counter()
        obs, _r, _d, infos = venv.step(get_actions(obs))
        barrier_wall = time.perf_counter() - t0
        w = np.array([float(i.get("worker_step_s", 0.0)) for i in infos])
        if w.max() <= 0.0:               # all workers hit no-sim (budget) steps
            n_idle_skipped += 1
            continue
        idle_fracs.append(1.0 - w.mean() / w.max())
        barrier_walls.append(barrier_wall)
        max_w.append(w.max())
        mean_w.append(w.mean())
    venv.close()

    if not idle_fracs:
        print("[straggler] no simulating steps captured — try more --steps.")
        return

    idle = np.array(idle_fracs)
    mw, mnw, bw = np.array(max_w), np.array(mean_w), np.array(barrier_walls)
    ipc = bw - mw                        # parent overhead beyond slowest worker

    print("\n── straggler tax (synchronous SubprocVecEnv barrier) ─────────")
    print(f"  measured macro-steps:        {len(idle)} "
          f"(+{n_idle_skipped} all-idle steps skipped)")
    print(f"  per-worker sim time:         max {mw.mean():.3f}s  "
          f"mean {mnw.mean():.3f}s")
    print(f"  idle fraction (waste):       {idle.mean()*100:5.1f}%  "
          f"± {idle.std()*100:.1f}%")
    print(f"  ⇒ parallel efficiency:       {(1-idle.mean())*100:5.1f}% "
          f"of ideal {args.n_envs}× rollout")
    print(f"  IPC/parent overhead:         {ipc.mean()*1e3:.0f}ms/step "
          f"(barrier_wall − slowest worker)")
    eff_speedup = args.n_envs * (mnw.mean() / mw.mean())
    print(f"  effective rollout speedup:   {eff_speedup:.2f}× "
          f"(vs nominal {args.n_envs}×)")

    if args.out:
        record = {
            **git_meta(),
            "args": vars(args),
            "metrics": {
                "n_envs": args.n_envs,
                "mode": "policy" if args.policy else "random",
                "measured_steps": len(idle),
                "idle_frac_mean": float(idle.mean()),
                "idle_frac_std": float(idle.std()),
                "parallel_eff": float(1 - idle.mean()),
                "max_worker_s": float(mw.mean()),
                "mean_worker_s": float(mnw.mean()),
                "ipc_ms": float(ipc.mean() * 1e3),
                "eff_speedup": float(eff_speedup),
            },
        }
        out = append_result(args.out, record)
        print(f"\n[straggler] appended result → {out}")


if __name__ == "__main__":
    main()

"""PPO trainer for Paradigm A (MRzero-in-the-step adaptive sequence design).

CPU smoke:
    python python/train_paradigm_a.py --timesteps 5000 --out runs/paradigm_a/smoke

GPU full:
    python python/train_paradigm_a.py --timesteps 500000 --out runs/paradigm_a/full

Use the MRzero venv:  source .venv_mrzero/bin/activate
"""
from __future__ import annotations
import os, sys, argparse, time
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))

from qalibremd_gym.env_paradigm_a import ParadigmAEnv


def make_env(seed):
    env = ParadigmAEnv(seed=seed)
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=5000)
    ap.add_argument("--out", type=str, default="runs/paradigm_a/smoke")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = os.path.join(ROOT, args.out) if not os.path.isabs(args.out) else args.out
    os.makedirs(out, exist_ok=True)

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    env = DummyVecEnv([lambda: make_env(args.seed)])

    tb_log = None
    try:
        import tensorboard  # noqa: F401
        tb_log = os.path.join(out, "tb")
    except ImportError:
        print("(tensorboard not installed; skipping TB logging)")
    model = PPO("MlpPolicy", env,
                n_steps=128, batch_size=64, n_epochs=4,
                learning_rate=3e-4, verbose=1,
                tensorboard_log=tb_log)
    t0 = time.time()
    model.learn(total_timesteps=args.timesteps)
    print(f"trained in {time.time()-t0:.1f} s")
    model.save(os.path.join(out, "policy"))
    print(f"saved {out}/policy")


if __name__ == "__main__":
    main()

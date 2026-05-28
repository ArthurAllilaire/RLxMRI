"""Evaluate a trained E2 PPO agent on held-out phantom configs.

Usage:
    python python/eval_e2.py --policy runs/e2/ppo/policy.zip \
                             --vecnorm runs/e2/ppo/vecnorm.pkl \
                             --episodes 50

Reports:
  - T1 MAPE per sphere (mean ± std across episodes)
  - Adaptive vs fixed-TI baseline comparison
  - TI-choice histogram (checks for non-trivial policy)
  - Noise robustness sweep (optional)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from qalibremd_gym.env_e2 import QalibreMDE2Env
from qalibremd_gym.schedules import log_ti_grid


# ── Fixed-TI grid baseline ─────────────────────────────────────────────────


def _fixed_grid_action(step: int, n_ti: int = 7) -> np.ndarray:
    """Cycle through a fixed log-spaced TI grid, TE=20ms, TR=4s, α=90°."""
    tis = log_ti_grid(n=n_ti)
    ti  = float(tis[step % n_ti])
    return np.array([ti, 0.020, 4.0, 90.0, 0.0], dtype=np.float32)


def evaluate_fixed_grid(env: QalibreMDE2Env, n_episodes: int,
                         seed_offset: int) -> dict:
    mapes, per_sphere = [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_offset + ep)
        done, step, info = False, 0, {}
        while not done:
            # Convert physical action → normalised action ([-1, 1])
            phys = _fixed_grid_action(step)
            norm = 2.0 * (phys - env._ACT_LO) / (env._ACT_HI - env._ACT_LO) - 1.0
            obs, _r, done, _trunc, info = env.step(norm)
            step += 1
        mapes.append(float(info.get("mape", np.nan)))
        per_sphere.append(np.abs(env.T1_est - env.T1_true) / env.T1_true)
    return {
        "mape_pct": float(np.nanmean(mapes)) * 100,
        "mape_p90_pct": float(np.nanpercentile(mapes, 90)) * 100,
        "per_sphere_mape_pct": np.nanmean(np.array(per_sphere), axis=0) * 100,
    }


def evaluate_policy(policy_path: Path, vecnorm_path: Optional[Path],
                    n_episodes: int, seed_offset: int,
                    **env_kwargs) -> dict:
    # Always step the raw env directly (per-episode reseed). If a VecNormalize
    # checkpoint is provided, use it only to normalize observations before
    # they reach the policy. This avoids DummyVecEnv's auto-reset semantics
    # and keeps per-episode seeds consistent with the baseline.
    raw_env = QalibreMDE2Env(rng_seed=seed_offset, **env_kwargs)
    vec_norm = None
    if vecnorm_path is not None and vecnorm_path.exists():
        venv_tmp = DummyVecEnv([lambda: QalibreMDE2Env(rng_seed=seed_offset, **env_kwargs)])
        vec_norm = VecNormalize.load(str(vecnorm_path), venv_tmp)
        vec_norm.training = False
        vec_norm.norm_reward = False

    model = PPO.load(str(policy_path))

    def _norm(o):
        if vec_norm is None:
            return o
        return vec_norm.normalize_obs(np.expand_dims(o, 0))[0]

    mapes, per_sphere, ti_choices, times = [], [], [], []
    # D2 readout: per-pool-index MAPE so we can see which T1 values fail.
    pool_apes: dict[int, list[float]] = {}

    for ep in range(n_episodes):
        obs, _ = raw_env.reset(seed=seed_offset + ep)
        done, info = False, {}
        ep_tis = []

        while not done:
            action, _ = model.predict(_norm(obs), deterministic=True)
            obs, _r, done, _trunc, info = raw_env.step(action)
            ep_tis.append(float(info.get("TI", np.nan)))

        mapes.append(float(info.get("mape", np.nan)))
        t1_est  = np.asarray(info.get("T1_est",  raw_env.T1_est),  dtype=np.float64)
        t1_true = np.asarray(info.get("T1_true", raw_env.T1_true), dtype=np.float64)
        sphere_idx = np.asarray(info.get("sphere_indices",
                                         np.arange(1, len(t1_true) + 1)),
                                dtype=np.int64)
        ape = np.abs(t1_est - t1_true) / t1_true
        per_sphere.append(ape)
        for pi, a in zip(sphere_idx, ape):
            pool_apes.setdefault(int(pi), []).append(float(a))
        ti_choices.extend(ep_tis)
        times.append(float(info.get("time_s", raw_env.time_used_s)))

    per_sphere_arr = np.nanmean(np.array(per_sphere), axis=0)
    per_pool = {pi: (float(np.nanmean(v)) * 100, len(v))
                for pi, v in sorted(pool_apes.items())}

    return {
        "mape_pct":           float(np.nanmean(mapes)) * 100,
        "mape_p90_pct":       float(np.nanpercentile(mapes, 90)) * 100,
        "success_5pct":       float(np.mean([m < 0.05 for m in mapes])),
        "per_sphere_mape_pct": per_sphere_arr * 100,
        "per_pool_mape_pct":  per_pool,
        "mean_scan_time_s":   float(np.mean(times)),
        "ti_choices_s":       ti_choices,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy",   type=Path, required=True)
    p.add_argument("--vecnorm",  type=Path, default=None)
    p.add_argument("--episodes", type=int,  default=50)
    p.add_argument("--seed",     type=int,  default=500_000)
    p.add_argument("--field",    type=str,  default="T3",
                   choices=["T3", "T15"])
    p.add_argument("--max-blocks", type=int, default=15)
    p.add_argument("--time-budget", type=float, default=120.0)
    p.add_argument("--subset-size", type=int, default=None,
                   help="Evaluate on random k-sphere subsets instead of the "
                        "full 14-sphere plate.")
    p.add_argument("--phase-sensitive", action="store_true")
    p.add_argument("--sigma-method", type=str, default="bootstrap",
                   choices=["asymptotic", "profile_likelihood", "bootstrap"])
    p.add_argument("--noise-sweep", action="store_true",
                   help="Sweep absolute σ ∈ {0, 0.002, 0.005, 0.01, 0.02}")
    p.add_argument("--simplified-action", action="store_true",
                   help="Required if the policy was trained with "
                        "--simplified-action (3-dim action space)")
    p.add_argument("--fix-te", action="store_true",
                   help="Required if the policy was trained with --fix-te "
                        "([TI, TR] action space; TE fixed at 20ms).")
    p.add_argument("--learn-alpha", action="store_true",
                   help="Required if the policy was trained with --learn-alpha "
                        "([TI, TR, α] action space).")
    p.add_argument("--log-ti-action", action="store_true",
                   help="Required if the policy was trained with "
                        "--log-ti-action (log-spaced TI mapping).")
    p.add_argument("--oracle-fit", action="store_true",
                   help="D2 diagnostic: narrow the fitter's T1 grid to a "
                        "log-band of ±oracle-band around T1_true per sphere. "
                        "Tests whether multimodal-SSE wrong-basin convergence "
                        "is the bottleneck. Cheats — never use for reported "
                        "results.")
    p.add_argument("--oracle-band", type=float, default=2.0,
                   help="Oracle grid half-width (multiplicative). 2.0 = ±1 octave.")
    p.add_argument("--fitter-n-grid", type=int, default=200,
                   help="n_grid for fit_t1_generalized_ir. §17.10 control: "
                        "use 2000 to separate grid coarseness from genuine "
                        "multimodal SSE under the baseline fitter.")
    p.add_argument("--include-image", action="store_true",
                   help="Required if the policy was trained with --include-image "
                        "(obs prepends the flattened recon image).")
    p.add_argument("--include-sigma", action="store_true",
                   help="Required if the policy was trained with --include-sigma "
                        "(obs appends the per-sphere fitter-σ channel).")
    p.add_argument("--water-model", type=str, default="bloch",
                   choices=["bloch", "cached_perline", "analytic"])
    p.add_argument("--noise-sigma-abs", type=float, default=50.0)
    p.add_argument("--reward-mode", type=str, default="neg_mape")
    p.add_argument("--terminal-bonus", type=float, default=0.5)
    p.add_argument("--mape-alpha", type=float, default=1.0)
    p.add_argument("--allow-stop", action="store_true")
    p.add_argument("--use-gpu", action="store_true")
    p.add_argument("--nfe", type=int, default=None)
    p.add_argument("--npe", type=int, default=None)
    args = p.parse_args()

    env_kwargs = dict(cfg_field=args.field,
                       max_blocks=args.max_blocks,
                       time_budget_s=args.time_budget,
                       subset_size=args.subset_size,
                       phase_sensitive=args.phase_sensitive,
                       sigma_method=args.sigma_method,
                       simplified_action=args.simplified_action,
                       fix_te=args.fix_te,
                       learn_alpha=args.learn_alpha,
                       log_ti_action=args.log_ti_action,
                       oracle_fit=args.oracle_fit,
                       oracle_band=args.oracle_band,
                       fitter_n_grid=args.fitter_n_grid,
                       include_image=args.include_image,
                       include_sigma=args.include_sigma,
                       water_model=args.water_model,
                       noise_sigma_abs=args.noise_sigma_abs,
                       reward_mode=args.reward_mode,
                       terminal_bonus=args.terminal_bonus,
                       mape_alpha=args.mape_alpha,
                       allow_stop=args.allow_stop,
                       use_gpu=args.use_gpu)
    if args.nfe is not None:
        env_kwargs["Nfe"] = args.nfe
    if args.npe is not None:
        env_kwargs["Npe"] = args.npe

    print("=" * 60)
    print(f"E2 Evaluation — policy: {args.policy}")
    print("=" * 60)

    # ── PPO agent results ────────────────────────────────────────────────
    print(f"\nEvaluating PPO agent on {args.episodes} held-out configs …")
    res = evaluate_policy(args.policy, args.vecnorm, args.episodes,
                          args.seed, **env_kwargs)
    print(f"  MAPE        = {res['mape_pct']:.2f}%")
    print(f"  p90 MAPE    = {res['mape_p90_pct']:.2f}%")
    print(f"  Success<5%  = {res['success_5pct']:.1%}")
    print(f"  Mean time   = {res['mean_scan_time_s']:.1f}s")

    ps = res["per_sphere_mape_pct"]
    print(f"\n  Per-active-slot MAPE [%]:")
    for i, v in enumerate(ps):
        print(f"    Slot {i+1:2d}: {v:.2f}%")

    pp = res.get("per_pool_mape_pct", {})
    if pp:
        print(f"\n  Per-pool-index MAPE [%] (T1_ARRAY index 1..14):")
        for pi, (mape_v, n_obs) in pp.items():
            print(f"    Pool {pi:2d}: {mape_v:7.2f}%   (n_eps = {n_obs})")

    tis = [t for t in res["ti_choices_s"] if not np.isnan(t)]
    if tis:
        print(f"\n  TI histogram (log10 s): mean={np.mean(tis):.3f}  "
              f"std={np.std(tis):.3f}")
        bins = np.histogram(np.log10(tis), bins=10)
        for lo, hi, cnt in zip(np.exp(bins[1][:-1]), np.exp(bins[1][1:]), bins[0]):
            bar = "█" * (cnt * 30 // (max(bins[0]) + 1) + 1)
            print(f"    [{lo:.3f}-{hi:.3f}s]: {bar}")

    # ── Fixed-grid baseline ───────────────────────────────────────────────
    # Baseline always uses the full 5-dim action space: it cycles through a
    # physical [TI, TE, TR, α, slice_z] vector by construction.
    print(f"\nEvaluating fixed-TI grid baseline on same configs …")
    base_kwargs = {k: v for k, v in env_kwargs.items()
                   if k not in ("simplified_action", "log_ti_action",
                                "fix_te", "learn_alpha")}
    env_base = QalibreMDE2Env(rng_seed=args.seed, **base_kwargs)
    base = evaluate_fixed_grid(env_base, args.episodes, args.seed)
    print(f"  MAPE        = {base['mape_pct']:.2f}%")
    print(f"  p90 MAPE    = {base['mape_p90_pct']:.2f}%")

    print(f"\n  Agent  MAPE = {res['mape_pct']:.2f}%  "
          f"(baseline = {base['mape_pct']:.2f}%, "
          f"speedup factor = {base['mape_pct']/max(res['mape_pct'], 0.01):.1f}×)")

    # ── Optional noise sweep ──────────────────────────────────────────────
    if args.noise_sweep:
        print("\nNoise robustness sweep:")
        print(f"  {'σ':>6}  {'Agent MAPE':>12}  {'Baseline MAPE':>14}")
        for sigma in [0.0, 0.002, 0.005, 0.01, 0.02]:
            kw = dict(cfg_field=args.field, noise_sigma_abs=sigma)
            r_s = evaluate_policy(args.policy, args.vecnorm,
                                  args.episodes // 2, args.seed, **kw)
            env_b = QalibreMDE2Env(rng_seed=args.seed, **kw)
            b_s = evaluate_fixed_grid(env_b, args.episodes // 2, args.seed)
            print(f"  {sigma:>6.2f}  {r_s['mape_pct']:>12.2f}%  "
                  f"{b_s['mape_pct']:>14.2f}%")

    # ── Save summary ──────────────────────────────────────────────────────
    out_dir = args.policy.parent
    summary = {
        "policy":        str(args.policy),
        "episodes":      args.episodes,
        "oracle_fit":    bool(args.oracle_fit),
        "oracle_band":   float(args.oracle_band),
        "agent_mape_pct":    res["mape_pct"],
        "agent_mape_p90_pct": res["mape_p90_pct"],
        "baseline_mape_pct": base["mape_pct"],
        "per_sphere":    res["per_sphere_mape_pct"].tolist(),
        "per_pool":      {str(k): list(v)
                           for k, v in res.get("per_pool_mape_pct", {}).items()},
    }
    out_name = "eval_summary_oracle.json" if args.oracle_fit else "eval_summary.json"
    with (out_dir / out_name).open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {out_dir / out_name}")


if __name__ == "__main__":
    main()

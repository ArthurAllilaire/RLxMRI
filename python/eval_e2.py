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
import os
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

# juliapkg env vars must be set before importing juliacall, and we import
# juliacall first (below) — earlier than the qalibremd_gym.env module that
# normally sets these. Without this, juliacall instantiates a fresh
# .venv/julia_env/ that lacks PythonCall/MRISystemPhantom. setdefault so an
# explicit env (e.g. run_e2.sh sourcing .envrc.local) still wins.
_runtime_proj = str(Path(__file__).resolve().parent / "julia_runtime")
os.environ.setdefault("PYTHON_JULIAPKG_PROJECT", _runtime_proj)
os.environ.setdefault("PYTHON_JULIAPKG_OFFLINE", "yes")

# Import juliacall before torch (pulled in by stable_baselines3). On macOS, Julia
# and torch both install mach exception-port handlers at init; whichever loads
# first wins, and torch-first makes Julia's init crash with EXC_GUARD. Harmless
# on Linux/WSL (no mach ports) but required here. See juliacall warning + the
# import order in train_e2_mf.py / baseline_e2.py.
import juliacall  # noqa: F401  (must precede stable_baselines3/torch)

import numpy as np
from e2_train_common import load_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from qalibremd_gym.env_e2 import QalibreMDE2Env
from qalibremd_gym.schedules import log_ti_grid
from baseline_e2 import _mape_uncertainty  # episode-level bootstrap CI/SEM
from e2_config import parse_int_csv


# ── Fixed-TI grid baseline ─────────────────────────────────────────────────


def _fixed_grid_physical(step: int, n_ti: int = 7) -> tuple[float, float, float]:
    """Fixed log-spaced TI grid, α=90° — the conventional multi-TI IR protocol.
    Returns the *physical* (TI, TR, α); the env builds the correctly scaled
    normalised action via physical_to_norm_action (honours
    fix_te/learn_alpha/log_ti_action). Grid spans the env's TI bounds.

    TR=1.7 s (not the textbook ≥5·T1): at Npe=32 / 240 s a longer TR fits only
    ONE block, so the fitter never gets its ≥2 samples and MAPE is forced to
    1.0 — the spurious "100% baseline" seen before. 1.7 s lets ~5 blocks fit the
    budget the agent also faces. The authoritative comparator is baseline_e2.py
    (CR-optimal / clinical schedules); this is a quick in-pipeline sanity grid."""
    tis = log_ti_grid(lo=0.05, hi=3.0, n=n_ti)
    return float(tis[step % n_ti]), 1.7, 90.0


def evaluate_fixed_grid(env: QalibreMDE2Env, n_episodes: int,
                         seed_offset: int) -> dict:
    mapes, per_sphere = [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_offset + ep)
        done, step, info = False, 0, {}
        while not done:
            ti, tr, alpha = _fixed_grid_physical(step)
            norm = env.physical_to_norm_action(ti, tr, alpha)
            obs, _r, done, _trunc, info = env.step(norm)
            step += 1
        mapes.append(float(info.get("mape", np.nan)))
        per_sphere.append(np.abs(env.T1_est - env.T1_true) / env.T1_true)
    return {
        "mape_pct": float(np.nanmean(mapes)) * 100,
        "mape_p90_pct": float(np.nanpercentile(mapes, 90)) * 100,
        **_mape_uncertainty(mapes),
        "per_sphere_mape_pct": np.nanmean(np.array(per_sphere), axis=0) * 100,
    }


def evaluate_policy(policy_path: Path, vecnorm_path: Optional[Path],
                    n_episodes: int, seed_offset: int,
                    recurrent: bool = False,
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

    model = load_policy(policy_path, recurrent=recurrent)

    def _norm(o):
        if vec_norm is None:
            return o
        return vec_norm.normalize_obs(np.expand_dims(o, 0))[0]

    mapes, per_sphere, ti_choices, times = [], [], [], []
    repair_flags, tr_lift_flags, te_clamp_flags = [], [], []
    tr_lift_amounts, te_clamp_amounts = [], []
    # D2 readout: per-pool-index MAPE so we can see which T1 values fail.
    pool_apes: dict[int, list[float]] = {}
    # Flat (realised T1, abs-pct-error) pairs across all spheres and episodes,
    # for an error-vs-T1 breakdown (the per-pool-index view is uninformative
    # under the continuous-T1 sampler, where each slot is an i.i.d. draw).
    t1_true_all, ape_all = [], []

    for ep in range(n_episodes):
        obs, _ = raw_env.reset(seed=seed_offset + ep)
        done, info = False, {}
        ep_tis = []
        # LSTM state threading; plain PPO accepts and ignores both kwargs.
        state = None
        episode_start = np.ones((1,), dtype=bool)

        while not done:
            action, state = model.predict(_norm(obs), state=state,
                                          episode_start=episode_start,
                                          deterministic=True)
            episode_start[0] = False
            obs, _r, done, _trunc, info = raw_env.step(action)
            ep_tis.append(float(info.get("TI", np.nan)))
            repair_flags.append(bool(info.get("action_repaired", False)))
            tr_lift_flags.append(bool(info.get("TR_lifted", False)))
            te_clamp_flags.append(bool(info.get("TE_clamped", False)))
            tr_lift_amounts.append(float(info.get("TR_lift_amount", 0.0)))
            te_clamp_amounts.append(float(info.get("TE_clamp_amount", 0.0)))

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
        t1_true_all.extend(t1_true.tolist())
        ape_all.extend(ape.tolist())
        ti_choices.extend(ep_tis)
        times.append(float(info.get("time_s", raw_env.time_used_s)))

    per_sphere_arr = np.nanmean(np.array(per_sphere), axis=0)
    per_pool = {pi: (float(np.nanmean(v)) * 100, len(v))
                for pi, v in sorted(pool_apes.items())}

    return {
        "mape_pct":           float(np.nanmean(mapes)) * 100,
        "mape_p90_pct":       float(np.nanpercentile(mapes, 90)) * 100,
        **_mape_uncertainty(mapes),
        "success_5pct":       float(np.mean([m < 0.05 for m in mapes])),
        "per_sphere_mape_pct": per_sphere_arr * 100,
        "per_pool_mape_pct":  per_pool,
        "t1_true_all_s":      t1_true_all,
        "ape_all":            ape_all,
        "mean_scan_time_s":   float(np.mean(times)),
        "ti_choices_s":       ti_choices,
        "action_repair_rate": float(np.mean(repair_flags)) if repair_flags else 0.0,
        "tr_lift_rate":       float(np.mean(tr_lift_flags)) if tr_lift_flags else 0.0,
        "te_clamp_rate":      float(np.mean(te_clamp_flags)) if te_clamp_flags else 0.0,
        "mean_tr_lift_s":     float(np.mean(tr_lift_amounts)) if tr_lift_amounts else 0.0,
        "max_tr_lift_s":      float(np.max(tr_lift_amounts)) if tr_lift_amounts else 0.0,
        "mean_te_clamp_s":    float(np.mean(te_clamp_amounts)) if te_clamp_amounts else 0.0,
        "max_te_clamp_s":     float(np.max(te_clamp_amounts)) if te_clamp_amounts else 0.0,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--from-run", type=Path, default=None,
                   help="Load the env config from this run dir's run_config.json "
                        "(so eval matches training exactly — no need to repeat "
                        "--field/--nfe/--npe/--fix-te/… by hand). Also defaults "
                        "--policy/--vecnorm to <dir>/policy.zip,vecnorm.pkl.")
    p.add_argument("--policy",   type=Path, default=None,
                   help="Policy .zip. Optional if --from-run is given.")
    p.add_argument("--vecnorm",  type=Path, default=None)
    p.add_argument("--recurrent", action="store_true",
                   help="Load the policy as sb3-contrib RecurrentPPO. "
                        "Inferred automatically from run_config.json when "
                        "--from-run is given.")
    p.add_argument("--episodes", type=int,  default=50)
    p.add_argument("--seed",     type=int,  default=500_000)
    p.add_argument("--field",    type=str,  default="T3",
                   choices=["T3", "T15"])
    p.add_argument("--max-blocks", type=int, default=15)
    p.add_argument("--time-budget", type=float, default=120.0)
    p.add_argument("--subset-size", type=int, default=None,
                   help="Evaluate on random k-sphere subsets instead of the "
                        "full 14-sphere plate.")
    p.add_argument("--forced-sphere-indices", type=str, default=None,
                   help="Comma-separated 1-based T1-pool labels active every "
                        "episode, e.g. 1,3,6,8,14.")
    p.add_argument("--t1-sampler", type=str, default=None,
                   choices=["lognormal", "linear_uniform_range"],
                   help="Override run/default T1 material sampler.")
    p.add_argument("--pose-mode", type=str, default=None,
                   choices=["auto", "fixed", "inplane_jitter"],
                   help="Override run/default pose mode.")
    p.add_argument("--translation-sigma-mm", type=float, default=None)
    p.add_argument("--rotation-sigma-rad", type=float, default=None)
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
    p.add_argument("--roi-radius", type=int, default=0,
                   help="Square ROI half-width for per-sphere signal extraction. "
                        "0 = centre pixel; 1 = 3x3 mean. Can be overlaid on "
                        "--from-run for eval-only ablations.")
    p.add_argument("--water-model", type=str, default="bloch",
                   choices=["bloch", "cached_perline", "analytic"])
    p.add_argument("--noise-sigma-abs", type=float, default=50.0)
    p.add_argument("--reward-mode", type=str, default="neg_mape")
    p.add_argument("--terminal-bonus", type=float, default=0.5)
    p.add_argument("--mape-alpha", type=float, default=1.0)
    p.add_argument("--allow-stop", action="store_true")
    p.add_argument("--use-gpu", action="store_true")
    p.add_argument("--cpu", action="store_true",
                   help="Force CPU simulation, overriding the run's saved "
                        "use_gpu (e.g. for a CPU-only machine).")
    p.add_argument("--nfe", type=int, default=None)
    p.add_argument("--npe", type=int, default=None)
    p.add_argument("--summary-name", type=str, default=None,
                   help="Output JSON filename in the policy directory. Defaults "
                        "to eval_summary.json or eval_summary_oracle.json.")
    p.add_argument("--skip-fixed-baseline", action="store_true",
                   help="Only evaluate the agent policy. Useful for quick "
                        "eval-only ablations where the fixed-grid baseline is "
                        "unchanged or not needed.")
    args = p.parse_args()

    if args.from_run is not None:
        # Inherit the exact env config from the run, so eval can't drift from
        # training. Default the policy/vecnorm to the run dir too.
        from e2_config import load_run_env_kwargs, load_run_recurrent
        env_kwargs = load_run_env_kwargs(args.from_run)
        args.recurrent = args.recurrent or load_run_recurrent(args.from_run)
        if args.policy is None:
            args.policy = args.from_run / "policy.zip"
        if args.vecnorm is None and (args.from_run / "vecnorm.pkl").exists():
            args.vecnorm = args.from_run / "vecnorm.pkl"
        # Diagnostic eval-only knobs aren't in the training config; overlay them.
        env_kwargs.update(oracle_fit=args.oracle_fit, oracle_band=args.oracle_band,
                          fitter_n_grid=args.fitter_n_grid,
                          roi_radius=args.roi_radius)
        if args.subset_size is not None:
            env_kwargs["subset_size"] = args.subset_size
        if args.forced_sphere_indices is not None:
            env_kwargs["forced_sphere_indices"] = parse_int_csv(args.forced_sphere_indices)
        if args.t1_sampler is not None:
            env_kwargs["t1_sampler"] = args.t1_sampler
        if args.pose_mode is not None:
            env_kwargs["pose_mode"] = args.pose_mode
        if args.translation_sigma_mm is not None:
            env_kwargs["translation_sigma_mm"] = args.translation_sigma_mm
        if args.rotation_sigma_rad is not None:
            env_kwargs["rotation_sigma_rad"] = args.rotation_sigma_rad
        print(f"[eval_e2] env config loaded from {args.from_run}/run_config.json")
    else:
        env_kwargs = dict(cfg_field=args.field,
                           max_blocks=args.max_blocks,
                           time_budget_s=args.time_budget,
                           subset_size=args.subset_size,
                           forced_sphere_indices=parse_int_csv(args.forced_sphere_indices),
                           t1_sampler=args.t1_sampler or "lognormal",
                           pose_mode=args.pose_mode or "auto",
                           translation_sigma_mm=(
                               5.0 if args.translation_sigma_mm is None
                               else args.translation_sigma_mm),
                           rotation_sigma_rad=(
                               0.15 if args.rotation_sigma_rad is None
                               else args.rotation_sigma_rad),
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
                           roi_radius=args.roi_radius,
                           water_model=args.water_model,
                           noise_sigma_abs=args.noise_sigma_abs,
                           reward_mode=args.reward_mode,
                           terminal_bonus=args.terminal_bonus,
                           mape_alpha=args.mape_alpha,
                           allow_stop=args.allow_stop,
                           use_gpu=args.use_gpu)
    # Resolution overrides apply in either mode (only if explicitly passed).
    if args.nfe is not None:
        env_kwargs["Nfe"] = args.nfe
    if args.npe is not None:
        env_kwargs["Npe"] = args.npe
    if args.cpu:
        env_kwargs["use_gpu"] = False
    if args.policy is None:
        p.error("--policy is required unless --from-run is given.")

    print("=" * 60)
    print(f"E2 Evaluation — policy: {args.policy}")
    print("=" * 60)

    # ── PPO agent results ────────────────────────────────────────────────
    print(f"\nEvaluating PPO agent on {args.episodes} held-out configs …")
    res = evaluate_policy(args.policy, args.vecnorm, args.episodes,
                          args.seed, recurrent=args.recurrent, **env_kwargs)
    _ci = res["mape_ci95_pct"]
    print(f"  MAPE        = {res['mape_pct']:.2f}%  "
          f"(95% CI {_ci[0]:.2f}–{_ci[1]:.2f}%, SEM ±{res['mape_sem_pct']:.2f}%)")
    print(f"  p90 MAPE    = {res['mape_p90_pct']:.2f}%")
    print(f"  Success<5%  = {res['success_5pct']:.1%}")
    print(f"  Mean time   = {res['mean_scan_time_s']:.1f}s")
    print(f"  Action repair = {res['action_repair_rate']:.1%} "
          f"(TR lift {res['tr_lift_rate']:.1%}, "
          f"mean ΔTR {res['mean_tr_lift_s']:.3f}s, "
          f"max ΔTR {res['max_tr_lift_s']:.3f}s; "
          f"TE clamp {res['te_clamp_rate']:.1%})")

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
    # Run the baseline in the SAME action mode as the policy: physical_to_norm_action
    # decodes the fixed [TI, TE, TR, α] schedule correctly under any mode, so we no
    # longer strip fix_te/learn_alpha/log_ti_action (doing so used to silently
    # mis-route action channels — see the 2026-06 fix).
    base = None
    if not args.skip_fixed_baseline:
        print(f"\nEvaluating fixed-TI grid baseline on same configs …")
        env_base = QalibreMDE2Env(rng_seed=args.seed, **env_kwargs)
        base = evaluate_fixed_grid(env_base, args.episodes, args.seed)
        _bci = base["mape_ci95_pct"]
        print(f"  MAPE        = {base['mape_pct']:.2f}%  "
              f"(95% CI {_bci[0]:.2f}–{_bci[1]:.2f}%, SEM ±{base['mape_sem_pct']:.2f}%)")
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
                                  args.episodes // 2, args.seed,
                                  recurrent=args.recurrent, **kw)
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
        "agent_mape_ci95_pct": res["mape_ci95_pct"],
        "agent_mape_sem_pct": res["mape_sem_pct"],
        "baseline_mape_pct": None if base is None else base["mape_pct"],
        "baseline_mape_ci95_pct": None if base is None else base.get("mape_ci95_pct"),
        "per_sphere":    res["per_sphere_mape_pct"].tolist(),
        "per_pool":      {str(k): list(v)
                           for k, v in res.get("per_pool_mape_pct", {}).items()},
        # Flat (realised T1 [s], abs-pct-error [fraction]) pairs for the
        # error-vs-T1 breakdown. Build the low/mid/high figure from these.
        "per_episode_t1_err": {
            "t1_true_s": res.get("t1_true_all_s", []),
            "ape":       res.get("ape_all", []),
        },
    }
    out_name = (args.summary_name if args.summary_name is not None else
                ("eval_summary_oracle.json" if args.oracle_fit
                 else "eval_summary.json"))
    with (out_dir / out_name).open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {out_dir / out_name}")


if __name__ == "__main__":
    main()

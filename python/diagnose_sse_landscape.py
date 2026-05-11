"""Plot SSE_prof(T1) for failing spheres under V12.

§19.5 item 1: the decisive test for whether truth is in the top-K SSE
candidates. For each of N rollouts, record the per-sphere block schedule
(TIs, TRs, alpha_excs) and measured magnitudes, then sweep a dense T1
grid computing SSE(T1) with closed-form amplitude A. Plot the SSE
landscape with markers for T1_true, the global SSE minimum (= what the
baseline fitter returns), and T1_true's rank in the SSE ordering.

Three observable shapes per sphere:

  (1) T1_true is the global SSE minimum. Truth basin is correctly the
      MLE; baseline-fitter failure (if any) was grid coarseness or LM
      step issues. Re-evaluate with K-restart-by-SSE — should help.
  (2) T1_true is a *local* SSE minimum, but the global minimum is at a
      wrong T1. Truth is in top-K; non-SSE-selector fitters (Bayesian
      prior, joint multi-sphere, profile-likelihood width) could in
      principle recover truth.
  (3) T1_true is not even a local SSE minimum. Likelihood is dominated
      by abs() flip artefacts. All SSE-driven fitters dead; only
      likelihood-side fixes remain (joint fit, prior, smaller noise,
      signed recon).

Usage:
    PYTHONPATH=python PYTHON_JULIAPKG_OFFLINE=yes python python/diagnose_sse_landscape.py \\
        --policy   runs/e2/e2_tractability_V12/best/best_policy.zip \\
        --vecnorm  runs/e2/e2_tractability_V12/best/best_vecnorm.pkl \\
        --episodes 10 --seed 500000 \\
        --simplified-action --log-ti-action \\
        --max-blocks 30 --time-budget 250.0 --subset-size 5 \\
        --out runs/e2/e2_tractability_V12/sse_landscape
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from qalibremd_gym.env_e2 import QalibreMDE2Env


# Reimplementation of `transient_mz_at_excite_npe` from src/fitting/fits.jl.
# F1+ closed-form transient over Npe shots starting from M0 = 1.
def transient_mz_npe(T1, TI, TR, theta_inv, alpha_exc, Npe):
    if not np.isfinite(TR) or TR <= 0:
        E1 = np.exp(-TI / T1)
        return 1.0 - (1.0 - np.cos(theta_inv)) * E1
    E1 = np.exp(-TI / T1)
    E2 = np.exp(-(TR - TI) / T1)
    a = np.cos(theta_inv)
    b = np.cos(alpha_exc)
    Mz_pre = 1.0
    accum = 0.0
    for _ in range(int(Npe)):
        Mz_at_TI = (1 - E1) + a * E1 * Mz_pre
        accum += Mz_at_TI
        Mz_pre = (1 - E2) + b * E2 * Mz_at_TI
    return accum / Npe


def sse_prof(T1_grid, TIs, TRs, theta_invs, alpha_excs, mags, Npe):
    """Profile SSE(T1) with closed-form amplitude A. Magnitude path."""
    n = len(TIs)
    SSE = np.empty_like(T1_grid)
    A_arr = np.empty_like(T1_grid)
    for i, T1 in enumerate(T1_grid):
        y = np.array([
            transient_mz_npe(T1, TIs[k], TRs[k], theta_invs[k], alpha_excs[k], Npe)
            for k in range(n)
        ])
        ay = np.abs(y)
        den = float(np.sum(ay * ay))
        if den <= 0:
            SSE[i] = np.inf
            A_arr[i] = np.nan
            continue
        A = float(np.sum(mags * ay) / den)
        r = A * ay - mags
        SSE[i] = float(np.sum(r * r))
        A_arr[i] = A
    return SSE, A_arr


def collect_episodes(policy_path, vecnorm_path, n_episodes, seed_offset, env_kwargs):
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

    eps = []
    for ep in range(n_episodes):
        obs, info0 = raw_env.reset(seed=seed_offset + ep)
        done = False
        while not done:
            action, _ = model.predict(_norm(obs), deterministic=True)
            obs, _r, done, _trunc, info = raw_env.step(action)

        # After episode, pull per-sphere block schedule + mags from Julia env.
        jl_env = raw_env._env
        n_spheres = int(jl_env.n_spheres)
        spheres = []
        # PythonCall exposes Julia Vector{...} with 0-based __getitem__.
        for i in range(n_spheres):
            tis    = np.array([float(x) for x in jl_env.block_TIs[i]])
            trs    = np.array([float(x) for x in jl_env.block_TRs[i]])
            alphas = np.array([float(x) for x in getattr(jl_env, "block_α_excs")[i]])
            mags   = np.array([float(x) for x in jl_env.block_mags[i]])
            spheres.append({
                "TIs": tis, "TRs": trs,
                "alpha_excs": alphas,
                "theta_invs": np.full_like(tis, np.pi),  # 180° inversion
                "mags": mags,
            })
        eps.append({
            "ep": ep,
            "T1_true": np.array([float(t) for t in jl_env.T1_true]),
            "T1_est":  np.array([float(t) for t in jl_env.T1_est]),
            "sphere_indices": np.array(
                [int(i) for i in jl_env.sphere_indices], dtype=np.int64),
            "spheres": spheres,
            "Npe": int(jl_env.Npe),
        })
    return eps


def plot_sphere(ax, T1_grid, SSE, T1_true, T1_est, pool_idx, ape_pct):
    """One subplot of SSE vs T1 with markers."""
    ax.semilogx(T1_grid, SSE, color="C0", linewidth=1.0)
    ax.axvline(T1_true, color="green", linestyle="-", linewidth=1.5,
               label=f"T1_true = {T1_true*1e3:.1f} ms")
    ax.axvline(T1_est, color="red", linestyle="--", linewidth=1.5,
               label=f"T1_est = {T1_est*1e3:.1f} ms")

    # Mark the global SSE minimum
    i_min = int(np.argmin(SSE))
    T1_argmin = T1_grid[i_min]
    ax.scatter([T1_argmin], [SSE[i_min]], color="black", marker="x", s=60,
               label=f"argmin SSE = {T1_argmin*1e3:.1f} ms",
               zorder=5)

    # Find SSE at T1_true (interpolated), report rank percentile
    sse_at_truth = float(np.interp(T1_true, T1_grid, SSE))
    rank = int(np.sum(SSE < sse_at_truth))
    pct = 100.0 * rank / len(SSE)
    ax.set_title(f"pool {pool_idx}  (T1_true = {T1_true*1e3:.1f} ms,  "
                 f"APE = {ape_pct:.0f} %)\n"
                 f"truth at SSE rank {rank}/{len(SSE)}  ({pct:.1f}-th pct, "
                 f"lower is better)",
                 fontsize=9)
    ax.set_xlabel("candidate T1 (s)")
    ax.set_ylabel("SSE (closed-form A)")
    ax.legend(loc="best", fontsize=7)
    ax.grid(True, alpha=0.3)
    return {"T1_argmin": float(T1_argmin),
            "SSE_argmin": float(SSE[i_min]),
            "SSE_at_truth": sse_at_truth,
            "rank_of_truth": rank,
            "pct_of_truth": pct,
            "n_grid": len(SSE)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy",   type=Path, required=True)
    p.add_argument("--vecnorm",  type=Path, default=None)
    p.add_argument("--episodes", type=int,  default=10)
    p.add_argument("--seed",     type=int,  default=500_000)
    p.add_argument("--field",    type=str,  default="T3", choices=["T3", "T15"])
    p.add_argument("--max-blocks", type=int, default=30)
    p.add_argument("--time-budget", type=float, default=250.0)
    p.add_argument("--subset-size", type=int, default=5)
    p.add_argument("--simplified-action", action="store_true")
    p.add_argument("--log-ti-action",     action="store_true")
    p.add_argument("--phase-sensitive",   action="store_true")
    p.add_argument("--sigma-method",      type=str, default="bootstrap",
                   choices=["asymptotic", "profile_likelihood", "bootstrap"])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-grid", type=int, default=2000,
                   help="Dense T1 grid resolution for SSE(T1) plots.")
    p.add_argument("--T1-range", type=float, nargs=2, default=[0.005, 5.0])
    p.add_argument("--ape-threshold", type=float, default=200.0,
                   help="Only plot spheres with baseline APE > this %.")
    p.add_argument("--max-plots", type=int, default=12)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    env_kwargs = dict(cfg_field=args.field,
                      max_blocks=args.max_blocks,
                      time_budget_s=args.time_budget,
                      subset_size=args.subset_size,
                      phase_sensitive=args.phase_sensitive,
                      sigma_method=args.sigma_method,
                      simplified_action=args.simplified_action,
                      log_ti_action=args.log_ti_action)

    print(f"[diagnose_sse] Collecting {args.episodes} V12 episodes …")
    eps = collect_episodes(args.policy, args.vecnorm,
                           args.episodes, args.seed, env_kwargs)
    print(f"[diagnose_sse] Done. Per-sphere APE summary (pool idx, APE %):")
    for ep in eps:
        for slot in range(len(ep["sphere_indices"])):
            t_t = ep["T1_true"][slot]
            t_e = ep["T1_est"][slot]
            ape = abs(t_e - t_t) / t_t * 100.0
            pi  = int(ep["sphere_indices"][slot])
            print(f"  ep {ep['ep']:2d}  pool {pi:2d}  "
                  f"T1_true={t_t*1e3:7.1f} ms  T1_est={t_e*1e3:7.1f} ms  "
                  f"APE={ape:7.1f}%")

    # Build the candidate list: any sphere with APE > threshold.
    candidates = []
    for ep in eps:
        for slot in range(len(ep["sphere_indices"])):
            t_t = ep["T1_true"][slot]
            t_e = ep["T1_est"][slot]
            ape = abs(t_e - t_t) / t_t * 100.0
            if ape >= args.ape_threshold:
                candidates.append({
                    "ep": ep["ep"],
                    "slot": slot,
                    "pool_idx": int(ep["sphere_indices"][slot]),
                    "T1_true": float(t_t),
                    "T1_est":  float(t_e),
                    "ape_pct": float(ape),
                    "sphere": ep["spheres"][slot],
                    "Npe": ep["Npe"],
                })
    candidates.sort(key=lambda c: -c["ape_pct"])
    candidates = candidates[: args.max_plots]
    print(f"[diagnose_sse] {len(candidates)} candidates with APE >= "
          f"{args.ape_threshold:.0f} % (showing up to {args.max_plots})")

    if not candidates:
        print("[diagnose_sse] No failing spheres above threshold; nothing to plot.")
        return

    T1_grid = np.exp(np.linspace(np.log(args.T1_range[0]),
                                  np.log(args.T1_range[1]),
                                  args.n_grid))

    n_panels = len(candidates)
    ncols = 3
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(5.5 * ncols, 3.6 * nrows),
                              squeeze=False)
    summary = []
    for k, cand in enumerate(candidates):
        ax = axes[k // ncols, k % ncols]
        s = cand["sphere"]
        SSE, _A = sse_prof(T1_grid, s["TIs"], s["TRs"],
                            s["theta_invs"], s["alpha_excs"], s["mags"],
                            cand["Npe"])
        # Stash into a JSON-friendly summary
        info = plot_sphere(ax, T1_grid, SSE,
                            cand["T1_true"], cand["T1_est"],
                            cand["pool_idx"], cand["ape_pct"])
        info.update({"ep": cand["ep"], "slot": cand["slot"],
                     "pool_idx": cand["pool_idx"],
                     "T1_true": cand["T1_true"],
                     "T1_est":  cand["T1_est"],
                     "ape_pct": cand["ape_pct"],
                     "n_blocks": len(s["TIs"]),
                     "TIs": s["TIs"].tolist(),
                     "TRs": s["TRs"].tolist(),
                     "mags": s["mags"].tolist()})
        summary.append(info)

    # Hide unused axes
    for k in range(n_panels, nrows * ncols):
        axes[k // ncols, k % ncols].axis("off")

    fig.suptitle(
        "SSE landscape vs candidate T1, V12 best policy on failing spheres "
        f"(APE ≥ {args.ape_threshold:.0f} %).\n"
        "Green = T1_true, Red = T1_est (baseline fitter), Black × = global SSE min.",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_png = args.out / "sse_landscape.png"
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"[diagnose_sse] Saved {out_png}")

    out_json = args.out / "sse_landscape_summary.json"
    with out_json.open("w") as f:
        json.dump({"args": {k: (str(v) if isinstance(v, Path) else v)
                              for k, v in vars(args).items()},
                   "candidates": summary,
                   "T1_grid_range_s": list(args.T1_range),
                   "n_grid": args.n_grid}, f, indent=2)
    print(f"[diagnose_sse] Saved {out_json}")

    # Aggregate stat: where does truth rank in SSE ordering?
    ranks = np.array([c["pct_of_truth"] for c in summary])
    print(f"[diagnose_sse] Truth's SSE percentile across {len(ranks)} failing "
          f"spheres: min={ranks.min():.1f}, median={np.median(ranks):.1f}, "
          f"mean={ranks.mean():.1f}, max={ranks.max():.1f}")


if __name__ == "__main__":
    main()

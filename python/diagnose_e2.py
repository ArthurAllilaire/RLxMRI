"""Diagnose a trained E2 policy: is it actually adaptive, or has it collapsed?

Generates four plots from N rollouts of a trained policy:

  1. TI choice vs block index (one line per episode) — does TI vary within
     and across episodes?
  2. TI histogram (log-scale x) — single-mode collapse vs spread.
  3. Mean(T1_est) trajectory within episodes — is the running fit usable
     for the agent to condition on?
  4. TI vs current mean(T1_est) at decision time (scatter) — does the
     policy actually adapt its TI choice to its running estimate?

Usage:
    python python/diagnose_e2.py \
        --policy  runs/e2/ppo_200k/policy.zip \
        --vecnorm runs/e2/ppo_200k/vecnorm.pkl \
        --episodes 30 \
        --out     runs/e2/ppo_200k/diagnostics
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


def collect(policy_path: Path, vecnorm_path: Path | None,
            n_episodes: int, seed_offset: int, **env_kwargs):
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

    episodes = []  # list of dicts per episode
    for ep in range(n_episodes):
        obs, _ = raw_env.reset(seed=seed_offset + ep)
        ep_record = {
            "TI":            [],
            "TE":            [],
            "TR":            [],
            "alpha_deg":     [],
            "mape":          [],
            "T1_est_mean":   [],
            "T1_est_at_decision":      [],   # mean (kept for back-compat)
            "T1_est_min_at_decision":  [],   # min finite per-sphere T1_est
            "T1_est_max_at_decision":  [],   # max finite per-sphere T1_est
            "T1_est_unc_at_decision":  [],   # T1_est of most-uncertain sphere
            "T1_est_per_sphere_at_decision": [],   # full vector per block
            "T1_sigma_at_decision":    [],   # full sigma vector per block
            "T1_true":       np.asarray(raw_env.T1_true, dtype=np.float64),
            "sphere_indices": np.asarray([], dtype=np.int64),
        }
        done = False
        while not done:
            # Per-sphere running T1_est the policy *sees* at decision time
            t1_est_now = np.asarray(raw_env.T1_est, dtype=np.float64)
            try:
                t1_sig_now = np.asarray(raw_env._env.T1_sigma, dtype=np.float64)
            except Exception:
                t1_sig_now = np.full_like(t1_est_now, np.nan)
            finite = np.isfinite(t1_est_now) & (t1_est_now > 0)
            ep_record["T1_est_at_decision"].append(
                float(np.nanmean(t1_est_now)) if finite.any() else np.nan
            )
            ep_record["T1_est_min_at_decision"].append(
                float(np.min(t1_est_now[finite])) if finite.any() else np.nan
            )
            ep_record["T1_est_max_at_decision"].append(
                float(np.max(t1_est_now[finite])) if finite.any() else np.nan
            )
            # Most-uncertain sphere's T1_est (highest finite sigma)
            sig_finite = np.isfinite(t1_sig_now) & finite
            if sig_finite.any():
                idx_unc = int(np.argmax(np.where(sig_finite, t1_sig_now, -np.inf)))
                ep_record["T1_est_unc_at_decision"].append(float(t1_est_now[idx_unc]))
            else:
                ep_record["T1_est_unc_at_decision"].append(np.nan)
            ep_record["T1_est_per_sphere_at_decision"].append(t1_est_now.copy())
            ep_record["T1_sigma_at_decision"].append(t1_sig_now.copy())
            action, _ = model.predict(_norm(obs), deterministic=True)
            obs, _r, done, _trunc, info = raw_env.step(action)
            ep_record["TI"].append(float(info.get("TI", np.nan)))
            ep_record["TE"].append(float(info.get("TE", np.nan)))
            ep_record["TR"].append(float(info.get("TR", np.nan)))
            ep_record["alpha_deg"].append(float(info.get("alpha_deg", np.nan)))
            ep_record["mape"].append(float(info.get("mape", np.nan)))
            t1_est_post = np.asarray(info.get("T1_est",
                                              raw_env.T1_est), dtype=np.float64)
            ep_record["T1_est_mean"].append(float(np.nanmean(t1_est_post)))
        ep_record["sphere_indices"] = np.asarray(
            info.get("sphere_indices", []), dtype=np.int64)
        ep_record["T1_true"] = np.asarray(
            info.get("T1_true", raw_env.T1_true), dtype=np.float64)
        episodes.append(ep_record)

    return episodes


def plot_ti_per_episode(episodes, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for ep in episodes:
        ax.plot(np.arange(1, len(ep["TI"]) + 1), ep["TI"],
                marker="o", alpha=0.45, linewidth=1)
    ax.set_xlabel("Block index within episode")
    ax.set_ylabel("TI [s]")
    ax.set_yscale("log")
    ax.set_ylim(0.008, 4.0)
    ax.set_title("TI choice vs block index (one line per episode)\n"
                 "Adaptive policy → varied lines; collapsed → flat horizontal cluster")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_ti_histogram(episodes, out_path):
    all_ti = np.concatenate([np.asarray(ep["TI"]) for ep in episodes])
    all_ti = all_ti[np.isfinite(all_ti) & (all_ti > 0)]
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.logspace(np.log10(0.01), np.log10(3.0), 30)
    ax.hist(all_ti, bins=bins, edgecolor="black")
    ax.set_xscale("log")
    ax.set_xlabel("TI [s] (log scale)")
    ax.set_ylabel("Count across all blocks of all episodes")
    ax.set_title(f"TI histogram across {len(episodes)} episodes "
                 f"(N={len(all_ti)} blocks)\n"
                 "Single-bin spike → degenerate policy")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_t1est_trajectory(episodes, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for ep in episodes:
        ax.plot(np.arange(1, len(ep["T1_est_mean"]) + 1), ep["T1_est_mean"],
                marker="s", alpha=0.5, linewidth=1)
    ax.set_xlabel("Block index within episode")
    ax.set_ylabel("mean(T1_est) across 14 spheres [s]")
    ax.set_yscale("log")
    ax.set_title("Running T1 estimate the policy observes\n"
                 "Wild jumps → the obs is too noisy to condition on")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _pearson_log(xs, ys):
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    mask = np.isfinite(xs) & np.isfinite(ys) & (xs > 0) & (ys > 0)
    if mask.sum() < 2:
        return None, 0
    lx = np.log(xs[mask]); ly = np.log(ys[mask])
    return float(np.corrcoef(lx, ly)[0, 1]), int(mask.sum())


def plot_ti_vs_t1est(episodes, out_path):
    """Scatter TI vs running per-sphere T1_est summaries at decision time.

    The mean(T1_est) summary collapses heterogeneous subsets and was the
    diagnostic gap in §9.2 of EXPERT_REPORT_TRAC.md. We now also plot:
      - TI vs min(T1_est)  — does the policy reach for short TIs when a
        short-T1 sphere is present?
      - TI vs T1_est of the most-uncertain sphere — does the policy target
        the sphere it knows least about?
    """
    summaries = {}
    for key, label in [
        ("T1_est_at_decision",     "mean(T1_est) [s]"),
        ("T1_est_min_at_decision", "min(T1_est) [s]"),
        ("T1_est_unc_at_decision", "T1_est of most-uncertain sphere [s]"),
    ]:
        xs, ys = [], []
        for ep in episodes:
            for i, ti in enumerate(ep["TI"]):
                xs.append(ep[key][i])
                ys.append(ti)
        r, n = _pearson_log(xs, ys)
        summaries[key] = {"pearson_log_r": r, "n": n, "label": label,
                          "xs": xs, "ys": ys}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, (key, info) in zip(axes, summaries.items()):
        ax.scatter(info["xs"], info["ys"], alpha=0.45, s=18)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(info["label"])
        r = info["pearson_log_r"]
        ax.set_title(f"r = {r:+.3f}" if r is not None else "insufficient data")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("TI chosen this block [s]")
    fig.suptitle("TI vs running T1_est at decision time "
                 "(log–log Pearson r per panel)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return {k: {"pearson_log_r": v["pearson_log_r"], "n": v["n"]}
            for k, v in summaries.items()}


def write_summary(episodes, out_path):
    all_ti = np.concatenate([np.asarray(ep["TI"]) for ep in episodes])
    all_ti = all_ti[np.isfinite(all_ti) & (all_ti > 0)]
    final_mapes = np.asarray([ep["mape"][-1] if ep["mape"] else np.nan
                               for ep in episodes])
    ep_lens = np.asarray([len(ep["TI"]) for ep in episodes])

    # Within-episode TI std vs across-episode TI std — adaptivity proxy
    intra_std = float(np.nanmean([np.nanstd(np.log(ep["TI"]))
                                    for ep in episodes if len(ep["TI"]) > 1]))
    inter_std = float(np.nanstd(np.log(all_ti)))

    # Collapse score: fraction of blocks within ±20% of the modal TI
    log_ti = np.log10(all_ti)
    hist, edges = np.histogram(log_ti, bins=30)
    mode_lo, mode_hi = edges[np.argmax(hist)], edges[np.argmax(hist) + 1]
    in_mode = ((log_ti >= mode_lo) & (log_ti <= mode_hi)).mean()

    summary = {
        "n_episodes":              len(episodes),
        "total_blocks":            int(len(all_ti)),
        "ep_len_mean":             float(np.mean(ep_lens)),
        "ep_len_std":              float(np.std(ep_lens)),
        "final_mape_mean_pct":     float(np.nanmean(final_mapes)) * 100,
        "final_mape_std_pct":      float(np.nanstd(final_mapes)) * 100,
        "ti_log10_intra_episode_std": intra_std / np.log(10),
        "ti_log10_inter_episode_std": inter_std / np.log(10),
        "ti_modal_bin_share":      float(in_mode),
        "ti_modal_bin_range_s":    [float(10**mode_lo), float(10**mode_hi)],
        # Raw TI lists (per episode) so plots_for_report.py can use them
        "all_ti_s":                all_ti.tolist(),
        "ep_tis":                  [ep["TI"] for ep in episodes],
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def _subset_bucket(ep):
    t1 = np.asarray(ep.get("T1_true", []), dtype=np.float64)
    if t1.size == 0 or not np.all(np.isfinite(t1)):
        return "unknown"
    mn, mx = float(np.min(t1)), float(np.max(t1))
    if mx >= 0.5 and mn >= 0.1:
        return "all_long"
    if mx < 0.2 and mn < 0.05:
        return "all_short"
    return "mixed"


def _ks_2sample(x, y):
    """Two-sample Kolmogorov-Smirnov statistic with asymptotic p-value.

    Avoids a scipy dependency for this lightweight diagnostic.
    """
    x = np.sort(np.asarray(x, dtype=np.float64))
    y = np.sort(np.asarray(y, dtype=np.float64))
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    n, m = x.size, y.size
    if n == 0 or m == 0:
        return {"D": None, "p": None, "n_x": int(n), "n_y": int(m)}
    data = np.concatenate([x, y])
    cdf_x = np.searchsorted(x, data, side="right") / n
    cdf_y = np.searchsorted(y, data, side="right") / m
    d = float(np.max(np.abs(cdf_x - cdf_y)))
    en = np.sqrt(n * m / (n + m))
    z = (en + 0.12 + 0.11 / max(en, 1e-12)) * d
    p = 2.0 * sum(((-1) ** (k - 1)) * np.exp(-2.0 * k * k * z * z)
                  for k in range(1, 101))
    return {"D": d, "p": float(np.clip(p, 0.0, 1.0)), "n_x": int(n), "n_y": int(m)}


def plot_subset_bucket_histograms(episodes, out_path):
    buckets = {"all_long": [], "all_short": [], "mixed": []}
    for ep in episodes:
        b = _subset_bucket(ep)
        if b in buckets:
            buckets[b].extend(ep["TI"])

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.logspace(np.log10(0.01), np.log10(3.0), 30)
    for name, vals in buckets.items():
        vals = np.asarray(vals, dtype=np.float64)
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if vals.size:
            ax.hist(vals, bins=bins, histtype="step", linewidth=2,
                    label=f"{name} (N={vals.size})")
    ax.set_xscale("log")
    ax.set_xlabel("TI [s] (log scale)")
    ax.set_ylabel("Count across blocks")
    ax.set_title("TI distributions by episode T1-subset bucket")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)

    return {
        "bucket_counts_episodes": {
            name: int(sum(_subset_bucket(ep) == name for ep in episodes))
            for name in buckets
        },
        "bucket_counts_blocks": {name: int(len(vals)) for name, vals in buckets.items()},
        "ks_all_long_vs_all_short": _ks_2sample(buckets["all_long"], buckets["all_short"]),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy",   type=Path, required=True)
    p.add_argument("--vecnorm",  type=Path, default=None)
    p.add_argument("--episodes", type=int,  default=30)
    p.add_argument("--seed",     type=int,  default=500_000)
    p.add_argument("--field",    type=str,  default="T3",
                   choices=["T3", "T15"])
    p.add_argument("--out",      type=Path, default=None,
                   help="Output dir; defaults to <policy parent>/diagnostics")
    p.add_argument("--simplified-action", action="store_true",
                   help="Required if the policy was trained with "
                        "--simplified-action (3-dim action space)")
    p.add_argument("--log-ti-action", action="store_true",
                   help="Required if the policy was trained with "
                        "--log-ti-action.")
    p.add_argument("--max-blocks", type=int, default=15)
    p.add_argument("--time-budget", type=float, default=120.0)
    p.add_argument("--subset-size", type=int, default=None,
                   help="Diagnose policy on random k-sphere subsets.")
    p.add_argument("--phase-sensitive", action="store_true")
    p.add_argument("--sigma-method", type=str, default="bootstrap",
                   choices=["asymptotic", "profile_likelihood", "bootstrap"])
    args = p.parse_args()

    out_dir = args.out or (args.policy.parent / "diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[diagnose] Collecting {args.episodes} rollouts from {args.policy} …")
    eps = collect(args.policy, args.vecnorm, args.episodes, args.seed,
                   cfg_field=args.field,
                   max_blocks=args.max_blocks,
                   time_budget_s=args.time_budget,
                   subset_size=args.subset_size,
                   phase_sensitive=args.phase_sensitive,
                   sigma_method=args.sigma_method,
                   simplified_action=args.simplified_action,
                   log_ti_action=args.log_ti_action)

    print("[diagnose] Plotting …")
    plot_ti_per_episode  (eps, out_dir / "ti_per_episode.png")
    plot_ti_histogram    (eps, out_dir / "ti_histogram.png")
    plot_t1est_trajectory(eps, out_dir / "t1est_trajectory.png")
    ti_vs_t1est_summary = plot_ti_vs_t1est(eps, out_dir / "ti_vs_t1est.png")
    summary = write_summary(eps, out_dir / "diagnose_summary.json")
    summary["ti_vs_t1est_correlations"] = ti_vs_t1est_summary
    bucket_summary = None
    if args.subset_size:
        bucket_summary = plot_subset_bucket_histograms(
            eps, out_dir / "ti_histogram_by_subset_bucket.png")
        summary["subset_bucket_diagnostic"] = bucket_summary
        with (out_dir / "diagnose_summary.json").open("w") as f:
            json.dump(summary, f, indent=2)

    print(f"[diagnose] Done. Plots and summary in {out_dir}")
    print(f"  ep_len_mean             = {summary['ep_len_mean']:.2f}")
    print(f"  final MAPE              = {summary['final_mape_mean_pct']:.2f}%")
    print(f"  TI intra-episode log-σ  = {summary['ti_log10_intra_episode_std']:.3f}")
    print(f"  TI inter-episode log-σ  = {summary['ti_log10_inter_episode_std']:.3f}")
    print(f"  Modal-bin share         = {summary['ti_modal_bin_share']:.1%}  "
          f"(range {summary['ti_modal_bin_range_s'][0]:.3f}-"
          f"{summary['ti_modal_bin_range_s'][1]:.3f}s)")
    for key, info in ti_vs_t1est_summary.items():
        r = info["pearson_log_r"]
        rstr = f"{r:+.3f}" if r is not None else "n/a"
        print(f"  log–log Pearson r ({key}) = {rstr}  (N={info['n']})")
    if bucket_summary is not None:
        ks = bucket_summary["ks_all_long_vs_all_short"]
        print(f"  Subset buckets          = {bucket_summary['bucket_counts_episodes']}")
        if ks["p"] is not None:
            print(f"  KS all_long vs all_short = D={ks['D']:.3f}, p={ks['p']:.3g} "
                  f"(N={ks['n_x']}/{ks['n_y']})")
        else:
            print("  KS all_long vs all_short = insufficient bucket samples")
    print()
    if summary["ti_log10_intra_episode_std"] < 0.05:
        print("  → DEGENERATE: policy picks essentially the same TI every block.")
    elif summary["ti_modal_bin_share"] > 0.50:
        print("  → STRONG MODE: policy mostly picks one TI but occasionally varies.")
    else:
        print("  → Spread across TI bins; could be genuinely adaptive — "
              "check ti_vs_t1est.png correlation.")


if __name__ == "__main__":
    main()

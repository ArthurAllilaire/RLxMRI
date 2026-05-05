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
            "T1_est_at_decision": [],   # mean estimate the policy could see
            "T1_true":       np.asarray(raw_env.T1_true, dtype=np.float64),
        }
        done = False
        while not done:
            # mean of running T1_est that the policy *sees* at decision time
            t1_est_now = np.asarray(raw_env.T1_est, dtype=np.float64)
            ep_record["T1_est_at_decision"].append(
                float(np.nanmean(t1_est_now)) if np.any(np.isfinite(t1_est_now)) else np.nan
            )
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


def plot_ti_vs_t1est(episodes, out_path):
    """If the policy is genuinely adaptive, TI choice should covary with the
    running T1_est at decision time. A flat scatter rules out adaptivity."""
    xs, ys = [], []
    for ep in episodes:
        n = len(ep["TI"])
        for i in range(n):
            t1_at_decision = ep["T1_est_at_decision"][i]
            if not np.isfinite(t1_at_decision) or t1_at_decision <= 0:
                continue
            xs.append(t1_at_decision)
            ys.append(ep["TI"][i])
    xs = np.asarray(xs); ys = np.asarray(ys)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(xs, ys, alpha=0.5, s=20)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Running mean(T1_est) at decision time [s]")
    ax.set_ylabel("TI chosen this block [s]")
    if len(xs) > 1:
        # Pearson correlation in log space — proxy for adaptivity
        lx = np.log(xs); ly = np.log(ys)
        lx = lx[np.isfinite(lx) & np.isfinite(ly)]
        ly = ly[np.isfinite(lx) & np.isfinite(ly)]
        if len(lx) > 1:
            r = float(np.corrcoef(lx, ly)[0, 1])
            ax.set_title(f"TI vs T1_est at decision (log–log Pearson r = {r:+.3f})\n"
                         "|r| ≈ 0 → policy is non-adaptive")
        else:
            ax.set_title("TI vs T1_est at decision")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


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
    args = p.parse_args()

    out_dir = args.out or (args.policy.parent / "diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[diagnose] Collecting {args.episodes} rollouts from {args.policy} …")
    eps = collect(args.policy, args.vecnorm, args.episodes, args.seed,
                   cfg_field=args.field,
                   simplified_action=args.simplified_action)

    print("[diagnose] Plotting …")
    plot_ti_per_episode  (eps, out_dir / "ti_per_episode.png")
    plot_ti_histogram    (eps, out_dir / "ti_histogram.png")
    plot_t1est_trajectory(eps, out_dir / "t1est_trajectory.png")
    plot_ti_vs_t1est     (eps, out_dir / "ti_vs_t1est.png")
    summary = write_summary(eps, out_dir / "diagnose_summary.json")

    print(f"[diagnose] Done. Plots and summary in {out_dir}")
    print(f"  ep_len_mean             = {summary['ep_len_mean']:.2f}")
    print(f"  final MAPE              = {summary['final_mape_mean_pct']:.2f}%")
    print(f"  TI intra-episode log-σ  = {summary['ti_log10_intra_episode_std']:.3f}")
    print(f"  TI inter-episode log-σ  = {summary['ti_log10_inter_episode_std']:.3f}")
    print(f"  Modal-bin share         = {summary['ti_modal_bin_share']:.1%}  "
          f"(range {summary['ti_modal_bin_range_s'][0]:.3f}-"
          f"{summary['ti_modal_bin_range_s'][1]:.3f}s)")
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

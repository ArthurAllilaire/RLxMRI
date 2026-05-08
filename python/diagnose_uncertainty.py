"""Diagnose per-sphere fit uncertainty (σ_T1) over the course of an episode.

Reads `env.T1_sigma` (asymptotic σ from the IR fit's covariance) at every
block of N rollouts and produces:

  1. sigma_trajectory.png  — log10(σ_T1/T1_est) per sphere vs block index,
                              one panel per episode (or aggregated across eps).
  2. sigma_final.png        — final-block σ_T1/T1_est per sphere, mean across
                              episodes with ±1 SD bars. Pairs naturally with
                              per_sphere_mape.png.
  3. sigma_summary.json     — raw per-block, per-sphere arrays.

Usage:
    python python/diagnose_uncertainty.py \
        --policy  runs/e2/20260506_111510/policy.zip \
        --vecnorm runs/e2/20260506_111510/vecnorm.pkl \
        --episodes 30
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
    raw = QalibreMDE2Env(rng_seed=seed_offset, **env_kwargs)
    vec_norm = None
    if vecnorm_path is not None and vecnorm_path.exists():
        venv_tmp = DummyVecEnv([lambda: QalibreMDE2Env(rng_seed=seed_offset, **env_kwargs)])
        vec_norm = VecNormalize.load(str(vecnorm_path), venv_tmp)
        vec_norm.training = False
        vec_norm.norm_reward = False

    model = PPO.load(str(policy_path))

    def _norm(o):
        return o if vec_norm is None else vec_norm.normalize_obs(np.expand_dims(o, 0))[0]

    episodes = []
    for ep in range(n_episodes):
        obs, _ = raw.reset(seed=seed_offset + ep)
        rec = {
            "sigma":    [], "T1_est": [], "T1_true": np.asarray(raw.T1_true),
            "TI": [], "TE": [], "TR": [], "alpha_deg": [],
            "block_time": [], "mape": [],
        }
        done = False
        while not done:
            action, _ = model.predict(_norm(obs), deterministic=True)
            obs, _r, done, _trunc, info = raw.step(action)
            rec["sigma"].append(np.asarray(raw.T1_sigma))
            rec["T1_est"].append(np.asarray(raw.T1_est))
            rec["TI"].append(float(info.get("TI", np.nan)))
            rec["TE"].append(float(info.get("TE", np.nan)))
            rec["TR"].append(float(info.get("TR", np.nan)))
            rec["alpha_deg"].append(float(info.get("alpha_deg", np.nan)))
            rec["block_time"].append(float(info.get("block_time", np.nan)))
            rec["mape"].append(float(info.get("mape", np.nan)))
        rec["sigma"]  = np.asarray(rec["sigma"])     # (n_blocks, n_spheres)
        rec["T1_est"] = np.asarray(rec["T1_est"])
        for k in ("TI", "TE", "TR", "alpha_deg", "block_time", "mape"):
            rec[k] = np.asarray(rec[k])
        episodes.append(rec)
    return episodes


def plot_action_trajectories(episodes, out_path):
    """4-panel: TI, TR, TE, α_exc vs block index, one line per episode."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True)
    keys   = ["TI", "TR", "TE", "alpha_deg"]
    titles = ["TI [s] (log)", "TR [s] (log)", "TE [s]", "α_exc [deg]"]
    log    = [True, True, False, False]
    for ax, k, t, lg in zip(axes.flat, keys, titles, log):
        for ep in episodes:
            ax.plot(np.arange(1, len(ep[k]) + 1), ep[k],
                    marker="o", alpha=0.40, linewidth=1, markersize=3)
        if lg:
            ax.set_yscale("log")
        ax.set_xlabel("Block index")
        ax.set_ylabel(t)
        ax.set_title(t)
        ax.grid(True, alpha=0.3, which="both")
    fig.suptitle("Action trajectories across episodes "
                 "(one line per evaluation episode)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_action_histograms(episodes, out_path):
    """4-panel histograms of TI, TR, TE, α across all blocks."""
    all_TI = np.concatenate([ep["TI"]        for ep in episodes])
    all_TR = np.concatenate([ep["TR"]        for ep in episodes])
    all_TE = np.concatenate([ep["TE"]        for ep in episodes])
    all_a  = np.concatenate([ep["alpha_deg"] for ep in episodes])
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))
    log_bins = np.logspace(np.log10(0.01), np.log10(3.0), 30)
    axes[0, 0].hist(all_TI[np.isfinite(all_TI) & (all_TI > 0)],
                    bins=log_bins, color="#1f77b4", edgecolor="black")
    axes[0, 0].set_xscale("log"); axes[0, 0].set_title("TI [s]")
    log_bins_TR = np.logspace(np.log10(0.5), np.log10(5.5), 30)
    axes[0, 1].hist(all_TR[np.isfinite(all_TR) & (all_TR > 0)],
                    bins=log_bins_TR, color="#2ca02c", edgecolor="black")
    axes[0, 1].set_xscale("log"); axes[0, 1].set_title("TR [s]")
    axes[1, 0].hist(all_TE[np.isfinite(all_TE)], bins=30,
                    color="#ff7f0e", edgecolor="black")
    axes[1, 0].set_title("TE [s]")
    axes[1, 1].hist(all_a[np.isfinite(all_a)], bins=30,
                    color="#d62728", edgecolor="black")
    axes[1, 1].set_title("α_exc [deg]")
    for ax in axes.flat:
        ax.set_ylabel("Block count")
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Action histograms across {len(episodes)} evaluation episodes "
                 f"(N_blocks = {len(all_TI)})", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_ti_vs_sigma(episodes, out_path):
    """For each block, scatter TI chosen vs the *prior* mean sigma the policy
    sees at decision time. Tests whether the agent uses the σ-channel."""
    xs, ys = [], []  # x = mean(σ/T1_est) before this block, y = TI chosen
    for ep in episodes:
        nb = len(ep["TI"])
        for b in range(nb):
            if b == 0:
                # prior σ before any block = NaN; skip
                continue
            with np.errstate(divide="ignore", invalid="ignore"):
                rel = ep["sigma"][b - 1] / np.maximum(ep["T1_est"][b - 1], 1e-9)
            m = np.nanmean(rel)
            if not np.isfinite(m) or m <= 0:
                continue
            xs.append(m); ys.append(ep["TI"][b])
    xs, ys = np.asarray(xs), np.asarray(ys)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    if len(xs):
        ax.scatter(xs, ys, alpha=0.5, s=22)
        lx, ly = np.log(xs), np.log(ys)
        m = np.isfinite(lx) & np.isfinite(ly)
        if m.sum() > 1:
            r = float(np.corrcoef(lx[m], ly[m])[0, 1])
            ax.set_title(f"TI chosen vs prior mean(σ/T1_est)  "
                         f"(log–log Pearson r = {r:+.3f})\n"
                         "|r|>0 → agent IS using uncertainty channel")
        else:
            ax.set_title("TI vs prior mean(σ/T1_est)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Mean(σ_T1 / T1_est) at decision time")
    ax.set_ylabel("TI chosen this block [s]")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_sigma_trajectory(episodes, out_path):
    """log10(σ/T1_est) per sphere vs block index — one line per sphere,
    averaged across episodes (with ±1 SD shading)."""
    max_blocks = max(len(ep["sigma"]) for ep in episodes)
    n_spheres  = episodes[0]["T1_true"].size

    # Stack to (n_episodes, max_blocks, n_spheres) with NaN padding
    pad = np.full((len(episodes), max_blocks, n_spheres), np.nan)
    for i, ep in enumerate(episodes):
        nb = len(ep["sigma"])
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = ep["sigma"] / np.where(ep["T1_est"] > 0, ep["T1_est"], np.nan)
        pad[i, :nb] = np.log10(np.clip(rel, 1e-3, 1e2))

    mean = np.nanmean(pad, axis=0)   # (max_blocks, n_spheres)
    std  = np.nanstd (pad, axis=0)

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    cmap = plt.get_cmap("viridis")
    blocks = np.arange(1, max_blocks + 1)
    for i in range(n_spheres):
        c = cmap(i / max(n_spheres - 1, 1))
        ax.plot(blocks, mean[:, i], "-o", color=c, linewidth=1.2,
                markersize=3.5, label=f"S{i+1}" if i in (0, 6, 13) else None)
        ax.fill_between(blocks, mean[:, i] - std[:, i], mean[:, i] + std[:, i],
                         color=c, alpha=0.10)

    ax.set_xlabel("Block index within episode")
    ax.set_ylabel("log10(σ_T1 / T1_est)")
    ax.set_title("Per-sphere fit uncertainty trajectory\n"
                 "(viridis: dark = sphere 1 longest T1, light = sphere 14 shortest)")
    ax.axhline(0, color="black", linestyle=":", linewidth=0.8,
               label="100% relative σ")
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.92)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_sigma_final(episodes, out_path):
    """Final-block σ/T1_est per sphere, mean ± 1 SD across episodes."""
    n_spheres = episodes[0]["T1_true"].size
    finals = np.full((len(episodes), n_spheres), np.nan)
    for i, ep in enumerate(episodes):
        if len(ep["sigma"]) == 0:
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = ep["sigma"][-1] / np.where(ep["T1_est"][-1] > 0, ep["T1_est"][-1], np.nan)
        finals[i] = np.clip(rel, 1e-3, 1e2)

    mean = np.nanmean(finals, axis=0) * 100
    std  = np.nanstd (finals, axis=0) * 100
    idx  = np.arange(1, n_spheres + 1)

    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.bar(idx, mean, yerr=std, capsize=4, color="#2ca02c",
           edgecolor="black", linewidth=0.5)
    ax.set_xticks(idx)
    ax.set_xlabel("Sphere index (1 = longest T1, 14 = shortest)")
    ax.set_ylabel("Final-block σ_T1 / T1_est  [%]")
    ax.set_title("Per-sphere fit uncertainty at episode end (mean ±1 SD across episodes)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def write_summary(episodes, out_path):
    out = {
        "n_episodes":  len(episodes),
        "per_episode": [],
    }
    for ep in episodes:
        out["per_episode"].append({
            "T1_true":   ep["T1_true"].tolist(),
            "T1_est":    ep["T1_est"].tolist(),
            "sigma":     ep["sigma"].tolist(),
            "TI":        ep["TI"].tolist(),
            "TE":        ep["TE"].tolist(),
            "TR":        ep["TR"].tolist(),
            "alpha_deg": ep["alpha_deg"].tolist(),
            "block_time": ep["block_time"].tolist(),
            "mape":      ep["mape"].tolist(),
        })
    with open(out_path, "w") as f:
        json.dump(out, f)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy",   type=Path, required=True)
    p.add_argument("--vecnorm",  type=Path, default=None)
    p.add_argument("--episodes", type=int,  default=30)
    p.add_argument("--seed",     type=int,  default=500_000)
    p.add_argument("--field",    type=str,  default="T3")
    p.add_argument("--out",      type=Path, default=None)
    p.add_argument("--simplified-action", action="store_true")
    args = p.parse_args()

    out_dir = args.out or (args.policy.parent / "diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[uncertainty] Collecting {args.episodes} rollouts from {args.policy} …")
    eps = collect(args.policy, args.vecnorm, args.episodes, args.seed,
                   cfg_field=args.field,
                   simplified_action=args.simplified_action)

    plot_sigma_trajectory  (eps, out_dir / "sigma_trajectory.png")
    plot_sigma_final       (eps, out_dir / "sigma_final.png")
    plot_action_trajectories(eps, out_dir / "action_trajectories.png")
    plot_action_histograms (eps, out_dir / "action_histograms.png")
    plot_ti_vs_sigma       (eps, out_dir / "ti_vs_sigma.png")
    write_summary          (eps, out_dir / "sigma_summary.json")

    # quick console snapshot
    final_rel = np.array([
        ep["sigma"][-1] / np.maximum(ep["T1_est"][-1], 1e-9)
        for ep in eps
    ])
    print(f"[uncertainty] Final relative σ across all sphere×episode cells:")
    print(f"  median = {np.nanmedian(final_rel)*100:.1f}%")
    print(f"  mean   = {np.nanmean(final_rel)*100:.1f}%")
    print(f"  p90    = {np.nanpercentile(final_rel, 90)*100:.1f}%")
    print(f"  >100% σ fraction = {np.nanmean(final_rel >= 1.0)*100:.1f}%")
    print(f"[uncertainty] Plots and summary in {out_dir}")


if __name__ == "__main__":
    main()

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
from qalibremd_gym import env as _env_mod


def collect(policy_path: Path, vecnorm_path: Path | None,
            n_episodes: int, seed_offset: int,
            forced_indices_list=None,
            snr_holder: dict | None = None, **env_kwargs):
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
    qmd = _env_mod._JL_QMD
    for ep in range(n_episodes):
        forced = forced_indices_list[ep] if forced_indices_list is not None else None
        obs, _ = raw_env.reset(seed=seed_offset + ep, forced_sphere_indices=forced)

        # Once per `collect()` run, on the first episode, capture the
        # NEMA MS-1 dual-acquisition SNR — the trustworthy noise-level
        # number we cite in the report. Costs 2 extra simulate() calls;
        # acceptable as a one-time measurement since σ is fixed.
        if ep == 0 and snr_holder is not None and not snr_holder:
            try:
                rep = qmd.e2_dual_acq_snr_report(raw_env._env)
                snr_holder.update({
                "ksp_rms":               float(rep.ksp_rms),
                "sigma_used":            float(rep.sigma_used),
                "snr_ksp":               float(rep.snr_ksp),
                "background_std_a":      float(rep.image.background_std_a),
                "background_std_b":      float(rep.image.background_std_b),
                "diff_roi_std":          float(rep.image.diff_roi_std),
                "sphere_mean_a":         [float(v) for v in rep.image.sphere_mean_a],
                "sphere_mean_b":         [float(v) for v in rep.image.sphere_mean_b],
                "sphere_means":          [float(v) for v in rep.image.sphere_means],
                "temporal_instability":  [float(v) for v in rep.image.temporal_instability],
                "snr_nema_per_sphere_a": [float(v) for v in rep.image.snr_nema_per_sphere_a],
                "snr_nema_per_sphere_b": [float(v) for v in rep.image.snr_nema_per_sphere_b],
                "snr_nema_peak_a":       float(rep.image.snr_nema_peak_a),
                "snr_nema_peak_b":       float(rep.image.snr_nema_peak_b),
                "snr_dual_per_sphere":   [float(v) for v in rep.image.snr_dual_per_sphere],
                "snr_dual_peak":         float(rep.image.snr_dual_peak),
                })
                print(f"[diagnose] SNR report (NEMA MS-1 dual-acq):")
                print(f"           σ = {snr_holder['sigma_used']:.4g}   "
                      f"snr_ksp = {snr_holder['snr_ksp']:.2f}   "
                      f"snr_nema_peak_a = {snr_holder['snr_nema_peak_a']:.2f}   "
                      f"snr_dual_peak = {snr_holder['snr_dual_peak']:.2f}  ← report figure")
            except Exception as _snr_err:
                print(f"[diagnose] SNR report unavailable ({_snr_err.__class__.__name__}): {_snr_err}")
                snr_holder.clear()
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
            "snr_nema_peak":  [],   # per-block single-image NEMA SNR
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
            # Cheap per-step NEMA single-image SNR on the reconstructed
            # image this block just produced. Aggregated into snr_vs_block.png.
            try:
                stats = qmd.e2_image_stats(raw_env._env)
                ep_record["snr_nema_peak"].append(float(stats.snr_peak))
            except Exception:
                ep_record["snr_nema_peak"].append(np.nan)
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


def plot_snr_vs_block(episodes, out_path, dual_acq_snr=None):
    """Per-block NEMA single-image SNR (one line per episode). If the dual-acq
    reference SNR is supplied, draws it as a horizontal line — that's the
    figure-of-merit we cite in the report (single-image NEMA is biased high
    on coarse grids by Gibbs ringing in the "background" region)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for ep in episodes:
        snrs = np.asarray(ep.get("snr_nema_peak", []), dtype=np.float64)
        if snrs.size == 0:
            continue
        ax.plot(np.arange(1, snrs.size + 1), snrs,
                marker="o", alpha=0.4, linewidth=1)
    if dual_acq_snr is not None and np.isfinite(dual_acq_snr):
        ax.axhline(dual_acq_snr, color="crimson", linestyle="--", linewidth=1.5,
                   label=f"dual-acq SNR (NEMA MS-1) = {dual_acq_snr:.1f}")
        ax.legend(loc="best")
    ax.set_xlabel("Block index within episode")
    ax.set_ylabel("NEMA single-image SNR (peak sphere)")
    ax.set_yscale("log")
    ax.set_title("Per-block image SNR\n"
                 "(single-image NEMA is biased by structured background;\n"
                 "the dashed dual-acq line is the trustworthy noise reference)")
    ax.grid(True, which="both", alpha=0.3)
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


def save_episodes(episodes, out_path):
    """Serialise full episode data so plots can be regenerated without Julia."""
    def _to_json(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, (np.floating, np.integer)):
            return v.item()
        return v

    serialised = [
        {k: _to_json(v) for k, v in ep.items()
         if k not in ("T1_est_per_sphere_at_decision", "T1_sigma_at_decision")}
        for ep in episodes
    ]
    with open(out_path, "w") as f:
        json.dump(serialised, f)
    print(f"[diagnose] Episodes saved to {out_path}")


def load_episodes(path):
    """Load episodes saved by save_episodes(); returns list of dicts."""
    raw = json.loads(Path(path).read_text())
    for ep in raw:
        for k in ("T1_true", "sphere_indices"):
            if k in ep:
                ep[k] = np.asarray(ep[k])
    return raw


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
    colors = {"all_long": "#1f77b4", "mixed": "#ff7f0e", "all_short": "#2ca02c"}
    for name, vals in buckets.items():
        vals = np.asarray(vals, dtype=np.float64)
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if vals.size:
            counts, _ = np.histogram(vals, bins=bins)
            pct = counts / counts.sum() * 100  # normalise to % of blocks in bucket
            bin_centres = np.sqrt(bins[:-1] * bins[1:])
            ax.step(bin_centres, pct, where="mid", linewidth=2,
                    label=f"{name} (n={vals.size} blocks)",
                    color=colors.get(name))
    ax.set_xscale("log")
    ax.set_xlabel("TI [s] (log scale)")
    ax.set_ylabel("% of blocks in bucket")
    ax.set_title("TI distributions by episode T1-subset bucket (normalised)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)

    bucket_tis = {name: np.asarray(vals, dtype=np.float64).tolist()
                  for name, vals in buckets.items()}
    return {
        "bucket_counts_episodes": {
            name: int(sum(_subset_bucket(ep) == name for ep in episodes))
            for name in buckets
        },
        "bucket_counts_blocks": {name: int(len(vals)) for name, vals in buckets.items()},
        "ks_all_long_vs_all_short": _ks_2sample(buckets["all_long"], buckets["all_short"]),
        "bucket_tis": bucket_tis,
    }


_T1_POOL_3T = np.array([
    1.838, 1.398, 0.9983, 0.7258, 0.5091, 0.367, 0.2587,
    0.1847, 0.1308, 0.0909, 0.0642, 0.04628, 0.03265, 0.02295,
])


def _all_subsets_by_bucket():
    """Return dict bucket→list of 0-based index tuples for all C(14,5)=2002 subsets."""
    from itertools import combinations
    buckets = {"all_long": [], "all_short": [], "mixed": []}
    for idxs in combinations(range(len(_T1_POOL_3T)), 5):
        t1 = _T1_POOL_3T[list(idxs)]
        b = _subset_bucket({"T1_true": t1})
        buckets[b].append(list(idxs))
    return buckets


def collect_stratified(policy_path, vecnorm_path, seed_offset,
                       target_per_bucket=30, mixed_target=100, **env_kwargs):
    """Collect episodes with forced sphere subsets to get balanced bucket counts.

    Precomputes all 2002 valid 5-of-14 subsets in Python, classifies them by
    bucket, then runs `target_per_bucket` episodes per bucket (and `mixed_target`
    mixed episodes) with the subset injected directly into the env.  T1 jitter
    and all other randomness still vary with the episode seed.
    """
    import random as _random
    all_buckets = _all_subsets_by_bucket()
    print(f"[diagnose] Bucket pool sizes: "
          f"all_long={len(all_buckets['all_long'])}, "
          f"all_short={len(all_buckets['all_short'])}, "
          f"mixed={len(all_buckets['mixed'])}")

    targets = {"all_long": target_per_bucket,
               "all_short": target_per_bucket,
               "mixed": mixed_target}
    selected = []  # list of (0-based indices, bucket_label)
    for bucket, tgt in targets.items():
        pool = all_buckets[bucket]
        chosen = _random.sample(pool, min(tgt, len(pool)))
        selected.extend((idxs, bucket) for idxs in chosen)

    forced_list = [idxs for idxs, _ in selected]
    bucket_labels = [b for _, b in selected]

    print(f"[diagnose] Running {len(selected)} stratified episodes "
          f"({target_per_bucket}/{target_per_bucket}/{mixed_target} "
          f"all_long/all_short/mixed) …")
    eps = collect(policy_path, vecnorm_path, len(selected), seed_offset,
                  forced_indices_list=forced_list, **env_kwargs)
    for ep, label in zip(eps, bucket_labels):
        ep["bucket"] = label
    return eps


def plot_ti_vs_subset_t1(episodes, out_path):
    """Scatter: mean(log TI per episode) vs mean(T1_true of subset).

    Tests the continuous version of the adaptivity hypothesis: does the policy
    choose longer TIs when the active subset has longer T1 values?
    Returns the log-log Pearson r and n.
    """
    xs, ys = [], []
    for ep in episodes:
        t1_true = np.asarray(ep.get("T1_true", []), dtype=np.float64)
        tis = np.asarray(ep["TI"], dtype=np.float64)
        tis = tis[np.isfinite(tis) & (tis > 0)]
        t1_true = t1_true[np.isfinite(t1_true) & (t1_true > 0)]
        if t1_true.size == 0 or tis.size == 0:
            continue
        xs.append(float(np.mean(t1_true)))
        ys.append(float(np.mean(np.log(tis))))

    if len(xs) < 3:
        print("[diagnose] Not enough data for subset-T1 correlation plot.")
        return {"pearson_log_r": None, "n": len(xs)}

    log_x = np.log(np.array(xs))
    y_arr = np.array(ys)
    r = float(np.corrcoef(log_x, y_arr)[0, 1])
    n = len(xs)

    # Regression line
    m, b = np.polyfit(log_x, y_arr, 1)
    x_line = np.linspace(log_x.min(), log_x.max(), 100)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(np.exp(log_x), np.exp(y_arr), alpha=0.55, s=25)
    ax.plot(np.exp(x_line), np.exp(m * x_line + b), color="crimson", linewidth=1.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Mean T1_true of active subset [s]")
    ax.set_ylabel("Geometric mean TI chosen this episode [s]")
    ax.set_title(f"Subset-T1 vs mean TI (log–log Pearson r = {r:+.3f}, N={n})\n"
                 "Negative r → policy targets shorter TIs for short-T1 subsets")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return {"pearson_log_r": r, "n": n}


def plot_alpha_vs_tr_vs_t1(episodes, out_path):
    """α-vs-TR scatter with Ernst-angle curves overlaid (ALPHA_DOF.md, Ernst diagnostic).

    Tests for Ernst-angle emergence under Run A: if the policy is behaving
    SNR-efficiently, its (TR, α) choices should track the Ernst curve
    α* = acos(exp(−TR/T1)) for the fleet's T1 range. Points are coloured by the
    mean running T1 estimate at decision time.
    """
    TRs, alphas, t1est = [], [], []
    for ep in episodes:
        tr = np.asarray(ep.get("TR", []), dtype=np.float64)
        al = np.asarray(ep.get("alpha_deg", []), dtype=np.float64)
        te = np.asarray(ep.get("T1_est_at_decision", []), dtype=np.float64)
        m = min(len(tr), len(al), len(te))
        if m == 0:
            continue
        TRs.extend(tr[:m]); alphas.extend(al[:m]); t1est.extend(te[:m])
    TRs = np.asarray(TRs); alphas = np.asarray(alphas); t1est = np.asarray(t1est)
    ok = np.isfinite(TRs) & np.isfinite(alphas)
    if not ok.any():
        return {"n": 0}

    fig, ax = plt.subplots(figsize=(8, 5))
    c = np.where(np.isfinite(t1est) & (t1est > 0), t1est, np.nan)
    sc = ax.scatter(TRs[ok], alphas[ok],
                    c=np.log10(c[ok]) if np.isfinite(c[ok]).any() else None,
                    cmap="viridis", alpha=0.55, s=22)
    if np.isfinite(c[ok]).any():
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label("log10(mean running T1_est at decision) [s]")

    tr_grid = np.linspace(0.5, 5.0, 200)
    for T1, style in ((0.046, ":"), (0.367, "--"), (1.398, "-")):
        ernst = np.degrees(np.arccos(np.clip(np.exp(-tr_grid / T1), 0.0, 1.0)))
        ax.plot(tr_grid, ernst, style, color="crimson", linewidth=1.5,
                label=f"Ernst (T1={T1:.3f}s)")

    ax.set_xlabel("TR [s]")
    ax.set_ylabel("excitation flip angle α [deg]")
    ax.set_title("Learned (TR, α) vs Ernst-angle curves")
    ax.set_ylim(0, 95)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return {"n": int(ok.sum()),
            "alpha_deg_mean": float(np.nanmean(alphas[ok])),
            "alpha_deg_std": float(np.nanstd(alphas[ok]))}


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
    p.add_argument("--fix-te", action="store_true",
                   help="Required if the policy was trained with --fix-te.")
    p.add_argument("--learn-alpha", action="store_true",
                   help="Required if the policy was trained with --learn-alpha.")
    p.add_argument("--log-ti-action", action="store_true",
                   help="Required if the policy was trained with "
                        "--log-ti-action.")
    p.add_argument("--include-image", action="store_true",
                   help="Required if the policy was trained with --include-image.")
    p.add_argument("--include-sigma", action="store_true",
                   help="Required if the policy was trained with --include-sigma.")
    p.add_argument("--max-blocks", type=int, default=15)
    p.add_argument("--time-budget", type=float, default=120.0)
    p.add_argument("--subset-size", type=int, default=None,
                   help="Diagnose policy on random k-sphere subsets.")
    p.add_argument("--phase-sensitive", action="store_true")
    p.add_argument("--sigma-method", type=str, default="bootstrap",
                   choices=["asymptotic", "profile_likelihood", "bootstrap"])
    p.add_argument("--from-json", type=Path, default=None, metavar="EPISODES_JSON",
                   help="Skip Julia rollout; load episodes from a previously saved "
                        "episodes.json and just regenerate plots.")
    p.add_argument("--stratified", action="store_true",
                   help="Use forced-injection stratified collection instead of "
                        "uniform random subsets. Requires --subset-size.")
    p.add_argument("--target-per-bucket", type=int, default=30,
                   help="Episodes per all_long/all_short bucket (--stratified only).")
    p.add_argument("--mixed-target", type=int, default=100,
                   help="Episodes for the mixed bucket (--stratified only).")
    p.add_argument("--water-model", type=str, default="bloch",
                   choices=["bloch", "cached_perline", "analytic"],
                   help="Must match training to keep obs distribution consistent.")
    p.add_argument("--noise-sigma-abs", type=float, default=50.0)
    p.add_argument("--reward-mode", type=str, default="neg_mape")
    p.add_argument("--terminal-bonus", type=float, default=0.5)
    p.add_argument("--mape-alpha", type=float, default=1.0)
    p.add_argument("--allow-stop", action="store_true")
    p.add_argument("--use-gpu", action="store_true")
    p.add_argument("--nfe", type=int, default=None)
    p.add_argument("--npe", type=int, default=None)
    args = p.parse_args()

    out_dir = args.out or (args.policy.parent / "diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)

    env_kwargs = dict(
        cfg_field=args.field,
        max_blocks=args.max_blocks,
        time_budget_s=args.time_budget,
        subset_size=args.subset_size,
        phase_sensitive=args.phase_sensitive,
        sigma_method=args.sigma_method,
        simplified_action=args.simplified_action,
        fix_te=args.fix_te,
        learn_alpha=args.learn_alpha,
        log_ti_action=args.log_ti_action,
        include_image=args.include_image,
        include_sigma=args.include_sigma,
        water_model=args.water_model,
        noise_sigma_abs=args.noise_sigma_abs,
        reward_mode=args.reward_mode,
        terminal_bonus=args.terminal_bonus,
        mape_alpha=args.mape_alpha,
        allow_stop=args.allow_stop,
        use_gpu=args.use_gpu,
    )
    if args.nfe is not None:
        env_kwargs["Nfe"] = args.nfe
    if args.npe is not None:
        env_kwargs["Npe"] = args.npe
    snr_measurement = {}
    if args.from_json is not None:
        print(f"[diagnose] Loading episodes from {args.from_json} (skipping Julia) …")
        eps = load_episodes(args.from_json)
    elif args.stratified:
        if not args.subset_size:
            raise ValueError("--stratified requires --subset-size")
        eps = collect_stratified(
            args.policy, args.vecnorm, args.seed,
            target_per_bucket=args.target_per_bucket,
            mixed_target=args.mixed_target,
            snr_holder=snr_measurement,
            **env_kwargs,
        )
        save_episodes(eps, out_dir / "episodes.json")
    else:
        print(f"[diagnose] Collecting {args.episodes} rollouts from {args.policy} …")
        eps = collect(args.policy, args.vecnorm, args.episodes, args.seed,
                      snr_holder=snr_measurement, **env_kwargs)
        save_episodes(eps, out_dir / "episodes.json")

    print("[diagnose] Plotting …")
    plot_ti_per_episode  (eps, out_dir / "ti_per_episode.png")
    plot_ti_histogram    (eps, out_dir / "ti_histogram.png")
    plot_t1est_trajectory(eps, out_dir / "t1est_trajectory.png")
    plot_snr_vs_block    (eps, out_dir / "snr_vs_block.png",
                          dual_acq_snr=snr_measurement.get("snr_dual_peak"))
    ti_vs_t1est_summary = plot_ti_vs_t1est(eps, out_dir / "ti_vs_t1est.png")
    alpha_ernst_summary = plot_alpha_vs_tr_vs_t1(
        eps, out_dir / "alpha_vs_tr_vs_t1.png")
    summary = write_summary(eps, out_dir / "diagnose_summary.json")
    summary["ti_vs_t1est_correlations"] = ti_vs_t1est_summary
    summary["alpha_ernst_diagnostic"] = alpha_ernst_summary
    if snr_measurement:
        summary["snr_report"] = snr_measurement
    bucket_summary = None
    subset_t1_corr = None
    if args.subset_size:
        bucket_summary = plot_subset_bucket_histograms(
            eps, out_dir / "ti_histogram_by_subset_bucket.png")
        summary["subset_bucket_diagnostic"] = bucket_summary
        subset_t1_corr = plot_ti_vs_subset_t1(
            eps, out_dir / "ti_vs_subset_t1.png")
        summary["subset_t1_correlation"] = subset_t1_corr
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
    if subset_t1_corr is not None:
        r = subset_t1_corr["pearson_log_r"]
        rstr = f"{r:+.3f}" if r is not None else "n/a"
        print(f"  Subset-T1 vs mean-TI r  = {rstr}  (N={subset_t1_corr['n']})")
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

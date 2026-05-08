"""V1 validation for E2.4 — replay E2.3 under the F1+ fitter and compare.

The E2.3 policy at `runs/e2/e2_3_A_C_delta/policy.zip` was trained against
the *steady-state* IR forward model (`steady_state_mz_at_excite`). After
`E2_4_PLAN.md` §2 lands, the env now uses the finite-Npe transient closed
form (F1+, `transient_mz_at_excite_npe`) and an absolute σ floor (Fix B).

This script runs the same policy + same eval seeds through the corrected
env and compares against the diagnostic snapshot recorded under the old
fitter at `runs/e2/e2_3_A_C_delta/diagnostics/sigma_summary.json`.

Caveat: this is *not* a pure fitter-ablation. The policy reads `T1_est`
and `log10(σ/T1_est)` from the observation, so a corrected fitter
changes the obs distribution and therefore the action sequence the
policy chooses. Both effects (better fit + better-informed policy
decisions) move MAPE in the same direction, so a drop here confirms H1
+ exposes the additional benefit; the *ceiling* of what F1+ can deliver
on this trained policy.

For a pure fitter-ablation (action sequence identical, fitter swapped)
we'd need to record per-sphere `block_mags` during diagnose so we could
re-fit offline. That requires extending `diagnose_uncertainty.py` and is
left for V2.

Usage:
    PYTHON_JULIAPKG_OFFLINE=yes python python/refit_e2_3.py \\
        --policy   runs/e2/e2_3_A_C_delta/policy.zip \\
        --vecnorm  runs/e2/e2_3_A_C_delta/vecnorm.pkl \\
        --old-summary runs/e2/e2_3_A_C_delta/diagnostics/sigma_summary.json \\
        --episodes 30 --simplified-action

Outputs (default `runs/e2/e2_3_A_C_delta/refit/`):
    refit_summary.json     per-episode + per-sphere old vs new
    per_sphere_mape.png    bar chart, old vs new MAPE per sphere
    sigma_calibration.png  scatter, σ_T1 vs |T1_est − T1_true|, old vs new
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


def collect_new(policy_path: Path, vecnorm_path: Path | None,
                 n_episodes: int, seed_offset: int, **env_kwargs):
    """Run the policy in the (F1+) env. Returns one dict per episode with
    the same shape as `diagnose_uncertainty.collect()` so plotting code can
    treat new and old uniformly."""
    raw = QalibreMDE2Env(rng_seed=seed_offset, **env_kwargs)
    vec_norm = None
    if vecnorm_path is not None and vecnorm_path.exists():
        venv_tmp = DummyVecEnv(
            [lambda: QalibreMDE2Env(rng_seed=seed_offset, **env_kwargs)]
        )
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
            "T1_true": np.asarray(raw.T1_true),
            "T1_est":  [], "sigma": [],
            "TI": [], "TE": [], "TR": [], "alpha_deg": [],
            "block_time": [], "mape": [],
        }
        done = False
        while not done:
            action, _ = model.predict(_norm(obs), deterministic=True)
            obs, _r, done, _trunc, info = raw.step(action)
            rec["T1_est"].append(np.asarray(raw.T1_est))
            rec["sigma"].append(np.asarray(raw.T1_sigma))
            rec["TI"].append(float(info.get("TI", np.nan)))
            rec["TE"].append(float(info.get("TE", np.nan)))
            rec["TR"].append(float(info.get("TR", np.nan)))
            rec["alpha_deg"].append(float(info.get("alpha_deg", np.nan)))
            rec["block_time"].append(float(info.get("block_time", np.nan)))
            rec["mape"].append(float(info.get("mape", np.nan)))
        rec["T1_est"] = np.asarray(rec["T1_est"])
        rec["sigma"]  = np.asarray(rec["sigma"])
        for k in ("TI", "TE", "TR", "alpha_deg", "block_time", "mape"):
            rec[k] = np.asarray(rec[k])
        episodes.append(rec)
    return episodes


def load_old(path: Path):
    """Read the snapshot from the previous diagnose_uncertainty run."""
    with open(path) as f:
        data = json.load(f)
    eps = []
    for ep in data["per_episode"]:
        eps.append({
            "T1_true":   np.asarray(ep["T1_true"]),
            "T1_est":    np.asarray(ep["T1_est"]),
            "sigma":     np.asarray(ep["sigma"]),
            "TI":        np.asarray(ep["TI"]),
            "TE":        np.asarray(ep["TE"]),
            "TR":        np.asarray(ep["TR"]),
            "alpha_deg": np.asarray(ep["alpha_deg"]),
            "block_time":np.asarray(ep["block_time"]),
            "mape":      np.asarray(ep["mape"]),
        })
    return eps


def per_sphere_mape(episodes):
    """Final-block per-sphere MAPE (%), mean across episodes."""
    n_spheres = episodes[0]["T1_true"].size
    mapes = np.full((len(episodes), n_spheres), np.nan)
    for i, ep in enumerate(episodes):
        if len(ep["T1_est"]) == 0:
            continue
        T1e = ep["T1_est"][-1]
        T1t = ep["T1_true"]
        with np.errstate(divide="ignore", invalid="ignore"):
            mapes[i] = np.abs(T1e - T1t) / np.maximum(T1t, 1e-9) * 100
    return np.nanmean(mapes, axis=0), np.nanstd(mapes, axis=0)


def sigma_calibration_points(episodes):
    """For every (episode, sphere, final-block) cell, return the pair
    (σ_T1, |T1_est − T1_true|). NaN/Inf-cleaned."""
    sigs, errs = [], []
    for ep in episodes:
        if len(ep["sigma"]) == 0:
            continue
        s   = ep["sigma"][-1]
        T1e = ep["T1_est"][-1]
        T1t = ep["T1_true"]
        for sph in range(T1t.size):
            if not (np.isfinite(s[sph]) and np.isfinite(T1e[sph]) and T1t[sph] > 0):
                continue
            sigs.append(float(s[sph]))
            errs.append(float(abs(T1e[sph] - T1t[sph])))
    return np.asarray(sigs), np.asarray(errs)


def overall_mape(episodes):
    """Final-block MAPE across all (episode, sphere) cells in %."""
    vals = []
    for ep in episodes:
        if len(ep["T1_est"]) == 0:
            continue
        T1e = ep["T1_est"][-1]
        T1t = ep["T1_true"]
        m = np.abs(T1e - T1t) / np.maximum(T1t, 1e-9) * 100
        vals.extend(m.tolist())
    vals = np.asarray(vals)
    return {
        "mean":   float(np.nanmean(vals)),
        "median": float(np.nanmedian(vals)),
        "p90":    float(np.nanpercentile(vals, 90)),
    }


def sigma_summary(episodes):
    """Final-block σ_T1/T1_est (relative) summary."""
    vals = []
    for ep in episodes:
        if len(ep["sigma"]) == 0:
            continue
        s   = ep["sigma"][-1]
        T1e = ep["T1_est"][-1]
        with np.errstate(divide="ignore", invalid="ignore"):
            r = s / np.maximum(T1e, 1e-9)
        vals.extend(r.tolist())
    vals = np.asarray(vals)
    finite = np.isfinite(vals)
    return {
        "median":  float(np.nanmedian(vals[finite])) if finite.any() else float("nan"),
        "p90":     float(np.nanpercentile(vals[finite], 90)) if finite.any() else float("nan"),
        "above_1": float(np.mean(vals[finite] >= 1.0)) if finite.any() else float("nan"),
        "n_finite": int(finite.sum()),
        "n_total":  int(vals.size),
    }


def plot_per_sphere_mape(old_eps, new_eps, out_path):
    n_spheres = old_eps[0]["T1_true"].size
    old_mean, old_std = per_sphere_mape(old_eps)
    new_mean, new_std = per_sphere_mape(new_eps)
    idx = np.arange(1, n_spheres + 1)
    width = 0.4
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(idx - width / 2, old_mean, width, yerr=old_std, capsize=3,
           color="#d62728", edgecolor="black", linewidth=0.4,
           label=f"E2.3 steady-state fit (mean overall {np.nanmean(old_mean):.0f}%)")
    ax.bar(idx + width / 2, new_mean, width, yerr=new_std, capsize=3,
           color="#2ca02c", edgecolor="black", linewidth=0.4,
           label=f"E2.4 F1+ fit (mean overall {np.nanmean(new_mean):.0f}%)")
    ax.set_xticks(idx)
    ax.set_xlabel("Sphere index (1 = longest T1, 14 = shortest)")
    ax.set_ylabel("Final-block MAPE [%]")
    ax.set_yscale("log")
    ax.set_title("Per-sphere MAPE — E2.3 (steady-state) vs E2.4 (F1+, same policy)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_sigma_calibration(old_eps, new_eps, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True, sharex=True)
    for ax, eps, label, color in [
        (axes[0], old_eps, "E2.3 steady-state", "#d62728"),
        (axes[1], new_eps, "E2.4 F1+",         "#2ca02c"),
    ]:
        sigs, errs = sigma_calibration_points(eps)
        # Avoid log(0)
        sigs_p = np.clip(sigs, 1e-6, None)
        errs_p = np.clip(errs, 1e-6, None)
        ax.scatter(sigs_p, errs_p, alpha=0.45, s=20, color=color, edgecolor="none")
        # Diagonal y=x reference
        lims = [1e-4, 1e2]
        ax.plot(lims, lims, "k--", linewidth=1, label="σ = |error|")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(lims);    ax.set_ylim(lims)
        ax.set_xlabel("Reported σ_T1 [s]")
        ax.set_title(label)
        ax.grid(True, which="both", alpha=0.3)
        # Calibration ratio: median(σ / |err|)
        if len(sigs) > 0:
            ratio = np.nanmedian(sigs / np.maximum(errs, 1e-9))
            ax.text(0.05, 0.95, f"median σ/|err| = {ratio:.3f}\nideal ≈ 1",
                    transform=ax.transAxes, va="top", fontsize=9,
                    bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray"))
    axes[0].set_ylabel("|T1_est − T1_true| [s]")
    fig.suptitle("σ calibration: reported uncertainty vs realised error")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def write_summary(old_eps, new_eps, out_path):
    out = {
        "n_episodes": len(new_eps),
        "old": {
            "mape":  overall_mape(old_eps),
            "sigma": sigma_summary(old_eps),
            "n_blocks_per_episode": [int(len(ep["TI"])) for ep in old_eps],
        },
        "new": {
            "mape":  overall_mape(new_eps),
            "sigma": sigma_summary(new_eps),
            "n_blocks_per_episode": [int(len(ep["TI"])) for ep in new_eps],
        },
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy",   type=Path, required=True)
    p.add_argument("--vecnorm",  type=Path, default=None)
    p.add_argument("--old-summary", type=Path, required=True,
                    help="Path to the diagnose_uncertainty sigma_summary.json")
    p.add_argument("--episodes", type=int,   default=30)
    p.add_argument("--seed",     type=int,   default=500_000,
                    help="Same default as diagnose_uncertainty.py")
    p.add_argument("--field",    type=str,   default="T3")
    p.add_argument("--simplified-action", action="store_true")
    p.add_argument("--out",      type=Path, default=None)
    args = p.parse_args()

    out_dir = args.out or (args.policy.parent / "refit")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[refit] Loading old summary  {args.old_summary}")
    old_eps = load_old(args.old_summary)
    if len(old_eps) < args.episodes:
        print(f"[refit] WARN: old summary has only {len(old_eps)} episodes "
              f"(< requested {args.episodes}); will compare what's available")

    print(f"[refit] Replaying {args.episodes} eps under F1+ env  policy={args.policy}")
    new_eps = collect_new(
        args.policy, args.vecnorm, args.episodes, args.seed,
        cfg_field=args.field, simplified_action=args.simplified_action,
    )

    summary = write_summary(old_eps, new_eps, out_dir / "refit_summary.json")
    plot_per_sphere_mape (old_eps, new_eps, out_dir / "per_sphere_mape.png")
    plot_sigma_calibration(old_eps, new_eps, out_dir / "sigma_calibration.png")

    # Console snapshot — the H1/H2 acceptance criteria for E2.4 V1.
    print("\n" + "=" * 60)
    print("E2.4 V1 — Refit-of-E2.3 acceptance check")
    print("=" * 60)
    print(f"  Episodes:  old={len(old_eps)}  new={len(new_eps)}")
    print()
    print("MAPE % across all (episode, sphere) final-block cells:")
    for label in ("old", "new"):
        m = summary[label]["mape"]
        print(f"  {label:>3}:  median={m['median']:7.1f}  "
              f"mean={m['mean']:8.1f}  p90={m['p90']:8.1f}")
    print()
    print("σ_T1 / T1_est across the same cells:")
    for label in ("old", "new"):
        s = summary[label]["sigma"]
        print(f"  {label:>3}:  median={s['median']*100:6.1f}%  "
              f"p90={s['p90']*100:7.1f}%  fraction≥100%={s['above_1']*100:5.1f}%  "
              f"(finite {s['n_finite']}/{s['n_total']})")
    print()

    h1 = summary["new"]["mape"]["mean"] < 50.0
    sigs_o, errs_o = sigma_calibration_points(old_eps)
    sigs_n, errs_n = sigma_calibration_points(new_eps)
    h2_old = float(np.nanmedian(sigs_o / np.maximum(errs_o, 1e-9))) if len(sigs_o) else float("nan")
    h2_new = float(np.nanmedian(sigs_n / np.maximum(errs_n, 1e-9))) if len(sigs_n) else float("nan")
    summary["old"]["sigma_calibration_median"] = h2_old
    summary["new"]["sigma_calibration_median"] = h2_new
    with open(out_dir / "refit_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"H1 (mean MAPE < 50%): {'PASS' if h1 else 'FAIL'}")
    # H2: median σ/|err| → 1.0 is ideal; "<<1" is over-confident (σ too small);
    # ">>1" is over-saturated (σ too large). Acceptance ≥ 0.3.
    print(f"H2 (median σ/|err|): old={h2_old:.3f}  new={h2_new:.3f}  "
          f"(ideal=1.0; old likely << 1)  "
          f"{'PASS' if h2_new >= 0.3 else 'FAIL'}")
    print(f"\nOutputs in {out_dir}")


if __name__ == "__main__":
    main()

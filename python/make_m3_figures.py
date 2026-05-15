"""Generate the two M3 meeting figures from saved JSON data (no Julia needed).

Figures produced:
  report_plots/E2_tractability_V12/v12_vs_fixed_anchors.png
      V12 mean MAPE and p90 MAPE vs CR-opt / log-grid / TR-matched baselines
      (replaces the old V9 version in M3.md §2.4 / §6)

  runs/e2/e2_tractability_V12/diagnostics/ti_histogram_by_subset_bucket.png
      Normalised TI distribution by T1-subset bucket (% of blocks, not counts).
      Requires diagnose_e2.py to have been run once (it saves episodes.json and
      bucket_tis in diagnose_summary.json). After that, regenerate instantly with:
          python python/diagnose_e2.py --policy ... --from-json .../episodes.json

Run from repo root:
    python python/make_m3_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "legend.fontsize":    9,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.25,
    "grid.linewidth":     0.6,
})


# ── Figure 1: V12 vs fixed anchors ──────────────────────────────────────────

def make_v12_vs_anchors():
    """Bar chart: V12 mean MAPE and p90 MAPE vs CR-opt and log-grid baselines."""

    # V12 N=200 paired eval (from runs/e2/e2_tractability_V12/eval_n200.log)
    v12_mean = 321.62
    v12_p90  = 746.74

    # Baselines from runs/e2/e2_tractability_baselines_n200/baseline_summary.json
    bl_path = REPO / "runs/e2/e2_tractability_baselines_n200/baseline_summary.json"
    if not bl_path.exists():
        print(f"[make_m3_figures] WARNING: baseline summary not found at {bl_path}")
        cr_mean, cr_p90 = 421.0, 1006.2
        lg_mean, lg_p90 = 534.4, 1317.1
        lgt_mean, lgt_p90 = 466.6, 1092.8
    else:
        bl = json.loads(bl_path.read_text())
        cr_mean  = bl["cr_optimal"]["mape_pct"]
        cr_p90   = bl["cr_optimal"]["mape_p90_pct"]
        lg_mean  = bl["log_grid"]["mape_pct"]
        lg_p90   = bl["log_grid"]["mape_p90_pct"]
        lgt_mean = bl["log_grid_trmatched"]["mape_pct"]
        lgt_p90  = bl["log_grid_trmatched"]["mape_p90_pct"]

    labels  = ["log-grid\n(TR=4 s)", "TR-matched\nlog-grid", "CR-optimal\n(fixed)", "V12 (RL)"]
    means   = [lg_mean, lgt_mean, cr_mean, v12_mean]
    p90s    = [lg_p90,  lgt_p90,  cr_p90,  v12_p90]
    colors  = ["#aec7e8", "#ffbb78", "#c5b0d5", "#2ca02c"]  # muted for fixed; green for RL

    x = np.arange(len(labels))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=False)

    for ax, vals, metric in zip(axes, [means, p90s], ["Mean MAPE [%]", "p90 MAPE [%]"]):
        bars = ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel(metric)
        ax.set_title(metric)
        # Annotate bars with value
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 5,
                    f"{v:.0f}%",
                    ha="center", va="bottom", fontsize=9)

    axes[0].set_title("Mean MAPE — V12 vs fixed schedules (N=200 paired episodes)")
    axes[1].set_title("p90 MAPE — V12 vs fixed schedules")

    fig.suptitle("V12 (RL, magnitude, log-TI) vs fixed-schedule anchors\n"
                 "5-of-14 random subset, 250 s budget, F1+ fitter",
                 fontsize=11)
    fig.tight_layout()

    out_dir = REPO / "report_plots/E2_tractability_V12"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "v12_vs_fixed_anchors.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[make_m3_figures] Saved {out_path}")


# ── Figure 2: normalised bucket histogram (from saved diagnose_summary.json) ──

def make_bucket_histogram_from_json():
    """Re-draw the normalised TI-by-subset-bucket histogram from saved JSON.

    Requires diagnose_summary.json to contain 'subset_bucket_diagnostic.bucket_tis',
    which is written by the updated diagnose_e2.py.
    """
    summary_path = (REPO / "runs/e2/e2_tractability_V12/diagnostics/diagnose_summary.json")
    if not summary_path.exists():
        print(f"[make_m3_figures] WARNING: {summary_path} not found — skipping bucket plot")
        return

    d = json.loads(summary_path.read_text())
    sbd = d.get("subset_bucket_diagnostic", {})
    bucket_tis = sbd.get("bucket_tis")
    if bucket_tis is None:
        print("[make_m3_figures] WARNING: bucket_tis not in diagnose_summary.json "
              "— re-run diagnose_e2.py first, then re-run this script")
        return

    bins = np.logspace(np.log10(0.01), np.log10(3.0), 30)
    colors = {"all_long": "#1f77b4", "mixed": "#ff7f0e", "all_short": "#2ca02c"}
    ep_counts = sbd.get("bucket_counts_episodes", {})

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, tis_list in bucket_tis.items():
        vals = np.asarray(tis_list, dtype=np.float64)
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if vals.size == 0:
            continue
        counts, _ = np.histogram(vals, bins=bins)
        pct = counts / counts.sum() * 100
        bin_centres = np.sqrt(bins[:-1] * bins[1:])
        n_eps = ep_counts.get(name, "?")
        ax.step(bin_centres, pct, where="mid", linewidth=2,
                label=f"{name} ({n_eps} eps, {vals.size} blocks)",
                color=colors.get(name))

    ax.set_xscale("log")
    ax.set_xlabel("TI [s] (log scale)")
    ax.set_ylabel("% of blocks in bucket")
    ax.set_title("TI distributions by T1-subset bucket — V12 (normalised)\n"
                 "KS p < 1e-6 (all-long vs all-short)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = summary_path.parent / "ti_histogram_by_subset_bucket.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[make_m3_figures] Saved {out_path}")


def make_subset_t1_scatter_from_json(summary_path=None):
    """Re-draw subset-T1 vs mean-TI scatter from saved diagnose_summary.json."""
    if summary_path is None:
        summary_path = (REPO / "runs/e2/e2_tractability_V12/diagnostics/diagnose_summary.json")
    summary_path = Path(summary_path)
    if not summary_path.exists():
        print(f"[make_m3_figures] WARNING: {summary_path} not found — skipping subset-T1 scatter")
        return

    d = json.loads(summary_path.read_text())
    corr = d.get("subset_t1_correlation")
    if corr is None:
        print("[make_m3_figures] WARNING: subset_t1_correlation not in summary "
              "— re-run diagnose_e2.py first")
        return

    r = corr.get("pearson_log_r")
    n = corr.get("n", "?")
    print(f"[make_m3_figures] subset_t1_correlation: r={r:+.3f} (N={n})")


if __name__ == "__main__":
    make_v12_vs_anchors()
    make_bucket_histogram_from_json()
    make_subset_t1_scatter_from_json()

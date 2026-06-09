"""Plot T1 fitted vs T1 true — scatter + per-sphere MAPE bar chart.

Left panel:  T1_fit vs T1_true scatter (diagonal = perfect), coloured by MAPE.
             Data from runs/t1_fit_vs_true/accurate/t1_fit_vs_true.csv (Julia clean run,
             no noise, no jitter, CR-optimal schedule).

Right panel: Per-sphere MAPE bar chart.  Always shows the Julia clean run.
             Optionally overlays E2-env baselines and/or an RL policy eval.

Usage examples:

  # Julia clean run only (default)
  python scripts/t1_fit_vs_true.py

  # Add E2 baseline comparison
  python scripts/t1_fit_vs_true.py \\
      --baselines runs/e2/e2_tractability_baselines_ps_n200/baseline_summary.json

  # Add RL policy on top
  python scripts/t1_fit_vs_true.py \\
      --baselines runs/e2/e2_tractability_baselines_ps_n200/baseline_summary.json \\
      --policy    runs/e2/e2_tractability_V12/eval_summary.json \\
      --policy-label "V12 PPO"
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import csv

here = os.path.dirname(os.path.abspath(__file__))
# Override with RUNS_ROOT to target a version folder (see pixel_grid_overlay.py).
runs_t1 = os.path.join(os.environ.get("RUNS_ROOT", os.path.join(here, "runs")),
                       "t1_fit_vs_true")


# ── Data loaders ─────────────────────────────────────────────────────────────

def load_julia(subdir="accurate"):
    path = os.path.join(runs_t1, subdir, "t1_fit_vs_true.csv")
    labels, T1_true, T1_fit, T1_sigma, mape, cx, cy = [], [], [], [], [], [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.append(row["label"])
            T1_true.append(float(row["T1_true_s"]))
            T1_fit.append(float(row["T1_fit_s"]))
            T1_sigma.append(float(row["T1_sigma_s"]))
            mape.append(float(row["mape_pct"]))
            cx.append(float(row["cx_m"]))
            cy.append(float(row["cy_m"]))
    return {
        "labels":   labels,
        "T1_true":  np.array(T1_true),
        "T1_fit":   np.array(T1_fit),
        "T1_sigma": np.array(T1_sigma),
        "mape":     np.array(mape),
        "cx":       np.array(cx),
        "cy":       np.array(cy),
    }


def load_run_config(subdir):
    """Load config.json from a run subdir; returns {} if absent."""
    path = os.path.join(runs_t1, subdir, "config.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def load_baselines(path):
    """Load baseline_summary.json → dict of {policy_name: per_pool_mape (14,)}."""
    with open(path) as f:
        raw = json.load(f)
    out = {}
    for name, v in raw.items():
        if name.startswith("_") or not isinstance(v, dict):
            continue  # skip "_run_meta" and any non-schedule entries
        arr = v.get("per_pool_sphere_mape_pct") or v.get("per_sphere_mape_pct")
        if arr is not None:
            out[name] = np.array(arr, dtype=float)
    return out


def load_policy(path):
    """Load eval_summary.json → per-sphere MAPE array (pool-aligned if 14 entries)."""
    with open(path) as f:
        raw = json.load(f)
    # eval_summary.json uses "per_sphere" (may be subset-length)
    # baseline_summary.json entries use "per_pool_sphere_mape_pct"
    arr = (raw.get("per_sphere")
           or raw.get("per_pool_sphere_mape_pct")
           or raw.get("per_sphere_mape_pct"))
    if arr is None:
        raise ValueError(f"Cannot find per-sphere MAPE array in {path}")
    return np.array(arr, dtype=float)


# ── Helpers ───────────────────────────────────────────────────────────────────

# Human-readable names for the standard baseline keys
BASELINE_LABELS = {
    "log_grid":           "Log-grid (TR=4 s)",
    "clinical_irse":      "Clinical IR-SE",
    "log_grid_trmatched": "Log-grid (TR-matched)",
    "cr_optimal":         "CR-optimal (E2 env)",
}

# Colours for the bar chart series
SERIES_COLORS = [
    "#4878cf",   # blue  — Julia clean
    "#6acc65",   # green — cr_optimal / first baseline
    "#d65f5f",   # red
    "#b47cc7",   # purple
    "#c4ad66",   # tan
    "#77bedb",   # light blue
]


# ── Spatial panel helper ──────────────────────────────────────────────────────

def _draw_spatial(ax, cx, cy, values, labels, vmin, vmax, cmap, title, sphere_r_m=0.0075):
    """Draw sphere layout coloured by `values` (e.g. T1_true or T1_fit)."""
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    for x, y, v, lbl in zip(cx, cy, values, labels):
        circle = plt.Circle((x * 100, y * 100), sphere_r_m * 100,
                             color=cmap(norm(v)), ec="k", lw=0.8, zorder=3)
        ax.add_patch(circle)
        ax.text(x * 100, y * 100, f"{v:.3f}",
                ha="center", va="center", fontsize=6.5, zorder=4,
                color="white" if norm(v) > 0.5 else "black")
    lim = 11
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("x [cm]")
    ax.set_ylabel("y [cm]")
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.2)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="T1 [s]", fraction=0.046, shrink=0.85)


# ── Plot ──────────────────────────────────────────────────────────────────────

def make_figure(julia, baselines, policy_mape, policy_label, args):
    fig = plt.figure(figsize=(18, 10))
    # Layout: top row = scatter + spatial pair; bottom row = bar chart (full width)
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)
    ax_scatter = fig.add_subplot(gs[0, 0])
    ax_sp_true = fig.add_subplot(gs[0, 1])
    ax_sp_fit  = fig.add_subplot(gs[0, 2])
    ax_bar     = fig.add_subplot(gs[1, :])

    T1_true = julia["T1_true"]
    T1_fit  = julia["T1_fit"]
    mape    = julia["mape"]
    cx      = julia["cx"]
    cy      = julia["cy"]
    labels  = julia["labels"]
    n       = len(T1_true)
    order   = np.argsort(T1_true)          # ascending T1 for x-axis labeling

    # ── Left: scatter T1_fit vs T1_true ──────────────────────────────────────
    norm  = plt.Normalize(vmin=0, vmax=min(max(mape), 30))
    cmap  = cm.RdYlGn_r
    colors = cmap(norm(mape))

    lims = (min(T1_true.min(), T1_fit.min()) * 0.8,
            max(T1_true.max(), T1_fit.max()) * 1.2)
    t_line = np.array(lims)
    ax_scatter.loglog(t_line, t_line, "k--", linewidth=1, alpha=0.5, label="perfect")
    ax_scatter.loglog(t_line, t_line * 1.1, ":", color="grey", linewidth=0.8, alpha=0.4)
    ax_scatter.loglog(t_line, t_line * 0.9, ":", color="grey", linewidth=0.8, alpha=0.4,
                      label="±10 % band")

    sc = ax_scatter.scatter(T1_true, T1_fit, c=mape, cmap=cmap, norm=norm,
                             s=70, zorder=3, edgecolors="k", linewidths=0.5)
    # label each point with T1_true
    for i in range(n):
        ax_scatter.annotate(f"{T1_true[i]:.3f}",
                            (T1_true[i], T1_fit[i]),
                            textcoords="offset points", xytext=(5, 3),
                            fontsize=7, color="dimgrey")

    plt.colorbar(sc, ax=ax_scatter, label="MAPE [%]")
    ax_scatter.set_xlabel("T1 true [s]")
    ax_scatter.set_ylabel("T1 fitted [s]")
    noise_tag = ("no noise" if args.noise_label is None
                 else f"noise σ={args.noise_label}")
    if args.phase_sensitive:
        noise_tag += ", phase-sensitive"
    if args.clean_recon:
        noise_tag += ", clean recon"
    ax_scatter.set_title(f"T1 fit vs true\n(CR-optimal schedule, {noise_tag})")
    ax_scatter.legend(fontsize=8)
    ax_scatter.grid(True, which="both", alpha=0.2)

    # ── Spatial T1 maps ───────────────────────────────────────────────────────
    sp_cmap = cm.plasma
    vmin = T1_true.min() * 0.9
    vmax = T1_true.max() * 1.1
    noise_tag = ("no noise" if args.noise_label is None
                 else f"noise σ={args.noise_label}")
    if args.phase_sensitive:
        noise_tag += ", phase-sensitive"
    if args.clean_recon:
        noise_tag += ", clean recon"
    _draw_spatial(ax_sp_true, cx, cy, T1_true, labels, vmin, vmax, sp_cmap,
                  "T1 true  [s]")
    _draw_spatial(ax_sp_fit,  cx, cy, T1_fit,  labels, vmin, vmax, sp_cmap,
                  f"T1 fitted [s]  ({noise_tag})")

    # ── Bottom: per-sphere MAPE bar chart ─────────────────────────────────────
    # Build list of (label, mape_array) series to plot.
    # All arrays must be length-14 and pool-aligned.
    series = [("CR-optimal\n(Julia, no noise)", mape)]

    for name, arr in baselines.items():
        label = BASELINE_LABELS.get(name, name)
        if len(arr) == n:
            series.append((label, arr))

    if policy_mape is not None and len(policy_mape) == n:
        series.append((policy_label, policy_mape))
    elif policy_mape is not None:
        print(f"  [warn] policy per_sphere length {len(policy_mape)} ≠ {n}; skipped from bar chart")

    n_series = len(series)
    x = np.arange(n)
    width = 0.8 / n_series

    for s_idx, (label, s_mape) in enumerate(series):
        offset = (s_idx - (n_series - 1) / 2) * width
        # Sort by ascending T1_true so x-axis goes short→long
        bars = ax_bar.bar(x[order] + offset, s_mape[order],
                          width=width,
                          color=SERIES_COLORS[s_idx % len(SERIES_COLORS)],
                          edgecolor="k", linewidth=0.4,
                          label=label, alpha=0.85)

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([f"{T1_true[i]:.3f}" for i in order],
                            rotation=45, ha="right", fontsize=8)
    ax_bar.set_xlabel("T1 true [s]  (ascending)")
    ax_bar.set_ylabel("MAPE [%]")
    ax_bar.set_title("Per-sphere MAPE by method")
    ax_bar.axhline(5, color="green",  linestyle="--", linewidth=1,
                   alpha=0.7, label="5 % target")
    ax_bar.axhline(10, color="orange", linestyle=":",  linewidth=1,
                   alpha=0.6, label="10 % threshold")
    ax_bar.legend(fontsize=8, loc="upper left")
    ax_bar.grid(axis="y", alpha=0.3)
    ax_bar.set_ylim(0)

    # Summary stats annotation
    mean_julia = np.mean(mape)
    fig.suptitle(
        f"T1 accuracy: CR-optimal Julia (mean MAPE={mean_julia:.1f}%)"
        + ("" if not series[1:] else
           "  |  " + "  ".join(
               f"{lbl.split(chr(10))[0]}={np.mean(arr):.1f}%"
               for lbl, arr in series[1:]
           )),
        fontsize=11, y=1.01,
    )

    plt.tight_layout()
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--subdir",       default="accurate",
                    help="Subdir under runs/t1_fit_vs_true/ containing t1_fit_vs_true.csv")
parser.add_argument("--baselines",    default=None,
                    help="Path to baseline_summary.json from baseline_e2.py")
parser.add_argument("--baseline-keys", nargs="*", default=None,
                    help="Which baseline keys to include (default: all)")
parser.add_argument("--policy",       default=None,
                    help="Path to eval_summary.json from eval_e2.py")
parser.add_argument("--policy-label", default="RL policy",
                    help="Display name for the policy series")
parser.add_argument("--noise-label",  default=None,
                    help="Noise level string for the scatter title, e.g. '0.05'")
parser.add_argument("--phase-sensitive", action="store_true",
                    help="Mark plot title as phase-sensitive (signed) recon")
parser.add_argument("--clean-recon",   action="store_true",
                    help="Mark plot title as clean recon (Hamming+zero-pad+ROI)")
parser.add_argument("--out",          default=None,
                    help="Output path (default: runs/t1_fit_vs_true/<subdir>/t1_fit_vs_true.png)")
args = parser.parse_args()

# Auto-populate annotation flags from config.json if present; explicit CLI flags win.
cfg = load_run_config(args.subdir)
if cfg:
    if args.noise_label is None:
        if cfg.get("noise_sigma_abs", 0.0) > 0:
            args.noise_label = str(cfg["noise_sigma_abs"])
    if not args.phase_sensitive and cfg.get("phase_sensitive", False):
        args.phase_sensitive = True
    if not args.clean_recon and cfg.get("clean_recon", False):
        args.clean_recon = True

julia = load_julia(args.subdir)

baselines = {}
if args.baselines:
    all_baselines = load_baselines(args.baselines)
    keys = args.baseline_keys or list(all_baselines.keys())
    baselines = {k: all_baselines[k] for k in keys if k in all_baselines}
    print(f"Loaded baselines: {list(baselines.keys())}")

policy_mape = None
if args.policy:
    policy_mape = load_policy(args.policy)
    print(f"Loaded policy '{args.policy_label}': mean MAPE = {np.mean(policy_mape):.1f}%")

fig = make_figure(julia, baselines, policy_mape, args.policy_label, args)

out = args.out or os.path.join(runs_t1, args.subdir, "t1_fit_vs_true.png")
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"Wrote {out}")

print()
print("  sphere   T1_true   MAPE(Julia)" +
      "".join(f"   MAPE({k})" for k in baselines) +
      (f"   MAPE({args.policy_label})" if policy_mape is not None else ""))
T1_true = julia["T1_true"]
for i in np.argsort(T1_true):
    row = f"  {i+1:6d}   {T1_true[i]:.4f}   {julia['mape'][i]:9.2f}%"
    for arr in baselines.values():
        row += f"   {arr[i]:9.2f}%"
    if policy_mape is not None and i < len(policy_mape):
        row += f"   {policy_mape[i]:9.2f}%"
    print(row)

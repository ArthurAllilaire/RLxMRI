"""Generate publication-quality plots for the FYP report.

Produces six PNGs in `report_plots/<TAG>/` from data already saved in
`runs/e2/*`. `<TAG>` defaults to `E2.1` (the current iteration); pass
`--tag E2.2` (or similar) when iterating so prior figures are not
overwritten. Each iteration's PNGs are committed alongside the report.

  1. mape_training_curve.png    Eval MAPE vs steps, E2 (neg_mape) vs E2.1 (delta_mape)
  2. ti_histogram_compare.png   TI histograms side-by-side
  3. per_sphere_mape.png        Per-sphere MAPE bars vs nominal T1, with TI modes
  4. information_landscape.png  |dS/dTI| heatmap over (T1, TI), with TI modes
  5. ir_signal_curves.png       IR magnitude curves per sphere with TI modes overlaid
  6. ep_length_compare.png      Episode-length histograms

All plots use a consistent style (figsize, fonts, palette) and are saved at
130 DPI which is sufficient for inclusion in an A4 report.

Run from repo root:
    python python/plots_for_report.py                 # writes report_plots/E2.1/
    python python/plots_for_report.py --tag E2.2      # writes report_plots/E2.2/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── style ───────────────────────────────────────────────────────────────────
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

REPO   = Path(__file__).resolve().parents[1]
OUT    = REPO / "report_plots"   # overwritten in main() with the tagged subdir

C_LEGACY = "#7f7f7f"   # E2 (neg_mape) — grey
C_DELTA  = "#1f77b4"   # E2.1 (delta_mape) — blue
C_TARGET = "#d62728"   # 5% MAPE target — red

# Nominal T1 values at 3T, in seconds (from src/materials/t1_array.jl, /1000)
T1_3T = np.array([
    1.838, 1.362, 0.998, 0.726, 0.509, 0.367, 0.259,
    0.185, 0.131, 0.091, 0.064, 0.046, 0.033, 0.023,
])

# TI modes observed in E2.1 diagnostic — empirical from raw TI lists.
# The policy clusters at the extremes of [0.01, 3.0]s with secondary mass
# in the 1–2.5s range; almost no mass in the informative middle (0.1–0.7s).
E2_1_TI_MODES = [0.010, 3.000]   # seconds

# ── helpers ──────────────────────────────────────────────────────────────────

def _load_eval_history(p):
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _load_eval_summary(p):
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _ir_signal(T1, TI):
    """Magnitude IR signal |1 − 2 exp(−TI/T1)|."""
    return np.abs(1 - 2 * np.exp(-TI / T1))


def _info_density(T1, TI):
    """|dS/dT1| — Fisher-information-style proxy for fit informativeness."""
    # d/dT1 [|1 − 2 e^(−TI/T1)|] = ±(2·TI/T1²) · e^(−TI/T1)
    inside = 1 - 2 * np.exp(-TI / T1)
    return (2 * TI / T1**2) * np.exp(-TI / T1) * np.sign(inside)


# ── plot 1 — MAPE training curves ────────────────────────────────────────────

def plot_mape_training_curve():
    legacy = _load_eval_history(REPO / "runs/e2/ppo_200k/eval_history.json")
    delta  = _load_eval_history(REPO / "runs/e2/e2_1_delta_100k/eval_history.json")
    if legacy is None or delta is None:
        print("[skip] mape_training_curve — missing eval_history.json")
        return

    fig, ax = plt.subplots(figsize=(7.5, 4.0))

    L = np.array([(d["step"], d["mape_pct"]) for d in legacy])
    D = np.array([(d["step"], d["mape_pct"]) for d in delta])

    ax.plot(L[:, 0] / 1000, L[:, 1], "o-", color=C_LEGACY,
            label="E2 (mean-MAPE reward, 5-dim action)", linewidth=1.6,
            markersize=5)
    ax.plot(D[:, 0] / 1000, D[:, 1], "s-", color=C_DELTA,
            label="E2.1 (delta-MAPE reward, 3-dim action)", linewidth=1.6,
            markersize=5)
    ax.axhline(5, color=C_TARGET, linestyle="--", linewidth=1.2,
               label="target (5%)")

    ax.set_yscale("log")
    ax.set_xlabel("Training steps (×1000)")
    ax.set_ylabel("Held-out eval MAPE [%]")
    ax.set_title("E2 vs E2.1 — eval MAPE during training (30 episodes per checkpoint)")
    ax.legend(loc="upper right", framealpha=0.95)
    ax.set_ylim(3, 2500)

    fig.tight_layout()
    fig.savefig(OUT / "mape_training_curve.png", dpi=130)
    plt.close(fig)
    print(f"[ok]   {OUT / 'mape_training_curve.png'}")


# ── plot 2 — TI histograms side-by-side ──────────────────────────────────────

def plot_ti_histogram_compare():
    L = _load_eval_summary(REPO / "runs/e2/ppo_200k/diagnostics/diagnose_summary.json")
    D = _load_eval_summary(REPO / "runs/e2/e2_1_delta_100k/diagnostics/diagnose_summary.json")
    if L is None or D is None or "all_ti_s" not in L or "all_ti_s" not in D:
        print("[skip] ti_histogram_compare — raw TIs missing; "
              "re-run diagnose_e2.py to populate")
        return

    legacy_ti = np.asarray(L["all_ti_s"])
    delta_ti  = np.asarray(D["all_ti_s"])

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), sharey=True)
    # Widen the leftmost bin so values exactly at TI_min=0.010s are visible
    # rather than collapsed into a sliver against the y-axis.
    bins = np.concatenate([[0.009],
                           np.logspace(np.log10(0.013), np.log10(3.0), 23)])

    w_legacy = np.ones_like(legacy_ti) / len(legacy_ti)
    w_delta  = np.ones_like(delta_ti)  / len(delta_ti)

    axes[0].hist(legacy_ti, bins=bins, weights=w_legacy, color=C_LEGACY,
                 edgecolor="black", linewidth=0.4)
    axes[0].set_title(f"E2 (mean-MAPE, 200k) — n={len(legacy_ti)} blocks\n"
                       f"{L['ti_modal_bin_share']:.0%} mass at TI_min "
                       "— policy collapsed")
    axes[1].hist(delta_ti, bins=bins, weights=w_delta, color=C_DELTA,
                 edgecolor="black", linewidth=0.4)
    axes[1].set_title(f"E2.1 (delta-MAPE, 100k) — n={len(delta_ti)} blocks\n"
                       f"modal bin {D['ti_modal_bin_share']:.0%} — "
                       "spread restored, peaks at TI extremes")

    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("TI [s]")
        ax.set_xlim(0.0085, 3.5)
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda y, _: f"{y*100:.0f}%"))
    axes[0].set_ylabel("Fraction of blocks across 30 eval episodes")

    fig.suptitle("How the policy distributes its TI choices",
                 y=1.02, fontsize=11.5)
    fig.tight_layout()
    fig.savefig(OUT / "ti_histogram_compare.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok]   {OUT / 'ti_histogram_compare.png'}")


# ── plot 3 — per-sphere MAPE bars ────────────────────────────────────────────

def plot_per_sphere_mape():
    summ = _load_eval_summary(REPO / "runs/e2/e2_1_delta_100k/eval_summary.json")
    if summ is None or "per_sphere" not in summ:
        print("[skip] per_sphere_mape — missing eval_summary.json")
        return

    per_sphere = np.array(summ["per_sphere"])
    n = len(per_sphere)
    sphere_idx = np.arange(1, n + 1)

    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    bars = ax.bar(sphere_idx, per_sphere, color=C_DELTA,
                  edgecolor="black", linewidth=0.5)

    ax.set_xticks(sphere_idx)
    # Secondary x-axis: nominal T1 in ms
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(sphere_idx)
    ax2.set_xticklabels([f"{int(t*1000)}" for t in T1_3T],
                        rotation=45, fontsize=8)
    ax2.set_xlabel("Nominal T1 at 3 T [ms]", fontsize=9)

    ax.set_xlabel("Sphere index (1 = longest T1, 14 = shortest)")
    ax.set_ylabel("MAPE [%]")
    ax.set_title("E2.1 per-sphere MAPE — policy concentrates TIs at the\n"
                 "extremes, leaving mid-T1 spheres badly fit")

    # Highlight the spheres in the middle of the T1 range — these are the
    # ones the policy ignores by clustering at TI extremes.
    for i, mape in enumerate(per_sphere):
        if mape > 150:  # threshold for "catastrophically misfit"
            ax.text(i + 1, mape + 10, f"{mape:.0f}%", ha="center",
                     fontsize=8, color="darkred", fontweight="bold")

    ax.axhline(5, color=C_TARGET, linestyle="--", linewidth=1.0,
               label="5% target")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_ylim(0, max(per_sphere) * 1.25)

    fig.tight_layout()
    fig.savefig(OUT / "per_sphere_mape.png", dpi=130)
    plt.close(fig)
    print(f"[ok]   {OUT / 'per_sphere_mape.png'}")


# ── plot 4 — information landscape ───────────────────────────────────────────

def plot_information_landscape():
    """Heatmap of |dS/dT1| over (T1, TI). Bright = informative TI for that T1."""
    T1_grid = np.logspace(np.log10(0.02), np.log10(2.0), 200)
    TI_grid = np.logspace(np.log10(0.01), np.log10(3.0), 200)
    T1m, TIm = np.meshgrid(T1_grid, TI_grid)
    info = np.abs(_info_density(T1m, TIm))

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    im = ax.pcolormesh(T1_grid, TI_grid, info,
                        shading="auto", cmap="magma",
                        vmin=0, vmax=np.percentile(info, 99))

    # Overlay the IR null line: TI = T1·ln 2 (where information peaks)
    null_line = T1_grid * np.log(2)
    ax.plot(T1_grid, null_line, "--", color="white", linewidth=1.2,
            label="IR null:  TI = T1·ln(2)")

    # Overlay the 14 sphere T1s
    ax.scatter(T1_3T, [0.012]*len(T1_3T), marker="v", color="cyan",
               s=40, edgecolor="black", linewidth=0.5, zorder=5,
               label="14 sphere T1s")

    # Overlay E2.1's two TI modes as horizontal bands
    for ti in E2_1_TI_MODES:
        ax.axhline(ti, color="lime", linestyle="-", linewidth=1.5, alpha=0.85)
    ax.axhline(E2_1_TI_MODES[0], color="lime", linestyle="-", linewidth=1.5,
               alpha=0.85, label=f"E2.1 TI modes (TI_min, TI_max)")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Sphere T1 [s]")
    ax.set_ylabel("TI [s]")
    ax.set_title("Information landscape — |∂S/∂T1| over (T1, TI)\n"
                 "Bright ridge = informative TI for that T1.  "
                 "E2.1's two TIs hit only the ends of the ridge.")
    cbar = fig.colorbar(im, ax=ax, label="|∂S/∂T1|  (a.u.)")
    ax.legend(loc="lower right", framealpha=0.92, fontsize=8.5)

    fig.tight_layout()
    fig.savefig(OUT / "information_landscape.png", dpi=130)
    plt.close(fig)
    print(f"[ok]   {OUT / 'information_landscape.png'}")


# ── plot 5 — IR signal curves per sphere ─────────────────────────────────────

def plot_ir_signal_curves():
    """For each sphere, plot |S(TI)| over TI, mark the agent's two TI modes."""
    TI_grid = np.logspace(np.log10(0.005), np.log10(4.0), 400)

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    cmap = plt.get_cmap("viridis")
    for i, T1 in enumerate(T1_3T):
        s = _ir_signal(T1, TI_grid)
        ax.plot(TI_grid, s, color=cmap(i / (len(T1_3T) - 1)),
                linewidth=1.0, alpha=0.85)

    for ti in E2_1_TI_MODES:
        ax.axvline(ti, color="red", linestyle="--", linewidth=1.4, alpha=0.8)
    ax.axvline(E2_1_TI_MODES[0], color="red", linestyle="--", linewidth=1.4,
               alpha=0.8, label=f"E2.1 TI modes ({E2_1_TI_MODES[0]:.2f}, "
                                 f"{E2_1_TI_MODES[1]:.2f} s)")

    ax.set_xscale("log")
    ax.set_xlabel("TI [s]")
    ax.set_ylabel("|S(TI)| / S0")
    ax.set_title("IR magnitude curves for the 14 spheres\n"
                 "(viridis: dark = sphere 1 longest T1, light = sphere 14 shortest)")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left", fontsize=9)

    # Annotate where each sphere's null falls
    for T1 in T1_3T:
        ax.scatter(T1 * np.log(2), 0, marker="|", color="black", s=80, zorder=5)
    ax.text(0.012, 0.05, "↑ each black tick = a sphere's null", fontsize=8.5,
            color="black")

    fig.tight_layout()
    fig.savefig(OUT / "ir_signal_curves.png", dpi=130)
    plt.close(fig)
    print(f"[ok]   {OUT / 'ir_signal_curves.png'}")


# ── plot 6 — episode length histogram ────────────────────────────────────────

def plot_ep_length_compare():
    L = _load_eval_summary(REPO / "runs/e2/ppo_200k/diagnostics/diagnose_summary.json")
    D = _load_eval_summary(REPO / "runs/e2/e2_1_delta_100k/diagnostics/diagnose_summary.json")
    if L is None or D is None:
        print("[skip] ep_length_compare — missing diagnose_summary.json")
        return

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    labels = ["E2\n(mean-MAPE)", "E2.1\n(delta-MAPE)"]
    means  = [L["ep_len_mean"], D["ep_len_mean"]]
    stds   = [L.get("ep_len_std", 0.0), D.get("ep_len_std", 0.0)]
    bars = ax.bar(labels, means, yerr=stds, capsize=6,
                   color=[C_LEGACY, C_DELTA], edgecolor="black", linewidth=0.5)
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width()/2, m + 0.25, f"{m:.2f}",
                ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Mean blocks per episode")
    ax.set_title("E2 vs E2.1 — episode length\n"
                 "Longer episodes → agent is genuinely exploring")
    ax.set_ylim(0, max(means) * 1.5)
    fig.tight_layout()
    fig.savefig(OUT / "ep_length_compare.png", dpi=130)
    plt.close(fig)
    print(f"[ok]   {OUT / 'ep_length_compare.png'}")


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="E2.1",
                        help="subfolder under report_plots/ to write into "
                             "(default: E2.1; use E2.2 etc. for new iterations)")
    args = parser.parse_args()

    OUT = REPO / "report_plots" / args.tag
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Writing figures to: {OUT}\n")

    plot_mape_training_curve()
    plot_ti_histogram_compare()
    plot_per_sphere_mape()
    plot_information_landscape()
    plot_ir_signal_curves()
    plot_ep_length_compare()
    print(f"\nAll plots in: {OUT}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Memory-mechanism ablation figures (5-sphere, 560 s) for the E2 chapter.

Reads the strict held-out (seed 600000) global-best eval summaries for the four
memory arms and the held-out fixed-schedule baselines, and emits two figures
matching the existing Run B style:

  figs/mf_runB_5sphere_memory_mape.png      overall held-out MAPE + 95% CI,
                                            with fixed-schedule reference lines
  figs/mf_runB_5sphere_memory_per_pool.png  per-active-pool MAPE, four arms

Run from the repo root:
    python report/e2_runs/make_memory_ablation_figs.py
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs" / "e2"
OUT = Path(__file__).resolve().parent / "figs"
OUT.mkdir(parents=True, exist_ok=True)

# Arms: label, run dir, bar colour. Ordered best -> worst (the narrative order).
ARMS = [
    ("TI histogram\n(R1)",   "mf_runB_5sphere_hist_560s_gpu",  "#4c9f70"),
    ("No memory\n(control)", "mf_runB_5sphere_560s_gpu",        "#4878cf"),
    ("σ-channel",            "mf_runB_5sphere_sigma_560s_gpu",  "#b07aa1"),
    ("LSTM\n(R2)",           "mf_runB_5sphere_lstm_560s_gpu",   "#d1654f"),
]
POOLS = ["1", "3", "6", "8", "14"]

# Held-out (seed 600000) fixed-schedule baselines, runs/e2/baselines_heldout_560s.
BASELINES = [
    ("fixed log-grid (log_grid_trmatched)", 6.04, "#555555", "--"),
    ("CR-optimal",                          7.74, "#999999", ":"),
]


def load(run_dir: str) -> dict:
    p = RUNS / run_dir / "global_best" / "eval_summary.json"
    with p.open() as f:
        return json.load(f)


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color="0.85", linewidth=0.8)
    ax.set_axisbelow(True)


def fig_overall(data):
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    xs = range(len(ARMS))
    for i, (label, _run, colour) in enumerate(ARMS):
        d = data[label]
        mape = d["agent_mape_pct"]
        lo, hi = d["agent_mape_ci95_pct"]
        ax.bar(i, mape, width=0.62, color=colour,
               edgecolor="white", linewidth=0.5, zorder=3)
        ax.errorbar(i, mape, yerr=[[mape - lo], [hi - mape]], fmt="none",
                    ecolor="0.2", elinewidth=1.4, capsize=4, zorder=4)
        ax.text(i, hi + 0.18, f"{mape:.2f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=colour, zorder=5)

    for blabel, bval, bcol, bstyle in BASELINES:
        ax.axhline(bval, color=bcol, linestyle=bstyle, linewidth=1.5, zorder=2,
                   label=f"{blabel}: {bval:.2f}%")

    ax.set_xticks(list(xs))
    ax.set_xticklabels([a[0] for a in ARMS])
    ax.set_ylabel("Held-out mean MAPE (%)")
    ax.set_ylim(0, max(9, max(data[a[0]]["agent_mape_ci95_pct"][1]
                              for a in ARMS) + 1.2))
    ax.set_title("Memory mechanism vs MAPE — 5-sphere, 560 s, strict held-out "
                 "(seed 600000)", fontsize=11)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    style_axes(ax)
    fig.tight_layout()
    out = OUT / "mf_runB_5sphere_memory_mape.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_per_pool(data):
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    n = len(ARMS)
    width = 0.8 / n
    for i, (label, _run, colour) in enumerate(ARMS):
        d = data[label]
        vals = [d["per_pool"][p][0] for p in POOLS]
        xs = [j + (i - (n - 1) / 2) * width for j in range(len(POOLS))]
        ax.bar(xs, vals, width=width, color=colour, label=label.replace("\n", " "),
               edgecolor="white", linewidth=0.4, zorder=3)

    ax.set_xticks(range(len(POOLS)))
    ax.set_xticklabels([f"Pool {p}" for p in POOLS])
    ax.set_ylabel("Per-active-pool MAPE (%)")
    ax.set_title("Per-pool MAPE by memory mechanism — 5-sphere, 560 s, strict "
                 "held-out", fontsize=11)
    ax.legend(loc="upper left", frameon=False, fontsize=9, ncol=2)
    style_axes(ax)
    fig.tight_layout()
    out = OUT / "mf_runB_5sphere_memory_per_pool.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main():
    data = {label: load(run) for label, run, _ in ARMS}
    o1 = fig_overall(data)
    o2 = fig_per_pool(data)
    for label, _run, _c in ARMS:
        d = data[label]
        print(f"  {label.replace(chr(10), ' '):22} "
              f"MAPE={d['agent_mape_pct']:5.2f}%  "
              f"CI={[round(x, 2) for x in d['agent_mape_ci95_pct']]}")
    print(f"Wrote {o1.relative_to(ROOT)}")
    print(f"Wrote {o2.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

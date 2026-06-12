"""Regenerate the two Run B result figures (5-sphere) for the E2 chapter, using
STRICT held-out (seed 600000) data throughout. The earlier figures used the
seed-500000 selection stream for the 240 s bars, which was optimistic and
inconsistent with the held-out 560 s bars; this version is held-out everywhere.

Outputs (overwrites the chapter images):
  report_latex/imgs/e2_rl/runB_mape_comparison.png
  report_latex/imgs/e2_rl/runB_per_pool.png

All numbers are strict seed-600000 24-episode evals:
  240 s no-sigma : runs/e2/mf_runB_cached3_cached_full3_full_gpu (eval_globalbest_24ep_heldout.log,
                    baselines_heldout/baseline_summary.json)
  560 s no-sigma : runs/e2/mf_runB_5sphere_560s_gpu  (eval_globalbest_24ep_heldout.log)
  560 s +sigma   : runs/e2/mf_runB_5sphere_sigma_560s_gpu
  fixed log-grid : 6.86%% (240 s) / 6.04%% (560 s), held-out baseline harness.
"""
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[2] / "report_latex" / "imgs" / "e2_rl"

RL_COLOR = "#4878cf"
BASE_COLOR = "#c44e52"
CR_COLOR = "#8c8c8c"

# --- left panel: overall MAPE, RL vs fixed log-grid vs CR-optimal, held-out ---
# (label, RL mean, RL ci, log-grid mean, log-grid ci, CR mean, CR ci)
GROUPS = [
    ("RL 240s\nno sigma", 4.62, (4.06, 5.19), 6.86, (5.83, 7.90), 15.96, (12.26, 19.81)),
    ("RL 560s\nno sigma", 4.16, (3.41, 4.99), 6.04, (5.36, 6.70), 7.74, (7.08, 8.42)),
    ("RL 560s\n+ sigma", 5.25, (4.32, 6.22), 6.04, (5.36, 6.70), 7.74, (7.08, 8.42)),
]


def ci_err(mean, ci):
    return [[mean - ci[0]], [ci[1] - mean]]


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color="0.85", linewidth=0.8)
    ax.set_axisbelow(True)


def fig_overall():
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(GROUPS))
    w = 0.27
    for i, (_, rl, rlci, base, baseci, cr, crci) in enumerate(GROUPS):
        for off, val, vci, col, lab in [
            (-w, rl, rlci, RL_COLOR, "RL global-best"),
            (0.0, base, baseci, BASE_COLOR, "fixed log-grid"),
            (w, cr, crci, CR_COLOR, "CR-optimal"),
        ]:
            ax.bar(x[i] + off, val, w, color=col,
                   yerr=ci_err(val, vci), capsize=3, ecolor="black",
                   label=lab if i == 0 else None)
            ax.text(x[i] + off, vci[1] + 0.3, f"{val:.1f}", ha="center",
                    va="bottom", fontsize=8, color="0.25")
    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in GROUPS])
    ax.set_ylabel("Mean MAPE (%)")
    ax.set_ylim(0, 22.0)
    ax.legend(loc="upper right", frameon=False)
    style_axes(ax)
    fig.tight_layout()
    out = OUT / "runB_mape_comparison.png"
    fig.savefig(out, dpi=150)
    print("wrote", out)


# --- right panel: per-active-pool MAPE, three RL policies, held-out -----------
POOLS = ["1", "3", "6", "8", "14"]
PER_POOL = {
    "RL 240s no sigma": ([5.27, 4.58, 4.36, 4.01, 4.91], "#4878cf"),
    "RL 560s no sigma": ([4.75, 4.21, 3.88, 2.90, 5.06], "#4c9f70"),
    "RL 560s + sigma": ([4.03, 4.98, 6.20, 5.92, 5.12], "#b07aa1"),
}


def fig_per_pool():
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(POOLS))
    n = len(PER_POOL)
    w = 0.8 / n
    for j, (label, (vals, color)) in enumerate(PER_POOL.items()):
        ax.bar(x + (j - (n - 1) / 2) * w, vals, w, color=color, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Pool {p}" for p in POOLS])
    ax.set_ylabel("Per-active-pool MAPE (%)")
    ax.set_ylim(0, 7.0)
    ax.legend(loc="upper left", frameon=False)
    style_axes(ax)
    fig.tight_layout()
    out = OUT / "runB_per_pool.png"
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    fig_overall()
    fig_per_pool()

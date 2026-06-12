"""Error-vs-realised-T1 figure for the memory ablation (5-sphere, 560 s,
strict held-out). Bins the per-episode (T1, abs-pct-error) pairs into three
clinical T1 bands and plots mean MAPE per band for each memory arm.

This is the meaningful replacement for the per-pool-index breakdown: under the
continuous-T1 sampler each slot is an i.i.d. draw, so error-vs-T1 (not
error-vs-slot) is what carries information.

Output: report_latex/imgs/e2_rl/runB_memory_t1bands.png
Reads:  runs/e2/<arm>/global_best/eval_t1err_heldout.json
"""
from pathlib import Path
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report_latex" / "imgs" / "e2_rl" / "runB_memory_t1bands.png"

# (label, run dir, colour) — same palette as the memory MAPE figure.
ARMS = [
    ("TI histogram (R1)", "mf_runB_5sphere_hist_560s_gpu", "#4c9f70"),
    ("no memory",         "mf_runB_5sphere_560s_gpu",      "#4878cf"),
    ("$\\sigma$-channel", "mf_runB_5sphere_sigma_560s_gpu", "#b07aa1"),
    ("LSTM (R2)",         "mf_runB_5sphere_lstm_560s_gpu",  "#d1654f"),
]
EDGES = [0.04, 0.5, 1.2, 1.9]
BAND_LABELS = ["short\n0.04–0.5 s", "mid\n0.5–1.2 s", "long\n1.2–1.9 s"]


def band_means(run_dir):
    p = ROOT / "runs" / "e2" / run_dir / "global_best" / "eval_t1err_heldout.json"
    pe = json.load(p.open())["per_episode_t1_err"]
    t1 = np.array(pe["t1_true_s"]); ape = np.array(pe["ape"]) * 100
    out = []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        m = (t1 >= lo) & (t1 < hi if hi != EDGES[-1] else t1 <= hi)
        out.append(float(ape[m].mean()))
    return out


def main():
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(BAND_LABELS))
    n = len(ARMS)
    w = 0.8 / n
    for j, (label, run, col) in enumerate(ARMS):
        vals = band_means(run)
        ax.bar(x + (j - (n - 1) / 2) * w, vals, w, color=col, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(BAND_LABELS)
    ax.set_xlabel("Realised $T_1$ band")
    ax.set_ylabel("Mean MAPE (%)")
    ax.set_ylim(0, 9.5)
    ax.legend(loc="upper left", frameon=False, ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color="0.85", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print("wrote", OUT)


if __name__ == "__main__":
    main()

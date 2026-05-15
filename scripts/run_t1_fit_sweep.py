"""scripts/run_t1_fit_sweep.py — parameter sweep over budget × Npe × SNR.

Fixed: Nfe=128, clean=false.
Sweep: budget ∈ {80,120,160,240}s  ×  Npe ∈ {8,16,32}  ×  SNR ∈ {0, 2.5}
Total: 24 runs.

After all simulations finish, writes a 2-panel comparison figure:
  runs/t1_fit_vs_true/sweep_comparison.png
  Left panel : SNR=0    — MAPE vs budget, one line per Npe
  Right panel: SNR=2.5  — same

Usage:
    python scripts/run_t1_fit_sweep.py
    python scripts/run_t1_fit_sweep.py --dry-run        # print commands only
    python scripts/run_t1_fit_sweep.py --plot-only       # skip sims, just replot
"""

from __future__ import annotations
import argparse
import csv
import itertools
import subprocess
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT   = Path(__file__).resolve().parent.parent
RUNS   = ROOT / "scripts" / "runs" / "t1_fit_vs_true"
JULIA  = ROOT / "scripts" / "t1_fit_vs_true.jl"
PYPLOT = ROOT / "scripts" / "t1_fit_vs_true.py"

# ── Sweep parameters ──────────────────────────────────────────────────────────

BUDGETS = [80, 120, 160, 240]
NPES    = [8, 16, 32]
SNRS    = [0, 2.5]
NFE     = 128   # fixed

# ── Label logic (mirrors t1_fit_vs_true.jl) ──────────────────────────────────

def noise_tag(snr: float) -> str:
    if snr <= 0:
        return "nonoise"
    s = str(snr)
    return "snr" + s.replace(".", "p")

def run_label(budget: int, npe: int, snr: float) -> str:
    return f"b{budget}s_{noise_tag(snr)}_npe{npe}fe{NFE}"

# ── Per-run helpers ───────────────────────────────────────────────────────────

def julia_cmd(budget: int, npe: int, snr: float) -> list[str]:
    cmd = [
        "julia", "--project=.", str(JULIA),
        "--budget", str(budget),
        "--npe", str(npe),
        "--nfe", str(NFE),
    ]
    cmd += ["--snr", str(snr)]   # always explicit; Julia default is 2.5, not 0
    return cmd

def python_cmd(budget: int, npe: int, snr: float) -> list[str]:
    cmd = [sys.executable, str(PYPLOT), "--subdir", run_label(budget, npe, snr)]
    if snr > 0:
        cmd += ["--noise-label", f"SNR={snr:g}"]
    return cmd

def csv_path(budget: int, npe: int, snr: float) -> Path:
    return RUNS / run_label(budget, npe, snr) / "t1_fit_vs_true.csv"

# ── Comparison plot ───────────────────────────────────────────────────────────

def load_mape_stats(budget: int, npe: int, snr: float) -> tuple[float, float] | None:
    p = csv_path(budget, npe, snr)
    if not p.exists():
        return None
    mapes = []
    with open(p) as f:
        for row in csv.DictReader(f):
            mapes.append(float(row["mape_pct"]))
    if not mapes:
        return None
    return float(np.mean(mapes)), float(np.median(mapes))

def make_comparison_figure(no_mean: bool = False) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    colours = [matplotlib.colormaps["tab10"](i) for i in range(len(NPES))]

    for ax, snr in zip(axes, SNRS):
        for colour, npe in zip(colours, NPES):
            stats = [load_mape_stats(b, npe, snr) for b in BUDGETS]
            valid  = [(b, s) for b, s in zip(BUDGETS, stats) if s is not None]
            if not valid:
                continue
            xs = [b for b, _ in valid]
            means   = [s[0] for _, s in valid]
            medians = [s[1] for _, s in valid]
            if not no_mean:
                ax.plot(xs, means,   marker="o", linestyle="--", color=colour,
                        label=f"Npe={npe} mean",   alpha=0.6)
            ax.plot(xs, medians, marker="o", linestyle="-",  color=colour,
                    label=f"Npe={npe}" + ("" if no_mean else " median"))

        label = "no noise" if snr == 0 else f"SNR={snr:g}"
        ax.set_title(label)
        ax.set_xlabel("Budget (s)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("MAPE (%)")
    fig.suptitle("T1 fit accuracy vs scan budget", fontweight="bold")
    fig.tight_layout()

    out = RUNS / "sweep_comparison.png"
    RUNS.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out

# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run",   action="store_true", help="Print commands without running")
    p.add_argument("--plot-only", action="store_true", help="Skip simulations, regenerate plot only")
    p.add_argument("--no-mean",   action="store_true", help="Omit mean lines from comparison plot")
    args = p.parse_args()

    combos = list(itertools.product(BUDGETS, NPES, SNRS))
    total  = len(combos)

    if not args.plot_only:
        for i, (budget, npe, snr) in enumerate(combos, 1):
            print(f"\n{'═'*56}")
            print(f"  [{i}/{total}]  budget={budget}s  Npe={npe}  SNR={snr:g}")
            print(f"{'═'*56}")

            if csv_path(budget, npe, snr).exists():
                print(f"  ↳ CSV exists — skipping simulation")
            else:
                jcmd = julia_cmd(budget, npe, snr)
                print(f"  $ {' '.join(jcmd)}")
                if not args.dry_run:
                    subprocess.run(jcmd, cwd=ROOT, check=True)

            pcmd = python_cmd(budget, npe, snr)
            print(f"  $ {' '.join(pcmd)}")
            if not args.dry_run:
                subprocess.run(pcmd, cwd=ROOT, check=True)

    print(f"\n{'─'*56}")
    print("  Generating comparison figure…")
    if not args.dry_run:
        out = make_comparison_figure(no_mean=args.no_mean)
        print(f"  → {out.relative_to(ROOT)}")
    else:
        print(f"  → runs/t1_fit_vs_true/sweep_comparison.png")

    print("\nDone.")


if __name__ == "__main__":
    main()

"""scripts/run_t1_fit_sweep.py — parameter sweep over budget × Npe × σ × water.

Fixed: Nfe=128, magnitude recon.
Sweep: budget ∈ {80,120,160,240,manual}  ×  Npe ∈ {32,64}  ×  {dry,water}
       σ (absolute k-space noise) ∈ {0,600} dry  /  {0,100} water.

The Julia script (t1_fit_vs_true.jl) now owns the σ loop: it simulates each
schedule's noise-free k-space ONCE and reuses it across all σ in --sigmas, then
auto-renders the per-σ figures (t1_fit_vs_true.py + plot_recovery_curves_koma.py)
itself. So this orchestrator issues just ONE Julia call per (budget, Npe, water)
— 5×2×2 = 20 runs, 40 fitted σ-dirs — and afterwards stitches a comparison grid:

  runs/t1_fit_vs_true/sweep_comparison.png
  2×2 panels: rows = {dry, water}, cols = {σ=0, σ=noisy}; one line per Npe,
  x = budget; the manual schedule is drawn as a star to the right.

Usage:
    PYTHON=.venv/bin/python python scripts/run_t1_fit_sweep.py
    python scripts/run_t1_fit_sweep.py --dry-run        # print commands only
    python scripts/run_t1_fit_sweep.py --plot-only       # skip sims, just replot
"""

from __future__ import annotations
import argparse
import csv
import itertools
import subprocess
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT   = Path(__file__).resolve().parent.parent
RUNS   = ROOT / "scripts" / "runs" / "t1_fit_vs_true"
JULIA  = ROOT / "scripts" / "t1_fit_vs_true.jl"

# ── Sweep parameters ──────────────────────────────────────────────────────────

NUMERIC_BUDGETS = [80, 120, 160, 240]
BUDGETS         = NUMERIC_BUDGETS + ["manual"]
NPES            = [32, 64]
NFE             = 128   # fixed
WATERS          = [False, True]
# σ list per water setting. Water collapses far earlier (see
# runs/snr_sweep_voxel_1mm_water), so it gets a gentler noisy point.
SIGMAS          = {False: [0, 600], True: [0, 100]}
NOISY_SIGMA     = {False: 600, True: 100}
MANUAL_X        = max(NUMERIC_BUDGETS) * 1.18   # x-position for the manual star

# ── Label logic (mirrors t1_fit_vs_true.jl) ──────────────────────────────────

def sigma_label(sigma: float) -> str:
    """Integer σ stays clean (600 → '600'); else replace dot (0.3 → '0p3')."""
    if float(sigma).is_integer():
        return str(int(sigma))
    return f"{sigma:g}".replace(".", "p")

def noise_tag(sigma: float) -> str:
    return "nonoise" if sigma <= 0 else f"noise{sigma_label(sigma)}"

def run_label(budget, npe: int, sigma: float, water: bool) -> str:
    budget_tag = "bMANUAL" if budget == "manual" else f"b{budget}s"
    grid_tag   = "" if (npe == 32 and NFE == 64) else f"_npe{npe}fe{NFE}"
    water_tag  = "_water" if water else ""
    return f"{budget_tag}_{noise_tag(sigma)}{grid_tag}{water_tag}"

# ── Per-run helpers ───────────────────────────────────────────────────────────

def julia_cmd(budget, npe: int, water: bool) -> list[str]:
    cmd = ["julia", "--project=.", str(JULIA), "--npe", str(npe), "--nfe", str(NFE)]
    if budget == "manual":
        cmd += ["--manual"]
    else:
        cmd += ["--budget", str(budget)]
    cmd += ["--sigmas", ",".join(str(s) for s in SIGMAS[water])]
    if water:
        cmd += ["--water"]
    return cmd

def csv_path(budget, npe: int, sigma: float, water: bool) -> Path:
    return RUNS / run_label(budget, npe, sigma, water) / "t1_fit_vs_true.csv"

def all_csvs_exist(budget, npe: int, water: bool) -> bool:
    return all(csv_path(budget, npe, s, water).exists() for s in SIGMAS[water])

# ── Comparison plot ───────────────────────────────────────────────────────────

def load_mape_stats(budget, npe: int, sigma: float, water: bool):
    p = csv_path(budget, npe, sigma, water)
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
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    colours = [matplotlib.colormaps["tab10"](i) for i in range(len(NPES))]

    for row, water in enumerate(WATERS):
        for col, sigma in enumerate([0, NOISY_SIGMA[water]]):
            ax = axes[row, col]
            for colour, npe in zip(colours, NPES):
                stats = [load_mape_stats(b, npe, sigma, water) for b in NUMERIC_BUDGETS]
                valid = [(b, s) for b, s in zip(NUMERIC_BUDGETS, stats) if s is not None]
                if valid:
                    xs      = [b for b, _ in valid]
                    means   = [s[0] for _, s in valid]
                    medians = [s[1] for _, s in valid]
                    if not no_mean:
                        ax.plot(xs, means, marker="o", linestyle="--", color=colour,
                                label=f"Npe={npe} mean", alpha=0.6)
                    ax.plot(xs, medians, marker="o", linestyle="-", color=colour,
                            label=f"Npe={npe}" + ("" if no_mean else " median"))
                # Manual schedule as a star to the right.
                m = load_mape_stats("manual", npe, sigma, water)
                if m is not None:
                    ax.scatter([MANUAL_X], [m[1]], marker="*", s=140, color=colour,
                               edgecolor="k", linewidth=0.4, zorder=5)

            ax.set_title(f"{'water' if water else 'dry'},  σ={sigma}"
                         + ("  (noise-free)" if sigma == 0 else ""))
            ax.grid(True, alpha=0.3)
            if row == 0 and col == 0:
                ax.legend(fontsize=7)

    for ax in axes[-1, :]:
        ax.set_xlabel("Budget (s)   [★ = manual]")
    for ax in axes[:, 0]:
        ax.set_ylabel("MAPE (%)")
    fig.suptitle("T1 fit accuracy vs scan budget (dry vs water, noise-free vs noisy)",
                 fontweight="bold")
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

    combos = list(itertools.product(BUDGETS, NPES, WATERS))
    total  = len(combos)

    if not args.plot_only:
        for i, (budget, npe, water) in enumerate(combos, 1):
            print(f"\n{'═'*56}")
            print(f"  [{i}/{total}]  budget={budget}  Npe={npe}  water={water}  "
                  f"σ={SIGMAS[water]}")
            print(f"{'═'*56}")

            if all_csvs_exist(budget, npe, water):
                print("  ↳ all σ CSVs exist — skipping")
                continue
            jcmd = julia_cmd(budget, npe, water)
            print(f"  $ {' '.join(jcmd)}")
            if not args.dry_run:
                subprocess.run(jcmd, cwd=ROOT, check=True)

    print(f"\n{'─'*56}")
    print("  Generating comparison figure…")
    if not args.dry_run:
        out = make_comparison_figure(no_mean=args.no_mean)
        print(f"  → {out.relative_to(ROOT)}")
    else:
        print("  → runs/t1_fit_vs_true/sweep_comparison.png")

    print("\nDone.")


if __name__ == "__main__":
    main()

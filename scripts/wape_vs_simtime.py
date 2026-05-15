#!/usr/bin/env python
"""Plot WAPE / Pearson r / energy-on-support vs total simulated time.

Run scripts/wape_vs_simtime.jl first to produce accurate/wape_vs_simtime.txt.

Usage:
    python wape_vs_simtime.py                    # accurate (default)
    python wape_vs_simtime.py --mode inaccurate
    python wape_vs_simtime.py --mode both        # overlay on same axes
"""
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))


def load_data(subdir):
    data = np.loadtxt(os.path.join(here, subdir, "wape_vs_simtime.txt"))
    return data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4]


def plot_series(ax_w, ax_r, TR, sim_time, wape, pearson, e_supp,
                color_w, color_p, color_e, label_suffix=""):
    ax_w.plot(sim_time, wape, "o-", color=color_w, linewidth=2,
              markersize=7, label=f"WAPE{label_suffix}")
    for t, w, tr in zip(sim_time, wape, TR):
        if tr in (4.0, 8.0, 20.0):
            ax_w.annotate(f"TR={tr:g}s\nWAPE={w:.0f}%",
                          xy=(t, w), xytext=(8, 8),
                          textcoords="offset points", fontsize=9,
                          arrowprops=dict(arrowstyle="-", color="grey", lw=0.5))
    ax_r.plot(sim_time, pearson, "s-", color=color_p, linewidth=2,
              markersize=6, label=f"Pearson r{label_suffix}")
    ax_r.plot(sim_time, e_supp / 100, "^-", color=color_e, linewidth=2,
              markersize=6, label=f"energy-on-support{label_suffix}")


parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["accurate", "inaccurate", "both"],
                    default="accurate")
args = parser.parse_args()

fig, (ax_w, ax_r) = plt.subplots(1, 2, figsize=(12, 4.6))

if args.mode == "both":
    TR_a, st_a, wape_a, pear_a, esupp_a = load_data("accurate")
    TR_i, st_i, wape_i, pear_i, esupp_i = load_data("inaccurate")
    xmax = max(st_a.max(), st_i.max()) + 20

    plot_series(ax_w, ax_r, TR_a, st_a, wape_a, pear_a, esupp_a,
                color_w="steelblue", color_p="steelblue", color_e="mediumseagreen",
                label_suffix=" (fixed)")
    plot_series(ax_w, ax_r, TR_i, st_i, wape_i, pear_i, esupp_i,
                color_w="firebrick", color_p="darkorange", color_e="darkgreen",
                label_suffix=" (buggy)")

    ax_w.axvspan(0, 70, color="green", alpha=0.08, label="safe zone (≤ 70 s)")
    ax_w.axvspan(70, xmax, color="red", alpha=0.05, label="KomaMRI drift zone")
    ax_r.axvspan(0, 70, color="green", alpha=0.08)
    ax_r.axvspan(70, xmax, color="red", alpha=0.05)

    fig.suptitle(
        "KomaMRI bug fix: WAPE/accuracy vs sim time  (blue = fixed, red = buggy)",
        fontsize=12, y=1.02,
    )
    out = os.path.join(here, "wape_vs_simtime_comparison.png")

else:
    TR, sim_time, wape, pearson, e_supp = load_data(args.mode)
    xmax = sim_time.max() + 20

    plot_series(ax_w, ax_r, TR, sim_time, wape, pearson, e_supp,
                color_w="firebrick" if args.mode == "inaccurate" else "steelblue",
                color_p="steelblue",
                color_e="darkgreen")

    ax_w.axvspan(0, 70, color="green", alpha=0.08, label="safe zone (≤ 70 s)")
    ax_w.axvspan(70, xmax, color="red", alpha=0.05, label="KomaMRI drift zone")
    ax_r.axvspan(0, 70, color="green", alpha=0.08)
    ax_r.axvspan(70, xmax, color="red", alpha=0.05)

    fig.suptitle(
        f"KomaMRI multi-shot recon: accuracy collapses past ~70 s of sim time  [{args.mode}]",
        fontsize=12, y=1.02,
    )
    out = os.path.join(here, args.mode, "wape_vs_simtime.png")

    print()
    print("  TR [s]   sim_time [s]   WAPE [%]   Pearson r   on-support [%]")
    print("  " + "─" * 64)
    data = np.column_stack([TR, sim_time, wape, pearson, e_supp])
    for row in data:
        print(f"  {row[0]:6.1f}   {row[1]:11.1f}   {row[2]:7.1f}   {row[3]:9.3f}   {row[4]:13.1f}")

ax_w.set_xlabel("total simulated time = TR × Npe  [s]")
ax_w.set_ylabel("WAPE  [%]")
ax_w.set_title("Recon accuracy vs sim time (TI = 3 s, Npe = 16)")
ax_w.set_ylim(0)
ax_w.set_xlim(0, xmax)
ax_w.legend(loc="lower right", fontsize=9)
ax_w.grid(alpha=0.3)

ax_r.set_xlabel("total simulated time = TR × Npe  [s]")
ax_r.set_ylabel("score (higher = better)")
ax_r.set_title("Geometric agreement metrics")
ax_r.set_ylim(0, 1.05)
ax_r.set_xlim(0, xmax)
ax_r.legend(loc="lower left", fontsize=9)
ax_r.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(out, dpi=130, bbox_inches="tight")
print(f"wrote {out}")

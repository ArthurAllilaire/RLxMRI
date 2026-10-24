#!/usr/bin/env python3
"""Generate the figures for the BIC-ISMRM 2026 abstract."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 10.0,
        "axes.titlesize": 10.0,
        "axes.labelsize": 10.0,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    }
)

BLUE = "#2F6B9A"
TEAL = "#17807E"
GREEN = "#3A8D5D"
ORANGE = "#D97732"
RED = "#B94C4C"
GREY = "#6D7378"
LIGHT_BLUE = "#DDECF2"


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.08,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        va="top",
    )


def draw_full_phantom(ax: plt.Axes) -> None:
    ax.set_aspect("equal")
    ax.add_patch(Circle((0, 0), 100, facecolor=LIGHT_BLUE, edgecolor=BLUE, lw=1.4, alpha=0.65))

    plate_specs = [
        (56.5, TEAL, r"$T_1$"),
        (16.5, ORANGE, r"$T_2$"),
        (-23.5, BLUE, "PD"),
    ]
    xs = np.array([-65, -48, -28, 0, 28, 48, 65])
    for z, colour, label in plate_specs:
        ax.add_patch(Ellipse((0, z), 145, 16, facecolor="white", edgecolor=colour, lw=1.0, alpha=0.9))
        for x in xs:
            ax.add_patch(Circle((x, z), 6.2, facecolor=colour, edgecolor="white", lw=0.35))
        ax.text(103, z, label, color=colour, va="center", ha="left", fontweight="bold")

    for x in (-80, -40, 0, 40, 80):
        for z in (-80, -40, 0, 40, 80):
            if x * x + z * z <= 92**2:
                ax.add_patch(Circle((x, z), 2.2, facecolor=GREY, edgecolor="none", alpha=0.6))

    ax.text(0, -112, "water-filled housing", color=GREY, ha="center", va="top")
    ax.set_xlim(-118, 128)
    ax.set_ylim(-120, 112)
    ax.axis("off")
    ax.set_title("MRISystemPhantom digital twin", pad=2)


def draw_t1_plate(ax: plt.Axes) -> None:
    t1_values = np.array(
        [1.879, 1.432, 1.027, 0.7513, 0.527, 0.3841, 0.2723,
         0.1945, 0.1378, 0.0947, 0.067, 0.04814, 0.03435, 0.02416]
    )
    theta_outer = np.linspace(0, 2 * np.pi, 10, endpoint=False)
    theta_inner = np.pi / 4 + np.linspace(0, 2 * np.pi, 4, endpoint=False)
    x = np.concatenate([65 * np.cos(theta_outer), 28 * np.cos(theta_inner)])
    y = np.concatenate([65 * np.sin(theta_outer), 28 * np.sin(theta_inner)])
    active = {0, 2, 5, 7, 13}

    ax.set_aspect("equal")
    ax.add_patch(Circle((0, 0), 96, facecolor=LIGHT_BLUE, edgecolor=BLUE, lw=1.1, alpha=0.45))
    norm = mpl.colors.LogNorm(vmin=t1_values.min(), vmax=t1_values.max())
    cmap = mpl.colormaps["viridis"]
    for idx, (xi, yi, value) in enumerate(zip(x, y, t1_values)):
        edge = "black" if idx in active else "white"
        lw = 1.5 if idx in active else 0.5
        ax.add_patch(Circle((xi, yi), 12.0, facecolor=cmap(norm(value)), edgecolor=edge, lw=lw))
    ax.text(0, -105, "black outline: five active spheres", ha="center", va="top", color=GREY)
    ax.set_xlim(-112, 112)
    ax.set_ylim(-115, 106)
    ax.axis("off")
    ax.set_title(r"$T_1$ plate geometry", pad=2)


def crop_simulated_image() -> np.ndarray:
    source = ROOT / "report_latex/imgs/mri_system_phantom/hamming_phantom_diff.png"
    image = Image.open(source).convert("RGB")
    # Central top panel: full-water IR-SE reconstruction with Hamming apodisation.
    return np.asarray(image.crop((825, 220, 1450, 820)))


def draw_workflow(ax: plt.Axes) -> None:
    ax.axis("off")
    labels = [
        "sample digital\nphantom",
        "simulate noisy\nIR-SE image",
        "reconstruct +\nfit $T_1$",
        "update 12-bin\nTI history",
        "policy selects\nnext TI/TR",
    ]
    colours = [LIGHT_BLUE, "#E8F3EF", "#F5EBDC", "#E9EEF7", "#E8F3EF"]
    lefts = np.linspace(0.01, 0.81, len(labels))
    width = 0.17
    for i, (left, label, colour) in enumerate(zip(lefts, labels, colours)):
        box = FancyBboxPatch(
            (left, 0.18), width, 0.62,
            boxstyle="round,pad=0.015,rounding_size=0.02",
            transform=ax.transAxes,
            facecolor=colour,
            edgecolor=BLUE if i in (0, 4) else GREY,
            lw=0.9,
        )
        ax.add_patch(box)
        ax.text(left + width / 2, 0.49, label, transform=ax.transAxes, ha="center", va="center", fontsize=8.3)
        if i < len(labels) - 1:
            arrow = FancyArrowPatch(
                (left + width, 0.49), (lefts[i + 1], 0.49),
                transform=ax.transAxes, arrowstyle="-|>", mutation_scale=9,
                lw=0.8, color=GREY,
            )
            ax.add_patch(arrow)


def make_method_figure() -> None:
    fig = plt.figure(figsize=(7.15, 3.05), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=[3.6, 1.15], width_ratios=[1.0, 1.0, 1.2])
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[0, 2])
    ax_d = fig.add_subplot(grid[1, :])

    draw_full_phantom(ax_a)
    draw_t1_plate(ax_b)
    ax_c.imshow(crop_simulated_image())
    ax_c.axis("off")
    ax_c.set_title("Simulated 2-D IR-SE image", pad=2)
    draw_workflow(ax_d)

    for ax, label in zip((ax_a, ax_b, ax_c, ax_d), ("A", "B", "C", "D")):
        panel_label(ax, label)

    fig.savefig(OUT / "method_overview.pdf")
    fig.savefig(OUT / "method_overview.png", dpi=300)
    plt.close(fig)


def load_heldout_data() -> dict:
    source = ROOT / "runs/e2/mf_runB_5sphere_hist_560s_gpu/global_best/eval_t1err_heldout.json"
    with source.open() as handle:
        return json.load(handle)


def make_results_figure() -> None:
    heldout = load_heldout_data()
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(7.15, 3.0), gridspec_kw={"width_ratios": [1.05, 1.0]}, constrained_layout=True
    )

    methods = ["RL + TI history", "RL, no history", "Fixed log grid", "CR-optimal fixed"]
    means = np.array([2.93095, 4.16, 6.03671, 7.74])
    ci_low = np.array([2.58019, 3.41, 5.36224, 7.08])
    ci_high = np.array([3.30213, 4.99, 6.69859, 8.42])
    success = [95.8, 66.7, 25.0, 8.3]
    colours = [GREEN, BLUE, GREY, ORANGE]
    y = np.arange(len(methods))[::-1]
    for yi, mean, lo, hi, colour in zip(y, means, ci_low, ci_high, colours):
        ax_a.errorbar(
            mean, yi, xerr=np.array([[mean - lo], [hi - mean]]), fmt="o",
            ms=6, color=colour, ecolor=colour, elinewidth=1.7, capsize=3, zorder=3,
        )
    ax_a.set_yticks(y, methods)
    ax_a.set_xlabel(r"Mean $T_1$ MAPE (%)  [lower is better]")
    ax_a.set_xlim(0, 9.5)
    ax_a.set_ylim(-0.45, 3.35)
    ax_a.grid(axis="x", color="#D9D9D9", lw=0.7)
    ax_a.axvspan(0, 5, color="#E9F4EC", alpha=0.65, zorder=0)
    for yi, mean, pct in zip(y, means, success):
        ax_a.text(mean + 0.24, yi + 0.12, f"{mean:.2f}%", color="#222222", va="center", fontsize=8.5)
        ax_a.text(mean + 0.24, yi - 0.16, f"{pct:.1f}% of episodes <5%", color=GREY, va="center", fontsize=7.4)
    ax_a.annotate(
        "51.4% lower",
        xy=(2.93, y[0] - 0.35), xytext=(6.04, y[0] - 0.35),
        ha="center", va="center", fontsize=8.2, color=GREEN,
        arrowprops={"arrowstyle": "<->", "color": GREEN, "lw": 1.0},
    )
    ax_a.set_title("560 s held-out evaluation (24 episodes)")
    panel_label(ax_a, "A")

    t1_true = np.asarray(heldout["per_episode_t1_err"]["t1_true_s"])
    ape_pct = 100 * np.asarray(heldout["per_episode_t1_err"]["ape"])
    ax_b.scatter(t1_true, ape_pct, s=16, color=BLUE, alpha=0.38, edgecolor="none", label="sphere measurement")
    bands = [(0.04, 0.5), (0.5, 1.2), (1.2, 1.9)]
    band_x = []
    band_y = []
    for lo, hi in bands:
        mask = (t1_true >= lo) & (t1_true < hi if hi < 1.9 else t1_true <= hi)
        band_x.append(np.exp(np.mean(np.log(t1_true[mask]))))
        band_y.append(np.mean(ape_pct[mask]))
    ax_b.plot(band_x, band_y, "o-", color=GREEN, lw=2.0, ms=6, label="mean by $T_1$ band")
    ax_b.axhline(5, color=RED, lw=1.0, ls="--", label="5% target")
    ax_b.set_xscale("log")
    ax_b.set_xlim(0.035, 2.05)
    ax_b.set_ylim(0, max(12.5, float(np.max(ape_pct)) + 0.5))
    ax_b.set_xlabel(r"Realised $T_1$ (s)")
    ax_b.set_ylabel("Absolute percentage error (%)")
    ax_b.grid(color="#E0E0E0", lw=0.6)
    ax_b.legend(frameon=False, loc="upper left")
    ax_b.set_title("TI-history policy across the sampled range")
    panel_label(ax_b, "B")

    fig.savefig(OUT / "heldout_results.pdf")
    fig.savefig(OUT / "heldout_results.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    make_method_figure()
    make_results_figure()
    print(f"Wrote figures to {OUT}")

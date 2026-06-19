#!/usr/bin/env python3
"""Generate the purpose-built talk figures (FIG-2, FIG-3, FIG-4).

Outputs PNGs into presentation/figs/ that drop into the deck placeholders.
Run:  python presentation/make_figs.py

(The env-loop diagram, FIG_envloop / fig_envloop.png, is exported separately
from the report's TikZ source via presentation/make_envloop.sh.)
"""
from matplotlib.patches import FancyArrowPatch, Rectangle, Circle
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figs")
os.makedirs(FIGS, exist_ok=True)

IMPERIAL = "#0000CD"   # report ImperialBlue (0,0,205)
NAVY = "#001E5A"
GREY = "#8a8a90"
LIGHT = "#eef0f8"
INK = "#1e1e23"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 13,
    "axes.edgecolor": "#444",
})


# ===========================================================================
# FIG-2a / FIG-2b — Two grouped related-work tables (replaces the single
#   capability matrix). The story is two tensions, one per table:
#     2a  adaptive qMRI exists, but never as a LEARNED multi-tissue policy
#     2b  RL controls MRI, but never for the QUANTITATIVE objective
#   THIS WORK is the only all-✓ row in BOTH tables — it supplies the one
#   property each group is missing. Method column shows the paradigm so
#   "learned RL" reads as one option among Bayesian / differentiable / etc.
# ===========================================================================
def _grouped_table(out, title, columns, rows, figsize=(11.0, 3.25),
                   footnote=None):
    """Render one related-work table. columns = [(header, frac, kind)] with
    kind in {"text","tick"}; tick cells take "y"/"n". Last row is highlighted
    as THIS WORK."""
    GREEN = "#1f8a3b"
    XGREY = "#b7b7bd"

    # column left-edges and centres from fractional widths
    L, R = 0.015, 0.985
    fracs = [c[1] for c in columns]
    span = R - L
    edges = [L]
    for f in fracs:
        edges.append(edges[-1] + f * span)
    pad = 0.012

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(L, 0.95, title, ha="left", va="top", fontsize=14.5,
            fontweight="bold", color=NAVY)

    # header row
    y_hdr = 0.78
    for j, (hdr, _f, kind) in enumerate(columns):
        if kind == "tick":
            xc = (edges[j] + edges[j + 1]) / 2
            ax.text(xc, y_hdr, hdr, ha="center", va="center", fontsize=11.5,
                    fontweight="bold", color=NAVY)
        else:
            ax.text(edges[j] + pad, y_hdr, hdr, ha="left", va="center",
                    fontsize=11.5, fontweight="bold", color=NAVY)
    ax.plot([L, R], [0.70, 0.70], color="#888", lw=1.4)

    n = len(rows)
    y0, dy = 0.60, min(0.135, 0.58 / n)
    for i, row in enumerate(rows):
        y = y0 - i * dy
        last = (i == n - 1)
        if last:
            ax.plot([L, R], [y + dy * 0.52, y + dy * 0.52], color="#888", lw=1.2)
            ax.add_patch(Rectangle((L - 0.008, y - dy * 0.45), span + 0.016,
                                   dy * 0.92, facecolor=IMPERIAL, alpha=0.10,
                                   edgecolor=IMPERIAL, lw=1.4, zorder=0))
        for j, (_hdr, _f, kind) in enumerate(columns):
            val = row[j]
            if kind == "tick":
                xc = (edges[j] + edges[j + 1]) / 2
                if val == "y":
                    ax.text(xc, y, "✓", ha="center", va="center", fontsize=19,
                            color=GREEN, fontweight="bold")
                else:
                    ax.text(xc, y, "✗", ha="center", va="center", fontsize=16,
                            color=XGREY)
            else:
                col = IMPERIAL if last else INK
                fw = "bold" if (last or j == 0) else "normal"
                ax.text(edges[j] + pad, y, val, ha="left", va="center",
                        fontsize=11.5, color=col, fontweight=fw)

    if footnote:
        y_foot = (y0 - (n - 1) * dy) - dy * 0.75
        ax.text(L, y_foot, footnote, ha="left", va="top", fontsize=8.6,
                color=GREY)

    fig.savefig(os.path.join(FIGS, out), dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("wrote", os.path.join(FIGS, out))


def fig2a_qmri_table():
    columns = [
        ("Quantitative MRI Sequence Design", 0.34, "text"),
        ("Method", 0.34, "text"),
        ("Adaptive?", 0.16, "tick"),
        ("Multi-voxel?", 0.16, "tick"),
    ]
    rows = [
        ["Beracha et al. (2023)", "Bayesian Posterior + CRLB Lookup", "y", "n"],
        ["MRF Schedule Optimisation", "Differentiable, Offline", "n", "y"],
        ["MIMOSA (2026)¹", "Optimised Multi-Echo, Offline", "n", "y"],
        ["THIS WORK", "Learned RL Policy (PPO)", "y", "y"],
    ]
    _grouped_table("fig2a_qmri.png",
                   "Adaptive qMRI exists — but never a learned multi-voxel policy",
                   columns, rows,
                   footnote="¹ Chen et al., MIMOSA, Magn. Reson. Med. 2026.")


def fig2b_rl_table():
    columns = [
        ("RL / Learned Control of MRI", 0.27, "text"),
        ("Method", 0.21, "text"),
        ("Objective", 0.34, "text"),
        ("Quantitative?", 0.18, "tick"),
    ]
    rows = [
        ["Walker-Samuel et al. (2023)", "Deep RL (DDPG)", "Shape Classification", "n"],
        ["Pineda et al. (2020)", "Deep RL (DQN)", "K-Space Recon Quality", "n"],
        ["AUTOSEQ (2018)", "Bayesian Optimisation", "Image MSE (1-D Toy)", "n"],
        ["DeepRF (2021)²", "Deep RL", "RF Pulse Waveform Design", "n"],
        ["THIS WORK", "Deep RL (PPO)", "Fitted-T1 Error", "y"],
    ]
    _grouped_table("fig2b_rl.png",
                   "RL controls MRI — but never for the quantitative objective",
                   columns, rows,
                   footnote="² Shin et al., DeepRF, Nat. Mach. Intell. 2021.")


# ===========================================================================
# FIG-2 — Capability matrix (works x {Learned, Adaptive, Quantitative})
#   Only THIS WORK ticks all three. No invented coordinates; the three
#   attributes are exactly the parts of the novelty claim.
# ===========================================================================
def fig2_capability():
    GREEN = "#1f8a3b"
    ORANGE = "#d98014"
    # (label, learned, adaptive, quantitative)  vals: 'y'/'n'/'p'
    prior = [
        ("Beracha '23  (adaptive model-based)", "n", "y", "y"),
        ("MRF schedule optimisation",           "n", "n", "y"),
        ("MRzero",                              "p", "n", "n"),
        ("AUTOSEQ",                             "y", "y", "n"),
        ("Walker-Samuel '19",                   "y", "y", "n"),
        ("Pineda '20",                          "y", "y", "n"),
    ]
    this = ("THIS WORK   —   RL agent for adaptive qMRI", "y", "y", "y")
    col_titles = ["Learned\n(RL policy)", "Adaptive\n(closed-loop)",
                  "Quantitative\n(T1/T2 estimation)"]
    col_x = [0.555, 0.730, 0.905]
    label_x = 0.03

    fig = plt.figure(figsize=(10.2, 6.6))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    def mark(x, y, v):
        if v == "y":
            ax.text(x, y, "✓", ha="center", va="center", fontsize=24,
                    color=GREEN, fontweight="bold")
        elif v == "p":
            ax.text(x, y, "∼", ha="center", va="center", fontsize=22,
                    color=ORANGE, fontweight="bold")
        else:
            ax.text(x, y, "✗", ha="center", va="center", fontsize=20,
                    color="#b7b7bd")

    # title
    ax.text(label_x, 0.965, "Prior work has at most two of the three —",
            fontsize=15, fontweight="bold", color=NAVY, va="top")
    ax.text(label_x, 0.915, "this is the first to combine all three",
            fontsize=15, fontweight="bold", color=IMPERIAL, va="top")

    # column headers
    y_hdr = 0.83
    for x, t in zip(col_x, col_titles):
        ax.text(x, y_hdr, t, ha="center", va="center", fontsize=12.5,
                fontweight="bold", color=NAVY)
    ax.plot([label_x, 0.97], [0.77, 0.77], color="#888", lw=1.4)

    # prior rows
    y0, dy = 0.71, 0.083
    for i, (lab, l, a, q) in enumerate(prior):
        y = y0 - i * dy
        ax.text(label_x, y, lab, ha="left", va="center", fontsize=12.5,
                color=INK)
        for x, v in zip(col_x, (l, a, q)):
            mark(x, y, v)

    # separator + THIS WORK row (highlighted)
    y_this = y0 - len(prior) * dy - 0.012
    ax.plot([label_x, 0.97], [y_this + 0.045, y_this + 0.045],
            color="#888", lw=1.4)
    ax.add_patch(plt.Rectangle((label_x - 0.015, y_this - 0.038), 0.985,
                               0.076, facecolor=IMPERIAL, alpha=0.10,
                               edgecolor=IMPERIAL, lw=1.6))
    lab, l, a, q = this
    ax.text(label_x, y_this, lab, ha="left", va="center", fontsize=13,
            fontweight="bold", color=IMPERIAL)
    for x, v in zip(col_x, (l, a, q)):
        mark(x, y_this, v)

    # footnotes
    fy = y_this - 0.075
    notes = [
        "Learned = a learned policy, not a hand-derived rule    ·    "
        "Adaptive = chooses each acquisition from observations so far    ·    "
        "Quantitative = objective is tissue-parameter (T1/T2) error",
        "∼  MRzero learns a sequence offline by supervised differentiable "
        "optimisation — not a closed-loop policy, and its objective is "
        "target-contrast fidelity, not parameter estimation.",
    ]
    for j, n in enumerate(notes):
        ax.text(label_x, fy - j * 0.045, n, ha="left", va="top",
                fontsize=9.3, color=GREY, wrap=True)

    out = os.path.join(FIGS, "fig2_capability.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out)


# ===========================================================================
# FIG-3 — Float-collapse: fixed absolute epsilon vs relative Float64 spacing
# ===========================================================================
def fig3_float_collapse():
    fig = plt.figure(figsize=(9.6, 5.4))

    # --- spacing vs fixed epsilon (log-log) ---
    ax = fig.add_subplot(1, 1, 1)
    t = np.logspace(0, 3, 600)            # 1 s .. 1000 s
    spacing_half = np.array([np.spacing(ti) / 2 for ti in t])
    eps_fixed = 1e-14

    ax.loglog(t, spacing_half, color=IMPERIAL, lw=2.4,
              label=r"half Float64 spacing  $\mathrm{eps}(t)/2 \approx t\cdot 2^{-53}$")
    ax.axhline(eps_fixed, color="#c0392b", lw=2.2, ls="--",
               label=r"fixed marker offset  $\epsilon = 10^{-14}$ s")

    # collapse region
    ax.axvspan(128, 1000, color="#c0392b", alpha=0.08)
    ax.axvline(128, color="#c0392b", lw=1.5, ls=":")
    ax.annotate("collapse threshold\n$t \\approx 128$ s",
                (128, 4e-14), (200, 1.1e-13),
                fontsize=11, color="#c0392b", fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color="#c0392b"))
    ax.text(420, 3e-15, "once $\\mathrm{eps}(t)/2 > \\epsilon$:\n$t+\\epsilon == t$  bit-for-bit",
            fontsize=10.5, color="#c0392b", ha="center")

    ax.set_xlabel("cumulative sequence time  $t$  (s)", fontsize=12)
    ax.set_ylabel("time gap (s)", fontsize=12)
    ax.set_xlim(1, 1000)
    ax.set_ylim(1e-16, 3e-13)
    ax.legend(loc="upper left", fontsize=10.5, framealpha=0.95)
    ax.grid(True, which="both", color="#eee", lw=0.6)
    ax.set_title("Fixed absolute $\\epsilon$ is swallowed by relative Float64 precision",
                 fontsize=14, fontweight="bold", color=NAVY, pad=8)

    fig.tight_layout()
    out = os.path.join(FIGS, "fig3_float_collapse.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out)


# ===========================================================================
# FIG-4 — Fidelity ladder: cost staircase + shrinking water-spin bars.
#   All numbers from Table tab:e2-fidelity-ladder in AdaptiveRLMRI.tex.
#   (The S = S_spheres + S_water linearity gets its own slide: fig4b below.)
# ===========================================================================
def fig4_ladder():
    WATER = "#9fb3e8"      # background water (the lever)
    SPHERE = IMPERIAL      # sphere spins the agent actually measures
    COST = "#d98014"       # measured s/step

    # rung order = cheap -> expensive (cost rises left to right)
    rungs = ["analytic", "cached3", "cached", "full3", "full"]
    water = [0,          0,         0,
             2103,    18919]   # water spins simulated
    sphere = [0,          4816,      4816,
              4816,    4816]    # sphere spins simulated
    cost = [0.034,      0.34,      0.34,     0.42,    1.03]    # s/step
    wmodel = ["closed\nform", "cached 3 mm", "cached 1 mm",
              "full 3 mm", "full 1 mm"]
    x = np.arange(len(rungs))

    fig = plt.figure(figsize=(9.0, 6.2))
    fig.subplots_adjust(left=0.10, right=0.90, top=0.90, bottom=0.13)

    # ---- stacked spin bars (linear) + cost staircase (log twin) ------------
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(x, sphere, width=0.62, color=SPHERE, label="sphere spins (measured)",
           zorder=3)
    ax.bar(x, water, width=0.62, bottom=sphere, color=WATER,
           label="water spins (the lever)", zorder=3)
    ax.set_ylabel("Bloch spins simulated / step", fontsize=12, color=NAVY)
    ax.set_ylim(0, 25500)
    ax.set_xticks(x)
    ax.set_xticklabels(rungs, fontsize=12.5, fontfamily="monospace")
    for xi, wm in zip(x, wmodel):
        ax.text(xi, -2300, wm, ha="center", va="top", fontsize=8.6, color=GREY)
    ax.tick_params(axis="y", labelcolor=NAVY)
    ax.spines["top"].set_visible(False)

    # total-spin annotations above each bar
    totals = [s + w for s, w in zip(sphere, water)]
    for xi, tot in zip(x, totals):
        txt = "0\n(no sim)" if tot == 0 else f"{tot:,}"
        ax.text(xi, tot + 600, txt, ha="center", va="bottom", fontsize=9,
                color=NAVY, fontweight="bold")

    # cost staircase on a log twin axis
    ax2 = ax.twinx()
    ax2.step(x, cost, where="mid", color=COST, lw=2.6, zorder=5)
    ax2.plot(x, cost, "o", color=COST, ms=7, zorder=6)
    ax2.set_yscale("log")
    ax2.set_ylabel("measured cost  (s / step, log)", fontsize=12, color=COST)
    ax2.tick_params(axis="y", labelcolor=COST)
    ax2.set_ylim(0.02, 3.0)
    ax2.spines["top"].set_visible(False)
    for xi, c in zip(x, cost):
        ax2.annotate(f"{c:g}s", (xi, c), textcoords="offset points",
                     xytext=(0, 11), ha="center", fontsize=9.5,
                     color=COST, fontweight="bold")

    # cost-range bracket (~30x)
    ax2.annotate("", xy=(4, 1.03), xytext=(4, 0.034),
                 arrowprops=dict(arrowstyle="<->", color=COST, lw=1.3))
    ax2.text(3.62, 0.21, "$\\sim$30$\\times$\ncost range", color=COST,
             fontsize=10.5, fontweight="bold", ha="right", va="center")
    # cached saving callout: ~3x end-to-end per step (0.34 vs 1.03 s, the plotted
    # staircase); the water SIM in isolation is 8x (4.47->0.56 s, AppendixA), but
    # the sphere sim + recon + refit floor dilutes that to ~3x per step.
    ax.annotate("cached rungs:\n$\\sim$3$\\times$ faster per step\n"
                "(8$\\times$ on sim)",
                (2, 6000), (2, 8000), fontsize=10, color=IMPERIAL,
                fontweight="bold", ha="center",
                arrowprops=dict(arrowstyle="-|>", color=IMPERIAL, lw=1.3))

    ax.legend(loc="upper left", fontsize=10, framealpha=0.95)
    ax.set_title("Fidelity ladder: cheaper rungs shrink the simulated water",
                 fontsize=14, fontweight="bold", color=NAVY, pad=10)
    ax.set_xlim(-0.62, 4.62)

    out = os.path.join(FIGS, "fig4_ladder.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out)


# ===========================================================================
# FIG-4b — Cached water: the linearity that powers it, equations only.
#   S_full = S_spheres + S_water, and the per-shot factorisation of S_water.
# ===========================================================================
def fig4b_water_linearity():
    GREEN = "#2a7d2a"
    fig = plt.figure(figsize=(11.0, 6.0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.93, "Cached water: the Bloch signal is linear in spins",
            ha="center", va="center", fontsize=18, fontweight="bold", color=NAVY)

    # Equation 1 — additivity
    ax.text(0.5, 0.70,
            "$S_\\mathrm{full} \\;=\\; S_\\mathrm{spheres} \\;+\\; S_\\mathrm{water}$",
            ha="center", va="center", fontsize=30, color=NAVY)
    ax.text(0.5, 0.585,
            "re-simulate only the sphere spins each step",
            ha="center", va="center", fontsize=13, color=GREY)

    # divider
    ax.plot([0.18, 0.82], [0.50, 0.50], color="#c7cfe8", lw=1.3)

    # Equation 2 — per-shot factorisation of the water term
    ax.text(0.5, 0.375,
            "$S_\\mathrm{water}[k] \\;=\\; "
            "M_z^{(k)}(\\mathrm{TI},\\mathrm{TR},\\alpha)\\,\\sin\\alpha "
            "\\;\\, W_\\alpha[k]$",
            ha="center", va="center", fontsize=28, color=IMPERIAL)
    ax.text(0.5, 0.255,
            "homogeneous water factorises — only the scalar "
            "$M_z^{(k)}$ depends on the timing,\n"
            "so the geometric template $W_\\alpha$ is built once and reused",
            ha="center", va="center", fontsize=13, color=GREY)

    ax.text(0.5, 0.08,
            "$\\sim$8$\\times$ cheaper per step  ·  matches full-Bloch "
            "$T_1$ fits to 0.12%  (worst 1.15%)",
            ha="center", va="center", fontsize=14, color=GREEN,
            fontweight="bold")

    out = os.path.join(FIGS, "fig4b_linearity.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out)


def fig9_positioning():
    """Acceleration vs the nearest quantitative published comparator (Beracha).

    Honest 'positioning' bar chart, NOT a like-for-like ranking. Our number is
    the precision^2->time heuristic applied to the TI-histogram policy:
    (6.04 / 2.93)^2 = 4.25x, with a delta-method 95% CI of [3.05, 5.92]
    (recovered from the Run B memory-ablation CIs, n=24; verified by Monte
    Carlo). Beracha rows are the ranges reported in adaptive_mri:23.
    """
    ORANGE = "#d98014"
    GREY2 = "#b9bcc9"

    # (label, central x, (lo, hi) whisker or None, colour, highlight)
    rows = [
        ("This work — TI-histogram\n($T_1$, 5 spheres, 560 s)",
         4.25, (3.05, 5.92), IMPERIAL, True),
        ("Beracha et al. — simulated $T_1$",
         2.85, (1.7, 4.0), ORANGE, False),
        ("Beracha et al. — in vivo\n(healthy volunteers)",
         2.5, None, GREY2, False),
    ]
    labels = {
        0: "4.25$\\times$   (95% CI 3.0–5.9$\\times$)",
        1: "1.7–4$\\times$ range",
        2: "2.5$\\times$",
    }

    y = np.arange(len(rows))[::-1]   # first row on top
    fig = plt.figure(figsize=(9.4, 5.0))
    fig.subplots_adjust(left=0.30, right=0.965, top=0.80, bottom=0.16)
    ax = fig.add_subplot(1, 1, 1)

    for yi, (lab, cx, whisk, col, hl) in zip(y, rows):
        ax.barh(yi, cx, height=0.56, color=col, zorder=3,
                edgecolor=NAVY if hl else "none", linewidth=1.6 if hl else 0)
        if whisk is not None:
            ax.errorbar(cx, yi, xerr=[[cx - whisk[0]], [whisk[1] - cx]],
                        fmt="none", ecolor=NAVY if hl else "#7a5a16",
                        elinewidth=1.8, capsize=6, capthick=1.8, zorder=4)
        idx = list(y).index(yi)
        ax.text(whisk[1] + 0.12 if whisk else cx + 0.12, yi, labels[idx],
                va="center", ha="left", fontsize=11.5,
                color=IMPERIAL if hl else INK, fontweight="bold" if hl else "normal")

    ax.set_ylim(-0.75, len(rows) - 0.25)
    ax.axvline(1.0, color=GREY, ls="--", lw=1.2, zorder=2)
    ax.text(1.0, -0.62, "no speedup", color=GREY, fontsize=9,
            ha="center", va="center")

    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=11.5, color=INK)
    ax.set_xlim(0, 7.0)
    ax.set_xlabel("Acceleration at matched precision  ($\\times$)",
                  fontsize=12.5, color=NAVY)
    ax.set_title("Positioning: acceleration vs published adaptive qMRI",
                 fontsize=14, fontweight="bold", color=NAVY, pad=35)
    ax.text(0.5, 1.025,
            "via Beracha's precision$^2\\!\\to$time heuristic  ·  tasks differ — "
            "heuristic positioning, not a like-for-like ranking",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=9.5, color=GREY, style="italic")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)

    out = os.path.join(FIGS, "fig9_positioning.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    fig2a_qmri_table()
    fig2b_rl_table()
    fig2_capability()
    fig3_float_collapse()
    fig4_ladder()
    fig4b_water_linearity()
    fig9_positioning()

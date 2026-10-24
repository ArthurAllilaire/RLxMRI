"""Shared style for the ISMRM 2027 abstract figures.

Colours follow the phantom viewer: viridis encodes T1 everywhere, and the
method/rung colours are drawn from viridis' teal and indigo so every figure
reads as one family. Change a colour here, never in a figure script.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
VANCOUVER = Path(__file__).resolve().parents[1]
FIGURES = VANCOUVER / "figures"

T1_CMAP = "viridis"
# One T1-to-colour mapping for every figure; covers the sampled range (0.024-1.88 s).
T1_NORM = (0.0, 2.0)
# Relative T1 error maps: blue underestimates, red overestimates, neutral at 0.
DIFF_CMAP = "RdBu_r"

# Validated with the dataviz palette checker (light surface, all checks pass).
ACCENT = "#0A9C8C"
RL_OTHER = "#4F5FA8"
# Kept lighter than the accent so a filled comparator bar never outweighs the hero bar.
FIXED_GRID = "#7D848B"
CRAMER_RAO = "#B3B8BD"

METHOD_COLOURS = {
    "hist": ACCENT,
    "nomem": RL_OTHER,
    "sigma": RL_OTHER,
    "lstm": RL_OTHER,
    "log_grid": FIXED_GRID,
    "cr_optimal": CRAMER_RAO,
}
METHOD_LABELS = {
    "hist": "RL + TI histogram",
    "nomem": "RL, no memory",
    "sigma": "RL + fitter uncertainty",
    "lstm": "RL, LSTM",
    "log_grid": "Fixed log TI grid",
    "cr_optimal": "Cramér-Rao optimal",
}
# Fixed protocols are also told apart by line style, so identity is never colour alone.
METHOD_LINESTYLES = {"log_grid": "--", "cr_optimal": ":"}

RUNGS = ["analytic", "cached3", "cached", "full3", "full"]
RUNG_LABELS = {
    "analytic": "analytic",
    "cached3": "cached 3 mm",
    "cached": "cached 1 mm",
    "full3": "full 3 mm",
    "full": "full 1 mm",
}
# Single-hue ordinal ramp, light (cheap) to dark (target fidelity).
RUNG_COLOURS = {
    "analytic": "#E6E8F5",
    "cached3": "#C9CEEB",
    "cached": "#A3ACDB",
    "full3": "#7884C4",
    "full": "#3F4C94",
}

INK = "#222222"
INK_SECONDARY = "#555555"
GRID = "#E5E5E5"


def apply() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            # Otherwise subscripts such as the 1 in T_1 fall back to DejaVu.
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "font.size": 9.0,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.0,
            "legend.frameon": False,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": INK_SECONDARY,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "lines.linewidth": 1.6,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "savefig.facecolor": "white",
            # TrueType instead of Type 3, so text stays editable in the PDFs.
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel_label(ax: plt.Axes, label: str, x: float = -0.12, y: float = 1.04, va: str = "bottom") -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=10.5, fontweight="bold", va=va, ha="left")


def save(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"{stem}.{ext}")

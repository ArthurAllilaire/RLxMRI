"""Figure 1, version 2: the twin, its training randomisation, the IR-SE shot and the fidelity ladder.

(A) The 3 T viewer re-export and (D) the two-axis fidelity ladder, as in fig1.py's
    fig1_B1_shotreexport.
(B) fig1.py's three training phantoms (B1), shrunk to a row over a table of their poses.
(C) One IR-SE shot (seqdiagram.py), generic: TI and TR are left to the policy.

Two layouts: fig1_v2 has A on the left and B, C, D stacked on the right; fig1_v2_alt puts
B under A, sharing A's T1 colourbar, and stacks C over D.

Run from the RLxMRI repo root, once fig1.py's inputs and fig3_sequence_waveforms.json exist:
  .venv/bin/python ismrm_latex/vancouver/scripts/fig1_v2.py
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch

import fig1
import figstyle
from seqdiagram import WAVEFORMS, ShotLabels, dimension_arrow, draw_ir_se_shot

SMALL = fig1.SMALL
# The policy's action bounds in python/qalibremd_gym/env_e2.py; importing that module would start Julia.
TI_RANGE_S = (0.01, 3.0)
TR_RANGE_S = (0.5, 5.0)
# On a shared baseline, so a subscript in the title cannot lift it above the panel letter.
HEADING = {"fontsize": SMALL + 1.5, "color": figstyle.INK, "va": "baseline", "ha": "left"}
NOTE = {"fontsize": SMALL, "color": figstyle.INK_SECONDARY}
# From a panel's top edge to its heading's baseline, and from there to the panel's content.
HEAD_H, HEAD_GAP = 0.17, 0.08
# Under A: the gap to its colourbar, the bar, then its tick labels and title.
A_CBAR_GAP, A_CBAR_H, A_CBAR_LABELS_H = 0.12, 0.07, 0.3
# Left of the ladder's axes for its spin label, right of it for its cost label, and under it for the
# rung names.
D_LEFT, D_RIGHT, D_LABELS_H = 0.45, 0.42, 0.42
# Row labels of B's pose table, whose column also holds the scale bar.
B_LABEL_W = 0.62
# B's two-line pose table and its gap below the slabs.
B_TABLE_H, B_TABLE_GAP = 0.25, 0.03
# The removed plate positions keep the dotted look of fig1.py at a weight that survives the smaller slabs.
B_REMOVED_LW = 0.75


@dataclass
class Canvas:
    fig: plt.Figure
    w: float
    h: float

    def axes(self, x: float, top: float, w: float, h: float) -> plt.Axes:
        """Axes from its left and top edges, in inches from the canvas' left and top."""
        return self.fig.add_axes((x / self.w, (self.h - top - h) / self.h, w / self.w, h / self.h))

    def text(self, x: float, top: float, s: str, **kw):
        return self.fig.text(x / self.w, (self.h - top) / self.h, s, **kw)

    def heading(self, x: float, top: float, letter: str, title: str) -> float:
        """Panel letter and title at fig1.py's size; returns where the panel's content starts."""
        self.text(x, top + HEAD_H, letter, fontweight="bold", **HEADING)
        self.text(x + 0.15, top + HEAD_H, title, **HEADING)
        return top + HEAD_H + HEAD_GAP


def new_canvas(h: float) -> Canvas:
    # savefig pads the tight bbox, so the canvas is narrower than the saved width.
    w = fig1.SAVED_W_IN - 2 * mpl.rcParams["savefig.pad_inches"]
    return Canvas(plt.figure(figsize=(w, h)), w, h)


@dataclass
class Inputs:
    panel_a: fig1.PanelA
    a_w: float
    rungs: list[fig1.Rung]
    slabs: Mapping[str, np.ndarray]
    slab_meta: dict
    t1_norm: Normalize
    wf: dict

    @property
    def a_h(self) -> float:
        return self.a_w * self.panel_a.image.shape[0] / self.panel_a.image.shape[1]


def load() -> Inputs:
    twin_meta = fig1.read_json(fig1.TWIN_META)
    return Inputs(fig1.panel_a_source("shotreexport", twin_meta), twin_meta["panel_width_in"], fig1.ladder(),
                  np.load(fig1.SLABS_NPZ), fig1.read_json(fig1.SLABS_META), Normalize(*twin_meta["t1_range_s"]),
                  fig1.read_json(WAVEFORMS))


# ---------------------------------------------------------------- panels


def a_block_h(inp: Inputs) -> float:
    return HEAD_H + HEAD_GAP + inp.a_h + A_CBAR_GAP + A_CBAR_H + A_CBAR_LABELS_H


def b_block_h(disc: float) -> float:
    return HEAD_H + HEAD_GAP + disc + B_TABLE_GAP + B_TABLE_H


def place_a(cv: Canvas, inp: Inputs, top: float) -> float:
    """A and its colourbar, laid out as in fig1.py; returns the bottom of the block."""
    y = cv.heading(0.0, top, "A", f"Digital twin, {fig1.FIELD_LABELS[inp.panel_a.field]}")
    cax = cv.axes(0.3, y + inp.a_h + A_CBAR_GAP, inp.a_w - 0.9, A_CBAR_H)
    fig1.panel_twin(cv.axes(0.0, y, inp.a_w, inp.a_h), cax, inp.panel_a, inp.t1_norm)
    return top + a_block_h(inp)


def place_b(cv: Canvas, inp: Inputs, x: float, top: float, w: float, disc: float) -> float:
    """The three slabs in a row over a table of their shift and rotation, whose row-label column also
    holds the scale bar, level with the slabs' bottom edge."""
    y = cv.heading(x, top, "B", "Training randomisation, T$_1$-plate slab")
    half = fig1.b1_half_width(inp.slabs, inp.slab_meta)
    slot = (w - B_LABEL_W) / 3
    centres = [x + B_LABEL_W + slot * (i + 0.5) for i in range(3)]
    for cx, d in zip(centres, inp.slab_meta["b1"]):
        fig1.draw_training_slab(cv.axes(cx - disc / 2, y, disc, disc), inp.slabs, d, inp.t1_norm, half,
                                removed_lw=B_REMOVED_LW)

    # On the slabs' mm scale, with the bar on the slabs' bottom edge.
    mm_per_in = 2 * half / disc
    bar = cv.axes(x, y, B_LABEL_W, disc)
    bar.set_xlim(0, B_LABEL_W * mm_per_in)
    bar.set_ylim(-half, half)
    bar.axis("off")
    bar_y = -half + 2
    bar.plot([0, 50], [bar_y, bar_y], color=figstyle.INK, lw=1.0, solid_capstyle="butt")
    bar.annotate("50 mm", (0, bar_y), xytext=(0, 1.5), textcoords="offset points", ha="left", va="bottom",
                 fontsize=SMALL, color=figstyle.INK)

    rows = [y + disc + B_TABLE_GAP + i * B_TABLE_H / 2 for i in range(2)]
    for ry, label in zip(rows, ("shift (mm)", "rotation")):
        cv.text(x, ry, label, ha="left", va="top", **NOTE)
    for cx, d in zip(centres, inp.slab_meta["b1"]):
        tx, ty = d["translation_mm"]
        cv.text(cx, rows[0], f"({fig1.fmt_signed(tx, 1)}, {fig1.fmt_signed(ty, 1)})", ha="center", va="top",
                **NOTE)
        cv.text(cx, rows[1], f"{fig1.fmt_signed(d['rotation_deg'], 1)}°", ha="center", va="top", **NOTE)
    return top + b_block_h(disc)


def policy_key(cv: Canvas, right: float, top: float) -> None:
    """A policy arrow and 'chosen by the policy', right-aligned on the heading's baseline."""
    label = cv.text(right, top + HEAD_H, "chosen by the policy", ha="right", va="baseline", **NOTE)
    left = label.get_window_extent(cv.fig.canvas.get_renderer()).transformed(cv.fig.transFigure.inverted()).x0
    # Arial's x-height is about half an em, so this is mid-height of the lower-case letters.
    y = (cv.h - top - HEAD_H + 0.26 * NOTE["fontsize"] / 72) / cv.h
    x1 = left - 0.06 / cv.w
    cv.fig.add_artist(FancyArrowPatch((x1 - 0.28 / cv.w, y), (x1, y), transform=cv.fig.transFigure,
                                      **dimension_arrow(policy=True)))


def place_c(cv: Canvas, inp: Inputs, x: float, top: float, w: float, h: float) -> float:
    y = cv.heading(x, top, "C", "IR-SE shot")
    policy_key(cv, x + w, top)
    labels = ShotLabels(rf"$\mathbf{{TI}}$ {TI_RANGE_S[0]:g}-{TI_RANGE_S[1]:g} s",
                        f"TE {inp.wf['TE_s'] * 1e3:.0f} ms (fixed)",
                        rf"$\mathbf{{TR}}$ {TR_RANGE_S[0]:g}-{TR_RANGE_S[1]:g} s "
                        f"(×{inp.wf['Npe']} phase-encode shots per block)")
    draw_ir_se_shot(cv.axes(x, y, w, h), inp.wf, labels)
    return y + h


def place_d(cv: Canvas, inp: Inputs, x: float, top: float, w: float, plot_h: float) -> None:
    y = cv.heading(x, top, "D", "Fidelity ladder, 14-sphere plate, CPU")
    fig1.panel_ladder_dual(cv.axes(x + D_LEFT, y, w - D_LEFT - D_RIGHT, plot_h), inp.rungs)


# ---------------------------------------------------------------- layouts


def main_layout(inp: Inputs) -> plt.Figure:
    """A left; B, C and D stacked on the right."""
    disc, shot_h, plot_h, gap = 0.62, 0.95, 0.92, 0.04
    c_h = HEAD_H + HEAD_GAP + shot_h
    d_h = HEAD_H + HEAD_GAP + plot_h + D_LABELS_H
    cv = new_canvas(b_block_h(disc) + gap + c_h + gap + d_h)
    x_r = inp.a_w + 0.2
    w_r = cv.w - x_r
    place_a(cv, inp, 0.0)
    y = place_b(cv, inp, x_r, 0.0, w_r, disc)
    y = place_c(cv, inp, x_r, y + gap, w_r, shot_h)
    place_d(cv, inp, x_r, y + gap, w_r, plot_h)
    return cv.fig


def alt_layout(inp: Inputs) -> plt.Figure:
    """A over B on the left; C over D on the right, the ladder taking the height B adds."""
    disc, shot_h, gap = 0.58, 1.3, 0.08
    cv = new_canvas(a_block_h(inp) + b_block_h(disc))
    plot_h = cv.h - (HEAD_H + HEAD_GAP + shot_h + gap + HEAD_H + HEAD_GAP + D_LABELS_H)
    x_r = inp.a_w + 0.2
    w_r = cv.w - x_r
    y = place_a(cv, inp, 0.0)
    place_b(cv, inp, 0.0, y, inp.a_w, disc)
    y = place_c(cv, inp, x_r, 0.0, w_r, shot_h)
    place_d(cv, inp, x_r, y + gap, w_r, plot_h)
    return cv.fig


def main() -> None:
    figstyle.apply()
    inp = load()
    for build, stem in ((main_layout, "fig1_v2"), (alt_layout, "fig1_v2_alt")):
        fig = build(inp)
        figstyle.save(fig, stem)
        plt.close(fig)
        print("wrote", figstyle.FIGURES / f"{stem}.png")


if __name__ == "__main__":
    main()

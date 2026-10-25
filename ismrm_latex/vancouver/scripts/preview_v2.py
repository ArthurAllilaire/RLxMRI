"""Preview figure, version 2: the digital twin beside the held-out T1 error.

Left: preview_v2_twin3d.jl's clean render of Figure 1's twin (no axes, legend or hover box, so it
stays legible as a thumbnail), over Figure 1's T1 colourbar.
Right: T1 MAPE with bootstrap 95% CI of the TI-histogram policy and the two fixed designs on
the held-out phantoms, from fig5_table_v3's panel A.

ISMRM shows the preview without a caption, at 3:2 and at most 1 MB, and people browse the
programme on phones, so every text is sized to stay legible at about 350 px wide.

Run from the RLxMRI repo root, once fig1.py's and fig5_table_v3.py's inputs exist:
  .venv/bin/python ismrm_latex/vancouver/scripts/preview_v2.py
"""

from __future__ import annotations

import io

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from PIL import Image

import fig1
import figstyle
from fig1_v2 import Canvas
from fig5_table import _round
from fig5_table_v3 import panel_a as fig5_rows

CLEAN_PNG = figstyle.FIGURES / "preview_v2_twin3d.png"
CLEAN_META = figstyle.FIGURES / "preview_v2_twin3d.json"
W_IN, H_IN, DPI = 6.0, 4.0, 300
MAX_BYTES = 1_000_000
BARS = {"hist": "RL policy", "log_grid": "Best fixed grid", "cr_optimal": "Cramér-Rao optimal"}
TITLE = r"RL in a digital twin learns better T$_\mathbf{1}$ protocols"

TITLE_PT, SUBTITLE_PT = 17.5, 13.0
LABEL_PT, VALUE_PT, TICK_PT = 14.5, 16.0, 13.0
PT = 1 / 72
# Layout in inches from the canvas' top-left corner. The colourbar and the bar chart share their
# bottom edge and tick styling, so their axis titles sit on one baseline.
TEXT_X = 0.17
TITLE_TOP = 0.12
CONTENT_TOP = 0.8
AXIS_BOTTOM = 3.43
CBAR_H, CBAR_GAP = 0.12, 0.11
CHART_X, CHART_W = 3.5, 1.9
TICKS = {"labelsize": TICK_PT, "length": 4, "width": 0.8, "pad": 2}
LABEL_PAD = 3
# Breaks the light grid lines behind text that sits over the plot area.
KNOCKOUT = {"facecolor": "white", "edgecolor": "none", "pad": 1.0}


# ---------------------------------------------------------------- data


def results() -> tuple[list, dict]:
    table, info = fig5_rows()
    budget = table[0].row.budget_s
    rows = {t.row.key: t.row for t in table if t.row.budget_s == budget}
    return [rows[k] for k in BARS], {"n_scans": info["n_scans"], "budget_s": budget}


def clean_render(t1_range_s: list[float]) -> fig1.PanelA:
    meta = fig1.read_json(CLEAN_META)
    if meta["t1_range_s"] != t1_range_s:
        raise ValueError("preview_v2_twin3d.jl and fig1.py disagree on the T1 colour range")
    # T1s above the range saturate in the render, as in panel A's re-export.
    return fig1.PanelA(fig1.trim_white(plt.imread(CLEAN_PNG)[..., :3]), meta["field"], "T$_1$ (s)", "max")


# ---------------------------------------------------------------- panels


def place_twin(cv: Canvas, panel: fig1.PanelA, t1_norm: Normalize) -> None:
    """The twin over its colourbar, centred left of the chart and filling the height between the title
    and the shared bottom edge."""
    cbar_top = AXIS_BOTTOM - CBAR_H
    img_h = cbar_top - CBAR_GAP - CONTENT_TOP
    img_w = img_h / panel.image.shape[0] * panel.image.shape[1]
    x = (CHART_X - img_w) / 2
    cax = cv.axes(x, cbar_top, img_w, CBAR_H)
    fig1.panel_twin(cv.axes(x, CONTENT_TOP, img_w, img_h), cax, panel, t1_norm)
    # fig1.py sizes the colourbar for print; here it has to survive a phone-width thumbnail.
    cax.tick_params(**TICKS)
    cax.set_xlabel(cax.get_xlabel(), fontsize=LABEL_PT, labelpad=LABEL_PAD)
    cax.spines["outline"].set_linewidth(0.8)


def place_bars(cv: Canvas, rows: list) -> None:
    """One bar per method, its name above it so the bars get the full width."""
    ax = cv.axes(CHART_X, CONTENT_TOP, CHART_W, AXIS_BOTTOM - CONTENT_TOP)
    ys = np.arange(len(rows))[::-1]
    bar_h = 0.4
    for y, r in zip(ys, rows):
        lo, hi = r.ci_pct
        ax.barh(y, r.mape_pct, height=bar_h, color=figstyle.METHOD_COLOURS[r.key], zorder=2)
        # A white halo keeps the whisker visible where it crosses a dark bar.
        for colour, lw, cap in (("white", 5.0, 8.5), (figstyle.INK, 2.0, 7.0)):
            ax.errorbar(r.mape_pct, y, xerr=[[r.mape_pct - lo], [hi - r.mape_pct]], fmt="none", ecolor=colour,
                        elinewidth=lw, capsize=cap, capthick=lw, zorder=3)
        weight = "bold" if r.key == "hist" else "normal"
        ax.annotate(f"{_round(r.mape_pct, 2)}%", (hi, y), xytext=(7, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=VALUE_PT, fontweight=weight, color=figstyle.INK,
                    bbox=KNOCKOUT, zorder=4)
        ax.annotate(BARS[r.key], (0, y + bar_h / 2), xytext=(-1, 5), textcoords="offset points", va="bottom",
                    ha="left", fontsize=LABEL_PT, fontweight=weight, color=figstyle.INK, bbox=KNOCKOUT, zorder=4)

    xmax = np.ceil(max(r.ci_pct[1] for r in rows) / 2) * 2
    ax.set_xlim(0, xmax)
    ax.set_xticks(np.arange(0, xmax + 1e-9, 2))
    ax.set_ylim(-0.4, len(rows) - 0.4)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="x", **TICKS)
    ax.xaxis.grid(True, zorder=0)
    ax.set_xlabel("T$_1$ error, MAPE (%)", fontsize=LABEL_PT, labelpad=LABEL_PAD)


# ---------------------------------------------------------------- layout


def check_inside(fig: plt.Figure) -> None:
    """Every text lies on the canvas: the fixed 3:2 canvas would silently crop anything outside it."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for t in fig.findobj(mpl.text.Text):
        if t.get_visible() and t.get_text():
            e = t.get_window_extent(renderer)
            assert fig.bbox.x0 <= e.x0 and e.x1 <= fig.bbox.x1 and fig.bbox.y0 <= e.y0 and e.y1 <= fig.bbox.y1, \
                (t.get_text(), e)


def compose(panel: fig1.PanelA, t1_norm: Normalize, rows: list, info: dict) -> plt.Figure:
    cv = Canvas(plt.figure(figsize=(W_IN, H_IN)), W_IN, H_IN)
    cv.text(TEXT_X, TITLE_TOP, TITLE, fontsize=TITLE_PT, fontweight="bold", va="top", ha="left",
            color=figstyle.INK)
    cv.text(TEXT_X, TITLE_TOP + 1.45 * TITLE_PT * PT,
            f"Simulated, {info['n_scans']} held-out phantoms, {_round(info['budget_s'])} s scans, "
            "mean with 95% CI",
            fontsize=SUBTITLE_PT, va="top", ha="left", color=figstyle.INK_SECONDARY)
    place_twin(cv, panel, t1_norm)
    place_bars(cv, rows)
    check_inside(cv.fig)
    return cv.fig


def save_png(fig: plt.Figure, stem: str) -> None:
    buf = io.BytesIO()
    # A fixed canvas keeps the required 3:2 aspect; a tight bbox would crop it.
    with mpl.rc_context({"savefig.bbox": "standard"}):
        fig.savefig(buf, format="png", dpi=DPI)
    plt.close(fig)
    img = Image.open(buf).convert("RGB")
    assert img.size == (round(W_IN * DPI), round(H_IN * DPI)) and 2 * img.size[0] == 3 * img.size[1], img.size
    path = figstyle.FIGURES / f"{stem}.png"
    img.save(path, optimize=True)
    size = path.stat().st_size
    assert size <= MAX_BYTES, (path, size)
    print(f"wrote {path} ({img.size[0]}x{img.size[1]} px, {size / 1e3:.0f} kB)")


def main() -> None:
    figstyle.apply()
    rows, info = results()
    twin_meta = fig1.read_json(fig1.TWIN_META)
    t1_norm = Normalize(*twin_meta["t1_range_s"])

    clean = clean_render(twin_meta["t1_range_s"])
    save_png(compose(clean, t1_norm, rows, info), "preview_v2")


if __name__ == "__main__":
    main()

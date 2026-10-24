"""Figure 1: digital twin, training randomisation and the fidelity ladder.

(A) 3D view of the twin. Default: static render (fig1_twin3d.jl), T1-plate spheres coloured
    by T1. With --viewer, the report's 3 T viewer screenshot
    (report_latex/imgs/mri_system_phantom/T1_voxel_phantom.png) cropped to drop its dropdown
    and slider (fig1_B1_shotcrop), and its print-sized Kaleido re-export on the shared T1
    scale (fig1_twin3d_viewer.jl, fig1_B1_shotreexport), which is paired with the
    two-axis ladder in (C).
(B) Two candidates, written as separate figures so one can be chosen:
    B1 (fig1_B1): T1-plate slab of three phantoms drawn by the training environment;
    B2 (fig1_B2): the nominal twin sliced through the T1, T2 and PD plates.
    Both are ground-truth spin maps from fig1_slabs.jl.
(C) Cost per RL step of each fidelity rung (Run A, 14-sphere plate, CPU), with the
    Bloch-simulated spin count of each rung: as labelled bars on one axis, or (with the
    re-export) as stacked spin bars on a left axis and a cost step line on a right axis.

Run from the RLxMRI repo root:
  julia --project=. ismrm_latex/vancouver/scripts/fig1_twin3d.jl
  julia --project=. ismrm_latex/vancouver/scripts/fig1_slabs.jl
  .venv/bin/python ismrm_latex/vancouver/scripts/fig1.py
and for the viewer candidates:
  .venv/bin/python ismrm_latex/vancouver/scripts/fig1_viewer_camera.py
  julia --project=. ismrm_latex/vancouver/scripts/fig1_twin3d_viewer.jl
  .venv/bin/python ismrm_latex/vancouver/scripts/fig1.py --viewer
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
from matplotlib.patches import Circle
from matplotlib.ticker import FuncFormatter

import figstyle

LADDER_TEX = figstyle.ROOT / "report_latex" / "chapters" / "AdaptiveRLMRI.tex"
SPIN_COUNTS = figstyle.ROOT / "scripts" / "e2_spin_counts.json"
RUN_A_HISTORY = figstyle.ROOT / "runs" / "e2" / "mf_v2_runA_cpu_9h" / "fidelity_history.json"
TWIN_PNG = figstyle.FIGURES / "fig1_twin3d.png"
TWIN_META = figstyle.FIGURES / "fig1_twin3d.json"
SLABS_NPZ = figstyle.FIGURES / "fig1_slabs.npz"
SLABS_META = figstyle.FIGURES / "fig1_slabs.json"
SHOT_PNG = figstyle.ROOT / "report_latex" / "imgs" / "mri_system_phantom" / "T1_voxel_phantom.png"
SHOT_CROP_PNG = figstyle.FIGURES / "fig1_twin3d_shotcrop.png"
VIEWER_PNG = figstyle.FIGURES / "fig1_twin3d_viewer.png"
VIEWER_META = figstyle.FIGURES / "fig1_twin3d_viewer.json"
# examples/plot_phantom_3d.jl, which produced the screenshot, uses PhantomConfig(field = :T3).
SHOT_FIELD = "T3"
# Viewer controls in the screenshot (device px): the property dropdown lies left of this
# column and the water-opacity slider below this row; nothing else does.
SHOT_DROPDOWN_RIGHT_PX = 172
SHOT_SLIDER_TOP_PX = 1095

CM = 1 / 2.54
SAVED_W_IN = 17.0 * CM
SMALL = 7.0
MINUS = "\u2212"
FIELD_LABELS = {"T15": "1.5 T", "T3": "3 T"}
# The housing as it appears in the 3D render (translucent water over white).
WATER_TINT = "#E3EDF8"
EMPTY = "#FFFFFF"
# Plate positions left empty (water) in a training phantom.
REMOVED_SPHERE = "#8A9096"
# Light-to-dark greys for the T2 and PD plates, which panel A shows in grey.
GREY_CMAP = LinearSegmentedColormap.from_list("plate_grey", ["#E9EBED", "#3A3F44"])
# Offset of the ladder's spin-count rows below its baseline, under the two-line rung names.
SPIN_ROW_PT = 20.0
# Two-axis ladder: sphere spins dark and water spins light from the rung ramp; axis tops
# chosen so the cost line runs above every bar and its count.
SPHERE_SPINS = figstyle.RUNG_COLOURS["full"]
WATER_SPINS = figstyle.RUNG_COLOURS["cached3"]
LADDER_SPIN_AXIS_MAX = 50_000
LADDER_COST_AXIS_MAX = 1.2


def read_json(path: Path):
    return json.loads(path.read_text())


@dataclass
class PanelA:
    image: np.ndarray
    field: str
    # Label of the 0-2 s colourbar under A: shared by A and B, except under the screenshot
    # crop, which carries its own 0-3 s colourbar.
    colourbar_label: str
    # "max" when A has T1 values above the range, which saturate.
    colourbar_extend: str = "neither"


def trim_white(img: np.ndarray) -> np.ndarray:
    ink = np.nonzero((img[..., :3] < 0.98).any(-1))
    return img[ink[0].min():ink[0].max() + 1, ink[1].min():ink[1].max() + 1]


def panel_a_source(name: str, twin_meta) -> PanelA:
    if name == "render":
        return PanelA(plt.imread(TWIN_PNG), twin_meta["field"], "T$_1$ (s)")
    if name == "shotcrop":
        crop = trim_white(plt.imread(SHOT_PNG)[:SHOT_SLIDER_TOP_PX, SHOT_DROPDOWN_RIGHT_PX + 1:, :3])
        plt.imsave(SHOT_CROP_PNG, crop)
        return PanelA(crop, SHOT_FIELD, "T$_1$ in B (s)")
    if name == "shotreexport":
        meta = read_json(VIEWER_META)
        # The re-export's fonts are sized for this panel width, so it is placed untrimmed.
        if (meta["panel_width_in"], meta["t1_range_s"]) != (twin_meta["panel_width_in"], twin_meta["t1_range_s"]):
            raise ValueError("fig1_twin3d_viewer.jl and fig1.py disagree on panel A's width or T1 range")
        return PanelA(plt.imread(VIEWER_PNG)[..., :3], meta["field"], "T$_1$ (s)", "max")
    raise ValueError(f"unknown panel A source {name!r}")


# ---------------------------------------------------------------- ladder data


@dataclass
class Rung:
    name: str
    cost_s: float
    decimals: int  # as printed in the report's table
    approx: bool
    sphere_spins: int
    water_spins: int | None  # None: water added from the cached template, not simulated


def _ladder_table() -> dict[str, dict]:
    """Rows of Table tab:e2-fidelity-ladder: Bloch spins and printed s/step per rung."""
    tex = LADDER_TEX.read_text()
    end = tex.index(r"\label{tab:e2-fidelity-ladder}")
    body = tex[tex.rindex(r"\begin{tabular}", 0, end):end]
    rows = {}
    for m in re.finditer(r"\\texttt\{(\w+)\}\s*&([^\n]*?)\\\\", body):
        cells = [c.strip() for c in m.group(2).split("&")]
        cost = cells[2]
        rows[m.group(1)] = {
            "spins": int(cells[1].replace("{,}", "")),
            "cost": float(re.sub(r"[^0-9.]", "", cost)),
            "approx": "approx" in cost,
            "decimals": len(re.sub(r"[^0-9.]", "", cost).split(".")[1]),
        }
    return rows


def _run_a_costs() -> dict[str, float]:
    """Seconds per RL step per stage of Run A.

    Each switch decision logs an EMA of s/step; the stage cost is their median, which is
    how the report's stage table reads them. The final rung logs no decisions, so its cost
    is the stage's wall time over its steps.
    """
    hist = read_json(RUN_A_HISTORY)
    costs = {}
    for stage in dict.fromkeys(e["stage"] for e in hist):
        spp = [e["sec_per_step"] for e in hist if e["stage"] == stage and e.get("kind") == "decision"]
        if spp:
            costs[stage] = statistics.median(spp)
            continue
        start = next(e for e in hist if e["stage"] == stage and e.get("kind") == "stage_start")
        end = next(e for e in hist if e["stage"] == stage and e.get("kind") == "stage_end")
        costs[stage] = (end["wall_s"] - start["wall_s"]) / (end["step"] - start["step"])
    return costs


def ladder() -> list[Rung]:
    table = _ladder_table()
    measured = _run_a_costs()
    counts = read_json(SPIN_COUNTS)
    sphere = counts["spheres_only"]
    water = {
        "analytic": 0,
        "cached3": None,
        "cached": None,
        "full3": counts["full_3mm"] - sphere,
        "full": counts["water_1mm"],
    }
    rungs = []
    for name in figstyle.RUNGS:
        row = table[name]
        n_sphere = 0 if name == "analytic" else sphere
        n_bloch = n_sphere + (water[name] or 0)
        if n_bloch != row["spins"]:
            raise ValueError(f"{name}: {n_bloch} Bloch spins from e2_spin_counts.json, table says {row['spins']}")
        # Run A skipped the cached 1 mm rung; the report gives it as approximately cached3.
        if name in measured:
            cost = measured[name]
            if round(cost, row["decimals"]) != row["cost"]:
                raise ValueError(f"{name}: Run A gives {cost:.4f} s/step, table says {row['cost']}")
        elif row["approx"]:
            cost = row["cost"]
        else:
            raise ValueError(f"{name}: no Run A measurement and no approximate table value")
        rungs.append(Rung(name, cost, row["decimals"], name not in measured, n_sphere, water[name]))
    return rungs


# ---------------------------------------------------------------- slab rendering


def slab_rgb(group: np.ndarray, value: np.ndarray, cmap, norm) -> np.ndarray:
    rgb = np.empty(group.shape + (3,))
    rgb[group < 0] = to_rgb(EMPTY)
    rgb[group == 0] = to_rgb(WATER_TINT)
    sphere = group > 0
    rgb[sphere] = plt.get_cmap(cmap)(norm(value[sphere]))[:, :3]
    return rgb


def draw_slab(ax, grid_mm, group, value, cmap, norm, spheres, half_mm):
    rgb = slab_rgb(group, value, cmap, norm)
    step = grid_mm[1] - grid_mm[0]
    extent = (grid_mm[0] - step / 2, grid_mm[-1] + step / 2) * 2
    ax.imshow(rgb, origin="lower", extent=extent, interpolation="none")
    for s in spheres:
        ax.add_patch(Circle(s["centre_mm"], s["radius_mm"], fill=False, lw=0.6, ec=figstyle.INK))
    ax.set_xlim(-half_mm, half_mm)
    ax.set_ylim(-half_mm, half_mm)
    ax.set_aspect("equal")
    ax.axis("off")


def scale_bar(ax, half_mm, length_mm=50):
    """Bottom-right corner, which the circular slab leaves empty."""
    x0, y0 = half_mm - 2 - length_mm, -half_mm + 2
    ax.plot([x0, x0 + length_mm], [y0, y0], color=figstyle.INK, lw=1.0, solid_capstyle="butt")
    ax.text(x0 + length_mm / 2, y0 + 2.5, f"{length_mm} mm", ha="center", va="bottom", fontsize=SMALL,
            color=figstyle.INK)


def occupied_half_width(groups, pad_mm, grid_mm):
    occ = np.any([g >= 0 for g in groups], axis=0)
    ys, xs = np.nonzero(occ)
    return max(np.abs(grid_mm[xs]).max(), np.abs(grid_mm[ys]).max()) + pad_mm


def fmt_signed(v: float, digits: int) -> str:
    s = f"{abs(v):.{digits}f}"
    return (MINUS + s) if round(v, digits) < 0 else s


# ---------------------------------------------------------------- panels


# Shortest form ("0", "0.5", "1") on every numeric scale in the figure.
SHORT_TICKS = FuncFormatter(lambda v, _: f"{v:g}")


def style_colourbar(cb, label, ticks):
    cb.set_ticks(ticks)
    cb.ax.xaxis.set_major_formatter(SHORT_TICKS)
    cb.ax.tick_params(labelsize=SMALL, length=2, pad=1.5, width=0.5)
    cb.outline.set_linewidth(0.5)
    cb.set_label(label, fontsize=SMALL, labelpad=1.5)


def t1_ticks(norm):
    return np.arange(norm.vmin, norm.vmax + 1e-9, 0.5)


def panel_twin(ax, cax, panel: PanelA, t1_norm):
    ax.imshow(panel.image, interpolation="lanczos")
    ax.axis("off")
    cb = plt.colorbar(ScalarMappable(norm=t1_norm, cmap=figstyle.T1_CMAP), cax=cax, orientation="horizontal",
                      extend=panel.colourbar_extend, extendfrac=0.04)
    style_colourbar(cb, panel.colourbar_label, t1_ticks(t1_norm))


def panel_b1(axes, slabs, meta, t1_norm, half):
    grid = slabs["grid_mm"]
    for i, (ax, d) in enumerate(zip(axes, meta["b1"])):
        s = d["seed"]
        draw_slab(ax, grid, slabs[f"b1_{s}_group"], slabs[f"b1_{s}_T1"], figstyle.T1_CMAP, t1_norm,
                  d["spheres"], half)
        active = {sp["label"] for sp in d["spheres"]}
        for pos in d["plate_positions"]:
            if pos["label"] not in active:
                ax.add_patch(Circle(pos["centre_mm"], pos["radius_mm"], fill=False, lw=0.6, ls=(0, (1, 1.5)),
                                    ec=REMOVED_SPHERE))
        ax.set_title(f"Training phantom {i + 1}", fontsize=SMALL + 0.5, pad=2)
        tx, ty = d["translation_mm"]
        ax.text(0.5, -0.02, f"shift ({fmt_signed(tx, 1)}, {fmt_signed(ty, 1)}) mm\n"
                f"rotation {fmt_signed(d['rotation_deg'], 1)}°",
                transform=ax.transAxes, ha="center", va="top", fontsize=SMALL, color=figstyle.INK_SECONDARY,
                linespacing=1.15)
    scale_bar(axes[-1], half)


def panel_b2(axes, caxes, slabs, meta, t1_norm, half):
    grid = slabs["grid_mm"]
    t2_top = float(np.ceil(max(s["T2_s"] for s in meta["b2"]["T2"]["spheres"])))
    scales = {
        "T1": ("T$_1$ plate", figstyle.T1_CMAP, t1_norm, "T$_1$ (s)", np.arange(t1_norm.vmin, t1_norm.vmax + 0.1, 1.0)),
        "T2": ("T$_2$ plate", GREY_CMAP, Normalize(0.0, t2_top), "T$_2$ (s)", np.arange(0.0, t2_top + 0.1, 1.0)),
        "PD": ("PD plate", GREY_CMAP, Normalize(0.0, 1.0), "PD (water fraction)", [0.0, 0.5, 1.0]),
    }
    for ax, cax, (plate, (title, cmap, norm, label, ticks)) in zip(axes, caxes, scales.items()):
        draw_slab(ax, grid, slabs[f"b2_{plate}_group"], slabs[f"b2_{plate}_value"], cmap, norm,
                  meta["b2"][plate]["spheres"], half)
        ax.set_title(title, fontsize=SMALL + 0.5, pad=2)
        cb = plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
        style_colourbar(cb, label, ticks)
    scale_bar(axes[-1], half)


def spin_label(r: Rung) -> str:
    water = "cached" if r.water_spins is None else f"{r.water_spins:,}"
    return f"{r.sphere_spins:,}\n+ {water}"


def panel_ladder(ax, rungs: list[Rung]):
    x = np.arange(len(rungs))
    costs = [r.cost_s for r in rungs]
    # The lightest rungs nearly vanish on white, so every bar carries a hairline in the darkest rung colour.
    ax.bar(x, costs, width=0.46, color=[figstyle.RUNG_COLOURS[r.name] for r in rungs],
           edgecolor=figstyle.RUNG_COLOURS["full"], linewidth=0.5, zorder=3)
    for xi, r in zip(x, rungs):
        ax.annotate(("≈" if r.approx else "") + f"{r.cost_s:.{r.decimals}f} s", (xi, r.cost_s), xytext=(0, 2),
                    textcoords="offset points", ha="center", va="bottom", fontsize=SMALL, color=figstyle.INK)
    ax.set_xticks(x)
    # Two lines ("cached" over "3 mm") so neighbouring rung names never touch.
    ax.set_xticklabels([figstyle.RUNG_LABELS[r.name].replace(" ", "\n", 1) for r in rungs],
                       fontsize=SMALL + 0.5, linespacing=1.0)
    ax.tick_params(axis="x", length=0, pad=2.5)
    spin_kw = dict(textcoords="offset points", va="top", fontsize=SMALL, color=figstyle.INK_SECONDARY,
                   linespacing=1.1, annotation_clip=False)
    for xi, r in zip(x, rungs):
        ax.annotate(spin_label(r), (xi, 0), xytext=(0, -SPIN_ROW_PT), ha="center", **spin_kw)
    ax.annotate("sphere spins\n+ water spins", (-0.5, 0), xytext=(-3, -SPIN_ROW_PT), ha="right", **spin_kw)
    ax.set_ylabel("Cost (s)", fontsize=SMALL + 0.5, labelpad=2)
    ax.set_ylim(0, max(costs) * 1.18)
    ax.set_yticks(np.arange(0, max(costs) + 0.01, 0.5))
    ax.yaxis.set_major_formatter(SHORT_TICKS)
    ax.tick_params(axis="y", labelsize=SMALL, length=2, width=0.5)
    ax.set_xlim(-0.5, len(rungs) - 0.5)


def k_ticks(v, _):
    return "0" if v == 0 else f"{v / 1000:g}k"


def panel_ladder_dual(ax, rungs: list[Rung]):
    """The presentation's ladder: stacked Bloch-spin bars (left axis), cost step line (right)."""
    x = np.arange(len(rungs))
    sphere = np.array([r.sphere_spins for r in rungs])
    water = np.array([r.water_spins or 0 for r in rungs])
    costs = np.array([r.cost_s for r in rungs])
    bar = dict(width=0.5, edgecolor="white", linewidth=0.6, zorder=2)
    ax.bar(x, sphere, color=SPHERE_SPINS, label="sphere spins", **bar)
    ax.bar(x, water, bottom=sphere, color=WATER_SPINS, label="water spins", **bar)
    for xi, r, total in zip(x, rungs, sphere + water):
        if total:
            ax.annotate(f"{total:,}", (xi, total), xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=SMALL, color=figstyle.INK)
    # Headroom keeps every bar and its label below the cost line, which rises with it.
    ax.set_ylim(0, LADDER_SPIN_AXIS_MAX)
    ax.yaxis.set_major_formatter(FuncFormatter(k_ticks))
    ax.set_ylabel("Bloch spins per step", fontsize=SMALL + 0.5, labelpad=2)
    ax.tick_params(axis="y", labelsize=SMALL, length=2, width=0.5)
    ax.set_xticks(x)
    labels = [figstyle.RUNG_LABELS[r.name].replace(" ", "\n", 1) for r in rungs]
    # The analytic rung simulates no spins, so its bar is empty; say why under it.
    labels[0] += "\n(closed form)"
    ax.set_xticklabels(labels, fontsize=SMALL + 0.5, linespacing=1.0, va="top")
    ax.tick_params(axis="x", length=0, pad=2.5)
    ax.set_xlim(-0.5, len(rungs) - 0.5)

    cost_ax = ax.twinx()
    cost_ax.spines["right"].set_visible(True)
    cost_ax.spines["right"].set_color(figstyle.ACCENT)
    cost_ax.step(x, costs, where="mid", color=figstyle.ACCENT, lw=1.4, zorder=4)
    cost_ax.plot(x, costs, "o", ms=4, color=figstyle.ACCENT, mec="white", mew=0.8, zorder=5,
                 label="cost per RL step")
    for xi, r in zip(x, rungs):
        cost_ax.annotate(("≈" if r.approx else "") + f"{r.cost_s:.{r.decimals}f} s", (xi, r.cost_s),
                         xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                         fontsize=SMALL, color=figstyle.ACCENT)
    cost_ax.set_ylim(0, LADDER_COST_AXIS_MAX)
    cost_ax.set_yticks(np.arange(0, LADDER_COST_AXIS_MAX + 1e-9, 0.5))
    cost_ax.yaxis.set_major_formatter(SHORT_TICKS)
    cost_ax.tick_params(axis="y", labelsize=SMALL, length=2, width=0.5, colors=figstyle.ACCENT)
    cost_ax.set_ylabel("Cost per RL step (s)", fontsize=SMALL + 0.5, color=figstyle.ACCENT, labelpad=3)

    handles = ax.get_legend_handles_labels()[0] + cost_ax.get_legend_handles_labels()[0]
    ax.legend(handles=handles, loc="upper left", fontsize=SMALL, handlelength=1.2, borderaxespad=0.2,
              labelspacing=0.3)


# ---------------------------------------------------------------- figure


def build(variant: str, panel_a: PanelA, rungs: list[Rung], slabs, slab_meta, twin_meta,
          dual_ladder: bool = False) -> plt.Figure:
    t1_norm = Normalize(*twin_meta["t1_range_s"])
    a_w = twin_meta["panel_width_in"]
    a_h = a_w * panel_a.image.shape[0] / panel_a.image.shape[1]

    # Vertical budget of the right column, top to bottom (inches).
    heading_h, title_h, slab, c_heading_h, c_plot_h, c_labels_h = 0.2, 0.17, 1.15, 0.24, 0.74, 0.6
    if dual_ladder:
        # No spin-count rows under the axis; the bars carry the counts.
        c_plot_h, c_labels_h = 0.92, 0.42
    under_slab_h = 0.3 if variant == "B1" else 0.44
    fig_h = heading_h + title_h + slab + under_slab_h + c_heading_h + c_plot_h + c_labels_h
    # savefig pads the tight bbox, so the canvas is narrower than the saved width.
    fig_w = SAVED_W_IN - 2 * mpl.rcParams["savefig.pad_inches"]
    fig = plt.figure(figsize=(fig_w, fig_h))

    def add(x, y, w, h):
        return fig.add_axes((x / fig_w, y / fig_h, w / fig_w, h / fig_h))

    top = fig_h - heading_h
    ax_a = add(0.0, top - a_h, a_w, a_h)
    cax_a = add(0.3, top - a_h - 0.12, a_w - 0.9, 0.07)
    panel_twin(ax_a, cax_a, panel_a, t1_norm)

    x_r = a_w + 0.2
    gap = (fig_w - x_r - 3 * slab) / 2
    slab_y = top - title_h - slab
    b_axes = [add(x_r + i * (slab + gap), slab_y, slab, slab) for i in range(3)]
    if variant == "B1":
        groups = [slabs[f"b1_{d['seed']}_group"] for d in slab_meta["b1"]]
        panel_b1(b_axes, slabs, slab_meta, t1_norm, occupied_half_width(groups, 2.0, slabs["grid_mm"]))
    else:
        groups = [slabs[f"b2_{p}_group"] for p in ("T1", "T2", "PD")]
        c_axes = [add(x_r + i * (slab + gap) + 0.15, slab_y - 0.13, slab - 0.3, 0.07) for i in range(3)]
        panel_b2(b_axes, c_axes, slabs, slab_meta, t1_norm, occupied_half_width(groups, 2.0, slabs["grid_mm"]))

    c_heading_y = slab_y - under_slab_h - c_heading_h
    if dual_ladder:
        ax_c = add(x_r + 0.45, c_labels_h, fig_w - x_r - 0.45 - 0.42, c_plot_h)
        panel_ladder_dual(ax_c, rungs)
    else:
        ax_c = add(x_r + 0.62, c_labels_h, fig_w - x_r - 0.62, c_plot_h)
        panel_ladder(ax_c, rungs)

    heading = dict(fontsize=SMALL + 1.5, color=figstyle.INK, va="bottom", ha="left")
    b_title = "Training randomisation, T$_1$-plate slab" if variant == "B1" else "Slices through the three plates"
    for x, y, letter, text in [
        (0.0, top, "A", f"Digital twin, {FIELD_LABELS[panel_a.field]}"),
        (x_r, top, "B", b_title),
        (x_r, c_heading_y + 0.04, "C", ("Fidelity ladder, 14-sphere plate, CPU" if dual_ladder
                                        else "Fidelity ladder: cost per RL step, 14-sphere plate, CPU")),
    ]:
        fig.text(x / fig_w, y / fig_h, letter, fontweight="bold", **heading)
        fig.text((x + 0.15) / fig_w, y / fig_h, text, **heading)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--viewer", action="store_true",
                        help="build the B1 figure with each viewer-screenshot candidate for panel A")
    args = parser.parse_args()

    figstyle.apply()
    rungs = ladder()
    for r in rungs:
        print(f"{r.name:9s} {r.cost_s:.4f} s/step  approx={r.approx}  spins={r.sphere_spins}+{r.water_spins}")
    slabs = np.load(SLABS_NPZ)
    slab_meta = read_json(SLABS_META)
    twin_meta = read_json(TWIN_META)
    if args.viewer:
        jobs = [("B1", "shotcrop", False, "fig1_B1_shotcrop"), ("B1", "shotreexport", True, "fig1_B1_shotreexport")]
    else:
        jobs = [(b, "render", False, f"fig1_{b}") for b in ("B1", "B2")]
    for variant, a, dual, stem in jobs:
        fig = build(variant, panel_a_source(a, twin_meta), rungs, slabs, slab_meta, twin_meta, dual)
        figstyle.save(fig, stem)
        plt.close(fig)
        print("wrote", figstyle.FIGURES / f"{stem}.png")


if __name__ == "__main__":
    main()

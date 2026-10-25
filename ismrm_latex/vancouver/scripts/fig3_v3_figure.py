"""Figure 3, v3: the TI-histogram policy over training, against the fixed protocols, and how its TIs
follow the scan's T1s.

(A) fig3_combo's training curve, with lighter rung shading under stronger faint elements.
(B) Executed (TI, TR) of the three checkpoints A follows, overlaid in one scatter.
(C) Running T1 MAPE within a scan for those checkpoints, the fixed log TI grid and the Cramer-Rao
    schedule, all on the 24 validation phantoms (seed 500000 + k).
(D) The selected policy's TI per block on the 24 validation scans (T1s spread as in training) and on
    24 clustered-T1 scans (control_narrow_band.jl, picked by evenly_by_centre), on one colour scale:
    the scan's mean sphere T1. The IR-SE shot inset of fig3_combo moves to Figure 1.

Inputs as fig3_ci_figure.py, plus figures/fig3_comparators_validation.json
(fig3_comparators_validation.jl) and figures/control_data/control_narrow_band.json.

Run from the RLxMRI repo root:
  .venv/bin/python ismrm_latex/vancouver/scripts/fig3_v3_figure.py

Writes figures/fig3_combo_v3.{png,pdf}.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import figstyle
import matplotlib.pyplot as plt
import numpy as np
from fig3_ci_figure import (
    MAX_CAPTION_CHARS,
    N_SCANS,
    Row,
    axis_max,
    evenly_by_centre,
    load_narrow_band,
    load_rows,
    plotted,
)
from fig3_figure import (
    CM,
    FIG_W,
    RUN,
    SMALL,
    PanelAContrast,
    ScanLines,
    ValidationScans,
    add_axes_cm,
    add_colourbar,
    draw_panel_a,
    draw_panel_b,
    load_scans,
    load_training_curve,
    read_json,
)
from fig3evo_figure import (
    Checkpoint,
    Reference,
    draw_running_mape,
    executed_ti_tr,
    format_hours,
    format_ti_tr_axes,
    load_checkpoints,
)
from matplotlib.colors import LogNorm

COMPARATORS = figstyle.FIGURES / "fig3_comparators_validation.json"
# figstyle key -> (name in fig3_comparators_validation.json, legend text), top to bottom as in A.
# Two-line labels keep C's legend in the empty corner right of the Cramer-Rao plateau.
REFERENCE_RUNS = {"cr_optimal": ("cr_optimal", "Cramér-Rao\noptimal"),
                  "log_grid": ("log_grid_trmatched", "Fixed log\nTI grid")}
CONTRAST = PanelAContrast(rung_tint=0.12, faint_tint=0.6, faint_lw=1.0, reference_lw=1.1)
# Unclipped, so a marker at the 3 s TI ceiling stays whole.
D_LINE_KW = {"lw": 0.9, "marker": "o", "ms": 1.8, "alpha": 0.9, "clip_on": False}
# Axes-fraction rectangle in D's left subplot, above its lines from block 8 on.
COLOURBAR_AT = (0.5, 0.8, 0.42, 0.035)
# D's subplot titles clear a marker at the 3 s TI ceiling.
D_TITLE_PAD_PT = 5.0
LEFT = 1.35
# Panel letters this far left of their axes, as draw_panel_a places A (-0.25 of its 4.5 cm axes).
LABEL_X_CM = 1.125


# ---------------------------------------------------------------- data


def load_references(cps: list[Checkpoint]) -> list[Reference]:
    data = read_json(COMPARATORS)
    assert data["seed_base"] == 500_000 and data["n_scans"] == N_SCANS
    refs = [Reference(key, data["comparators"][name]["episodes"], label)
            for key, (name, label) in REFERENCE_RUNS.items()]
    for ref in refs:
        for cp in cps:
            assert all(a["seed"] == b["seed"] and a["T1_true"] == b["T1_true"] for a, b in zip(ref.episodes, cp.episodes))
    return refs


@dataclass
class PanelDData:
    spread: ValidationScans
    clustered: ValidationScans
    norm: LogNorm


def load_panel_d() -> PanelDData:
    """The validation scans, and 24 clustered-T1 scans picked by band centre; both coloured by mean sphere T1."""
    data, every = load_narrow_band()
    mean_t1 = {c: float(np.mean(r["T1_true"])) for c, r in zip(data["T1_band_centre"], data["arms"]["adaptive"])}
    assert all(abs(m / c - 1) <= data["t1_band_rel"] for c, m in mean_t1.items())
    picked = evenly_by_centre(every, N_SCANS)
    clustered = replace(picked, scans=[replace(s, colour_t1_s=mean_t1[s.colour_t1_s]) for s in picked.scans])
    return PanelDData(load_scans(), clustered, LogNorm(*data["centre_range_s"]))


def log_correlation(vs: ValidationScans) -> float:
    """Pearson r between log shortest TI and log mean sphere T1, over the scans."""
    return float(np.corrcoef(np.log([s.ti_s.min() for s in vs.scans]), np.log([s.colour_t1_s for s in vs.scans]))[0, 1])


# ---------------------------------------------------------------- panels


def draw_ti_tr_overlaid(ax: plt.Axes, cps: list[Checkpoint]) -> None:
    # The selected checkpoint last, so its cloud stays on top.
    for i, cp in enumerate(cps):
        ax.scatter(*executed_ti_tr(cp), s=8, color=cp.colour, alpha=0.6, lw=0, zorder=2 + i, clip_on=False,
                   label=format_hours(cp.hours))
    format_ti_tr_axes(ax)
    ax.set_ylabel("TR (s)")
    ax.legend(loc="upper left", fontsize=SMALL, handletextpad=0.1, borderaxespad=0.3, markerscale=1.6,
              title="checkpoint", title_fontsize=SMALL, alignment="left")


def draw_panel_d(fig: plt.Figure, left: float, bottom: float, width: float, height: float, gap: float,
                 d: PanelDData) -> list[plt.Axes]:
    """Two subplots on one block axis and one TI axis, with one colourbar."""
    n_blocks = max(len(s.ti_s) for vs in (d.spread, d.clustered) for s in vs.scans)
    ticks = [t for t in (0.03, 0.1, 0.3, 1.0) if d.norm.vmin <= t <= d.norm.vmax]
    lines = ScanLines(d.norm, "scan's mean sphere T$_1$ (s)", ticks, [f"{t:g}" for t in ticks], colourbar_at=None,
                      line_kw=D_LINE_KW)
    w = (width - gap) / 2
    titles = ("T$_1$s spread (as in training)", "clustered T$_1$s (not seen in training)")
    axes = []
    for i, (vs, title) in enumerate(zip((d.spread, d.clustered), titles)):
        ax = add_axes_cm(fig, left + i * (w + gap), bottom, w, height)
        draw_panel_b(ax, vs, lines=lines, n_blocks=n_blocks, vertical_probe_label=True)
        ax.set_title(title, fontsize=SMALL + 1, loc="left", pad=D_TITLE_PAD_PT)
        ax.set_xticks([1, *range(5, n_blocks + 1, 5)])
        ax.set_xticks(range(1, n_blocks + 1), minor=True)
        ax.set_xlabel("")
        x_after_probe = (vs.opening_probe_blocks + 0.5 - ax.get_xlim()[0]) / np.diff(ax.get_xlim())[0] + 0.02
        ax.text(x_after_probe, 0.03, f"shortest TI vs mean T$_1$ (log-log): r = {log_correlation(vs):.2f}, "
                f"n = {len(vs.scans)}", transform=ax.transAxes, ha="left", va="bottom", fontsize=SMALL,
                color=figstyle.INK)
        axes.append(ax)
    axes[1].set_ylabel("")
    axes[1].tick_params(labelleft=False)
    add_colourbar(axes[0], lines, COLOURBAR_AT)
    figstyle.panel_label(axes[0], "D", x=-LABEL_X_CM / w, y=1 + D_TITLE_PAD_PT / 72 * 2.54 / height, va="baseline")
    fw, fh = fig.get_size_inches() / CM
    fig.text((left + width / 2) / fw, (bottom - 0.62) / fh, f"Block ({d.spread.shots_per_block} shots each)",
             ha="center", va="top", fontsize=9)
    return axes


# ---------------------------------------------------------------- layout


def v3_figure(rows: list[Row], d: PanelDData) -> plt.Figure:
    cps = load_checkpoints()
    kw = read_json(RUN / "run_config.json")["base_env_kwargs"]
    fig = plt.figure(figsize=(FIG_W * CM, 16.7 * CM))
    draw_panel_a(fig, left=LEFT, bottom=13.3, height=3.0, data=load_training_curve(),
                 checkpoints=plotted(rows, True), ymax=axis_max(rows), contrast=CONTRAST)

    row_b, row_h = 8.25, 3.35
    w_b, gap, w_c = 6.2, 1.55, 6.9
    ax_b = add_axes_cm(fig, LEFT, row_b, w_b, row_h)
    draw_ti_tr_overlaid(ax_b, cps)
    figstyle.panel_label(ax_b, "B", x=-LABEL_X_CM / w_b, y=1.04)
    ax_c = add_axes_cm(fig, LEFT + w_b + gap, row_b, w_c, row_h)
    draw_running_mape(ax_c, cps, float(kw["time_budget_s"]), int(kw["Npe"]), note="median and IQR, 24 scans",
                      references=load_references(cps), reference_lw=CONTRAST.reference_lw)
    figstyle.panel_label(ax_c, "C", x=-LABEL_X_CM / w_c, y=1.04)

    draw_panel_d(fig, LEFT, 1.0, FIG_W - LEFT - 0.2, 5.55, 0.35, d)
    return fig


def caption(rows: list[Row]) -> str:
    early = next(r for r in rows if r.name == "stage0_analytic_end")
    sel = next(r for r in rows if r.kind == "selected")
    return (
        f"TI-histogram policy over training, {N_SCANS} validation phantoms unless stated. (A) T1 MAPE of each "
        "saved checkpoint, full Bloch (mean, bootstrap 95% CI); faint: logged probes; star, lines: selected "
        "checkpoint, fixed protocols on 24 held-out phantoms. (B) Executed (TI, TR) of coloured checkpoints: "
        f"pairing short TIs with short TRs fits {sel.blocks:.0f} blocks, not {early.blocks:.0f}. (C) T1 MAPE within "
        "a scan (median, IQR); grey: fixed protocols. (D) Selected policy's TIs on spread and clustered-T1 scans, "
        "coloured by mean sphere T1."
    )


def main() -> None:
    figstyle.apply()
    rows = load_rows()
    d = load_panel_d()
    fig = v3_figure(rows, d)
    figstyle.save(fig, "fig3_combo_v3")
    plt.close(fig)

    for name, vs in (("spread", d.spread), ("clustered", d.clustered)):
        print(f"{name}: r = {log_correlation(vs):.3f} over {len(vs.scans)} scans")
    cap = caption(rows)
    assert len(cap) <= MAX_CAPTION_CHARS, len(cap)
    print(f"\nfig3_combo_v3 caption ({len(cap)} characters):\n{cap}")


if __name__ == "__main__":
    main()

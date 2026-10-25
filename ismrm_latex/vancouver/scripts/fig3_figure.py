"""Figure 3: training curve, TI choices and the IR-SE shot the policy controls.

(A) Full-Bloch T1 MAPE versus training wall-clock for the TI-histogram run (560 s),
    with the fidelity rungs shaded and the selected checkpoint's held-out MAPE.
(B) TI chosen at each block of the 24 validation scans (seed 500000).
(C) Inset in (B): one shot of the IR-SE block, from the waveforms that
    fig3_sequence_dump.jl samples out of the real KomaMRI sequence.

Writes figures/fig3.{png,pdf} and the single panels fig3_A, fig3_B (with the
inset) and fig3_C.

Run from the RLxMRI repo root:
  julia --project=. ismrm_latex/vancouver/scripts/fig3_sequence_dump.jl
  .venv/bin/python ismrm_latex/vancouver/scripts/fig3_figure.py
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, to_hex, to_rgb
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D

import figstyle

RUN = figstyle.ROOT / "runs" / "e2" / "mf_runB_5sphere_hist_560s_gpu"
BASELINES = figstyle.ROOT / "runs" / "e2" / "baselines_heldout_560s" / "baseline_summary.json"
WAVEFORMS = figstyle.FIGURES / "fig3_sequence_waveforms.json"

CM = 1 / 2.54
FIG_W = 17.0
SMALL = 7.0
T1_NORM = Normalize(*figstyle.T1_NORM)
MAPE_YMAX = 12.5
# TI axis floor: below the smallest TI chosen (0.051 s), above the action floor
# (0.01 s), so the inset gets the empty top of the panel at a legible size.
TI_YLIM = (0.03, 3.0)
INSET_FROM_BLOCK = 7.6
PROBE_SHADE = "#EDEDED"
# Rung backgrounds are RUNG_COLOURS blended toward white: the full rung fills most
# of panel A, so even the darkest step must stay a pale tint behind the data.
RUNG_TINT = 0.2
# The logged probes and evaluations, once re-rolled checkpoints carry the curve.
FAINT_TINT = 0.4


@dataclass(frozen=True)
class PanelAContrast:
    """Panel A's rung shading and the weight of the elements drawn over it."""
    rung_tint: float = RUNG_TINT
    faint_tint: float = FAINT_TINT
    faint_lw: float = 0.8
    reference_lw: float = 0.9


DEFAULT_CONTRAST = PanelAContrast()


def read_json(path: Path):
    return json.loads(path.read_text())


def tint(colour: str, strength: float) -> str:
    """Blend a colour toward white; strength 1 keeps it, 0 gives white."""
    rgb = np.asarray(to_rgb(colour))
    return to_hex(1.0 - strength * (1.0 - rgb))


# ---------------------------------------------------------------- data


@dataclass
class TrainingCurve:
    rungs: list[tuple[str, float, float]]
    probe_h: np.ndarray
    probe_mape: np.ndarray
    eval_h: np.ndarray
    eval_mape: np.ndarray
    selected_h: float
    selected_mape: float
    n_probe: int
    n_eval: int
    heldout_mape: float
    n_heldout: int
    baselines: dict[str, float]


def ppo_start_wall_s() -> float:
    # SB3 opens the first tensorboard event file (named by Unix time) when PPO
    # starts learning; fidelity_history.json only begins at the first probe.
    files = (RUN / "tb").glob("*/PPO_*/events.out.tfevents.*")
    return float(min(int(f.name.split(".")[3]) for f in files))


def load_training_curve() -> TrainingCurve:
    config = read_json(RUN / "run_config.json")
    plan = config["plan"]
    history = read_json(RUN / "fidelity_history.json")
    t0 = ppo_start_wall_s()

    def hours(wall_s: float) -> float:
        return (wall_s - t0) / 3600.0

    starts = {r["stage"]: r["wall_s"] for r in history if r["kind"] == "stage_start"}
    starts[plan[0]] = t0
    bounds = [starts[s] for s in plan] + [max(r["wall_s"] for r in history)]
    rungs = [(s, hours(bounds[i]), hours(bounds[i + 1])) for i, s in enumerate(plan)]

    # A stage-end probe re-scores the checkpoint of that stage's last decision on
    # the same seeds, so keep one point per (stage, step).
    probes: dict[tuple[str, int], tuple[float, float]] = {}
    for r in history:
        if r["kind"] == "decision":
            probes[(r["stage"], r["step"])] = (r["wall_s"], r["mape_H_pct"])
    for r in history:
        if r["kind"] == "stage_end":
            probes.setdefault((r["stage"], r["step"]), (r["wall_s"], r["full_mape_pct"]))
    probe = np.array(sorted(probes.values()))

    final_stage = plan[-1]
    evals = read_json(RUN / f"stage{len(plan) - 1}_{final_stage}" / "eval_history.json")
    # eval_history wall_s counts from the eval callback's creation at the stage start.
    eval_h = np.array([hours(starts[final_stage] + e["wall_s"]) for e in evals])
    eval_mape = np.array([e["mape_pct"] for e in evals])

    best = read_json(RUN / "global_best" / "best_meta.json")
    i_sel = next(i for i, e in enumerate(evals) if e["step"] == best["step"])

    heldout = read_json(RUN / "global_best" / "eval_summary.json")
    base = read_json(BASELINES)
    return TrainingCurve(
        rungs=rungs,
        probe_h=np.array([hours(w) for w in probe[:, 0]]),
        probe_mape=probe[:, 1],
        eval_h=eval_h,
        eval_mape=eval_mape,
        selected_h=float(eval_h[i_sel]),
        selected_mape=float(eval_mape[i_sel]),
        n_probe=int(config["mf"]["mf_probe_episodes_full"]),
        n_eval=int(best["source_n_eval_episodes"]),
        heldout_mape=float(heldout["agent_mape_pct"]),
        n_heldout=int(heldout["episodes"]),
        baselines={
            "log_grid": float(base["log_grid_trmatched"]["mape_pct"]),
            "cr_optimal": float(base["cr_optimal"]["mape_pct"]),
        },
    )


@dataclass
class CheckpointMape:
    """A checkpoint re-rolled on the validation scans: mean MAPE and its 95% CI, in %."""
    hours: float
    mape: float
    lo: float
    hi: float
    n_scans: int
    selected: bool
    colour: str = figstyle.ACCENT
    # Featured checkpoints are filled in their colour; the rest are open, so a few can be followed
    # across panels without the others competing.
    featured: bool = True
    # Direct label for a featured checkpoint that other panels follow (e.g. its training time).
    label: str | None = None


@dataclass
class Scan:
    ti_s: np.ndarray
    # The T1 its line is coloured by: the scan's mean sphere T1, or a narrow band's centre.
    colour_t1_s: float


@dataclass
class ValidationScans:
    scans: list[Scan]
    shots_per_block: int
    opening_probe_blocks: int


@dataclass
class ScanLines:
    """How panel B draws each scan: the T1 scale that colours it (Scan.colour_t1_s) and the line style."""
    norm: Normalize
    colourbar_label: str
    colourbar_ticks: Sequence[float]
    # None keeps the colourbar's own tick labels.
    colourbar_tick_labels: Sequence[str] | None = None
    # Axes-fraction rectangle of the colourbar; None leaves it to the caller.
    colourbar_at: Sequence[float] | None = (0.72, 0.1, 0.25, 0.03)
    line_kw: dict = field(default_factory=lambda: {"lw": 1.0, "marker": "o", "ms": 2.2, "alpha": 0.9})


MEAN_T1_LINES = ScanLines(T1_NORM, "scan's mean sphere T$_1$ (s)", np.arange(T1_NORM.vmin, T1_NORM.vmax + 1e-9, 0.5))


def opening_probe_blocks(scans: list[Scan]) -> int:
    """Leading blocks with the same TI in every scan."""
    n = 0
    while all(len(s.ti_s) > n and np.isclose(s.ti_s[n], scans[0].ti_s[n], rtol=1e-6) for s in scans):
        n += 1
    return n


def load_scans() -> ValidationScans:
    kw = read_json(RUN / "run_config.json")["base_env_kwargs"]
    budget_s, npe = float(kw["time_budget_s"]), int(kw["Npe"])
    episodes = read_json(RUN / "global_best" / "diagnostics" / "episodes.json")
    scans = []
    for ep in episodes:
        n = len(ep["TI"])
        # diagnose_e2.py also records the final proposal that the budget guard
        # rejected (julia/rl/e2.jl:888-893); it was never acquired.
        if npe * sum(ep["TR"]) > budget_s:
            n -= 1
        scans.append(Scan(np.asarray(ep["TI"][:n]), float(np.mean(ep["T1_true"]))))
    return ValidationScans(scans, npe, opening_probe_blocks(scans))


# ---------------------------------------------------------------- panel A


def add_axes_cm(fig: plt.Figure, left: float, bottom: float, width: float, height: float) -> plt.Axes:
    fw, fh = fig.get_size_inches() / CM
    return fig.add_axes([left / fw, bottom / fh, width / fw, height / fh])


def draw_break(ax_left: plt.Axes, ax_right: plt.Axes, size_cm: float = 0.12) -> None:
    for ax, x in ((ax_left, 1.0), (ax_right, 0.0)):
        bbox = ax.get_position()
        fw, fh = ax.figure.get_size_inches() / CM
        dx, dy = size_cm / (bbox.width * fw), size_cm / (bbox.height * fh)
        ax.plot([x - dx / 2, x + dx / 2], [-dy, dy], transform=ax.transAxes, color=figstyle.INK_SECONDARY,
                lw=0.8, clip_on=False)


def draw_panel_a(fig: plt.Figure, left: float, bottom: float, height: float, data: TrainingCurve,
                 checkpoints: list[CheckpointMape] | None = None, ymax: float = MAPE_YMAX,
                 contrast: PanelAContrast = DEFAULT_CONTRAST) -> None:
    """Broken time axis: the cheap rungs stretched, then the full rung, then the held-out result.

    With checkpoints, the logged probes and evaluations recede and the re-rolled checkpoints
    (mean and 95% CI) become the curve.
    """
    w_cheap, w_gap, w_full, w_ho_gap, w_ho = 4.5, 0.25, 6.75, 0.35, 0.8
    ax_cheap = add_axes_cm(fig, left, bottom, w_cheap, height)
    ax_full = add_axes_cm(fig, left + w_cheap + w_gap, bottom, w_full, height)
    ax_ho = add_axes_cm(fig, left + w_cheap + w_gap + w_full + w_ho_gap, bottom, w_ho, height)

    full_start, end_h = data.rungs[-1][1], data.rungs[-1][2]
    ax_cheap.set_xlim(0, full_start)
    ax_full.set_xlim(full_start, end_h + 0.12)
    ax_ho.set_xlim(0, 1)
    for ax in (ax_cheap, ax_full, ax_ho):
        ax.set_ylim(0, ymax)
    ax_full.spines["left"].set_visible(False)
    ax_full.tick_params(axis="y", left=False, labelleft=False)
    for side in ("left", "bottom"):
        ax_ho.spines[side].set_visible(False)
    ax_ho.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)

    label_kw = dict(fontsize=SMALL, color=figstyle.INK, linespacing=1.0)
    for name, h0, h1 in data.rungs:
        ax = ax_full if h0 >= full_start else ax_cheap
        ax.axvspan(h0, h1, color=tint(figstyle.RUNG_COLOURS[name], contrast.rung_tint), lw=0, zorder=0)
        if h0 > ax.get_xlim()[0]:
            ax.axvline(h0, color="white", lw=1.0, zorder=0.5)
        label = figstyle.RUNG_LABELS[name]
        span = (h1 - h0) / np.diff(ax.get_xlim())[0]
        if span < 0.1:
            ax.annotate(label, xy=(h0, 0.25), xytext=(1.5, 0), textcoords="offset points", rotation=90,
                        ha="left", va="bottom", **label_kw)
        elif ax is ax_full:
            ax.text(h1 - 0.1, 0.35, label, ha="right", va="bottom", **label_kw)
        else:
            ax.text((h0 + h1) / 2, 0.35, label.replace(" ", "\n", 1), ha="center", va="bottom", **label_kw)

    for key, value in data.baselines.items():
        for ax in (ax_cheap, ax_full, ax_ho):
            ax.axhline(value, color=figstyle.METHOD_COLOURS[key], ls=figstyle.METHOD_LINESTYLES[key],
                       lw=contrast.reference_lw, zorder=1)
    upper, lower = sorted(data.baselines, key=data.baselines.get, reverse=True)
    for key, va, dy in ((upper, "bottom", 0.15), (lower, "top", -0.15)):
        value = data.baselines[key]
        ax_ho.text(1.08, value + dy, f"{figstyle.METHOD_LABELS[key]}\n{value:.2f}%",
                   transform=ax_ho.get_yaxis_transform(), ha="left", va=va, **label_kw)

    faint = checkpoints is not None
    probe_kw = dict(marker="o", ms=3.6, mfc="white", mec=figstyle.ACCENT, mew=1.0, ls="none")
    eval_kw = dict(color=figstyle.ACCENT, lw=1.3, marker="o", ms=2.6)
    if faint:
        probe_kw.update(ms=3.0, mew=0.7, mec=tint(figstyle.ACCENT, contrast.faint_tint))
        eval_kw.update(lw=contrast.faint_lw, ms=0, color=tint(figstyle.ACCENT, contrast.faint_tint))
    for ax in (ax_cheap, ax_full):
        h0, h1 = ax.get_xlim()
        inside = (data.probe_h >= h0) & (data.probe_h < h1)
        shown = inside & (data.probe_mape <= ymax)
        ax.plot(data.probe_h[shown], data.probe_mape[shown], zorder=3, clip_on=False, **probe_kw)
        for h, m in zip(data.probe_h[inside & ~shown], data.probe_mape[inside & ~shown]):
            ax.plot([h], [ymax], zorder=3, clip_on=False, **{**probe_kw, "marker": "^", "ms": 4.4})
            ax.annotate(f"{m:.1f}%", xy=(h, ymax), xytext=(4, 0), textcoords="offset points",
                        ha="left", va="center", fontsize=SMALL, color=figstyle.INK)
    ax_full.plot(data.eval_h, data.eval_mape, zorder=4, **eval_kw)

    sel_h, sel_mape, sel_label = data.selected_h, data.selected_mape, None
    ckpt_handle = None
    followed = []
    for cp in checkpoints or []:
        ax = ax_full if cp.hours >= full_start else ax_cheap
        style = dict(mfc=cp.colour, mec="white", mew=0.6, ms=4.6) if cp.featured else \
            dict(mfc="white", mec=cp.colour, mew=0.9, ms=3.8)
        handle = ax.errorbar([cp.hours], [cp.mape], yerr=[[cp.mape - cp.lo], [cp.hi - cp.mape]], fmt="o",
                             color=cp.colour, elinewidth=1.1, capsize=2.2, capthick=1.1, zorder=6 if cp.featured else 5,
                             clip_on=False, **style)
        if ckpt_handle is None or not cp.featured:
            ckpt_handle = handle
        if cp.label is not None:
            followed.append(cp)
        if cp.selected:
            sel_h, sel_mape, sel_label = cp.hours, cp.mape, cp.label
        elif cp.label is not None:
            x0, x1 = ax.get_xlim()
            if (cp.hours - x0) / (x1 - x0) > 0.85:
                ax.annotate(cp.label, xy=(cp.hours, cp.hi), xytext=(-2, 3), textcoords="offset points",
                            ha="right", va="bottom", **label_kw)
            else:
                ax.annotate(cp.label, xy=(cp.hours, cp.mape), xytext=(5, 0), textcoords="offset points",
                            ha="left", va="center", **label_kw)
    ring_ms = 9.0 if faint else 7.5
    ax_full.plot([sel_h], [sel_mape], marker="o", ms=ring_ms, mfc="none", mec=figstyle.INK, mew=1.0, zorder=7)
    sel_text = "selected\ncheckpoint" + (f", {sel_label}" if sel_label else "")
    ax_full.annotate(sel_text, xy=(sel_h, sel_mape), xytext=(sel_h + 0.35, 0.744 * ymax),
                     ha="center", va="bottom",
                     arrowprops=dict(arrowstyle="-", lw=0.6, color=figstyle.INK, shrinkA=1,
                                     shrinkB=ring_ms / 2 + 0.25),
                     **label_kw)

    ax_ho.plot([0.4], [data.heldout_mape], marker="*", ms=10, color=figstyle.ACCENT, mec="white", mew=0.6,
               zorder=6)
    if ymax > MAPE_YMAX:
        # A taller value range packs the reference-line labels down onto the star's value.
        ax_ho.annotate(f"{data.heldout_mape:.2f}%", xy=(0.4, data.heldout_mape), xytext=(0, -6),
                       textcoords="offset points", ha="center", va="top", fontsize=SMALL, color=figstyle.INK,
                       fontweight="bold")
    else:
        ax_ho.text(1.08, data.heldout_mape, f"{data.heldout_mape:.2f}%", transform=ax_ho.get_yaxis_transform(),
                   ha="left", va="center", fontsize=SMALL, color=figstyle.INK, fontweight="bold")
    ax_ho.text(0.4, -0.05, f"held-out\n{data.n_heldout} scans", transform=ax_ho.get_xaxis_transform(),
               ha="center", va="top", **label_kw)

    ax_cheap.set_xticks([0, 0.2, 0.4, 0.6])
    ax_full.set_xticks([1, 3, 5, 7, 9])
    ax_cheap.set_yticks(range(0, int(ymax) + 1, 2 if ymax <= 14 else 5))
    ax_cheap.set_ylabel("T$_1$ MAPE (%)")
    draw_break(ax_cheap, ax_full)
    fw, fh = fig.get_size_inches() / CM
    fig.text((left + (w_cheap + w_gap + w_full) / 2) / fw, (bottom - 0.62) / fh, "Training wall-clock (h)",
             ha="center", va="top", fontsize=9)

    handles = [
        Line2D([], [], label=f"validation, {data.n_probe} scans", **probe_kw),
        Line2D([], [], label=f"validation, {data.n_eval} scans", **eval_kw),
    ]
    labels = [h.get_label() for h in handles]
    if faint:
        handles.insert(0, ckpt_handle)
        labels.insert(0, f"validation, {checkpoints[0].n_scans} scans, 95% CI")
    if followed:
        handles.insert(1, tuple(Line2D([], [], ls="none", marker="o", ms=4.6, mfc=cp.colour, mec="white", mew=0.6)
                                for cp in followed))
        labels.insert(1, "colour: followed in B and C")
    ax_full.legend(handles=handles, labels=labels, loc="upper left", bbox_to_anchor=(0.0, 1.03),
                   ncol=1 if faint else len(handles), fontsize=SMALL, handlelength=1.5, handletextpad=0.4,
                   columnspacing=1.0, borderaxespad=0.0, handler_map={tuple: HandlerTuple(ndivide=None, pad=0.1)})
    figstyle.panel_label(ax_cheap, "A", x=-0.25, y=1.0)


# ---------------------------------------------------------------- panel B


def draw_panel_b(ax: plt.Axes, vs: ValidationScans, compact: bool = False, lines: ScanLines = MEAN_T1_LINES,
                 n_blocks: int | None = None, vertical_probe_label: bool = False) -> None:
    """n_blocks: the block axis length, by default the longest scan's; vertical_probe_label runs the probe's
    label up the shaded band, for blocks too narrow to hold it across."""
    n_max = n_blocks or max(len(s.ti_s) for s in vs.scans)
    ax.axvspan(0.5, vs.opening_probe_blocks + 0.5, color=PROBE_SHADE, lw=0, zorder=0)
    cmap = plt.get_cmap(figstyle.T1_CMAP)
    for s in sorted(vs.scans, key=lambda s: s.colour_t1_s):
        blocks = np.arange(1, len(s.ti_s) + 1)
        ax.plot(blocks, s.ti_s, color=cmap(lines.norm(s.colour_t1_s)), zorder=2, **lines.line_kw)
    ax.set_yscale("log")
    ax.set_ylim(*TI_YLIM)
    ax.set_xlim(0.5, n_max + 0.5)
    ax.set_xticks(range(1, n_max + 1))
    ticks = [t for t in (0.01, 0.03, 0.1, 0.3, 1, 3) if TI_YLIM[0] <= t <= TI_YLIM[1]]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:g}" for t in ticks])
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel(f"Block ({vs.shots_per_block} shots each)")
    ax.set_ylabel("TI (s)")
    probe_x = (vs.opening_probe_blocks + 1) / 2
    if vertical_probe_label:
        # Every scan's probe TIs lie above 0.3 s, so the band's lower half is free.
        ax.text(probe_x, 0.03, "fixed opening probe", transform=ax.get_xaxis_transform(), rotation=90,
                ha="center", va="bottom", fontsize=SMALL, color=figstyle.INK)
    else:
        ax.text(probe_x, 0.97, "fixed\nopening\nprobe", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=SMALL, color=figstyle.INK, linespacing=1.0)
    if lines.colourbar_at is not None:
        add_colourbar(ax, lines, lines.colourbar_at, title_loc="right" if compact else "left")


def add_colourbar(ax: plt.Axes, lines: ScanLines, rect: Sequence[float], title_loc: str = "left") -> plt.Axes:
    """lines' T1 colour scale as a horizontal bar at the axes-fraction rect, titled above."""
    cax = ax.inset_axes(rect)
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=lines.norm, cmap=plt.get_cmap(figstyle.T1_CMAP)), cax=cax,
                      orientation="horizontal")
    cb.outline.set_visible(False)
    cb.set_ticks(lines.colourbar_ticks, labels=lines.colourbar_tick_labels)
    # A log scale would otherwise add unlabelled minor ticks.
    cb.minorticks_off()
    cb.ax.tick_params(labelsize=SMALL, length=2, pad=1)
    cax.set_title(lines.colourbar_label, fontsize=SMALL, pad=2.5, loc=title_loc)
    return cax


def inset_rect(ax: plt.Axes, scans: list[Scan], x_start: float, clearance_decades: float = 0.05) -> list[float]:
    """Axes-fraction rectangle right of x_start that clears every TI line."""
    x0, x1 = ax.get_xlim()
    y0, y1 = np.log10(ax.get_ylim())
    xs = np.linspace(x_start, x1, 400)
    top = -np.inf
    for s in scans:
        blocks = np.arange(1, len(s.ti_s) + 1)
        inside = (xs >= blocks[0]) & (xs <= blocks[-1])
        if inside.any():
            top = max(top, np.interp(xs[inside], blocks, np.log10(s.ti_s)).max())
    bottom = (top + clearance_decades - y0) / (y1 - y0)
    left = (x_start - x0) / (x1 - x0)
    return [left, bottom, 1.0 - left, 1.0 - bottom]


# ---------------------------------------------------------------- panel C


class BrokenTime:
    """Piecewise time axis: every visible window at one scale, each gap at a fixed width."""

    def __init__(self, windows_ms: list[tuple[float, float]], gap: float):
        self.windows = windows_ms
        self.gap = gap
        self.offsets = []
        x = 0.0
        for t0, t1 in windows_ms:
            self.offsets.append(x)
            x += (t1 - t0) + gap
        self.width = x - gap

    def __call__(self, t_ms: float) -> float:
        for (t0, t1), off in zip(self.windows, self.offsets):
            if t0 - 1e-9 <= t_ms <= t1 + 1e-9:
                return off + t_ms - t0
        raise ValueError(f"t = {t_ms} ms is not in a visible window")

    def spans(self) -> list[tuple[float, float]]:
        return [(off, off + t1 - t0) for (t0, t1), off in zip(self.windows, self.offsets)]

    def gaps(self) -> list[tuple[float, float]]:
        return [(b, b + self.gap) for _, b in self.spans()[:-1]]


def draw_sequence(ax: plt.Axes, wf: dict, compact: bool = False) -> None:
    """compact drops the example values from the TI and TE labels so they fit a narrow inset."""
    ms = 1e3
    inv, exc, ref = (e["centre_s"] * ms for e in wf["rf_events"])
    nxt = wf["next_inversion"]["centre_s"] * ms
    adc_c, adc_d = wf["adc"]["centre_s"] * ms, wf["adc"]["dur_s"] * ms
    pad = 0.9
    bt = BrokenTime([(inv - pad, inv + pad), (exc - pad, adc_c + adc_d / 2 + pad), (nxt - pad, nxt + pad)],
                    gap=7.0)

    rows = {"RF": 4.8, "Gx": 3.4, "Gy": 2.0, "ADC": 0.6}
    amp = {"RF": 0.95, "Gx": 0.85, "Gy": 0.65, "ADC": 0.55}
    ink = figstyle.INK_SECONDARY

    for name, y in rows.items():
        for a, b in bt.spans():
            ax.plot([a, b], [y, y], color=ink, lw=0.6, solid_capstyle="butt")
        for a, b in bt.gaps():
            ax.plot([a, b], [y, y], color=ink, lw=0.6, ls=(0, (1, 2.5)))
        ax.text(-0.8, y, name, ha="right", va="center", fontsize=SMALL, color=figstyle.INK)
    for a, b in bt.gaps():
        for xc in ((a + b) / 2 - 0.45, (a + b) / 2 + 0.45):
            ax.plot([xc - 0.35, xc + 0.35], [rows["ADC"] - 0.5, rows["RF"] + 0.5], color=ink, lw=0.6)

    def draw_wave(key: str, row: str) -> None:
        t = np.asarray(wf[key]["t_s"]) * ms
        a = np.asarray(wf[key]["A"])
        a = a / np.abs(a).max() * amp[row]
        # KomaMRI samples each hard pulse or lobe as four corners (0, A, A, 0).
        for k in range(0, len(t), 4):
            ax.fill([bt(v) for v in t[k:k + 4]], rows[row] + a[k:k + 4], color=ink, lw=0.6, zorder=2)

    draw_wave("rf", "RF")
    draw_wave("gx", "Gx")

    gy = np.asarray(wf["gy_pe_amplitude_per_shot_T_per_m"])
    gy = gy / np.abs(gy).max() * amp["Gy"]
    pe0 = wf["pe_block"]["start_s"] * ms
    xa, xb = bt(pe0), bt(pe0 + wf["pe_block"]["dur_s"] * ms)
    for g in gy:
        ax.plot([xa, xb], [rows["Gy"] + g] * 2, color=ink, lw=0.3, zorder=2)
    ax.plot([xa, xa, xb, xb, xa], rows["Gy"] + np.array([gy.min(), gy.max(), gy.max(), gy.min(), gy.min()]),
            color=ink, lw=0.6, zorder=2)

    adc_t = np.asarray(wf["adc_t_s"]) * ms
    ax.add_patch(plt.Rectangle((bt(adc_t[0]), rows["ADC"]), bt(adc_t[-1]) - bt(adc_t[0]), amp["ADC"],
                               facecolor=ink, lw=0, zorder=2))

    flips = [e["flip_deg"] for e in wf["rf_events"]] + [wf["rf_events"][0]["flip_deg"]]
    for c, f in zip((inv, exc, ref, nxt), flips):
        ax.text(bt(c), rows["RF"] + amp["RF"] + 0.15, f"{f:.0f}°", ha="center", va="bottom", fontsize=SMALL,
                color=figstyle.INK)

    def dimension(x0: float, x1: float, y: float, label: str, policy: bool) -> None:
        colour, lw = (figstyle.ACCENT, 1.3) if policy else (figstyle.INK_SECONDARY, 0.7)
        ax.annotate("", xy=(x1, y), xytext=(x0, y),
                    arrowprops=dict(arrowstyle="<|-|>", color=colour, lw=lw, shrinkA=0, shrinkB=0,
                                    mutation_scale=5))
        ax.text((x0 + x1) / 2, y, label, ha="center", va="center", fontsize=SMALL, color=figstyle.INK,
                fontweight="bold" if policy else "normal",
                bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="none"))

    # In a short inset text is taller relative to the rows, so the TE line (centred over the
    # refocusing pulse's label at TE/2) needs more headroom.
    y_top = rows["RF"] + amp["RF"] + (1.9 if compact else 1.45)
    dimension(bt(inv), bt(exc), y_top, "TI" if compact else f"TI {wf['TI_s']:.2f} s", policy=True)
    te = f"TE {wf['TE_s'] * ms:.0f} ms"
    dimension(bt(exc), bt(adc_c), y_top, te if compact else f"{te} (fixed)", policy=False)
    y_bot = rows["ADC"] - 0.95
    dimension(bt(inv), bt(nxt), y_bot, f"TR {wf['TR_s']:.2f} s (×{wf['Npe']} shots per block)", policy=True)

    # Room for the row labels, which keep their point size while a compact inset shrinks.
    ax.set_xlim(-5.2 if compact else -3.4, bt.width + 1.2)
    ax.set_ylim(y_bot - 0.75, y_top + 0.75)
    ax.axis("off")


def frame_inset(ax: plt.Axes) -> None:
    ax.axis("on")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(figstyle.GRID)
        spine.set_linewidth(0.8)


# ---------------------------------------------------------------- assembly


def add_panel_b_with_inset(fig: plt.Figure, rect_cm: tuple[float, float, float, float], vs: ValidationScans,
                           wf: dict, labels: tuple[str, str | None] = ("B", "C"), label_x: float = -0.075,
                           compact: bool = False, lines: ScanLines = MEAN_T1_LINES,
                           inset: Sequence[float] | None = None) -> plt.Axes:
    """inset: the shot inset's axes-fraction rectangle, reaching the right edge; by default the largest one
    right of INSET_FROM_BLOCK that clears every TI line."""
    ax_b = add_axes_cm(fig, *rect_cm)
    draw_panel_b(ax_b, vs, compact=compact, lines=lines)
    if inset is None:
        inset = inset_rect(ax_b, vs.scans, INSET_FROM_BLOCK)
    else:
        # A given inset only has to clear the lines, not keep the default placement's margin.
        x0, x1 = ax_b.get_xlim()
        clear = inset_rect(ax_b, vs.scans, x0 + inset[0] * (x1 - x0), clearance_decades=0.0)
        assert inset[1] >= clear[1], "the inset would hide TI lines"
    ax_c = ax_b.inset_axes(inset)
    draw_sequence(ax_c, wf, compact=compact)
    frame_inset(ax_c)
    figstyle.panel_label(ax_b, labels[0], x=label_x, y=1.0)
    if labels[1] is not None:
        figstyle.panel_label(ax_c, labels[1], x=0.008, y=0.97, va="top")
    return ax_b


def full_figure(curve: TrainingCurve, vs: ValidationScans, wf: dict,
                checkpoints: list[CheckpointMape] | None = None, ymax: float = MAPE_YMAX) -> plt.Figure:
    fig = plt.figure(figsize=(FIG_W * CM, 14.0 * CM))
    draw_panel_a(fig, left=1.35, bottom=10.0, height=3.7, data=curve, checkpoints=checkpoints, ymax=ymax)
    add_panel_b_with_inset(fig, (1.35, 1.0, FIG_W - 1.35 - 0.2, 7.8), vs, wf)
    return fig


def main() -> None:
    figstyle.apply()
    curve = load_training_curve()
    vs = load_scans()
    wf = read_json(WAVEFORMS)

    fig = full_figure(curve, vs, wf)
    figstyle.save(fig, "fig3")
    plt.close(fig)

    fig = plt.figure(figsize=(FIG_W * CM, 5.4 * CM))
    draw_panel_a(fig, left=1.35, bottom=1.25, height=3.7, data=curve)
    figstyle.save(fig, "fig3_A")
    plt.close(fig)

    fig = plt.figure(figsize=(FIG_W * CM, 9.2 * CM))
    add_panel_b_with_inset(fig, (1.35, 1.0, FIG_W - 1.35 - 0.2, 7.8), vs, wf)
    figstyle.save(fig, "fig3_B")
    plt.close(fig)

    fig = plt.figure(figsize=(12.0 * CM, 4.2 * CM))
    draw_sequence(add_axes_cm(fig, 0.0, 0.0, 12.0, 4.2), wf)
    figstyle.save(fig, "fig3_C")
    plt.close(fig)

    print(f"rungs (h): {[(n, round(a, 3), round(b, 3)) for n, a, b in curve.rungs]}")
    print(f"selected checkpoint at {curve.selected_h:.2f} h, {curve.selected_mape:.2f}% ({curve.n_eval} scans)")
    print(f"held-out {curve.heldout_mape:.2f}%; opening probe = {vs.opening_probe_blocks} blocks, "
          f"TI {vs.scans[0].ti_s[:vs.opening_probe_blocks].round(4)} s")


if __name__ == "__main__":
    main()

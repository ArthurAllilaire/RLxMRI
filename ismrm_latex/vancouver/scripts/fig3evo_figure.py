"""Training-evolution candidate figure: the TI-histogram policy at three checkpoints.

(A) Executed blocks of 24 validation scans in the (TI, TR) plane, one panel per checkpoint.
(B) Running T1 MAPE versus scan time: median and interquartile range over the 24 scans.

A third panel (measured ROI signal against the fitter's model at each sampled (TI, TR)
pair) was tried and dropped: at the sampled points a wrong fit and the true model agree
to within noise, so the panel could not show why the early fit is wrong.

Inputs: figures/fig3evo_rollout_<checkpoint>_0-23.json from fig3evo_rollout.jl, and the
run directory for training times, budget and shots per block.

Run from the RLxMRI repo root:
  .venv/bin/python ismrm_latex/vancouver/scripts/fig3evo_figure.py
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

import figstyle
from fig3_figure import CM, FIG_W, RUN, SMALL, add_axes_cm, load_training_curve, read_json

# Earlier checkpoints recede toward light indigo; the selected one carries the accent.
CHECKPOINTS = {
    "stage0_analytic_end": ("End of analytic rung", figstyle.RUNG_COLOURS["cached"]),
    "stage3_full3_end": ("End of full 3 mm rung", figstyle.RL_OTHER),
    "global_best": ("Selected checkpoint", figstyle.ACCENT),
}
# TI spans the action space (julia/rl/e2.jl:386-387); TR starts at 0 so blocks at the
# 0.5 s TR floor stay visible.
TI_LIM = (0.01, 3.0)
TR_LIM = (0.0, 5.0)
MAPE_YMAX = 50.0


@dataclass
class Checkpoint:
    name: str
    label: str
    colour: str
    hours: float
    episodes: list[dict]

    def executed(self, ep: dict) -> list[dict]:
        return [b for b in ep["blocks"] if b["executed"]]

    @property
    def blocks_per_scan(self) -> float:
        return float(np.mean([len(self.executed(ep)) for ep in self.episodes]))

    @property
    def per_scan_mape(self) -> np.ndarray:
        return np.array([ep["blocks"][-1]["mape"] for ep in self.episodes])

    @property
    def mape_pct(self) -> float:
        return 100.0 * float(np.mean(self.per_scan_mape))


def training_hours() -> dict[str, float]:
    curve = load_training_curve()
    rung_end = {name: h1 for name, _, h1 in curve.rungs}
    return {"stage0_analytic_end": rung_end["analytic"], "stage3_full3_end": rung_end["full3"],
            "global_best": curve.selected_h}


def load_checkpoints() -> list[Checkpoint]:
    hours = training_hours()
    out = []
    for name, (label, colour) in CHECKPOINTS.items():
        data = read_json(figstyle.FIGURES / f"fig3evo_rollout_{name}_0-23.json")
        out.append(Checkpoint(name, label, colour, hours[name], data["episodes"]))
    return out


def format_hours(h: float) -> str:
    return f"{h * 60:.0f} min" if h < 1 else f"{h:.1f} h"


# ---------------------------------------------------------------- panel A


def checkpoint_stats(cp: Checkpoint) -> str:
    return f"{cp.blocks_per_scan:.1f} blocks/scan\nMAPE {cp.mape_pct:.1f}%"


def draw_ti_tr(axes: list[plt.Axes], cps: list[Checkpoint],
               stats: Callable[[Checkpoint], str] = checkpoint_stats, stats_y: float = 0.96) -> None:
    for ax, cp in zip(axes, cps):
        ti = [b["TI"] for ep in cp.episodes for b in cp.executed(ep)]
        tr = [b["TR"] for ep in cp.episodes for b in cp.executed(ep)]
        ax.scatter(ti, tr, s=9, color=cp.colour, alpha=0.55, lw=0, zorder=2, clip_on=False)
        ax.set_xscale("log")
        ax.set_xlim(*TI_LIM)
        ax.set_ylim(*TR_LIM)
        ticks = [0.03, 0.1, 0.3, 1]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:g}" for t in ticks])
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
        ax.set_title(f"{cp.label}, {format_hours(cp.hours)}", fontsize=SMALL + 1, loc="left", pad=3)
        ax.text(0.97, stats_y, stats(cp), transform=ax.transAxes, ha="right", va="top", fontsize=SMALL,
                color=figstyle.INK, linespacing=1.1)
        ax.set_xlabel("TI (s)")
    axes[0].set_ylabel("TR (s)")
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)


# ---------------------------------------------------------------- panel B


def running_mape(cp: Checkpoint, t_grid: np.ndarray, shots: int) -> np.ndarray:
    """Per-scan MAPE after the blocks finished by each time; NaN before the first fit."""
    rows = []
    for ep in cp.episodes:
        blocks = cp.executed(ep)
        t_end = shots * np.cumsum([b["TR"] for b in blocks])
        mape = np.array([100.0 * b["mape"] for b in blocks])
        # The env reports 100% until two blocks exist, i.e. before any fit.
        mape[:1] = np.nan
        idx = np.searchsorted(t_end, t_grid, side="right") - 1
        rows.append(np.where(idx >= 0, mape[np.clip(idx, 0, None)], np.nan))
    return np.array(rows)


def draw_running_mape(ax: plt.Axes, cps: list[Checkpoint], budget_s: float, shots: int,
                      note: str = "line: median, band: interquartile range (24 scans)") -> None:
    t = np.linspace(0, budget_s, 1121)
    ends = []
    for cp in cps:
        m = running_mape(cp, t, shots)
        ok = np.all(np.isfinite(m), axis=0)
        q25, q50, q75 = np.percentile(m[:, ok], [25, 50, 75], axis=0)
        ax.fill_between(t[ok], q25, q75, color=cp.colour, alpha=0.18, lw=0, step="post")
        ax.step(t[ok], q50, where="post", color=cp.colour, lw=1.6)
        ends.append((q50[-1], format_hours(cp.hours)))
    # End labels sit at their line's final value, nudged apart just enough not to touch.
    min_gap = 0.075 * MAPE_YMAX
    placed = []
    for y, label in sorted(ends):
        y = max(y, placed[-1] + min_gap) if placed else y
        placed.append(y)
        ax.text(budget_s + 6, y, label, ha="left", va="center", fontsize=SMALL, color=figstyle.INK)
    ax.set_xlim(0, budget_s)
    ax.set_ylim(0, MAPE_YMAX)
    ax.set_xlabel("Scan time (s)")
    ax.set_ylabel("Running T$_1$ MAPE (%)")
    ax.text(1.0, 1.0, note, transform=ax.transAxes, ha="right", va="bottom", fontsize=SMALL,
            color=figstyle.INK_SECONDARY)


# ---------------------------------------------------------------- assembly


def main() -> None:
    figstyle.apply()
    cps = load_checkpoints()
    kw = read_json(RUN / "run_config.json")["base_env_kwargs"]
    budget_s, shots = float(kw["time_budget_s"]), int(kw["Npe"])

    fig = plt.figure(figsize=(FIG_W * CM, 11.0 * CM))
    left, gap, width = 1.35, 0.45, FIG_W - 1.35 - 0.3
    w_a = (width - 2 * gap) / 3
    axes_a = [add_axes_cm(fig, left + i * (w_a + gap), 6.1, w_a, 4.0) for i in range(3)]
    draw_ti_tr(axes_a, cps)
    figstyle.panel_label(axes_a[0], "A", x=-0.3, y=1.08)

    ax_b = add_axes_cm(fig, left, 1.0, width - 1.3, 3.6)
    draw_running_mape(ax_b, cps, budget_s, shots)
    figstyle.panel_label(ax_b, "B", x=-0.1, y=1.04)
    figstyle.save(fig, "fig3evo")
    plt.close(fig)

    for cp in cps:
        print(f"{cp.name:22s} {format_hours(cp.hours):>7s}  {cp.blocks_per_scan:5.2f} blocks/scan  "
              f"MAPE {cp.mape_pct:5.2f}%")


if __name__ == "__main__":
    main()

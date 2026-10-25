"""One IR-SE shot as a pulse-sequence diagram (RF, Gx, Gy and ADC rows) on a broken time axis.

Drawn from the waveforms that fig3_sequence_dump.jl samples out of the real KomaMRI sequence
(figures/fig3_sequence_waveforms.json). The interval labels are parameters, so the same diagram can
quote one block's TI and TR or stay generic.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

import figstyle

WAVEFORMS = figstyle.FIGURES / "fig3_sequence_waveforms.json"
SMALL = 7.0
# Room, in diagram units, left of the time axis for the row labels and right of it for the last
# pulse's label.
X_ROOM = (3.4, 1.2)
# From the RF row's peak to the TI and TE dimensions: the TE label sits over the refocusing pulse's
# label (at TE/2), and text keeps its point size however short the axes are.
TOP_HEADROOM = 1.9


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


def dimension_arrow(policy: bool) -> dict:
    """Arrow style of an interval: heavy and in the accent colour when the policy chooses it."""
    colour, lw = (figstyle.ACCENT, 1.3) if policy else (figstyle.INK_SECONDARY, 0.7)
    return {"arrowstyle": "<|-|>", "color": colour, "lw": lw, "shrinkA": 0, "shrinkB": 0, "mutation_scale": 5}


@dataclass(frozen=True)
class ShotLabels:
    """Text of the three interval dimensions; TI and TR are the policy's, TE is fixed. Each label sets
    its own weight, with mathtext for a bold part."""
    ti: str
    te: str
    tr: str


def draw_ir_se_shot(ax: plt.Axes, wf: dict, labels: ShotLabels) -> None:
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
        ax.annotate("", xy=(x1, y), xytext=(x0, y), arrowprops=dimension_arrow(policy))
        ax.text((x0 + x1) / 2, y, label, ha="center", va="center", fontsize=SMALL, color=figstyle.INK,
                bbox={"boxstyle": "square,pad=0.2", "fc": "white", "ec": "none"})

    y_top = rows["RF"] + amp["RF"] + TOP_HEADROOM
    dimension(bt(inv), bt(exc), y_top, labels.ti, policy=True)
    dimension(bt(exc), bt(adc_c), y_top, labels.te, policy=False)
    y_bot = rows["ADC"] - 0.95
    dimension(bt(inv), bt(nxt), y_bot, labels.tr, policy=True)

    ax.set_xlim(-X_ROOM[0], bt.width + X_ROOM[1])
    ax.set_ylim(y_bot - 0.75, y_top + 0.75)
    ax.axis("off")

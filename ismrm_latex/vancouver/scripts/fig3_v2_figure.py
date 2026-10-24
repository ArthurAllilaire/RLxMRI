"""Figure 3 v2: what the TI-histogram policy learned, on the 24 validation scans (seed 500000).

(A) Executed (TI, TR) pairs at three checkpoints (fig3evo_figure.draw_ti_tr), with the
    T1 MAPE's percentile-bootstrap 95% CI over the scans (python/baseline_e2.py:41-58).
(B) TI chosen at each block, with (C) the IR-SE shot inset (fig3_figure.add_panel_b_with_inset).

Inputs: figures/fig3evo_rollout_<checkpoint>_0-23.json (fig3evo_rollout.jl), the run's
global_best/diagnostics/episodes.json and figures/fig3_sequence_waveforms.json.

Run from the RLxMRI repo root:
  .venv/bin/python ismrm_latex/vancouver/scripts/fig3_v2_figure.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import figstyle
from control_summary import boot_ci
from fig3_figure import (CM, FIG_W, WAVEFORMS, ValidationScans, add_axes_cm, add_panel_b_with_inset,
                         load_scans, read_json)
from fig3evo_figure import Checkpoint, draw_ti_tr, load_checkpoints

MAX_CAPTION_CHARS = 500
FIG_H = 14.4
LEFT = 1.35
A_BOTTOM, A_HEIGHT, A_GAP = 9.95, 3.6, 0.45
B_BOTTOM, B_HEIGHT = 1.0, 7.4
# add_panel_b_with_inset puts B at x = -0.075 of panel B's width; A gets the same page offset.
B_LABEL_X = -0.075
# Clears panel B's widest TI tick label (0.03).
YLABEL_OFFSET_CM = 1.0
# Raised from fig3evo's 0.96 so the CI line clears the TR 4 s outlier of the middle panel by
# about as much as it clears the panel title (~1.4 mm each).
STATS_Y = 0.99


def mape_ci_pct(cp: Checkpoint) -> tuple[float, float]:
    lo, hi = boot_ci(cp.per_scan_mape)
    return 100 * lo, 100 * hi


def mape_stats(cp: Checkpoint) -> str:
    lo, hi = mape_ci_pct(cp)
    return f"{cp.blocks_per_scan:.1f} blocks/scan\nMAPE {cp.mape_pct:.1f}% [{lo:.1f}, {hi:.1f}]"


def caption(cps: list[Checkpoint], vs: ValidationScans) -> str:
    early = np.mean([cp.blocks_per_scan for cp in cps[:-1]])
    late = cps[-1].blocks_per_scan
    return (
        f"TI-histogram policy, 560 s budget, {len(vs.scans)} validation scans. (A) Executed (TI, TR) "
        "pairs at three checkpoints, with blocks per scan and T1 MAPE [95% CI]. Early checkpoints use "
        f"long TRs at any TI; the selected one learned to pair short TIs with short TRs, fitting "
        f"~{late:.0f} blocks instead of ~{early:.0f}. (B) TI per block, coloured by the scan's mean sphere "
        f"T1: a fixed {vs.opening_probe_blocks}-block opening probe (shaded), then per-scan choices. "
        f"(C) One IR-SE shot: the policy sets TI and TR; {vs.shots_per_block} shots per block, one "
        "k-space line each."
    )


def main() -> None:
    figstyle.apply()
    cps = load_checkpoints()
    vs = load_scans()
    wf = read_json(WAVEFORMS)
    assert cps[-1].name == "global_best"
    # Panel A's selected checkpoint is the pure-Julia rollout of the scans panel B reads.
    assert np.isclose(cps[-1].blocks_per_scan, np.mean([len(s.ti_s) for s in vs.scans]))

    fig = plt.figure(figsize=(FIG_W * CM, FIG_H * CM))
    width = FIG_W - LEFT - 0.3
    w_a = (width - 2 * A_GAP) / 3
    axes_a = [add_axes_cm(fig, LEFT + i * (w_a + A_GAP), A_BOTTOM, w_a, A_HEIGHT) for i in range(3)]
    draw_ti_tr(axes_a, cps, stats=mape_stats, stats_y=STATS_Y)
    w_b = FIG_W - LEFT - 0.2
    figstyle.panel_label(axes_a[0], "A", x=B_LABEL_X * w_b / w_a, y=1.08)
    add_panel_b_with_inset(fig, (LEFT, B_BOTTOM, w_b, B_HEIGHT), vs, wf)
    ax_b = next(ax for ax in fig.axes if ax not in axes_a)
    # fig.align_ylabels skips axes placed with add_axes, so align the two y labels by hand.
    for ax, w in ((axes_a[0], w_a), (ax_b, w_b)):
        ax.yaxis.set_label_coords(-YLABEL_OFFSET_CM / w, 0.5)
    figstyle.save(fig, "fig3_v2")
    plt.close(fig)

    cap = caption(cps, vs)
    assert len(cap) <= MAX_CAPTION_CHARS, len(cap)
    for cp in cps:
        lo, hi = mape_ci_pct(cp)
        print(f"{cp.name:22s} {cp.blocks_per_scan:5.2f} blocks/scan  MAPE {cp.mape_pct:.2f}% [{lo:.2f}, {hi:.2f}]")
    print(f"Caption ({len(cap)} characters):\n{cap}")


if __name__ == "__main__":
    main()

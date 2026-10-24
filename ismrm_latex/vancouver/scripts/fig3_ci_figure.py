"""Figure 3 with 95% CIs on the training curve, in two layouts.

Every saved checkpoint of the TI-histogram run is re-rolled on the 24 validation scans
(seed 500000 + k) by fig3evo_rollout.jl; each gets its mean T1 MAPE with a percentile-bootstrap
95% CI (control_summary.boot_ci: 5000 resamples, default_rng(0), as python/baseline_e2.py:41-58)
and its training time on fig3's axis (PPO start = first tensorboard event file).

  fig3_ci     fig3 with the new panel A (A training curve, B TI per block, C shot inset)
  fig3_combo  evolution over training: A training curve, B (TI, TR) at three checkpoints,
              C running MAPE within a scan at those checkpoints, D the selected policy's TI per
              block with the shot as an inset; one colour per featured checkpoint throughout

Also writes figures/fig3_ci_checkpoints.{json,md}.

Run from the RLxMRI repo root (after the rollouts):
  .venv/bin/python ismrm_latex/vancouver/scripts/fig3_ci_figure.py
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

import figstyle
from control_summary import boot_ci
from fig3_figure import (CM, FIG_W, RUN, WAVEFORMS, CheckpointMape, add_axes_cm, add_panel_b_with_inset,
                         draw_panel_a, full_figure, load_scans, load_training_curve, ppo_start_wall_s, read_json)
from fig3_v2_figure import mape_stats
from fig3evo_figure import CHECKPOINTS as FEATURED
from fig3evo_figure import draw_running_mape, draw_ti_tr, format_hours, load_checkpoints

MAX_CAPTION_CHARS = 500
N_SCANS = 24
# name -> (rung, how the checkpoint was saved)
CHECKPOINTS = {
    "stage0_analytic_end": ("analytic", "end of rung"),
    "stage1_cached3_best": ("cached3", "best in rung"),
    "stage1_cached3_end": ("cached3", "end of rung"),
    "stage2_cached_best": ("cached", "best in rung"),
    "stage2_cached_end": ("cached", "end of rung"),
    "stage3_full3_best": ("full3", "best in rung"),
    "stage3_full3_end": ("full3", "end of rung"),
    "stage4_full_best": ("full", "best in rung"),
    "stage4_full_end": ("full", "end of rung"),
    "global_best": ("full", "selected"),
}


@dataclass
class Row:
    name: str
    rung: str
    kind: str
    step: int
    hours: float
    blocks: float
    mape: float
    lo: float
    hi: float
    same_weights_as: str | None


def checkpoint_hours() -> dict[str, float]:
    """Training time at which each checkpoint was saved, on fig3 A's axis."""
    t0 = ppo_start_wall_s()
    plan = read_json(RUN / "run_config.json")["plan"]
    history = read_json(RUN / "fidelity_history.json")
    stage_start = {r["stage"]: r["wall_s"] for r in history if r["kind"] == "stage_start"}
    stage_end = {r["stage"]: r["wall_s"] for r in history if r["kind"] == "stage_end"}
    hours = {}
    for i, rung in enumerate(plan):
        stage_dir = RUN / f"stage{i}_{rung}"
        hours[f"stage{i}_{rung}_end"] = (stage_end[rung] - t0) / 3600
        best = stage_dir / "best" / "best_meta.json"
        if best.exists():
            step = read_json(best)["step"]
            evals = read_json(stage_dir / "eval_history.json")
            # eval_history wall_s counts from the eval callback's creation at the stage start.
            wall = next(e["wall_s"] for e in evals if e["step"] == step)
            start = stage_start.get(rung, t0)
            hours[f"stage{i}_{rung}_best"] = (start + wall - t0) / 3600
    hours["global_best"] = load_training_curve().selected_h
    return hours


def load_rows() -> list[Row]:
    hours = checkpoint_hours()
    index = read_json(figstyle.FIGURES / "fig3evo_policies.json")
    weights = {}
    rows = []
    for name, (rung, kind) in CHECKPOINTS.items():
        eps = read_json(figstyle.FIGURES / f"fig3evo_rollout_{name}_0-{N_SCANS - 1}.json")["episodes"]
        assert len(eps) == N_SCANS
        per_scan = np.array([ep["blocks"][-1]["mape"] for ep in eps])
        lo, hi = boot_ci(per_scan)
        blocks = np.mean([sum(b["executed"] for b in ep["blocks"]) for ep in eps])
        npz = np.load(figstyle.FIGURES / index[name]["npz"])
        w = tuple(npz[k].tobytes() for k in sorted(npz.files) if k.startswith(("W", "b")))
        twin = next((other for other, ow in weights.items() if ow == w), None)
        weights[name] = w
        rows.append(Row(name, rung, kind, index[name]["num_timesteps"], hours[name], float(blocks),
                        100 * float(per_scan.mean()), 100 * lo, 100 * hi, twin))
    return sorted(rows, key=lambda r: r.hours)


def write_table(rows: list[Row]) -> None:
    (figstyle.FIGURES / "fig3_ci_checkpoints.json").write_text(json.dumps([r.__dict__ for r in rows], indent=2))
    lines = ["| Checkpoint | Rung | Saved as | Step | Training time (h) | Blocks/scan | MAPE % [95% CI] |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        note = f" (same weights as {r.same_weights_as})" if r.same_weights_as else ""
        lines.append(f"| {r.name}{note} | {figstyle.RUNG_LABELS[r.rung]} | {r.kind} | {r.step} | {r.hours:.3f} | "
                     f"{r.blocks:.2f} | {r.mape:.2f} [{r.lo:.2f}, {r.hi:.2f}] |")
    (figstyle.FIGURES / "fig3_ci_checkpoints.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def plotted(rows: list[Row], follow_featured: bool = False) -> list[CheckpointMape]:
    """One marker per distinct policy; optionally coloured to follow the featured checkpoints."""
    out = []
    # A "best in rung" saved at the next rung's first evaluation has the previous rung's final
    # weights; plotting both would draw one policy twice at the same time.
    for r in rows:
        if r.same_weights_as is not None:
            continue
        cp = CheckpointMape(r.hours, r.mape, r.lo, r.hi, N_SCANS, r.kind == "selected")
        if follow_featured:
            featured = r.name in FEATURED
            cp.colour = FEATURED[r.name][1] if featured else figstyle.INK_SECONDARY
            cp.featured = featured
            # The same text as panel B's titles and panel C's line ends.
            cp.label = format_hours(r.hours) if featured else None
        out.append(cp)
    return out


def axis_max(rows: list[Row]) -> float:
    return 5.0 * math.ceil(max(r.hi for r in rows) * 1.08 / 5.0)


# ---------------------------------------------------------------- layouts


def combo_figure(rows: list[Row]) -> plt.Figure:
    curve = load_training_curve()
    vs = load_scans()
    wf = read_json(WAVEFORMS)
    cps = load_checkpoints()
    kw = read_json(RUN / "run_config.json")["base_env_kwargs"]
    left, right = 1.35, 0.2
    fig = plt.figure(figsize=(FIG_W * CM, 16.7 * CM))
    draw_panel_a(fig, left=left, bottom=13.25, height=3.0, data=curve, checkpoints=plotted(rows, True),
                 ymax=axis_max(rows))

    gap = 0.45
    w_b = (FIG_W - left - 0.3 - 2 * gap) / 3
    axes_b = [add_axes_cm(fig, left + i * (w_b + gap), 8.3, w_b, 3.2) for i in range(3)]
    draw_ti_tr(axes_b, cps, stats=mape_stats, stats_y=0.99)
    figstyle.panel_label(axes_b[0], "B", x=-1.2 / w_b, y=1.1)

    w_c, row_h = 4.2, 5.9
    ax_c = add_axes_cm(fig, left, 1.0, w_c, row_h)
    draw_running_mape(ax_c, cps, float(kw["time_budget_s"]), int(kw["Npe"]), note="median and IQR, 24 scans")
    figstyle.panel_label(ax_c, "C", x=-1.2 / w_c, y=1.04)

    d_left = left + w_c + 2.3
    w_d = FIG_W - d_left - right
    add_panel_b_with_inset(fig, (d_left, 1.0, w_d, row_h), vs, wf, labels=("D", None), label_x=-1.2 / w_d,
                           compact=True)
    return fig


def caption_ci(rows: list[Row]) -> str:
    return (
        "TI-histogram policy, 560 s budget. (A) T1 MAPE during training (one GPU, 9 h): saved checkpoints "
        f"re-rolled on {N_SCANS} validation phantoms with full Bloch (mean, bootstrap 95% CI; faint: logged 4- "
        "and 20-scan values); shading: fidelity rung. Star: selected checkpoint on 24 held-out phantoms; lines: "
        "fixed log-TI grid and Cramér-Rao schedule on the same phantoms. (B) TI per block, coloured by mean sphere "
        "T1, after a fixed two-block probe (shaded). (C) One IR-SE shot; the policy sets TI and TR."
    )


def caption_combo(rows: list[Row]) -> str:
    early = next(r for r in rows if r.name == "stage0_analytic_end")
    sel = next(r for r in rows if r.kind == "selected")
    return (
        f"TI-histogram policy over training, {N_SCANS} validation phantoms unless stated. (A) T1 MAPE of every "
        "saved checkpoint, full Bloch (mean, bootstrap 95% CI); faint: logged probes; star and lines: selected "
        "checkpoint and fixed protocols, 24 held-out phantoms. Coloured: checkpoints followed in B and C. (B) Executed "
        f"(TI, TR): pairing short TIs with short TRs fits {sel.blocks:.0f} blocks, not {early.blocks:.0f}. (C) T1 "
        "MAPE within a scan (median, IQR). (D) Selected policy's TI per block, coloured by mean "
        "sphere T1; inset: one IR-SE shot."
    )


def main() -> None:
    figstyle.apply()
    rows = load_rows()
    write_table(rows)

    fig = full_figure(load_training_curve(), load_scans(), read_json(WAVEFORMS), checkpoints=plotted(rows),
                      ymax=axis_max(rows))
    figstyle.save(fig, "fig3_ci")
    plt.close(fig)

    fig = combo_figure(rows)
    figstyle.save(fig, "fig3_combo")
    plt.close(fig)

    for name, cap in (("fig3_ci", caption_ci(rows)), ("fig3_combo", caption_combo(rows))):
        assert len(cap) <= MAX_CAPTION_CHARS, (name, len(cap))
        print(f"\n{name} caption ({len(cap)} characters):\n{cap}")


if __name__ == "__main__":
    main()

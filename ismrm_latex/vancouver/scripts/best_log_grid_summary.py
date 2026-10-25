"""Summarise the abstract's fixed log TI grid comparator (best_log_grid.jl) into figures/best_log_grid.md.

The grid is fixed_schedules.jl's best_log_grid_schedule: TR 2.0 s, the best row of grid_tr_sweep.md, with
the top TI capped so that no block's TR is lifted. It replaces log_grid_trmatched (TR 1.7 s), whose rows are
kept here for reference. Statistics are control_summary.py's: percentile bootstrap CIs, 5000 resamples over
the phantoms, numpy default_rng(0); paired differences (grid minus the other arm, per phantom) resample the
phantoms the same way.

    .venv/bin/python ismrm_latex/vancouver/scripts/best_log_grid_summary.py
"""

from __future__ import annotations

import json

import numpy as np

import control_narrow_band_summary as nb
import figstyle
from control_summary import (
    N_BOOT,
    SUCCESS_THRESHOLD,
    arm_stats,
    boot_ci,
    fmt_paired,
    paired_stats,
)
from fig5_table_v2 import DISTILLED_ARM, FIXED_REPLAY
from grid_tr_sweep_summary import RL_RUNS, rl_per_scan

HELDOUT = figstyle.FIGURES / "control_data" / "best_log_grid_heldout.json"
NARROW = figstyle.FIGURES / "control_data" / "best_log_grid_narrow_band.json"
SWEEP = figstyle.FIGURES / "control_data" / "grid_tr_sweep.json"
OUT = figstyle.FIGURES / "best_log_grid.md"
GRID = "best_log_grid"
# The fixed schedules the abstract compares the policy with, by key; the replaced grid is reference only.
FIXED = {GRID: "Fixed log TI grid, TR 2.0 s", "cr_optimal": "Cramer-Rao optimal", "distilled": "Distilled fixed protocol"}
OLD_GRID = ("log_grid_trmatched", "Fixed log TI grid, TR 1.7 s (replaced)")


def per_scan(scans: list[dict]) -> tuple[np.ndarray, dict]:
    mape = np.array([s["mape"] for s in scans])
    return mape, arm_stats(mape, mape, [s["n_blocks"] for s in scans], [s["time_s"] for s in scans])


def rl_stats(mape: np.ndarray) -> dict:
    # The held-out evaluations saved no per-scan block counts or scan times.
    return {"mape": 100 * mape.mean(), "ci": tuple(100 * v for v in boot_ci(mape)), "n_runs": mape.size,
            "success": float(np.mean(mape < SUCCESS_THRESHOLD)), "worst": 100 * mape.max()}


def fmt_ci(s: dict) -> str:
    return f"{s['mape']:.2f} [{s['ci'][0]:.2f}, {s['ci'][1]:.2f}]"


def stats_row(label: str, s: dict) -> str:
    extra = f"{s['blocks']:.2f} | {s['time_s']:.0f}" if "blocks" in s else "- | -"
    return (f"| {label} | {fmt_ci(s)} | {100 * s['success']:.1f}% ({round(s['success'] * s['n_runs'])}/{s['n_runs']}) | "
            f"{s['worst']:.2f} | {extra} |")


def paired_row(label: str, p: dict) -> str:
    return f"| {label} | {fmt_paired(p)} | {p['median']:+.2f} | {p['worse']}/{p['n']} |"


def check_phantoms(t1_true: np.ndarray, arms: dict[str, list[dict]]) -> None:
    for name, scans in arms.items():
        assert np.array_equal([s["T1_true"] for s in scans], t1_true), name


def main() -> None:
    ho, nbd = json.loads(HELDOUT.read_text()), json.loads(NARROW.read_text())
    n, seed = ho["n_scans"], ho["seed_base"]
    assert [s["seed"] for s in ho["scans"]] == list(range(seed, seed + n))
    assert ho["reproduces"]["per_scan_identical"] and nbd["regression_check"]["per_scan_identical"]
    sweep = json.loads(SWEEP.read_text())
    row = next(p for p in sweep["series"]["capped"] if p["TR_s"] == ho["TR_s"] and p["label"] is None)
    assert row["scans"] == ho["scans"] and row["schedule"] == ho["schedule"]

    # Held-out phantoms: the grid, the other fixed schedules and the RL policies, scan by scan.
    replay = json.loads(FIXED_REPLAY.read_text())["budgets"][str(round(ho["budget_s"]))]
    wide = json.loads(nb.WIDE.read_text())
    fixed_ho = {GRID: ho["scans"], "cr_optimal": replay["cr_optimal"]["scans"],
                "distilled": wide["arms"][DISTILLED_ARM], OLD_GRID[0]: replay[OLD_GRID[0]]["scans"]}
    t1_true = np.array([s["T1_true"] for s in ho["scans"]])
    check_phantoms(t1_true, fixed_ho)
    mape_ho, stats_ho = {}, {}
    for k, scans in fixed_ho.items():
        mape_ho[k], stats_ho[k] = per_scan(scans)
    for k, (_, run) in RL_RUNS.items():
        mape_ho[k] = rl_per_scan(run, t1_true)
        stats_ho[k] = rl_stats(mape_ho[k])
    paired_ho = {k: paired_stats(mape_ho[GRID], mape_ho[k]) for k in ("rl_560", "distilled", "rl_240", OLD_GRID[0])}

    # Clustered-T1 phantoms: the same arms of control_narrow_band.json, on the same phantoms and noise.
    nbc = json.loads(nb.DATA.read_text())
    assert nbd["n_scans"] == nbc["n_scans"] and nbd["T1_band_centre"] == nbc["T1_band_centre"]
    fixed_nb = {GRID: nbd["scans"], **{k: nbc["arms"][k] for k in ("cr_optimal", "distilled", OLD_GRID[0], "adaptive")}}
    check_phantoms(np.array([s["T1_true"] for s in nbc["arms"]["adaptive"]]), fixed_nb)
    mape_nb, stats_nb = {}, {}
    for k, scans in fixed_nb.items():
        mape_nb[k], stats_nb[k] = per_scan(scans)
    paired_nb = paired_stats(mape_nb[GRID], mape_nb["adaptive"])
    centre = np.array(nbc["T1_band_centre"])
    bins = np.digitize(centre, nb.BIN_EDGES_S)
    e0, e1 = nb.BIN_EDGES_S
    by_bin = []
    for b, label in enumerate((f"c < {e0} s", f"{e0} <= c < {e1} s", f"c >= {e1} s")):
        sel = np.flatnonzero(bins == b)
        by_bin.append((f"{label} (n = {sel.size})", per_scan([fixed_nb[GRID][i] for i in sel])[1],
                       paired_stats(mape_nb[GRID][sel], mape_nb["adaptive"][sel])))

    worst = [stats_ho[k]["worst"] for k in FIXED]
    clustered = [stats_nb[k]["mape"] for k in FIXED]
    g_ho, g_nb = stats_ho[GRID], stats_nb[GRID]
    sched = ho["schedule"]
    ti = ", ".join(f"{t:.3f}" for t in sched["TI_s"])
    labels_ho = {**FIXED, OLD_GRID[0]: OLD_GRID[1], **{k: label for k, (label, _) in RL_RUNS.items()}}
    labels_nb = {**FIXED, OLD_GRID[0]: OLD_GRID[1], "adaptive": nb.ADAPTIVE_ARM[0]}
    head = "| Arm | Mean MAPE % [95% CI] | Scans < 5% | Worst scan MAPE % | Mean blocks | Mean scan time (s) |"
    paired_head = "| Comparison | Grid - arm, mean [95% CI] | Median per-phantom difference (pp) | Phantoms where the grid is worse |"
    lines = [
        "# Fixed log TI grid comparator: TR 2.0 s",
        "",
        "The abstract's fixed log TI grid is `best_log_grid_schedule` (`scripts/fixed_schedules.jl`): the best row of the TR sweep (`grid_tr_sweep.md`, TI-capped series), replacing `log_grid_trmatched` (TR 1.7 s).",
        f"Schedule: {len(sched['TI_s'])} blocks at TR {ho['TR_s']:.1f} s, {ho['Npe']} shots each ({ho['Npe'] * sum(sched['TR_s']):.0f} s of the {ho['budget_s']:.0f} s budget), TIs log-spaced over {sched['TI_s'][0]:g}-{sched['TI_s'][-1]:g} s ({ti} s); the top TI is capped at 0.9 TR - TE - 1 ms, so no block's TR is lifted.",
        f"Replayed by `scripts/best_log_grid.jl` as the fixed comparators were (same E2Env, generalised-IR T1 fitter, per-seed noise and Python action round trip, {ho['numpy_promotion']} promotion rules).",
        "",
        "## Reproduction checks",
        "",
        f"- Held-out: the replay reproduces the sweep's capped TR {ho['TR_s']:.1f} s row (`{ho['reproduces']['file']}`) bit for bit on all {n} scans (every field: T1 true and estimates, MAPE after each block, blocks, scan time, executed timings).",
        f"- Clustered-T1: before the grid, `log_grid_trmatched` replayed on the {nbd['n_scans']} phantoms reproduces its arm of `{nbd['regression_check']['file']}` bit for bit, so the grid ran on the same phantoms and noise as that file's arms.",
        "",
        f"## Held-out ({n} phantoms, seeds {seed}-{seed + n - 1})",
        "",
        head,
        "|---|---|---|---|---|---|",
        *(stats_row(labels_ho[k], stats_ho[k]) for k in (GRID, "cr_optimal", "distilled", OLD_GRID[0], *RL_RUNS)),
        "",
        paired_head,
        "|---|---|---|---|",
        *(paired_row(f"Grid - {labels_ho[k]}", p) for k, p in paired_ho.items()),
        "",
        f"## Clustered-T1 ({nbd['n_scans']} phantoms, seeds {nbd['seed_base']}-{nbd['seed_base'] + nbd['n_scans'] - 1}, T1s within +-{100 * nbd['t1_band_rel']:g}% of a random centre)",
        "",
        head,
        "|---|---|---|---|---|---|",
        *(stats_row(labels_nb[k], stats_nb[k]) for k in (GRID, "cr_optimal", "distilled", OLD_GRID[0], "adaptive")),
        "",
        paired_head,
        "|---|---|---|---|",
        paired_row(f"Grid - {labels_nb['adaptive']}", paired_nb),
        "",
        f"By band centre c (bins of `control_narrow_band.md`, edges {e0} and {e1} s); CIs and paired differences resample the phantoms within each bin.",
        "",
        "| Centre bin | Grid MAPE % [95% CI] | Scans < 5% | Worst scan MAPE % | Grid - Adaptive [95% CI] |",
        "|---|---|---|---|---|",
        *(f"| {label} | {fmt_ci(s)} | {100 * s['success']:.1f}% ({round(s['success'] * s['n_runs'])}/{s['n_runs']}) | "
          f"{s['worst']:.2f} | {fmt_paired(p)} |" for label, s, p in by_bin),
        "",
        "## For the text",
        "",
        f"- Held-out, grid: {fmt_ci(g_ho)}%, {round(g_ho['success'] * n)}/{n} scans < 5%, worst {g_ho['worst']:.2f}%, {g_ho['blocks']:.0f} blocks, {g_ho['time_s']:.0f} s.",
        *(f"- Held-out, grid - {labels_ho[k]}: {fmt_paired(paired_ho[k])}." for k in ("rl_560", "distilled", "rl_240")),
        f"- Clustered-T1, grid: {fmt_ci(g_nb)}%, {round(g_nb['success'] * nbd['n_scans'])}/{nbd['n_scans']} scans < 5%, worst {g_nb['worst']:.2f}%; grid - adaptive {fmt_paired(paired_nb)}.",
        f"- Worst-scan MAPE of the three fixed schedules on held-out (grid, Cramer-Rao, distilled): {min(worst):.2f}-{max(worst):.2f}%.",
        f"- Mean MAPE of the three fixed schedules on clustered-T1 (grid, Cramer-Rao, distilled): {min(clustered):.2f}-{max(clustered):.2f}%.",
        "",
        "MAPE: mean over the five spheres of |T1 estimate - T1 true| / T1 true, averaged over scans.",
        f"CI: percentile bootstrap, {N_BOOT} resamples over the phantoms, numpy `default_rng(0)` (as `python/baseline_e2.py:41-58`). Paired: grid minus the other arm per phantom, bootstrapped the same way; positive means the grid is less accurate.",
        "The TR 2.0 s grid was picked as the best of the sweep's rows on these same 24 held-out phantoms, so its held-out MAPE is optimistic; the clustered-T1 phantoms played no part in the choice.",
        "",
        "## Regenerate",
        "",
        "From the RLxMRI repo root:",
        "",
        "```bash",
        "julia -t 6 --project=. ismrm_latex/vancouver/scripts/best_log_grid.jl",
        ".venv/bin/python ismrm_latex/vancouver/scripts/best_log_grid_summary.py",
        "```",
        "",
        "Outputs: `figures/control_data/best_log_grid_heldout.json`, `figures/control_data/best_log_grid_narrow_band.json` (per-scan results) and this file.",
        "",
    ]
    OUT.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()

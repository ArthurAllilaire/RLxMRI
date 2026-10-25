"""Figure 5 v3 (candidate): the held-out table (A) above the narrow-band control (B).

A is fig5_table_v2's table without the flip-angle mark. B is the narrow-band control of
control_narrow_band.jl (96 phantoms, same TI-histogram checkpoint, not retrained), with the
statistics of control_narrow_band_summary.py; every other B cell is checked against
figures/control_narrow_band.md.
In both panels the fixed log TI grid is best_log_grid.jl's replay of best_log_grid_schedule (TR 2.0 s,
the best row of grid_tr_sweep.md), which replaces v2's log_grid_trmatched (TR 1.7 s).

    julia -t 6 --project=. ismrm_latex/vancouver/scripts/best_log_grid.jl
    .venv/bin/python ismrm_latex/vancouver/scripts/control_narrow_band_summary.py
    .venv/bin/python ismrm_latex/vancouver/scripts/fig5_table_v3.py
"""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

import control_narrow_band_summary as nb
import figstyle
from control_summary import arm_stats
from fig5_table import (
    DAGGER,
    DOUBLE_DAGGER,
    MAX_CAPTION_CHARS,
    RL_RUNS,
    RUNS,
    Row,
    _round,
    _tex_escape,
)
from fig5_table_v2 import (
    COLUMNS,
    DISTILLED_ARM,
    HEADER,
    REPLAY_SCHEDULES,
    SCAN_MAPE_CAP,
    SECTION,
    TABLE_HEAD,
    TableRow,
    body_tex,
    build,
    cells,
    document_tex,
    note_marks,
    notes_tex,
    render,
)

STEM = "fig5_table_v3"
MARK_ORDER = (DAGGER, SECTION)
# MAPE and its CI are separate TeX columns.
N_TEX_COLUMNS = len(HEADER) + 1
TRAINING_SAMPLER = "linear_uniform_range"
BEST_GRID_HELDOUT = figstyle.FIGURES / "control_data" / "best_log_grid_heldout.json"
BEST_GRID_NARROW = figstyle.FIGURES / "control_data" / "best_log_grid_narrow_band.json"
SWEEP = figstyle.FIGURES / "control_data" / "grid_tr_sweep.json"


def best_grid_row(path: Path, t1_true: np.ndarray) -> tuple[TableRow, dict]:
    """The fixed log TI grid row: best_log_grid.jl's replay, on the phantoms with these sphere T1s."""
    d = json.loads(path.read_text())
    scans = d["scans"]
    assert np.array_equal([s["T1_true"] for s in scans], t1_true), path.name
    mape = np.array([s["mape"] for s in scans])
    s = arm_stats(mape, mape, [r["n_blocks"] for r in scans], [r["time_s"] for r in scans])
    row = Row(key="log_grid", method=figstyle.METHOD_LABELS["log_grid"], memory="", budget_s=d["budget_s"],
              scan_time_s=s["time_s"], mape_pct=s["mape"], ci_pct=s["ci"], success=s["success"])
    return TableRow(row, mape), d


def swept_trs(grid_mape_pct: float) -> tuple[float, float]:
    """The TR range of grid_tr_sweep.json, whose lowest mean MAPE the caption says the grid has."""
    sweep = json.loads(SWEEP.read_text())
    best = min(100 * np.mean([s["mape"] for s in p["scans"]]) for rows in sweep["series"].values() for p in rows)
    assert best == grid_mape_pct, (best, grid_mape_pct)
    trs = [p["TR_s"] for rows in sweep["series"].values() for p in rows]
    return min(trs), max(trs)


def panel_a() -> tuple[list[TableRow], dict]:
    table, info = build()
    for key in RL_RUNS:
        cfg = json.loads((RUNS / RL_RUNS[key] / "run_config.json").read_text())
        assert cfg["base_env_kwargs"]["t1_sampler"] == TRAINING_SAMPLER, key
    # The flip angle is left out of this figure; v2 keeps its note.
    table = [replace(t, row=replace(t.row, mark="")) if t.row.mark == DOUBLE_DAGGER else t for t in table]
    # The shorter budget keeps only its RL row, which the text compares with the full-budget fixed designs.
    full_budget = table[0].row.budget_s
    table = [t for t in table if t.row.budget_s == full_budget or t.row.method == "RL"]
    hist = next(t for t in table if t.row.key == "hist")
    grid, d = best_grid_row(BEST_GRID_HELDOUT, np.reshape(hist.row.per_scan_t1_true, (info["n_scans"], -1)))
    assert d["seed_base"] == info["seed"] and grid.row.budget_s == full_budget
    table = [grid if t.row.key == "log_grid" else t for t in table]
    info["tr_s"], info["swept_trs_s"] = d["TR_s"], swept_trs(grid.row.mape_pct)
    info["heading"] = (f"{info['n_scans']} held-out phantoms (T\\textsubscript{{1}}s uniform over the "
                       "T\\textsubscript{1}-plate range, as in training)")
    return table, info


def narrow_band_md() -> dict[str, list[str]]:
    """Rows of the Results table of control_narrow_band.md, keyed by arm label."""
    lines = nb.OUT.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("## Results"))
    rows = {}
    for line in lines[start + 1:]:
        if line.startswith("## "):
            break
        if line.startswith("| ") and not line.startswith("| Arm |"):
            c = [x.strip() for x in line.strip().strip("|").split("|")]
            rows[c[0]] = c
    return rows


def check_md(t: TableRow, md: list[str]) -> None:
    _, _, mape_ci, success, worst, _, time_s, _ = md
    c = cells(t)
    n = t.per_scan_mape.size
    assert c[4] == mape_ci, (c[4], mape_ci)
    assert f"{c[5]} ({round(t.row.success * n)}/{n})" == success, (c[5], success)
    assert _round(t.worst_pct, 2) == worst, (t.worst_pct, worst)
    assert c[3] == f"{time_s} s", (c[3], time_s)


def panel_b(a_table: list[TableRow]) -> tuple[list[TableRow], dict]:
    d = json.loads(nb.DATA.read_text())
    hist = RUNS / RL_RUNS["hist"]
    assert d["env_run"] == str(hist.relative_to(figstyle.ROOT)) and d["checkpoint"] == "global_best"
    env = json.loads((hist / "run_config.json").read_text())["base_env_kwargs"]
    assert d["budget_s"] == env["time_budget_s"] and d["t1_sampler"] != env["t1_sampler"]
    control = json.loads(nb.WIDE.read_text())
    assert d["protocols"]["distilled"] == control["median_protocols"][DISTILLED_ARM]
    # The caption says these phantoms were never used to pick the checkpoint either.
    best = json.loads((hist / "global_best" / "best_meta.json").read_text())
    seeds = set(range(d["seed_base"], d["seed_base"] + d["n_scans"]))
    for offset, n in ((best["seed_offset"], best["n_eval_episodes"]),
                      (best["source_seed_offset"], best["source_n_eval_episodes"])):
        assert not seeds & set(range(offset, offset + n)), (offset, n)

    arms = nb.load_arms(d)
    stats = nb.arm_table(arms, np.arange(d["n_scans"]))
    a = {t.row.key: t for t in a_table if t.row.budget_s == d["budget_s"]}
    # key: (arm, method, memory, method mark, label in control_narrow_band.md)
    spec = {
        "hist": ("adaptive", a["hist"].row.method, a["hist"].row.memory, "", nb.ADAPTIVE_ARM[0]),
        "distilled": ("distilled", a["distilled"].row.method, "", a["distilled"].method_mark,
                      nb.FIXED_ARMS["distilled"][0]),
        # The log TI grid row is best_log_grid.jl's, not control_narrow_band.jl's TR 1.7 s grid.
        **{key: (arm, a[key].row.method, "", "", nb.FIXED_ARMS[arm][0]) for key, arm in REPLAY_SCHEDULES.items()
           if key != "log_grid"},
    }
    md = narrow_band_md()
    rows = {"log_grid": best_grid_row(BEST_GRID_NARROW, np.array([r["T1_true"] for r in d["arms"]["adaptive"]]))[0]}
    for key, (arm, method, memory, mark, md_label) in spec.items():
        s = stats[arm]
        runs = arms.mape[arm]
        row = Row(key=key, method=method, memory=memory, budget_s=d["budget_s"], scan_time_s=s["time_s"],
                  mape_pct=s["mape"], ci_pct=s["ci"], success=s["success"])
        t = TableRow(row, runs, method_mark=mark)
        assert t.worst_pct == s["worst"] and runs.size == s["n_runs"], key
        check_md(t, md[md_label])
        rows[key] = t
    table = [rows[t.row.key] for t in a_table if t.row.key in rows]

    # Only swapped schedules reached the environment's scan MAPE cap, so these rows need no cap note.
    assert all(t.per_scan_mape.max() < SCAN_MAPE_CAP for t in table)
    info = {
        "n_scans": d["n_scans"],
        "seed": d["seed_base"],
        "band_pct": 100 * d["t1_band_rel"],
        "best_key": min(table, key=lambda t: t.row.mape_pct).row.key,
        "notes": {},
    }
    info["heading"] = (f"{info['n_scans']} clustered-T\\textsubscript{{1}} phantoms (T\\textsubscript{{1}}s within "
                       f"±{info['band_pct']:g}% of a random centre; policy not retrained)")
    return table, info


def heading_tex(label: str, text: str) -> str:
    return (rf"\multicolumn{{{N_TEX_COLUMNS}}}{{@{{}}l@{{}}}}{{\fontsize{{10.5}}{{12}}\selectfont\bfseries "
            rf"{label}\hspace{{0.55em}}\fontsize{{9.5}}{{11}}\selectfont\mdseries {_tex_escape(text)}}} \\[2.5pt]")


def table_tex(a_table: list[TableRow], a_info: dict, b_table: list[TableRow], b_info: dict) -> str:
    tabular = "\n".join([
        rf"\begin{{tabular}}{{{COLUMNS}}}",
        heading_tex("A", a_info["heading"]),
        TABLE_HEAD,
        body_tex(a_table, a_info["best_key"]),
        r"\bottomrule",
        r"\addlinespace[11pt]",
        heading_tex("B", b_info["heading"]),
        TABLE_HEAD,
        body_tex(b_table, b_info["best_key"]),
        r"\bottomrule",
        r"\end{tabular}",
    ])
    notes = notes_tex({**a_info["notes"], **b_info["notes"]}, note_marks(a_table + b_table, MARK_ORDER))
    return document_tex(tabular, notes)


def caption(a_info: dict, b_info: dict) -> str:
    lo, hi = a_info["swept_trs_s"]
    return (
        "T1 error on phantoms identical across methods, unseen in training or checkpoint selection: "
        f"(A) {a_info['n_scans']} held-out; (B) {b_info['n_scans']} clustered-T1 (T1s within "
        f"±{b_info['band_pct']:g}% of a random centre). MAPE: mean absolute percentage error over a scan's "
        "spheres, averaged over scans (bootstrap 95% CI); worst: highest scan MAPE. Fixed grid: log-spaced "
        f"TIs, TR {a_info['tr_s']:.1f} s (best of a {lo:g}-{hi:g} s sweep), top TI capped to fit. Cramér-Rao: "
        "optimised for the plate's T1 range. Distilled: policy's per-block median TI and TR. Same T1 fitter for all."
    )


def _plain(heading: str) -> str:
    return heading.replace(r"T\textsubscript{1}", "T1")


def write_text(a_table: list[TableRow], a_info: dict, b_table: list[TableRow], b_info: dict, cap: str) -> None:
    # The BOM lets Excel and Word detect UTF-8, so the marks and e-acute survive a paste.
    with open(figstyle.FIGURES / f"{STEM}.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["Panel", *HEADER])
        writer.writerows([panel, *cells(t)] for panel, table in (("A", a_table), ("B", b_table)) for t in table)

    def md_table(table: list[TableRow], best_key: str) -> list[str]:
        out = ["| " + " | ".join(HEADER) + " |", "|" + "---|" * len(HEADER)]
        for t in table:
            c = cells(t)
            if t.row.key == best_key:
                c = [f"**{x}**" for x in c]
            out.append("| " + " | ".join(c) + " |")
        return out

    notes = {**a_info["notes"], **b_info["notes"]}
    lines = [
        "# Figure 5 v3 (candidate) - held-out table and narrow-band control",
        "",
        "Generated by `scripts/fig5_table_v3.py`; bold is the best row of each panel.",
        (f"A: `fig5_table_v2.py`'s table (held-out seeds {a_info['seed']}-"
         f"{a_info['seed'] + a_info['n_scans'] - 1}) with v2's {DOUBLE_DAGGER} mark and note removed."),
        (f"B: `control_data/control_narrow_band.json` (seeds {b_info['seed']}-"
         f"{b_info['seed'] + b_info['n_scans'] - 1}), statistics from `control_narrow_band_summary.py`; "
         "every cell but the fixed log TI grid's is checked against `control_narrow_band.md`."),
        (f"Fixed log TI grid, both panels: `best_log_grid_schedule` (TR {a_info['tr_s']:.1f} s, the best row of "
         "`grid_tr_sweep.md`) as replayed by `best_log_grid.jl` (`control_data/best_log_grid_heldout.json`, "
         "`control_data/best_log_grid_narrow_band.json`), with the statistics of `control_summary.py`; "
         "v2's TR 1.7 s grid is replaced."),
        "",
        f"## A - {_plain(a_info['heading'])}",
        "",
        *md_table(a_table, a_info["best_key"]),
        "",
        f"## B - {_plain(b_info['heading'])}",
        "",
        *md_table(b_table, b_info["best_key"]),
        "",
        *[f"{m} {notes[m]}" for m in note_marks(a_table + b_table, MARK_ORDER)],
        "",
        f"## Caption ({len(cap)} characters)",
        "",
        cap,
        "",
    ]
    (figstyle.FIGURES / f"{STEM}.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    figstyle.apply()
    a_table, a_info = panel_a()
    b_table, b_info = panel_b(a_table)
    render(table_tex(a_table, a_info, b_table, b_info), STEM)
    cap = caption(a_info, b_info)
    assert len(cap) <= MAX_CAPTION_CHARS, len(cap)
    write_text(a_table, a_info, b_table, b_info, cap)
    for panel, table in (("A", a_table), ("B", b_table)):
        for t in table:
            print(panel, " | ".join(cells(t)))
    print(f"Caption ({len(cap)} characters):\n{cap}")


if __name__ == "__main__":
    main()

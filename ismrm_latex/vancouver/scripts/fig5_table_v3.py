"""Figure 5 v3 (candidate): the held-out table (A) above the narrow-band control (B).

A is fig5_table_v2's table without the flip-angle mark. B is the narrow-band control of
control_narrow_band.jl (96 phantoms, same TI-histogram checkpoint, not retrained), with the
statistics of control_narrow_band_summary.py; every B cell is checked against
figures/control_narrow_band.md.

    .venv/bin/python ismrm_latex/vancouver/scripts/control_narrow_band_summary.py
    .venv/bin/python ismrm_latex/vancouver/scripts/fig5_table_v3.py
"""

from __future__ import annotations

import csv
import json
from dataclasses import replace

import numpy as np

import control_narrow_band_summary as nb
import figstyle
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
        **{key: (arm, a[key].row.method, "", "", nb.FIXED_ARMS[arm][0]) for key, arm in REPLAY_SCHEDULES.items()},
    }
    md = narrow_band_md()
    table = []
    for key, (arm, method, memory, mark, md_label) in spec.items():
        s = stats[arm]
        runs = arms.mape[arm]
        row = Row(key=key, method=method, memory=memory, budget_s=d["budget_s"], scan_time_s=s["time_s"],
                  mape_pct=s["mape"], ci_pct=s["ci"], success=s["success"])
        t = TableRow(row, runs, method_mark=mark)
        assert t.worst_pct == s["worst"] and runs.size == s["n_runs"], key
        check_md(t, md[md_label])
        table.append(t)

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
    return (
        "T1 error on phantoms identical across methods, unseen in training or checkpoint selection: "
        f"(A) {a_info['n_scans']} held-out; (B) {b_info['n_scans']} clustered-T1 (T1s within "
        f"±{b_info['band_pct']:g}% of a random centre). MAPE: mean absolute percentage error over a scan's "
        "spheres, averaged over scans (bootstrap 95% CI); worst: highest scan MAPE. Fixed grid: log-spaced "
        f"TIs, TR {a_info['tr_s']:g} s. Cramér-Rao: optimised for the plate's T1 range. Distilled: per-block "
        "median TI and TR of the policy. Same T1 fitter for all."
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
         "every cell is checked against `control_narrow_band.md`."),
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

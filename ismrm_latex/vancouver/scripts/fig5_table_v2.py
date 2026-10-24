"""Figure 5 v2: the held-out results table with a worst-scan column and the distilled fixed protocol.

Rows and aggregates come from fig5_table.collect(). The worst scan is the largest of the
24 per-scan MAPEs:
- RL arms: global_best/eval_t1err_heldout.json (5 spheres per scan);
- fixed log TI grid and Cramer-Rao: replayed per scan by fixed_comparators_heldout.jl,
  which reproduces each baseline_summary.json mean (checked again here);
- distilled fixed protocol: the validation-derived median arm of control_schedule_swap.jl.

    julia -t 8 --project=. ismrm_latex/vancouver/scripts/fixed_comparators_heldout.jl
    .venv/bin/python ismrm_latex/vancouver/scripts/fig5_table_v2.py
"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import figstyle
from control_summary import boot_ci
from fig5_table import (DAGGER, DOUBLE_DAGGER, MAX_CAPTION_CHARS, NUMBER_WORDS, RL_RUNS, RUNS,
                        SUCCESS_THRESHOLD, Row, _round, _tex_escape, collect)

STEM = "fig5_table_v2"
FIXED_REPLAY = figstyle.FIGURES / "fig5_v2_data" / "fixed_comparators_heldout.json"
CONTROL = figstyle.FIGURES / "control_data" / "control_results.json"
DISTILLED_ARM = "fixed_median_validation"
SECTION = "§"
HEADER = ["Method", "Memory", "Budget", "Mean scan time", "MAPE % [95% CI]", "Scans < 5%",
          "Worst scan MAPE %"]
REPLAY_SCHEDULES = {"log_grid": "log_grid_trmatched", "cr_optimal": "cr_optimal"}
# The replay must reproduce the recorded means; anything looser is a different run.
MATCH_TOL_PCT = 1e-6


@dataclass(frozen=True)
class TableRow:
    row: Row
    per_scan_mape: np.ndarray
    method_mark: str = ""

    @property
    def worst_pct(self) -> float:
        return 100.0 * float(self.per_scan_mape.max())


def rl_per_scan(key: str, n_scans: int) -> np.ndarray:
    t1err = json.loads((RUNS / RL_RUNS[key] / "global_best" / "eval_t1err_heldout.json").read_text())
    ape = np.asarray(t1err["per_episode_t1_err"]["ape"]).reshape(n_scans, -1)
    return np.clip(ape.mean(axis=1), 0.0, 1.0)


def fixed_per_scan(row: Row, replay: dict, n_scans: int) -> np.ndarray:
    arm = replay["budgets"][_round(row.budget_s)][REPLAY_SCHEDULES[row.key]]
    scans = arm["scans"]
    assert [s["seed"] for s in scans] == list(range(replay["seed_base"], replay["seed_base"] + n_scans))
    mape = np.array([s["mape"] for s in scans])
    assert abs(100 * mape.mean() - row.mape_pct) <= MATCH_TOL_PCT, (row.key, row.budget_s)
    assert np.isclose(np.mean([s["time_s"] for s in scans]), row.scan_time_s, rtol=0, atol=1e-6)
    assert np.mean(mape < SUCCESS_THRESHOLD) == row.success
    ci = [100 * v for v in boot_ci(mape)]
    assert np.allclose(ci, row.ci_pct, rtol=0, atol=MATCH_TOL_PCT), (ci, row.ci_pct)
    return mape


def distilled_row(seed: int, n_scans: int) -> tuple[TableRow, dict]:
    control = json.loads(CONTROL.read_text())
    assert control["seed_base"] == seed and control["n_scans"] == n_scans
    scans = control["arms"][DISTILLED_ARM]
    mape = np.array([s["mape"] for s in scans])
    protocol = control["median_protocols"][DISTILLED_ARM]
    row = Row(
        key="distilled",
        method="Fixed, distilled from RL policy",
        memory="",
        budget_s=control["budget_s"],
        scan_time_s=float(np.mean([s["time_s"] for s in scans])),
        mape_pct=100 * float(mape.mean()),
        ci_pct=tuple(100 * v for v in boot_ci(mape)),
        success=float(np.mean(mape < SUCCESS_THRESHOLD)),
    )
    note = {"n_validation": protocol["n_scans_at_block"][0], "n_blocks": len(protocol["TI_s"]),
            "time_s": protocol["time_s"]}
    return TableRow(row, mape, method_mark=SECTION), note


def build() -> tuple[list[TableRow], dict]:
    rows, info = collect()
    n = info["n_scans"]
    replay = json.loads(FIXED_REPLAY.read_text())
    assert replay["seed_base"] == info["seed"]

    table = []
    for r in rows:
        per_scan = rl_per_scan(r.key, n) if r.method == "RL" else fixed_per_scan(r, replay, n)
        assert np.isclose(100 * per_scan.mean(), r.mape_pct, rtol=0, atol=1e-9), r.key
        table.append(TableRow(r, per_scan))

    distilled, note = distilled_row(info["seed"], n)
    budget = table[0].row.budget_s
    assert distilled.row.budget_s == budget
    last_rl = max(i for i, t in enumerate(table) if t.row.method == "RL" and t.row.budget_s == budget)
    table.insert(last_rl + 1, distilled)

    info["notes"][SECTION] = (
        f"One fixed schedule: per-block median TI and TR of the TI-histogram policy's schedules on "
        f"{note['n_validation']} validation scans ({note['n_blocks']} blocks, {_round(note['time_s'])} s)."
    )
    info["replay"] = {f"{b} s {s}": replay["budgets"][b][s]["max_abs_diff"]["mape_pct"]
                      for b in sorted(replay["budgets"], key=int, reverse=True)
                      for s in REPLAY_SCHEDULES.values()}
    info["numpy_promotion"] = replay["numpy_promotion"]
    return table, info


def cells(t: TableRow) -> list[str]:
    r = t.row
    lo, hi = r.ci_pct
    return [
        r.method + t.method_mark,
        r.memory + r.mark or "-",
        f"{_round(r.budget_s)} s",
        f"{_round(r.scan_time_s)} s",
        f"{_round(r.mape_pct, 2)} [{_round(lo, 2)}, {_round(hi, 2)}]",
        f"{_round(r.success * 100, 1)}%",
        _round(t.worst_pct, 2),
    ]


def _sup(mark: str) -> str:
    return rf"\textsuperscript{{{mark}}}" if mark else ""


def table_tex(table: list[TableRow], info: dict) -> str:
    body = []
    for i, t in enumerate(table):
        r = t.row
        if i and r.budget_s != table[i - 1].row.budget_s:
            body.append(r"\midrule")
        _, _, budget, time, _, success, worst = cells(t)
        lo, hi = r.ci_pct
        tex = [_tex_escape(r.method) + _sup(t.method_mark),
               (_tex_escape(r.memory) + _sup(r.mark)) if r.memory else "--",
               budget, time, _round(r.mape_pct, 2), f"[{_round(lo, 2)}, {_round(hi, 2)}]",
               _tex_escape(success), worst]
        if r.key == info["best_key"]:
            tex = [rf"\textbf{{{c}}}" for c in tex]
        body.append(" & ".join(tex) + r" \\")

    marks = {t.row.mark for t in table} | {t.method_mark for t in table}
    notes = "\n".join(_sup(m) + rf"\,{_tex_escape(info['notes'][m])}\par"
                      for m in (DAGGER, DOUBLE_DAGGER, SECTION) if m in marks)
    rule = "\n".join(body)
    ink = figstyle.INK.lstrip("#")
    return rf"""\documentclass[border=1pt]{{standalone}}
\usepackage{{fontspec}}
\setmainfont{{Arial}}
\usepackage{{booktabs}}
\usepackage{{xcolor}}
\definecolor{{ink}}{{HTML}}{{{ink}}}
\newsavebox{{\tbl}}
\begin{{document}}
\fontsize{{9}}{{11}}\selectfont\color{{ink}}%
\setlength{{\tabcolsep}}{{4.5pt}}%
\begin{{lrbox}}{{\tbl}}%
\begin{{tabular}}{{@{{}}llrrr@{{\hspace{{3pt}}}}lrr@{{}}}}
\toprule
 & & & Mean scan & & & Scans & Worst scan \\
Method & Memory & Budget & time & \multicolumn{{2}}{{c}}{{MAPE \% [95\% CI]}} & \textless{{}}\,5\% & MAPE \% \\
\midrule
{rule}
\bottomrule
\end{{tabular}}%
\end{{lrbox}}%
\begin{{minipage}}{{\wd\tbl}}
\usebox{{\tbl}}\par\vspace{{2.5pt}}
{{\fontsize{{7.5}}{{9}}\selectfont\raggedright
{notes}}}
\end{{minipage}}
\end{{document}}
"""


def render(table: list[TableRow], info: dict) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "table.tex").write_text(table_tex(table, info))
        proc = subprocess.run(["lualatex", "-interaction=nonstopmode", "-halt-on-error", "table.tex"],
                              cwd=tmp, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"lualatex failed:\n{proc.stdout[-3000:]}")
        shutil.copy(tmp / "table.pdf", figstyle.FIGURES / f"{STEM}.pdf")
        subprocess.run(["pdftoppm", "-r", "300", "-png", "-singlefile", str(tmp / "table.pdf"),
                        str(figstyle.FIGURES / STEM)], check=True)


def caption(info: dict) -> str:
    return (
        f"T1 error on {info['n_scans']} held-out random phantoms, identical across methods, unseen in "
        "training or checkpoint selection. MAPE: mean absolute percentage error of a scan's "
        f"{NUMBER_WORDS[info['n_spheres']]} spheres, averaged over scans (bootstrap 95% CI). Worst scan: "
        f"highest scan MAPE. Fixed grid: log-spaced TIs, TR {info['tr_s']:g} s. Cramér-Rao: (TI, TR) "
        "optimised for the plate's T1 range. Distilled: one fixed schedule from the policy's validation "
        f"scans. Same T1 fitter for all. {DAGGER} LSTM checkpoint selected before any full-Bloch training."
    )


def write_text(table: list[TableRow], info: dict, cap: str) -> None:
    # The BOM lets Excel and Word detect UTF-8, so the marks, sigma and e-acute survive a paste.
    with open(figstyle.FIGURES / f"{STEM}.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        writer.writerows(cells(t) for t in table)

    marks = [m for m in (DAGGER, DOUBLE_DAGGER, SECTION)
             if any(m in (t.row.mark, t.method_mark) for t in table)]
    replay = ", ".join(f"{k} {v:.1g} pp" for k, v in info["replay"].items())
    lines = [
        "# Figure 5 v2 - main results table with worst scan",
        "",
        f"Generated by `scripts/fig5_table_v2.py`; held-out seeds {info['seed']}-"
        f"{info['seed'] + info['n_scans'] - 1}; bold is the best row.",
        "Worst scan: the largest of the 24 per-scan MAPEs.",
        "The fixed comparators' per-scan values were replayed by `scripts/fixed_comparators_heldout.jl` "
        f"({info['numpy_promotion']} promotion rules for the wrapper's action round trip); "
        f"largest difference of the replayed mean from `baseline_summary.json`: {replay}.",
        "",
        "| " + " | ".join(HEADER) + " |",
        "|" + "---|" * len(HEADER),
    ]
    for t in table:
        c = cells(t)
        if t.row.key == info["best_key"]:
            c = [f"**{x}**" for x in c]
        lines.append("| " + " | ".join(c) + " |")
    lines += ["", *[f"{m} {info['notes'][m]}" for m in marks], "",
              f"## Caption ({len(cap)} characters)", "", cap, ""]
    (figstyle.FIGURES / f"{STEM}.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    figstyle.apply()
    table, info = build()
    render(table, info)
    cap = caption(info)
    assert len(cap) <= MAX_CAPTION_CHARS, len(cap)
    write_text(table, info, cap)
    for t in table:
        print(" | ".join(cells(t)))
    print(f"Caption ({len(cap)} characters):\n{cap}")


if __name__ == "__main__":
    main()

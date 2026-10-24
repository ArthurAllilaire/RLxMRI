"""Figure 5 (held-out results table) and the caption-less preview figure.

Every number is read from the run outputs under runs/e2/ and cross-checked
between the files that report it; nothing is typed in.

    .venv/bin/python ismrm_latex/vancouver/scripts/fig5_table.py
"""

from __future__ import annotations

import csv
import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, replace
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import figstyle

RUNS = figstyle.ROOT / "runs" / "e2"
SUCCESS_THRESHOLD = 0.05
MAX_CAPTION_CHARS = 500

RL_RUNS = {
    "hist": "mf_runB_5sphere_hist_560s_gpu",
    "nomem": "mf_runB_5sphere_560s_gpu",
    "sigma": "mf_runB_5sphere_sigma_560s_gpu",
    "lstm": "mf_runB_5sphere_lstm_560s_gpu",
    "nomem_240": "mf_runB_cached3_cached_full3_full_gpu",
}
BASELINES = {
    560: (RUNS / "baselines_heldout_560s" / "baseline_summary.json",
          RUNS / "baselines_heldout_560s" / "baseline_heldout.log"),
    240: (RUNS / RL_RUNS["nomem_240"] / "baselines_heldout" / "baseline_summary.json",
          RUNS / RL_RUNS["nomem_240"] / "baseline_heldout.log"),
}
BASELINE_SCHEDULES = {"log_grid": "log_grid_trmatched", "cr_optimal": "cr_optimal"}

NUMBER_WORDS = {5: "five"}
DAGGER = "†"
DOUBLE_DAGGER = "‡"
HEADER = ["Method", "Memory", "Budget", "Mean scan time", "MAPE % [95% CI]", "Scans < 5%"]


@dataclass(frozen=True)
class Row:
    key: str
    method: str
    memory: str
    budget_s: float
    scan_time_s: float
    mape_pct: float
    ci_pct: tuple[float, float]
    success: float
    per_scan_t1_true: tuple[float, ...] = ()
    mark: str = ""


def _round(x: float, places: int = 0) -> str:
    # Half-up, so a log value such as 547.5 s cannot flip with binary rounding.
    q = Decimal(1).scaleb(-places)
    return str(Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP))


def _parse_eval_log(path: Path) -> dict[str, float]:
    text = path.read_text()
    # The first MAPE line is the agent; a later one is eval_e2.py's quick grid sanity check.
    return {
        "mape_pct": float(re.search(r"MAPE\s+=\s+([\d.]+)%", text).group(1)),
        "success_pct": float(re.search(r"Success<5%\s+=\s+([\d.]+)%", text).group(1)),
        "scan_time_s": float(re.search(r"Mean time\s+=\s+([\d.]+)s", text).group(1)),
    }


def _memory_label(cfg: dict) -> str:
    env = cfg["base_env_kwargs"]
    if cfg.get("recurrent"):
        return "LSTM"
    if env.get("include_ti_history"):
        return "TI histogram"
    if env.get("include_sigma"):
        return "fitter uncertainty (σ)"
    return "none"


def load_rl(key: str) -> tuple[Row, dict]:
    run = RUNS / RL_RUNS[key]
    cfg = json.loads((run / "run_config.json").read_text())
    summary = json.loads((run / "global_best" / "eval_summary.json").read_text())
    t1err = json.loads((run / "global_best" / "eval_t1err_heldout.json").read_text())
    log = _parse_eval_log(run / "eval_globalbest_24ep_heldout.log")

    n_scans = summary["episodes"]
    ape = np.asarray(t1err["per_episode_t1_err"]["ape"]).reshape(n_scans, -1)
    # Per-scan MAPE is clamped after averaging the spheres (e2.jl, e2_step!).
    per_scan = np.clip(ape.mean(axis=1), 0.0, 1.0)
    success = float(np.mean(per_scan < SUCCESS_THRESHOLD))

    assert np.isclose(per_scan.mean() * 100, summary["agent_mape_pct"], atol=1e-9), key
    assert np.isclose(t1err["agent_mape_pct"], summary["agent_mape_pct"], atol=1e-9), key
    assert abs(log["mape_pct"] - summary["agent_mape_pct"]) < 0.005 + 1e-9, key
    assert abs(log["success_pct"] - success * 100) < 0.05 + 1e-9, key

    row = Row(
        key=key,
        method="RL",
        memory=_memory_label(cfg),
        budget_s=cfg["base_env_kwargs"]["time_budget_s"],
        scan_time_s=log["scan_time_s"],
        mape_pct=summary["agent_mape_pct"],
        ci_pct=tuple(summary["agent_mape_ci95_pct"]),
        success=success,
        per_scan_t1_true=tuple(t1err["per_episode_t1_err"]["t1_true_s"]),
    )
    best = json.loads((run / "global_best" / "best_meta.json").read_text())
    selection = [(best["seed_offset"], best["n_eval_episodes"])]
    if "source_seed_offset" in best:
        selection.append((best["source_seed_offset"], best["source_n_eval_episodes"]))
    meta = {"n_scans": n_scans, "n_spheres": ape.shape[1], "cfg": cfg, "selection_seeds": selection,
            "best_stage": best["stage"]}
    return row, meta


def load_baselines(budget: int) -> tuple[list[Row], dict]:
    summary_path, log_path = BASELINES[budget]
    summary = json.loads(summary_path.read_text())
    env = summary["_run_meta"]["env_config"]
    assert env["time_budget_s"] == budget
    log = log_path.read_text()
    tr_s = float(re.search(r"log_grid_trmatched: .*TR=([\d.]+)s", log).group(1))
    seeds = re.search(r"Evaluating 'log_grid_trmatched' on (\d+) eps \(seeds (\d+)", log)

    rows = []
    for key, schedule in BASELINE_SCHEDULES.items():
        b = summary[schedule]
        rows.append(Row(
            key=key,
            method=figstyle.METHOD_LABELS[key],
            memory="",
            budget_s=env["time_budget_s"],
            scan_time_s=b["mean_scan_time_s"],
            mape_pct=b["mape_pct"],
            ci_pct=tuple(b["mape_ci95_pct"]),
            success=b["success_5pct"],
        ))
    meta = {
        "n_scans": summary["_run_meta"]["episodes"],
        "seed": summary["_run_meta"]["seed"],
        "log_seed": int(seeds.group(2)),
        "log_n_scans": int(seeds.group(1)),
        "tr_s": tr_s,
    }
    return rows, meta


def full_bloch_steps(key: str) -> int:
    history = json.loads((RUNS / RL_RUNS[key] / "fidelity_history.json").read_text())
    full = {e["kind"]: e["step"] for e in history
            if e["stage"] == "full" and e["kind"] in ("stage_start", "stage_end")}
    return full["stage_end"] - full["stage_start"]


def collect() -> tuple[list[Row], dict]:
    rl, rl_meta = {}, {}
    for key in RL_RUNS:
        rl[key], rl_meta[key] = load_rl(key)
    base560, meta560 = load_baselines(560)
    base240, meta240 = load_baselines(240)

    # Same held-out phantoms for every RL arm, and the same scan count for every method.
    ref = rl["hist"].per_scan_t1_true
    assert all(r.per_scan_t1_true == ref for r in rl.values())
    n_scans = {m["n_scans"] for m in (*rl_meta.values(), meta560, meta240)}
    n_scans |= {meta560["log_n_scans"], meta240["log_n_scans"]}
    assert len(n_scans) == 1, n_scans
    assert meta560["seed"] == meta240["seed"] == meta560["log_seed"] == meta240["log_seed"]
    assert meta560["tr_s"] == meta240["tr_s"]
    n_spheres = {m["n_spheres"] for m in rl_meta.values()}
    assert len(n_spheres) == 1
    # The caption claims the held-out seeds were never used to pick a checkpoint.
    held_out = range(meta560["seed"], meta560["seed"] + min(n_scans))
    for m in rl_meta.values():
        for offset, n in m["selection_seeds"]:
            assert not set(range(offset, offset + n)) & set(held_out), (offset, n)

    lstm_steps, hist_steps = full_bloch_steps("lstm"), full_bloch_steps("hist")
    rl["lstm"] = replace(rl["lstm"], mark=DAGGER)
    flip_angle_arms = [k for k, m in rl_meta.items() if m["cfg"]["base_env_kwargs"]["learn_alpha"]]
    for k in flip_angle_arms:
        rl[k] = replace(rl[k], mark=DOUBLE_DAGGER)

    rows = [rl["hist"], rl["nomem"], rl["sigma"], rl["lstm"], *base560,
            rl["nomem_240"], *base240]
    info = {
        "n_scans": n_scans.pop(),
        "n_spheres": n_spheres.pop(),
        "seed": meta560["seed"],
        "tr_s": meta560["tr_s"],
        "lstm_steps": lstm_steps,
        "hist_steps": hist_steps,
        "lstm_best_stage": rl_meta["lstm"]["best_stage"],
        "best_key": min(rows, key=lambda r: r.mape_pct).key,
    }
    # The LSTM's selected checkpoint predates full-Bloch training, which is the real reason it lags.
    assert info["lstm_best_stage"] != "full"
    info["notes"] = {
        DAGGER: (f"LSTM: selected checkpoint came from the {info['lstm_best_stage']} rung, before any "
                 f"full-Bloch training; the run had {hist_steps / lstm_steps:.1f}× fewer full-Bloch "
                 f"steps than the TI-histogram run ({lstm_steps:,} vs {hist_steps:,})."),
        DOUBLE_DAGGER: "Policy also chose the flip angle.",
    }
    return rows, info


def cells(row: Row) -> list[str]:
    lo, hi = row.ci_pct
    return [
        row.method,
        row.memory + row.mark or "-",
        f"{_round(row.budget_s)} s",
        f"{_round(row.scan_time_s)} s",
        f"{_round(row.mape_pct, 2)} [{_round(lo, 2)}, {_round(hi, 2)}]",
        f"{_round(row.success * 100, 1)}%",
    ]


def _tex_escape(s: str) -> str:
    return s.replace("%", r"\%").replace("<", r"\textless{}")


def table_tex(rows: list[Row], info: dict) -> str:
    body = []
    for i, row in enumerate(rows):
        if i and row.budget_s != rows[i - 1].budget_s:
            body.append(r"\midrule")
        method, _, budget, time, _, success = cells(row)
        memory = row.memory + (rf"\textsuperscript{{{row.mark}}}" if row.mark else "") or "--"
        lo, hi = row.ci_pct
        tex = [method, memory, budget, time, _round(row.mape_pct, 2),
               f"[{_round(lo, 2)}, {_round(hi, 2)}]", success]
        tex = [_tex_escape(c) for c in tex]
        if row.key == info["best_key"]:
            tex = [rf"\textbf{{{c}}}" for c in tex]
        body.append(" & ".join(tex) + r" \\")

    marks = {r.mark for r in rows if r.mark}
    notes = "\n".join(rf"\textsuperscript{{{m}}}\,{_tex_escape(info['notes'][m])}\par"
                      for m in (DAGGER, DOUBLE_DAGGER) if m in marks)
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
\begin{{tabular}}{{@{{}}llrrr@{{\hspace{{3pt}}}}lr@{{}}}}
\toprule
 & & & Mean scan & & & Scans \\
Method & Memory & Budget & time & \multicolumn{{2}}{{c}}{{MAPE \% [95\% CI]}} & \textless{{}}\,5\% \\
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


def render_table(rows: list[Row], info: dict, stem: str) -> None:
    figstyle.FIGURES.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "table.tex").write_text(table_tex(rows, info))
        proc = subprocess.run(["lualatex", "-interaction=nonstopmode", "-halt-on-error", "table.tex"],
                              cwd=tmp, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"lualatex failed for {stem}:\n{proc.stdout[-3000:]}")
        shutil.copy(tmp / "table.pdf", figstyle.FIGURES / f"{stem}.pdf")
        subprocess.run(["pdftoppm", "-r", "300", "-png", "-singlefile", str(tmp / "table.pdf"),
                        str(figstyle.FIGURES / stem)], check=True)


def caption(info: dict) -> str:
    threshold = f"{SUCCESS_THRESHOLD * 100:g}%"
    return (
        f"T1 error on {info['n_scans']} held-out random phantoms, identical for every method and "
        "never used for training or checkpoint selection. MAPE: mean absolute percentage error "
        f"over the {NUMBER_WORDS[info['n_spheres']]} spheres of a scan, averaged over scans, "
        "with bootstrap 95% CI. "
        f"Scans < {threshold}: share of scans with MAPE under {threshold}. Fixed grid: log-spaced TIs, "
        f"TR {info['tr_s']:g} s. Cramér-Rao: (TI, TR) schedule optimised for the plate's T1 range. "
        f"All methods use the same T1 fitter. {DAGGER} LSTM checkpoint selected before any "
        "full-Bloch training."
    )


def write_text_tables(rows: list[Row], info: dict, cap: str) -> None:
    # The BOM lets Excel and Word detect UTF-8, so the daggers, sigma and e-acute survive a paste.
    with open(figstyle.FIGURES / "fig5_table.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        writer.writerows(cells(r) for r in rows)

    marks = [m for m in (DAGGER, DOUBLE_DAGGER) if any(r.mark == m for r in rows)]
    n_first = sum(r.budget_s == rows[0].budget_s for r in rows)
    lines = [
        "# Figure 5 - main results table",
        "",
        f"Generated by `scripts/fig5_table.py`; held-out seeds {info['seed']}-"
        f"{info['seed'] + info['n_scans'] - 1}.",
        f"The {_round(rows[0].budget_s)} s-only variant is the first {n_first} rows; "
        "bold is the best row.",
        "",
        "| " + " | ".join(HEADER) + " |",
        "|" + "---|" * len(HEADER),
    ]
    for r in rows:
        c = cells(r)
        if r.key == info["best_key"]:
            c = [f"**{x}**" for x in c]
        lines.append("| " + " | ".join(c) + " |")
    lines += ["", *[f"{m} {info['notes'][m]}" for m in marks], "",
              f"## Caption ({len(cap)} characters)", "", cap, ""]
    (figstyle.FIGURES / "fig5_table.md").write_text("\n".join(lines), encoding="utf-8")


def preview(rows: list[Row]) -> None:
    by_key = {r.key: r for r in rows if r.budget_s == rows[0].budget_s}
    keys = ["hist", "nomem", "log_grid", "cr_optimal"]
    budget = _round(rows[0].budget_s)

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    fig.subplots_adjust(left=0.405, right=0.95, top=0.80, bottom=0.20)
    ys = np.arange(len(keys))[::-1]
    for y, key in zip(ys, keys):
        r = by_key[key]
        lo, hi = r.ci_pct
        ax.barh(y, r.mape_pct, height=0.62, color=figstyle.METHOD_COLOURS[key], zorder=2)
        # A white halo keeps the whisker visible where it crosses a dark bar.
        for colour, width, cap in (("white", 5.0, 8.5), (figstyle.INK, 2.0, 7.0)):
            ax.errorbar(r.mape_pct, y, xerr=[[r.mape_pct - lo], [hi - r.mape_pct]], fmt="none",
                        ecolor=colour, elinewidth=width, capsize=cap, capthick=width, zorder=3)
        ax.text(hi + 0.25, y, f"{_round(r.mape_pct, 1)}%", va="center", ha="left", fontsize=17,
                fontweight="bold" if key == "hist" else "normal", color=figstyle.INK)

    ax.set_yticks(ys, [figstyle.METHOD_LABELS[k] for k in keys], fontsize=17, color=figstyle.INK)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.tick_params(axis="x", labelsize=14, length=4)
    ax.spines["left"].set_visible(False)
    xmax = max(by_key[k].ci_pct[1] for k in keys)
    ax.set_xlim(0, np.ceil(xmax / 2 + 0.75) * 2)
    ax.set_ylim(-0.6, len(keys) - 0.4)
    ax.xaxis.grid(True, zorder=0)
    ax.set_xlabel("T1 error, MAPE (%)", fontsize=15)
    fig.text(0.035, 0.955, "Adaptive T1 mapping: error on held-out phantoms",
             fontsize=17, fontweight="bold", va="top", color=figstyle.INK)
    fig.text(0.035, 0.875, f"{budget} s scans, simulated, mean with 95% CI",
             fontsize=14, va="top", color=figstyle.INK_SECONDARY)
    # A fixed canvas keeps the required 3:2 aspect; a tight bbox would crop it.
    with mpl.rc_context({"savefig.bbox": "standard"}):
        figstyle.save(fig, "preview")
    plt.close(fig)


def main() -> None:
    figstyle.apply()
    rows, info = collect()
    rows_560 = [r for r in rows if r.budget_s == rows[0].budget_s]
    render_table(rows_560, info, "fig5_table_560")
    render_table(rows, info, "fig5_table_with240")
    # Default until the 240 s rows are decided: the full table, which the current title needs.
    render_table(rows, info, "fig5_table")
    cap = caption(info)
    assert len(cap) <= MAX_CAPTION_CHARS, len(cap)
    write_text_tables(rows, info, cap)
    preview(rows)
    print(f"Caption ({len(cap)} characters):\n{cap}")


if __name__ == "__main__":
    main()

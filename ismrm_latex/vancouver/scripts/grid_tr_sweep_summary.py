"""Summarise the log TI grid TR sweep (grid_tr_sweep.jl) into figures/grid_tr_sweep.md.

Statistics are control_summary.py's: percentile bootstrap CIs, 5000 resamples over the 24
phantoms, numpy default_rng(0); paired differences resample the phantoms the same way. The
reference is the as-built grid at TR 1.7 s, the abstract's comparator. The RL rows are the
held-out evaluations of the 560 s TI-histogram policy and of the 240 s policy, which ran on the
same 24 phantoms.

    .venv/bin/python ismrm_latex/vancouver/scripts/grid_tr_sweep_summary.py
"""

from __future__ import annotations

import json

import numpy as np

import figstyle
from control_summary import (
    N_BOOT,
    SUCCESS_THRESHOLD,
    arm_stats,
    boot_ci,
    fmt_paired,
    paired_stats,
)

DATA = figstyle.FIGURES / "control_data" / "grid_tr_sweep.json"
OUT = figstyle.FIGURES / "grid_tr_sweep.md"
RUNS = figstyle.ROOT / "runs" / "e2"
BASELINE_SUMMARY = RUNS / "baselines_heldout_560s" / "baseline_summary.json"
RL_RUNS = {"rl_560": ("TI-histogram policy, 560 s", "mf_runB_5sphere_hist_560s_gpu"),
           "rl_240": ("RL policy, 240 s", "mf_runB_cached3_cached_full3_full_gpu")}
SERIES = {"as_built": "As built (primary)", "capped": "TI capped (secondary)"}


def rl_per_scan(run: str, t1_true: np.ndarray) -> np.ndarray:
    """Per-scan MAPE of a policy's held-out evaluation, checked against its recorded CI and phantoms."""
    rec = json.loads((RUNS / run / "global_best" / "eval_t1err_heldout.json").read_text())
    n_scans, n_spheres = t1_true.shape
    # eval_e2.py appends each episode's spheres in turn, so rows are scans.
    ape = np.reshape(rec["per_episode_t1_err"]["ape"], (n_scans, n_spheres))
    assert np.array_equal(np.reshape(rec["per_episode_t1_err"]["t1_true_s"], (n_scans, n_spheres)), t1_true), run
    mape = ape.mean(axis=1)
    assert np.allclose(100 * np.array(boot_ci(mape)), rec["agent_mape_ci95_pct"], rtol=0, atol=1e-9), run
    return mape


def tr_cell(p: dict) -> str:
    return f"{p['TR_s']:.3f} ({p['label']})" if p["label"] else f"{p['TR_s']:g}"


def blocks_cell(f: dict) -> str:
    notes = [f"{f['n_lifted']} lifted"] if f["n_lifted"] else []
    if f["n_schedule_blocks_not_run"]:
        notes.append(f"{f['n_schedule_blocks_not_run']} not run")
    if f["ended_by"] == "max_blocks":
        notes.append("max_blocks")
    return f"{f['n_executed']} of {f['n_schedule_blocks']}" + (f" ({', '.join(notes)})" if notes else "")


def ci_cell(s: dict) -> str:
    return f"{s['mape']:.2f} [{s['ci'][0]:.2f}, {s['ci'][1]:.2f}]"


def main() -> None:
    d = json.loads(DATA.read_text())
    ref_series, ref_tr = d["reference"]["series"], d["reference"]["TR_s"]
    points = {s: d["series"][s] for s in SERIES}
    mape = {s: [np.array([r["mape"] for r in p["scans"]]) for p in pts] for s, pts in points.items()}
    ref_i = next(i for i, p in enumerate(points[ref_series]) if p["TR_s"] == ref_tr and not p["label"])
    ref = mape[ref_series][ref_i]

    recorded = json.loads(BASELINE_SUMMARY.read_text())["log_grid_trmatched"]
    assert abs(100 * ref.mean() - recorded["mape_pct"]) < 1e-9
    assert np.allclose(100 * np.array(boot_ci(ref)), recorded["mape_ci95_pct"], rtol=0, atol=1e-9)

    t1_true = np.array([r["T1_true"] for r in points[ref_series][ref_i]["scans"]])
    rl = {k: rl_per_scan(run, t1_true) for k, (_, run) in RL_RUNS.items()}
    # The held-out evaluations saved no per-scan block counts, which arm_stats would also report.
    rl_stats = {k: {"mape": 100 * m.mean(), "ci": tuple(100 * v for v in boot_ci(m)),
                    "success": float(np.mean(m < SUCCESS_THRESHOLD)), "worst": 100 * m.max()} for k, m in rl.items()}

    stats = {s: [arm_stats(m, m, [r["n_blocks"] for r in p["scans"]], [r["time_s"] for r in p["scans"]])
                 for p, m in zip(pts, mape[s])] for s, pts in points.items()}
    vs_ref = {s: [paired_stats(m, ref) for m in mape[s]] for s in SERIES}
    vs_rl = {k: {s: [paired_stats(m, rl[k]) for m in mape[s]] for s in SERIES} for k in rl}
    best = {s: int(np.argmin([st["mape"] for st in stats[s]])) for s in SERIES}

    n_scans, budget, npe = d["n_scans"], d["budget_s"], d["Npe"]
    seeds = f"{d['seed_base']}-{d['seed_base'] + n_scans - 1}"
    pol, pol_ho = d["policy_TRs"]["validation"], d["policy_TRs"]["heldout_for_reference"]
    r_ref = stats[ref_series][ref_i]
    r240, r560 = rl_stats["rl_240"], rl_stats["rl_560"]

    def table(s: str) -> list[str]:
        rows = [("| TR (s) | TI range run (s) | Blocks run of grid | Scan time (s) | Mean MAPE % [95% CI] | "
                 "Scans < 5% | Worst scan % | Paired vs as-built TR 1.7 s (pp) [95% CI] | Paired vs RL 240 s (pp) [95% CI] |"),
                "|---|---|---|---|---|---|---|---|---|"]
        for i, p in enumerate(points[s]):
            f, st = p["feasibility"], stats[s][i]
            lo, hi = f["TI_executed_range_s"]
            bold = "**" if i == best[s] else ""
            paired = "reference" if (s, i) == (ref_series, ref_i) else fmt_paired(vs_ref[s][i])
            rows.append(f"| {bold}{tr_cell(p)}{bold} | {lo:.3f}-{hi:.3f} | {blocks_cell(f)} | {f['scan_time_s']:.0f} | "
                        f"{bold}{ci_cell(st)}{bold} | {round(st['success'] * n_scans)}/{n_scans} | {st['worst']:.2f} | "
                        f"{paired} | {fmt_paired(vs_rl['rl_240'][s][i])} |")
        return rows

    def lifts(s: str) -> list[str]:
        out = []
        for p in points[s]:
            f = p["feasibility"]
            lifted = "; ".join(f"block {b['block']} TI {b['TI_s']:.3f} s, TR {b['TR_requested_s']:.3f} -> "
                               f"{b['TR_executed_s']:.3f} s" for b in f["lifted_blocks"]) or "none"
            rej = f["rejected_block"]
            ended = (f"{f['ended_by']} (TI {rej['TI_s']:.3f} s, needs TR {rej['TR_needed_s']:.3f} s)"
                     if rej and rej["block"] <= f["n_schedule_blocks"] else f["ended_by"])
            out.append(f"| {tr_cell(p)} | {f['n_schedule_blocks']} | {f['n_executed']} | {lifted} | "
                       f"{f['n_schedule_blocks_not_run']} | {ended} | {f['scan_time_s']:.1f} |")
        return out

    def at(s: str, i: int) -> str:
        f = points[s][i]["feasibility"]
        return f"{ci_cell(stats[s][i])}% ({f['n_executed']} blocks, {f['scan_time_s']:.0f} s)"

    def row_of(s: str, label: str | None, tr: float | None = None) -> int:
        return next(i for i, p in enumerate(points[s]) if p["label"] == label and (tr is None or p["TR_s"] == tr))

    def names(sel: list[tuple[str, int]]) -> str:
        return ", ".join(f"{tr_cell(points[s][i])} s {s.replace('_', '-')}" for s, i in sel) or "none"

    all_pts = [(s, i) for s in SERIES for i in range(len(points[s]))]
    s_min, i_min = min(all_pts, key=lambda si: stats[si[0]][si[1]]["mape"])
    ref_rank = 1 + sum(st["mape"] < r_ref["mape"] for st in stats[ref_series])
    better_than_ref = [(s, i) for s, i in all_pts if vs_ref[s][i]["hi"] < 0]
    rl240_not_better = [(s, i) for s, i in all_pts if vs_rl["rl_240"][s][i]["lo"] <= 0]
    rl560_not_better = [(s, i) for s, i in all_pts if vs_rl["rl_560"][s][i]["lo"] <= 0]
    closest_240 = min(range(len(points[ref_series])), key=lambda i: vs_rl["rl_240"][ref_series][i]["mean"])
    capped_ref = row_of("capped", None, ref_tr)
    policy_rows = {lab: {s: row_of(s, lab) for s in SERIES} for lab in ("policy mean TR", "policy median TR")}
    same_schedule = [tr_cell(a) for a, c in zip(points["as_built"], points["capped"]) if a["schedule"] == c["schedule"]]
    block_cap_binds = ", ".join(f"TR {tr_cell(p)} s ({int(budget // (npe * p['TR_s']))} blocks fit the budget)"
                                for p in points["capped"] if budget // (npe * p["TR_s"]) > d["max_blocks"]) or "no TR"
    n_sim = sum(r["n_blocks"] for pts in points.values() for p in pts for r in p["scans"])
    wall = sum(p["wall_s"] for pts in points.values() for p in pts)

    b_as, b_cap = best["as_built"], best["capped"]
    as_better = [si for si in better_than_ref if si[0] == "as_built"]
    below_240 = [(s, i) for s, i in all_pts if stats[s][i]["mape"] < r240["mape"]]
    as_built_verdict = ("No as-built TR is significantly better than 1.7 s, so the reported comparator is close to the "
                        "best TR of its own construction." if not as_better
                        else f"As-built TRs significantly better than 1.7 s: {names(as_better)}.")
    policy_matched = "; ".join(
        f"at the {lab} ({points['as_built'][rows['as_built']]['TR_s']:.3f} s) the grid reaches "
        f"{stats['as_built'][rows['as_built']]['mape']:.2f}% as built and {stats['capped'][rows['capped']]['mape']:.2f}% capped"
        for lab, rows in policy_rows.items())
    rl240_as_built = (
        f"The 240 s policy is significantly better than every as-built row (closest: {tr_cell(points[ref_series][closest_240])} s, "
        f"grid - RL {fmt_paired(vs_rl['rl_240'][ref_series][closest_240])})"
        if all(v["lo"] > 0 for v in vs_rl["rl_240"][ref_series])
        else "The 240 s policy is not significantly better than every as-built row")
    rl240_exceptions = (", but not than " + ", ".join(f"{names([(s, i)])} (grid - RL {fmt_paired(vs_rl['rl_240'][s][i])})"
                                                    for s, i in rl240_not_better) if rl240_not_better else "")
    answers = [
        (f"- **As built (primary)**: the best TR is {tr_cell(points['as_built'][b_as])} s at {at('as_built', b_as)}, "
         f"{fmt_paired(vs_ref['as_built'][b_as])} against TR 1.7 s, which ranks {ref_rank} of {len(points[ref_series])}. "
         f"{as_built_verdict}"),
        (f"- **TI capped (secondary)**: the best TR is {tr_cell(points['capped'][b_cap])} s at {at('capped', b_cap)}, "
         f"{fmt_paired(vs_ref['capped'][b_cap])} against the reported comparator. Rows significantly better than the reported "
         f"comparator: {names(better_than_ref)}. Capping alone at TR 1.7 s gives {at('capped', capped_ref)}, "
         f"{fmt_paired(vs_ref['capped'][capped_ref])}."),
        (f"- **Policy-matched TRs**: {policy_matched}. Short TRs make this grid much worse, so matching the policy's TR "
         "does not bring it near the policy; the TI placement matters as well, consistent with the schedule-shape control "
         "(`control_summary.md`)."),
        (f"- **The 240 s claim ({r240['mape']:.2f}%)**: rows below it in mean MAPE: {names(below_240)}; the lowest is "
         f"{stats[s_min][i_min]['mape']:.2f}% ({names([(s_min, i_min)])}). {rl240_as_built}{rl240_exceptions}."),
        (f"- **The main result ({r560['mape']:.2f}%)**: the closest row is {names([(s_min, i_min)])} at "
         f"{stats[s_min][i_min]['mape']:.2f}%, grid - RL {fmt_paired(vs_rl['rl_560'][s_min][i_min])}; "
         + ("every row is significantly worse than the 560 s policy." if not rl560_not_better
            else f"rows not significantly worse: {names(rl560_not_better)}.")),
        (f"- The best TRs are picked after seeing {len(all_pts)} rows on the same {n_scans} phantoms, so their MAPEs and "
         "paired gains are optimistic."),
    ]
    if rl240_not_better:
        answers.append(
            "- For the abstract: the sentence that the 240 s policy beat both fixed designs at 560 s holds against the reported "
            f"comparator (grid - RL {fmt_paired(vs_rl['rl_240'][ref_series][ref_i])}) and in mean against every grid here, "
            f"but not significantly against {names(rl240_not_better)}; if that grid is cited, say the 240 s policy matched it "
            "rather than beat it.")

    lines = [
        "# Control: TR of the fixed log TI grid comparator",
        "",
        "Question: the abstract's main fixed comparator, a log-spaced TI grid at TR 1.7 s (`log_grid_trmatched`), took its TR from an earlier agent's average rather than from a sweep. Is 1.7 s close to the best TR for this grid, or does another TR give a clearly lower MAPE?",
        f"Phantoms: the {n_scans} held-out scans, seeds {seeds}, with the env config of `{d['env_run']}` (`run_config.json` -> `base_env_kwargs`), as the recorded comparator used.",
        f"Every TR runs on the same phantoms (same T1s and pose) with the same per-seed noise stream, the same E2Env forward model (full Bloch, CPU), the same generalised-IR T1 fitter and the same Python action round trip ({d['numpy_promotion']} promotion rules) as the recorded comparators; only the grid's TR and TIs change.",
        f"Budget {budget:.0f} s, {npe} phase-encode shots per block, TE {1e3 * d['TE_s']:.0f} ms, flip angle 90 deg, at most {d['max_blocks']} blocks.",
        "",
        "## Step 1: reproduction of the recorded comparator",
        "",
        f"The as-built grid at TR 1.7 s reproduces the recorded per-scan results (`{d['reference']['recorded']}`) bit for bit on all {n_scans} scans: T1 true, T1 estimates, MAPE, block count, scan time and executed (TI, TR) blocks.",
        f"Its mean MAPE {r_ref['mape']:.4f}% and bootstrap CI [{r_ref['ci'][0]:.4f}, {r_ref['ci'][1]:.4f}] match `runs/e2/baselines_heldout_560s/baseline_summary.json` to 1e-9.",
        "The RL references below are read from each policy's `global_best/eval_t1err_heldout.json`; their phantoms' true T1s equal the sweep's exactly and their recomputed CIs match the recorded ones, so the paired differences are over the same phantoms.",
        "",
        "## TRs and feasibility rule",
        "",
        f"Grid TRs: {', '.join(f'{t:g}' for t in d['grid_TRs'])} s.",
        f"Policy-matched TRs, from the TI-histogram policy's executed TRs on its {pol['n_scans']} validation scans (`global_best/diagnostics/episodes.json` without each episode's budget-rejected final proposal, {pol['n_blocks']} blocks), so no held-out information shapes a comparator: block-weighted mean {pol['mean_s']:.4f} s (policy mean TR, equalises block count with the policy) and median {pol['median_s']:.4f} s (policy median TR).",
        f"For reference only, the same statistics on the held-out schedules are {pol_ho['mean_s']:.4f} s and {pol_ho['median_s']:.4f} s.",
        "",
        f"`make_log_grid_trmatched` (`python/baseline_e2.py:86-101`) builds floor({budget:.0f} / ({npe} x TR)) blocks at one TR, with TIs log-spaced over {d['TI_range_s'][0]:g}-{d['TI_range_s'][1]:g} s whatever the TR; it does not cap TI by TR.",
        f"The env lifts a block's TR to (TI + TE) / {d['tr_headroom']:g} when TI + TE exceeds {d['tr_headroom']:g} TR, and its budget guard rejects the first block that would overrun {budget:.0f} s, which ends the episode.",
        "",
        "- **As built** (primary): the construction exactly as the reported comparator ran, lifts and rejections included; the TR 1.7 s row is the abstract's comparator. Lifted blocks, grid blocks never run and the reason each episode ended are listed below.",
        f"- **TI capped** (secondary): top TI = min({d['TI_range_s'][1]:g} s, {d['tr_headroom']:g} TR - TE - {1e3 * d['cap_margin_s']:.0f} ms) and at most {d['max_blocks']} blocks, so every block runs at the nominal TR and every grid TI is executed (the script stops otherwise). The 1 ms keeps the top TI clear of the env's lift test after the float32 action round trip. The block cap binds at {block_cap_binds}.",
        "",
        "## Results, as built (primary)",
        "",
        *table("as_built"),
        "",
        "## Results, TI capped (secondary)",
        "",
        *table("capped"),
        "",
        "RL references on the same phantoms: " + "; ".join(
            f"{label} (`{run}`) {ci_cell(rl_stats[k])}%, {round(rl_stats[k]['success'] * n_scans)}/{n_scans} scans < 5%, "
            f"worst {rl_stats[k]['worst']:.2f}%" for k, (label, run) in RL_RUNS.items()) + ".",
        "MAPE: mean over the five spheres of |T1 estimate - T1 true| / T1 true, averaged over scans. Bold: the lowest mean MAPE of the series.",
        f"CI: percentile bootstrap, {N_BOOT} resamples over the {n_scans} phantoms, numpy `default_rng(0)` (as `python/baseline_e2.py:41-58`). Paired: grid minus the other arm per phantom, bootstrapped the same way; negative means the grid row is more accurate.",
        "Blocks run of grid: executed blocks out of the construction's TIs; lifted blocks ran at a longer TR than the grid's, and blocks not run did not fit the remaining budget.",
        f"Both series run the same schedule at TR {', '.join(same_schedule)} s, where the cap does not bind." if same_schedule else "",
        "",
        "## Lifted and rejected blocks",
        "",
        "Every scan executes the same blocks (lifts and the budget guard depend only on the timings; the script checks it).",
        "",
        *(line for s in SERIES for line in [
            f"{SERIES[s]}:",
            "",
            "| TR (s) | Grid TIs | Run | Lifted blocks (TR requested -> executed) | Grid blocks not run | Episode ended by | Scan time (s) |",
            "|---|---|---|---|---|---|---|",
            *lifts(s),
            ""]),
        "## Interpretation",
        "",
        *answers,
        "",
        "## Runtime",
        "",
        (f"Local CPU, Julia {d['julia_version']}, {d['threads']} threads: {wall / 60:.0f} min for {n_sim} full-Bloch blocks "
         f"({wall / n_sim:.2f} s per block including the fit)."),
        "",
        "## Regenerate",
        "",
        "From the RLxMRI repo root:",
        "",
        "```bash",
        "julia -t 8 --project=. ismrm_latex/vancouver/scripts/grid_tr_sweep.jl",
        ".venv/bin/python ismrm_latex/vancouver/scripts/grid_tr_sweep_summary.py",
        "```",
        "",
        "Outputs: `figures/control_data/grid_tr_sweep.json` (per-scan results, schedules and feasibility of every row) and this file.",
        "",
    ]
    OUT.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()

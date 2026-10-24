"""Summarise the schedule-swap control (control_schedule_swap.jl) into figures/control_summary.md.

Bootstrap CIs follow python/baseline_e2.py:_mape_uncertainty (5000 percentile
resamples over scans, numpy default_rng(0)); paired differences resample the 24
phantoms the same way. Reference rows are read from the held-out baseline run.

    .venv/bin/python ismrm_latex/vancouver/scripts/control_summary.py
"""

from __future__ import annotations

import json

import numpy as np

import figstyle

RUN = figstyle.ROOT / "runs" / "e2" / "mf_runB_5sphere_hist_560s_gpu"
BASELINE_DIR = figstyle.ROOT / "runs" / "e2" / "baselines_heldout_560s"
DATA = figstyle.FIGURES / "control_data" / "control_results.json"
OUT = figstyle.FIGURES / "control_summary.md"
SUCCESS_THRESHOLD = 0.05
N_BOOT = 5000
N_SHORTEST = 5
# Post hoc: the phantoms with the shortest sphere T1s, where every non-adaptive schedule failed.
N_TAIL = 2
# python/baseline_e2.py:_LOG_GRID_TR and log_ti_grid's range; julia/rl/e2.jl:e2_tr_headroom.
LOG_GRID_TR_S = 1.7
LOG_GRID_TI_RANGE_S = (0.05, 3.0)
TE_S = 0.020
TR_HEADROOM = 0.90

MEDIAN_ARMS = {"fixed_median": ("Fixed median protocol", "medians of held-out schedules"),
               "fixed_median_validation": ("Fixed median protocol, validation-derived",
                                           "medians of validation schedules")}


def boot_ci(x: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    boot = x[rng.integers(0, x.size, size=(N_BOOT, x.size))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi)


def executed_blocks(tis, trs, budget_s: float, npe: int) -> tuple[int, float]:
    """Blocks a fixed schedule runs in E2Env: TR lifted for TI + TE, stop at the first overrun."""
    t, n = 0.0, 0
    for ti, tr in zip(tis, trs):
        tr = max(tr, (ti + TE_S) / TR_HEADROOM)
        if t + npe * tr > budget_s:
            break
        t, n = t + npe * tr, n + 1
    return n, t


def arm_stats(mape_runs: np.ndarray, mape_per_phantom: np.ndarray, blocks, times) -> dict:
    return {"mape": 100 * mape_per_phantom.mean(), "ci": tuple(100 * v for v in boot_ci(mape_per_phantom)),
            "success": float(np.mean(mape_runs < SUCCESS_THRESHOLD)), "n_runs": mape_runs.size,
            "worst": 100 * mape_runs.max(), "blocks": float(np.mean(blocks)), "time_s": float(np.mean(times))}


def paired_stats(x: np.ndarray, adaptive: np.ndarray) -> dict:
    diff = x - adaptive
    lo, hi = boot_ci(diff)
    return {"mean": 100 * diff.mean(), "lo": 100 * lo, "hi": 100 * hi, "median": 100 * np.median(diff),
            "worse": int(np.sum(diff > 0)), "n": diff.size}


def references(d: dict) -> dict:
    summary = json.loads((BASELINE_DIR / "baseline_summary.json").read_text())
    cr = json.loads((BASELINE_DIR / "cr_optimal_schedule.json").read_text())
    budget, npe = d["budget_s"], d["Npe"]
    n_grid = int(np.floor(budget / (npe * LOG_GRID_TR_S)))
    grid_tis = np.exp(np.linspace(*np.log(LOG_GRID_TI_RANGE_S), n_grid))
    schedules = {"log_grid_trmatched": (grid_tis, [LOG_GRID_TR_S] * n_grid),
                 "cr_optimal": (cr["TIs"], cr["TRs"])}
    out = {}
    for key, (tis, trs) in schedules.items():
        s = summary[key]
        n, t = executed_blocks(tis, trs, budget, npe)
        # The rebuilt schedule must spend exactly the scan time the baseline run logged.
        assert abs(t - s["mean_scan_time_s"]) < 1e-2, (key, t, s["mean_scan_time_s"])
        out[key] = {"mape": s["mape_pct"], "ci": tuple(s["mape_ci95_pct"]), "success": s["success_5pct"],
                    "n_runs": s["n_episodes"], "blocks": float(n), "time_s": t}
    return out


def fmt_row(label: str, source: str, s: dict) -> str:
    # The baseline run saved no per-scan MAPEs, so reference rows have no worst scan.
    worst = f"{s['worst']:.2f}" if "worst" in s else "-"
    return (f"| {label} | {source} | {s['mape']:.2f} [{s['ci'][0]:.2f}, {s['ci'][1]:.2f}] | "
            f"{100 * s['success']:.1f}% ({round(s['success'] * s['n_runs'])}/{s['n_runs']}) | {worst} | "
            f"{s['blocks']:.2f} | {s['time_s']:.0f} |")


def fmt_paired(p: dict) -> str:
    return f"{p['mean']:+.2f} pp [{p['lo']:+.2f}, {p['hi']:+.2f}]"


def main() -> None:
    d = json.loads(DATA.read_text())
    arms, shifts, n_scans = d["arms"], d["shifts"], d["n_scans"]
    mape = {k: np.array([r["mape"] for r in v]) for k, v in arms.items()}
    blocks = {k: [r["n_blocks"] for r in v] for k, v in arms.items()}
    times = {k: [r["time_s"] for r in v] for k, v in arms.items()}

    rec = json.loads((RUN / "global_best" / "eval_t1err_heldout.json").read_text())
    ci_adaptive = boot_ci(mape["adaptive"])
    assert np.allclose(100 * np.array(ci_adaptive), rec["agent_mape_ci95_pct"], rtol=0, atol=1e-9), ci_adaptive
    assert np.array_equal(mape["identity"], mape["adaptive"])

    swap_keys = [f"swap_{s}" for s in shifts]
    swap_runs = np.concatenate([mape[k] for k in swap_keys])
    mape["swap_mean"] = np.mean([mape[k] for k in swap_keys], axis=0)
    swap_label = f"Swapped, mean of {len(shifts)} shifts"

    stats = {"adaptive": arm_stats(mape["adaptive"], mape["adaptive"], blocks["adaptive"], times["adaptive"])}
    for k in swap_keys:
        stats[k] = arm_stats(mape[k], mape[k], blocks[k], times[k])
    stats["swap_mean"] = arm_stats(swap_runs, mape["swap_mean"], sum((blocks[k] for k in swap_keys), []),
                                   sum((times[k] for k in swap_keys), []))
    for k in MEDIAN_ARMS:
        stats[k] = arm_stats(mape[k], mape[k], blocks[k], times[k])
    refs = references(d)

    rows = [("Adaptive (policy in the loop)", "own, chosen online", stats["adaptive"])]
    rows += [(f"Swapped, shift {s}", f"scan (k + {s}) mod {n_scans}", stats[k]) for s, k in zip(shifts, swap_keys)]
    rows.append((swap_label, "another scan's", stats["swap_mean"]))
    rows += [(label, source, stats[k]) for k, (label, source) in MEDIAN_ARMS.items()]
    rows.append(("Fixed log TI grid, TR 1.7 s (reference)", "fixed", refs["log_grid_trmatched"]))
    rows.append(("Cramer-Rao optimal (reference)", "fixed", refs["cr_optimal"]))

    compared = [*swap_keys, "swap_mean", *MEDIAN_ARMS]
    labels = {**{k: f"Swapped, shift {s}" for s, k in zip(shifts, swap_keys)}, "swap_mean": swap_label,
              **{k: label for k, (label, _) in MEDIAN_ARMS.items()}}
    paired = {k: paired_stats(mape[k], mape["adaptive"]) for k in compared}

    def shortest_sphere_ape(arm: str, i: int) -> float:
        r = arms[arm][i]
        return 100 * r["ape"][int(np.argmin(r["T1_true"]))]

    min_t1 = np.array([min(r["T1_true"]) for r in arms["adaptive"]])
    min_ti = np.array([min(s["TI_s"]) for s in d["schedules"]])
    r_min = float(np.corrcoef(np.log(min_t1), np.log(min_ti))[0, 1])
    by_min_t1 = [int(i) for i in np.argsort(min_t1)]
    tail = by_min_t1[:N_TAIL]
    keep = np.setdiff1d(np.arange(n_scans), tail)
    paired_rest = {k: paired_stats(mape[k][keep], mape["adaptive"][keep]) for k in compared}
    tail_ape_fixed = " and ".join(" / ".join(f"{shortest_sphere_ape(k, i):.1f}%" for k in MEDIAN_ARMS) + f" on phantom {i}'s"
                                  for i in tail)
    swap_diff = mape["swap_mean"] - mape["adaptive"]
    swap_top = int(np.argmax(swap_diff))
    swap_top_share = 100 * swap_diff[swap_top] / swap_diff.sum()

    protocols = d["median_protocols"]
    ph, pv = protocols["fixed_median"], protocols["fixed_median_validation"]
    wall, sim_blocks = d["wall_s"], d["simulated_blocks"]
    total_wall, total_blocks = sum(wall.values()), sum(sim_blocks.values())
    repro = d["reproduction"]
    a, fm, fv, sw = stats["adaptive"], stats["fixed_median"], stats["fixed_median_validation"], stats["swap_mean"]
    lg, cr = refs["log_grid_trmatched"], refs["cr_optimal"]
    seeds = f"{d['seed_base']}-{d['seed_base'] + n_scans - 1}"
    tail_names = " and ".join(str(i) for i in tail)
    tail_t1s = " and ".join(f"{1e3 * min_t1[i]:.0f}" for i in tail)
    tail_tis = " and ".join(f"{1e3 * min_ti[i]:.0f}" for i in tail)
    tail_ape_adaptive = " and ".join(f"{shortest_sphere_ape('adaptive', i):.1f}%" for i in tail)

    lines = [
        "# Control: adaptation versus schedule shape",
        "",
        "Question: does the TI-histogram policy's held-out T1 accuracy come from adapting the schedule to each scan, or only from the shape of its schedules (short TIs paired with short TRs, about 13 blocks in 560 s)?",
        f"Policy: `{RUN.relative_to(figstyle.ROOT)}`, checkpoint `global_best`.",
        f"Phantoms: the {n_scans} held-out scans, seeds {seeds}, as in `global_best/eval_t1err_heldout.json`.",
        "Every arm runs on the same phantoms (same T1s and pose) with the same per-seed noise stream, the same E2Env forward model (full Bloch, CPU) and the same generalised-IR T1 fitter; only the (TI, TR) schedule differs.",
        "",
        "## Step 1: reproduction of the held-out evaluation",
        "",
        f"The pure-Julia rollout of `global_best` on seeds {seeds} reproduces `eval_t1err_heldout.json` exactly.",
        f"Largest per-sphere APE difference: {repro['max_abs_diff_ape']:.1e}; largest true-T1 difference: {repro['max_abs_diff_t1_true_s']:.1e} s.",
        f"Mean MAPE {repro['mape_pct_rollout']:.4f}% (recorded {repro['mape_pct_recorded']:.4f}%), and the bootstrap CI below matches `eval_summary.json`.",
        "Replaying each scan's own executed schedule without the policy (identity replay) reproduces every scan's final T1 estimates bit for bit, so the replay path is the same pipeline the policy was scored on.",
        "",
        "## Arms",
        "",
        "- **Adaptive**: each phantom gets the schedule the policy chose for it online.",
        f"- **Swapped**: phantom k gets the executed schedule of scan (k + s) mod {n_scans}, for s = {', '.join(map(str, shifts))} ({len(shifts) * n_scans} runs, no policy in the loop).",
        "  Each shift is a derangement, so every schedule is used once per shift and never on its own phantom; block counts and scan times are therefore the same as Adaptive.",
        f"- **Fixed median protocol**: per block index, the median log TI and median TR over the {n_scans} adaptive held-out schedules (over the scans that reached that block), truncated so that {d['Npe']} x sum(TR) <= {d['budget_s']:.0f} s; the same schedule on all {n_scans} phantoms.",
        "- **Fixed median protocol, validation-derived** (added robustness check): the same recipe applied to the policy's executed schedules on the 24 validation scans (seed 500000, `global_best/diagnostics/episodes.json` without the budget-rejected final proposals), so no held-out information shapes the protocol.",
        "- **References**: the held-out baseline run (`runs/e2/baselines_heldout_560s/baseline_summary.json`); per-scan values were not saved, so they have no worst scan or paired difference.",
        "",
        f"## Results ({n_scans} held-out phantoms)",
        "",
        "| Arm | Schedule | Mean MAPE % [95% CI] | Scans < 5% | Worst scan MAPE % | Mean blocks | Mean scan time (s) |",
        "|---|---|---|---|---|---|---|",
        *(fmt_row(*row) for row in rows),
        "",
        "MAPE: mean over the five spheres of |T1 estimate - T1 true| / T1 true, averaged over scans.",
        f"CI: percentile bootstrap, {N_BOOT} resamples over the {n_scans} phantoms, numpy `default_rng(0)` (as `python/baseline_e2.py:41-58`).",
        f"For the swap mean, each phantom's {len(shifts)} swapped MAPEs are averaged before resampling; its success share and worst scan are over all {swap_runs.size} runs.",
        "",
        "## Paired differences to Adaptive",
        "",
        f"| Arm | Mean difference, arm - Adaptive [95% CI] | Median per-phantom difference (pp) | Phantoms where the arm is worse | Without phantoms {tail_names} (post hoc) |",
        "|---|---|---|---|---|",
        *(f"| {labels[k]} | {fmt_paired(paired[k])} | {paired[k]['median']:+.2f} | {paired[k]['worse']}/{paired[k]['n']} | "
          f"{fmt_paired(paired_rest[k])} |" for k in compared),
        "",
        f"Paired bootstrap over the phantoms ({N_BOOT} resamples, `default_rng(0)`); for the swap mean, the {len(shifts)} shifts are averaged per phantom first.",
        f"The last column drops the {N_TAIL} phantoms with the shortest sphere T1 (chosen after seeing the results; see below).",
        "",
        "## Where adaptation does work (post hoc)",
        "",
        f"The {N_SHORTEST} phantoms with the shortest sphere T1; APE is that sphere's error, MAPE the scan's.",
        f"Across all {n_scans} held-out scans, the policy's shortest TI tracks the phantom's shortest T1 (Pearson r = {r_min:+.2f} on log scales).",
        f"The fixed median protocols' shortest TIs are {1e3 * min(ph['TI_s']):.0f} ms (held-out-derived) and {1e3 * min(pv['TI_s']):.0f} ms (validation-derived).",
        "",
        "| Phantom | Shortest T1 (s) | Policy's shortest TI (s) | Shortest-sphere APE %: adaptive / swapped mean / fixed median / fixed median (val.) | Scan MAPE %: adaptive / swapped mean / fixed median / fixed median (val.) |",
        "|---|---|---|---|---|",
    ]
    for i in by_min_t1[:N_SHORTEST]:
        ape_sw = np.mean([shortest_sphere_ape(k, i) for k in swap_keys])
        ape_fixed = " / ".join(f"{shortest_sphere_ape(k, i):.1f}" for k in MEDIAN_ARMS)
        mape_fixed = " / ".join(f"{100 * mape[k][i]:.2f}" for k in MEDIAN_ARMS)
        lines.append(f"| {i} | {min_t1[i]:.3f} | {min_ti[i]:.3f} | {shortest_sphere_ape('adaptive', i):.1f} / {ape_sw:.1f} / "
                     f"{ape_fixed} | {100 * mape['adaptive'][i]:.2f} / {100 * mape['swap_mean'][i]:.2f} / {mape_fixed} |")
    lines += [
        "",
        "## Interpretation",
        "",
        "The mean accuracy gain over the fixed comparators comes mostly from the shape of the policy's schedules, not from tailoring each schedule to its scan.",
        f"A single fixed protocol built from the per-block medians of the policy's held-out schedules reaches {fm['mape']:.2f}% [{fm['ci'][0]:.2f}, {fm['ci'][1]:.2f}] against {a['mape']:.2f}% [{a['ci'][0]:.2f}, {a['ci'][1]:.2f}] for the adaptive policy (paired {fmt_paired(paired['fixed_median'])}); built only from validation scans it reaches {fv['mape']:.2f}% [{fv['ci'][0]:.2f}, {fv['ci'][1]:.2f}] (paired {fmt_paired(paired['fixed_median_validation'])}).",
        f"Both are well below the fixed log TI grid ({lg['mape']:.2f}%) and the Cramer-Rao schedule ({cr['mape']:.2f}%), and neither mean differs significantly from the adaptive policy at n = {n_scans}.",
        f"Giving each phantom another scan's schedule costs {fmt_paired(paired['swap_mean'])} ({sw['mape']:.2f}% [{sw['ci'][0]:.2f}, {sw['ci'][1]:.2f}], median per phantom {paired['swap_mean']['median']:+.2f} pp): a borderline penalty, {swap_top_share:.0f}% of it from phantom {swap_top}.",
        f"Per-scan adaptation shows in the tail, not the mean: phantoms {tail_names} have spheres of {tail_t1s} ms, the policy placed TIs down to {tail_tis} ms and measured those spheres to {tail_ape_adaptive}, while the fixed median protocols (held-out-derived / validation-derived) got {tail_ape_fixed}.",
        f"As a result, the worst scan is {a['worst']:.2f}% for the adaptive policy against {fm['worst']:.2f}% and {fv['worst']:.2f}% for the fixed median protocols and {sw['worst']:.2f}% for swapped schedules.",
        f"Without those {N_TAIL} phantoms (a post hoc cut), the fixed median protocols match or slightly beat the adaptive policy ({fmt_paired(paired_rest['fixed_median'])} and {fmt_paired(paired_rest['fixed_median_validation'])}).",
        f"With only {N_TAIL} such phantoms, this robustness effect is an observation, not a measured rate.",
        f"For the abstract: claim that RL found a schedule shape (short TIs paired with short TRs, about 13 blocks in 560 s) that cuts the error of the best fixed comparator from {lg['mape']:.2f}% to {a['mape']:.2f}%, and that a fixed protocol distilled from it keeps most of that gain; claim per-scan adaptation only as a robustness effect seen on rare very short T1s, not as the source of the mean gain.",
        "",
        "## Fixed median protocols",
        "",
        "| Block | Held-out-derived TI (s) | TR (s) | Scans reaching block | Validation-derived TI (s) | TR (s) | Scans reaching block |",
        "|---|---|---|---|---|---|---|",
    ]

    def cells(p: dict, b: int) -> str:
        if b >= len(p["TI_s"]):
            return " | | "
        return f"{p['TI_s'][b]:.4f} | {p['TR_s'][b]:.4f} | {p['n_scans_at_block'][b]}"

    for b in range(max(len(ph["TI_s"]), len(pv["TI_s"]))):
        lines.append(f"| {b + 1} | {cells(ph, b)} | {cells(pv, b)} |")
    lines += [
        f"| Total | {len(ph['TI_s'])} blocks, {ph['time_s']:.1f} s | | | {len(pv['TI_s'])} blocks, {pv['time_s']:.1f} s | | |",
        "",
        "## Runtime",
        "",
        f"Local CPU, Julia {d['julia_version']}, {d['threads']} threads: {total_wall / 60:.1f} min for {total_blocks} full-Bloch blocks ({total_wall / total_blocks:.2f} s per block including the fit).",
        "",
        "| Arm | Simulated blocks | Wall time (s) |",
        "|---|---|---|",
        *(f"| {k} | {sim_blocks[k]} | {wall[k]:.0f} |" for k in ["adaptive", "identity", *swap_keys, *MEDIAN_ARMS]),
        "",
        "## Regenerate",
        "",
        "From the RLxMRI repo root (the policy arrays come from `fig3evo_export_policies.py`, already in `figures/`):",
        "",
        "```bash",
        "julia -t 8 --project=. ismrm_latex/vancouver/scripts/control_schedule_swap.jl",
        ".venv/bin/python ismrm_latex/vancouver/scripts/control_summary.py",
        "```",
        "",
        "Outputs: `figures/control_data/control_adaptive_heldout.json` (full adaptive rollouts), `figures/control_data/control_results.json` (per-phantom results of every arm), and this file.",
        "",
    ]
    OUT.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()

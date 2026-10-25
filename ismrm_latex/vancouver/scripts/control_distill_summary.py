"""Summarise the distilled-protocol control (control_distill_variants.jl) into
figures/control_distill_variants.md.

Statistics are control_summary.py's: percentile bootstrap CIs, 5000 resamples over the
24 held-out phantoms, numpy default_rng(0); paired differences resample the phantoms the
same way. Each policy's adaptive reference is its own eval_t1err_heldout.json.

    .venv/bin/python ismrm_latex/vancouver/scripts/control_distill_summary.py
"""

from __future__ import annotations

import json

import numpy as np

import figstyle
from control_summary import (
    N_BOOT,
    N_TAIL,
    SUCCESS_THRESHOLD,
    boot_ci,
    fmt_paired,
    paired_stats,
    references,
)

DATA = figstyle.FIGURES / "control_data" / "control_distill_variants.json"
OUT = figstyle.FIGURES / "control_distill_variants.md"
LABELS = {"histogram": "TI histogram", "no_memory": "No memory", "sigma": "Fitter uncertainty"}
VARIANTS = list(LABELS)
REF = "histogram"
# julia/rl/e2.jl:e2_action_lo.
TR_FLOOR_S = 0.5
REF_ROWS = {"log_grid_trmatched": "Fixed log TI grid, TR 1.7 s", "cr_optimal": "Cramer-Rao optimal"}


def scan_stats(m: np.ndarray) -> dict:
    lo, hi = boot_ci(m)
    return {"mape": 100 * m.mean(), "ci": (100 * lo, 100 * hi), "success": float(np.mean(m < SUCCESS_THRESHOLD)),
            "n": m.size, "worst": 100 * m.max()}


def fmt_stats(s: dict) -> str:
    return (f"{s['mape']:.2f} [{s['ci'][0]:.2f}, {s['ci'][1]:.2f}] | "
            f"{100 * s['success']:.1f}% ({round(s['success'] * s['n'])}/{s['n']}) | {s['worst']:.2f}")


def training_diff(runs: dict) -> list[str]:
    """run_config.json differences outside base_env_kwargs (the Julia script checks those)."""
    cfgs = {v: json.loads((figstyle.ROOT / r / "run_config.json").read_text()) for v, r in runs.items()}
    flat = {v: {"plan": c["plan"], **{f"mf.{k}": x for k, x in c["mf"].items()}} for v, c in cfgs.items()}
    keys = sorted(set().union(*flat.values()))
    return [f"`{k}`: " + ", ".join(f"{LABELS[v]} {flat[v].get(k, 'unset')}" for v in VARIANTS)
            for k in keys if len({json.dumps(flat[v].get(k)) for v in VARIANTS}) > 1]


def main() -> None:
    d = json.loads(DATA.read_text())
    var, n_scans = d["variants"], d["n_scans"]
    runs = {v: var[v]["run"] for v in VARIANTS}

    adaptive = {v: np.array(var[v]["adaptive"]["mape"]) for v in VARIANTS}
    distilled = {v: np.array([r["mape"] for r in var[v]["distilled"]]) for v in VARIANTS}
    for v in VARIANTS:
        rec = json.loads((figstyle.ROOT / runs[v] / "global_best" / "eval_t1err_heldout.json").read_text())
        ci = 100 * np.array(boot_ci(adaptive[v]))
        assert np.allclose(ci, rec["agent_mape_ci95_pct"], rtol=0, atol=1e-9), (v, ci)
    stats = {(v, arm): scan_stats(m[v]) for v in VARIANTS for arm, m in (("adaptive", adaptive), ("distilled", distilled))}
    proto = {v: var[v]["protocol"] for v in VARIANTS}

    t1_true = np.array([r["T1_true"] for r in var[REF]["distilled"]])
    min_t1 = t1_true.min(axis=1)
    tail = [int(i) for i in np.argsort(min_t1)[:N_TAIL]]
    keep = np.setdiff1d(np.arange(n_scans), tail)

    def shortest_ape(v: str, arm: str, i: int) -> float:
        if arm == "adaptive":
            ape, t1 = var[v]["adaptive"]["ape"][i], var[v]["adaptive"]["T1_true"][i]
        else:
            ape, t1 = var[v]["distilled"][i]["ape"], var[v]["distilled"][i]["T1_true"]
        return 100 * ape[int(np.argmin(t1))]

    arm_mape = {(v, arm): m[v] for v in VARIANTS for arm, m in (("adaptive", adaptive), ("distilled", distilled))}
    comparisons = [((v, "distilled"), (v, "adaptive")) for v in VARIANTS]
    for v in VARIANTS[1:]:
        comparisons += [((v, "adaptive"), (REF, "adaptive")), ((v, "distilled"), (REF, "distilled")),
                        ((v, "distilled"), (REF, "adaptive"))]
    paired = {c: paired_stats(arm_mape[c[0]], arm_mape[c[1]]) for c in comparisons}
    paired_rest = {c: paired_stats(arm_mape[c[0]][keep], arm_mape[c[1]][keep]) for c in comparisons}
    # The paired no-memory gap splits exactly into its own adaptation cost, the shape
    # difference of the distilled protocols, and the TI-histogram policy's adaptation gain.
    terms = [(("no_memory", "adaptive"), ("no_memory", "distilled")), (("no_memory", "distilled"), (REF, "distilled")),
             ((REF, "distilled"), (REF, "adaptive"))]
    split = [paired_stats(arm_mape[x], arm_mape[y]) for x, y in terms]
    split_rest = [paired_stats(arm_mape[x][keep], arm_mape[y][keep]) for x, y in terms]
    refs = references(d)

    def name(key: tuple[str, str]) -> str:
        return f"{LABELS[key[0]]} {key[1]}"

    env_lines = []
    for v in VARIANTS[1:]:
        diff = var[v]["env_diff"]
        items = ", ".join(f"`{k}` {json.dumps(x['variant'])} (TI histogram {json.dumps(x['histogram'])})"
                          for k, x in sorted(diff.items()))
        env_lines.append(f"- {LABELS[v]} (`{runs[v]}`): {items}.")
    sanity = d["sanity_check"]
    seeds = f"{d['seed_base']}-{d['seed_base'] + n_scans - 1}"
    wall, sim_blocks = d["wall_s"], d["simulated_blocks"]
    total_wall, total_blocks = sum(wall.values()), sum(sim_blocks.values())
    a, f = ({v: stats[(v, arm)] for v in VARIANTS} for arm in ("adaptive", "distilled"))
    tail_names = " and ".join(str(i) for i in tail)

    nm, sg = "no_memory", "sigma"
    nm_own, sg_own = (paired[((v, "distilled"), (v, "adaptive"))] for v in (nm, sg))
    nm_vs_ref = paired[((nm, "distilled"), (REF, "distilled"))]
    gap = paired[((nm, "adaptive"), (REF, "adaptive"))]
    along = 100 * (f[nm]["mape"] - f[REF]["mape"]) / (a[nm]["mape"] - f[REF]["mape"])

    def tail_apes(v: str) -> str:
        return " and ".join(f"{shortest_ape(v, 'distilled', i):.1f}%" for i in tail)

    def ape_by_t1_rank(ape, t1) -> np.ndarray:
        return 100 * np.take_along_axis(np.asarray(ape), np.argsort(t1, axis=1), axis=1).mean(axis=0)

    sg_rank = zip(ape_by_t1_rank(var[sg]["adaptive"]["ape"], var[sg]["adaptive"]["T1_true"]),
                  ape_by_t1_rank([r["ape"] for r in var[sg]["distilled"]], [r["T1_true"] for r in var[sg]["distilled"]]))
    sg_floor = sum(abs(tr - TR_FLOOR_S) < 1e-6 for tr in proto[sg]["TR_s"])
    sg_last = proto[sg]["n_scans_at_block"][-3:]
    interpretation = [
        f"Decision rule: a distilled no-memory protocol near the TI-histogram run's distilled {f[REF]['mape']:.2f}% would mean both policies found an equally good schedule and no-memory loses accuracy through its adaptation; near its own adaptive {a[nm]['mape']:.2f}% or worse would mean its schedule shape is worse.",
        f"Result: {f[nm]['mape']:.2f}% [{f[nm]['ci'][0]:.2f}, {f[nm]['ci'][1]:.2f}], in between ({along:.0f}% of the way from {f[REF]['mape']:.2f}% to {a[nm]['mape']:.2f}%), and the intervals are too wide to call it either way at n = {n_scans} with one training seed per run.",
        f"Paired, it is {fmt_paired(nm_vs_ref)} against the TI-histogram distilled protocol and {fmt_paired(nm_own)} against the no-memory adaptive policy; both intervals include zero.",
        f"The paired gap between the two adaptive policies, {fmt_paired(gap)}, splits exactly into three paired terms: the no-memory policy's adaptation cost (its adaptive - distilled) {fmt_paired(split[0])}, the shape difference (no-memory distilled - TI-histogram distilled) {fmt_paired(split[1])}, and the TI-histogram policy's adaptation gain (its distilled - adaptive) {fmt_paired(split[2])}.",
        f"On all {n_scans} phantoms the point estimates put most of the gap on adaptation and little on shape, but no term is resolved, and the shape term is held down by the TI-histogram protocol's failures on phantoms {tail_names}.",
        f"Without those {N_TAIL} phantoms (post hoc), the terms become {fmt_paired(split_rest[0])}, {fmt_paired(split_rest[1])} and {fmt_paired(split_rest[2])}: on typical phantoms the no-memory schedule shape is worse by about 1 pp ({100 * distilled[nm][keep].mean():.2f}% against {100 * distilled[REF][keep].mean():.2f}%), the only resolved term.",
        f"Removing its adaptation does make the no-memory policy more reliable: scans below 5% rise from {round(a[nm]['success'] * n_scans)}/{n_scans} to {round(f[nm]['success'] * n_scans)}/{n_scans}, the worst scan falls from {a[nm]['worst']:.2f}% to {f[nm]['worst']:.2f}%, and the distilled protocol is better on {int(np.sum(distilled[nm] < adaptive[nm]))}/{n_scans} phantoms.",
        f"Its shorter distilled TIs (down to {1e3 * min(proto[nm]['TI_s']):.0f} ms against {1e3 * min(proto[REF]['TI_s']):.0f} ms) also make it more robust than the TI-histogram distilled protocol on phantoms {tail_names} (shortest-sphere APE {tail_apes(nm)} against {tail_apes(REF)}).",
        "Under the decision rule the answer is in between: the no-memory policy loses some accuracy through over-adaptation (about half a point, not significant) and, on typical phantoms, its schedule shape is also worse (about 1 pp, post hoc); this control cannot assign the gap to one cause.",
        f"The fitter-uncertainty policy does not distil: its fixed protocol reaches {f[sg]['mape']:.2f}% [{f[sg]['ci'][0]:.2f}, {f[sg]['ci'][1]:.2f}], {fmt_paired(sg_own)} against its own adaptive policy (worse on {sg_own['worse']}/{sg_own['n']} phantoms) and above the fixed log TI grid ({refs['log_grid_trmatched']['mape']:.2f}%).",
        "Its mean APE by sphere T1 rank, shortest to longest, adaptive / distilled: " + ", ".join(f"{x:.1f} / {y:.1f}%" for x, y in sg_rank) + ".",
        f"Either this policy relies on per-scan adaptation, or per-block medians do not summarise its schedules: {sg_floor} of its {len(proto[sg]['TR_s'])} median blocks sit at the {TR_FLOOR_S} s TR floor, and its last three blocks are medians over only {sg_last[0]}, {sg_last[1]} and {sg_last[2]} of the {n_scans} validation scans; this control cannot tell which.",
        f"For the abstract: the schedule-shape claim holds for the TI-histogram policy and is consistent with the no-memory policy (its distilled protocol, {f[nm]['mape']:.2f}%, also stays far below the fixed log TI grid and the Cramer-Rao schedule), but do not attribute the no-memory deficit to over-adaptation alone, and do not extend the claim to the fitter-uncertainty policy.",
        f"Caveats: {n_scans} phantoms and one training seed per run; the cut of phantoms {tail_names} was chosen in `control_summary.md` after seeing results; per-block medians are one crude distillation among many; the no-memory and fitter-uncertainty adaptive references are their recorded GPU evaluations, not local rollouts (for the TI-histogram run the local CPU rollout reproduced its GPU evaluation exactly).",
    ]

    lines = [
        "# Control: distilled protocols of the no-memory and fitter-uncertainty policies",
        "",
        f"Question: the no-memory policy scores {a['no_memory']['mape']:.2f}% on the held-out phantoms against {a['histogram']['mape']:.2f}% for the TI-histogram policy.",
        "Is its schedule shape worse, or does it lose accuracy by adapting each schedule too strongly?",
        ("Method: the validation-derived fixed median recipe of `control_summary.md`, applied to each policy: per block index, the median log TI and median TR of its `global_best` schedules on its 24 validation scans "
         f"(seed {d['validation_seed']}, `global_best/diagnostics/episodes.json` without the budget-rejected final proposals), truncated so that {d['Npe']} x sum(TR) <= {d['budget_s']:.0f} s, then replayed with no policy in the loop."),
        f"Phantoms: the {n_scans} held-out scans, seeds {seeds}.",
        f"Every replay uses the TI-histogram run's E2Env (`{d['env_run']}/run_config.json`), so all arms share phantoms, pose, per-seed noise stream, full-Bloch forward model (CPU) and generalised-IR T1 fitter; only the (TI, TR) schedule differs.",
        "Adaptive rows are each policy's own held-out evaluation (`global_best/eval_t1err_heldout.json`), paired per phantom.",
        "",
        "## Checks",
        "",
        "- **Env settings**: `base_env_kwargs` match the TI-histogram run in every forward-model setting (budget, Npe, noise, fidelity, spheres, T1 sampler, TE, flip angle, pose); the only differences are observation channels, which `julia/rl/e2.jl` reads only to build the observation:",
        *(f"  {line}" for line in env_lines),
        "  `null` means unset (E2Env default, off). The fitter computes its bootstrap sigma in every run with its own fixed seed, so `include_sigma` does not touch the noise stream.",
        "- **Training config** (outside `base_env_kwargs`, no effect on the replay): " + ("; ".join(training_diff(runs)) or "identical") + ".",
        f"- **Sanity check**: rerunning the TI-histogram run's validation-derived protocol reproduces `control_results.json` (`arms.fixed_median_validation`) exactly: protocol {'identical' if sanity['protocol_identical'] else 'different'}, every phantom's T1 estimates and MAPE identical (max |dT1_est| {sanity['max_abs_diff_T1_est_s']:.1e} s, max |dMAPE| {sanity['max_abs_diff_mape']:.1e}).",
        f"- **Pairing**: each run's `eval_t1err_heldout.json` was scored on the replayed phantoms (max |dT1_true| {max(var[v]['adaptive']['max_abs_diff_t1_true_s'] for v in VARIANTS):.1e} s for all three), and the bootstrap CIs of its per-scan MAPEs match the recorded `agent_mape_ci95_pct`.",
        "",
        f"## Results ({n_scans} held-out phantoms)",
        "",
        "| Policy | Arm | Mean MAPE % [95% CI] | Scans < 5% | Worst scan MAPE % | Blocks | Scan time (s) |",
        "|---|---|---|---|---|---|---|",
    ]
    for v in VARIANTS:
        lines.append(f"| {LABELS[v]} | Adaptive (policy in the loop) | {fmt_stats(a[v])} | - | - |")
        lines.append(f"| {LABELS[v]} | Distilled (fixed, validation-derived) | {fmt_stats(f[v])} | {len(proto[v]['TI_s'])} | "
                     f"{proto[v]['time_s']:.1f} |")
    for k, label in REF_ROWS.items():
        s = refs[k]
        lines.append(f"| Reference | {label} | {s['mape']:.2f} [{s['ci'][0]:.2f}, {s['ci'][1]:.2f}] | "
                     f"{100 * s['success']:.1f}% ({round(s['success'] * s['n_runs'])}/{s['n_runs']}) | - | "
                     f"{s['blocks']:.0f} | {s['time_s']:.1f} |")
    lines += [
        "",
        "MAPE: mean over the five spheres of |T1 estimate - T1 true| / T1 true, averaged over scans.",
        f"CI: percentile bootstrap, {N_BOOT} resamples over the {n_scans} phantoms, numpy `default_rng(0)` (as `python/baseline_e2.py:41-58`).",
        "`eval_e2.py` saved no per-scan block counts, so adaptive rows have none; each distilled protocol runs all its blocks on every phantom.",
        "Reference rows are the held-out baseline run of `control_summary.md`; per-scan values were not saved, so they have no worst scan.",
        "",
        "## Paired differences",
        "",
        f"| Comparison | Mean difference [95% CI] | Median per-phantom difference (pp) | Phantoms where the first is worse | Without phantoms {tail_names} (post hoc) |",
        "|---|---|---|---|---|",
        *(f"| {name(c[0])} - {name(c[1])} | {fmt_paired(paired[c])} | {paired[c]['median']:+.2f} | "
          f"{paired[c]['worse']}/{paired[c]['n']} | {fmt_paired(paired_rest[c])} |" for c in comparisons),
        "",
        f"Paired bootstrap over the phantoms ({N_BOOT} resamples, `default_rng(0)`).",
        f"The last column drops the {N_TAIL} phantoms with the shortest sphere T1, as in `control_summary.md`.",
        "",
        "## Short-T1 phantoms (post hoc)",
        "",
        "The distilled protocols' shortest TIs are " + ", ".join(f"{1e3 * min(proto[v]['TI_s']):.0f} ms ({LABELS[v]})" for v in VARIANTS) + ".",
        "",
        "| Phantom | Shortest T1 (ms) | " + " | ".join(f"Shortest-sphere APE %, {LABELS[v]}: adaptive / distilled" for v in VARIANTS)
        + " | " + " | ".join(f"Scan MAPE %, {LABELS[v]}: adaptive / distilled" for v in VARIANTS) + " |",
        "|---|---|" + "---|" * (2 * len(VARIANTS)),
        *(f"| {i} | {1e3 * min_t1[i]:.0f} | "
          + " | ".join(f"{shortest_ape(v, 'adaptive', i):.1f} / {shortest_ape(v, 'distilled', i):.1f}" for v in VARIANTS) + " | "
          + " | ".join(f"{100 * adaptive[v][i]:.2f} / {100 * distilled[v][i]:.2f}" for v in VARIANTS) + " |" for i in tail),
        "",
        "## Interpretation",
        "",
        *interpretation,
        "",
        "## Distilled protocols",
        "",
        "| Block | " + " | ".join(f"{LABELS[v]} TI (s) | TR (s) | Scans reaching block" for v in VARIANTS) + " |",
        "|---|" + "---|---|---|" * len(VARIANTS),
    ]

    def cells(p: dict, b: int) -> str:
        if b >= len(p["TI_s"]):
            return " | | "
        return f"{p['TI_s'][b]:.4f} | {p['TR_s'][b]:.4f} | {p['n_scans_at_block'][b]}"

    for b in range(max(len(p["TI_s"]) for p in proto.values())):
        lines.append(f"| {b + 1} | " + " | ".join(cells(proto[v], b) for v in VARIANTS) + " |")
    lines += [
        "| Total | " + " | ".join(f"{len(proto[v]['TI_s'])} blocks, {proto[v]['time_s']:.1f} s | | " for v in VARIANTS) + " |",
        "",
        "## Runtime",
        "",
        f"Local CPU, Julia {d['julia_version']}, {d['threads']} threads: {total_wall / 60:.1f} min for {total_blocks} full-Bloch blocks ({total_wall / total_blocks:.2f} s per block including the fit).",
        "",
        "## Regenerate",
        "",
        "From the RLxMRI repo root (needs `figures/control_data/control_results.json` from `control_schedule_swap.jl` for the sanity check):",
        "",
        "```bash",
        "julia -t 8 --project=. ismrm_latex/vancouver/scripts/control_distill_variants.jl",
        ".venv/bin/python ismrm_latex/vancouver/scripts/control_distill_summary.py",
        "```",
        "",
        "Outputs: `figures/control_data/control_distill_variants.json` (protocols, per-phantom results and adaptive references of every policy) and this file.",
        "",
    ]
    OUT.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()

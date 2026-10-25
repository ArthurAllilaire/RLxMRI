"""Summarise the narrow-band control (control_narrow_band.jl) into figures/control_narrow_band.md.

Statistics are control_summary.py's: percentile bootstrap CIs, 5000 resamples over the
phantoms, numpy default_rng(0); paired differences resample the phantoms the same way.
Wide-range references are the same arms on the 24 default held-out phantoms.

    .venv/bin/python ismrm_latex/vancouver/scripts/control_narrow_band_summary.py
"""

from __future__ import annotations

import json

import numpy as np

import figstyle
from control_summary import (
    N_BOOT,
    arm_stats,
    fmt_paired,
    fmt_row,
    paired_stats,
    references,
)

DATA = figstyle.FIGURES / "control_data" / "control_narrow_band.json"
WIDE = figstyle.FIGURES / "control_data" / "control_results.json"
OUT = figstyle.FIGURES / "control_narrow_band.md"
BIN_EDGES_S = (0.1, 0.5)
FIXED_ARMS = {"distilled": ("Distilled fixed protocol", "medians of validation schedules"),
              "log_grid_trmatched": ("Fixed log TI grid, TR 1.7 s", "fixed"),
              "cr_optimal": ("Cramer-Rao optimal", "fixed")}


def boot_r(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Pearson r with a percentile bootstrap CI over scans (5000 resamples, default_rng(0))."""
    rng = np.random.default_rng(0)
    idx = rng.integers(0, x.size, size=(N_BOOT, x.size))
    boot = [np.corrcoef(x[i], y[i])[0, 1] for i in idx]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(np.corrcoef(x, y)[0, 1]), float(lo), float(hi)


def fmt_r(r: tuple[float, float, float]) -> str:
    return f"{r[0]:+.2f} [{r[1]:+.2f}, {r[2]:+.2f}]"


def fmt_ci(s: dict) -> str:
    return f"{s['mape']:.2f} [{s['ci'][0]:.2f}, {s['ci'][1]:.2f}]"


def main() -> None:
    d = json.loads(DATA.read_text())
    wide = json.loads(WIDE.read_text())
    arms, shifts, n_scans = d["arms"], d["shifts"], d["n_scans"]
    mape = {k: np.array([r["mape"] for r in v]) for k, v in arms.items()}
    blocks = {k: [r["n_blocks"] for r in v] for k, v in arms.items()}
    times = {k: [r["time_s"] for r in v] for k, v in arms.items()}

    swap_keys = [f"swap_{s}" for s in shifts]
    swap_runs = np.concatenate([mape[k] for k in swap_keys])
    mape["swap_mean"] = np.mean([mape[k] for k in swap_keys], axis=0)
    swap_label = f"Swapped, mean of {len(shifts)} shifts"

    def arm_table(sel: np.ndarray) -> dict:
        """Per-arm stats over the phantoms in sel; the swap mean pools its shifts' runs."""
        out = {k: arm_stats(mape[k][sel], mape[k][sel], [blocks[k][i] for i in sel], [times[k][i] for i in sel])
               for k in ["adaptive", *swap_keys, *FIXED_ARMS]}
        out["swap_mean"] = arm_stats(np.concatenate([mape[k][sel] for k in swap_keys]), mape["swap_mean"][sel],
                                     [blocks[k][i] for k in swap_keys for i in sel],
                                     [times[k][i] for k in swap_keys for i in sel])
        return out

    stats = arm_table(np.arange(n_scans))
    compared = [*swap_keys, "swap_mean", *FIXED_ARMS]
    labels = {**{k: f"Swapped, shift {s}" for s, k in zip(shifts, swap_keys)}, "swap_mean": swap_label,
              **{k: label for k, (label, _) in FIXED_ARMS.items()}}
    paired = {k: paired_stats(mape[k], mape["adaptive"]) for k in compared}

    wide_mape = {k: np.array([r["mape"] for r in v]) for k, v in wide["arms"].items()}
    wide_swap = np.mean([wide_mape[f"swap_{s}"] for s in wide["shifts"]], axis=0)
    wide_refs = references(wide)
    wide_ref = {"adaptive": 100 * wide_mape["adaptive"].mean(), "swap_mean": 100 * wide_swap.mean(),
                "distilled": 100 * wide_mape["fixed_median_validation"].mean(),
                **{k: wide_refs[k]["mape"] for k in ("log_grid_trmatched", "cr_optimal")}}
    wide_paired = {"swap_mean": paired_stats(wide_swap, wide_mape["adaptive"]),
                   "distilled": paired_stats(wide_mape["fixed_median_validation"], wide_mape["adaptive"])}

    centre = np.array(d["T1_band_centre"])
    bins = np.digitize(centre, BIN_EDGES_S)
    sched_ti = [np.array(s["TI_s"]) for s in d["schedules"]]
    min_ti = np.array([s.min() for s in sched_ti])
    med_ti = np.array([np.median(s) for s in sched_ti])
    min_t1 = np.array([min(r["T1_true"]) for r in arms["adaptive"]])
    log_c = np.log(centre)
    r_med, r_min, r_t1 = boot_r(log_c, np.log(med_ti)), boot_r(log_c, np.log(min_ti)), boot_r(np.log(min_t1), np.log(min_ti))
    slope_med, slope_min = (float(np.polyfit(log_c, np.log(ti), 1)[0]) for ti in (med_ti, min_ti))
    wide_min_t1 = np.array([min(r["T1_true"]) for r in wide["arms"]["adaptive"]])
    wide_min_ti = np.array([min(s["TI_s"]) for s in wide["schedules"]])
    r_wide = boot_r(np.log(wide_min_t1), np.log(wide_min_ti))
    proto = d["protocols"]["distilled"]

    bin_arms = ["adaptive", "swap_mean", *FIXED_ARMS]
    bin_names = {"adaptive": "Adaptive", **labels}
    by_bin = []
    e0, e1 = BIN_EDGES_S
    for b, label in enumerate((f"c < {e0} s", f"{e0} <= c < {e1} s", f"c >= {e1} s")):
        sel = np.flatnonzero(bins == b)
        assert sel.size > 0, f"no phantom has {label}"
        by_bin.append({"label": label, "sel": sel, "stats": arm_table(sel),
                       "paired": {k: paired_stats(mape[k][sel], mape["adaptive"][sel]) for k in bin_arms[1:]}})

    a, ds = stats["adaptive"], stats["distilled"]
    lg, cr = stats["log_grid_trmatched"], stats["cr_optimal"]
    repro = d["regression_check"]["adaptive"]
    wall, sim_blocks = d["wall_s"], d["simulated_blocks"]
    check_wall = d["regression_check"]["wall_s"]
    total_wall, total_blocks = sum(wall.values()), sum(sim_blocks.values())
    seeds = f"{d['seed_base']}-{d['seed_base'] + n_scans - 1}"
    lo_c, hi_c = d["centre_range_s"]
    w = d["t1_band_rel"]

    def bin_row(bb: dict, k: str) -> str:
        s = bb["stats"][k]
        diff = "-" if k == "adaptive" else fmt_paired(bb["paired"][k])
        return (f"| {bb['label']} (n = {bb['sel'].size}) | {bin_names[k]} | {fmt_ci(s)} | "
                f"{100 * s['success']:.1f}% ({round(s['success'] * s['n_runs'])}/{s['n_runs']}) | {s['worst']:.2f} | {diff} |")

    def policy_row(bb: dict) -> str:
        sel = bb["sel"]
        return (f"| {bb['label']} | {sel.size} | {1e3 * np.median(centre[sel]):.0f} | {np.mean([blocks['adaptive'][i] for i in sel]):.1f} | "
                f"{1e3 * np.median(min_ti[sel]):.0f} | {1e3 * np.median(med_ti[sel]):.0f} |")

    rows = [("Adaptive (policy in the loop)", "own, chosen online", stats["adaptive"], wide_ref["adaptive"])]
    rows += [(f"Swapped, shift {s}", f"scan (k + {s}) mod {n_scans}", stats[k], None) for s, k in zip(shifts, swap_keys)]
    rows.append((swap_label, "another scan's", stats["swap_mean"], wide_ref["swap_mean"]))
    rows += [(label, source, stats[k], wide_ref[k]) for k, (label, source) in FIXED_ARMS.items()]

    short, mid, long_ = by_bin
    sp, mp, lp = (bb["paired"] for bb in by_bin)
    n_bin = [bb["sel"].size for bb in by_bin]
    bin_blocks = [np.mean([blocks["adaptive"][i] for i in bb["sel"]]) for bb in by_bin]
    rest = np.flatnonzero(centre >= e0)
    rest_ds = paired_stats(mape["distilled"][rest], mape["adaptive"][rest])
    rest_sw = paired_stats(mape["swap_mean"][rest], mape["adaptive"][rest])
    ds_fail = mape["distilled"] > 0.10
    swap_fail = [(k, r["schedule_scan"]) for key in swap_keys for k, r in enumerate(arms[key]) if r["mape"] > 0.20]
    assert all(centre[k] < e0 for k, _ in swap_fail)
    swap_capped = int(np.sum(swap_runs >= 1.0))
    mid_short = mid["sel"][min_ti[mid["sel"]] < 0.1]
    mid_ad, mid_ds = mid["stats"]["adaptive"], mid["stats"]["distilled"]
    # The wording below is for this outcome of the decision rule.
    assert paired["swap_mean"]["lo"] > 0 and paired["distilled"]["lo"] > 0
    interpretation = [
        "Decision rule (set before the run): adaptation pays off if the adaptive policy clearly beats both the swapped schedules and the distilled protocol (paired 95% CIs exclude zero); if it is no better than either, the policy does not exploit narrow bands and the idea needs a policy trained on them.",
        f"Result: it beats both, so by the rule the policy's per-scan adaptation pays off when scans are atypical: another scan's schedule costs {fmt_paired(paired['swap_mean'])} and the distilled protocol {fmt_paired(paired['distilled'])}, against {fmt_paired(wide_paired['swap_mean'])} and {fmt_paired(wide_paired['distilled'])} on the wide-range held-out phantoms.",
        f"Out of its training distribution and without retraining, the policy stays as accurate as on wide-range scans ({a['mape']:.2f}% [{a['ci'][0]:.2f}, {a['ci'][1]:.2f}] against {wide_ref['adaptive']:.2f}%), with {round(a['success'] * n_scans)}/{n_scans} scans below 5% and a worst scan of {a['worst']:.2f}%; the fixed log TI grid ({lg['mape']:.2f}%) and the Cramer-Rao schedule ({cr['mape']:.2f}%) trail it by {fmt_paired(paired['log_grid_trmatched'])} and {fmt_paired(paired['cr_optimal'])}.",
        f"The result is mixed across the centre range, and the pooled gain over the distilled protocol ({ds['mape']:.2f}%) is a tail effect: that protocol is better than the policy on the median phantom ({paired['distilled']['median']:+.2f} pp, worse on only {paired['distilled']['worse']}/{n_scans}) but fails (scan MAPE above 10%) on {int(ds_fail.sum())} phantoms, all with c below {1e3 * centre[ds_fail].max():.0f} ms.",
        f"Short bands, {short['label']} ({n_bin[0]} phantoms): every alternative fails; swapped {fmt_paired(sp['swap_mean'])}, distilled {fmt_paired(sp['distilled'])}, log TI grid {fmt_paired(sp['log_grid_trmatched'])}, Cramer-Rao {fmt_paired(sp['cr_optimal'])}.",
        f"All {len(swap_fail)} catastrophic swapped runs (scan MAPE above 20%, {swap_capped} capped at 100%) are such phantoms given a schedule whose shortest TI is {1e3 * min(min_ti[j] for _, j in swap_fail):.0f} ms or more, while the policy chose about {bin_blocks[0]:.0f} blocks with TIs down to about {1e3 * np.median(min_ti[short['sel']]):.0f} ms.",
        f"Long bands, {long_['label']} ({n_bin[2]} phantoms): the policy switches to fewer, longer blocks ({bin_blocks[2]:.1f} blocks, median TI {1e3 * np.median(med_ti[long_['sel']]):.0f} ms); swapped schedules still cost {fmt_paired(lp['swap_mean'])}, but the distilled protocol matches the policy ({fmt_paired(lp['distilled'])}).",
        f"Mid bands, {mid['label']} ({n_bin[1]} phantoms): adaptation buys nothing; swapped schedules match the policy ({fmt_paired(mp['swap_mean'])}) and the distilled protocol beats it ({fmt_paired(mp['distilled'])}; {mid_ds['mape']:.2f}% against {mid_ad['mape']:.2f}%).",
        f"On {mid_short.size}/{n_bin[1]} of these phantoms, up to c = {1e3 * centre[mid_short].max():.0f} ms, the policy keeps a short-T1 schedule (shortest TI below 100 ms), where the distilled protocol's longer TIs (shortest {1e3 * min(proto['TI_s']):.0f} ms) do better.",
        f"Without the {n_bin[0]} phantoms with c < {e0} s (a post hoc cut), the distilled protocol is better than the policy (paired {fmt_paired(rest_ds)}), while another scan's schedule still costs {fmt_paired(rest_sw)}: outside short bands the policy's tailoring is worth more than another scan's schedule but less than a well-placed fixed protocol.",
        f"The policy's TIs do track the band, more tightly than on wide-range scans: r = {fmt_r(r_min)} between log c and the log of its shortest TI, and {fmt_r(r_t1)} between log shortest T1 and log shortest TI, against {fmt_r(r_wide)} on the wide-range held-out scans.",
        f"The slopes ({slope_min:.2f} for the shortest TI, {slope_med:.2f} for the median TI) show its TIs moving less than half as far as the band in log units (mean blocks {bin_blocks[0]:.1f} / {bin_blocks[1]:.1f} / {bin_blocks[2]:.1f} from short to long bands).",
        f"For the abstract: on narrow-band scans the TI-histogram policy, without retraining, tailors its schedule to the scan; another scan's schedule costs {paired['swap_mean']['mean']:+.1f} pp and the distilled fixed protocol {paired['distilled']['mean']:+.1f} pp, almost all of it in short-T1 bands, while a fixed protocol is better for mid-range bands; claim adaptation as robustness to atypical scans, not as a uniform accuracy gain.",
        f"Caveats: one training seed; out-of-distribution evaluation of a policy trained on wide-range scans; one band width (w = {w}); the centre bins hold {n_bin[0]}, {n_bin[1]} and {n_bin[2]} phantoms and the c >= {e0} s cut is post hoc; E2Env caps a scan's MAPE at 100%, which can only understate the failing arms' means; the distilled protocol and the Cramer-Rao schedule were designed for wide-range scans, and a fixed protocol tuned to narrow bands at unknown centres (for example distilled from the policy's narrow-band schedules on separate seeds) was not tested and is the stronger fixed comparator for this question.",
    ]

    lines = [
        "# Control: per-scan adaptation on narrow-band phantoms",
        "",
        "Question: when each scan's five sphere T1s sit in a narrow band at an unknown position, does the TI-histogram policy's per-scan adaptation pay off against fixed schedules?",
        "Training and the held-out evaluation draw each sphere T1 independently and uniformly over the T1-plate range, so every scan spans most of it and needs similar TIs (`control_summary.md`: adaptation helped only on two phantoms with a very short T1 sphere).",
        f"Policy: `{d['env_run']}`, checkpoint `{d['checkpoint']}`, evaluated out of its training distribution; no retraining.",
        f"Phantoms: {n_scans} scans, seeds {seeds}, `t1_sampler = :narrow_band` (`julia/rl/e2.jl`): a centre c drawn log-uniformly over the T1-plate range {lo_c:.3f}-{hi_c:.3f} s, then each sphere's T1 uniformly over c(1 +- {w:.2f}), with T2 and T2* keeping each sphere's nominal T2/T1 ratio.",
        "Every other env setting is the histogram run's `base_env_kwargs`.",
        "The sampler draws the centre from its own stream, so phantom k keeps held-out scan k's pose and per-seed noise stream and differs from it only in its T1s.",
        "Every arm runs on the same phantoms with the same noise stream, the same E2Env forward model (full Bloch, CPU) and the same generalised-IR T1 fitter; only the (TI, TR) schedule differs.",
        "",
        "## Step 1: regression check on the default sampler",
        "",
        f"After the sampler change, the pure-Julia rollout of `global_best` on the default held-out seeds still reproduces `eval_t1err_heldout.json` exactly: largest per-sphere APE difference {repro['max_abs_diff_ape']:.1e}, largest true-T1 difference {repro['max_abs_diff_t1_true_s']:.1e} s, mean MAPE {repro['mape_pct_rollout']:.4f}% (recorded {repro['mape_pct_recorded']:.4f}%).",
        "Replaying the distilled protocol and the two fixed comparators on those phantoms reproduces their recorded per-scan T1 estimates, MAPEs and block counts bit for bit (`control_results.json` `arms.fixed_median_validation`; `fig5_v2_data/fixed_comparators_heldout.json`).",
        "",
        "## Arms",
        "",
        "- **Adaptive**: the policy in the loop, rolled out as in `control_schedule_swap.jl`.",
        f"- **Swapped**: phantom k gets the executed schedule of scan (k + s) mod {n_scans}, for s = {', '.join(map(str, shifts))} ({len(shifts) * n_scans} runs, no policy in the loop); the direct test of per-scan tailoring.",
        f"- **Distilled**: the fixed protocol `median_protocols.fixed_median_validation` of `control_results.json` ({len(proto['TI_s'])} blocks, {proto['time_s']:.1f} s, shortest TI {1e3 * min(proto['TI_s']):.0f} ms), as recorded, not re-derived.",
        "- **Fixed log TI grid** and **Cramer-Rao optimal**: the 560 s schedules of the abstract's comparisons, replayed as `fixed_comparators_heldout.jl` replays them (`fixed_schedules.jl`).",
        "  The Cramer-Rao schedule is not re-optimised: the band's position is unknown, so the full-range design is the fair fixed comparator.",
        "",
        f"## Results ({n_scans} narrow-band phantoms)",
        "",
        "| Arm | Schedule | Mean MAPE % [95% CI] | Scans < 5% | Worst scan MAPE % | Mean blocks | Mean scan time (s) | Wide-range held-out MAPE % |",
        "|---|---|---|---|---|---|---|---|",
        *(fmt_row(label, source, s) + (f" {ref:.2f} |" if ref is not None else " - |") for label, source, s, ref in rows),
        "",
        "MAPE: mean over the five spheres of |T1 estimate - T1 true| / T1 true, averaged over scans.",
        f"CI: percentile bootstrap, {N_BOOT} resamples over the {n_scans} phantoms, numpy `default_rng(0)` (as `python/baseline_e2.py:41-58`).",
        f"For the swap mean, each phantom's {len(shifts)} swapped MAPEs are averaged before resampling; its success share and worst scan are over all {swap_runs.size} runs.",
        "Wide-range held-out: the same arm on the 24 default held-out phantoms (`control_summary.md`; shifts 1, 8, 16 there).",
        "",
        "## Paired differences to Adaptive",
        "",
        "| Arm | Mean difference, arm - Adaptive [95% CI] | Median per-phantom difference (pp) | Phantoms where the arm is worse |",
        "|---|---|---|---|",
        *(f"| {labels[k]} | {fmt_paired(paired[k])} | {paired[k]['median']:+.2f} | {paired[k]['worse']}/{paired[k]['n']} |"
          for k in compared),
        "",
        f"Paired bootstrap over the phantoms ({N_BOOT} resamples, `default_rng(0)`); for the swap mean, the {len(shifts)} shifts are averaged per phantom first.",
        f"On the wide-range held-out phantoms the same comparisons were {fmt_paired(wide_paired['swap_mean'])} (swapped mean) and {fmt_paired(wide_paired['distilled'])} (distilled).",
        "",
        "## By band centre",
        "",
        f"Bins of the band centre c with edges {BIN_EDGES_S[0]} and {BIN_EDGES_S[1]} s (fixed before the run); CIs and paired differences resample the phantoms within each bin.",
        "",
        "| Centre bin | Arm | Mean MAPE % [95% CI] | Scans < 5% | Worst scan MAPE % | Paired difference, arm - Adaptive [95% CI] |",
        "|---|---|---|---|---|---|",
        *(bin_row(bb, k) for bb in by_bin for k in bin_arms),
        "",
        "What the policy did in each bin (medians over scans of each scan's shortest and median executed TI):",
        "",
        "| Centre bin | Phantoms | Median centre (ms) | Mean blocks | Shortest TI (ms) | Median TI (ms) |",
        "|---|---|---|---|---|---|",
        *(policy_row(bb) for bb in by_bin),
        f"| Distilled protocol | - | - | {len(proto['TI_s'])} | {1e3 * min(proto['TI_s']):.0f} | {1e3 * np.median(proto['TI_s']):.0f} |",
        "",
        "## Does the policy track the band?",
        "",
        f"Pearson r on log scales over the {n_scans} narrow-band scans, with percentile bootstrap CIs ({N_BOOT} resamples, `default_rng(0)`); the slope is the least-squares slope of log TI on log c (1 = TIs scale with the band).",
        "",
        "| Pair | Narrow band: r [95% CI] | Slope | Wide-range held-out (24 scans): r [95% CI] |",
        "|---|---|---|---|",
        f"| log c vs log median TI | {fmt_r(r_med)} | {slope_med:.2f} | - |",
        f"| log c vs log shortest TI | {fmt_r(r_min)} | {slope_min:.2f} | - |",
        f"| log shortest T1 vs log shortest TI | {fmt_r(r_t1)} | - | {fmt_r(r_wide)} |",
        "",
        "## Interpretation",
        "",
        *interpretation,
        "",
        "## Runtime",
        "",
        f"Local CPU, Julia {d['julia_version']}, {d['threads']} threads: {total_wall / 60:.1f} min for the {total_blocks} full-Bloch blocks of the narrow-band arms ({total_wall / total_blocks:.2f} s per block including the fit), plus {sum(check_wall.values()) / 60:.1f} min for the regression check.",
        "",
        "| Arm | Simulated blocks | Wall time (s) |",
        "|---|---|---|",
        *(f"| {k} | {sim_blocks[k]} | {wall[k]:.0f} |" for k in ["adaptive", *swap_keys, *FIXED_ARMS]),
        "",
        "## Regenerate",
        "",
        "From the RLxMRI repo root (needs `figures/control_data/control_results.json` from `control_schedule_swap.jl` and `figures/fig5_v2_data/fixed_comparators_heldout.json` from `fixed_comparators_heldout.jl`):",
        "",
        "```bash",
        f"julia -t 8 --project=. ismrm_latex/vancouver/scripts/control_narrow_band.jl {n_scans}",
        ".venv/bin/python ismrm_latex/vancouver/scripts/control_narrow_band_summary.py",
        "```",
        "",
        "Outputs: `figures/control_data/control_narrow_band.json` (band centres, executed adaptive schedules and per-phantom results of every arm) and this file.",
        "",
    ]
    OUT.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()

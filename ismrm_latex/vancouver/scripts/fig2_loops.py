"""Build Figure 2 (inner and outer training loops) for the ISMRM 2027 abstract.

The TikZ source is figures/fig2_loops.tex. Every number and colour it shows is
read here from the run config, the training and environment code, and
figstyle.py, written to figures/fig2_values.tex, and then the figure is compiled
with lualatex and rasterised at 300 dpi.

    python ismrm_latex/vancouver/scripts/fig2_loops.py [--build-dir DIR]
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import figstyle

ROOT = figstyle.ROOT
FIGURES = figstyle.FIGURES
RUN = ROOT / "runs/e2/mf_runB_5sphere_hist_560s_gpu"
TEX = FIGURES / "fig2_loops.tex"
VALUES = FIGURES / "fig2_values.tex"
RUNG_KEYS = "ABCDE"


def class_constant(tree: ast.Module, cls: str, name: str) -> ast.expr:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for stmt in node.body:
                if isinstance(stmt, ast.Assign) and any(
                        isinstance(t, ast.Name) and t.id == name for t in stmt.targets):
                    return stmt.value
    raise LookupError(f"{cls}.{name} not found")


def module_constant(tree: ast.Module, name: str) -> ast.expr:
    for stmt in tree.body:
        targets = stmt.targets if isinstance(stmt, ast.Assign) else (
            [stmt.target] if isinstance(stmt, ast.AnnAssign) else [])
        if any(isinstance(t, ast.Name) and t.id == name for t in targets):
            return stmt.value
    raise LookupError(f"{name} not found")


def np_array_literal(node: ast.expr) -> list[float]:
    """Value of `np.array([...], dtype=...)`."""
    assert isinstance(node, ast.Call), ast.dump(node)
    return list(ast.literal_eval(node.args[0]))


def fixed_flip_deg(tree: ast.Module) -> float:
    """Flip angle the env applies when TE is fixed and alpha is not learned:
    the `else` branch of `if self._learn_alpha:` in `_denorm_action`."""
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef) and fn.name == "_denorm_action":
            for node in ast.walk(fn):
                if (isinstance(node, ast.If) and isinstance(node.test, ast.Attribute)
                        and node.test.attr == "_learn_alpha"):
                    (assign,) = node.orelse
                    return float(ast.literal_eval(assign.value))
    raise LookupError("fixed flip angle not found in _denorm_action")


def net_arch(path: Path) -> list[int]:
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.keyword) and node.arg == "net_arch":
            return list(ast.literal_eval(node.value))
    raise LookupError("net_arch not found")


def num(x: float) -> str:
    return f"{x:g}"


def thousands(n: int) -> str:
    return f"{n:,}"


def luminance(hex_colour: str) -> float:
    rgb = [int(hex_colour.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def contrast(a: str, b: str) -> float:
    la, lb = sorted((luminance(a), luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def rank_threshold_text(rungs: list[str], thresholds: dict[str, float]) -> str:
    """'0.7 (analytic), 0.8 (cached), 0.9 (full 3 mm)'. Consecutive rungs that
    share a threshold are named by their common first word, and spaces become
    TeX ties so a threshold never wraps away from its rung name."""
    groups: list[tuple[float, list[str]]] = []
    for rung in rungs:
        thr = thresholds[rung]
        if groups and groups[-1][0] == thr:
            groups[-1][1].append(figstyle.RUNG_LABELS[rung])
        else:
            groups.append((thr, [figstyle.RUNG_LABELS[rung]]))
    parts = []
    for thr, labels in groups:
        name = labels[0] if len(labels) == 1 else labels[0].split(" ")[0]
        parts.append(f"{num(thr)} ({name})".replace(" ", "~"))
    return ", ".join(parts)


def collect_values() -> dict[str, str]:
    cfg = json.loads((RUN / "run_config.json").read_text())
    env, mf = cfg["base_env_kwargs"], cfg["mf"]

    # The figure's wording describes this configuration; fail loudly if it changes.
    assert cfg["plan"] == figstyle.RUNGS, cfg["plan"]
    assert env["fix_te"] and not env["learn_alpha"] and env["log_ti_action"]
    assert env["include_ti_history"] and not env["include_sigma"] and not env["include_image"]
    assert env["reward_mode"] == "delta_log_mape" and env["mape_alpha"] == 1.0
    assert env["pose_mode"] == "inplane_jitter" and not env["allow_stop"]
    assert mf["schedule"] == "criterion" and mf["n_envs"] == 1
    assert mf["mf_use_lookahead"] and mf["mf_reset_reward_norm"]
    assert not mf["mf_reset_optimizer_on_switch"]

    env_tree = ast.parse((ROOT / "python/qalibremd_gym/env_e2.py").read_text())
    act_lo = np_array_literal(class_constant(env_tree, "QalibreMDE2Env", "_ACT_LO"))
    act_hi = np_array_literal(class_constant(env_tree, "QalibreMDE2Env", "_ACT_HI"))
    te_s = ast.literal_eval(class_constant(env_tree, "QalibreMDE2Env", "_FIXED_TE_S"))
    rank_thr = ast.literal_eval(module_constant(
        ast.parse((ROOT / "python/train_e2_mf.py").read_text()), "RANKCORR_THRESH"))
    headroom = re.search(r"e2_tr_headroom\(::E2Env\)\s*=\s*([0-9.]+)",
                         (ROOT / "julia/rl/e2.jl").read_text()).group(1)
    arch = net_arch(ROOT / "python/e2_train_common.py")
    assert len(set(arch)) == 1, arch

    cheap = cfg["plan"][:-1]
    min_steps = [int(s) for s in mf["mf_min_steps"].split(",")][:len(cheap)]
    max_steps = [int(s) for s in mf["mf_max_steps"].split(",")][:len(cheap)]
    decision_steps = mf["mf_decision_rollouts"] * mf["n_steps"] * mf["n_envs"]

    return {
        "ValTIlo": num(act_lo[0]), "ValTIhi": num(act_hi[0]),
        "ValTRlo": num(act_lo[2]), "ValTRhi": num(act_hi[2]),
        "ValHeadroom": num(float(headroom)),
        "ValTEms": num(te_s * 1e3),
        "ValFlip": num(fixed_flip_deg(env_tree)),
        "ValNpe": str(env["Npe"]), "ValNfe": str(env["Nfe"]),
        "ValBudget": num(env["time_budget_s"]),
        "ValROI": str(2 * env["roi_radius"] + 1),
        "ValSpheres": str(env["subset_size"]),
        "ValHistBins": str(env["ti_hist_bins"]),
        "ValLayers": str(len(arch)), "ValHidden": str(arch[0]),
        "ValRollout": thousands(mf["n_steps"] * mf["n_envs"]),
        "ValDecisionRollouts": str(mf["mf_decision_rollouts"]),
        "ValDecisionSteps": thousands(decision_steps),
        "ValProbeEps": str(mf["mf_probe_episodes_full"]),
        "ValPlateauPct": num(100 * mf["mf_plateau_delta"]),
        "ValPlateauWindow": str(mf["mf_plateau_window"]),
        "ValLookRollouts": str(mf["mf_lookahead_rollouts"]),
        "ValLookMargin": num(mf["mf_lookahead_margin"]),
        "ValRankThresholds": rank_threshold_text(cheap, rank_thr),
        "ValRankMinProbes": str(mf["mf_rankcorr_min_ckpts"]),
        "ValBiasMeanPP": num(100 * mf["mf_bias_mape_abs"]),
        "ValBiasPNinetyPP": num(100 * mf["mf_bias_p90_abs"]),
        "ValReservePct": num(100 * mf["mf_full_reserve_frac"]),
        "ValWallHours": num(mf["mf_budget_hours"]),
        "ValMinSteps": f"{thousands(min(min_steps))}-{thousands(max(min_steps))}",
        "ValMaxSteps": f"{thousands(min(max_steps))}-{thousands(max(max_steps))}",
        "ValBestEps": str(mf["mf_global_best_episodes"]),
        "ValFinalRung": figstyle.RUNG_LABELS[cfg["plan"][-1]],
    }


def colour_definitions() -> list[str]:
    lines = [
        rf"\definecolor{{accent}}{{HTML}}{{{figstyle.ACCENT.lstrip('#')}}}",
        rf"\definecolor{{ink}}{{HTML}}{{{figstyle.INK.lstrip('#')}}}",
        rf"\definecolor{{inksec}}{{HTML}}{{{figstyle.INK_SECONDARY.lstrip('#')}}}",
        rf"\definecolor{{gridgrey}}{{HTML}}{{{figstyle.GRID.lstrip('#')}}}",
    ]
    assert len(RUNG_KEYS) == len(figstyle.RUNGS)
    for key, rung in zip(RUNG_KEYS, figstyle.RUNGS):
        fill = figstyle.RUNG_COLOURS[rung]
        text = figstyle.INK if contrast(fill, figstyle.INK) >= contrast(fill, "#FFFFFF") else "#FFFFFF"
        head, _, tail = figstyle.RUNG_LABELS[rung].partition(" ")
        label = rf"{head}\\{tail}" if tail else head
        lines += [
            rf"\definecolor{{rung{key}}}{{HTML}}{{{fill.lstrip('#')}}}",
            rf"\definecolor{{rungtext{key}}}{{HTML}}{{{text.lstrip('#')}}}",
            rf"\newcommand{{\RungLabel{key}}}{{{label}}}",
        ]
    return lines


def write_values() -> None:
    lines = ["% Generated by scripts/fig2_loops.py; do not edit by hand."]
    lines += [rf"\newcommand{{\{k}}}{{{v}}}" for k, v in collect_values().items()]
    lines += colour_definitions()
    VALUES.write_text("\n".join(lines) + "\n")


def build(build_dir: Path) -> None:
    build_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["lualatex", "-interaction=nonstopmode", "-halt-on-error",
         f"-output-directory={build_dir}", TEX.name],
        cwd=FIGURES, check=True, stdout=subprocess.DEVNULL)
    pdf = FIGURES / "fig2_loops.pdf"
    shutil.copyfile(build_dir / "fig2_loops.pdf", pdf)
    subprocess.run(["pdftoppm", "-r", "300", "-png", "-singlefile", str(pdf),
                    str(FIGURES / "fig2_loops")], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=None,
                        help="where lualatex writes its artefacts (default: a temp dir)")
    args = parser.parse_args()
    write_values()
    if args.build_dir is None:
        with tempfile.TemporaryDirectory() as tmp:
            build(Path(tmp))
    else:
        build(args.build_dir)
    print(f"wrote {FIGURES / 'fig2_loops.pdf'} and {FIGURES / 'fig2_loops.png'}")


if __name__ == "__main__":
    main()

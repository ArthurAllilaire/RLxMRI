"""Sweep E2 sim-cost levers and chart the results.

One axis at a time around a baseline config (defined by the usual E2 flags, same
as bench_e2.py). Reuses ONE warm Julia runtime for the *cheap* axes (npe, voxel,
water) so Julia's per-process JIT (~tens of seconds on the first simulate) is paid
once per axis. The nthreads axis is boot-fixed (PYTHON_JULIACALL_THREADS must be
set before juliacall starts), so each value gets its own worker process; n_envs
uses SubprocVecEnv and is capped at {1,2} (higher OOMs on a 7.6 GiB box — bump
.wslconfig memory to go further).

Timing uses a controlled best-of-N simulate() (the thread_probe.py method), not
bench_e2's single-shot mean_sim, which is too load-noisy for a clean bar chart.

Usage:
    PYTHON_JULIAPKG_OFFLINE=yes python python/scripts/sweep_e2.py \
        --fix-te --learn-alpha --field T15 --noise 50 --axis npe --values 16,32,64
    ... --axis voxel  --values 1.0,1.5,2.0,3.0
    ... --axis water                 # values default to on,off
    ... --axis nthreads --values 1,2,4
    ... --axis n-envs   --values 1,2
    ... --axis all                   # default sweep over every axis
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from e2_config import add_e2_env_args, e2_env_kwargs
from _bench_io import git_meta, append_result, RUNS_DIR

_MARKER = "@@SWEEP_JSON@@"
_CHEAP = ("npe", "voxel", "water", "water_voxel")
_DEFAULT_VALUES = {
    "npe": "16,32,64",
    "voxel": "1.0,1.5,2.0,3.0",
    "water": "on,off",
    "water_voxel": "1.0,2.0,3.0",
    "nthreads": "1,2,4",
    "n-envs": "1,2",
}


def _parse_values(axis: str, values: str | None):
    raw = values or _DEFAULT_VALUES[axis]
    items = [v.strip() for v in raw.split(",") if v.strip()]
    if axis == "npe":
        return [int(v) for v in items]
    if axis in ("voxel", "water_voxel"):
        return [float(v) for v in items]
    if axis in ("nthreads", "n-envs"):
        return [int(v) for v in items]
    if axis == "water":
        return [v.lower() in ("on", "true", "1", "yes") for v in items]
    raise ValueError(axis)


def _override(axis: str, value, base_kwargs: dict) -> dict:
    """Return env_kwargs with the swept axis applied to the baseline."""
    kw = dict(base_kwargs)
    if axis == "npe":
        kw["Npe"] = int(value)
    elif axis == "voxel":
        kw["voxel_size_mm"] = float(value)
    elif axis == "water":
        kw["include_water"] = bool(value)
    elif axis == "water_voxel":
        kw["water_voxel_size_mm"] = float(value)
        kw["include_water"] = True
    return kw


# ── worker: runs in ONE warm Julia process, times a list of configs ──────────

_PROBE_SETUP = '''
using KomaMRI
function _sweep_probe(eobj, repeats::Int)
    seq = MRISystemPhantom.ir_se_2d_sequence(0.8, 0.02, 3.0;
            α_exc = deg2rad(90.0), FOV = eobj.FOV, Nfe = eobj.Nfe, Npe = eobj.Npe)
    scn = MRISystemPhantom.scanner_for_field(eobj.cfg_field)
    sp  = Dict{String,Any}("gpu" => false)   # Nthreads defaults to nthreads()
    redirect_stdout(devnull) do
        redirect_stderr(devnull) do
            simulate(eobj.phantom, seq, scn; sim_params = sp)   # JIT warm
            best = Inf
            for _ in 1:repeats
                t = @elapsed simulate(eobj.phantom, seq, scn; sim_params = sp)
                best = min(best, t)
            end
            return best
        end
    end
end
'''


def _run_worker(args) -> None:
    """Time each config of `args.axis` in this one process; emit a JSON line."""
    from qalibremd_gym.env_e2 import QalibreMDE2Env

    base_kwargs = e2_env_kwargs(args)
    values = _parse_values(args.axis, args.values)
    records: list[dict] = []

    jl = None
    for value in values:
        kw = _override(args.axis, value, base_kwargs) if args.axis in _CHEAP \
            else base_kwargs
        env = QalibreMDE2Env(rng_seed=args.seed, **kw)
        env.reset(seed=args.seed)
        if jl is None:                       # first env booted Julia → set up probe
            from juliacall import Main as _jl
            jl = _jl
            jl.seval(_PROBE_SETUP)
            nthreads = int(jl.seval("Threads.nthreads()"))
        n_spins = int(len(env._env.phantom.x))
        sim_s = float(jl._sweep_probe(env._env, args.repeats))
        label = ("on" if (args.axis == "water" and value) else
                 "off" if args.axis == "water" else
                 f"{value:.1f}mm" if args.axis == "water_voxel" else value)
        records.append({
            "axis": args.axis, "value": label, "metric": "sim_s",
            "metric_value": sim_s, "n_spins": n_spins, "nthreads": nthreads,
        })
        print(f"[worker] {args.axis}={label}: n_spins={n_spins:,} "
              f"sim={sim_s:.3f}s (nthreads={nthreads})", file=sys.stderr)
    print(_MARKER + json.dumps(records))


# ── orchestrator ─────────────────────────────────────────────────────────────

def _spawn_worker(axis: str, values: str, extra_env: dict | None,
                  passthrough: list[str]) -> list[dict]:
    """Run this script in --worker mode as a subprocess; parse its JSON line."""
    cmd = [sys.executable, str(Path(__file__).resolve()),
           "--worker", "--axis", axis, "--values", values, *passthrough]
    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"worker failed (axis={axis}): exit {proc.returncode}")
    for line in proc.stdout.splitlines():
        if line.startswith(_MARKER):
            return json.loads(line[len(_MARKER):])
    raise RuntimeError(f"worker produced no result line (axis={axis})")


def _measure_n_envs(values, base_kwargs, seed, warmup, repeats) -> list[dict]:
    """Effective per-step wall via SubprocVecEnv (reuses bench_e2's harness)."""
    from bench_e2 import _time_vec, _time_single
    records = []
    for n in values:
        if n <= 1:
            per_step, _ep, _ns, _spins = _time_single(
                base_kwargs, seed, warmup, max(8, repeats * 4))
            eff = float(per_step.mean())
        else:
            eff = float(_time_vec(base_kwargs, seed, n, warmup,
                                  max(8, repeats * 4)))
        records.append({"axis": "n-envs", "value": n, "metric": "eff_step_s",
                        "metric_value": eff, "n_spins": None, "nthreads": 1})
        print(f"[sweep] n-envs={n}: eff_step={eff:.3f}s", file=sys.stderr)
    return records


def _passthrough_args(argv: list[str]) -> list[str]:
    """The baseline E2 flags to forward to workers (drop sweep-only flags)."""
    drop = {"--axis", "--values", "--out", "--figdir", "--repeats", "--worker"}
    out, skip = [], False
    for tok in argv:
        if skip:
            skip = False
            continue
        if tok in drop:
            skip = tok != "--worker"      # --worker takes no value
            continue
        out.append(tok)
    return out


def _orchestrate(args, argv: list[str]) -> None:
    axes = [a for a in ("npe", "voxel", "water", "water_voxel", "nthreads", "n-envs")] \
        if args.axis == "all" else [args.axis]
    base_kwargs = e2_env_kwargs(args)
    passthrough = _passthrough_args(argv)
    meta = git_meta()

    for axis in axes:
        values = _parse_values(axis, args.values if args.axis != "all" else None)
        print(f"\n[sweep] === axis: {axis} ({values}) ===", file=sys.stderr)
        if axis in _CHEAP:
            vstr = args.values if (args.axis == axis and args.values) \
                else _DEFAULT_VALUES[axis]
            recs = _spawn_worker(axis, vstr, None, passthrough)
        elif axis == "nthreads":
            recs = []
            for v in values:
                env = {"PYTHON_JULIACALL_THREADS": str(v),
                       "PYTHON_JULIACALL_HANDLE_SIGNALS": "yes",
                       "JULIA_NUM_THREADS": str(v)} if v > 1 else None
                recs += _spawn_worker("nthreads", str(v), env, passthrough)
        elif axis == "n-envs":
            recs = _measure_n_envs(values, base_kwargs, args.seed,
                                   args.warmup, args.repeats)
        else:
            raise ValueError(axis)

        for r in recs:
            append_result(args.out, {**meta, "args": vars(args), **r})
        _plot_axis(axis, args.out, args.figdir)


def _plot_axis(axis: str, results_json: str, figdir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    records = json.loads(Path(results_json).read_text())
    rows = [r for r in records if r.get("axis") == axis]
    if not rows:
        return
    by_val: dict = defaultdict(list)
    spins: dict = {}
    for r in rows:
        by_val[str(r["value"])].append(r["metric_value"])
        if r.get("n_spins") is not None:
            spins[str(r["value"])] = r["n_spins"]
    labels = list(by_val.keys())
    means = [float(np.mean(by_val[k])) for k in labels]
    stds = [float(np.std(by_val[k])) for k in labels]
    metric = rows[-1]["metric"]

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=4, color="#4C72B0")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel(axis)
    ax.set_ylabel("effective per-step wall [s]" if metric == "eff_step_s"
                  else "best-of-N simulate() [s]")
    n_rep = max(len(v) for v in by_val.values())
    ax.set_title(f"E2 sweep — {axis}  (mean±std over {n_rep} run(s))")
    if spins:
        for xi, k in zip(x, labels):
            if k in spins:
                ax.annotate(f"{spins[k]:,} spins", (xi, means[labels.index(k)]),
                            ha="center", va="bottom", fontsize=7, rotation=0)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = Path(figdir) / f"{axis}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[sweep] chart → {out}", file=sys.stderr)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--axis", required=True,
                   choices=["npe", "voxel", "water", "water_voxel", "nthreads", "n-envs", "all"])
    p.add_argument("--values", type=str, default=None,
                   help="Comma-separated axis values; per-axis defaults if omitted.")
    p.add_argument("--repeats", type=int, default=3,
                   help="best-of-N simulate() samples per config (controlled timing).")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str,
                   default=str(RUNS_DIR / "sweep" / "sweep_results.json"))
    p.add_argument("--figdir", type=str,
                   default=str(RUNS_DIR / "sweep" / "figures"))
    p.add_argument("--worker", action="store_true",
                   help="Internal: run one warm-process sweep and emit JSON.")
    add_e2_env_args(p)
    args = p.parse_args()

    if args.worker:
        _run_worker(args)
    else:
        _orchestrate(args, sys.argv[1:])


if __name__ == "__main__":
    main()

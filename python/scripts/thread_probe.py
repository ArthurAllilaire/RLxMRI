"""Probe whether Koma's CPU Bloch sim actually parallelises over threads.

Boots one juliacall runtime with N Julia threads, builds the E2 env exactly as
bench/train do, then times a *single* simulate() at several thread counts. Use
it to decide whether the --nthreads lever in bench_e2.py is worth setting on a
given box before committing to a long run.

Two findings this script exists to demonstrate:
  1. PYTHON_JULIACALL_HANDLE_SIGNALS=yes is MANDATORY with >1 thread. Without it
     the multithreaded sim runs with no speedup or segfaults ("dumped core").
  2. The env path (sim_params={"gpu":…}, no "Nthreads" key) threads fine — Koma's
     default Nthreads tracks Threads.nthreads(), so just booting with more
     threads is enough; no Julia-side change needed.

Usage:
    PYTHON_JULIAPKG_OFFLINE=yes python python/scripts/thread_probe.py
    PYTHON_JULIAPKG_OFFLINE=yes python python/scripts/thread_probe.py --nthreads 8
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# --nthreads is read here (not argparse) because the env vars must be set BEFORE
# the Julia runtime boots, which happens on first env construction below.
_n = 4
for i, a in enumerate(sys.argv):
    if a == "--nthreads" and i + 1 < len(sys.argv):
        _n = int(sys.argv[i + 1])
if _n > 1:
    os.environ["PYTHON_JULIACALL_THREADS"] = str(_n)
    os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"
    os.environ.setdefault("JULIA_NUM_THREADS", str(_n))

# python/ on the path so qalibremd_gym + e2_config import regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qalibremd_gym.env_e2 import QalibreMDE2Env
from e2_config import e2_env_kwargs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--nthreads", type=int, default=4)
    args = p.parse_args()

    env_args = argparse.Namespace(
        field="T15", max_blocks=30, time_budget=160.0, subset_size=None,
        noise=50.0, Nfe=64, Npe=32, voxel_mm=1.0, use_gpu=False,
        reward_mode="neg_mape", log_ti_action=False, simplified_action=False,
        fix_te=True, learn_alpha=True, terminal_bonus=0.5, mape_alpha=1.0,
        phase_sensitive=False, sigma_method="bootstrap", include_image=False,
        include_sigma=False, include_water=True,
    )
    env = QalibreMDE2Env(rng_seed=0, **e2_env_kwargs(env_args))
    env.reset(seed=0)

    from juliacall import Main as jl
    print("runtime Threads.nthreads() =", jl.seval("Threads.nthreads()"))
    jl.seval("using KomaMRI")
    jl.eobj = env._env
    print("n_spins =", jl.seval("length(eobj.phantom.x)"))

    jl.seval('''
    seq = QalibreMDPhantom.ir_se_2d_sequence(0.8, 0.02, 3.0;
            α_exc=deg2rad(90.0), FOV=eobj.FOV, Nfe=eobj.Nfe, Npe=eobj.Npe)
    scn = QalibreMDPhantom.scanner_for_field(eobj.cfg_field)
    function probe_time(sp)
        redirect_stdout(devnull) do
            redirect_stderr(devnull) do
                simulate(eobj.phantom, seq, scn; sim_params=sp)  # JIT warm
                best = Inf
                for _ in 1:3
                    t = @elapsed simulate(eobj.phantom, seq, scn; sim_params=sp)
                    best = min(best, t)
                end
                return best
            end
        end
    end
    probe_explicit(nth) = probe_time(Dict{String,Any}("gpu"=>false, "Nthreads"=>nth))
    probe_envpath()     = probe_time(Dict{String,Any}("gpu"=>false))
    ''')

    print(f"env path (no Nthreads key): {float(jl.probe_envpath()):.3f}s")
    for nth in (1, 2, 4, args.nthreads):
        print(f"explicit Nthreads={nth}: best-of-3 = "
              f"{float(jl.probe_explicit(nth)):.3f}s")


if __name__ == "__main__":
    main()

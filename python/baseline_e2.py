"""Fixed-schedule baselines for E2 — the missing yardstick for §§8-9 in
EXPERT_REPORT_E2_4.md.

Runs two non-RL policies through QalibreMDE2Env on the same eval seeds
used by train_e2.py (default 500_000 + i):

  - "log_grid"      : 7-block log-spaced TI ∈ [0.05, 3.0] s, TR=4 s, α=90°.
                      Same schedule embedded in eval_e2.py:_fixed_grid_action.
  - "clinical_irse" : 6-block radiographer-style TI ∈ {0.05, 0.15, 0.4,
                      0.9, 1.6, 2.5} s, TR=5 s, α=90°.

Reports per-policy: mean / p90 MAPE, success rate, mean scan time, and
the per-sphere MAPE vector (so the report can compare against V5's
short-T1-tail diagnosis in §9.9).

Usage:
    PYTHON_JULIAPKG_OFFLINE=yes python python/baseline_e2.py \
        --episodes 30 --out runs/e2/baselines
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from qalibremd_gym.env_e2 import QalibreMDE2Env


def _phys_to_norm(env: QalibreMDE2Env, phys: np.ndarray) -> np.ndarray:
    return 2.0 * (phys - env._ACT_LO) / (env._ACT_HI - env._ACT_LO) - 1.0


def _log_grid(step: int) -> np.ndarray:
    tis = np.exp(np.linspace(np.log(0.05), np.log(3.0), 7))
    return np.array([float(tis[step % 7]), 0.020, 4.0, 90.0, 0.0],
                    dtype=np.float32)


def _clinical_irse(step: int) -> np.ndarray:
    tis = [0.05, 0.15, 0.4, 0.9, 1.6, 2.5]
    return np.array([float(tis[step % len(tis)]), 0.020, 5.0, 90.0, 0.0],
                    dtype=np.float32)


def _log_grid_trmatched(step: int) -> np.ndarray:
    """log_grid with TR matched to V5's average (1.7 s).

    Same TI grid as `_log_grid`, but TR shortened from 4.0 s → 1.7 s so the
    schedule fits ≈ 8 blocks into the 120 s budget — the same data-points-
    per-sphere count V5 achieves. Isolates "RL win is TR efficiency" from
    "RL win is per-sphere TI targeting".
    """
    tis = np.exp(np.linspace(np.log(0.05), np.log(3.0), 7))
    return np.array([float(tis[step % 7]), 0.020, 1.7, 90.0, 0.0],
                    dtype=np.float32)


SCHEDULES = {
    "log_grid":            _log_grid,
    "clinical_irse":       _clinical_irse,
    "log_grid_trmatched":  _log_grid_trmatched,
}


def make_cr_optimal_schedule(T1s, budget_s, Npe=8,
                              n_block_grid=(4, 6, 8, 10, 14, 18),
                              n_starts=1000, n_refine=10, rng_seed=0):
    """Solve the CR-optimal fixed-block schedule for the given fleet via Julia.

    Returns a (TIs_list, TRs_list, n_blocks, L) tuple. TE = 20 ms, alpha = 90°
    are fixed (matching the F1+ Jacobian assumption).
    """
    from qalibremd_gym import env as _env_mod  # Julia bridge
    _env_mod._ensure_julia(None)
    jl  = _env_mod._JL
    qmd = _env_mod._JL_QMD
    # Convert Python lists to typed Julia arrays — juliacall infers PyList{Any}
    # for plain Python lists, which Julia keyword args reject when typed.
    T1s_jl = jl.Vector[jl.Float64](list(map(float, T1s)))
    nbg_jl = jl.Vector[jl.Int](list(int(x) for x in n_block_grid))
    res = qmd.cr_optimize_sweep(
        T1s_jl,
        budget_s     = float(budget_s),
        Npe          = int(Npe),
        n_block_grid = nbg_jl,
        n_starts     = int(n_starts),
        n_refine     = int(n_refine),
    )
    TIs = [float(t) for t in res.schedule.TIs]
    TRs = [float(t) for t in res.schedule.TRs]
    n_blocks = int(res.n_blocks)
    L = float(res.schedule.L)
    return TIs, TRs, n_blocks, L


def _cr_optimal_factory(TIs, TRs):
    """Build a step → action callable that cycles through the precomputed
    CR-optimal (TI, TR) schedule. TE = 20 ms, alpha = 90°, slice_z = 0."""
    n = len(TIs)
    def _sched(step: int) -> np.ndarray:
        k = step % n
        return np.array([TIs[k], 0.020, TRs[k], 90.0, 0.0], dtype=np.float32)
    return _sched


def evaluate(name: str, sched_fn, n_episodes: int, seed_offset: int) -> dict:
    env = QalibreMDE2Env(rng_seed=seed_offset)
    mapes, per_sphere, times, ep_lens = [], [], [], []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_offset + ep)
        done, step, info = False, 0, {}
        while not done:
            phys = sched_fn(step)
            obs, _r, done, _trunc, info = env.step(_phys_to_norm(env, phys))
            step += 1
        mapes.append(float(info.get("mape", np.nan)))
        per_sphere.append(np.abs(env.T1_est - env.T1_true) / env.T1_true)
        times.append(env.time_used_s)
        ep_lens.append(step)

    arr = np.array(per_sphere)
    return {
        "schedule":           name,
        "n_episodes":         n_episodes,
        "mape_pct":           float(np.nanmean(mapes)) * 100,
        "mape_p90_pct":       float(np.nanpercentile(mapes, 90)) * 100,
        "success_5pct":       float(np.mean([m < 0.05 for m in mapes
                                              if not np.isnan(m)])),
        "mean_scan_time_s":   float(np.mean(times)),
        "mean_ep_len":        float(np.mean(ep_lens)),
        "per_sphere_mape_pct": (np.nanmean(arr, axis=0) * 100).tolist(),
        "per_sphere_p90_pct":  (np.nanpercentile(arr, 90, axis=0) * 100).tolist(),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--seed",     type=int, default=500_000)
    p.add_argument("--out",      type=Path, default=Path("runs/e2/baselines"))
    p.add_argument("--cr-optimal", action="store_true",
                   help="Also solve and run the CR-optimal fixed schedule on "
                        "the 14-sphere T3 fleet at the env's default budget.")
    p.add_argument("--cr-budget", type=float, default=120.0,
                   help="Scan-time budget for the CR-optimal solve [s].")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    schedules = dict(SCHEDULES)

    if args.cr_optimal:
        # T3 14-sphere fleet — should match T1_ARRAY[:T3] in src/materials/t1_array.jl
        T1s_T3 = [1.838, 1.398, 0.998, 0.726, 0.509, 0.367, 0.259, 0.185,
                  0.131, 0.091, 0.064, 0.046, 0.033, 0.023]
        print(f"\n[baseline_e2] Solving CR-optimal schedule "
              f"(fleet=14 spheres, budget={args.cr_budget}s) …")
        TIs, TRs, n_blocks, L = make_cr_optimal_schedule(
            T1s_T3, budget_s=args.cr_budget)
        print(f"  best n_blocks = {n_blocks}, L = {L:.4f}")
        print(f"  TIs (sorted)  = {sorted(round(t, 4) for t in TIs)}")
        print(f"  TRs (sorted)  = {sorted(round(t, 4) for t in TRs)}")
        schedules["cr_optimal"] = _cr_optimal_factory(TIs, TRs)
        # Persist the CR-opt schedule for reproducibility
        with (args.out / "cr_optimal_schedule.json").open("w") as f:
            json.dump({"TIs": TIs, "TRs": TRs, "n_blocks": n_blocks, "L": L,
                       "T1s": T1s_T3, "budget_s": args.cr_budget}, f, indent=2)

    results = {}
    for name, fn in schedules.items():
        print(f"\n[baseline_e2] Evaluating '{name}' on {args.episodes} eps "
              f"(seeds {args.seed}…{args.seed + args.episodes - 1})")
        r = evaluate(name, fn, args.episodes, args.seed)
        results[name] = r
        print(f"  MAPE       = {r['mape_pct']:.2f}%")
        print(f"  p90 MAPE   = {r['mape_p90_pct']:.2f}%")
        print(f"  Success<5% = {r['success_5pct']:.1%}")
        print(f"  Mean time  = {r['mean_scan_time_s']:.1f}s "
              f"({r['mean_ep_len']:.1f} blocks)")
        print(f"  Per-sphere MAPE [%] (long-T1 → short-T1):")
        for i, v in enumerate(r['per_sphere_mape_pct']):
            print(f"    sphere {i:2d}: {v:7.1f}%")

    out_path = args.out / "baseline_summary.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[baseline_e2] Summary → {out_path}")


if __name__ == "__main__":
    main()

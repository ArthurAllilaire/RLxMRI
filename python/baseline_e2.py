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
from qalibremd_gym.schedules import log_ti_grid


def _active_nominal_t1s(env: QalibreMDE2Env) -> list[float]:
    return [float(d.T1) for d in env._env.active_base_descs]


def _phys_to_norm(env: QalibreMDE2Env, phys: np.ndarray) -> np.ndarray:
    return 2.0 * (phys - env._ACT_LO) / (env._ACT_HI - env._ACT_LO) - 1.0


def _log_grid(step: int) -> np.ndarray:
    tis = log_ti_grid()
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
    tis = log_ti_grid()
    return np.array([float(tis[step % 7]), 0.020, 1.7, 90.0, 0.0],
                    dtype=np.float32)


SCHEDULES = {
    "log_grid":            _log_grid,
    "clinical_irse":       _clinical_irse,
    "log_grid_trmatched":  _log_grid_trmatched,
}


def nominal_fleet_t1s(field: str) -> list[float]:
    """Full 14-sphere nominal T1 fleet for `field`, read from Julia's
    `T1_ARRAY` (the single source of truth — no Python copy)."""
    from qalibremd_gym import env as _env_mod  # Julia bridge
    _env_mod._ensure_julia(None)
    jl  = _env_mod._JL
    qmd = _env_mod._JL_QMD
    return [float(t) for t in qmd.T1_ARRAY[jl.Symbol(field)]]


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


def make_cr_optimal_alpha_schedule(T1s, budget_s, Npe=8,
                                    n_block_grid=(4, 6, 8, 10, 14, 18),
                                    n_starts=1000, n_refine=10):
    """α-aware CR-optimal schedule (ALPHA_DOF.md, reference point (c)). Optimises (TI, TR, α)
    jointly; α is a design variable, the Fisher stays 2×2 over (T1, A).

    Returns (TIs, TRs, alphas_deg, n_blocks, L). TE = 20 ms is fixed.
    """
    from qalibremd_gym import env as _env_mod
    _env_mod._ensure_julia(None)
    jl  = _env_mod._JL
    qmd = _env_mod._JL_QMD
    T1s_jl = jl.Vector[jl.Float64](list(map(float, T1s)))
    nbg_jl = jl.Vector[jl.Int](list(int(x) for x in n_block_grid))
    res = qmd.cr_optimize_sweep_alpha(
        T1s_jl,
        budget_s     = float(budget_s),
        Npe          = int(Npe),
        n_block_grid = nbg_jl,
        n_starts     = int(n_starts),
        n_refine     = int(n_refine),
    )
    TIs = [float(t) for t in res.schedule.TIs]
    TRs = [float(t) for t in res.schedule.TRs]
    alphas_deg = [float(np.degrees(a)) for a in res.schedule.αs]
    n_blocks = int(res.n_blocks)
    L = float(res.schedule.L)
    return TIs, TRs, alphas_deg, n_blocks, L


def _cr_optimal_alpha_factory(TIs, TRs, alphas_deg):
    """Cycle through the α-aware CR-optimal (TI, TR, α) schedule.
    TE = 20 ms, slice_z = 0; α is the optimised per-block flip angle."""
    n = len(TIs)
    def _sched(step: int) -> np.ndarray:
        k = step % n
        return np.array([TIs[k], 0.020, TRs[k], alphas_deg[k], 0.0],
                        dtype=np.float32)
    return _sched


def _ernst_fixed_factory(TIs, TRs, T1_ref):
    """Reuse the CR-opt (TI, TR) timing but set α to the Ernst angle at the
    reference (fleet-median) T1 per block (ALPHA_DOF.md, reference point (b)). Delegates to the
    Julia `ernst_fixed_schedule` so the physics lives in the package."""
    from qalibremd_gym import env as _env_mod
    _env_mod._ensure_julia(None)
    jl  = _env_mod._JL
    qmd = _env_mod._JL_QMD
    TIs_jl = jl.Vector[jl.Float64](list(map(float, TIs)))
    TRs_jl = jl.Vector[jl.Float64](list(map(float, TRs)))
    alphas_rad = qmd.ernst_fixed_schedule(TIs_jl, TRs_jl, float(T1_ref))
    alphas_deg = [float(np.degrees(a)) for a in alphas_rad]
    n = len(TIs)
    def _sched(step: int) -> np.ndarray:
        k = step % n
        return np.array([TIs[k], 0.020, TRs[k], alphas_deg[k], 0.0],
                        dtype=np.float32)
    return _sched


def evaluate(name: str, sched_fn, n_episodes: int, seed_offset: int,
             *, max_steps: int | None = None, **env_kwargs) -> dict:
    """Roll a fixed schedule through the env for `n_episodes` paired seeds.

    `max_steps` makes the schedule run *once* (its designed n_blocks) rather
    than cycling to fill the budget: a CR-optimal schedule is a one-shot design,
    so re-measuring into leftover slack would give it more data than the solver
    optimised for. Conventional repeating protocols (log_grid, clinical_irse)
    pass max_steps=None and keep cycling.
    """
    env = QalibreMDE2Env(rng_seed=seed_offset, **env_kwargs)
    mapes, per_sphere, times, ep_lens = [], [], [], []
    pool_errs = {i: [] for i in range(1, 15)}
    subset_indices = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_offset + ep)
        done, step, info = False, 0, {}
        while not done:
            phys = sched_fn(step)
            obs, _r, done, _trunc, info = env.step(_phys_to_norm(env, phys))
            step += 1
            if max_steps is not None and step >= max_steps:
                break
        mapes.append(float(info.get("mape", np.nan)))
        errs = np.abs(env.T1_est - env.T1_true) / env.T1_true
        per_sphere.append(errs)
        idxs = np.asarray(info.get("sphere_indices", []), dtype=np.int64)
        if idxs.size == errs.size:
            subset_indices.append(idxs.tolist())
            for idx, err in zip(idxs, errs):
                pool_errs[int(idx)].append(float(err))
        times.append(env.time_used_s)
        ep_lens.append(step)

    max_slots = max(len(x) for x in per_sphere)
    arr = np.full((len(per_sphere), max_slots), np.nan)
    for i, errs in enumerate(per_sphere):
        arr[i, :len(errs)] = errs
    result = {
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
        "subset_indices":     subset_indices,
    }
    pool_mean = []
    pool_p90 = []
    pool_n = []
    for i in range(1, 15):
        vals = np.asarray(pool_errs[i], dtype=np.float64)
        pool_n.append(int(vals.size))
        pool_mean.append(float(np.nanmean(vals)) * 100 if vals.size else None)
        pool_p90.append(float(np.nanpercentile(vals, 90)) * 100 if vals.size else None)
    result["per_pool_sphere_mape_pct"] = pool_mean
    result["per_pool_sphere_p90_pct"] = pool_p90
    result["per_pool_sphere_n"] = pool_n
    return result


def evaluate_cr_oracle(n_episodes: int, seed_offset: int,
                       cr_budget: float, cr_block_grid: list[int],
                       cr_starts: int, cr_refine: int,
                       **env_kwargs) -> dict:
    """Evaluate Formulation B: solve a CR-optimal schedule for each subset.

    This is an oracle lower bound for non-adaptive fixed schedules because it
    sees the episode's sphere subset before choosing the schedule.
    """
    env = QalibreMDE2Env(rng_seed=seed_offset, **env_kwargs)
    cr_npe = int(env_kwargs.get("Npe", 32))
    cache: dict[tuple[int, ...], tuple[list[float], list[float], int, float]] = {}

    def _solve_for_current_subset(info):
        idxs = tuple(int(i) for i in info["sphere_indices"])
        if idxs not in cache:
            # Oracle sees the subset identity, not the episode's jittered T1.
            T1s = _active_nominal_t1s(env)
            cache[idxs] = make_cr_optimal_schedule(
                T1s,
                budget_s=cr_budget,
                Npe=cr_npe,
                n_block_grid=tuple(cr_block_grid),
                n_starts=cr_starts,
                n_refine=cr_refine,
            )
        return cache[idxs]

    mapes, per_sphere, times, ep_lens, subset_indices = [], [], [], [], []
    pool_errs = {i: [] for i in range(1, 15)}
    schedules = {}

    for ep in range(n_episodes):
        obs, reset_info = env.reset(seed=seed_offset + ep)
        TIs, TRs, n_blocks, L = _solve_for_current_subset(reset_info)
        sched_fn = _cr_optimal_factory(TIs, TRs)
        idxs = tuple(int(i) for i in reset_info["sphere_indices"])
        schedules[str(idxs)] = {"TIs": TIs, "TRs": TRs,
                                "n_blocks": n_blocks, "L": L}

        done, step, info = False, 0, {}
        while not done:
            phys = sched_fn(step)
            obs, _r, done, _trunc, info = env.step(_phys_to_norm(env, phys))
            step += 1
            if step >= n_blocks:        # run-once: the oracle schedule is a one-shot design
                break

        errs = np.abs(env.T1_est - env.T1_true) / env.T1_true
        mapes.append(float(info.get("mape", np.nan)))
        per_sphere.append(errs)
        subset_indices.append(list(idxs))
        for idx, err in zip(idxs, errs):
            pool_errs[int(idx)].append(float(err))
        times.append(env.time_used_s)
        ep_lens.append(step)

    max_slots = max(len(x) for x in per_sphere)
    arr = np.full((len(per_sphere), max_slots), np.nan)
    for i, errs in enumerate(per_sphere):
        arr[i, :len(errs)] = errs

    pool_mean, pool_p90, pool_n = [], [], []
    for i in range(1, 15):
        vals = np.asarray(pool_errs[i], dtype=np.float64)
        pool_n.append(int(vals.size))
        pool_mean.append(float(np.nanmean(vals)) * 100 if vals.size else None)
        pool_p90.append(float(np.nanpercentile(vals, 90)) * 100 if vals.size else None)

    return {
        "schedule": "cr_oracle",
        "formulation": "oracle_per_subset",
        "n_episodes": n_episodes,
        "n_unique_subsets_solved": len(cache),
        "mape_pct": float(np.nanmean(mapes)) * 100,
        "mape_p90_pct": float(np.nanpercentile(mapes, 90)) * 100,
        "success_5pct": float(np.mean([m < 0.05 for m in mapes
                                        if not np.isnan(m)])),
        "mean_scan_time_s": float(np.mean(times)),
        "mean_ep_len": float(np.mean(ep_lens)),
        "per_sphere_mape_pct": (np.nanmean(arr, axis=0) * 100).tolist(),
        "per_sphere_p90_pct": (np.nanpercentile(arr, 90, axis=0) * 100).tolist(),
        "per_pool_sphere_mape_pct": pool_mean,
        "per_pool_sphere_p90_pct": pool_p90,
        "per_pool_sphere_n": pool_n,
        "subset_indices": subset_indices,
        "schedules_by_subset": schedules,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--seed",     type=int, default=500_000)
    p.add_argument("--out",      type=Path, default=Path("runs/e2/baselines"))
    p.add_argument("--field",    type=str, default="T3", choices=["T3", "T15"])
    p.add_argument("--max-blocks", type=int, default=15)
    p.add_argument("--time-budget", type=float, default=120.0)
    p.add_argument("--subset-size", type=int, default=None,
                   help="Evaluate schedules on random k-sphere subsets.")
    p.add_argument("--noise", type=float, default=50.0,
                   help="Absolute complex-Gaussian σ on k-space (FIX_SIM_PLAN §2). "
                        "Default σ*=50 → NEMA dual-acq SNR ≈ 25 (E2_RERUN_PLAN §3.1).")
    p.add_argument("--nfe", type=int, default=64,
                   help="Frequency-encode samples (image width). Must match the "
                        "RL env resolution for an apples-to-apples yardstick.")
    p.add_argument("--npe", type=int, default=32,
                   help="Phase-encode steps (image height) AND the per-block shot "
                        "count: block scan-time = Npe·TR. Threaded into the CR "
                        "solver so its budget/Fisher use the same Npe as the env.")
    p.add_argument("--phase-sensitive", action="store_true")
    p.add_argument("--sigma-method", type=str, default="bootstrap",
                   choices=["asymptotic", "profile_likelihood", "bootstrap"])
    p.add_argument("--cr-optimal", action="store_true",
                   help="Also solve and run the CR-optimal fixed schedule on "
                        "the 14-sphere T3 fleet at the env's default budget.")
    p.add_argument("--cr-oracle", action="store_true",
                   help="Also run oracle CR-optimal Formulation B: solve one "
                        "fixed schedule per sampled subset. Use with "
                        "--subset-size for the E2-tractability lower bound.")
    p.add_argument("--ernst-baseline", action="store_true",
                   help="Also run the Ernst-fixed-α baseline (ALPHA_DOF "
                        "§3b): CR-opt (TI,TR) timing with α set to the Ernst "
                        "angle at the fleet-median T1. Requires --cr-optimal.")
    p.add_argument("--cr-optimize-alpha", action="store_true",
                   help="Also solve and run the α-aware CR-optimal schedule "
                        "(ALPHA_DOF.md, reference point (c)): jointly optimises (TI,TR,α). "
                        "The 'does CR-opt-with-α close the gap?' control.")
    p.add_argument("--cr-budget", type=float, default=None,
                   help="Scan-time budget for the CR-optimal solve [s].")
    p.add_argument("--cr-block-grid", type=int, nargs="+",
                   default=[4, 6, 8, 10, 14, 18],
                   help="n_blocks values swept by the CR optimiser.")
    p.add_argument("--cr-starts", type=int, default=1000)
    p.add_argument("--cr-refine", type=int, default=10)
    p.add_argument("--cr-load", type=Path, default=None,
                   help="Load CR-optimal schedule from a JSON (TIs, TRs) "
                        "instead of re-solving. Skips the multi-start optimiser.")
    p.add_argument("--cr-only", action="store_true",
                   help="Skip the default fixed schedules; run only --cr-optimal "
                        "and/or --cr-oracle.")
    p.add_argument("--oracle-fit", action="store_true",
                   help="D2: narrow the fitter T1 grid to ±oracle-band around "
                        "T1_true per sphere. Diagnostic — never report.")
    p.add_argument("--oracle-band", type=float, default=2.0)
    p.add_argument("--fitter-n-grid", type=int, default=200,
                   help="n_grid for fit_t1_generalized_ir. §17.10 control: "
                        "use 2000 to test grid coarseness vs multimodal SSE.")
    p.add_argument("--include-image", action="store_true",
                   help="Prepend the flattened recon image to the obs "
                        "(E2_RERUN_PLAN §6.2). Default off; does not affect "
                        "fixed/CR baselines (they ignore the obs).")
    p.add_argument("--include-sigma", action="store_true",
                   help="Append the per-sphere fitter-σ channel to the obs "
                        "(E2_RERUN_PLAN §6.3). Default off.")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    schedules = {} if args.cr_only else dict(SCHEDULES)
    # Per-schedule step cap. One-shot CR designs run exactly their n_blocks
    # (run-once); conventional repeating protocols stay uncapped (cycle-to-fill).
    schedule_max_steps: dict[str, int] = {}
    env_kwargs = dict(
        cfg_field=args.field,
        Nfe=args.nfe,
        Npe=args.npe,
        max_blocks=args.max_blocks,
        time_budget_s=args.time_budget,
        subset_size=args.subset_size,
        noise_sigma_abs=args.noise,
        phase_sensitive=args.phase_sensitive,
        sigma_method=args.sigma_method,
        oracle_fit=args.oracle_fit,
        oracle_band=args.oracle_band,
        fitter_n_grid=args.fitter_n_grid,
        include_image=args.include_image,
        include_sigma=args.include_sigma,
    )
    cr_budget = args.time_budget if args.cr_budget is None else args.cr_budget

    # 14-sphere nominal fleet for the chosen field, from Julia's T1_ARRAY.
    T1s_fleet = nominal_fleet_t1s(args.field)
    T1_median = float(np.median(T1s_fleet))

    if args.ernst_baseline and not args.cr_optimal:
        raise ValueError("--ernst-baseline reuses the CR-opt (TI,TR) timing; "
                         "pass --cr-optimal too.")

    if args.cr_optimal:
        if args.cr_load is not None:
            print(f"\n[baseline_e2] Loading CR-optimal schedule from "
                  f"{args.cr_load} (skipping optimiser) …")
            with args.cr_load.open() as f:
                cr_data = json.load(f)
            TIs = list(cr_data["TIs"])
            TRs = list(cr_data["TRs"])
            n_blocks = int(cr_data["n_blocks"])
            L = float(cr_data["L"])
        else:
            print(f"\n[baseline_e2] Solving CR-optimal schedule "
                  f"(expected random-subset fleet=14 spheres, budget={cr_budget}s) …")
            TIs, TRs, n_blocks, L = make_cr_optimal_schedule(
                T1s_fleet, budget_s=cr_budget, Npe=args.npe,
                n_block_grid=tuple(args.cr_block_grid),
                n_starts=args.cr_starts,
                n_refine=args.cr_refine)
        print(f"  best n_blocks = {n_blocks}, L = {L:.4f}")
        print(f"  TIs (sorted)  = {sorted(round(t, 4) for t in TIs)}")
        print(f"  TRs (sorted)  = {sorted(round(t, 4) for t in TRs)}")
        schedules["cr_optimal"] = _cr_optimal_factory(TIs, TRs)
        schedule_max_steps["cr_optimal"] = n_blocks
        # Persist the CR-opt schedule for reproducibility
        with (args.out / "cr_optimal_schedule.json").open("w") as f:
            json.dump({"TIs": TIs, "TRs": TRs, "n_blocks": n_blocks, "L": L,
                       "T1s": T1s_fleet, "budget_s": cr_budget,
                       "formulation": "expected_loss_all_14_pool",
                       "subset_size_eval": args.subset_size,
                       "n_block_grid": args.cr_block_grid}, f, indent=2)

        if args.ernst_baseline:
            print(f"\n[baseline_e2] Building Ernst-fixed-α baseline "
                  f"(fleet-median T1 = {T1_median:.3f}s) on the CR-opt timing …")
            schedules["ernst_fixed"] = _ernst_fixed_factory(TIs, TRs, T1_median)
            schedule_max_steps["ernst_fixed"] = n_blocks

    if args.cr_optimize_alpha:
        print(f"\n[baseline_e2] Solving α-aware CR-optimal schedule "
              f"(joint TI,TR,α; budget={cr_budget}s) …")
        TIs_a, TRs_a, alphas_a, n_blocks_a, L_a = make_cr_optimal_alpha_schedule(
            T1s_fleet, budget_s=cr_budget, Npe=args.npe,
            n_block_grid=tuple(args.cr_block_grid),
            n_starts=args.cr_starts, n_refine=args.cr_refine)
        print(f"  best n_blocks = {n_blocks_a}, L = {L_a:.4f}")
        print(f"  α (deg, sorted) = {sorted(round(a, 1) for a in alphas_a)}")
        schedules["cr_optimal_alpha"] = _cr_optimal_alpha_factory(
            TIs_a, TRs_a, alphas_a)
        schedule_max_steps["cr_optimal_alpha"] = n_blocks_a
        with (args.out / "cr_optimal_alpha_schedule.json").open("w") as f:
            json.dump({"TIs": TIs_a, "TRs": TRs_a, "alphas_deg": alphas_a,
                       "n_blocks": n_blocks_a, "L": L_a, "T1s": T1s_fleet,
                       "budget_s": cr_budget,
                       "n_block_grid": args.cr_block_grid}, f, indent=2)

    if args.cr_oracle and args.subset_size is None:
        raise ValueError("--cr-oracle is intended for --subset-size k evaluations")

    results = {}
    for name, fn in schedules.items():
        print(f"\n[baseline_e2] Evaluating '{name}' on {args.episodes} eps "
              f"(seeds {args.seed}…{args.seed + args.episodes - 1})")
        r = evaluate(name, fn, args.episodes, args.seed,
                     max_steps=schedule_max_steps.get(name), **env_kwargs)
        results[name] = r
        print(f"  MAPE       = {r['mape_pct']:.2f}%")
        print(f"  p90 MAPE   = {r['mape_p90_pct']:.2f}%")
        print(f"  Success<5% = {r['success_5pct']:.1%}")
        print(f"  Mean time  = {r['mean_scan_time_s']:.1f}s "
              f"({r['mean_ep_len']:.1f} blocks)")
        label = "active slot" if args.subset_size else "sphere"
        print(f"  Per-{label} MAPE [%] (long-T1 → short-T1 within active episode):")
        for i, v in enumerate(r['per_sphere_mape_pct']):
            print(f"    {label} {i:2d}: {v:7.1f}%")
        if args.subset_size:
            print(f"  Per-pool-sphere MAPE [%] (original 1-based T1_ARRAY index):")
            for i, (v, n) in enumerate(zip(r["per_pool_sphere_mape_pct"],
                                           r["per_pool_sphere_n"]), start=1):
                if v is not None:
                    print(f"    pool {i:2d}: {v:7.1f}%  (n={n})")

    if args.cr_oracle:
        print(f"\n[baseline_e2] Evaluating oracle CR-optimal on "
              f"{args.episodes} sampled subsets")
        r = evaluate_cr_oracle(
            args.episodes, args.seed, cr_budget, args.cr_block_grid,
            args.cr_starts, args.cr_refine, **env_kwargs)
        results["cr_oracle"] = r
        print(f"  MAPE       = {r['mape_pct']:.2f}%")
        print(f"  p90 MAPE   = {r['mape_p90_pct']:.2f}%")
        print(f"  Success<5% = {r['success_5pct']:.1%}")
        print(f"  Mean time  = {r['mean_scan_time_s']:.1f}s "
              f"({r['mean_ep_len']:.1f} blocks)")
        print(f"  Unique subsets solved = {r['n_unique_subsets_solved']}")
        with (args.out / "cr_oracle_schedules.json").open("w") as f:
            json.dump(r["schedules_by_subset"], f, indent=2)

    out_path = args.out / "baseline_summary.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[baseline_e2] Summary → {out_path}")


if __name__ == "__main__":
    main()

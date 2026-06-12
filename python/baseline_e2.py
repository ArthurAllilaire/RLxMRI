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
from e2_config import parse_int_csv


def _active_nominal_t1s(env: QalibreMDE2Env) -> list[float]:
    return [float(d.T1) for d in env._env.active_base_descs]


def _mape_uncertainty(mapes: list[float], n_boot: int = 5000,
                      seed: int = 0) -> dict:
    """Episode-level uncertainty on the mean MAPE: standard error of the mean +
    a bootstrap 95% CI (percentile method). Lets the reader judge whether the
    episode count resolves a given gap between schedules. All values in %."""
    m = np.asarray([x for x in mapes if not np.isnan(x)], dtype=np.float64)
    n = m.size
    if n < 2:
        return {"mape_sem_pct": float("nan"), "mape_ci95_pct": [float("nan")] * 2,
                "n_eff": int(n)}
    rng = np.random.default_rng(seed)
    boot = m[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "mape_sem_pct":  float(m.std(ddof=1) / np.sqrt(n)) * 100,
        "mape_ci95_pct": [float(lo) * 100, float(hi) * 100],
        "n_eff":         int(n),
    }


def _phys_to_norm(env: QalibreMDE2Env, phys: np.ndarray) -> np.ndarray:
    # Delegate to the env so the schedule decodes to the requested physical
    # sequence under ANY action mode (fix_te / learn_alpha / log_ti_action).
    # A bare 5-dim linear inverse is only correct in the full-action default and
    # silently mis-routes channels otherwise (see eval_e2.py fix, 2026-06).
    return env.physical_to_norm_action(
        ti_s=float(phys[0]), tr_s=float(phys[2]), alpha_deg=float(phys[3]),
        te_s=float(phys[1]))


def _log_grid(step: int) -> np.ndarray:
    tis = log_ti_grid()
    return np.array([float(tis[step % 7]), 0.020, 4.0, 90.0],
                    dtype=np.float32)


def _clinical_irse(step: int) -> np.ndarray:
    tis = [0.05, 0.15, 0.4, 0.9, 1.6, 2.5]
    return np.array([float(tis[step % len(tis)]), 0.020, 5.0, 90.0],
                    dtype=np.float32)


_LOG_GRID_TR = 1.7  # s — TR used by the trmatched baseline


def make_log_grid_trmatched(budget_s: float, npe: int, tr_s: float = _LOG_GRID_TR):
    """log_grid with TR matched to the RL agent's average (default 1.7 s).

    The number of TI grid points equals the number of blocks that fit in the
    budget (floor(budget / (Npe * TR))), log-spaced over [0.05, 3.0] s. This
    avoids the fixed-7 bug where a 560 s budget would repeat the first 3 TIs,
    or a 240 s budget would miss the upper TI range entirely. Pass ``tr_s`` to
    match the policy's *realised* mean TR (e.g. 1.24 s), which equalises block
    count and isolates the TI-placement gain from the block-count gain.
    """
    n_steps = max(1, int(np.floor(budget_s / (npe * tr_s))))
    tis = log_ti_grid(n=n_steps)
    def _sched(step: int) -> np.ndarray:
        return np.array([float(tis[step % n_steps]), 0.020, tr_s, 90.0],
                        dtype=np.float32)
    return _sched


SCHEDULES = {
    "log_grid":            _log_grid,
    "clinical_irse":       _clinical_irse,
    # log_grid_trmatched is budget-dependent; patched into `schedules` in main()
    # after env_kwargs is resolved.
}

LONG_TR_FIXED_SCHEDULES = {
    "log_grid": 4.0,
    "clinical_irse": 5.0,
}


def _skip_too_short_long_tr_schedules(
    schedules: dict, *, time_budget_s: float, max_blocks: int, npe: int,
) -> dict[str, dict]:
    """Drop long-TR fixed protocols when the budget fits ≤2 blocks.

    Those runs are not meaningful baselines for E2: with only one or two
    measurements the T1 fit is underdetermined/noisy, and the resulting MAPE just
    says "TR was too long for this budget". Keep log_grid_trmatched and the CR
    schedules, which are designed for the budget.
    """
    skipped = {}
    for name, tr_s in LONG_TR_FIXED_SCHEDULES.items():
        if name not in schedules:
            continue
        max_by_time = int(np.floor(float(time_budget_s) / (int(npe) * tr_s)))
        max_possible = min(int(max_blocks), max_by_time)
        if max_possible <= 2:
            schedules.pop(name)
            skipped[name] = {
                "reason": "fixed TR fits <=2 blocks in the scan budget",
                "TR_s": tr_s,
                "Npe": int(npe),
                "time_budget_s": float(time_budget_s),
                "max_blocks_by_time": max_by_time,
                "max_possible_blocks": max_possible,
            }
            print(f"[baseline_e2] Skipping '{name}': TR={tr_s:g}s with "
                  f"Npe={int(npe)} and budget={float(time_budget_s):.1f}s "
                  f"fits only {max_possible} block(s).")
    return skipped


def _cr_timing_constraints_from_env(env_kwargs: dict, *, seed: int) -> dict[str, float]:
    """Read CR timing constraints from the live E2 environment wrapper."""
    env = QalibreMDE2Env(rng_seed=seed, **env_kwargs)
    return env.cr_timing_constraints()


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
                              n_starts=1000, n_refine=10, rng_seed=0,
                              tr_lo_floor=0.0,
                              te_s=0.0,
                              tr_headroom=1.0):
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
        TR_lo_floor  = float(tr_lo_floor),
        TE_s         = float(te_s),
        TR_headroom  = float(tr_headroom),
    )
    TIs = [float(t) for t in res.schedule.TIs]
    TRs = [float(t) for t in res.schedule.TRs]
    n_blocks = int(res.n_blocks)
    L = float(res.schedule.L)
    return TIs, TRs, n_blocks, L


def _cr_optimal_factory(TIs, TRs):
    """Build a step → action callable that cycles through the precomputed
    CR-optimal (TI, TR) schedule. TE = 20 ms, alpha = 90°."""
    n = len(TIs)
    def _sched(step: int) -> np.ndarray:
        k = step % n
        return np.array([TIs[k], 0.020, TRs[k], 90.0], dtype=np.float32)
    return _sched


def make_cr_optimal_alpha_schedule(T1s, budget_s, Npe=8,
                                    n_block_grid=(4, 6, 8, 10, 14, 18),
                                    n_starts=1000, n_refine=10,
                                    tr_lo_floor=0.0,
                                    te_s=0.0,
                                    tr_headroom=1.0):
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
        TR_lo_floor  = float(tr_lo_floor),
        TE_s         = float(te_s),
        TR_headroom  = float(tr_headroom),
    )
    TIs = [float(t) for t in res.schedule.TIs]
    TRs = [float(t) for t in res.schedule.TRs]
    alphas_deg = [float(np.degrees(a)) for a in res.schedule.αs]
    n_blocks = int(res.n_blocks)
    L = float(res.schedule.L)
    return TIs, TRs, alphas_deg, n_blocks, L


def _cr_optimal_alpha_factory(TIs, TRs, alphas_deg):
    """Cycle through the α-aware CR-optimal (TI, TR, α) schedule.
    TE = 20 ms; α is the optimised per-block flip angle."""
    n = len(TIs)
    def _sched(step: int) -> np.ndarray:
        k = step % n
        return np.array([TIs[k], 0.020, TRs[k], alphas_deg[k]],
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
        return np.array([TIs[k], 0.020, TRs[k], alphas_deg[k]],
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
        **_mape_uncertainty(mapes),
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
                       cr_timing: dict[str, float],
                       **env_kwargs) -> dict:
    """Evaluate Formulation B: solve a CR-optimal schedule for each subset.

    This is an oracle lower bound for non-adaptive fixed schedules because it
    sees the episode's true sampled T1 values before choosing the schedule.
    Under `t1_sampler=linear_uniform_range`, those values vary continuously
    episode-to-episode, so this intentionally solves one schedule per distinct
    truth vector rather than reusing a nominal pool schedule.
    """
    env = QalibreMDE2Env(rng_seed=seed_offset, **env_kwargs)
    cr_npe = int(env_kwargs.get("Npe", 32))
    cache: dict[tuple, tuple[list[float], list[float], int, float]] = {}

    def _solve_for_current_subset(info):
        idxs = tuple(int(i) for i in info["sphere_indices"])
        T1s = [float(t) for t in info["T1_true"]]
        # Key by labels and rounded truth values. Nominal/lognormal/linear
        # samplers all flow through this path; continuous T1 tasks generally
        # produce one unique key per episode, which is what a true oracle means.
        key = (idxs, tuple(round(t, 9) for t in T1s))
        if key not in cache:
            cache[key] = make_cr_optimal_schedule(
                T1s,
                budget_s=cr_budget,
                Npe=cr_npe,
                n_block_grid=tuple(cr_block_grid),
                n_starts=cr_starts,
                n_refine=cr_refine,
                **cr_timing,
            )
        return cache[key]

    mapes, per_sphere, times, ep_lens, subset_indices = [], [], [], [], []
    pool_errs = {i: [] for i in range(1, 15)}
    schedules = {}

    for ep in range(n_episodes):
        obs, reset_info = env.reset(seed=seed_offset + ep)
        TIs, TRs, n_blocks, L = _solve_for_current_subset(reset_info)
        sched_fn = _cr_optimal_factory(TIs, TRs)
        idxs = tuple(int(i) for i in reset_info["sphere_indices"])
        T1s_true = [float(t) for t in reset_info["T1_true"]]
        sched_key = json.dumps({
            "sphere_indices": list(idxs),
            "T1_true": [round(t, 9) for t in T1s_true],
        })
        schedules[sched_key] = {"TIs": TIs, "TRs": TRs,
                                "n_blocks": n_blocks, "L": L,
                                "T1s": T1s_true,
                                "timing_constraints": cr_timing}

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
    p.add_argument("--match-run", type=Path, default=None,
                   help="Inherit the env config from this run dir's "
                        "run_config.json so the baseline matches a policy run "
                        "exactly (field/nfe/npe/noise/budget/water/action-mode). "
                        "Avoids hand-repeating flags; guarantees apples-to-apples.")
    p.add_argument("--field",    type=str, default="T3", choices=["T3", "T15"])
    p.add_argument("--max-blocks", type=int, default=15)
    p.add_argument("--time-budget", type=float, default=120.0)
    p.add_argument("--subset-size", type=int, default=None,
                   help="Evaluate schedules on random k-sphere subsets.")
    p.add_argument("--forced-sphere-indices", type=str, default=None,
                   help="Comma-separated 1-based T1-pool labels active every "
                        "episode, e.g. 1,3,6,8,14.")
    p.add_argument("--t1-sampler", type=str, default=None,
                   choices=["lognormal", "linear_uniform_range"],
                   help="Override matched/default T1 material sampler.")
    p.add_argument("--pose-mode", type=str, default=None,
                   choices=["auto", "fixed", "inplane_jitter"],
                   help="Override matched/default pose mode.")
    p.add_argument("--translation-sigma-mm", type=float, default=None)
    p.add_argument("--rotation-sigma-rad", type=float, default=None)
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
    p.add_argument("--use-gpu", action="store_true",
                   help="Run KomaMRI's Bloch simulation on the GPU "
                        "(requires a CUDA-enabled Julia runtime/project).")
    p.add_argument("--cpu", action="store_true",
                   help="Force CPU simulation, overriding a --match-run's saved "
                        "use_gpu (e.g. on a CPU-only machine).")
    p.add_argument("--log-grid-tr", type=float, default=_LOG_GRID_TR,
                   help="TR (s) for the log_grid_trmatched baseline. Default "
                        f"{_LOG_GRID_TR}. Set to the policy's realised mean TR "
                        "(e.g. 1.24) to equalise block count and isolate the "
                        "TI-placement gain from the block-count gain.")
    p.add_argument("--fix-te", action="store_true",
                   help="Match the RL action mode: pin TE=20ms (TI,TR[,α] only). "
                        "Required for an apples-to-apples comparison with a "
                        "policy trained with --fix-te.")
    p.add_argument("--learn-alpha", action="store_true",
                   help="Match the RL action mode: expose α (requires --fix-te). "
                        "Schedules still set their own α; only the action layout "
                        "changes so it decodes correctly.")
    p.add_argument("--log-ti-action", action="store_true",
                   help="Match the RL action mode: log-spaced TI action axis. "
                        "Schedules are unaffected (they pass physical TIs); the "
                        "env just decodes them on the correct axis.")
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
    p.add_argument("--roi-radius", type=int, default=0,
                   help="Square ROI half-width for per-sphere signal extraction. "
                        "0 = centre pixel; 1 = 3x3 mean. Can be overlaid on "
                        "--match-run for eval-only ablations.")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    schedules = {} if args.cr_only else dict(SCHEDULES)
    # Per-schedule step cap. One-shot CR designs run exactly their n_blocks
    # (run-once); conventional repeating protocols stay uncapped (cycle-to-fill).
    schedule_max_steps: dict[str, int] = {}
    if args.match_run is not None:
        from e2_config import load_run_env_kwargs
        env_kwargs = load_run_env_kwargs(args.match_run)
        # Overlay baseline-only diagnostic knobs (not in a training config), and
        # let an explicit --subset-size override (for oracle/subset evals).
        env_kwargs.update(oracle_fit=args.oracle_fit, oracle_band=args.oracle_band,
                          fitter_n_grid=args.fitter_n_grid,
                          roi_radius=args.roi_radius)
        if args.subset_size is not None:
            env_kwargs["subset_size"] = args.subset_size
        if args.forced_sphere_indices is not None:
            env_kwargs["forced_sphere_indices"] = parse_int_csv(args.forced_sphere_indices)
        if args.t1_sampler is not None:
            env_kwargs["t1_sampler"] = args.t1_sampler
        if args.pose_mode is not None:
            env_kwargs["pose_mode"] = args.pose_mode
        if args.translation_sigma_mm is not None:
            env_kwargs["translation_sigma_mm"] = args.translation_sigma_mm
        if args.rotation_sigma_rad is not None:
            env_kwargs["rotation_sigma_rad"] = args.rotation_sigma_rad
        if args.use_gpu:
            env_kwargs["use_gpu"] = True
        print(f"[baseline_e2] env config loaded from "
              f"{args.match_run}/run_config.json")
    else:
        env_kwargs = dict(
            cfg_field=args.field,
            Nfe=args.nfe,
            Npe=args.npe,
            use_gpu=args.use_gpu,
            max_blocks=args.max_blocks,
            time_budget_s=args.time_budget,
            subset_size=args.subset_size,
            forced_sphere_indices=parse_int_csv(args.forced_sphere_indices),
            t1_sampler=args.t1_sampler or "lognormal",
            pose_mode=args.pose_mode or "auto",
            translation_sigma_mm=(
                5.0 if args.translation_sigma_mm is None
                else args.translation_sigma_mm),
            rotation_sigma_rad=(
                0.15 if args.rotation_sigma_rad is None
                else args.rotation_sigma_rad),
            noise_sigma_abs=args.noise,
            phase_sensitive=args.phase_sensitive,
            sigma_method=args.sigma_method,
            oracle_fit=args.oracle_fit,
            oracle_band=args.oracle_band,
            fitter_n_grid=args.fitter_n_grid,
            include_image=args.include_image,
            include_sigma=args.include_sigma,
            roi_radius=args.roi_radius,
            fix_te=args.fix_te,
            learn_alpha=args.learn_alpha,
            log_ti_action=args.log_ti_action,
        )
    if args.cpu:
        env_kwargs["use_gpu"] = False
    # CR solver budget defaults to the env's scan-time budget (matched or args).
    cr_budget = (env_kwargs["time_budget_s"] if args.cr_budget is None
                 else args.cr_budget)
    # Field/Npe for the CR fleet+solver: take from the (possibly matched) env
    # config so they can't disagree with what's being evaluated.
    field = env_kwargs.get("cfg_field", args.field)
    npe = int(env_kwargs.get("Npe", args.npe))
    if not args.cr_only:
        budget_s = float(env_kwargs["time_budget_s"])
        schedules["log_grid_trmatched"] = make_log_grid_trmatched(
            budget_s, npe, args.log_grid_tr)
        n_trmatched = max(1, int(np.floor(budget_s / (npe * args.log_grid_tr))))
        print(f"[baseline_e2] log_grid_trmatched: {n_trmatched} TI points "
              f"for budget={budget_s:.0f}s, Npe={npe}, TR={args.log_grid_tr}s")
    cr_timing = _cr_timing_constraints_from_env(env_kwargs, seed=args.seed)
    print("[baseline_e2] CR timing constraints from E2 env: "
          f"TR_lo_floor={cr_timing['tr_lo_floor']:.3f}s, "
          f"TE={cr_timing['te_s']:.3f}s, "
          f"TR_headroom={cr_timing['tr_headroom']:.2f}")
    skipped_schedules = _skip_too_short_long_tr_schedules(
        schedules,
        time_budget_s=float(env_kwargs["time_budget_s"]),
        max_blocks=int(env_kwargs["max_blocks"]),
        npe=npe,
    )

    # 14-sphere nominal fleet for the chosen field, from Julia's T1_ARRAY.
    T1s_fleet = nominal_fleet_t1s(field)
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
                T1s_fleet, budget_s=cr_budget, Npe=npe,
                n_block_grid=tuple(args.cr_block_grid),
                n_starts=args.cr_starts,
                n_refine=args.cr_refine,
                **cr_timing)
        print(f"  best n_blocks = {n_blocks}, L = {L:.4f}")
        print(f"  TIs (sorted)  = {sorted(round(t, 4) for t in TIs)}")
        print(f"  TRs (sorted)  = {sorted(round(t, 4) for t in TRs)}")
        schedules["cr_optimal"] = _cr_optimal_factory(TIs, TRs)
        schedule_max_steps["cr_optimal"] = n_blocks
        # Persist the CR-opt schedule for reproducibility
        with (args.out / "cr_optimal_schedule.json").open("w") as f:
            json.dump({"TIs": TIs, "TRs": TRs, "n_blocks": n_blocks, "L": L,
                       "T1s": T1s_fleet, "budget_s": cr_budget,
                       "timing_constraints": cr_timing,
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
            T1s_fleet, budget_s=cr_budget, Npe=npe,
            n_block_grid=tuple(args.cr_block_grid),
            n_starts=args.cr_starts, n_refine=args.cr_refine,
            **cr_timing)
        print(f"  best n_blocks = {n_blocks_a}, L = {L_a:.4f}")
        print(f"  α (deg, sorted) = {sorted(round(a, 1) for a in alphas_a)}")
        schedules["cr_optimal_alpha"] = _cr_optimal_alpha_factory(
            TIs_a, TRs_a, alphas_a)
        schedule_max_steps["cr_optimal_alpha"] = n_blocks_a
        with (args.out / "cr_optimal_alpha_schedule.json").open("w") as f:
            json.dump({"TIs": TIs_a, "TRs": TRs_a, "alphas_deg": alphas_a,
                       "n_blocks": n_blocks_a, "L": L_a, "T1s": T1s_fleet,
                       "budget_s": cr_budget,
                       "timing_constraints": cr_timing,
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
        ci = r["mape_ci95_pct"]
        print(f"  MAPE       = {r['mape_pct']:.2f}%  "
              f"(95% CI {ci[0]:.2f}–{ci[1]:.2f}%, SEM ±{r['mape_sem_pct']:.2f}%)")
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
            args.cr_starts, args.cr_refine, cr_timing, **env_kwargs)
        results["cr_oracle"] = r
        print(f"  MAPE       = {r['mape_pct']:.2f}%")
        print(f"  p90 MAPE   = {r['mape_p90_pct']:.2f}%")
        print(f"  Success<5% = {r['success_5pct']:.1%}")
        print(f"  Mean time  = {r['mean_scan_time_s']:.1f}s "
              f"({r['mean_ep_len']:.1f} blocks)")
        print(f"  Unique oracle schedules solved = {r['n_unique_subsets_solved']}")
        with (args.out / "cr_oracle_schedules.json").open("w") as f:
            json.dump(r["schedules_by_subset"], f, indent=2)

    out_path = args.out / "baseline_summary.json"
    # Record the exact env config so the run is reproducible and can be diffed
    # against a policy's run_config.json["base_env_kwargs"] to confirm an
    # apples-to-apples comparison (same noise, field, resolution, budget, …).
    # Stored under "_run_meta" so per-schedule entries stay top-level (consumers
    # iterate them); the leading "_" marks it as non-schedule metadata.
    results["_run_meta"] = {
        "env_config": {k: (str(v) if isinstance(v, Path) else v)
                       for k, v in env_kwargs.items()},
        "episodes": args.episodes,
        "seed": args.seed,
        "skipped_schedules": skipped_schedules,
    }
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[baseline_e2] Summary → {out_path}")


if __name__ == "__main__":
    main()

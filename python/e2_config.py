"""Shared E2 environment CLI flags + kwargs builder.

Both `train_e2.py` and `bench_e2.py` register the *same* env-construction flags
via `add_e2_env_args` and build the `QalibreMDE2Env` kwargs via `e2_env_kwargs`,
so a benchmark command is the training command with the script name swapped.

Training-only flags (timesteps, eval cadence, checkpoints, …) stay in the
individual scripts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_run_env_kwargs(run_dir) -> dict:
    """Load the saved QalibreMDE2Env constructor kwargs from a run's
    run_config.json. Supports both trainers: train_e2 writes them under
    "env_kwargs", train_e2_mf under "base_env_kwargs" (the full-fidelity target
    config, before per-stage fidelity overrides). Use this so eval/baseline runs
    inherit the exact env config and can't drift from the training command."""
    run_dir = Path(run_dir)
    cfg_path = run_dir / "run_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"No run_config.json in {run_dir} — cannot load env config.")
    cfg = json.loads(cfg_path.read_text())
    kw = cfg.get("base_env_kwargs") or cfg.get("env_kwargs")
    if kw is None:
        raise ValueError(
            f"{cfg_path} has neither 'base_env_kwargs' nor 'env_kwargs'.")
    return dict(kw)


def add_e2_env_args(p: argparse.ArgumentParser) -> None:
    """Register every flag that maps onto a QalibreMDE2Env constructor kwarg."""
    p.add_argument("--field", type=str, default="T3", choices=["T3", "T15"])
    p.add_argument("--max-blocks", type=int, default=15)
    p.add_argument("--time-budget", type=float, default=120.0,
                   help="Episode scan-time budget in seconds.")
    p.add_argument("--subset-size", type=int, default=None,
                   help="Draw this many T1 spheres without replacement per "
                        "episode. Omit for the full 14-sphere plate.")
    p.add_argument("--noise", type=float, default=50.0,
                   help="Absolute complex-Gaussian σ on k-space (FIX_SIM_PLAN §2). "
                        "Default σ*=50 → NEMA dual-acq SNR ≈ 25 (E2_RERUN_PLAN §3.1).")

    # ── resolution / phantom perf knobs (E2_RERUN_PLAN §6.1, run-time plan) ──
    # These are the dominant simulate-cost levers: per-step cost ≈ Npe·TR·n_spins
    # plus Nfe ADC samples/shot. Defaults match the env defaults (64×32, 1 mm).
    p.add_argument("--Nfe", type=int, default=64,
                   help="Frequency-encode samples (readout). Cost ∝ Nfe ADC/shot.")
    p.add_argument("--Npe", type=int, default=32,
                   help="Phase-encode shots. Dominant cost lever: sim integrates "
                        "≈ Npe·TR seconds of Bloch dynamics per step.")
    p.add_argument("--voxel-mm", type=float, default=1.0,
                   help="Phantom voxel size [mm]. Smaller = more spins (∝ 1/voxel² "
                        "in-slab); 1 mm matches the σ*=50 SNR calibration.")
    p.add_argument("--water-voxel-mm", type=float, default=None,
                   help="Optional background-water-only voxel size [mm]. Omit to "
                        "use --voxel-mm for water as well. Coarser water keeps "
                        "sphere voxels unchanged while reducing water spin count.")
    p.add_argument("--use-gpu", action="store_true",
                   help="Run KomaMRI's Bloch simulation on the GPU "
                        "(sim_params['gpu']=true). Requires a CUDA backend loaded "
                        "in the Julia runtime; no-op fallback to CPU otherwise.")

    p.add_argument("--reward-mode", type=str, default="neg_mape",
                   choices=["neg_mape", "delta_mape", "neg_log_mape",
                            "delta_log_mape", "terminal_only"],
                   help="neg_mape (legacy level) | delta_mape (per-step progress) "
                        "| neg_log_mape (log-level, sub-1%% gradient) | "
                        "delta_log_mape (log-ratio progress) | terminal_only "
                        "(−final MAPE at episode end only)")
    p.add_argument("--time-penalty", type=float, default=0.0,
                   help="λ: subtract λ·(block_time/budget) from the per-step "
                        "reward, on top of any --reward-mode. Only applied with "
                        "--allow-stop (under a fixed budget total time ≈ budget, "
                        "so the term is a near-constant offset). Sweep to trace "
                        "the accuracy-vs-time Pareto. Default 0.0.")
    p.add_argument("--allow-stop", action="store_true",
                   help="Expose a learned STOP decision: the action gains a stop "
                        "gate (last dim) and the agent ends the episode when "
                        "another block's accuracy gain no longer beats its scan "
                        "time (pair with --time-penalty). Default off (fixed "
                        "budget). Keep --max-blocks/--time-budget high as a safety "
                        "cap so episodes stay bounded.")
    p.add_argument("--log-ti-action", action="store_true",
                   help="Log-spaced TI action mapping (constant density per "
                        "decade). See EXPERT_REPORT_TRAC §9.2.")
    p.add_argument("--simplified-action", action="store_true",
                   help="3-dim action [TI, TE, TR]; fixes α_exc=90°")
    p.add_argument("--fix-te", action="store_true",
                   help="Fix TE=20ms and expose [TI, TR] (2-dim) — the Run A0 "
                        "'without α' ablation (ALPHA_DOF.md, action-space ablation). With "
                        "--learn-alpha, exposes [TI, TR, α] (3-dim).")
    p.add_argument("--learn-alpha", action="store_true",
                   help="Add the excitation flip angle α∈[5°,90°] as a learned "
                        "action dim. Requires --fix-te (Run A).")
    p.add_argument("--terminal-bonus", type=float, default=0.0,
                   help="Set to 0.0 to disable (E1-style degenerate-policy "
                        "driver — see EXPERT_REPORT §15)")
    p.add_argument("--mape-alpha", type=float, default=1.0,
                   help="MAPE aggregation: α·mean + (1−α)·max. "
                        "1.0 = legacy mean; 0.5 = §16.4 Option A")
    p.add_argument("--phase-sensitive", action="store_true",
                   help="Use signed real-part image reconstruction instead "
                        "of magnitude (cr_explainer.md §14, EXPERT_REPORT_E2_4 "
                        "§15).")
    p.add_argument("--sigma-method", type=str, default="bootstrap",
                   choices=["asymptotic", "profile_likelihood", "bootstrap"],
                   help="σ_T1 estimation method (E2_5_PLAN.md §3 / §15).")
    p.add_argument("--include-image", action="store_true",
                   help="Prepend the flattened recon image (Nfe*Npe) to the "
                        "observation (E2_RERUN_PLAN §6.2). Default off.")
    p.add_argument("--include-sigma", action="store_true",
                   help="Append the per-sphere fitter-σ channel to the "
                        "observation (E2_RERUN_PLAN §6.3). Default off.")
    p.add_argument("--no-water", dest="include_water", action="store_false",
                   help="Build the phantom WITHOUT background-water spins "
                        "(spheres only). Default keeps water. Mainly a bench "
                        "sweep knob — water dominates the spin count / sim cost.")
    p.add_argument("--water-model", type=str, default="bloch",
                   choices=["bloch", "cached_perline"],
                   help="Water simulation model (src/water_cache.jl). bloch (default) "
                        "full-Bloch sims water every step; cached_perline Bloch-sims "
                        "only the spheres and adds water from a cached Koma template "
                        "rescaled per k-line (~8× per-step, T1-grid-floor accurate). "
                        "Requires water; cache scope follows --include-image.")
    p.add_argument("--forward-model", type=str, default="bloch",
                   choices=["bloch", "analytic"],
                   help="bloch (default) full KomaMRI Bloch sim + 2D recon every "
                        "step; analytic skips Koma and synthesises per-sphere "
                        "signals from transient_mz_at_excite_npe (~µs/step, fits "
                        "noise-limited only). Fast reward SCREENING surrogate — "
                        "T1-only obs, not for absolute MAPE.")
    p.add_argument("--analytic-noise", type=float, default=0.04,
                   help="Signal-space σ for --forward-model analytic (signal scale "
                        "O(1); default 0.04 ≈ SNR 25 at the reference operating "
                        "point). Sweep to probe noise robustness.")


def e2_env_kwargs(args: argparse.Namespace) -> dict:
    """Build the QalibreMDE2Env constructor kwargs from parsed args.

    Mirrors the dict previously inlined in train_e2.main(); the only additions
    are the resolution/voxel/gpu perf knobs from `add_e2_env_args`.
    """
    return dict(
        cfg_field=args.field,
        Nfe=args.Nfe,
        Npe=args.Npe,
        voxel_size_mm=args.voxel_mm,
        water_voxel_size_mm=args.water_voxel_mm,
        use_gpu=args.use_gpu,
        max_blocks=args.max_blocks,
        time_budget_s=args.time_budget,
        subset_size=args.subset_size,
        noise_sigma_abs=args.noise,
        reward_mode=args.reward_mode,
        time_penalty_coef=args.time_penalty,
        allow_stop=args.allow_stop,
        forward_model=args.forward_model,
        analytic_noise_sigma=args.analytic_noise,
        simplified_action=args.simplified_action,
        fix_te=args.fix_te,
        learn_alpha=args.learn_alpha,
        log_ti_action=args.log_ti_action,
        terminal_bonus=args.terminal_bonus,
        mape_alpha=args.mape_alpha,
        phase_sensitive=args.phase_sensitive,
        sigma_method=args.sigma_method,
        include_image=args.include_image,
        include_sigma=args.include_sigma,
        include_water=args.include_water,
        water_model=args.water_model,
    )

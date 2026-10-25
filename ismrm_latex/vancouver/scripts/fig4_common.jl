# Shared setup for the Fig 4 Julia scripts (fig4_replay.jl, fig4_pixelfit.jl).

using MRISystemPhantom, KomaMRI

const ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
include(joinpath(ROOT, "julia", "rl_boot.jl"))

using JSON, NPZ, Printf

const RUNS = joinpath(ROOT, "runs", "e2")
const DATA_DIR = joinpath(@__DIR__, "..", "figures", "fig4_data")
const RL_RUNS = (hist = "mf_runB_5sphere_hist_560s_gpu", nomem = "mf_runB_5sphere_560s_gpu")
# baselines_heldout_560s/baseline_heldout.log: the baselines used the hist run's env config.
const BASELINE_RUN = RL_RUNS.hist

# python/qalibremd_gym/env_e2.py defaults for E2Env kwargs absent from run_config.json.
const WRAPPER_DEFAULTS = Dict{Symbol,Any}(:FOV => 0.2, :success_tol => 0.05, :T1_sigma_rel => 0.05,
                                          :oracle_fit => false, :oracle_band => 2.0, :fitter_n_grid => 200)
# Action-layout flags the Python wrapper consumes itself (TE fixed at 20 ms, alpha 90 deg, log-TI axis).
const WRAPPER_ONLY = ("simplified_action", "fix_te", "learn_alpha", "log_ti_action")
const SYMBOL_KEYS = ("cfg_field", "t1_sampler", "pose_mode", "reward_mode", "sigma_method",
                     "water_model", "forward_model")

"""E2Env built as python/qalibremd_gym/env_e2.py:186-231 builds it from a run's config, on CPU."""
function env_from_run(run::AbstractString)
    cfg = JSON.parsefile(joinpath(RUNS, run, "run_config.json"))["base_env_kwargs"]
    kw = copy(WRAPPER_DEFAULTS)
    for (k, v) in cfg
        (k in WRAPPER_ONLY || v === nothing) && continue
        kw[Symbol(k)] = k in SYMBOL_KEYS ? Symbol(v) :
                        k == "forced_sphere_indices" ? Int.(v) : v
    end
    kw[:use_gpu] = false
    E2Env(; kw...)
end

# Scan tags: "k15" is validation scan 15 (seed 500015), "heldout_k07" held-out scan 7 (seed 600007),
# "heldout_k07_bestgrid" the same scan with the log TI grid row run as best_log_grid_schedule.
scan_tag(k::Int; heldout::Bool = false, best_grid::Bool = false) =
    (heldout ? "heldout_" : "") * @sprintf("k%02d", k) * (best_grid ? "_bestgrid" : "")

"""Command line of fig4_replay.jl and fig4_pixelfit.jl: [--heldout [--best-grid]] [scan numbers]."""
function scan_args(args)
    flags = filter(startswith("--"), args)
    flags ⊆ ("--heldout", "--best-grid") || error("unknown option among $flags")
    heldout, best_grid = "--heldout" in flags, "--best-grid" in flags
    best_grid && !heldout && error("--best-grid replaces the held-out log TI grid row; add --heldout")
    (heldout = heldout, best_grid = best_grid, nums = parse.(Int, filter(!startswith("--"), args)))
end
replay_path(tag::AbstractString) = joinpath(DATA_DIR, "fig4_replay_$(tag).npz")
maps_path(tag::AbstractString) = joinpath(DATA_DIR, "fig4_maps_$(tag).npz")

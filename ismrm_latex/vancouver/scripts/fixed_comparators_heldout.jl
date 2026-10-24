# Per-scan held-out results of the two fixed comparators, which
# runs/e2/baselines_heldout_560s and .../mf_runB_cached3_cached_full3_full_gpu/baselines_heldout
# only stored as aggregates (python/baseline_e2.py:evaluate):
#   log_grid_trmatched  log-spaced TIs over 0.05-3.0 s, TR 1.7 s, cycled until the budget
#                       guard rejects a block (make_log_grid_trmatched, baseline_e2.py:86-101);
#   cr_optimal          the saved Cramer-Rao schedule, run once (max_steps = n_blocks).
# Each scan replays the baseline's held-out seed (600000 + k) through the same E2Env config,
# and the script stops unless the replayed means reproduce baseline_summary.json.
#
# The baselines handed physical actions through the Python wrapper's physical_to_norm_action
# and _denorm_action (python/qalibremd_gym/env_e2.py:310-386), which moves TI and TR by about
# one float32 ulp; python_action() mirrors that round trip. Python floats meeting float32
# scalars promote to float64 under numpy 1 but stay float32 under numpy 2 (NEP 50), so the
# promotion rules are an argument.
#
# Run from the RLxMRI repo root:
#   julia -t 8 --project=. ismrm_latex/vancouver/scripts/fixed_comparators_heldout.jl [numpy1|numpy2]
#
# Writes figures/fig5_v2_data/fixed_comparators_heldout.json.

include(joinpath(@__DIR__, "fig4_common.jl"))
using Statistics

const HELDOUT_SEED = 600_000
const OUT = normpath(joinpath(@__DIR__, "..", "figures", "fig5_v2_data", "fixed_comparators_heldout.json"))
const RUN_240 = "mf_runB_cached3_cached_full3_full_gpu"
const BUDGETS = (
    (budget_s = 560, run = BASELINE_RUN, dir = joinpath(RUNS, "baselines_heldout_560s")),
    (budget_s = 240, run = RUN_240, dir = joinpath(RUNS, RUN_240, "baselines_heldout")),
)
# python/baseline_e2.py:_LOG_GRID_TR and qalibremd_gym/schedules.py:log_ti_grid defaults.
const LOG_GRID_TR_S = 1.7
const LOG_GRID_TI_RANGE_S = (0.05, 3.0)
# env_e2.py: _FIXED_TE_S, _ALPHA_LO_DEG, _ALPHA_HI_DEG; the schedules ask for alpha = 90 deg.
const TE_S = 0.020
const ALPHA_LO_DEG, ALPHA_HI_DEG = 5.0, 90.0
const ALPHA_DEG = 90.0
# A replayed mean that differs by more than this cannot be the recorded run.
const MATCH_TOL_PCT = 1e-6

"""Physical action after the wrapper's physical -> normalised -> physical round trip (fix_te mode)."""
function python_action(env::E2Env, ti, tr; learn_alpha::Bool, numpy2::Bool)
    lo, hi = Float32.(e2_action_lo(env)), Float32.(e2_action_hi(env))
    # The schedule functions build a float32 array; physical_to_norm_action receives float(phys[i]).
    phys = Float32[ti, TE_S, tr, ALPHA_DEG]
    ti_s, tr_s, alpha_s = Float64(phys[1]), Float64(phys[3]), Float64(phys[4])
    ti_s = clamp(ti_s, Float64(lo[1]), Float64(hi[1]))
    u_ti = numpy2 ? log(Float32(ti_s) / lo[1]) / log(hi[1] / lo[1]) :
                    log(ti_s / Float64(lo[1])) / Float64(log(hi[1] / lo[1]))
    u_tr = numpy2 ? (Float32(tr_s) - lo[3]) / (hi[3] - lo[3]) :
                    (tr_s - Float64(lo[3])) / Float64(hi[3] - lo[3])
    u = Float32[u_ti, u_tr]
    learn_alpha && push!(u, Float32((alpha_s - ALPHA_LO_DEG) / (ALPHA_HI_DEG - ALPHA_LO_DEG)))
    norm = 2.0f0 .* clamp.(u, 0.0f0, 1.0f0) .- 1.0f0

    v = (clamp.(norm, -1.0f0, 1.0f0) .+ 1.0f0) ./ 2.0f0
    ti_out = Float32(Float64(lo[1]) * (Float64(hi[1]) / Float64(lo[1]))^Float64(v[1]))
    tr_out = lo[3] + v[2] * (hi[3] - lo[3])
    alpha_out = !learn_alpha ? Float32(ALPHA_DEG) :
                numpy2 ? Float32(ALPHA_LO_DEG) + v[3] * Float32(ALPHA_HI_DEG - ALPHA_LO_DEG) :
                         Float32(ALPHA_LO_DEG + Float64(v[3]) * (ALPHA_HI_DEG - ALPHA_LO_DEG))
    Float64[ti_out, Float32(TE_S), tr_out, alpha_out]
end

function log_grid_schedule(budget_s::Real, npe::Integer)
    n = max(1, floor(Int, budget_s / (npe * LOG_GRID_TR_S)))
    tis = exp.(range(log(LOG_GRID_TI_RANGE_S[1]), log(LOG_GRID_TI_RANGE_S[2]); length = n))
    (TIs = collect(tis), TRs = fill(LOG_GRID_TR_S, n), max_steps = nothing)
end

function cr_schedule(dir::AbstractString)
    s = JSON.parsefile(joinpath(dir, "cr_optimal_schedule.json"))
    (TIs = Float64.(s["TIs"]), TRs = Float64.(s["TRs"]), max_steps = Int(s["n_blocks"]))
end

"""One episode as baseline_e2.py:evaluate runs it: cycle the schedule until done or max_steps."""
function run_episode(env::E2Env, seed::Integer, sched; learn_alpha::Bool, numpy2::Bool)
    e2_reset!(env; rng_seed = seed)
    n = length(sched.TIs)
    step, done, info = 0, false, Dict{String,Any}()
    executed = Vector{Float64}[]
    while !done
        a = python_action(env, sched.TIs[step % n + 1], sched.TRs[step % n + 1];
                          learn_alpha = learn_alpha, numpy2 = numpy2)
        _, _, done, info = e2_step!(env, a)
        get(info, "budget_exceeded", false) || push!(executed, [info["TI"], info["TR"], info["alpha_deg"]])
        step += 1
        sched.max_steps !== nothing && step >= sched.max_steps && break
    end
    Dict{String,Any}("seed" => seed, "mape" => info["mape"],
                     "ape" => abs.(env.T1_est .- env.T1_true) ./ env.T1_true,
                     "T1_true" => copy(env.T1_true), "T1_est" => copy(env.T1_est),
                     "n_blocks" => env.n_blocks, "time_s" => env.time_used_s, "ep_len" => step,
                     "executed_TI_TR_alpha" => executed)
end

function check_env_config(env::E2Env, cfg::AbstractDict)
    bad = String[]
    for (k, v) in cfg
        (k == "use_gpu" || v === nothing || !hasproperty(env, Symbol(k))) && continue
        got = getproperty(env, Symbol(k))
        want = got isa Symbol ? Symbol(v) : v
        isequal(got, want) || (got isa AbstractVector && isequal(collect(got), Int.(want))) ||
            push!(bad, "$k: env $got, baseline $v")
    end
    isempty(bad) || error("E2Env differs from the baseline run's env_config:\n" * join(bad, "\n"))
end

function compare(name::AbstractString, scans, rec::AbstractDict)
    mape = [s["mape"] for s in scans]
    ape = reduce(hcat, [s["ape"] for s in scans])
    replayed = Dict("mape_pct" => 100 * mean(mape), "success_5pct" => mean(mape .< 0.05),
                    "mean_scan_time_s" => mean(s["time_s"] for s in scans),
                    "mean_ep_len" => mean(s["ep_len"] for s in scans),
                    "per_sphere_mape_pct" => vec(100 * mean(ape; dims = 2)))
    diffs = Dict(k => maximum(abs.(replayed[k] .- rec[k])) for k in keys(replayed))
    @printf("%-20s replayed MAPE %.12f%% recorded %.12f%% | max |diff|: MAPE %.2g pp, per-sphere %.2g pp, time %.2g s, steps %.2g, success %.2g\n",
            name, replayed["mape_pct"], rec["mape_pct"], diffs["mape_pct"], diffs["per_sphere_mape_pct"],
            diffs["mean_scan_time_s"], diffs["mean_ep_len"], diffs["success_5pct"])
    replayed, diffs
end

function main(args)
    rules = isempty(args) ? "numpy2" : args[1]
    rules in ("numpy1", "numpy2") || error("promotion rules must be numpy1 or numpy2, got $rules")
    numpy2 = rules == "numpy2"
    results = Dict{String,Any}()
    for b in BUDGETS
        summary = JSON.parsefile(joinpath(b.dir, "baseline_summary.json"))
        meta = summary["_run_meta"]
        cfg = meta["env_config"]
        cfg["time_budget_s"] == b.budget_s || error("$(b.dir) is not the $(b.budget_s) s baseline")
        meta["seed"] == HELDOUT_SEED || error("$(b.dir) did not run on the held-out seeds")
        env = env_from_run(b.run)
        check_env_config(env, cfg)
        learn_alpha = cfg["learn_alpha"]
        schedules = Dict("log_grid_trmatched" => log_grid_schedule(env.time_budget_s, env.Npe),
                         "cr_optimal" => cr_schedule(b.dir))
        out = Dict{String,Any}("baseline_dir" => relpath(b.dir, ROOT), "env_run" => b.run,
                               "learn_alpha_action_layout" => learn_alpha)
        for name in ("log_grid_trmatched", "cr_optimal")
            sched = schedules[name]
            t0 = time()
            scans = [run_episode(env, meta["seed"] + k, sched; learn_alpha = learn_alpha, numpy2 = numpy2)
                     for k in 0:(meta["episodes"] - 1)]
            replayed, diffs = compare("$(b.budget_s) s $name", scans, summary[name])
            out[name] = Dict("scans" => scans, "replayed" => replayed, "max_abs_diff" => diffs,
                             "recorded" => Dict(k => summary[name][k] for k in keys(replayed)),
                             "schedule" => Dict("TI_s" => sched.TIs, "TR_s" => sched.TRs,
                                                "max_steps" => sched.max_steps),
                             "wall_s" => time() - t0)
        end
        results["$(b.budget_s)"] = out
    end

    worst = maximum(results[b][n]["max_abs_diff"]["mape_pct"] for b in keys(results)
                    for n in ("log_grid_trmatched", "cr_optimal"))
    worst <= MATCH_TOL_PCT ||
        error(@sprintf("replayed means differ from baseline_summary.json by up to %.3g pp; not written", worst))
    mkpath(dirname(OUT))
    open(OUT, "w") do io
        JSON.print(io, Dict("seed_base" => HELDOUT_SEED, "numpy_promotion" => rules,
                            "julia_version" => string(VERSION), "threads" => Threads.nthreads(),
                            "budgets" => results))
    end
    println("wrote ", OUT)
end

main(ARGS)

# The two fixed comparator schedules of python/baseline_e2.py, replayed through an E2Env
# exactly as baseline_e2.py:evaluate runs them:
#   log_grid_trmatched  log-spaced TIs over 0.05-3.0 s, TR 1.7 s, cycled until the budget
#                       guard rejects a block (make_log_grid_trmatched, baseline_e2.py:86-101);
#   cr_optimal          the saved Cramer-Rao schedule, run once (max_steps = n_blocks).
#
# The baselines handed physical actions through the Python wrapper's physical_to_norm_action
# and _denorm_action (python/qalibremd_gym/env_e2.py:310-386), which moves TI and TR by about
# one float32 ulp; python_action() mirrors that round trip. Python floats meeting float32
# scalars promote to float64 under numpy 1 but stay float32 under numpy 2 (NEP 50), so the
# promotion rules are an argument.
#
# Needs the E2Env API in scope (julia/rl_boot.jl); fixed_comparators_heldout.jl and
# control_narrow_band.jl include it.

using JSON

# python/baseline_e2.py:_LOG_GRID_TR and qalibremd_gym/schedules.py:log_ti_grid defaults.
const LOG_GRID_TR_S = 1.7
const LOG_GRID_TI_RANGE_S = (0.05, 3.0)
# env_e2.py: _FIXED_TE_S, _ALPHA_LO_DEG, _ALPHA_HI_DEG; the schedules ask for alpha = 90 deg.
const TE_S = 0.020
const ALPHA_LO_DEG, ALPHA_HI_DEG = 5.0, 90.0
const ALPHA_DEG = 90.0

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

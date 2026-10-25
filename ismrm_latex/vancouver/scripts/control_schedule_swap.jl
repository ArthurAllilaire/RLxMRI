# Control for the ISMRM abstract: does the TI-histogram policy's held-out T1 accuracy
# come from adapting the schedule to each scan, or from the shape of its schedules?
#
# 1. Roll out global_best on the held-out seeds 600000-600023 (eval_e2.py's seeds) and
#    stop unless its per-sphere errors match global_best/eval_t1err_heldout.json.
# 2. With no policy in the loop, replay executed (TI, TR) schedules through the same
#    E2Env, fitter and per-seed noise stream on each held-out phantom k:
#      identity  phantom k gets its own schedule (must reproduce step 1 exactly);
#      swap_s    phantom k gets scan (k + s) mod 24's schedule, s = 1, 8, 16;
#      fixed_median  per block index, the median log TI and median TR over the 24
#                adaptive schedules, truncated to fit the budget, on every phantom;
#      fixed_median_validation  the same recipe applied to the policy's schedules on
#                the 24 validation scans (seed 500000), so no held-out data shapes it.
#
# Run from the RLxMRI repo root, then summarise with control_summary.py:
#   julia -t 8 --project=. ismrm_latex/vancouver/scripts/control_schedule_swap.jl
#
# Writes figures/control_data/control_adaptive_heldout.json (full adaptive rollouts) and
# figures/control_data/control_results.json (per-phantom results of every arm).

include(joinpath(@__DIR__, "fig3evo_rollout.jl"))
using Statistics

const HELDOUT_SEED = 600_000
const N_SCANS = 24
const SHIFTS = (1, 8, 16)
const CONTROL_DATA = joinpath(FIGURES, "control_data")
const MATCH_TOL = 1e-9

executed_schedule(ep) = [(Float64(b["TI"]), Float64(b["TR"])) for b in ep["blocks"] if b["executed"]]

function validation_schedules(env::E2Env, run::AbstractString)
    map(JSON.parsefile(joinpath(run, "global_best", "diagnostics", "episodes.json"))) do ep
        s = [(Float64(ti), Float64(tr)) for (ti, tr) in zip(ep["TI"], ep["TR"])]
        # diagnose_e2.py also records the final proposal the budget guard rejected.
        env.Npe * sum(last.(s)) > env.time_budget_s ? s[1:end-1] : s
    end
end

function final_result(env::E2Env, mape::Real)
    Dict{String,Any}("mape" => mape, "ape" => abs.(env.T1_est .- env.T1_true) ./ env.T1_true,
                     "T1_true" => copy(env.T1_true), "T1_est" => copy(env.T1_est),
                     "n_blocks" => env.n_blocks, "time_s" => env.time_used_s)
end

function replay(env::E2Env, seed::Integer, schedule)
    e2_reset!(env; rng_seed = seed)
    mape = NaN
    for (b, (ti, tr)) in enumerate(schedule)
        env.done && error("seed $seed: episode ended after $(b - 1) of $(length(schedule)) blocks")
        # TE and alpha as decode() passes them, so a replayed block is the executed block.
        _, _, _, info = e2_step!(env, Float64[ti, Float32(FIXED_TE_S), tr, Float32(FIXED_ALPHA_DEG)])
        get(info, "budget_exceeded", false) && error("seed $seed: block $b rejected by the budget guard")
        info["TR"] == tr || error("seed $seed: block $b TR lifted from $tr to $(info["TR"])")
        mape = info["mape"]
    end
    final_result(env, mape)
end

function check_reproduction(adaptive)
    rec = JSON.parsefile(joinpath(RUN, "global_best", "eval_t1err_heldout.json"))
    n = length(adaptive[1]["ape"])
    # eval_e2.py appends each episode's five spheres in turn, so columns are scans.
    ape_rec = reshape(Float64.(rec["per_episode_t1_err"]["ape"]), n, N_SCANS)
    t1_rec = reshape(Float64.(rec["per_episode_t1_err"]["t1_true_s"]), n, N_SCANS)
    d_ape = maximum(maximum(abs.(r["ape"] .- ape_rec[:, k])) for (k, r) in enumerate(adaptive))
    d_t1 = maximum(maximum(abs.(r["T1_true"] .- t1_rec[:, k])) for (k, r) in enumerate(adaptive))
    mape_rec = vec(mean(ape_rec; dims = 1))
    mape = [mean(r["ape"]) for r in adaptive]
    d_mape = maximum(abs.(mape .- mape_rec))
    @printf("reproduction: max |dAPE| %.3g, max |dT1_true| %.3g s, max |dMAPE| %.3g; mean MAPE %.4f%% (recorded %.4f%%, eval_summary %.4f%%)\n",
            d_ape, d_t1, d_mape, 100 * mean(mape), 100 * mean(mape_rec), rec["agent_mape_pct"])
    d_ape <= MATCH_TOL && d_t1 <= MATCH_TOL ||
        error("the held-out rollout does not reproduce eval_t1err_heldout.json; stopping before the controls")
    Dict("max_abs_diff_ape" => d_ape, "max_abs_diff_t1_true_s" => d_t1, "max_abs_diff_mape" => d_mape,
         "mape_pct_rollout" => 100 * mean(mape), "mape_pct_recorded" => 100 * mean(mape_rec),
         "per_scan_mape_rollout" => mape, "per_scan_mape_recorded" => mape_rec)
end

function median_protocol(env::E2Env, schedules)
    TIs, TRs, n_at = Float64[], Float64[], Int[]
    for b in 1:maximum(length, schedules)
        at_b = [s[b] for s in schedules if length(s) >= b]
        # Float32, as any schedule handed over by the Python wrapper arrives.
        ti = Float64(Float32(exp(median(log.(first.(at_b))))))
        tr = Float64(Float32(median(last.(at_b))))
        tr >= (ti + Float32(FIXED_TE_S)) / e2_tr_headroom(env) || error("median block $b would need a TR lift")
        env.Npe * (sum(TRs) + tr) > env.time_budget_s && break
        push!(TIs, ti); push!(TRs, tr); push!(n_at, length(at_b))
    end
    Dict("TI_s" => TIs, "TR_s" => TRs, "n_scans_at_block" => n_at, "time_s" => env.Npe * sum(TRs))
end

function run_adaptive!(results, wall, env, actor; n_scans = N_SCANS)
    t0 = time()
    episodes, adaptive = Dict{String,Any}[], Dict{String,Any}[]
    for k in 0:(n_scans - 1)
        ep = rollout(env, actor, HELDOUT_SEED + k)
        r = final_result(env, ep["blocks"][end]["mape"])
        r["schedule_scan"] = k
        push!(episodes, ep)
        push!(adaptive, r)
        @printf("%-12s phantom %2d: %2d blocks, %.1f s, MAPE %.4f%%\n", "adaptive", k, r["n_blocks"],
                r["time_s"], 100 * r["mape"])
        flush(stdout)
    end
    wall["adaptive"] = time() - t0
    results["adaptive"] = adaptive
    episodes
end

function run_arm!(results, wall, name, env, schedule_of; n_scans = N_SCANS)
    t0 = time()
    out = Dict{String,Any}[]
    for k in 0:(n_scans - 1)
        src, schedule = schedule_of(k)
        r = replay(env, HELDOUT_SEED + k, schedule)
        r["schedule_scan"] = src
        push!(out, r)
        @printf("%-12s phantom %2d <- schedule %s: %2d blocks, %.1f s, MAPE %.4f%%\n", name, k,
                src === nothing ? "-" : string(src), r["n_blocks"], r["time_s"], 100 * r["mape"])
        flush(stdout)
    end
    results[name] = out
    wall[name] = time() - t0
    out
end

function main()
    mkpath(CONTROL_DATA)
    env = build_env(JSON.parsefile(joinpath(RUN, "run_config.json"))["base_env_kwargs"])
    actor = load_actor("global_best")
    results, wall = Dict{String,Any}(), Dict{String,Float64}()

    episodes = run_adaptive!(results, wall, env, actor)
    adaptive = results["adaptive"]
    open(io -> JSON.print(io, Dict("checkpoint" => "global_best", "seed_base" => HELDOUT_SEED,
                                   "episodes" => episodes)),
         joinpath(CONTROL_DATA, "control_adaptive_heldout.json"), "w")
    reproduction = check_reproduction(adaptive)

    schedules = executed_schedule.(episodes)
    identity = run_arm!(results, wall, "identity", env, k -> (k, schedules[k + 1]))
    for (a, r) in zip(adaptive, identity)
        a["mape"] == r["mape"] && a["T1_est"] == r["T1_est"] ||
            error("replaying a scan's own schedule does not reproduce its adaptive result")
    end
    println("identity replay reproduces every adaptive scan exactly")

    for s in SHIFTS
        run_arm!(results, wall, "swap_$s", env, k -> (mod(k + s, N_SCANS), schedules[mod(k + s, N_SCANS) + 1]))
    end

    protocols = Dict("fixed_median" => median_protocol(env, schedules),
                     "fixed_median_validation" => median_protocol(env, validation_schedules(env, RUN)))
    for name in ("fixed_median", "fixed_median_validation")
        proto = collect(zip(protocols[name]["TI_s"], protocols[name]["TR_s"]))
        run_arm!(results, wall, name, env, k -> (nothing, proto))
    end

    blocks = Dict(name => sum(r["n_blocks"] for r in arm) for (name, arm) in results)
    open(joinpath(CONTROL_DATA, "control_results.json"), "w") do io
        JSON.print(io, Dict("checkpoint" => "global_best", "seed_base" => HELDOUT_SEED, "n_scans" => N_SCANS,
                            "shifts" => collect(SHIFTS), "budget_s" => env.time_budget_s, "Npe" => env.Npe,
                            "threads" => Threads.nthreads(), "julia_version" => string(VERSION),
                            "reproduction" => reproduction, "median_protocols" => protocols,
                            "schedules" => [Dict("TI_s" => first.(s), "TR_s" => last.(s)) for s in schedules],
                            "arms" => results, "wall_s" => wall, "simulated_blocks" => blocks))
    end
    println("wrote ", joinpath(CONTROL_DATA, "control_results.json"))
end

# control_distill_variants.jl and control_narrow_band.jl include this file for the replay code.
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

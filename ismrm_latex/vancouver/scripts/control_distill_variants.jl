# Control for the ISMRM abstract: does the no-memory policy lose held-out T1 accuracy to the
# TI-histogram policy because its schedule shape is worse, or because it over-adapts?
#
# Applies control_schedule_swap.jl's fixed_median_validation recipe to the no-memory and
# fitter-uncertainty runs: per block index, the median log TI and median TR of the run's
# global_best schedules on its 24 validation scans (seed 500000), truncated to the budget,
# replayed with no policy in the loop on the 24 held-out phantoms (seeds 600000-600023).
# Every replay uses the TI-histogram run's E2Env, so all arms share phantoms, noise stream,
# forward model and fitter; the runs' env settings may differ only in observation channels.
#
# Stops unless the TI-histogram run's own distilled protocol reproduces its
# fixed_median_validation arm in control_results.json exactly, and unless each run's
# eval_t1err_heldout.json (the adaptive reference) was scored on the phantoms replayed here.
#
# Run from the RLxMRI repo root, then summarise with control_distill_summary.py:
#   julia -t 8 --project=. ismrm_latex/vancouver/scripts/control_distill_variants.jl
#
# Writes figures/control_data/control_distill_variants.json.

include(joinpath(@__DIR__, "control_schedule_swap.jl"))

const VARIANTS = ["histogram" => RUN,
                  "no_memory" => joinpath(ROOT, "runs", "e2", "mf_runB_5sphere_560s_gpu"),
                  "sigma" => joinpath(ROOT, "runs", "e2", "mf_runB_5sphere_sigma_560s_gpu")]
# julia/rl/e2.jl reads these only to build the observation.
const OBS_ONLY = ("include_sigma", "include_ti_history", "ti_hist_bins")

function env_diff(ref::AbstractDict, kw::AbstractDict)
    diff = Dict{String,Any}()
    for k in union(keys(ref), keys(kw))
        a, b = get(ref, k, nothing), get(kw, k, nothing)
        a == b && continue
        k in OBS_ONLY || error("env setting $k differs from the TI-histogram run ($a vs $b)")
        diff[k] = Dict("histogram" => a, "variant" => b)
    end
    diff
end

function check_histogram(protocol, arm)
    ref = JSON.parsefile(joinpath(CONTROL_DATA, "control_results.json"))
    ref_protocol, ref_arm = ref["median_protocols"]["fixed_median_validation"], ref["arms"]["fixed_median_validation"]
    same_protocol = all(protocol[k] == ref_protocol[k] for k in ("TI_s", "TR_s", "n_scans_at_block", "time_s"))
    d_est = maximum(maximum(abs.(r["T1_est"] .- q["T1_est"])) for (r, q) in zip(arm, ref_arm))
    d_mape = maximum(abs(r["mape"] - q["mape"]) for (r, q) in zip(arm, ref_arm))
    exact = same_protocol && all(r[k] == q[k] for (r, q) in zip(arm, ref_arm)
                                 for k in ("mape", "ape", "T1_true", "T1_est", "n_blocks", "time_s"))
    @printf("histogram sanity check: protocol %s, max |dT1_est| %.3g s, max |dMAPE| %.3g\n",
            same_protocol ? "identical" : "differs", d_est, d_mape)
    exact || error("the TI-histogram run's distilled protocol does not reproduce control_results.json; stopping")
    Dict("protocol_identical" => same_protocol, "per_phantom_identical" => exact,
         "max_abs_diff_T1_est_s" => d_est, "max_abs_diff_mape" => d_mape)
end

function adaptive_reference(run::AbstractString, arm)
    rec = JSON.parsefile(joinpath(run, "global_best", "eval_t1err_heldout.json"))
    n = length(arm[1]["ape"])
    # eval_e2.py appends each episode's five spheres in turn, so columns are scans.
    ape = reshape(Float64.(rec["per_episode_t1_err"]["ape"]), n, N_SCANS)
    t1 = reshape(Float64.(rec["per_episode_t1_err"]["t1_true_s"]), n, N_SCANS)
    d_t1 = maximum(maximum(abs.(r["T1_true"] .- t1[:, k])) for (k, r) in enumerate(arm))
    d_t1 <= MATCH_TOL || error("$(basename(run)): eval_t1err_heldout.json was scored on other phantoms (max |dT1_true| $d_t1 s)")
    Dict("ape" => collect.(eachcol(ape)), "T1_true" => collect.(eachcol(t1)), "mape" => vec(mean(ape; dims = 1)),
         "mape_pct_recorded" => rec["agent_mape_pct"], "max_abs_diff_t1_true_s" => d_t1)
end

function main()
    kw = JSON.parsefile(joinpath(RUN, "run_config.json"))["base_env_kwargs"]
    env = build_env(kw)
    diffs = Dict(name => env_diff(kw, JSON.parsefile(joinpath(run, "run_config.json"))["base_env_kwargs"])
                 for (name, run) in VARIANTS)
    protocols = Dict(name => median_protocol(env, validation_schedules(env, run)) for (name, run) in VARIANTS)
    results, wall = Dict{String,Any}(), Dict{String,Float64}()

    function replay_protocol!(name)
        proto = collect(zip(protocols[name]["TI_s"], protocols[name]["TR_s"]))
        run_arm!(results, wall, name, env, k -> (nothing, proto))
    end
    sanity = check_histogram(protocols["histogram"], replay_protocol!("histogram"))
    foreach(replay_protocol!, ("no_memory", "sigma"))

    variants = Dict(name => Dict("run" => relpath(run, ROOT), "env_diff" => diffs[name],
                                 "protocol" => protocols[name], "distilled" => results[name],
                                 "adaptive" => adaptive_reference(run, results[name]))
                    for (name, run) in VARIANTS)
    out = joinpath(CONTROL_DATA, "control_distill_variants.json")
    open(out, "w") do io
        JSON.print(io, Dict("env_run" => relpath(RUN, ROOT), "seed_base" => HELDOUT_SEED, "n_scans" => N_SCANS,
                            "validation_seed" => VALIDATION_SEED, "budget_s" => env.time_budget_s, "Npe" => env.Npe,
                            "threads" => Threads.nthreads(), "julia_version" => string(VERSION),
                            "sanity_check" => sanity, "variants" => variants, "wall_s" => wall,
                            "simulated_blocks" => Dict(k => sum(r["n_blocks"] for r in v) for (k, v) in results)))
    end
    println("wrote ", out)
end

main()

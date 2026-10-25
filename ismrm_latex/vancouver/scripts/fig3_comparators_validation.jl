# Running T1 MAPE of the two fixed 560 s comparators (log_grid_trmatched and cr_optimal,
# fixed_schedules.jl) on the 24 validation phantoms of Figure 3's panel C: seeds 500000 + k,
# the phantoms fig3evo_rollout.jl rolled the featured checkpoints out on.
#
# The comparators run as control_narrow_band.jl runs them: same E2Env, fitter, per-seed noise
# and Python action round trip. The script stops unless that first reproduces their recorded
# held-out results (fig5_v2_data/fixed_comparators_heldout.json) scan for scan, and unless every
# validation phantom's T1_true matches the checkpoints' rollouts.
#
# Regenerate from the RLxMRI repo root (about 4 min on 8 threads), then rebuild the figure:
#   julia -t 8 --project=. ismrm_latex/vancouver/scripts/fig3_comparators_validation.jl
#   .venv/bin/python ismrm_latex/vancouver/scripts/fig3_v3_figure.py
#
# Writes figures/fig3_comparators_validation.json: per phantom, the executed blocks with the MAPE
# after each, in fig3evo_rollout.jl's episode format.

include(joinpath(@__DIR__, "control_narrow_band.jl"))

const FEATURED = ("stage0_analytic_end", "stage3_full3_end", "global_best")
const OUT_VALIDATION = joinpath(FIGURES, "fig3_comparators_validation.json")

function as_episode(r)
    # The budget guard's rejected block leaves the estimate, so the final MAPE is the last executed one.
    r["executed_mape"][end] == r["mape"] || error("seed $(r["seed"]): final MAPE is not the last block's")
    Dict("seed" => r["seed"], "T1_true" => r["T1_true"], "T1_est" => r["T1_est"], "mape" => r["mape"],
         "blocks" => [Dict("TI" => b[1], "TR" => b[2], "mape" => m, "executed" => true)
                      for (b, m) in zip(r["executed_TI_TR_alpha"], r["executed_mape"])])
end

function main()
    env = build_env(JSON.parsefile(joinpath(RUN, "run_config.json"))["base_env_kwargs"])
    fixed = recorded_comparators(env)
    comparator!(name, results, wall; kw...) = run_comparator!(results, wall, name, env, fixed.schedules[name];
                                                        n_scans = N_SCANS, learn_alpha = fixed.learn_alpha,
                                                        numpy2 = fixed.numpy2, kw...)

    check, check_wall = Dict{String,Any}(), Dict{String,Float64}()
    for name in COMPARATORS
        comparator!(name, check, check_wall)
        check_identical(name, check[name], fixed.scans[name])
    end

    results, wall = Dict{String,Any}(), Dict{String,Float64}()
    for name in COMPARATORS
        comparator!(name, results, wall; seed_base = VALIDATION_SEED)
    end
    for ckpt in FEATURED
        eps = JSON.parsefile(joinpath(FIGURES, "fig3evo_rollout_$(ckpt)_0-$(N_SCANS - 1).json"))["episodes"]
        all(ep["seed"] == r["seed"] && Float64.(ep["T1_true"]) == r["T1_true"]
            for arm in values(results) for (ep, r) in zip(eps, arm)) ||
            error("the comparators did not run on $ckpt's validation phantoms")
    end
    println("T1_true matches the rollouts of ", join(FEATURED, ", "), " on all $N_SCANS phantoms")

    comparators = Dict(name => Dict(
        "mape_pct" => 100 * mean(r["mape"] for r in results[name]),
        "schedule" => Dict("TI_s" => s.TIs, "TR_s" => s.TRs, "max_steps" => s.max_steps),
        "episodes" => as_episode.(results[name])) for (name, s) in fixed.schedules)
    for (name, c) in comparators
        @printf("%-18s validation MAPE %.2f%% over %d phantoms\n", name, c["mape_pct"], N_SCANS)
    end
    open(OUT_VALIDATION, "w") do io
        JSON.print(io, Dict(
            "seed_base" => VALIDATION_SEED, "n_scans" => N_SCANS, "env_run" => relpath(RUN, ROOT),
            "budget_s" => env.time_budget_s, "Npe" => env.Npe, "numpy_promotion" => fixed.numpy_promotion,
            "t1_true_matches" => collect(FEATURED), "julia_version" => string(VERSION),
            "threads" => Threads.nthreads(),
            "regression_check" => Dict("heldout_seed_base" => HELDOUT_SEED, "per_scan_identical" => collect(COMPARATORS),
                                       "wall_s" => check_wall),
            "comparators" => comparators, "wall_s" => wall))
    end
    println("wrote ", OUT_VALIDATION)
end

main()

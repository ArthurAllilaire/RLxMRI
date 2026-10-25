# Per-scan held-out results of the two fixed comparators (log_grid_trmatched and cr_optimal,
# replayed by fixed_schedules.jl), which runs/e2/baselines_heldout_560s and
# .../mf_runB_cached3_cached_full3_full_gpu/baselines_heldout only stored as aggregates
# (python/baseline_e2.py:evaluate).
# Each scan replays the baseline's held-out seed (600000 + k) through the same E2Env config,
# and the script stops unless the replayed means reproduce baseline_summary.json.
#
# Run from the RLxMRI repo root:
#   julia -t 8 --project=. ismrm_latex/vancouver/scripts/fixed_comparators_heldout.jl [numpy1|numpy2]
#
# Writes figures/fig5_v2_data/fixed_comparators_heldout.json.

include(joinpath(@__DIR__, "fig4_common.jl"))
include(joinpath(@__DIR__, "fixed_schedules.jl"))
using Statistics

const HELDOUT_SEED = 600_000
const OUT = normpath(joinpath(@__DIR__, "..", "figures", "fig5_v2_data", "fixed_comparators_heldout.json"))
const RUN_240 = "mf_runB_cached3_cached_full3_full_gpu"
const BUDGETS = (
    (budget_s = 560, run = BASELINE_RUN, dir = joinpath(RUNS, "baselines_heldout_560s")),
    (budget_s = 240, run = RUN_240, dir = joinpath(RUNS, RUN_240, "baselines_heldout")),
)
# A replayed mean that differs by more than this cannot be the recorded run.
const MATCH_TOL_PCT = 1e-6

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

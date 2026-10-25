# Control for the ISMRM abstract: does the TI-histogram policy's per-scan adaptation pay off
# when a scan's five sphere T1s sit in a narrow band at an unknown position?
#
# Training and the held-out evaluation draw every sphere T1 uniformly over the T1-plate range,
# so each scan spans most of it and needs similar TIs. Here each scan draws a centre c
# log-uniformly over that range and its five T1s uniformly over c(1 +- 0.15)
# (t1_sampler = :narrow_band, julia/rl/e2.jl), which the policy never saw in training.
#
# 1. Regression check on the default sampler, held-out seeds 600000-600023: the rollout of
#    global_best must still reproduce global_best/eval_t1err_heldout.json exactly, and the
#    distilled protocol and fixed comparators must reproduce their recorded per-scan results
#    (control_results.json, fig5_v2_data/fixed_comparators_heldout.json) bit for bit.
# 2. Narrow-band phantoms, seeds 600000 + k. The sampler keeps the default sampler's pose and
#    per-seed noise stream, so phantom k differs from held-out scan k only in its T1s.
#    Every arm runs on the same N phantoms:
#      adaptive   the policy in the loop;
#      swap_s     phantom k gets scan (k + s) mod N's executed schedule, s = 1, N/3, 2N/3;
#      distilled  control_results.json's fixed_median_validation protocol, as recorded;
#      log_grid_trmatched, cr_optimal  the fixed 560 s comparators (fixed_schedules.jl).
#
# Run from the RLxMRI repo root, then summarise with control_narrow_band_summary.py:
#   julia -t 8 --project=. ismrm_latex/vancouver/scripts/control_narrow_band.jl [N_SCANS]
#
# Writes figures/control_data/control_narrow_band.json.

include(joinpath(@__DIR__, "control_schedule_swap.jl"))
include(joinpath(@__DIR__, "fixed_schedules.jl"))

const T1_BAND_REL = 0.15
const N_NARROW_DEFAULT = 96
const FIXED_HELDOUT = joinpath(FIGURES, "fig5_v2_data", "fixed_comparators_heldout.json")
const COMPARATORS = ("log_grid_trmatched", "cr_optimal")
const OUT = joinpath(CONTROL_DATA, "control_narrow_band.json")

"""The recorded 560 s comparators: their schedules, the action round trip they ran with, and their
per-scan held-out results."""
function recorded_comparators(env::E2Env)
    rec = JSON.parsefile(FIXED_HELDOUT)
    fixed = rec["budgets"][string(round(Int, env.time_budget_s))]
    fixed["env_run"] == basename(RUN) || error("the recorded comparators used another run's env")
    (schedules = Dict("log_grid_trmatched" => log_grid_schedule(env.time_budget_s, env.Npe),
                      "cr_optimal" => cr_schedule(joinpath(ROOT, fixed["baseline_dir"]))),
     learn_alpha = fixed["learn_alpha_action_layout"], numpy_promotion = rec["numpy_promotion"],
     numpy2 = rec["numpy_promotion"] == "numpy2", scans = Dict(name => fixed[name]["scans"] for name in COMPARATORS))
end

function run_comparator!(results, wall, name, env, sched; n_scans, learn_alpha, numpy2, seed_base = HELDOUT_SEED)
    t0 = time()
    results[name] = map(0:(n_scans - 1)) do k
        r = run_episode(env, seed_base + k, sched; learn_alpha = learn_alpha, numpy2 = numpy2)
        @printf("%-18s phantom %2d: %2d blocks, %.1f s, MAPE %.4f%%\n", name, k, r["n_blocks"], r["time_s"],
                100 * r["mape"])
        flush(stdout)
        r
    end
    wall[name] = time() - t0
end

function check_identical(name, got, recorded)
    length(got) == length(recorded) &&
        all(r[k] == q[k] for (r, q) in zip(got, recorded) for k in ("T1_true", "T1_est", "mape", "n_blocks")) ||
        error("$name on the default sampler does not reproduce its recorded per-scan results; stopping")
    println("$name on the default sampler reproduces its recorded per-scan results exactly")
end

function main(args)
    n_scans = isempty(args) ? N_NARROW_DEFAULT : parse(Int, args[1])
    kw = JSON.parsefile(joinpath(RUN, "run_config.json"))["base_env_kwargs"]
    actor = load_actor("global_best")
    recorded = JSON.parsefile(joinpath(CONTROL_DATA, "control_results.json"))
    distilled = recorded["median_protocols"]["fixed_median_validation"]
    protocol = collect(zip(Float64.(distilled["TI_s"]), Float64.(distilled["TR_s"])))

    default_env = build_env(kw)
    fixed = recorded_comparators(default_env)
    schedules_fixed, learn_alpha, numpy2 = fixed.schedules, fixed.learn_alpha, fixed.numpy2

    check, check_wall = Dict{String,Any}(), Dict{String,Float64}()
    run_adaptive!(check, check_wall, default_env, actor)
    reproduction = check_reproduction(check["adaptive"])
    reproduction["max_abs_diff_ape"] == 0 && reproduction["max_abs_diff_t1_true_s"] == 0 ||
        error("the held-out rollout reproduces eval_t1err_heldout.json only to within tolerance; stopping")
    run_arm!(check, check_wall, "distilled", default_env, k -> (nothing, protocol))
    check_identical("distilled", check["distilled"], recorded["arms"]["fixed_median_validation"])
    for name in COMPARATORS
        run_comparator!(check, check_wall, name, default_env, schedules_fixed[name];
                        n_scans = N_SCANS, learn_alpha = learn_alpha, numpy2 = numpy2)
        check_identical(name, check[name], fixed.scans[name])
    end

    env = build_env(merge(kw, Dict{String,Any}("t1_sampler" => "narrow_band", "t1_band_rel" => T1_BAND_REL)))
    phantoms = map(0:(n_scans - 1)) do k
        e2_reset!(env; rng_seed = HELDOUT_SEED + k)
        (centre = env.T1_band_centre, T1_true = copy(env.T1_true))
    end
    results, wall = Dict{String,Any}(), Dict{String,Float64}()
    episodes = run_adaptive!(results, wall, env, actor; n_scans = n_scans)
    schedules = executed_schedule.(episodes)
    shifts = (1, n_scans ÷ 3, 2 * n_scans ÷ 3)
    allunique(shifts) && all(0 .< shifts .< n_scans) || error("$n_scans scans are too few for three distinct shifts")
    for s in shifts
        run_arm!(results, wall, "swap_$s", env, k -> (mod(k + s, n_scans), schedules[mod(k + s, n_scans) + 1]);
                 n_scans = n_scans)
    end
    run_arm!(results, wall, "distilled", env, k -> (nothing, protocol); n_scans = n_scans)
    for name in COMPARATORS
        run_comparator!(results, wall, name, env, schedules_fixed[name];
                        n_scans = n_scans, learn_alpha = learn_alpha, numpy2 = numpy2)
    end
    all(r["T1_true"] == p.T1_true for arm in values(results) for (r, p) in zip(arm, phantoms)) ||
        error("the arms did not run on the same phantoms")

    open(OUT, "w") do io
        JSON.print(io, Dict(
            "checkpoint" => "global_best", "env_run" => relpath(RUN, ROOT), "seed_base" => HELDOUT_SEED,
            "n_scans" => n_scans, "t1_sampler" => "narrow_band", "t1_band_rel" => T1_BAND_REL,
            "centre_range_s" => collect(extrema(T1_ARRAY[env.cfg_field])), "shifts" => collect(shifts),
            "budget_s" => env.time_budget_s, "Npe" => env.Npe, "threads" => Threads.nthreads(),
            "julia_version" => string(VERSION), "numpy_promotion" => fixed.numpy_promotion,
            "regression_check" => Dict("adaptive" => reproduction, "per_scan_identical" => ["distilled", COMPARATORS...],
                                       "wall_s" => check_wall),
            "T1_band_centre" => [p.centre for p in phantoms],
            "protocols" => Dict("distilled" => distilled,
                                (name => Dict("TI_s" => s.TIs, "TR_s" => s.TRs, "max_steps" => s.max_steps)
                                 for (name, s) in schedules_fixed)...),
            "schedules" => [Dict("TI_s" => first.(s), "TR_s" => last.(s)) for s in schedules],
            "arms" => results, "wall_s" => wall,
            "simulated_blocks" => Dict(name => sum(r["n_blocks"] for r in arm) for (name, arm) in results)))
    end
    println("wrote ", OUT)
end

# fig3_comparators_validation.jl includes this file for the comparator setup and runner.
if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end

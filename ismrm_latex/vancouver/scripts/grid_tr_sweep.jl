# Robustness check for the ISMRM abstract: is TR 1.7 s close to the best TR for the fixed log TI
# grid comparator (log_grid_trmatched, python/baseline_e2.py:make_log_grid_trmatched)?
#
# The construction runs floor(budget / (Npe TR)) blocks at one TR, with TIs log-spaced over
# 0.05-3.0 s whatever the TR. It does not cap TI by TR: a block whose TI + TE exceeds 0.9 TR has
# its TR lifted by the env (julia/rl/e2.jl:e2_step!), and the budget guard rejects the first block
# that would overrun the budget, which is why the comparator ran 9 blocks in 504 s at TR 1.7 s.
# The TRs are GRID_TRS plus the block-weighted mean and the median of the TI-histogram policy's
# executed TRs on its 24 validation scans (global_best/diagnostics/episodes.json without the
# budget-rejected final proposals), so no held-out information shapes a comparator; the mean
# equalises block count with the policy. For each TR, two series run on the 24 held-out phantoms
# (seeds 600000-600023) through the comparator replay of fixed_schedules.jl (same E2Env config,
# fitter, per-seed noise and Python action round trip as the recorded comparators):
#   as_built  the construction itself, lifts and rejections included, as the reported baseline ran;
#   capped    top TI lowered to 0.9 TR - TE - 1 ms and at most max_blocks TIs, so every block runs
#             at the nominal TR and every TI of the grid is executed.
# The script stops unless as_built at TR 1.7 s first reproduces the recorded per-scan results
# (fig5_v2_data/fixed_comparators_heldout.json) bit for bit.
#
# Run from the RLxMRI repo root (about 70 min on 8 threads), then summarise:
#   julia -t 8 --project=. ismrm_latex/vancouver/scripts/grid_tr_sweep.jl
#   .venv/bin/python ismrm_latex/vancouver/scripts/grid_tr_sweep_summary.py
#
# Writes figures/control_data/grid_tr_sweep.json.

include(joinpath(@__DIR__, "control_narrow_band.jl"))

const GRID_TRS = (0.8, 1.0, 1.2, 1.35, 1.5, 1.7, 2.0, 2.5, 3.0, 4.0)
const SERIES = ("as_built", "capped")
const OUT_SWEEP = joinpath(CONTROL_DATA, "grid_tr_sweep.json")

function series_schedule(env::E2Env, series::AbstractString, tr)
    series == "as_built" && return log_grid_schedule(env.time_budget_s, env.Npe; tr_s = tr)
    log_grid_schedule(env.time_budget_s, env.Npe; tr_s = tr, ti_hi_s = capped_ti_hi(env, tr),
                      max_blocks = env.max_blocks)
end

"""What the env did to a schedule. Lifts and the budget guard depend only on the timings, so every
scan must have executed the same blocks."""
function feasibility(env::E2Env, sched, scans; learn_alpha::Bool, numpy2::Bool)
    first_scan = scans[1]
    ex, ep_len = first_scan["executed_TI_TR_alpha"], first_scan["ep_len"]
    all(s["executed_TI_TR_alpha"] == ex && s["time_s"] == first_scan["time_s"] && s["ep_len"] == ep_len
        for s in scans) || error("scans executed different blocks")
    n, n_ex = length(sched.TIs), length(ex)
    requested = [python_action(env, sched.TIs[mod1(j, n)], sched.TRs[mod1(j, n)];
                               learn_alpha = learn_alpha, numpy2 = numpy2) for j in 1:n_ex]
    all(b[1] == a[1] for (b, a) in zip(ex, requested)) || error("the env changed an executed TI")
    lifted = findall([b[2] != a[3] for (b, a) in zip(ex, requested)])
    rejected = nothing
    if ep_len > n_ex
        ended_by = ep_len <= n ? "budget guard rejected schedule block $ep_len" :
                                 "budget guard rejected the repeat of block $(mod1(ep_len, n))"
        a = python_action(env, sched.TIs[mod1(ep_len, n)], sched.TRs[mod1(ep_len, n)];
                          learn_alpha = learn_alpha, numpy2 = numpy2)
        # The TR e2_step! would have run the rejected block at, after its lift.
        rejected = Dict("block" => ep_len, "TI_s" => a[1], "TR_requested_s" => a[3],
                        "TR_needed_s" => max(a[3], (a[1] + a[2]) / e2_tr_headroom(env)))
    else
        ended_by = n_ex >= env.max_blocks ? "max_blocks" : "no time left for a block at the TR floor"
    end
    Dict{String,Any}(
        "n_schedule_blocks" => n, "n_executed" => n_ex, "n_lifted" => length(lifted),
        "n_schedule_blocks_not_run" => max(0, n - n_ex), "n_repeated_blocks" => max(0, n_ex - n),
        "ended_by" => ended_by, "rejected_block" => rejected, "scan_time_s" => first_scan["time_s"],
        "TI_executed_range_s" => collect(extrema(first.(ex))),
        "TR_executed_range_s" => collect(extrema(getindex.(ex, 2))),
        "lifted_blocks" => [Dict("block" => j, "TI_s" => ex[j][1], "TR_requested_s" => requested[j][3],
                                 "TR_executed_s" => ex[j][2]) for j in lifted])
end

"""Block-weighted mean and median of a policy's executed TRs, pooled over scans."""
function policy_trs(tr_lists)
    trs = reduce(vcat, tr_lists)
    Dict("mean_s" => mean(trs), "median_s" => median(trs), "n_blocks" => length(trs), "n_scans" => length(tr_lists))
end

function main()
    env = build_env(JSON.parsefile(joinpath(RUN, "run_config.json"))["base_env_kwargs"])
    fixed = recorded_comparators(env)
    policy = Dict(
        "validation" => policy_trs([last.(s) for s in validation_schedules(env, RUN)]),
        "heldout_for_reference" => policy_trs([Float64.(s["TR_s"]) for s in
                                          JSON.parsefile(joinpath(CONTROL_DATA, "control_results.json"))["schedules"]]))
    points = sort([[(tr, nothing) for tr in GRID_TRS];
                   (policy["validation"]["mean_s"], "policy mean TR");
                   (policy["validation"]["median_s"], "policy median TR")]; by = first)
    @printf("policy TRs on the validation scans: mean %.4f s, median %.4f s over %d blocks\n",
            policy["validation"]["mean_s"], policy["validation"]["median_s"], policy["validation"]["n_blocks"])
    scans, wall = Dict{String,Any}(), Dict{String,Float64}()
    sweep!(label, sched) = run_comparator!(scans, wall, label, env, sched; n_scans = N_SCANS,
                                           learn_alpha = fixed.learn_alpha, numpy2 = fixed.numpy2)
    label(series, tr) = "$series TR $(round(tr; digits = 3))"

    ref = label("as_built", LOG_GRID_TR_S)
    sched_ref = series_schedule(env, "as_built", LOG_GRID_TR_S)
    rec_sched = JSON.parsefile(FIXED_HELDOUT)["budgets"][string(round(Int, env.time_budget_s))]["log_grid_trmatched"]["schedule"]
    sched_ref.TIs == rec_sched["TI_s"] && sched_ref.TRs == rec_sched["TR_s"] ||
        error("the as-built TR $(LOG_GRID_TR_S) s grid is not the recorded comparator's schedule")
    sweep!(ref, sched_ref)
    recorded = fixed.scans["log_grid_trmatched"]
    check_identical(ref, scans[ref], recorded)
    all(r["time_s"] == q["time_s"] && r["executed_TI_TR_alpha"] == q["executed_TI_TR_alpha"]
        for (r, q) in zip(scans[ref], recorded)) || error("$ref does not reproduce the recorded blocks; stopping")

    out = Dict{String,Any}()
    for series in SERIES
        out[series] = map(points) do (tr, point_label)
            sched = series_schedule(env, series, tr)
            haskey(scans, label(series, tr)) || sweep!(label(series, tr), sched)
            f = feasibility(env, sched, scans[label(series, tr)]; learn_alpha = fixed.learn_alpha,
                            numpy2 = fixed.numpy2)
            series == "capped" && (f["n_lifted"] + f["n_schedule_blocks_not_run"] + f["n_repeated_blocks"] > 0) &&
                error("the capped grid at TR $tr s did not run as built: $f")
            Dict("TR_s" => tr, "label" => point_label, "schedule" => Dict("TI_s" => sched.TIs, "TR_s" => sched.TRs),
                 "feasibility" => f, "scans" => scans[label(series, tr)], "wall_s" => wall[label(series, tr)])
        end
    end
    all(r["T1_true"] == q["T1_true"] for arm in values(scans) for (r, q) in zip(arm, recorded)) ||
        error("the series did not run on the comparator's phantoms")

    open(OUT_SWEEP, "w") do io
        JSON.print(io, Dict(
            "env_run" => relpath(RUN, ROOT), "seed_base" => HELDOUT_SEED, "n_scans" => N_SCANS,
            "budget_s" => env.time_budget_s, "Npe" => env.Npe, "max_blocks" => env.max_blocks,
            "TE_s" => TE_S, "tr_headroom" => e2_tr_headroom(env), "cap_margin_s" => CAP_MARGIN_S,
            "TI_range_s" => collect(LOG_GRID_TI_RANGE_S), "grid_TRs" => collect(GRID_TRS),
            "policy_TRs" => Dict("checkpoint" => "global_best", "used" => "validation", policy...),
            "reference" => Dict("series" => "as_built", "TR_s" => LOG_GRID_TR_S,
                                "recorded" => relpath(FIXED_HELDOUT, ROOT), "per_scan_identical" => true),
            "numpy_promotion" => fixed.numpy_promotion, "learn_alpha_action_layout" => fixed.learn_alpha,
            "julia_version" => string(VERSION), "threads" => Threads.nthreads(), "series" => out))
    end
    println("wrote ", OUT_SWEEP)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

# The abstract's fixed log TI grid comparator, best_log_grid_schedule (fixed_schedules.jl): TR 2.0 s,
# the best row of the TR sweep (grid_tr_sweep.md), with the top TI capped so that no block's TR is
# lifted. It replaces log_grid_trmatched (TR 1.7 s) and is replayed as control_narrow_band.jl replays
# the fixed comparators (same E2Env, fitter, per-seed noise and Python action round trip) on
#   held-out     the 24 held-out phantoms, seeds 600000 + k; the script stops unless this reproduces
#                the sweep's capped TR 2.0 s row (grid_tr_sweep.json) bit for bit;
#   narrow band  the clustered-T1 phantoms of control_narrow_band.json, seeds 600000 + k; the script
#                stops unless log_grid_trmatched, replayed on them first, reproduces that file's arm
#                bit for bit, so the new grid sees the same phantoms and noise as the old comparators.
# fig3_comparators_validation.jl runs it on the validation phantoms.
#
# Run from the RLxMRI repo root (about 10 min on 6 threads), then summarise:
#   julia -t 6 --project=. ismrm_latex/vancouver/scripts/best_log_grid.jl
#   .venv/bin/python ismrm_latex/vancouver/scripts/best_log_grid_summary.py
#
# Writes figures/control_data/best_log_grid_heldout.json and best_log_grid_narrow_band.json:
# per-scan results in run_episode's format.

include(joinpath(@__DIR__, "control_narrow_band.jl"))

const BEST_GRID = "best_log_grid"
const SWEEP = joinpath(CONTROL_DATA, "grid_tr_sweep.json")
const NARROW_BAND = joinpath(CONTROL_DATA, "control_narrow_band.json")
const OUT_BEST_HELDOUT = joinpath(CONTROL_DATA, "best_log_grid_heldout.json")
const OUT_BEST_NARROW = joinpath(CONTROL_DATA, "best_log_grid_narrow_band.json")

schedule_dict(s) = Dict("TI_s" => s.TIs, "TR_s" => s.TRs)

"""The grid for env, and its held-out per-scan results as this script recorded them."""
function recorded_best_grid(env::E2Env)
    rec = JSON.parsefile(OUT_BEST_HELDOUT)
    sched = best_log_grid_schedule(env)
    rec["schedule"] == schedule_dict(sched) || error("$OUT_BEST_HELDOUT holds another schedule")
    (schedule = sched, scans = rec["scans"])
end

"""Every field of every recorded scan must equal the replayed one; control_narrow_band.json predates
run_episode's executed_mape, so a recorded file may hold fewer fields than the replay."""
function check_bit_identical(what, got, recorded)
    length(got) == length(recorded) &&
        all(all(haskey(r, k) && r[k] == q[k] for k in keys(q)) for (r, q) in zip(got, recorded)) ||
        error("$what does not reproduce its recorded per-scan results bit for bit; stopping")
    println("$what reproduces its recorded per-scan results bit for bit")
end

function write_results(path, env::E2Env, sched, fixed, scans, wall_s; extra...)
    open(path, "w") do io
        JSON.print(io, Dict(
            "comparator" => "best_log_grid_schedule", "env_run" => relpath(RUN, ROOT), "seed_base" => HELDOUT_SEED,
            "n_scans" => length(scans), "t1_sampler" => string(env.t1_sampler), "budget_s" => env.time_budget_s,
            "Npe" => env.Npe, "TR_s" => BEST_GRID_TR_S, "schedule" => schedule_dict(sched),
            "numpy_promotion" => fixed.numpy_promotion, "julia_version" => string(VERSION),
            "threads" => Threads.nthreads(), "scans" => scans, "wall_s" => wall_s,
            (string(k) => v for (k, v) in extra)...))
    end
    println("wrote ", path)
end

function main()
    kw = JSON.parsefile(joinpath(RUN, "run_config.json"))["base_env_kwargs"]
    env = build_env(kw)
    fixed = recorded_comparators(env)
    replay!(results, wall, name, env, sched, n_scans) =
        run_comparator!(results, wall, name, env, sched; n_scans = n_scans, learn_alpha = fixed.learn_alpha,
                        numpy2 = fixed.numpy2)

    sched = best_log_grid_schedule(env)
    row = only(p for p in JSON.parsefile(SWEEP)["series"]["capped"] if p["TR_s"] == BEST_GRID_TR_S && p["label"] === nothing)
    row["schedule"] == schedule_dict(sched) || error("best_log_grid_schedule is not the sweep's capped TR $(BEST_GRID_TR_S) s grid")
    results, wall = Dict{String,Any}(), Dict{String,Float64}()
    replay!(results, wall, "$BEST_GRID heldout", env, sched, N_SCANS)
    check_bit_identical("$BEST_GRID on the held-out phantoms", results["$BEST_GRID heldout"], row["scans"])
    write_results(OUT_BEST_HELDOUT, env, sched, fixed, results["$BEST_GRID heldout"], wall["$BEST_GRID heldout"];
                  reproduces = Dict("file" => relpath(SWEEP, ROOT), "series" => "capped", "TR_s" => BEST_GRID_TR_S,
                                    "per_scan_identical" => true))

    nb = JSON.parsefile(NARROW_BAND)
    nb["t1_sampler"] == "narrow_band" && nb["t1_band_rel"] == T1_BAND_REL && nb["seed_base"] == HELDOUT_SEED ||
        error("$NARROW_BAND is not control_narrow_band.jl's setup")
    env_nb = build_env(merge(kw, Dict{String,Any}("t1_sampler" => nb["t1_sampler"], "t1_band_rel" => nb["t1_band_rel"])))
    best_log_grid_schedule(env_nb) == sched || error("the narrow-band env gives another grid")
    replay!(results, wall, "log_grid_trmatched narrow band", env_nb, fixed.schedules["log_grid_trmatched"], nb["n_scans"])
    check_bit_identical("log_grid_trmatched on the narrow-band phantoms", results["log_grid_trmatched narrow band"],
                        nb["arms"]["log_grid_trmatched"])
    replay!(results, wall, "$BEST_GRID narrow band", env_nb, sched, nb["n_scans"])
    write_results(OUT_BEST_NARROW, env_nb, sched, fixed, results["$BEST_GRID narrow band"], wall["$BEST_GRID narrow band"];
                  t1_band_rel = nb["t1_band_rel"], T1_band_centre = nb["T1_band_centre"],
                  regression_check = Dict("file" => relpath(NARROW_BAND, ROOT), "arm" => "log_grid_trmatched",
                                          "per_scan_identical" => true,
                                          "wall_s" => wall["log_grid_trmatched narrow band"]))
end

# fig3_comparators_validation.jl includes this file for recorded_best_grid.
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

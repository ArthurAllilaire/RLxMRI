# Fig 4 replay: re-run scans of 560 s protocols through the E2 simulator on CPU and keep
# every block's noisy magnitude image, because training only ever kept the latest image
# and per-sphere ROI fits.
#
#   julia -t 8 --project=<RLxMRI> fig4_replay.jl                  # validation scans 0-23
#   julia -t 8 --project=<RLxMRI> fig4_replay.jl 6 14             # selected validation scans
#   julia -t 8 --project=<RLxMRI> fig4_replay.jl --heldout 7      # selected held-out scans
#   julia -t 8 --project=<RLxMRI> fig4_replay.jl --heldout --best-grid 7
#
# Validation scans (seed 500000 + k): TI-histogram and no-memory policies (schedules from
# their global_best/diagnostics/episodes.json), fixed log TI grid, Cramer-Rao.
# Held-out scans (seed 600000 + k): TI-histogram policy (its held-out schedule from
# control_adaptive_heldout.json), the fixed protocol distilled from its validation
# schedules (control_results.json), fixed log TI grid and Cramer-Rao (their executed
# held-out schedules from fig5_v2_data/fixed_comparators_heldout.json). Every held-out
# replay must reproduce the final T1 estimates recorded in those files.
# --best-grid: the log TI grid row is fixed_schedules.jl's best_log_grid_schedule instead (the
# capped TR 2.0 s grid of grid_tr_sweep.md), which must reproduce its scan in
# control_data/grid_tr_sweep.json.
#
# Writes figures/fig4_data/fig4_replay_<tag>.npz per scan (tags as in fig4_common.jl).

include(joinpath(@__DIR__, "fig4_common.jl"))
include(joinpath(@__DIR__, "fixed_schedules.jl"))

const VALIDATION_SEED0 = 500_000
const HELDOUT_SEED0 = 600_000
const N_SCANS = 24
const CR_SCHEDULE = joinpath(RUNS, "baselines_heldout_560s", "cr_optimal_schedule.json")
const CONTROL_DATA = joinpath(@__DIR__, "..", "figures", "control_data")
const FIXED_HELDOUT = joinpath(@__DIR__, "..", "figures", "fig5_v2_data", "fixed_comparators_heldout.json")
const GRID_SWEEP = joinpath(CONTROL_DATA, "grid_tr_sweep.json")

# The wrapper hands Julia a float32 action vector, so fixed schedules arrive float32-rounded.
f32(x) = Float64(Float32(x))
const TE_WRAPPER = f32(0.020)
action(ti, tr; alpha_deg = 90.0) = Float64[ti, TE_WRAPPER, tr, alpha_deg]

"""One protocol to replay. `check(r)` compares the replay with its recording (erroring on a
mismatch) and returns the recorded values to save beside it plus the recorded final MAPE."""
struct Plan
    method::Symbol
    env::E2Env
    actions::Vector{Vector{Float64}}
    check::Function
end

no_record(_) = (Dict{String,Any}(), nothing)

function rl_actions(run::AbstractString, k::Int)
    ep = JSON.parsefile(joinpath(RUNS, run, "global_best", "diagnostics", "episodes.json"))[k + 1]
    actions = [Float64[ep["TI"][b], ep["TE"][b], ep["TR"][b], ep["alpha_deg"][b]] for b in eachindex(ep["TI"])]
    actions, ep
end

function log_grid_actions(budget_s::Float64, npe::Int; tr_s::Float64 = 1.7)
    # python/baseline_e2.py:make_log_grid_trmatched with qalibremd_gym/schedules.py:log_ti_grid
    n = max(1, floor(Int, budget_s / (npe * tr_s)))
    tis = exp.(range(log(0.05), log(3.0); length = n))
    [action(f32(ti), f32(tr_s)) for ti in tis]
end

function cr_actions()
    s = JSON.parsefile(CR_SCHEDULE)
    [action(f32(ti), f32(tr)) for (ti, tr) in zip(s["TIs"], s["TRs"])]
end

function validation_plans(envs, k::Int)
    function rl_plan(m, env, run)
        actions, ep = rl_actions(run, k)
        check = r -> begin
            nb = length(r.images)
            # The recorded episode ends with the guard-rejected proposal exactly when 32 * sum(TR) > budget.
            (length(ep["TR"]) > nb) == r.rejected ||
                error("validation scan $k $m: budget-guard outcome differs from the recording")
            (Dict{String,Any}("mape_recorded" => Float64.(ep["mape"][1:nb]),
                              "T1_est_mean_recorded" => Float64.(ep["T1_est_mean"][1:nb]),
                              "snr_nema_peak_recorded" => Float64.(ep["snr_nema_peak"][1:nb])),
             Float64(ep["mape"][nb]))
        end
        Plan(m, env, actions, check), Float64.(ep["T1_true"])
    end
    hist, t1 = rl_plan(:hist, envs.hist, RL_RUNS.hist)
    nomem, t1_nomem = rl_plan(:nomem, envs.nomem, RL_RUNS.nomem)
    t1_nomem == t1 || error("validation scan $k: the two RL arms saw different phantoms")
    plans = [hist, nomem,
             Plan(:log_grid, envs.baseline, log_grid_actions(envs.baseline.time_budget_s, envs.baseline.Npe), no_record),
             Plan(:cr_optimal, envs.baseline, cr_actions(), no_record)]
    plans, t1
end

"""best_log_grid_schedule with the action round trip grid_tr_sweep.jl ran it with, and that sweep's
record of held-out scan k. The sweep stops unless every block of its capped grids ran once at its
nominal TR, so one pass of the schedule is what it executed."""
function best_grid_actions(env::E2Env, k::Int)
    sw = JSON.parsefile(GRID_SWEEP)
    basename(sw["env_run"]) == BASELINE_RUN || error("grid_tr_sweep.json ran another run's env")
    point = only(p for p in sw["series"]["capped"] if p["TR_s"] == BEST_GRID_TR_S)
    sched = best_log_grid_schedule(env)
    sched.TIs == point["schedule"]["TI_s"] && sched.TRs == point["schedule"]["TR_s"] ||
        error("best_log_grid_schedule is not grid_tr_sweep.json's capped TR $(BEST_GRID_TR_S) s schedule")
    rec = point["scans"][k + 1]
    Int(rec["seed"]) == HELDOUT_SEED0 + k || error("grid_tr_sweep.json scan $k is not seed $(HELDOUT_SEED0 + k)")
    numpy2 = sw["numpy_promotion"] == "numpy2"
    actions = [python_action(env, ti, tr; learn_alpha = sw["learn_alpha_action_layout"], numpy2 = numpy2)
               for (ti, tr) in zip(sched.TIs, sched.TRs)]
    actions, rec
end

function heldout_plans(envs, k::Int; best_grid::Bool = false)
    ad = JSON.parsefile(joinpath(CONTROL_DATA, "control_adaptive_heldout.json"))["episodes"][k + 1]
    Int(ad["seed"]) == HELDOUT_SEED0 + k || error("control_adaptive_heldout.json episode $k is not seed $(HELDOUT_SEED0 + k)")
    ctrl = JSON.parsefile(joinpath(CONTROL_DATA, "control_results.json"))
    fixed = JSON.parsefile(FIXED_HELDOUT)["budgets"]["560"]
    distilled = ctrl["median_protocols"]["fixed_median_validation"]

    function final_check(m, rec)
        r -> begin
            T1_rec = Float64.(rec["T1_est"])
            r.T1_est[end] == T1_rec && r.mape[end] == rec["mape"] ||
                error("held-out scan $k $m: replayed T1 $(r.T1_est[end]) / MAPE $(r.mape[end]) differ from recorded $(T1_rec) / $(rec["mape"])")
            (Dict{String,Any}("final_T1_est_recorded" => T1_rec), Float64(rec["mape"]))
        end
    end
    executed(scan) = [action(x[1], x[2]; alpha_deg = x[3]) for x in scan["executed_TI_TR_alpha"]]
    grid, cr = fixed["log_grid_trmatched"]["scans"][k + 1], fixed["cr_optimal"]["scans"][k + 1]
    grid_actions, grid_rec = best_grid ? best_grid_actions(envs.baseline, k) : (executed(grid), grid)
    plans = [
        Plan(:hist, envs.hist, [action(b["TI"], b["TR"]) for b in ad["blocks"] if b["executed"]],
             final_check(:hist, ctrl["arms"]["adaptive"][k + 1])),
        Plan(:distilled, envs.baseline, [action(ti, tr) for (ti, tr) in zip(distilled["TI_s"], distilled["TR_s"])],
             final_check(:distilled, ctrl["arms"]["fixed_median_validation"][k + 1])),
        Plan(:log_grid, envs.baseline, grid_actions, final_check(:log_grid, grid_rec)),
        Plan(:cr_optimal, envs.baseline, executed(cr), final_check(:cr_optimal, cr)),
    ]
    plans, Float64.(ad["T1_true"])
end

# Spin-count fraction of each pixel belonging to each active sphere; sphere spins are
# identified by their sampled T1, which no other material shares.
function sphere_fractions(env::E2Env)
    ph = env.phantom
    occ = phantom_occupancy(ph, env.Npe, env.Nfe, env.FOV)
    fr = zeros(Float64, env.n_spheres, env.Npe, env.Nfe)
    for i in 1:env.n_spheres
        cnt = phantom_parameter_map(ph, Float64.(ph.T1 .== env.T1_true[i]), env.Npe, env.Nfe, env.FOV) .* occ
        fr[i, :, :] = cnt ./ max.(occ, 1.0)
    end
    fr, occ
end

function sphere_centres_mm(env::E2Env)
    ph = env.phantom
    c = zeros(env.n_spheres, 2)
    for i in 1:env.n_spheres
        sel = ph.T1 .== env.T1_true[i]
        c[i, 1] = 1e3 * sum(ph.x[sel]) / count(sel)
        c[i, 2] = 1e3 * sum(ph.y[sel]) / count(sel)
    end
    c
end

"""Step one protocol through a freshly reset episode; return per-block records."""
function replay(env::E2Env, seed::Int, actions, T1_true_expected)
    e2_reset!(env; rng_seed = seed)
    rel = maximum(abs.(env.T1_true .- T1_true_expected) ./ T1_true_expected)
    rel < 1e-9 || error("seed $seed: phantom T1s $(env.T1_true) differ from recorded $(T1_true_expected)")
    imgs = Matrix{Float32}[]
    T1_est = Vector{Float64}[]
    mape = Float64[]
    snr = Float64[]
    executed = Vector{Float64}[]
    rejected = false
    for a in actions
        env.done && break
        _, _, _, info = e2_step!(env, a)
        if get(info, "budget_exceeded", false)
            rejected = true
            break
        end
        push!(imgs, copy(reshape(env.last_image_mag, env.Npe, env.Nfe)))
        push!(T1_est, copy(env.T1_est))
        push!(mape, info["mape"])
        push!(snr, e2_image_stats(env).snr_peak)
        push!(executed, [info["TI"], info["TE"], info["TR"], info["alpha_deg"]])
    end
    (images = imgs, T1_est = T1_est, mape = mape, snr = snr, actions = executed,
     rejected = rejected, time_s = env.time_used_s)
end

stack3(v) = permutedims(cat(v...; dims = 3), (3, 1, 2))
rows(v) = permutedims(reduce(hcat, v))

function run_scan(tag::AbstractString, k::Int, seed::Int, plans, T1_expected)
    out = Dict{String,Any}()
    for (j, plan) in enumerate(plans)
        t0 = time()
        r = replay(plan.env, seed, plan.actions, T1_expected)
        recorded, rec_mape = plan.check(r)
        p = string(plan.method)
        out["$(p)_images"] = stack3(r.images)
        out["$(p)_T1_est"] = rows(r.T1_est)
        out["$(p)_mape"] = r.mape
        out["$(p)_snr_nema_peak"] = r.snr
        out["$(p)_actions"] = rows(r.actions)
        out["$(p)_rejected_final"] = Int(r.rejected)
        for (key, v) in recorded
            out["$(p)_$(key)"] = v
        end
        if j == 1
            env = plan.env
            ph = env.phantom
            fr, occ = sphere_fractions(env)
            out["T1_true"] = copy(env.T1_true)
            out["sphere_indices"] = copy(env.sphere_indices)
            out["sphere_px"] = rows([[px[1], px[2]] for px in env.sphere_px])
            out["sphere_centre_mm"] = sphere_centres_mm(env)
            out["sphere_radius_mm"] = 1e3 * CONTRAST_RADIUS_M
            out["water_T1_s"] = BACKGROUND_WATER[env.cfg_field].T1
            out["truth_T1"] = phantom_parameter_map(ph, ph.T1, env.Npe, env.Nfe, env.FOV)
            out["occupancy"] = occ
            out["sphere_fraction"] = fr
            out["background_mask"] = Int.(env.background_mask)
            out["fov_m"] = env.FOV
            out["roi_radius"] = env.roi_radius
            out["noise_sigma_abs"] = env.noise_sigma_abs
        end
        @printf("%s %-10s %2d blocks %6.1f s  final MAPE %.4f%s  (%.0f s wall)\n",
                tag, p, length(r.images), r.time_s, r.mape[end],
                rec_mape === nothing ? "" : @sprintf(" (recorded %.4f)", rec_mape), time() - t0)
        flush(stdout)
    end
    out["scan"] = k
    out["seed"] = seed
    mkpath(DATA_DIR)
    npzwrite(replay_path(tag), out)
end

function main(args)
    a = scan_args(args)
    scans = isempty(a.nums) ? collect(0:N_SCANS-1) : a.nums
    envs = (hist = env_from_run(RL_RUNS.hist), nomem = env_from_run(RL_RUNS.nomem),
            baseline = env_from_run(BASELINE_RUN))
    for k in scans
        plans, t1 = a.heldout ? heldout_plans(envs, k; best_grid = a.best_grid) : validation_plans(envs, k)
        seed = (a.heldout ? HELDOUT_SEED0 : VALIDATION_SEED0) + k
        run_scan(scan_tag(k; heldout = a.heldout, best_grid = a.best_grid), k, seed, plans, t1)
    end
end

main(ARGS)

# Roll out exported PPO checkpoints of the TI-histogram run in pure Julia.
#
# The actor (two tanh layers + linear head) and its VecNormalize statistics come
# from fig3evo_export_policies.py. Observations come from the same E2Env code the
# training used (julia/rl/e2.jl); the action decoding mirrors
# python/qalibremd_gym/env_e2.py:_denorm_action for this run's action mode
# (fix_te, log TI, alpha fixed), including its float32 rounding.
#
# Run from the RLxMRI repo root:
#   julia -t 4 --project=. ismrm_latex/vancouver/scripts/fig3evo_rollout.jl CHECKPOINTS FIRST_SCAN N_SCANS [MAX_BLOCKS]
# CHECKPOINTS is comma-separated; global_best is also checked against its
# recorded global_best/diagnostics/episodes.json. The figure uses:
#   julia -t 4 --project=. ismrm_latex/vancouver/scripts/fig3evo_rollout.jl \
#       stage0_analytic_end,stage3_full3_end,global_best 0 24

using JSON
using NPZ
using Printf
using MRISystemPhantom

const ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
const RUN = joinpath(ROOT, "runs", "e2", "mf_runB_5sphere_hist_560s_gpu")
const FIGURES = normpath(joinpath(@__DIR__, "..", "figures"))
const VALIDATION_SEED = 500_000
# env_e2.py fixes these in Python for fix_te runs without learn_alpha.
const FIXED_TE_S = 0.020
const FIXED_ALPHA_DEG = 90.0

include(joinpath(ROOT, "julia", "rl_boot.jl"))

struct Actor
    Ws::Vector{Matrix{Float32}}
    bs::Vector{Vector{Float32}}
    W_out::Matrix{Float32}
    b_out::Vector{Float32}
    obs_mean::Vector{Float64}
    obs_var::Vector{Float64}
    epsilon::Float64
    clip_obs::Float64
end

function load_actor(name::AbstractString)
    d = npzread(joinpath(FIGURES, "fig3evo_policy_$(name).npz"))
    n = count(k -> occursin(r"^W\d+$", k), keys(d))
    Actor([d["W$i"] for i in 1:n], [d["b$i"] for i in 1:n], d["W_out"], d["b_out"],
          d["obs_mean"], d["obs_var"], d["epsilon"], d["clip_obs"])
end

function act(a::Actor, obs::AbstractVector)
    # VecNormalize works in float64; the torch policy then runs in float32.
    x = Float32.(clamp.((Float64.(obs) .- a.obs_mean) ./ sqrt.(a.obs_var .+ a.epsilon), -a.clip_obs, a.clip_obs))
    for (W, b) in zip(a.Ws, a.bs)
        x = tanh.(W * x .+ b)
    end
    clamp.(a.W_out * x .+ a.b_out, -1f0, 1f0)
end

function decode(env::E2Env, a::AbstractVector{Float32})
    lo, hi = Float32.(e2_action_lo(env)), Float32.(e2_action_hi(env))
    u = (a .+ 1f0) ./ 2f0
    ti = Float32(Float64(lo[1]) * (Float64(hi[1]) / Float64(lo[1]))^Float64(u[1]))
    tr = lo[3] + u[2] * (hi[3] - lo[3])
    Float64[ti, Float32(FIXED_TE_S), tr, Float32(FIXED_ALPHA_DEG)]
end

function build_env(kw::AbstractDict)
    kw["fix_te"] && kw["log_ti_action"] && !kw["learn_alpha"] && !kw["simplified_action"] &&
        !kw["allow_stop"] || error("decode() only mirrors the fix_te + log-TI action mode")
    python_only = ("fix_te", "learn_alpha", "log_ti_action", "simplified_action", "use_gpu")
    symbols = ("cfg_field", "t1_sampler", "pose_mode", "reward_mode", "sigma_method", "water_model",
               "forward_model")
    args = Dict{Symbol,Any}()
    for (k, v) in kw
        (k in python_only || v === nothing) && continue
        args[Symbol(k)] = k in symbols ? Symbol(v) :
                          k == "forced_sphere_indices" ? Int.(v) : v
    end
    E2Env(; args..., use_gpu = false, rng_seed = VALIDATION_SEED)
end

function model_vs_data(env::E2Env)
    # Refit each sphere exactly as _e2_update_t1_estimates! does, to keep the amplitude A,
    # then evaluate the fitter's model at every sampled (TI, TR): with the fitted T1, and
    # with the true T1 at its own least-squares amplitude (the fitter's closed form).
    map(1:env.n_spheres) do i
        TIs, TRs, αs, m = env.block_TIs[i], env.block_TRs[i], env.block_α_excs[i], env.block_mags[i]
        fit = fit_t1_generalized_ir(TIs, fill(π, length(TIs)), m; TRs = TRs, α_excs = αs, Npe = env.Npe,
                                    T1_range = env.T1_sample_range, n_grid = env.fitter_n_grid,
                                    abs_noise_sigma = _sin_corrected_abs_noise(env.noise_sigma_abs, αs),
                                    sigma_method = :asymptotic, signed = env.phase_sensitive)
        model(T1) = abs.(transient_mz_at_excite_npe.(T1, TIs, TRs, π, αs; Npe = env.Npe))
        y_true = model(env.T1_true[i])
        A_true = sum(m .* y_true) / sum(y_true .^ 2)
        Dict("T1_fit" => fit.T1, "A_fit" => fit.A, "model_fit" => abs(fit.A) .* model(fit.T1),
             "T1_true" => env.T1_true[i], "A_true" => A_true, "model_true" => A_true .* y_true)
    end
end

function rollout(env::E2Env, actor::Actor, seed::Integer; max_blocks::Integer = typemax(Int))
    obs = e2_reset!(env; rng_seed = seed)
    blocks = Dict{String,Any}[]
    done = false
    while !done && length(blocks) < max_blocks
        a = act(actor, obs)
        t0 = time()
        obs, _, done, info = e2_step!(env, decode(env, a), false)
        push!(blocks, Dict("action" => a, "TI" => info["TI"], "TR" => info["TR"], "mape" => info["mape"],
                           "executed" => !get(info, "budget_exceeded", false),
                           "T1_est" => copy(env.T1_est), "step_s" => time() - t0))
    end
    Dict("seed" => seed, "T1_true" => copy(env.T1_true), "blocks" => blocks,
         "TIs" => copy(env.block_TIs[1]), "TRs" => copy(env.block_TRs[1]),
         "mags" => [copy(m) for m in env.block_mags],
         "model_vs_data" => length(env.block_TIs[1]) >= 2 ? model_vs_data(env) : nothing)
end

function compare_with_recorded(ep::AbstractDict, rec::AbstractDict)
    println("block  TI replay      TI recorded    TR replay      TR recorded    MAPE replay  recorded")
    for (b, blk) in enumerate(ep["blocks"])
        b > length(rec["TI"]) && break
        @printf("%3d    %.10f   %.10f   %.10f   %.10f   %.5f      %.5f\n", b, blk["TI"], rec["TI"][b],
                blk["TR"], rec["TR"][b], blk["mape"], rec["mape"][b])
    end
end

function main(args)
    names = split(args[1], ",")
    first_scan = parse(Int, args[2])
    n_scans = parse(Int, args[3])
    max_blocks = length(args) >= 4 ? parse(Int, args[4]) : typemax(Int)

    env = build_env(JSON.parsefile(joinpath(RUN, "run_config.json"))["base_env_kwargs"])
    for name in names
        actor = load_actor(name)
        recorded = name == "global_best" ?
            JSON.parsefile(joinpath(RUN, "global_best", "diagnostics", "episodes.json")) : nothing
        episodes = Dict{String,Any}[]
        t_ckpt = time()
        for k in first_scan:(first_scan + n_scans - 1)
            t0 = time()
            ep = rollout(env, actor, VALIDATION_SEED + k; max_blocks = max_blocks)
            push!(episodes, ep)
            @printf("%s scan %d: %d blocks, final MAPE %.2f%%, %.1f s\n", name, k, length(ep["blocks"]),
                    100 * ep["blocks"][end]["mape"], time() - t0)
            recorded === nothing || compare_with_recorded(ep, recorded[k + 1])
        end
        out = joinpath(FIGURES, "fig3evo_rollout_$(name)_$(first_scan)-$(first_scan + n_scans - 1).json")
        open(io -> JSON.print(io, Dict("checkpoint" => name, "episodes" => episodes,
                                       "wall_s" => time() - t_ckpt, "threads" => Threads.nthreads())), out, "w")
        println("wrote $out")
    end
end

# control_schedule_swap.jl includes this file for the actor and rollout code.
if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end

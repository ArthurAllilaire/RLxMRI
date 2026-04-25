# E1 — single-sphere T1 estimation with a discrete IR-prep action set.
# PLAN.md §4 E1. The environment holds one episode's progression; Python
# drives it via juliacall (`reset!` → returns obs; `step!` → returns
# (obs, reward, done, info_dict)).
#
# Action space: cartesian product of TI_set × α_set (default 6 × 3 = 18).
# Each action is a parameterised IR-prep block. Signal generation uses
# the closed-form `generalized_ir_signal` by default (training hot path,
# PLAN §7); flipping `backend = :simulate` routes through `ir_sequence`
# + `simulate` for fidelity validation.

"""
    E1Env

Stateful environment struct for E1. Mutable because episodes are driven
step-by-step from Python — see `e1_reset!` / `e1_step!`.
"""
mutable struct E1Env
    # --- static config (set at construction) ---
    TI_set::Vector{Float64}
    α_set::Vector{Float64}              # radians
    action_table::Vector{Tuple{Float64,Float64}}   # (TI, α) per discrete action
    n_adc::Int
    dur_adc::Float64
    max_blocks::Int
    time_budget_s::Float64
    λ_time::Float64
    terminal_bonus::Float64             # awarded iff |err| < success_tol at done
    success_tol::Float64
    T1_sample_range::NTuple{2,Float64}
    T2_factor_range::NTuple{2,Float64}
    backend::Symbol                     # :analytical (default) or :simulate

    # --- episode state (mutated by reset!/step!) ---
    rng::MersenneTwister
    T1_true::Float64
    T2_true::Float64
    TIs_used::Vector{Float64}
    αs_used::Vector{Float64}
    mags_used::Vector{Float64}          # magnitude at first ADC sample
    last_signal::Vector{Float64}        # |ADC| of last block, length n_adc
    action_counts::Vector{Int}          # plays-per-action, length |actions|
    n_blocks::Int
    time_used_s::Float64
    T1_est::Float64
    done::Bool
end

"""
    E1Env(; …)

Construct a fresh environment. All tunables are keyword arguments so
Python can override per experiment. Defaults match PLAN.md §4 E1.
"""
function E1Env(;
    TI_set::AbstractVector{<:Real} =
        [10e-3, 30e-3, 100e-3, 300e-3, 1000e-3, 3000e-3],
    α_set_deg::AbstractVector{<:Real} = [10.0, 90.0, 180.0],
    n_adc::Int = 64,
    dur_adc::Real = 2e-3,
    max_blocks::Int = 12,
    time_budget_s::Real = 20.0,
    λ_time::Real = 1.0,
    terminal_bonus::Real = 1.0,
    success_tol::Real = 0.03,
    T1_sample_range::NTuple{2,<:Real} = (20e-3, 2.0),
    T2_factor_range::NTuple{2,<:Real} = (0.3, 1.0),
    backend::Symbol = :analytical,
    rng_seed::Integer = 0,
)
    TIs = Float64.(collect(TI_set))
    αs  = Float64.(deg2rad.(collect(α_set_deg)))
    action_table = [(ti, α) for ti in TIs for α in αs]
    n_actions = length(action_table)
    backend ∈ (:analytical, :simulate) ||
        error("backend must be :analytical or :simulate")

    E1Env(
        TIs, αs, action_table,
        Int(n_adc), Float64(dur_adc),
        Int(max_blocks), Float64(time_budget_s),
        Float64(λ_time), Float64(terminal_bonus), Float64(success_tol),
        Float64.(T1_sample_range), Float64.(T2_factor_range), backend,
        MersenneTwister(rng_seed),
        NaN, NaN,
        Float64[], Float64[], Float64[],
        zeros(Float64, n_adc), zeros(Int, n_actions),
        0, 0.0, NaN, false,
    )
end

"Number of discrete actions (|TI_set| × |α_set|)."
e1_n_actions(env::E1Env) = length(env.action_table)

"""
    e1_obs_dim(env) -> Int

Length of the observation vector. See `_e1_observation` for layout.
"""
e1_obs_dim(env::E1Env) = env.n_adc + 3 + e1_n_actions(env)

"""
    e1_action_table(env)

Return the (TI, α) tuple list so Python can render / debug. `action_index
- 1` in Python → this index.
"""
e1_action_table(env::E1Env) = env.action_table

# Observation layout: [last_signal (n_adc), T1_est_log (1), blocks_frac (1),
#                      time_frac (1), action_counts_normalised (n_actions)]
function _e1_observation(env::E1Env)
    # Normalise last signal by its own peak (pure shape information).
    pk = maximum(env.last_signal)
    sig = pk > 0 ? env.last_signal ./ pk : env.last_signal
    T1_est = isnan(env.T1_est) ? 1.0 : env.T1_est
    T1_est_log = log10(clamp(T1_est, 1e-3, 10.0))
    n_frac     = env.n_blocks / env.max_blocks
    t_frac     = min(1.0, env.time_used_s / env.time_budget_s)
    counts_norm = env.action_counts ./ max(env.max_blocks, 1)
    vcat(Float64.(sig), Float64(T1_est_log), Float64(n_frac), Float64(t_frac),
         Float64.(counts_norm))
end

"Approximate simulated duration of a block (TI dominates; 10 ms RF+ADC overhead)."
@inline _block_duration(TI::Real) = TI + 0.01

function _block_signal(env::E1Env, TI::Real, α::Real)
    if env.backend === :analytical
        return generalized_ir_signal(env.T1_true, env.T2_true;
                                     TI = TI, α = α,
                                     n_adc = env.n_adc,
                                     dur_adc = env.dur_adc)
    else
        # :simulate — route through KomaMRI on a single-spin phantom.
        # amp_T is scaled so a π/2 pulse fits inside the TI envelope.
        amp_T = 20e-6
        obj = single_spin_phantom(T1 = env.T1_true, T2 = env.T2_true)
        d_prep = rf_duration(α;   amp_T = amp_T)
        d_exc  = rf_duration(π/2; amp_T = amp_T)
        seq  = Sequence()
        seq += RF(amp_T, d_prep)
        seq += Delay(max(TI - d_prep/2 - d_exc/2, 1e-6))
        seq += RF(amp_T, d_exc)
        seq += ADC(env.n_adc, env.dur_adc, 0.0)
        raw = Suppressor.@suppress simulate(obj, seq, Scanner())
        return Float64.(abs.(raw.profiles[1].data[:, 1]))
    end
end

"""
    e1_reset!(env; rng_seed = nothing) -> obs

Start a new episode. Re-seeds the env RNG if `rng_seed` is given
(useful for deterministic evaluation). Samples (T1_true, T2_true) per
`T1_sample_range` (log-uniform) and `T2_factor_range` (uniform).
"""
function e1_reset!(env::E1Env; rng_seed::Union{Nothing,Integer} = nothing)
    if rng_seed !== nothing
        env.rng = MersenneTwister(rng_seed)
    end

    # log-uniform T1
    lo, hi  = log.(env.T1_sample_range)
    env.T1_true = exp(lo + rand(env.rng) * (hi - lo))

    # T2 = u·T1, u uniform in T2_factor_range (so T2 ≤ T1, physical)
    u = env.T2_factor_range[1] +
        rand(env.rng) * (env.T2_factor_range[2] - env.T2_factor_range[1])
    env.T2_true = u * env.T1_true

    empty!(env.TIs_used)
    empty!(env.αs_used)
    empty!(env.mags_used)
    fill!(env.last_signal, 0.0)
    fill!(env.action_counts, 0)
    env.n_blocks    = 0
    env.time_used_s = 0.0
    env.T1_est      = NaN
    env.done        = false

    _e1_observation(env)
end

"""
    e1_step!(env, action) -> (obs, reward, done, info)

Play one block (1-based `action` index) on the current episode. `info`
is a plain `Dict{String,Any}` so it serialises cleanly to Python.
"""
function e1_step!(env::E1Env, action::Integer)
    env.done && error("Episode already done; call e1_reset! first.")
    a = Int(action)
    (1 ≤ a ≤ e1_n_actions(env)) ||
        error("Action $a out of range 1..$(e1_n_actions(env))")

    TI, α = env.action_table[a]

    sig = _block_signal(env, TI, α)
    env.last_signal = sig
    push!(env.TIs_used, TI)
    push!(env.αs_used, α)
    push!(env.mags_used, sig[1])
    env.action_counts[a] += 1
    env.n_blocks += 1
    block_time = _block_duration(TI)
    env.time_used_s += block_time

    # Update running T1 estimate — needs ≥ 2 data points.
    if length(env.mags_used) >= 2
        fit = fit_t1_generalized_ir(env.TIs_used, env.αs_used, env.mags_used;
                                    T1_range = env.T1_sample_range,
                                    n_grid = 200)
        env.T1_est = fit.T1
    else
        env.T1_est = sqrt(prod(env.T1_sample_range))   # neutral prior
    end

    err     = abs(env.T1_est - env.T1_true) / env.T1_true
    reward  = -err - env.λ_time * block_time / env.time_budget_s

    if env.n_blocks >= env.max_blocks || env.time_used_s >= env.time_budget_s
        env.done = true
        err < env.success_tol && (reward += env.terminal_bonus)
    end

    info = Dict{String,Any}(
        "T1_true"  => env.T1_true,
        "T2_true"  => env.T2_true,
        "T1_est"   => env.T1_est,
        "err"      => err,
        "n_blocks" => env.n_blocks,
        "time_s"   => env.time_used_s,
        "TI"       => TI,
        "alpha_deg"=> rad2deg(α),
    )
    obs = _e1_observation(env)
    (obs, reward, env.done, info)
end

# Aliases for juliacall (Python can't reach names ending in `!`).
const e1_reset_b = e1_reset!
const e1_step_b  = e1_step!

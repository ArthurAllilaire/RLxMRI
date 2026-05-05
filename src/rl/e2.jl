# E2 — Full-plate multi-sphere T1 mapping with gradient-encoded 2D imaging.
# PLAN.md §4 E2 / E2_PLAN.md.
#
# The environment wraps a full T1-plate phantom (14 spheres, coarse voxels).
# At each step the agent submits a 5-parameter IR-SE action; the env builds a
# 2D gradient-encoded sequence, calls KomaMRI.simulate(), adds complex
# Gaussian noise, reconstructs a magnitude image, extracts per-sphere ROI
# signals, and updates running T1 estimates (Levenberg-Marquardt grid search).
# Reward = −mean MAPE across 14 spheres (dense, per step).

import FFTW: ifft

"""
    E2Env

Stateful environment for E2. Python drives it step-by-step via
`e2_reset!` / `e2_step!`.
"""
mutable struct E2Env
    # ── static config ────────────────────────────────────────────────────────
    cfg_field::Symbol                      # :T3 or :T15
    voxel_size_mm::Float64                 # phantom voxelisation
    FOV::Float64                           # image field of view [m]
    Nfe::Int                               # frequency-encode samples
    Npe::Int                               # phase-encode steps
    n_spheres::Int                         # 14 for the T1 plate
    max_blocks::Int
    time_budget_s::Float64
    terminal_bonus::Float64
    success_tol::Float64                   # MAPE threshold for terminal bonus
    noise_sigma_rel::Float64               # noise σ as fraction of signal RMS
    T1_sigma_rel::Float64                  # per-sphere T1 jitter (relative)
    translation_sigma_mm::Float64          # pose translation σ per axis [mm]
    rotation_sigma_rad::Float64            # pose rotation σ per axis [rad]
    T1_sample_range::NTuple{2,Float64}     # T1 search range for fitting [s]
    reward_mode::Symbol                    # :neg_mape | :delta_mape

    # ── sphere base info (fixed at construction) ──────────────────────────
    sphere_centres_base::Vector{NTuple{3,Float64}}  # original centres [m]
    T1_base::Vector{Float64}               # nominal T1 at cfg_field [s]
    T2_ratio::Vector{Float64}              # T2/T1 ratio per sphere

    # ── episode state ────────────────────────────────────────────────────────
    rng::MersenneTwister
    phantom::Any                           # KomaMRI Phantom (cached per episode)
    T1_true::Vector{Float64}               # per-sphere true T1 this episode
    sphere_px::Vector{NTuple{2,Int}}       # (i_pe, i_fe) per sphere in image
    episode_rotation::NTuple{3,Float64}    # Euler XYZ [rad]
    episode_translation_m::NTuple{3,Float64}  # [m]

    # accumulated per sphere
    block_TIs::Vector{Vector{Float64}}     # TI values used, per sphere
    block_mags::Vector{Vector{Float64}}    # magnitudes (sin(α_exc)-corrected), per sphere
    T1_est::Vector{Float64}                # running T1 estimate per sphere

    # episode progress
    n_blocks::Int
    time_used_s::Float64
    done::Bool
    prev_mape::Float64                     # for :delta_mape reward shaping

    # last image (for observation)
    last_image_mag::Vector{Float32}        # flattened [Npe × Nfe] magnitude
end

"""
    E2Env(; …)

Construct a fresh E2 environment. Defaults match E2_PLAN.md training config.
"""
function E2Env(;
    cfg_field::Symbol             = :T3,
    voxel_size_mm::Real           = 3.0,
    FOV::Real                     = 0.2,
    Nfe::Int                      = 16,
    Npe::Int                      = 8,
    max_blocks::Int                = 15,
    time_budget_s::Real            = 120.0,
    terminal_bonus::Real           = 0.5,
    success_tol::Real              = 0.05,
    noise_sigma_rel::Real          = 0.05,
    T1_sigma_rel::Real             = 0.05,
    translation_sigma_mm::Real     = 5.0,
    rotation_sigma_rad::Real       = 0.15,   # ~8.6°
    T1_sample_range::NTuple{2,<:Real} = (0.01, 3.0),
    reward_mode::Symbol            = :neg_mape,
    rng_seed::Integer              = 0,
)
    cfg_field ∈ (:T3, :T15) || error("cfg_field must be :T3 or :T15")
    reward_mode ∈ (:neg_mape, :delta_mape) ||
        error("reward_mode must be :neg_mape or :delta_mape")

    # Base sphere info (no rotation/translation/jitter)
    base_cfg   = PhantomConfig(field = cfg_field, voxel_size_mm = 1.0,
                                include_plates = [:T1])
    base_descs = sphere_descriptors(:T1, base_cfg;
                                    rng = MersenneTwister(0))
    n_spheres  = length(base_descs)
    centres    = [d.centre for d in base_descs]
    T1_base    = [d.T1 for d in base_descs]
    T2_ratio   = T2_OF_T1_ARRAY[cfg_field] ./ T1_ARRAY[cfg_field]

    E2Env(
        cfg_field, Float64(voxel_size_mm),
        Float64(FOV), Nfe, Npe, n_spheres,
        Int(max_blocks), Float64(time_budget_s),
        Float64(terminal_bonus), Float64(success_tol),
        Float64(noise_sigma_rel), Float64(T1_sigma_rel),
        Float64(translation_sigma_mm), Float64(rotation_sigma_rad),
        Float64.(T1_sample_range),
        reward_mode,
        centres, Float64.(T1_base), Float64.(T2_ratio),
        MersenneTwister(rng_seed),
        nothing,                            # phantom (filled at reset)
        zeros(n_spheres), fill((1, 1), n_spheres),
        (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
        [Float64[] for _ in 1:n_spheres],
        [Float64[] for _ in 1:n_spheres],
        zeros(n_spheres),
        0, 0.0, false, 0.0,
        zeros(Float32, Npe * Nfe),
    )
end

"Observation dimension: image (Nfe*Npe) + T1 estimates (n_spheres) + budget (3)."
e2_obs_dim(env::E2Env) = env.Nfe * env.Npe + env.n_spheres + 3

"Action space bounds: [TI_lo, TE_lo, TR_lo, α_deg_lo, slice_z_mm_lo], same for hi."
e2_action_lo(::E2Env) = Float64[0.010, 0.005, 0.5,   5.0,  -60.0]
e2_action_hi(::E2Env) = Float64[3.000, 0.080, 5.0, 180.0,   60.0]

# ── Internal helpers ─────────────────────────────────────────────────────────

function _e2_build_episode_phantom(env::E2Env, rng_seed::Int)
    rng_ep = MersenneTwister(rng_seed)

    # Per-sphere T1 jitter (log-normal, centred on nominal values)
    T1_ep = [env.T1_base[i] * exp(env.T1_sigma_rel * randn(rng_ep))
             for i in 1:env.n_spheres]

    # Per-sphere T2 (preserve T2/T1 ratio)
    T2_ep = T1_ep .* env.T2_ratio

    # Build custom sphere map: override T1 and T2 per sphere
    base_cfg  = PhantomConfig(field = env.cfg_field, voxel_size_mm = 1.0,
                               include_plates = [:T1])
    base_descs = sphere_descriptors(:T1, base_cfg; rng = MersenneTwister(0))
    cmap = Dict{Symbol,Any}()
    for (i, d) in enumerate(base_descs)
        cmap[d.label] = SphereDescriptor(
            d.centre, d.radius, d.ρ,
            T1_ep[i], T2_ep[i], T2_ep[i], d.delta_w, d.label,
        )
    end

    # Episode rotation and translation (domain randomisation)
    rx = env.rotation_sigma_rad * randn(rng_ep)
    ry = env.rotation_sigma_rad * randn(rng_ep)
    rz = env.rotation_sigma_rad * randn(rng_ep)
    tx = env.translation_sigma_mm * randn(rng_ep)
    ty = env.translation_sigma_mm * randn(rng_ep)
    tz = env.translation_sigma_mm * randn(rng_ep)

    cfg = PhantomConfig(
        field                = env.cfg_field,
        voxel_size_mm        = env.voxel_size_mm,
        include_plates       = [:T1],
        rotation             = (rx, ry, rz),
        translation_mm       = (tx, ty, tz),
        augment              = AugmentConfig(B0_sigma_Hz = 5.0),
        custom_sphere_map    = cmap,
        rng_seed             = rng_seed,
    )
    phantom = build_phantom(cfg)

    # Compute transformed sphere centres for pixel mapping
    R = rotation_matrix(rx, ry, rz)
    t_m = [tx, ty, tz] .* 1e-3
    sphere_px = NTuple{2,Int}[]
    for c in env.sphere_centres_base
        c_trans = R * collect(c) .+ t_m
        cx, cy = c_trans[1], c_trans[2]
        ife = mod(round(Int, cx * env.Nfe / env.FOV), env.Nfe) + 1
        ipe = mod(round(Int, cy * env.Npe / env.FOV), env.Npe) + 1
        push!(sphere_px, (ipe, ife))
    end

    phantom, T1_ep, sphere_px,
    (rx, ry, rz), (tx * 1e-3, ty * 1e-3, tz * 1e-3)
end

function _e2_observation(env::E2Env)
    # 1. Normalised magnitude image (flat)
    img_max = maximum(env.last_image_mag)
    img_norm = img_max > 0f0 ? env.last_image_mag ./ img_max :
                                env.last_image_mag

    # 2. Running T1 estimates (log10 scale, 0 for uninitialised)
    t1_obs = [isnan(env.T1_est[i]) ? 0f0 :
              Float32(log10(clamp(env.T1_est[i], 1e-4, 10.0)))
              for i in 1:env.n_spheres]

    # 3. Budget state
    t_frac  = Float32(min(1.0, env.time_used_s / env.time_budget_s))
    n_frac  = Float32(env.n_blocks / env.max_blocks)
    bgt     = Float32[t_frac, n_frac, 1.0f0]

    vcat(Float32.(img_norm), Float32.(t1_obs), bgt)
end

function _e2_simulate_step(env::E2Env, TI::Real, TE::Real, TR::Real,
                            α_exc_deg::Real)
    α_exc = deg2rad(Float64(α_exc_deg))
    seq = Suppressor.@suppress ir_se_2d_sequence(
        TI, TE, TR;
        α_exc = α_exc,
        FOV   = env.FOV,
        Nfe   = env.Nfe,
        Npe   = env.Npe,
    )
    raw = Suppressor.@suppress simulate(env.phantom, seq, Scanner())

    # Build k-space: rows = PE steps, cols = FE samples
    Npe, Nfe = env.Npe, env.Nfe
    ksp = zeros(ComplexF32, Npe, Nfe)
    for k in 1:Npe
        if k <= length(raw.profiles)
            ksp[k, :] = ComplexF32.(raw.profiles[k].data[:, 1])
        end
    end

    # Add complex Gaussian noise (relative to signal RMS)
    all_sig  = vec(ksp)
    sig_rms  = sqrt(sum(abs2, all_sig) / length(all_sig))
    σ = Float32(env.noise_sigma_rel) * sig_rms
    if σ > 0
        ksp .+= σ .* (randn(Float32, Npe, Nfe) .+ im .* randn(Float32, Npe, Nfe))
    end

    # TODO: UNDERSTAND THIS + what the raw output of simulate acc is
    # Reconstruct magnitude image via 2D IFFT
    # abs.(ifft(ksp)) gives |m(x,y)| with x=(i_fe-1)*FOV/Nfe, y=(i_pe-1)*FOV/Npe
    image_mag = Float32.(abs.(ifft(ksp, (1, 2))))

    image_mag, ksp
end

function _e2_update_t1_estimates!(env::E2Env, image_mag::Matrix{Float32},
                                   TI::Real, α_exc::Real)
    # Excitation flip angle scales transverse signal by sin(α_exc) per shot.
    # The fitter uses a single amplitude A across all TIs, so we normalise
    # the recorded magnitude by sin(α_exc) here. Floor at 1e-3 avoids divide-
    # by-zero for near-zero excitation (action lower bound is 5° → sin≈0.087).
    sin_α = max(abs(sin(Float64(α_exc))), 1e-3)
    for i in 1:env.n_spheres
        ipe, ife = env.sphere_px[i]
        mag_i = Float64(image_mag[ipe, ife]) / sin_α
        push!(env.block_TIs[i],   Float64(TI))
        push!(env.block_mags[i],  mag_i)

        if length(env.block_TIs[i]) >= 2
            # α = π for standard 180° inversion prep
            αs = fill(π, length(env.block_TIs[i]))
            fit = fit_t1_generalized_ir(
                env.block_TIs[i], αs, env.block_mags[i];
                T1_range = env.T1_sample_range, n_grid = 200,
            )
            env.T1_est[i] = fit.T1
        else
            # Neutral prior: geometric mean of sample range
            env.T1_est[i] = sqrt(env.T1_sample_range[1] * env.T1_sample_range[2])
        end
    end
end

function _e2_mape(env::E2Env)
    # Mean absolute percentage error across spheres with a valid estimate
    errs = [abs(env.T1_est[i] - env.T1_true[i]) / env.T1_true[i]
            for i in 1:env.n_spheres]
    mean(errs)
end

# ── Public API ────────────────────────────────────────────────────────────────

"""
    e2_reset!(env; rng_seed = nothing) → obs_vector

Start a new episode. Rebuilds the phantom with fresh domain randomisation.
"""
function e2_reset!(env::E2Env; rng_seed::Union{Nothing,Integer} = nothing)
    if rng_seed !== nothing
        env.rng = MersenneTwister(Int(rng_seed))
    end
    ep_seed = rand(env.rng, 0:typemax(Int32))

    phantom, T1_ep, sphere_px, rot, trans =
        _e2_build_episode_phantom(env, ep_seed)

    env.phantom              = phantom
    env.T1_true              = T1_ep
    env.sphere_px            = sphere_px
    env.episode_rotation     = rot
    env.episode_translation_m = trans

    for i in 1:env.n_spheres
        empty!(env.block_TIs[i])
        empty!(env.block_mags[i])
        env.T1_est[i] = NaN
    end
    fill!(env.last_image_mag, 0f0)
    env.n_blocks    = 0
    env.time_used_s = 0.0
    env.done        = false
    env.prev_mape   = 1.0   # neutral prior for delta-MAPE shaping (≈ 100% error)

    _e2_observation(env)
end

"""
    e2_step!(env, action_vec) → (obs, reward, done, info_dict)

`action_vec` is a 5-element vector [TI_s, TE_s, TR_s, α_deg, slice_z_mm].
All values should lie within `e2_action_lo`/`e2_action_hi`.
"""
function e2_step!(env::E2Env, action_vec)
    env.done && error("Episode done; call e2_reset! first.")

    TI        = Float64(action_vec[1])
    TE        = Float64(action_vec[2])
    TR        = Float64(action_vec[3])
    α_exc_deg = Float64(action_vec[4])
    # action_vec[5] = slice_z_mm — stored but not yet used in sequence

    # Ensure TR can accommodate the requested TI + TE with recovery headroom.
    # Lift TR up rather than capping TI down — capping TI silently removed
    # the long-T1 regime when the agent chose a small TR (E1-style failure).
    TR = max(TR, (TI + TE) / 0.90)
    TE = min(TE, TR * 0.30)

    # Simulate and reconstruct
    image_mag, _ksp = _e2_simulate_step(env, TI, TE, TR, α_exc_deg)

    # Store image and update T1 estimates
    env.last_image_mag = vec(image_mag)
    _e2_update_t1_estimates!(env, image_mag, TI, deg2rad(α_exc_deg))

    # Scan time for this block (Npe shots × TR per shot)
    block_time      = env.Npe * TR
    env.time_used_s += block_time
    env.n_blocks    += 1

    # Reward shaping
    #   :neg_mape   → r_t = −MAPE_t                       (dense per-step level)
    #   :delta_mape → r_t = MAPE_{t-1} − MAPE_t           (dense per-step progress)
    # Per-step delta-MAPE rewards informational *progress* and assigns zero
    # reward to redundant actions, breaking the E1-style degenerate-policy
    # equilibrium where the agent collects the same uninformative samples.
    mape = env.n_blocks >= 2 ? _e2_mape(env) : 1.0
    if env.reward_mode === :delta_mape
        reward = env.prev_mape - mape
    else                          # :neg_mape (legacy E2 behaviour)
        reward = -(env.n_blocks >= 2 ? mape : 0.0)
    end
    env.prev_mape = mape

    # Episode termination
    if env.n_blocks >= env.max_blocks || env.time_used_s >= env.time_budget_s
        env.done = true
        if env.n_blocks >= 2 && _e2_mape(env) < env.success_tol
            reward += env.terminal_bonus
        end
    end

    info = Dict{String,Any}(
        "mape"        => mape,
        "T1_true"     => copy(env.T1_true),
        "T1_est"      => copy(env.T1_est),
        "n_blocks"    => env.n_blocks,
        "time_s"      => env.time_used_s,
        "block_time"  => block_time,
        "TI"          => TI,
        "TE"          => TE,
        "TR"          => TR,
        "alpha_deg"   => α_exc_deg,
    )

    obs = _e2_observation(env)
    (obs, reward, env.done, info)
end

# Aliases for juliacall (Python can't reach names ending in `!`)
const e2_reset_b = e2_reset!
const e2_step_b  = e2_step!

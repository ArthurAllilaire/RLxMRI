# E2 — Full-plate multi-sphere T1 mapping with gradient-encoded 2D imaging.
# PLAN.md §4 E2 / E2_PLAN.md.
#
# The environment wraps a full T1-plate phantom (14 spheres, coarse voxels).
# At each step the agent submits a 5-parameter IR-SE action; the env builds a
# 2D gradient-encoded sequence, calls KomaMRI.simulate(), adds complex
# Gaussian noise, reconstructs a magnitude image, extracts per-sphere ROI
# signals, and updates running T1 estimates (Levenberg-Marquardt grid search).
# Reward = −mean MAPE across 14 spheres (dense, per step).
#
# ── Physical assumptions and contingencies (see EXPERT_REPORT_E2_4.md §12) ──
# * Action range TI ∈ [0.01, 3.0] s, TR ∈ [0.5, 5.0] s — set in env_e2.py.
#   - TI lower bound is well above the physics floor: at amp_T = 20 μT
#     (default in `ir_se_2d_sequence`), the inversion pulse is d180 ≈ 0.59 ms
#     and the 90° excitation is d90 ≈ 0.29 ms (rf_duration = α / (2π·γ·B1)
#     with γ = 42.58 MHz/T). Strict pulse-non-overlap requires TI ≥ ~0.5 ms;
#     the 10 ms floor adds ~20× cushion for gradient activity, simulator
#     stability, and clinical relevance. Lower bound does NOT bind for the
#     shortest phantom T1: T1 = 0.024 s → optimal TI = T1·ln(2) ≈ 0.017 s,
#     already above the floor. The floor IS binding for hypothetical
#     T1 < 0.007 s spheres, which we don't have.
#   - TI upper bound 3.0 s is past optimal for the longest sphere
#     (T1 = 1.84 s → TI_opt ≈ 1.28 s); higher TIs would be post-saturation
#     and add no T1 information.
#   - TR upper bound 5.0 s is "5·T1_max" — gives ≥99 % Mz recovery between
#     blocks under the legacy steady-state convention. F1+ uses this less
#     aggressively (cross-shot recovery is now modelled, not assumed).
#
# * Noise (line 260): σ_kspace = noise_sigma_rel × RMS(full ksp). This is a
#   simulation hack to keep noise meaningful across phantom configs — NOT a
#   physical claim. Real MRI noise is hardware-determined (thermal,
#   ∝ √(k_B·T·BW·R) / G), absolute, scene-independent. In practice our
#   model is roughly absolute-noise-like because RMS(ksp) is dominated by
#   the loudest spheres (long-T1 in the inverted regime), so per-pixel σ
#   ends up ~constant — but if the phantom composition changed, the per-
#   sphere effective SNR would shift. Sim-to-real comparisons need to
#   either (a) replace this with an absolute σ_kspace constant, or (b)
#   calibrate noise_sigma_rel against a reference scan.
#
# * Forward model (`fits.jl::transient_mz_at_excite_npe`, F1+) assumes
#   perfect transverse spoiling between PE rows. Holds for our test
#   phantom (T2 = 20 ms ≪ TR − TI typically). Breaks on real-T2 tissue
#   (T2 ≈ 80–2200 ms) — would need EPG (E2_4_PLAN.md §2.5).
#
# * Single-compartment relaxation. One (T1, T2) per voxel. No MT, no
#   off-resonance distribution, no slice-profile, no B1 inhomogeneity.
#
# * Sequential PE ordering. F1+'s "average over Npe shots" assumes uniform
#   PE encoding (all rows equally weighted). Centric or other orderings
#   would change the per-shot weighting in the analytic forward model.
#
# * One TI per block, shared across all 14 spheres. No per-sphere
#   targeting — `slice_z` axis exists in the action but is unused.
#   Structural limitation; addressable only via slice-selective excitation
#   or a multi-action-per-block MDP (E3+).
#
# * abs() in the signal magnitude (no phase-sensitive reconstruction)
#   creates multimodal SSE in the fit. Two T1 values typically give
#   equivalent |S| at saturated TIs → LM solver picks a basin by init.
#   Profile-likelihood σ (E2_5_PLAN.md §3) reports the resulting ambiguity
#   honestly without changing the point estimate.
#
# * Fitter noise floor (line 298) currently uses noise_sigma_rel ×
#   |first-block magnitude per sphere| — per-sphere relative. Because
#   the env's effective noise floor is roughly absolute (set by scene
#   RMS), this UNDER-estimates σ for short-T1 spheres by 5–30×. To be
#   replaced with profile-likelihood σ (E2_5_PLAN.md §3); using a scene-
#   level absolute floor would also work but overfits to the env's
#   specific noise model.

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
    n_spheres::Int                         # active spheres per episode
    subset_size::Union{Nothing,Int}        # nothing = all spheres; k = random k-subset
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
    mape_alpha::Float64                    # MAPE aggregation: α·mean + (1−α)·max
    phase_sensitive::Bool                  # false = magnitude image (clinical default,
                                            # creates abs() multimodal SSE — see §14 of
                                            # cr_explainer.md). true = signed real-part
                                            # reconstruction (sim-only, eliminates abs()
                                            # ambiguity but assumes a known reference phase
                                            # that real scanners need a calibration scan
                                            # for). Default false for back-compat with
                                            # all V1–V5 runs.
    sigma_method::Symbol                   # :asymptotic (default, V1–V5 compat) |
                                            # :profile_likelihood | :bootstrap. Controls
                                            # how `T1_sigma` is computed in the fitter.

    # ── sphere pool info (fixed at construction) ──────────────────────────
    sphere_centres_pool::Vector{NTuple{3,Float64}}  # all original centres [m]
    T1_base_pool::Vector{Float64}          # nominal pool T1 at cfg_field [s]
    T2_ratio_pool::Vector{Float64}         # T2/T1 ratio per pool sphere

    # ── active sphere info (changes at reset when subset_size is set) ─────
    sphere_indices::Vector{Int}            # 1-based indices into the 14-sphere pool
    sphere_centres_base::Vector{NTuple{3,Float64}}
    T1_base::Vector{Float64}
    T2_ratio::Vector{Float64}

    # ── episode state ────────────────────────────────────────────────────────
    rng::MersenneTwister
    phantom::Any                           # KomaMRI Phantom (cached per episode)
    T1_true::Vector{Float64}               # per-sphere true T1 this episode
    sphere_px::Vector{NTuple{2,Int}}       # (i_pe, i_fe) per sphere in image
    episode_rotation::NTuple{3,Float64}    # Euler XYZ [rad]
    episode_translation_m::NTuple{3,Float64}  # [m]

    # accumulated per sphere
    block_TIs::Vector{Vector{Float64}}     # TI values used, per sphere
    block_TRs::Vector{Vector{Float64}}     # TR values used, per sphere (for steady-state fit)
    block_α_excs::Vector{Vector{Float64}}  # excitation flip angles [rad], per sphere
    block_mags::Vector{Vector{Float64}}    # magnitudes (sin(α_exc)-corrected), per sphere
    T1_est::Vector{Float64}                # running T1 estimate per sphere
    T1_sigma::Vector{Float64}              # asymptotic σ on T1_est (NaN until ≥2 samples)

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
    mape_alpha::Real               = 1.0,
    rng_seed::Integer              = 0,
    phase_sensitive::Bool          = false,            # cr_explainer.md §14
    sigma_method::Symbol           = :profile_likelihood,  # E2_5_PLAN.md §3
    subset_size::Union{Nothing,Integer} = nothing,
)
    cfg_field ∈ (:T3, :T15) || error("cfg_field must be :T3 or :T15")
    reward_mode ∈ (:neg_mape, :delta_mape) ||
        error("reward_mode must be :neg_mape or :delta_mape")
    sigma_method ∈ (:asymptotic, :profile_likelihood, :bootstrap) ||
        error("sigma_method must be :asymptotic, :profile_likelihood, or :bootstrap")
    0.0 <= Float64(mape_alpha) <= 1.0 ||
        error("mape_alpha must be in [0, 1]")

    # Base sphere info (no rotation/translation/jitter)
    base_cfg   = PhantomConfig(field = cfg_field, voxel_size_mm = 1.0,
                                include_plates = [:T1])
    base_descs = sphere_descriptors(:T1, base_cfg;
                                    rng = MersenneTwister(0))
    n_pool      = length(base_descs)
    subset_size !== nothing &&
        (1 <= Int(subset_size) <= n_pool ||
         error("subset_size must be between 1 and $n_pool, or nothing"))
    n_spheres  = subset_size === nothing ? n_pool : Int(subset_size)
    pool_centres = [d.centre for d in base_descs]
    pool_T1      = [d.T1 for d in base_descs]
    pool_ratio   = Float64.(T2_OF_T1_ARRAY[cfg_field] ./ T1_ARRAY[cfg_field])
    active_idx   = collect(1:n_spheres)
    centres      = pool_centres[active_idx]
    T1_base      = pool_T1[active_idx]
    T2_ratio     = pool_ratio[active_idx]

    E2Env(
        cfg_field, Float64(voxel_size_mm),
        Float64(FOV), Nfe, Npe, n_spheres, subset_size === nothing ? nothing : n_spheres,
        Int(max_blocks), Float64(time_budget_s),
        Float64(terminal_bonus), Float64(success_tol),
        Float64(noise_sigma_rel), Float64(T1_sigma_rel),
        Float64(translation_sigma_mm), Float64(rotation_sigma_rad),
        Float64.(T1_sample_range),
        reward_mode,
        Float64(mape_alpha),
        Bool(phase_sensitive),
        sigma_method,
        pool_centres, Float64.(pool_T1), pool_ratio,
        active_idx, centres, Float64.(T1_base), T2_ratio,
        MersenneTwister(rng_seed),
        nothing,                            # phantom (filled at reset)
        zeros(n_spheres), fill((1, 1), n_spheres),
        (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
        [Float64[] for _ in 1:n_spheres],   # block_TIs
        [Float64[] for _ in 1:n_spheres],   # block_TRs
        [Float64[] for _ in 1:n_spheres],   # block_α_excs
        [Float64[] for _ in 1:n_spheres],   # block_mags
        zeros(n_spheres),
        fill(NaN, n_spheres),               # T1_sigma
        0, 0.0, false, 0.0,
        zeros(Float32, Npe * Nfe),
    )
end

"Observation dimension: image (Nfe*Npe) + T1 estimates (n_spheres) + T1 σ-channel (n_spheres) + budget (3)."
e2_obs_dim(env::E2Env) = env.Nfe * env.Npe + 2 * env.n_spheres + 3

"Action space bounds: [TI_lo, TE_lo, TR_lo, α_deg_lo, slice_z_mm_lo], same for hi."
e2_action_lo(::E2Env) = Float64[0.010, 0.005, 0.5,   5.0,  -60.0]
e2_action_hi(::E2Env) = Float64[3.000, 0.080, 5.0, 180.0,   60.0]

# ── Internal helpers ─────────────────────────────────────────────────────────

function _e2_build_episode_phantom(env::E2Env, rng_seed::Int)
    rng_ep = MersenneTwister(rng_seed)

    if env.subset_size === nothing
        env.sphere_indices = collect(eachindex(env.T1_base_pool))
    else
        env.sphere_indices = sort(randperm(rng_ep, length(env.T1_base_pool))[1:env.subset_size])
    end
    env.sphere_centres_base = env.sphere_centres_pool[env.sphere_indices]
    env.T1_base             = env.T1_base_pool[env.sphere_indices]
    env.T2_ratio            = env.T2_ratio_pool[env.sphere_indices]

    # Per-sphere T1 jitter (log-normal, centred on nominal values)
    T1_ep = [env.T1_base[i] * exp(env.T1_sigma_rel * randn(rng_ep))
             for i in 1:env.n_spheres]

    # Per-sphere T2 (preserve T2/T1 ratio)
    T2_ep = T1_ep .* env.T2_ratio

    # Build a custom active-sphere list: subset_size episodes contain only the
    # selected T1 spheres, while the default path keeps the full 14-sphere plate.
    base_cfg  = PhantomConfig(field = env.cfg_field, voxel_size_mm = 1.0,
                               include_plates = [:T1])
    base_descs = sphere_descriptors(:T1, base_cfg; rng = MersenneTwister(0))
    active_descs = SphereDescriptor[]
    for (i, pool_i) in enumerate(env.sphere_indices)
        d = base_descs[pool_i]
        push!(active_descs, SphereDescriptor(
            d.centre, d.radius, d.ρ,
            T1_ep[i], T2_ep[i], T2_ep[i], d.delta_w, d.label,
        ))
    end

    # Episode rotation and translation (domain randomisation)
    rx = env.rotation_sigma_rad * randn(rng_ep)
    ry = env.rotation_sigma_rad * randn(rng_ep)
    rz = env.rotation_sigma_rad * randn(rng_ep)
    tx = env.translation_sigma_mm * randn(rng_ep)
    ty = env.translation_sigma_mm * randn(rng_ep)
    tz = env.translation_sigma_mm * randn(rng_ep)

    delta_x = env.voxel_size_mm * 1e-3
    parts = Phantom[]
    for d in active_descs
        p = build_sphere(d, delta_x)
        length(p.x) > 0 && push!(parts, p)
    end
    phantom = isempty(parts) ? _empty_phantom("e2_subset") : reduce(+, parts)
    phantom.name = "e2_subset"
    phantom = apply_transform!(phantom, (rx, ry, rz), (tx, ty, tz) .* 1e-3)
    phantom = apply_per_spin_noise!(phantom, AugmentConfig(B0_sigma_Hz = 5.0), rng_ep)

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

    # 3. Per-sphere relative uncertainty in log10 scale.
    # log10(σ_T1 / T1_est), clamped to [-3, 0]. 0 means "fully uncertain"
    # (relative σ ≥ 100%) and is the sentinel for "no estimate yet" — that
    # way the channel encodes a coherent prior at episode start.
    sig_obs = [_e2_sigma_channel(env.T1_sigma[i], env.T1_est[i])
               for i in 1:env.n_spheres]

    # 4. Budget state
    t_frac  = Float32(min(1.0, env.time_used_s / env.time_budget_s))
    n_frac  = Float32(env.n_blocks / env.max_blocks)
    bgt     = Float32[t_frac, n_frac, 1.0f0]

    vcat(Float32.(img_norm), Float32.(t1_obs), Float32.(sig_obs), bgt)
end

@inline function _e2_sigma_channel(σ::Float64, T1::Float64)
    if !isfinite(σ) || !isfinite(T1) || T1 <= 0 || σ <= 0
        return 0f0                          # fully-uncertain sentinel
    end
    Float32(clamp(log10(σ / T1), -3.0, 0.0))
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

    # Reconstruct image via 2D IFFT.
    # - Magnitude (default, env.phase_sensitive = false): clinical convention,
    #   produces non-negative real image. Throws away phase. Creates the abs()-
    #   induced multimodal SSE in T1 fitting (cr_explainer.md §14, §15).
    # - Phase-sensitive (env.phase_sensitive = true): take the real part of
    #   the IFFT'd complex image. Sim-only — assumes a known reference phase
    #   (which a real scanner needs a phase-calibration scan for, see
    #   PSIR — Phase-Sensitive Inversion Recovery). The signed signal model
    #   `S = A · (1 − 2·exp(−TI/T1))` is monotonic in T1, so the SSE has a
    #   single basin per sphere, removing the multimodal-SSE failure mode at
    #   the source.
    img_complex = ifft(ksp, (1, 2))
    image_mag = if env.phase_sensitive
        # The complex image has a baseline phase (FFT shift convention etc.)
        # we don't model directly. In simulation the IR signal aligns along
        # one axis (set by the excitation phase 90°), so the relevant signed
        # quantity is the real part. For non-trivial reference phases, this
        # would need a phase-correction step.
        Float32.(real.(img_complex))
    else
        Float32.(abs.(img_complex))
    end

    image_mag, ksp
end

function _e2_update_t1_estimates!(env::E2Env, image_mag::Matrix{Float32},
                                   TI::Real, TR::Real, α_exc::Real)
    # Excitation flip angle scales transverse signal by sin(α_exc) per shot.
    # The fitter uses a single amplitude A across all TIs, so we normalise
    # the recorded magnitude by sin(α_exc) here. Floor at 1e-3 avoids divide-
    # by-zero for near-zero excitation (action lower bound is 5° → sin≈0.087).
    # Note: cos(α_exc) still matters via the steady-state Mz_pre, so α_exc
    # is also passed through to the fitter via `α_excs`.
    sin_α = max(abs(sin(Float64(α_exc))), 1e-3)
    for i in 1:env.n_spheres
        ipe, ife = env.sphere_px[i]
        mag_i = Float64(image_mag[ipe, ife]) / sin_α
        push!(env.block_TIs[i],     Float64(TI))
        push!(env.block_TRs[i],     Float64(TR))
        push!(env.block_α_excs[i],  Float64(α_exc))
        push!(env.block_mags[i],    mag_i)

        if length(env.block_TIs[i]) >= 2
            # θ_inv = π for standard 180° inversion prep
            αs = fill(π, length(env.block_TIs[i]))
            # Fix B (`docs/T1_FIT_AND_KOMA_TESTS.md` §7.3): use an *absolute*
            # noise floor anchored to the first block's magnitude per sphere
            # so σ_T1 stops coupling to the agent's per-fit data RMS. Any
            # deterministic per-sphere reference works; first-block magnitude
            # is the simplest stable choice.
            abs_noise = env.noise_sigma_rel * abs(env.block_mags[i][1])
            # F1+ (`E2_4_PLAN.md` §2.2): pass env.Npe so the fitter uses the
            # finite-Npe transient closed form that matches what
            # ir_se_2d_sequence actually feeds the simulator.
            fit = fit_t1_generalized_ir(
                env.block_TIs[i], αs, env.block_mags[i];
                TRs    = env.block_TRs[i],
                α_excs = env.block_α_excs[i],
                Npe    = env.Npe,
                T1_range = env.T1_sample_range, n_grid = 200,
                abs_noise_sigma = abs_noise,
                sigma_method    = env.sigma_method,
                signed          = env.phase_sensitive,
            )
            env.T1_est[i]   = fit.T1
            env.T1_sigma[i] = fit.T1_sigma
        else
            # Neutral prior: geometric mean of sample range
            env.T1_est[i]   = sqrt(env.T1_sample_range[1] * env.T1_sample_range[2])
            env.T1_sigma[i] = NaN
        end
    end
end

function _e2_mape(env::E2Env)
    # Aggregate per-sphere absolute percentage errors as
    #   α · mean(errs) + (1 − α) · max(errs)
    # α = 1.0 recovers plain mean MAPE (legacy behaviour).
    # α < 1.0 mixes in worst-case to penalise the policy for ignoring any
    # one sphere — addresses the §16.2 mid-T1 gap.
    errs = [abs(env.T1_est[i] - env.T1_true[i]) / env.T1_true[i]
            for i in 1:env.n_spheres]
    α = env.mape_alpha
    α * mean(errs) + (1 - α) * maximum(errs)
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
        empty!(env.block_TRs[i])
        empty!(env.block_α_excs[i])
        empty!(env.block_mags[i])
        env.T1_est[i]   = NaN
        env.T1_sigma[i] = NaN
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
    _e2_update_t1_estimates!(env, image_mag, TI, TR, deg2rad(α_exc_deg))

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
        "sphere_indices" => copy(env.sphere_indices),
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

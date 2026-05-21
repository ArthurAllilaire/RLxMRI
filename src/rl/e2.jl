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
# * Noise: absolute complex Gaussian on k-space (FIX_SIM_PLAN §2). Real and
#   imaginary parts are independently N(0, σ²) with σ = env.noise_sigma_abs,
#   added via `add_noise!`. This matches the physical model: MRI thermal
#   noise is hardware-determined (∝ √(k_B·T·BW·R) / G), absolute, and
#   scene-independent. After `abs()` the residuals are Rician — the
#   Gaussian likelihood in the fitter is therefore misspecified at low SNR
#   (the short-T1 regime). Right fix is phase-sensitive recon
#   (`phase_sensitive=true`); documented as a Ch4 limitation.
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
# * Fitter noise floor: `abs_noise_sigma = env.noise_sigma_abs` is passed
#   directly — the same absolute σ the env injects on k-space, propagated
#   to the fitter for σ_T1 estimation. Profile-likelihood σ is used by
#   default (E2_5_PLAN §3) which is more robust to misspecification than
#   the asymptotic J^T J formula.




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
    noise_sigma_abs::Float64               # absolute complex-Gaussian σ on k-space
    target_snr::Float64                    # if > 0, σ was derived at construction from
                                            # a reference acquisition on a nominal phantom
                                            # (ksp_rms / target_snr). 0 = σ used as-is.
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
    oracle_fit::Bool                       # D2 diagnostic (EXPERT_REPORT_TRAC §17): when
                                            # true, the fitter's T1 grid is narrowed to a
                                            # log-band of ±oracle_band around the true T1
                                            # of each sphere. Tests "is wrong-basin LM
                                            # convergence the bottleneck?" — never deploy.
    oracle_band::Float64                   # multiplicative half-width for oracle grid.
                                            # 2.0 = ±1 octave. Only used if oracle_fit.
    fitter_n_grid::Int                     # n_grid passed to fit_t1_generalized_ir.
                                            # Default 200. Higher values test whether
                                            # baseline MAPE is grid-resolution-limited
                                            # (§17.10 control).

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

    # background pixel mask (cached per episode in e2_reset!) — used by
    # `e2_image_stats` / `e2_dual_acq_snr_report` for NEMA SNR. nothing until
    # first reset.
    background_mask::Union{Nothing,BitMatrix}
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
    noise_sigma_abs::Real          = 0.005,
    target_snr::Union{Nothing,Real} = nothing,  # if given, override noise_sigma_abs by
                                                 # calibrating once at construction off a
                                                 # reference IR-SE block on a nominal phantom
                                                 # (σ = ksp_rms / target_snr). Scene-relative
                                                 # knob; the resulting σ is fixed across
                                                 # episodes — preserves physical model
                                                 # (hardware-determined σ, see header §Noise).
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
    oracle_fit::Bool               = false,            # D2 diagnostic only
    oracle_band::Real              = 2.0,
    fitter_n_grid::Integer         = 200,
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

    snr_for_struct = target_snr === nothing ? 0.0 : Float64(target_snr)

    env = E2Env(
        cfg_field, Float64(voxel_size_mm),
        Float64(FOV), Nfe, Npe, n_spheres, subset_size === nothing ? nothing : n_spheres,
        Int(max_blocks), Float64(time_budget_s),
        Float64(terminal_bonus), Float64(success_tol),
        Float64(noise_sigma_abs), snr_for_struct, Float64(T1_sigma_rel),
        Float64(translation_sigma_mm), Float64(rotation_sigma_rad),
        Float64.(T1_sample_range),
        reward_mode,
        Float64(mape_alpha),
        Bool(phase_sensitive),
        sigma_method,
        Bool(oracle_fit),
        Float64(oracle_band),
        Int(fitter_n_grid),
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
        nothing,                            # background_mask (filled at reset)
    )

    # If target_snr was requested, calibrate noise_sigma_abs ONCE here off a
    # reference acquisition on a nominal phantom (no jitter, no pose). This
    # gives an interpretable SNR knob for sweeps while keeping σ fixed and
    # scene-independent during training — matches the physical noise model
    # documented in §Noise. Reference block: TI=0.5 s, TR=3.0 s, α=90°,
    # TE=0.020 s — produces signal across most spheres.
    if target_snr !== nothing && Float64(target_snr) > 0.0
        _e2_calibrate_snr!(env, Float64(target_snr))
    end

    env
end

"""
    _e2_calibrate_snr!(env, target_snr) → env

Calibrate `env.noise_sigma_abs` so a reference IR-SE acquisition on a nominal
phantom (no jitter, no pose) has k-space RMS / σ = `target_snr`. Run once at
construction; the resulting σ stays fixed for all episodes (hardware-determined
noise model). Reference block parameters are deterministic so the same
(target_snr, phantom_config) always produces the same σ.
"""
function _e2_calibrate_snr!(env::E2Env, target_snr::Real)
    # Build nominal phantom: same plate config as training, no jitter or pose.
    base_cfg = PhantomConfig(field = env.cfg_field,
                              voxel_size_mm = env.voxel_size_mm,
                              include_plates = [:T1])
    nominal_phantom = build_phantom(base_cfg)

    # Reference acquisition (matches docstring above).
    TI_ref, TE_ref, TR_ref = 0.5, 0.020, 3.0
    α_ref = deg2rad(90.0)
    seq = Suppressor.@suppress ir_se_2d_sequence(
        TI_ref, TE_ref, TR_ref;
        α_exc = α_ref,
        FOV   = env.FOV,
        Nfe   = env.Nfe,
        Npe   = env.Npe,
    )
    scanner = scanner_for_field(env.cfg_field)
    raw_a = Suppressor.@suppress simulate(nominal_phantom, seq, scanner)
    ksp_a = raw_to_kspace(raw_a, env.Npe, env.Nfe)

    ksp_rms = sqrt(sum(abs2, ksp_a) / length(ksp_a))
    σ = ksp_rms / Float64(target_snr)
    env.noise_sigma_abs = σ

    # Build per-sphere pixel locations on the nominal phantom (no pose
    # transform), and a background mask, so we can compute NEMA and dual-acq
    # SNR right here at construction.
    sphere_px = NTuple{2,Int}[]
    for c in env.sphere_centres_base
        cx, cy = c[1], c[2]
        ife = mod(round(Int, cx * env.Nfe / env.FOV) + env.Nfe ÷ 2, env.Nfe) + 1
        ipe = mod(round(Int, cy * env.Npe / env.FOV) + env.Npe ÷ 2, env.Npe) + 1
        push!(sphere_px, (ipe, ife))
    end
    bg = background_mask(nominal_phantom, env.Npe, env.Nfe, env.FOV; erosion_px = 1)

    # Inject noise on a *copy* of ksp_a so the calibration figure for
    # `ksp_rms` stays on the noiseless k-space (clean reference); the
    # reconstructed image we use for NEMA/dual must include noise.
    ksp_a_noisy = copy(ksp_a)
    rng_cal = MersenneTwister(0)
    add_noise!(ksp_a_noisy, σ; rng = rng_cal)
    img_a = kspace_to_image(ksp_a_noisy; phase_sensitive = env.phase_sensitive)

    rep = snr_report(nominal_phantom, seq, scanner;
                     σ = σ,
                     sphere_px = sphere_px,
                     bg_mask = bg,
                     ksp_a = ksp_a_noisy,
                     img_a = img_a,
                     rng = rng_cal,
                     phase_sensitive = env.phase_sensitive,
                     roi_radius = 0)

    @info "E2Env SNR calibration" target_snr σ ksp_rms snr_ksp=rep.snr_ksp snr_nema_peak_a=rep.image.snr_nema_peak_a snr_nema_peak_b=rep.image.snr_nema_peak_b snr_dual_peak=rep.image.snr_dual_peak
    env
end

"Observation dimension: image (Nfe*Npe) + T1 estimates (n_spheres) + T1 σ-channel (n_spheres) + budget (3)."
e2_obs_dim(env::E2Env) = env.Nfe * env.Npe + 2 * env.n_spheres + 3

"Action space bounds: [TI_lo, TE_lo, TR_lo, α_deg_lo, slice_z_mm_lo], same for hi."
e2_action_lo(::E2Env) = Float64[0.010, 0.005, 0.5,   5.0,  -60.0]
e2_action_hi(::E2Env) = Float64[3.000, 0.080, 5.0, 180.0,   60.0]

# ── Internal helpers ─────────────────────────────────────────────────────────

function _e2_build_episode_phantom(env::E2Env, rng_seed::Int; forced_indices=nothing)
    rng_ep = MersenneTwister(rng_seed)

    if forced_indices !== nothing
        env.sphere_indices = sort(Int.(forced_indices))
    elseif env.subset_size === nothing
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
        # Centred indexing: image centre (FOV origin) maps to pixel
        # (Npe÷2+1, Nfe÷2+1) after fftshift; offsets are added relative to
        # that. Matches the recon convention in `_e2_simulate_step`.
        ife = mod(round(Int, cx * env.Nfe / env.FOV) + env.Nfe ÷ 2, env.Nfe) + 1
        ipe = mod(round(Int, cy * env.Npe / env.FOV) + env.Npe ÷ 2, env.Npe) + 1
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
    # KomaMRI multi-shot drift: per-shot signals start diverging from
    # steady-state physics past ~70 s of total simulated time per
    # `simulate()` call. See KOMA_BUG_REPRO.md §1.2.5. We warn rather than
    # error because the cutoff is fuzzy and there are use cases (eg
    # debug / one-shot diagnostics) where the user accepts the drift.
    sim_time = Float64(TR) * env.Npe
    if sim_time > 60.0
        @warn "TR × Npe = $(round(sim_time, digits=1)) s exceeds KomaMRI safe zone (~60 s). Per-shot signals will drift; see KOMA_BUG_REPRO.md." maxlog = 1
    end

    α_exc = deg2rad(Float64(α_exc_deg))
    seq = Suppressor.@suppress ir_se_2d_sequence(
        TI, TE, TR;
        α_exc = α_exc,
        FOV   = env.FOV,
        Nfe   = env.Nfe,
        Npe   = env.Npe,
    )
    raw = Suppressor.@suppress simulate(env.phantom, seq, scanner_for_field(env.cfg_field))

    ksp = raw_to_kspace(raw, env.Npe, env.Nfe)

    # Absolute complex Gaussian noise (FIX_SIM_PLAN §2). σ is hardware-set,
    # not scene-relative — same magnitude every step regardless of phantom.
    add_noise!(ksp, env.noise_sigma_abs; rng = env.rng)

    # Reconstruct via 2D IFFT. See `kspace_to_image` for convention notes.
    # phase_sensitive=true uses signed real part (PSIR-style, sim-only);
    # false (default) uses magnitude — clinical convention, abs() induces
    # multimodal SSE in T1 fitting (cr_explainer.md §14, §15).
    image_mag = kspace_to_image(ksp; phase_sensitive=env.phase_sensitive)

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
            # Fitter noise floor = the same absolute σ injected on k-space
            # (FIX_SIM_PLAN §2.2). Decoupled from per-sphere data RMS.
            abs_noise = env.noise_sigma_abs
            # F1+ (`E2_4_PLAN.md` §2.2): pass env.Npe so the fitter uses the
            # finite-Npe transient closed form that matches what
            # ir_se_2d_sequence actually feeds the simulator.
            fit = fit_t1_generalized_ir(
                env.block_TIs[i], αs, env.block_mags[i];
                TRs    = env.block_TRs[i],
                α_excs = env.block_α_excs[i],
                Npe    = env.Npe,
                T1_range = env.T1_sample_range, n_grid = env.fitter_n_grid,
                abs_noise_sigma = abs_noise,
                sigma_method    = env.sigma_method,
                signed          = env.phase_sensitive,
                T1_oracle       = env.oracle_fit ? env.T1_true[i] : nothing,
                oracle_band     = env.oracle_band,
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
function e2_reset!(env::E2Env; rng_seed::Union{Nothing,Integer} = nothing,
                   forced_indices = nothing)
    if rng_seed !== nothing
        env.rng = MersenneTwister(Int(rng_seed))
    end
    ep_seed = rand(env.rng, 0:typemax(Int32))

    phantom, T1_ep, sphere_px, rot, trans =
        _e2_build_episode_phantom(env, ep_seed; forced_indices=forced_indices)

    env.phantom              = phantom
    env.T1_true              = T1_ep
    env.sphere_px            = sphere_px
    env.episode_rotation     = rot
    env.episode_translation_m = trans
    # Cache background pixel mask (zero-occupancy pixels, eroded by 1 px)
    # so per-step `e2_image_stats` and `e2_dual_acq_snr_report` don't
    # re-derive it on every call. Phantom is constant within an episode.
    env.background_mask      = background_mask(phantom, env.Npe, env.Nfe,
                                               env.FOV; erosion_px = 1)

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
    image_mag, _ = _e2_simulate_step(env, TI, TE, TR, α_exc_deg)

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

"""
    e2_image_stats(env; roi_radius=0) → NamedTuple

Single-image NEMA SNR stats on the most recent step's reconstructed image.
Cheap (no extra simulation). Requires that at least one `e2_step!` has run
since the last reset. See `nema_stats` for the returned fields and the
`/0.6551` Rayleigh-correction convention.
"""
function e2_image_stats(env::E2Env; roi_radius::Integer = 0)
    env.background_mask === nothing &&
        error("e2_image_stats: call e2_reset! first")
    img = reshape(env.last_image_mag, env.Npe, env.Nfe)
    nema_stats(img, env.sphere_px, env.background_mask; roi_radius = Int(roi_radius))
end

"""
    e2_dual_acq_snr_report(env; TI=0.5, TE=0.02, TR=3.0, α_deg=90.0,
                            roi_radius=0, seed=0) → SNRReport

Run a **dual acquisition** (two independent noise realisations of the same
sequence) on the current episode phantom with the current `noise_sigma_abs`,
and return a full `SNRReport`. Costs 2 simulator calls — intended to be
called once per `diagnose_e2` run, not per step. Uses default reference
parameters (TI=0.5 s, TR=3.0 s, α=90°, TE=0.02 s) so the number is comparable
across runs; override via kwargs if you want a different operating point.
"""
function e2_dual_acq_snr_report(env::E2Env;
                                 TI::Real = 0.5,
                                 TE::Real = 0.020,
                                 TR::Real = 3.0,
                                 α_deg::Real = 90.0,
                                 roi_radius::Integer = 0,
                                 seed::Integer = 0)
    env.phantom === nothing && error("e2_dual_acq_snr_report: call e2_reset! first")
    env.background_mask === nothing &&
        error("e2_dual_acq_snr_report: missing background mask (call e2_reset!)")

    α_exc = deg2rad(Float64(α_deg))
    seq = Suppressor.@suppress ir_se_2d_sequence(
        Float64(TI), Float64(TE), Float64(TR);
        α_exc = α_exc, FOV = env.FOV, Nfe = env.Nfe, Npe = env.Npe,
    )
    scanner = scanner_for_field(env.cfg_field)
    rng = MersenneTwister(Int(seed))

    raw_a = Suppressor.@suppress simulate(env.phantom, seq, scanner)
    ksp_a = raw_to_kspace(raw_a, env.Npe, env.Nfe)
    add_noise!(ksp_a, env.noise_sigma_abs; rng = rng)
    img_a = kspace_to_image(ksp_a; phase_sensitive = env.phase_sensitive)

    snr_report(env.phantom, seq, scanner;
               σ = env.noise_sigma_abs,
               sphere_px = env.sphere_px,
               bg_mask = env.background_mask,
               ksp_a = ksp_a,
               img_a = img_a,
               rng = rng,
               phase_sensitive = env.phase_sensitive,
               roi_radius = Int(roi_radius))
end

# Aliases for juliacall (Python can't reach names ending in `!`)
const e2_reset_b = e2_reset!
const e2_step_b  = e2_step!
const e2_image_stats_b = e2_image_stats
const e2_dual_acq_snr_report_b = e2_dual_acq_snr_report

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
#   targeting (no slice-selective excitation in the forward model).
#   Structural limitation; addressable only via slice-selective excitation
#   or a multi-action-per-block MDP (E3+).
#
# * abs() in the signal magnitude (no phase-sensitive reconstruction)
#   creates multimodal SSE in the fit. Two T1 values typically give
#   equivalent |S| at saturated TIs → LM solver picks a basin by init.
#   Profile-likelihood σ (E2_5_PLAN.md §3) reports the resulting ambiguity
#   honestly without changing the point estimate.
#
# * Fitter noise floor: the env injects an absolute σ, but the fitter receives
#   magnitudes divided by sin(α_exc). The noise floor passed to the fit is
#   therefore the RMS of σ / sin(α_exc,k) over the samples in that sphere's
#   running fit. Profile-likelihood σ is used by default (E2_5_PLAN §3), which
#   is more robust to misspecification than the asymptotic J^T J formula.




"""
    E2Env

Stateful environment for E2. Python drives it step-by-step via
`e2_reset!` / `e2_step!`.
"""
mutable struct E2Env
    # ── static config ────────────────────────────────────────────────────────
    cfg_field::Symbol                      # :T3 or :T15
    voxel_size_mm::Float64                 # phantom voxelisation
    water_voxel_size_mm::Union{Nothing,Float64}  # optional background-water voxelisation
    FOV::Float64                           # image field of view [m]
    Nfe::Int                               # frequency-encode samples
    Npe::Int                               # phase-encode steps
    use_gpu::Bool                          # run KomaMRI Bloch sim on GPU. Passed as
                                            # sim_params["gpu"]; KomaMRI falls back to
                                            # CPU when no GPU backend is loaded, so
                                            # false (default) == prior CPU behaviour.
    n_spheres::Int                         # active spheres per episode
    subset_size::Union{Nothing,Int}        # nothing = all spheres; k = random k-subset
    forced_sphere_indices::Vector{Int}     # fixed 1-based T1-pool labels; empty = none
    max_blocks::Int
    time_budget_s::Float64
    terminal_bonus::Float64
    success_tol::Float64                   # MAPE threshold for terminal bonus
    noise_sigma_abs::Float64               # absolute complex-Gaussian σ on k-space
    T1_sigma_rel::Float64                  # per-sphere T1 jitter (relative)
    t1_sampler::Symbol                     # :lognormal | :linear_uniform_range
    translation_sigma_mm::Float64          # pose translation σ per axis [mm]
    rotation_sigma_rad::Float64            # pose rotation σ per axis [rad]
    pose_mode::Symbol                      # :auto | :fixed | :inplane_jitter
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
    include_image::Bool                    # include flattened recon image in the obs.
                                            # Default false (E2_RERUN_PLAN §6.2): the
                                            # in-env fit already distils the image into
                                            # T1_est, so the raw pixels are largely
                                            # redundant and inflate the obs to ~2048 dims.
    include_sigma::Bool                    # include the per-sphere fitter-σ channel in
                                            # the obs. Default false (E2_RERUN_PLAN §6.3):
                                            # the σ-uncertainty machinery was a red herring
                                            # of the pre-fix sim.
    roi_radius::Int                        # square ROI half-width for per-sphere signal
                                            # extraction from the reconstructed image.
                                            # 0 = legacy centre pixel; 1 = 3×3 mean.
    include_water::Bool                    # include background-water spins in the phantom
                                            # slab. Default true. Set false to benchmark
                                            # sim cost without the water background, or for
                                            # a spheres-only phantom (the T1 sphere spins
                                            # remain via custom_sphere_descriptors).
    water_model::Symbol                    # :bloch (default) full-Bloch sims the water with
                                            # the spheres every step. :cached_perline
                                            # Bloch-sims only the spheres and adds the
                                            # background water from a cached Koma template
                                            # (CachedWaterModel) rescaled analytically per
                                            # k-line — ~8× per-step, reproduces full sim to
                                            # the T1-grid floor (src/water_cache.jl). Requires
                                            # include_water. Cache scope follows include_image:
                                            # global (one water sim, fixed pose) when the obs
                                            # is T1-only, per-episode (rebuilt each reset, pose
                                            # randomised) when the image is in the obs.
    forward_model::Symbol                  # :bloch (default) full KomaMRI Bloch sim + 2D recon
                                            # every step. :analytic skips Koma entirely and
                                            # generates per-sphere magnitudes from the SAME closed
                                            # form the fitter inverts (transient_mz_at_excite_npe),
                                            # so steps are ~µs. Fits become noise-limited only
                                            # (no water-bleed / B0σ / spatial recon cross-talk) —
                                            # a fast surrogate for SCREENING reward behaviour, not
                                            # for absolute MAPE. include_image/water_model are
                                            # ignored under :analytic.
    analytic_noise_sigma::Float64          # complex-Gaussian σ on the analytic per-sphere signal
                                            # (signal scale is O(1); default 0.04 ≈ SNR 25 at the
                                            # reference operating point). Sweepable. Only used when
                                            # forward_model = :analytic.
    time_penalty_coef::Float64             # λ: subtract λ·(block_time / time_budget_s) from the
                                            # per-step reward. ONLY applied when allow_stop is true
                                            # (with a fixed budget total time ≈ budget, so the term
                                            # is a near-constant offset and just clutters the reward).
                                            # With a learned stop the episode length is variable and
                                            # Σ λ·block_time/budget = λ·total_time/budget — a genuine
                                            # accuracy-vs-time price. Default 0.0.
    allow_stop::Bool                       # expose a learned STOP decision to the agent. When true,
                                            # e2_step! takes a `stop` flag (the wrapper appends a
                                            # thresholded gate to the action); a stop ends the episode
                                            # after the current block. No n_blocks guard — a premature
                                            # stop (<2 blocks) yields final_mape=1.0, so the agent
                                            # learns to avoid it. Default false = legacy fixed-budget.

    # ── sphere pool info (fixed at construction) ──────────────────────────
    base_descs_pool::Vector{SphereDescriptor}       # full nominal T1 descriptor pool

    # ── active sphere info (filled at reset) ───────────────────────────────
    sphere_indices::Vector{Int}            # 1-based indices into the 14-sphere pool
    active_base_descs::Vector{SphereDescriptor}

    # ── episode state ────────────────────────────────────────────────────────
    rng::MersenneTwister
    phantom::Any                           # KomaMRI Phantom Bloch-simulated each step.
                                            # :bloch → spheres+water; :cached_perline → dry
                                            # (spheres-only); the water is added analytically.
    cached_water::Union{Nothing,CachedWaterModel}  # water template for :cached_perline.
                                            # Built once (global scope) or per reset
                                            # (per-episode scope); nothing under :bloch.
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
    cfg_field::Symbol             = :T15,
    voxel_size_mm::Real           = 1.0,
    water_voxel_size_mm::Union{Nothing,Real} = nothing,
    FOV::Real                     = 0.2,
    Nfe::Int                      = 64,
    Npe::Int                      = 32,
    use_gpu::Bool                  = false,
    max_blocks::Int                = 15,
    time_budget_s::Real            = 120.0,
    terminal_bonus::Real           = 0.5,
    success_tol::Real              = 0.05,
    noise_sigma_abs::Real          = 50.0,    # σ* for NEMA dual-acq SNR ≈ 25 (E2_RERUN_PLAN §3.1)
    T1_sigma_rel::Real             = 0.05,
    t1_sampler::Symbol             = :lognormal,
    forced_sphere_indices = Int[],
    translation_sigma_mm::Real     = 5.0,
    rotation_sigma_rad::Real       = 0.15,   # ~8.6°
    pose_mode::Symbol              = :auto,
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
    include_image::Bool            = false,            # E2_RERUN_PLAN §6.2
    include_sigma::Bool            = false,            # E2_RERUN_PLAN §6.3
    roi_radius::Integer             = 0,                # 0 = centre pixel; 1 = 3×3 mean
    include_water::Bool            = true,             # background-water spins on/off
    water_model::Symbol            = :bloch,           # :bloch | :cached_perline (water_cache.jl)
    forward_model::Symbol          = :bloch,           # :bloch | :analytic (fast surrogate)
    analytic_noise_sigma::Real     = 0.04,             # σ on analytic signal (≈ SNR 25)
    time_penalty_coef::Real        = 0.0,              # λ on block_time/time_budget (only with allow_stop)
    allow_stop::Bool               = false,            # learned STOP decision (optimal stopping)
)
    cfg_field ∈ (:T3, :T15) || error("cfg_field must be :T3 or :T15")
    reward_mode ∈ (:neg_mape, :delta_mape, :neg_log_mape, :delta_log_mape, :terminal_only) ||
        error("reward_mode must be one of :neg_mape, :delta_mape, :neg_log_mape, " *
              ":delta_log_mape, :terminal_only")
    forward_model ∈ (:bloch, :analytic) ||
        error("forward_model must be :bloch or :analytic")
    sigma_method ∈ (:asymptotic, :profile_likelihood, :bootstrap) ||
        error("sigma_method must be :asymptotic, :profile_likelihood, or :bootstrap")
    water_model ∈ (:bloch, :cached_perline) ||
        error("water_model must be :bloch or :cached_perline")
    t1_sampler ∈ (:lognormal, :linear_uniform_range) ||
        error("t1_sampler must be :lognormal or :linear_uniform_range")
    pose_mode ∈ (:auto, :fixed, :inplane_jitter) ||
        error("pose_mode must be :auto, :fixed, or :inplane_jitter")
    (water_model === :bloch || include_water) ||
        error("water_model = :cached_perline requires include_water = true")
    water_voxel_size_mm === nothing || Float64(water_voxel_size_mm) > 0.0 ||
        error("water_voxel_size_mm must be > 0")
    0.0 <= Float64(mape_alpha) <= 1.0 ||
        error("mape_alpha must be in [0, 1]")
    Int(roi_radius) >= 0 ||
        error("roi_radius must be >= 0")

    # Base sphere info (no rotation/translation/jitter)
    base_cfg = PhantomConfig(field = cfg_field, include_plates = [:T1])
    base_descs = sphere_descriptors(:T1, base_cfg)
    n_pool      = length(base_descs)
    subset_size !== nothing &&
        (1 <= Int(subset_size) <= n_pool ||
         error("subset_size must be between 1 and $n_pool, or nothing"))
    forced_indices = sort(unique(Int.(forced_sphere_indices)))
    all(i -> 1 <= i <= n_pool, forced_indices) ||
        error("forced_sphere_indices must be 1-based labels in 1:$n_pool")
    subset_size !== nothing && length(forced_indices) > Int(subset_size) &&
        error("forced_sphere_indices length cannot exceed subset_size")
    n_spheres  = subset_size === nothing ?
                 (isempty(forced_indices) ? n_pool : length(forced_indices)) :
                 Int(subset_size)

    env = E2Env(
        cfg_field, Float64(voxel_size_mm),
        water_voxel_size_mm === nothing ? nothing : Float64(water_voxel_size_mm),
        Float64(FOV), Nfe, Npe, Bool(use_gpu),
        n_spheres, subset_size === nothing ? nothing : n_spheres,
        forced_indices,
        Int(max_blocks), Float64(time_budget_s),
        Float64(terminal_bonus), Float64(success_tol),
        Float64(noise_sigma_abs), Float64(T1_sigma_rel),
        t1_sampler,
        Float64(translation_sigma_mm), Float64(rotation_sigma_rad),
        pose_mode,
        Float64.(T1_sample_range),
        reward_mode,
        Float64(mape_alpha),
        Bool(phase_sensitive),
        sigma_method,
        Bool(oracle_fit),
        Float64(oracle_band),
        Int(fitter_n_grid),
        Bool(include_image),
        Bool(include_sigma),
        Int(roi_radius),
        Bool(include_water),
        water_model,
        forward_model,
        Float64(analytic_noise_sigma),
        Float64(time_penalty_coef),
        Bool(allow_stop),
        base_descs,
        Int[], SphereDescriptor[],
        MersenneTwister(rng_seed),
        nothing,                            # phantom (filled at reset)
        nothing,                            # cached_water (filled at reset if cached)
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

    env
end


"""
Observation dimension. The T1-estimate channel (n_spheres) and budget (3) are
always present; the flattened image (Nfe*Npe) and the per-sphere σ-channel
(n_spheres) are gated by `include_image` / `include_sigma` (both default false,
E2_RERUN_PLAN §6.2–6.3). Default obs = n_spheres + 3.
"""
e2_obs_dim(env::E2Env) =
    (env.include_image ? env.Nfe * env.Npe : 0) +
    env.n_spheres +
    (env.include_sigma ? env.n_spheres : 0) + 3

"Action space bounds: [TI_lo, TE_lo, TR_lo, α_deg_lo], same for hi."
e2_action_lo(::E2Env) = Float64[0.010, 0.005, 0.5,   5.0]
e2_action_hi(::E2Env) = Float64[3.000, 0.080, 5.0, 180.0]

"Fraction of TR available to contain TI + TE before recovery headroom."
e2_tr_headroom(::E2Env) = 0.90

# ── Internal helpers ─────────────────────────────────────────────────────────

function _e2_build_episode_phantom(env::E2Env, rng_seed::Int; forced_indices=nothing)
    # Episode-level domain randomisation via the MRISystemPhantom random-phantom
    # API. The selector → material → pose RNG consumption order (subset randperm →
    # per-sphere T1 randn → pose randn) matches the previous hand-rolled version,
    # so episode selection, T1 jitter and pose are byte-identical for a given seed.
    # See MRISystemPhantom.jl/RANDOM_PHANTOM_PLAN.md.
    cache_globally = _e2_cache_globally(env)

    # Deterministic build settings. Contrast plate :T1 is generated then moved
    # into custom_sphere_descriptors by the API; :water (if any) and the slab
    # stay deterministic, so the resulting cfg matches the old phantom_cfg.
    base = PhantomConfig(
        field               = env.cfg_field,
        voxel_size_mm       = env.voxel_size_mm,
        water_voxel_size_mm = env.water_voxel_size_mm,
        include_plates      = env.include_water ? [:T1, :water] : [:T1],
        augment             = AugmentConfig(B0_sigma_Hz = 5.0),
        slice_thickness_mm  = env.voxel_size_mm,
        slice_center_mm     = (0.0, 0.0, PLATE_Z_MM.T1),
    )

    active_forced = forced_indices !== nothing ? sort(Int.(forced_indices)) :
                    env.forced_sphere_indices

    # Sphere selection: explicit forced set, all spheres, or a random k-subset.
    selector = !isempty(active_forced) ?
                   E2SphereSelector(subset_size = env.subset_size === nothing ?
                                                  length(active_forced) : env.subset_size,
                                    forced_indices = active_forced) :
               env.subset_size === nothing ? nothing :
                   E2SphereSelector(subset_size = env.subset_size)

    # IN-PLANE ONLY pose: we acquire a single thin axial slab, so out-of-plane
    # tilt (rx, ry) and z-translation (tz) would move spheres out of the slab.
    # In-plane rotation rz + translation tx, ty still relocate every sphere on the
    # image each episode — enough to stop the agent memorising fixed pixels.
    #
    # Under :cached_perline with a T1-only observation we cache the water template
    # ONCE globally, which requires a fixed geometry — so the pose is pinned to
    # zero. The agent never sees pixel positions in that mode, so fixing the pose
    # costs no exploitable information.
    pose = if env.pose_mode === :fixed || cache_globally
        FixedPose()
    else
        InPlanePoseSampler(rotation_sigma_rad   = env.rotation_sigma_rad,
                           translation_sigma_mm = env.translation_sigma_mm)
    end

    material_sampler = if env.t1_sampler === :linear_uniform_range
        vals = T1_ARRAY[env.cfg_field]
        MaterialDistributionSampler(
            T1  = Uniform(minimum(vals), maximum(vals)),
            T2  = PreserveNominalRatio(:T2, :T1),
            T2s = PreserveNominalRatio(:T2s, :T1),
        )
    else
        RatioPreservingLogNormalT1(env.T1_sigma_rel)
    end

    rpcfg = RandomPhantomConfig(
        base             = base,
        sphere_selector  = selector,
        material_sampler = material_sampler,
        pose_sampler     = pose,
    )
    episode = sample_phantom_config(rpcfg; rng_seed = rng_seed)

    # Env side effects + episode truth. `custom_sphere_descriptors` holds the
    # sampled T1 descriptors in stable label order, aligned with sphere_indices.
    env.sphere_indices    = sort(episode.truth.active_indices_by_plate[:T1])
    env.active_base_descs = env.base_descs_pool[env.sphere_indices]

    active_descs = episode.cfg.custom_sphere_descriptors
    T1_ep        = [d.T1 for d in active_descs]

    episode_rotation      = episode.truth.rotation
    episode_translation_m = episode.truth.translation_mm .* 1e-3
    phantom_cfg           = episode.cfg

    # :bloch → simulate spheres+water together. :cached_perline → simulate only the
    # spheres (dry, keeping B0σ) and add the cached water k-space; split the same cfg
    # into its dry + water blocks via the library helper (exploits the
    # [spheres…, water…] layout + KomaMRI's linearity in spins). The water template
    # is built at B0σ=0 (per-spin off-resonance makes the cache TI-phase-dependent
    # and is unrecoverable analytically); the Bloch-simulated spheres keep B0σ.
    water_phantom = nothing
    if env.forward_model === :analytic
        # Analytic surrogate: signals come from the closed form in
        # _e2_analytic_signals, so no Koma phantom is needed. Skip the build.
        phantom = nothing
    elseif env.water_model === :cached_perline
        phantom, water_phantom = build_dry_and_water(phantom_cfg; water_B0_sigma_Hz = 0.0)
        phantom.name = "e2_subset"
    else
        phantom = build_phantom(phantom_cfg)
        phantom.name = "e2_subset"
    end

    active_descs_scanner = transform_descriptors(active_descs,
                                                 episode_rotation,
                                                 episode_translation_m)
    sphere_px = sphere_descriptor_pixels(active_descs_scanner,
                                         env.Npe, env.Nfe, env.FOV)

    phantom, water_phantom, T1_ep, sphere_px,
    episode_rotation, episode_translation_m
end

"True when the water template should be cached once globally rather than rebuilt
per episode: cached water model AND a T1-only observation (no image to exploit
pose), so the geometry can be pinned and one water sim reused across episodes."
_e2_cache_globally(env::E2Env) =
    env.water_model === :cached_perline && !env.include_image

function _e2_observation(env::E2Env)
    # Running T1 estimates (log10 scale, 0 for uninitialised) — always present.
    t1_obs = [isnan(env.T1_est[i]) ? 0f0 :
              Float32(log10(clamp(env.T1_est[i], 1e-4, 10.0)))
              for i in 1:env.n_spheres]

    # Budget state — always present.
    t_frac  = Float32(min(1.0, env.time_used_s / env.time_budget_s))
    n_frac  = Float32(env.n_blocks / env.max_blocks)
    bgt     = Float32[t_frac, n_frac, 1.0f0]

    parts = Vector{Vector{Float32}}()

    # Optional flattened normalised magnitude image (E2_RERUN_PLAN §6.2).
    if env.include_image
        img_max = maximum(env.last_image_mag)
        img_norm = img_max > 0f0 ? env.last_image_mag ./ img_max :
                                    env.last_image_mag
        push!(parts, Float32.(img_norm))
    end

    push!(parts, Float32.(t1_obs))

    # Optional per-sphere relative-uncertainty channel (E2_RERUN_PLAN §6.3).
    # log10(σ_T1 / T1_est), clamped to [-3, 0]; 0 = "fully uncertain / no
    # estimate yet" — a coherent prior at episode start.
    if env.include_sigma
        sig_obs = [_e2_sigma_channel(env.T1_sigma[i], env.T1_est[i])
                   for i in 1:env.n_spheres]
        push!(parts, Float32.(sig_obs))
    end

    push!(parts, bgt)

    vcat(parts...)
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
    raw = Suppressor.@suppress simulate(
        env.phantom, seq, scanner_for_field(env.cfg_field);
        sim_params = Dict{String,Any}("gpu" => env.use_gpu))

    ksp = raw_to_kspace(raw, env.Npe, env.Nfe)

    # :cached_perline — env.phantom held only the spheres; add the background
    # water from the cached Koma template (per-k-line transient + sin α + TE
    # correction). Added on the clean k-space, before noise, exactly as a full
    # water sim would contribute (KomaMRI is linear in spins). See water_cache.jl.
    if env.water_model === :cached_perline
        ksp .+= cached_water_ksp(env.cached_water, TI, TR, α_exc, TE)
    end

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

"""
    _e2_sphere_signals(env, TI, TE, TR, α_exc_deg) -> Vector{Float64}

Per-sphere reconstructed magnitude for one block. Dispatches on `forward_model`:
`:bloch` runs the full KomaMRI sim + 2D recon and samples each sphere with
`roi_mean(...; r=env.roi_radius)` (and caches the image in `last_image_mag`);
`:analytic` synthesises the signal directly from `transient_mz_at_excite_npe` —
the same closed form the fitter inverts — with no Koma call.
"""
function _e2_sphere_signals(env::E2Env, TI::Real, TE::Real, TR::Real,
                            α_exc_deg::Real)
    if env.forward_model === :analytic
        return _e2_analytic_signals(env, TI, TE, TR, α_exc_deg)
    end
    image_mag, _ = _e2_simulate_step(env, TI, TE, TR, α_exc_deg)
    env.last_image_mag = vec(image_mag)
    sig = Vector{Float64}(undef, env.n_spheres)
    for i in 1:env.n_spheres
        ipe, ife = env.sphere_px[i]
        sig[i] = roi_mean(image_mag, ipe, ife; r = env.roi_radius)
    end
    sig
end

"""
    _e2_analytic_signals(env, TI, TE, TR, α_exc_deg) -> Vector{Float64}

Analytic surrogate for `_e2_sphere_signals`. For each sphere the clean signal is
`ρ · Mz_at_excite · sin(α_exc) · exp(-TE/T2)` (units: fraction of M0, scale O(1)),
where `Mz_at_excite = transient_mz_at_excite_npe(T1, TI, TR, π, α_exc; Npe)`.
Complex Gaussian noise of σ = `analytic_noise_sigma` is added in real+imag, then
`abs()` (or signed real part under `phase_sensitive`) — so the Rician/abs()
multimodality at low SNR is reproduced. No image is produced (`last_image_mag`
stays at its reset zeros; analytic mode is for T1-only observations).
"""
function _e2_analytic_signals(env::E2Env, TI::Real, TE::Real, TR::Real,
                              α_exc_deg::Real)
    α_exc = deg2rad(Float64(α_exc_deg))
    sin_α = sin(α_exc)
    σ     = env.analytic_noise_sigma
    sig   = Vector{Float64}(undef, env.n_spheres)
    for i in 1:env.n_spheres
        base = env.active_base_descs[i]
        T1_i = env.T1_true[i]
        T2_i = T1_i * base.T2 / base.T1
        mz   = transient_mz_at_excite_npe(T1_i, Float64(TI), Float64(TR),
                                          π, α_exc; Npe = env.Npe)
        clean = base.ρ * mz * sin_α * exp(-Float64(TE) / T2_i)
        re    = clean + σ * randn(env.rng)
        im    = σ * randn(env.rng)
        sig[i] = env.phase_sensitive ? re : hypot(re, im)
    end
    sig
end

function _sin_corrected_abs_noise(abs_noise_base::Real,
                                  α_excs::AbstractVector{<:Real})
    isempty(α_excs) && return Float64(abs_noise_base)
    return sqrt(mean((Float64(abs_noise_base) /
                      max(abs(sin(Float64(a))), 1e-3))^2 for a in α_excs))
end

function _e2_update_t1_estimates!(env::E2Env, signals::AbstractVector{<:Real},
                                   TI::Real, TR::Real, α_exc::Real)
    # Excitation flip angle scales transverse signal by sin(α_exc) per shot.
    # The fitter uses a single amplitude A across all TIs, so we normalise
    # the recorded magnitude by sin(α_exc) here. Floor at 1e-3 avoids divide-
    # by-zero for near-zero excitation (action lower bound is 5° → sin≈0.087).
    # Note: cos(α_exc) still matters via the steady-state Mz_pre, so α_exc
    # is also passed through to the fitter via `α_excs`.
    sin_α = max(abs(sin(Float64(α_exc))), 1e-3)
    for i in 1:env.n_spheres
        mag_i = Float64(signals[i]) / sin_α
        push!(env.block_TIs[i],     Float64(TI))
        push!(env.block_TRs[i],     Float64(TR))
        push!(env.block_α_excs[i],  Float64(α_exc))
        push!(env.block_mags[i],    mag_i)

        if length(env.block_TIs[i]) >= 2
            # θ_inv = π for standard 180° inversion prep
            αs = fill(π, length(env.block_TIs[i]))
            # Fitter noise floor after the sin(α) magnitude correction above.
            # The measured signal has absolute noise σ, but the fit sees
            # signal/sin(α), so each sample's σ scales by 1/sin(α). The fitter
            # currently uses a scalar floor, so pass the RMS corrected σ for
            # this sphere's accumulated mixed-α measurements.
            abs_noise_base = env.forward_model === :analytic ?
                             env.analytic_noise_sigma : env.noise_sigma_abs
            abs_noise = _sin_corrected_abs_noise(abs_noise_base,
                                                 env.block_α_excs[i])
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

# log-MAPE in [-3, 0] (floor at 1e-3 = 0.1 % error). More negative = smaller
# error. The 1e-3 floor sets both the resolution and the neg_log offset below.
@inline _e2_lc(m::Real) = log10(clamp(Float64(m), 1e-3, 1.0))

"""
    _e2_reward(env, mape, prev_mape, block_time, done) -> Float64

Single source of truth for the per-step reward, called by both the normal-step
and budget-exit paths of `e2_step!`. `mape`/`prev_mape` are already clamped to
[0,1]; `done` is whether this step ends the episode.

Base term by `reward_mode`:
  :neg_mape       −mape                                   ∈ [-1, 0]
  :delta_mape     prev_mape − mape                        (per-step progress)
  :neg_log_mape   −log10(clamp(mape,1e-3,1)) − 3          ∈ [-3, 0] (low-MAPE gradient)
  :delta_log_mape log10(prev) − log10(mape)               (per-step log-ratio progress)
  :terminal_only  −mape on the terminal step, else 0      (sparse control)

Then a uniform time cost `−time_penalty_coef · block_time / time_budget_s` is
subtracted, but ONLY when `allow_stop` is set: with a learned stop the episode
length is variable so this sums to `λ·total_time/budget` (a real time price);
under a fixed budget total time ≈ budget, so the term is a near-constant offset
and is dropped to keep the reward clean. The budget-exit path passes block_time=0
so the discarded block is not charged.
"""
function _e2_reward(env::E2Env, mape::Float64, prev_mape::Float64,
                    block_time::Float64, done::Bool)
    base =
        env.reward_mode === :neg_mape       ? -mape :
        env.reward_mode === :delta_mape     ? prev_mape - mape :
        env.reward_mode === :neg_log_mape   ? -_e2_lc(mape) - 3.0 :
        env.reward_mode === :delta_log_mape ? _e2_lc(prev_mape) - _e2_lc(mape) :
        env.reward_mode === :terminal_only  ? (done ? -mape : 0.0) :
        error("unknown reward_mode $(env.reward_mode)")
    time_cost = env.allow_stop ?
        env.time_penalty_coef * (block_time / env.time_budget_s) : 0.0
    base - time_cost
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

    phantom, water_phantom, T1_ep, sphere_px, rot, trans =
        _e2_build_episode_phantom(env, ep_seed; forced_indices=forced_indices)

    env.phantom              = phantom
    env.T1_true              = T1_ep
    env.sphere_px            = sphere_px
    env.episode_rotation     = rot
    env.episode_translation_m = trans

    # Cached-water template bank. Global scope (T1-only obs): build once on the
    # first reset — the pose is pinned so it is valid for every episode.
    # Per-episode scope (image in obs): rebuild each reset because the pose changes.
    # The α grid spans the action's flip-angle range so any step α is interpolated
    # (never extrapolated); a single fixed α would be wrong for learned-α runs.
    if env.forward_model === :bloch && env.water_model === :cached_perline &&
       (env.cached_water === nothing || !_e2_cache_globally(env))
        α_grid = deg2rad.(e2_action_lo(env)[4]:5.0:e2_action_hi(env)[4])
        env.cached_water = build_cached_water_model(
            water_phantom, scanner_for_field(env.cfg_field);
            FOV = env.FOV, Nfe = env.Nfe, Npe = env.Npe,
            α_grid = α_grid, use_gpu = env.use_gpu)
    end
    # Cache background pixel mask (zero-occupancy pixels, eroded by 1 px)
    # so per-step `e2_image_stats` and `e2_dual_acq_snr_report` don't
    # re-derive it on every call. Phantom is constant within an episode.
    # No image under :analytic → no mask.
    env.background_mask      = phantom === nothing ? nothing :
        background_mask(phantom, env.Npe, env.Nfe, env.FOV; erosion_px = 1)

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
    e2_step!(env, action_vec, stop=false) → (obs, reward, done, info_dict)

`action_vec` is a 4-element vector [TI_s, TE_s, TR_s, α_deg].
All values should lie within `e2_action_lo`/`e2_action_hi`.

`stop` is the learned STOP decision (only honoured when `env.allow_stop`): the
current block is still executed and the fit updated, then the episode ends. There
is no minimum-block guard — a stop before a valid fit (<2 blocks) ends the episode
with `mape=1.0`, so the agent is penalised for stopping too early and learns to
avoid it.
"""
function e2_step!(env::E2Env, action_vec, stop::Bool = false)
    env.done && error("Episode done; call e2_reset! first.")
    stop_now = env.allow_stop && stop

    TI        = Float64(action_vec[1])
    TE        = Float64(action_vec[2])
    TR        = Float64(action_vec[3])
    α_exc_deg = Float64(action_vec[4])
    TI_requested = TI
    TE_requested = TE
    TR_requested = TR

    # Ensure TR can accommodate the requested TI + TE with recovery headroom.
    # Lift TR up rather than capping TI down — capping TI silently removed
    # the long-T1 regime when the agent chose a small TR (E1-style failure).
    TR_min_required = (TI + TE) / e2_tr_headroom(env)
    TR = max(TR, TR_min_required)
    TR_lift_amount = TR - TR_requested
    TR_lifted = TR_lift_amount > 1e-12
    TE_max_allowed = TR * 0.30
    TE = min(TE, TR * 0.30)
    TE_clamped = TE < TE_requested - 1e-12
    TE_clamp_amount = TE_requested - TE
    action_repaired = TR_lifted || TE_clamped

    # Scan time for this block (Npe shots × TR per shot).
    block_time = env.Npe * TR

    # Scan-time budget + fitter-viability guard. End the episode WITHOUT
    # simulating this block when either:
    #  • it would overrun the budget; or
    #  • it would be a lone first block with no room for a second (F1+ needs
    #    ≥2 samples to fit T1 and A).
    # The discarded action is not executed; no simulate() call is made.
    tr_floor     = e2_action_lo(env)[3]
    min_followup = env.Npe * tr_floor
    overruns   = env.time_used_s + block_time > env.time_budget_s
    lone_first = env.n_blocks == 0 &&
                 env.time_used_s + block_time + min_followup > env.time_budget_s
    if overruns || lone_first
        env.done = true
        final_mape = env.n_blocks >= 2 ? clamp(_e2_mape(env), 0.0, 1.0) : 1.0
        # Discarded block is not executed → not charged scan time (block_time=0).
        reward = _e2_reward(env, final_mape, env.prev_mape, 0.0, true)
        env.prev_mape = final_mape
        info = Dict{String,Any}(
            "mape"            => final_mape,
            "T1_true"         => copy(env.T1_true),
            "T1_est"          => copy(env.T1_est),
            "sphere_indices"  => copy(env.sphere_indices),
            "n_blocks"        => env.n_blocks,
            "time_s"          => env.time_used_s,
            "block_time"      => 0.0,
            "TI"              => TI,
            "TE"              => TE,
            "TR"              => TR,
            "TI_requested"    => TI_requested,
            "TE_requested"    => TE_requested,
            "TR_requested"    => TR_requested,
            "TI_executed"     => TI,
            "TE_executed"     => TE,
            "TR_executed"     => TR,
            "TR_min_required" => TR_min_required,
            "TR_lifted"       => TR_lifted,
            "TR_lift_amount"  => TR_lift_amount,
            "TE_max_allowed"  => TE_max_allowed,
            "TE_clamped"      => TE_clamped,
            "TE_clamp_amount" => TE_clamp_amount,
            "action_repaired" => action_repaired,
            "alpha_deg"       => α_exc_deg,
            "budget_exceeded" => true,
            "stop_requested"  => stop_now,
        )
        return (_e2_observation(env), reward, true, info)
    end

    # Synthesise per-sphere signals (Koma sim + recon in :bloch, closed-form in
    # :analytic) and update the running T1 estimates.
    signals = _e2_sphere_signals(env, TI, TE, TR, α_exc_deg)
    _e2_update_t1_estimates!(env, signals, TI, TR, deg2rad(α_exc_deg))

    env.time_used_s += block_time
    env.n_blocks    += 1

    # mape=1.0 until ≥2 samples (no valid fit yet); clamp to [0,1].
    mape = env.n_blocks >= 2 ? clamp(_e2_mape(env), 0.0, 1.0) : 1.0

    # Episode termination — determined BEFORE reward so :terminal_only knows
    # whether this is the terminal step. Beyond max_blocks, also stop once no
    # further block could fit the remaining budget even at the TR floor
    # (e2_action_lo), so we don't burn a step the budget guard would reject.
    if env.n_blocks >= env.max_blocks ||
       env.time_used_s + min_followup > env.time_budget_s ||
       stop_now
        env.done = true
    end

    reward = _e2_reward(env, mape, env.prev_mape, block_time, env.done)
    env.prev_mape = mape

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
        "TI_requested" => TI_requested,
        "TE_requested" => TE_requested,
        "TR_requested" => TR_requested,
        "TI_executed" => TI,
        "TE_executed" => TE,
        "TR_executed" => TR,
        "TR_min_required" => TR_min_required,
        "TR_lifted" => TR_lifted,
        "TR_lift_amount" => TR_lift_amount,
        "TE_max_allowed" => TE_max_allowed,
        "TE_clamped" => TE_clamped,
        "TE_clamp_amount" => TE_clamp_amount,
        "action_repaired" => action_repaired,
        "alpha_deg"   => α_exc_deg,
        "stop_requested" => stop_now,
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
    env.forward_model === :analytic &&
        error("e2_image_stats: no image under forward_model = :analytic")
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
and return a full `SNRReport`. Costs 1 simulator call; the clean k-space is
cached and both noisy A/B images are generated from it. Uses default reference
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
    env.forward_model === :analytic &&
        error("e2_dual_acq_snr_report: no image under forward_model = :analytic")
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

    raw = Suppressor.@suppress simulate(
        env.phantom, seq, scanner;
        sim_params = Dict{String,Any}("gpu" => env.use_gpu))
    ksp_clean = raw_to_kspace(raw, env.Npe, env.Nfe)

    # In :cached_perline, env.phantom is spheres-only — add the cached background
    # water so the SNR reflects the same scene the training steps reconstruct.
    if env.water_model === :cached_perline
        ksp_clean = ksp_clean .+ cached_water_ksp(env.cached_water,
                                                  Float64(TI), Float64(TR), α_exc,
                                                  Float64(TE))
    end

    snr_report_from_clean(ksp_clean, env.noise_sigma_abs;
                          sphere_px = env.sphere_px,
                          bg_mask = env.background_mask,
                          rng = rng,
                          phase_sensitive = env.phase_sensitive,
                          roi_radius = Int(roi_radius))
end

# Aliases for juliacall (Python can't reach names ending in `!`)
const e2_reset_b = e2_reset!
const e2_step_b  = e2_step!
const e2_image_stats_b = e2_image_stats
const e2_dual_acq_snr_report_b = e2_dual_acq_snr_report

#!/usr/bin/env julia
# scripts/snr_sweep.jl
#
# Sweep absolute noise σ across a clinically-relevant range and dump
# (σ, measured snr_ksp / snr_nema / snr_dual, fit MAPE) for each level,
# plus reconstructed images per (σ, block) so we can visually inspect the
# effect of noise on recon quality. Outputs are consumed by
# scripts/snr_sweep.py for the report-ready figures.
#
# ═════════════════════════════════════════════════════════════════════════════
# WHY σ IS THE INPUT
# ═════════════════════════════════════════════════════════════════════════════
#
# The previous version calibrated σ from the FIRST block's k-space RMS divided
# by a user-facing SNR target. That coupled σ to whichever TI/TR happened to
# sit in slot 1: a short-TI block-1 has low signal -> small σ -> the remaining
# blocks look artificially high-SNR. Comparisons across schedules were not
# apples-to-apples.
#
# `simulate()` is deterministic, so this script now caches one noise-free
# k-space per block, then adds noise of magnitude σ per sweep point — the
# sweep variable is σ itself. We REPORT snr_ksp post-hoc as the mean over
# all blocks: snr_ksp_measured = mean_b(ksp_rms[b]) / σ.
#
# ═════════════════════════════════════════════════════════════════════════════
# THREE SNR DEFINITIONS — only one is comparable to clinical MRI literature.
# ═════════════════════════════════════════════════════════════════════════════
#
#   (a) `snr_ksp_measured = mean(ksp_rms[b]) / σ`.  Internal calibration
#       metric; non-standard.  By Parseval, ksp_rms is image-domain RMS
#       averaged over EVERY pixel — dominated by empty background on a sparse
#       phantom, so this understates the in-tissue SNR by roughly
#       (image area)/(phantom area).
#
#   (b) `snr_nema_peak_a`:  NEMA single-image SNR — mean(sphere ROI) /
#       (std(background)/0.6551). On coarse grids (Nfe×Npe ≲ 64²) Gibbs
#       ringing contaminates "background" pixels, so this reads
#       SYSTEMATICALLY LOW. Erosion helps but doesn't eliminate it.
#
#   (c) `snr_dual_peak`:  NEMA MS-1 dual-acquisition SNR. Two independent
#       noise realisations A,B; (A−B) cancels structured signal exactly,
#       leaving zero-mean Gaussian noise. Divide by √2. GOLD STANDARD —
#       use this when comparing to clinical references.
#
#   Expected ordering on coarse grids: snr_dual > snr_nema > snr_ksp.
#
# ═════════════════════════════════════════════════════════════════════════════
# Usage
# ─────
# julia --project=. scripts/snr_sweep.jl                       # default sweep
# julia --project=. scripts/snr_sweep.jl --sigmas 1e-4,1e-3,1e-2,1e-1
# julia --project=. scripts/snr_sweep.jl --budget 160 --npe 32 --nfe 64
# julia --project=. scripts/snr_sweep.jl --field T3            # 3 T fleet + scanner
# julia --project=. scripts/snr_sweep.jl --clean-recon         # Hamming+ROI
#
# Output: runs/snr_sweep/snr_sweep.csv           (one row per σ)
#         runs/snr_sweep/per_sphere_mape.csv     (one row per sphere×σ)
#         runs/snr_sweep/run_sigma_<σ>.json      (full per-run dump)
#         runs/snr_sweep/config.json             (sweep params for repro)
#         runs/snr_sweep/images/noise_free.npz   (block_<b> ⇒ Float32 [H,W])
#         runs/snr_sweep/images/sigma_<σ>.npz    (block_<b> ⇒ noisy recon)
# ═════════════════════════════════════════════════════════════════════════════

# TODO: LOOK AT HOW CLEAN RECON CHANGES NOISE VALUES

using QalibreMDPhantom, KomaMRI, Suppressor
using DelimitedFiles, Random, Statistics, Printf, JSON, NPZ
using FFTW: ifft, fftshift, ifftshift

# ─── Defaults (match scripts/t1_fit_vs_true.jl) ──────────────────────────────
const FOV           = 0.2      # m
const VOXEL_MM      = 1.0
const ALPHA_EXC_DEG = 90.0
const TE_DEFAULT    = 0.020    # s

# ─── CLI ─────────────────────────────────────────────────────────────────────
struct SweepConfig
    sigmas::Vector{Float64}
    budget_s::Float64
    Npe::Int
    Nfe::Int
    clean_recon::Bool
    outdir::String
    n_seeds::Int
    voxel_mm::Float64
    water::Bool
    slice_mm::Float64
    slice_center_mm::Float64
    field::Symbol
end

function parse_args()
    # Defaults: budget/grid match scripts/t1_fit_vs_true.jl. The σ list is
    # picked AFTER --voxel-mm is parsed (see below) because per-voxel signal
    # — and therefore the σ at which the fit collapses — scales as the
    # number of spins per voxel, i.e. as (1/voxel_mm)³. The 3 mm list is
    # the empirical baseline (collapse around σ≈30–40); 1 mm and 0.5 mm
    # are this list multiplied by (3/voxel_mm)³ (27× and 216×).
    sigmas      = Float64[]   # filled in after --voxel-mm parsing
    budget_s    = 160.0
    Npe         = 32
    Nfe         = 64
    clean_recon = false
    outdir      = nothing
    n_seeds     = 15
    voxel_mm    = VOXEL_MM
    water       = false
    slice_mm        = 0.0    # --slice-mm: slab thickness; 0 → auto (voxel_mm)
    slice_center_mm = NaN    # --slice-center-mm: slab centre; NaN → auto (PLATE_Z_MM.T1)
    field           = :T15   # --field: :T15 (1.5 T) or :T3 (3 T); sets T1/T2 fleet + scanner B0
    sigmas_user = false
    i = 1
    while i <= length(ARGS)
        if ARGS[i] == "--sigmas" && i < length(ARGS)
            sigmas = [parse(Float64, s) for s in split(ARGS[i+1], ",")]
            sigmas_user = true; i += 2
        elseif ARGS[i] == "--budget" && i < length(ARGS)
            budget_s = parse(Float64, ARGS[i+1]); i += 2
        elseif ARGS[i] == "--npe" && i < length(ARGS)
            Npe = parse(Int, ARGS[i+1]); i += 2
        elseif ARGS[i] == "--nfe" && i < length(ARGS)
            Nfe = parse(Int, ARGS[i+1]); i += 2
        elseif ARGS[i] == "--clean-recon"
            clean_recon = true; i += 1
        elseif ARGS[i] == "--out" && i < length(ARGS)
            outdir = ARGS[i+1]; i += 2
        elseif ARGS[i] == "--seeds" && i < length(ARGS)
            n_seeds = parse(Int, ARGS[i+1]); i += 2
        elseif ARGS[i] == "--voxel-mm" && i < length(ARGS)
            voxel_mm = parse(Float64, ARGS[i+1]); i += 2
        elseif ARGS[i] == "--water"
            water = true; i += 1
        elseif ARGS[i] == "--slice-mm" && i < length(ARGS)
            slice_mm = parse(Float64, ARGS[i+1]); i += 2
        elseif ARGS[i] == "--slice-center-mm" && i < length(ARGS)
            slice_center_mm = parse(Float64, ARGS[i+1]); i += 2
        elseif ARGS[i] == "--field" && i < length(ARGS)
            field = Symbol(ARGS[i+1]); i += 2
        else
            i += 1
        end
    end
    field ∈ (:T15, :T3) || error("--field must be T15 or T3, got $field")
    # Default outdir is voxel-size-suffixed so 1 mm and 3 mm sweeps don't
    # overwrite each other (the SNR characteristics differ drastically).
    if outdir === nothing
        vox_tag   = replace(@sprintf("%gmm", voxel_mm), "." => "p")
        water_tag = water ? "_water" : ""
        field_tag = field === :T15 ? "" : "_$(field)"
        outdir    = joinpath(@__DIR__, "runs", "snr_sweep_voxel_$(vox_tag)$(water_tag)$(field_tag)")
    end
    # Default σ list (when --sigmas not given): scale the 3 mm baseline by
    # (3/voxel_mm)³ — per-voxel signal grows cubically with spin count, so
    # the σ at which the fit collapses tracks the same scaling.
    if !sigmas_user
        baseline_3mm = [0.0, 1.0, 3.0, 8.0, 12.0, 16.0, 22.0, 30.0, 40.0, 45.0, 50.0, 65.0]
        scale = (3.0 / voxel_mm)^3
        sigmas = [round(σ * scale; sigdigits = 3) for σ in baseline_3mm]
    end
    SweepConfig(sigmas, budget_s, Npe, Nfe, clean_recon, outdir, n_seeds, voxel_mm, water,
                slice_mm, slice_center_mm, field)
end

# Filesystem-friendly σ label used in filenames + JSON keys.
σ_label(σ::Real) = replace(@sprintf("%.4g", σ), "e+0" => "e+", "e-0" => "e-")

# ─── Per-σ run ───────────────────────────────────────────────────────────────
"""
    run_one(σ; cfg, ksp_clean, ...) → NamedTuple

Run the fit pipeline at a single σ. Adds fresh noise to cached `ksp_clean[b]`
arrays and re-fits per sphere. Returns NamedTuple with metrics, per-block
noisy magnitude images, and a `MultiBlockSNRReport` (per-block + pooled
across the whole schedule) — only when `want_snr_report` is true.
"""
function run_one(σ::Float64; seed::Int, cfg::SweepConfig,
                  ksp_clean::Vector{<:AbstractMatrix{<:Complex}},
                  ksp_rms_mean::Float64,
                  sphere_px, sphere_px_raw,
                  bg_mask_raw, descs,
                  TIs::Vector{Float64}, TRs::Vector{Float64},
                  want_snr_report::Bool = true)
    n_blocks  = length(ksp_clean)
    n_spheres = length(descs)
    T1_true   = [d.T1 for d in descs]

    img_pad    = cfg.clean_recon ? 2 : 1
    roi_radius = cfg.clean_recon ? 1 : 0

    α_exc = deg2rad(ALPHA_EXC_DEG)
    sin_α = abs(sin(α_exc))

    block_TIs    = [Float64[] for _ in 1:n_spheres]
    block_TRs    = [Float64[] for _ in 1:n_spheres]
    block_α_excs = [Float64[] for _ in 1:n_spheres]
    block_mags   = [Float64[] for _ in 1:n_spheres]

    block_imgs = Vector{Matrix{Float32}}(undef, n_blocks)

    rng = MersenneTwister(seed * 100003 + round(Int, 1e6 * σ))

    for blk in 1:n_blocks
        ksp = copy(ksp_clean[blk])
        add_noise!(ksp, σ; rng = rng)

        img = cfg.clean_recon ?
            kspace_to_image(ksp; pad_factor = img_pad, hamming = true,
                                  phase_sensitive = false) :
            kspace_to_image(ksp; phase_sensitive = false)
        block_imgs[blk] = Float32.(img)

        for i in 1:n_spheres
            ipe, ife = sphere_px[i]
            mag_i = roi_mean(img, ipe, ife; r = roi_radius) / sin_α
            push!(block_TIs[i],    TIs[blk])
            push!(block_TRs[i],    TRs[blk])
            push!(block_α_excs[i], α_exc)
            push!(block_mags[i],   mag_i)
        end
    end

    # ─── Multi-block SNR report (per-block + pooled across schedule) ─────────
    # Independent A/B noise realisations per block, no `simulate()` calls —
    # we already have `ksp_clean[b]`. Uses the raw (non-padded, non-hamming)
    # recon for SNR measurement even when `clean_recon=true`, so the
    # background/diff statistics are taken on the same image space as
    # `bg_mask_raw`/`sphere_px_raw`. Only seed 1 builds it; downstream seeds
    # share the same characterisation of the schedule.
    mb_rep::Union{Nothing,MultiBlockSNRReport} = nothing
    if want_snr_report
        per_block_reports = Vector{SNRReport}(undef, n_blocks)
        imgs_a = Vector{Matrix{Float32}}(undef, n_blocks)
        imgs_b = Vector{Matrix{Float32}}(undef, n_blocks)
        for b in 1:n_blocks
            ab = QalibreMDPhantom._make_ab_images(ksp_clean[b], σ; rng = rng,
                                                  phase_sensitive = false)
            imgs_a[b] = Float32.(ab.img_a)
            imgs_b[b] = Float32.(ab.img_b)
            ksp_rms_b = sqrt(sum(abs2, ksp_clean[b]) / length(ksp_clean[b]))
            snr_ksp_b = σ > 0 ? ksp_rms_b / Float64(σ) : 0.0
            image_rep = image_snr_report(ab.img_a, ab.img_b,
                                         sphere_px_raw, bg_mask_raw)
            per_block_reports[b] = SNRReport(image_rep, ksp_rms_b,
                                             Float64(σ), snr_ksp_b)
        end
        pooled = pooled_image_snr_report(imgs_a, imgs_b,
                                         sphere_px_raw, bg_mask_raw)
        mb_rep = MultiBlockSNRReport(
            per_block_reports, pooled,
            [r.image.snr_dual_peak    for r in per_block_reports],
            [r.image.snr_nema_peak_a  for r in per_block_reports],
        )
    end

    # ─── Fit per sphere ─────────────────────────────────────────────────────
    T1_fit   = zeros(n_spheres)
    T1_sigma = fill(NaN, n_spheres)
    mapes    = zeros(n_spheres)
    α_inv_vec = fill(π, n_blocks)
    # The profile-likelihood needs a positive σ; at σ=0 use a tiny ε just
    # to keep the noise model well-defined (the data is noise-free anyway).
    fit_sigma = σ > 0 ? σ : 1e-9
    for i in 1:n_spheres
        fit = fit_t1_generalized_ir(
            block_TIs[i], α_inv_vec, block_mags[i];
            TRs = block_TRs[i], α_excs = block_α_excs[i], Npe = cfg.Npe,
            T1_range = (0.01, 3.0), n_grid = 500,
            abs_noise_sigma = fit_sigma,
            sigma_method = :profile_likelihood, signed = false,
        )
        T1_fit[i]   = fit.T1
        T1_sigma[i] = fit.T1_sigma
        mapes[i]    = abs(fit.T1 - T1_true[i]) / T1_true[i] * 100
    end

    snr_ksp_measured = ksp_rms_mean / σ   # Inf at σ=0 — kept as-is, no special case

    (; σ, snr_ksp_measured, mb_rep, T1_true, T1_fit, T1_sigma, mapes,
       mape_mean = mean(mapes), mape_max = maximum(mapes),
       mape_median = median(mapes), block_imgs)
end

# ─── Main ────────────────────────────────────────────────────────────────────
function main()
    cfg = parse_args()
    mkpath(cfg.outdir)
    img_dir = joinpath(cfg.outdir, "images")
    mkpath(img_dir)

    println("="^60)
    println("SNR sweep (σ-input)")
    println("  sigmas        = $(cfg.sigmas)")
    println("  budget        = $(cfg.budget_s) s")
    println("  Nfe × Npe     = $(cfg.Nfe) × $(cfg.Npe)")
    println("  field         = $(cfg.field)")
    println("  clean_recon   = $(cfg.clean_recon)")
    println("  n_seeds       = $(cfg.n_seeds)")
    println("  voxel_mm      = $(cfg.voxel_mm)")
    println("  water         = $(cfg.water)")
    println("  output dir    = $(cfg.outdir)")
    println("="^60)

    # Build phantom + sphere geometry ONCE.
    # Resolve slice geometry. Centre on the T1 plate z; slab thickness defaults
    # to one voxel so spins and recon agree on a single in-plane slice. Both
    # values are forwarded to PhantomConfig so the builder applies the z-mask —
    # we don't post-filter phantom.z ourselves.
    slice_center = isnan(cfg.slice_center_mm) ? QalibreMDPhantom.PLATE_Z_MM.T1 : cfg.slice_center_mm
    slice_thick  = cfg.slice_mm > 0 ? cfg.slice_mm : cfg.voxel_mm
    pcfg     = PhantomConfig(field = cfg.field, voxel_size_mm = cfg.voxel_mm,
                              include_plates     = cfg.water ? [:T1, :water] : [:T1],
                              slice_thickness_mm = slice_thick,
                              slice_center_mm    = slice_center)
    phantom = build_phantom(pcfg)
    println("Slab: z=$(slice_center) ± $(slice_thick/2) mm")
    scanner = scanner_for_field(pcfg)
    descs   = sphere_descriptors(:T1, pcfg; rng = MersenneTwister(0))
    T1_true = [d.T1 for d in descs]
    centres = [d.centre for d in descs]
    n_spheres = length(descs)
    println("Phantom: $(length(phantom.x)) spins, $n_spheres spheres")

    # Sphere pixel locations on image grid (used for fit) and raw grid (SNR)
    img_pad = cfg.clean_recon ? 2 : 1
    Npe_img = cfg.Npe * img_pad
    Nfe_img = cfg.Nfe * img_pad
    sphere_px = NTuple{2,Int}[]
    for c in centres
        push!(sphere_px, (phys_to_pixel(c[2], Npe_img, FOV), phys_to_pixel(c[1], Nfe_img, FOV)))
    end
    sphere_px_raw = NTuple{2,Int}[]
    for c in centres
        push!(sphere_px_raw, (phys_to_pixel(c[2], cfg.Npe, FOV), phys_to_pixel(c[1], cfg.Nfe, FOV)))
    end
    bg_mask_raw = background_mask(phantom, cfg.Npe, cfg.Nfe, FOV; erosion_px = 1)
    println("Background mask: $(sum(bg_mask_raw)) / $(length(bg_mask_raw)) pixels")

    # Phantom-corrected SNR scaling.  ksp_rms averages over EVERY image pixel
    # (Parseval), but most pixels are empty background — diluting snr_ksp.
    # Multiply by √(N_total / N_phantom_pixels) to convert to in-tissue RMS
    # (so `snr_ksp_phantom = snr_ksp_measured · phantom_corr`).
    occ = phantom_occupancy(phantom, cfg.Npe, cfg.Nfe, FOV)
    n_phantom_px = sum(occ .> 0)
    n_total_px   = cfg.Npe * cfg.Nfe
    phantom_corr = sqrt(n_total_px / n_phantom_px)
    @printf("Phantom occupancy: %d / %d pixels  →  phantom_corr = %.3f\n",
            n_phantom_px, n_total_px, phantom_corr)

    # CR-optimal schedule
    println("\nOptimising CR schedule (budget=$(cfg.budget_s) s, Npe=$(cfg.Npe))…")
    sched = cr_optimize_sweep(T1_true;
        budget_s = cfg.budget_s, Npe = cfg.Npe,
        n_block_grid = [4, 6, 8, 10],   # 14/18 never beat the floor — dropped for speed
        n_starts = 1000, n_refine = 10)
    TIs_opt = sched.schedule.TIs
    TRs_opt = sched.schedule.TRs
    n_blocks = sched.n_blocks
    println("  n_blocks = $n_blocks  CR obj = $(round(sched.schedule.L, sigdigits=4))")
    println("  TIs: $(round.(TIs_opt, digits=3))")
    println("  TRs: $(round.(TRs_opt, digits=3))")

    # ─── Noise-free pre-pass: simulate each block ONCE, cache ksp_clean[b] ──
    println("\nNoise-free pre-pass ($n_blocks simulate calls)…")
    α_exc = deg2rad(ALPHA_EXC_DEG)
    ksp_clean = Vector{Matrix{ComplexF32}}(undef, n_blocks)
    ksp_rms   = zeros(Float64, n_blocks)
    for blk in 1:n_blocks
        seq = Suppressor.@suppress ir_se_2d_sequence(
            TIs_opt[blk], TE_DEFAULT, TRs_opt[blk];
            α_exc = α_exc, FOV = FOV, Nfe = cfg.Nfe, Npe = cfg.Npe,
        )
        raw = Suppressor.@suppress simulate(phantom, seq, scanner)
        ksp = raw_to_kspace(raw, cfg.Npe, cfg.Nfe)
        ksp_clean[blk] = ksp
        ksp_rms[blk]   = sqrt(sum(abs2, ksp) / length(ksp))
        @printf("  block %2d  TI=%.3f  TR=%.3f  ksp_rms=%.5f\n",
                blk, TIs_opt[blk], TRs_opt[blk], ksp_rms[blk])
    end
    ksp_rms_mean = mean(ksp_rms)
    println("  mean ksp_rms = $(round(ksp_rms_mean, sigdigits=5))")

    # Save noise-free reference images (one per block) ----------------------
    noise_free_npz = Dict{String,Array{Float32,2}}()
    for blk in 1:n_blocks
        img = cfg.clean_recon ?
            kspace_to_image(ksp_clean[blk]; pad_factor = img_pad,
                            hamming = true, phase_sensitive = false) :
            kspace_to_image(ksp_clean[blk]; phase_sensitive = false)
        noise_free_npz["block_$blk"] = Float32.(img)
    end
    npzwrite(joinpath(img_dir, "noise_free.npz"), noise_free_npz)
    println("  wrote images/noise_free.npz  (H×W = $(size(first(values(noise_free_npz)))))")

    # ─── Sweep loop ──────────────────────────────────────────────────────────
    # Each σ is repeated `cfg.n_seeds` times (different RNG seeds, same
    # cached ksp_clean). Per-sphere MAPE is averaged across seeds; the
    # per-σ summary then mean/median/max-reduces over spheres.
    rows = Vector{NTuple{20,Float64}}()
    per_sphere_rows = Tuple{Float64,String,Float64,Float64,Float64,Float64,Float64}[]

    for σ in cfg.sigmas
        println("\n── σ = $σ  (n_seeds = $(cfg.n_seeds)) ───────────")
        seed_results = [
            run_one(σ; seed = s, cfg = cfg, ksp_clean = ksp_clean,
                    ksp_rms_mean = ksp_rms_mean,
                    sphere_px = sphere_px, sphere_px_raw = sphere_px_raw,
                    bg_mask_raw = bg_mask_raw,
                    descs = descs, TIs = TIs_opt, TRs = TRs_opt,
                    want_snr_report = (s == 1))
            for s in 1:cfg.n_seeds
        ]
        mb_rep = seed_results[1].mb_rep
        pooled = mb_rep.pooled
        # NB. At σ=0 the dual-acq metric divides by std(A−B)≈0 (floating-
        # point noise between independent noise draws), giving an enormous
        # but not infinite snr_dual. Keep the raw value; downstream
        # formatting falls back to scientific notation.
        finite_or_nan(x) = isfinite(x) ? x : NaN
        bg_std    = finite_or_nan(pooled.background_std_a)
        snr_nema  = finite_or_nan(pooled.snr_nema_peak_a)
        diff_std  = finite_or_nan(pooled.diff_roi_std)
        snr_dual  = finite_or_nan(pooled.snr_dual_peak)   # = pooled across blocks
        snr_dual_per_sphere = [finite_or_nan(x) for x in pooled.snr_dual_per_sphere]
        # Per-block snr_dual range — characterises how SNR varies through the
        # schedule (block 1 vs middle vs last). Useful diagnostic alongside
        # the headline pooled value.
        block_dual = [finite_or_nan(x) for x in mb_rep.block_snr_dual_peak]
        block_dual_finite = filter(isfinite, block_dual)
        snr_dual_block_min    = isempty(block_dual_finite) ? NaN : minimum(block_dual_finite)
        snr_dual_block_median = isempty(block_dual_finite) ? NaN : median(block_dual_finite)
        snr_dual_block_max    = isempty(block_dual_finite) ? NaN : maximum(block_dual_finite)

        # Per-sphere averages across seeds (used for max-MAPE and per-sphere CSV)
        mapes_avg = [mean([sr.mapes[i] for sr in seed_results]) for i in 1:n_spheres]
        T1_fit_avg = [mean([sr.T1_fit[i] for sr in seed_results]) for i in 1:n_spheres]
        T1_sigma_avg = [mean([isnan(sr.T1_sigma[i]) ? 0.0 : sr.T1_sigma[i]
                              for sr in seed_results]) for i in 1:n_spheres]
        # Mean curve: per-seed sphere-mean → grand mean across seeds, with
        # ASYMMETRIC one-sided std error bars (lower from samples below the
        # mean, upper from samples above). MAPE's seed distribution is
        # right-skewed at high σ (rare catastrophic fits), so a symmetric
        # std would imply implausibly large lower whiskers.
        # Median curve: per-seed sphere-median → grand mean across seeds.
        # The median already collapses the catastrophic-fit tail, so a
        # single symmetric std is fine.
        per_seed_sphere_mean   = [mean(sr.mapes)   for sr in seed_results]
        per_seed_sphere_median = [median(sr.mapes) for sr in seed_results]

        function asym_std(xs::AbstractVector, c::Real)
            lo_xs = filter(x -> x < c, xs)
            hi_xs = filter(x -> x > c, xs)
            lo = isempty(lo_xs) ? 0.0 : sqrt(mean((x - c)^2 for x in lo_xs))
            hi = isempty(hi_xs) ? 0.0 : sqrt(mean((x - c)^2 for x in hi_xs))
            (lo, hi)
        end

        mape_mean   = mean(per_seed_sphere_mean)
        mape_median = mean(per_seed_sphere_median)
        mape_mean_std_lo, mape_mean_std_hi =
            cfg.n_seeds > 1 ? asym_std(per_seed_sphere_mean, mape_mean) : (0.0, 0.0)
        mape_median_std = cfg.n_seeds > 1 ? std(per_seed_sphere_median) : 0.0
        mape_max    = maximum(mapes_avg)

        @printf("  snr_ksp=%.3g  snr_ksp_phantom=%.3g  snr_nema_a=%.3g  snr_dual=%.3g  (per-block %.3g–%.3g)  MAPE mean=%.2f (-%.2f/+%.2f)%%  max=%.2f%%\n",
                seed_results[1].snr_ksp_measured,
                seed_results[1].snr_ksp_measured * phantom_corr,
                snr_nema, snr_dual,
                snr_dual_block_min, snr_dual_block_max,
                mape_mean, mape_mean_std_lo, mape_mean_std_hi, mape_max)

        snr_ksp_meas = seed_results[1].snr_ksp_measured
        push!(rows, (
            σ, ksp_rms_mean,
            snr_ksp_meas, snr_ksp_meas * phantom_corr,
            bg_std, snr_nema, diff_std, snr_dual,
            snr_dual_block_min, snr_dual_block_median, snr_dual_block_max,
            mape_mean, mape_median, mape_max,
            mape_mean_std_lo, mape_mean_std_hi, mape_median_std,
            Float64(cfg.n_seeds), Float64(n_blocks), Float64(cfg.clean_recon),
        ))
        for i in 1:n_spheres
            push!(per_sphere_rows,
                  (σ, String(descs[i].label),
                   seed_results[1].T1_true[i], T1_fit_avg[i], T1_sigma_avg[i],
                   mapes_avg[i], snr_dual_per_sphere[i]))
        end

        # Per-σ noisy images: save seed-1 reconstruction
        npz = Dict{String,Array{Float32,2}}()
        for blk in 1:n_blocks
            npz["block_$blk"] = seed_results[1].block_imgs[blk]
        end
        npzwrite(joinpath(img_dir, "sigma_$(σ_label(σ)).npz"), npz)

        # Per-σ JSON dump
        open(joinpath(cfg.outdir, "run_sigma_$(σ_label(σ)).json"), "w") do io
            JSON.print(io, Dict(
                "sigma"             => σ,
                "snr_ksp_measured"  => seed_results[1].snr_ksp_measured,
                "ksp_rms_mean"      => ksp_rms_mean,
                "snr_report"        => multi_block_snr_report_to_dict(mb_rep),
                "T1_true"           => seed_results[1].T1_true,
                "T1_fit_seed_mean"  => T1_fit_avg,
                "T1_sigma_seed_mean" => T1_sigma_avg,
                "mape_pct_seed_mean" => mapes_avg,
                "mape_mean_pct"     => mape_mean,
                "mape_mean_pct_std_lo" => mape_mean_std_lo,
                "mape_mean_pct_std_hi" => mape_mean_std_hi,
                "mape_median_pct_std" => mape_median_std,
                "mape_max_pct"      => mape_max,
                "n_seeds"           => cfg.n_seeds,
                "per_seed_sphere_mean_mape" => per_seed_sphere_mean,
                "n_blocks"          => n_blocks,
                "TIs_s"             => TIs_opt,
                "TRs_s"             => TRs_opt,
                "Nfe"               => cfg.Nfe,
                "Npe"               => cfg.Npe,
                "clean_recon"       => cfg.clean_recon,
                "budget_s"          => cfg.budget_s,
            ), 2)
            println(io)
        end
    end

    # ─── Write summary CSV ───────────────────────────────────────────────────
    summary_path = joinpath(cfg.outdir, "snr_sweep.csv")
    open(summary_path, "w") do io
        println(io, "sigma,ksp_rms_mean,snr_ksp_measured,snr_ksp_phantom," *
                    "background_std_a,snr_nema_peak_a,diff_roi_std,snr_dual_peak," *
                    "snr_dual_block_min,snr_dual_block_median,snr_dual_block_max," *
                    "mape_mean_pct,mape_median_pct,mape_max_pct," *
                    "mape_mean_pct_std_lo,mape_mean_pct_std_hi,mape_median_pct_std," *
                    "n_seeds,n_blocks,clean_recon")
        for r in rows
            println(io, join(r, ","))
        end
    end
    println("\nWrote $summary_path")

    ps_path = joinpath(cfg.outdir, "per_sphere_mape.csv")
    open(ps_path, "w") do io
        println(io, "sigma,label,T1_true_s,T1_fit_s,T1_sigma_s,mape_pct,snr_dual")
        for r in per_sphere_rows
            println(io, r[1], ",", r[2], ",", r[3], ",", r[4], ",",
                     r[5], ",", r[6], ",", r[7])
        end
    end
    println("Wrote $ps_path")

    # ─── Reproducibility config ─────────────────────────────────────────────
    cfg_path = joinpath(cfg.outdir, "config.json")
    open(cfg_path, "w") do io
        JSON.print(io, Dict(
            "sigmas"        => cfg.sigmas,
            "sigma_labels"  => [σ_label(σ) for σ in cfg.sigmas],
            "budget_s"      => cfg.budget_s,
            "Nfe"           => cfg.Nfe,
            "Npe"           => cfg.Npe,
            "FOV_m"         => FOV,
            "field"         => String(cfg.field),
            "voxel_mm"      => cfg.voxel_mm,
            "clean_recon"   => cfg.clean_recon,
            "water"         => cfg.water,
            "slice_thickness_mm" => slice_thick,
            "slice_center_mm"    => slice_center,
            "n_blocks"      => n_blocks,
            "TIs_s"         => TIs_opt,
            "TRs_s"         => TRs_opt,
            "alpha_exc_deg" => ALPHA_EXC_DEG,
            "TE_s"          => TE_DEFAULT,
            "ksp_rms_mean"  => ksp_rms_mean,
            "ksp_rms_per_block" => ksp_rms,
            "n_phantom_px"  => n_phantom_px,
            "n_total_px"    => n_total_px,
            "phantom_corr"  => phantom_corr,
            "n_seeds"       => cfg.n_seeds,
        ), 2)
        println(io)
    end
    println("Wrote $cfg_path")

    println("\nDone. Run scripts/snr_sweep.py to plot.")
end

main()

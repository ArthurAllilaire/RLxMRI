#!/usr/bin/env julia
# scripts/snr_sweep.jl
#
# Sweep `target_snr` across a clinically-relevant range and dump
# (NEMA MS-1 dual-acquisition SNR, fit MAPE) for each level. The output CSV
# is consumed by scripts/snr_sweep.py to produce the report-ready figure.
#
# ═════════════════════════════════════════════════════════════════════════════
# WHY THIS SWEEP EXISTS — context the report needs in §SNR-methodology
# ═════════════════════════════════════════════════════════════════════════════
#
# 1. THREE SNR DEFINITIONS, ONE TRUTHFUL NUMBER.
#
#    We use three SNR metrics; only one of them is comparable to clinical
#    MRI literature.
#
#    (a) `snr_ksp = ksp_rms / σ`. This is the internal knob the env uses to
#        calibrate noise σ from a user-facing "target_snr". It is NOT a
#        standard MRI SNR definition. By Parseval's theorem `ksp_rms` is the
#        image-domain RMS averaged over EVERY pixel — dominated by empty
#        background on a sparse phantom, so `snr_ksp` understates the
#        in-tissue SNR by roughly the ratio (image area)/(phantom area).
#
#    (b) `snr_nema_peak`: NEMA single-image SNR — mean(signal in sphere ROI)
#        divided by std(background) with the standard /0.6551 Rayleigh
#        correction (Henkelman 1985, Gudbjartsson–Patz 1995). On coarse
#        image grids (Nfe × Npe ≲ 64²) the "background" is not actually
#        signal-free: Gibbs ringing from the sharp sphere edges leaks
#        oscillations into nominally-empty pixels. The std of those pixels
#        is dominated by structured ringing, not noise, so single-image
#        NEMA reads SYSTEMATICALLY LOW. Erosion of the background mask
#        helps but doesn't eliminate it on small grids.
#
#    (c) `snr_dual_peak`: NEMA MS-1 dual-acquisition SNR (NEMA Standards
#        Publication MS-1, 2014). Take two independent noise realisations
#        A, B of the same sequence; (A − B) cancels the structured signal
#        (including Gibbs ringing) exactly, leaving zero-mean Gaussian
#        noise. Divide by √2 to undo the noise-doubling from differencing.
#        This is the GOLD STANDARD for reproducible image SNR and is what
#        clinical literature reports. Use this number — and only this
#        number — when comparing against clinical references.
#
#    Expected ordering on coarse grids: snr_dual > snr_nema > snr_ksp,
#    sometimes by a factor of ~10×. The gap between snr_dual and snr_nema
#    is the "Gibbs bias"; the gap between snr_nema and snr_ksp is the
#    "background fraction" bias.
#
# 2. CLINICAL REFERENCE RANGE.
#
#    Routine 1.5 T / 3 T quantitative T1 mapping of brain or phantom tissue
#    typically reports image SNR in the range 20–50 (NEMA MS-1 dual-acq,
#    in-tissue ROI). MR-Linac and small-voxel acquisitions push lower
#    (10–20); high-field research scans push higher (50–100). This sweep
#    spans target_snr values chosen so that the resulting `snr_dual_peak`
#    covers ~5 → 300, bracketing every plausible clinical operating point
#    and our previous training noise levels.
#
# 3. WHAT THIS SWEEP DOES NOT MEASURE.
#
#    * Per-block SNR variation (we report the dual-acq number for ONE
#      reference block per run; in training, each block has a different
#      effective SNR because signal depends on TI/TR).
#    * Per-sphere SNR — only the peak-sphere SNR is shown; the per-sphere
#      array is still dumped into config.json for the report.
#    * Pose / B0 / B1 inhomogeneity — pure noise-floor sweep.
#
# ═════════════════════════════════════════════════════════════════════════════
# Usage
# ─────
# julia --project=. scripts/snr_sweep.jl                # default sweep
# julia --project=. scripts/snr_sweep.jl --snrs 50,20,10,5,2.5,1.5,1.0
# julia --project=. scripts/snr_sweep.jl --budget 160 --npe 32 --nfe 64
# julia --project=. scripts/snr_sweep.jl --clean-recon  # apply Hamming+ROI
#
# Output: runs/snr_sweep/snr_sweep.csv          (one row per target_snr)
#         runs/snr_sweep/per_sphere_mape.csv    (one row per sphere×SNR)
#         runs/snr_sweep/run_<target_snr>.json  (full per-run dump)
# ═════════════════════════════════════════════════════════════════════════════

using QalibreMDPhantom, KomaMRI, Suppressor
using DelimitedFiles, Random, Statistics, Printf, JSON
using FFTW: ifft, fftshift, ifftshift

# ─── Defaults (match scripts/t1_fit_vs_true.jl) ──────────────────────────────
const FOV           = 0.2      # m
const VOXEL_MM      = 3.0
const ALPHA_EXC_DEG = 90.0
const TE_DEFAULT    = 0.020    # s

# ─── CLI ─────────────────────────────────────────────────────────────────────
struct SweepConfig
    snrs::Vector{Float64}
    budget_s::Float64
    Npe::Int
    Nfe::Int
    clean_recon::Bool
    outdir::String
end

function parse_args()
    snrs        = [50.0, 20.0, 10.0, 7.0, 5.0, 3.0, 2.0, 1.0]
    budget_s    = 160.0
    Npe         = 32
    Nfe         = 64
    clean_recon = false
    outdir      = joinpath(@__DIR__, "runs", "snr_sweep")
    i = 1
    while i <= length(ARGS)
        if ARGS[i] == "--snrs" && i < length(ARGS)
            snrs = [parse(Float64, s) for s in split(ARGS[i+1], ",")]; i += 2
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
        else
            i += 1
        end
    end
    SweepConfig(snrs, budget_s, Npe, Nfe, clean_recon, outdir)
end

# ─── Hamming + zero-pad recon (lifted from t1_fit_vs_true.jl) ────────────────
function clean_kspace_to_image(ksp::Matrix{ComplexF32}; pad_factor::Int = 2,
                                phase_sensitive::Bool = false)
    Npe_k, Nfe_k = size(ksp)
    w_pe = Float32.(0.54 .- 0.46 .* cos.(2π .* (0:Npe_k-1) ./ (Npe_k-1)))
    w_fe = Float32.(0.54 .- 0.46 .* cos.(2π .* (0:Nfe_k-1) ./ (Nfe_k-1)))
    ksp_w = ksp .* (w_pe .* transpose(w_fe))
    if pad_factor > 1
        Npe_pad = Npe_k * pad_factor
        Nfe_pad = Nfe_k * pad_factor
        ksp_padded = zeros(ComplexF32, Npe_pad, Nfe_pad)
        pe0 = (Npe_pad - Npe_k) ÷ 2 + 1
        fe0 = (Nfe_pad - Nfe_k) ÷ 2 + 1
        @views ksp_padded[pe0:pe0+Npe_k-1, fe0:fe0+Nfe_k-1] = ksp_w
        ksp_w = ksp_padded
    end
    img = fftshift(ifft(ifftshift(ksp_w, (1, 2)), (1, 2)), (1, 2))
    phase_sensitive ? Float32.(real.(img)) : Float32.(abs.(img))
end

# ─── Per-target_snr run ──────────────────────────────────────────────────────
"""
    run_one(target_snr; cfg, descs, phantom, sphere_px, sphere_px_raw,
             bg_mask_raw) → NamedTuple

Run the full {schedule → simulate × n_blocks → fit} pipeline at a single
target_snr. Returns NamedTuple with SNR metrics + per-sphere fit results.
Shares the precomputed phantom / sphere indices across all SNRs (the only
thing that changes between calls is the noise σ and RNG seed).
"""
function run_one(target_snr::Float64; cfg::SweepConfig,
                  descs, phantom, sphere_px, sphere_px_raw, bg_mask_raw,
                  n_blocks::Int, TIs::Vector{Float64}, TRs::Vector{Float64})
    n_spheres = length(descs)
    T1_true   = [d.T1 for d in descs]

    img_pad     = cfg.clean_recon ? 2 : 1
    roi_radius  = cfg.clean_recon ? 1 : 0

    α_exc = deg2rad(ALPHA_EXC_DEG)
    sin_α = abs(sin(α_exc))

    block_TIs    = [Float64[] for _ in 1:n_spheres]
    block_TRs    = [Float64[] for _ in 1:n_spheres]
    block_α_excs = [Float64[] for _ in 1:n_spheres]
    block_mags   = [Float64[] for _ in 1:n_spheres]

    rng = MersenneTwister(42)
    noise_sigma_abs::Float64 = 0.0
    snr_rep::Union{Nothing,SNRReport} = nothing

    for blk in 1:n_blocks
        TI = TIs[blk]; TR = TRs[blk]
        seq = Suppressor.@suppress ir_se_2d_sequence(
            TI, TE_DEFAULT, TR;
            α_exc = α_exc, FOV = FOV, Nfe = cfg.Nfe, Npe = cfg.Npe,
        )
        raw = Suppressor.@suppress simulate(phantom, seq, Scanner())
        ksp = raw_to_kspace(raw, cfg.Npe, cfg.Nfe)

        if blk == 1
            ksp_rms = sqrt(sum(abs2, ksp) / length(ksp))
            noise_sigma_abs = ksp_rms / target_snr
        end

        add_noise!(ksp, noise_sigma_abs; rng = rng)

        img = cfg.clean_recon ?
            clean_kspace_to_image(ksp; pad_factor = img_pad,
                                        phase_sensitive = false) :
            kspace_to_image(ksp; phase_sensitive = false)

        # On block 1, also produce a RAW recon for the SNR diagnostic.
        # Under --clean-recon the windowing makes pixel noise non-stationary
        # which biases NEMA / dual-acq estimates; the raw recon avoids that.
        if blk == 1
            img_raw = cfg.clean_recon ?
                kspace_to_image(ksp; phase_sensitive = false) :
                img
            snr_rep = snr_report(phantom, seq, Scanner();
                σ = noise_sigma_abs,
                sphere_px = sphere_px_raw,
                bg_mask = bg_mask_raw,
                ksp_a = ksp, img_a = img_raw,
                rng = rng, phase_sensitive = false, roi_radius = 0,
            )
        end

        for i in 1:n_spheres
            ipe, ife = sphere_px[i]
            mag_i = if roi_radius == 0
                Float64(img[ipe, ife]) / sin_α
            else
                pe_lo = clamp(ipe - roi_radius, 1, size(img, 1))
                pe_hi = clamp(ipe + roi_radius, 1, size(img, 1))
                fe_lo = clamp(ife - roi_radius, 1, size(img, 2))
                fe_hi = clamp(ife + roi_radius, 1, size(img, 2))
                Float64(mean(img[pe_lo:pe_hi, fe_lo:fe_hi])) / sin_α
            end
            push!(block_TIs[i],    TI)
            push!(block_TRs[i],    TR)
            push!(block_α_excs[i], α_exc)
            push!(block_mags[i],   mag_i)
        end
    end

    # ─── Fit per sphere ─────────────────────────────────────────────────────
    T1_fit  = zeros(n_spheres)
    T1_sigma = fill(NaN, n_spheres)
    mapes   = zeros(n_spheres)
    α_inv_vec = fill(π, n_blocks)
    for i in 1:n_spheres
        fit = fit_t1_generalized_ir(
            block_TIs[i], α_inv_vec, block_mags[i];
            TRs = block_TRs[i], α_excs = block_α_excs[i], Npe = cfg.Npe,
            T1_range = (0.01, 3.0), n_grid = 500,
            abs_noise_sigma = noise_sigma_abs,
            sigma_method = :profile_likelihood, signed = false,
        )
        T1_fit[i]   = fit.T1
        T1_sigma[i] = fit.T1_sigma
        mapes[i]    = abs(fit.T1 - T1_true[i]) / T1_true[i] * 100
    end

    (; target_snr, noise_sigma_abs, snr_rep, T1_true, T1_fit, T1_sigma,
       mapes, mape_mean = mean(mapes), mape_max = maximum(mapes),
       mape_median = median(mapes))
end

# ─── Main ────────────────────────────────────────────────────────────────────
function main()
    cfg = parse_args()
    mkpath(cfg.outdir)
    println("="^60)
    println("SNR sweep")
    println("  target_snrs   = $(cfg.snrs)")
    println("  budget        = $(cfg.budget_s) s")
    println("  Nfe × Npe     = $(cfg.Nfe) × $(cfg.Npe)")
    println("  clean_recon   = $(cfg.clean_recon)")
    println("  output dir    = $(cfg.outdir)")
    println("="^60)

    # Build phantom + sphere geometry ONCE (deterministic; shared across SNRs)
    pcfg     = PhantomConfig(field = :T15, voxel_size_mm = VOXEL_MM,
                              include_plates = [:T1])
    phantom = build_phantom(pcfg)
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
        ife = mod(round(Int, c[1] * Nfe_img / FOV) + Nfe_img ÷ 2, Nfe_img) + 1
        ipe = mod(round(Int, c[2] * Npe_img / FOV) + Npe_img ÷ 2, Npe_img) + 1
        push!(sphere_px, (ipe, ife))
    end
    sphere_px_raw = NTuple{2,Int}[]
    for c in centres
        ife = mod(round(Int, c[1] * cfg.Nfe / FOV) + cfg.Nfe ÷ 2, cfg.Nfe) + 1
        ipe = mod(round(Int, c[2] * cfg.Npe / FOV) + cfg.Npe ÷ 2, cfg.Npe) + 1
        push!(sphere_px_raw, (ipe, ife))
    end
    bg_mask_raw = background_mask(phantom, cfg.Npe, cfg.Nfe, FOV; erosion_px = 1)
    println("Background mask: $(sum(bg_mask_raw)) / $(length(bg_mask_raw)) pixels")

    # CR-optimal schedule (shared across all SNRs — same acquisition, different
    # noise). The CR objective doesn't depend on noise scale, only on T1/TR/TI
    # ratios, so this is the right way to keep the comparison apples-to-apples.
    println("\nOptimising CR schedule (budget=$(cfg.budget_s) s, Npe=$(cfg.Npe))…")
    sched = cr_optimize_sweep(T1_true;
        budget_s = cfg.budget_s, Npe = cfg.Npe,
        n_block_grid = [4, 6, 8, 10, 14, 18],
        n_starts = 1000, n_refine = 10)
    TIs_opt = sched.schedule.TIs
    TRs_opt = sched.schedule.TRs
    n_blocks = sched.n_blocks
    println("  n_blocks = $n_blocks  CR obj = $(round(sched.schedule.L, sigdigits=4))")
    println("  TIs: $(round.(TIs_opt, digits=3))")
    println("  TRs: $(round.(TRs_opt, digits=3))")

    # ─── Sweep loop ──────────────────────────────────────────────────────────
    rows = NTuple{12,Float64}[]   # one summary row per SNR
    per_sphere_rows = Tuple{Float64,String,Float64,Float64,Float64,Float64,Float64}[]

    for snr in cfg.snrs
        println("\n── target_snr = $snr ─────────────────────────────────────")
        r = run_one(snr; cfg = cfg, descs = descs, phantom = phantom,
                    sphere_px = sphere_px, sphere_px_raw = sphere_px_raw,
                    bg_mask_raw = bg_mask_raw,
                    n_blocks = n_blocks, TIs = TIs_opt, TRs = TRs_opt)
        rep = r.snr_rep
        @printf("  σ=%.4f  snr_ksp=%.2f  snr_nema=%.2f  snr_dual=%.2f  MAPE mean=%.2f%%  max=%.2f%%\n",
                r.noise_sigma_abs, rep.snr_ksp, rep.snr_nema_peak,
                rep.snr_dual_peak, r.mape_mean, r.mape_max)

        push!(rows, (
            snr, r.noise_sigma_abs,
            rep.ksp_rms, rep.snr_ksp,
            rep.background_std, rep.snr_nema_peak,
            rep.diff_roi_std,   rep.snr_dual_peak,
            r.mape_mean, r.mape_median, r.mape_max, Float64(n_blocks),
        ))
        for i in 1:n_spheres
            push!(per_sphere_rows,
                  (snr, String(descs[i].label), r.T1_true[i], r.T1_fit[i],
                   isnan(r.T1_sigma[i]) ? 0.0 : r.T1_sigma[i],
                   r.mapes[i], rep.snr_dual_per_sphere[i]))
        end

        # Per-run JSON dump (full SNRReport + fit array)
        open(joinpath(cfg.outdir, "run_target_snr_$(snr).json"), "w") do io
            JSON.print(io, Dict(
                "target_snr"      => snr,
                "noise_sigma_abs" => r.noise_sigma_abs,
                "snr_report"      => snr_report_to_dict(rep),
                "T1_true"         => r.T1_true,
                "T1_fit"          => r.T1_fit,
                "T1_sigma"        => [isnan(x) ? 0.0 : x for x in r.T1_sigma],
                "mape_pct"        => r.mapes,
                "mape_mean_pct"   => r.mape_mean,
                "mape_max_pct"    => r.mape_max,
                "n_blocks"        => n_blocks,
                "TIs_s"           => TIs_opt,
                "TRs_s"           => TRs_opt,
                "Nfe"             => cfg.Nfe,
                "Npe"             => cfg.Npe,
                "clean_recon"     => cfg.clean_recon,
                "budget_s"        => cfg.budget_s,
            ), 2)
            println(io)
        end
    end

    # ─── Write summary CSV ───────────────────────────────────────────────────
    summary_path = joinpath(cfg.outdir, "snr_sweep.csv")
    open(summary_path, "w") do io
        println(io, "target_snr,noise_sigma_abs,ksp_rms,snr_ksp," *
                    "background_std,snr_nema_peak,diff_roi_std,snr_dual_peak," *
                    "mape_mean_pct,mape_median_pct,mape_max_pct,n_blocks")
        for r in rows
            println(io, join(r, ","))
        end
    end
    println("\nWrote $summary_path")

    ps_path = joinpath(cfg.outdir, "per_sphere_mape.csv")
    open(ps_path, "w") do io
        println(io, "target_snr,label,T1_true_s,T1_fit_s,T1_sigma_s,mape_pct,snr_dual")
        for r in per_sphere_rows
            println(io, r[1], ",", r[2], ",", r[3], ",", r[4], ",",
                     r[5], ",", r[6], ",", r[7])
        end
    end
    println("Wrote $ps_path")

    println("\nDone. Run scripts/snr_sweep.py to plot.")
end

main()

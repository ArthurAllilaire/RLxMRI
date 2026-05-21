# Compare T1 fitted (from 2D image pipeline) vs T1 true (phantom ground truth).
#
# Uses the CR-optimal schedule built from the true T1 fleet — the best
# possible fixed-block sequence — so without noise this should be very close
# to perfect (sub-1% MAPE per sphere).
#
# Pipeline mirrors E2Env exactly:
#   build_phantom → cr_optimize → ir_se_2d_sequence × n_blocks
#   → simulate → kspace_to_image → per-sphere ROI pixel → fit_t1_generalized_ir
#
# No pose jitter, no T1 jitter, noise_sigma_abs = 0 by default.
# Noise can be set via --noise (absolute k-space sigma) or --snr (auto-computes
# sigma = ksp_rms / snr from the first block). See FIX_SIM_PLAN §2.
#
# Usage:
#   julia --project=. scripts/t1_fit_vs_true.jl
#   julia --project=. scripts/t1_fit_vs_true.jl --budget 250 --snr 20
#   julia --project=. scripts/t1_fit_vs_true.jl --budget 250 --noise 0.3
#   julia --project=. scripts/t1_fit_vs_true.jl --npe 64 --nfe 128
#   julia --project=. scripts/t1_fit_vs_true.jl --budget 160 --npe 64 --nfe 128 --clean-recon

# TODO: ADD processing to e2 steps so simulator has them
# increasing Nfe to 128 or 256 would help with leaking from other spins - worth a try
# need to re

using QalibreMDPhantom, KomaMRI, Suppressor
using DelimitedFiles, Random, Statistics, Printf, JSON
using FFTW: ifft, fftshift, ifftshift

# ── Config (match E2Env defaults) ────────────────────────────────────────────
const FOV           = 0.2      # m
const VOXEL_MM      = 3.0
const ALPHA_EXC_DEG = 90.0
const TE            = 0.020    # s

# Parse CLI args
let
    global budget_s        = 160.0
    global Npe             = 32     # --npe: phase-encode lines (affects sim time)
    global Nfe             = 64     # --nfe: frequency-encode samples (negligible cost)
    global noise_sigma_abs = 0.0   # set directly via --noise, or derived via --snr
    # Default target_snr = 2.5 → NEMA MS-1 dual-acq SNR ≈ 30 (mid-clinical
    # 1.5T/3T range 20–50). See scripts/snr_sweep/ for the calibration table
    # and scripts/snr_sweep.jl for the methodology comment. Override with
    # --snr <value>; pass --snr 0 to disable noise entirely.
    global target_snr      = 2.5
    global phase_sensitive = false   # --phase-sensitive → signed real-part recon + signed fit
    global unlimited       = false   # --unlimited → effectively-infinite budget, large n_blocks grid
    global manual          = false   # --manual → hand-crafted log-spaced schedule, bypasses cr_optimize
    global clean_recon     = false   # --clean-recon → Hamming window + 2× zero-pad + 3×3 ROI averaging
    i = 1
    while i <= length(ARGS)
        if ARGS[i] == "--budget" && i < length(ARGS)
            global budget_s = parse(Float64, ARGS[i+1]); i += 2
        elseif ARGS[i] == "--npe" && i < length(ARGS)
            global Npe = parse(Int, ARGS[i+1]); i += 2
        elseif ARGS[i] == "--nfe" && i < length(ARGS)
            global Nfe = parse(Int, ARGS[i+1]); i += 2
        elseif ARGS[i] == "--noise" && i < length(ARGS)
            global noise_sigma_abs = parse(Float64, ARGS[i+1]); i += 2
        elseif ARGS[i] == "--snr" && i < length(ARGS)
            global target_snr = parse(Float64, ARGS[i+1]); i += 2
        elseif ARGS[i] == "--phase-sensitive"
            global phase_sensitive = true; i += 1
        elseif ARGS[i] == "--unlimited"
            global unlimited = true; i += 1
        elseif ARGS[i] == "--manual"
            global manual = true; i += 1
        elseif ARGS[i] == "--clean-recon"
            global clean_recon = true; i += 1
        else
            i += 1
        end
    end
end

if unlimited
    budget_s = 1.0e6   # any block schedule the grid produces will fit
end

# Treat --snr 0 (or negative) as "disable SNR calibration; use --noise / no
# noise instead". This makes 0 the obvious off-switch without needing a
# separate flag.
if target_snr !== nothing && target_snr <= 0.0
    target_snr = nothing
end

# Build run label — SNR path gets a clean tag, --noise path uses raw sigma value
noise_tag = if target_snr !== nothing
    snr_str = isinteger(target_snr) ? string(Int(target_snr)) :
              replace(string(target_snr), "." => "p")
    "snr$(snr_str)"
elseif noise_sigma_abs > 0.0
    "noise$(noise_sigma_abs)"
else
    "nonoise"
end
budget_tag  = manual ? "bMANUAL" : (unlimited ? "bUNLIM" : "b$(round(Int, budget_s))s")
grid_tag    = (Npe == 32 && Nfe == 64) ? "" : "_npe$(Npe)fe$(Nfe)"
ps_tag      = phase_sensitive ? "_ps" : ""
clean_tag   = clean_recon ? "_clean" : ""
run_label   = "$(budget_tag)_$(noise_tag)$(grid_tag)$(ps_tag)$(clean_tag)"

println("="^60)
println("T1 fit vs true — budget=$(budget_s) s  noise=$(noise_tag)  " *
        "recon=$(phase_sensitive ? "phase-sensitive (signed)" : "magnitude")" *
        "$(clean_recon ? " + Hamming/zero-pad/ROI" : "")")
println("="^60)

# ── Build phantom (Scanner() defaults to B0 = 1.5 T, so use :T15 T1 values) ───
cfg     = PhantomConfig(field = :T15, voxel_size_mm = VOXEL_MM,
                        include_plates = [:T1])
phantom = build_phantom(cfg)
println("Phantom: $(length(phantom.x)) spins  (1.5 T calibration)")

# ── Sphere ground truth ───────────────────────────────────────────────────────
descs       = sphere_descriptors(:T1, cfg; rng = MersenneTwister(0))
n_spheres   = length(descs)
T1_true     = [d.T1 for d in descs]
centres     = [d.centre for d in descs]   # (x, y, z) tuples, metres

# Recon grid size depends on --clean-recon (2× zero-pad)
img_pad     = clean_recon ? 2 : 1
Npe_img     = Npe * img_pad
Nfe_img     = Nfe * img_pad
roi_radius  = clean_recon ? 1 : 0   # 1 → 3×3 ROI mean, 0 → single pixel

# Pixel mapping — uses image grid (Npe_img × Nfe_img)
sphere_px = NTuple{2,Int}[]
for c in centres
    cx, cy = c[1], c[2]
    ife = mod(round(Int, cx * Nfe_img / FOV) + Nfe_img ÷ 2, Nfe_img) + 1
    ipe = mod(round(Int, cy * Npe_img / FOV) + Npe_img ÷ 2, Npe_img) + 1
    push!(sphere_px, (ipe, ife))
end

# Sphere pixel mapping on the RAW (Npe × Nfe) grid — used for the SNR report
# regardless of whether --clean-recon is on, so the dual-acq SNR number is
# measured on stationary-noise pixels (Hamming windowing makes noise
# non-stationary and would bias the reported SNR low).
sphere_px_raw = NTuple{2,Int}[]
for c in centres
    cx, cy = c[1], c[2]
    ife = mod(round(Int, cx * Nfe / FOV) + Nfe ÷ 2, Nfe) + 1
    ipe = mod(round(Int, cy * Npe / FOV) + Npe ÷ 2, Npe) + 1
    push!(sphere_px_raw, (ipe, ife))
end

println("\nSphere fleet (T1 true values):")
for (i, d) in enumerate(descs)
    @printf("  sphere %2d  T1 = %.4f s  label = %s\n", i, d.T1, d.label)
end

# ── Schedule generators ───────────────────────────────────────────────────────

"""
    manual_schedule() → (TIs, TRs, n_blocks)

Hand-crafted log-spaced schedule designed to give the magnitude-mode fitter
unambiguous sign info for every sphere:
  - TIs span each sphere's null T1*ln(2):
      T1_14=0.023 → null 0.016 ;  T1_1=1.838 → null 1.274
  - Anchor TIs at 2.0 and 2.8 s put every sphere clearly past its null
    (positive-recovering side), breaking the abs() ambiguity.
  - TR = 5 s for all blocks → ~2.7×T1_max recovery between blocks.
"""
function manual_schedule()
    TIs = [0.015, 0.022, 0.032, 0.045, 0.065, 0.10, 0.15, 0.22,
           0.35, 0.55, 0.85, 1.30, 2.00, 2.80]
    TRs = fill(5.0, length(TIs))
    println("\nManual schedule: $(length(TIs)) blocks, log-spaced TIs, TR=5 s.")
    return TIs, TRs, length(TIs)
end

"""
    cr_optimal_schedule(T1_true, Npe; budget_s, unlimited) → (TIs, TRs, n_blocks)

Wraps `cr_optimize_sweep`. When `unlimited` is true uses a larger n_block_grid
and an effectively-infinite budget so the optimizer is free to pick long-TR
blocks for the long-T1 spheres.
"""
function cr_optimal_schedule(T1_true, Npe; budget_s, unlimited)
    n_block_grid = unlimited ? [14, 18, 24, 32] : [4, 6, 8, 10, 14, 18]
    println("\nOptimising CR schedule (budget=$(unlimited ? "∞" : "$(budget_s) s"), " *
            "Npe=$Npe, n_block_grid=$n_block_grid)…")
    sched = cr_optimize_sweep(
        T1_true;
        budget_s     = budget_s,
        Npe          = Npe,
        n_block_grid = n_block_grid,
        n_starts     = 1000,
        n_refine     = 10,
    )
    println("  CR obj = $(round(sched.schedule.L, sigdigits=4))")
    return sched.schedule.TIs, sched.schedule.TRs, sched.n_blocks
end

# ── Pick schedule ─────────────────────────────────────────────────────────────
TIs_opt, TRs_opt, n_blocks = manual ?
    manual_schedule() :
    cr_optimal_schedule(T1_true, Npe; budget_s = budget_s, unlimited = unlimited)

scan_time = sum(TRs_opt) * Npe
println("  n_blocks = $n_blocks  scan_time = $(round(scan_time, digits=1)) s")
println("  TIs: $(round.(TIs_opt, digits=3))")
println("  TRs: $(round.(TRs_opt, digits=3))")

# ── Simulate each block and accumulate per-sphere signals ─────────────────────
println("\nSimulating $n_blocks blocks…")

block_TIs   = [Float64[] for _ in 1:n_spheres]
block_TRs   = [Float64[] for _ in 1:n_spheres]
block_α_excs = [Float64[] for _ in 1:n_spheres]
block_mags  = [Float64[] for _ in 1:n_spheres]

rng = MersenneTwister(42)
α_exc = deg2rad(ALPHA_EXC_DEG)
sin_α = abs(sin(α_exc))   # = 1.0 for 90°

# Background mask on the RAW (Npe × Nfe) grid — see comment on
# `sphere_px_raw` above. SNR is always measured on the non-windowed recon.
bg_mask_raw = background_mask(phantom, Npe, Nfe, FOV; erosion_px = 1)
snr_rep::Union{Nothing,SNRReport} = nothing

for blk in 1:n_blocks
    TI = TIs_opt[blk]
    TR = TRs_opt[blk]

    seq = Suppressor.@suppress ir_se_2d_sequence(
        TI, TE, TR;
        α_exc = α_exc,
        FOV   = FOV,
        Nfe   = Nfe,
        Npe   = Npe,
    )
    raw = Suppressor.@suppress simulate(phantom, seq, scanner_for_field(phantom))
    ksp = raw_to_kspace(raw, Npe, Nfe)

    # TODO: don't use the first block should be done with a fixed sequence I thought we fixed this?
    # If --snr was given, derive noise_sigma_abs from the first block's k-space RMS.
    if blk == 1 && target_snr !== nothing
        ksp_rms = sqrt(sum(abs2, ksp) / length(ksp))
        global noise_sigma_abs = ksp_rms / target_snr
        @printf("  ksp_rms = %.4f  →  noise_sigma_abs = %.4f  (SNR=%g)\n",
                ksp_rms, noise_sigma_abs, target_snr)
    end

    # Absolute complex Gaussian noise on k-space (FIX_SIM_PLAN §2)
    add_noise!(ksp, noise_sigma_abs; rng = rng)

    img = clean_recon ?
        kspace_to_image(ksp; pad_factor = img_pad, hamming = true,
                              phase_sensitive = phase_sensitive) :
        kspace_to_image(ksp; phase_sensitive = phase_sensitive)

    # NEMA single-image + MS-1 dual-acquisition SNR on the first block — the
    # reference block all noise calibration is anchored on. Costs 1 extra
    # simulate() call total. SNR is measured on the RAW (non-windowed,
    # non-zero-padded) recon so the dual-acq noise estimate is on stationary
    # Gaussian noise pixels. Under --clean-recon, the fit still uses the
    # windowed image — only the SNR diagnostic is on the raw recon.
    if blk == 1 && !phase_sensitive
        img_raw = kspace_to_image(ksp; phase_sensitive = false)
        global snr_rep = snr_report(phantom, seq, Scanner();
            σ = noise_sigma_abs,
            sphere_px = sphere_px_raw,
            bg_mask = bg_mask_raw,
            ksp_a = ksp,
            img_a = img_raw,
            rng = rng,
            phase_sensitive = false,
            roi_radius = 0,
        )
        println()
        print_snr_report(snr_rep;
            label = clean_recon ?
                "SNR report (block 1, raw recon — fit uses windowed recon)" :
                "SNR report (block 1 reference)")
    end

    for i in 1:n_spheres
        ipe, ife = sphere_px[i]
        # ROI mean over (2·roi_radius+1)² pixels around the sphere centre.
        # roi_radius=0 → single-pixel sample (legacy); =1 → 3×3 mean.
        mag_i = roi_mean(img, ipe, ife; r = roi_radius) / sin_α
        push!(block_TIs[i],    TI)
        push!(block_TRs[i],    TR)
        push!(block_α_excs[i], α_exc)
        push!(block_mags[i],   mag_i)
    end
    @printf("  block %2d/%d  TI=%.3f s  TR=%.3f s  done\n", blk, n_blocks, TI, TR)
end

# ── Fit T1 per sphere ─────────────────────────────────────────────────────────
println("\nFitting T1 per sphere…")

T1_fit   = zeros(n_spheres)
T1_sigma = fill(NaN, n_spheres)
M0_fit   = zeros(n_spheres)
mapes    = zeros(n_spheres)

α_inv_vec = fill(π, n_blocks)   # standard 180° inversion pulse

for i in 1:n_spheres
    abs_noise = noise_sigma_abs > 0 ? noise_sigma_abs : nothing

    fit = fit_t1_generalized_ir(
        block_TIs[i], α_inv_vec, block_mags[i];
        TRs           = block_TRs[i],
        α_excs        = block_α_excs[i],
        Npe           = Npe,
        T1_range      = (0.01, 3.0),
        n_grid        = 500,
        abs_noise_sigma = abs_noise,
        sigma_method  = :profile_likelihood,
        signed        = phase_sensitive,
    )
    T1_fit[i]   = fit.T1
    T1_sigma[i] = fit.T1_sigma
    M0_fit[i]   = fit.A
    mapes[i]    = abs(fit.T1 - T1_true[i]) / T1_true[i] * 100
end

# ── Results table ─────────────────────────────────────────────────────────────
println()
println("  sphere   T1_true [s]   T1_fit [s]   T1_σ [s]   MAPE [%]")
println("  " * "─"^58)
for i in 1:n_spheres
    @printf("  %6s   %10.4f   %10.4f   %8.4f   %7.2f\n",
            descs[i].label, T1_true[i], T1_fit[i],
            isnan(T1_sigma[i]) ? 0.0 : T1_sigma[i],
            mapes[i])
end
println("  " * "─"^58)
@printf("  %6s   %10s   %10s   %8s   %7.2f\n",
        "MEAN", "", "", "", mean(mapes))
@printf("  %6s   %10s   %10s   %8s   %7.2f\n",
        "MAX", "", "", "", maximum(mapes))

outdir = joinpath(@__DIR__, "runs", "t1_fit_vs_true", run_label)
mkpath(outdir)

open(joinpath(outdir, "config.json"), "w") do io
    cfg_dict = Dict{String,Any}(
        "budget_s"        => budget_s,
        "Npe"             => Npe,
        "Nfe"             => Nfe,
        "noise_sigma_abs" => noise_sigma_abs,
        "target_snr"      => target_snr,
        "phase_sensitive" => phase_sensitive,
        "clean_recon"     => clean_recon,
        "unlimited"       => unlimited,
        "manual"          => manual,
        "n_blocks"        => n_blocks,
        "scan_time_s"     => round(scan_time, digits=1),
        "TIs_s"           => round.(TIs_opt, digits=4),
        "TRs_s"           => round.(TRs_opt, digits=4),
    )
    if snr_rep !== nothing
        cfg_dict["snr_report"] = snr_report_to_dict(snr_rep)
    end
    JSON.print(io, cfg_dict, 2)
    println(io)
end

csv_path = joinpath(outdir, "t1_fit_vs_true.csv")
open(csv_path, "w") do io
    println(io, "label,T1_true_s,T1_fit_s,T1_sigma_s,M0_fit,mape_pct,cx_m,cy_m")
    for i in 1:n_spheres
        cx, cy = centres[i][1], centres[i][2]
        σ = isnan(T1_sigma[i]) ? 0.0 : T1_sigma[i]
        println(io, "$(descs[i].label),$(T1_true[i]),$(T1_fit[i]),$σ,$(M0_fit[i]),$(mapes[i]),$cx,$cy")
    end
end
println("\nWrote $csv_path")

# Per-block per-sphere observed magnitudes (what the fitter actually saw),
# for downstream plotting via scripts/plot_recovery_curves.py --source koma.
signals_path = joinpath(outdir, "block_signals.csv")
open(signals_path, "w") do io
    println(io, "label,block,TI_s,TR_s,mag")
    for i in 1:n_spheres
        for k in 1:length(block_TIs[i])
            println(io, "$(descs[i].label),$k,$(block_TIs[i][k]),$(block_TRs[i][k]),$(block_mags[i][k])")
        end
    end
end
println("Wrote $signals_path")

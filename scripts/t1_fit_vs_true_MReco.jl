# Compare T1 fitted (from 2D image pipeline) vs T1 true (phantom ground truth).
# MRIReco port of scripts/t1_fit_vs_true.jl — replaces in-house FFTW with MRIReco.
#
# Uses the CR-optimal schedule built from the true T1 fleet — the best
# possible fixed-block sequence — so without noise this should be very close
# to perfect (sub-1% MAPE per sphere).
#
# Pipeline mirrors E2Env exactly:
#   build_phantom → cr_optimize → ir_se_2d_sequence × n_blocks
#   → simulate → AcquisitionData → reconstruction → per-sphere ROI pixel
#   → fit_t1_generalized_ir
#
# No pose jitter, no T1 jitter, noise_sigma_abs = 0 by default.
# Noise can be set via --noise (absolute k-space sigma) or --snr (auto-computes
# sigma = ksp_rms / snr from the first block). See FIX_SIM_PLAN §2.
#
# Usage:
#   julia --project=. scripts/t1_fit_vs_true_MReco.jl
#   julia --project=. scripts/t1_fit_vs_true_MReco.jl --budget 250 --snr 20
#   julia --project=. scripts/t1_fit_vs_true_MReco.jl --budget 250 --noise 0.3
#
# ─────────────────────────────────────────────────────────────────────────────
# STATUS / FINDINGS (compared to the in-house FFTW version)
# ─────────────────────────────────────────────────────────────────────────────
# Github: https://github.com/MagneticResonanceImaging/MRIReco.jl
# Paper: https://pubmed.ncbi.nlm.nih.gov/33817833/
#
# Performance: ~60% slower per run (88 s vs 55 s for 4-block b160 case).
#   MRIReco's reconstruction() builds encoding operators and handles trajectory
#   metadata on every call; in-house is a single ifft() call. Not worth it for
#   a simple Cartesian IFFT.
#
# Correctness: basic recon (no --clean-recon) matches in-house to exact float
#   parity. Hamming-windowed recon (--clean-recon) is comparable (~7.9% vs
#   ~9.1% mean MAPE) but OMITS zero-padding (see limitation below).
#
# LIMITATION — zero-padding not supported in this port:
#   MRIReco's direct-FFT path does not support reconSize > encodedSize for
#   Cartesian data without first calling changeEncodingSize2D!(acq, newSize).
#   That function rescales the trajectory nodes by encodedSize/newSize, placing
#   the existing k-space samples at the centre of the larger grid (zero-pad).
#   Not yet wired up here; the Hamming window alone gives the main sidelobe
#   benefit (-13 dB → -43 dB) without zero-padding.
#   Fix when needed:
#     changeEncodingSize2D!(acq, (Nfe_img, Npe_img))  # before reconstruction()
#     params[:reconSize] = (Nfe_img, Npe_img)
#
# GOTCHA — must set seq.DEF["Nx"]/["Ny"] before simulate():
#   KomaMRI's signal_to_raw_data reads Nx/Ny from seq.DEF (default 1 if unset).
#   Without these, encodedSize=[1,1,1] → subsampleIndices computes negative
#   k-space line indices → BoundsError in samplingDensity. Fix applied here.
#
# GOTCHA — must set t.cartesian = true after AcquisitionData(raw):
#   KomaMRI hardcodes trajectory="other" in the raw header, so AcquisitionData
#   always creates a Custom/cartesian=false trajectory. Without flipping this,
#   MRIReco routes through the NFFT density-compensation pipeline, which
#   requires normalised float nodes on a Chebyshev grid (not satisfied by
#   KomaMRI's Cartesian trajectory nodes).
#
# GOTCHA — image is FE×PE not PE×FE:
#   reconstruction() returns shape (Nfe, Npe, ...) — FE-first, matching
#   encodedSize=[Nfe,Npe]. In-house kspace_to_image returns PE×FE (ksp is
#   assembled with phase-encode as rows). Permute with permutedims(img2d,(2,1)).
#
# FUTURE USECASES where MRIReco would be worth the overhead:
#   1. Parallel imaging (SENSE/GRAPPA): reconstruction(:multiCoil) with
#      coil sensitivity maps (senseMaps param) — no in-house equivalent.
#   2. Iterative / compressed-sensing recon: reco=:standard/:multiEcho with
#      L1/TV regularisation; relevant if moving to undersampled k-space (E4+).
#   3. Non-Cartesian trajectories (radial, spiral): NFFT path handles arbitrary
#      nodes — needed if the RL agent learns to design non-Cartesian readouts.
#   4. Off-resonance correction (B0 maps): correctionMap param in
#      reconstruction_direct; useful for sim-to-real with field inhomogeneity.
#   5. 3D volumetric recon: same API, just set ndims=3 in signal_to_raw_data
#      and reconSize=(Nx,Ny,Nz).
# ─────────────────────────────────────────────────────────────────────────────
# For report: tried this then used in-house for speed for simulations

using QalibreMDPhantom, KomaMRI, Suppressor
using DelimitedFiles, Random, Statistics, Printf
using MRIReco

# ── Config (match E2Env defaults) ────────────────────────────────────────────
const FOV           = 0.2      # m
const Npe           = 32
const Nfe           = 64
const VOXEL_MM      = 3.0
const ALPHA_EXC_DEG = 90.0
const TE            = 0.020    # s
# Npe is configurable via --npe (default 32). At Npe=32 with TR=5s, simulation
# time per block hits the KomaMRI multi-shot drift wall (>60s). Drop Npe for
# long-TR schedules — but watch out for cross-sphere contamination at Npe<16.

# Parse CLI args
let
    global budget_s        = 160.0
    global noise_sigma_abs = 0.0   # set directly via --noise, or derived via --snr
    global target_snr      = nothing  # nothing = not used
    global phase_sensitive = false   # --phase-sensitive → signed real-part recon + signed fit
    global unlimited       = false   # --unlimited → effectively-infinite budget, large n_blocks grid
    global manual          = false   # --manual → hand-crafted log-spaced schedule, bypasses cr_optimize
    global clean_recon     = false   # --clean-recon → Hamming window + 2× zero-pad + 3×3 ROI averaging
    i = 1
    while i <= length(ARGS)
        if ARGS[i] == "--budget" && i < length(ARGS)
            global budget_s = parse(Float64, ARGS[i+1]); i += 2
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

# Build run label — SNR path gets a clean tag, --noise path uses raw sigma value
noise_tag = if target_snr !== nothing
    "snr$(round(Int, target_snr))"
elseif noise_sigma_abs > 0.0
    "noise$(noise_sigma_abs)"
else
    "nonoise"
end
budget_tag  = manual ? "bMANUAL" : (unlimited ? "bUNLIM" : "b$(round(Int, budget_s))s")
ps_tag      = phase_sensitive ? "_ps" : ""
clean_tag   = clean_recon ? "_clean" : ""
run_label   = "$(budget_tag)_$(noise_tag)$(ps_tag)$(clean_tag)"

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

# MRIReco direct-FFT path always reconstructs at the encoded resolution (Npe×Nfe).
# Zero-padding is not used (see comment near reconstruction() call below).
# roi_radius=1 (3×3 mean) still helps with sub-pixel sphere centering.
Npe_img     = Npe
Nfe_img     = Nfe
roi_radius  = clean_recon ? 1 : 0   # 1 → 3×3 ROI mean, 0 → single pixel

# Pixel mapping — uses image grid (Npe × Nfe)
sphere_px = NTuple{2,Int}[]
for c in centres
    cx, cy = c[1], c[2]
    ife = mod(round(Int, cx * Nfe_img / FOV) + Nfe_img ÷ 2, Nfe_img) + 1
    ipe = mod(round(Int, cy * Npe_img / FOV) + Npe_img ÷ 2, Npe_img) + 1
    push!(sphere_px, (ipe, ife))
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

# ── Recon helper: Hamming window applied in-place on AcquisitionData ─────────
"""
    apply_hamming!(acq, Npe, Nfe)

Multiply each contrast's k-space samples by a separable Hamming window
(0.54 − 0.46·cos), suppressing the rect-truncation sidelobes from −13 dB to
about −43 dB. The samples in `acq.kdata[c]` are stored as a `(numSamples,
numCoils)` matrix flattened in row-major (PE-outer, FE-inner) order, so the
2D window is flattened to match.
"""
function apply_hamming!(acq::AcquisitionData, Npe::Int, Nfe::Int)
    w_pe = Float32.(0.54 .- 0.46 .* cos.(2π .* (0:Npe-1) ./ (Npe-1)))
    w_fe = Float32.(0.54 .- 0.46 .* cos.(2π .* (0:Nfe-1) ./ (Nfe-1)))
    W2D  = w_pe .* transpose(w_fe)            # (Npe, Nfe)
    W    = ComplexF32.(vec(W2D))              # flattened, length Npe*Nfe
    for c in eachindex(acq.kdata)
        acq.kdata[c] .*= W                     # broadcast across coil dim
    end
    return acq
end

# ── Simulate each block and accumulate per-sphere signals ─────────────────────
println("\nSimulating $n_blocks blocks…")

block_TIs   = [Float64[] for _ in 1:n_spheres]
block_TRs   = [Float64[] for _ in 1:n_spheres]
block_α_excs = [Float64[] for _ in 1:n_spheres]
block_mags  = [Float64[] for _ in 1:n_spheres]

rng = MersenneTwister(42)
α_exc = deg2rad(ALPHA_EXC_DEG)
sin_α = abs(sin(α_exc))   # = 1.0 for 90°

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
    # KomaMRI's signal_to_raw_data reads Nx/Ny from seq.DEF (default=1).
    # Without these, encodedSize=[1,1,1] and subsampleIndices gets negative
    # line indices, causing a BoundsError in samplingDensity.
    seq.DEF["Nx"] = Nfe
    seq.DEF["Ny"] = Npe
    raw = Suppressor.@suppress simulate(phantom, seq, Scanner())
    acq = AcquisitionData(raw)
    # KomaMRI tags the trajectory as Custom/cartesian=false. Override to use
    # MRIReco's direct-FFT path instead of the NFFT pipeline.
    for t in acq.traj
        t.cartesian = true
    end

    # If --snr was given, derive noise_sigma_abs from the first block's k-space RMS.
    if blk == 1 && target_snr !== nothing
        ksp_flat = vcat([vec(acq.kdata[c]) for c in eachindex(acq.kdata)]...)
        ksp_rms  = sqrt(sum(abs2, ksp_flat) / length(ksp_flat))
        global noise_sigma_abs = ksp_rms / target_snr
        @printf("  ksp_rms = %.4f  →  noise_sigma_abs = %.4f  (SNR=%.0f)\n",
                ksp_rms, noise_sigma_abs, target_snr)
    end

    # Absolute complex Gaussian noise on k-space (FIX_SIM_PLAN §2)
    if noise_sigma_abs > 0
        for c in eachindex(acq.kdata)
            acq.kdata[c] .+= Float32(noise_sigma_abs) .*
                              randn(rng, ComplexF32, size(acq.kdata[c]))
        end
    end

    clean_recon && apply_hamming!(acq, Npe, Nfe)

    # reconSize in (FE, PE) order — must match KomaMRI's encodedSize=[Nfe,Npe].
    # Note: MRIReco's direct-FFT path doesn't support reconSize > encodedSize for
    # Cartesian data without changeEncodingSize2D. Zero-padding is omitted here;
    # the Hamming window alone already suppresses the −13 dB rect-truncation lobes.
    params   = Dict{Symbol,Any}(:reco => "direct", :reconSize => (Nfe, Npe))
    img_cplx = reconstruction(acq, params)
    # img_cplx shape: Nfe × Npe (FE-first from MRIReco).
    # Permute to Npe × Nfe so img[ipe, ife] indexing below is correct.
    img2d_raw = img_cplx[:, :, 1, 1, 1, 1]
    img2d     = permutedims(img2d_raw, (2, 1))
    img       = phase_sensitive ? Float32.(real.(img2d)) : Float32.(abs.(img2d))

    for i in 1:n_spheres
        ipe, ife = sphere_px[i]
        # ROI mean over (2·roi_radius+1)² pixels around the sphere centre.
        # roi_radius=0 → single-pixel sample (legacy); =1 → 3×3 mean.
        if roi_radius == 0
            mag_i = Float64(img[ipe, ife]) / sin_α
        else
            pe_lo = clamp(ipe - roi_radius, 1, size(img, 1))
            pe_hi = clamp(ipe + roi_radius, 1, size(img, 1))
            fe_lo = clamp(ife - roi_radius, 1, size(img, 2))
            fe_hi = clamp(ife + roi_radius, 1, size(img, 2))
            mag_i = Float64(mean(img[pe_lo:pe_hi, fe_lo:fe_hi])) / sin_α
        end
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
csv_path = joinpath(outdir, "t1_fit_vs_true.csv")
open(csv_path, "w") do io
    println(io, "label,T1_true_s,T1_fit_s,T1_sigma_s,mape_pct,cx_m,cy_m")
    for i in 1:n_spheres
        cx, cy = centres[i][1], centres[i][2]
        σ = isnan(T1_sigma[i]) ? 0.0 : T1_sigma[i]
        println(io, "$(descs[i].label),$(T1_true[i]),$(T1_fit[i]),$σ,$(mapes[i]),$cx,$cy")
    end
end
println("\nWrote $csv_path")

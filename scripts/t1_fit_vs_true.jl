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
# Noise is swept via --sigmas: the noise-free k-space for each block is
# simulated ONCE (deterministic `simulate()`), then for each σ in the list we
# add fresh complex Gaussian noise of magnitude σ, re-fit, and write a separate
# run-dir. This avoids re-simulating per noise level (the schedule is identical).
#
# After each σ's run-dir is written the two report figures are auto-rendered
# (scripts/t1_fit_vs_true.py + scripts/plot_recovery_curves_koma.py), fail-soft.
# The dense recovery curves are pre-computed here and dumped to
# recovery_curves.npz so the plotter doesn't need to boot a second Julia.
#
# Usage:
#   julia --project=. scripts/t1_fit_vs_true.jl
#   julia --project=. scripts/t1_fit_vs_true.jl --budget 160 --sigmas 0,600
#   julia --project=. scripts/t1_fit_vs_true.jl --water --sigmas 0,100
#   julia --project=. scripts/t1_fit_vs_true.jl --manual --sigmas 0
#   julia --project=. scripts/t1_fit_vs_true.jl --budget 160 --npe 64 --nfe 128 --clean-recon

using QalibreMDPhantom, KomaMRI, Suppressor
using DelimitedFiles, Random, Statistics, Printf, JSON, NPZ
using FFTW: ifft, fftshift, ifftshift

# Shared fit pipeline (accumulate_block_mags, fit_fleet, print_fit_table,
# write_rundir, try_render, render_t1_figures).
include(joinpath(@__DIR__, "t1_fit_lib.jl"))

# ── Config (match E2Env defaults) ────────────────────────────────────────────
const FOV           = 0.2      # m
const VOXEL_MM      = 1.0
const ALPHA_EXC_DEG = 90.0
const TE            = 0.020    # s

# Parse CLI args
let
    global budget_s        = 160.0
    global Npe             = 32     # --npe: phase-encode lines (affects sim time)
    global Nfe             = 64     # --nfe: frequency-encode samples (negligible cost)
    global sigmas          = [0.0]  # --sigmas: comma-list of absolute k-space noise σ
    global phase_sensitive = false   # --phase-sensitive → signed real-part recon + signed fit
    global unlimited       = false   # --unlimited → effectively-infinite budget, large n_blocks grid
    global manual          = false   # --manual → hand-crafted log-spaced schedule, bypasses cr_optimize
    global clean_recon     = false   # --clean-recon → Hamming window + 2× zero-pad + 3×3 ROI averaging
    global spoil           = false   # --spoil → Gz crusher pair around 180° refocus + TR spoiler
    global water           = false   # --water → include water background plate in phantom
    global slice_mm        = 0.0     # --slice-mm: slab thickness; 0 → auto (VOXEL_MM)
    global slice_center_mm = NaN     # --slice-center-mm: slab centre; NaN → auto (PLATE_Z_MM.T1)
    global no_render       = false   # --no-render → skip the auto Python figure calls
    global refresh_cache   = false   # --refresh-cache → ignore + overwrite the k-space cache
    i = 1
    while i <= length(ARGS)
        if ARGS[i] == "--budget" && i < length(ARGS)
            global budget_s = parse(Float64, ARGS[i+1]); i += 2
        elseif ARGS[i] == "--npe" && i < length(ARGS)
            global Npe = parse(Int, ARGS[i+1]); i += 2
        elseif ARGS[i] == "--nfe" && i < length(ARGS)
            global Nfe = parse(Int, ARGS[i+1]); i += 2
        elseif ARGS[i] == "--sigmas" && i < length(ARGS)
            global sigmas = [parse(Float64, s) for s in split(ARGS[i+1], ",")]; i += 2
        elseif ARGS[i] == "--phase-sensitive"
            global phase_sensitive = true; i += 1
        elseif ARGS[i] == "--unlimited"
            global unlimited = true; i += 1
        elseif ARGS[i] == "--manual"
            global manual = true; i += 1
        elseif ARGS[i] == "--clean-recon"
            global clean_recon = true; i += 1
        elseif ARGS[i] == "--spoil"
            global spoil = true; i += 1
        elseif ARGS[i] == "--water"
            global water = true; i += 1
        elseif ARGS[i] == "--slice-mm" && i < length(ARGS)
            global slice_mm = parse(Float64, ARGS[i+1]); i += 2
        elseif ARGS[i] == "--slice-center-mm" && i < length(ARGS)
            global slice_center_mm = parse(Float64, ARGS[i+1]); i += 2
        elseif ARGS[i] == "--no-render"
            global no_render = true; i += 1
        elseif ARGS[i] == "--refresh-cache"
            global refresh_cache = true; i += 1
        else
            i += 1
        end
    end
end

if unlimited
    budget_s = 1.0e6   # any block schedule the grid produces will fit
end

# Filesystem-friendly σ label: integers stay clean (600 → "600"), non-integers
# replace the dot (0.3 → "0p3"). Python's run_t1_fit_sweep mirrors this rule.
σlabel(σ::Real) = isinteger(σ) ? string(Int(σ)) : replace(@sprintf("%g", σ), "." => "p")
noise_tag(σ::Real) = σ > 0 ? "noise$(σlabel(σ))" : "nonoise"

# Run-label tags that don't depend on σ (the noise tag is appended per σ).
budget_tag  = manual ? "bMANUAL" : (unlimited ? "bUNLIM" : "b$(round(Int, budget_s))s")
grid_tag    = (Npe == 32 && Nfe == 64) ? "" : "_npe$(Npe)fe$(Nfe)"
ps_tag      = phase_sensitive ? "_ps" : ""
clean_tag   = clean_recon ? "_clean" : ""
spoil_tag   = spoil ? "_spoil" : ""
water_tag   = water ? "_water" : ""

println("="^60)
println("T1 fit vs true — budget=$(budget_s) s  sigmas=$(sigmas)  " *
        "recon=$(phase_sensitive ? "phase-sensitive (signed)" : "magnitude")" *
        "$(clean_recon ? " + Hamming/zero-pad/ROI" : "")")
println("="^60)

# ── Build phantom (Scanner() defaults to B0 = 1.5 T, so use :T15 T1 values) ───
# Resolve slice geometry. Centre on the T1 plate z; slab thickness defaults to
# one voxel so spins and recon agree on a single in-plane slice. Both values
# are forwarded to PhantomConfig so the builder applies the z-mask — we don't
# post-filter phantom.z ourselves.
slice_center = isnan(slice_center_mm) ? QalibreMDPhantom.PLATE_Z_MM.T1 : slice_center_mm
slice_thick  = slice_mm > 0 ? slice_mm : VOXEL_MM
cfg     = PhantomConfig(field = :T15, voxel_size_mm = VOXEL_MM,
                        include_plates     = water ? [:T1, :water] : [:T1],
                        slice_thickness_mm = slice_thick,
                        slice_center_mm    = slice_center)
phantom = build_phantom(cfg)
println("Phantom: $(length(phantom.x)) spins  (1.5 T calibration, " *
        "slab z=$(slice_center) ± $(slice_thick/2) mm)")

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
    push!(sphere_px, (phys_to_pixel(cy, Npe_img, FOV), phys_to_pixel(cx, Nfe_img, FOV)))
end

# Sphere pixel mapping on the RAW (Npe × Nfe) grid — used for the SNR report
# regardless of whether --clean-recon is on, so the dual-acq SNR number is
# measured on stationary-noise pixels (Hamming windowing makes noise
# non-stationary and would bias the reported SNR low).
sphere_px_raw = NTuple{2,Int}[]
for c in centres
    cx, cy = c[1], c[2]
    push!(sphere_px_raw, (phys_to_pixel(cy, Npe, FOV), phys_to_pixel(cx, Nfe, FOV)))
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
    # 14/18 are feasible at large budgets but never beat the CR-optimal floor,
    # so they're dropped from the default sweep (pure optimize-time saving). The
    # --unlimited path keeps the large counts — that mode wants them.
    n_block_grid = unlimited ? [14, 18, 24, 32] : [4, 6, 8, 10]
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
# A small budget at large Npe can be physically infeasible (even the smallest
# block schedule at the TR floor overruns the budget — see cr_optimal.jl:274).
# That's a legitimate "this cell can't exist" rather than an error, so exit
# cleanly (code 0) with a note: the sweep orchestrator then keeps going and the
# comparison plot simply omits the missing cell. Genuine errors still rethrow.
function pick_schedule()
    try
        return manual ?
            manual_schedule() :
            cr_optimal_schedule(T1_true, Npe; budget_s = budget_s, unlimited = unlimited)
    catch e
        if occursin("no valid n_blocks fits budget", sprint(showerror, e))
            @warn "Skipping run: budget=$(budget_s)s infeasible at Npe=$(Npe) " *
                  "(smallest schedule overruns budget). No output written."
            exit(0)
        end
        rethrow()
    end
end
α_exc = deg2rad(ALPHA_EXC_DEG)
sin_α = abs(sin(α_exc))   # = 1.0 for 90°

# ── Noise-free k-space: load from disk cache, or simulate then cache ──────────
# The cached k-space depends ONLY on the simulate inputs + schedule — not on σ,
# --phase-sensitive or --clean-recon (applied later at recon/fit). So a run that
# changes only the noise list reuses the cache and skips simulate entirely. The
# schedule (TIs/TRs) is cached too, so a hit also skips the CR optimize and is
# immune to its random-start non-determinism.
p_(x) = replace(string(x), "." => "p")
cache_key  = "$(budget_tag)_npe$(Npe)fe$(Nfe)_$(water ? "water" : "dry")_" *
             "$(spoil ? "spoil" : "nospoil")_vox$(p_(VOXEL_MM))_" *
             "sl$(p_(slice_thick))c$(p_(slice_center))"
cache_dir  = joinpath(@__DIR__, "runs", "t1_fit_vs_true", "_kspace_cache")
cache_path = joinpath(cache_dir, cache_key * ".npz")

if isfile(cache_path) && !refresh_cache
    println("\nLoaded cached k-space: $(cache_path)")
    d         = npzread(cache_path)
    TIs_opt   = Float64.(d["TIs"])
    TRs_opt   = Float64.(d["TRs"])
    ksp_arr   = d["ksp"]                 # (n_blocks, Npe, Nfe) ComplexF32
    n_blocks  = size(ksp_arr, 1)
    ksp_clean = [ComplexF32.(ksp_arr[b, :, :]) for b in 1:n_blocks]
else
    TIs_opt, TRs_opt, n_blocks = pick_schedule()
    println("\nNoise-free pre-pass ($n_blocks simulate calls)…")
    ksp_clean = Vector{Matrix{ComplexF32}}(undef, n_blocks)
    for blk in 1:n_blocks
        seq = Suppressor.@suppress ir_se_2d_sequence(
            TIs_opt[blk], TE, TRs_opt[blk];
            α_exc = α_exc, FOV = FOV, Nfe = Nfe, Npe = Npe,
            spoiler = SpoilerConfig(enabled = spoil),
        )
        raw = Suppressor.@suppress simulate(phantom, seq, scanner_for_field(cfg))
        ksp_clean[blk] = raw_to_kspace(raw, Npe, Nfe)
        @printf("  block %2d/%d  TI=%.3f s  TR=%.3f s  done\n",
                blk, n_blocks, TIs_opt[blk], TRs_opt[blk])
    end
    mkpath(cache_dir)
    ksp_arr = Array{ComplexF32,3}(undef, n_blocks, Npe, Nfe)
    for b in 1:n_blocks
        ksp_arr[b, :, :] = ksp_clean[b]
    end
    npzwrite(cache_path, Dict("TIs" => TIs_opt, "TRs" => TRs_opt, "ksp" => ksp_arr))
    println("Cached k-space → $(cache_path)")
end

scan_time = sum(TRs_opt) * Npe
println("  n_blocks = $n_blocks  scan_time = $(round(scan_time, digits=1)) s")
println("  TIs: $(round.(TIs_opt, digits=3))")
println("  TRs: $(round.(TRs_opt, digits=3))")

# Per-sphere block metadata (identical for every σ — built once).
block_TIs    = [copy(TIs_opt) for _ in 1:n_spheres]
block_TRs    = [copy(TRs_opt) for _ in 1:n_spheres]
block_α_excs = [fill(α_exc, n_blocks) for _ in 1:n_spheres]
α_inv_vec    = fill(π, n_blocks)   # standard 180° inversion pulse

# Background mask on the RAW (Npe × Nfe) grid — see comment on `sphere_px_raw`.
bg_mask_raw = background_mask(phantom, Npe, Nfe, FOV; erosion_px = 1)

# Dense recovery-curve abscissa (shared across σ; curves themselves depend on
# the per-σ fit). Mirrors plot_recovery_curves_koma.py's geomspace + median TR.
TR_eff   = median(TRs_opt)
TI_dense = 10 .^ range(log10(1e-3), log10(max(maximum(TRs_opt), 3.0)); length = 400)

# ── Per-σ loop ────────────────────────────────────────────────────────────────
for σ in sigmas
    println("\n", "─"^60)
    println("σ = $σ  ($(noise_tag(σ)))")
    println("─"^60)

    rng = MersenneTwister(round(Int, 1e6 * σ) + 42)

    # NEMA single-image + MS-1 dual-acquisition SNR on block 1, built straight
    # from the cached clean k-space (no simulate, no seq). Uses its own RNG so
    # the diagnostic doesn't perturb the fit's noise stream below.
    snr_rep::Union{Nothing,SNRReport} = nothing
    if !phase_sensitive
        snr_rep = snr_report_from_clean(ksp_clean[1], σ;
            sphere_px = sphere_px_raw, bg_mask = bg_mask_raw,
            rng = MersenneTwister(round(Int, 1e6 * σ) + 7),
            phase_sensitive = false, roi_radius = 0)
        println()
        print_snr_report(snr_rep;
            label = clean_recon ?
                "SNR report (block 1, raw recon — fit uses windowed recon)" :
                "SNR report (block 1 reference)")
    end

    block_mags = accumulate_block_mags(ksp_clean, sphere_px, sin_α;
        σ = σ, rng = rng, clean_recon = clean_recon, img_pad = img_pad,
        roi_radius = roi_radius, phase_sensitive = phase_sensitive)

    # ── Fit T1 per sphere ────────────────────────────────────────────────────
    println("\nFitting T1 per sphere…")
    abs_noise = σ > 0 ? σ : nothing
    fit = fit_fleet(block_TIs, α_inv_vec, block_mags, T1_true, Npe;
        block_TRs = block_TRs, block_α_excs = block_α_excs,
        abs_noise = abs_noise, phase_sensitive = phase_sensitive,
        T1_range = (0.01, 3.0), n_grid = 500, sigma_method = :profile_likelihood)

    print_fit_table(descs, T1_true, fit)

    # ── Write run-dir (config + CSV + signals + recovery curves) ──────────────
    run_label = "$(budget_tag)_$(noise_tag(σ))$(grid_tag)$(ps_tag)$(clean_tag)$(spoil_tag)$(water_tag)"
    outdir    = joinpath(@__DIR__, "runs", "t1_fit_vs_true", run_label)
    cfg_dict  = Dict{String,Any}(
        "budget_s"        => budget_s,
        "Npe"             => Npe,
        "Nfe"             => Nfe,
        "noise_sigma_abs" => σ,
        "phase_sensitive" => phase_sensitive,
        "clean_recon"     => clean_recon,
        "spoil"           => spoil,
        "water"           => water,
        "slice_thickness_mm" => slice_thick,
        "slice_center_mm"    => slice_center,
        "unlimited"       => unlimited,
        "manual"          => manual,
        "n_blocks"        => n_blocks,
        "scan_time_s"     => round(scan_time, digits=1),
        "TIs_s"           => round.(TIs_opt, digits=4),
        "TRs_s"           => round.(TRs_opt, digits=4),
        "TR_eff_s"        => TR_eff,
    )
    write_rundir(outdir, descs, T1_true, fit, centres,
                 block_TIs, block_TRs, block_mags, cfg_dict, TI_dense, TR_eff,
                 Npe, α_exc; snr_rep = snr_rep)

    # ── Auto-render figures (fail-soft) ──────────────────────────────────────
    !no_render && render_t1_figures(run_label)
end

println("\nDone.")

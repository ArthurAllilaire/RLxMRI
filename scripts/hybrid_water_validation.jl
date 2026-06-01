# Hybrid-water validation (noiseless).
#
# Water is ~90% of the phantom spin count but is one homogeneous on-resonance
# material, so its k-space contribution factorises: a single Bloch envelope ×
# the FFT of the water occupancy. This script checks whether simulating ONLY the
# spheres and ADDING an analytic water k-space term (`ir_se_theory_image`)
# reproduces the full-sim image and the downstream T1 fits — and how much faster
# the spheres-only sim is.
#
# Decomposition used (exact): build_phantom([:T1,:water]) lays spins out as
# [spheres…, water…]; build_phantom([:T1]) is the same sphere block. So the water
# spins E2 actually sees = full[length(dry)+1 : end] (carries the same pose / B0
# jitter), and by Bloch linearity ksp_full = ksp_dry + ksp_water_koma exactly.
# The ONLY approximation under test is analytic-water vs koma-water.
#
# Reuses the shared fit pipeline (scripts/t1_fit_lib.jl) and the existing Python
# figure renderers (t1_fit_vs_true.py, plot_recovery_curves_koma.py for the fits;
# pixel_grid_overlay.py for the image / k-space diff panels).
#
# Usage:
#   julia --project=. scripts/hybrid_water_validation.jl
#   julia --project=. scripts/hybrid_water_validation.jl --b0-sigma 5 --overlay-ti 0.5
#   PYTHON=.venv/bin/python PYTHON_JULIAPKG_OFFLINE=yes julia --project=. scripts/hybrid_water_validation.jl

using MRISystemPhantom, KomaMRI, Suppressor
using Random, Statistics, Printf, JSON, NPZ
include(joinpath(@__DIR__, "t1_fit_lib.jl"))

# ── Config (match E2Env / t1_fit_vs_true.jl) ──────────────────────────────────
const FOV           = 0.2      # m
const VOXEL_MM      = 1.0
const ALPHA_EXC_DEG = 90.0
const TE            = 0.020    # s

let
    global Npe        = 32
    global Nfe        = 64
    global b0_sigma   = 0.0    # --b0-sigma: per-spin off-resonance σ [Hz]; E2 uses 5.0
    global overlay_ti = 0.5    # --overlay-ti: TI [s] for the image/k-space diff panels
    global no_render  = false  # --no-render
    i = 1
    while i <= length(ARGS)
        if ARGS[i] == "--npe" && i < length(ARGS)
            global Npe = parse(Int, ARGS[i+1]); i += 2
        elseif ARGS[i] == "--nfe" && i < length(ARGS)
            global Nfe = parse(Int, ARGS[i+1]); i += 2
        elseif ARGS[i] == "--b0-sigma" && i < length(ARGS)
            global b0_sigma = parse(Float64, ARGS[i+1]); i += 2
        elseif ARGS[i] == "--overlay-ti" && i < length(ARGS)
            global overlay_ti = parse(Float64, ARGS[i+1]); i += 2
        elseif ARGS[i] == "--no-render"
            global no_render = true; i += 1
        else
            i += 1
        end
    end
end

const α_exc   = deg2rad(ALPHA_EXC_DEG)
const sin_α   = abs(sin(α_exc))
const scanner = scanner_for_field(:T15)

println("="^64)
println("hybrid-water validation   Npe=$Npe Nfe=$Nfe  B0σ=$(b0_sigma) Hz  " *
        "(noiseless)")
println("="^64)

# ── Phantoms: spheres-only (dry), full (spheres+water); derive water ──────────
slice_center = MRISystemPhantom.PLATE_Z_MM.T1
slice_thick  = VOXEL_MM
augment      = AugmentConfig(B0_sigma_Hz = b0_sigma)
cfg_common   = (field = :T15, voxel_size_mm = VOXEL_MM,
                slice_thickness_mm = slice_thick, slice_center_mm = slice_center,
                augment = augment)
cfg_full = PhantomConfig(; include_plates = [:T1, :water], cfg_common...)
cfg_dry  = PhantomConfig(; include_plates = [:T1],          cfg_common...)

full = build_phantom(cfg_full)
dry  = build_phantom(cfg_dry)
n_dry = length(dry.x)
n_full = length(full.x)

# Verify the layout invariant so the water extraction below is exact.
@assert full.x[1:n_dry] ≈ dry.x && full.y[1:n_dry] ≈ dry.y && full.z[1:n_dry] ≈ dry.z "phantom layout is not [spheres…, water…]; water extraction would be wrong"

water_mask = falses(n_full); water_mask[(n_dry+1):end] .= true
water = full[water_mask]
n_water = length(water.x)
@printf("spins:  full=%d   spheres=%d   water=%d  (water = %.1f%%)\n",
        n_full, n_dry, n_water, 100 * n_water / n_full)

# ── Sphere ground truth + pixel mapping (mirrors t1_fit_vs_true.jl) ───────────
descs     = sphere_descriptors(:T1, cfg_full; rng = MersenneTwister(0))
n_spheres = length(descs)
T1_true   = [d.T1 for d in descs]
centres   = [d.centre for d in descs]
sphere_px = [(phys_to_pixel(c[2], Npe, FOV), phys_to_pixel(c[1], Nfe, FOV)) for c in centres]

# ── Manual log-spaced schedule (matches t1_fit_vs_true.jl manual_schedule) ────
TIs = [0.015, 0.022, 0.032, 0.045, 0.065, 0.10, 0.15, 0.22,
       0.35, 0.55, 0.85, 1.30, 2.00, 2.80]
TRs = fill(5.0, length(TIs))
n_blocks = length(TIs)
println("schedule: $n_blocks blocks, log-spaced TIs, TR=5 s, α=$(ALPHA_EXC_DEG)°")

# ── Helpers ───────────────────────────────────────────────────────────────────
seq_for(TI, TR) = Suppressor.@suppress ir_se_2d_sequence(
    TI, TE, TR; α_exc = α_exc, FOV = FOV, Nfe = Nfe, Npe = Npe)

sim_ksp(phantom, TI, TR) =
    raw_to_kspace(Suppressor.@suppress(simulate(phantom, seq_for(TI, TR), scanner)),
                  Npe, Nfe)

analytic_water_ksp(TI, TR) = ir_se_theory_image(
    water; TI = TI, TR = TR, α_exc = α_exc, θ_inv = π,
    FOV = FOV, Nfe = Nfe, Npe = Npe, voxel_mm = VOXEL_MM).ksp

relerr(a, b) = sum(abs.(a .- b)) / max(sum(abs.(b)), eps())

# ── Cached Koma water template (built ONCE per geometry) ──────────────────────
# One reference Koma water sim captures the true per-line spatial structure
# (edges, spoiling, B0, per-shot transient ramp). Water is one material, so for a
# FIXED geometry its k-space scales with the IR transient magnetisation
# m_w(TI,TR,α) = transient_mz_at_excite_npe(...)·sin(α) — the SAME forward model
# the fitter uses (fits.jl:163), reused here. Cache W = ksp_water_ref / m_w_ref
# once; rescale by m_w for any (TI,TR,α) with NO re-simulation.
#
# Single mean-transient scalar: exact at α=90° (cos α = 0 flattens the per-shot
# Mz ramp, so every k-line shares one scalar). When α varies, the per-shot ramp
# returns and the rescale should weight each k-line by its own shot's transient
# (recoverable from transient_mz_at_excite_npe via finite differences of the
# cumulative mean — no new recurrence). Revisit when sweeping α.
const T1_WATER = MRISystemPhantom.BACKGROUND_WATER[:T15].T1
const TI0_REF  = 0.10
const TR0_REF  = 5.0
mw(TI, TR; α = α_exc) =
    transient_mz_at_excite_npe(T1_WATER, TI, TR, π, α; Npe = Npe) * sin(α)

ksp_water_ref = sim_ksp(water, TI0_REF, TR0_REF)          # one sim per geometry
w_geom        = ksp_water_ref ./ ComplexF32(mw(TI0_REF, TR0_REF))   # geometric template
cached_water_ksp(TI, TR; α = α_exc) = w_geom .* ComplexF32(mw(TI, TR; α = α))

# ── Per-block: full + dry sims, analytic water, hybrid ────────────────────────
ksp_full          = Vector{Matrix{ComplexF32}}(undef, n_blocks)
ksp_hybrid_analy  = Vector{Matrix{ComplexF32}}(undef, n_blocks)
ksp_hybrid_cached = Vector{Matrix{ComplexF32}}(undef, n_blocks)
relerr_analy  = zeros(n_blocks)   # analytic water vs koma (= full − dry)
relerr_cached = zeros(n_blocks)   # cached-template water vs koma

println("\nSimulating $n_blocks blocks (full + spheres-only); water from analytic + cached template…")
for blk in 1:n_blocks
    kf = sim_ksp(full, TIs[blk], TRs[blk])
    kd = sim_ksp(dry,  TIs[blk], TRs[blk])
    kw_koma   = kf .- kd                                   # exact koma water
    kw_analy  = ComplexF32.(analytic_water_ksp(TIs[blk], TRs[blk]))
    kw_cached = cached_water_ksp(TIs[blk], TRs[blk])

    ksp_full[blk]          = kf
    ksp_hybrid_analy[blk]  = kd .+ kw_analy
    ksp_hybrid_cached[blk] = kd .+ kw_cached
    relerr_analy[blk]  = relerr(kw_analy,  kw_koma)
    relerr_cached[blk] = relerr(kw_cached, kw_koma)
    @printf("  block %2d/%d  TI=%.3f s   water relerr  analytic=%.3e  cached=%.3e\n",
            blk, n_blocks, TIs[blk], relerr_analy[blk], relerr_cached[blk])
end

# ── Fit T1 on the full / analytic-hybrid / cached-hybrid stacks ───────────────
block_TIs    = [copy(TIs) for _ in 1:n_spheres]
block_TRs    = [copy(TRs) for _ in 1:n_spheres]
block_α_excs = [fill(α_exc, n_blocks) for _ in 1:n_spheres]
α_inv_vec    = fill(π, n_blocks)

mags_full   = accumulate_block_mags(ksp_full,          sphere_px, sin_α)
mags_analy  = accumulate_block_mags(ksp_hybrid_analy,  sphere_px, sin_α)
mags_cached = accumulate_block_mags(ksp_hybrid_cached, sphere_px, sin_α)

fitf(m) = fit_fleet(block_TIs, α_inv_vec, m, T1_true, Npe;
                    block_TRs = block_TRs, block_α_excs = block_α_excs)
fit_full   = fitf(mags_full)
fit_analy  = fitf(mags_analy)
fit_cached = fitf(mags_cached)

println("\n### FULL-SIM fit");            print_fit_table(descs, T1_true, fit_full)
println("\n### HYBRID (cached) fit");     print_fit_table(descs, T1_true, fit_cached)

# ── Per-sphere hybrid-vs-full T1 agreement (the RL-relevant metric) ───────────
println("\n  sphere   T1_true   T1_full   |Δanaly|/full%   |Δcached|/full%")
println("  " * "─"^62)
rel_analy  = zeros(n_spheres)
rel_cached = zeros(n_spheres)
for i in 1:n_spheres
    rel_analy[i]  = abs(fit_analy.T1_fit[i]  - fit_full.T1_fit[i]) / fit_full.T1_fit[i] * 100
    rel_cached[i] = abs(fit_cached.T1_fit[i] - fit_full.T1_fit[i]) / fit_full.T1_fit[i] * 100
    @printf("  %6s   %7.4f   %7.4f      %9.3f         %9.3f\n",
            descs[i].label, T1_true[i], fit_full.T1_fit[i], rel_analy[i], rel_cached[i])
end
println("  " * "─"^62)
@printf("  |ΔT1|/T1 vs full   analytic: mean=%.3f%% max=%.3f%%   cached: mean=%.3f%% max=%.3f%%\n",
        mean(rel_analy), maximum(rel_analy), mean(rel_cached), maximum(rel_cached))
@printf("  water k-space relerr vs koma   analytic: mean=%.3e   cached: mean=%.3e\n",
        mean(relerr_analy), mean(relerr_cached))
@printf("  MAPE vs true   full: %.2f%%   analytic-hybrid: %.2f%%   cached-hybrid: %.2f%%\n",
        mean(fit_full.mapes), mean(fit_analy.mapes), mean(fit_cached.mapes))

# ── Write fit run-dirs + render the existing T1 figures ───────────────────────
TR_eff   = median(TRs)
TI_dense = 10 .^ range(log10(1e-3), log10(max(maximum(TRs), 3.0)); length = 400)
function fit_cfg_dict(label)
    Dict{String,Any}("source" => label, "Npe" => Npe, "Nfe" => Nfe,
        "noise_sigma_abs" => 0.0, "phase_sensitive" => false, "manual" => true,
        "n_blocks" => n_blocks, "b0_sigma_Hz" => b0_sigma,
        "TIs_s" => round.(TIs, digits=4), "TRs_s" => round.(TRs, digits=4),
        "TR_eff_s" => TR_eff)
end
for (label, fit, mags) in (("hybridval_full",   fit_full,   mags_full),
                           ("hybridval_cached", fit_cached, mags_cached))
    outdir = joinpath(@__DIR__, "runs", "t1_fit_vs_true", label)
    write_rundir(outdir, descs, T1_true, fit, centres,
                 block_TIs, block_TRs, mags, fit_cfg_dict(label),
                 TI_dense, TR_eff, Npe, α_exc)
    no_render || render_t1_figures(label)
end

# ── Image / k-space diff panels at a representative TI (reuse overlay renderer)─
# Dumps a pixel_grid_overlay-format run-dir and renders sim-vs-theory + k-space.
function dump_overlay(label, sim_ksp_, theory_ksp_, src_phantom; ti)
    outdir = joinpath(@__DIR__, "runs", "pixel_grid_overlay", label)
    mkpath(outdir)
    img_sim    = kspace_to_image(sim_ksp_)             # magnitude (default)
    img_theory = kspace_to_image(theory_ksp_)
    npzwrite(joinpath(outdir, "kspace.npy"),         Array(sim_ksp_))
    npzwrite(joinpath(outdir, "theory_kspace.npy"),  Array(theory_ksp_))
    npzwrite(joinpath(outdir, "image.npy"),          Array(img_sim))
    npzwrite(joinpath(outdir, "theory_magnitude.npy"), Array(abs.(img_theory)))
    npzwrite(joinpath(outdir, "diff_image.npy"),     Array(abs.(img_sim) .- abs.(img_theory)))
    npzwrite(joinpath(outdir, "occupancy.npy"),      Array(phantom_occupancy(src_phantom, Npe, Nfe, FOV)))
    npzwrite(joinpath(outdir, "spins_xy.npy"),       hcat(src_phantom.x, src_phantom.y))
    sphere_centres = Matrix{Float64}(undef, n_spheres, 2)
    sphere_px_m    = Matrix{Int}(undef, n_spheres, 2)
    for (i, d) in enumerate(descs)
        sphere_centres[i, :] = [d.centre[1], d.centre[2]]
        sphere_px_m[i, :]    = [phys_to_pixel(d.centre[2], Npe, FOV),
                                phys_to_pixel(d.centre[1], Nfe, FOV)]
    end
    npzwrite(joinpath(outdir, "sphere_centres.npy"), sphere_centres)
    npzwrite(joinpath(outdir, "sphere_px.npy"),      sphere_px_m)
    open(joinpath(outdir, "sphere_meta.json"), "w") do io
        JSON.print(io, [Dict("label" => String(d.label), "radius_m" => d.radius,
                             "T1_s" => d.T1, "centre_xyz_m" => collect(d.centre))
                        for d in descs], 2); println(io)
    end
    open(joinpath(outdir, "config.json"), "w") do io
        JSON.print(io, Dict{String,Any}(
            "FOV_m" => FOV, "voxel_size_mm" => VOXEL_MM, "Nfe" => Nfe, "Npe" => Npe,
            "plate" => "T1", "z_plate_m" => slice_center / 1000,
            "slice_half_m" => slice_thick / 1000 / 2,
            "n_spins_total" => length(src_phantom.x),
            "n_spins_in_slice" => length(src_phantom.x),
            "dx_fe_m" => FOV / Nfe, "dy_pe_m" => FOV / Npe,
            "image" => Dict{String,Any}("rendered" => true, "TI_s" => ti,
                "TE_s" => TE, "TR_s" => 5.0, "clean" => false, "spoil" => false,
                "img_npe" => Npe, "img_nfe" => Nfe),
        ), 2); println(io)
    end
    no_render || try_render("pixel_grid_overlay.py", ["--run", label])
    outdir
end

println("\nRendering image / k-space diff panels at TI=$(overlay_ti) s…")
tag = replace(string(overlay_ti), "." => "p")
kf_ov = sim_ksp(full, overlay_ti, 5.0)
kd_ov = sim_ksp(dry,  overlay_ti, 5.0)
# (a) water diagnostic: koma water (= full − dry) vs cached-template water
dump_overlay("hybridval_water_TI$(tag)",
             kf_ov .- kd_ov, cached_water_ksp(overlay_ti, 5.0), water; ti = overlay_ti)
# (b) full vs cached-hybrid (full = "sim" slot, hybrid = "theory" slot)
dump_overlay("hybridval_fullvshybrid_TI$(tag)",
             kf_ov, kd_ov .+ cached_water_ksp(overlay_ti, 5.0), full; ti = overlay_ti)

# ── Timing: per-step cost (spheres-only + rescale) vs full sim ─────────────────
# Per-step the agent pays a spheres-only sim + a cached-template rescale; the one
# water reference sim is amortised over the whole episode.
println("\nTiming (after warmup)…")
sim_ksp(full, 0.5, 5.0); sim_ksp(dry, 0.5, 5.0)   # warmup JIT
t_full   = @elapsed sim_ksp(full, 0.5, 5.0)
t_dry    = @elapsed sim_ksp(dry,  0.5, 5.0)
t_cached = @elapsed cached_water_ksp(0.5, 5.0)
@printf("  full=%.3f s   spheres-only=%.3f s   cached-water rescale=%.3e s\n",
        t_full, t_dry, t_cached)
@printf("  per-step speedup=%.2f×  (spin ratio %.2f×);  one %.3f s water ref sim amortised/episode\n",
        t_full / (t_dry + t_cached), n_full / n_dry, t_full)

println("\nDone.")

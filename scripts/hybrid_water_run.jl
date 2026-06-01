# Hybrid-water data harness — produces ONE run folder of arrays + metadata for a
# single configuration (field, α, B0, TI/TR schedule). Figures are generated
# separately by scripts/hybrid_water_figs.py (single run) and stitched across
# runs by scripts/hybrid_water_stitch.py. Julia writes data only — no plots.
#
# Four image-forming variants (each a full Npe×Nfe k-space + magnitude image per
# schedule block), so any can be reconstructed/fitted downstream:
#   koma_full        full Koma sim (spheres + water)           ← GROUND TRUTH
#   hybrid_cached    Koma spheres + cached Koma-water template (rescaled by m_w)
#   hybrid_analytic  Koma spheres + analytic water (ir_se_theory_image)
#   theory_full      pure analytic forward model on the full phantom
# Plus three water-only variants (water_koma = full−dry, water_cache, water_theory).
#
# Layout:
#   runs/hybrid_water/<label>/
#     config.json
#     arrays/   TIs.npy TRs.npy <variant>_ksp.npy(complex, n_blk×Npe×Nfe) <variant>_img.npy
#               sphere_centres.npy sphere_px.npy spins_xy_full.npy spins_xy_water.npy
#               relerr.csv  t1fits.csv
#     figures/  (filled by hybrid_water_figs.py)
#
# Usage:
#   julia --project=. scripts/hybrid_water_run.jl
#   julia --project=. scripts/hybrid_water_run.jl --alpha 30 --b0-sigma 5 --label a30_b0
#   julia --project=. scripts/hybrid_water_run.jl --npe 64 --nfe 128

using MRISystemPhantom, KomaMRI, Suppressor
using Random, Statistics, Printf, JSON, NPZ
include(joinpath(@__DIR__, "t1_fit_lib.jl"))

# ── Config ────────────────────────────────────────────────────────────────────
const FOV   = 0.2
const VOXEL = 1.0
const TE    = 0.020
const TI0_REF = 0.10          # cached-template reference TI (water far from null)
const TR0_REF = 5.0

let
    global Npe = 32; global Nfe = 64
    global α_deg = 90.0; global b0_sigma = 0.0
    global field = :T15
    global label = ""
    i = 1
    while i <= length(ARGS)
        a = ARGS[i]
        if     a == "--npe"      && i < length(ARGS); global Npe = parse(Int, ARGS[i+1]); i += 2
        elseif a == "--nfe"      && i < length(ARGS); global Nfe = parse(Int, ARGS[i+1]); i += 2
        elseif a == "--alpha"    && i < length(ARGS); global α_deg = parse(Float64, ARGS[i+1]); i += 2
        elseif a == "--b0-sigma" && i < length(ARGS); global b0_sigma = parse(Float64, ARGS[i+1]); i += 2
        elseif a == "--label"    && i < length(ARGS); global label = ARGS[i+1]; i += 2
        else i += 1 end
    end
end

const α_exc   = deg2rad(α_deg)
const sin_α   = abs(sin(α_exc))
const scanner = scanner_for_field(field)
p_(x) = replace(string(x), "." => "p")
isempty(label) && (label = "a$(p_(α_deg))_b0$(p_(b0_sigma))_npe$(Npe)fe$(Nfe)")

run_dir = joinpath(@__DIR__, "runs", "hybrid_water", label)
arr_dir = joinpath(run_dir, "arrays")
mkpath(arr_dir); mkpath(joinpath(run_dir, "figures"))

println("="^64)
println("hybrid_water_run  label=$label  α=$(α_deg)°  B0σ=$(b0_sigma) Hz  Npe=$Npe Nfe=$Nfe")
println("="^64)

# ── Phantoms: full, spheres-only (dry); derive the water spins exactly ────────
slice_center = MRISystemPhantom.PLATE_Z_MM.T1
augment = AugmentConfig(B0_sigma_Hz = b0_sigma)
cfg_kw  = (field = field, voxel_size_mm = VOXEL,
           slice_thickness_mm = VOXEL, slice_center_mm = slice_center, augment = augment)
cfg_full = PhantomConfig(; include_plates = [:T1, :water], cfg_kw...)
cfg_dry  = PhantomConfig(; include_plates = [:T1],          cfg_kw...)

full = build_phantom(cfg_full)
dry  = build_phantom(cfg_dry)
n_dry, n_full = length(dry.x), length(full.x)
@assert full.x[1:n_dry] ≈ dry.x "phantom layout not [spheres…, water…]"
water_mask = falses(n_full); water_mask[(n_dry+1):end] .= true
water = full[water_mask]
n_water = length(water.x)
@printf("spins: full=%d spheres=%d water=%d (%.1f%%)\n", n_full, n_dry, n_water, 100n_water/n_full)

# ── Sphere ground truth + pixel map ───────────────────────────────────────────
descs   = sphere_descriptors(:T1, cfg_full; rng = MersenneTwister(0))
n_sph   = length(descs)
T1_true = [d.T1 for d in descs]
centres = [d.centre for d in descs]
sphere_px = [(phys_to_pixel(c[2], Npe, FOV), phys_to_pixel(c[1], Nfe, FOV)) for c in centres]

# ── Schedule (manual log-spaced TIs; α/B0 from config) ────────────────────────
TIs = [0.015,0.022,0.032,0.045,0.065,0.10,0.15,0.22,0.35,0.55,0.85,1.30,2.00,2.80]
TRs = fill(5.0, length(TIs))
n_blk = length(TIs)

# ── Helpers ───────────────────────────────────────────────────────────────────
seq_for(TI,TR) = Suppressor.@suppress ir_se_2d_sequence(TI,TE,TR; α_exc=α_exc, FOV=FOV, Nfe=Nfe, Npe=Npe)
sim_ksp(ph,TI,TR) = raw_to_kspace(Suppressor.@suppress(simulate(ph, seq_for(TI,TR), scanner)), Npe, Nfe)
theory_ksp(ph,TI,TR) = ComplexF32.(ir_se_theory_image(ph; TI=TI, TR=TR, α_exc=α_exc, θ_inv=π,
                                    FOV=FOV, Nfe=Nfe, Npe=Npe, voxel_mm=VOXEL).ksp)
relerr(a,b) = sum(abs.(a .- b)) / max(sum(abs.(b)), eps())

# ── Cached Koma-water template (one sim per geometry) ─────────────────────────
# Row k of the raw k-space stack is acquired in shot k, weighted by that shot's
# transient magnetisation. Two rescaling models share one reference sim:
#   scalar  : one mean-transient scalar for the whole acquisition (assumes a flat
#             per-shot ramp — exact only at α=90° where cos α = 0).
#   perline : rescale each row k by its OWN shot's transient — tracks the ramp
#             exactly for any α / TI, removing the dominant single-scalar residual.
# Both reuse the fitter's forward model (transient_mz_at_excite_npe, fits.jl:163).
# The per-line model is now the library `CachedWaterModel` (src/water_cache.jl) so
# this harness and the E2 env exercise one implementation; the scalar variant is
# kept inline as the comparison baseline. The single-α grid here matches this run's
# α_exc (the env uses a multi-α bank for learned-α; see water_cache.jl header).
const T1_WATER = MRISystemPhantom.BACKGROUND_WATER[field].T1
mw(TI,TR; α=α_exc) = transient_mz_at_excite_npe(T1_WATER, TI, TR, π, α; Npe=Npe) * sin(α)

ksp_water_ref = sim_ksp(water, TI0_REF, TR0_REF)            # one sim per geometry
# (1) scalar template (inline baseline)
w_geom = ksp_water_ref ./ ComplexF32(mw(TI0_REF, TR0_REF))
cached_water_scalar(TI,TR; α=α_exc) = w_geom .* ComplexF32(mw(TI,TR; α=α))
# (2) per-line template via the library (CachedWaterModel + cached_water_ksp)
water_model = build_cached_water_model(water, scanner;
    FOV=FOV, Nfe=Nfe, Npe=Npe, α_grid=[α_exc],
    TI_ref=TI0_REF, TR_ref=TR0_REF, TE_ref=TE)
cached_water_perline(TI,TR; α=α_exc) = cached_water_ksp(water_model, TI, TR, α, TE)

# ── Per-block: compute all variants ───────────────────────────────────────────
img_variants   = ["koma_full","hybrid_cached","hybrid_cached_perline","hybrid_analytic","theory_full"]
water_variants = ["water_koma","water_cache","water_cache_perline","water_theory"]
ksp = Dict(v => Vector{Matrix{ComplexF32}}(undef, n_blk) for v in vcat(img_variants, water_variants))

println("\nComputing $n_blk blocks × $(length(img_variants)) image variants…")
for b in 1:n_blk
    TI, TR = TIs[b], TRs[b]
    kf = sim_ksp(full, TI, TR)
    kd = sim_ksp(dry,  TI, TR)
    kw_koma     = kf .- kd
    kw_scalar   = cached_water_scalar(TI, TR)
    kw_perline  = cached_water_perline(TI, TR)
    kw_theory   = theory_ksp(water, TI, TR)

    ksp["koma_full"][b]             = kf
    ksp["hybrid_cached"][b]         = kd .+ kw_scalar
    ksp["hybrid_cached_perline"][b] = kd .+ kw_perline
    ksp["hybrid_analytic"][b]       = kd .+ kw_theory
    ksp["theory_full"][b]           = theory_ksp(full, TI, TR)
    ksp["water_koma"][b]            = kw_koma
    ksp["water_cache"][b]           = kw_scalar
    ksp["water_cache_perline"][b]   = kw_perline
    ksp["water_theory"][b]          = kw_theory
    @printf("  blk %2d/%d TI=%.3f\n", b, n_blk, TI)
end

# ── Reconstruct magnitude images (identical operator for every variant) ───────
img = Dict(v => [kspace_to_image(ksp[v][b]) for b in 1:n_blk] for v in keys(ksp))

# ── relerr.csv: per-block per-variant vs ground truth ─────────────────────────
open(joinpath(arr_dir, "relerr.csv"), "w") do io
    println(io, "block,TI_s,target,variant,ksp_relerr,img_relerr")
    for b in 1:n_blk
        for v in img_variants[2:end]                      # skip koma_full (truth)
            kr = relerr(ksp[v][b], ksp["koma_full"][b])
            ir = relerr(img[v][b], img["koma_full"][b])
            println(io, "$b,$(TIs[b]),koma_full,$v,$kr,$ir")
        end
        for v in [w for w in water_variants if w != "water_koma"]
            kr = relerr(ksp[v][b], ksp["water_koma"][b])
            ir = relerr(img[v][b], img["water_koma"][b])
            println(io, "$b,$(TIs[b]),water_koma,$v,$kr,$ir")
        end
    end
end
println("wrote relerr.csv")

# ── T1 fits for all four image variants ───────────────────────────────────────
block_TIs    = [copy(TIs) for _ in 1:n_sph]
block_TRs    = [copy(TRs) for _ in 1:n_sph]
block_α_excs = [fill(α_exc, n_blk) for _ in 1:n_sph]
α_inv_vec    = fill(π, n_blk)
fits = Dict{String,Any}()
open(joinpath(arr_dir, "t1fits.csv"), "w") do io
    println(io, "variant,label,T1_true_s,T1_fit_s,T1_sigma_s,mape_vs_true_pct,rel_vs_koma_pct")
    koma_fit = nothing
    for v in img_variants
        mags = accumulate_block_mags(ksp[v], sphere_px, sin_α)
        f = fit_fleet(block_TIs, α_inv_vec, mags, T1_true, Npe; block_TRs=block_TRs, block_α_excs=block_α_excs)
        fits[v] = f
        v == "koma_full" && (koma_fit = f)
        for i in 1:n_sph
            rel = koma_fit === nothing ? 0.0 : abs(f.T1_fit[i]-koma_fit.T1_fit[i])/koma_fit.T1_fit[i]*100
            sig = isnan(f.T1_sigma[i]) ? 0.0 : f.T1_sigma[i]
            println(io, "$v,$(descs[i].label),$(T1_true[i]),$(f.T1_fit[i]),$sig,$(f.mapes[i]),$rel")
        end
    end
end
println("wrote t1fits.csv")
for v in img_variants
    f = fits[v]
    rel = [abs(f.T1_fit[i]-fits["koma_full"].T1_fit[i])/fits["koma_full"].T1_fit[i]*100 for i in 1:n_sph]
    @printf("  %-16s  MAPE_vs_true=%.2f%%   |ΔT1|/koma mean=%.3f%% max=%.3f%%\n",
            v, mean(f.mapes), mean(rel), maximum(rel))
end

# ── Save stacked arrays ───────────────────────────────────────────────────────
stack(v) = (A = Array{ComplexF32,3}(undef, n_blk, Npe, Nfe); for b in 1:n_blk; A[b,:,:]=ksp[v][b]; end; A)
stacki(v) = (A = Array{Float32,3}(undef, n_blk, Npe, Nfe); for b in 1:n_blk; A[b,:,:]=img[v][b]; end; A)
for v in keys(ksp)
    npzwrite(joinpath(arr_dir, "$(v)_ksp.npy"), stack(v))
    npzwrite(joinpath(arr_dir, "$(v)_img.npy"), stacki(v))
end
npzwrite(joinpath(arr_dir, "TIs.npy"), TIs)
npzwrite(joinpath(arr_dir, "TRs.npy"), TRs)
npzwrite(joinpath(arr_dir, "sphere_centres.npy"), [c[j] for c in centres, j in 1:2])
npzwrite(joinpath(arr_dir, "sphere_px.npy"), [sphere_px[i][j] for i in 1:n_sph, j in 1:2])
npzwrite(joinpath(arr_dir, "spins_xy_full.npy"),  hcat(full.x, full.y))
npzwrite(joinpath(arr_dir, "spins_xy_water.npy"), hcat(water.x, water.y))
npzwrite(joinpath(arr_dir, "occupancy_full.npy"),  Array(phantom_occupancy(full,  Npe, Nfe, FOV)))
npzwrite(joinpath(arr_dir, "occupancy_water.npy"), Array(phantom_occupancy(water, Npe, Nfe, FOV)))
println("wrote stacked arrays")

# ── Timing (per-step cost) ────────────────────────────────────────────────────
sim_ksp(full,0.5,5.0); sim_ksp(dry,0.5,5.0)             # warmup
cached_water_scalar(0.5,5.0); cached_water_perline(0.5,5.0)
t_full    = @elapsed sim_ksp(full,0.5,5.0)
t_dry     = @elapsed sim_ksp(dry,0.5,5.0)
t_scalar  = @elapsed cached_water_scalar(0.5,5.0)
t_perline = @elapsed cached_water_perline(0.5,5.0)
t_theoryw = @elapsed theory_ksp(water,0.5,5.0)
spd_scalar  = t_full / (t_dry + t_scalar)
spd_perline = t_full / (t_dry + t_perline)
@printf("timing: full=%.3fs dry=%.3fs theory_water=%.3fs\n", t_full, t_dry, t_theoryw)
@printf("  cached rescale  scalar=%.2es  perline=%.2es  (perline/scalar=%.1f×)\n",
        t_scalar, t_perline, t_perline/max(t_scalar,eps()))
@printf("  per-step speedup  scalar=%.2f×  perline=%.2f×  (spin ratio %.2f×)\n",
        spd_scalar, spd_perline, n_full/n_dry)

# ── config.json ───────────────────────────────────────────────────────────────
open(joinpath(run_dir, "config.json"), "w") do io
    JSON.print(io, Dict{String,Any}(
        "label"=>label, "field"=>String(field), "FOV_m"=>FOV, "voxel_size_mm"=>VOXEL,
        "Nfe"=>Nfe, "Npe"=>Npe, "alpha_deg"=>α_deg, "b0_sigma_Hz"=>b0_sigma, "TE_s"=>TE,
        "schedule"=>"manual", "n_blocks"=>n_blk, "TIs_s"=>TIs, "TRs_s"=>TRs,
        "ref_TI_s"=>TI0_REF, "ref_TR_s"=>TR0_REF, "T1_water_s"=>T1_WATER,
        "n_spins"=>Dict("full"=>n_full,"spheres"=>n_dry,"water"=>n_water),
        "img_variants"=>img_variants, "water_variants"=>water_variants,
        "ground_truth"=>"koma_full",
        "dz_fe_m"=>FOV/Nfe, "dy_pe_m"=>FOV/Npe, "z_plate_m"=>slice_center/1000,
        "timing_s"=>Dict("full"=>t_full,"spheres"=>t_dry,
                         "cached_rescale_scalar"=>t_scalar,"cached_rescale_perline"=>t_perline,
                         "theory_water"=>t_theoryw,
                         "per_step_speedup_scalar"=>spd_scalar,"per_step_speedup_perline"=>spd_perline,
                         "spin_ratio"=>n_full/n_dry),
    ), 2); println(io)
end
println("\nwrote $run_dir/config.json")
println("Done.")

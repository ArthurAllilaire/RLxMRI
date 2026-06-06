# Dump phantom spin positions + sphere geometry for the pixel-grid overlay
# diagnostic. Pure geometry — no simulation, no fitting.
#
# Companion to scripts/pixel_grid_overlay.py, which renders the figure.
#
# Usage:
#   julia --project=. scripts/pixel_grid_overlay.jl
#   julia --project=. scripts/pixel_grid_overlay.jl --npe 32 --nfe 64 --voxel-mm 1.0
#   julia --project=. scripts/pixel_grid_overlay.jl --npe 32 --nfe 64 --voxel-mm 1.0 --ti 0.1 --spoil 
#   julia --project=. scripts/pixel_grid_overlay.jl --npe 64 --nfe 128 --voxel-mm 1.0 --ti 0.5 --spoil --water
using MRISystemPhantom, KomaMRI, Suppressor
using JSON, NPZ, Random

# ── Defaults (match scripts/t1_fit_vs_true.jl) ────────────────────────────────
FOV       = 0.2
Nfe       = 64
Npe       = 32
VOXEL_MM  = 1.0
PLATE     = :T1            # imaging plate (always T1 for now)
WATER     = false          # --water: include water background in PhantomConfig
SLICE_MM  = 0.0            # 0 → auto: voxel_mm (slab thickness)
SLICE_CENTER_MM = NaN      # NaN → auto: PLATE_Z_MM.T1
TI_IMG    = NaN            # --ti <s>: if set, simulate one IR-SE block and dump |image|
TE_IMG    = 0.020
TR_IMG    = 5.0
CLEAN     = false          # --clean-recon: Hamming window + 2× zero-pad
SPOIL     = false          # --spoil: run BOTH spoil and no-spoil sims, and ALWAYS emit clean recons
                            #          (gives 4 image variants for the comparison figure)

# ── CLI ───────────────────────────────────────────────────────────────────────
let i = 1
    while i <= length(ARGS)
        if ARGS[i] == "--nfe" && i < length(ARGS)
            global Nfe = parse(Int, ARGS[i+1]); i += 2
        elseif ARGS[i] == "--npe" && i < length(ARGS)
            global Npe = parse(Int, ARGS[i+1]); i += 2
        elseif ARGS[i] == "--fov" && i < length(ARGS)
            global FOV = parse(Float64, ARGS[i+1]); i += 2
        elseif ARGS[i] == "--voxel-mm" && i < length(ARGS)
            global VOXEL_MM = parse(Float64, ARGS[i+1]); i += 2
        elseif ARGS[i] == "--slice-mm" && i < length(ARGS)
            global SLICE_MM = parse(Float64, ARGS[i+1]); i += 2
        elseif ARGS[i] == "--slice-center-mm" && i < length(ARGS)
            global SLICE_CENTER_MM = parse(Float64, ARGS[i+1]); i += 2
        elseif ARGS[i] == "--water"
            global WATER = true; i += 1
        elseif ARGS[i] == "--ti" && i < length(ARGS)
            global TI_IMG = parse(Float64, ARGS[i+1]); i += 2
        elseif ARGS[i] == "--te" && i < length(ARGS)
            global TE_IMG = parse(Float64, ARGS[i+1]); i += 2
        elseif ARGS[i] == "--tr" && i < length(ARGS)
            global TR_IMG = parse(Float64, ARGS[i+1]); i += 2
        elseif ARGS[i] == "--clean-recon"
            global CLEAN = true; i += 1
        elseif ARGS[i] == "--spoil"
            global SPOIL = true; i += 1
        else
            i += 1
        end
    end
end

fov_tag   = replace(string(FOV),      "." => "p")
vox_tag   = replace(string(VOXEL_MM), "." => "p")
ti_tag    = isnan(TI_IMG) ? "" : "_TI" * replace(string(TI_IMG), "." => "p")
# Only tag TR when it differs from default (5 s) — keeps existing run dirs
# (which were all TR=5) from changing names.
tr_tag    = (!isnan(TI_IMG) && TR_IMG != 5.0) ?
                "_TR" * replace(string(TR_IMG), "." => "p") : ""
clean_tag = CLEAN ? "_clean" : ""
spoil_tag = SPOIL ? "_spoil" : ""
# Water tag — keep dry runs unlabelled so existing run dirs stay the same.
water_tag = WATER ? "_water" : ""
run_label = "npe$(Npe)_nfe$(Nfe)_fov$(fov_tag)_vox$(vox_tag)mm$(water_tag)$(ti_tag)$(tr_tag)$(clean_tag)$(spoil_tag)"

println("="^60)
println("pixel_grid_overlay  $(run_label)")
println("  FOV=$(FOV) m  voxel=$(VOXEL_MM) mm  Npe=$(Npe)  Nfe=$(Nfe)  plate=$(PLATE)")
println("="^60)

# Resolve slice geometry. Centre on the T1 plate z; slab thickness defaults to
# one voxel so spins and recon agree on a single in-plane slice. Both values
# are forwarded to PhantomConfig so the builder applies the z-mask — we don't
# post-filter phantom.z ourselves.
slice_center_mm = isnan(SLICE_CENTER_MM) ? MRISystemPhantom.PLATE_Z_MM.T1 : SLICE_CENTER_MM
slice_thick_mm  = SLICE_MM > 0 ? SLICE_MM : VOXEL_MM

include_plates = WATER ? [PLATE, :water] : [PLATE]
cfg     = PhantomConfig(
    field              = :T15,
    voxel_size_mm      = VOXEL_MM,
    include_plates     = include_plates,
    slice_thickness_mm = slice_thick_mm,
    slice_center_mm    = (0.0, 0.0, slice_center_mm),  # (x,y,z) point on slab plane
)
phantom = build_phantom(cfg)
n_total = length(phantom.x)
println("phantom: $(n_total) spins (slab z=$(slice_center_mm) ± $(slice_thick_mm/2) mm, water=$(WATER))")

# Sphere descriptors — same RNG as scripts/t1_fit_vs_true.jl:118.
descs = sphere_descriptors(PLATE, cfg; rng = MersenneTwister(0))

xs = phantom.x
ys = phantom.y
n_slice = length(xs)
z_plate = slice_center_mm / 1000
slice_half = slice_thick_mm / 1000 / 2
println("slice center = $(slice_center_mm) mm   thickness $(slice_thick_mm) mm   →   $n_slice spins")

# Per-sphere pixel mapping (mirrors scripts/t1_fit_vs_true.jl:129-148 and
# src/geometry/projection.jl:11-19 — both index conventions identical).
sphere_centres = Matrix{Float64}(undef, length(descs), 2)
sphere_px      = Matrix{Int}(undef,     length(descs), 2)
for (i, d) in enumerate(descs)
    cx, cy = d.centre[1], d.centre[2]
    sphere_centres[i, 1] = cx
    sphere_centres[i, 2] = cy
    sphere_px[i, 1] = phys_to_pixel(cy, Npe, FOV)  # ipe
    sphere_px[i, 2] = phys_to_pixel(cx, Nfe, FOV)  # ife
end

# Override with RUNS_ROOT to target a version folder; inherited by the Python
# render hooks spawned below so they write to the same place.
runs_root = get(ENV, "RUNS_ROOT", joinpath(@__DIR__, "runs"))
outdir = joinpath(runs_root, "pixel_grid_overlay", run_label)
mkpath(outdir)

# Optional: simulate one IR-SE block at chosen TI and dump |image|. Same code
# path as scripts/t1_fit_vs_true.jl so the Gibbs/leakage pattern matches.
img_meta = Dict{String,Any}("rendered" => false)
# Simulate one IR-SE acquisition (one spoiler config), reconstruct raw + clean
# images, and dump everything tagged with `tag` (e.g. "" or "_spoil"). Returns
# the raw recon image so the caller can use it for diff_image vs theory.
function simulate_and_dump_variant(; phantom, scanner, outdir, tag,
                                     TI, TE, TR, α_exc, FOV, Nfe, Npe,
                                     spoil_enabled)
    seq = Suppressor.@suppress ir_se_2d_sequence(
        TI, TE, TR; α_exc = α_exc, FOV = FOV, Nfe = Nfe, Npe = Npe,
        spoiler = SpoilerConfig(enabled = spoil_enabled),
    )
    raw = Suppressor.@suppress simulate(phantom, seq, scanner)
    ksp = raw_to_kspace(raw, Npe, Nfe)

    img       = kspace_to_image(ksp)
    img_clean = kspace_to_image(ksp; pad_factor = 2, hamming = true)
    # Dump the Hamming-windowed k-space too (on the original grid, no padding),
    # so the Python figure can display it without redefining the window.
    ksp_clean = ksp .* hamming_window_2d(size(ksp)...)

    npzwrite(joinpath(outdir, "image$(tag).npy"),        Array(img))
    npzwrite(joinpath(outdir, "image$(tag)_clean.npy"),  Array(img_clean))
    npzwrite(joinpath(outdir, "kspace$(tag).npy"),       Array(ksp))
    npzwrite(joinpath(outdir, "kspace$(tag)_clean.npy"), Array(ksp_clean))
    img
end

if !isnan(TI_IMG)
    println("simulating IR-SE block: TI=$(TI_IMG) s  TE=$(TE_IMG) s  TR=$(TR_IMG) s  spoil=$(SPOIL)")

    # When --spoil is set, run BOTH no-spoil and spoil sims (gives all 4 variants
    # for the comparison figure). Otherwise just the no-spoil sim. Clean recon
    # is always emitted — it's cheap and the figures want it.
    sim_modes = SPOIL ? [(false, ""), (true, "_spoil")] : [(false, "")]

    scanner = scanner_for_field(cfg)
    npzwrite(joinpath(outdir, "occupancy.npy"), Array(phantom_occupancy(phantom, Npe, Nfe, FOV)))

    # `global` so the no-spoil image survives the for-loop's local scope and
    # is reachable for diff_image below.
    global img_raw = nothing
    for (enable_spoil, tag) in sim_modes
        img_v = simulate_and_dump_variant(;
            phantom, scanner, outdir, tag,
            TI = TI_IMG, TE = TE_IMG, TR = TR_IMG, α_exc = π/2,
            FOV = FOV, Nfe = Nfe, Npe = Npe,
            spoil_enabled = enable_spoil,
        )
        enable_spoil || (global img_raw = img_v)
    end

    theory = ir_se_theory_image(phantom;
        TI = TI_IMG, TR = TR_IMG, α_exc = π/2, θ_inv = π,
        FOV = FOV, Nfe = Nfe, Npe = Npe, voxel_mm = VOXEL_MM,
    )
    diff_image = abs.(img_raw) .- abs.(theory.image)

    npzwrite(joinpath(outdir, "T1_map.npy"),           theory.T1_map)
    npzwrite(joinpath(outdir, "rho_map.npy"),          theory.rho_map)
    npzwrite(joinpath(outdir, "theory_image.npy"),     theory.image)
    npzwrite(joinpath(outdir, "theory_kspace.npy"),    theory.ksp)
    npzwrite(joinpath(outdir, "theory_magnitude.npy"), Array(abs.(theory.image)))
    npzwrite(joinpath(outdir, "diff_image.npy"),       Array(diff_image))

    img_meta["rendered"] = true
    img_meta["TI_s"]     = TI_IMG
    img_meta["TE_s"]     = TE_IMG
    img_meta["TR_s"]     = TR_IMG
    img_meta["clean"]    = true
    img_meta["spoil"]    = SPOIL
    img_meta["img_npe"]  = size(img_raw, 1)
    img_meta["img_nfe"]  = size(img_raw, 2)
end

npzwrite(joinpath(outdir, "spins_xy.npy"),       hcat(xs, ys))
npzwrite(joinpath(outdir, "sphere_centres.npy"), sphere_centres)
npzwrite(joinpath(outdir, "sphere_px.npy"),      sphere_px)

open(joinpath(outdir, "sphere_meta.json"), "w") do io
    arr = [Dict("label" => String(d.label),
                "radius_m" => d.radius,
                "T1_s" => d.T1,
                "centre_xyz_m" => collect(d.centre)) for d in descs]
    JSON.print(io, arr, 2); println(io)
end

open(joinpath(outdir, "config.json"), "w") do io
    JSON.print(io, Dict{String,Any}(
        "FOV_m"          => FOV,
        "voxel_size_mm"  => VOXEL_MM,
        "Nfe"            => Nfe,
        "Npe"            => Npe,
        "plate"          => String(PLATE),
        "z_plate_m"      => z_plate,
        "slice_half_m"   => slice_half,
        "n_spins_total"  => n_total,
        "n_spins_in_slice" => n_slice,
        "dx_fe_m"        => FOV / Nfe,
        "dy_pe_m"        => FOV / Npe,
        "image"          => img_meta,
    ), 2); println(io)
end

println("Wrote to $(outdir)/")

# ── Auto-render Python figures ───────────────────────────────────────────────
# Cheap enough to always run; fail soft so the script still succeeds if the
# venv / python isn't on PATH.
function try_render(script_name, args)
    python = get(ENV, "PYTHON", "python")
    cmd = `$python $(joinpath(@__DIR__, script_name)) $args`
    try
        run(cmd)
        println("rendered via $(script_name)")
    catch e
        println("skipped $(script_name) ($e) — to render manually: $(cmd)")
    end
end

# Single Python entrypoint: renders the per-variant overlays plus (when all
# four spoil/clean variants are present in this run dir) the 5×2 comparison.
try_render("pixel_grid_overlay.py", ["--run", run_label])

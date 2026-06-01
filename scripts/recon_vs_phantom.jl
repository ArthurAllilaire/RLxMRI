# Side-by-side comparison: input phantom geometry vs reconstructed image.
#
# Uses MRISystemPhantom's raw_to_kspace / kspace_to_image / phantom_occupancy
# (the same code path as E2Env._e2_simulate_step) to ensure the diagnostic
# reflects real training conditions.
#
# Writes to scripts/runs/recon_vs_phantom/<run_label>/ alongside a config.json
# describing the run.  Run scripts/recon_vs_phantom.py to render.
#
# Usage:
#   julia --project=. scripts/recon_vs_phantom.jl
#   julia --project=. scripts/recon_vs_phantom.jl --nfe 128 --npe 64 --fov 0.2 --voxel-mm 3.0
using MRISystemPhantom, KomaMRI, Suppressor
using JSON, NPZ

# ── Defaults ─────────────────────────────────────────────────────────────────
FOV       = 0.2
Nfe       = 128
Npe       = 32
VOXEL_MM  = 3.0
WATER     = false
TI_VAL    = 3.0
TE_VAL    = 0.02
TR_VAL    = 4.0

# ── Parse CLI args ────────────────────────────────────────────────────────────
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
        elseif ARGS[i] == "--water"
            global WATER = true; i += 1
        elseif ARGS[i] == "--ti" && i < length(ARGS)
            global TI_VAL = parse(Float64, ARGS[i+1]); i += 2
        elseif ARGS[i] == "--tr" && i < length(ARGS)
            global TR_VAL = parse(Float64, ARGS[i+1]); i += 2
        elseif ARGS[i] == "--te" && i < length(ARGS)
            global TE_VAL = parse(Float64, ARGS[i+1]); i += 2
        else
            i += 1
        end
    end
end

# ── Run label ────────────────────────────────────────────────────────────────
fov_tag = replace(string(FOV),      "." => "p")
vox_tag = replace(string(VOXEL_MM), "." => "p")
water_tag = WATER ? "_water" : ""
run_label = "npe$(Npe)_nfe$(Nfe)_fov$(fov_tag)_vox$(vox_tag)mm$(water_tag)"

println("="^60)
println("recon_vs_phantom  $(run_label)")
println("  FOV=$(FOV) m  voxel=$(VOXEL_MM) mm  Npe=$(Npe)  Nfe=$(Nfe)")
println("  TI=$(TI_VAL) s  TR=$(TR_VAL) s  TE=$(TE_VAL) s  water=$(WATER)")
println("="^60)

plates = WATER ? [:T1, :water] : [:T1]
cfg = PhantomConfig(field = :T15, voxel_size_mm = VOXEL_MM, include_plates = plates)
phantom = build_phantom(cfg)
println("phantom: $(length(phantom.x)) spins, $(length(unique(phantom.T1))) unique T1 values")

seq = Suppressor.@suppress ir_se_2d_sequence(
    TI_VAL, TE_VAL, TR_VAL; α_exc = π/2, FOV = FOV, Nfe = Nfe, Npe = Npe,
)
raw = Suppressor.@suppress simulate(phantom, seq, scanner_for_field(cfg))

ksp = raw_to_kspace(raw, Npe, Nfe)
img = kspace_to_image(ksp)
occ = phantom_occupancy(phantom, Npe, Nfe, FOV)

outdir = joinpath(@__DIR__, "runs", "recon_vs_phantom", run_label)
mkpath(outdir)
npzwrite(joinpath(outdir, "occupancy.npy"), Array(occ))
npzwrite(joinpath(outdir, "image.npy"),     Array(img))
npzwrite(joinpath(outdir, "kspace.npy"),    Array(ksp))

open(joinpath(outdir, "config.json"), "w") do io
    cfg_dict = Dict{String,Any}(
        "FOV_m"        => FOV,
        "voxel_size_mm" => VOXEL_MM,
        "Nfe"          => Nfe,
        "Npe"          => Npe,
        "TI_s"         => TI_VAL,
        "TR_s"         => TR_VAL,
        "TE_s"         => TE_VAL,
        "water"        => WATER,
        "n_spins"      => length(phantom.x),
    )
    JSON.print(io, cfg_dict, 2)
    println(io)
end

println("Wrote to $(outdir)/")

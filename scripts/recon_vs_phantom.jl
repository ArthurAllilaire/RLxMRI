# Side-by-side comparison: input phantom geometry vs reconstructed image.
#
# Uses QalibreMDPhantom's raw_to_kspace / kspace_to_image / phantom_occupancy
# (the same code path as E2Env._e2_simulate_step) to ensure the diagnostic
# reflects real training conditions.
#
# Writes to scripts/accurate/ — run scripts/recon_vs_phantom.py to render.

using QalibreMDPhantom, KomaMRI, Suppressor
using DelimitedFiles

const FOV    = 0.2
const Nfe    = 64
const Npe    = 16
const TI_VAL = 3.0
const TE_VAL = 0.02
const TR_VAL = 4.0

cfg = PhantomConfig(field = :T3, voxel_size_mm = 2.0, include_plates = [:T1])
phantom = build_phantom(cfg)
println("phantom: $(length(phantom.x)) spins, $(length(unique(phantom.T1))) unique T1 values")

seq = Suppressor.@suppress ir_se_2d_sequence(
    TI_VAL, TE_VAL, TR_VAL; α_exc = π/2, FOV = FOV, Nfe = Nfe, Npe = Npe,
)
raw = Suppressor.@suppress simulate(phantom, seq, Scanner())

ksp          = raw_to_kspace(raw, Npe, Nfe)
img          = kspace_to_image(ksp)
occ          = phantom_occupancy(phantom, Npe, Nfe, FOV)
per_shot_dc  = [abs(raw.profiles[k].data[Nfe ÷ 2, 1]) for k in 1:Npe]

outdir = joinpath(@__DIR__, "accurate")
mkpath(outdir)
writedlm(joinpath(outdir, "recon_vs_phantom_occupancy.txt"), occ)
writedlm(joinpath(outdir, "recon_vs_phantom_image.txt"),     img)
writedlm(joinpath(outdir, "recon_vs_phantom_pershot.txt"),   per_shot_dc)
writedlm(joinpath(outdir, "recon_vs_phantom_meta.txt"),      [TI_VAL TR_VAL Nfe Npe])

println("Wrote to $(outdir)/")
println()
println("Per-shot DC (should be roughly constant after fix):")
for (k, v) in enumerate(per_shot_dc)
    println("  shot $k: $(round(v, digits = 2))")
end

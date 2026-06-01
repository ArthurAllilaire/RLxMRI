# Sweep TR (and hence total sim time) and dump per-TR WAPE / Pearson /
# energy-on-support against the T1-plate phantom.
#
# Uses MRISystemPhantom's raw_to_kspace / kspace_to_image / phantom_occupancy
# (the same code path as E2Env._e2_simulate_step) so the sweep reflects
# real training conditions.
#
# Writes to scripts/accurate/ — run scripts/wape_vs_simtime.py to plot.

using MRISystemPhantom, KomaMRI, Suppressor
using DelimitedFiles

const FOV = 0.2
const Nfe = 64
const Npe = 16
const TI  = 3.0
const TE  = 0.02

cfg = PhantomConfig(field = :T3, voxel_size_mm = 3.0, include_plates = [:T1])
phantom = build_phantom(cfg)

occ   = phantom_occupancy(phantom, Npe, Nfe, FOV)
occ_n = occ ./ sum(occ)

TRs = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 10.0, 15.0, 20.0]

results = Array{Float64}(undef, length(TRs), 5)
# columns: TR, sim_time, WAPE, Pearson_r, energy_on_support
for (idx, TR) in enumerate(TRs)
    seq = Suppressor.@suppress ir_se_2d_sequence(
        TI, TE, TR; α_exc = π/2, FOV = FOV, Nfe = Nfe, Npe = Npe,
    )
    raw   = Suppressor.@suppress simulate(phantom, seq, scanner_for_field(cfg))
    ksp   = raw_to_kspace(raw, Npe, Nfe)
    img   = kspace_to_image(ksp)
    img_n = img ./ sum(img)

    wape   = sum(abs.(img_n .- occ_n)) / sum(occ_n) * 100.0
    r      = let a = vec(occ), b = vec(img)
        sum((a .- sum(a)/length(a)) .* (b .- sum(b)/length(b))) /
            (sqrt(sum((a .- sum(a)/length(a)).^2) * sum((b .- sum(b)/length(b)).^2)) + eps())
    end
    e_supp = sum(img[occ .> 0]) / sum(img) * 100.0

    results[idx, :] = [TR, TR*Npe, wape, r, e_supp]
    println("TR=$TR  sim_time=$(TR*Npe)s  WAPE=$(round(wape,digits=1))%  r=$(round(r,digits=3))  e_supp=$(round(e_supp,digits=1))%")
end

outdir = joinpath(@__DIR__, "accurate")
mkpath(outdir)
writedlm(joinpath(outdir, "wape_vs_simtime.txt"), results)
println("\nWrote ", joinpath(outdir, "wape_vs_simtime.txt"))

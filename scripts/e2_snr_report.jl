# Dual-acquisition (NEMA MS-1) SNR at the E2 operating point.
#
# Reproduces the number diagnose_e2.py prints, but reports the MEAN dual-acq
# SNR across the active spheres (and seeds) alongside the peak (brightest
# sphere) the report figure currently cites. Pure Julia: builds the same E2 env
# the trainer uses, resets it, and calls e2_dual_acq_snr_report at its default
# reference block (TI=0.5 s, TE=20 ms, TR=3 s, α=90°).
#
# Run: julia --project=. scripts/e2_snr_report.jl

using MRISystemPhantom
using Statistics, Printf, JSON
include(joinpath(@__DIR__, "..", "julia", "rl_boot.jl"))

const SIGMA   = 50.0
const NSEEDS  = 8            # average over a few phantom poses/materials
const ROI_R   = 1            # match the Run B reports (ROI radius 1)

env = E2Env(
    cfg_field       = :T15,
    voxel_size_mm   = 1.0,
    Nfe = 64, Npe = 32,
    noise_sigma_abs = SIGMA,
    include_water   = true,
    water_model     = :bloch,
    forward_model   = :bloch,
)

per_sphere = Float64[]   # pooled per-sphere dual SNR across seeds
peaks      = Float64[]

for s in 0:(NSEEDS - 1)
    e2_reset!(env; rng_seed = s)
    rep = e2_dual_acq_snr_report(env; roi_radius = ROI_R, seed = s)
    append!(per_sphere, Float64.(rep.image.snr_dual_per_sphere))
    push!(peaks, Float64(rep.image.snr_dual_peak))
end

mean_snr = mean(per_sphere)
med_snr  = median(per_sphere)
mean_pk  = mean(peaks)

@printf("\nE2 dual-acq (NEMA MS-1) SNR  —  σ=%.0f, T15, 64x32, 1mm, ROI=%d, %d seeds\n\n",
        SIGMA, ROI_R, NSEEDS)
@printf("  mean over spheres   = %.2f\n", mean_snr)
@printf("  median over spheres = %.2f\n", med_snr)
@printf("  peak (brightest)    = %.2f  (report figure)\n", mean_pk)
@printf("  n sphere samples    = %d\n", length(per_sphere))

open(joinpath(@__DIR__, "e2_snr_report.json"), "w") do io
    JSON.print(io, Dict(
        "sigma" => SIGMA, "n_seeds" => NSEEDS, "roi_radius" => ROI_R,
        "snr_dual_mean_over_spheres" => mean_snr,
        "snr_dual_median_over_spheres" => med_snr,
        "snr_dual_peak_mean" => mean_pk,
        "n_sphere_samples" => length(per_sphere),
    ), 2)
end
println("\nwrote ", joinpath(@__DIR__, "e2_snr_report.json"))

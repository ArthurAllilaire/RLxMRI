# Experiment 02 — is the time-driven jump a one-off, or periodic?
#
# In experiment 01 we saw a single step jump at sim_time ∈ [66, 80] s in three
# configurations. Question: if we keep simulating past that point, do additional
# jumps occur at fixed intervals (~80 s spacing), or does the signal just settle
# at the new level after one jump?
#
# Setup: TR=2s, Npe=150 → 300 s of cumulative sim time. Same z-line phantom.
# Look at the per-shot |peak| trajectory and locate all step events.

using KomaMRI, QalibreMDPhantom, Suppressor, Printf

n_z      = 21
zs       = collect(LinRange(-1e-3, 1e-3, n_z))
phantom  = Phantom(
    name = "z_line",
    x  = fill(1e-3, n_z), y = fill(1e-3, n_z), z = zs,
    T1 = fill(1.0,  n_z), T2 = fill(0.5,  n_z), T2s = fill(0.5, n_z),
    ρ  = ones(n_z),       Δw = zeros(n_z),
)
scanner = Scanner()

TI, TE, TR   = 0.3, 0.02, 2.0
Nfe, Npe     = 16, 150
FOV          = 0.2

println("="^72)
println("Experiment 02 — long-time trajectory, looking for repeated jumps")
println("="^72)
@printf("TR=%g s, Npe=%d → total sim time ≈ %.0f s\n", TR, Npe, TR*Npe)
println()

seq = Suppressor.@suppress ir_se_2d_sequence(
    TI, TE, TR; α_exc = π/2, FOV = FOV, Nfe = Nfe, Npe = Npe,
    spoiler = SpoilerConfig(enabled = false),
)
raw = Suppressor.@suppress simulate(phantom, seq, scanner)
centre_kx = Nfe ÷ 2 + 1
peaks = [abs(raw.profiles[k].data[centre_kx, 1]) for k in 1:Npe]

# Detect step events: |Δ vs previous| > 1 %.
function find_step_events(peaks, TR, threshold_pct=1.0)
    println("Step events (|Δ vs previous| > $(threshold_pct) %):")
    println("  shot │ sim_time │ |peak|     │ Δ vs prev")
    println("  ─────┼──────────┼────────────┼──────────")
    n = 0
    for k in 2:length(peaks)
        Δrel = 100 * (peaks[k] - peaks[k-1]) / peaks[k-1]
        if abs(Δrel) > threshold_pct
            @printf("  %4d │ %7.1fs │ %10.5g │ %+7.3f%%\n", k, k*TR, peaks[k], Δrel)
            n += 1
        end
    end
    n
end

n_events = find_step_events(peaks, TR)
@printf("\nTotal step events: %d (out of %d shots)\n", n_events, Npe)

# Save trajectory to a CSV so we can plot if needed.
open(joinpath(@__DIR__, "02_trajectory.csv"), "w") do io
    println(io, "shot,sim_time_s,peak_abs")
    for (k, v) in enumerate(peaks)
        @printf(io, "%d,%g,%g\n", k, k*TR, v)
    end
end
println("\nWrote trajectory to 02_trajectory.csv")

# Print every 10th value for an at-a-glance scan.
println("\nEvery-10th sample of the trajectory:")
println("  shot │ sim_time │ |peak|")
println("  ─────┼──────────┼─────────────")
for k in 1:10:Npe
    @printf("  %4d │ %7.1fs │ %12.5g\n", k, k*TR, peaks[k])
end

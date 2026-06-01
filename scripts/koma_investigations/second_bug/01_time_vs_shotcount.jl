# Experiment 01 — is drift driven by total sim time, shot count, or per-shot TR?
#
# Three configs share total simulated time but differ in (TR, Npe):
#   A: TR=10s,  Npe=8     → 80s total,  8 shots
#   B: TR=2s,   Npe=40    → 80s total, 40 shots
#   C: TR=5s,   Npe=16    → 80s total, 16 shots
#
# If drift size & pattern match across A/B/C → cumulative sim time is the driver (H3).
# If shorter-TR configs drift more → steady-state convergence (H2).
# If drift grows with shot count regardless of TR → coherence-pathway buildup (H1).
#
# Phantom: line of 21 spins along z at fixed (x, y) so the Gz crusher has spatial
# extent (matching scripts/test_coherence_pathways.jl). T1=1s, T2=0.5s.
# At TR=10s, Mz recovery is exp(-10) ≈ 1.0 (full); TR=5 → ~0.99; TR=2 → ~0.86.
# So H2 (steady-state) should affect TR=2 noticeably but not TR=5 or TR=10.
#
# We report the per-shot |peak| at echo centre for unspoiled. Spoiled is included
# as a side reference but isn't the focus here (it had separate Koma issues last
# time — we'll come back to it in a later experiment).

using KomaMRI, MRISystemPhantom, Suppressor, Printf

n_z      = 21
zs       = collect(LinRange(-1e-3, 1e-3, n_z))
phantom  = Phantom(
    name = "z_line",
    x  = fill(1e-3, n_z), y = fill(1e-3, n_z), z = zs,
    T1 = fill(1.0,  n_z), T2 = fill(0.5,  n_z), T2s = fill(0.5, n_z),
    ρ  = ones(n_z),       Δw = zeros(n_z),
)
scanner = Scanner()

TI, TE       = 0.3, 0.02
Nfe          = 16
FOV          = 0.2
α_exc        = π / 2

function run_variant(TR::Real, Npe::Int, spoil::Bool)
    seq = Suppressor.@suppress ir_se_2d_sequence(
        TI, TE, TR;
        α_exc = α_exc, FOV = FOV, Nfe = Nfe, Npe = Npe,
        spoiler = SpoilerConfig(enabled = spoil),
    )
    raw = Suppressor.@suppress simulate(phantom, seq, scanner)
    centre_kx = Nfe ÷ 2 + 1
    peaks = [abs(raw.profiles[k].data[centre_kx, 1]) for k in 1:Npe]
    return peaks
end

# Drift score = (last shot − early steady value) / (early steady value).
# "Early steady" = mean of shots 2..min(5, Npe-2) to skip the shot-1 transient.
function drift_metrics(peaks)
    last_peak  = peaks[end]
    n          = length(peaks)
    early_end  = min(5, n - 2)
    early_mean = early_end >= 2 ? sum(peaks[2:early_end]) / (early_end - 1) : peaks[1]
    rel_drift  = 100 * (last_peak - early_mean) / early_mean
    monotonic  = all(peaks[i] >= peaks[i-1] - 1e-6 for i in 3:n) ||
                 all(peaks[i] <= peaks[i-1] + 1e-6 for i in 3:n)
    (early_mean = early_mean,
     last_peak  = last_peak,
     rel_drift  = rel_drift,
     monotonic  = monotonic,
     min_peak   = minimum(peaks),
     max_peak   = maximum(peaks))
end

function print_table(label, peaks)
    @printf("\n  %s\n", label)
    println("  shot │ |peak|      │ Δ vs prev")
    println("  ─────┼─────────────┼──────────")
    for (k, v) in enumerate(peaks)
        Δ = k == 1 ? 0.0 : 100 * (v - peaks[k-1]) / peaks[k-1]
        @printf("  %4d │ %11.6g │ %+7.3f%%\n", k, v, Δ)
    end
end

println("="^72)
println("Experiment 01 — sim time vs shot count vs TR")
println("="^72)
println("All three configs run for ~80 s of cumulative sim time.")
println()

configs = [
    ("A: TR=10s,  Npe=8 ", 10.0,  8),
    ("B: TR=2s,   Npe=40", 2.0,  40),
    ("C: TR=5s,   Npe=16", 5.0,  16),
]

results = []
for (label, TR, Npe) in configs
    println("─"^72)
    println(label, "   (sim time ≈ ", TR*Npe, "s)")
    println("─"^72)
    peaks_ns = run_variant(TR, Npe, false)
    print_table("UNSPOILED", peaks_ns)
    m = drift_metrics(peaks_ns)
    @printf("  → unspoiled: early_mean=%.4g  last=%.4g  rel_drift=%+.2f%%  monotonic=%s\n",
            m.early_mean, m.last_peak, m.rel_drift, m.monotonic)
    push!(results, (label = label, TR = TR, Npe = Npe, peaks = peaks_ns, metrics = m))
end

println()
println("="^72)
println("Comparison")
println("="^72)
@printf("  %-18s  %-6s  %-5s  %-12s  %-12s  %-10s  %-10s\n",
        "config", "TR", "Npe", "early_mean", "last_peak", "rel_drift", "monotonic?")
println("  " * "─"^88)
for r in results
    @printf("  %-18s  %-6.1f  %-5d  %-12.5g  %-12.5g  %+9.2f%%  %-10s\n",
            r.label, r.TR, r.Npe, r.metrics.early_mean, r.metrics.last_peak,
            r.metrics.rel_drift, string(r.metrics.monotonic))
end

println()
println("Interpretation hints:")
println("  - If all three rel_drift values are similar → total sim time drives drift (H3).")
println("  - If TR=2s drift is much larger → steady-state convergence (H2). Should affect early")
println("    shots only; would show in shot-1 vs shot-2..N gap, not shot-N vs shot-(N/2).")
println("  - If rel_drift grows with Npe → coherence-pathway accumulation (H1).")

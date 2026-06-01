# Test whether the multi-shot signal drift we see in `ir_se_2d_sequence`
# (when unspoiled) is multi-shot coherence pathway buildup, the textbook
# reason gradient spoiling exists — not Koma simulator drift.
#
# `scripts/koma_bug_minimal.jl` already established that *without* gradients
# Koma returns a flat per-shot signal across many seconds of simulation
# (after the user's fix to the previously-reported time-driven jump bug).
# So if we add gradients and see a per-shot drift that the SpoilerConfig
# kills, the mechanism is real spin physics, not a Koma artifact.
#
# Setup:
#   - Single position (x=1 mm, y=1 mm) repeated as a *line of spins along z*
#     in [-1, +1] mm so the Gz crusher pair has spatial extent to dephase
#     across. All spins share the same (x, y) so the spin echo formed at
#     ADC midpoint is independent of shot index *in theory* (single point
#     in the FE/PE plane → constant |peak|).
#   - The sequence is `ir_se_2d_sequence` unchanged. We toggle SpoilerConfig.
#   - We read the per-shot peak |signal| at the ADC midpoint (echo centre).
#
# Predictions:
#   - Unspoiled: per-shot |peak| drifts monotonically with shot index
#     (coherence pathways from earlier shots refocusing into later ADC
#     windows). Magnitude grows with TR — more Mz available to be stored.
#   - Spoiled:   per-shot |peak| stays flat to numerical precision (each
#     shot's transverse coherence is dephased before the next shot's RF,
#     breaking the pathway chain).
#
# Run with:
#   julia --project=. scripts/test_coherence_pathways.jl

using KomaMRI, MRISystemPhantom, Suppressor, Printf

# ── Phantom: line of 21 spins along z at fixed (x,y) ─────────────────────────
# z extent matches the slice thickness our 2D sequence implicitly relies on.
# Crusher moment: 30 mT/m × 5 ms × 2 mm ≈ 13 cycles of dephasing across the
# line. Plenty for clean spoiling.
n_z      = 21
zs       = collect(LinRange(-1e-3, 1e-3, n_z))
phantom  = Phantom(
    name = "z_line",
    x  = fill(1e-3, n_z), y = fill(1e-3, n_z), z = zs,
    T1 = fill(1.0,  n_z), T2 = fill(0.5,  n_z), T2s = fill(0.5, n_z),
    ρ  = ones(n_z),       Δw = zeros(n_z),
)
scanner = Scanner()

# ── Sequence parameters ──────────────────────────────────────────────────────
TI, TE       = 0.3, 0.02
Npe, Nfe     = 16, 16
FOV          = 0.2
α_exc        = π / 2

function run_variant(label::String, spoil::Bool, TR::Real)
    seq = Suppressor.@suppress ir_se_2d_sequence(
        TI, TE, TR;
        α_exc = α_exc, FOV = FOV, Nfe = Nfe, Npe = Npe,
        spoiler = SpoilerConfig(enabled = spoil),
    )
    raw = Suppressor.@suppress simulate(phantom, seq, scanner)

    # Per-shot peak |signal| — at the centre kx column (the spin-echo peak).
    centre_kx = Nfe ÷ 2 + 1
    peaks = [abs(raw.profiles[k].data[centre_kx, 1]) for k in 1:Npe]

    @printf("\n  %s  (sim time = %.1fs)\n", label, TR * Npe)
    println("  shot │ |peak|      │ dev vs shot 1")
    println("  ─────┼─────────────┼──────────────")
    for (k, v) in enumerate(peaks)
        dev  = 100 * (v - peaks[1]) / peaks[1]
        flag = abs(dev) > 1.0 ? "  ← DRIFT" : ""
        @printf("  %4d │ %11.6g │ %+7.3f%%%s\n", k, v, dev, flag)
    end
    peaks
end

function verdict(peaks_ns, peaks_s, TR)
    drift_ns = 100 * (peaks_ns[end] - peaks_ns[1]) / peaks_ns[1]
    drift_s  = 100 * (peaks_s[end]  - peaks_s[1])  / peaks_s[1]
    @printf("\n  TR=%4.1fs  drift across %d shots:  unspoiled = %+6.2f%%   spoiled = %+6.3f%%\n",
            TR, Npe, drift_ns, drift_s)
end

println("="^72)
println("Test: do coherence pathways accumulate across unspoiled IR-SE shots?")
println("="^72)
println("Phantom: $(n_z) spins along z in [-1, +1] mm at (x=1 mm, y=1 mm)")
println("Sequence: ir_se_2d_sequence  TI=$(TI)s  TE=$(TE)s  Npe=$(Npe)")
println()
println("Prediction:  unspoiled drift grows with TR  |  spoiled drift ≈ 0")
println("="^72)

# Sweep TR so we can see if drift scales with available Mz (which scales
# with TR via 1 − exp(−TR/T1)). For T1=1s: TR=0.5 → 39%, TR=2 → 86%, TR=5 → 99%.
results = []
for TR in (0.5, 2.0, 5.0)
    println("\n" * "─"^72)
    println("TR = $(TR) s")
    println("─"^72)
    peaks_ns = run_variant("UNSPOILED", false, TR)
    peaks_s  = run_variant("SPOILED  ", true,  TR)
    push!(results, (TR, peaks_ns, peaks_s))
end

println("\n" * "="^72)
println("Summary")
println("="^72)
for (TR, p_ns, p_s) in results
    verdict(p_ns, p_s, TR)
end
println()
println("Hypothesis confirmed if unspoiled drift is large (>5%) AND grows with TR,")
println("while spoiled drift stays small (<1%) at all TR.")

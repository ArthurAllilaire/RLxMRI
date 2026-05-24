# Minimal reproducer for the *residual* Koma drift bug that survived the
# earlier fix (the one verified by scripts/koma_bug_minimal.jl).
#
# Pattern: identical IR-90° shots with a 1-sample ADC, no gradients of any
# kind. Sequence per shot is exactly the koma_bug_minimal pattern:
#
#     180° inversion → TI delay → 90° excitation → 1-sample ADC → TR pad
#
# Expected (under correct physics): per-shot |signal| is constant across all
# shots — every shot is the same physical event on the same (steady-state)
# spin. The koma_bug_minimal at N=16, TR=15 s (240 s total) does indeed
# return 16 identical values after the user's fix.
#
# Observed: extend the same sequence to N=20 at TR=15 s (300 s total) and
# a sharp jump appears around shot 18 (sim_time ≈ 270 s). With longer sim
# runs, more jumps appear afterwards. The "fix" raised the threshold
# but didn't eliminate the underlying numerical drift.
#
# Threshold depends on per-shot complexity:
#   - 1-sample ADC, no 180° refocus, TR=15 s: jump at ~270 s
#   - 16-sample ADC + 180° refocus, TR=15 s:  jump at ~285 s
#   - Full ir_se_2d_sequence (gradients), TR=2-10 s:  jump at ~70 s
# so the bug *scales* with how much arithmetic Koma does per unit sim time.
#
# Run with:
#   julia --project=. scripts/koma_investigation/koma_bug_residual.jl

using KomaMRI, Suppressor, Printf

const B1_AMP  = 20e-6     # 20 µT envelope → 180° in 1 ms
const T_RF    = 1e-3
const T_ADC   = 1e-3
const TI      = 3.0
const TR      = 15.0
const N_SHOTS = 24        # 24 × 15 s = 360 s; well past the threshold

# Single off-origin spin — but position is irrelevant since no gradients.
obj = Phantom(
    name = "spin",
    x = [0.0], y = [0.0], z = [0.0],
    T1 = [1.0], T2 = [1.0], T2s = [1.0],
    ρ  = [1.0], Δw = [0.0],
)

# Same block primitives as koma_bug_minimal.jl.
function rf_block(α, T)
    amp = B1_AMP * (α / π)
    Sequence(reshape([Grad(0.0, T), Grad(0.0, T), Grad(0.0, T)], 3, 1),
             reshape([RF(amp, T)], 1, 1),
             [ADC(0, 0.0)])
end

function adc_block(T)
    Sequence(reshape([Grad(0.0, T), Grad(0.0, T), Grad(0.0, T)], 3, 1),
             reshape([RF(0.0, T)], 1, 1),
             [ADC(1, T)])
end

function build_seq(N::Int; TI::Real, TR::Real)
    seq = Sequence()
    tr_pad = TR - TI - T_RF - T_ADC
    for _ in 1:N
        seq += rf_block(π,   T_RF)        # 180° inversion
        seq += Delay(TI - T_RF)           # TI
        seq += rf_block(π/2, T_RF)        # 90° excitation
        seq += adc_block(T_ADC)           # 1-sample ADC
        tr_pad > 1e-9 && (seq += Delay(tr_pad))
    end
    seq
end

println("KomaMRI version: ", pkgversion(KomaMRI))
println()
println("Per-shot |signal| for N=$(N_SHOTS) identical IR shots (TI=$(TI) s, TR=$(TR) s).")
println("Expected: $(N_SHOTS) identical values after the shot-1 transient.")
println("Observed: a sharp jump should appear when sim_time exceeds ~270 s")
println("          (i.e. around shot $(Int(ceil(270/TR)))).")
println()

seq = build_seq(N_SHOTS; TI = TI, TR = TR)
raw = Suppressor.@suppress simulate(obj, seq, Scanner();
                                      sim_params = Dict{String,Any}("Nthreads" => 1))

vals = [abs(p.data[1, 1]) for p in raw.profiles]
ref  = vals[2]    # shot 2 — past the shot-1 transient

println("  shot │ sim_time [s] │ |signal|     │ dev vs shot 2")
println("  ─────┼──────────────┼──────────────┼──────────────")
for (k, v) in enumerate(vals)
    sim_t = k * TR
    dev   = 100 * (v - ref) / ref
    flag  = abs(dev) > 1.0 ? "  ← DRIFT" : ""
    @printf("  %4d │ %12g │ %12.6g │ %+7.3f%%%s\n", k, sim_t, v, dev, flag)
end

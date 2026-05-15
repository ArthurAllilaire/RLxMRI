# Minimal reproducer for KomaMRI multi-shot amplitude drift.
#
# Builds N identical inversion-recovery shots on a single voxel with no
# spatial encoding (no gradients, no phase encoding). Every shot is the
# same physical event, so the per-shot signal should be a constant.
#
# Expected:   |signal[k]|  ≈ const  for k = 1..N
# Observed:   sharp jumps once cumulative simulated time exceeds ~70-130 s
#
# 22% amplitude drift above 100s
# Run with:
#   julia --project=. scripts/koma_bug_minimal.jl

# 
# shot  1: 0.47627
# shot  2: 0.47627    ← identical, as expected
# ...
# shot  9: 0.47627
# shot 10: 0.5849     ← +22.8 % jump
# shot 11: 0.5849
# ...
# shot 16: 0.5849
# 

# The jump always lands in the **70–150 s** band of cumulative simulated
# time, regardless of how that time is split between TR and shot count:

# | TR (s) | shots | sim time at jump | jump size |
# |---|---|---|---|
# | 4   | none in 16 shots (64 s sim) | — | — |
# | 8   | shot 9 / 16  | ~72 s  | 19 % |
# | 15  | shot 6 / 16  | ~90 s  | 21 % |
# | 30  | shot 4 / 8   | ~120 s | 21 % |

# The threshold is **time-driven, not shot-count-driven**.

using KomaMRI

const B1_AMP = 20e-6           # 20 µT envelope; gives 180° in 1 ms
const T_RF   = 1e-3            # 1 ms hard pulse

# Build a hard pulse of nominal flip α and duration T.
function rf_block(α, T)
    amp = B1_AMP * (α / π)     # linear: 180° at B1_AMP, scale others
    RFb = reshape([RF(amp, T)], 1, 1)
    GR  = reshape([Grad(0.0, T), Grad(0.0, T), Grad(0.0, T)], 3, 1)
    Sequence(GR, RFb, [ADC(0, 0.0)])
end

# Single sampled point at the end of the readout window (instead of a true
# gradient-echo readout) — we just want one number per shot.
function adc_block(T)
    GR  = reshape([Grad(0.0, T), Grad(0.0, T), Grad(0.0, T)], 3, 1)
    RFb = reshape([RF(0.0, T)], 1, 1)
    Sequence(GR, RFb, [ADC(1, T)])
end

# One IR shot:  180 → TI delay → 90 → echo-time delay → 1-sample ADC → TR pad.
function build_seq(N; TI=3.0, TR=15.0)
    seq = Sequence()
    T_adc = 1e-3
    TR_pad = TR - TI - T_RF - T_adc
    for _ in 1:N
        seq += rf_block(π,   T_RF)             # 180° inversion
        seq += Delay(TI - T_RF)                # TI delay
        seq += rf_block(π/2, T_RF)             # 90° excitation
        seq += adc_block(T_adc)                # single ADC sample
        TR_pad > 0 && (seq += Delay(TR_pad))   # TR fill
    end
    seq
end

obj = Phantom(
    name = "single_voxel",
    x = [0.0], y = [0.0], z = [0.0],
    T1 = [1.0], T2 = [1.0], T2s = [1.0],
    ρ = [1.0], Δw = [0.0],
)

println("KomaMRI version: ", pkgversion(KomaMRI))
println()
println("Per-shot |signal| for N=16 identical IR shots (TI=3 s, TR=15 s).")
println("Expected: 16 identical values. Observed: jumps once sim time > ~70 s.")
println()

seq = build_seq(16; TI = 3.0, TR = 15.0)
raw = simulate(obj, seq, Scanner(); sim_params = Dict{String,Any}("Nthreads" => 1))

vals = [abs(p.data[1, 1]) for p in raw.profiles]
ref  = vals[1]
println("  shot | sim time [s] | |signal|     | dev vs shot 1")
println("  ─────┼──────────────┼──────────────┼──────────────")
for (k, v) in enumerate(vals)
    sim_t = k * 15.0
    dev   = 100 * (v - ref) / ref
    flag  = abs(dev) > 1.0 ? "  ← DRIFT" : ""
    println("   $(lpad(k,3)) | $(lpad(sim_t,12))s | $(rpad(round(v, digits=5),13)) | $(lpad(round(dev, digits=2),7)) %$flag")
end

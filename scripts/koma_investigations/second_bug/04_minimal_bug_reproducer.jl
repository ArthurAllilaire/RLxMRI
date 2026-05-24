# Experiment 04 — minimal reproducer for the residual Koma bug.
#
# koma_bug_minimal.jl (after the user's fix) is stable for the pattern:
#   { 180° inversion → TI delay → 90° excitation → 1-sample ADC → TR pad } × N
#
# Our Exp 03 L0 (no gradients) drifts at sim_time≈66 s with the pattern:
#   { 180° inv → TI → 90° exc → Δ → Δ → 180° refocus → Δ → 16-sample ADC → TR pad } × N
#
# Question: which single addition over koma_bug_minimal causes the drift?
# Test four configurations, all on a single off-origin spin, all extended to
# sim_time ≈ 300 s so we're well past the 66 s onset.
#
#   A: koma_bug_minimal verbatim style              (control — expect stable)
#   B: A + 180° refocus pulse                       (adds 1 extra RF per shot)
#   C: A + 16-sample ADC instead of 1-sample        (adds longer ADC)
#   D: A + 180° refocus + 16-sample ADC             (= Exp 03 L0 — expect drift)
#
# The first level that shows drift = the minimal trigger.

using KomaMRI, Suppressor, Printf

const B1_AMP = 20e-6
const T_RF   = 1e-3
const TI     = 3.0
const TR     = 15.0
const T_ADC  = 1e-3
const TE     = 0.02
const N_SHOTS = 20             # 20 × 15 s = 300 s, past the 66 s threshold

# Same off-origin spin as Exp 03 (so any gradients we add later have effect;
# here we have none yet so position is irrelevant, but matches the lineage).
obj = Phantom(
    name = "spin",
    x = [1e-3], y = [1e-3], z = [0.0],
    T1 = [1.0],  T2 = [0.5], T2s = [0.5],
    ρ  = [1.0],  Δw = [0.0],
)

# Helpers (mirror koma_bug_minimal's style: gradient-only "blocks" with all zeros).
function rf_block(α, T)
    amp = B1_AMP * (α / π)
    Sequence(reshape([Grad(0.0, T), Grad(0.0, T), Grad(0.0, T)], 3, 1),
             reshape([RF(amp, T)], 1, 1),
             [ADC(0, 0.0)])
end

function adc_block(N::Int, T::Real)
    Sequence(reshape([Grad(0.0, T), Grad(0.0, T), Grad(0.0, T)], 3, 1),
             reshape([RF(0.0, T)], 1, 1),
             [ADC(N, T)])
end

# Build one shot for a given config.
function build_shot(config::Symbol)
    seq = Sequence()
    if config == :A
        # koma_bug_minimal verbatim: 180 → TI → 90 → 1-sample ADC → TR pad
        seq += rf_block(π,   T_RF)
        seq += Delay(TI - T_RF)
        seq += rf_block(π/2, T_RF)
        seq += adc_block(1, T_ADC)
        tr_pad = TR - TI - T_RF - T_ADC
        tr_pad > 1e-9 && (seq += Delay(tr_pad))
    elseif config == :B
        # A + 180° refocus pulse (no extra ADC samples)
        seq += rf_block(π,   T_RF)
        seq += Delay(TI - T_RF)
        seq += rf_block(π/2, T_RF)
        seq += Delay(TE/2 - T_RF/2 - T_RF/2)        # to 180° refocus centre
        seq += rf_block(π,   T_RF)                  # 180° refocus
        seq += Delay(TE/2 - T_RF/2 - T_ADC/2)       # to ADC midpoint
        seq += adc_block(1, T_ADC)
        tr_pad = TR - TI - 2*T_RF - TE - T_ADC
        tr_pad > 1e-9 && (seq += Delay(tr_pad))
    elseif config == :C
        # A + 16-sample ADC (no extra refocus)
        seq += rf_block(π,   T_RF)
        seq += Delay(TI - T_RF)
        seq += rf_block(π/2, T_RF)
        seq += adc_block(16, T_ADC)
        tr_pad = TR - TI - T_RF - T_ADC
        tr_pad > 1e-9 && (seq += Delay(tr_pad))
    elseif config == :D
        # B + 16-sample ADC (= Exp 03 L0)
        seq += rf_block(π,   T_RF)
        seq += Delay(TI - T_RF)
        seq += rf_block(π/2, T_RF)
        seq += Delay(TE/2 - T_RF/2 - T_RF/2)
        seq += rf_block(π,   T_RF)
        seq += Delay(TE/2 - T_RF/2 - T_ADC/2)
        seq += adc_block(16, T_ADC)
        tr_pad = TR - TI - 2*T_RF - TE - T_ADC
        tr_pad > 1e-9 && (seq += Delay(tr_pad))
    end
    seq
end

function build_full(config::Symbol)
    seq = Sequence()
    for _ in 1:N_SHOTS
        seq += build_shot(config)
    end
    seq
end

function run_config(label::String, config::Symbol)
    seq = Suppressor.@suppress build_full(config)
    raw = Suppressor.@suppress simulate(obj, seq, Scanner();
                                          sim_params = Dict{String,Any}("Nthreads" => 1))
    # Use the centre sample of each ADC (or the only sample for N=1).
    peaks = Float64[]
    for p in raw.profiles
        n = size(p.data, 1)
        push!(peaks, abs(p.data[n ÷ 2 + 1, 1]))
    end

    # Step events: |Δ vs previous| > 1 %.
    n_steps = 0
    first_step = 0
    for k in 2:length(peaks)
        Δrel = 100 * (peaks[k] - peaks[k-1]) / max(peaks[k-1], 1e-12)
        if abs(Δrel) > 1.0
            n_steps += 1
            if first_step == 0; first_step = k; end
        end
    end

    drift = 100 * (peaks[end] - peaks[2]) / max(peaks[2], 1e-12)   # skip shot-1 transient
    @printf("  %-40s  shot-2=%.5g  end=%.5g  drift=%+.2f%%  n_steps=%d  first_step@%s\n",
            label, peaks[2], peaks[end], drift, n_steps,
            first_step == 0 ? "(none)" : "shot $(first_step) (sim_time=$(first_step*TR)s)")
    peaks
end

println("="^96)
println("Experiment 04 — minimal reproducer hunt")
println("="^96)
@printf("N_SHOTS=%d, TR=%g s → sim_time=%g s (past the 66 s threshold)\n\n", N_SHOTS, TR, N_SHOTS*TR)

println("Verdicts:")
A = run_config("A: bug_minimal verbatim (1-samp ADC, no 180°)",   :A)
B = run_config("B: A + 180° refocus  (still 1-samp ADC)",          :B)
C = run_config("C: A + 16-samp ADC   (still no 180°)",             :C)
D = run_config("D: A + 180° + 16-samp ADC (= Exp 03 L0)",          :D)

println("\nThe minimal reproducer is the *simplest* config that drifts.")

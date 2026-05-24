# Experiment 03 — minimum gradient pattern that triggers chaotic drift.
#
# We know:
#   - koma_bug_minimal (RF + delays only, no gradients) is stable.
#   - ir_se_2d_sequence (full RF + Gx + Gy gradients) goes chaotic past ~80 s.
#
# Question: which gradient feature(s) trigger it?
#
# Test ladder, building up from the gradient-free baseline:
#   L0: RF only (replica of koma_bug_minimal style, no gradients).             ← control
#   L1: RF + Gx readout during ADC.
#   L2: RF + Gx prewinder + Gx readout (no Gy).
#   L3: RF + Gx prewinder + per-shot-varying Gy + Gx readout.
#   L4: full ir_se_2d_sequence.                                                ← known chaotic
#
# Each level adds one gradient feature. Find the first level that drifts.
#
# Phantom: single spin at (1 mm, 1 mm, 0) so gradients have effect (off-origin).
# T1=1 s, T2=0.5 s, Δw=0.
# Sequence: same IR-SE backbone (180° → TI → 90° → TE/2 → 180° → TE/2 → ADC → TR delay),
# only the gradient overlays differ between levels.

using KomaMRI, QalibreMDPhantom, Suppressor, Printf

# Single off-origin spin.
phantom = Phantom(
    name = "single_off_origin",
    x = [1e-3], y = [1e-3], z = [0.0],
    T1 = [1.0], T2 = [0.5], T2s = [0.5],
    ρ  = [1.0], Δw = [0.0],
)
scanner = Scanner()

const TI      = 0.3
const TE      = 0.02
const TR      = 2.0
const Npe     = 150       # 300 s total → past the bug threshold
const FOV     = 0.2
const Nfe     = 16
const amp_T   = 20e-6

# Helpers to build event blocks.
function rf_pulse(α; amp = amp_T)
    d = rf_duration(α; amp_T = amp)
    Sequence(reshape([Grad(0.0, d), Grad(0.0, d), Grad(0.0, d)], 3, 1),
             reshape([RF(amp, d)], 1, 1),
             [ADC(0, 0.0)])
end

# A Gx readout-style block with ADC. `Gx_amp` may be 0 (no encoding).
function adc_block(; Gx_amp = 0.0, dur_adc = 1e-3)
    Sequence(reshape([Grad(Gx_amp, dur_adc), Grad(0.0, dur_adc), Grad(0.0, dur_adc)], 3, 1),
             reshape([RF(0.0, dur_adc)], 1, 1),
             [ADC(Nfe, dur_adc)])
end

# Gx prewinder + (optional) Gy phase-encode block, no RF, no ADC.
function pe_block(; Gx_amp = 0.0, Gy_amp = 0.0, dur_pe = 5e-4)
    Sequence(reshape([Grad(Gx_amp, dur_pe), Grad(Gy_amp, dur_pe), Grad(0.0, dur_pe)], 3, 1),
             reshape([RF(0.0, dur_pe)], 1, 1),
             [ADC(0, 0.0)])
end

# Build one shot of the IR-SE backbone with a configurable gradient overlay.
# `gradient_mode` ∈ (:none, :readout_only, :prewinder, :pe).
function build_shot(gradient_mode::Symbol, shot_k::Int)
    γ_Hz   = 42.577e6
    kmax_x = Nfe / (2.0 * FOV)
    dur_adc = 1e-3
    dur_pe  = dur_adc / 2

    Gx_pre = kmax_x / (γ_Hz * dur_pe)
    Gx_ro  = 2.0 * kmax_x / (γ_Hz * dur_adc)
    Δky    = 1.0 / FOV
    ky_k   = (shot_k - 1 - Npe ÷ 2) * Δky
    Gy_k   = -ky_k / (γ_Hz * dur_pe)

    d_inv = rf_duration(π;   amp_T = amp_T)
    d_exc = rf_duration(π/2; amp_T = amp_T)
    d_ref = rf_duration(π;   amp_T = amp_T)
    ti_d  = max(TI  - d_inv/2 - d_exc/2, 0.0)
    te1_d = max(TE/2 - d_exc/2 - dur_pe - d_ref/2, 0.0)
    te2_d = max(TE/2 - d_ref/2 - dur_adc/2, 0.0)

    seq = Sequence()
    seq += rf_pulse(π)              # inversion
    ti_d > 1e-9 && (seq += Delay(ti_d))
    seq += rf_pulse(π/2)            # excitation

    if gradient_mode == :none
        # No prewinder, no Gy, no readout gradient. Pure delays + RF + ADC.
        seq += Delay(dur_pe)
        te1_d > 1e-9 && (seq += Delay(te1_d))
        seq += rf_pulse(π)          # refocus
        te2_d > 1e-9 && (seq += Delay(te2_d))
        seq += adc_block(Gx_amp = 0.0)
    elseif gradient_mode == :readout_only
        seq += Delay(dur_pe)
        te1_d > 1e-9 && (seq += Delay(te1_d))
        seq += rf_pulse(π)
        te2_d > 1e-9 && (seq += Delay(te2_d))
        seq += adc_block(Gx_amp = Gx_ro)
    elseif gradient_mode == :prewinder
        seq += pe_block(Gx_amp = Gx_pre, Gy_amp = 0.0)
        te1_d > 1e-9 && (seq += Delay(te1_d))
        seq += rf_pulse(π)
        te2_d > 1e-9 && (seq += Delay(te2_d))
        seq += adc_block(Gx_amp = Gx_ro)
    elseif gradient_mode == :pe
        seq += pe_block(Gx_amp = Gx_pre, Gy_amp = Gy_k)
        te1_d > 1e-9 && (seq += Delay(te1_d))
        seq += rf_pulse(π)
        te2_d > 1e-9 && (seq += Delay(te2_d))
        seq += adc_block(Gx_amp = Gx_ro)
    else
        error("unknown gradient_mode $gradient_mode")
    end

    # TR pad.
    shot_time = sum(seq.DUR)
    tr_d = max(TR - shot_time, 0.0)
    tr_d > 1e-9 && (seq += Delay(tr_d))
    seq
end

function build_full_seq(gradient_mode::Symbol)
    seq = Sequence()
    for k in 1:Npe
        seq += build_shot(gradient_mode, k)
    end
    seq
end

function run_level(label, gradient_mode)
    seq = Suppressor.@suppress build_full_seq(gradient_mode)
    raw = Suppressor.@suppress simulate(phantom, seq, scanner)
    centre_kx = Nfe ÷ 2 + 1
    peaks = [abs(raw.profiles[k].data[centre_kx, 1]) for k in 1:Npe]

    # Count step events of |Δ| > 1 %.
    n_steps = 0
    max_step = 0.0
    first_step_shot = 0
    for k in 2:Npe
        Δrel = 100 * (peaks[k] - peaks[k-1]) / max(peaks[k-1], 1e-9)
        if abs(Δrel) > 1.0
            n_steps += 1
            if abs(Δrel) > abs(max_step); max_step = Δrel; end
            if first_step_shot == 0; first_step_shot = k; end
        end
    end

    # Drift relative to early steady state.
    early_mean = sum(peaks[3:30]) / 28        # skip transient
    drift      = 100 * (peaks[end] - early_mean) / max(early_mean, 1e-9)

    @printf("\n  %-22s  early=%.5g  end=%.5g  drift=%+.2f%%  n_steps=%d  first_step@shot=%d  max_step=%+.2f%%\n",
            label, early_mean, peaks[end], drift, n_steps, first_step_shot, max_step)
    peaks
end

println("="^88)
println("Experiment 03 — minimum gradient pattern that triggers chaotic drift")
println("="^88)
@printf("Phantom: single spin at (1mm, 1mm, 0)\n")
@printf("TR=%g s, Npe=%d → total sim time ≈ %.0f s (well past previous ~80 s threshold)\n\n",
        TR, Npe, TR*Npe)

@printf("  %-22s  %-12s  %-12s  %-10s  %-8s  %-14s  %-10s\n",
        "level", "early_mean", "end_peak", "drift", "n_steps", "first_step_shot", "max_step")
println("  " * "─"^102)

results = Dict{Symbol, Vector{Float64}}()
for (label, mode) in [("L0 — no gradients",         :none),
                      ("L1 — Gx readout only",       :readout_only),
                      ("L2 — prewinder + readout",   :prewinder),
                      ("L3 — prewinder+Gy+readout",  :pe)]
    results[mode] = run_level(label, mode)
end

# Save trajectories for cross-comparison.
open(joinpath(@__DIR__, "03_trajectories.csv"), "w") do io
    println(io, "shot,sim_time_s,L0_none,L1_readout,L2_prewinder,L3_pe")
    for k in 1:Npe
        @printf(io, "%d,%g,%g,%g,%g,%g\n",
                k, k*TR,
                results[:none][k], results[:readout_only][k],
                results[:prewinder][k], results[:pe][k])
    end
end
println("\nWrote 03_trajectories.csv")

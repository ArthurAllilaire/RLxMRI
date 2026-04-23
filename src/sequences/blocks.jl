# Pulseq-style sequence building blocks used by E0 and the later RL
# experiments. Blocks are parameterised functions returning a `Sequence`
# that can be concatenated with `+=` or simulated directly.
#
# RF timing convention matches `01-FID.jl`:
#   α = 2π · γ · B1 · duration
# where γ is in Hz/T (42.58 MHz/T for protons) and B1 in T.

"""
    rf_duration(α; amp_T)

Duration (s) of a rectangular RF pulse that produces flip angle `α`
(radians) at B1 = `amp_T` tesla, assuming on-resonance.
"""
rf_duration(α; amp_T = 2e-6) = α / (2π * γ * amp_T)

"""
    ir_sequence(TI; amp_T, n_adc, dur_adc)

Inversion-recovery sequence: 180° inversion → delay TI → 90° excitation →
ADC. Returns a `Sequence`. The first ADC sample is the FID amplitude at
the start of readout and, for `amp_T` small enough that T2 decay during
the pulses is negligible, tracks `|1 − 2·exp(−TI/T1)|` for a
Mz-equilibrium spin.
"""
function ir_sequence(TI::Real;
                     amp_T::Real = 2e-6,
                     n_adc::Int  = 16,
                     dur_adc::Real = 2e-3,
                     adc_delay::Real = 0.0)
    d90  = rf_duration(π/2;  amp_T = amp_T)
    d180 = rf_duration(π;    amp_T = amp_T)
    seq  = Sequence()
    seq += RF(amp_T, d180)
    seq += Delay(TI)
    seq += RF(amp_T, d90)
    seq += ADC(n_adc, dur_adc, adc_delay)
    seq
end

"""
    se_sequence(TE; amp_T, n_adc, dur_adc)

Spin-echo sequence: 90° → delay TE/2 → 180° → delay TE/2 → ADC centred
near the echo. The middle ADC sample approximates the echo peak, whose
magnitude tracks `S0·exp(−TE/T2)`.
"""
function se_sequence(TE::Real;
                     amp_T::Real = 20e-6,    # harder pulses → short d180
                     n_adc::Int  = 33,       # odd → one sample at centre
                     dur_adc::Real = min(2e-3, TE/4))
    d90  = rf_duration(π/2; amp_T = amp_T)
    d180 = rf_duration(π;   amp_T = amp_T)

    # Center ADC on the echo: between the end of the 180° and the ADC
    # midpoint we want TE/2 − d180/2 (from pulse centre to echo) minus
    # half the ADC window.
    pre  = TE/2 - d90/2  - d180/2
    post = TE/2 - d180/2 - dur_adc/2
    pre  <= 0 && error("TE too short for 90° pulse duration; got TE=$TE")
    post <= 0 && error("TE too short for ADC window; got TE=$TE")

    seq  = Sequence()
    seq += RF(amp_T, d90)
    seq += Delay(pre)
    seq += RF(amp_T, d180)
    seq += Delay(post)
    seq += ADC(n_adc, dur_adc, 0.0)
    seq
end

"""
    single_spin_phantom(; T1, T2, ρ = 1.0)

Zero-dimensional phantom (one spin at the origin) with the given
relaxation times. Fastest possible simulate-target — the right thing
for non-spatial E0 measurements, mirroring `01-FID.jl`.
"""
function single_spin_phantom(; T1::Real, T2::Real, ρ::Real = 1.0)
    Phantom(x = [0.0], T1 = [Float64(T1)], T2 = [Float64(T2)],
            ρ = [Float64(ρ)])
end

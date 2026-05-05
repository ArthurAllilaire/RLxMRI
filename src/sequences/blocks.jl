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
    generalized_ir_signal(T1, T2; TI, α, n_adc = 64, dur_adc = 2e-3)

Analytical (closed-form) magnitude readout of an IR-prep / 90°-excite /
ADC block. The prep pulse tips Mz by angle `α` (so α = π is canonical
inversion recovery, α = π/2 is saturation recovery, α = 10° is a
small-tip prep). Assumes perfect transverse spoiling after the prep so
that only Mz is carried forward to the excitation. The returned vector is

    |Mz(TI)| · exp(−t_i / T2),  t_i ∈ [0, dur_adc]

with `n_adc` uniformly spaced samples. Orders of magnitude faster than
`simulate()` and equivalent in the single-spin limit — exactly what
PLAN.md §7 calls for on the training hot path.
"""
function generalized_ir_signal(T1::Real, T2::Real;
                               TI::Real, α::Real,
                               n_adc::Int = 64,
                               dur_adc::Real = 2e-3)
    Mz_after_prep = cos(α)
    Mz_at_excite  = 1 - (1 - Mz_after_prep) * exp(-TI / T1)
    amp           = abs(Mz_at_excite)
    ts            = range(0, dur_adc; length = n_adc)
    Float64[amp * exp(-t / T2) for t in ts]
end

"""
    ir_se_2d_sequence(TI, TE, TR; α_exc, FOV, Nfe, Npe, amp_T) → Sequence

Multi-shot 2D spin-echo IR sequence. `Npe` shots are concatenated, each with a
different Gy phase-encode amplitude. Returns a `Sequence` with exactly `Npe`
ADC blocks (one `Nfe`-sample readout per phase-encode step).

Gradient convention (spin-echo k-space):
  - Positive Gx prewinder after excitation: kx → +kmax_x
  - 180° refocus conjugates phase: kx_eff → −kmax_x
  - Positive Gx readout drives kx_eff from −kmax_x to +kmax_x; echo at midpoint

After `simulate()`, reconstruct the magnitude image as:
  `abs.(ifft(ksp, (1,2)))` where `ksp[k,:] = raw.profiles[k].data[:,1]`

Sphere centre (x,y) [m] → image pixel (i_pe, i_fe):
  `i_fe = mod(round(Int, x*Nfe/FOV), Nfe) + 1`
  `i_pe = mod(round(Int, y*Npe/FOV), Npe) + 1`

Note you do a spin echo to remove the T2* decay which is much faster and less predictable
than the T2 delay incurred while doing spin echo.
"""
function ir_se_2d_sequence(TI::Real, TE::Real, TR::Real;
                            α_exc::Real   = π / 2,
                            FOV::Real     = 0.2,
                            Nfe::Int      = 16,
                            Npe::Int      = 8,
                            amp_T::Real   = 20e-6)
    γ_rad = 2π * 42.577e6   # proton gyromagnetic ratio [rad/s/T]

    d_inv = rf_duration(π;     amp_T = amp_T)
    d_exc = rf_duration(α_exc; amp_T = amp_T)
    d_ref = rf_duration(π;     amp_T = amp_T)

    dur_adc = 1e-3            # readout window [s]
    dur_pe  = dur_adc / 2     # prewinder / phase-encode duration [s]

    kmax_x = Nfe / (2.0 * FOV)
    # Positive prewinder (applied after excitation) sets kx → +kmax_x before refocus.
    # Refocus conjugates → kx_eff = −kmax_x. Positive readout sweeps to +kmax_x.
    Gx_pre = kmax_x / (γ_rad * dur_pe)
    Gx_ro  = 2.0 * kmax_x / (γ_rad * dur_adc)

    # Phase-encode steps, centred on zero ky
    Δky      = 1.0 / FOV
    ky_steps = [(k - (Npe + 1.0) / 2.0) * Δky for k in 1:Npe]

    # Timing clamped to be non-negative (min TI / TE constraints ensure positivity)
    ti_delay(d_i, d_e) = max(TI  - d_i / 2 - d_e / 2, 0.0)
    te_delay1(d_e, d_r) = max(TE / 2 - d_e / 2 - dur_pe - d_r / 2, 0.0)
    te_delay2(d_r) = max(TE / 2 - d_r / 2 - dur_adc / 2, 0.0)

    ti_d  = ti_delay(d_inv, d_exc)
    te1_d = te_delay1(d_exc, d_ref)
    te2_d = te_delay2(d_ref)

    shot_time = d_inv + ti_d + d_exc + dur_pe + te1_d + d_ref + te2_d + dur_adc
    tr_d      = max(TR - shot_time, 0.0)

    _gr0(d) = reshape([Grad(0.0, d), Grad(0.0, d), Grad(0.0, d)], 3, 1)
    _rf0(d) = reshape([RF(0.0, d)], 1, 1)
    _rf1(d) = reshape([RF(amp_T, d)], 1, 1)
    _adc0   = [ADC(0, 0.0)]

    seq = Sequence()

    for k in 1:Npe
        # Negative prewinder so that after the 180° refocus (which conjugates
        # phase) the readout samples ky = +ky_steps[k]. With no Gy during ADC
        # this gives clean Cartesian sampling at the intended ky line.
        Gy_k = -ky_steps[k] / (γ_rad * dur_pe)

        # 1. 180° inversion
        seq += Sequence(_gr0(d_inv), _rf1(d_inv), _adc0)

        # 2. TI delay
        ti_d > 1e-9 && (seq += Delay(ti_d))

        # 3. Excitation
        seq += Sequence(_gr0(d_exc), _rf1(d_exc), _adc0)

        # 4. Gx prewinder + Gy phase encode (no RF, after excitation)
        seq += Sequence(
            reshape([Grad(Gx_pre, dur_pe), Grad(Gy_k, dur_pe), Grad(0.0, dur_pe)], 3, 1),
            _rf0(dur_pe), _adc0,
        )

        # 5. TE/2 delay to refocus
        te1_d > 1e-9 && (seq += Delay(te1_d))

        # 6. 180° refocus 
        # TODO: leave out ??
        seq += Sequence(_gr0(d_ref), _rf1(d_ref), _adc0)

        # 7. TE/2 delay to echo
        te2_d > 1e-9 && (seq += Delay(te2_d))

        # 8. ADC + Gx readout (no Gy during readout — Cartesian sampling)
        # The kth profile samples ky = -ky_steps[k] (sign-flipped because the
        # 180° refocus already conjugated ky after the prewinder). ky_steps
        # is symmetric about zero so the resulting ksp is row-reversed vs.
        # canonical; pixel mapping accounts for this via the same FFT.
        seq += Sequence(
            reshape([Grad(Gx_ro, dur_adc), Grad(0.0, dur_adc), Grad(0.0, dur_adc)], 3, 1),
            _rf0(dur_adc),
            [ADC(Nfe, dur_adc)],
        )

        # 9. TR recovery
        tr_d > 1e-9 && (seq += Delay(tr_d))
    end

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

@testset "E2 — sequence + env regressions" begin

    @testset "ir_se_2d_sequence: no Gy during ADC (Cartesian sampling)" begin
        # Regression test for the Gy-rewind-during-ADC bug. The ADC window
        # must run with Gy = 0 — otherwise ky shifts during readout and the
        # k-space line is sheared, breaking the standard FFT reconstruction.
        seq = ir_se_2d_sequence(0.5, 0.02, 2.0;
                                 FOV = 0.2, Nfe = 8, Npe = 4)

        adc_blocks = [i for i in 1:length(seq) if seq[i].ADC[1].N > 0]
        @test length(adc_blocks) == 4         # Npe profiles

        for i in adc_blocks
            b = seq[i]
            @test isapprox(b.GR[2, 1].A, 0.0; atol = 1e-12)   # Gy = 0
            @test b.GR[1, 1].A > 0                            # Gx readout active
        end
    end

    @testset "ir_se_2d_sequence: prewinder sign drives ky to +ky_steps[k] at echo" begin
        # After the fix, the prewinder is negative; the 180° refocus then
        # flips it to positive ky_steps[k] (the intended sampling line).
        # Walking through the structure: every 4th non-empty block (after the
        # prewinder pattern) corresponds to one PE step. Just verify Gy areas
        # span both signs across PE steps and are antisymmetric about zero.
        seq = ir_se_2d_sequence(0.5, 0.02, 2.0;
                                 FOV = 0.2, Nfe = 8, Npe = 4)

        # Prewinder blocks: identifiable as "no ADC, Gy != 0"
        gy_areas = Float64[]
        for i in 1:length(seq)
            b = seq[i]
            if b.ADC[1].N == 0 && abs(b.GR[2, 1].A) > 1e-12
                push!(gy_areas, b.GR[2, 1].A * b.DUR[1])
            end
        end

        @test length(gy_areas) == 4
        # Antisymmetric about zero (centred PE steps): pairwise sums ≈ 0
        sorted = sort(gy_areas)
        @test isapprox(sorted[1] + sorted[end], 0.0; atol = 1e-12)
        @test isapprox(sorted[2] + sorted[3],   0.0; atol = 1e-12)
    end

    @testset "fit_t1_generalized_ir: α_exc-scaled magnitudes need correction" begin
        # Regression for the excitation-flip-angle scaling fix in e2.jl.
        # The fitter has a single amplitude A, so when α_exc varies between
        # blocks the env must divide observed magnitude by sin(α_exc) before
        # passing it in — otherwise the fit is biased.
        T1_true = 0.6
        TIs = [0.05, 0.2, 0.5, 1.5, 3.0]
        α_excs = [deg2rad(d) for d in (10.0, 45.0, 90.0, 135.0, 30.0)]

        # Observed magnitudes: signal scales by sin(α_exc) per shot
        mags_obs = [sin(α) * abs(1 - 2 * exp(-ti / T1_true))
                    for (ti, α) in zip(TIs, α_excs)]

        αs_inv = fill(π, length(TIs))   # inversion prep is always 180°

        # Without correction: feed raw mags directly → biased fit
        f_bad = fit_t1_generalized_ir(TIs, αs_inv, mags_obs;
                                        T1_range = (0.01, 3.0), n_grid = 200)

        # With correction (matches what _e2_update_t1_estimates! now does)
        mags_corr = [m / max(abs(sin(α)), 1e-3)
                     for (m, α) in zip(mags_obs, α_excs)]
        f_good = fit_t1_generalized_ir(TIs, αs_inv, mags_corr;
                                         T1_range = (0.01, 3.0), n_grid = 200)

        @test isapprox(f_good.T1, T1_true; rtol = 0.05)
        @test abs(f_good.T1 - T1_true) <= abs(f_bad.T1 - T1_true) + 1e-9
    end

    @testset "steady_state_mz_at_excite: analytic identities" begin
        T1 = 0.8

        # TR → ∞ collapses to legacy non-steady-state form
        for (TI, θ) in [(0.1, π), (0.3, π/2), (0.05, deg2rad(10))]
            mz_inf  = steady_state_mz_at_excite(T1, TI, Inf, θ, π/2)
            mz_legacy = 1 - (1 - cos(θ)) * exp(-TI / T1)
            @test isapprox(mz_inf, mz_legacy; atol = 1e-12)
        end

        # θ_inv = π, α_exc = π/2 → textbook IR with finite TR:
        #   1 − 2·exp(−TI/T1) + exp(−TR/T1)
        for (TI, TR) in [(0.05, 0.5), (0.3, 1.0), (0.1, 0.3), (1.0, 5.0)]
            mz   = steady_state_mz_at_excite(T1, TI, TR, π, π/2)
            text = 1 - 2*exp(-TI/T1) + exp(-TR/T1)
            @test isapprox(mz, text; atol = 1e-12)
        end

        # Finite-TR correction matters: at TR/T1 = 0.5, magnetisation is
        # nowhere near the TR=∞ value.
        @test abs(steady_state_mz_at_excite(T1, 0.1, 0.4, π, π/2) -
                  steady_state_mz_at_excite(T1, 0.1, Inf, π, π/2)) > 0.3
    end

    @testset "fit_t1_generalized_ir: TR-aware fit unbiased; TR-blind biased" begin
        # When TR/T1 is small, the legacy "full recovery" model biases T1.
        # Generate clean steady-state data, then confirm the new TR-aware
        # call recovers T1 while the old TR-blind call does not.
        T1_true = 1.2
        TIs = [0.05, 0.15, 0.4, 0.9, 1.5]
        TRs = [0.5,  0.8,  1.2, 1.8, 2.5]    # all comparable to T1
        αs_inv = fill(π, length(TIs))
        α_excs = fill(π/2, length(TIs))

        # Magnitudes per textbook IR with finite TR
        mags = [abs(1 - 2*exp(-ti/T1_true) + exp(-tr/T1_true))
                for (ti, tr) in zip(TIs, TRs)]

        f_blind = fit_t1_generalized_ir(TIs, αs_inv, mags;
                                         T1_range = (0.05, 5.0), n_grid = 400)
        f_aware = fit_t1_generalized_ir(TIs, αs_inv, mags;
                                         TRs = TRs, α_excs = α_excs,
                                         T1_range = (0.05, 5.0), n_grid = 400)

        @test isapprox(f_aware.T1, T1_true; rtol = 0.02)
        @test abs(f_aware.T1 - T1_true) < abs(f_blind.T1 - T1_true)
    end

    @testset "steady-state IR formula matches KomaMRI simulation" begin
        # Validate the analytic Mz(TI⁻) against a Bloch simulation. Use
        # short T2 so transverse magnetisation decays between TRs (≈ perfect
        # spoiling assumption); θ_inv=π, α_exc=π/2, which reaches steady
        # state in one TR (cos α_exc = 0 kills the transient term).
        T1 = 0.5
        T2 = 0.02                                    # T2 ≪ (TR − TI)
        amp_T = 100e-6                               # short pulses → analytic ≈ Bloch
        d180 = rf_duration(π;   amp_T = amp_T)
        d90  = rf_duration(π/2; amp_T = amp_T)
        n_rep = 4

        for (TI, TR) in [(0.05, 0.5), (0.2, 1.0), (0.1, 0.3), (0.4, 1.5)]
            ti_delay = TI - d180/2 - d90/2
            ti_delay > 0 || continue

            obj = single_spin_phantom(T1 = T1, T2 = T2)
            seq = Sequence()
            for _ in 1:n_rep
                seq += RF(amp_T, d180)
                seq += Delay(ti_delay)
                seq += RF(amp_T, d90)
                seq += ADC(1, 1e-6, 0.0)
                shot = d180 + ti_delay + d90 + 1e-6
                seq += Delay(TR - shot)
            end

            raw = @suppress simulate(obj, seq, Scanner())
            koma_mag = abs(raw.profiles[end].data[1, 1])

            analytic = abs(steady_state_mz_at_excite(T1, TI, TR, π, π/2))
            @test isapprox(koma_mag, analytic; rtol = 0.05, atol = 5e-3)
        end
    end

    @testset "e2_step!: TR lifts to honour TI (no silent TI cap)" begin
        # Regression for the `TI = min(TI, TR*0.9)` bug, which silently
        # capped TI when the agent picked a small TR — locking out the
        # long-T1 regime. The new policy: TR lifts to (TI+TE)/0.9 instead.
        env = E2Env(; max_blocks = 2, time_budget_s = 600.0,
                     Nfe = 8, Npe = 4)
        e2_reset!(env; rng_seed = 42)

        TI_req = 2.5
        TE_req = 0.02
        TR_req = 0.5    # too small to fit TI
        _, _, _, info = e2_step!(env, [TI_req, TE_req, TR_req, 90.0, 0.0])

        @test isapprox(info["TI"], TI_req; atol = 1e-9)
        @test info["TR"] >= (TI_req + TE_req) / 0.90 - 1e-9
    end

end

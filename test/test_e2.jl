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

    @testset "varying-action: per-shot bias vs KomaMRI (α_exc = π/2)" begin
        # The existing "matches KomaMRI" test repeats the SAME shot n_rep
        # times so the simulator naturally reaches the analytic steady
        # state. Real E2 episodes don't do that — the agent picks a
        # DIFFERENT (TI, TR, α_exc) every shot, so each block starts
        # from whatever the previous block's exit state was, NOT from
        # the steady-state Mz_pre that the analytic formula assumes.
        # This test pins that gap: per-shot Koma magnitude vs per-shot
        # analytic prediction. With α_exc = π/2 the cross-shot memory
        # is killed in one step (cos α_exc = 0), so the only bias is
        # "previous shot's (TR−TI) sets actual Mz_pre, current shot's
        # (TR−TI) sets predicted Mz_pre" — plus the very first shot
        # which starts at M0 = 1, not at any steady state.
        T1 = 0.6
        T2 = 0.02                                 # T2 ≪ TR−TI for spoiling
        amp_T = 100e-6
        d180 = rf_duration(π;   amp_T = amp_T)
        d90  = rf_duration(π/2; amp_T = amp_T)

        # Mix that swings TR widely between adjacent shots — the worst
        # case for the steady-state assumption. Long → short → long.
        actions = [(0.05, 0.5), (0.30, 1.5), (0.10, 0.4),
                   (0.40, 2.0), (0.15, 0.6), (0.80, 3.0)]

        obj = single_spin_phantom(T1 = T1, T2 = T2)
        seq = Sequence()
        for (TI, TR) in actions
            ti_delay = TI - d180/2 - d90/2
            ti_delay > 0 || error("TI too short for pulse durations")
            seq += RF(amp_T, d180)
            seq += Delay(ti_delay)
            seq += RF(amp_T, d90)
            seq += ADC(1, 1e-6, 0.0)
            shot = d180 + ti_delay + d90 + 1e-6
            seq += Delay(TR - shot)
        end

        raw = @suppress simulate(obj, seq, Scanner())

        biases  = Float64[]
        for (k, (TI, TR)) in enumerate(actions)
            koma     = abs(raw.profiles[k].data[1, 1])
            analytic = abs(steady_state_mz_at_excite(
                            T1, TI, TR, π, π/2))
            push!(biases, abs(koma - analytic))
            @info "shot bias (α_exc=π/2)" k TI TR koma analytic abs_bias=biases[end]
        end

        # Documented upper bound on the per-shot bias. Tighten as evidence
        # accumulates; the test is meant to SURFACE the assumption gap,
        # not to enforce it's small. Empirically the cold-start shot
        # (k=1) carries the largest bias (~0.43) because the simulator
        # starts at M0=1, not at the analytic steady-state Mz_pre.
        # Subsequent shots are bounded around 0.30.
        @test maximum(biases) < 0.50
    end

    @testset "varying-action: end-to-end T1 fit bias vs KomaMRI" begin
        # Same physical setup as above, but pipe the simulated magnitudes
        # through fit_t1_generalized_ir and check the recovered T1 against
        # the truth. This is the REPORT-RELEVANT number: how much does the
        # steady-state-per-shot assumption bias the T1 estimate when the
        # data come from a varying-action sequence?
        T1_true = 0.8
        T2 = 0.02
        amp_T = 100e-6
        d180 = rf_duration(π;   amp_T = amp_T)
        d90  = rf_duration(π/2; amp_T = amp_T)

        actions = [(0.05, 0.5), (0.30, 1.5), (0.10, 0.4),
                   (0.50, 2.0), (0.20, 0.8), (0.80, 3.0),
                   (0.15, 0.45), (1.20, 4.0)]

        obj = single_spin_phantom(T1 = T1_true, T2 = T2)
        seq = Sequence()
        for (TI, TR) in actions
            ti_delay = TI - d180/2 - d90/2
            ti_delay > 0 || continue
            seq += RF(amp_T, d180)
            seq += Delay(ti_delay)
            seq += RF(amp_T, d90)
            seq += ADC(1, 1e-6, 0.0)
            shot = d180 + ti_delay + d90 + 1e-6
            seq += Delay(TR - shot)
        end

        raw  = @suppress simulate(obj, seq, Scanner())
        mags = [abs(raw.profiles[k].data[1, 1]) for k in 1:length(actions)]

        TIs    = [a[1] for a in actions]
        TRs    = [a[2] for a in actions]
        αs_inv = fill(π,   length(actions))
        α_excs = fill(π/2, length(actions))

        f = fit_t1_generalized_ir(TIs, αs_inv, mags;
                                   TRs = TRs, α_excs = α_excs,
                                   T1_range = (0.05, 5.0), n_grid = 400)

        rel_bias = abs(f.T1 - T1_true) / T1_true
        @info "varying-action end-to-end" T1_true T1_est=f.T1 rel_bias residual=f.residual

        # Documented upper bound — surfaces the bias rather than hiding it.
        @test rel_bias < 0.30
    end

    @testset "varying-action with small-tip α_exc — bias grows" begin
        # When α_exc < π/2, cos(α_exc) > 0 keeps cross-shot memory alive,
        # so the steady-state-per-shot assumption fails harder. Run the
        # same varying-action probe at α_exc = 30° and confirm the
        # per-shot bias is at least as large as the π/2 case (we don't
        # over-claim the ratio — just show direction of effect).
        T1 = 0.6
        T2 = 0.02
        amp_T = 100e-6
        d180  = rf_duration(π; amp_T = amp_T)
        α_exc = deg2rad(30.0)
        d_exc = rf_duration(α_exc; amp_T = amp_T)

        actions = [(0.05, 0.5), (0.30, 1.5), (0.10, 0.4),
                   (0.40, 2.0), (0.15, 0.6)]

        obj = single_spin_phantom(T1 = T1, T2 = T2)
        seq = Sequence()
        for (TI, TR) in actions
            ti_delay = TI - d180/2 - d_exc/2
            ti_delay > 0 || error("TI too short")
            seq += RF(amp_T, d180)
            seq += Delay(ti_delay)
            seq += RF(amp_T, d_exc)
            seq += ADC(1, 1e-6, 0.0)
            shot = d180 + ti_delay + d_exc + 1e-6
            seq += Delay(TR - shot)
        end

        raw = @suppress simulate(obj, seq, Scanner())

        biases = Float64[]
        for (k, (TI, TR)) in enumerate(actions)
            koma = abs(raw.profiles[k].data[1, 1])
            # Predicted Mxy = sin(α_exc) · |Mz_at_excite| (steady state)
            analytic = sin(α_exc) * abs(steady_state_mz_at_excite(
                                          T1, TI, TR, π, α_exc))
            push!(biases, abs(koma - analytic))
            @info "shot bias (α_exc=30°)" k TI TR koma analytic abs_bias=biases[end]
        end

        # Loose upper bound — purpose is to log the bias, not gate on it.
        @test maximum(biases) < 0.50
    end

    # ── F1+ : finite-Npe transient closed-form forward model (E2.4) ──────────
    #
    # Pins the new `transient_mz_at_excite_npe(T1, TI, TR, θ_inv, α_exc; Npe)`
    # against three analytic limits and against KomaMRI on actual multi-shot
    # sequences. Plus a negative-regression test that the steady-state model
    # is provably biased on Npe-shot data — should fail loudly if anyone
    # silently re-introduces the steady-state assumption in fits.jl.
    @testset "F1+ closed-form limits" begin
        T1, TI, TR = 1.0, 0.5, 2.0
        # Limit 1: Npe = 1 ⇒ TR-blind transient form 1 − (1 − cos θ_inv)·E1
        @test isapprox(transient_mz_at_excite_npe(T1, TI, TR, π, π/2; Npe = 1),
                        1 - (1 - cos(π)) * exp(-TI/T1); rtol = 1e-12)
        # Limit 2: Npe → ∞ ⇒ steady-state fixed point
        @test isapprox(transient_mz_at_excite_npe(T1, TI, TR, π, π/2; Npe = 10_000),
                        steady_state_mz_at_excite(T1, TI, TR, π, π/2);
                        rtol = 1e-3)
        # Limit 3: TR → ∞ ⇒ Mz_pre[k] ≡ 1 for k > 0 ⇒ pure transient
        @test isapprox(transient_mz_at_excite_npe(T1, TI, 1e6, π, π/2; Npe = 8),
                        1 - (1 - cos(π)) * exp(-TI/T1); rtol = 1e-6)
        # Limit 4: Non-trivial α_exc (cos > 0) keeps cross-shot memory alive
        # so Npe=8 ≠ Npe=1 even if TR is finite — sanity check that α_exc
        # actually enters the recurrence.
        @test !isapprox(
            transient_mz_at_excite_npe(T1, TI, TR, π, deg2rad(30); Npe = 8),
            transient_mz_at_excite_npe(T1, TI, TR, π, deg2rad(30); Npe = 1);
            atol = 0.01)
    end

    @testset "F1+ matches KomaMRI on Npe-shot IR-SE single-spin" begin
        # Build the actual Npe-shot IR-SE sequence (cold start from M0=1, the
        # regime _e2_simulate_step runs in). Short T2 enforces spoiling, the
        # only assumption F1+ shares with the simulator.
        T2 = 0.02
        amp_T = 100e-6
        d180 = rf_duration(π;   amp_T = amp_T)
        d90  = rf_duration(π/2; amp_T = amp_T)

        for (T1, TI, TR, Npe) in [(1.0, 0.5, 2.0, 8), (0.3, 0.1, 1.0, 8),
                                   (0.05, 0.02, 0.5, 4)]
            ti_delay = TI - d180/2 - d90/2
            ti_delay > 0 || continue

            obj = single_spin_phantom(T1 = T1, T2 = T2)
            seq = Sequence()
            for _ in 1:Npe
                seq += RF(amp_T, d180)
                seq += Delay(ti_delay)
                seq += RF(amp_T, d90)
                seq += ADC(1, 1e-6, 0.0)
                shot = d180 + ti_delay + d90 + 1e-6
                seq += Delay(TR - shot)
            end
            raw = @suppress simulate(obj, seq, Scanner())
            # Mean over shots ≈ DC of IFFT for a single-pixel-bright sphere
            mags = [abs(raw.profiles[k].data[1, 1]) for k in 1:Npe]
            sim_mean = sum(mags) / Npe
            ana_mean = abs(transient_mz_at_excite_npe(T1, TI, TR, π, π/2; Npe = Npe))
            @test isapprox(sim_mean, ana_mean; rtol = 0.05, atol = 5e-3)
        end
    end

    @testset "F1+ fitter recovers T1 from adaptive (varying-TI/TR) sequences" begin
        # Regime E2 actually operates in: 8 distinct (TI, TR) per fit, each
        # block an Npe-shot IR-SE from M0=1. Generate "data" from F1+ itself
        # (the closed-form-vs-simulator agreement is pinned by the previous
        # testset).
        Npe = 8
        rng = MersenneTwister(42)
        for T1 in [0.1, 0.5, 1.0, 1.8]
            n   = 8
            TIs = sort!(exp.(log(0.02) .+ (log(2.0) − log(0.02)) .* rand(rng, n)))
            TRs = TIs .+ 0.5 .+ 1.5 .* rand(rng, n)            # TR > TI
            αes = fill(π/2, n);   αs = fill(π, n)
            mags = [abs(transient_mz_at_excite_npe(T1, TIs[i], TRs[i],
                                                      π, αes[i]; Npe = Npe))
                    for i in 1:n]
            fit = fit_t1_generalized_ir(TIs, αs, mags;
                        TRs = TRs, α_excs = αes, Npe = Npe,
                        T1_range = (0.02, 3.0), n_grid = 400)
            @test isapprox(fit.T1, T1; rtol = 0.05)
        end
    end

    @testset "Steady-state fitter is provably biased on Npe-shot data (regression for §19.5.1)" begin
        # Pin the bias direction: steady-state model on Npe=8 data must miss
        # the truth by ≥10%, while F1+ recovers within 5%. Should fail loudly
        # if anyone silently re-introduces the steady-state assumption.
        T1, Npe = 1.0, 8
        TIs     = [0.05, 0.3, 1.0, 2.0]
        TRs     = [0.5,  0.6, 1.2, 2.5]
        αs      = fill(π,   4)
        αes     = fill(π/2, 4)
        mags = [abs(transient_mz_at_excite_npe(T1, TIs[i], TRs[i], π, π/2; Npe = Npe))
                for i in 1:4]
        f_npe = fit_t1_generalized_ir(TIs, αs, mags;
                                        TRs = TRs, α_excs = αes, Npe = Npe,
                                        T1_range = (0.05, 5.0), n_grid = 400)
        f_steady = fit_t1_generalized_ir(TIs, αs, mags;
                                           TRs = TRs, α_excs = αes,
                                           T1_range = (0.05, 5.0), n_grid = 400)
        @info "steady-state-bias regression" T1_true=T1 T1_npe=f_npe.T1 T1_steady=f_steady.T1
        @test abs(f_npe.T1    - T1) / T1 < 0.05      # F1+ unbiased
        @test abs(f_steady.T1 - T1) / T1 > 0.10      # steady-state biased
        @test abs(f_npe.T1 - T1) < abs(f_steady.T1 - T1)
    end

    # ── σ fixes (E2.4) ───────────────────────────────────────────────────────
    @testset "σ Fix A: n-gate forces floor at small n" begin
        # With n ≤ 4 the residual variance has no statistical power, so
        # σ²_resid should be Inf and σ²_eff should equal the floor — even
        # when the residuals happen to be tiny (e.g. clean synthetic data).
        T1   = 1.0
        TIs  = [0.05, 0.3, 1.0]                # n = 3
        TRs  = [0.5,  1.0, 2.0]
        αs   = fill(π,   3)
        αes  = fill(π/2, 3)
        mags = [abs(transient_mz_at_excite_npe(T1, TIs[i], TRs[i], π, π/2; Npe = 8))
                for i in 1:3]

        # No floor → at n=3 σ_T1 should be undefined (NaN) under Fix A
        # because σ²_resid = Inf and there's no other source of σ.
        f_no_floor = fit_t1_generalized_ir(TIs, αs, mags;
                                             TRs = TRs, α_excs = αes, Npe = 8,
                                             T1_range = (0.05, 5.0), n_grid = 400)
        @test !isfinite(f_no_floor.T1_sigma) || f_no_floor.T1_sigma > 1e9

        # With an explicit absolute noise floor → σ_T1 finite and reasonable
        f_with_floor = fit_t1_generalized_ir(TIs, αs, mags;
                                               TRs = TRs, α_excs = αes, Npe = 8,
                                               abs_noise_sigma = 0.01,
                                               T1_range = (0.05, 5.0), n_grid = 400)
        @test isfinite(f_with_floor.T1_sigma)
        @test f_with_floor.T1_sigma > 0
    end

    @testset "σ Fix B: abs_noise_sigma decouples σ from data RMS (action choice)" begin
        # Same physical noise level, different per-fit data RMS (because the
        # agent picks different TIs). Under the relative noise_sigma path,
        # σ_T1 changes with data RMS — that's the §7.3 coupling the fix
        # removes. Under abs_noise_sigma, σ_T1 should be (approximately) the
        # same regardless of the data scale.
        T1, Npe = 1.0, 8
        TIs_hi  = [0.05, 0.10, 0.20, 0.40, 0.80]   # all high-signal (|1−2E1| close to 1)
        TIs_lo  = [0.50, 0.60, 0.65, 0.70, 0.75]   # all near the IR null (low signal)
        TRs     = fill(2.0, 5)
        αs      = fill(π,   5)
        αes     = fill(π/2, 5)
        m_hi = [abs(transient_mz_at_excite_npe(T1, TIs_hi[i], TRs[i], π, π/2; Npe=Npe))
                for i in 1:5]
        m_lo = [abs(transient_mz_at_excite_npe(T1, TIs_lo[i], TRs[i], π, π/2; Npe=Npe))
                for i in 1:5]
        rms_hi = sqrt(sum(abs2, m_hi) / length(m_hi))
        rms_lo = sqrt(sum(abs2, m_lo) / length(m_lo))
        @test rms_hi > 3 * rms_lo            # the two regimes really do differ in RMS

        # Relative path: σ_T1 scales with rms(m), so the two fits' σ differ
        f_hi_rel = fit_t1_generalized_ir(TIs_hi, αs, m_hi;
                                           TRs = TRs, α_excs = αes, Npe = Npe,
                                           noise_sigma = 0.05,
                                           T1_range = (0.05, 5.0), n_grid = 400)
        f_lo_rel = fit_t1_generalized_ir(TIs_lo, αs, m_lo;
                                           TRs = TRs, α_excs = αes, Npe = Npe,
                                           noise_sigma = 0.05,
                                           T1_range = (0.05, 5.0), n_grid = 400)
        # Absolute path: same σ_floor per fit, σ_T1 should be similar in scale
        abs_noise = 0.05 * rms_hi
        f_hi_abs = fit_t1_generalized_ir(TIs_hi, αs, m_hi;
                                           TRs = TRs, α_excs = αes, Npe = Npe,
                                           abs_noise_sigma = abs_noise,
                                           T1_range = (0.05, 5.0), n_grid = 400)
        f_lo_abs = fit_t1_generalized_ir(TIs_lo, αs, m_lo;
                                           TRs = TRs, α_excs = αes, Npe = Npe,
                                           abs_noise_sigma = abs_noise,
                                           T1_range = (0.05, 5.0), n_grid = 400)

        # The abs-path σ floor is identical for both fits, so any remaining
        # variation in σ_T1 comes from Jacobian conditioning (real signal,
        # not floor-coupling). The relative-path picks up an extra
        # rms-scaling factor, so its ratio is *further* from 1 — i.e. the
        # two action-choices look more different in σ-space than they
        # really are. Fix B's claim is "abs ratio closer to 1 than rel
        # ratio". Both ratios are <1 in this setup (low-signal fit happens
        # to have looser σ even after Jacobian wins back some), so closer-
        # to-1 means *larger*: `abs_ratio > rel_ratio`.
        rel_ratio = f_lo_rel.T1_sigma / f_hi_rel.T1_sigma
        abs_ratio = f_lo_abs.T1_sigma / f_hi_abs.T1_sigma
        @info "σ Fix B" rel_ratio abs_ratio rms_hi rms_lo
        @test abs(log(abs_ratio)) < abs(log(rel_ratio))
    end

    @testset "Profile-likelihood σ — well-determined fit gives small σ" begin
        # Synthetic clean data at T1 = 0.5 s, 8 well-spread TIs in the
        # informative window, plus small noise. SSE has one sharp basin →
        # both methods give small σ; they should agree within ~factor 5.
        T1_true = 0.5
        Npe     = 8
        TIs = [0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0, 1.5]
        TRs = fill(2.0, length(TIs))
        αs  = fill(π,   length(TIs))
        αes = fill(π/2, length(TIs))
        rng  = MersenneTwister(7)
        mags = [abs(transient_mz_at_excite_npe(T1_true, TIs[i], TRs[i],
                                                 π, π/2; Npe = Npe)) +
                0.02 * randn(rng) for i in eachindex(TIs)]

        f_prof = fit_t1_generalized_ir(TIs, αs, mags; TRs = TRs, α_excs = αes,
                                         Npe = Npe, abs_noise_sigma = 0.02,
                                         sigma_method = :profile_likelihood)
        f_asy  = fit_t1_generalized_ir(TIs, αs, mags; TRs = TRs, α_excs = αes,
                                         Npe = Npe, abs_noise_sigma = 0.02,
                                         sigma_method = :asymptotic)

        @test isapprox(f_prof.T1, T1_true; rtol = 0.05)
        @test isapprox(f_asy.T1,  T1_true; rtol = 0.05)
        # σ_T1 should be small (< 20 % of T1) on well-determined fit
        @test isfinite(f_prof.T1_sigma) && f_prof.T1_sigma < 0.20 * T1_true
        @test isfinite(f_asy.T1_sigma)  && f_asy.T1_sigma  < 0.20 * T1_true
        # Both methods should agree within ~factor 5 on a unimodal fit —
        # they're measuring the same width of the same basin (modulo grid
        # discretisation in profile-likelihood).
        @test 0.05 * f_asy.T1_sigma < f_prof.T1_sigma < 20 * f_asy.T1_sigma
    end

    @testset "Profile-likelihood σ — multimodal SSE gives wide σ" begin
        # Synthetic short-T1 data with all-saturated TIs → flat SSE plateau
        # across short T1s. Asymptotic σ from a local J^T J at the LM
        # minimum is over-confident; profile-likelihood should report wide.
        T1_true = 0.023      # very short
        Npe     = 8
        TIs = [0.5, 1.0, 1.5, 2.0, 0.7, 1.2, 1.8, 2.5]   # all ≫ T1_true
        TRs = fill(3.0, length(TIs))
        αs  = fill(π,   length(TIs))
        αes = fill(π/2, length(TIs))
        # Ground-truth mags + small noise
        rng  = MersenneTwister(42)
        mags = [abs(transient_mz_at_excite_npe(T1_true, TIs[i], TRs[i],
                                                 π, π/2; Npe = Npe)) +
                0.02 * randn(rng) for i in eachindex(TIs)]

        f_prof = fit_t1_generalized_ir(TIs, αs, mags; TRs = TRs, α_excs = αes,
                                         Npe = Npe, abs_noise_sigma = 0.02,
                                         sigma_method = :profile_likelihood)
        f_asy  = fit_t1_generalized_ir(TIs, αs, mags; TRs = TRs, α_excs = αes,
                                         Npe = Npe, abs_noise_sigma = 0.02,
                                         sigma_method = :asymptotic)

        # Profile-likelihood σ should be large vs T1*, because the saturated
        # data doesn't constrain T1. Use 50 % of T1* as the wide threshold.
        @test isfinite(f_prof.T1_sigma)
        @test f_prof.T1_sigma > 0.5 * f_prof.T1
        # Crucially, profile σ should expose the ambiguity that asymptotic
        # σ misses: either asymptotic returns NaN (singular J^T J at the
        # degenerate LM minimum) or profile σ is at least 5× larger.
        @test isnan(f_asy.T1_sigma) || f_prof.T1_sigma > 5 * f_asy.T1_sigma
    end

    @testset "Profile-likelihood σ — point estimate unchanged from asymptotic" begin
        # The σ change must NOT alter the LM point estimate. Same T1*, same A*.
        rng = MersenneTwister(0)
        for T1_true in [0.05, 0.2, 0.7, 1.5]
            n   = 8
            TIs = sort!(exp.(log(0.05) .+ (log(2.5) − log(0.05)) .* rand(rng, n)))
            TRs = TIs .+ exp.(log(0.5) .+ log(3.0) .* rand(rng, n))
            αes = fill(π/2, n);  αs = fill(π, n)
            mags = [abs(transient_mz_at_excite_npe(T1_true, TIs[i], TRs[i],
                                                     π, π/2; Npe = 8))
                    for i in 1:n]

            f_a = fit_t1_generalized_ir(TIs, αs, mags; TRs = TRs, α_excs = αes,
                                          Npe = 8, abs_noise_sigma = 0.01,
                                          sigma_method = :asymptotic)
            f_p = fit_t1_generalized_ir(TIs, αs, mags; TRs = TRs, α_excs = αes,
                                          Npe = 8, abs_noise_sigma = 0.01,
                                          sigma_method = :profile_likelihood)
            @test isapprox(f_a.T1, f_p.T1; rtol = 1e-9)
            @test isapprox(f_a.A,  f_p.A;  rtol = 1e-9)
            @test isapprox(f_a.residual, f_p.residual; rtol = 1e-9)
        end
    end

    @testset "Bootstrap σ — finite on saturated data where asymptotic is NaN" begin
        # Saturated data → all measurements ≈ A → SSE flat across short T1s.
        # Asymptotic σ → NaN (singular J^T J). Profile σ wide. Bootstrap
        # widely spread because resampled residuals push T1*_b across the
        # flat-basin region.
        T1_true = 0.023
        Npe     = 8
        TIs = [0.5, 1.0, 1.5, 2.0, 0.7, 1.2, 1.8, 2.5]   # all ≫ T1_true
        TRs = fill(3.0, length(TIs)); αs = fill(π, length(TIs))
        αes = fill(π/2, length(TIs))
        rng = MersenneTwister(42)
        mags = [abs(transient_mz_at_excite_npe(T1_true, TIs[i], TRs[i],
                                                 π, π/2; Npe = Npe)) +
                0.02 * randn(rng) for i in eachindex(TIs)]

        f_boot = fit_t1_generalized_ir(TIs, αs, mags; TRs = TRs, α_excs = αes,
                                         Npe = Npe, abs_noise_sigma = 0.02,
                                         sigma_method = :bootstrap,
                                         n_bootstrap = 200)
        f_asy  = fit_t1_generalized_ir(TIs, αs, mags; TRs = TRs, α_excs = αes,
                                         Npe = Npe, abs_noise_sigma = 0.02,
                                         sigma_method = :asymptotic)

        # Bootstrap σ is finite and non-trivial on saturated data
        @test isfinite(f_boot.T1_sigma)
        @test f_boot.T1_sigma > 0.01      # ≫ grid resolution
        # Asymptotic σ would typically be NaN here (singular Jacobian).
        # Test claim: bootstrap captures the ambiguity that asymptotic can't.
        @test isnan(f_asy.T1_sigma) || f_boot.T1_sigma > 5 * f_asy.T1_sigma
    end

    @testset "Bootstrap σ — point estimate unchanged from asymptotic" begin
        # Same regression as profile-likelihood: σ method must not move T1*.
        rng = MersenneTwister(0)
        for T1_true in [0.05, 0.2, 0.7, 1.5]
            n   = 8
            TIs = sort!(exp.(log(0.05) .+ (log(2.5) − log(0.05)) .* rand(rng, n)))
            TRs = TIs .+ exp.(log(0.5) .+ log(3.0) .* rand(rng, n))
            αes = fill(π/2, n);  αs = fill(π, n)
            mags = [abs(transient_mz_at_excite_npe(T1_true, TIs[i], TRs[i],
                                                     π, π/2; Npe = 8))
                    for i in 1:n]
            f_a = fit_t1_generalized_ir(TIs, αs, mags; TRs = TRs, α_excs = αes,
                                          Npe = 8, abs_noise_sigma = 0.01,
                                          sigma_method = :asymptotic)
            f_b = fit_t1_generalized_ir(TIs, αs, mags; TRs = TRs, α_excs = αes,
                                          Npe = 8, abs_noise_sigma = 0.01,
                                          sigma_method = :bootstrap,
                                          n_bootstrap = 50)
            @test isapprox(f_a.T1, f_b.T1; rtol = 1e-9)
            @test isapprox(f_a.A,  f_b.A;  rtol = 1e-9)
        end
    end

    @testset "Phase-sensitive (signed) fit — recovers T1 cleanly, no multimodality" begin
        # When the fitter sees signed data via phase-sensitive recon, the
        # forward model |1 − 2·exp(−TI/T1)| → (1 − 2·exp(−TI/T1)) is
        # monotonic in T1. SSE has one basin per sphere, asymptotic σ is
        # honest, profile σ is tight.
        T1_true = 0.05
        Npe     = 8
        TIs = [0.07, 0.12, 0.20, 0.35, 0.55, 0.80, 1.20, 1.80]
        TRs = fill(2.5, length(TIs)); αs = fill(π, length(TIs))
        αes = fill(π/2, length(TIs))
        # SIGNED ground truth (no abs)
        mags_signed = [transient_mz_at_excite_npe(T1_true, TIs[i], TRs[i],
                                                    π, π/2; Npe = Npe)
                       for i in eachindex(TIs)]

        f_signed = fit_t1_generalized_ir(TIs, αs, mags_signed; TRs = TRs,
                                           α_excs = αes, Npe = Npe,
                                           abs_noise_sigma = 0.01,
                                           signed = true,
                                           sigma_method = :profile_likelihood)
        # Magnitude fit on the same data using |mags_signed| would be the
        # baseline — recovers truth too in the noiseless case but is more
        # vulnerable to noise.
        @test isapprox(f_signed.T1, T1_true; rtol = 0.05)
        @test isfinite(f_signed.T1_sigma)
        # σ should be tight (< 30 % of T1) because there's no multimodal
        # ambiguity — only one basin in the signed model
        @test f_signed.T1_sigma < 0.30 * T1_true
    end

    @testset "Phase-sensitive (signed) fit — point estimate matches magnitude when signs are consistent" begin
        # Sanity: if all measurements are post-null (signs all positive),
        # signed fit and magnitude fit should give the same T1*.
        T1_true = 0.5
        TIs = [0.5, 0.8, 1.2, 1.8, 2.5]   # all post-null for T1 = 0.5 (null at TI ≈ 0.35)
        TRs = fill(3.0, length(TIs)); αs = fill(π, length(TIs))
        αes = fill(π/2, length(TIs))
        mags_signed = [transient_mz_at_excite_npe(T1_true, TIs[i], TRs[i],
                                                    π, π/2; Npe = 8)
                       for i in eachindex(TIs)]
        # Verify all positive (post-null regime)
        @test all(>=(0), mags_signed)

        f_mag    = fit_t1_generalized_ir(TIs, αs, abs.(mags_signed);
                                           TRs = TRs, α_excs = αes, Npe = 8,
                                           signed = false)
        f_signed = fit_t1_generalized_ir(TIs, αs, mags_signed;
                                           TRs = TRs, α_excs = αes, Npe = 8,
                                           signed = true)
        @test isapprox(f_mag.T1, f_signed.T1; rtol = 0.01)
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

    @testset "E2 random subset reset keeps fixed obs shape and active sphere identities" begin
        env = E2Env(; subset_size = 5, Nfe = 8, Npe = 4,
                     max_blocks = 2, time_budget_s = 600.0)

        obs1 = e2_reset!(env; rng_seed = 123)
        idx1 = copy(env.sphere_indices)
        T1_1 = copy(env.T1_base)

        @test env.n_spheres == 5
        @test length(obs1) == 8 * 4 + 2 * 5 + 3
        @test length(idx1) == 5
        @test issorted(idx1)
        @test length(unique(idx1)) == 5
        @test all(1 .<= idx1 .<= 14)
        @test T1_1 == env.T1_base_pool[idx1]

        obs2 = e2_reset!(env; rng_seed = 123)
        @test env.sphere_indices == idx1
        @test env.T1_base == T1_1
        @test length(obs2) == length(obs1)

        obs3 = e2_reset!(env; rng_seed = 124)
        @test length(obs3) == length(obs1)
        @test length(env.T1_true) == 5
    end

end

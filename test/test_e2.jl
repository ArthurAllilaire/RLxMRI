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

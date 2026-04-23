@testset "sequences & fitting (E0 building blocks)" begin

    @testset "analytical IR fit" begin
        # 3-param magnitude IR fit should recover T1 from a clean signal
        for T1_true in (0.05, 0.2, 0.5, 1.0, 2.0)
            TIs  = adaptive_TI_schedule(T1_true)
            mags = [abs(1 - 2 * exp(-ti / T1_true)) for ti in TIs]
            f = fit_t1_ir(TIs, mags;
                          T1_range = (T1_true / 4, T1_true * 4), n_grid = 200)
            @test isapprox(f.T1, T1_true; rtol = 0.02)
        end
    end

    @testset "analytical SE fit" begin
        for T2_true in (0.05, 0.2, 0.5, 1.5)
            TEs  = adaptive_TE_schedule(T2_true)
            mags = [exp(-te / T2_true) for te in TEs]
            f = fit_t2_se(TEs, mags)
            @test isapprox(f.T2, T2_true; rtol = 0.01)
        end
    end

    @testset "fit_t2_se rejects pathological input" begin
        @test_throws ErrorException fit_t2_se([1.0, 2.0], [1.0])
        @test_throws ErrorException fit_t2_se([1.0, 2.0], [1.0, 2.0])  # non-decaying
    end

    @testset "fit_t1_ir rejects too few samples" begin
        @test_throws ErrorException fit_t1_ir([1e-3, 1e-2], [0.1, 0.5])
    end

    @testset "sequence builders" begin
        ir = ir_sequence(100e-3)
        @test length(ir) == 4                   # 180, delay, 90, ADC
        se = se_sequence(50e-3)
        @test length(se) == 5                   # 90, delay, 180, delay, ADC
        @test_throws ErrorException se_sequence(1e-5)  # TE shorter than RF
    end

    @testset "single_spin_phantom" begin
        p = single_spin_phantom(T1 = 0.5, T2 = 0.1, ρ = 1.0)
        @test length(p.x) == 1
        @test p.T1 == [0.5]
        @test p.T2 == [0.1]
    end

    @testset "simulated single-sphere T1 and T2 recovery (subset)" begin
        # Pick a mid-range T1-array sphere and a mid-range T2-array sphere
        # and check we recover T1/T2 within 5 % through the full
        # simulate → fit pipeline. Covers the whole stack without spending
        # minutes on all 14 spheres.
        for field in (:T3, :T15)
            i = 5
            T1_tr  = T1_ARRAY[field][i]
            T2_tr1 = T2_OF_T1_ARRAY[field][i]
            r = measure_t1(T1_tr, T2_tr1)
            @test isapprox(r.T1_est, T1_tr; rtol = 0.05)

            j = 7
            T2_tr = T2_ARRAY[field][j]
            r = measure_t2(3.0, T2_tr)
            @test isapprox(r.T2_est, T2_tr; rtol = 0.05)
        end
    end

    @testset "run_e0 MAPE bound (3T, smoke)" begin
        # Full run over all 14 T1 + 14 T2 spheres at 3 T. This is the
        # simulator sanity check PLAN.md §4 E0 calls for.
        res = run_e0(; field = :T3, verbose = false)
        @test res.T1_MAPE_pct < 3.0
        @test res.T2_MAPE_pct < 3.0
    end
end

@testset "material tables" begin
    # Table lengths — all 14 contrast spheres per array
    @test length(T1_ARRAY[:T15]) == 14
    @test length(T1_ARRAY[:T3])  == 14
    @test length(T2_OF_T1_ARRAY[:T15]) == 14
    @test length(T2_OF_T1_ARRAY[:T3])  == 14
    @test length(T2_ARRAY[:T15]) == 14
    @test length(T2_ARRAY[:T3])  == 14
    @test length(PD_FRACTIONS)   == 14

    # Spot-check manual values (4 sig fig) at 3 T
    @test T1_ARRAY[:T3][1]         == 1.838
    @test T1_ARRAY[:T3][14]        == 0.02295
    @test T2_OF_T1_ARRAY[:T3][7]   == 0.1893
    @test T2_ARRAY[:T3][1]         == 2.756
    @test T2_ARRAY[:T3][14]        == 0.0869

    # And at 1.5 T
    @test T1_ARRAY[:T15][5]        == 0.527
    @test T2_ARRAY[:T15][10]       == 0.2610

    # PD is monotonically increasing; endpoints match manual table
    @test PD_FRACTIONS[1]  == 0.05
    @test PD_FRACTIONS[14] == 1.00
    @test issorted(PD_FRACTIONS)

    # T1-array values are monotonically decreasing (doping increases → T1 falls)
    @test issorted(T1_ARRAY[:T3];  rev = true)
    @test issorted(T1_ARRAY[:T15]; rev = true)
    @test issorted(T2_ARRAY[:T3];  rev = true)
    @test issorted(T2_ARRAY[:T15]; rev = true)

    # T2 < T1 on the T1-array (physical sanity)
    @test all(T2_OF_T1_ARRAY[:T3]  .< T1_ARRAY[:T3])
    @test all(T2_OF_T1_ARRAY[:T15] .< T1_ARRAY[:T15])

    # Background water sane
    @test BACKGROUND_WATER[:T3].T1  > 1.0
    @test BACKGROUND_WATER[:T15].T1 > 1.0
    @test BACKGROUND_WATER[:T3].ρ   == 1.0

    # Fiducial properties defined for both fields
    @test haskey(FIDUCIAL_PROPS, :T15)
    @test haskey(FIDUCIAL_PROPS, :T3)
    @test FIDUCIAL_PROPS[:T3].ρ == 1.0

    # Legacy T1 table is populated
    @test length(T1_ARRAY_LEGACY[:T3]) == 14
end

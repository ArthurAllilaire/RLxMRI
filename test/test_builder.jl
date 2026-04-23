_one_t1_descriptor() = sphere_descriptors(:T1, PhantomConfig())[1]

@testset "builder" begin
    cfg = PhantomConfig(voxel_size_mm = 3.0)   # coarse, fast

    @testset "sphere descriptors" begin
        t1s  = sphere_descriptors(:T1, cfg)
        t2s  = sphere_descriptors(:T2, cfg)
        pds  = sphere_descriptors(:PD, cfg)
        fids = sphere_descriptors(:fiducials, cfg)

        @test length(t1s)  == 14
        @test length(t2s)  == 14
        @test length(pds)  == 14
        @test length(fids) == 57
        @test length(t1s) + length(t2s) + length(pds) == 42

        # Labels are unique within each plate
        for descs in (t1s, t2s, pds, fids)
            @test length(unique(d -> d.label, descs)) == length(descs)
        end

        # Contrast sphere radii
        @test all(d -> d.radius == CONTRAST_RADIUS_M, t1s)
        @test all(d -> d.radius == CONTRAST_RADIUS_M, t2s)
        @test all(d -> d.radius == CONTRAST_RADIUS_M, pds)
        @test all(d -> d.radius == FIDUCIAL_RADIUS_M, fids)

        # T1 descriptor T1 values match manual @ 3T
        @test [d.T1 for d in t1s] == T1_ARRAY[:T3]

        # PD descriptor ρ values match H₂O fractions
        @test [d.ρ for d in pds] == PD_FRACTIONS

        @test_throws ErrorException sphere_descriptors(:bogus, cfg)
    end

    @testset "build_sphere produces a populated Phantom" begin
        d = _one_t1_descriptor()
        p = build_sphere(d, 1e-3)
        @test length(p.x) > 0
        @test length(p.x) == length(p.T1) == length(p.T2) == length(p.ρ)
        @test all(p.T1 .== d.T1)
        @test all(p.T2 .== d.T2)
    end

    @testset "build_plate concatenates sphere voxels" begin
        p = build_plate(:T1, cfg)
        @test length(p.x) > 0
        # 14 T1 spheres should produce spins spanning a range of T1 values
        @test length(unique(p.T1)) == 14
        # z positions all on the T1 plate, within a sphere-radius margin
        @test all(abs.(p.z .- PLATE_Z_MM.T1 * 1e-3) .<= CONTRAST_RADIUS_M + 1e-9)
    end

    @testset "build_phantom full assembly" begin
        obj = build_phantom(cfg)
        @test length(obj.x) > 0
        @test length(obj.x) == length(obj.T1) == length(obj.T2)
        # With water included, a lot more spins than plates-only
        obj_noWater = build_phantom(PhantomConfig(
            voxel_size_mm = 3.0,
            include_plates = [:T1, :T2, :PD, :fiducials]))
        @test length(obj.x) > length(obj_noWater.x)
        # Named
        @test obj.name == "qalibremd"
    end

    @testset "plate subset" begin
        cfg2 = PhantomConfig(voxel_size_mm = 3.0, include_plates = [:T1])
        obj = build_phantom(cfg2)
        # Only T1 array → unique T1 values is exactly 14
        @test length(unique(obj.T1)) == 14
    end

    @testset "field selection matters" begin
        cfgA = PhantomConfig(voxel_size_mm = 3.0, field = :T3,
                             include_plates = [:T1])
        cfgB = PhantomConfig(voxel_size_mm = 3.0, field = :T15,
                             include_plates = [:T1])
        A = build_phantom(cfgA)
        B = build_phantom(cfgB)
        @test sort(unique(A.T1)) != sort(unique(B.T1))
    end

    @testset "drop_sphere_p removes spheres" begin
        cfgD = PhantomConfig(
            voxel_size_mm = 3.0,
            include_plates = [:T1],
            augment = AugmentConfig(drop_sphere_p = 1.0))   # drop all
        descs = sphere_descriptors(:T1, cfgD)
        @test isempty(descs)
    end

    @testset "background water excludes contrast volumes" begin
        cfgW = PhantomConfig(voxel_size_mm = 3.0,
                             include_plates = [:T1, :T2, :PD, :fiducials, :water])
        water = build_background_water(cfgW)
        # No water voxel should lie strictly inside any contrast/fiducial sphere
        for d in all_sphere_descriptors(cfgW)
            dx = water.x .- d.centre[1]
            dy = water.y .- d.centre[2]
            dz = water.z .- d.centre[3]
            @test !any(@.(dx^2 + dy^2 + dz^2) .<= d.radius^2 - 1e-12)
        end
    end
end

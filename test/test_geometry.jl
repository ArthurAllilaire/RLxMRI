@testset "geometry primitives" begin
    @testset "voxelise_sphere volume" begin
        # At 1 mm voxels, voxelised volume should track (4/3)πr³ within 2 %
        # for a 15 mm diameter sphere (radius 7.5 mm).
        r = 7.5e-3
        delta_x = 1e-3
        xs, ys, zs = voxelise_sphere((0.0, 0.0, 0.0), r, delta_x)
        voxel_vol = length(xs) * delta_x^3
        @test isapprox(voxel_vol, sphere_volume(r); rtol = 0.02)

        # Coordinates are in metres and lie inside the sphere
        @test all(@.(xs^2 + ys^2 + zs^2) .<= r^2 + 1e-12)

        # Finer grid is more accurate
        delta_x2 = 0.5e-3
        xs2, _, _ = voxelise_sphere((0.0, 0.0, 0.0), r, delta_x2)
        voxel_vol2 = length(xs2) * delta_x2^3
        @test abs(voxel_vol2 - sphere_volume(r)) <
              abs(voxel_vol  - sphere_volume(r))

        # Non-centred sphere returns the same count (invariance under translation)
        xs3, _, _ = voxelise_sphere((0.1, -0.05, 0.03), r, delta_x)
        @test length(xs3) == length(xs)
    end

    @testset "contrast plate layout" begin
        for z in (56.5, 16.5, -23.5)
            cs = contrast_plate_centres(z)
            @test length(cs) == 14

            # All at the plate's z
            @test all(c -> isapprox(c[3], z * 1e-3), cs)

            # Outer ring radius 65 mm, inner ring 28 mm (first 10 outer, last 4 inner)
            rings = [sqrt(c[1]^2 + c[2]^2) for c in cs]
            @test all(isapprox.(rings[1:10],  65e-3))
            @test all(isapprox.(rings[11:14], 28e-3))
        end
    end

    @testset "fiducial grid" begin
        pts = fiducial_grid_centres()
        @test length(pts) == 57

        # Each point is within the clipping sphere
        rmax = 95e-3
        @test all(p -> p[1]^2 + p[2]^2 + p[3]^2 <= rmax^2 + 1e-12, pts)

        # 40 mm spacing — every coordinate is a multiple of 0.04 m
        for p in pts
            for ci in p
                @test isapprox(round(ci / 0.04) * 0.04, ci; atol = 1e-12)
            end
        end
    end
end

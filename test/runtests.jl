using Test
using Random
using QalibreMDPhantom
using KomaMRI

@testset "QalibreMDPhantom" begin
    include("test_materials.jl")
    include("test_geometry.jl")
    include("test_builder.jl")
    include("test_augment.jl")
    include("test_determinism.jl")
    include("test_simulation.jl")
    include("test_baseline.jl")
end

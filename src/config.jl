Base.@kwdef struct AugmentConfig
    T1_sigma_rel::Float64       = 0.0   # fractional Gaussian jitter on T1
    T2_sigma_rel::Float64       = 0.0   # fractional Gaussian jitter on T2
    PD_sigma_abs::Float64       = 0.0   # absolute Gaussian jitter on ρ
    position_sigma_mm::Float64  = 0.0   # per-spin position noise (mm)
    B0_sigma_Hz::Float64        = 0.0   # per-spin off-resonance stdev (Hz)
    drop_sphere_p::Float64      = 0.0   # Bernoulli prob of dropping a sphere
end

"""
    PhantomConfig

Single contract between the RL training loop and the builder. Everything the
builder needs to construct a phantom lives in this struct — there is no
hidden state.
"""
Base.@kwdef struct PhantomConfig
    field::Symbol                     = :T3            # :T15 or :T3
    voxel_size_mm::Float64            = 2.0            # isotropic voxel edge
    include_plates::Vector{Symbol}    = [:T1, :T2, :PD, :fiducials, :water]
    serial_number_class::Symbol       = :new           # :new (≥0042) or :legacy
    temperature_C::Float64            = 20.0
    rotation::NTuple{3,Float64}       = (0.0, 0.0, 0.0)   # Euler XYZ rad
    translation_mm::NTuple{3,Float64} = (0.0, 0.0, 0.0)
    augment::AugmentConfig            = AugmentConfig()
    rng_seed::Int                     = 0
    custom_sphere_map::Dict{Symbol,Any} = Dict{Symbol,Any}()
end

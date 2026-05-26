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
    field::Symbol                     = :T15           # :T15 or :T3
    voxel_size_mm::Float64            = 2.0            # isotropic voxel edge
    include_plates::Vector{Symbol}    = [:T1, :T2, :PD, :fiducials, :water]
    serial_number_class::Symbol       = :new           # :new (≥0042) or :legacy
    temperature_C::Float64            = 20.0
    rotation::NTuple{3,Float64}       = (0.0, 0.0, 0.0)   # Euler XYZ rad
    translation_mm::NTuple{3,Float64} = (0.0, 0.0, 0.0)
    augment::AugmentConfig            = AugmentConfig()
    rng_seed::Int                     = 0
    custom_sphere_map::Dict{Symbol,Any} = Dict{Symbol,Any}()
    keep_sphere_labels::Union{Nothing,Vector{Symbol}} = nothing
    drop_sphere_labels::Vector{Symbol} = Symbol[]
    custom_sphere_descriptors::Vector{SphereDescriptor} = SphereDescriptor[]
    # Z-slab mask: when slice_thickness_mm !== nothing, keep only spins with
    # |z_scanner − slice_center_mm·1e-3| ≤ slice_thickness_mm·1e-3 / 2. Applied
    # in scanner-frame, after pose transform. Set slice_center_mm to a plate's
    # PLATE_Z_MM entry (e.g. PLATE_Z_MM.T1 = 56.5) to isolate that plate.
    # If slice_thickness_mm < voxel_size_mm the slab may end up empty.
    slice_thickness_mm::Union{Nothing,Float64} = nothing
    slice_center_mm::Float64                   = 0.0
end

"""
    scanner_for_field(field::Symbol) -> Scanner
    scanner_for_field(cfg::PhantomConfig) -> Scanner

Return a `Scanner` whose `B0` matches the phantom field strength: 3.0 T for
`:T3`, 1.5 T for `:T15`. Use this instead of bare `Scanner()` whenever the
phantom was built with a `PhantomConfig` so that the Larmor frequency in the
Bloch simulation is consistent with the relaxation values.
"""
scanner_for_field(field::Symbol) = Scanner(B0 = field === :T3 ? 3.0 : 1.5)
scanner_for_field(cfg::PhantomConfig) = scanner_for_field(cfg.field)

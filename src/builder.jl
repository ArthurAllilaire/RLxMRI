"Fiducial sphere radius in metres (10 mm OD)."
const FIDUCIAL_RADIUS_M = 5.0e-3
"Contrast sphere radius in metres (15 mm OD)."
const CONTRAST_RADIUS_M = 7.5e-3
"Water housing sphere radius in metres (200 mm ID hemispheres joined)."
const HOUSING_RADIUS_M = 100e-3

"""
    with_sphere_relaxation(d, T1, T2; T2s=T2)

Return a copy of sphere descriptor `d` with updated relaxation values.
Geometry, density, off-resonance, and label are preserved.
"""
function with_sphere_relaxation(d::SphereDescriptor, T1::Real, T2::Real;
                                T2s::Real = T2)
    SphereDescriptor(
        d.centre, d.radius, d.ρ,
        Float64(T1), Float64(T2), Float64(T2s), d.delta_w, d.label,
    )
end

"""
    transform_descriptor(d, euler, translation)

Return a copy of sphere descriptor `d` whose centre has been rotated by
Euler angles `euler` and translated by `translation` (metres). Material
properties and radius are unchanged.
"""
function transform_descriptor(d::SphereDescriptor,
                              euler::NTuple{3,<:Real},
                              translation::NTuple{3,<:Real})
    transform_descriptor(d, rotation_matrix(euler...), translation)
end

function transform_descriptor(d::SphereDescriptor,
                              R::AbstractMatrix{<:Real},
                              translation::NTuple{3,<:Real})
    c_vec = R * collect(d.centre)
    c = (
        Float64(c_vec[1] + translation[1]),
        Float64(c_vec[2] + translation[2]),
        Float64(c_vec[3] + translation[3]),
    )
    SphereDescriptor(c, d.radius, d.ρ, d.T1, d.T2, d.T2s, d.delta_w, d.label)
end

function transform_descriptors(descs::AbstractVector{<:SphereDescriptor},
                               euler::NTuple{3,<:Real},
                               translation::NTuple{3,<:Real})
    [transform_descriptor(d, euler, translation) for d in descs]
end

function transform_descriptors(descs::AbstractVector{<:SphereDescriptor},
                               R::AbstractMatrix{<:Real},
                               translation::NTuple{3,<:Real})
    [transform_descriptor(d, R, translation) for d in descs]
end

function _t1_table(field::Symbol, class::Symbol)
    class === :legacy ? T1_ARRAY_LEGACY[field] : T1_ARRAY[field]
end

function _build_t1_descriptors(cfg::PhantomConfig)
    t1 = _t1_table(cfg.field, cfg.serial_number_class)
    t2 = T2_OF_T1_ARRAY[cfg.field]
    centres = contrast_plate_centres(PLATE_Z_MM.T1)
    descs = SphereDescriptor[]
    for i in eachindex(centres)
        push!(descs, SphereDescriptor(
            centres[i], CONTRAST_RADIUS_M, 1.0,
            t1[i], t2[i], t2[i], 0.0, Symbol("T1_$i")))
    end
    descs
end

function _build_t2_descriptors(cfg::PhantomConfig)
    t2 = T2_ARRAY[cfg.field]
    centres = contrast_plate_centres(PLATE_Z_MM.T2)
    descs = SphereDescriptor[]
    for i in eachindex(centres)
        push!(descs, SphereDescriptor(
            centres[i], CONTRAST_RADIUS_M, 1.0,
            T1_OF_T2_ARRAY[cfg.field][i], t2[i], t2[i], 0.0, Symbol("T2_$i")))
    end
    descs
end

function _build_pd_descriptors(cfg::PhantomConfig)
    centres = contrast_plate_centres(PLATE_Z_MM.PD)
    descs = SphereDescriptor[]
    for i in eachindex(centres)
        ρ = PD_FRACTIONS[i]
        T1 = pd_t1(ρ, cfg.field)
        T2 = pd_t2(ρ, cfg.field)
        push!(descs, SphereDescriptor(
            centres[i], CONTRAST_RADIUS_M, ρ,
            T1, T2, T2, 0.0, Symbol("PD_$i")))
    end
    descs
end

function _build_fiducial_descriptors(cfg::PhantomConfig)
    props = FIDUCIAL_PROPS[cfg.field]
    centres = fiducial_grid_centres()
    descs = SphereDescriptor[]
    for (i, c) in enumerate(centres)
        push!(descs, SphereDescriptor(
            c, FIDUCIAL_RADIUS_M, props.ρ,
            props.T1, props.T2, props.T2, 0.0, Symbol("fid_$i")))
    end
    descs
end

"""
    sphere_descriptors(plate, cfg)

Return the descriptor list for one plate, applying `drop_sphere_p` and any
entries from `cfg.custom_sphere_map` keyed on the sphere's label.
"""
function sphere_descriptors(plate::Symbol, cfg::PhantomConfig;
                            rng::AbstractRNG = Random.MersenneTwister(cfg.rng_seed))::Vector{SphereDescriptor}
    descs = plate === :T1        ? _build_t1_descriptors(cfg) :
            plate === :T2        ? _build_t2_descriptors(cfg) :
            plate === :PD        ? _build_pd_descriptors(cfg) :
            plate === :fiducials ? _build_fiducial_descriptors(cfg) :
            error("Unknown plate $plate")

    # Custom override per label
    if !isempty(cfg.custom_sphere_map)
        descs = map(descs) do d
            get(cfg.custom_sphere_map, d.label, d)
        end
    end

    if cfg.keep_sphere_labels !== nothing
        keep_labels = Set(cfg.keep_sphere_labels)
        descs = [d for d in descs if d.label in keep_labels]
    end

    if !isempty(cfg.drop_sphere_labels)
        drop_labels = Set(cfg.drop_sphere_labels)
        descs = [d for d in descs if !(d.label in drop_labels)]
    end

    # Sphere dropout (only if p > 0)
    p = cfg.augment.drop_sphere_p
    if p > 0
        keep = [rand(rng) > p for _ in descs]
        descs = descs[keep]
    end

    descs
end

function _generated_sphere_descriptors(cfg::PhantomConfig;
                                       rng::AbstractRNG = Random.MersenneTwister(cfg.rng_seed))
    descs = SphereDescriptor[]
    for plate in (:T1, :T2, :PD, :fiducials)
        plate ∈ cfg.include_plates || continue
        append!(descs, sphere_descriptors(plate, cfg; rng))
    end
    descs
end

"""
    all_sphere_descriptors(cfg)

Every contrast + fiducial sphere that the full phantom would contain.
Used by `build_background_water` to cut sphere volumes out of water.
"""
function all_sphere_descriptors(cfg::PhantomConfig;
                                rng::AbstractRNG = Random.MersenneTwister(cfg.rng_seed))
    descs = _generated_sphere_descriptors(cfg; rng)
    append!(descs, cfg.custom_sphere_descriptors)
    descs
end

"""
    sphere_descriptor_pixel(d, Npe, Nfe, FOV) -> (i_pe, i_fe)

Map a descriptor centre to the corresponding phase/frequency image pixel.
"""
function sphere_descriptor_pixel(d::SphereDescriptor, Npe::Int, Nfe::Int,
                                 FOV::Real)::NTuple{2,Int}
    (phys_to_pixel(d.centre[2], Npe, FOV),
     phys_to_pixel(d.centre[1], Nfe, FOV))
end

function sphere_descriptor_pixels(descs::AbstractVector{<:SphereDescriptor},
                                  Npe::Int, Nfe::Int, FOV::Real)
    [sphere_descriptor_pixel(d, Npe, Nfe, FOV) for d in descs]
end

function _empty_phantom(name::AbstractString = "empty")
    Phantom(name = String(name), x = Float64[])
end

function build_phantom_from_descriptors(descs::AbstractVector{<:SphereDescriptor},
                                        delta_x::Real;
                                        name::AbstractString = "spheres",
                                        z_range::Union{Nothing,Tuple{<:Real,<:Real}} = nothing)
    isempty(descs) && return _empty_phantom(name)
    parts = Phantom[]
    for d in descs
        p = build_sphere(d, delta_x; z_range = z_range)
        length(p.x) > 0 && push!(parts, p)
    end
    isempty(parts) && return _empty_phantom(name)
    obj = reduce(+, parts)
    obj.name = String(name)
    obj
end

function build_sphere(d::SphereDescriptor, delta_x::Real;
                      name::AbstractString = String(d.label),
                      z_range::Union{Nothing,Tuple{<:Real,<:Real}} = nothing)
    x, y, z = voxelise_sphere(d.centre, d.radius, delta_x; z_range = z_range)
    n = length(x)
    n == 0 && return _empty_phantom(name)
    Phantom(
        name = String(name),
        x = x, y = y, z = z,
        ρ   = fill(d.ρ,   n),
        T1  = fill(d.T1,  n),
        T2  = fill(d.T2,  n),
        T2s = fill(d.T2s, n),
        Δw  = fill(d.delta_w, n),
    )
end

"""
    build_plate(plate, cfg) -> Phantom

Voxelise the sphere descriptors of one plate and concatenate into a single
`Phantom`. Returns an empty phantom if the plate is empty after augmentation.
"""
function build_plate(plate::Symbol, cfg::PhantomConfig)
    rng = Random.MersenneTwister(cfg.rng_seed + hash(plate) % 10_000)
    descs = sphere_descriptors(plate, cfg; rng)
    delta_x = cfg.voxel_size_mm * 1e-3
    build_phantom_from_descriptors(descs, delta_x; name = String(plate))
end

"""
    build_background_water(cfg; cutout_descs=nothing) -> Phantom

Voxelise a 100 mm-radius water sphere and cut out all contrast + fiducial
sphere volumes. Spins are labelled as bulk water.
"""
function build_background_water(cfg::PhantomConfig; cutout_descs = nothing)
    delta_x = cfg.voxel_size_mm * 1e-3
    xs, ys, zs = voxelise_sphere((0.0, 0.0, 0.0), HOUSING_RADIUS_M, delta_x)
    isempty(xs) && return _empty_phantom("water")
    # Apply slice mask here — before allocating spin-property arrays — so that
    # the full-3D water sphere (up to 33M spins at 0.5 mm) is never fully
    # materialised in memory when a thin slice is requested.
    if cfg.slice_thickness_mm !== nothing
        z_half   = (cfg.slice_thickness_mm * 1e-3) / 2
        z_centre = cfg.slice_center_mm * 1e-3
        slice_ok = abs.(zs .- z_centre) .≤ z_half + 1e-9
        xs, ys, zs = xs[slice_ok], ys[slice_ok], zs[slice_ok]
    end
    isempty(xs) && return _empty_phantom("water")
    keep = trues(length(xs))
    cutouts = cutout_descs === nothing ? all_sphere_descriptors(cfg) : cutout_descs
    for d in cutouts
        @. keep &= ((xs - d.centre[1])^2 + (ys - d.centre[2])^2 +
                    (zs - d.centre[3])^2) > d.radius^2
    end
    sum(keep) == 0 && return _empty_phantom("water")
    props = BACKGROUND_WATER[cfg.field]
    n = sum(keep)
    Phantom(
        name = "water",
        x = xs[keep], y = ys[keep], z = zs[keep],
        ρ   = fill(props.ρ,  n),
        T1  = fill(props.T1, n),
        T2  = fill(props.T2, n),
        T2s = fill(props.T2, n),
        Δw  = fill(0.0,      n),
    )
end

"""
    build_phantom(cfg) -> Phantom

Assemble the full QalibreMD Model 130 digital twin from a `PhantomConfig`.
Pipeline: descriptors → voxelise → background water → rotate/translate →
per-spin noise. The returned `Phantom` can be fed directly to
`KomaMRI.simulate`.
"""
function build_phantom(cfg::PhantomConfig = PhantomConfig())
    rng = Random.MersenneTwister(cfg.rng_seed)
    sphere_descs = all_sphere_descriptors(cfg; rng)
    parts = Phantom[]
    if !isempty(sphere_descs)
        push!(parts, build_phantom_from_descriptors(
            sphere_descs, cfg.voxel_size_mm * 1e-3; name = "spheres"))
    end
    :water ∈ cfg.include_plates &&
        push!(parts, build_background_water(cfg; cutout_descs = sphere_descs))

    filter!(p -> length(p.x) > 0, parts)
    obj = isempty(parts) ? _empty_phantom("qalibremd") : reduce(+, parts)
    obj.name = "qalibremd"
    obj = apply_transform!(obj, cfg.rotation, cfg.translation_mm .* 1e-3)
    if cfg.slice_thickness_mm !== nothing && length(obj.x) > 0
        z_half   = (cfg.slice_thickness_mm * 1e-3) / 2
        z_centre = cfg.slice_center_mm * 1e-3
        # +1 nm float-tolerance so voxels landing exactly on the slab edge
        # (common when slice_thickness_mm == voxel_size_mm) survive roundoff.
        keep = abs.(obj.z .- z_centre) .≤ z_half + 1e-9
        obj = obj[keep]
        obj.name = "qalibremd"
    end
    obj = apply_per_spin_noise!(obj, cfg.augment, rng)
    obj
end

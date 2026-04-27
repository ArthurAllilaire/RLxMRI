struct SphereDescriptor
    centre::NTuple{3,Float64}   # m
    radius::Float64             # m
    ρ::Float64
    T1::Float64                 # s
    T2::Float64                 # s
    T2s::Float64                # s
    delta_w::Float64            # rad/s
    label::Symbol
end

"Fiducial sphere radius in metres (10 mm OD)."
const FIDUCIAL_RADIUS_M = 5.0e-3
"Contrast sphere radius in metres (15 mm OD)."
const CONTRAST_RADIUS_M = 7.5e-3
"Water housing sphere radius in metres (200 mm ID hemispheres joined)."
const HOUSING_RADIUS_M = 100e-3

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
                            rng::AbstractRNG = Random.MersenneTwister(cfg.rng_seed))
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

    # Sphere dropout (only if p > 0)
    p = cfg.augment.drop_sphere_p
    if p > 0
        keep = [rand(rng) > p for _ in descs]
        descs = descs[keep]
    end

    descs
end

"""
    all_sphere_descriptors(cfg)

Every contrast + fiducial sphere that the full phantom would contain.
Used by `build_background_water` to cut sphere volumes out of water.
"""
function all_sphere_descriptors(cfg::PhantomConfig)
    rng = Random.MersenneTwister(cfg.rng_seed)
    descs = SphereDescriptor[]
    for plate in (:T1, :T2, :PD, :fiducials)
        plate ∈ cfg.include_plates || continue
        append!(descs, sphere_descriptors(plate, cfg; rng))
    end
    descs
end

function _empty_phantom(name::AbstractString = "empty")
    Phantom(name = String(name), x = Float64[])
end

function build_sphere(d::SphereDescriptor, delta_x::Real;
                      name::AbstractString = String(d.label))
    x, y, z = voxelise_sphere(d.centre, d.radius, delta_x)
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
    isempty(descs) && return _empty_phantom(String(plate))
    parts = Phantom[]
    for d in descs
        p = build_sphere(d, delta_x)
        length(p.x) > 0 && push!(parts, p)
    end
    isempty(parts) && return _empty_phantom(String(plate))
    obj = reduce(+, parts)
    obj.name = String(plate)
    obj
end

"""
    build_background_water(cfg) -> Phantom

Voxelise a 100 mm-radius water sphere and cut out all contrast + fiducial
sphere volumes. Spins are labelled as bulk water.
"""
function build_background_water(cfg::PhantomConfig)
    delta_x = cfg.voxel_size_mm * 1e-3
    xs, ys, zs = voxelise_sphere((0.0, 0.0, 0.0), HOUSING_RADIUS_M, delta_x)
    isempty(xs) && return _empty_phantom("water")
    keep = trues(length(xs))
    for d in all_sphere_descriptors(cfg)
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
    parts = Phantom[]
    :fiducials ∈ cfg.include_plates && push!(parts, build_plate(:fiducials, cfg))
    :T1        ∈ cfg.include_plates && push!(parts, build_plate(:T1, cfg))
    :T2        ∈ cfg.include_plates && push!(parts, build_plate(:T2, cfg))
    :PD        ∈ cfg.include_plates && push!(parts, build_plate(:PD, cfg))
    :water     ∈ cfg.include_plates && push!(parts, build_background_water(cfg))

    filter!(p -> length(p.x) > 0, parts)
    obj = isempty(parts) ? _empty_phantom("qalibremd") : reduce(+, parts)
    obj.name = "qalibremd"
    obj = apply_transform!(obj, cfg.rotation, cfg.translation_mm .* 1e-3)
    obj = apply_per_spin_noise!(obj, cfg.augment, rng)
    obj
end

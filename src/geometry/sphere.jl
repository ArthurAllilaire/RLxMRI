"""
    voxelise_sphere(centre, radius, delta_x) -> (xs, ys, zs)

Return voxel-centre coordinates (in metres, `Vector{Float64}` each) that lie
inside a sphere of given `centre` (m) and `radius` (m), sampled on an
isotropic cubic lattice of spacing `delta_x` (m). The lattice for each sphere is
locally aligned to its centre; points from different spheres are not
guaranteed to share a global grid, but that is fine for a bag-of-spins
phantom.
"""
function voxelise_sphere(centre::NTuple{3,<:Real}, radius::Real, delta_x::Real;
                         z_range::Union{Nothing,Tuple{<:Real,<:Real}} = nothing)
    cx, cy, cz = Float64.(centre)
    r = Float64(radius)
    h = Float64(delta_x)
    xs = cx - r : h : cx + r
    ys = cy - r : h : cy + r
    zs = cz - r : h : cz + r
    # Optional z-slab restriction. Filter the full lattice (rather than starting
    # the range at z_range) so the kept voxels are *exactly* those a full
    # voxelisation followed by a post-hoc z-mask would keep — and we never
    # materialise the out-of-slab voxels.
    if z_range !== nothing
        zlo, zhi = Float64(z_range[1]), Float64(z_range[2])
        zs = filter(z -> zlo - 1e-9 <= z <= zhi + 1e-9, collect(zs))
    end
    X = [x for x in xs, _ in ys, _ in zs]
    Y = [y for _ in xs, y in ys, _ in zs]
    Z = [z for _ in xs, _ in ys, z in zs]
    mask = @. (X - cx)^2 + (Y - cy)^2 + (Z - cz)^2 <= r^2
    return X[mask], Y[mask], Z[mask]
end

"""
    sphere_volume(radius)

Analytic sphere volume (m³) — used by tests as a ground truth for the
voxelised count.
"""
sphere_volume(radius::Real) = (4/3) * π * radius^3

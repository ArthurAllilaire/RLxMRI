"""
    voxelise_sphere(centre, radius, delta_x) -> (xs, ys, zs)

Return voxel-centre coordinates (in metres, `Vector{Float64}` each) that lie
inside a sphere of given `centre` (m) and `radius` (m), sampled on an
isotropic cubic lattice of spacing `delta_x` (m). The lattice for each sphere is
locally aligned to its centre; points from different spheres are not
guaranteed to share a global grid, but that is fine for a bag-of-spins
phantom.
"""
function voxelise_sphere(centre::NTuple{3,<:Real}, radius::Real, delta_x::Real)
    cx, cy, cz = Float64.(centre)
    r = Float64(radius)
    h = Float64(delta_x)
    xs = cx - r : h : cx + r
    ys = cy - r : h : cy + r
    zs = cz - r : h : cz + r
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

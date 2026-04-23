"""
    voxelise_sphere(centre, radius, Δx) -> (xs, ys, zs)

Return voxel-centre coordinates (in metres, `Vector{Float64}` each) that lie
inside a sphere of given `centre` (m) and `radius` (m), sampled on an
isotropic cubic lattice of spacing `Δx` (m). The lattice for each sphere is
locally aligned to its centre; points from different spheres are not
guaranteed to share a global grid, but that is fine for a bag-of-spins
phantom.
"""
function voxelise_sphere(centre::NTuple{3,<:Real}, radius::Real, Δx::Real)
    cx, cy, cz = Float64.(centre)
    r = Float64(radius)
    h = Float64(Δx)
    xs = cx - r : h : cx + r
    ys = cy - r : h : cy + r
    zs = cz - r : h : cz + r
    X = reshape(collect(xs), :, 1, 1)
    Y = reshape(collect(ys), 1, :, 1)
    Z = reshape(collect(zs), 1, 1, :)
    mask = @. (X - cx)^2 + (Y - cy)^2 + (Z - cz)^2 <= r^2
    O = zero(X) .+ zero(Y) .+ zero(Z)
    X3 = X .+ O
    Y3 = Y .+ O
    Z3 = Z .+ O
    return X3[mask], Y3[mask], Z3[mask]
end

"""
    sphere_volume(radius)

Analytic sphere volume (m³) — used by tests as a ground truth for the
voxelised count.
"""
sphere_volume(radius::Real) = (4/3) * π * radius^3

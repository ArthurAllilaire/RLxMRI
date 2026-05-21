"""
    phantom_occupancy(phantom, Npe, Nfe, FOV) → Matrix{Float64}

Project phantom spin positions onto a 2-D image pixel grid and count spins per
pixel. Uses the same centred-indexing convention as `kspace_to_image`:
DC at array index (Npe÷2+1, Nfe÷2+1), positions wrap modulo the grid.

`phantom.x` maps to the frequency-encode (column) axis; `phantom.y` maps to
the phase-encode (row) axis.
"""
function phantom_occupancy(phantom, Npe::Int, Nfe::Int, FOV::Real)
    occ = zeros(Float64, Npe, Nfe)
    for k in eachindex(phantom.x)
        ife = mod(round(Int, phantom.x[k] * Nfe / FOV) + Nfe ÷ 2, Nfe) + 1
        ipe = mod(round(Int, phantom.y[k] * Npe / FOV) + Npe ÷ 2, Npe) + 1
        occ[ipe, ife] += 1.0
    end
    occ
end

"""
    phantom_occupancy(phantom, Npe, Nfe, Nsl, FOV_xy, FOV_z) → Array{Float64,3}

3-D overload. Projects phantom spins onto a (Npe × Nfe × Nsl) voxel grid.
`phantom.x` → frequency-encode, `phantom.y` → phase-encode,
`phantom.z` → slice axis. All positions wrap modulo their respective grid.
"""
function phantom_occupancy(phantom, Npe::Int, Nfe::Int, Nsl::Int,
                            FOV_xy::Real, FOV_z::Real)
    occ = zeros(Float64, Npe, Nfe, Nsl)
    for k in eachindex(phantom.x)
        ife = mod(round(Int, phantom.x[k] * Nfe / FOV_xy) + Nfe ÷ 2, Nfe) + 1
        ipe = mod(round(Int, phantom.y[k] * Npe / FOV_xy) + Npe ÷ 2, Npe) + 1
        isl = mod(round(Int, phantom.z[k] * Nsl / FOV_z) + Nsl ÷ 2, Nsl) + 1
        occ[ipe, ife, isl] += 1.0
    end
    occ
end

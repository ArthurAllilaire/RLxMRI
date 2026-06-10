# Exact spin counts for the E2 fidelity ladder.
#
# Builds the T1-plate slab exactly as the E2 env does
# (julia/rl/e2.jl `_e2_build_episode_phantom`: field T15, 1 mm voxels,
# 1 mm slab at PLATE_Z_MM.T1, B0σ=5 Hz) and reports how many spins each
# fidelity rung actually simulates per step:
#
#   full / full3  — full Bloch on spheres + water (1 mm / 3 mm water grid)
#   cached*       — only the spheres are Bloch-simulated; water is added from
#                   the cached template, so its Bloch cost is the spheres-only
#                   count regardless of water grid.
#
# Run: julia --project=. scripts/e2_spin_counts.jl
# (writes a small JSON next to the script so the report numbers can be rerun.)

using MRISystemPhantom
using Printf
using JSON

const FIELD          = :T15
const SPHERE_VOXEL   = 1.0   # mm
const SLICE_MM       = 1.0   # mm (single axial slab through the T1 plate)

"""Spin count of the T1-plate slab at the given water grid.
`water_voxel_mm === nothing` reproduces the env's full 1 mm water; pass a
value to coarsen. `include_water=false` gives the spheres-only count that the
cached rungs Bloch-simulate."""
function slab_spins(; water_voxel_mm::Union{Nothing,Float64}, include_water::Bool)
    cfg = PhantomConfig(
        field               = FIELD,
        voxel_size_mm       = SPHERE_VOXEL,
        water_voxel_size_mm = water_voxel_mm,
        include_plates      = include_water ? [:T1, :water] : [:T1],
        augment             = AugmentConfig(B0_sigma_Hz = 5.0),
        slice_thickness_mm  = SLICE_MM,
        slice_center_mm     = (0.0, 0.0, PLATE_Z_MM.T1),
    )
    length(build_phantom(cfg).x)
end

spheres_only = slab_spins(water_voxel_mm = nothing, include_water = false)
full_1mm     = slab_spins(water_voxel_mm = nothing, include_water = true)
full_2mm     = slab_spins(water_voxel_mm = 2.0,     include_water = true)
full_3mm     = slab_spins(water_voxel_mm = 3.0,     include_water = true)

rows = [
    ("spheres only (cached rungs Bloch-sim this)", spheres_only),
    ("full, 1 mm water  (`full` rung)",            full_1mm),
    ("full, 2 mm water",                           full_2mm),
    ("full, 3 mm water  (`full3` rung)",           full_3mm),
]

println("\nE2 spin counts — field $FIELD, $(SPHERE_VOXEL) mm spheres, $(SLICE_MM) mm slab\n")
for (name, n) in rows
    frac = full_1mm == 0 ? 0.0 : 100 * (n - spheres_only) / full_1mm
    @printf("  %-44s %8d   water %4.0f%%\n", name, n, frac)
end
water_1mm = full_1mm - spheres_only
@printf("\n  water is %.1f%% of the full 1 mm phantom (%d / %d spins)\n",
        100 * water_1mm / full_1mm, water_1mm, full_1mm)
@printf("  3 mm water grid reduces water spins by %.1fx (%d -> %d)\n",
        water_1mm / (full_3mm - spheres_only), water_1mm, full_3mm - spheres_only)

out = Dict(
    "field" => String(FIELD), "sphere_voxel_mm" => SPHERE_VOXEL, "slice_mm" => SLICE_MM,
    "spheres_only" => spheres_only, "full_1mm" => full_1mm,
    "full_2mm" => full_2mm, "full_3mm" => full_3mm,
    "water_1mm" => water_1mm, "water_frac_pct" => 100 * water_1mm / full_1mm,
)
open(joinpath(@__DIR__, "e2_spin_counts.json"), "w") do io
    JSON.print(io, out, 2)
end
println("\nwrote ", joinpath(@__DIR__, "e2_spin_counts.json"))

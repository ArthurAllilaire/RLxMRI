# Ground-truth slab maps for Fig 1(B), read by fig1.py.
#   B1: the T1-plate slab of three phantoms drawn by the training environment
#       (E2Env + e2_reset! with the TI-histogram run's base_env_kwargs).
#   B2: the nominal twin (1.5 T) sliced through the T1, T2 and PD plates.
#
# Run from the RLxMRI root:
#   julia --project=. ismrm_latex/vancouver/scripts/fig1_slabs.jl

using MRISystemPhantom
using JSON, NPZ

const ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
include(joinpath(ROOT, "julia", "rl_boot.jl"))

const RUN_DIR = joinpath(ROOT, "runs", "e2", "mf_runB_5sphere_hist_560s_gpu")
const FIG_DIR = normpath(joinpath(@__DIR__, "..", "figures"))
const SEEDS = [0, 1, 2]
const PIXEL_M = 0.5e-3
# The housing is 100 mm in radius; fig1.py crops each panel to its own slab.
const HALF_FOV_M = 101e-3
# Spins sit on a 1 mm lattice, so every pixel inside the phantom has a spin within
# half a diagonal; a 1 mm reach covers that without bridging gaps between spins.
const REACH_M = 1.0e-3

"Keyword arguments for E2Env from a run's saved config, at full fidelity on CPU."
function env_kwargs(run_dir)
    saved = JSON.parsefile(joinpath(run_dir, "run_config.json"))["base_env_kwargs"]
    accepted = Set(Iterators.flatten(Base.kwarg_decl.(methods(E2Env))))
    dropped = sort([k for k in keys(saved) if Symbol(k) ∉ accepted])
    @info "base_env_kwargs handled by the Python wrapper only (action mapping)" dropped
    # The "full" rung's overrides (python/train_e2_mf.py FIDELITIES) equal the base here.
    saved["forward_model"] == "bloch" && saved["water_model"] == "bloch" &&
        saved["include_water"] || error("hist run base config is not the full rung")
    kw = Dict{Symbol,Any}()
    for (k, v) in saved
        Symbol(k) in accepted || continue
        kw[Symbol(k)] = v isa AbstractString ? Symbol(v) :
                        v isa AbstractVector ? Int.(v) : v
    end
    kw[:use_gpu] = false
    kw
end

"Index of the spin nearest to each pixel centre (0 = no spin within REACH_M)."
function nearest_spin_raster(x, y, grid)
    cell(v) = floor(Int, v / REACH_M)
    buckets = Dict{NTuple{2,Int},Vector{Int}}()
    for k in eachindex(x)
        push!(get!(buckets, (cell(x[k]), cell(y[k])), Int[]), k)
    end
    none = Int[]
    idx = zeros(Int, length(grid), length(grid))
    for (j, gx) in enumerate(grid), (i, gy) in enumerate(grid)
        best, bestd = 0, REACH_M^2
        for dx in -1:1, dy in -1:1
            for k in get(buckets, (cell(gx) + dx, cell(gy) + dy), none)
                d = (x[k] - gx)^2 + (y[k] - gy)^2
                d < bestd && ((best, bestd) = (k, d))
            end
        end
        idx[i, j] = best
    end
    idx
end

"Per-spin sphere membership: k for spins inside descs[k], 0 for water."
function sphere_membership(obj, descs)
    g = zeros(Int, length(obj.x))
    for (k, d) in enumerate(descs)
        cx, cy, cz = d.centre
        inside = @. (obj.x - cx)^2 + (obj.y - cy)^2 + (obj.z - cz)^2 <= d.radius^2 * (1 + 1e-9)
        g[inside] .= k
    end
    g
end

"Rasterise a slab phantom: group map (-1 empty, 0 water, k sphere) and per-pixel spin values."
function rasterise(obj, descs, props)
    grid = collect(-HALF_FOV_M:PIXEL_M:HALF_FOV_M)
    idx = nearest_spin_raster(obj.x, obj.y, grid)
    member = sphere_membership(obj, descs)
    group = Int8[k == 0 ? -1 : member[k] for k in idx]
    vals = Dict(String(p) => Float32[k == 0 ? NaN : getproperty(obj, p)[k] for k in idx]
                for p in props)
    grid, group, vals
end

sphere_meta(descs) = [Dict("label" => String(d.label),
                           "centre_mm" => collect(d.centre[1:2]) .* 1e3,
                           "radius_mm" => d.radius * 1e3,
                           "T1_s" => d.T1, "T2_s" => d.T2, "PD" => d.ρ) for d in descs]

arrays = Dict{String,Any}()
meta = Dict{String,Any}("pixel_mm" => PIXEL_M * 1e3, "run_dir" => relpath(RUN_DIR, ROOT))

# ── B1: training-distribution phantoms ─────────────────────────────────────
kw = env_kwargs(RUN_DIR)
env = E2Env(; kw...)
b1 = Any[]
for s in SEEDS
    e2_reset!(env; rng_seed = s)
    # Active spheres in the scanner frame, carrying this episode's sampled T1.
    posed = transform_descriptors(env.active_base_descs, env.episode_rotation,
                                  env.episode_translation_m)
    posed = [with_sphere_relaxation(d, t1, d.T2) for (d, t1) in zip(posed, env.T1_true)]
    grid, group, vals = rasterise(env.phantom, posed, (:T1,))
    arrays["b1_$(s)_group"] = group
    arrays["b1_$(s)_T1"] = vals["T1"]
    arrays["grid_mm"] = grid .* 1e3
    @info "B1 seed $s" env.sphere_indices env.T1_true env.episode_rotation env.episode_translation_m
    push!(b1, Dict("seed" => s,
                   "sphere_indices" => env.sphere_indices,
                   "T1_true_s" => env.T1_true,
                   "rotation_deg" => rad2deg(env.episode_rotation[3]),
                   "translation_mm" => collect(env.episode_translation_m[1:2]) .* 1e3,
                   "spheres" => sphere_meta(posed),
                   # All 14 plate positions under this pose; the unselected ones hold water.
                   "plate_positions" => sphere_meta(transform_descriptors(
                       env.base_descs_pool, env.episode_rotation, env.episode_translation_m))))
end
meta["b1"] = b1
meta["t1_sample_range_s"] = collect(extrema(T1_ARRAY[kw[:cfg_field]]))
meta["translation_sigma_mm"] = kw[:translation_sigma_mm]
meta["rotation_sigma_rad"] = kw[:rotation_sigma_rad]

# ── B2: nominal twin sliced through each contrast plate ────────────────────
b2 = Dict{String,Any}()
for plate in (:T1, :T2, :PD)
    cfg = PhantomConfig(field = kw[:cfg_field], voxel_size_mm = kw[:voxel_size_mm],
                        slice_thickness_mm = kw[:voxel_size_mm],
                        slice_center_mm = (0.0, 0.0, getproperty(PLATE_Z_MM, plate)))
    descs = sphere_descriptors(plate, cfg)
    # Each plate is shown in the parameter its array varies.
    prop = plate === :PD ? :ρ : plate
    _, group, vals = rasterise(build_phantom(cfg), descs, (prop,))
    arrays["b2_$(plate)_group"] = group
    arrays["b2_$(plate)_value"] = vals[String(prop)]
    b2[String(plate)] = Dict("z_mm" => getproperty(PLATE_Z_MM, plate),
                             "spheres" => sphere_meta(descs))
end
meta["b2"] = b2
meta["field"] = String(kw[:cfg_field])

mkpath(FIG_DIR)
npzwrite(joinpath(FIG_DIR, "fig1_slabs.npz"), arrays)
open(joinpath(FIG_DIR, "fig1_slabs.json"), "w") do io
    JSON.print(io, meta, 1)
end
@info "wrote" joinpath(FIG_DIR, "fig1_slabs.npz") joinpath(FIG_DIR, "fig1_slabs.json")

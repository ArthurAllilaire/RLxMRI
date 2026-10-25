# The digital twin for the ISMRM preview figure (preview_v2.py): fig1_twin3d_viewer.jl's phantom,
# traces, camera direction and T1 colour range, without the axes, legend and hover box, and
# zoomed in so the phantom fills a square render.
#
# Run from the RLxMRI root (after fig1_viewer_camera.py):
#   julia --project=. ismrm_latex/vancouver/scripts/preview_v2_twin3d.jl

using MRISystemPhantom
using JSON, Random
# PlotlyJS 0.18 (and PlotlyKaleido for static export) are in the project manifest via KomaMRI.
const PlotlyJS = Base.require(Base.PkgId(Base.UUID("f0f68f2c-4968-5e81-91da-67840de0976a"), "PlotlyJS"))

const FIG_DIR = normpath(joinpath(@__DIR__, "..", "figures"))
const CAMERA = JSON.parsefile(joinpath(FIG_DIR, "fig1_viewer_camera.json"))

# As in fig1_twin3d_viewer.jl.
const CFG = PhantomConfig(field = :T3, voxel_size_mm = 1.5, water_voxel_size_mm = 3.0)
const MAX_WATER_POINTS = 200_000
const MAX_SPHERE_POINTS = 400_000
const WATER_OPACITY = 0.08
const WATER_MARKER_PX = 1.5
const SPHERE_MARKER_PX = 2.5
const T1_RANGE_S = (0.0, 2.0)

const PAGE_PX = 800
const SCALE = 2
# The screenshot's eye direction, moved in until the phantom nearly fills the frame; with no
# tick labels to keep inside the scene, nothing else needs the room.
const EYE_ZOOM = 0.72
# Seen through the translucent water, its regular voxel lattice beats into moire stripes over
# the spheres and the phantom's rim; moving each water point randomly within its voxel breaks
# the lattice without changing the look of a uniform medium.
const WATER_JITTER_SEED = 1

subsample(idx, maxn) = length(idx) <= maxn ? idx :
    unique(idx[round.(Int, range(1, length(idx); length = maxn))])

function group_mask(obj, descs)
    m = falses(length(obj.x))
    for d in descs
        cx, cy, cz = d.centre
        @. m |= (obj.x - cx)^2 + (obj.y - cy)^2 + (obj.z - cz)^2 <= d.radius^2
    end
    m
end

obj = build_phantom(CFG)
groups = Vector{Int}[]
assigned = falses(length(obj.x))
for plate in (:T1, :T2, :PD, :fiducials)
    m = group_mask(obj, sphere_descriptors(plate, CFG)) .& .!assigned
    assigned .|= m
    push!(groups, subsample(findall(m), MAX_SPHERE_POINTS))
end
water = subsample(findall(.!assigned), MAX_WATER_POINTS)
rng = Xoshiro(WATER_JITTER_SEED)
jitter() = CFG.water_voxel_size_mm * 1e-3 .* (rand(rng, length(water)) .- 0.5)
water_xyz = (obj.x[water] .+ jitter(), obj.y[water] .+ jitter(), obj.z[water] .+ jitter())

# Marker sizes are screen px, so they follow the scene's on-screen size to keep the
# screenshot's look of overlapping spin dots.
marker_scale = PAGE_PX / (EYE_ZOOM * CAMERA["plot_area_css"]["height"])
markers((x, y, z); kw...) = PlotlyJS.scatter3d(
    x = x, y = y, z = z, mode = "markers", hoverinfo = "skip", showlegend = false,
    marker = PlotlyJS.attr(; kw...))
traces = [markers(water_xyz; size = WATER_MARKER_PX * marker_scale, color = "lightblue", opacity = WATER_OPACITY)]
for idx in groups
    push!(traces, markers((obj.x[idx], obj.y[idx], obj.z[idx]); size = SPHERE_MARKER_PX * marker_scale,
                          color = obj.T1[idx], coloraxis = "coloraxis"))
end

eye = EYE_ZOOM .* [CAMERA["eye"][c] for c in ("x", "y", "z")]
hidden = PlotlyJS.attr(visible = false)
layout = PlotlyJS.Layout(
    width = PAGE_PX, height = PAGE_PX, margin = PlotlyJS.attr(l = 0, r = 0, t = 0, b = 0),
    paper_bgcolor = "white",
    scene = PlotlyJS.attr(aspectmode = "data", xaxis = hidden, yaxis = hidden, zaxis = hidden,
                          camera = PlotlyJS.attr(eye = PlotlyJS.attr(x = eye[1], y = eye[2], z = eye[3]))),
    coloraxis = PlotlyJS.attr(colorscale = "Viridis", cmin = T1_RANGE_S[1], cmax = T1_RANGE_S[2],
                              showscale = false))

out = joinpath(FIG_DIR, "preview_v2_twin3d.png")
PlotlyJS.savefig(PlotlyJS.plot(traces, layout), out; width = PAGE_PX, height = PAGE_PX, scale = SCALE)
open(joinpath(FIG_DIR, "preview_v2_twin3d.json"), "w") do io
    JSON.print(io, Dict("field" => String(CFG.field), "t1_range_s" => collect(T1_RANGE_S)), 1)
end
@info "wrote" out

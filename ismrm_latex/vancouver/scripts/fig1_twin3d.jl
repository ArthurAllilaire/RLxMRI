# Static 3D render of the digital twin for Fig 1(A), composed into the figure by fig1.py.
#
# Starts from plot_phantom_html (MRISystemPhantom.jl/ext/MRISystemPhantomPlotlyJSExt.jl) but
# draws each sphere descriptor as a shaded surface instead of its voxelised spins: at
# abstract size the spin clouds of neighbouring spheres merge into flat blobs with no depth
# cue. Panel B shows the spin-level slabs. There is no dropdown, slider, hover text, title,
# legend or colourbar; fig1.py draws one T1 colourbar shared with panel B. Fiducials are
# left out so the three contrast plates read clearly.
#
# Run from the RLxMRI root:
#   julia --project=. ismrm_latex/vancouver/scripts/fig1_twin3d.jl

using MRISystemPhantom
using JSON
# PlotlyJS 0.18 (and PlotlyKaleido for static export) are in the project manifest via KomaMRI.
const PlotlyJS = Base.require(Base.PkgId(Base.UUID("f0f68f2c-4968-5e81-91da-67840de0976a"), "PlotlyJS"))

const FIG_DIR = normpath(joinpath(@__DIR__, "..", "figures"))

const FIELD = :T15
# Viridis range shared with panel B; fig1.py reads it from the sidecar JSON.
const T1_RANGE_S = (0.0, 2.0)
const WATER = "#7FB2E5"
const GREY_PLATE = "#B4B8BD"
const INK = "#222222"
const INK_SECONDARY = "#555555"
# Printed width of panel A; labels are sized so they print at LABEL_PT.
const PANEL_WIDTH_IN = 2.7
const LABEL_PT = 7.5
const WIDTH_PX = 700
const HEIGHT_PX = 600
const SCALE = 4
const EYE = (x = 1.0, y = -0.8, z = 0.6)
const LEADER_PX = 60
# Height of the housing point the water label points at, below the PD plate.
const WATER_LABEL_Z_MM = -60.0
const LIGHTING = PlotlyJS.attr(ambient = 0.7, diffuse = 0.45, specular = 0.08,
                               roughness = 0.7, fresnel = 0.05)

function uv_sphere(centre, radius; n = 28)
    θ = range(0, π; length = n)
    φ = range(0, 2π; length = 2n)
    x = [centre[1] + radius * sin(t) * cos(p) for t in θ, p in φ]
    y = [centre[2] + radius * sin(t) * sin(p) for t in θ, p in φ]
    z = [centre[3] + radius * cos(t) for t in θ, _ in φ]
    x .* 1e3, y .* 1e3, z .* 1e3
end

flat(colour) = [[0, colour], [1, colour]]

function sphere_trace(centre, radius; value = 0.0, colorscale = flat(GREY_PLATE), n = 28,
                      opacity = 1.0, lighting = LIGHTING)
    x, y, z = uv_sphere(centre, radius; n)
    PlotlyJS.surface(x = x, y = y, z = z, surfacecolor = fill(value, size(x)),
                     colorscale = colorscale, cmin = T1_RANGE_S[1], cmax = T1_RANGE_S[2],
                     showscale = false, opacity = opacity, hoverinfo = "skip",
                     lighting = lighting,
                     lightposition = PlotlyJS.attr(x = -1e5, y = -1e5, z = 2e5))
end

cfg = PhantomConfig(field = FIELD)
traces = [sphere_trace((0.0, 0.0, 0.0), HOUSING_RADIUS_M; colorscale = flat(WATER), n = 64,
                       opacity = 0.12,
                       lighting = PlotlyJS.attr(ambient = 1.0, diffuse = 0.0, specular = 0.0))]
for p in (:PD, :T2), d in sphere_descriptors(p, cfg)
    push!(traces, sphere_trace(d.centre, d.radius))
end
for d in sphere_descriptors(:T1, cfg)
    push!(traces, sphere_trace(d.centre, d.radius; value = d.T1, colorscale = "Viridis"))
end

# Labels hang off leaders from screen-right points: each plate's rightmost sphere, and the
# housing below the plates.
right = let r = (-EYE.y, EYE.x)
    r ./ hypot(r...)
end
leader_label(point_m, text; leader_px = LEADER_PX) = PlotlyJS.attr(
    x = point_m[1] * 1e3, y = point_m[2] * 1e3, z = point_m[3] * 1e3, text = text,
    showarrow = true, arrowhead = 0, arrowwidth = 1.6, arrowcolor = INK_SECONDARY,
    ax = leader_px, ay = 0, xanchor = "left",
    font = PlotlyJS.attr(family = "Arial", color = INK, size = LABEL_PT * WIDTH_PX / (PANEL_WIDTH_IN * 72)))
annotations = map([(:T1, "T<sub>1</sub> plate"), (:T2, "T<sub>2</sub> plate"), (:PD, "PD plate")]) do (p, text)
    d = argmax(d -> d.centre[1] * right[1] + d.centre[2] * right[2], sphere_descriptors(p, cfg))
    leader_label((d.centre[1] + right[1] * d.radius, d.centre[2] + right[2] * d.radius, d.centre[3]), text)
end
water_z = WATER_LABEL_Z_MM * 1e-3
water_r = sqrt(HOUSING_RADIUS_M^2 - water_z^2)
# This point lies inside the housing's silhouette, so its leader is longer to clear the edge.
push!(annotations, leader_label((right[1] * water_r, right[2] * water_r, water_z), "water";
                                leader_px = LEADER_PX + 14))

hidden_axis = PlotlyJS.attr(visible = false, showbackground = false)
layout = PlotlyJS.Layout(
    width = WIDTH_PX, height = HEIGHT_PX, showlegend = false, paper_bgcolor = "white",
    margin = PlotlyJS.attr(l = 0, r = 0, t = 0, b = 0),
    scene = PlotlyJS.attr(
        aspectmode = "data", xaxis = hidden_axis, yaxis = hidden_axis, zaxis = hidden_axis,
        domain = PlotlyJS.attr(x = [0.0, 0.8], y = [0.0, 1.0]),
        camera = PlotlyJS.attr(eye = EYE, up = PlotlyJS.attr(x = 0, y = 0, z = 1)),
        annotations = annotations))

mkpath(FIG_DIR)
out = joinpath(FIG_DIR, "fig1_twin3d.png")
PlotlyJS.savefig(PlotlyJS.plot(traces, layout), out;
                 width = WIDTH_PX, height = HEIGHT_PX, scale = SCALE)
open(joinpath(FIG_DIR, "fig1_twin3d.json"), "w") do io
    JSON.print(io, Dict("field" => String(FIELD), "t1_range_s" => collect(T1_RANGE_S),
                        "panel_width_in" => PANEL_WIDTH_IN), 1)
end
@info "wrote" out

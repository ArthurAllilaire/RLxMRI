# Static re-export of the report's 3D viewer screenshot for Fig 1(A)
# (report_latex/imgs/mri_system_phantom/T1_voxel_phantom.png), laid out for the abstract.
#
# Kept from the screenshot: the phantom and traces of examples/plot_phantom_3d.jl through
# plot_phantom_html (MRISystemPhantom.jl/ext/MRISystemPhantomPlotlyJSExt.jl), the camera
# (fitted by fig1_viewer_camera.py), the marker style, the legend and the hover box on the
# same T1-plate spin. Changed for print: all text is sized to print at LABEL_PT in a
# PANEL_WIDTH_IN-wide panel; the colour range is fig1.py's shared T1 range, so the viewer's
# own colourbar is dropped (fig1.py draws the shared one); the hover box sits beside the
# scene on a leader line instead of covering it; the dropdown, slider and hover spike lines
# are gone.
#
# Run from the RLxMRI root (after fig1_viewer_camera.py):
#   julia --project=. ismrm_latex/vancouver/scripts/fig1_twin3d_viewer.jl

using MRISystemPhantom
using JSON, LinearAlgebra, Printf
# PlotlyJS 0.18 (and PlotlyKaleido for static export) are in the project manifest via KomaMRI.
const PlotlyJS = Base.require(Base.PkgId(Base.UUID("f0f68f2c-4968-5e81-91da-67840de0976a"), "PlotlyJS"))
const ColorSchemes = Base.require(Base.PkgId(Base.UUID("35d6a980-a343-548e-a6ea-1d62b119f2f4"), "ColorSchemes"))

const FIG_DIR = normpath(joinpath(@__DIR__, "..", "figures"))
const CAMERA = JSON.parsefile(joinpath(FIG_DIR, "fig1_viewer_camera.json"))

# examples/plot_phantom_3d.jl and the plot_phantom_html defaults.
const CFG = PhantomConfig(field = :T3, voxel_size_mm = 1.5, water_voxel_size_mm = 3.0)
const MAX_WATER_POINTS = 200_000
const MAX_SPHERE_POINTS = 400_000
const WATER_OPACITY = 0.08
const WATER_MARKER_PX = 1.5
const SPHERE_MARKER_PX = 2.5
const PLATE_LABEL = Dict(:T1 => "T1 plate", :T2 => "T2 plate", :PD => "PD plate",
                         :fiducials => "fiducials")
# Shared with panel B; fig1.py checks it against its own range.
const T1_RANGE_S = (0.0, 2.0)

# Print geometry. fig1.py places this image at PANEL_WIDTH_IN, so one CSS px prints at
# PT_PER_PX points and every font is set in points.
const PANEL_WIDTH_IN = 2.7
const LABEL_PT = 7.5
const PAGE_WIDTH_PX = 800
const PT_PER_PX = PANEL_WIDTH_IN * 72 / PAGE_WIDTH_PX
px(pt) = pt / PT_PER_PX
const SCALE = 4
# The scene is drawn this tall (print inches); marker sizes scale with it so the spheres
# keep the screenshot's look of overlapping spin dots.
const SCENE_HEIGHT_IN = 1.93
# The eye sits this much further out than in the screenshot (same direction), so the
# enlarged tick labels at the front corner stay inside the scene instead of being cut.
const EYE_ZOOM = 1.1
# The scene is centred in its plot area but its labels reach further left than right, so
# the plot area is wider than the page's scene column and the hover box overlaps its empty
# right part.
const SCENE_CENTRE_X_IN = 0.98
const SCENE_TOP_IN = 0.0
const COLUMN_X_IN = 1.84
const GAP_IN = 0.06

# The spin the screenshot's hover box describes (x, y, z in mm, as printed in it).
const HOVER_MM = (-14.086, 66.319, 56.5)
const HOVER_PAD_PX = 3
# gl3d draws tick labels at a size that grows faster than tickfont.size (and depends on the
# axis-title font), so this was set by measuring rendered digit heights against the
# LABEL_PT legend text (both about 7.5 pt in print).
const TICK_FONT_PX = 19
const LEADER = "#555555"
const INK = "#2a3f5f"  # plotly's default text colour, as in the screenshot
const TICK_STEP_M = 0.1
# Where two axes meet at a box corner their end labels collide, so only one keeps it:
# z drops -0.1 (corner shared with y) and x drops 0.1 (corner shared with y).
const Z_TICKS_M = [0.0, 0.1]
const X_TICKS_M = [-0.1, 0.0]

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

# d3-format "g" keeps trailing zeros, which C's %g drops; '#' restores them.
d3g(v) = @sprintf("%#.5g", v)

"""
Screen position (px) of a data point in a plotly.js 2.35 gl3d scene, as modelled in
fig1_viewer_camera.py: aspectmode "data" scales every axis by the geometric mean of the
data extents, autorange pads each axis by 1/32 of its extent, and the view is
lookAt(eye, origin, +z) with fovy = pi/4 over a plot area of height h centred at (cx, cy).
"""
function scene_to_screen(p, lo, hi, eye, cx, cy, h)
    extent = hi .- lo
    scale = cbrt(prod(extent)) * (1 + 2 / 32)
    q = (p .- (lo .+ hi) ./ 2) ./ scale
    fwd = -eye ./ norm(eye)
    side = normalize(cross(fwd, [0.0, 0.0, 1.0]))
    up = cross(side, fwd)
    d = q .- eye
    depth = dot(fwd, d)
    f = 1 / tan(π / 8)
    (cx + f * dot(side, d) / depth * h / 2, cy - f * dot(up, d) / depth * h / 2)
end

obj = build_phantom(CFG)
groups = Tuple{Symbol,Vector{Int}}[]
assigned = falses(length(obj.x))
for plate in (:T1, :T2, :PD, :fiducials)
    m = group_mask(obj, sphere_descriptors(plate, CFG)) .& .!assigned
    assigned .|= m
    push!(groups, (plate, subsample(findall(m), MAX_SPHERE_POINTS)))
end
water = subsample(findall(.!assigned), MAX_WATER_POINTS)

scene_h = px(SCENE_HEIGHT_IN * 72)
marker_scale = scene_h / (EYE_ZOOM * CAMERA["plot_area_css"]["height"])
markers(idx, name; kw...) = PlotlyJS.scatter3d(
    x = obj.x[idx], y = obj.y[idx], z = obj.z[idx], mode = "markers", name = name,
    hoverinfo = "skip", marker = PlotlyJS.attr(; kw...))
traces = [markers(water, "water"; size = WATER_MARKER_PX * marker_scale, color = "lightblue",
                  opacity = WATER_OPACITY)]
for (plate, idx) in groups
    push!(traces, markers(idx, PLATE_LABEL[plate]; size = SPHERE_MARKER_PX * marker_scale,
                          color = obj.T1[idx], coloraxis = "coloraxis"))
end

t1_idx = last(groups[findfirst(g -> first(g) === :T1, groups)])
k = t1_idx[argmin([sum(abs2, 1e3 .* (obj.x[i], obj.y[i], obj.z[i]) .- HOVER_MM) for i in t1_idx])]
offset_mm = maximum(abs.(1e3 .* (obj.x[k], obj.y[k], obj.z[k]) .- HOVER_MM))
offset_mm < 1e-3 || error("no T1-plate spin at the screenshot's hover position (off by $offset_mm mm)")
hover_text = join([
    "<b>$(PLATE_LABEL[:T1])</b>",
    @sprintf("x: %.3f mm", 1e3 * obj.x[k]),
    @sprintf("y: %.3f mm", 1e3 * obj.y[k]),
    @sprintf("z: %.3f mm", 1e3 * obj.z[k]),
    "T1: $(d3g(obj.T1[k])) s", "T2: $(d3g(obj.T2[k])) s", "T2s: $(d3g(obj.T2s[k])) s",
    "ρ: $(d3g(obj.ρ[k]))", "Δw: $(d3g(obj.Δw[k])) Hz"], "<br>")
# plotly fills a hover box with the hovered marker's colour.
t = (obj.T1[k] - T1_RANGE_S[1]) / (T1_RANGE_S[2] - T1_RANGE_S[1])
c = ColorSchemes.get(ColorSchemes.viridis, clamp(t, 0, 1))
hover_bg = @sprintf("#%02X%02X%02X", round.(Int, 255 .* (c.r, c.g, c.b))...)

lo = [minimum(obj.x), minimum(obj.y), minimum(obj.z)]
hi = [maximum(obj.x), maximum(obj.y), maximum(obj.z)]
eye = EYE_ZOOM .* [CAMERA["eye"][c] for c in ("x", "y", "z")]
centre_x = px(SCENE_CENTRE_X_IN * 72)
scene_top = px(SCENE_TOP_IN * 72)
spin_xy = scene_to_screen([obj.x[k], obj.y[k], obj.z[k]], lo, hi, eye, centre_x,
                          scene_top + scene_h / 2, scene_h)

column_x = px(COLUMN_X_IN * 72)
gap = px(GAP_IN * 72)
font = PlotlyJS.attr(size = px(LABEL_PT), color = INK)
legend_rows = length(traces)
# plotly.js legend rows are about 1.3 lines tall plus the legend's own padding.
legend_h = legend_rows * 1.3 * px(LABEL_PT) + 2 * 10
box_top = legend_h + gap
hover_box = PlotlyJS.attr(
    x = obj.x[k], y = obj.y[k], z = obj.z[k], text = hover_text, align = "left",
    xanchor = "left", yanchor = "top", ax = column_x - spin_xy[1], ay = box_top - spin_xy[2],
    showarrow = true, arrowhead = 6, arrowsize = 1, arrowwidth = px(0.6), arrowcolor = LEADER,
    bgcolor = hover_bg, bordercolor = "white", borderwidth = 1, borderpad = HOVER_PAD_PX,
    font = PlotlyJS.attr(family = "Arial, sans-serif", size = px(LABEL_PT), color = "white"))
box_bottom = box_top + 9 * 1.3 * px(LABEL_PT) + 2 * (HOVER_PAD_PX + 1)

page_h = round(Int, max(scene_top + scene_h, box_bottom) + gap)
axis = PlotlyJS.attr(tick0 = 0, dtick = TICK_STEP_M,
                     tickfont = PlotlyJS.attr(size = TICK_FONT_PX, color = INK),
                     title = PlotlyJS.attr(font = PlotlyJS.attr(size = px(LABEL_PT + 0.5), color = INK)))
plot_w = 2 * centre_x
layout = PlotlyJS.Layout(
    width = PAGE_WIDTH_PX, height = page_h, font = font,
    margin = PlotlyJS.attr(l = 0, t = round(Int, scene_top), r = round(Int, PAGE_WIDTH_PX - plot_w),
                           b = round(Int, page_h - scene_top - scene_h)),
    scene = PlotlyJS.attr(aspectmode = "data", yaxis = axis,
                          xaxis = merge(axis, PlotlyJS.attr(tickvals = X_TICKS_M)),
                          zaxis = merge(axis, PlotlyJS.attr(tickvals = Z_TICKS_M)),
                          camera = PlotlyJS.attr(eye = PlotlyJS.attr(x = eye[1], y = eye[2], z = eye[3])),
                          annotations = [hover_box]),
    coloraxis = PlotlyJS.attr(colorscale = "Viridis", cmin = T1_RANGE_S[1], cmax = T1_RANGE_S[2],
                              showscale = false),
    legend = PlotlyJS.attr(xref = "container", yref = "container", x = column_x / PAGE_WIDTH_PX,
                           y = 1.0, xanchor = "left", yanchor = "top", font = font))

fig = PlotlyJS.plot(traces, layout)
out = joinpath(FIG_DIR, "fig1_twin3d_viewer.png")
PlotlyJS.savefig(fig, out; width = PAGE_WIDTH_PX, height = page_h, scale = SCALE)
open(joinpath(FIG_DIR, "fig1_twin3d_viewer.json"), "w") do io
    JSON.print(io, Dict("field" => String(CFG.field), "t1_range_s" => collect(T1_RANGE_S),
                        "panel_width_in" => PANEL_WIDTH_IN, "label_pt" => LABEL_PT), 1)
end
@info "wrote" out page_h spin_xy hover_bg

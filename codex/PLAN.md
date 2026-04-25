Yes — and the right way to do this for KomaMRI is **not** to hard-code one giant blob of coordinates.

For your use case, the best design is:

1. **an analytic phantom spec**  
   - shell, inserts, fiducials, wedges, resolution modules
   - named regions with geometry + MR properties

2. **a sampler**  
   - converts that spec into a KomaMRI-compatible spin cloud / phantom arrays

3. **augmentation layers**  
   - rotation, translation, sphere perturbation, Gaussian property noise
   - later: k-space noise, B0 offsets, temperature drift, different configs

That gives you a maintainable digital twin you can keep extending for RL.

---

# What the manual gives us reliably

From the pages you shared, the QalibreMD / NIST-ISMRM System Standard Model 130 includes:

- **outer spherical shell**
  - deionized-water filled
  - **200 mm inner diameter**

- **5 internal plates**

- **57 fiducial spheres**
  - sphere inner diameter: **10.0 ± 0.1 mm**
  - arranged on a **3D 40 ± 0.1 mm grid**

- arrays:
  - **Plate 5:** 14 T1 spheres + resolution inset
  - **Plate 4:** 14 T2 spheres + 2 resolution insets
  - **Plate 3:** 14 proton density spheres
  - **Plate 2:** slice profile wedges
  - **Plate 1:** fiducials

- sequence slice locations from origin:
  - **T1 plane:** `V = 56.5 mm`
  - **T2 plane:** `V = 16.5 mm`
  - **PD plane:** `V = -23.5 mm`
  - **resolution inset planes:** `V = 58 mm`, `29.5 mm`, `18 mm`
  - **section thickness inset:** `V = -60 mm`

- material tables:
  - T1 array values
  - T2 array values
  - PD water fractions
  - 1.5T and 3T values
  - serial-number-specific values for **SN 0042+**

---

# What is still ambiguous from the manual alone

This matters because you asked for a **properly researched** build.

The manual is enough to build a **very good digital twin**, but not a perfect CAD-faithful one unless you also have:

- vendor CAD/STL
- measured localizer DICOMs
- or a real scan from your own phantom for calibration

The main uncertainties from the manual alone are:

- exact 3D coordinates of every contrast sphere center
- exact plate mounting offsets in z
- exact geometry of the resolution inset “coffins”
- exact wedge body thickness / acrylic support placement
- exact shell wall thickness and any support hardware
- exact orientation conventions beyond the qualitative description

So the practical recommendation is:

## Phase 1
Build a **quantitative insert twin**
- shell
- background fill
- T1 / T2 / PD spheres
- key fiducials
- slice planes matching the manual

## Phase 2
Add:
- full fiducial plates
- wedges
- resolution insets

## Phase 3
Calibrate against real images if you want a truly high-fidelity twin

---

# How phantom design should be done for KomaMRI

Even without relying on exact API details, KomaMRI-style simulators generally want:

- positions of spins / isochromats
- proton density / magnetization
- T1
- T2 / T2*
- off-resonance / Δω
- possibly diffusion or other properties depending on the simulator path

So for KomaMRI, the clean architecture is:

## 1. Keep geometry analytic
Do **not** make the point cloud your source of truth.

Source of truth should be:
- spheres
- shell
- plates
- wedges
- insets
- transforms

## 2. Sample later
Then turn analytic regions into:
- coarse spin clouds for fast RL
- finer spin clouds for validation
- maybe voxel grids if needed

## 3. Separate augmentation from physics
Three distinct augmentation types:

### Geometry augmentation
- rotate phantom
- translate phantom
- jitter sphere centers
- change sphere radii slightly
- swap or randomize sphere layouts

### Material augmentation
- add noise to T1/T2/ρ
- apply global temperature correction
- global scale factors
- B0 offset fields

### Acquisition augmentation
- add **complex Gaussian noise in raw signal / k-space**
- not image-domain Gaussian if you want MRI-realistic magnitude noise
- vary coil sensitivities later if needed

That separation is ideal for RL.

---

# Recommended package structure

I would build this as a small Julia package, not a single notebook file.

```text
QalibreMD130/
  Project.toml
  src/
    QalibreMD130.jl
    types.jl
    materials.jl
    geometry.jl
    layout.jl
    builders.jl
    sampling.jl
    augment.jl
    koma_adapter.jl
  test/
    runtests.jl
```

---

# Core design

Below is the structure I recommend.

## Types

```julia
module QalibreMD130

using StaticArrays, Random

export MRProps, SphereRegion, PhantomSpec,
       build_qalibre130, augment, sample_spins, to_koma_arrays

struct MRProps{T}
    ρ::T          # relative proton density
    T1_ms::T
    T2_ms::T
    T2s_ms::T
    Δf_Hz::T
end

struct SphereRegion{T}
    id::Symbol
    center_mm::SVector{3,T}
    radius_mm::T
    props::MRProps{T}
    group::Symbol         # :t1, :t2, :pd, :fiducial, :fill, :wedge, ...
end

struct PhantomSpec{T}
    inner_radius_mm::T
    regions::Vector{SphereRegion{T}}
    meta::Dict{Symbol,Any}
end

end
```

---

# Material tables

Use the manual’s **SN 0042+** table by default, since it’s the most standardized one in the document.

## `materials.jl`

```julia
module Materials

export material_table, pd_fractions

const T1_ARRAY_1P5T = (
    T1_ms = [1879.0, 1432.0, 1027.0, 751.3, 527.0, 384.1, 272.3, 194.5,
             137.8, 94.7, 67.0, 48.14, 34.35, 24.16],
    T2_ms = [1542.0, 1196.0, 871.7, 646.1, 457.5, 335.3, 238.4, 170.6,
             121.6, 83.7, 59.1, 42.6, 30.4, 21.3]
)

const T1_ARRAY_3T = (
    T1_ms = [1838.0, 1398.0, 998.3, 725.8, 509.1, 367.0, 258.7, 184.7,
             130.8, 90.9, 64.2, 46.28, 32.65, 22.95],
    T2_ms = [1354.0, 1035.0, 728.3, 524.4, 368.6, 266.7, 189.3, 134.1,
             93.8, 65.7, 46.8, 33.11, 23.69, 16.73]
)

const T2_ARRAY_1P5T = (
    T1_ms = [2640.0, 2292.0, 1923.0, 1489.0, 1245.0, 1004.0, 733.9, 533.1,
             400.3, 261.0, 189.8, 154.7, 102.1, 79.65],
    T2_ms = [1044.0, 623.9, 428.3, 258.4, 186.1, 137.0, 89.52, 62.82,
             43.84, 27.28, 19.24, 15.44, 10.05, 7.79]
)

const T2_ARRAY_3T = (
    T1_ms = [2756.0, 2281.0, 1961.0, 1552.0, 1341.0, 1017.0, 782.1, 589.7,
             443.8, 299.8, 237.8, 170.5, 121.8, 86.9],
    T2_ms = [645.8, 423.6, 286.0, 184.8, 134.1, 94.40, 62.51, 44.98,
             30.95, 20.10, 15.40, 10.85, 7.59, 5.35]
)

const PD_FRACTIONS = [
    0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35,
    0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00
]

function pd_fractions()
    return PD_FRACTIONS
end

function material_table(field_T::Real)
    if isapprox(field_T, 1.5; atol=0.2)
        return (t1 = T1_ARRAY_1P5T, t2 = T2_ARRAY_1P5T, pd = PD_FRACTIONS)
    elseif isapprox(field_T, 3.0; atol=0.2)
        return (t1 = T1_ARRAY_3T, t2 = T2_ARRAY_3T, pd = PD_FRACTIONS)
    else
        error("Only 1.5T and 3T tables are defined from the manual.")
    end
end

end
```

---

# Geometry helpers

The fiducial plate counts strongly suggest grid-based layouts:

- 5-point cross
- 13-point grid-in-disk
- 21-point 5×5 minus 4 corners

That part is actually nicely reproducible.

## `geometry.jl`

```julia
module Geometry

using StaticArrays
export grid5_xy, grid13_xy, grid21_xy, ring_xy

const MM = 1.0

grid5_xy(spacing=40.0) = SVector{2,Float64}[
    SA[0.0, 0.0],
    SA[ spacing, 0.0],
    SA[-spacing, 0.0],
    SA[0.0,  spacing],
    SA[0.0, -spacing],
]

function grid13_xy(spacing=40.0)
    pts = SVector{2,Float64}[]
    for x in (-2, -1, 0, 1, 2), y in (-2, -1, 0, 1, 2)
        if x^2 + y^2 <= 4
            push!(pts, SA[x*spacing, y*spacing])
        end
    end
    return pts
end

function grid21_xy(spacing=40.0)
    pts = SVector{2,Float64}[]
    for x in (-2, -1, 0, 1, 2), y in (-2, -1, 0, 1, 2)
        if !(abs(x) == 2 && abs(y) == 2)
            push!(pts, SA[x*spacing, y*spacing])
        end
    end
    return pts
end

function ring_xy(n::Int, radius_mm::Real; start_deg=90.0, clockwise=true)
    sign = clockwise ? -1.0 : 1.0
    return [
        SA[
            radius_mm * cosd(start_deg + sign*(k-1)*(360/n)),
            radius_mm * sind(start_deg + sign*(k-1)*(360/n))
        ]
        for k in 1:n
    ]
end

end
```

---

# Important note on insert geometry

The manual is **clear on material values**, but **not precise enough for exact 3D sphere centers** of all contrast arrays.

So I recommend making layout configurable:

```julia
struct LayoutConfig{T}
    z_t1_mm::T
    z_t2_mm::T
    z_pd_mm::T
    sphere_diameter_mm::T

    t1_outer_radius_mm::T
    t1_inner_radius_mm::T
    t2_outer_radius_mm::T
    t2_inner_radius_mm::T

    pd_positions_xy_mm::Vector{SVector{2,T}}   # configurable, not hard-coded forever
end
```

Use manual-validated defaults for z:

- `z_t1 = 56.5`
- `z_t2 = 16.5`
- `z_pd = -23.5`

For T1/T2 xy positions, use a ring approximation first.  
For PD, I would keep positions configurable from day one.

That is much safer than pretending the manual gave exact CAD coordinates.

---

# Builder

## `builders.jl`

```julia
module Builders

using StaticArrays
using ..Materials: material_table
using ..Geometry: ring_xy
using ..QalibreMD130: MRProps, SphereRegion, PhantomSpec

export default_layout, build_qalibre130

struct LayoutConfig{T}
    z_t1_mm::T
    z_t2_mm::T
    z_pd_mm::T
    sphere_diameter_mm::T
    t1_outer_radius_mm::T
    t1_inner_radius_mm::T
    t2_outer_radius_mm::T
    t2_inner_radius_mm::T
    pd_positions_xy_mm::Vector{SVector{2,T}}
    include_fill::Bool
end

function default_layout(T=Float64)
    # PD positions are only approximate unless you calibrate them from real data/CAD.
    pd_xy = SVector{2,T}[
        SA[T(0),   T(48)],
        SA[T(-24), T(24)], SA[T(0), T(24)], SA[T(24), T(24)],
        SA[T(-48), T(0)],  SA[T(-16), T(0)], SA[T(16), T(0)], SA[T(48), T(0)],
        SA[T(-24), T(-24)], SA[T(0), T(-24)], SA[T(24), T(-24)],
        SA[T(-48), T(-48)], SA[T(0), T(-48)], SA[T(48), T(-48)],
    ]
    return LayoutConfig(
        T(56.5), T(16.5), T(-23.5),
        T(10.0),
        T(64.0), T(28.0),
        T(44.0), T(18.0),
        pd_xy,
        true
    )
end

function build_qalibre130(; field_T=3.0, layout=default_layout(), fill_props=nothing)
    mats = material_table(field_T)
    regions = SphereRegion{Float64}[]
    r = layout.sphere_diameter_mm / 2

    # Background fill
    if layout.include_fill
        fp = isnothing(fill_props) ?
            MRProps(1.0, 2500.0, 1200.0, 1200.0, 0.0) :
            fill_props
        push!(regions, SphereRegion(
            :fill, SA[0.0, 0.0, 0.0], 100.0, fp, :fill
        ))
    end

    # T1 spheres: 10 outer + 4 inner
    t1_xy = vcat(
        ring_xy(10, layout.t1_outer_radius_mm; start_deg=90.0, clockwise=true),
        ring_xy(4, layout.t1_inner_radius_mm; start_deg=45.0, clockwise=true),
    )
    for i in 1:14
        props = MRProps(
            1.0,
            mats.t1.T1_ms[i],
            mats.t1.T2_ms[i],
            mats.t1.T2_ms[i],
            0.0
        )
        c = SA[t1_xy[i][1], t1_xy[i][2], layout.z_t1_mm]
        push!(regions, SphereRegion(Symbol("T1_$i"), c, r, props, :t1))
    end

    # T2 spheres: 10 outer + 4 inner
    t2_xy = vcat(
        ring_xy(10, layout.t2_outer_radius_mm; start_deg=90.0, clockwise=true),
        ring_xy(4, layout.t2_inner_radius_mm; start_deg=45.0, clockwise=true),
    )
    for i in 1:14
        props = MRProps(
            1.0,
            mats.t2.T1_ms[i],
            mats.t2.T2_ms[i],
            mats.t2.T2_ms[i],
            0.0
        )
        c = SA[t2_xy[i][1], t2_xy[i][2], layout.z_t2_mm]
        push!(regions, SphereRegion(Symbol("T2_$i"), c, r, props, :t2))
    end

    # PD spheres
    for i in 1:14
        ρ = mats.pd[i]
        props = MRProps(ρ, 2500.0, 1200.0, 1200.0, 0.0)
        xy = layout.pd_positions_xy_mm[i]
        c = SA[xy[1], xy[2], layout.z_pd_mm]
        push!(regions, SphereRegion(Symbol("PD_$i"), c, r, props, :pd))
    end

    meta = Dict{Symbol,Any}(
        :field_T => field_T,
        :layout => layout,
        :source => "QalibreMD System Standard Model 130 manual",
        :notes => "Contrast values from SN0042+ tables; geometry partly approximate unless calibrated."
    )

    return PhantomSpec(100.0, regions, meta)
end

end
```

---

# Why this design is good for RL

Because it lets you generate families of phantoms, not just one.

Examples:
- rotate phantom by ±5°
- translate by a few mm
- vary proton density slightly
- perturb T1/T2 by Gaussian factors
- change PD sphere layout
- disable background fill for ablation
- swap 1.5T / 3T materials
- later add bubbles or fill inhomogeneity

---

# Augmentation layer

## `augment.jl`

```julia
module Augment

using StaticArrays, Random, LinearAlgebra
using ..QalibreMD130: MRProps, SphereRegion, PhantomSpec

export augment

function rotmat_xyz(rx_deg, ry_deg, rz_deg)
    cx, sx = cosd(rx_deg), sind(rx_deg)
    cy, sy = cosd(ry_deg), sind(ry_deg)
    cz, sz = cosd(rz_deg), sind(rz_deg)

    Rx = SA[1.0 0.0 0.0; 0.0 cx -sx; 0.0 sx cx]
    Ry = SA[cy 0.0 sy; 0.0 1.0 0.0; -sy 0.0 cy]
    Rz = SA[cz -sz 0.0; sz cz 0.0; 0.0 0.0 1.0]
    return Rz * Ry * Rx
end

function augment(spec::PhantomSpec;
                 rotation_deg=(0.0, 0.0, 0.0),
                 translation_mm=(0.0, 0.0, 0.0),
                 rho_sigma=0.0,
                 t1_rel_sigma=0.0,
                 t2_rel_sigma=0.0,
                 center_sigma_mm=0.0,
                 radius_sigma_mm=0.0,
                 rng=Random.default_rng())

    R = rotmat_xyz(rotation_deg...)
    t = SA[translation_mm...]

    new_regions = map(spec.regions) do reg
        c = R * reg.center_mm + t
        c += center_sigma_mm .* SA[randn(rng), randn(rng), randn(rng)]

        p = reg.props
        newp = MRProps(
            max(0.0, p.ρ + rho_sigma * randn(rng)),
            max(1e-6, p.T1_ms * (1 + t1_rel_sigma * randn(rng))),
            max(1e-6, p.T2_ms * (1 + t2_rel_sigma * randn(rng))),
            max(1e-6, p.T2s_ms * (1 + t2_rel_sigma * randn(rng))),
            p.Δf_Hz
        )

        newr = max(0.1, reg.radius_mm + radius_sigma_mm * randn(rng))
        SphereRegion(reg.id, c, newr, newp, reg.group)
    end

    return PhantomSpec(spec.inner_radius_mm, collect(new_regions), copy(spec.meta))
end

end
```

---

# Sampling to a spin cloud

For KomaMRI, this is the key bridge.

You want two modes:

- **fast**: low-density sampling for RL loops
- **reference**: denser sampling for validation

## `sampling.jl`

```julia
module Sampling

using StaticArrays, Random
using ..QalibreMD130: PhantomSpec

export sample_spins

function points_in_sphere(center, radius_mm; spacing_mm=2.0)
    pts = SVector{3,Float64}[]
    r2 = radius_mm^2
    xs = collect(-radius_mm:spacing_mm:radius_mm)
    for x in xs, y in xs, z in xs
        if x*x + y*y + z*z <= r2
            push!(pts, center + SA[x, y, z])
        end
    end
    return pts
end

"""
Returns named vectors that are easy to adapt into KomaMRI:
x_mm, y_mm, z_mm, rho, T1_ms, T2_ms, T2s_ms, df_Hz, label
"""
function sample_spins(spec::PhantomSpec; spacing_mm=2.0, groups=(:t1, :t2, :pd))
    x = Float64[]
    y = Float64[]
    z = Float64[]
    rho = Float64[]
    T1 = Float64[]
    T2 = Float64[]
    T2s = Float64[]
    df = Float64[]
    label = Symbol[]

    for reg in spec.regions
        reg.group in groups || continue
        pts = points_in_sphere(reg.center_mm, reg.radius_mm; spacing_mm=spacing_mm)
        for p in pts
            push!(x, p[1]); push!(y, p[2]); push!(z, p[3])
            push!(rho, reg.props.ρ)
            push!(T1, reg.props.T1_ms)
            push!(T2, reg.props.T2_ms)
            push!(T2s, reg.props.T2s_ms)
            push!(df, reg.props.Δf_Hz)
            push!(label, reg.id)
        end
    end

    return (
        x_mm = x, y_mm = y, z_mm = z,
        rho = rho, T1_ms = T1, T2_ms = T2, T2s_ms = T2s, df_Hz = df,
        label = label
    )
end

end
```

---

# KomaMRI adapter

I’m being careful here: KomaMRI has evolved, and I don’t want to fake an exact constructor signature without your installed version.

So the safe, maintainable pattern is:

- keep a `to_koma_arrays(...)` adapter
- then a very thin version-specific wrapper in your local project

## `koma_adapter.jl`

```julia
module KomaAdapter

export to_koma_arrays

"""
Convert to SI-like arrays commonly needed by MRI simulators.
Check your installed KomaMRI constructor and plug these in.
"""
function to_koma_arrays(spins)
    return (
        x_m   = spins.x_mm .* 1e-3,
        y_m   = spins.y_mm .* 1e-3,
        z_m   = spins.z_mm .* 1e-3,
        rho   = spins.rho,
        T1_s  = spins.T1_ms .* 1e-3,
        T2_s  = spins.T2_ms .* 1e-3,
        T2s_s = spins.T2s_ms .* 1e-3,
        df_Hz = spins.df_Hz
    )
end

end
```

Then your local KomaMRI bridge is something like:

```julia
spec  = build_qalibre130(field_T=3.0)
spec2 = augment(spec; rotation_deg=(2.0, -1.0, 3.0), t1_rel_sigma=0.02)
spins = sample_spins(spec2; spacing_mm=2.0)
arrs  = to_koma_arrays(spins)

# Plug arrs into your installed KomaMRI phantom constructor here.
# Exact constructor depends on your KomaMRI version.
```

---

# Noise: do this the MRI-correct way

Since you mentioned Gaussian noise:

## Best practice
Add noise to the **complex simulated signal / k-space**, not to the final magnitude image.

If `s` is your complex raw data:

```julia
function add_complex_gaussian_noise(s, σ; rng=Random.default_rng())
    n = (randn(rng, size(s)) .+ 1im .* randn(rng, size(s))) .* (σ / sqrt(2))
    return s .+ n
end
```

Why?
- complex Gaussian noise in raw data leads to realistic magnitude noise after reconstruction
- direct magnitude-image Gaussian noise is not the same MRI noise model

For RL, this distinction usually matters.

---

# What I would model first

## Minimum useful twin
- T1 spheres
- T2 spheres
- PD spheres
- optional background fill
- global rotation / translation
- raw-signal Gaussian noise

This is enough for many RL experiments.

## Next
- fiducials
- 1.5T vs 3T modes
- serial-specific material tables
- temperature correction option

## Later
- wedges
- resolution insets
- fill bubbles
- B0 inhomogeneity
- imperfect slice profiles
- coil sensitivity variation

---

# Validation strategy

Before trusting it for RL, validate in layers:

## Geometry validation
- sphere count = 42 for T1/T2/PD arrays
- sphere diameter = 10 mm
- T1/T2/PD plane z-values match manual slice positions

## Contrast validation
- monotonic T1 ordering
- monotonic T2 ordering
- PD proportionality to H2O fraction

## Image validation
Compare simulated coronal slices to manual pages:
- T1 page
- T2 page
- PD page

You do not need perfect visual agreement on support plastic initially, but:
- ordering
- approximate spacing
- slice level
- signal ranking  
should all agree.

---

# One important recommendation

Because the manual alone leaves some geometry uncertain, I would explicitly store:

```julia
meta[:geometry_confidence] = Dict(
    :materials => :high,
    :sphere_diameter => :high,
    :slice_z_positions => :high,
    :fiducial_grid => :high,
    :contrast_xy_positions => :medium,
    :wedge_geometry => :medium,
    :resolution_insets => :medium_low
)
```

That makes the phantom honest and maintainable.

---

# My recommendation for your project

If you want this to survive iteration, do **not** build “the phantom” as one file.

Build:

- `build_qalibre130(...)`
- `augment(...)`
- `sample_spins(...)`
- `to_koma_arrays(...)`

That API will let you generate:
- baseline phantom
- rotated phantom
- noisy phantom
- different field strengths
- altered sphere configs
- later even domain-randomized training sets

---

# Best next step

If you want, I can do the next step in one of two ways:

## Option A — practical
I write you a **complete Julia mini-package** with:
- the module files above
- tests
- 1.5T/3T tables included
- a clean KomaMRI adapter stub

## Option B — exact-to-your-install
If you tell me:
- your KomaMRI version
- how you currently instantiate a phantom in KomaMRI
- whether you want point-cloud sampling or voxelized sampling

then I can write the adapter so it is **directly runnable** in your environment.

If you want, I can now produce:
1. a full `src/` tree, or  
2. a single-file prototype you can run immediately.
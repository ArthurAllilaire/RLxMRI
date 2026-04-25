# QalibreMDPhantom.jl

A programmatic, parameterised digital twin of the **QalibreMD NIST/ISMRM
System Standard Model 130** phantom for [KomaMRI.jl](https://github.com/JuliaHealth/KomaMRI.jl).
Single-call `build_phantom(cfg)` produces a `KomaMRI.Phantom` that can be
fed straight into `simulate(obj, seq, sys)`, with every knob an RL training
loop might want (field strength, voxel size, rotation, translation,
per-property jitter, sphere drop-out) controlled by one `PhantomConfig`.

```julia
using KomaMRI, QalibreMDPhantom
cfg = PhantomConfig(field = :T3, voxel_size_mm = 2.0)
obj = build_phantom(cfg)
raw = simulate(obj, seq, Scanner())
```

Design goals: **faithful** (values match the manual), **maintainable**
(tables in plain `.jl` data files, no magic numbers inside builders),
**flexible for RL** (single config contract), and **composable** (plates
can be built and simulated in isolation via `a + b`). The full design
document is in [`PLAN.md`](./PLAN.md).

---

## Install

```julia
] activate .
] instantiate
```

KomaMRI is a required dependency — first-time precompilation takes a few
minutes.

---

## Quick start

```julia
using KomaMRI, QalibreMDPhantom

# Nominal 3 T twin, 2 mm isotropic voxels, everything included
obj = build_phantom(PhantomConfig())

# 1.5 T, only the T1 plate, rotated 10° about z, mild T1 jitter
cfg = PhantomConfig(
    field          = :T15,
    include_plates = [:T1],
    rotation       = (0.0, 0.0, deg2rad(10)),
    augment        = AugmentConfig(T1_sigma_rel = 0.02, B0_sigma_Hz = 3.0),
    rng_seed       = 42,
)
obj = build_phantom(cfg)
```

RL rollout pattern:

```julia
function sample_cfg(rng)
    PhantomConfig(
        field          = rand(rng, (:T15, :T3)),
        voxel_size_mm  = 2.0,
        rotation       = Tuple(rand(rng, 3) .* 2π),
        translation_mm = Tuple(randn(rng, 3) .* 5.0),
        augment        = AugmentConfig(
            T1_sigma_rel  = 0.03,
            T2_sigma_rel  = 0.03,
            PD_sigma_abs  = 0.01,
            B0_sigma_Hz   = 5.0,
            drop_sphere_p = 0.05,
        ),
        rng_seed = rand(rng, Int),
    )
end
```

---

## Specs extracted from the manual

Source: *QalibreMD System Standard Model 130 T1/T2/PD User Manual* (SN 0042+).
The manual PDF lives at the repo root.

### Overall geometry

| Quantity | Value |
|---|---|
| Housing | Two 200 mm **inner-diameter** hemispheres joined to a sphere |
| Fill | Deionised / distilled water |
| Internal scaffold | 5 rigid acrylic plates (Plates 1–5) on positioning rods |
| Origin | Centre of the central fiducial sphere ≈ geometric centre |
| Axes | `V` (Superior–Inferior) → z; coronal plate plane → x–y; notch marks S/I & R/L |
| Temperature | Nominal values at **20 °C**; T1 ≈ 0.4 %/°C (NiCl₂), T2 ≈ 0.9 %/°C (MnCl₂) |

### Plates

| Plate | z [mm]  | Contents |
|------:|--------:|----------|
| 1 | —       | 5 fiducial spheres (blue) |
| 2 | −60     | 13 fiducial spheres + slice-profile wedges (green) |
| 3 | **−23.5** | 21 fiducial spheres + **14 PD spheres** (yellow, H₂O/D₂O) |
| 4 | **+16.5** | 13 fiducial spheres + **14 T2 spheres** (red, MnCl₂) + resolution insets |
| 5 | **+56.5** | 4 fiducial spheres + **14 T1 spheres** (blue-green, NiCl₂) + resolution inset |

**Note on the per-plate fiducial counts.** The manual's table above lists
`5, 13, 21, 13, 4 = 56` fiducials by plate, yet elsewhere it states 57
total. This is a book-keeping artefact, not a geometric disagreement.
The manual (§2.3) is explicit that the 57 fiducials sit on a **40 mm
3-D cubic grid** clipped to the ~95 mm housing radius. The plate
z-positions `{−60, −23.5, +16.5, +56.5}` mm are *not* on that lattice
(the grid's z-slices are at `{−80, −40, 0, +40, +80}` mm), so the
manual's per-plate breakdown reflects which support plate each fiducial
is mechanically attached to — not a geometric fact about the lattice.

Our code models the geometric lattice directly. Enumerating integer
offsets `(i, j, k) ∈ {−2, ±1, 0} ^3` subject to `i² + j² + k² ≤ 5`
(the clip) gives exactly `5, 13, 21, 13, 5 = 57` points per z-slice —
symmetric about the equator, as expected from a cubic lattice clipped
to a sphere. `fiducial_grid_centres()` in
`src/geometry/plate_layouts.jl` produces precisely this set, and
`test/test_geometry.jl` asserts both the total (57) and that every
point lies inside the 95 mm clip. Treat the manual's "6 and 4" pole
counts as mounting-plate labelling; the geometry is 5 / 13 / 21 / 13 / 5.

Additional resolution-inset coffins at V = +19 mm and V = +58 mm are
omitted from v1 (small features, mostly for geometric-distortion assessment).

### Sphere dimensions

| Type | OD | r (m) |
|---|---|---|
| Contrast (T1 / T2 / PD) | **15 ± 0.1 mm** | 7.5e-3 |
| Fiducial | **10 ± 0.1 mm** | 5.0e-3 |
| Fiducial grid | 40 ± 0.1 mm cubic spacing, 57 total inside r ≤ 95 mm | — |

In-plate layout: two concentric rings — outer 10 and inner 4 — at r_outer = 65 mm,
r_inner = 28 mm. Exact measured (x, y) per sphere is not tabulated in the
manual; `data/plate_layouts.jl` is the place to drop them in when they land.

### Material property tables — **SN ≥ 0042** (current shipping recipe)

**T1 array — NiCl₂ in water** (spheres T1-1 … T1-14):

| # | T1@1.5T [ms] | T2@1.5T [ms] | T1@3T [ms] | T2@3T [ms] |
|--:|--:|--:|--:|--:|
| 1  | 1879  | 1542  | 1838  | 1354  |
| 2  | 1432  | 1196  | 1362  | 1039  |
| 3  | 1027  | 871.7 | 998.3 | 718.3 |
| 4  | 751.3 | 646.1 | 725.8 | 524.4 |
| 5  | 527.0 | 457.5 | 509.1 | 368.6 |
| 6  | 384.1 | 335.3 | 367.0 | 266.7 |
| 7  | 272.3 | 238.4 | 258.7 | 189.3 |
| 8  | 194.5 | 170.6 | 184.7 | 134.1 |
| 9  | 137.8 | 121.6 | 130.8 | 93.8  |
| 10 | 94.7  | 83.7  | 90.9  | 65.7  |
| 11 | 67.0  | 59.2  | 64.2  | 46.8  |
| 12 | 48.14 | 42.6  | 46.28 | 33.15 |
| 13 | 34.35 | 30.4  | 32.65 | 23.69 |
| 14 | 24.16 | 21.3  | 22.95 | 16.73 |

SN 0001–0041 legacy T1 values are stored separately as `T1_ARRAY_LEGACY`.

**T2 array — MnCl₂ in water** (spheres T2-1 … T2-14):

| # | T2@1.5T [ms] | T2@3T [ms] |
|--:|--:|--:|
| 1  | 2640  | 2756  |
| 2  | 2292  | 2281  |
| 3  | 1923  | 1961  |
| 4  | 1609  | 1552  |
| 5  | 1245  | 1341  |
| 6  | 1004  | 1017  |
| 7  | 782.9 | 782.1 |
| 8  | 533.1 | 589.7 |
| 9  | 400.5 | 443.6 |
| 10 | 261.0 | 314.8 |
| 11 | 189.8 | 237.4 |
| 12 | 154.7 | 170.1 |
| 13 | 100.2 | 123.8 |
| 14 | 79.65 | 86.9  |

T1 of the MnCl₂ spheres is long (~water); v1 defaults to `T1 = 3.0 s`
(`T1_OF_T2_ARRAY_DEFAULT`), overrideable per sphere via
`PhantomConfig.custom_sphere_map`.

**PD array — H₂O / D₂O mixtures** (spheres PD-1 … PD-14):

| # | H₂O % | ρ    |
|--:|--:|--:|
| 1  | 5   | 0.05 |
| 2  | 10  | 0.10 |
| 3  | 15  | 0.15 |
| 4  | 20  | 0.20 |
| 5  | 25  | 0.25 |
| 6  | 30  | 0.30 |
| 7  | 35  | 0.35 |
| 8  | 40  | 0.40 |
| 9  | 50  | 0.50 |
| 10 | 60  | 0.60 |
| 11 | 70  | 0.70 |
| 12 | 80  | 0.80 |
| 13 | 90  | 0.90 |
| 14 | 100 | 1.00 |

T1 / T2 of the PD spheres are parameterised via `pd_t1(ρ, field)` /
`pd_t2(ρ, field)` so a future calibration can swap in a ρ-dependent model
without touching the builder. v1 uses bulk water T1 / T2 for all mixtures.

**Fiducial spheres** (aqueous CuSO₄): defaults `T1 ≈ 180 ms`, `T2 ≈ 120 ms`,
`ρ = 1.0` at both field strengths. Stored in `FIDUCIAL_PROPS[:T15]` /
`[:T3]` — override as calibration data arrives.

**Background water** (deionised):
`BACKGROUND_WATER[:T15] = (T1 = 2.8 s, T2 = 2.2 s, ρ = 1.0)`,
`BACKGROUND_WATER[:T3]  = (T1 = 3.0 s, T2 = 2.0 s, ρ = 1.0)`.

---

## Configuration

```julia
Base.@kwdef struct PhantomConfig
    field::Symbol                     = :T3              # :T15 or :T3
    voxel_size_mm::Float64            = 2.0              # isotropic voxel
    include_plates::Vector{Symbol}    = [:T1, :T2, :PD, :fiducials, :water]
    serial_number_class::Symbol       = :new             # :new or :legacy
    temperature_C::Float64            = 20.0
    rotation::NTuple{3,Float64}       = (0.0, 0.0, 0.0)  # Euler XYZ [rad]
    translation_mm::NTuple{3,Float64} = (0.0, 0.0, 0.0)
    augment::AugmentConfig            = AugmentConfig()
    rng_seed::Int                     = 0
    custom_sphere_map::Dict{Symbol,Any} = Dict{Symbol,Any}()
end

Base.@kwdef struct AugmentConfig
    T1_sigma_rel::Float64       = 0.0
    T2_sigma_rel::Float64       = 0.0
    PD_sigma_abs::Float64       = 0.0
    position_sigma_mm::Float64  = 0.0
    B0_sigma_Hz::Float64        = 0.0
    drop_sphere_p::Float64      = 0.0
end
```

---

## Tests

```julia
] test
```

The suite (373 tests) covers:

* Material-table values and table shapes (spot-checks against the manual).
* Geometry primitives: voxelised sphere volume vs `(4/3)πr³`; fiducial grid
  producing exactly 57 points at 40 mm / 95 mm clipping; plate-ring layout.
* Builder end-to-end: sphere counts (42 contrast + 57 fiducials), per-plate
  build, full-phantom assembly, background-water volume cut-outs.
* Augmentations: rotation orthogonality + `|r|` invariance, translation,
  Gaussian T1/T2 jitter stddev, PD clipping, B0 off-resonance.
* Determinism: same `rng_seed` → bit-for-bit identical phantom.
* Simulation smoke test: FID on a T1-array sphere; exponential fit
  recovers nominal T2 within 10 %.

---

## Baseline experiment (E0)

Non-RL sanity check — PLAN.md §4 E0. Runs inversion-recovery on each of
the 14 T1-array spheres and multi-TE spin-echo on each of the 14
T2-array spheres, then fits monoexponentials and compares against the
manual values. Target: MAPE < 3 %.

```bash
julia --project=. examples/e0_baseline.jl
```

Programmatic access:

```julia
using QalibreMDPhantom
res = run_e0(; field = :T3)
res.T1_MAPE_pct   # mean absolute percentage error on the 14 T1 spheres
res.T2_MAPE_pct   # on the 14 T2 spheres
```

Key primitives exposed for later RL use:

| Function | Purpose |
|----------|---------|
| `ir_sequence(TI)` / `se_sequence(TE)` | Pulseq-style RF-pulse blocks |
| `single_spin_phantom(; T1, T2)` | 0-D phantom — fast simulate target |
| `measure_ir_signal` / `measure_se_signal` | one TI/TE → one magnitude |
| `fit_t1_ir(TIs, mags)` / `fit_t2_se(TEs, mags)` | monoexponential fits |
| `adaptive_TI_schedule(T1_hint)` / `adaptive_TE_schedule(T2_hint)` | log-spaced sweeps sized to the target |
| `run_e0(; field)` | full 14 + 14 sphere sweep |

---

## E1 — single-sphere T1 estimation (RL)

First RL experiment on the ladder. One episode = one unknown (T1, T2)
sphere. Agent picks from a discrete set of IR-prep blocks; a Levenberg-like
running T1 fit is updated after every block. Reward = per-step parameter
error + terminal bonus for |err| < 3 % − a scan-time cost (PLAN.md §4 E1).

Anti-memorisation comes free via `E1Env`: each episode's T1 is drawn
log-uniformly from `[20 ms, 2 s]` and T2 = U(0.3, 1.0)·T1, so the agent
never sees the 14 manual values as a finite set.

### Action space (default)

| Action index | TI [ms] | α [°] |
|---|---|---|
| 1–3 | 10 | 10 / 90 / 180 |
| 4–6 | 30 | 10 / 90 / 180 |
| 7–9 | 100 | 10 / 90 / 180 |
| 10–12 | 300 | 10 / 90 / 180 |
| 13–15 | 1000 | 10 / 90 / 180 |
| 16–18 | 3000 | 10 / 90 / 180 |

α = 180° → canonical IR; α = 90° → saturation recovery; α = 10° →
small-tip prep (weak). All feed the same generalized-IR fit (see
`fit_t1_generalized_ir`) which handles mixed α natively.

### Training

Python side uses [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
PPO and Julia is called via [juliacall](https://juliapy.github.io/PythonCall.jl/).
One Julia runtime per Python worker, `:analytical` signal backend for the
training hot path (closed-form, orders of magnitude faster than
`simulate()`; PLAN.md §7).

```bash
# one-time: install Python deps
pip install --user juliacall gymnasium 'stable-baselines3[extra]'

# fixed 8-block IR grid baseline (no learning)
python python/baseline_e1.py --episodes 200

# train PPO; writes runs/e1/ppo/policy.zip + eval_history.json
python python/train_e1.py --timesteps 50000 --out runs/e1/ppo

# head-to-head on held-out seeds
python python/eval_e1.py --policy runs/e1/ppo/policy.zip --episodes 500
```

### Fidelity validation

The training backend is analytical; swap in the full KomaMRI solver for
a spot check (slow, do not use for training):

```python
env = QalibreMDE1Env(backend="simulate")
```

See `test/test_e1.jl` for an automated cross-check that the two backends
agree within the ~0.5 % RF-duration bias observed in E0.

---

## Visualising the twin

```bash
julia --project=. examples/plot_phantom.jl
```

writes interactive Plotly HTML files into `src/assets/`:

| File | Content |
|------|---------|
| `phantom_T1_3T_3d.html` | Full phantom, T1 map, 3 T (3D, ≤ 250 k spins) |
| `phantom_T2_3T_3d.html` | Full phantom, T2 map, 3 T (3D) |
| `phantom_ρ_3T_3d.html`  | Full phantom, ρ map, 3 T (3D) |
| `plate_T1_slice_3T.html` | T1 plate, 2D slab at z = +56.5 mm (ring layout) |
| `plate_T2_slice_3T.html` | T2 plate, 2D slab at z = +16.5 mm |
| `plate_PD_slice_3T.html` | PD plate, 2D slab at z = −23.5 mm, coloured by ρ |
| `fiducials_z0_slice.html` | Fiducial grid, 2D slab through z = 0 |
| `phantom_T1_augmented_3T.html` | 30° rotated + translated + jittered |

Resolution knobs in `examples/plot_phantom.jl`:

* `voxel_size_mm` in the `PhantomConfig` controls how densely each sphere
  is voxelised — 1.0 mm gives a crisp view, 2.0 mm is faster.
* `max_spins` kwarg on `plot_phantom_map` bumps the Plotly display cap
  (default 20 000 is sparse for the full phantom).
* `view_2d = true` renders a 2D x-y scatter; combined with a z-slab
  (see `slab(obj, z_mm)` helper) this gives the clearest view of each
  plate's ring layout for comparison with the manual photos.

Open them in a browser.

---

## Layout

```
src/
├── QalibreMDPhantom.jl    # top-level module; re-exports public API
├── config.jl              # PhantomConfig, AugmentConfig
├── builder.jl             # SphereDescriptor, build_sphere/plate/phantom
├── augment.jl             # rotate/translate + per-spin Gaussian jitter
├── materials/             # T1 / T2 / PD tables, fiducial + water constants
└── geometry/              # voxelise_sphere, contrast_plate_centres, fiducial_grid_centres

test/
├── runtests.jl
├── test_materials.jl
├── test_geometry.jl
├── test_builder.jl
├── test_augment.jl
├── test_determinism.jl
└── test_simulation.jl

examples/
└── plot_phantom.jl        # renders HTML phantom maps to src/assets/
```

---

## Known gaps

* **Exact in-plate sphere coordinates** — not in the manual; v1 uses a
  parametric two-ring layout that matches the photographs. Exact measured
  (x, y) per sphere will slot into `src/geometry/plate_layouts.jl`.
* **Fiducial CuSO₄ relaxation** — not tabulated in the manual excerpt;
  defaults are typical CuSO₄ values, overrideable via `FIDUCIAL_PROPS`.
* **Resolution-inset coffins** — omitted from v1 (mostly useful for
  geometric-distortion assessments, not T1/T2 RL).
* **Temperature correction** — the manual gives linear coefficients at
  1.5 T only. Planned as an optional post-processing step so the default
  at 20 °C stays a single source of truth.
* **Motion / flow** — not required by the RL task; future extension via
  `KomaMRI.Motion` / `FlowPath`.

---

## References

* QalibreMD System Standard Model 130 User Manual — PDF at repo root, SN 0042+.
* KomaMRI.jl docs —
  [Phantom API](https://juliahealth.org/KomaMRI.jl/stable/reference/2-koma-base/),
  [Create your own phantom](https://juliahealth.org/KomaMRI.jl/stable/how-to/3-create-your-own-phantom/).
* NIST/ISMRM system phantom characterisation — ISMRM 2012 abstract #2456.
* MRIStandards `SystemPhantom` repo (different phantom, design-decision inspiration):
  https://github.com/MRIStandards/SystemPhantom

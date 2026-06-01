# Coarser background-water voxelisation plan

## Context

In the E2 setup the phantom is sliced to a thin slab centred on the T1 plate.
At 1 mm resolution the background-water ball dominates the spin count: the
slice has far more water spins than contrast-sphere spins, so water dominates
Bloch simulation cost.

The goal is to keep contrast spheres fine, because contrast detail matters, but
coarsen the background water in-plane. Since the E2 slice is effectively 2D,
coarsening water by 3x in-plane should reduce water spin count by roughly 9x
while preserving the total water signal by reweighting each coarse spin.

This is Julia-only for now. Python/E2 environment wiring and policy checks can
come later.

## Semantics

The slice plane should be phantom-attached, not scanner-fixed.

Define `slice_normal` and `slice_center_mm` in phantom coordinates. Build and
clip the phantom against that plane before pose. Then apply the normal
`apply_transform!` rotation/translation to every surviving spin.

This matches the desired mental model:

- Add a slice plane to the phantom.
- Add a rotation/translation to the phantom.
- The returned spins lie on the rotated/translated slice plane.

Water is not transformed differently from spheres. Water is only sampled
differently:

- Spheres: fine 3D voxelisation, then signed-distance slice clip.
- Sliced water: 2D in-plane grid on the same phantom-frame slice plane.
  `water_voxel_size_mm` controls in-plane spacing; `nothing` uses
  `voxel_size_mm`. `water_throughplane_voxel_size_mm` controls sheet spacing;
  `nothing` uses a single weighted centre sheet.
- Unsliced water: 3D sphere voxelisation, optionally at `water_voxel_size_mm`.
- All surviving spins: same `apply_transform!`.

## Config changes

Add to `PhantomConfig` in `src/config.jl`:

```julia
water_voxel_size_mm::Union{Nothing,Float64} = nothing
water_throughplane_voxel_size_mm::Union{Nothing,Float64} = nothing
slice_center_mm::NTuple{3,Float64} = (0.0, 0.0, 0.0)
slice_normal::NTuple{3,Float64} = (0.0, 0.0, 1.0)
```

Interpretation:

- `water_voxel_size_mm === nothing`: use `voxel_size_mm` for water spacing.
  Unsliced water keeps current behaviour; sliced water still uses the new 2D
  plane representation.
- `water_voxel_size_mm !== nothing`: use the water-specific voxel size.
- `water_throughplane_voxel_size_mm === nothing`: use one weighted centre sheet
  for sliced water.
- `water_throughplane_voxel_size_mm !== nothing`: stack sliced-water sheets at
  that through-plane spacing.
- `slice_normal`: phantom-frame slice normal.
- `slice_center_mm`: phantom-frame point on the slice plane, in mm.
- `slice_thickness_mm`: slab thickness around the oriented plane, in mm.

With the default normal `(0, 0, 1)`,
`slice_center_mm = (0, 0, PLATE_Z_MM.T1)` remains the current T1-plate z slice.

Validation:

- `voxel_size_mm > 0`
- `water_voxel_size_mm > 0` when provided
- `water_throughplane_voxel_size_mm > 0` when provided
- `slice_normal` must have nonzero norm
- no upper or lower clamp on `water_voxel_size_mm`; water may be coarser or
  finer than the sphere voxel size

Documentation note:

- This is water-only in-plane coarsening.
- Coarse water `rho` is a quadrature weight, not just material proton density.
- Recommend `water_voxel_size_mm <= FOV / Npe` once E2/Python wiring is added.

## Geometry helpers

Add `src/geometry/plane.jl` and include it from `src/QalibreMDPhantom.jl`.

Suggested helpers:

```julia
slice_basis(normal) -> (n_hat, u_hat, v_hat)
signed_distance(x, y, z, centre, n_hat) -> distance
voxelise_plane(centre, n_hat, u_hat, v_hat, in_plane_dx, radius) -> (xs, ys, zs)
```

`slice_basis`:

- Normalise the normal.
- Error if the norm is zero or nearly zero.
- Pick an axis that is most orthogonal to the normal.
- Use cross products to build an orthonormal in-plane basis.

`signed_distance`:

- Compute `dot(point - centre, n_hat)`.
- Used for all oriented-slab clipping.

`voxelise_plane`:

- Generate a 2D lattice on the plane:

  ```julia
  p = centre + i * dx * u_hat + j * dx * v_hat
  ```

- Keep only points inside the housing sphere:

  ```julia
  dot(p, p) <= radius^2
  ```

### Plane footprint radius

The housing is a sphere of radius `R`. A plane through the origin intersects it
in a circle of radius `R`, but an offset plane intersects it in a smaller
circle.

For plane centre `p` and unit normal `n_hat`, the perpendicular distance from
the housing-sphere origin to the plane is:

```julia
d = dot(p, n_hat)
```

The intersection-circle radius is:

```julia
r_plane = sqrt(R^2 - d^2)
```

Cases:

- `abs(d) == 0`: full radius `R`
- `abs(d) < R`: circle radius `sqrt(R^2 - d^2)`
- `abs(d) == R`: tangent, radius zero
- `abs(d) > R`: no intersection, return empty arrays

Use `r_plane` to choose the 2D lattice bounds instead of always laying out
`[-R, R]^2`. For the T1 plane at 56.5 mm in a 100 mm housing sphere, the water
cross-section radius is about 82.5 mm, not 100 mm.

Still apply the final `dot(p, p) <= R^2` keep mask, because the square lattice
bounding box contains points outside the circular footprint.

## Builder changes

### Shared plane definition

Add a small internal helper in `src/builder.jl` or `plane.jl`:

```julia
slice_plane(cfg) -> (centre, n_hat, u_hat, v_hat)
```

For phantom-frame slicing:

```julia
n_hat, u_hat, v_hat = slice_basis(cfg.slice_normal)
centre = cfg.slice_center_mm .* 1e-3
```

Use this same helper for:

- coarse water plane generation
- fine/default water prefilter
- sphere/final slab clipping
- tests

### `build_background_water`

Compute:

```julia
sphere_dx = cfg.voxel_size_mm * 1e-3
water_dx_mm = cfg.water_voxel_size_mm === nothing ?
    cfg.voxel_size_mm : cfg.water_voxel_size_mm
water_dx = water_dx_mm * 1e-3
through_dx_mm = cfg.water_throughplane_voxel_size_mm === nothing ?
    cfg.slice_thickness_mm : cfg.water_throughplane_voxel_size_mm
```

Branches:

1. **Sliced water**

   Generate water as one or more phantom-frame slice-parallel planes with
   `voxelise_plane`, using `water_dx` in-plane. Sheet edges partition the slab
   thickness; each sheet is centred in its interval.

   ```julia
   ρ_weight = (water_dx / sphere_dx)^2 * (sheet_thickness / sphere_dx)
   ρ_eff = props.ρ * ρ_weight
   ```

   This preserves total water signal and can represent through-plane extent.
   With `water_throughplane_voxel_size_mm === nothing`, there is one sheet with
   `sheet_thickness = slice_thickness_m`.

2. **Unsliced water**

   Generate a 3D water ball with `voxelise_sphere(..., water_dx)`.

   ```julia
   ρ_weight = cfg.water_voxel_size_mm === nothing ? 1 :
       (water_dx / sphere_dx)^3
   ρ_eff = props.ρ * ρ_weight
   ```

After water generation:

- Apply sphere cutouts as today.
- Fill water `rho`/`ρ` with `ρ_eff`.
- Do not special-case water during pose.

### Slab thickness options

`slice_thickness_mm` can continue to mean a slab around the oriented plane:

```julia
abs(signed_distance(point, centre, n_hat)) <= slice_thickness_mm * 1e-3 / 2
```

For fine spheres and default/fine water, this is sufficient.

The config should remain in mm. If a caller wants a thickness in image pixels,
convert it before building the phantom, for example
`slice_thickness_mm = n_pixels * (FOV / Npe) * 1e3`.

For sliced water, `water_throughplane_voxel_size_mm === nothing` gives the cheap
single weighted plane. Setting it to `voxel_size_mm` gives stacked sheets at the
same through-plane resolution as the contrast sphere lattice.

### Generalised clips

Replace axis-aligned z-slab masks with signed-distance masks:

- the general `slab` option in `voxelise_sphere`
- final/global clip if it remains

Because the slice is phantom-attached, prefer clipping before pose. If the final
clip remains as a safety check, it must test against the transformed plane:

```julia
p_scanner = R * p_phantom + t
n_scanner = R * n_phantom
```

It should not be needed for coarse water correctness if all construction-time
clips use the phantom-frame plane.

## PD jitter interaction

Current `apply_per_spin_noise!` clamps proton density after PD jitter:

```julia
obj.ρ[i] = clamp(obj.ρ[i] + noise, 0.0, 1.0)
```

That is physically reasonable for material proton-density fractions, but coarse
water `rho`/`ρ` becomes a quadrature weight. For example, 3x in-plane
coarsening can give water `ρ_eff = 9`. Clamping that to 1 would destroy signal
conservation.

Implement the material-PD model immediately, rather than guarding or deferring
PD jitter support.

Keep two concepts separate:

- `ρ_material`: physical proton-density fraction, clamped to `[0, 1]`
- `ρ_weight`: quadrature weight from voxel coarsening

The stored KomaMRI spin `ρ` should be:

```julia
ρ = ρ_material * ρ_weight
```

For unweighted spins, `ρ_weight = 1`. For coarse water:

```julia
ρ_weight = (water_dx / sphere_dx)^2 * (sheet_thickness / sphere_dx)
                                            # sliced sheet water
ρ_weight = (water_dx / sphere_dx)^3        # unsliced 3D water
```

PD jitter should be applied as:

```julia
ρ_material = obj.ρ[i] / ρ_weight[i]
ρ_material = clamp(ρ_material + σ_PD * randn(rng), 0.0, 1.0)
obj.ρ[i] = ρ_material * ρ_weight[i]
```

Implementation options:

1. Add an optional `ρ_weight` vector to `apply_per_spin_noise!`, defaulting to
   all ones. The default path remains current behaviour.

   ```julia
   apply_per_spin_noise!(obj, aug, rng; ρ_weight = nothing)
   ```

2. In `build_phantom`, construct the matching `ρ_weight` sidecar while
   concatenating parts:

   - spheres: ones
   - default/fine water: ones
   - coarse water: fill with the water quadrature weight

3. Pass the sidecar into `apply_per_spin_noise!`.

This preserves the existing physical clamp while allowing coarse water spins to
carry weights greater than 1.

## Tests

Add or update tests in `test/test_geometry.jl` and `test/test_builder.jl`.

### Geometry

1. `slice_basis`

   - normal is unit length
   - basis vectors are mutually orthogonal
   - zero normal errors

2. `signed_distance`

   - axis-aligned plane reproduces `z - z0`
   - oblique points on the plane have near-zero distance

3. `voxelise_plane`

   - axis-aligned normal gives constant z equal to plane offset
   - footprint radius matches `sqrt(R^2 - d^2)`
   - plane outside housing sphere returns empty arrays
   - oblique normal returns planar points inside `|p| <= R`

### Builder

1. Backward compatibility:

   - unsliced `water_voxel_size_mm = nothing`
   - `slice_normal = (0, 0, 1)`
   - no rotation
   - non-sliced golden counts remain unchanged
   - water `rho`/`ρ` equals background water `rho`/`ρ`

2. Coarse sliced water count:

   - 1 mm sphere dx, 3 mm water dx
   - count is approximately `1/9` of a fine 2D water-plane reference

3. Signal conservation:

   - compare `sum(coarse.ρ)` to a fine 2D plane reference
   - use loose tolerance, for example `rtol = 0.1`, because lattice boundaries
     differ

4. Spheres untouched:

   - sphere spin count and sphere `rho`/`ρ` match the fine-water build

5. Phantom-attached oblique plane survives pose:

   - use non-default `slice_normal`
   - use nonzero `rotation` and optionally translation
   - after `build_phantom`, test water against the transformed plane:

     ```julia
     p_s = R * p_p + t
     n_s = R * n_p
     ```

   - all water distances are near zero for single-plane coarse water, or within
     half thickness for slab water

6. Non-sliced 3D coarse water:

   - 2x water dx gives roughly `1/8` count
   - `sum(ρ)` conserved with loose tolerance, for example `rtol = 0.15`

7. Input validation:

   - `water_voxel_size_mm <= 0` errors
   - zero `slice_normal` errors
   - water finer than spheres works and can give `ρ_eff < props.ρ`
   - `water_throughplane_voxel_size_mm <= 0` errors when provided

8. PD jitter with weighted water:

   - coarse water plus `PD_sigma_abs > 0` does not clamp weighted water to 1
   - material PD remains within `[0, 1]` after dividing by `ρ_weight`
   - with `PD_sigma_abs = 0`, `sum(ρ)` conservation tests still pass
   - default unweighted phantoms retain the existing `[0, 1]` clamp behaviour

## Verification command

Run:

```bash
julia --project=. test/runtests.jl
```

Existing tests should remain green.

## Risks and follow-ups

- Generalised slice clipping touches shared builder behaviour. The main
  regression surface is default-normal sliced builds.
- Coarse water preserves total signal approximately, but boundary discretisation
  changes count times weight by a few percent.
- Absolute SNR should be revalidated in E2 with `noise_sigma_abs = 50` before
  trusting final SNR numbers.
- Python/E2 wiring is deferred. When added, gate or warn when
  `water_voxel_size_mm > FOV / Npe`.
- For thick slabs, set `water_throughplane_voxel_size_mm` explicitly instead of
  using the default single weighted centre sheet.

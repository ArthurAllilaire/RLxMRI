# QalibreMDPhantom.jl — Digital Twin Library

## 0. The physical phantom

The QalibreMD NIST/ISMRM System Standard Model 130 (manufactured by Caliber MRI, formerly QalibreMD) is a hardware phantom designed to mimic the geometry of a human head and to provide a wide, calibrated range of quantitative MRI tissue parameters — T1, T2, and proton density — in a single, reproducible object. It is the standard reference device for quantitative MRI system validation in clinical and research settings, and its design is described in the *QalibreMD System Standard Model 130 User Manual*.

The phantom consists of a spherical water-filled housing (100 mm internal radius) containing four internal structures at fixed axial positions:

| Plate | z (mm) | Contents |
|-------|--------|----------|
| T1 array | +56.5 | 14 NiCl₂ spheres (15 mm radius), T1 range ~24 ms – 1.9 s at 3 T |
| T2 array | +16.5 | 14 MnCl₂ spheres (15 mm radius), T2 range ~87 ms – 2.8 s at 3 T |
| PD array | −23.5 | 14 spheres of varying proton density (15 mm radius) |
| Fiducial grid | distributed | 57 small spheres (5 mm radius) on a 40 mm cubic lattice, clipped to a 95 mm sphere |

Each contrast plate arranges its 14 spheres in two concentric rings: 10 on an outer ring of radius 65 mm and 4 on an inner ring of radius 28 mm. This geometry matches the phantom manual photographs and was verified against the physical device.

The relaxation values are field-strength-dependent (doped aqueous solutions do not follow a simple scaling law), so separate tables are provided for 1.5 T and 3 T. The library also distinguishes two serial classes: serial numbers ≥ 0042 use a revised recipe with significantly different T1 values compared to the legacy (0001–0041) class. For example, T1-sphere 1 at 3 T is 1.84 s on the current recipe versus 2.38 s on legacy.

---

## 1. Design goals and architecture

A digital twin of this phantom is needed for two reasons. First, it provides a ground-truth environment for RL training: the agent needs to query a simulator that returns physically meaningful signals and can report the true tissue parameters for reward computation. Second, it must be **flexible** — the RL training loop needs to randomise pose, field strength, noise level, and sphere properties without modifying library code.

The library is structured around a single contract: a `PhantomConfig` struct that carries everything the builder needs, passed to a single function `build_phantom` that returns a `KomaMRI.Phantom`. The phantom can then be fed directly to `KomaMRI.simulate` with no further setup. This design was chosen deliberately over a mutable phantom object or a collection of keyword arguments: the config is serialisable, diffable, and reproducible (the RNG seed is embedded), and the builder has no hidden state.

The pipeline from config to simulation is:

```
PhantomConfig
    │
    ├─ sphere_descriptors(:T1, cfg)   ┐
    ├─ sphere_descriptors(:T2, cfg)   │ one SphereDescriptor per contrast sphere
    ├─ sphere_descriptors(:PD, cfg)   │ (geometry + material, no voxels yet)
    ├─ sphere_descriptors(:fiducials) ┘
    │
    ├─ voxelise_sphere(d, delta_x)    → KomaMRI.Phantom per sphere
    │
    ├─ build_background_water(cfg)    → KomaMRI.Phantom (water housing, sphere cutouts)
    │
    ├─ apply_transform!(obj, rotation, translation)
    │
    └─ apply_per_spin_noise!(obj, augment, rng)
                                      → KomaMRI.Phantom (ready for simulate())
```

The two-stage design — descriptors first, voxels second — means all geometry and material decisions happen before any memory is allocated for spin arrays. The descriptor layer is also directly useful for imaging: `sphere_descriptor_pixels` maps descriptor centres to image pixel indices, so ROI extraction after reconstruction does not require a separate geometry calculation.

---

## 2. Geometry

### 2.1 Sphere voxelisation

Each sphere is voxelised on a cubic grid at spacing `delta_x = voxel_size_mm × 1e-3` metres. A spin is included if its grid point falls within the sphere radius:

$$
(x_i - c_x)^2 + (y_i - c_y)^2 + (z_i - c_z)^2 \leq r^2
$$

The grid is constructed by iterating over the bounding box `[c − r, c + r]` on each axis. For a typical 15 mm radius sphere at 2 mm voxels this produces around 500 spins per sphere, which is small enough that the full set of 14 spheres per plate fits comfortably in memory and the per-step simulate call is fast.

### 2.2 Slicing

For 2D imaging experiments, including only the spins in a thin slab reduces simulation cost dramatically. A `Slab` is defined by a centre point, a unit normal, and a half-thickness. Voxelisation is gated on the signed distance from the slab midplane:

$$
|(\mathbf{p} - \mathbf{c}) \cdot \hat{n}| \leq t/2
$$

The normal defaults to $\hat{z}$ (axial slice), but any orientation is supported. A floating-point tolerance of 0.1 µm is applied to the boundary condition to avoid rejecting voxels that land exactly on the slab edge due to roundoff. The slice centre `slice_center_mm = (0, 0, PLATE_Z_MM.T1)` isolates the T1 plate axially; the RL environment uses this to reduce simulation cost by an order of magnitude without changing the imaging geometry visible to the agent.

### 2.3 Background water

The housing is voxelised as a 100 mm-radius sphere at a coarser grid spacing (`water_voxel_size_mm`, defaulting to `voxel_size_mm`) with all contrast and fiducial sphere volumes cut out. The cutout is a simple exclusion test on squared distance to each sphere centre — the same condition as the voxelisation but negated. The water proton density is reweighted by `(water_dx / sphere_dx)³` to conserve the total water spin count relative to what a uniform fine-grid voxelisation would give, so the bulk water signal amplitude is physically correct regardless of which coarsening factor is chosen.

When a through-plane slice is selected, the water housing is replaced by a stack of plane-sheets (one per `water_throughplane_voxel_size_mm` step through the slab) so that each sheet carries the signal from the column of water it represents. This avoids the artefact that would arise from using a single plane at the slab centre: the plane would carry the signal of one voxel but the actual water signal integrates over the full slice thickness.

### 2.4 Pose transforms

`apply_transform!` rotates and translates all spin positions by the Euler XYZ rotation and translation specified in `PhantomConfig.rotation` and `translation_mm`. The rotation matrix is built from three successive Rx, Ry, Rz multiplications. The transform is applied after voxelisation and before augmentation noise so that the sphere geometry is exact before any stochastic perturbation.

---

## 3. Material tables

The relaxation values are read from four module-level constants defined from the phantom manual:

| Constant | Spheres | Source |
|----------|---------|--------|
| `T1_ARRAY[:T3]` / `[:T15]` | 14 T1-plate spheres | NiCl₂ recipe, manual table, SN ≥ 0042 |
| `T1_ARRAY_LEGACY[:T3]` / `[:T15]` | same spheres | pre-0042 recipe (substantially different values) |
| `T2_OF_T1_ARRAY` | T2 companions for T1 plate | same manual table |
| `T2_ARRAY[:T3]` / `[:T15]` | 14 T2-plate spheres | MnCl₂ recipe |
| `T1_OF_T2_ARRAY` | T1 companions for T2 plate | same manual table |
| `PD_FRACTIONS` | 14 PD-plate spheres | proton density fractions |
| `BACKGROUND_WATER` | housing water | distilled water at 20 °C |

The T1 range at 3 T spans two decades: 23 ms to 1.84 s for the NiCl₂ plate, covering the full range of biological tissue T1 values from white matter to cerebrospinal fluid. The T2 range spans ~87 ms to 2.76 s. These wide ranges are why the phantom is used for system validation: a single acquisition must produce reliable estimates across the full physiological range.

The `serial_number_class` field switches the T1 table between current and legacy recipes. The difference is large enough to matter: the highest T1 sphere changes from 1.84 s (current) to 2.38 s (legacy) at 3 T. Using the wrong table introduces a systematic bias in all T1 estimates that no amount of RL optimisation can overcome.

---

## 4. Augmentation

`AugmentConfig` enables domain randomisation for RL training. All fields default to zero. When non-zero values are provided, `apply_per_spin_noise!` draws from the relevant distributions using the config's `rng_seed`:

| Field | Effect |
|-------|--------|
| `T1_sigma_rel` | Multiplies each sphere's T1 by `exp(σ·z)` where `z ~ N(0,1)`, drawn once per sphere |
| `T2_sigma_rel` | Same for T2 |
| `PD_sigma_abs` | Adds `N(0, σ²)` to each spin's proton density |
| `position_sigma_mm` | Adds `N(0, σ²)` to each spin's (x, y, z) position independently |
| `B0_sigma_Hz` | Adds `N(0, σ²)` per-spin off-resonance, stored in `Δw` (rad/s after ×2π) |
| `drop_sphere_p` | Drops each sphere independently with Bernoulli probability p before voxelisation |

The T1/T2 jitter is applied at the sphere level (one draw per sphere, not per spin) to simulate manufacturing variability in the doping concentration — a per-spin draw would be unphysical since all spins in a sphere are in the same solution. The position noise is per-spin and models rigid-body placement uncertainty at the sub-voxel scale.

The `rng_seed` is the only source of stochasticity: fixing it makes `build_phantom` deterministic, so the same configuration always produces the same phantom. Changing the seed between RL episodes provides statistically independent training samples without any global mutable state.

---

## 5. KomaMRI integration

`QalibreMDPhantom.jl` is built entirely on top of `KomaMRI.jl` and returns standard `KomaMRI.Phantom` objects. The concatenation operator `+` on `Phantom` is used throughout to combine plates: each plate's voxelised spheres are concatenated into one `Phantom`, then the water housing is concatenated with the sphere stack. The result is a single flat spin array with heterogeneous relaxation values, which is exactly what `KomaMRI.simulate` expects.

The matching scanner is obtained from `scanner_for_field(cfg)`, which returns a `Scanner` with B0 set to 3.0 T for `:T3` or 1.5 T for `:T15`. Using any other `Scanner` with a phantom built at a different field would silently use the wrong relaxation values, so this helper is provided to make the coupling explicit.

### Known issue: floating-point accumulation in KomaMRI

During development two floating-point accumulation bugs were discovered in `KomaMRIBase.jl`. The first caused per-shot signal drift in long sequences (> ~270 s total duration) when the gradient moment accumulator overflowed. The second was the same root cause at a different accumulation site. Both were diagnosed, reported (GitHub issue #788), and fixed in a fork (`ArthurAllilaire/KomaMRI.jl`, branch `fix/grad-fp`, PR #789). The library currently depends on this fork at a pinned commit; the fix is awaiting upstream merge. This dependency is the only deviation from the registered Julia package registry.

---

## 6. Library availability

`QalibreMDPhantom.jl` is developed as a standalone open-source Julia package at `github.com/ArthurAllilaire/QalibreMDPhantom.jl` with API documentation hosted at `arthurallilaire.github.io/QalibreMDPhantom.jl`. The package includes a complete test suite (~373 tests), example scripts for T1 mapping and SNR calibration, and documentation pages covering phantom construction, sequence design, parameter fitting, and SNR diagnostics. Making the library independently installable was a deliberate choice: the digital twin is a contribution in its own right, separable from the RL experiments it was built to support, and a public release allows the methodology to be reproduced and extended by other groups working on quantitative MRI system validation or simulation-based sequence optimisation.

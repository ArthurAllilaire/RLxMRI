"""
    QalibreMDPhantom

Programmatic, parameterised digital twin of the **QalibreMD NIST/ISMRM
System Standard Model 130** phantom. Builds a `KomaMRI.Phantom` from a
single `PhantomConfig` so an RL training loop can sample field strength,
rotation, voxel size, per-property jitter, etc. without touching library
code.

    using QalibreMDPhantom
    obj = build_phantom(PhantomConfig(field = :T3, voxel_size_mm = 2.0))
"""
module QalibreMDPhantom

using KomaMRI
using Random
using LinearAlgebra
import Suppressor

# --- materials (pure data) ------------------------------------------------
include("materials/fiducial.jl")    # defines Relax, FIDUCIAL_PROPS
include("materials/background.jl")  # uses Relax; defines BACKGROUND_WATER
include("materials/t1_array.jl")
include("materials/t2_array.jl")
include("materials/pd_array.jl")    # uses BACKGROUND_WATER

# --- geometry primitives --------------------------------------------------
include("geometry/sphere.jl")
include("geometry/plate_layouts.jl")

# --- configs, builder, augmentations --------------------------------------
include("config.jl")
include("augment.jl")
include("builder.jl")

# --- sequences, fitting, baseline experiments -----------------------------
include("sequences/blocks.jl")
include("fitting/fits.jl")
include("baselines/e0.jl")

export PhantomConfig, AugmentConfig, SphereDescriptor,
       build_phantom, build_plate, build_sphere, build_background_water,
       sphere_descriptors, all_sphere_descriptors,
       voxelise_sphere, sphere_volume,
       contrast_plate_centres, fiducial_grid_centres,
       rotation_matrix, apply_transform, apply_per_spin_noise!,
       T1_ARRAY, T2_OF_T1_ARRAY, T1_ARRAY_LEGACY,
       T2_ARRAY, T1_OF_T2_ARRAY_DEFAULT,
       PD_FRACTIONS, pd_t1, pd_t2,
       FIDUCIAL_PROPS, BACKGROUND_WATER, Relax,
       PLATE_Z_MM, CONTRAST_RADIUS_M, FIDUCIAL_RADIUS_M, HOUSING_RADIUS_M,
       # sequences
       rf_duration, ir_sequence, se_sequence, single_spin_phantom,
       # fitting
       fit_t1_ir, fit_t2_se,
       # E0 baseline
       measure_ir_signal, measure_se_signal,
       measure_t1, measure_t2,
       default_TI_schedule, default_TE_schedule,
       adaptive_TI_schedule, adaptive_TE_schedule,
       run_e0

end # module

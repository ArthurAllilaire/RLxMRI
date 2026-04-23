# T2-array spheres — MnCl₂ in water (T2-1 … T2-14)
# SN ≥ 0042 recipe, T2 values in seconds.
# T1 of these spheres is long (~water) compared to T2; default T1 = 3.0 s
# matches the v1 assumption in PLAN.md §2.4.2 and is overrideable.

const T2_ARRAY = Dict(
    :T15 => [2.640, 2.292, 1.923, 1.609, 1.245, 1.004, 0.7829, 0.5331,
             0.4005, 0.2610, 0.1898, 0.1547, 0.1002, 0.07965],
    :T3  => [2.756, 2.281, 1.961, 1.552, 1.341, 1.017, 0.7821, 0.5897,
             0.4436, 0.3148, 0.2374, 0.1701, 0.1238, 0.0869],
)

const T1_OF_T2_ARRAY_DEFAULT = 3.0

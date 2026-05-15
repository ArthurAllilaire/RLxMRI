"""QalibreMD phantom tissue values (Python mirror of src/materials/{t1,t2,pd}_array.jl).

Values are at 3T (the field used in Ch5). Used by Gate 6.4 to fit a prior on
(log T1, log T2, log PD) for the QalibreMD phantom value distribution.

Sources:
- src/materials/t1_array.jl  T1_ARRAY[:T3], T2_OF_T1_ARRAY[:T3]
- src/materials/t2_array.jl  T2_ARRAY[:T3], T1_OF_T2_ARRAY[:T3]
- src/materials/pd_array.jl  PD_FRACTIONS (T1,T2 fall back to background water)
"""
from __future__ import annotations
import numpy as np

# T1-array (NiCl2 spheres) at 3T
T1_ARRAY_T3 = np.array([
    1.838, 1.398, 0.9983, 0.7258, 0.5091, 0.367, 0.2587, 0.1847,
    0.1308, 0.0909, 0.0642, 0.04628, 0.03265, 0.02295,
])
T2_OF_T1_ARRAY_T3 = np.array([
    1.354, 1.035, 0.7283, 0.5244, 0.3686, 0.2667, 0.1893, 0.1341,
    0.0938, 0.0657, 0.0468, 0.03315, 0.02369, 0.01673,
])

# T2-array (MnCl2 spheres) at 3T
T2_ARRAY_T3 = np.array([
    2.756, 2.281, 1.961, 1.552, 1.341, 1.017, 0.7821, 0.5897,
    0.4436, 0.3148, 0.2374, 0.1701, 0.1238, 0.0869,
])
T1_OF_T2_ARRAY_T3 = np.array([
    2.756, 2.281, 1.961, 1.552, 1.341, 1.017, 0.7821, 0.5897,
    0.4438, 0.2998, 0.2378, 0.1705, 0.1218, 0.0869,
])

# PD spheres: PD fractions; T1/T2 fall back to bulk water at 3T (~3.0/2.0 s).
PD_FRACTIONS = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
                         0.50, 0.60, 0.70, 0.80, 0.90, 1.00])
BG_WATER_T1_3T = 3.0
BG_WATER_T2_3T = 2.0


def phantom_value_table() -> np.ndarray:
    """Return an (N, 3) array of (T1, T2, PD) for all calibration spheres at 3T."""
    t1_rows = np.stack([T1_ARRAY_T3, T2_OF_T1_ARRAY_T3,
                        np.ones_like(T1_ARRAY_T3)], axis=1)
    t2_rows = np.stack([T1_OF_T2_ARRAY_T3, T2_ARRAY_T3,
                        np.ones_like(T2_ARRAY_T3)], axis=1)
    pd_rows = np.stack([np.full_like(PD_FRACTIONS, BG_WATER_T1_3T),
                        np.full_like(PD_FRACTIONS, BG_WATER_T2_3T),
                        PD_FRACTIONS], axis=1)
    return np.concatenate([t1_rows, t2_rows, pd_rows], axis=0)


def sample_from_prior(n: int, rng: np.random.Generator,
                      jitter_log: float = 0.15) -> np.ndarray:
    """Sample (T1, T2, PD) from the phantom value distribution with log-normal
    jitter around each calibration sphere. Used for prior-fitting and training
    data generation.

    jitter_log: stddev in natural log space (0.15 ~ ~15% multiplicative jitter).
    """
    table = phantom_value_table()  # (N, 3)
    idx = rng.integers(0, len(table), size=n)
    base = table[idx]
    log_base = np.log(base)
    log_sample = log_base + rng.normal(0.0, jitter_log, size=log_base.shape)
    # Clip PD into [0.02, 1.2] in linear space.
    sample = np.exp(log_sample)
    sample[:, 2] = np.clip(sample[:, 2], 0.02, 1.2)
    sample[:, 0] = np.clip(sample[:, 0], 0.01, 5.0)
    sample[:, 1] = np.clip(sample[:, 1], 0.005, 3.0)
    return sample


def sample_ood(n: int, rng: np.random.Generator) -> np.ndarray:
    """Out-of-distribution: take an in-dist sample and ±50% multiplicatively jitter."""
    base = sample_from_prior(n, rng, jitter_log=0.15)
    mult = np.exp(rng.uniform(np.log(0.5), np.log(1.5), size=base.shape))
    return np.clip(base * mult, [0.01, 0.005, 0.02], [5.0, 3.0, 1.2])

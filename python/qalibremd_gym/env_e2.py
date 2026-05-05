"""Gymnasium wrapper for E2: full-plate multi-sphere T1 mapping.

Action space : Box(5), continuous, normalised to [-1, 1] by the env wrapper.
               Physical ranges:
                 TI      [0.01, 3.00]  s
                 TE      [0.005, 0.08] s
                 TR      [0.50, 5.00]  s
                 alpha   [5, 180]      deg  (excitation flip angle)
                 slice_z [-60, 60]     mm   (future use)

Observation  : float32 Box of length Nfe*Npe + 2*n_spheres + 3
                 = flattened magnitude image (normalised to [0,1])
                 + log10(running T1 estimate) per sphere (0 = no estimate)
                 + log10(σ_T1 / T1_est) per sphere, clamped [-3, 0]
                   (0 = fully uncertain / no estimate yet)
                 + (time_fraction, block_fraction, 1.0)

Reward       : −aggMAPE across 14 spheres per step (0 for first block),
               where aggMAPE = α·mean + (1−α)·max (α=mape_alpha; 1.0 = legacy mean).
               Terminal bonus if aggMAPE < success_tol at episode end.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from . import env as _env_mod
from .env import _ensure_julia, _to_py


class QalibreMDE2Env(gym.Env):
    """Multi-sphere T1 mapping RL environment (E2).

    See module docstring for space dimensions and reward definition.
    """

    metadata = {"render_modes": [], "render_fps": 0}

    # Physical action bounds (full 5-dim space)
    _ACT_LO = np.array([0.010, 0.005, 0.500,   5.0, -60.0], dtype=np.float32)
    _ACT_HI = np.array([3.000, 0.080, 5.000, 180.0,  60.0], dtype=np.float32)

    def __init__(
        self,
        *,
        rng_seed: int = 0,
        cfg_field: str = "T3",
        voxel_size_mm: float = 3.0,
        Nfe: int = 16,
        Npe: int = 8,
        FOV: float = 0.2,
        max_blocks: int = 15,
        time_budget_s: float = 120.0,
        terminal_bonus: float = 0.5,
        success_tol: float = 0.05,
        noise_sigma_rel: float = 0.05,
        T1_sigma_rel: float = 0.05,
        translation_sigma_mm: float = 5.0,
        rotation_sigma_rad: float = 0.15,
        reward_mode: str = "neg_mape",       # "neg_mape" | "delta_mape"
        mape_alpha: float = 1.0,             # α in α·mean + (1−α)·max; 1.0 = mean only
        simplified_action: bool = False,     # drop slice_z, fix α_exc=90°
        project_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        _ensure_julia(project_dir)

        jl = _env_mod._JL
        qmd = _env_mod._JL_QMD

        self._env = qmd.E2Env(
            rng_seed              = rng_seed,
            cfg_field             = jl.Symbol(cfg_field),
            voxel_size_mm         = float(voxel_size_mm),
            FOV                   = float(FOV),
            Nfe                   = int(Nfe),
            Npe                   = int(Npe),
            max_blocks            = int(max_blocks),
            time_budget_s         = float(time_budget_s),
            terminal_bonus        = float(terminal_bonus),
            success_tol           = float(success_tol),
            noise_sigma_rel       = float(noise_sigma_rel),
            T1_sigma_rel          = float(T1_sigma_rel),
            translation_sigma_mm  = float(translation_sigma_mm),
            rotation_sigma_rad    = float(rotation_sigma_rad),
            reward_mode           = jl.Symbol(reward_mode),
            mape_alpha            = float(mape_alpha),
        )

        obs_dim = int(qmd.e2_obs_dim(self._env))

        # Normalised action space: agent outputs [-1, 1], wrapper rescales.
        # Simplified mode exposes only [TI, TE, TR] (3-dim); α_exc is fixed at
        # 90° and slice_z at 0 — both were either coordinated-but-wasted (α
        # interacted with the fitter) or completely unused (slice_z).
        self._simplified_action = bool(simplified_action)
        n_act = 3 if self._simplified_action else 5
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(n_act,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self._Nfe = Nfe
        self._Npe = Npe

    # ── physical ↔ normalised action conversion ──────────────────────────

    def _denorm_action(self, action: np.ndarray) -> np.ndarray:
        """Map agent output in [-1, 1] to a physical 5-vector for Julia."""
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        if self._simplified_action:
            # Lift 3-dim agent output → 5-dim env action: α_exc=90°, slice_z=0
            full = np.zeros(5, dtype=np.float32)
            lo3 = self._ACT_LO[[0, 1, 2]]
            hi3 = self._ACT_HI[[0, 1, 2]]
            full[[0, 1, 2]] = lo3 + (a + 1.0) / 2.0 * (hi3 - lo3)
            full[3] = 90.0
            full[4] = 0.0
            return full
        return self._ACT_LO + (a + 1.0) / 2.0 * (self._ACT_HI - self._ACT_LO)

    # ── Gymnasium API ──────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        qmd = _env_mod._JL_QMD
        if seed is not None:
            obs = qmd.e2_reset_b(self._env, rng_seed=int(seed))
        else:
            obs = qmd.e2_reset_b(self._env)
        obs = np.asarray(obs, dtype=np.float32)
        info = {
            "T1_true": np.array([float(v) for v in self._env.T1_true]),
        }
        return obs, info

    def step(self, action):
        phys = self._denorm_action(action)
        obs, reward, done, info_dict = _env_mod._JL_QMD.e2_step_b(
            self._env,
            list(phys.astype(float)),
        )
        obs  = np.asarray(obs, dtype=np.float32)
        info = {}
        for k, v in info_dict.items():
            k_str = str(k)
            try:
                arr = np.asarray(v, dtype=np.float64)
                info[k_str] = arr if arr.ndim > 0 else float(arr)
            except Exception:
                info[k_str] = _to_py(v)
        return obs, float(reward), bool(done), False, info

    def render(self):
        return None

    def close(self):
        pass

    # ── convenience accessors ─────────────────────────────────────────────

    @property
    def T1_true(self) -> np.ndarray:
        return np.array([float(v) for v in self._env.T1_true])

    @property
    def T1_est(self) -> np.ndarray:
        return np.array([float(v) for v in self._env.T1_est])

    @property
    def n_blocks(self) -> int:
        return int(self._env.n_blocks)

    @property
    def time_used_s(self) -> float:
        return float(self._env.time_used_s)

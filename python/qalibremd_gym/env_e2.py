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

    # Physical action bounds (full 5-dim space): [TI, TE, TR, alpha_deg, slice_z_mm]
    #
    # Where these come from (see src/rl/e2.jl header for the full version
    # and EXPERT_REPORT_E2_4.md §12):
    #
    # TI ∈ [0.010, 3.000] s
    #   Lower bound 10 ms is ~20× above the physics floor: at amp_T = 20 μT
    #   (default in ir_se_2d_sequence), inversion pulse d180 = π/(2π·γ·B1)
    #   ≈ 0.59 ms and excitation d90 ≈ 0.29 ms with γ = 42.58 MHz/T. Strict
    #   pulse-non-overlap requires TI ≥ ~0.5 ms; 10 ms adds margin for
    #   gradient activity, simulator stability, and clinical relevance.
    #   Lower bound does not bind for our shortest phantom sphere
    #   (T1 = 0.024 s, optimal TI = T1·ln(2) ≈ 0.017 s, already above 10 ms).
    #   Upper bound 3.0 s is past optimal for the longest sphere
    #   (T1 = 1.84 s, optimal TI ≈ 1.28 s); higher TIs are post-saturation
    #   and carry no T1 information.
    #
    # TE ∈ [0.005, 0.080] s
    #   Echo time. Short TE preserves signal (T2 decay). Lower bound 5 ms
    #   is roughly the readout duration; upper bound 80 ms is past where
    #   T2 = 20 ms phantoms have signal left.
    #
    # TR ∈ [0.500, 5.000] s
    #   Repetition time. Lower bound 500 ms keeps TR > TI for typical
    #   choices; upper bound 5 s is "5·T1_max" (full Mz recovery under the
    #   legacy steady-state convention). F1+ models partial recovery, so
    #   shorter TRs are now genuinely informative.
    #
    # alpha_deg ∈ [5, 180]°
    #   Excitation flip angle. 5° lower bound avoids sin(α) → 0 numerical
    #   issues; 180° upper bound permits inversion if the agent wants it
    #   (it usually picks 90°).
    #
    # slice_z ∈ [-60, +60] mm
    #   Slab centre offset. Currently UNUSED in the env (the F1+ forward
    #   model is non-slice-selective). Reserved for E3 per-sphere targeting.
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
        noise_sigma_abs: float = 0.005,
        # If set, override noise_sigma_abs by calibrating once at env
        # construction: simulate one reference IR-SE block (TI=0.5 s, TR=3.0 s,
        # α=90°) on a nominal (no-jitter, no-pose) phantom, then set
        # σ = ksp_rms / target_snr. σ is then fixed for the env's lifetime
        # (scene-independent, matches the physical hardware-noise model).
        target_snr: Optional[float] = None,
        T1_sigma_rel: float = 0.05,
        translation_sigma_mm: float = 5.0,
        rotation_sigma_rad: float = 0.15,
        reward_mode: str = "neg_mape",       # "neg_mape" | "delta_mape"
        # α in α·mean + (1−α)·max; 1.0 = mean only
        mape_alpha: float = 1.0,
        simplified_action: bool = False,     # drop slice_z, fix α_exc=90°
        # True = signed real() recon (cr_explainer.md §14)
        phase_sensitive: bool = False,
        # "asymptotic" | "profile_likelihood" | "bootstrap"
        sigma_method: str = "bootstrap",
        subset_size: Optional[int] = None,
        log_ti_action: bool = False,
        # D2 diagnostic (EXPERT_REPORT_TRAC §17): narrows the fitter's T1
        # grid to a log-band of ±oracle_band around T1_true per sphere.
        oracle_fit: bool = False,
        oracle_band: float = 2.0,
        # §17.10 control: bump fitter T1 grid resolution to test whether
        # baseline MAPE is grid-coarseness-limited.
        fitter_n_grid: int = 200,
        project_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        _ensure_julia(project_dir)

        jl = _env_mod._JL
        qmd = _env_mod._JL_QMD

        env_kwargs = dict(
            rng_seed=rng_seed,
            cfg_field=jl.Symbol(cfg_field),
            voxel_size_mm=float(voxel_size_mm),
            FOV=float(FOV),
            Nfe=int(Nfe),
            Npe=int(Npe),
            max_blocks=int(max_blocks),
            time_budget_s=float(time_budget_s),
            terminal_bonus=float(terminal_bonus),
            success_tol=float(success_tol),
            noise_sigma_abs=float(noise_sigma_abs),
            T1_sigma_rel=float(T1_sigma_rel),
            translation_sigma_mm=float(translation_sigma_mm),
            rotation_sigma_rad=float(rotation_sigma_rad),
            reward_mode=jl.Symbol(reward_mode),
            mape_alpha=float(mape_alpha),
            phase_sensitive=bool(phase_sensitive),
            sigma_method=jl.Symbol(sigma_method),
            oracle_fit=bool(oracle_fit),
            oracle_band=float(oracle_band),
            fitter_n_grid=int(fitter_n_grid),
        )
        if subset_size is not None:
            env_kwargs["subset_size"] = int(subset_size)
        if target_snr is not None:
            env_kwargs["target_snr"] = float(target_snr)
        self._env = qmd.E2Env(**env_kwargs)

        obs_dim = int(qmd.e2_obs_dim(self._env))

        # Normalised action space: agent outputs [-1, 1], wrapper rescales.
        # Simplified mode exposes only [TI, TE, TR] (3-dim); α_exc is fixed at
        # 90° and slice_z at 0 — both were either coordinated-but-wasted (α
        # interacted with the fitter) or completely unused (slice_z).
        self._simplified_action = bool(simplified_action)
        self._log_ti_action = bool(log_ti_action)
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

    def _map_ti(self, u01: float, lo: float, hi: float) -> float:
        """Map u01 ∈ [0, 1] to a TI in [lo, hi]. Log-spaced if enabled, so the
        agent's action density is constant per decade — short-T1 spheres need
        TIs in the lowest 1.5 % of the linear range, see EXPERT_REPORT_TRAC §9.2."""
        if self._log_ti_action:
            return float(lo) * (float(hi) / float(lo)) ** float(u01)
        return float(lo) + float(u01) * (float(hi) - float(lo))

    def _denorm_action(self, action: np.ndarray) -> np.ndarray:
        """Map agent output in [-1, 1] to a physical 5-vector for Julia."""
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        u = (a + 1.0) / 2.0  # [0, 1]
        if self._simplified_action:
            full = np.zeros(5, dtype=np.float32)
            full[0] = self._map_ti(u[0], self._ACT_LO[0], self._ACT_HI[0])
            full[1] = self._ACT_LO[1] + u[1] * (self._ACT_HI[1] - self._ACT_LO[1])
            full[2] = self._ACT_LO[2] + u[2] * (self._ACT_HI[2] - self._ACT_LO[2])
            full[3] = 90.0
            full[4] = 0.0
            return full
        full = self._ACT_LO + u * (self._ACT_HI - self._ACT_LO)
        full[0] = self._map_ti(u[0], self._ACT_LO[0], self._ACT_HI[0])
        return full.astype(np.float32)

    # ── Gymnasium API ──────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        forced_sphere_indices=None,
    ):
        super().reset(seed=seed)
        qmd = _env_mod._JL_QMD
        kwargs = {}
        if seed is not None:
            kwargs["rng_seed"] = int(seed)
        if forced_sphere_indices is not None:
            # Julia expects 1-based indices; caller passes 0-based Python indices
            kwargs["forced_indices"] = [int(i) + 1 for i in forced_sphere_indices]
        obs = qmd.e2_reset_b(self._env, **kwargs)
        obs = np.asarray(obs, dtype=np.float32)
        info = {
            "T1_true": np.array([float(v) for v in self._env.T1_true]),
            "sphere_indices": np.array([int(v) for v in self._env.sphere_indices],
                                       dtype=np.int64),
        }
        return obs, info

    def step(self, action):
        phys = self._denorm_action(action)
        obs, reward, done, info_dict = _env_mod._JL_QMD.e2_step_b(
            self._env,
            list(phys.astype(float)),
        )
        obs = np.asarray(obs, dtype=np.float32)
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
    def T1_sigma(self) -> np.ndarray:
        """Asymptotic per-sphere σ on T1_est from the IR fit's J^T J inverse.
        NaN entries mean "fewer than 2 samples" or "fit failed"."""
        return np.array([float(v) for v in self._env.T1_sigma])

    @property
    def n_blocks(self) -> int:
        return int(self._env.n_blocks)

    @property
    def time_used_s(self) -> float:
        return float(self._env.time_used_s)

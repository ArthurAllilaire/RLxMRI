"""Gymnasium wrapper for E2: full-plate multi-sphere T1 mapping.

Action space : Box(4), continuous, normalised to [-1, 1] by the env wrapper.
               Physical ranges:
                 TI      [0.01, 3.00]  s
                 TE      [0.005, 0.08] s
                 TR      [0.50, 5.00]  s
                 alpha   [5, 180]      deg  (excitation flip angle)

Observation  : float32 Box. The T1-estimate channel and budget are always
               present; the image and σ channels are optional (default off,
               E2_RERUN_PLAN §6.2–6.3). Default obs = n_spheres + 3:
                 + log10(running T1 estimate) per sphere (0 = no estimate)
                 + (time_fraction, block_fraction, 1.0)
               include_image prepends the flattened magnitude image (Nfe*Npe,
               normalised to [0,1]); include_sigma appends log10(σ_T1 / T1_est)
               per sphere, clamped [-3, 0] (0 = fully uncertain / no estimate).

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

    # Physical action bounds (full 4-dim space): [TI, TE, TR, alpha_deg].
    #
    # SOURCE OF TRUTH: julia/rl/e2.jl `e2_action_lo`/`e2_action_hi`. This array
    # is a Python *mirror* — __init__ reads the live Julia bounds and verifies
    # them against this constant, failing fast on drift. Keep the two in sync.
    # (Held as a class constant so the Julia-free unit tests in
    # tests/test_alpha_action_modes.py can map actions without booting Julia.)
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
    # (A slice_z slab-offset action existed historically but was always unused —
    # the F1+ forward model is non-slice-selective — so it has been removed.)
    _ACT_LO = np.array([0.010, 0.005, 0.500,   5.0], dtype=np.float32)
    _ACT_HI = np.array([3.000, 0.080, 5.000, 180.0], dtype=np.float32)

    def __init__(
        self,
        *,
        rng_seed: int = 0,
        cfg_field: str = "T15",
        voxel_size_mm: float = 1.0,
        Nfe: int = 64,
        Npe: int = 32,
        FOV: float = 0.2,
        use_gpu: bool = False,
        max_blocks: int = 15,
        time_budget_s: float = 120.0,
        terminal_bonus: float = 0.5,
        success_tol: float = 0.05,
        noise_sigma_abs: float = 50.0,   # σ* for NEMA dual-acq SNR ≈ 25 (E2_RERUN_PLAN §3.1)
        T1_sigma_rel: float = 0.05,
        translation_sigma_mm: float = 5.0,
        rotation_sigma_rad: float = 0.15,
        reward_mode: str = "neg_mape",       # "neg_mape" | "delta_mape"
        # α in α·mean + (1−α)·max; 1.0 = mean only
        mape_alpha: float = 1.0,
        simplified_action: bool = False,     # [TI, TE, TR] only, fix α_exc=90°
        # Fix TE at 20 ms and expose only [TI, TR] (2-dim). With learn_alpha,
        # exposes [TI, TR, α] (3-dim). Isolates α as the single new DOF for the
        # Run A0/A ablation (ALPHA_DOF.md, action-space ablation). Takes precedence over
        # simplified_action when set.
        fix_te: bool = False,
        learn_alpha: bool = False,
        # True = signed real() recon (cr_explainer.md §14)
        phase_sensitive: bool = False,
        # "asymptotic" | "profile_likelihood" | "bootstrap"
        sigma_method: str = "bootstrap",
        subset_size: Optional[int] = None,
        forced_sphere_indices: Optional[list[int]] = None,
        log_ti_action: bool = False,
        t1_sampler: str = "lognormal",
        pose_mode: str = "auto",
        # D2 diagnostic (EXPERT_REPORT_TRAC §17): narrows the fitter's T1
        # grid to a log-band of ±oracle_band around T1_true per sphere.
        oracle_fit: bool = False,
        oracle_band: float = 2.0,
        # §17.10 control: bump fitter T1 grid resolution to test whether
        # baseline MAPE is grid-coarseness-limited.
        fitter_n_grid: int = 200,
        # Observation channels (E2_RERUN_PLAN §6.2–6.3). Both default off, so
        # obs = [log10(T1_est) per sphere, budget(3)]. Enable include_image to
        # prepend the flattened normalised recon image (Nfe*Npe dims);
        # include_sigma to append the per-sphere fitter-σ channel (n_spheres).
        include_image: bool = False,
        include_sigma: bool = False,
        roi_radius: int = 0,
        include_water: bool = True,
        water_voxel_size_mm: Optional[float] = None,
        # Background-water simulation model (src/water_cache.jl):
        #   "bloch"          — full-Bloch sim the water with the spheres every step.
        #   "cached_perline" — Bloch-sim only the spheres; add the water from a cached
        #                      Koma template rescaled analytically per k-line (~8×
        #                      per-step, reproduces the full sim to the T1-grid floor).
        #                      Requires include_water. Cache scope follows include_image
        #                      (global when T1-only obs, per-episode when image-in-obs).
        water_model: str = "bloch",
        # Forward-model surrogate (src/rl/e2.jl). "bloch" (default) runs the full
        # KomaMRI Bloch sim + 2D recon every step. "analytic" skips Koma and
        # synthesises per-sphere magnitudes from the same closed form the fitter
        # inverts (transient_mz_at_excite_npe) — ~µs/step, fits are noise-limited
        # only. For fast reward SCREENING, not absolute MAPE. Forces a T1-only obs
        # (no image). analytic_noise_sigma sets the signal-space σ (≈SNR at the
        # reference operating point; default 0.04 ≈ SNR 25, sweepable).
        forward_model: str = "bloch",
        analytic_noise_sigma: float = 0.04,
        # λ on scan time: subtracts time_penalty_coef·(block_time/budget) from the
        # per-step reward, on top of ANY reward_mode. Makes scan time a first-class
        # cost (the base reward is per-block and implicitly favours fat high-TR
        # blocks). Only applied when allow_stop is set (see allow_stop).
        time_penalty_coef: float = 0.0,
        # Learned STOP decision (optimal stopping). When True, the action space
        # gains one extra "stop gate" dim (the last component); a value > 0 ends
        # the episode after the current block. Lets the agent choose when the
        # accuracy gain from another block no longer justifies its scan time
        # (paired with time_penalty_coef). Default False = fixed-budget episodes.
        allow_stop: bool = False,
        project_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        if bool(learn_alpha) and not bool(fix_te):
            raise ValueError("learn_alpha requires fix_te=True (Run A action mode)")
        _ensure_julia(project_dir, use_gpu=bool(use_gpu))

        jl = _env_mod._JL
        qmd = _env_mod._JL_QMD

        env_kwargs = dict(
            rng_seed=rng_seed,
            cfg_field=jl.Symbol(cfg_field),
            voxel_size_mm=float(voxel_size_mm),
            FOV=float(FOV),
            Nfe=int(Nfe),
            Npe=int(Npe),
            use_gpu=bool(use_gpu),
            max_blocks=int(max_blocks),
            time_budget_s=float(time_budget_s),
            terminal_bonus=float(terminal_bonus),
            success_tol=float(success_tol),
            noise_sigma_abs=float(noise_sigma_abs),
            T1_sigma_rel=float(T1_sigma_rel),
            t1_sampler=jl.Symbol(t1_sampler),
            forced_sphere_indices=(
                [] if forced_sphere_indices is None
                else [int(i) for i in forced_sphere_indices]
            ),
            translation_sigma_mm=float(translation_sigma_mm),
            rotation_sigma_rad=float(rotation_sigma_rad),
            pose_mode=jl.Symbol(pose_mode),
            reward_mode=jl.Symbol(reward_mode),
            mape_alpha=float(mape_alpha),
            phase_sensitive=bool(phase_sensitive),
            sigma_method=jl.Symbol(sigma_method),
            oracle_fit=bool(oracle_fit),
            oracle_band=float(oracle_band),
            fitter_n_grid=int(fitter_n_grid),
            include_image=bool(include_image),
            include_sigma=bool(include_sigma),
            roi_radius=int(roi_radius),
            include_water=bool(include_water),
            water_model=jl.Symbol(water_model),
            forward_model=jl.Symbol(forward_model),
            analytic_noise_sigma=float(analytic_noise_sigma),
            time_penalty_coef=float(time_penalty_coef),
            allow_stop=bool(allow_stop),
        )
        if water_voxel_size_mm is not None:
            env_kwargs["water_voxel_size_mm"] = float(water_voxel_size_mm)
        if subset_size is not None:
            env_kwargs["subset_size"] = int(subset_size)
        self._env = qmd.E2Env(**env_kwargs)

        # Action bounds: Julia's e2_action_lo/hi are authoritative. Read them
        # from the live env and use those; verify the Python class mirror matches
        # so the two definitions can never silently drift apart.
        jl_lo = np.asarray(qmd.e2_action_lo(self._env), dtype=np.float32)
        jl_hi = np.asarray(qmd.e2_action_hi(self._env), dtype=np.float32)
        if not (jl_lo.shape == self._ACT_LO.shape
                and np.allclose(jl_lo, self._ACT_LO)
                and np.allclose(jl_hi, self._ACT_HI)):
            raise ValueError(
                "Action-bound drift between Julia and Python: Julia "
                f"e2_action_lo/hi = {jl_lo}/{jl_hi}, Python mirror "
                f"_ACT_LO/_ACT_HI = {self._ACT_LO}/{self._ACT_HI}. "
                "Sync env_e2._ACT_LO/_ACT_HI with julia/rl/e2.jl.")
        self._ACT_LO = jl_lo
        self._ACT_HI = jl_hi

        obs_dim = int(qmd.e2_obs_dim(self._env))

        # Normalised action space: agent outputs [-1, 1], wrapper rescales.
        # Simplified mode exposes only [TI, TE, TR] (3-dim); α_exc is fixed at
        # 90° (it was coordinated-but-wasted — α interacted with the fitter).
        self._simplified_action = bool(simplified_action)
        self._fix_te = bool(fix_te)
        self._learn_alpha = bool(learn_alpha)
        self._log_ti_action = bool(log_ti_action)
        self._allow_stop = bool(allow_stop)
        if self._fix_te:
            # [TI, TR] (2-dim) or [TI, TR, α] (3-dim) — TE fixed at 20 ms.
            n_act = 3 if self._learn_alpha else 2
        elif self._simplified_action:
            n_act = 3
        else:
            n_act = 4  # [TI, TE, TR, α]
        # allow_stop appends one "stop gate" component (the LAST dim): the agent
        # raw output > 0 ⇒ stop after this block. Peeled off in _denorm_action.
        if self._allow_stop:
            n_act += 1
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

    # Fixed-TE action modes (ALPHA_DOF.md, action-space ablation): TE pinned, α window narrowed
    # to [5°, 90°] so the Ernst regime sits mid-range.
    _FIXED_TE_S = 0.020
    _ALPHA_LO_DEG = 5.0
    _ALPHA_HI_DEG = 90.0

    def cr_timing_constraints(self) -> dict[str, float]:
        """Timing constraints a fixed-TE CR schedule must satisfy to match E2.

        The TR floor comes from the live Julia action bounds. The headroom comes
        from the Julia E2 step logic. Fixed TE is owned by this Python action
        wrapper because it is the one that pins TE before passing actions to
        Julia.
        """
        return {
            "tr_lo_floor": float(self._ACT_LO[2]),
            "te_s": float(self._FIXED_TE_S),
            "tr_headroom": float(_env_mod._JL_QMD.e2_tr_headroom(self._env)),
        }

    def _denorm_action(self, action: np.ndarray) -> tuple[np.ndarray, bool]:
        """Map agent output in [-1, 1] to a (physical 4-vector, stop) pair.
        The physical vector is [TI, TE, TR, α_deg].

        When allow_stop, the LAST action component is the stop gate (raw > 0 ⇒
        stop) and is stripped before the physical mapping; otherwise stop=False."""
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        stop = False
        if self._allow_stop:
            stop = bool(a[-1] > 0.0)
            a = a[:-1]
        u = (a + 1.0) / 2.0  # [0, 1]
        if self._fix_te:
            # [TI, TR] (α=90° fixed) or [TI, TR, α]; TE fixed.
            full = np.zeros(4, dtype=np.float32)
            full[0] = self._map_ti(u[0], self._ACT_LO[0], self._ACT_HI[0])
            full[1] = self._FIXED_TE_S
            full[2] = self._ACT_LO[2] + u[1] * (self._ACT_HI[2] - self._ACT_LO[2])
            if self._learn_alpha:
                full[3] = self._ALPHA_LO_DEG + u[2] * (
                    self._ALPHA_HI_DEG - self._ALPHA_LO_DEG)
            else:
                full[3] = 90.0
            return full, stop
        if self._simplified_action:
            full = np.zeros(4, dtype=np.float32)
            full[0] = self._map_ti(u[0], self._ACT_LO[0], self._ACT_HI[0])
            full[1] = self._ACT_LO[1] + u[1] * (self._ACT_HI[1] - self._ACT_LO[1])
            full[2] = self._ACT_LO[2] + u[2] * (self._ACT_HI[2] - self._ACT_LO[2])
            full[3] = 90.0
            return full, stop
        full = self._ACT_LO + u * (self._ACT_HI - self._ACT_LO)
        full[0] = self._map_ti(u[0], self._ACT_LO[0], self._ACT_HI[0])
        return full.astype(np.float32), stop

    def _inv_map_ti(self, ti: float, lo: float, hi: float) -> float:
        """Inverse of _map_ti: physical TI → u01 ∈ [0, 1]. Must stay in sync
        with _map_ti so fixed-protocol baselines hit the TI they request."""
        ti = float(np.clip(ti, lo, hi))
        if self._log_ti_action:
            return float(np.log(ti / lo) / np.log(hi / lo))
        return float((ti - lo) / (hi - lo))

    def physical_to_norm_action(
        self, ti_s: float, tr_s: float, alpha_deg: float = 90.0,
        te_s: Optional[float] = None, stop: bool = False,
    ) -> np.ndarray:
        """Build a normalised action (in this env's action layout) that decodes
        back to the requested physical sequence. Inverse of _denorm_action; used
        by fixed-protocol baselines so they honour fix_te / learn_alpha /
        log_ti_action instead of assuming the full 5-dim linear space."""
        u = np.zeros(self.action_space.shape[0]
                     - (1 if self._allow_stop else 0), dtype=np.float32)
        if self._fix_te:
            # [TI, TR] or [TI, TR, α]; TE fixed, slice fixed.
            u[0] = self._inv_map_ti(ti_s, self._ACT_LO[0], self._ACT_HI[0])
            u[1] = (tr_s - self._ACT_LO[2]) / (self._ACT_HI[2] - self._ACT_LO[2])
            if self._learn_alpha:
                u[2] = ((alpha_deg - self._ALPHA_LO_DEG)
                        / (self._ALPHA_HI_DEG - self._ALPHA_LO_DEG))
        elif self._simplified_action:
            # [TI, TE, TR]; α fixed at 90°.
            te = self._FIXED_TE_S if te_s is None else te_s
            u[0] = self._inv_map_ti(ti_s, self._ACT_LO[0], self._ACT_HI[0])
            u[1] = (te - self._ACT_LO[1]) / (self._ACT_HI[1] - self._ACT_LO[1])
            u[2] = (tr_s - self._ACT_LO[2]) / (self._ACT_HI[2] - self._ACT_LO[2])
        else:
            # Full [TI, TE, TR, α].
            te = self._FIXED_TE_S if te_s is None else te_s
            u[0] = self._inv_map_ti(ti_s, self._ACT_LO[0], self._ACT_HI[0])
            u[1] = (te - self._ACT_LO[1]) / (self._ACT_HI[1] - self._ACT_LO[1])
            u[2] = (tr_s - self._ACT_LO[2]) / (self._ACT_HI[2] - self._ACT_LO[2])
            u[3] = (alpha_deg - self._ACT_LO[3]) / (self._ACT_HI[3] - self._ACT_LO[3])
        norm = (2.0 * np.clip(u, 0.0, 1.0) - 1.0).astype(np.float32)
        if self._allow_stop:
            norm = np.append(norm, np.float32(1.0 if stop else -1.0))
        return norm

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
        phys, stop = self._denorm_action(action)
        obs, reward, done, info_dict = _env_mod._JL_QMD.e2_step_b(
            self._env,
            list(phys.astype(float)),
            bool(stop),
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

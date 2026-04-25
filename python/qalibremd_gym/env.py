"""Gymnasium ↔ Julia bridge for E1.

One Julia subinterpreter is spun up per Python process (lazily, at first
env construction). `reset`/`step` proxy into `QalibreMDPhantom.e1_reset!`
/ `e1_step!`. juliacall handles NumPy ↔ Julia array conversion; we force
`np.float32` on the way out because Stable-Baselines3 expects that dtype
for Box observation spaces.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# Shared Julia module — loaded once per Python process.
_JL: Any = None
_JL_QMD: Any = None


def _ensure_julia(project_dir: Optional[str] = None) -> None:
    """Boot Julia and import QalibreMDPhantom once per process.

    juliacall 0.9.31 requires Julia ≤ 1.11 (its bundled PythonCall does
    not yet load on 1.12). We run it against a dedicated Julia project
    at `python/julia_runtime/` that dev-depends on the repo's
    QalibreMDPhantom, resolved with a juliaup-installed 1.11. The main
    repo project can stay on 1.12 for direct Julia use (tests, E0,
    figure scripts).
    """
    global _JL, _JL_QMD
    if _JL is not None:
        return

    repo_root = Path(__file__).resolve().parents[2]
    runtime_proj = project_dir or str(repo_root / "python" / "julia_runtime")

    # Pick Julia 1.11 from juliaup if installed there.
    j11 = Path.home() / ".julia" / "juliaup" / "julia-1.11.9+0.x64.linux.gnu" / "bin" / "julia"
    if j11.exists():
        os.environ.setdefault("PYTHON_JULIAPKG_EXE", str(j11))
    os.environ.setdefault("PYTHON_JULIAPKG_OFFLINE", "yes")
    os.environ.setdefault("PYTHON_JULIAPKG_PROJECT", runtime_proj)

    from juliacall import Main as jl       # imported lazily
    jl.seval("using QalibreMDPhantom")
    _JL = jl
    _JL_QMD = jl.QalibreMDPhantom


class QalibreMDE1Env(gym.Env):
    """Single-sphere T1 estimation (PLAN.md §4 E1).

    Observation : float32 Box of length `n_adc + 3 + n_actions`
                  = normalised last-block ADC magnitudes,
                    log10(running T1 estimate), blocks-fraction,
                    time-fraction, and per-action play counts / max_blocks.
    Action      : Discrete(n_actions), default n_actions = |TI_set| × |α_set|
                  = 6 × 3 = 18.
    Reward      : -|T̂₁ − T₁_true|/T₁_true per step, minus
                  λ · block_time / budget, plus a terminal bonus if the
                  terminal error is below `success_tol`.
    Episode     : up to `max_blocks` steps OR `time_budget_s` simulated
                  scan-time exceeded.
    """

    metadata = {"render_modes": [], "render_fps": 0}

    def __init__(
        self,
        *,
        rng_seed: int = 0,
        max_blocks: int = 12,
        time_budget_s: float = 20.0,
        lambda_time: float = 1.0,
        terminal_bonus: float = 1.0,
        success_tol: float = 0.03,
        backend: str = "analytical",
        project_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        _ensure_julia(project_dir)

        self._env = _JL_QMD.E1Env(
            rng_seed=rng_seed,
            max_blocks=max_blocks,
            time_budget_s=time_budget_s,
            λ_time=lambda_time,
            terminal_bonus=terminal_bonus,
            success_tol=success_tol,
            backend=_JL.Symbol(backend),
        )
        n_actions = int(_JL_QMD.e1_n_actions(self._env))
        obs_dim   = int(_JL_QMD.e1_obs_dim(self._env))
        self.action_space = spaces.Discrete(n_actions)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

    # ---------------- gymnasium API ----------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,   # noqa: ARG002
    ):
        super().reset(seed=seed)
        if seed is not None:
            obs = _JL_QMD.e1_reset_b(self._env, rng_seed=int(seed))
        else:
            obs = _JL_QMD.e1_reset_b(self._env)
        obs = np.asarray(obs, dtype=np.float32)
        info = {
            "T1_true": float(self._env.T1_true),
            "T2_true": float(self._env.T2_true),
        }
        return obs, info

    def step(self, action):
        a_julia = int(action) + 1          # Python 0-based → Julia 1-based
        obs, reward, done, info_dict = _JL_QMD.e1_step_b(self._env, a_julia)
        obs = np.asarray(obs, dtype=np.float32)
        info = {str(k): _to_py(v) for k, v in info_dict.items()}
        return obs, float(reward), bool(done), False, info

    def render(self):
        # No visual render; pull state via attributes for logging.
        return None

    def close(self):
        pass

    # ---------------- accessors (debug / logging) ----------------

    @property
    def T1_true(self) -> float:
        return float(self._env.T1_true)

    @property
    def T1_est(self) -> float:
        val = float(self._env.T1_est)
        return val

    @property
    def n_blocks(self) -> int:
        return int(self._env.n_blocks)

    @property
    def time_used_s(self) -> float:
        return float(self._env.time_used_s)

    @property
    def action_table(self):
        """Return the list of (TI_s, α_rad) tuples, one per action index."""
        tbl = _JL_QMD.e1_action_table(self._env)
        return [(float(t[0]), float(t[1])) for t in tbl]


def _to_py(x: Any) -> Any:
    """Best-effort Julia → Python scalar / float conversion."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return x

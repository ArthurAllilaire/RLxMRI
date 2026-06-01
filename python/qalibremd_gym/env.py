"""Gymnasium ↔ Julia bridge for E1.

One Julia subinterpreter is spun up per Python process (lazily, at first
env construction). `reset`/`step` proxy into `MRISystemPhantom.e1_reset!`
/ `e1_step!`. juliacall handles NumPy ↔ Julia array conversion; we force
`np.float32` on the way out because Stable-Baselines3 expects that dtype
for Box observation spaces.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Any, Optional

# ── juliapkg env vars must be set before importing juliacall ──────────────────
# juliacall triggers juliapkg on import. If PYTHON_JULIAPKG_PROJECT is not set
# first, juliapkg creates a fresh env at .venv/julia_env/ with just PythonCall
# and MRISystemPhantom is not found. Setting it here ensures our pre-instantiated
# python/julia_runtime/ project is used regardless of import order.
#
# These are set with setdefault so run_e2.sh (which sources .envrc.local first)
# always wins — module-level code only fills in what's missing.
_repo_root = Path(__file__).resolve().parents[2]
_runtime_proj = str(_repo_root / "python" / "julia_runtime")
os.environ.setdefault("PYTHON_JULIAPKG_PROJECT", _runtime_proj)
os.environ.setdefault("PYTHON_JULIAPKG_OFFLINE", "yes")
if "PYTHON_JULIAPKG_EXE" not in os.environ:
    _j11_candidates = (
        glob.glob(str(Path.home() / ".julia" / "juliaup" / "julia-1.11*" / "bin" / "julia")) +
        glob.glob(str(Path.home() / ".juliaup" / "julia-1.11*" / "bin" / "julia"))
    )
    if _j11_candidates:
        os.environ["PYTHON_JULIAPKG_EXE"] = _j11_candidates[0]

# juliacall's multithreading support is experimental and segfault-prone unless
# Julia's signal handlers are disabled — running with JULIA_NUM_THREADS>1 (e.g. to
# multithread the KomaMRI Bloch sim) otherwise crashes SubprocVecEnv workers with a
# bare EOFError. juliacall reads PYTHON_JULIACALL_HANDLE_SIGNALS at import, so set it
# here (before the juliacall import) when threads>1. setdefault → an explicit user
# value still wins. Only enabled for threads>1 because the side effect is that
# Python's Ctrl-C no longer raises KeyboardInterrupt.
def _julia_num_threads() -> int:
    v = os.environ.get("JULIA_NUM_THREADS", "1")
    if v == "auto":
        return os.cpu_count() or 1
    try:                                  # may be "N" or "N,M" (compute,interactive)
        return int(str(v).split(",")[0])
    except ValueError:
        return 1
if _julia_num_threads() > 1:
    os.environ.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes")

# WSL2/Ubuntu's system libcrypto is too old for OpenSSL_jll ≥ 3.3 (libssl needs
# OPENSSL_3.3.0 symbols not present in /lib/x86_64-linux-gnu/libcrypto.so.3).
# glibc reads LD_LIBRARY_PATH at process exec time and caches it, so mutating
# os.environ after Python has started has no effect — we must re-exec ourselves
# with the JLL lib dir on the search path *before* juliacall triggers the
# libjulia → libssl dlopen chain. A sentinel env var prevents an exec loop.
if os.environ.get("_QMD_LD_REEXEC") != "1":
    _ossl_candidates = glob.glob(str(Path.home() / ".julia" / "artifacts" / "*" / "lib" / "libssl.so"))
    if _ossl_candidates:
        _ossl_dir = os.path.dirname(max(_ossl_candidates, key=os.path.getmtime))
        if _ossl_dir not in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
            # Reconstruct the real argv from /proc/self/cmdline — sys.argv loses
            # the `python -c "<code>"` form (the -c content is consumed before
            # sys.argv is built). Fall back to sys.argv on non-Linux.
            try:
                with open("/proc/self/cmdline", "rb") as _f:
                    _argv = _f.read().rstrip(b"\x00").split(b"\x00")
                _argv = [a.decode("utf-8", errors="replace") for a in _argv]
            except OSError:
                import sys as _sys
                _argv = [_sys.executable] + _sys.argv
            _new_env = os.environ.copy()
            _new_env["LD_LIBRARY_PATH"] = f"{_ossl_dir}:{_new_env.get('LD_LIBRARY_PATH', '')}"
            _new_env["_QMD_LD_REEXEC"] = "1"
            os.execvpe(_argv[0], _argv, _new_env)

# juliacall must also be imported before torch (pulled in by stable-baselines3)
# to avoid a segfault on some PyTorch builds.
# See https://github.com/pytorch/pytorch/issues/78829
import juliacall  # noqa: F401

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# Shared Julia module — loaded once per Python process.
_JL: Any = None
_JL_QMD: Any = None


def _ensure_julia(project_dir: Optional[str] = None,
                  use_gpu: bool = False) -> None:
    """Boot Julia and import MRISystemPhantom once per process.

    juliacall 0.9.31 requires Julia ≤ 1.11 (its bundled PythonCall does
    not yet load on 1.12). We run it against a dedicated Julia project
    at `python/julia_runtime/` that dev-depends on the repo's
    MRISystemPhantom, resolved with a juliaup-installed 1.11. The main
    repo project can stay on 1.12 for direct Julia use (tests, E0,
    figure scripts).

    When use_gpu is set, also `using CUDA` so KomaMRI registers the CUDA
    backend — without a backend package loaded, sim_params["gpu"]=true
    silently falls back to the CPU. Boot is once-per-process, so the
    first env's use_gpu wins (all envs in a run share it). If CUDA is
    unavailable we warn and continue on the CPU rather than crash, so the
    same code runs on a CPU-only box.
    """
    global _JL, _JL_QMD
    if _JL is not None:
        return

    from juliacall import Main as jl
    jl.seval("using MRISystemPhantom")
    if use_gpu:
        try:
            jl.seval("using CUDA")
        except Exception as e:  # noqa: BLE001
            import warnings
            warnings.warn(
                f"use_gpu=True but `using CUDA` failed ({e}); KomaMRI will "
                "fall back to the CPU. Add CUDA to python/julia_runtime.",
                RuntimeWarning,
            )
    _JL = jl
    _JL_QMD = jl.MRISystemPhantom


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

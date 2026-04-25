"""Smoke-test the Python Gymnasium wrapper for E1.

Run: `python -m pytest python/test_wrapper.py -q`
(or just `python python/test_wrapper.py` for ad-hoc use without pytest).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from qalibremd_gym.env import QalibreMDE1Env


def test_spaces_and_reset():
    env = QalibreMDE1Env(rng_seed=0)
    assert env.action_space.n == 18
    assert env.observation_space.shape == (18 + 3 + 64,)   # default
    obs, info = env.reset(seed=42)
    assert obs.shape == env.observation_space.shape
    assert obs.dtype == np.float32
    assert 20e-3 <= info["T1_true"] <= 2.0
    assert 0.3 * info["T1_true"] - 1e-9 <= info["T2_true"] <= info["T1_true"] + 1e-9


def test_step_and_terminate():
    env = QalibreMDE1Env(rng_seed=1, max_blocks=6)
    env.reset(seed=100)
    n_done = 0
    for step in range(6):
        obs, r, done, trunc, info = env.step(step % env.action_space.n)
        assert obs.shape == env.observation_space.shape
        assert isinstance(r, float)
        assert isinstance(done, bool)
        assert "T1_true" in info and "T1_est" in info and "err" in info
        if done:
            n_done += 1
    assert n_done == 1


def test_determinism_same_seed():
    """Two envs, same reset seed → same T1_true and same first signal."""
    env_a = QalibreMDE1Env(rng_seed=7)
    env_b = QalibreMDE1Env(rng_seed=7)
    obs_a, info_a = env_a.reset(seed=11)
    obs_b, info_b = env_b.reset(seed=11)
    assert info_a["T1_true"] == info_b["T1_true"]
    oa, _, _, _, ia = env_a.step(2)      # TI=10ms, α=180°
    ob, _, _, _, ib = env_b.step(2)
    np.testing.assert_allclose(oa, ob)
    assert ia["T1_est"] == ib["T1_est"]


def _run_ir_ladder(env: QalibreMDE1Env, seed: int) -> float:
    """Play IR (α=180°) over six TIs, return terminal err."""
    env.reset(seed=seed)
    ir_actions = [2, 5, 8, 11, 14, 17]     # indices for α=180° (0-based)
    last_err = 1.0
    for a in ir_actions * 2:
        _, _, done, _, info = env.step(a)
        last_err = info["err"]
        if done:
            break
    return last_err


def test_ir_ladder_accuracy():
    env = QalibreMDE1Env(rng_seed=99, max_blocks=12)
    errs = [_run_ir_ladder(env, seed=s) for s in range(20)]
    mape = float(np.mean(errs))
    assert mape < 0.1, f"IR ladder MAPE too high: {mape:.4f}"


def main():
    tests = [
        test_spaces_and_reset,
        test_step_and_terminate,
        test_determinism_same_seed,
        test_ir_ladder_accuracy,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: {e}")
    if failed:
        raise SystemExit(f"{failed} test(s) failed")
    print(f"all {len(tests)} tests passed")


if __name__ == "__main__":
    main()

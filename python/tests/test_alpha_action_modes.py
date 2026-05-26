"""Smoke tests for the fix_te / learn_alpha action modes (see ALPHA_DOF.md).

The pure-mapping tests construct the env via __new__ and set only the attributes
_denorm_action needs, so they run without starting Julia. The env-construction
tests (action_space shapes, the fail-fast guard) need juliacall and are skipped
if the bridge can't be initialised.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qalibremd_gym.env_e2 import QalibreMDE2Env


def _make_bare(*, fix_te, learn_alpha, simplified=False, log_ti=False):
    """A QalibreMDE2Env with only the attributes _denorm_action reads, no Julia."""
    env = QalibreMDE2Env.__new__(QalibreMDE2Env)
    env._fix_te = fix_te
    env._learn_alpha = learn_alpha
    env._simplified_action = simplified
    env._log_ti_action = log_ti
    return env


# 2-dim fix_te mode denorms to [TI, 0.020, TR, 90.0, 0.0] with TI/TR in bounds.
def test_denorm_2dim_fix_te():
    env = _make_bare(fix_te=True, learn_alpha=False)
    # mid action [0,0] → u=0.5 for both TI and TR
    full = env._denorm_action(np.array([0.0, 0.0], dtype=np.float32))
    assert full.shape == (5,)
    assert full[1] == pytest.approx(0.020)        # TE fixed
    assert full[3] == pytest.approx(90.0)         # α fixed at 90
    assert full[4] == pytest.approx(0.0)          # slice_z
    # TI within bounds, TR within bounds
    assert QalibreMDE2Env._ACT_LO[0] <= full[0] <= QalibreMDE2Env._ACT_HI[0]
    assert QalibreMDE2Env._ACT_LO[2] <= full[2] <= QalibreMDE2Env._ACT_HI[2]


# 3-dim learn_alpha mode: α hits 5°/90° at the action extremes, stays in [5,90]
# for interior actions, TE pinned to 0.020.
def test_denorm_3dim_learn_alpha_bounds():
    env = _make_bare(fix_te=True, learn_alpha=True)
    lo = env._denorm_action(np.array([-1.0, -1.0, -1.0], dtype=np.float32))
    hi = env._denorm_action(np.array([1.0, 1.0, 1.0], dtype=np.float32))
    assert lo[1] == pytest.approx(0.020) and hi[1] == pytest.approx(0.020)
    assert lo[3] == pytest.approx(QalibreMDE2Env._ALPHA_LO_DEG)   # α floor = 5°
    assert hi[3] == pytest.approx(QalibreMDE2Env._ALPHA_HI_DEG)   # α ceil = 90°
    # α stays inside [5, 90] for an interior action
    mid = env._denorm_action(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    assert 5.0 <= mid[3] <= 90.0


# Existing modes are untouched: without fix_te, simplified still pins α=90°.
def test_fix_te_unchanged_existing_modes():
    # Without fix_te, the simplified mode still pins α=90 and TE-from-range.
    env = _make_bare(fix_te=False, learn_alpha=False, simplified=True)
    full = env._denorm_action(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    assert full.shape == (5,)
    assert full[3] == pytest.approx(90.0)


# ── env-construction tests (need juliacall) ────────────────────────────────────

_RUN_JULIA = os.environ.get("RUN_JULIA_TESTS") == "1"


# learn_alpha without fix_te is rejected at construction (fail-fast, pre-Julia).
def test_learn_alpha_requires_fix_te_fails_fast():
    # This guard runs before any Julia work, so it must raise regardless.
    with pytest.raises(ValueError):
        QalibreMDE2Env(learn_alpha=True, fix_te=False)


# action_space is (2,) for fix_te and (3,) for learn_alpha (needs the Julia bridge).
@pytest.mark.skipif(not _RUN_JULIA, reason="set RUN_JULIA_TESTS=1 to run")
def test_action_space_shapes():
    e2 = QalibreMDE2Env(fix_te=True, learn_alpha=False, max_blocks=2,
                        time_budget_s=1e6, Nfe=8, Npe=4)
    assert e2.action_space.shape == (2,)
    e3 = QalibreMDE2Env(fix_te=True, learn_alpha=True, max_blocks=2,
                        time_budget_s=1e6, Nfe=8, Npe=4)
    assert e3.action_space.shape == (3,)

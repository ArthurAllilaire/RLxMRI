"""Pure-Python unit tests for the multi-fidelity switch logic (no Julia / no SB3).

Run: python -m pytest python/tests/test_mf_switch.py
"""

import math
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mf_switch import (  # noqa: E402
    SwitchThresholds, SwitchState, DecisionPoint, decide_switch,
    relative_target_gain, score, should_run_lookahead, slope_collapse,
    spearman, target_slope_per_cost,
)


def _state(mape_f, mape_H, p90_f=None, p90_H=None):
    p90_f = p90_f if p90_f is not None else mape_f
    p90_H = p90_H if p90_H is not None else mape_H
    st = SwitchState()
    for i, (f, h, pf, ph) in enumerate(zip(mape_f, mape_H, p90_f, p90_H)):
        st.add(DecisionPoint(step=1000 * (i + 1), mape_f=f, mape_H=h,
                             p90_f=pf, p90_H=ph))
    return st


# ── primitives ──────────────────────────────────────────────────────────────

def test_score_monotone_and_clamped():
    assert score(0.10) < score(0.01)            # lower MAPE → higher score
    assert score(1e-9, floor=1e-3) == score(1e-3, floor=1e-3)   # floor clamp
    assert score(2.0) == pytest.approx(0.0)     # clamp at 1.0


def test_spearman_perfect():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert spearman(x, x) == pytest.approx(1.0)
    assert spearman(x, x[::-1]) == pytest.approx(-1.0)
    assert math.isnan(spearman(np.array([1.0]), np.array([1.0])))


def test_target_slope_insufficient_history_is_inf():
    assert target_slope_per_cost([1.0, 2.0], window=3, sec_per_step=1.0,
                                 steps_per_window=10) == float("inf")


def test_target_slope_value():
    # P rose 0.3 over the window; cost = 2 s/step * 10 steps = 20 s.
    s = target_slope_per_cost([1.0, 1.1, 1.2, 1.3], window=3,
                              sec_per_step=2.0, steps_per_window=10)
    assert s == pytest.approx(0.3 / 20.0)


def test_relative_target_gain_value():
    assert relative_target_gain([1.0, 1.1, 1.2, 1.3], window=3) == pytest.approx(0.3)
    assert relative_target_gain([1.0, 1.1], window=3) == float("inf")


def test_slope_collapse_needs_min_points():
    collapsed, reason = slope_collapse([0.1, 0.02], 0.25, 3)
    assert not collapsed and reason == ""


def test_slope_collapse_triggers_against_previous_best():
    collapsed, reason = slope_collapse(
        [float("inf"), 0.10, 0.08, 0.015], collapse_frac=0.25, min_points=3)
    assert collapsed and reason == "slope_collapse"


def test_lookahead_near_plateau_triggers_before_hard_plateau():
    th = SwitchThresholds(plateau_window=3, plateau_delta=0.02,
                          near_plateau_frac=2.0, min_lookahead_points=4)
    # Relative gain = (1.03 - 1.00) / 1.00 = 0.03, above hard plateau
    # threshold 0.02 but inside the near-plateau trigger band 0.04.
    st = SwitchState([
        DecisionPoint(1, 0.3, 0.10, 0.3, 0.10),
        DecisionPoint(2, 0.3, 10 ** -1.01, 0.3, 10 ** -1.01),
        DecisionPoint(3, 0.3, 10 ** -1.02, 0.3, 10 ** -1.02),
        DecisionPoint(4, 0.3, 10 ** -1.03, 0.3, 10 ** -1.03),
    ])
    run, reason = should_run_lookahead(st, th, slope_hist=[0.1, 0.09, 0.08, 0.07])
    assert run and reason == "near_plateau"


def test_lookahead_ignores_insufficient_history():
    th = SwitchThresholds(min_lookahead_points=4)
    st = _state([0.3, 0.3, 0.3], [0.2, 0.19, 0.18])
    run, reason = should_run_lookahead(st, th, slope_hist=[0.1, 0.01, 0.001])
    assert not run and reason == ""


# ── decision rule ───────────────────────────────────────────────────────────

def test_min_steps_gate_blocks_promotion():
    th = SwitchThresholds(min_steps=10_000)
    st = _state([0.5] * 5, [0.5] * 5)
    promote, reason = decide_switch(st, th, stage_steps=2_000, below_reserve=False)
    assert not promote and reason == "min_steps_gate"


def test_budget_reserve_overrides_everything():
    th = SwitchThresholds(min_steps=10_000)
    st = _state([0.5], [0.5])
    promote, reason = decide_switch(st, th, stage_steps=0, below_reserve=True)
    assert promote and reason == "budget_reserve"


def test_target_plateau_fires_on_flat_fullsim():
    # Full-sim MAPE flat across >window points ⇒ P_H plateau ⇒ promote.
    th = SwitchThresholds(plateau_window=3, plateau_delta=0.02, min_steps=0)
    st = _state([0.30, 0.29, 0.28, 0.27, 0.26],   # cheap sim keeps "improving"
                [0.20, 0.20, 0.20, 0.20, 0.20])    # full sim flat
    promote, reason = decide_switch(st, th, stage_steps=50_000, below_reserve=False)
    assert promote and reason == "target_plateau"


def test_rising_fullsim_holds():
    # Full sim still improving, low bias, well-ranked ⇒ keep training.
    th = SwitchThresholds(plateau_window=3, plateau_delta=0.02, min_steps=0,
                          bias_mape_abs=0.05, bias_p90_abs=0.10)
    m = [0.30, 0.25, 0.20, 0.15, 0.10]
    st = _state(m, [x + 0.01 for x in m])   # cheap tracks full closely
    promote, reason = decide_switch(st, th, stage_steps=50_000, below_reserve=False)
    assert not promote and reason == ""


def test_rank_breakdown_needs_min_ckpts():
    # mape_f rises while mape_H falls → anti-correlated ranking. Full sim is
    # IMPROVING so no plateau; bias kept under tolerance so the trigger is rank.
    th = SwitchThresholds(plateau_window=3, plateau_delta=0.02, min_steps=0,
                          rankcorr_min_ckpts=4, rankcorr_thresh=0.7,
                          bias_mape_abs=1.0, bias_p90_abs=1.0)
    mape_H = [0.40, 0.30, 0.20, 0.10]      # improving (no plateau)
    mape_f = [0.10, 0.20, 0.30, 0.40]      # anti-correlated

    # With only 3 points: not enough to trust rank-corr → must NOT fire on rank.
    st3 = _state(mape_f[:3], mape_H[:3])
    promote3, reason3 = decide_switch(st3, th, stage_steps=50_000, below_reserve=False)
    assert reason3 != "rank_breakdown"

    # With 4 points: rank-corr is strongly negative → fire.
    st4 = _state(mape_f, mape_H)
    promote4, reason4 = decide_switch(st4, th, stage_steps=50_000, below_reserve=False)
    assert promote4 and reason4 == "rank_breakdown"


def test_bias_intolerance_fires():
    # Few points (no plateau, no rank-corr), full sim improving, but the cheap
    # sim is wildly optimistic vs full → bias trigger.
    th = SwitchThresholds(plateau_window=3, min_steps=0,
                          bias_mape_abs=0.01, bias_p90_abs=0.02)
    st = _state([0.05, 0.04], [0.30, 0.25])   # 21% mean gap
    promote, reason = decide_switch(st, th, stage_steps=50_000, below_reserve=False)
    assert promote and reason == "bias_intolerance"

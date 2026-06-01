"""Pure multi-fidelity switch logic for the E2 curriculum (no Julia / no SB3).

Isolated here so the promotion rule can be unit-tested on synthetic metric
histories without spinning up KomaMRI. `train_e2_mf.py` owns the env/PPO
plumbing and calls `decide_switch` at each decision point.

Naming note (deliberate): the implemented quantity is `target_slope_per_cost`,
an *information-gain-per-cost-inspired heuristic* — NOT the posterior
information gain of Sifaou & Simeone 2025 (we do not implement their
CQL/bootstrap-ensemble posterior). The switch is driven by held-out **full-sim**
validation, never by the current cheap simulator's own score.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ── thresholds ──────────────────────────────────────────────────────────────

@dataclass
class SwitchThresholds:
    mape_floor: float = 1e-3        # clamp for P = -log10(clamp(mape, floor, 1))
    plateau_window: int = 3         # # of decision points spanned by the slope
    plateau_delta: float = 0.02     # relative ΔP_H below this over the window ⇒ plateau
    rankcorr_min_ckpts: int = 4     # need ≥ this many points before trusting rank-corr
    rankcorr_thresh: float = 0.7    # Spearman(mape_f, mape_H) below this ⇒ ranking breakdown
    bias_mape_abs: float = 0.01     # |mean MAPE gap| (fraction) above this ⇒ bias intolerance
    bias_p90_abs: float = 0.02      # |p90 MAPE gap| (fraction) above this ⇒ bias intolerance
    min_steps: int = 0              # don't promote before this many env steps in the stage
    slope_collapse_frac: float = 0.25  # recent slope below this × previous best ⇒ look ahead
    near_plateau_frac: float = 2.0     # look ahead when rel ΔP_H is near the plateau threshold
    min_lookahead_points: int = 4      # need enough checkpoints before lookahead triggers


# ── primitives ──────────────────────────────────────────────────────────────

def score(mape: float, floor: float = 1e-3) -> float:
    """P = -log10(clamp(mape, floor, 1)). Higher is better; the floor models the
    fidelity's accuracy floor so a saturated cheap sim reads as a flat P."""
    return -math.log10(min(max(float(mape), floor), 1.0))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation via Pearson on ranks (no scipy dependency).
    Returns NaN for <2 points or zero-variance input."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2 or b.size < 2:
        return float("nan")
    ra = _rankdata(a)
    rb = _rankdata(b)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Average-rank of x (ties share the mean rank), matching scipy.rankdata."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)
    # average tied groups
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    return (sums / counts)[inv]


def target_slope_per_cost(p_hist: list[float], window: int,
                          sec_per_step: float, steps_per_window: float) -> float:
    """Δ⁺P_H over the last `window` decision points, divided by the wallclock the
    window cost (≈ sec_per_step · steps_per_window). An IG-per-cost-heuristic:
    improvement in the full-sim target per second at this fidelity."""
    if len(p_hist) <= window:
        return float("inf")          # not enough history → keep training here
    gain = max(0.0, p_hist[-1] - p_hist[-1 - window])
    cost = max(1e-9, float(sec_per_step) * float(steps_per_window))
    return gain / cost


def relative_target_gain(p_hist: list[float], window: int) -> float:
    """Relative ΔP_H over `window` decision points.

    This is the dimensionless target-plateau statistic used by `decide_switch`.
    It is factored out so the V2 lookahead trigger can fire when the old
    fallback is *near* plateau without duplicating the arithmetic in the trainer.
    """
    if len(p_hist) <= window:
        return float("inf")
    base = float(p_hist[-1 - window])
    return (float(p_hist[-1]) - base) / max(abs(base), 1e-6)


def slope_collapse(slope_hist: list[float], collapse_frac: float,
                   min_points: int) -> tuple[bool, str]:
    """Return true when recent finite slope has collapsed vs previous best.

    `slope_hist` contains target-slope-per-cost measurements at decision
    points. Infinite warm-up entries and NaNs are ignored. The latest finite
    slope is compared to the best earlier finite slope, which makes the trigger
    scale-free across hardware and resolution.
    """
    if len(slope_hist) < int(min_points):
        return False, ""
    finite = [float(s) for s in slope_hist if math.isfinite(float(s))]
    if len(finite) < 2:
        return False, ""
    latest = finite[-1]
    previous_best = max(finite[:-1], default=float("nan"))
    if not math.isfinite(previous_best) or previous_best <= 0.0:
        return False, ""
    if latest < float(collapse_frac) * previous_best:
        return True, "slope_collapse"
    return False, ""


def should_run_lookahead(state: SwitchState, th: SwitchThresholds,
                         slope_hist: list[float]) -> tuple[bool, str]:
    """Decide whether a next-fidelity lookahead probe is worth spending.

    The trigger is intentionally narrower than promotion itself: it only asks
    whether we should pay for a clone-and-train lookahead. Promotion remains a
    separate decision based on lookahead superiority or existing fallbacks.
    """
    if len(state.points) < th.min_lookahead_points:
        return False, ""

    collapsed, reason = slope_collapse(
        slope_hist, th.slope_collapse_frac, th.min_lookahead_points)
    if collapsed:
        return True, reason

    rel = relative_target_gain(state.p_hist, th.plateau_window)
    if math.isfinite(rel) and rel >= 0.0 and rel < th.near_plateau_frac * th.plateau_delta:
        return True, "near_plateau"
    return False, ""


# ── decision ────────────────────────────────────────────────────────────────

@dataclass
class DecisionPoint:
    step: int
    mape_f: float          # mean MAPE on the CURRENT-fidelity probe (fraction)
    mape_H: float          # mean MAPE on the FULL-sim probe (fraction)
    p90_f: float
    p90_H: float

    @property
    def P_H(self) -> float:
        return score(self.mape_H)


@dataclass
class SwitchState:
    points: list[DecisionPoint] = field(default_factory=list)

    def add(self, p: DecisionPoint) -> None:
        self.points.append(p)

    @property
    def p_hist(self) -> list[float]:
        return [p.P_H for p in self.points]


def decide_switch(state: SwitchState, th: SwitchThresholds, *,
                  stage_steps: int, below_reserve: bool) -> tuple[bool, str]:
    """Return (should_promote, reason). Reasons (priority order):

    - "budget_reserve" : remaining wallclock has fallen to the full-sim reserve.
    - "min_steps_gate" : (NOT a promotion) too early — never promote yet.
    - "target_plateau" : full-sim score P_H has stopped improving (primary).
    - "rank_breakdown" : Spearman(mape_f, mape_H) below guardrail (≥min ckpts).
    - "bias_intolerance": |mean or p90 MAPE gap| vs full exceeds tolerance.
    - ""               : keep training at the current fidelity.
    """
    # Budget reserve overrides the min-steps gate: if we're out of cheap-stage
    # budget we must hand the rest to the full sim regardless.
    if below_reserve:
        return True, "budget_reserve"

    if stage_steps < th.min_steps:
        return False, "min_steps_gate"

    pts = state.points
    if not pts:
        return False, ""

    # 1) Target plateau — full-sim P_H slope ~ 0 over the window.
    p_hist = state.p_hist
    if len(p_hist) > th.plateau_window:
        if relative_target_gain(p_hist, th.plateau_window) < th.plateau_delta:
            return True, "target_plateau"

    # 2) Ranking breakdown — only once we have enough checkpoints to trust it.
    if len(pts) >= th.rankcorr_min_ckpts:
        rc = spearman(np.array([p.mape_f for p in pts]),
                      np.array([p.mape_H for p in pts]))
        if not math.isnan(rc) and rc < th.rankcorr_thresh:
            return True, "rank_breakdown"

    # 3) Bias intolerance — cheap-sim eval has drifted too far from full sim.
    last = pts[-1]
    if (abs(last.mape_f - last.mape_H) > th.bias_mape_abs or
            abs(last.p90_f - last.p90_H) > th.bias_p90_abs):
        return True, "bias_intolerance"

    return False, ""

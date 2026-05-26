"""Shared fixed-schedule helpers for the E2 baselines.

Single definition of the log-spaced TI grid, used by both
`baseline_e2.py` (the fixed-schedule yardsticks) and `eval_e2.py`
(the fixed-grid baseline run) so the grid can't drift between them.
"""

from __future__ import annotations

import numpy as np


def log_ti_grid(lo: float = 0.05, hi: float = 3.0, n: int = 7) -> np.ndarray:
    """Log-spaced inversion-time grid in seconds (n points over [lo, hi])."""
    return np.exp(np.linspace(np.log(lo), np.log(hi), n))

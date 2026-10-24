"""Recompute the TI-versus-running-T1-estimate correlation for the RL arms.

The report (AdaptiveRLMRI.tex:1290-1291) quotes r = -0.12 for the TI-histogram
arm and r = +0.78 for the LSTM arm. Those numbers come from
python/diagnose_e2.py:plot_ti_vs_t1est: Pearson r between log(TI) and
log(mean of the five running T1 estimates the policy saw at decision time),
pooled over every recorded block of the 24 validation scans (seed 500000).
That recipe keeps the rejected final proposals, and it keeps block 2, whose
decision is made before any fit exists: with one block acquired, every running
estimate is the neutral prior sqrt(0.01 * 3.0) = 0.173 s (julia/rl/e2.jl:723).
This script reproduces the recipe and reports the variants without those entries.

Run from the RLxMRI repo root:
  .venv/bin/python ismrm_latex/vancouver/scripts/fig3_ti_correlation.py
"""

from __future__ import annotations

import json

import numpy as np

import figstyle

RUNS = figstyle.ROOT / "runs" / "e2"
ARMS = {
    "hist": "mf_runB_5sphere_hist_560s_gpu",
    "nomem": "mf_runB_5sphere_560s_gpu",
    "sigma": "mf_runB_5sphere_sigma_560s_gpu",
    "lstm": "mf_runB_5sphere_lstm_560s_gpu",
}
# Running estimates exist only once two blocks have been fitted (julia/rl/e2.jl:723).
FIRST_FITTED_DECISION = 2


def load_episodes(run: str) -> list[dict]:
    return json.loads((RUNS / run / "global_best" / "diagnostics" / "episodes.json").read_text())


def budget_s_and_npe(run: str) -> tuple[float, int]:
    kw = json.loads((RUNS / run / "run_config.json").read_text())["base_env_kwargs"]
    return float(kw["time_budget_s"]), int(kw["Npe"])


def n_executed(ep: dict, budget_s: float, npe: int) -> int:
    """Blocks actually acquired: the budget guard rejects a final proposal that overruns."""
    n = len(ep["TI"])
    return n - 1 if npe * sum(ep["TR"]) > budget_s else n


def log_pearson(episodes: list[dict], budget_s: float, npe: int,
                drop_rejected: bool, drop_prior: bool) -> tuple[float, int]:
    xs, ys = [], []
    for ep in episodes:
        n = n_executed(ep, budget_s, npe) if drop_rejected else len(ep["TI"])
        start = FIRST_FITTED_DECISION if drop_prior else 0
        xs += ep["T1_est_at_decision"][start:n]
        ys += ep["TI"][start:n]
    x, y = np.asarray(xs, float), np.asarray(ys, float)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    return float(np.corrcoef(np.log(x[ok]), np.log(y[ok]))[0, 1]), int(ok.sum())


def main() -> None:
    variants = {
        "diagnose_e2 recipe (all recorded blocks)": (False, False),
        "executed blocks only": (True, False),
        "executed blocks decided on a fitted estimate": (True, True),
    }
    results = {}
    for arm, run in ARMS.items():
        path = RUNS / run / "global_best" / "diagnostics" / "episodes.json"
        if not path.exists():
            continue
        episodes = load_episodes(run)
        budget_s, npe = budget_s_and_npe(run)
        results[arm] = {}
        for name, (drop_rejected, drop_prior) in variants.items():
            r, n = log_pearson(episodes, budget_s, npe, drop_rejected, drop_prior)
            results[arm][name] = {"pearson_log_r": r, "n_blocks": n}
            print(f"{arm:6s} {name:46s} r = {r:+.3f}  (n = {n})")
        prior = {round(ep["T1_est_at_decision"][1], 6) for ep in episodes}
        results[arm]["block2_T1_est_at_decision_values_s"] = sorted(prior)

    out = figstyle.FIGURES / "fig3_ti_correlation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

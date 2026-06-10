"""Multi-fidelity curriculum trainer for E2 (C2: scalable sim-in-the-loop RL).

Trains one PPO agent up a ladder of forward-model fidelities — cheap-but-biased
to slow-but-true — warm-starting each stage from the previous and switching
fidelity by a principled rule rather than a hand-tuned schedule:

    analytic  →  cached_perline water  →  full Bloch        (default plan)

Coarse-water intermediate stages are also available:
    cached3   = cached_perline water with water_voxel_size_mm=3
    full3     = full-Bloch water with water_voxel_size_mm=3

By default this runs in one process (n_envs=1 / DummyVecEnv) so the in-process
Julia (juliacall) stays warm across stages. On CPU, `--n-envs > 1` uses
SubprocVecEnv: each worker owns a separate Julia runtime. Julia/Koma threads are
controlled by environment variables set before Python starts.

The switch is driven by held-out **full-Bloch** validation, never the current
cheap simulator's own score (see mf_switch.py and report/e2_runs/multi_fidelity.md).

Example (smoke):
    PYTHON_JULIAPKG_OFFLINE=yes PYTHON_JULIACALL_HANDLE_SIGNALS=yes \
    PYTHON_JULIACALL_THREADS=6 JULIA_NUM_THREADS=6 \
    PYTHON_JULIAPKG_EXE=~/.julia/juliaup/julia-1.11.9+0.x64.linux.gnu/bin/julia \
    python python/train_e2_mf.py --out runs/e2/mf_smoke --multi-fidelity \
      --mf-plan analytic,cached3,full3,full --reward-mode delta_log_mape \
      --fix-te --learn-alpha --n-envs 2 \
      --field T15 --Nfe 16 --Npe 8 --voxel-mm 3 --mf-budget-hours 0.2 \
      --mf-min-steps 2000 --mf-max-steps 6000 --mf-decision-rollouts 4

Run A-equivalent CPU V2 curriculum (no GPU, 9h wall budget):
    PYTHON_JULIAPKG_OFFLINE=yes PYTHON_JULIACALL_HANDLE_SIGNALS=yes \
    PYTHON_JULIACALL_THREADS=4 JULIA_NUM_THREADS=4 \
    PYTHON_JULIAPKG_EXE=~/.julia/juliaup/julia-1.11.9+0.x64.linux.gnu/bin/julia \
      PYTHONUNBUFFERED=1 python -u python/train_e2_mf.py \
        --out runs/e2/mf_v2_runA_cpu_9h \
        --multi-fidelity --mf-plan analytic,cached3,full3,full \
        --reward-mode delta_log_mape --mape-alpha 1.0 \
        --fix-te --learn-alpha \
        --n-envs 1 \
        --field T15 --time-budget 240 --max-blocks 20 \
        --mf-budget-hours 9 --mf-full-reserve-frac 0.20 \
        --mf-min-steps 8192,32768,32768,0 \
        --mf-max-steps 50000,125000,125000,300000 \
        --n-steps 512 --batch-size 64 \
        --eval-interval 10000 --eval-episodes 8 \
        --mf-decision-rollouts 4 --mf-probe-episodes-full 4 \
        --mf-global-best-episodes 12 \
        --mf-use-lookahead --mf-lookahead-rollouts 1 \
        --mf-lookahead-margin 1.15 --mf-slope-collapse-frac 0.25 \
        2>&1 | tee runs/e2/mf_v2_runA_cpu_9h/run.log
"""

from __future__ import annotations

import copy
import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mf_switch import (
    SwitchThresholds, SwitchState, DecisionPoint, decide_switch,
    relative_target_gain, score, should_run_lookahead, target_slope_per_cost,
)
from qalibremd_gym.env_e2 import QalibreMDE2Env
from e2_train_common import (
    build_model, build_vec_env, rollout_eval, summarise_eval,
    E2EvalCallback, ProgressETACallback,
)
from e2_config import add_e2_env_args, e2_env_kwargs
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np


# ── fidelity registry ───────────────────────────────────────────────────────
# Each entry overrides ONLY the forward-model / water knobs on the base env
# kwargs; everything else (obs/action layout, reward, noise, resolution) is held
# fixed across the curriculum so warm-start weights stay compatible.
FIDELITIES: dict[str, dict] = {
    "analytic": {"forward_model": "analytic"},
    "dry":      {"forward_model": "bloch", "water_model": "bloch", "include_water": False},
    "cached":   {"forward_model": "bloch", "water_model": "cached_perline", "include_water": True},
    "cached3":  {"forward_model": "bloch", "water_model": "cached_perline", "include_water": True,
                 "water_voxel_size_mm": 3.0},
    "full3":    {"forward_model": "bloch", "water_model": "bloch", "include_water": True,
                 "water_voxel_size_mm": 3.0},
    "full":     {"forward_model": "bloch", "water_model": "bloch", "include_water": True},
}
# Per-fidelity ranking guardrail: how faithfully the cheap sim must rank policies
# like full sim before we trust it (looser for the most-biased analytic stage).
RANKCORR_THRESH = {
    "analytic": 0.7, "dry": 0.75, "cached": 0.8, "cached3": 0.8,
    "full3": 0.9, "full": 1.0,
}


@dataclass
class FidelitySpec:
    name: str
    env_overrides: dict
    min_steps: int
    max_steps: int
    is_final: bool = False
    thresholds: SwitchThresholds = field(default_factory=SwitchThresholds)


# ── callbacks ───────────────────────────────────────────────────────────────

class WallBudgetCallback(BaseCallback):
    """Hard stop when the global wallclock deadline is reached (caps the whole
    curriculum at --mf-budget-hours regardless of step counts)."""

    def __init__(self, deadline: float, verbose: int = 0):
        super().__init__(verbose)
        self.deadline = deadline

    def _on_step(self) -> bool:
        if time.time() >= self.deadline:
            if self.verbose:
                print("[MF] global wall budget exhausted — stopping stage")
            return False
        return True


class FullStageEarlyStopCallback(BaseCallback):
    """Stop final full-Bloch training after repeated eval non-improvement.

    The curriculum already promotes out of cheap fidelities when full-sim progress
    plateaus. Once the stage *is* the target full simulator, there is nowhere to
    promote, so this callback uses the existing full-stage eval history to avoid
    spending the remaining wallclock on circular PPO updates.
    """

    def __init__(self, *, eval_cb: E2EvalCallback, stage: str,
                 history_path: Path, patience: int, min_evals: int,
                 min_steps: int, delta_pct: float, verbose: int = 1):
        super().__init__(verbose)
        self.eval_cb = eval_cb
        self.stage = stage
        self.history_path = history_path
        self.patience = int(patience)
        self.min_evals = int(min_evals)
        self.min_steps = int(min_steps)
        self.delta_pct = float(delta_pct)
        self._stage_start = 0
        self._seen_evals = 0
        self._best_mape = float("inf")
        self._best_eval_idx = -1

    def _on_training_start(self) -> None:
        self._stage_start = self.num_timesteps
        hist = self.eval_cb.history
        self._seen_evals = len(hist)
        for idx, row in enumerate(hist):
            mape = float(row.get("mape_pct", float("inf")))
            if np.isfinite(mape) and mape < self._best_mape - self.delta_pct:
                self._best_mape = mape
                self._best_eval_idx = idx

    def _on_step(self) -> bool:
        if self.patience <= 0:
            return True

        hist = self.eval_cb.history
        if len(hist) <= self._seen_evals:
            return True
        self._seen_evals = len(hist)

        latest = hist[-1]
        mape = float(latest.get("mape_pct", float("inf")))
        eval_idx = self._seen_evals - 1
        if np.isfinite(mape) and mape < self._best_mape - self.delta_pct:
            self._best_mape = mape
            self._best_eval_idx = eval_idx
            return True

        stage_steps = self.num_timesteps - self._stage_start
        evals_since_best = eval_idx - self._best_eval_idx
        if (self._seen_evals >= self.min_evals and
                stage_steps >= self.min_steps and
                evals_since_best >= self.patience):
            entry = {
                "stage": self.stage,
                "kind": "full_early_stop",
                "step": int(self.num_timesteps),
                "wall_s": time.time(),
                "current_mape_pct": mape,
                "best_mape_pct": self._best_mape,
                "evals_seen": int(self._seen_evals),
                "evals_since_best": int(evals_since_best),
                "patience": int(self.patience),
                "min_evals": int(self.min_evals),
                "min_steps": int(self.min_steps),
                "delta_pct": float(self.delta_pct),
            }
            _append(self.history_path, entry)
            if self.verbose:
                print(f"[MF full early-stop] no MAPE improvement > "
                      f"{self.delta_pct:.3g} pp for {evals_since_best} evals "
                      f"(best={self._best_mape:.2f}%, current={mape:.2f}%)")
            return False
        return True


class GlobalBestFullSim:
    """Track the best policy seen on the target full-Bloch simulator.

    Per-stage best checkpoints are selected on each stage's own eval env. For a
    multi-fidelity curriculum that is not enough: a cached-stage policy can be the
    best policy under the target full simulator even if later stages regress. This
    tracker uses cheap full-Bloch probes only as a screen, then re-evaluates
    candidates on a larger held-out confirmation set before overwriting the
    run-level best checkpoint.
    """

    def __init__(self, best_dir: Path, *, target_env_kwargs: dict,
                 target_env: QalibreMDE2Env, seed_offset: int,
                 n_eval_episodes: int):
        self.best_dir = best_dir
        self.target_env_kwargs = target_env_kwargs
        self.target_env = target_env
        self.seed_offset = int(seed_offset)
        self.n_eval_episodes = int(n_eval_episodes)
        self.best_mape = float("inf")
        self.meta_path = self.best_dir / "best_meta.json"
        if self.meta_path.exists():
            try:
                self.best_mape = float(json.loads(
                    self.meta_path.read_text()).get("mape_pct", float("inf")))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                self.best_mape = float("inf")

    def maybe_save(self, *, model, vec_norm, metrics: dict, stage: str,
                   source: str, step: int, extra: dict | None = None) -> bool:
        screen_mape = float(metrics["mape_pct"])
        if not np.isfinite(screen_mape) or screen_mape >= self.best_mape:
            return False

        mapes, times = rollout_eval(
            model, self.target_env, self.n_eval_episodes, self.seed_offset,
            vec_norm=vec_norm)
        confirmed = summarise_eval(mapes, times)
        confirmed_mape = float(confirmed["mape_pct"])
        if not np.isfinite(confirmed_mape) or confirmed_mape >= self.best_mape:
            print(f"[MF global-best] candidate from {stage}/{source} screened "
                  f"at {screen_mape:.2f}% but confirmed at "
                  f"{confirmed_mape:.2f}% >= current best {self.best_mape:.2f}%")
            return False

        self.best_mape = confirmed_mape
        self.best_dir.mkdir(parents=True, exist_ok=True)
        model.save(str(self.best_dir / "best_policy"))
        if vec_norm is not None:
            vec_norm.save(str(self.best_dir / "best_vecnorm.pkl"))

        meta = {
            "stage": stage,
            "source": source,
            "step": int(step),
            "wall_s": time.time(),
            "mape_pct": confirmed_mape,
            "p90_pct": float(confirmed.get("p90_pct", np.nan)),
            "success_rate": float(confirmed.get("success_rate", np.nan)),
            "mean_time_s": float(confirmed.get("mean_time_s", np.nan)),
            "screen_mape_pct": screen_mape,
            "screen_p90_pct": float(metrics.get("p90_pct", np.nan)),
            "screen_success_rate": float(metrics.get("success_rate", np.nan)),
            "screen_mean_time_s": float(metrics.get("mean_time_s", np.nan)),
            "seed_offset": self.seed_offset,
            "n_eval_episodes": self.n_eval_episodes,
            "target_env_kwargs": self.target_env_kwargs,
        }
        if extra:
            for key, value in extra.items():
                if key in meta:
                    meta[f"source_{key}"] = value
                else:
                    meta[key] = value
        self.meta_path.write_text(json.dumps(meta, indent=2, default=str))
        print(f"[MF global-best] {confirmed_mape:.2f}% confirmed full-Bloch "
              f"MAPE saved from {stage}/{source} @ step {step} "
              f"(screen={screen_mape:.2f}%) → {self.best_dir}")
        return True


class FidelitySwitchCallback(BaseCallback):
    """Decides when to promote out of the current (non-final) fidelity.

    Every `decision_every` env steps it probes the policy on the current-fidelity
    eval env and the full-Bloch eval env (each capped at `probe_episodes`),
    records a DecisionPoint, and applies the mf_switch rule. Returns False to end
    `model.learn()` on promotion.
    """

    def __init__(self, *, spec: FidelitySpec, eval_env: QalibreMDE2Env,
                 gt_env: QalibreMDE2Env, eta_cb: ProgressETACallback,
                 decision_every: int, probe_episodes: int, probe_seed: int,
                 history_path: Path, global_deadline: float, reserve_s: float,
                 next_spec: FidelitySpec | None = None,
                 base_env_kwargs: dict | None = None,
                 n_envs: int = 1, n_steps: int = 512, batch_size: int = 64,
                 train_seed: int = 0,
                 global_best: GlobalBestFullSim | None = None,
                 lookahead_enabled: bool = False,
                 lookahead_rollouts: int = 1,
                 lookahead_probe_episodes: int | None = None,
                 lookahead_margin: float = 1.15,
                 lookahead_budget_s: float = 0.0,
                 lookahead_n_envs: int = 1,
                 lookahead_min_ram_gb: float = 0.0,
                 verbose: int = 1):
        super().__init__(verbose)
        self.spec = spec
        self.next_spec = next_spec
        self.base_env_kwargs = base_env_kwargs or {}
        self.eval_env = eval_env
        self.gt_env = gt_env  # ground truth env
        self.eta_cb = eta_cb
        self.decision_every = decision_every
        self.probe_episodes = probe_episodes
        self.probe_seed = probe_seed
        self.history_path = history_path
        self.global_deadline = global_deadline
        self.reserve_s = reserve_s
        self.n_envs = n_envs
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.train_seed = train_seed
        self.global_best = global_best
        self.lookahead_enabled = lookahead_enabled
        self.lookahead_rollouts = lookahead_rollouts
        self.lookahead_probe_episodes = (
            lookahead_probe_episodes
            if lookahead_probe_episodes is not None else probe_episodes)
        self.lookahead_margin = lookahead_margin
        self.lookahead_budget_s = lookahead_budget_s
        self.lookahead_n_envs = max(1, int(lookahead_n_envs))
        self.lookahead_min_ram_gb = float(lookahead_min_ram_gb)
        self.lookahead_wall_s = 0.0
        self.state = SwitchState()
        self.slope_hist: list[float] = []
        self._stage_start = 0
        self._last_decision = 0

    def _on_training_start(self) -> None:
        self._stage_start = self.num_timesteps
        self._last_decision = self.num_timesteps

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_decision < self.decision_every:
            return True
        self._last_decision = self.num_timesteps
        return self._decide()

    def _decide(self) -> bool:
        vec_norm = self.model.get_vec_normalize_env()
        # Current-fidelity probe (cheap) + held-out full-sim probe (the anchor).
        mf, tf = rollout_eval(self.model, self.eval_env, self.probe_episodes,
                              self.probe_seed, vec_norm=vec_norm)
        mh, th = rollout_eval(self.model, self.gt_env, self.probe_episodes,
                              self.probe_seed, vec_norm=vec_norm)
        sf, sh = summarise_eval(mf, tf), summarise_eval(mh, th)
        if self.global_best is not None:
            self.global_best.maybe_save(
                model=self.model,
                vec_norm=vec_norm,
                metrics=sh,
                stage=self.spec.name,
                source="switch_decision_full_probe",
                step=int(self.num_timesteps),
                extra={"current_fidelity_mape_pct": sf["mape_pct"]},
            )

        self.state.add(DecisionPoint(
            step=self.num_timesteps,
            mape_f=sf["mape_pct"] / 100, mape_H=sh["mape_pct"] / 100,
            p90_f=sf["p90_pct"] / 100,   p90_H=sh["p90_pct"] / 100,
        ))

        stage_steps = self.num_timesteps - self._stage_start
        below_reserve = time.time() >= (self.global_deadline - self.reserve_s)
        spp = self.eta_cb.sec_per_step or 0.0
        slope = target_slope_per_cost(
            self.state.p_hist, self.spec.thresholds.plateau_window,
            spp, self.decision_every * self.spec.thresholds.plateau_window)
        self.slope_hist.append(float(slope))
        rel_gain = relative_target_gain(
            self.state.p_hist, self.spec.thresholds.plateau_window)

        lookahead = {
            "lookahead_trigger": "",
            "lookahead_ran": False,
            "lookahead_stage": None,
            "lookahead_wall_s": 0.0,
            "lookahead_mape_pre_pct": None,
            "lookahead_mape_post_pct": None,
            "lookahead_slope_per_cost": None,
            "lookahead_promote": False,
            "lookahead_skip_reason": "",
            "lookahead_available_ram_gb": _available_ram_gb(),
        }

        promote, reason = False, ""
        if below_reserve or stage_steps < self.spec.thresholds.min_steps:
            promote, reason = decide_switch(
                self.state, self.spec.thresholds,
                stage_steps=stage_steps, below_reserve=below_reserve)
        else:
            run_lh, trigger = should_run_lookahead(
                self.state, self.spec.thresholds, self.slope_hist)
            if self.lookahead_enabled and run_lh and self.next_spec is not None:
                lookahead = self._run_lookahead(
                    vec_norm=vec_norm,
                    full_mape_pre_pct=sh["mape_pct"],
                    trigger=trigger)
                lh_slope = lookahead["lookahead_slope_per_cost"]
                current_slope = slope if np.isfinite(slope) else 0.0
                if (lh_slope is not None and
                        lh_slope > self.lookahead_margin * max(current_slope, 0.0)):
                    promote = True
                    reason = f"lookahead_better:{trigger}"
                    lookahead["lookahead_promote"] = True

            if not promote:
                promote, reason = decide_switch(
                    self.state, self.spec.thresholds,
                    stage_steps=stage_steps, below_reserve=below_reserve)

        rc = (None if len(self.state.points) < self.spec.thresholds.rankcorr_min_ckpts
              else _safe(np.corrcoef(
                  _rank([p.mape_f for p in self.state.points]),
                  _rank([p.mape_H for p in self.state.points]))[0, 1]))

        self._log({
            "stage": self.spec.name, "kind": "decision",
            "step": int(self.num_timesteps), "wall_s": time.time(),
            "mape_f_pct": sf["mape_pct"], "mape_H_pct": sh["mape_pct"],
            "p90_f_pct": sf["p90_pct"], "p90_H_pct": sh["p90_pct"],
            "bias_pct": sf["mape_pct"] - sh["mape_pct"],
            "rank_corr": rc, "target_slope_per_cost": _safe(slope),
            "relative_target_gain": _safe(rel_gain),
            "sec_per_step": spp, "below_reserve": bool(below_reserve),
            "promote": bool(promote), "switch_reason": reason,
            **lookahead,
        })
        if self.verbose:
            print(f"[MF {self.spec.name} @ {self.num_timesteps}] "
                  f"MAPE_f={sf['mape_pct']:.2f}% MAPE_full={sh['mape_pct']:.2f}% "
                  f"bias={sf['mape_pct']-sh['mape_pct']:+.2f}% "
                  f"rankcorr={rc if rc is None else round(rc,2)} "
                  f"→ {'PROMOTE('+reason+')' if promote else 'stay'}")
        return not promote

    def _run_lookahead(self, *, vec_norm, full_mape_pre_pct: float,
                       trigger: str) -> dict:
        entry = {
            "lookahead_trigger": trigger,
            "lookahead_ran": False,
            "lookahead_stage": self.next_spec.name if self.next_spec else None,
            "lookahead_wall_s": 0.0,
            "lookahead_mape_pre_pct": full_mape_pre_pct,
            "lookahead_mape_post_pct": None,
            "lookahead_slope_per_cost": None,
            "lookahead_promote": False,
        }
        if self.next_spec is None:
            return entry
        avail_ram_gb = _available_ram_gb()
        entry["lookahead_available_ram_gb"] = avail_ram_gb
        if (self.lookahead_min_ram_gb > 0.0 and avail_ram_gb is not None and
                avail_ram_gb < self.lookahead_min_ram_gb):
            entry["lookahead_skip_reason"] = "low_ram"
            return entry
        remaining_lookahead_s = self.lookahead_budget_s - self.lookahead_wall_s
        reserve_deadline = self.global_deadline - self.reserve_s
        if remaining_lookahead_s <= 0 or time.time() >= reserve_deadline:
            entry["lookahead_skip_reason"] = "budget"
            return entry

        next_kwargs = {**self.base_env_kwargs, **self.next_spec.env_overrides}
        next_vec_env = build_vec_env(
            next_kwargs, n_envs=self.lookahead_n_envs,
            train_seed=self.train_seed + 100_000 + int(self.num_timesteps))
        if vec_norm is not None:
            next_vec_env.obs_rms = copy.deepcopy(vec_norm.obs_rms)
        next_vec_env.ret_rms = RunningMeanStd(shape=())

        t0 = time.time()
        try:
            clone = build_model(
                next_vec_env, n_steps=self.n_steps, batch_size=self.batch_size)
            clone.policy.load_state_dict(
                copy.deepcopy(self.model.policy.state_dict()))

            lookahead_steps = (
                max(1, int(self.lookahead_rollouts)) *
                self.n_steps * self.lookahead_n_envs)
            deadline = min(reserve_deadline, time.time() + remaining_lookahead_s)
            clone.learn(
                total_timesteps=lookahead_steps,
                callback=[WallBudgetCallback(deadline)],
                reset_num_timesteps=True)
            mpost, tpost = rollout_eval(
                clone, self.gt_env, self.lookahead_probe_episodes,
                self.probe_seed, vec_norm=next_vec_env)
            spost = summarise_eval(mpost, tpost)
        finally:
            next_vec_env.close()
        wall_s = time.time() - t0
        self.lookahead_wall_s += wall_s

        gain = max(
            0.0,
            score(spost["mape_pct"] / 100) - score(full_mape_pre_pct / 100))
        entry.update({
            "lookahead_ran": True,
            "lookahead_wall_s": wall_s,
            "lookahead_mape_post_pct": spost["mape_pct"],
            "lookahead_slope_per_cost": gain / max(wall_s, 1e-9),
        })
        return entry

    def _log(self, entry: dict) -> None:
        hist = (json.loads(self.history_path.read_text())
                if self.history_path.exists() else [])
        hist.append(entry)
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        self.history_path.write_text(json.dumps(hist, indent=2))


def _rank(x):
    from mf_switch import _rankdata
    return _rankdata(np.asarray(x, dtype=float))


def _safe(v):
    v = float(v)
    return None if (np.isnan(v) or np.isinf(v)) else v


def _available_ram_gb() -> float | None:
    """Best-effort available system RAM from /proc/meminfo."""
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) / (1024.0 ** 2)
    except OSError:
        return None
    return None


# ── plan construction ───────────────────────────────────────────────────────

def _parse_int_list(s: str, n: int, name: str) -> list[int]:
    """Parse '2000' (broadcast) or '2000,4000,6000' (per-stage) into n ints."""
    parts = [int(x) for x in str(s).split(",") if x != ""]
    if len(parts) == 1:
        return parts * n
    if len(parts) != n:
        raise ValueError(f"--{name} expects 1 or {n} values, got {len(parts)}")
    return parts


def build_plan(args) -> list[FidelitySpec]:
    names = [s.strip() for s in args.mf_plan.split(",") if s.strip()]
    mins = _parse_int_list(args.mf_min_steps, len(names), "mf-min-steps")
    maxs = _parse_int_list(args.mf_max_steps, len(names), "mf-max-steps")
    specs = []
    for i, name in enumerate(names):
        if name not in FIDELITIES:
            raise ValueError(
                f"unknown fidelity '{name}'; choose from {list(FIDELITIES)}")
        th = SwitchThresholds(
            mape_floor=args.mf_mape_floor,
            plateau_window=args.mf_plateau_window,
            plateau_delta=args.mf_plateau_delta,
            rankcorr_min_ckpts=args.mf_rankcorr_min_ckpts,
            rankcorr_thresh=RANKCORR_THRESH.get(name, 0.8),
            bias_mape_abs=args.mf_bias_mape_abs,
            bias_p90_abs=args.mf_bias_p90_abs,
            min_steps=mins[i],
            slope_collapse_frac=args.mf_slope_collapse_frac,
            near_plateau_frac=args.mf_near_plateau_frac,
            min_lookahead_points=(
                args.mf_min_lookahead_points
                if args.mf_min_lookahead_points is not None
                else args.mf_plateau_window + 1),
        )
        specs.append(FidelitySpec(
            name=name, env_overrides=FIDELITIES[name],
            min_steps=mins[i], max_steps=maxs[i],
            is_final=(i == len(names) - 1), thresholds=th,
        ))
    return specs


# ── main ────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("runs/e2/mf"))
    p.add_argument("--n-envs", type=int, default=1,
                   help="Parallel training envs. n_envs=1 keeps one Julia "
                        "runtime warm; n_envs>1 uses SubprocVecEnv with one "
                        "Julia runtime per worker. PPO rollout size is "
                        "n_steps×n_envs.")
    p.add_argument("--n-steps", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--train-seed", type=int, default=0)
    p.add_argument("--eval-seed", type=int, default=500_000)
    p.add_argument("--eval-episodes", type=int, default=16)
    p.add_argument("--eval-interval", type=int, default=5_000)

    # ── multi-fidelity controller ──
    p.add_argument("--multi-fidelity", action="store_true",
                   help="Run the curriculum (required; present for explicitness).")
    p.add_argument("--mf-plan", type=str, default="analytic,cached,full",
                   help="Comma list of fidelities from "
                        f"{{{','.join(FIDELITIES)}}}.")
    p.add_argument("--mf-budget-hours", type=float, default=24.0,
                   help="Total wallclock budget Γ for the whole curriculum.")
    p.add_argument("--mf-full-reserve-frac", type=float, default=0.3,
                   help="Fraction of Γ reserved for the final full-Bloch stage.")
    p.add_argument("--mf-decision-rollouts", type=int, default=4,
                   help="Run a switch decision every this many PPO rollouts.")
    p.add_argument("--mf-probe-episodes-full", type=int, default=4,
                   help="Episodes per probe (current AND full) at each decision. "
                        "Caps the full-sim probe cost — keep small.")
    p.add_argument("--mf-global-best-episodes", type=int, default=12,
                   help="Held-out full-Bloch episodes used to confirm a candidate "
                        "before overwriting <run>/global_best. Switch probes still "
                        "use --mf-probe-episodes-full.")
    p.add_argument("--mf-global-best-seed", type=int, default=None,
                   help="Seed offset for global-best confirmation evals. Defaults "
                        "to --eval-seed + 10000 so it is independent of the switch "
                        "probe seeds.")
    p.add_argument("--mf-min-steps", type=str, default="0",
                   help="Per-stage min env steps before a promotion is allowed "
                        "(single value broadcasts, or comma list per stage).")
    p.add_argument("--mf-max-steps", type=str, default="200000",
                   help="Per-stage hard step cap (force-promote). Single or comma list.")
    p.add_argument("--mf-mape-floor", type=float, default=1e-3)
    p.add_argument("--mf-plateau-window", type=int, default=3)
    p.add_argument("--mf-plateau-delta", type=float, default=0.02)
    p.add_argument("--mf-rankcorr-min-ckpts", type=int, default=4)
    p.add_argument("--mf-bias-mape-abs", type=float, default=0.01)
    p.add_argument("--mf-bias-p90-abs", type=float, default=0.02)
    p.add_argument("--mf-use-lookahead", action="store_true",
                   help="Enable V2 next-fidelity lookahead slope-per-cost "
                        "promotion when current-fidelity progress collapses "
                        "or nears the plateau fallback.")
    p.add_argument("--mf-lookahead-rollouts", type=int, default=1,
                   help="PPO rollouts to train the cloned next-fidelity policy.")
    p.add_argument("--mf-lookahead-probe-episodes", type=int, default=None,
                   help="Full-sim episodes for lookahead post-eval. Defaults to "
                        "--mf-probe-episodes-full.")
    p.add_argument("--mf-lookahead-margin", type=float, default=1.15,
                   help="Promote when lookahead slope-per-cost exceeds current "
                        "slope-per-cost by this factor.")
    p.add_argument("--mf-slope-collapse-frac", type=float, default=0.25,
                   help="Trigger lookahead when recent slope falls below this "
                        "fraction of the stage's previous best finite slope.")
    p.add_argument("--mf-near-plateau-frac", type=float, default=2.0,
                   help="Trigger lookahead when relative target gain is within "
                        "this multiple of --mf-plateau-delta.")
    p.add_argument("--mf-min-lookahead-points", type=int, default=None,
                   help="Minimum decision points before lookahead can trigger. "
                        "Defaults to --mf-plateau-window + 1.")
    p.add_argument("--mf-lookahead-max-frac", type=float, default=0.10,
                   help="Cap cumulative lookahead wallclock at this fraction of "
                        "the global MF budget.")
    p.add_argument("--mf-lookahead-n-envs", type=int, default=1,
                   help="Vectorized env count for temporary lookahead training. "
                        "Default 1 to cap RAM even when main --n-envs is larger.")
    p.add_argument("--mf-lookahead-min-ram-gb", type=float, default=0.0,
                   help="Skip lookahead if /proc/meminfo MemAvailable is below "
                        "this many GiB. 0 disables the guard.")
    p.add_argument("--mf-reset-optimizer-on-switch", action="store_true",
                   help="Rebuild PPO (fresh Adam) and copy only policy weights at "
                        "each switch — safer than carrying stale optimizer state.")
    p.add_argument("--mf-reset-reward-norm", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Reset VecNormalize reward stats at each switch (default on; "
                        "reward distribution shifts between fidelities). Disable with "
                        "--no-mf-reset-reward-norm.")
    p.add_argument("--mf-full-early-stop-patience", type=int, default=0,
                   help="Final full-stage early stopping patience in eval points. "
                        "0 disables. Only applies once the current fidelity is "
                        "'full'.")
    p.add_argument("--mf-full-early-stop-min-evals", type=int, default=5,
                   help="Minimum number of full-stage evals before final-stage "
                        "early stopping can fire.")
    p.add_argument("--mf-full-early-stop-min-steps", type=int, default=0,
                   help="Minimum full-stage training steps before final-stage "
                        "early stopping can fire.")
    p.add_argument("--mf-full-early-stop-delta", type=float, default=0.10,
                   help="Required absolute MAPE improvement, in percentage "
                        "points, to reset final-stage early-stop patience.")
    p.add_argument("--schedule", choices=["criterion", "fixed"], default="criterion",
                   help="criterion = literature switch rule; fixed = equal wallclock "
                        "thirds (ablation baseline).")
    p.add_argument("--mf-linear-ti-action", action="store_true",
                   help="Opt out of the MF default log-spaced continuous TI "
                        "mapping and use the base linear TI action mapping.")

    add_e2_env_args(p)
    args = p.parse_args()

    if args.use_gpu and args.n_envs > 1:
        raise ValueError("--use-gpu should not be combined with --n-envs > 1; "
                         "one env generally saturates a single GPU and workers "
                         "would contend for the same device.")

    base_kwargs = e2_env_kwargs(args)
    if not args.mf_linear_ti_action:
        base_kwargs["log_ti_action"] = True
    if base_kwargs.get("include_image"):
        raise ValueError("--include-image is incompatible with the analytic stage "
                         "(T1-only obs). Run the curriculum with image obs OFF so "
                         "warm-start layouts match across fidelities.")
    args.out.mkdir(parents=True, exist_ok=True)

    specs = build_plan(args)
    total_budget_s = args.mf_budget_hours * 3600.0
    reserve_s = args.mf_full_reserve_frac * total_budget_s
    t_start = time.time()
    global_deadline = t_start + total_budget_s
    history_path = args.out / "fidelity_history.json"
    decision_every = args.mf_decision_rollouts * args.n_steps * args.n_envs

    (args.out / "run_config.json").write_text(json.dumps({
        "base_env_kwargs": base_kwargs,
        "plan": [s.name for s in specs],
        "mf": {k: v for k, v in vars(args).items()
               if k.startswith("mf_") or k in ("schedule", "n_envs", "n_steps", "batch_size")},
    }, indent=2, default=str))

    # One shared full-Bloch probe env (ground truth for every stage's decisions).
    print("[MF] Building full-Bloch ground-truth probe env …")
    gt_kwargs = {**base_kwargs, **FIDELITIES["full"]}
    gt_env = QalibreMDE2Env(rng_seed=args.eval_seed + 7, **gt_kwargs)
    global_best_seed = (args.mf_global_best_seed
                        if args.mf_global_best_seed is not None
                        else args.eval_seed + 10_000)
    global_best = GlobalBestFullSim(
        args.out / "global_best",
        target_env_kwargs=gt_kwargs,
        target_env=gt_env,
        seed_offset=global_best_seed,
        n_eval_episodes=args.mf_global_best_episodes,
    )

    model = None
    prev_obs_rms = None
    prev_final_mape = None
    final_vec_env = None

    for i, spec in enumerate(specs):
        stage_dir = args.out / f"stage{i}_{spec.name}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        env_kwargs = {**base_kwargs, **spec.env_overrides}
        print(f"\n[MF] ===== stage {i}: {spec.name}  "
              f"({'final' if spec.is_final else 'cheap'}) =====")

        vec_env = build_vec_env(env_kwargs, n_envs=args.n_envs,
                                train_seed=args.train_seed)
        eval_env = QalibreMDE2Env(rng_seed=args.eval_seed, **env_kwargs)

        if model is None:
            model = build_model(vec_env, n_steps=args.n_steps,
                                batch_size=args.batch_size,
                                tensorboard_log=str(args.out / "tb" / spec.name))
        else:
            # Warm-start: carry obs stats (per-sphere T1 in seconds — physically
            # comparable across fidelities). Reward stats / optimizer are reset by
            # default because the reward distribution shifts between fidelities.
            vec_env.obs_rms = copy.deepcopy(prev_obs_rms)
            if args.mf_reset_reward_norm:
                vec_env.ret_rms = RunningMeanStd(shape=())
            if args.mf_reset_optimizer_on_switch:
                fresh = build_model(vec_env, n_steps=args.n_steps,
                                    batch_size=args.batch_size,
                                    tensorboard_log=str(args.out / "tb" / spec.name))
                fresh.policy.load_state_dict(
                    model.policy.state_dict())  # weights only
                model = fresh
            else:
                model.set_env(vec_env)

            # Fidelity gap: held-out full-sim eval of the carried policy BEFORE
            # any training at this stage — the transfer diagnostic.
            mg, tg = rollout_eval(model, gt_env, args.mf_probe_episodes_full,
                                  args.eval_seed, vec_norm=vec_env)
            sg = summarise_eval(mg, tg)
            global_best.maybe_save(
                model=model,
                vec_norm=vec_env,
                metrics=sg,
                stage=spec.name,
                source="stage_start_full_probe",
                step=int(model.num_timesteps),
            )
            gap = (None if prev_final_mape is None
                   else sg["mape_pct"] - prev_final_mape)
            _append(history_path, {
                "stage": spec.name, "kind": "stage_start",
                "step": int(model.num_timesteps), "wall_s": time.time(),
                "full_mape_pre_pct": sg["mape_pct"],
                "fidelity_gap_pct": gap,
            })
            print(f"[MF] {spec.name} warm-start full-sim MAPE_pre="
                  f"{sg['mape_pct']:.2f}%  (gap vs prev final: "
                  f"{gap if gap is None else round(gap,2)}%)")

        # ── callbacks for this stage ──
        eta_cb = ProgressETACallback(
            total_timesteps=model.num_timesteps + spec.max_steps)
        eval_cb = E2EvalCallback(
            eval_env=eval_env, every_n_steps=args.eval_interval,
            n_eval_episodes=args.eval_episodes, seed_offset=args.eval_seed,
            log_path=stage_dir / "eval_history.json",
            best_dir=stage_dir / "best", env_kwargs=env_kwargs,
            global_best=(global_best if env_kwargs == gt_kwargs else None),
            global_best_stage=spec.name,
            global_best_source="stage_eval_full")
        callbacks: list[BaseCallback] = [eval_cb, eta_cb,
                                         WallBudgetCallback(global_deadline)]
        if spec.name == "full" and args.mf_full_early_stop_patience > 0:
            callbacks.append(FullStageEarlyStopCallback(
                eval_cb=eval_cb,
                stage=spec.name,
                history_path=history_path,
                patience=args.mf_full_early_stop_patience,
                min_evals=args.mf_full_early_stop_min_evals,
                min_steps=args.mf_full_early_stop_min_steps,
                delta_pct=args.mf_full_early_stop_delta))

        # Per-stage wall deadline: criterion uses the global one (switch rule
        # ends cheap stages early); fixed splits Γ into equal slices.
        if args.schedule == "fixed":
            slice_s = total_budget_s / len(specs)
            callbacks.append(WallBudgetCallback(time.time() + slice_s))
        elif not spec.is_final:
            next_spec = specs[i + 1] if i + 1 < len(specs) else None
            callbacks.append(FidelitySwitchCallback(
                spec=spec, eval_env=eval_env, gt_env=gt_env, eta_cb=eta_cb,
                decision_every=decision_every,
                probe_episodes=args.mf_probe_episodes_full,
                probe_seed=args.eval_seed, history_path=history_path,
                global_deadline=global_deadline, reserve_s=reserve_s,
                next_spec=next_spec, base_env_kwargs=base_kwargs,
                n_envs=args.n_envs,
                n_steps=args.n_steps, batch_size=args.batch_size,
                train_seed=args.train_seed,
                global_best=global_best,
                lookahead_enabled=args.mf_use_lookahead,
                lookahead_rollouts=args.mf_lookahead_rollouts,
                lookahead_probe_episodes=args.mf_lookahead_probe_episodes,
                lookahead_margin=args.mf_lookahead_margin,
                lookahead_budget_s=args.mf_lookahead_max_frac * total_budget_s,
                lookahead_n_envs=args.mf_lookahead_n_envs,
                lookahead_min_ram_gb=args.mf_lookahead_min_ram_gb))

        t0 = time.time()
        # `max_steps` is a PER-STAGE budget. SB3's learn(reset_num_timesteps=False)
        # ADDS the current counter to total_timesteps internally, so passing
        # spec.max_steps gives this stage exactly max_steps NEW steps while the
        # global counter stays monotonic across the curriculum (only stage 0
        # resets it to 0). Callbacks gate on their own stage-relative offset, so
        # the cumulative counter is purely for continuity/tensorboard.
        model.learn(total_timesteps=spec.max_steps, callback=callbacks,
                    reset_num_timesteps=(i == 0))
        print(f"[MF] stage {spec.name} done in {(time.time()-t0)/60:.1f} min")

        # Persist stage artifacts + record this stage's final full-sim MAPE.
        model.save(stage_dir / "policy")
        vec_env.save(stage_dir / "vecnorm.pkl")
        mfin, tfin = rollout_eval(model, gt_env, args.mf_probe_episodes_full,
                                  args.eval_seed, vec_norm=vec_env)
        sfin = summarise_eval(mfin, tfin)
        global_best.maybe_save(
            model=model,
            vec_norm=vec_env,
            metrics=sfin,
            stage=spec.name,
            source="stage_end_full_probe",
            step=int(model.num_timesteps),
        )
        prev_final_mape = sfin["mape_pct"]
        _append(history_path, {
            "stage": spec.name, "kind": "stage_end",
            "step": int(model.num_timesteps), "wall_s": time.time(),
            "full_mape_pct": sfin["mape_pct"], "full_p90_pct": sfin["p90_pct"],
        })
        prev_obs_rms = copy.deepcopy(vec_env.obs_rms)
        final_vec_env = vec_env
        if time.time() >= global_deadline:
            print("[MF] global budget reached — ending curriculum early")
            break

        # Close subprocess Julia workers before the next stage builds a new
        # SubprocVecEnv. The copied obs_rms above is all we need to warm-start
        # normalization, and explicit close avoids noisy SIGTERM stack dumps at
        # process teardown.
        if not spec.is_final:
            vec_env.close()

    model.save(args.out / "policy")
    if final_vec_env is not None:
        final_vec_env.save(args.out / "vecnorm.pkl")
        final_vec_env.close()
    print(f"\n[MF] Curriculum done in {(time.time()-t_start)/3600:.2f} h. "
          f"Final full-sim MAPE={prev_final_mape:.2f}%")
    print(f"Policy → {args.out / 'policy.zip'}")
    print(f"Global best → {args.out / 'global_best' / 'best_policy.zip'}")


def _append(path: Path, entry: dict) -> None:
    hist = json.loads(path.read_text()) if path.exists() else []
    hist.append(entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(hist, indent=2))


if __name__ == "__main__":
    main()

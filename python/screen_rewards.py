"""Reward-screening harness for E2, on the fast analytic forward model.

Accuracy-vs-time STOP sweep — run one λ first (cached for the full run), then all:

    # 1) single point first (λ=0.1, 200k, save model+vecnorm):
    PYTHON_JULIAPKG_OFFLINE=yes PYTHON_JULIAPKG_EXE=~/.julia/juliaup/julia-1.11.9+0.x64.linux.gnu/bin/julia \
      PYTHONUNBUFFERED=1 python -u python/screen_rewards.py \
        --out runs/e2/stop_sweep --stop-sweep --save-models \
        --lambdas 0.1 --seeds 16 --ppo-timesteps 200000 \
        2>&1 | tee runs/e2/stop_sweep/run.log

    # 2) full λ sweep into the SAME --out (reuses the cached λ=0.1 row):
    PYTHON_JULIAPKG_OFFLINE=yes PYTHON_JULIAPKG_EXE=~/.julia/juliaup/julia-1.11.9+0.x64.linux.gnu/bin/julia \
      PYTHONUNBUFFERED=1 python -u python/screen_rewards.py \
        --out runs/e2/stop_sweep --stop-sweep --save-models \
        --seeds 16 --ppo-timesteps 200000 \
        2>&1 | tee -a runs/e2/stop_sweep/run.log

Cheaply characterises how candidate reward functions shape agent behaviour
*before* committing to a multi-day full-Bloch training run. Runs on
`forward_model="analytic"` (≈0.4 ms/step vs seconds for Koma), action space
[TI, TR] (`fix_te=True`).

Two levels:

  Level 1 — policy-ranking probe (no training, instant).
    Roll a handful of hand-written open-loop policies under each reward config
    and record return + outcome metrics. The key diagnostic is return-vs-final-
    MAPE *alignment*: a sound reward should rank the policy with the LOWEST
    final MAPE highest. A reward is flagged pathological if it instead ranks a
    few-fat-block (high-TR) or a dawdling/degenerate policy at the top — that is
    the per-block-reward failure mode (the reward is blind to whether extra
    blocks bought accuracy).

  Level 2 — short PPO screen (--ppo-timesteps, default 30k).
    For each reward config, train a short PPO on the analytic env and evaluate
    the learned policy. Confirms what an agent actually learns to exploit.

Outputs to --out: results.csv (one row per config × policy/agent), a probe
ranking summary, and a MAPE-vs-scan-time scatter (draft Pareto) for the PPO
agents.

Examples:
    source .venv/bin/activate

    # full screen (probe + PPO over all default configs):
    PYTHON_JULIAPKG_OFFLINE=yes python python/screen_rewards.py \
        --seeds 16 --ppo-timesteps 30000 --out runs/e2/reward_screen

    # probe only (seconds, no training):
    PYTHON_JULIAPKG_OFFLINE=yes python python/screen_rewards.py --skip-ppo

    # PPO only over the default configs (skip the probe); caching skips configs
    # already in results.csv, so this trains only the new ones:
    PYTHON_JULIAPKG_OFFLINE=yes PYTHONUNBUFFERED=1 python -u python/screen_rewards.py \
        --out runs/e2/reward_screen --skip-probe \
        --seeds 16 --ppo-timesteps 30000 \
        2>&1 | tee runs/e2/reward_screen/ppo_run.log

    # discount comparison for one reward (separate dir):
    PYTHON_JULIAPKG_OFFLINE=yes PYTHONUNBUFFERED=1 python -u python/screen_rewards.py \
        --out runs/e2/reward_screen_gamma \
        --seeds 16 --ppo-timesteps 30000 \
        --ppo-modes delta_log_mape --ppo-alphas 1.0 \
        --gammas 0.99 1.0

    # accuracy-vs-time, experiment 1 — learned STOP, sweep λ (priority):
    PYTHON_JULIAPKG_OFFLINE=yes PYTHONUNBUFFERED=1 python -u python/screen_rewards.py \
        --out runs/e2/stop_sweep --stop-sweep --save-models \
        --seeds 16 --ppo-timesteps 200000

    # accuracy-vs-time, experiment 2 — fixed-budget sweep + combined overlay:
    PYTHON_JULIAPKG_OFFLINE=yes PYTHONUNBUFFERED=1 python -u python/screen_rewards.py \
        --out runs/e2/budget_sweep --budget-sweep --save-models \
        --seeds 16 --ppo-timesteps 200000 \
        --combine runs/e2/stop_sweep/results.csv

Findings (analytic env, σ=0.04≈SNR 25, 14-sphere T1.5 plate; as of 2026-05-27):

  Level-1 probe — reward alignment:
    * `neg_mape` and `neg_log_mape` are MISALIGNED. Summing −MAPE per block
      penalises every informative step, so they rank a near-do-nothing/random
      or few-fat-block policy above the most accurate one. The time penalty λ
      does NOT rescue them (within the always-spent budget, λ·time is ~constant).
    * `delta_mape`, `delta_log_mape`, `terminal_only` are ALIGNED — all rank the
      lowest-final-MAPE policy top. Best scripted policies: `balanced` 0.136,
      `log_sweep` 0.173, `many_block_lo_tr` 0.196 MAPE (these are the open-loop
      accuracy floor on this env).

  Level-2 PPO (30k steps) — what the agent actually learns:
    * Every aligned reward plateaus at ~0.83–0.89 MAPE — ~4× worse than the
      scripted sweeps. The agent collapses to 7 blocks at min TR with NO TI
      diversity (identical block structure to `many_block_same_ti`, 0.906).
    * So at 30k the bottleneck is TI-DIVERSITY LEARNING, not the reward choice;
      the α aggregation knob barely separates configs while stuck here.
    * Discount: delta_log_mape|α=1 gives final MAPE 0.830 (γ=0.99) vs 0.888
      (γ=1.0). γ=1 converges slower (un-discounted, higher-variance returns) —
      expected — but both are pre-convergence and the gap is within single-seed
      noise, so this does NOT rank the discounts. Needs more timesteps and 2–3
      training seeds to be conclusive.

  Caveat: analytic == the fitter's own forward model, so these fits are
  noise-limited only (no water-bleed / B0σ / recon cross-talk). The numbers
  screen reward-induced BEHAVIOUR, not absolute achievable MAPE — re-validate
  finalists on water_model=:cached_perline, then full Bloch.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from qalibremd_gym.env_e2 import QalibreMDE2Env

# The E2 training discount: mirrors the PPO gamma in train_e2.py (the value real
# E2 runs use). The γ sweep below always includes it as the baseline alongside
# γ=1 (the value at which delta_mape's return equals 1 − final_MAPE exactly).
DEFAULT_GAMMA = 0.99

# CSV schema. The numeric columns are cast to float on load so cached rows can
# be ranked/plotted alongside freshly computed ones. allow_stop/include_sigma/
# time_budget are env-knob columns (parameterised like gamma): kept as metadata,
# back-filled with defaults for pre-existing rows so old data isn't lost.
CSV_FIELDS = ["level", "config", "reward_mode", "time_penalty", "mape_alpha",
              "gamma", "allow_stop", "include_sigma", "time_budget", "policy",
              "mean_return", "final_mape", "mean_tr", "total_time", "n_blocks",
              "mape_spread", "action_repeat"]
NUMERIC_FIELDS = ["mean_return", "final_mape", "mean_tr", "total_time",
                  "n_blocks", "mape_spread", "action_repeat"]
# Defaults back-filled into rows that predate the env-knob columns (legacy runs
# were fixed-budget 120 s, no stop, no σ-channel).
LEGACY_DEFAULTS = {"allow_stop": 0, "include_sigma": 0, "time_budget": 120.0}


def row_key(row: dict) -> tuple:
    """Identity of a result row: (level, config-label, policy)."""
    return (row["level"], row["config"], row.get("policy", ""))


def load_existing(path: Path, force: bool) -> dict:
    """Load prior results.csv into {row_key: row}. Empty if --force or no file.

    Numeric fields are cast to float so cached rows rank/plot like new ones.
    Lets a rerun skip configs already computed (cache); pass --force to redo."""
    if force or not path.exists():
        return {}
    out: dict = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            for k in NUMERIC_FIELDS:
                try:
                    row[k] = float(row[k])
                except (TypeError, ValueError):
                    row[k] = float("nan")
            # Back-fill env-knob columns for rows written before they existed.
            for k, default in LEGACY_DEFAULTS.items():
                if not row.get(k):          # missing or empty string
                    row[k] = default
            out[row_key(row)] = row
    return out


# Physical action bounds for the [TI, TR] sub-space, derived from the single
# source (env_e2._ACT_LO/HI, itself verified against julia/rl/e2.jl at env
# construction). Indices: 0 = TI, 2 = TR in the [TI, TE, TR, α] vector.
TI_LO, TI_HI = float(QalibreMDE2Env._ACT_LO[0]), float(QalibreMDE2Env._ACT_HI[0])
TR_LO, TR_HI = float(QalibreMDE2Env._ACT_LO[2]), float(QalibreMDE2Env._ACT_HI[2])

# A spread of informative inversion times covering the plate's T1·ln2 range
# (short ~0.05 s to long ~1.4 s). Open-loop policies cycle through these.
INFO_TIS = np.array([0.05, 0.10, 0.20, 0.40, 0.70, 1.00, 1.40])
# Log-spaced TIs over the same range — denser at short TI, matching the
# decade-wide T1 distribution better than the (linear-ish) INFO_TIS grid.
LOG_TIS = np.geomspace(0.03, 1.60, 7)


def _u(phys: float, lo: float, hi: float) -> float:
    """Physical value → normalised action component in [-1, 1]."""
    return float(np.clip(2.0 * (phys - lo) / (hi - lo) - 1.0, -1.0, 1.0))


def _act(ti_s: float, tr_s: float) -> np.ndarray:
    return np.array([_u(ti_s, TI_LO, TI_HI), _u(tr_s, TR_LO, TR_HI)],
                    dtype=np.float32)


# ── scripted open-loop policies: (block_index) -> normalised [u_TI, u_TR] ──────
# Npe=32, budget=120 s ⇒ block_time = 32·TR, so TR sets how many blocks fit:
#   TR=0.5 → 16 s → ~7 blocks;  TR≈1.8 → 58 s → 2 blocks;  TR≥3.8 → 0–1 blocks.

def pol_many_block_lo_tr(t: int) -> np.ndarray:
    # Min TR (~7 blocks), swept informative TIs → most data, expected lowest MAPE.
    return _act(INFO_TIS[t % len(INFO_TIS)], TR_LO)


def pol_balanced(t: int) -> np.ndarray:
    # Mid TR (~4 blocks), swept TIs — a sensible accuracy/recovery compromise.
    return _act(INFO_TIS[t % len(INFO_TIS)], 0.8)


def pol_few_block_hi_tr(t: int) -> np.ndarray:
    # High TR → only ~2 blocks fit; 2 distinct TIs. The suspected reward exploit:
    # fewer scored blocks for a per-block reward, at the cost of final accuracy.
    return _act(INFO_TIS[[1, 4][t % 2]], 1.8)


def pol_dawdler(t: int) -> np.ndarray:
    # Two informative blocks, then repeat the 2nd action forever (redundant,
    # no new information) at short TR. Tests whether the reward penalises
    # spending blocks that don't improve the estimate.
    return _act(INFO_TIS[1] if t == 0 else INFO_TIS[4], TR_LO)


def pol_single_repeat(t: int) -> np.ndarray:
    # Same informative action every block (E1-style degenerate policy).
    return _act(INFO_TIS[3], TR_LO)


def pol_log_sweep(t: int) -> np.ndarray:
    # Log-spaced TIs at mid TR — best coverage of the T1 decade; the strongest
    # hand-designed open-loop policy, a near-floor reference for final MAPE.
    return _act(LOG_TIS[t % len(LOG_TIS)], 0.8)


def pol_two_point(t: int) -> np.ndarray:
    # Just two well-separated informative TIs (short + long) at higher TR
    # (higher SNR/block). Tests information-content-per-block vs block count:
    # can 2 good points rival 7 mediocre ones?
    return _act([0.08, 1.00][t % 2], 1.5)


def pol_many_block_same_ti(t: int) -> np.ndarray:
    # Many blocks (min TR) but ALL at one TI — the behaviour PPO collapsed to at
    # 30k steps: 'active' (7 blocks) yet information-starved (no TI diversity).
    # A sound reward must rank this far below a TI-sweeping policy DESPITE the
    # identical block count. The sharpest test of "rewards info, not activity".
    return _act(0.40, TR_LO)


def pol_random(t: int, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(-1.0, 1.0, size=2).astype(np.float32)


POLICIES = {
    "log_sweep":          pol_log_sweep,
    "many_block_lo_tr":   pol_many_block_lo_tr,
    "balanced":           pol_balanced,
    "two_point":          pol_two_point,
    "few_block_hi_tr":    pol_few_block_hi_tr,
    "many_block_same_ti": pol_many_block_same_ti,
    "dawdler":            pol_dawdler,
    "single_repeat":      pol_single_repeat,
    "random":             pol_random,
}

# Policies that are "active but uninformative" or degenerate — if a reward
# ranks any of these top, it is rewarding activity/block-count over actual
# information gain (the failure mode the 30k PPO run exhibited).
DEGENERATE_POLICIES = ("few_block_hi_tr", "many_block_same_ti", "dawdler",
                       "single_repeat")


@dataclass
class RewardCfg:
    reward_mode: str
    time_penalty: float
    mape_alpha: float
    allow_stop: bool = False
    include_sigma: bool = False
    time_budget: float = 120.0

    @property
    def label(self) -> str:
        # Suffix only NON-default env knobs, so legacy configs keep byte-identical
        # labels (the cache still hits) while new sweep points stay distinct.
        s = f"{self.reward_mode}|λ={self.time_penalty:g}|α={self.mape_alpha:g}"
        if self.include_sigma:
            s += "|σ=1"
        if self.allow_stop:
            s += "|stop=1"
        if self.time_budget != 120.0:
            s += f"|T={self.time_budget:g}"
        return s


@dataclass
class EpisodeStats:
    ret: float            # episode return (sum of rewards)
    final_mape: float     # final aggregated MAPE
    mean_tr: float        # mean TR chosen over executed blocks
    total_time: float     # total scan time used [s]
    n_blocks: int         # executed blocks
    # max - mean per-sphere |%err| at episode end (mean-vs-max)
    mape_spread: float
    # fraction of consecutive duplicate actions (degeneracy)
    action_repeat: float


def roll_episode(env: QalibreMDE2Env, policy, seed: int,
                 rng: np.random.Generator) -> EpisodeStats:
    obs, _ = env.reset(seed=seed)
    ret = 0.0
    trs: list[float] = []
    acts: list[tuple] = []
    last_info: dict = {}
    done = False
    t = 0
    while not done:
        try:
            a = policy(t, rng)          # random policy needs the rng
        except TypeError:
            a = policy(t)
        acts.append(tuple(np.round(a, 4)))
        obs, r, term, trunc, info = env.step(a)
        ret += r
        last_info = info
        if "TR" in info and not info.get("budget_exceeded", False):
            trs.append(float(info["TR"]))
        done = term or trunc
        t += 1

    t1_true = np.asarray(last_info.get("T1_true", []), dtype=float)
    t1_est = np.asarray(last_info.get("T1_est", []), dtype=float)
    if t1_true.size and np.all(t1_true > 0):
        errs = np.abs(t1_est - t1_true) / t1_true
        spread = float(errs.max() - errs.mean())
    else:
        spread = float("nan")
    repeat = (sum(acts[i] == acts[i - 1] for i in range(1, len(acts)))
              / max(1, len(acts) - 1))
    return EpisodeStats(
        ret=ret,
        final_mape=float(last_info.get("mape", float("nan"))),
        mean_tr=float(np.mean(trs)) if trs else float("nan"),
        total_time=float(last_info.get("time_s", 0.0)),
        n_blocks=int(last_info.get("n_blocks", 0)),
        mape_spread=spread,
        action_repeat=repeat,
    )


def mean_stats(rows: list[EpisodeStats]) -> dict:
    return {
        "mean_return": float(np.mean([r.ret for r in rows])),
        "final_mape":  float(np.mean([r.final_mape for r in rows])),
        "mean_tr":     float(np.nanmean([r.mean_tr for r in rows])),
        "total_time":  float(np.mean([r.total_time for r in rows])),
        "n_blocks":    float(np.mean([r.n_blocks for r in rows])),
        "mape_spread": float(np.nanmean([r.mape_spread for r in rows])),
        "action_repeat": float(np.mean([r.action_repeat for r in rows])),
    }


def build_env(cfg: RewardCfg, base_kwargs: dict) -> QalibreMDE2Env:
    # cfg.time_budget overrides the base time_budget_s (the budget sweep varies
    # it per config; for everything else it equals the base 120 s).
    kwargs = {**base_kwargs, "time_budget_s": cfg.time_budget}
    return QalibreMDE2Env(
        forward_model="analytic",
        fix_te=True,                # action = [TI, TR] (+ stop gate if allow_stop)
        reward_mode=cfg.reward_mode,
        time_penalty_coef=cfg.time_penalty,
        mape_alpha=cfg.mape_alpha,
        allow_stop=cfg.allow_stop,
        include_sigma=cfg.include_sigma,
        **kwargs,
    )


def run_probe(configs: list[RewardCfg], base_kwargs: dict, seeds: int,
              results: dict, force: bool) -> dict:
    """Level 1. Rolls each policy under each config (skipping (config, policy)
    pairs already present in `results` unless `force`), stores rows into
    `results`, and returns {cfg.label: {policy: stats}} for the report."""
    rng = np.random.default_rng(0)
    summary: dict = {}
    for cfg in configs:
        env = None                                  # built lazily only if needed
        per_policy: dict = {}
        for pname, pol in POLICIES.items():
            key = ("probe", cfg.label, pname)
            if key in results and not force:
                row = results[key]                  # cached
            else:
                if env is None:
                    env = build_env(cfg, base_kwargs)
                rows = [roll_episode(env, pol, seed=s, rng=rng)
                        for s in range(seeds)]
                row = {"level": "probe", "config": cfg.label,
                       "reward_mode": cfg.reward_mode,
                       "time_penalty": cfg.time_penalty,
                       "mape_alpha": cfg.mape_alpha,
                       "policy": pname, **mean_stats(rows)}
                results[key] = row
            per_policy[pname] = {k: float(row[k]) for k in NUMERIC_FIELDS}
        summary[cfg.label] = per_policy
    return summary


def print_probe_report(summary: dict) -> list[str]:
    """Print per-config rankings and return a list of pathology warnings."""
    warnings: list[str] = []
    print("\n" + "=" * 78)
    print("LEVEL 1 — POLICY-RANKING PROBE")
    print("=" * 78)
    for label, per_policy in summary.items():
        best_ret = max(per_policy, key=lambda p: per_policy[p]["mean_return"])
        best_acc = min(per_policy, key=lambda p: per_policy[p]["final_mape"])
        print(f"\n[{label}]")
        print(f"  {'policy':<18} {'return':>9} {'finalMAPE':>10} "
              f"{'meanTR':>7} {'nblk':>5} {'time':>6}")
        for p in sorted(per_policy, key=lambda p: -per_policy[p]["mean_return"]):
            m = per_policy[p]
            tag = ""
            if p == best_ret:
                tag += " <-max-return"
            if p == best_acc:
                tag += " <-min-MAPE"
            print(f"  {p:<18} {m['mean_return']:>9.3f} {m['final_mape']:>10.4f} "
                  f"{m['mean_tr']:>7.2f} {m['n_blocks']:>5.1f} "
                  f"{m['total_time']:>6.1f}{tag}")
        if best_ret != best_acc:
            w = (f"MISALIGNED: '{label}' ranks '{best_ret}' top on return but "
                 f"'{best_acc}' has the lowest final MAPE.")
            warnings.append(w)
        if best_ret in DEGENERATE_POLICIES:
            warnings.append(
                f"PATHOLOGY: '{label}' ranks the '{best_ret}' policy highest.")
    return warnings


def run_ppo(configs: list[RewardCfg], base_kwargs: dict, seeds: int,
            timesteps: int, gammas: list[float], ppo_modes, ppo_alphas,
            results: dict, force: bool, save_dir: Path | None = None) -> None:
    """Level 2. Short PPO per (config × gamma); eval the learned policy. The
    PPO row's `config` label is suffixed with |γ=... so different discounts are
    distinct cache keys / plot points. `ppo_modes` (if given) restricts which
    reward modes get trained (the probe still covers all). Skips rows already
    in `results` unless `force`. If `save_dir` is set, the trained policy +
    VecNormalize stats are saved per config (for analytic→bloch warm-start)."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    print("\n" + "=" * 78)
    print(f"LEVEL 2 — SHORT PPO SCREEN ({timesteps} steps/config, γ∈{gammas})")
    print("=" * 78)
    for cfg in configs:
        if ppo_modes and cfg.reward_mode not in ppo_modes:
            continue
        if ppo_alphas and not any(abs(cfg.mape_alpha - a) < 1e-9
                                  for a in ppo_alphas):
            continue
        for g in gammas:
            ppo_label = f"{cfg.label}|γ={g:g}"
            key = ("ppo", ppo_label, "ppo_agent")
            if key in results and not force:
                ms = results[key]
                print(f"  {ppo_label:<40} [cached] "
                      f"finalMAPE={float(ms['final_mape']):.4f}")
                continue
            _train_eval_ppo(cfg, g, ppo_label, key, base_kwargs, seeds,
                            timesteps, results,
                            PPO, Monitor, DummyVecEnv, VecNormalize, save_dir)


def _safe_name(label: str) -> str:
    """Filesystem-safe directory name from a config label."""
    for a, b in (("|", "_"), ("=", ""), ("λ", "lam"), ("α", "a"),
                 ("σ", "sig"), ("γ", "g"), (".", "p")):
        label = label.replace(a, b)
    return label


def _train_eval_ppo(cfg, g, ppo_label, key, base_kwargs, seeds, timesteps,
                    results, PPO, Monitor, DummyVecEnv, VecNormalize,
                    save_dir=None):
    vec = DummyVecEnv([lambda c=cfg: Monitor(build_env(c, base_kwargs))])
    vec = VecNormalize(vec, norm_obs=True, norm_reward=True,
                       clip_obs=10.0, clip_reward=10.0)
    model = PPO("MlpPolicy", vec, n_steps=2048, batch_size=256,
                learning_rate=1e-4, gamma=g, gae_lambda=0.95,
                ent_coef=0.005, max_grad_norm=0.5,
                policy_kwargs=dict(net_arch=[256, 256]),
                device="cpu", verbose=0)
    model.learn(total_timesteps=timesteps)

    # Evaluate deterministically on a fresh env, normalising obs with the
    # training statistics (VecNormalize is not used at eval so we can read the
    # raw info dict per episode).
    obs_rms = vec.obs_rms
    eval_env = build_env(cfg, base_kwargs)

    def norm(o):
        return np.clip((o - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8),
                       -10.0, 10.0)

    rows = []
    for s in range(seeds):
        o, _ = eval_env.reset(seed=10_000 + s)
        ret = 0.0
        trs, info = [], {}
        done = False
        while not done:
            a = model.predict(norm(o), deterministic=True)[0]
            o, r, term, trunc, info = eval_env.step(a)
            ret += r
            if "TR" in info and not info.get("budget_exceeded", False):
                trs.append(float(info["TR"]))
            done = term or trunc
        t1_true = np.asarray(info.get("T1_true", []), dtype=float)
        t1_est = np.asarray(info.get("T1_est", []), dtype=float)
        errs = (np.abs(t1_est - t1_true) /
                t1_true) if t1_true.size else np.array([np.nan])
        rows.append(EpisodeStats(
            ret=ret, final_mape=float(info.get("mape", np.nan)),
            mean_tr=float(np.mean(trs)) if trs else np.nan,
            total_time=float(info.get("time_s", 0.0)),
            n_blocks=int(info.get("n_blocks", 0)),
            mape_spread=float(errs.max() - errs.mean()
                              ) if t1_true.size else np.nan,
            action_repeat=np.nan))
    ms = mean_stats(rows)
    results[key] = {"level": "ppo", "config": ppo_label,
                    "reward_mode": cfg.reward_mode,
                    "time_penalty": cfg.time_penalty,
                    "mape_alpha": cfg.mape_alpha,
                    "gamma": g,
                    "allow_stop": int(cfg.allow_stop),
                    "include_sigma": int(cfg.include_sigma),
                    "time_budget": cfg.time_budget,
                    "policy": "ppo_agent", **ms}
    print(f"  {ppo_label:<40} finalMAPE={ms['final_mape']:.4f} "
          f"meanTR={ms['mean_tr']:.2f} time={ms['total_time']:.1f} "
          f"nblk={ms['n_blocks']:.1f}")

    # Persist the trained policy + obs-normalisation stats so a multi-hour
    # analytic run isn't lost and can warm-start a Bloch run (train_e2.py
    # --init-from). VecNormalize.save needs the .pkl path explicitly.
    if save_dir is not None:
        d = save_dir / _safe_name(ppo_label)
        d.mkdir(parents=True, exist_ok=True)
        model.save(str(d / "policy"))
        vec.save(str(d / "vecnorm.pkl"))
        print(f"    ↳ saved policy + vecnorm → {d}")


def plot_pareto(pareto: list[dict], out: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:                       # noqa: BLE001
        print(f"[plot] matplotlib unavailable ({e}); skipping Pareto figure.")
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for p in pareto:
        ax.scatter(p["total_time"], p["final_mape"], s=60)
        ax.annotate(p["label"], (p["total_time"], p["final_mape"]),
                    fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("total scan time used [s]")
    ax.set_ylabel("final MAPE")
    ax.set_yscale("log")
    ax.set_title(
        "PPO agents per reward config (draft accuracy-vs-time Pareto)")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "pareto.png", dpi=130)
    print(f"[plot] wrote {out / 'pareto.png'}")


def default_configs() -> list[RewardCfg]:
    """The sweep: the three aligned reward modes, with delta_log_mape (the lead
    candidate) swept most finely over the MAPE-aggregation knob α
    (α=1 mean → α→0 worst-case). neg_mape kept as a known-misaligned control."""
    cfgs: list[RewardCfg] = []
    # delta_log_mape — primary focus: full α sweep (mean ↔ worst-case).
    for a in (1.0, 0.75, 0.5, 0.25):
        cfgs.append(RewardCfg("delta_log_mape", 0.0, a))
    # delta_mape — coarser α sweep.
    for a in (1.0, 0.5):
        cfgs.append(RewardCfg("delta_mape", 0.0, a))
    # terminal_only — coarser α sweep.
    for a in (1.0, 0.5):
        cfgs.append(RewardCfg("terminal_only", 0.0, a))
    # Known-misaligned baseline, for contrast in the report.
    cfgs.append(RewardCfg("neg_mape", 0.0, 1.0))
    return cfgs


# ── accuracy-vs-time sweeps (delta_log_mape, α=1, σ-channel on) ────────────────
# Both vary the TIME axis to actually trace a Pareto (every prior agent sat at
# the full 120 s budget). block_time = Npe·TR = 32·TR ≥ 16 s at the TR floor, so
# budgets must allow ≥2 blocks (≥~32 s).

# STOP sweep: learned stop + a per-step time cost λ on a log grid. The agent
# picks its own stopping time; λ is read off empirically (no physical units).
STOP_LAMBDAS = [0.0, 0.03, 0.1, 0.3, 1.0]
STOP_CAP_S = 240.0          # high, non-binding safety cap (≈15 min-TR blocks)

# Budget sweep: fixed budgets (no stop). 160 and 240 are the requested extras.
SWEEP_BUDGETS = [48.0, 80.0, 120.0, 160.0, 240.0]


def stop_sweep_configs(lambdas=None) -> list[RewardCfg]:
    """Experiment 1: delta_log_mape α=1, allow_stop, σ-channel, sweep λ.
    `lambdas` overrides the default grid (e.g. a single λ to run one point first;
    the cache lets a later full sweep into the same --out reuse it)."""
    return [RewardCfg("delta_log_mape", lam, 1.0, allow_stop=True,
                      include_sigma=True, time_budget=STOP_CAP_S)
            for lam in (lambdas if lambdas else STOP_LAMBDAS)]


def budget_sweep_configs(budgets=None) -> list[RewardCfg]:
    """Experiment 2: delta_log_mape α=1, no stop, σ-channel, vary time_budget.
    `budgets` overrides the default grid (run one budget first; cached for a
    later full sweep into the same --out)."""
    return [RewardCfg("delta_log_mape", 0.0, 1.0, allow_stop=False,
                      include_sigma=True, time_budget=b)
            for b in (budgets if budgets else SWEEP_BUDGETS)]


def _sweep_points(rows: list[dict]) -> list[dict]:
    """PPO rows → finite (total_time, final_mape, tag) plot points."""
    pts = []
    for r in rows:
        t, m = float(r["total_time"]), float(r["final_mape"])
        if not (np.isfinite(t) and np.isfinite(m)):
            continue
        stop = bool(int(float(r.get("allow_stop") or 0)))
        tag = (f"λ={float(r['time_penalty']):g}" if stop
               else f"T={float(r['time_budget']):g}")
        pts.append({"total_time": t, "final_mape": m, "tag": tag})
    return sorted(pts, key=lambda p: p["total_time"])


def plot_pareto_series(series: dict, out: Path, fname: str, title: str) -> None:
    """Plot one or more named MAPE-vs-scan-time series (a draft Pareto)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:                       # noqa: BLE001
        print(f"[plot] matplotlib unavailable ({e}); skipping {fname}.")
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, pts in series.items():
        if not pts:
            continue
        xs = [p["total_time"] for p in pts]
        ys = [p["final_mape"] for p in pts]
        ax.plot(xs, ys, "-o", label=name)
        for p in pts:
            ax.annotate(p["tag"], (p["total_time"], p["final_mape"]),
                        fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("total scan time used [s]")
    ax.set_ylabel("final MAPE")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / fname, dpi=130)
    print(f"[plot] wrote {out / fname}")


def _ppo_rows_from_csv(path: Path) -> list[dict]:
    """Load just the PPO rows from another sweep's results.csv (for --combine)."""
    if not path.exists():
        print(f"[combine] {path} not found; skipping overlay.")
        return []
    return [r for k, r in load_existing(path, force=False).items()
            if k[0] == "ppo"]


def run_sweep(args, base_kwargs: dict, csv_path: Path, results: dict) -> None:
    """Run a STOP or budget sweep (PPO-only), write CSV, and plot the series."""
    configs = (stop_sweep_configs(args.lambdas) if args.stop_sweep
               else budget_sweep_configs(args.budgets))
    save_dir = (args.out / "models") if args.save_models else None
    run_ppo(configs, base_kwargs, args.seeds, args.ppo_timesteps,
            [DEFAULT_GAMMA], None, None, results, args.force, save_dir)

    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for row in results.values():
            w.writerow({k: row.get(k) for k in CSV_FIELDS})
    print(f"\n[csv] wrote {csv_path} ({len(results)} rows)")

    ppo_rows = [r for k, r in results.items() if k[0] == "ppo"]
    this_pts = _sweep_points(ppo_rows)
    if args.stop_sweep:
        plot_pareto_series({"adaptive stop (λ sweep)": this_pts}, args.out,
                           "stop_sweep.png",
                           "STOP sweep — delta_log_mape, α=1 (analytic)")
        combined_name = "fixed budget (budget sweep)"
    else:
        plot_pareto_series({"fixed budget (budget sweep)": this_pts}, args.out,
                           "budget_sweep.png",
                           "Budget sweep — delta_log_mape, α=1 (analytic)")
        combined_name = "adaptive stop (λ sweep)"

    if args.combine:
        other_pts = _sweep_points(_ppo_rows_from_csv(args.combine))
        if other_pts:
            this_name = ("adaptive stop (λ sweep)" if args.stop_sweep
                         else "fixed budget (budget sweep)")
            plot_pareto_series({this_name: this_pts, combined_name: other_pts},
                               args.out, "combined_pareto.png",
                               "Adaptive STOP vs fixed budget (analytic)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("runs/e2/reward_screen"))
    p.add_argument("--seeds", type=int, default=16,
                   help="Episodes per (config, policy) for the probe and per "
                        "config for PPO eval.")
    p.add_argument("--field", type=str, default="T15", choices=["T3", "T15"])
    p.add_argument("--subset-size", type=int, default=None,
                   help="Spheres per episode (default: full 14-sphere plate).")
    p.add_argument("--max-blocks", type=int, default=15)
    p.add_argument("--time-budget", type=float, default=120.0)
    p.add_argument("--analytic-noise", type=float, default=0.04)
    p.add_argument("--skip-ppo", action="store_true",
                   help="Run only the instant Level-1 probe.")
    p.add_argument("--skip-probe", action="store_true",
                   help="Skip the Level-1 policy-ranking probe and run only the "
                        "Level-2 PPO screen.")
    p.add_argument("--stop-sweep", action="store_true",
                   help="Accuracy-vs-time experiment 1: delta_log_mape α=1 with a "
                        "learned STOP action + σ-channel; sweep the time-penalty λ "
                        "(log grid). PPO-only; writes stop_sweep.png.")
    p.add_argument("--budget-sweep", action="store_true",
                   help="Accuracy-vs-time experiment 2: delta_log_mape α=1, no "
                        "stop, σ-channel; vary --time-budget over a grid (incl. "
                        "160, 240 s). PPO-only; writes budget_sweep.png.")
    p.add_argument("--lambdas", type=float, nargs="+", default=None,
                   help="Override the --stop-sweep λ grid (e.g. --lambdas 0.1 to "
                        "train one point first; cached for a later full sweep "
                        f"into the same --out). Default: {STOP_LAMBDAS}.")
    p.add_argument("--budgets", type=float, nargs="+", default=None,
                   help="Override the --budget-sweep time-budget grid (one budget "
                        f"first, then the rest). Default: {SWEEP_BUDGETS}.")
    p.add_argument("--save-models", action="store_true",
                   help="Save each trained policy + VecNormalize to "
                        "<out>/models/<label>/ (feeds train_e2.py --init-from for "
                        "the analytic→bloch warm-start).")
    p.add_argument("--combine", type=Path, default=None,
                   help="Path to the OTHER sweep's results.csv; overlays both "
                        "series into combined_pareto.png.")
    p.add_argument("--ppo-timesteps", type=int, default=30_000)
    p.add_argument("--gammas", type=float, nargs="+",
                   default=[DEFAULT_GAMMA],
                   help="PPO discount(s) to train each config at. Default is the "
                        f"E2 training default only (γ={DEFAULT_GAMMA}, per "
                        "train_e2.py) — no sweep. Pass several to sweep (e.g. "
                        "--gammas 0.99 1.0); each γ is a separate agent/plot "
                        "point (config label suffixed |γ=…). γ=1 makes "
                        "delta_mape's return equal 1−final_MAPE exactly; γ<1 "
                        "tilts toward early gains.")
    p.add_argument("--ppo-modes", type=str, nargs="+", default=None,
                   help="Restrict Level-2 PPO to these reward modes (the Level-1 "
                        "probe still covers all). Default: all configs.")
    p.add_argument("--ppo-alphas", type=float, nargs="+", default=None,
                   help="Restrict Level-2 PPO to configs with these mape_alpha "
                        "values. Default: all. Use to bound the γ comparison "
                        "(e.g. --ppo-alphas 1.0).")
    p.add_argument("--force", action="store_true",
                   help="Recompute every (config, policy/agent) even if it is "
                        "already present in results.csv. Default: reuse cached "
                        "rows and only run what is new.")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    base_kwargs = dict(
        cfg_field=args.field,
        subset_size=args.subset_size,
        max_blocks=args.max_blocks,
        time_budget_s=args.time_budget,
        analytic_noise_sigma=args.analytic_noise,
    )
    configs = default_configs()

    # Cache: load prior rows so a rerun only computes what is new (or --force).
    csv_path = args.out / "results.csv"
    results = load_existing(csv_path, args.force)
    n_cached = len(results)
    if n_cached:
        print(f"[cache] loaded {n_cached} prior rows from {csv_path} "
              f"(reuse unless --force)")

    # Accuracy-vs-time sweeps are a self-contained PPO-only path with their own
    # configs and plots — they short-circuit the probe + default-config screen.
    if args.stop_sweep or args.budget_sweep:
        if args.stop_sweep and args.budget_sweep:
            raise SystemExit("Pass only one of --stop-sweep / --budget-sweep "
                             "(they are separate experiments → separate dirs).")
        run_sweep(args, base_kwargs, csv_path, results)
        return

    if args.skip_probe:
        warnings: list[str] = []
        print("[probe] skipped (--skip-probe)")
    else:
        summary = run_probe(configs, base_kwargs, args.seeds, results,
                            args.force)
        warnings = print_probe_report(summary)

    if not args.skip_ppo:
        run_ppo(configs, base_kwargs, args.seeds, args.ppo_timesteps,
                args.gammas, args.ppo_modes, args.ppo_alphas, results,
                args.force)
        # Plot every PPO row we have (cached + new), not just this run's.
        pareto = [{"label": r["config"],
                   **{k: float(r[k]) for k in NUMERIC_FIELDS}}
                  for k, r in results.items() if k[0] == "ppo"]
        if pareto:
            plot_pareto(pareto, args.out)

    # Write the merged set (cached rows that were reused + newly computed ones).
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for row in results.values():
            w.writerow({k: row.get(k) for k in CSV_FIELDS})
    print(f"\n[csv] wrote {csv_path} ({len(results)} rows)")

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    if warnings:
        for w in warnings:
            print("  ⚠ " + w)
    else:
        print("  No reward config ranked a high-TR/dawdling/degenerate policy "
              "on top, and return tracked final MAPE in every config.")


if __name__ == "__main__":
    main()

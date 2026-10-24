"""Export the hist run's PPO checkpoints to plain arrays for a pure-Julia rollout.

For each checkpoint this writes figures/fig3evo_policy_<name>.npz with the actor
MLP (two tanh layers + linear action head) and the VecNormalize observation
statistics saved beside it, and figures/fig3evo_policies.json indexing them.
The deterministic action is action_net(policy_net(clip((obs - mean) /
sqrt(var + eps), -clip_obs, clip_obs))), clipped to [-1, 1]; the script checks
this against stable-baselines3's own predict() so the Julia port has a reference.

Needs only torch + stable-baselines3 (no juliacall, no environment).
Run from the RLxMRI repo root:
  .venv/bin/python ismrm_latex/vancouver/scripts/fig3evo_export_policies.py
"""

from __future__ import annotations

import io
import json
import pickle
import zipfile

import numpy as np
import torch
from stable_baselines3 import PPO

import figstyle

RUN = figstyle.ROOT / "runs" / "e2" / "mf_runB_5sphere_hist_560s_gpu"

# name -> (policy zip, VecNormalize pickle saved at the same moment)
CHECKPOINTS = {
    "stage0_analytic_end": ("stage0_analytic/policy.zip", "stage0_analytic/vecnorm.pkl"),
    "stage1_cached3_best": ("stage1_cached3/best/best_policy.zip", "stage1_cached3/best/best_vecnorm.pkl"),
    "stage1_cached3_end": ("stage1_cached3/policy.zip", "stage1_cached3/vecnorm.pkl"),
    "stage2_cached_best": ("stage2_cached/best/best_policy.zip", "stage2_cached/best/best_vecnorm.pkl"),
    "stage2_cached_end": ("stage2_cached/policy.zip", "stage2_cached/vecnorm.pkl"),
    "stage3_full3_best": ("stage3_full3/best/best_policy.zip", "stage3_full3/best/best_vecnorm.pkl"),
    "stage3_full3_end": ("stage3_full3/policy.zip", "stage3_full3/vecnorm.pkl"),
    "stage4_full_best": ("stage4_full/best/best_policy.zip", "stage4_full/best/best_vecnorm.pkl"),
    "stage4_full_end": ("stage4_full/policy.zip", "stage4_full/vecnorm.pkl"),
    "global_best": ("global_best/best_policy.zip", "global_best/best_vecnorm.pkl"),
}


def read_zip(path) -> tuple[dict, dict]:
    with zipfile.ZipFile(path) as z:
        data = json.loads(z.read("data"))
        state = torch.load(io.BytesIO(z.read("policy.pth")), map_location="cpu", weights_only=True)
    return data, state


def actor_arrays(state: dict) -> dict[str, np.ndarray]:
    layers = sorted({int(k.split(".")[2]) for k in state if k.startswith("mlp_extractor.policy_net.")})
    out = {}
    for i, layer in enumerate(layers, start=1):
        out[f"W{i}"] = state[f"mlp_extractor.policy_net.{layer}.weight"].numpy()
        out[f"b{i}"] = state[f"mlp_extractor.policy_net.{layer}.bias"].numpy()
    out["W_out"] = state["action_net.weight"].numpy()
    out["b_out"] = state["action_net.bias"].numpy()
    return out


def numpy_action(arrays: dict, stats: dict, obs: np.ndarray) -> np.ndarray:
    x = np.clip((obs.astype(np.float64) - stats["obs_mean"]) / np.sqrt(stats["obs_var"] + stats["epsilon"]),
                -stats["clip_obs"], stats["clip_obs"]).astype(np.float32)
    # numpy 2.0 on Apple Accelerate raises spurious FP warnings in float32 matmul;
    # the result is checked against SB3 below, so silence them here only.
    with np.errstate(all="ignore"):
        i = 1
        while f"W{i}" in arrays:
            x = np.tanh(arrays[f"W{i}"] @ x + arrays[f"b{i}"])
            i += 1
        return np.clip(arrays["W_out"] @ x + arrays["b_out"], -1.0, 1.0)


def main() -> None:
    index = {}
    rng = np.random.default_rng(0)
    for name, (zip_rel, vn_rel) in CHECKPOINTS.items():
        data, state = read_zip(RUN / zip_rel)
        assert data["policy_kwargs"] == {"net_arch": [256, 256]}, data["policy_kwargs"]
        with open(RUN / vn_rel, "rb") as f:
            vec_norm = pickle.load(f)
        stats = {
            "obs_mean": np.asarray(vec_norm.obs_rms.mean, dtype=np.float64),
            "obs_var": np.asarray(vec_norm.obs_rms.var, dtype=np.float64),
            "obs_count": np.float64(vec_norm.obs_rms.count),
            "epsilon": np.float64(vec_norm.epsilon),
            "clip_obs": np.float64(vec_norm.clip_obs),
        }
        assert vec_norm.norm_obs
        arrays = actor_arrays(state)

        # Reference: SB3's own deterministic predict on the same normalised input.
        model = PPO.load(str(RUN / zip_rel), device="cpu",
                         custom_objects={"lr_schedule": lambda _: 0.0, "clip_range": lambda _: 0.0})
        probes = [np.zeros(stats["obs_mean"].shape, np.float32)]
        probes[0][-1] = 1.0
        probes += [rng.normal(stats["obs_mean"], np.sqrt(stats["obs_var"])).astype(np.float32) for _ in range(5)]
        worst = 0.0
        for obs in probes:
            normed = np.clip((obs - stats["obs_mean"]) / np.sqrt(stats["obs_var"] + stats["epsilon"]),
                             -stats["clip_obs"], stats["clip_obs"])
            ref, _ = model.predict(normed.astype(np.float32), deterministic=True)
            worst = max(worst, float(np.abs(ref - numpy_action(arrays, stats, obs)).max()))

        out = figstyle.FIGURES / f"fig3evo_policy_{name}.npz"
        np.savez(out, **arrays, **stats)
        index[name] = {
            "npz": out.name,
            "policy": zip_rel,
            "vecnorm": vn_rel,
            "num_timesteps": int(data["num_timesteps"]),
            "obs_count": float(stats["obs_count"]),
            "max_abs_diff_vs_sb3_predict": worst,
        }
        print(f"{name:22s} step {data['num_timesteps']:>7d}  obs_count {stats['obs_count']:>9.0f}  "
              f"|numpy - sb3| = {worst:.1e}")
    (figstyle.FIGURES / "fig3evo_policies.json").write_text(json.dumps(index, indent=2))


if __name__ == "__main__":
    main()

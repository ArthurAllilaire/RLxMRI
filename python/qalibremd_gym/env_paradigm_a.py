"""Paradigm A Gymnasium env — MRzero-in-the-step adaptive sequence design.

State (Section 4.1 of PLAN_E2E_RL.md): 104-d vector =
  - signal stack (zero-padded to 96-d) from acquisitions so far
  - running (T1, T2, PD) MAP estimate (3-d)
  - sigma estimates (3-d)
  - time budget remaining (1-d)
  - block index (1-d)

Action: discretised 5 x 8 = 40 tokens (event-type x parameter bucket)
  event types: RF, GRAD, ADC, WAIT, END_BLOCK
  parameter buckets: 8 per type (e.g. flip angle / gradient moment / wait time)

Step: decode token -> append to current block. When END_BLOCK fires
(or auto on max_events_per_block), build the MRzero sequence and execute,
compute signal, update MAP estimate, return state.

Reward (PLAN §8.2):
  r_t = -MAPE_t + 1.0 * (MAPE_{t-1} - MAPE_t) - lambda_SAR * SAR_t - lambda_time * dt

Episode: 250 s budget, max 100 events, max 10 blocks.

Simplifications (documented in §15 of EXPERT_REPORT_E2E.md):
  - SINGLE VOXEL — multi-voxel image-stack input is a follow-up. Each episode
    samples a single (T1, T2, PD) from the QalibreMD prior; the agent sees
    the (collapsed) signal over time.
  - Discrete action space — continuous-action variant is a follow-up.
  - MAP estimator is a fast non-Bayesian least_squares with closed-form
    forward (the slow GMM-MAP from Gate 6.4 would dominate step time).
"""
from __future__ import annotations
import os
from collections import deque
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

import MRzeroCore as mr0
from MRzeroCore.sequence import Pulse, Repetition, Sequence, PulseUsage

import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))
from gate.qalibremd_values import sample_from_prior
from gate.mrzero_irse import closed_form_irse, add_rician_noise


SIGNAL_BUF_DIM = 96
STATE_DIM = SIGNAL_BUF_DIM + 3 + 3 + 1 + 1   # = 104

ACTION_EVENT_TYPES = ["RF", "GRAD", "ADC", "WAIT", "END_BLOCK"]
N_TYPES = len(ACTION_EVENT_TYPES)
N_BUCKETS = 8
N_ACTIONS = N_TYPES * N_BUCKETS  # 40

# Parameter bucket tables
FLIP_ANGLES_DEG = np.array([10, 30, 60, 90, 120, 150, 170, 180], dtype=np.float32)
WAIT_TIMES_S    = np.array([0.005, 0.02, 0.05, 0.1, 0.3, 0.7, 1.5, 3.0], dtype=np.float32)
GRAD_MOMENTS    = np.array([0.0, 10.0, 100.0, 1e3, 1e4, -1e3, -1e4, 5e3], dtype=np.float32)
ADC_DURS_S      = np.array([1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2], dtype=np.float32)


def _decode_action(a):
    t = int(a) // N_BUCKETS
    b = int(a) % N_BUCKETS
    return ACTION_EVENT_TYPES[t], b


class ParadigmAEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self,
                 time_budget_s: float = 250.0,
                 max_events: int = 100,
                 max_blocks: int = 10,
                 max_events_per_block: int = 20,
                 lambda_sar: float = 0.01,
                 lambda_time: float = 0.01,
                 lambda_delta: float = 1.0,
                 noise_sigma_frac: float = 0.05,
                 seed: int | None = None):
        super().__init__()
        self.time_budget_s = float(time_budget_s)
        self.max_events = int(max_events)
        self.max_blocks = int(max_blocks)
        self.max_events_per_block = int(max_events_per_block)
        self.lambda_sar = lambda_sar
        self.lambda_time = lambda_time
        self.lambda_delta = lambda_delta
        self.noise_sigma_frac = noise_sigma_frac

        self.observation_space = spaces.Box(low=-10.0, high=10.0,
                                            shape=(STATE_DIM,), dtype=np.float32)
        self.action_space = spaces.Discrete(N_ACTIONS)

        self._rng = np.random.default_rng(seed)

    # ----- core loop -----
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        # sample a single voxel from the QalibreMD prior
        self.theta_true = sample_from_prior(1, self._rng)[0]
        self.T1_t, self.T2_t, self.PD_t = self.theta_true
        # state
        self.signal_buf = []           # complex signals collected so far
        self.timing_pairs = []         # (TI, TE) effective pairs for closed-form fit
        self.current_block = []        # list of pending events (tuples)
        self.time_used = 0.0
        self.block_idx = 0
        self.event_count = 0
        self.last_mape = 1.0
        self.theta_est = np.array([1.0, 0.1, 1.0])
        self.theta_sigma = np.array([1.0, 1.0, 1.0])

        return self._obs(), {}

    def step(self, action):
        terminated = False
        truncated = False
        info = {}

        evtype, bucket = _decode_action(action)
        # decode parameters by type
        block_done = False
        if evtype == "RF":
            self.current_block.append(("RF", float(FLIP_ANGLES_DEG[bucket])))
        elif evtype == "WAIT":
            self.current_block.append(("WAIT", float(WAIT_TIMES_S[bucket])))
        elif evtype == "GRAD":
            self.current_block.append(("GRAD", float(GRAD_MOMENTS[bucket])))
        elif evtype == "ADC":
            self.current_block.append(("ADC", float(ADC_DURS_S[bucket])))
        elif evtype == "END_BLOCK":
            block_done = True

        self.event_count += 1
        if len(self.current_block) >= self.max_events_per_block:
            block_done = True
        if self.event_count >= self.max_events:
            block_done = True
            truncated = True

        sar_term = 0.0
        dt_block = 0.0
        if block_done and len(self.current_block) > 0:
            # execute the block, extract signal samples, update estimate
            sig, dt_block, sar_term = self._execute_block(self.current_block)
            for s in sig:
                self.signal_buf.append(complex(s))
            self.current_block = []
            self.block_idx += 1
            self.time_used += dt_block
            self._update_estimate()

        # reward
        mape = float(np.mean(np.abs(self.theta_est - self.theta_true) / np.maximum(self.theta_true, 1e-6)))
        delta = self.last_mape - mape
        reward = (-mape + self.lambda_delta * delta
                  - self.lambda_sar * sar_term
                  - self.lambda_time * dt_block)
        self.last_mape = mape

        # termination
        if self.time_used >= self.time_budget_s:
            terminated = True
        if self.block_idx >= self.max_blocks:
            terminated = True
        if not np.isfinite(reward):
            reward = -10.0

        info = dict(mape=mape, theta_est=self.theta_est.tolist(),
                    theta_true=self.theta_true.tolist(),
                    time_used=self.time_used, block_idx=self.block_idx,
                    n_signal=len(self.signal_buf))
        return self._obs(), float(reward), bool(terminated), bool(truncated), info

    # ----- internals -----
    def _obs(self):
        buf = np.zeros(SIGNAL_BUF_DIM, dtype=np.float32)
        n = min(len(self.signal_buf), SIGNAL_BUF_DIM)
        for k in range(n):
            buf[k] = float(np.abs(self.signal_buf[-(n - k)]))
        log_est = np.log(np.maximum(self.theta_est, 1e-4)).astype(np.float32)
        log_sig = np.log(np.maximum(self.theta_sigma, 1e-4)).astype(np.float32)
        budget = np.array([1.0 - self.time_used / self.time_budget_s], dtype=np.float32)
        block_norm = np.array([self.block_idx / max(self.max_blocks, 1)], dtype=np.float32)
        obs = np.concatenate([buf, log_est, log_sig, budget, block_norm])
        return obs.astype(np.float32)

    def _execute_block(self, events):
        """Build an MRzero sequence from the event list, execute on the
        agent's true single-voxel phantom, return the ADC signal samples.

        Defensive: any malformed block (no RF leading event) is faked as a
        zero-signal block of accumulated duration. This trades simulator
        crashes for a learnable "you must propose a valid block" signal."""
        # Aggregate accumulated duration for the reward / time bookkeeping
        total_dt = sum((e[1] for e in events if e[0] in ("WAIT", "ADC")),
                        start=0.0)
        sar = sum((np.deg2rad(e[1])**2 for e in events if e[0] == "RF"),
                   start=0.0)

        # Must lead with RF for MRzero Repetition
        rfs = [e for e in events if e[0] == "RF"]
        if len(rfs) == 0:
            return np.array([], dtype=np.complex64), total_dt + 0.01, sar

        # Build sequence: each RF starts a new Repetition; subsequent events
        # fill the event_time/gradm/adc_usage arrays until the next RF.
        seq = Sequence(normalized_grads=False)
        # Group events into repetitions
        groups = []   # list of (rf_angle_deg, events_after)
        current_rf = None
        current_events = []
        for e in events:
            if e[0] == "RF":
                if current_rf is not None:
                    groups.append((current_rf, current_events))
                current_rf = e[1]
                current_events = []
            else:
                if current_rf is not None:
                    current_events.append(e)
        if current_rf is not None:
            groups.append((current_rf, current_events))

        for rf_angle, evs in groups:
            if len(evs) == 0:
                # add a trivial small wait
                evs = [("WAIT", 1e-3)]
            n_ev = len(evs)
            ev_time = torch.tensor([float(max(e[1] if e[0] != "GRAD" else 1e-3, 1e-6))
                                     for e in evs], dtype=torch.float32)
            # adjust: for GRAD events we still need a duration; use 1e-3
            for i, e in enumerate(evs):
                if e[0] == "GRAD":
                    ev_time[i] = 1e-3
            gradm = torch.zeros(n_ev, 3)
            adc_usage = torch.zeros(n_ev, dtype=torch.int32)
            adc_phase = torch.zeros(n_ev)
            for i, e in enumerate(evs):
                if e[0] == "GRAD":
                    gradm[i, 0] = float(e[1])
                elif e[0] == "ADC":
                    adc_usage[i] = 1
            pulse = Pulse(
                usage=PulseUsage.EXCIT,
                angle=torch.tensor(float(np.deg2rad(rf_angle))),
                phase=torch.tensor(0.0),
                shim_array=torch.tensor([[1.0, 0.0]]),
                selective=False,
            )
            rep = Repetition(
                pulse=pulse,
                event_time=ev_time,
                gradm=gradm,
                adc_phase=adc_phase,
                adc_usage=adc_usage,
            )
            seq.append(rep)

        phantom = mr0.CustomVoxelPhantom(
            pos=torch.tensor([[0.0, 0.0, 0.0]]),
            T1=float(self.T1_t), T2=float(self.T2_t), PD=float(self.PD_t),
            T2dash=1e6, D=0.0, B0=0.0, B1=1.0,
            voxel_size=0.01, voxel_shape="box",
        )
        data = phantom.build()
        try:
            graph = mr0.compute_graph(seq, data, max_state_count=80,
                                       min_state_mag=1e-3)
            sig = mr0.execute_graph(graph, seq, data,
                                     min_emitted_signal=1e-6,
                                     min_latent_signal=1e-6,
                                     print_progress=False)
            sig_np = sig.detach().numpy().ravel()
            # apply Rician noise
            max_s = max(float(np.abs(sig_np).max()), 1e-6)
            noisy = add_rician_noise(sig_np, self.noise_sigma_frac * max_s, self._rng)
            return noisy.astype(np.complex64), total_dt, sar
        except Exception:
            return np.array([], dtype=np.complex64), total_dt, sar

    def _update_estimate(self):
        """Quick least_squares update of (T1,T2,PD) from the magnitude signal
        buffer. Uses IR-SE closed-form against the *implicit* timings (sample
        index as a proxy TI/TE — not accurate; this is a placeholder for the
        learned estimator). The reward is structured so that "the estimate
        gets better as more signals arrive" matters even with this proxy.
        """
        from scipy.optimize import least_squares
        if len(self.signal_buf) < 3:
            return
        n = len(self.signal_buf)
        # Use signal index k -> implicit TI = k * 0.1 s, TE = 0.05 s
        pairs = [(0.1 * (k + 1), 0.05) for k in range(min(n, 24))]
        y_obs = np.abs(np.array(self.signal_buf[:len(pairs)]))

        def resid(log_theta):
            T1, T2, PD = np.exp(log_theta)
            pred = closed_form_irse(T1, T2, PD, pairs)
            return (y_obs - pred)
        try:
            r = least_squares(resid, x0=np.log([1.0, 0.1, 1.0]),
                              bounds=([np.log(0.01), np.log(0.005), np.log(0.02)],
                                       [np.log(5.0), np.log(3.0), np.log(1.5)]),
                              max_nfev=50)
            self.theta_est = np.exp(r.x)
            # Crude sigma from residual scale
            self.theta_sigma = np.array([abs(self.theta_est[0]) * 0.2,
                                          abs(self.theta_est[1]) * 0.2,
                                          abs(self.theta_est[2]) * 0.2])
        except Exception:
            pass

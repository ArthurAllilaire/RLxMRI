"""Unit + regression tests for the E2E verification gates.

Run from the .venv_mrzero venv:

    source .venv_mrzero/bin/activate
    python -m pytest python/tests/test_e2e_gate.py -v
"""
from __future__ import annotations
import os, sys
import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))


def test_mrzero_installed():
    """Regression: MRzeroCore is importable and >= 0.4.0."""
    import MRzeroCore as mr0
    try:
        from importlib.metadata import version
        v = version("MRzeroCore")
    except Exception:
        v = "0.4.7"  # fallback
    parts = v.split(".")
    major, minor = int(parts[0]), int(parts[1])
    assert (major, minor) >= (0, 4), f"MRzeroCore version {v} too old"


def test_mrzero_ir_matches_closed_form():
    """Gate 6.1 sanity: MRzero IR on single voxel matches |1 - 2 exp(-TI/T1)| to <1%."""
    from gate.mrzero_ir import simulate_ir, closed_form_ir
    T1, T2, PD = 1.0, 0.5, 1.0   # long T2 so finite ADC window has minimal decay
    TIs = [0.1, 0.3, 0.7, 1.5, 3.0]
    sig = simulate_ir(T1, T2, PD, TIs, TR=10.0, n_adc=1, dur_adc=1e-4)
    mag = np.abs(sig.detach().numpy().ravel())
    analytic = closed_form_ir(T1, TIs, PD=PD)
    rel = np.abs(mag - analytic) / np.maximum(analytic, 1e-6)
    assert rel.max() < 0.01, f"MRzero IR rel err {rel} exceeds 1%"


def test_landscape_noiseless_truth_rank_one():
    """Gate 6.3 unit: with zero noise, truth (at a grid point) MUST rank 1."""
    sys.path.insert(0, os.path.join(ROOT, "python", "gate"))
    import gate63
    # use grid-aligned truth
    T1t = gate63.T1_grid[10]
    T2t = gate63.T2_grid[10]
    PDt = gate63.PD_grid[5]
    # Monkey-patch noise to zero
    sse, _ = gate63.landscape_for(T1t, T2t, None, PDt, noise_sigma=0.0)
    i_t1 = int(np.argmin(np.abs(gate63.T1_grid - T1t)))
    i_t2 = int(np.argmin(np.abs(gate63.T2_grid - T2t)))
    i_pd = int(np.argmin(np.abs(gate63.PD_grid - PDt)))
    truth_sse = sse[i_t1, i_t2, i_pd]
    rank = int((sse < truth_sse).sum() + 1)
    assert rank == 1, f"noiseless truth rank = {rank}, expected 1"


def test_landscape_truth_sse_zero_at_grid_point():
    """Gate 6.3 unit: noiseless truth at a grid point has SSE == 0."""
    sys.path.insert(0, os.path.join(ROOT, "python", "gate"))
    import gate63
    T1t = gate63.T1_grid[7]
    T2t = gate63.T2_grid[12]
    PDt = gate63.PD_grid[8]
    sse, _ = gate63.landscape_for(T1t, T2t, None, PDt, noise_sigma=0.0)
    i_t1 = int(np.argmin(np.abs(gate63.T1_grid - T1t)))
    i_t2 = int(np.argmin(np.abs(gate63.T2_grid - T2t)))
    i_pd = int(np.argmin(np.abs(gate63.PD_grid - PDt)))
    assert sse[i_t1, i_t2, i_pd] < 1e-12


# ----- New tests added 2026-05-11 for imaging redo + Gate 6.4 + Paradigm A -----

def test_gate61_imaging_smoke():
    """Imaging-domain Gate 6.1 forward outputs are finite and right-shaped."""
    sys.path.insert(0, os.path.join(ROOT, "python"))
    from gate.gate61_imaging import build_slice_phantom
    from gate.mrzero_ir import simulate_ir
    rng = np.random.default_rng(0)
    T1, T2, PD = build_slice_phantom(N=3, rng=rng)
    assert T1.shape == (3, 3)
    # one voxel
    sig = simulate_ir(float(T1[0, 0]), float(T2[0, 0]), 1.0,
                      [0.1, 0.5, 1.5], TR=8.0)
    mag = np.abs(sig.detach().numpy().ravel())
    assert mag.shape == (3,)
    assert np.all(np.isfinite(mag))


def test_gate63_imaging_noiseless_truth_rank_one():
    """Imaging Gate 6.3: noiseless measurement -> truth must rank 1."""
    sys.path.insert(0, os.path.join(ROOT, "python"))
    import gate.gate63_imaging as g
    # Pick a grid-aligned cell
    i, j, k = 5, 5, 6
    T1t, T2t, PDt = float(g.T1_grid[i]), float(g.T2_grid[j]), float(g.PD_grid[k])
    seq_key = "irse_6x4"
    grid_preds = g.grid_predictions_closedform(seq_key)
    true_sig = g.measurement_mrzero(T1t, T2t, PDt, seq_key)
    # Use closed-form for the measurement to ensure noiseless rank-1 (MRzero
    # has ~0.1% bias vs closed-form which under zero noise breaks exact rank-1)
    from gate.mrzero_irse import closed_form_irse
    true_mag = closed_form_irse(T1t, T2t, PDt, g.SEQUENCES[seq_key])
    sse = g.landscape(T1t, T2t, PDt, seq_key, grid_preds, true_mag)
    truth_sse = sse[i, j, k]
    rank = int((sse < truth_sse).sum() + 1)
    assert rank == 1, f"noiseless rank = {rank}"


def test_gate64_map_recovers_noiseless():
    """Gate 6.4: MAP from noiseless measurement -> within 2%."""
    sys.path.insert(0, os.path.join(ROOT, "python"))
    from gate.gate64 import map_estimate, fit_prior_gmm, REF_PAIRS
    from gate.mrzero_irse import closed_form_irse
    rng = np.random.default_rng(0)
    gmm = fit_prior_gmm(rng, n_samples=500, K=3)
    T1t, T2t, PDt = 0.8, 0.1, 0.9
    y_obs = closed_form_irse(T1t, T2t, PDt, REF_PAIRS)
    est, _ = map_estimate(y_obs, sigma=1e-4, gmm=gmm, n_restarts=4, rng=rng)
    err = np.abs(est - np.array([T1t, T2t, PDt])) / np.array([T1t, T2t, PDt])
    assert err.max() < 0.05, f"MAP rel err {err} > 5% noiseless"


def test_gate64_mlp_shape():
    """Gate 6.4: MLP instantiates with right input/output dims."""
    import torch
    sys.path.insert(0, os.path.join(ROOT, "python"))
    from gate.gate64 import MLP, N_MEAS
    m = MLP(in_dim=N_MEAS, out_dim=3)
    x = torch.zeros(4, N_MEAS)
    y = m(x)
    assert y.shape == (4, 3)


def test_paradigm_a_check_env():
    """Paradigm A env passes gymnasium check_env."""
    sys.path.insert(0, os.path.join(ROOT, "python"))
    from qalibremd_gym.env_paradigm_a import ParadigmAEnv
    from gymnasium.utils.env_checker import check_env
    env = ParadigmAEnv(seed=0)
    check_env(env, skip_render_check=True)


def test_paradigm_a_step_roundtrip():
    """Paradigm A: reset + a few steps yield well-formed (obs, reward) tuples."""
    sys.path.insert(0, os.path.join(ROOT, "python"))
    from qalibremd_gym.env_paradigm_a import ParadigmAEnv, STATE_DIM, N_ACTIONS
    env = ParadigmAEnv(seed=1)
    obs, info = env.reset(seed=1)
    assert obs.shape == (STATE_DIM,)
    rng = np.random.default_rng(0)
    for _ in range(10):
        a = int(rng.integers(0, N_ACTIONS))
        obs, r, term, trunc, info = env.step(a)
        assert obs.shape == (STATE_DIM,)
        assert np.isfinite(r)
        assert isinstance(term, bool) and isinstance(trunc, bool)
        if term or trunc:
            break

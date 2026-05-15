"""MRzero IR-SE forward model for joint (T1, T2, PD) estimation.

Builds a multi-(TI, TE) IR-prep + SE-readout sequence:
  for each (TI, TE):
    180 inversion -> wait TI -> 90 excite -> TE/2 -> 180 refocus -> TE/2 -> ADC -> TR pad

Used by Gate 6.3 (joint landscape with MRzero forward), Gate 6.4 (estimator
training/eval), and Paradigm A reward computation.

Supports both single voxel and multi-voxel phantoms.
"""
from __future__ import annotations
import torch
import numpy as np
import MRzeroCore as mr0
from MRzeroCore.sequence import Pulse, Repetition, Sequence, PulseUsage


def _pulse(angle: float, phase: float = 0.0,
           usage: PulseUsage = PulseUsage.UNDEF) -> Pulse:
    return Pulse(
        usage=usage,
        angle=torch.tensor(float(angle)),
        phase=torch.tensor(float(phase)),
        shim_array=torch.tensor([[1.0, 0.0]]),
        selective=False,
    )


def build_irse_sequence(ti_te_pairs: list[tuple[float, float]],
                        TR: float = 8.0) -> Sequence:
    """Build a multi-(TI,TE) IR-SE sequence with hard pulses.

    Each (TI, TE) is encoded as 4 repetitions sharing one TR cycle:
      1) 180 inversion, wait TI
      2) 90 excitation, wait TE/2 (no ADC)
      3) 180 refocus, wait TE/2 (no ADC)
      4) dummy 0-angle pulse, ADC sample then long spoil pad to recover

    Implementation note: each Repetition begins with a pulse. The ADC sample is
    placed at the start of the 4th repetition (immediately after the SE peak).
    """
    seq = Sequence(normalized_grads=False)
    spoil_grad = torch.zeros(2, 3)
    spoil_grad[1, 0] = 1e4
    for TI, TE in ti_te_pairs:
        # 1) inversion + TI delay
        rep_inv = Repetition(
            pulse=_pulse(np.pi, usage=PulseUsage.UNDEF),
            event_time=torch.tensor([float(TI)]),
            gradm=torch.zeros(1, 3),
            adc_phase=torch.zeros(1),
            adc_usage=torch.zeros(1, dtype=torch.int32),
        )
        seq.append(rep_inv)

        # 2) 90 excite, wait TE/2
        rep_exc = Repetition(
            pulse=_pulse(np.pi / 2, usage=PulseUsage.EXCIT),
            event_time=torch.tensor([float(TE) / 2.0]),
            gradm=torch.zeros(1, 3),
            adc_phase=torch.zeros(1),
            adc_usage=torch.zeros(1, dtype=torch.int32),
        )
        seq.append(rep_exc)

        # 3) 180 refocus, wait TE/2
        rep_ref = Repetition(
            pulse=_pulse(np.pi, phase=np.pi / 2, usage=PulseUsage.REFOC),
            event_time=torch.tensor([float(TE) / 2.0]),
            gradm=torch.zeros(1, 3),
            adc_phase=torch.zeros(1),
            adc_usage=torch.zeros(1, dtype=torch.int32),
        )
        seq.append(rep_ref)

        # 4) dummy small-angle pulse + ADC + spoil/TR pad
        ev_time = torch.tensor([1e-5, float(TR)])
        adc_usage = torch.zeros(2, dtype=torch.int32)
        adc_usage[0] = 1
        rep_adc = Repetition(
            pulse=_pulse(1e-4, usage=PulseUsage.UNDEF),  # negligible
            event_time=ev_time,
            gradm=spoil_grad.clone(),
            adc_phase=torch.zeros(2),
            adc_usage=adc_usage,
        )
        seq.append(rep_adc)
    return seq


def simulate_irse_voxel(T1: float, T2: float, PD: float,
                        ti_te_pairs: list[tuple[float, float]],
                        TR: float = 8.0,
                        max_state_count: int = 80,
                        min_state_mag: float = 1e-3) -> torch.Tensor:
    """Simulate IR-SE on a single-voxel phantom. Returns complex signal vector of
    length len(ti_te_pairs)."""
    seq = build_irse_sequence(ti_te_pairs, TR=TR)
    phantom = mr0.CustomVoxelPhantom(
        pos=[[0.0, 0.0, 0.0]],
        T1=float(T1), T2=float(T2), PD=float(PD),
        T2dash=1e6, D=0.0, B0=0.0, B1=1.0,
        voxel_size=0.1, voxel_shape="box",
    )
    data = phantom.build()
    graph = mr0.compute_graph(seq, data,
                              max_state_count=max_state_count,
                              min_state_mag=min_state_mag)
    signal = mr0.execute_graph(graph, seq, data,
                               min_emitted_signal=1e-6,
                               min_latent_signal=1e-6,
                               print_progress=False)
    return signal.ravel()


def simulate_irse_phantom(T1: torch.Tensor, T2: torch.Tensor, PD: torch.Tensor,
                          positions: torch.Tensor,
                          ti_te_pairs: list[tuple[float, float]],
                          TR: float = 8.0,
                          max_state_count: int = 80,
                          min_state_mag: float = 1e-3) -> torch.Tensor:
    """Simulate IR-SE on a multi-voxel phantom (each voxel has own T1/T2/PD).

    Returns complex signal of shape (n_meas,) since gradients=0 -> all voxels
    coherent; this gives the SUMMED signal, which is what an unspatially-encoded
    ADC sees. For per-voxel signals, run separate single-voxel sims (slow but
    parallelizable).
    """
    seq = build_irse_sequence(ti_te_pairs, TR=TR)
    phantom = mr0.CustomVoxelPhantom(
        pos=positions,
        T1=T1, T2=T2, PD=PD,
        T2dash=1e6, D=0.0, B0=0.0, B1=1.0,
        voxel_size=0.1, voxel_shape="box",
    )
    data = phantom.build()
    graph = mr0.compute_graph(seq, data,
                              max_state_count=max_state_count,
                              min_state_mag=min_state_mag)
    signal = mr0.execute_graph(graph, seq, data,
                               min_emitted_signal=1e-6,
                               min_latent_signal=1e-6,
                               print_progress=False)
    return signal.ravel()


def add_rician_noise(signal: np.ndarray, sigma: float,
                     rng: np.random.Generator) -> np.ndarray:
    """Add Rician noise: y = |x + n_r + i n_i| with n_r, n_i ~ N(0, sigma^2).

    `signal` may be real (treated as |x|, no original phase) or complex.
    Returns real magnitude array.
    """
    sig = np.asarray(signal)
    if np.iscomplexobj(sig):
        re = sig.real
        im = sig.imag
    else:
        re = sig
        im = np.zeros_like(sig)
    re_n = re + rng.normal(0.0, sigma, size=re.shape)
    im_n = im + rng.normal(0.0, sigma, size=im.shape)
    return np.hypot(re_n, im_n)


# Closed-form IR-SE for fast grid evaluations
def closed_form_irse(T1, T2, PD, ti_te_pairs):
    """Closed-form magnitude signal: |1 - 2 exp(-TI/T1)| * PD * exp(-TE/T2)."""
    T1 = np.asarray(T1, dtype=float)
    T2 = np.asarray(T2, dtype=float)
    PD = np.asarray(PD, dtype=float)
    out = []
    for TI, TE in ti_te_pairs:
        s = np.abs(1.0 - 2.0 * np.exp(-TI / T1)) * PD * np.exp(-TE / T2)
        out.append(s)
    return np.stack(out, axis=-1)

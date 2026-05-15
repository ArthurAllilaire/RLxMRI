"""MRzero inversion-recovery sequence builder.

Builds an IR-prep sequence: 180 inversion -> wait TI -> 90 excitation -> ADC.
Uses *non-selective* (instantaneous) hard pulses to match the simple Bloch
analytical model `|1 - 2*exp(-TI/T1)|` at the first ADC sample.

Notes on MRzero conventions
---------------------------
- A `Repetition` starts with a pulse and contains a stream of events. Each
  event has a duration, a gradient moment, and ADC config.
- We do NOT use gradient moments (we want spectroscopy-like FID, not k-space).
- We disable normalized gradients so events run in SI seconds.
"""
from __future__ import annotations
import torch
import MRzeroCore as mr0
from MRzeroCore.sequence import Pulse, Repetition, Sequence, PulseUsage


def build_ir_sequence(
    TIs: list[float],
    TR: float = 8.0,
    n_adc: int = 1,
    dur_adc: float = 1e-4,
) -> Sequence:
    """Build a multi-TI inversion-recovery sequence.

    Each TI is one repetition: 180 inv -> TI delay -> 90 excite -> ADC -> TR pad.
    Pulses are instantaneous (selective=False, zero duration).

    Returns a `Sequence` with `len(TIs)` repetitions, each emitting one ADC.
    """
    seq = Sequence(normalized_grads=False)
    for TI in TIs:
        # The Repetition's `pulse` is the LEADING pulse. So we need TWO repetitions
        # per TI: one starting with 180, one starting with 90.  We pack as:
        #   Rep A (180): event 0 has duration TI (the inversion delay), zero ADC.
        #   Rep B (90):  event 0 dur_adc with ADC, then TR-recovery dur.
        # 180 inversion repetition
        inv = Pulse(
            usage=PulseUsage.UNDEF,
            angle=torch.tensor(torch.pi),
            phase=torch.tensor(0.0),
            shim_array=torch.tensor([[1.0, 0.0]]),
            selective=False,
        )
        rep_inv = Repetition(
            pulse=inv,
            event_time=torch.tensor([float(TI)]),
            gradm=torch.zeros(1, 3),
            adc_phase=torch.zeros(1),
            adc_usage=torch.zeros(1, dtype=torch.int32),
        )
        seq.append(rep_inv)

        # 90 excitation repetition with ADC
        exc = Pulse(
            usage=PulseUsage.EXCIT,
            angle=torch.tensor(torch.pi / 2),
            phase=torch.tensor(0.0),
            shim_array=torch.tensor([[1.0, 0.0]]),
            selective=False,
        )
        # n_adc samples followed by a TR pad event
        n_events = n_adc + 1
        ev_time = torch.full((n_events,), dur_adc / max(n_adc, 1))
        ev_time[-1] = TR  # spoil-pad event (large) -> drives steady state to fully recover
        gradm = torch.zeros(n_events, 3)
        # Big gradient moment on the final event to spoil residual transverse magnetisation
        gradm[-1, 0] = 1e4
        adc_phase = torch.zeros(n_events)
        adc_usage = torch.zeros(n_events, dtype=torch.int32)
        adc_usage[:n_adc] = 1
        rep_exc = Repetition(
            pulse=exc,
            event_time=ev_time,
            gradm=gradm,
            adc_phase=adc_phase,
            adc_usage=adc_usage,
        )
        seq.append(rep_exc)
    return seq


def simulate_ir(
    T1: float, T2: float, PD: float,
    TIs: list[float],
    TR: float = 8.0,
    n_adc: int = 1,
    dur_adc: float = 1e-4,
) -> torch.Tensor:
    """Simulate IR sequence on a single-voxel phantom.

    Returns a tensor of shape (len(TIs)*n_adc,) of complex signal samples.
    """
    seq = build_ir_sequence(TIs, TR=TR, n_adc=n_adc, dur_adc=dur_adc)
    phantom = mr0.CustomVoxelPhantom(
        pos=[[0.0, 0.0, 0.0]],
        T1=float(T1), T2=float(T2), PD=float(PD),
        T2dash=1e6,   # disable T2* dephasing
        D=0.0, B0=0.0, B1=1.0,
        voxel_size=0.1, voxel_shape="box",
    )
    data = phantom.build()
    graph = mr0.compute_graph(seq, data, max_state_count=200, min_state_mag=1e-5)
    signal = mr0.execute_graph(graph, seq, data,
                               min_emitted_signal=1e-6,
                               min_latent_signal=1e-6,
                               print_progress=False)
    return signal


def closed_form_ir(T1: float, TIs, PD: float = 1.0):
    import numpy as np
    TIs = np.asarray(TIs, dtype=float)
    return PD * np.abs(1.0 - 2.0 * np.exp(-TIs / T1))

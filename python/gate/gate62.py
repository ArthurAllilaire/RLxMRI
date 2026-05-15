"""Gate 6.2 — PDG simulation cost on a 200-event FISP-like sequence.

CPU-adjusted pass condition: forward+backward < 2 s per simulation (10x the
200ms GPU target), peak host RAM < 4 GB. Reports forward-only and
forward+backward timings, N=10 trials each, median + std.
"""
from __future__ import annotations
import os, sys, time, json, gc, tracemalloc
import numpy as np
import torch
import MRzeroCore as mr0
from MRzeroCore.sequence import Pulse, Repetition, Sequence, PulseUsage

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUTDIR = os.path.join(ROOT, "runs", "e2e_gate", "gate2")
os.makedirs(OUTDIR, exist_ok=True)

N_EVENTS = 200
N_REPS = 10


def build_fisp(flip_angles: torch.Tensor, TR: float = 12e-3,
                TE: float = 5e-3, n_adc: int = 1) -> Sequence:
    """FISP-like train: N reps of (RF alpha_k -> readout -> spoil), variable alpha.

    `flip_angles`: tensor of shape (N_EVENTS,), radians. Differentiable.
    """
    seq = Sequence(normalized_grads=False)
    N = flip_angles.shape[0]
    for k in range(N):
        pulse = Pulse(
            usage=PulseUsage.EXCIT,
            angle=flip_angles[k],
            phase=torch.tensor(0.0),
            shim_array=torch.tensor([[1.0, 0.0]]),
            selective=False,
        )
        # 3 events per rep: prephase, ADC readout, spoil/TR-pad
        n_events = 3
        ev_time = torch.tensor([TE - 1e-4, 1e-4, TR - TE])
        gradm = torch.zeros(n_events, 3)
        # FLASH-style RF-spoiled readout (SI units, normalized_grads=False).
        # 1e4 rad/m moment dephases > 2pi across a 0.1 m voxel -> clean spoil.
        gradm[2, 0] = 1e4
        adc_phase = torch.zeros(n_events)
        adc_usage = torch.zeros(n_events, dtype=torch.int32)
        adc_usage[1] = 1  # one ADC during readout event
        rep = Repetition(
            pulse=pulse,
            event_time=ev_time,
            gradm=gradm,
            adc_phase=adc_phase,
            adc_usage=adc_usage,
        )
        seq.append(rep)
    return seq


def build_phantom(nx: int = 16, ny: int = 16) -> mr0.SimData:
    """Synthetic small-grid phantom with three concentric circles of varying tissue."""
    xs = torch.linspace(-0.08, 0.08, nx)
    ys = torch.linspace(-0.08, 0.08, ny)
    X, Y = torch.meshgrid(xs, ys, indexing="ij")
    R = torch.sqrt(X**2 + Y**2)
    T1 = torch.where(R < 0.025, torch.tensor(0.3),
          torch.where(R < 0.05, torch.tensor(1.0), torch.tensor(1.8)))
    T2 = torch.where(R < 0.025, torch.tensor(0.04),
          torch.where(R < 0.05, torch.tensor(0.1), torch.tensor(0.4)))
    PD_full = torch.where(R < 0.07, torch.tensor(1.0), torch.tensor(0.0))
    mask = PD_full.flatten() > 0
    positions = torch.stack([X.flatten(), Y.flatten(), torch.zeros(nx*ny)], dim=-1)[mask]
    T1f = T1.flatten()[mask]; T2f = T2.flatten()[mask]; PDf = PD_full.flatten()[mask]
    phantom = mr0.CustomVoxelPhantom(
        pos=positions,
        T1=T1f, T2=T2f, PD=PDf,
        T2dash=torch.full_like(T1f, 30e-3),
        D=0.0, B0=0.0, B1=1.0,
        voxel_size=0.01, voxel_shape="box",
    )
    return phantom.build()


def main():
    n_events_actual = N_EVENTS
    nx = ny = 16
    print(f"Gate 6.2: {n_events_actual}-event FISP-like seq, {nx}x{ny} voxel grid (no GPU)")
    print(f"torch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}")

    data = build_phantom(nx, ny)
    print(f"phantom built: {data.PD.numel()} voxels (after non-PD mask)")

    # warm-up
    fa_warm = torch.full((10,), torch.deg2rad(torch.tensor(10.0)).item())
    fa_warm.requires_grad_(True)
    seq_warm = build_fisp(fa_warm)
    g = mr0.compute_graph(seq_warm, data, max_state_count=80, min_state_mag=1e-3)
    s = mr0.execute_graph(g, seq_warm, data, min_emitted_signal=1e-3,
                          min_latent_signal=1e-3, print_progress=False)
    s.abs().sum().backward()

    # Forward only
    fwd_times = []
    for trial in range(N_REPS):
        fa = torch.deg2rad(10 + 30*torch.rand(n_events_actual))
        fa.requires_grad_(False)
        seq = build_fisp(fa)
        gc.collect()
        t0 = time.time()
        g = mr0.compute_graph(seq, data, max_state_count=80, min_state_mag=1e-3)
        sig = mr0.execute_graph(g, seq, data, min_emitted_signal=1e-3,
                                min_latent_signal=1e-3, print_progress=False)
        torch.zeros(1)  # force flush
        dt = time.time() - t0
        fwd_times.append(dt)
        print(f"  forward {trial}: {dt:.3f}s  signal len={sig.numel()}")

    # Forward + backward
    bwd_times = []
    peak_mems = []
    for trial in range(N_REPS):
        fa = torch.deg2rad(10 + 30*torch.rand(n_events_actual))
        fa.requires_grad_(True)
        seq = build_fisp(fa)
        gc.collect()
        tracemalloc.start()
        t0 = time.time()
        g = mr0.compute_graph(seq, data, max_state_count=80, min_state_mag=1e-3)
        sig = mr0.execute_graph(g, seq, data, min_emitted_signal=1e-3,
                                min_latent_signal=1e-3, print_progress=False)
        loss = sig.abs().sum()
        loss.backward()
        dt = time.time() - t0
        cur, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        bwd_times.append(dt)
        peak_mems.append(peak / 1024**2)
        print(f"  fwd+bwd {trial}: {dt:.3f}s  peak_alloc={peak/1024**2:.1f} MB  |grad|={fa.grad.abs().sum().item():.3e}")

    fwd_arr = np.array(fwd_times); bwd_arr = np.array(bwd_times)
    summary = dict(
        n_events=n_events_actual, n_voxels=int(data.PD.numel()),
        fwd_median=float(np.median(fwd_arr)), fwd_std=float(fwd_arr.std()),
        fwd_min=float(fwd_arr.min()), fwd_max=float(fwd_arr.max()),
        bwd_median=float(np.median(bwd_arr)), bwd_std=float(bwd_arr.std()),
        bwd_min=float(bwd_arr.min()), bwd_max=float(bwd_arr.max()),
        peak_mem_MB_median=float(np.median(peak_mems)),
        peak_mem_MB_max=float(np.max(peak_mems)),
    )
    print("\n=== Gate 6.2 summary ===")
    for k, v in summary.items():
        print(f"  {k} = {v}")

    cpu_pass = (summary["bwd_median"] < 2.0) and (summary["peak_mem_MB_max"] < 4096)
    summary["pass_cpu_adjusted"] = bool(cpu_pass)
    # GPU extrapolation factor — PyTorch CPU vs typical mid-range GPU on this workload ~5-20x
    summary["estimated_gpu_speedup_lo"] = 5
    summary["estimated_gpu_speedup_hi"] = 20
    summary["pass_gpu_optimistic"] = bool((summary["bwd_median"] / 20.0) < 0.2)
    summary["pass_gpu_pessimistic"] = bool((summary["bwd_median"] / 5.0) < 0.2)

    print(f"\nGate 6.2 CPU-adjusted (<2s fwd+bwd, <4GB): {'PASS' if cpu_pass else 'FAIL'}")
    print(f"  GPU-optimistic ({summary['bwd_median']/20:.3f}s < 0.2s): {'PASS' if summary['pass_gpu_optimistic'] else 'FAIL'}")
    print(f"  GPU-pessimistic ({summary['bwd_median']/5:.3f}s < 0.2s): {'PASS' if summary['pass_gpu_pessimistic'] else 'FAIL'}")

    with open(os.path.join(OUTDIR, "gate62_results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved {OUTDIR}/gate62_results.json")
    return cpu_pass


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)

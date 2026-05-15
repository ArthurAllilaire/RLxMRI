# MRzero / MRzeroCore — research notes

Notes accumulated during the M3 verification-gate experiment (2026-05-11). All
code lives in `python/gate/`; outputs in `runs/e2e_gate/`.

## Origin

- **Paper**: Loktyushin, Herz, Dang, Krotov, Hennel, Bender, Glang, Doerfler, Endt, Heigi, Scheffler, Zaiss. *MRzero — Automated discovery of MRI sequences using supervised learning*. Magn. Reson. Med. 86:709–724 (2021).
- **Codebase**: `MRzero-Project/MRzero-Core` on GitHub (installed via `pip install MRzeroCore` — package version 0.4.7 used here).
- **Underlying physics engine**: a **Phase Distribution Graph (PDG)** Bloch-Torrey solver implemented as a Rust pre-pass (`pyo3` bindings) that constructs the coherence-pathway tree, followed by a PyTorch-tensor forward pass through that tree. The whole forward pass is autograd-traceable, so gradients flow into sequence parameters (flip angles, gradient moments, event times), into tissue parameters (T1, T2, PD, B0, B1), and into spatial positions.

## PDG vs EPG — the correction we owe the interim report

The Imperial interim report (chapter 2, Section 2.5) describes MRzero as an **EPG** (Extended Phase Graph) simulator. This is incorrect and must be fixed in the final report.

- **EPG** tracks magnetisation in a 1D frequency-bin lattice (k_x states only), assuming on-resonance hard pulses and ideal spoilers. Configuration states $F_n^+, F_n^-, Z_n$ are dephasing modes along a single spatial direction.
- **PDG** (Endt et al. 2022) generalises EPG to a *tree of coherence pathways* parameterised by (k_x, k_y, k_z, dephasing-time τ, T2'-mode, ...). Each pulse splits a node into up to three children (the F+, F−, Z paths after an RF rotation), and the pre-pass prunes nodes whose amplitude is below `min_state_mag`. The tree captures off-resonance, diffusion, T2*, and arbitrary 3D dephasing — none of which EPG handles in a single bookkeeping pass.
- **Practical consequence**: MRzero produces an exact (within pruning) Bloch solution for arbitrary sequences and arbitrary phantoms; EPG would fail on the IR-SE 2D sequence in Ch4 because of the multi-axis gradient encoding.

## API surface used by the verification gate

```python
import MRzeroCore as mr0
from MRzeroCore.sequence import Pulse, Repetition, Sequence, PulseUsage

# 1. Build a Pulse (instantaneous; selective=False)
pulse = Pulse(usage=PulseUsage.EXCIT,
              angle=torch.tensor(0.26), phase=torch.tensor(0.0),
              shim_array=torch.tensor([[1.0, 0.0]]),  # single-Tx
              selective=False)

# 2. Build a Repetition (= one pulse + a stream of inter-pulse events)
rep = Repetition(pulse=pulse,
                 event_time=torch.tensor([dt0, dt1, ...]),   # seconds
                 gradm=torch.zeros(n_events, 3),             # rad/m (SI if normalized_grads=False)
                 adc_phase=torch.zeros(n_events),
                 adc_usage=torch.zeros(n_events, dtype=torch.int32))  # >0 = sampled

# 3. Append into a Sequence
seq = Sequence(normalized_grads=False)   # important: SI units
seq.append(rep)

# 4. Build a phantom (CustomVoxelPhantom for arbitrary positions/values)
phantom = mr0.CustomVoxelPhantom(
    pos=[[0,0,0]],                       # one voxel at origin
    T1=1.0, T2=0.1, PD=1.0,
    T2dash=30e-3, D=0.0, B0=0.0, B1=1.0,
    voxel_size=0.1, voxel_shape="box",
)
data = phantom.build()                   # -> SimData (PyTorch tensors)

# 5. Pre-pass + forward
graph = mr0.compute_graph(seq, data,
                          max_state_count=200, min_state_mag=1e-4)
signal = mr0.execute_graph(graph, seq, data,
                           min_emitted_signal=1e-3,
                           min_latent_signal=1e-3,
                           print_progress=False)
# signal: torch.complex64 tensor (n_adc_samples, n_coils)

# 6. Backprop into anything that's a leaf tensor in seq or data
signal.abs().sum().backward()
```

Important gotchas (each found by trial-and-error on 2026-05-11):

1. **`normalized_grads=True` triggers Rust-side NaN panics** on small single-voxel phantoms when the gradient moment exceeds the inverse voxel size. The Rust pre-pass divides by phantom Nyquist tensors and underflows. Workaround: pass `normalized_grads=False` and provide gradient moments in SI rad/m (e.g. `1e4` rad/m for a clean spoil over a 0.1 m voxel).
2. **`requires_grad=True` on `Pulse.angle` is fine for `execute_graph` but the Rust pre-pass calls `.item()` on the angle**, which produces a UserWarning but works. Do *not* pass non-leaf grad-bearing tensors into the pre-pass.
3. **`compute_graph` is invariant under per-pulse angle perturbations** in the small-FA limit — the tree topology is constructed once, then `execute_graph` re-walks it with the actual tensor values. So you can build the graph once with detached angles and reuse it across optimisation steps with autograd-bearing angles.
4. **`PD = 0` voxels produce NaN** in the pre-pass (a "sort + dists by mag" divide-by-zero). Filter voxels by `PD > 0` before constructing `CustomVoxelPhantom`.
5. **Signal ordering**: `execute_graph` returns shape `(total_adc_samples, n_coils)`. The samples are flattened in sequence order; you must reshape per-repetition yourself.

## Tradeoffs vs KomaMRI (Bloch ODE)

| | KomaMRI | MRzero (PDG) |
|---|---|---|
| Forward physics | per-spin Bloch ODE | coherence-pathway tree with PyTorch ops |
| Differentiability | No (finite differences only) | Yes (PyTorch autograd) |
| Fidelity | Exact (modulo ODE-solver step) | Exact within pruning thresholds |
| Speed on 200-event seq, 140 vox (CPU) | ~50 s (single TI, ~0.8 s/TI extrapolated) | 1.6 s fwd / 3.6 s fwd+bwd |
| Pulse shapes | Arbitrary (uses pulse duration explicitly) | Instantaneous + optional selective slice profile |
| Off-resonance | Direct (B0 field) | Per-voxel B0 + T2' off-resonance distribution |
| Slice-selective pulses | Native via gradients | Approximate via `selective=True` flag + slice profile |
| Ecosystem | Julia + Pulseq + KomaMRIPlots | Python + PyPulseq + scikit-image |
| **Best use** | Sim-to-real validation, high-fidelity ground truth | Differentiable RL/meta-optimisation, fast iteration |

The headline tradeoff for our project: **MRzero buys differentiability and ~30× speed at the cost of instantaneous-pulse approximation and a small (~1%) per-TI bias vs KomaMRI's finite-pulse model** (Gate 6.1 result: median 0.94% rel-diff, max 1.55%).

## Why Paradigm B is feasible if PDG cost stays bounded

Gate 6.2 on CPU: forward+backward 3.6 s per 200-event simulation on 140 voxels. PyTorch CPU is typically 5–20× slower than a single-GPU run of the same graph (most of the cost is gather/scatter into the path-tree tensor; this vectorises well on CUDA). Extrapolating:

- GPU forward+backward: 0.18 s (optimistic) – 0.72 s (pessimistic).
- The plan's 200 ms target sits at the optimistic end of that range; a borderline pass under the assumption of a mid-range GPU.

For Paradigm B (gradient meta-opt of a sequence schedule), this is workable: ~1000 outer steps * 0.72 s = 12 minutes per schedule optimisation on a single GPU. For Paradigm A (RL with thousands of trajectories), it is borderline; mitigation is shorter skills (Paradigm C, ≤50 events) which would cut wall time roughly proportionally.

## Open MRzero questions deferred past the gate

1. How does MRzero handle finite-duration RF pulses that overlap with gradient events? The `selective=True` flag is documented but we haven't verified the slice-profile fidelity.
2. The `Graph.max_state_count` pruning threshold (default 200) — what's the empirical pruning rate on a 1000-event MRF schedule? May need to be raised for long sequences.
3. RF spoiling: MRzero supports it via per-repetition `adc_phase` rotation but the bookkeeping convention differs from EPG papers; care needed when comparing to MRF literature.

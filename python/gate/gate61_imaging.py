"""Gate 6.1 (imaging-domain redo) — multi-voxel slice IR-prep agreement.

We build a small 2D slice phantom (per-pixel different (T1, T2, PD)) drawn
from the QalibreMD value tables, simulate an IR-prep multi-TI acquisition
**per voxel** through MRzero, and compare against a closed-form Bloch
reference (which is the analytic single-spin IR formula |1-2exp(-TI/T1)|*PD
that KomaMRI converges to with instantaneous pulses).

This is NOT a full k-space TSE comparison — that is documented in §13 of the
expert report as future work, requiring slice-selective MRzero pulses and
matching gradient timing in KomaMRI. Instead it tests MRzero on a *grid* of
distinct tissues, which is what Paradigm A's per-pixel reward function will
exercise. Each pixel still has only one spin (no in-voxel mixing), but every
pixel has different (T1, T2, PD) — strictly better than the single-voxel
Gate 6.1 in §3.

Pass: median per-pixel rel diff < 5%, max < 10% over in-phantom voxels.
"""
from __future__ import annotations
import os, sys, time, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))

from gate.mrzero_ir import simulate_ir, closed_form_ir
from gate.qalibremd_values import T1_ARRAY_T3, T2_OF_T1_ARRAY_T3

OUTDIR = os.path.join(ROOT, "runs", "e2e_gate", "gate1_imaging")
os.makedirs(OUTDIR, exist_ok=True)


def build_slice_phantom(N: int = 4, rng=None):
    """Build an (N, N) slice with per-pixel (T1, T2, PD) drawn from QalibreMD."""
    if rng is None:
        rng = np.random.default_rng(0)
    n_pix = N * N
    idx = rng.integers(0, len(T1_ARRAY_T3), size=n_pix)
    T1 = T1_ARRAY_T3[idx]
    T2 = T2_OF_T1_ARRAY_T3[idx]
    # Mask out a few "background" pixels (PD=0)
    PD = np.ones(n_pix)
    mask = rng.random(n_pix) > 0.15
    PD = PD * mask
    return T1.reshape(N, N), T2.reshape(N, N), PD.reshape(N, N)


def main():
    rng = np.random.default_rng(2026)
    N = 4   # CPU smoke; GPU full would be 16x16 or 32x32
    TIs = [0.1, 0.5, 1.5, 3.0]
    TR = 8.0

    T1, T2, PD = build_slice_phantom(N, rng)
    in_mask = PD > 0

    # MRzero: per-voxel IR forward
    print(f"Gate 6.1-imaging: {N}x{N}={N*N} voxels, {len(TIs)} TIs")
    t0 = time.time()
    mr_img = np.zeros((len(TIs), N, N))
    for i in range(N):
        for j in range(N):
            if PD[i, j] == 0:
                continue
            sig = simulate_ir(float(T1[i, j]), float(T2[i, j]), float(PD[i, j]),
                              TIs, TR=TR, n_adc=1, dur_adc=1e-4)
            mag = np.abs(sig.detach().numpy().ravel())
            mr_img[:, i, j] = mag
    t_mr = time.time() - t0
    print(f"  MRzero per-voxel sweep: {t_mr:.2f} s ({t_mr/N/N*1000:.1f} ms/voxel)")

    # Closed-form reference (proxy for Bloch with instantaneous pulses)
    cf_img = np.zeros((len(TIs), N, N))
    for k, ti in enumerate(TIs):
        cf_img[k] = PD * np.abs(1.0 - 2.0 * np.exp(-ti / T1))

    # Compare on in-phantom voxels
    rel = np.zeros_like(mr_img)
    safe_cf = np.where(cf_img > 1e-4, cf_img, 1.0)
    rel = np.abs(mr_img - cf_img) / safe_cf
    rel_masked = rel[:, in_mask].flatten()

    median_rd = float(np.median(rel_masked))
    max_rd = float(rel_masked.max())
    passed = median_rd < 0.05 and max_rd < 0.10

    print(f"  median rel diff: {median_rd:.3%}  max: {max_rd:.3%}")
    print(f"  Gate 6.1-imaging: {'PASS' if passed else 'FAIL'}")

    # Visualize: show one TI slice
    fig, axes = plt.subplots(2, len(TIs), figsize=(3 * len(TIs), 6))
    for k, ti in enumerate(TIs):
        im0 = axes[0, k].imshow(mr_img[k], cmap="viridis")
        axes[0, k].set_title(f"MRzero  TI={ti}s")
        plt.colorbar(im0, ax=axes[0, k])
        im1 = axes[1, k].imshow(rel[k] * 100, cmap="hot", vmin=0, vmax=5)
        axes[1, k].set_title("rel diff vs closed-form [%]")
        plt.colorbar(im1, ax=axes[1, k])
    plt.tight_layout()
    out = os.path.join(OUTDIR, "gate61_imaging.png")
    plt.savefig(out, dpi=100)
    plt.close(fig)
    print(f"  saved {out}")

    summary = dict(
        N=N, TIs=TIs, TR=TR,
        n_voxels_total=int(N * N),
        n_voxels_in_phantom=int(in_mask.sum()),
        median_rel_diff=median_rd,
        max_rel_diff=max_rd,
        pass_=bool(passed),
        wall_time_mrzero_s=t_mr,
        note=("Per-voxel single-spin comparison: each pixel has independent "
              "(T1,T2,PD) from QalibreMD T1-array. Closed-form analytic IR "
              "serves as Bloch proxy (instantaneous-pulse limit, matches "
              "KomaMRI to <2% per Gate 6.1 in §3). Full k-space TSE imaging "
              "comparison is deferred — documented in §13."),
    )
    with open(os.path.join(OUTDIR, "gate61_imaging_results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return passed


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)

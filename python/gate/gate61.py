"""Gate 6.1 — MRzero <-> KomaMRI agreement on an IR-prep sequence.

Picks the QalibreMD long-T1 sphere (T1=1.838 s, T2=0.5244 s at 3T; from T1_ARRAY[:T3][1]
and a representative T2). Simulates 6 log-spaced TIs in both simulators and reports
per-TI relative difference.

Pass: median rel-diff < 5%, max < 10%.
"""
from __future__ import annotations
import json, os, subprocess, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))

from gate.mrzero_ir import simulate_ir, closed_form_ir

OUTDIR = os.path.join(ROOT, "runs", "e2e_gate", "gate1")
os.makedirs(OUTDIR, exist_ok=True)


def run_koma_ir(T1: float, T2: float, TIs: list[float]) -> np.ndarray:
    out = os.path.join(OUTDIR, "koma_ir.json")
    cmd = [
        "julia", f"--project={ROOT}", os.path.join(ROOT, "python", "gate", "run_koma_ir.jl"),
        str(T1), str(T2), out, ",".join(str(t) for t in TIs),
    ]
    subprocess.check_call(cmd, cwd=ROOT)
    with open(out) as f:
        d = json.load(f)
    return np.asarray(d["mag"])


def main():
    # Long-T1 sphere from QalibreMD T1-array at 3T (sphere #2, mid-range)
    # Use long-T1 + long-T2 to avoid T2 decay during finite pulses
    T1, T2, PD = 1.398, 1.035, 1.0
    TIs = list(np.logspace(np.log10(0.05), np.log10(3.0), 6))

    print(f"Gate 6.1: T1={T1}s, T2={T2}s, PD={PD}, TIs={TIs}")

    print("Running MRzero...")
    t0 = time.time()
    sig_mr = simulate_ir(T1, T2, PD, TIs, TR=10.0, n_adc=1, dur_adc=1e-4)
    mr_mag = np.abs(sig_mr.detach().numpy().ravel())
    t_mr = time.time() - t0
    print(f"  MRzero mag: {mr_mag} ({t_mr:.2f}s)")

    print("Running KomaMRI...")
    t0 = time.time()
    koma_mag = run_koma_ir(T1, T2, TIs)
    t_koma = time.time() - t0
    print(f"  KomaMRI mag: {koma_mag} ({t_koma:.2f}s)")

    analytic = closed_form_ir(T1, TIs, PD=PD)
    print(f"  Analytic   : {analytic}")

    rel_mr_koma = np.abs(mr_mag - koma_mag) / np.maximum(koma_mag, 1e-6)
    rel_mr_an = np.abs(mr_mag - analytic) / np.maximum(analytic, 1e-6)
    rel_koma_an = np.abs(koma_mag - analytic) / np.maximum(analytic, 1e-6)

    print(f"\nRel diff MRzero vs KomaMRI:  median={np.median(rel_mr_koma):.3%}, max={rel_mr_koma.max():.3%}")
    print(f"Rel diff MRzero vs analytic: median={np.median(rel_mr_an):.3%}, max={rel_mr_an.max():.3%}")
    print(f"Rel diff KomaMRI vs analytic: median={np.median(rel_koma_an):.3%}, max={rel_koma_an.max():.3%}")

    median_rd = float(np.median(rel_mr_koma))
    max_rd = float(rel_mr_koma.max())
    passed = (median_rd < 0.05) and (max_rd < 0.10)
    print(f"\nGate 6.1: {'PASS' if passed else 'FAIL'} (median {median_rd:.2%} < 5%, max {max_rd:.2%} < 10%)")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax = axes[0]
    ax.semilogx(TIs, mr_mag, "o-", label="MRzero")
    ax.semilogx(TIs, koma_mag, "s--", label="KomaMRI")
    ax.semilogx(TIs, analytic, "k:", label=r"closed-form $|1-2e^{-TI/T_1}|$")
    ax.set_xlabel("TI [s]")
    ax.set_ylabel("magnitude")
    ax.set_title(f"IR signal, T1={T1}s T2={T2}s")
    ax.legend()
    ax.grid(alpha=0.3)
    ax = axes[1]
    ax.semilogx(TIs, 100 * rel_mr_koma, "o-", label="MRzero vs KomaMRI")
    ax.semilogx(TIs, 100 * rel_mr_an, "s--", label="MRzero vs analytic")
    ax.semilogx(TIs, 100 * rel_koma_an, "d:", label="KomaMRI vs analytic")
    ax.axhline(5, c="r", lw=0.7, ls="--", label="5% line")
    ax.set_xlabel("TI [s]")
    ax.set_ylabel("relative diff [%]")
    ax.set_title("Per-TI agreement")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_png = os.path.join(OUTDIR, "gate61_ir_agreement.png")
    plt.savefig(out_png, dpi=120)
    print(f"saved {out_png}")

    with open(os.path.join(OUTDIR, "gate61_results.json"), "w") as f:
        json.dump({
            "T1": T1, "T2": T2, "PD": PD,
            "TIs": list(TIs),
            "mrzero_mag": mr_mag.tolist(),
            "koma_mag": koma_mag.tolist(),
            "analytic_mag": analytic.tolist(),
            "rel_mr_koma": rel_mr_koma.tolist(),
            "rel_mr_analytic": rel_mr_an.tolist(),
            "rel_koma_analytic": rel_koma_an.tolist(),
            "median_rel": median_rd, "max_rel": max_rd,
            "pass": bool(passed),
            "t_mrzero_s": t_mr, "t_koma_s": t_koma,
        }, f, indent=2)
    return passed


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)

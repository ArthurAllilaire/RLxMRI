"""Simplified T1-recovery curve plotter — Koma observed signals only.

For each sphere, shows three things:
  1. Blue solid:  M0_fit · |M_z(TI)| at T1_true
  2. Red dashed:  M0_fit · |M_z(TI)| at T1_fit  (what the fitter believes)
  3. Black dots:  Koma pixel magnitudes from block_signals.csv

M0_fit is the amplitude the fitter found (field 'A' in fit_t1_generalized_ir),
saved per-sphere in t1_fit_vs_true.csv. Both curves use the same M0_fit so any
gap between blue and red is purely a T1 error, not an amplitude mismatch.

Dense curves are drawn at TR = median(TRs); actual per-block TRs are used
only for computing the dot predictions.

Usage:
  python scripts/plot_recovery_curves_koma.py --run bMANUAL_nonoise_npe33fe65
  python scripts/plot_recovery_curves_koma.py --run b240s_nonoise_npe33fe65
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
RUNS_DIR = HERE / "runs" / "t1_fit_vs_true"


def init_julia():
    sys.path.insert(0, str(HERE.parent / "python"))
    from qalibremd_gym import env as _env_mod
    _env_mod._ensure_julia(None)
    return _env_mod._JL_QMD


def mz_abs(jl_qmd, T1: float, TIs, TRs, Npe: int) -> np.ndarray:
    """|transient M_z| just before the excitation pulse for each (TI_k, TR_k)."""
    return np.array([
        abs(float(jl_qmd.transient_mz_at_excite_npe(
            float(T1), float(ti), float(tr),
            float(np.pi), float(np.pi / 2), Npe=int(Npe))))
        for ti, tr in zip(TIs, TRs)
    ])


def load_run(run: str):
    with open(RUNS_DIR / run / "config.json") as f:
        cfg = json.load(f)
    rows = []
    with open(RUNS_DIR / run / "t1_fit_vs_true.csv") as f:
        for r in csv.DictReader(f):
            rows.append({
                "label":   r["label"],
                "T1_true": float(r["T1_true_s"]),
                "T1_fit":  float(r["T1_fit_s"]),
                "M0_fit":  float(r["M0_fit"]),
                "mape":    float(r["mape_pct"]),
            })
    signals: dict[str, list[tuple[float, float, float]]] = {}
    with open(RUNS_DIR / run / "block_signals.csv") as f:
        for r in csv.DictReader(f):
            signals.setdefault(r["label"], []).append(
                (float(r["TI_s"]), float(r["TR_s"]), float(r["mag"])))
    return cfg, rows, signals


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True,
                   help="run subdir under scripts/runs/t1_fit_vs_true/")
    p.add_argument("--out", default=None,
                   help="output PNG (default: <run-dir>/recovery_curves_koma.png)")
    args = p.parse_args()

    cfg, rows, signals = load_run(args.run)
    TIs    = np.asarray(cfg["TIs_s"], dtype=float)
    TRs    = np.asarray(cfg["TRs_s"], dtype=float)
    Npe    = int(cfg["Npe"])
    TR_eff = float(cfg.get("TR_eff_s", np.median(TRs)))

    # Fast path: if the Julia run pre-computed the dense recovery curves, load
    # them and skip booting a second Julia via juliacall entirely. The curves
    # already have M0_fit baked in (y = M0_fit·|Mz|) and their rows align with
    # the CSV / descs order, so y_*_all[i] matches rows[i].
    npz_path = RUNS_DIR / args.run / "recovery_curves.npz"
    precomputed = npz_path.exists()
    if precomputed:
        print(f"Loading pre-computed recovery curves from {npz_path.name}")
        rc = np.load(npz_path)
        TI_dense = rc["TI_dense"]
        y_true_all = rc["y_true"]
        y_fit_all = rc["y_fit"]
        jl_qmd = None
    else:
        print("Initialising Julia bridge...")
        jl_qmd = init_julia()
        print("OK.")
        TI_dense = np.geomspace(1e-3, max(float(np.max(TRs)), 3.0), 400)
    TR_dense = np.full_like(TI_dense, TR_eff)

    T1_true = np.array([r["T1_true"] for r in rows])
    order   = np.argsort(T1_true)

    n     = len(rows)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.2 * ncols, 3.0 * nrows),
                             sharex=True)
    axes = np.atleast_2d(axes).reshape(nrows, ncols)

    for slot, i in enumerate(order):
        ax  = axes[slot // ncols, slot % ncols]
        r   = rows[i]
        T1t = r["T1_true"]
        T1f = r["T1_fit"]
        M0  = r["M0_fit"]

        if precomputed:
            y_true = y_true_all[i]
            y_fit  = y_fit_all[i]
        else:
            y_true = M0 * mz_abs(jl_qmd, T1t, TI_dense, TR_dense, Npe)
            y_fit  = M0 * mz_abs(jl_qmd, T1f, TI_dense, TR_dense, Npe)

        ax.plot(TI_dense, y_true, color="C0", lw=1.4,
                label=f"|Mz| T1_true={T1t:.3f} s")
        ax.plot(TI_dense, y_fit,  color="C3", lw=1.4, linestyle="--",
                label=f"|Mz| T1_fit={T1f:.3f} s")

        if r["label"] in signals:
            triples = signals[r["label"]]
            TIs_obs = np.array([t[0] for t in triples])
            mags    = np.array([t[2] for t in triples])
            ax.scatter(TIs_obs, mags, color="k", s=40, zorder=5,
                       label="Koma pixel mag")

        ax.axhline(0.0, color="gray", lw=0.5)
        ax.set_xscale("log")
        ax.set_title(f"{r['label']}  MAPE={r['mape']:.1f}%", fontsize=9)
        ax.grid(True, which="both", alpha=0.25)
        if slot == 0:
            ax.legend(fontsize=7, loc="lower right")

    for slot in range(n, nrows * ncols):
        axes[slot // ncols, slot % ncols].axis("off")

    for ax in axes[-1, :]:
        ax.set_xlabel("TI [s]")
    for ax in axes[:, 0]:
        ax.set_ylabel("signal (M0·|Mz|)")

    explainer = (
        "How each line is calculated\n"
        "─────────────────────────────────────────────\n"
        "Blue solid  — M0_fit · |Mz(TI)| at T1_true\n"
        "  transient_mz_at_excite_npe(T1_true, TI, TR, π, π/2, Npe)\n"
        "  integrates recovery across all Npe PE lines, then |·|.\n"
        "\n"
        "Red dashed  — M0_fit · |Mz(TI)| at T1_fit\n"
        "  same formula at the fitted T1.  Both curves share M0_fit\n"
        "  so any gap is purely a T1 error, not amplitude mismatch.\n"
        "\n"
        "Black dots  — Koma pixel magnitudes (block_signals.csv)\n"
        "  |IFFT(k-space)[ipe, ife]| / sin(α_exc)\n"
        "  the actual numbers handed to fit_t1_generalized_ir.\n"
        "\n"
        "M0_fit = amplitude free parameter from the fitter (field A).\n"
        f"Dense curves at TR_eff = {TR_eff:.3f} s (median TR)\n"
        f"Npe = {Npe}"
    )
    last_slot = n
    if last_slot < nrows * ncols:
        ax_ex = axes[last_slot // ncols, last_slot % ncols]
        ax_ex.axis("off")
        ax_ex.text(0.05, 0.95, explainer,
                   transform=ax_ex.transAxes,
                   fontsize=7.5, va="top", ha="left",
                   fontfamily="monospace",
                   bbox=dict(boxstyle="round", facecolor="lightyellow",
                             edgecolor="gray", alpha=0.8))
        for slot in range(last_slot + 1, nrows * ncols):
            axes[slot // ncols, slot % ncols].axis("off")

    tis_str = ", ".join(f"{ti:.3f}" for ti in TIs)
    fig.suptitle(f"{args.run}   TIs=[{tis_str}]", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out = Path(args.out) if args.out else RUNS_DIR / args.run / "recovery_curves_koma.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

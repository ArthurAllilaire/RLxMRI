"""scripts/snr_sweep.py — visualise the noise-sensitivity sweep.

Reads `scripts/snr_sweep/snr_sweep.csv` (produced by `snr_sweep.jl`) and
generates the report-ready figure:

    Left panel : Fit MAPE (mean / median / max) vs NEMA MS-1 dual-acq SNR
                 on a log-log axis, with a power-law line of best fit
                 (MAPE = a · SNR^b) drawn through mean-MAPE.
    Right panel: SNR-metric divergence vs target_snr — shows the three
                 SNR numbers (snr_ksp, snr_nema_peak, snr_dual_peak) we
                 compute, demonstrating why the dual-acq number is the
                 one to cite (single-image is Gibbs-biased on coarse grids).

═══════════════════════════════════════════════════════════════════════════
REPORT-USEFUL FACTS — copy these into the §SNR-methodology subsection
═══════════════════════════════════════════════════════════════════════════

1. We measure SNR using three methods. Only one is comparable to clinical
   literature: NEMA MS-1 dual-acquisition (NEMA Standards Publication
   MS-1, 2014).

2. NEMA single-image SNR uses mean(ROI)/std(background), with a 1/0.6551
   Rayleigh correction for magnitude images of zero-mean complex Gaussian
   noise (Henkelman 1985; Gudbjartsson & Patz 1995). On the coarse 32 × 64
   image grid we use, this reads systematically LOW because Gibbs ringing
   from sphere edges contaminates the "background" pixels. Erosion of the
   background mask reduces but does not eliminate the bias.

3. NEMA MS-1 dual-acquisition takes two independent noise realisations
   (A, B) of the same sequence. The difference image (A − B) exactly
   cancels structured signal — including Gibbs ringing — leaving only
   zero-mean Gaussian noise. Noise std is computed from (A − B) pooled
   across the signal ROIs (high-SNR regime where magnitude is locally
   Gaussian) and divided by √2 to undo the noise-doubling. This is the
   gold-standard reproducible SNR metric.

4. Clinical reference: routine 1.5 T / 3 T quantitative T1 mapping
   reports image SNR (NEMA MS-1) in the range 20–50. MR-Linac and
   small-voxel acquisitions push toward 10–20; high-field research scans
   go up to 50–100.

5. The fit-MAPE vs dual-acq SNR plot follows an approximate power law,
   MAPE ≈ a · SNR^b with b ≈ −0.5 to −1 for the well-conditioned
   spheres. The exponent quantifies how forgiving the CR-optimal
   baseline is to noise — useful as a "MAPE budget" for the RL policy
   to beat.

Usage:
    python scripts/snr_sweep.py
    python scripts/snr_sweep.py --indir scripts/snr_sweep --out scripts/snr_sweep/snr_sweep.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent


# ─── Data loading ────────────────────────────────────────────────────────────

def load_summary(csv_path: Path) -> dict[str, np.ndarray]:
    """Load snr_sweep.csv into a dict of numpy arrays keyed by column name."""
    cols: dict[str, list[float]] = {}
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                cols.setdefault(k, []).append(float(v))
    return {k: np.asarray(v, dtype=np.float64) for k, v in cols.items()}


# ─── Power-law fit ───────────────────────────────────────────────────────────

def power_law_fit(x: np.ndarray, y: np.ndarray):
    """Fit MAPE = a · SNR^b in log space. Returns (a, b, r2, fitline_x, fitline_y).

    Filters out non-positive entries (log undefined) and any inf/nan."""
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    lx, ly = np.log(x[mask]), np.log(y[mask])
    if lx.size < 2:
        return None, None, None, None, None
    b, log_a = np.polyfit(lx, ly, 1)
    a = float(np.exp(log_a))
    # R² on log scale
    ly_hat = b * lx + log_a
    ss_res = float(np.sum((ly - ly_hat) ** 2))
    ss_tot = float(np.sum((ly - np.mean(ly)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    x_line = np.geomspace(x[mask].min(), x[mask].max(), 100)
    y_line = a * x_line ** b
    return a, float(b), r2, x_line, y_line


# ─── Plot ────────────────────────────────────────────────────────────────────

def make_figure(data: dict[str, np.ndarray], out_path: Path):
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(15, 6))

    target_snr   = data["target_snr"]
    snr_ksp      = data["snr_ksp"]
    snr_nema     = data["snr_nema_peak"]
    snr_dual     = data["snr_dual_peak"]
    mape_mean    = data["mape_mean_pct"]
    mape_median  = data["mape_median_pct"]
    mape_max     = data["mape_max_pct"]
    sigma        = data["noise_sigma_abs"]

    # ── Left: MAPE vs NEMA MS-1 dual-acq SNR (the report figure) ────────────
    a, b, r2, x_line, y_line = power_law_fit(snr_dual, mape_mean)

    ax_l.loglog(snr_dual, mape_mean,   "o-", color="#1f77b4",
                label="mean MAPE", linewidth=1.6, markersize=7)
    ax_l.loglog(snr_dual, mape_median, "s--", color="#2ca02c",
                label="median MAPE", linewidth=1.2, markersize=6, alpha=0.85)
    ax_l.loglog(snr_dual, mape_max,    "^:",  color="#d62728",
                label="max MAPE (worst sphere)", linewidth=1.2,
                markersize=6, alpha=0.85)
    if x_line is not None:
        ax_l.loglog(x_line, y_line, "-", color="black", linewidth=1.0,
                    alpha=0.55,
                    label=fr"fit: MAPE = {a:.2f} · SNR$^{{{b:+.2f}}}$  "
                          fr"(R²={r2:.3f})")

    # Clinical SNR band (20–50, NEMA MS-1)
    ax_l.axvspan(20.0, 50.0, color="orange", alpha=0.15,
                 label="clinical SNR band\n(NEMA MS-1, 20–50)")
    # 5% / 10% MAPE reference lines
    ax_l.axhline(5.0, color="green",  linestyle="--", linewidth=1, alpha=0.5,
                 label="5 % MAPE target")
    ax_l.axhline(10.0, color="orange", linestyle=":", linewidth=1, alpha=0.5,
                 label="10 % MAPE threshold")

    ax_l.set_xlabel("NEMA MS-1 dual-acq SNR  (peak-sphere)")
    ax_l.set_ylabel("Fit MAPE [%]  (CR-optimal baseline)")
    ax_l.set_title("Fit MAPE vs clinical SNR\n"
                   "(power-law fit on mean MAPE; lower-right = clinical regime)")
    ax_l.grid(True, which="both", alpha=0.3)
    ax_l.legend(fontsize=8, loc="upper right")

    # Annotate each point with target_snr (the internal knob value)
    for x, y, ts in zip(snr_dual, mape_mean, target_snr):
        ax_l.annotate(f"ts={ts:g}", (x, y),
                      textcoords="offset points", xytext=(6, 6),
                      fontsize=7, color="dimgrey")

    # ── Right: SNR-metric divergence vs target_snr ──────────────────────────
    ax_r.loglog(target_snr, snr_ksp,  "o-",  color="#888888",
                label="snr_ksp = ksp_rms/σ  (internal knob, non-standard)",
                linewidth=1.4, markersize=6)
    ax_r.loglog(target_snr, snr_nema, "s-",  color="#ff7f0e",
                label="snr_nema_peak  (single-image, Gibbs-biased)",
                linewidth=1.4, markersize=6)
    ax_r.loglog(target_snr, snr_dual, "^-",  color="#1f77b4",
                label="snr_dual_peak  (NEMA MS-1, REPORT FIGURE)",
                linewidth=1.8, markersize=7)
    # Diagonal: snr_metric = target_snr (for reference)
    diag = np.geomspace(target_snr.min(), target_snr.max(), 50)
    ax_r.loglog(diag, diag, "k--", alpha=0.3, linewidth=0.9,
                label="y = target_snr")
    # Clinical band
    ax_r.axhspan(20.0, 50.0, color="orange", alpha=0.15)

    ax_r.set_xlabel("target_snr  (internal calibration knob)")
    ax_r.set_ylabel("measured SNR  (three methods)")
    ax_r.set_title("Three SNR metrics vs the same internal knob\n"
                   "(gap snr_dual↔snr_nema is the Gibbs bias)")
    ax_r.grid(True, which="both", alpha=0.3)
    ax_r.legend(fontsize=8, loc="upper left")

    # ── Suptitle with a quick recommendation ─────────────────────────────────
    # Find the target_snr whose snr_dual_peak lands inside 20–50.
    in_band = (snr_dual >= 20) & (snr_dual <= 50)
    rec_msg = ""
    if in_band.any():
        idxs = np.where(in_band)[0]
        ts_lo, ts_hi = target_snr[idxs].min(), target_snr[idxs].max()
        rec_msg = (f"  →  target_snr ≈ {ts_lo:g}–{ts_hi:g} "
                   f"lands in the clinical band (snr_dual_peak 20–50)")
    fig.suptitle(
        "Noise sensitivity of the CR-optimal T1 fit"
        + rec_msg,
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ─── Plain-text summary (printed to stdout, mirrors the figure numbers) ──────

def print_text_summary(data: dict[str, np.ndarray]):
    print()
    print("  target_snr   σ        snr_ksp   snr_nema   snr_dual   "
          "MAPE_mean   MAPE_med    MAPE_max")
    print("  " + "─" * 88)
    for i in range(len(data["target_snr"])):
        print(f"  {data['target_snr'][i]:8.2f}   "
              f"{data['noise_sigma_abs'][i]:6.3f}   "
              f"{data['snr_ksp'][i]:7.2f}   "
              f"{data['snr_nema_peak'][i]:8.2f}   "
              f"{data['snr_dual_peak'][i]:8.2f}   "
              f"{data['mape_mean_pct'][i]:8.2f}%   "
              f"{data['mape_median_pct'][i]:7.2f}%   "
              f"{data['mape_max_pct'][i]:8.2f}%")
    print()


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--indir", type=Path, default=HERE / "snr_sweep",
                   help="Directory containing snr_sweep.csv")
    p.add_argument("--out",   type=Path, default=None,
                   help="Output plot path (default: <indir>/snr_sweep.png)")
    args = p.parse_args()

    csv_path = args.indir / "snr_sweep.csv"
    if not csv_path.exists():
        raise SystemExit(
            f"Missing {csv_path}. Run "
            "`julia --project=. scripts/snr_sweep.jl` first.")

    data = load_summary(csv_path)
    out_path = args.out or (args.indir / "snr_sweep.png")
    make_figure(data, out_path)
    print(f"Wrote {out_path}")
    print_text_summary(data)


if __name__ == "__main__":
    main()

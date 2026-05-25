"""scripts/snr_sweep.py — visualise the noise-sensitivity sweep.

Reads `runs/snr_sweep/{config.json, snr_sweep.csv, images/*.npz}` produced by
`snr_sweep.jl` and emits:

    figures/mape_vs_sigma.png  — MAPE (mean / median / max) on a log-y axis
                                  vs σ on the primary x-axis, with three
                                  twiny axes across the top for the three
                                  measured SNR conventions (snr_ksp,
                                  snr_nema_peak_a, snr_dual_peak).
    figures/grid_one_block.png — one representative block (middle of the
                                  schedule) reconstructed at each σ + a
                                  noise-free reference column.
    figures/grid_all_blocks.png — every (σ, block) recon panel laid out as a
                                  grid: rows = σ (low → high), cols = block
                                  index (low TI → high TI).

═══════════════════════════════════════════════════════════════════════════
REPORT-USEFUL FACTS — copy these into the §SNR-methodology subsection
═══════════════════════════════════════════════════════════════════════════

1. We measure SNR using three methods. Only one is comparable to clinical
   literature: NEMA MS-1 dual-acquisition (NEMA Standards Publication
   MS-1, 2014).

2. NEMA single-image SNR uses mean(ROI)/std(background), with a 1/0.6551
   Rayleigh correction for magnitude images of zero-mean complex Gaussian
   noise (Henkelman 1985; Gudbjartsson & Patz 1995). On the coarse
   32 × 64 image grid we use this reads SYSTEMATICALLY LOW because Gibbs
   ringing from sphere edges contaminates the "background" pixels.

3. NEMA MS-1 dual-acquisition takes two independent noise realisations.
   The difference image (A − B) exactly cancels structured signal —
   including Gibbs ringing — leaving only zero-mean Gaussian noise. Noise
   std is computed from (A − B) pooled across the signal ROIs and divided
   by √2. Gold-standard reproducible SNR.

4. Clinical reference: routine 1.5 T / 3 T quantitative T1 mapping
   reports image SNR (NEMA MS-1) in the range 20–50.

Usage:
    python scripts/snr_sweep.py
    python scripts/snr_sweep.py --indir scripts/runs/snr_sweep_voxel_0p5mm
    python scripts/snr_sweep.py --indir scripts/runs/snr_sweep_voxel_1mm
    python scripts/snr_sweep.py --indir scripts/runs/snr_sweep_voxel_3mm
    python scripts/snr_sweep.py --indir scripts/runs/snr_sweep_voxel_0p5mm_water
    python scripts/snr_sweep.py --indir scripts/runs/snr_sweep_voxel_1mm_water
    python scripts/snr_sweep.py --indir scripts/runs/snr_sweep_voxel_3mm_water
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

HERE = Path(__file__).resolve().parent
RUNS_SNR = HERE / "runs" / "snr_sweep"


# ─── Data loading ────────────────────────────────────────────────────────────

def load_summary(csv_path: Path) -> dict[str, np.ndarray]:
    cols: dict[str, list[float]] = {}
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                cols.setdefault(k, []).append(float(v))
    return {k: np.asarray(v, dtype=np.float64) for k, v in cols.items()}


def load_config(cfg_path: Path) -> dict:
    with cfg_path.open() as f:
        return json.load(f)


def load_images_npz(npz_path: Path, n_blocks: int) -> list[np.ndarray]:
    with np.load(npz_path) as z:
        return [np.asarray(z[f"block_{b}"]) for b in range(1, n_blocks + 1)]


# ─── MAPE-vs-σ figure (with 4 stacked x-axes) ────────────────────────────────

def make_mape_vs_sigma(data: dict[str, np.ndarray], out_path: Path,
                       xscale: str = "log", yscale: str = "log",
                       drop_top_n: int = 0, show_max: bool = True):
    σ = data["sigma"]
    snr_ksp = data["snr_ksp_measured"]
    snr_ksp_phantom = data.get(
        "snr_ksp_phantom", np.full_like(snr_ksp, np.nan))
    snr_nema = data["snr_nema_peak_a"]
    snr_dual = data["snr_dual_peak"]
    m_mean = data["mape_mean_pct"]
    m_med = data["mape_median_pct"]
    m_max = data["mape_max_pct"]
    m_std_lo = data.get("mape_mean_pct_std_lo", np.zeros_like(m_mean))
    m_std_hi = data.get("mape_mean_pct_std_hi", np.zeros_like(m_mean))
    m_med_std = data.get("mape_median_pct_std", np.zeros_like(m_mean))

    # Sort by σ ascending. `drop_top_n` removes the largest N σ values —
    # the fitter typically collapses at high noise and those points squash
    # the rest of the curve when plotting on a linear x-axis. On a log
    # x-axis we additionally drop σ=0 (log undefined).
    order = np.argsort(σ)
    if drop_top_n > 0 and len(order) > drop_top_n:
        order = order[:-drop_top_n]
    if xscale == "log":
        order = order[σ[order] > 0]
    σ = σ[order]
    snr_ksp = snr_ksp[order]
    snr_ksp_phantom = snr_ksp_phantom[order]
    snr_nema = snr_nema[order]
    snr_dual = snr_dual[order]
    m_mean, m_med, m_max = m_mean[order], m_med[order], m_max[order]
    m_std_lo, m_std_hi = m_std_lo[order], m_std_hi[order]
    m_med_std = m_med_std[order]

    fig, ax = plt.subplots(figsize=(11, 7))
    if xscale == "log":
        ax.set_xscale("log")
    if yscale == "log":
        ax.set_yscale("log")

    # Mean bars are ASYMMETRIC one-sided std (lower from samples below the
    # mean, upper from samples above) — MAPE's seed distribution is
    # right-skewed at high σ. Clipped at the mean so the lower whisker
    # never crosses 0.
    yerr_lo = np.minimum(m_std_lo, m_mean)
    ax.errorbar(σ, m_mean, yerr=[yerr_lo, m_std_hi], fmt="o-",
                color="#1f77b4", lw=1.6, ms=7, capsize=3,
                label="mean MAPE  (±one-sided std across seeds)")
    # Median (mean-of-medians) doesn't suffer from the right-tail skew;
    # symmetric std is fine.
    yerr_med_lo = np.minimum(m_med_std, m_med)
    ax.errorbar(σ, m_med, yerr=[yerr_med_lo, m_med_std], fmt="s--",
                color="#2ca02c", lw=1.2, ms=6, alpha=0.85, capsize=3,
                label="median MAPE  (mean-of-medians ±std across seeds)")
    if show_max:
        ax.plot(σ, m_max, "^:", color="#d62728", lw=1.2, ms=6, alpha=0.85,
                label="max MAPE (worst sphere)")

    ax.axhline(5.0,  color="green",  ls="--",
               lw=1, alpha=0.5, label="5 % MAPE")
    ax.axhline(10.0, color="orange", ls=":",
               lw=1, alpha=0.5, label="10 % MAPE")

    ax.set_xlabel(r"noise $\sigma$  (k-space, absolute)")
    ax.set_ylabel("Fit MAPE [%]  (CR-optimal baseline)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")

    # ── Three twiny axes across the top for the three SNR conventions ──────
    # Each ax shares the σ x-coords but relabels them with one of the
    # measured SNR series. We use ax.twiny() with the same data, then hide
    # the line and only show ticks at the σ positions.
    def add_top_axis(parent, offset_pts, label, snr_vals, color):
        tw = parent.twiny()
        if xscale == "log":
            tw.set_xscale("log")
        tw.set_xlim(parent.get_xlim())  # match data range
        # Place ticks at every σ; label them with the SNR value at that σ.
        # Skip σ values whose SNR metric is undefined (NaN / inf). Use %.3g
        # so giant values (e.g. snr_dual at σ=0, where the diff std is
        # floating-point noise ≈1e-16) render in scientific notation.
        # Drop σ=0 from the top axes: snr_ksp/snr_ksp_phantom/snr_dual are
        # ∞ there (and would crowd the σ=1 label on a linear x-axis),
        # and snr_nema's σ=0 value (the Gibbs floor) is captured in the
        # axis title instead.
        finite = np.isfinite(snr_vals) & (σ > 0)
        tw.set_xticks(σ[finite])
        tw.set_xticklabels([f"{v:.3g}" for v in snr_vals[finite]],
                           fontsize=8, color=color)
        tw.tick_params(axis="x", which="minor", top=False)
        tw.tick_params(axis="x", which="major", colors=color, top=True,
                       bottom=False, labelbottom=False, labeltop=True,
                       direction="out", length=4)
        tw.spines["top"].set_position(("outward", offset_pts))
        tw.spines["top"].set_color(color)
        tw.set_xlabel(label, color=color, fontsize=9)
        return tw

    # σ-limits must be set on the primary axis BEFORE we copy them
    if xscale == "log":
        ax.set_xlim(σ.min() * 0.8, σ.max() * 1.25)
    else:
        pad = 0.05 * (σ.max() - σ.min())
        ax.set_xlim(σ.min() - pad, σ.max() + pad)
    # Look up the σ=0 value for each SNR series (if σ=0 is in the data) so
    # we can call it out in the axis title — those points are dropped from
    # the ticks themselves to avoid crowding.
    zero_idx = np.where(σ == 0)[0]

    def at_zero(arr, *, inf_label="∞"):
        if len(zero_idx) == 0:
            return ""
        v = arr[zero_idx[0]]
        if not np.isfinite(v) or v > 1e6:
            return f"  [σ=0: {inf_label}]"
        return f"  [σ=0: {v:.3g}]"

    add_top_axis(ax,   0,
                 "snr_ksp_measured  (= mean ksp_rms / σ, all pixels)"
                 + at_zero(snr_ksp),
                 snr_ksp, "#888888")
    add_top_axis(ax,  36,
                 "snr_ksp_phantom  (= snr_ksp · √(N_total/N_phantom))"
                 + at_zero(snr_ksp_phantom),
                 snr_ksp_phantom, "#444444")
    add_top_axis(ax,  72,
                 "snr_nema_peak_a  (single-image, Gibbs-biased)"
                 + at_zero(snr_nema),
                 snr_nema, "#ff7f0e")
    add_top_axis(ax, 108,
                 "snr_dual_peak  (NEMA MS-1, pooled across blocks — REPORT FIGURE)"
                 + at_zero(snr_dual),
                 snr_dual, "#1f77b4")

    fig.suptitle("Fit MAPE vs noise — three SNR conventions on the top axis",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ─── Image grids ─────────────────────────────────────────────────────────────

def _imshow(ax, img, title=None, vmax=None):
    ax.imshow(img, cmap="gray", origin="lower", vmin=0,
              vmax=vmax if vmax is not None else img.max())
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=8)


def make_grid_one_block(cfg: dict, data: dict[str, np.ndarray],
                        indir: Path, out_path: Path):
    """One representative block per σ, plus a noise-free reference column."""
    n_blocks = cfg["n_blocks"]
    sigmas = list(zip(cfg["sigmas"], cfg["sigma_labels"]))
    sigmas.sort(key=lambda x: x[0])  # ascending σ
    mid = (n_blocks + 1) // 2  # 1-based block index, middle of schedule

    noise_free = load_images_npz(indir / "images" / "noise_free.npz", n_blocks)
    ref = noise_free[mid - 1]
    vmax = float(ref.max())

    n = len(sigmas) + 1
    fig, axes = plt.subplots(1, n, figsize=(2.0 * n, 2.5))

    _imshow(axes[0], ref,
            title=f"noise-free\n(block {mid}, TI={cfg['TIs_s'][mid-1]:.2f}s)",
            vmax=vmax)

    # Map σ → measured snr_dual for annotation
    snr_dual_by_sigma = {s: d for s, d in zip(
        data["sigma"], data["snr_dual_peak"])}

    for k, (σ, label) in enumerate(sigmas, start=1):
        imgs = load_images_npz(
            indir / "images" / f"sigma_{label}.npz", n_blocks)
        sd = snr_dual_by_sigma.get(σ, float("nan"))
        _imshow(axes[k], imgs[mid - 1],
                title=f"σ={σ:.2g}\nsnr_dual={sd:.1f}",
                vmax=vmax)

    fig.suptitle(f"Recon of block {mid} across the σ sweep", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def make_grid_all_blocks(cfg: dict, indir: Path, out_path: Path):
    """Full grid: rows = σ (incl. noise-free at top), cols = block index."""
    n_blocks = cfg["n_blocks"]
    sigmas = list(zip(cfg["sigmas"], cfg["sigma_labels"]))
    sigmas.sort(key=lambda x: x[0])

    noise_free = load_images_npz(indir / "images" / "noise_free.npz", n_blocks)
    vmax = float(max(im.max() for im in noise_free))

    n_rows = 1 + len(sigmas)
    n_cols = n_blocks
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(1.5 * n_cols, 1.6 * n_rows),
                             squeeze=False)

    # Row 0: noise-free
    for b in range(n_blocks):
        title = (f"block {b+1}\nTI={cfg['TIs_s'][b]:.2f}s "
                 f"TR={cfg['TRs_s'][b]:.2f}s") if b == 0 else \
                (f"block {b+1}\nTI={cfg['TIs_s'][b]:.2f}s "
                 f"TR={cfg['TRs_s'][b]:.2f}s")
        _imshow(axes[0, b], noise_free[b], title=title, vmax=vmax)
    axes[0, 0].set_ylabel("noise-free", fontsize=9)

    # Subsequent rows: per-σ
    for r, (σ, label) in enumerate(sigmas, start=1):
        imgs = load_images_npz(
            indir / "images" / f"sigma_{label}.npz", n_blocks)
        for b in range(n_blocks):
            _imshow(axes[r, b], imgs[b], vmax=vmax)
        axes[r, 0].set_ylabel(f"σ={σ:.2g}", fontsize=9)

    fig.suptitle("All (σ, block) reconstructions", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ─── Plain-text summary ──────────────────────────────────────────────────────

def print_text_summary(data: dict[str, np.ndarray]):
    print()
    has_phantom = "snr_ksp_phantom" in data
    has_block_range = ("snr_dual_block_min" in data
                       and "snr_dual_block_max" in data)
    header = ("  σ          snr_ksp   "
              + ("snr_ksp_ph   " if has_phantom else "")
              + "snr_nema   snr_dual"
              + ("    (block min–max)   " if has_block_range else "   ")
              + "MAPE_mean   MAPE_med    MAPE_max")
    print(header)
    print("  " + "─" * len(header))
    order = np.argsort(data["sigma"])
    for i in order:
        ph_col = (f"{data['snr_ksp_phantom'][i]:>9.3g}    "
                  if has_phantom else "")
        range_col = (
            f"  ({data['snr_dual_block_min'][i]:>6.3g}–{data['snr_dual_block_max'][i]:<6.3g})   "
            if has_block_range else "   "
        )
        print(f"  {data['sigma'][i]:8.4g}   "
              f"{data['snr_ksp_measured'][i]:>7.3g}   "
              f"{ph_col}"
              f"{data['snr_nema_peak_a'][i]:>8.3g}   "
              f"{data['snr_dual_peak'][i]:>9.3g}"
              f"{range_col}"
              f"{data['mape_mean_pct'][i]:8.2f}%   "
              f"{data['mape_median_pct'][i]:7.2f}%   "
              f"{data['mape_max_pct'][i]:8.2f}%")
    print()


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--indir", type=Path, default=RUNS_SNR,
                   help="Directory containing snr_sweep.csv + config.json")
    args = p.parse_args()

    csv_path = args.indir / "snr_sweep.csv"
    cfg_path = args.indir / "config.json"
    if not csv_path.exists() or not cfg_path.exists():
        raise SystemExit(
            f"Missing {csv_path} or {cfg_path}. "
            "Run `julia --project=. scripts/snr_sweep.jl` first.")

    data = load_summary(csv_path)
    cfg = load_config(cfg_path)

    figdir = args.indir / "figures"
    figdir.mkdir(exist_ok=True)

    make_mape_vs_sigma(data, figdir / "mape_vs_sigma.png",
                       xscale="log", drop_top_n=0)
    print(f"Wrote {figdir/'mape_vs_sigma.png'}")
    make_mape_vs_sigma(data, figdir / "mape_vs_sigma_linear.png",
                       xscale="linear", yscale="linear",
                       drop_top_n=1, show_max=True)
    print(f"Wrote {figdir/'mape_vs_sigma_linear.png'}")
    make_mape_vs_sigma(data, figdir / "mape_vs_sigma_linear_nomax.png",
                       xscale="linear", yscale="linear",
                       drop_top_n=2, show_max=False)
    print(f"Wrote {figdir/'mape_vs_sigma_linear_nomax.png'}")
    make_mape_vs_sigma(data, figdir / "mape_vs_sigma_linear_zoom.png",
                       xscale="linear", yscale="linear",
                       drop_top_n=5, show_max=False)
    print(f"Wrote {figdir/'mape_vs_sigma_linear_zoom.png'}")

    make_grid_one_block(cfg, data, args.indir, figdir / "grid_one_block.png")
    print(f"Wrote {figdir/'grid_one_block.png'}")

    make_grid_all_blocks(cfg, args.indir, figdir / "grid_all_blocks.png")
    print(f"Wrote {figdir/'grid_all_blocks.png'}")

    print_text_summary(data)


if __name__ == "__main__":
    main()

"""scripts/voxel_snr_collapse.py — cross-voxel SNR / MAPE collapse plots.

Reads the per-voxel sweep outputs produced by `snr_sweep.jl` (one indir
per voxel size: `snr_sweep.csv` + `config.json`) and emits two families
of comparison figures:

  1. **SNR vs σ·v³.**  `voxel_size_mm` here is the spin-discretisation
     pitch — total spin count in the phantom scales as 1/v³, so MORE
     spins fall inside a pixel when v is small, and per-pixel signal
     scales as 1/v³. Injected k-space noise σ is voxel-independent, so
     SNR ∝ 1/(σ·v³). Plotting SNR against σ·v³ collapses all voxel
     sizes onto one curve y ∝ 1/x. (The Julia sweep's default σ list
     uses σ = σ_baseline · (3/v)³, so σ·v³ is identical across voxels
     at matched baseline indices — the collapse table should report a
     ratio ≈ 1.)

  2. **MAPE vs SNR.**  If the fit cares about image-domain SNR (not σ
     directly), MAPE-vs-snr_dual should also collapse across voxel
     sizes. The inflection where MAPE explodes is then a single
     "this fit needs snr_dual ≳ X" statement.

The script is read-only over the existing CSVs; no Julia re-runs.

Water-phantom sweeps (--water-indirs) are plotted separately AND overlaid
on the combined figures as dashed lines (same colour per voxel size).

═══════════════════════════════════════════════════════════════════════════
OLD RESULTS — pre-Koma bug fixes (0.5 / 1 / 3 mm sweeps, σ scaled (3/v)³)
═══════════════════════════════════════════════════════════════════════════

At matched σ·v³ values across voxels, the SNR ratio (max/min over the
three voxel sizes) was:

  σ·v³    snr_ksp_measured   snr_dual_peak   snr_nema_peak_a
  ─────────────────────────────────────────────────────────────
  27          1.079              1.007            1.471
  81          1.079              1.341            1.463
  810         1.079              1.197            1.220
  1080        1.079              1.215            1.205
  1350        1.079              1.439            1.200

Old takeaways (marked for reference only — superseded by new results):

* `snr_ksp_measured` collapsed essentially exactly (ratio = 1.079, constant
  across all σ·v³).

* `snr_dual_peak` collapsed well (≤ 22 % spread) through the working
  range.

* `snr_nema_peak_a` collapsed worst (~20–47 % spread): 3 mm reads
  systematically LOW due to Gibbs bleed on coarse grids.

* MAPE-vs-snr_dual showed the fit breaking down at snr_dual ≈ 13–17,
  consistently across voxel sizes.

═══════════════════════════════════════════════════════════════════════════
UPDATED RESULTS — post-Koma bug fixes (0.5 / 1 / 3 mm normal + 1 / 3 mm water)
═══════════════════════════════════════════════════════════════════════════

Phantom sweeps (NiCl₂/MnCl₂):

  σ·v³    snr_ksp_measured (0.5mm / 1mm / 3mm)         ratio
  ──────────────────────────────────────────────────────────────
  27       35.41  /  35.09  /  32.75                   1.081
  81       11.80  /  11.70  /  10.92                   1.081
  810       1.18  /   1.17  /   1.09                   1.081
  1080      0.89  /   0.88  /   0.82                   1.081
  1350      0.71  /   0.70  /   0.65                   1.081

New takeaways (phantom sweeps):

* `snr_ksp_measured` still collapses essentially exactly: ratio = 1.081
  (fixed ~8 % offset, unchanged from pre-fix runs). The v³ signal scaling
  hypothesis is confirmed to be robust to the Koma bug fixes.

* MAPE inflection band is approximately snr_dual ≈ 13–18 — consistent
  with the old 13–17 finding. Sharp transitions:
    0.5 mm: MAPE jumps 15 % → 134 % between snr_dual = 18 and 15
    1 mm:   MAPE jumps 12 % →  77 % between snr_dual = 21 and 15
    3 mm:   MAPE jumps 15 % →  78 % between snr_dual = 16 and 13

* The working regime (MAPE < ~15 %) requires snr_dual ≳ 18 across all
  voxel sizes — a safe operating threshold for the RL training noise level.

Water phantom sweeps (0.5 mm + 1 mm + 3 mm):

* Water signal is substantially weaker than the NiCl₂/MnCl₂ phantom:
    0.5 mm: water ksp_rms_mean = 640  vs  phantom = 7649  (11.9× lower)
    1 mm:   water ksp_rms_mean = 319  vs  phantom =  947   (3.0× lower)
    3 mm:   water ksp_rms_mean =  24  vs  phantom =   33   (1.4× lower)
  The large and voxel-size-dependent gap (12× at 0.5 mm vs 1.4× at 3 mm)
  confirms the water signal does NOT scale as v³: the T1 plate spheres
  contribute fixed-T1 signal independent of the water background, so the
  relative weight of water vs NiCl₂ spins changes with voxel size.

* Water `snr_ksp_measured` does NOT collapse with σ·v³:
  ratio = 8.08 across all σ·v³ points (a constant factor). For comparison
  the phantom collapses to ratio = 1.081. σ·v³ is not the right collapse
  variable for the water phantom.

* Water MAPE at zero noise:
    0.5 mm water:  7.50 % (vs 5.95 % for 0.5 mm phantom)
    1 mm water:    7.49 % (vs 5.86 % for 1 mm phantom)
    3 mm water:    5.29 % (vs 6.70 % for 3 mm phantom)
  T1 fitting is harder from the water slice at 0.5–1 mm (long water T1 is
  poorly constrained by short TIs). The 0.5 mm water run collapses almost
  immediately: at the first non-zero σ (σ=216, snr_dual=13.4) MAPE is
  already 15.5 %, and at σ=648 (snr_dual=4.5) it explodes to 236 %.

NOTE — 0.5 mm water OOM fix (applied in src/builder.jl):
  `build_background_water` previously voxelised the full 100 mm-radius
  housing sphere in 3D before `build_phantom` applied the z-slice mask.
  At 0.5 mm that is ~33 M spins (~800 MB), causing SIGKILL. Fix: apply the
  slice mask inside `build_background_water` immediately after calling
  `voxelise_sphere`, reducing the 0.5 mm water phantom to 85 498 spins.
  The 0.5 mm water run therefore used --seeds 5 (vs 15 for other sweeps).

Usage:
    python scripts/voxel_snr_collapse.py \\
        scripts/runs/snr_sweep_voxel_0p5mm scripts/runs/snr_sweep_voxel_1mm \\
        scripts/runs/snr_sweep_voxel_3mm \\
        --water-indirs \\
            scripts/runs/snr_sweep_voxel_1mm_water \\
            scripts/runs/snr_sweep_voxel_3mm_water
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from snr_sweep import load_config, load_summary


@dataclass(frozen=True)
class SnrMetric:
    """One column to plot as a v³-collapse curve.

    `band_lo` / `band_hi` are optional CSV column names for a shaded
    range around the line (currently only `snr_dual_peak` has these —
    its per-block min/max columns).
    """
    column: str
    label: str
    band_lo: str | None = None
    band_hi: str | None = None

HERE = Path(__file__).resolve().parent

# Stable voxel→colour map so curves are easy to track between figures.
VOXEL_COLOURS = {
    0.5: "#1f77b4",
    1.0: "#2ca02c",
    3.0: "#d62728",
}


def _voxel_colour(v_mm: float) -> str:
    return VOXEL_COLOURS.get(round(v_mm, 3), "#888888")


def _voxel_label(v_mm: float) -> str:
    return f"v = {v_mm:g} mm"


def load_indir(indir: Path) -> tuple[float, dict[str, np.ndarray]]:
    csv_path = indir / "snr_sweep.csv"
    cfg_path = indir / "config.json"
    if not csv_path.exists() or not cfg_path.exists():
        raise SystemExit(f"Missing {csv_path} or {cfg_path}.")
    cfg  = load_config(cfg_path)
    data = load_summary(csv_path)
    return float(cfg["voxel_mm"]), data


# ─── SNR vs σ·v³ ─────────────────────────────────────────────────────────────

def _drop_zero_sigma(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    mask = data["sigma"] > 0
    return {k: v[mask] for k, v in data.items()}


def _draw_snr_collapse(
    ax,
    runs: list[tuple[float, dict[str, np.ndarray]]],
    metric: SnrMetric,
    *,
    water_runs: list[tuple[float, dict[str, np.ndarray]]] | None = None,
    legend: bool = True,
):
    """Render one SNR-vs-σ·v³ collapse panel into the given axis.

    Normal runs are solid lines; water runs (if provided) are dashed,
    same colour per voxel size.
    """
    ax.set_xscale("log")
    ax.set_yscale("log")

    def _plot_one(v_mm, data, *, dashed: bool):
        d   = _drop_zero_sigma(data)
        x   = d["sigma"] * (v_mm ** 3)
        y   = d[metric.column]
        ok  = np.isfinite(x) & np.isfinite(y) & (y > 0)
        order = np.argsort(x[ok])
        x_s, y_s = x[ok][order], y[ok][order]
        col = _voxel_colour(v_mm)
        ls  = "--" if dashed else "-"
        suffix = " (water)" if dashed else ""
        if not dashed and metric.band_lo and metric.band_hi \
                and metric.band_lo in d and metric.band_hi in d:
            lo = d[metric.band_lo][ok][order]
            hi = d[metric.band_hi][ok][order]
            band_ok = np.isfinite(lo) & np.isfinite(hi) & (lo > 0) & (hi > 0)
            if band_ok.any():
                ax.fill_between(x_s[band_ok], lo[band_ok], hi[band_ok],
                                color=col, alpha=0.15, linewidth=0)
        ax.plot(x_s, y_s, f"o{ls}", color=col, lw=1.6, ms=5,
                label=_voxel_label(v_mm) + suffix)

    for v_mm, data in runs:
        _plot_one(v_mm, data, dashed=False)
    for v_mm, data in (water_runs or []):
        _plot_one(v_mm, data, dashed=True)

    ax.set_xlabel(r"$\sigma \cdot v_{\mathrm{mm}}^3$  "
                  r"(σ scaled by voxel volume — collapse variable)")
    ax.set_ylabel(metric.label)
    ax.grid(True, which="both", alpha=0.3)
    if legend:
        ax.legend(fontsize=8, loc="best")


def make_snr_collapse(
    runs: list[tuple[float, dict[str, np.ndarray]]],
    metric: SnrMetric,
    out_path: Path,
    *,
    water_runs: list[tuple[float, dict[str, np.ndarray]]] | None = None,
    title_suffix: str = "",
):
    """One SNR metric, all voxel sizes overlaid as functions of σ·v³."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    _draw_snr_collapse(ax, runs, metric, water_runs=water_runs)
    ax.set_title(f"{metric.label}  vs  σ·v³  —  v³ collapse check{title_suffix}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def make_snr_collapse_grid(
    runs: list[tuple[float, dict[str, np.ndarray]]],
    metrics: list[SnrMetric],
    out_path: Path,
    *,
    water_runs: list[tuple[float, dict[str, np.ndarray]]] | None = None,
    suptitle: str = "All SNR metrics  vs  σ·v³  —  v³ collapse check",
):
    """2×2 grid with one SNR metric per panel, all voxels overlaid."""
    n = len(metrics)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(6.5 * ncols, 4.8 * nrows),
                             squeeze=False)
    for ax_idx, m in enumerate(metrics):
        ax = axes[ax_idx // ncols, ax_idx % ncols]
        _draw_snr_collapse(ax, runs, m,
                           water_runs=water_runs,
                           legend=(ax_idx == 0))
        ax.set_title(m.label, fontsize=10)
    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")
    fig.suptitle(suptitle, fontsize=12, y=1.00)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ─── MAPE vs SNR ─────────────────────────────────────────────────────────────

def make_mape_vs_snr(
    runs: list[tuple[float, dict[str, np.ndarray]]],
    snr_col: str, snr_label: str,
    out_path: Path,
    *,
    water_runs: list[tuple[float, dict[str, np.ndarray]]] | None = None,
    inflection_hint: tuple[float, float] | None = (13.0, 17.0),
    title_suffix: str = "",
):
    """MAPE on y (log), `snr_col` on x (log), all voxels overlaid.

    Normal runs are solid; water runs are dashed, same colour per voxel.
    """
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.set_xscale("log")
    ax.set_yscale("log")

    def _plot_one(v_mm, data, *, dashed: bool):
        d  = _drop_zero_sigma(data)
        x  = d[snr_col]
        y  = d["mape_mean_pct"]
        ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        order = np.argsort(x[ok])
        ls = "--" if dashed else "-"
        suffix = " (water)" if dashed else ""
        ax.plot(x[ok][order], y[ok][order], f"o{ls}",
                color=_voxel_colour(v_mm), lw=1.6, ms=6,
                label=_voxel_label(v_mm) + suffix)

    for v_mm, data in runs:
        _plot_one(v_mm, data, dashed=False)
    for v_mm, data in (water_runs or []):
        _plot_one(v_mm, data, dashed=True)

    if inflection_hint:
        lo, hi = inflection_hint
        ax.axvspan(lo, hi, color="#999999", alpha=0.15,
                   label=f"observed collapse band ({lo:g}–{hi:g})")
    ax.axhline(5.0,  color="green",  ls="--", lw=1, alpha=0.5)
    ax.axhline(10.0, color="orange", ls=":",  lw=1, alpha=0.5)

    ax.set_xlabel(snr_label + "  (higher = less noise)")
    ax.set_ylabel("Fit MAPE [%]  (mean across spheres × seeds)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_title(f"MAPE  vs  {snr_label}  —  cross-voxel collapse{title_suffix}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ─── Collapse-quality text summary ───────────────────────────────────────────

def print_collapse_table(
    runs: list[tuple[float, dict[str, np.ndarray]]],
    metric_col: str, label: str,
):
    """For each x = σ·v³ point present in *all* voxel runs (within a
    tolerance), print SNR values side-by-side + the max/min ratio.

    With σ defaults scaled exactly by (3/v)³ across the three voxel
    sweeps, x = σ·v³ matches to 4+ sig figs for the canonical points.
    """
    runs = sorted(runs, key=lambda r: r[0])
    voxels = [v for v, _ in runs]
    # Build {x_key: {v: y}}
    by_x: dict[float, dict[float, float]] = {}
    for v, data in runs:
        d = _drop_zero_sigma(data)
        x = d["sigma"] * (v ** 3)
        y = d[metric_col]
        for xi, yi in zip(x, y):
            key = round(float(xi), 6)
            by_x.setdefault(key, {})[v] = float(yi)

    # Keep only x points present in every voxel run.
    aligned = sorted(k for k, vs in by_x.items() if len(vs) == len(voxels))
    if not aligned:
        print(f"\n[{label}] no σ·v³ points matched across all voxel runs.")
        return

    print()
    print(f"  Collapse-quality table for {label}")
    header = "  σ·v³        " + "  ".join(f"v={v:<4g}" for v in voxels) \
             + "    ratio (max/min)"
    print(header)
    print("  " + "─" * (len(header) - 2))
    for x in aligned:
        cells = by_x[x]
        vals  = np.array([cells[v] for v in voxels])
        finite = vals[np.isfinite(vals)]
        if len(finite) >= 2 and finite.min() > 0:
            ratio = f"{finite.max() / finite.min():.3f}"
        else:
            ratio = "n/a"
        row = (f"  {x:<10.4g}  "
               + "  ".join(f"{v:>6.3g}" for v in vals)
               + f"    {ratio}")
        print(row)
    print()


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("indirs", type=Path, nargs="+",
                   help="≥2 snr_sweep output directories (one per voxel size)")
    p.add_argument("--water-indirs", type=Path, nargs="+", default=[],
                   metavar="DIR",
                   help="Water-phantom sweep directories (one per voxel size). "
                        "Plotted separately and overlaid as dashed lines.")
    p.add_argument("--outdir", type=Path,
                   default=HERE / "runs" / "voxel_collapse" / "figures",
                   help="Where to write the figures.")
    args = p.parse_args()
    if len(args.indirs) < 2:
        raise SystemExit("Need ≥ 2 indirs to plot a collapse.")

    runs = [load_indir(d) for d in args.indirs]
    runs.sort(key=lambda r: r[0])

    water_runs: list[tuple[float, dict[str, np.ndarray]]] = []
    if args.water_indirs:
        water_runs = [load_indir(d) for d in args.water_indirs]
        water_runs.sort(key=lambda r: r[0])

    args.outdir.mkdir(parents=True, exist_ok=True)
    print(f"Voxel sizes (mm):       {[v for v, _ in runs]}")
    if water_runs:
        print(f"Water voxel sizes (mm): {[v for v, _ in water_runs]}")
    print(f"Output dir              : {args.outdir}")

    snr_metrics = [
        SnrMetric("snr_dual_peak",
                  "snr_dual_peak  (NEMA MS-1, pooled across blocks)",
                  band_lo="snr_dual_block_min",
                  band_hi="snr_dual_block_max"),
        SnrMetric("snr_nema_peak_a",  "snr_nema_peak_a  (single-image NEMA)"),
        SnrMetric("snr_ksp_measured", "snr_ksp_measured  (internal k-space)"),
        SnrMetric("snr_ksp_phantom",  "snr_ksp_phantom  (in-tissue corrected)"),
    ]

    # ── (a) Normal sweeps only ─────────────────────────────────────────────────
    for m in snr_metrics:
        out = args.outdir / f"{m.column}_vs_sigma_times_v3.png"
        make_snr_collapse(runs, m, out)
        print(f"Wrote {out}")
    grid_out = args.outdir / "snr_all_metrics_vs_sigma_times_v3.png"
    make_snr_collapse_grid(runs, snr_metrics, grid_out)
    print(f"Wrote {grid_out}")

    make_mape_vs_snr(runs, "snr_dual_peak", "snr_dual_peak (pooled)",
                     args.outdir / "mape_vs_snr_dual.png")
    print(f"Wrote {args.outdir / 'mape_vs_snr_dual.png'}")
    make_mape_vs_snr(runs, "snr_nema_peak_a", "snr_nema_peak_a",
                     args.outdir / "mape_vs_snr_nema.png",
                     inflection_hint=None)
    print(f"Wrote {args.outdir / 'mape_vs_snr_nema.png'}")

    # ── (b) Water sweeps only (if provided) ───────────────────────────────────
    if water_runs:
        wdir = args.outdir / "water"
        wdir.mkdir(parents=True, exist_ok=True)
        for m in snr_metrics:
            out = wdir / f"{m.column}_vs_sigma_times_v3_water.png"
            make_snr_collapse(water_runs, m, out, title_suffix="  [water]")
            print(f"Wrote {out}")
        grid_out_w = wdir / "snr_all_metrics_vs_sigma_times_v3_water.png"
        make_snr_collapse_grid(
            water_runs, snr_metrics, grid_out_w,
            suptitle="All SNR metrics  vs  σ·v³  —  water phantom",
        )
        print(f"Wrote {grid_out_w}")

        make_mape_vs_snr(water_runs, "snr_dual_peak", "snr_dual_peak (pooled)",
                         wdir / "mape_vs_snr_dual_water.png",
                         title_suffix="  [water]")
        print(f"Wrote {wdir / 'mape_vs_snr_dual_water.png'}")
        make_mape_vs_snr(water_runs, "snr_nema_peak_a", "snr_nema_peak_a",
                         wdir / "mape_vs_snr_nema_water.png",
                         inflection_hint=None, title_suffix="  [water]")
        print(f"Wrote {wdir / 'mape_vs_snr_nema_water.png'}")

        # ── (c) Combined overlay (normal solid, water dashed) ─────────────────
        cdir = args.outdir / "combined"
        cdir.mkdir(parents=True, exist_ok=True)
        for m in snr_metrics:
            out = cdir / f"{m.column}_vs_sigma_times_v3_combined.png"
            make_snr_collapse(runs, m, out,
                              water_runs=water_runs,
                              title_suffix="  [phantom vs water]")
            print(f"Wrote {out}")
        grid_out_c = cdir / "snr_all_metrics_vs_sigma_times_v3_combined.png"
        make_snr_collapse_grid(
            runs, snr_metrics, grid_out_c,
            water_runs=water_runs,
            suptitle="All SNR metrics  vs  σ·v³  —  phantom (solid) vs water (dashed)",
        )
        print(f"Wrote {grid_out_c}")

        make_mape_vs_snr(runs, "snr_dual_peak", "snr_dual_peak (pooled)",
                         cdir / "mape_vs_snr_dual_combined.png",
                         water_runs=water_runs,
                         title_suffix="  [phantom vs water]")
        print(f"Wrote {cdir / 'mape_vs_snr_dual_combined.png'}")
        make_mape_vs_snr(runs, "snr_nema_peak_a", "snr_nema_peak_a",
                         cdir / "mape_vs_snr_nema_combined.png",
                         water_runs=water_runs,
                         inflection_hint=None,
                         title_suffix="  [phantom vs water]")
        print(f"Wrote {cdir / 'mape_vs_snr_nema_combined.png'}")

    # ── (d) Collapse-quality tables ───────────────────────────────────────────
    print_collapse_table(runs, "snr_dual_peak",    "snr_dual_peak")
    print_collapse_table(runs, "snr_nema_peak_a",  "snr_nema_peak_a")
    print_collapse_table(runs, "snr_ksp_measured", "snr_ksp_measured")
    if water_runs:
        print("\n  === Water phantom collapse tables ===")
        print_collapse_table(water_runs, "snr_dual_peak",    "snr_dual_peak [water]")
        print_collapse_table(water_runs, "snr_nema_peak_a",  "snr_nema_peak_a [water]")
        print_collapse_table(water_runs, "snr_ksp_measured", "snr_ksp_measured [water]")


if __name__ == "__main__":
    main()

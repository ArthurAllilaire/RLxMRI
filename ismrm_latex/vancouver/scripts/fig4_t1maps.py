"""Fig 4: T1 maps of one scan under 560 s protocols, beside the ground truth.

Reads the replays from fig4_replay.jl and the per-pixel fits from fig4_pixelfit.jl.

Validation version (default): checks the replay of all 24 validation scans against the
recorded training-time diagnostics, picks the most representative scan and draws
  fig4_t1maps      per-pixel T1 fit (the map a scanner user would see)
  fig4_t1maps_roi  each sphere painted with the scored 3x3-pixel ROI estimate
Held-out version (--heldout K): one held-out scan with the TI-histogram policy, the fixed
protocol distilled from it, the fixed log TI grid and Cramer-Rao; draws
  fig4_t1maps_pK          per-pixel T1 fit and per-pixel difference
  fig4_t1maps_pK_roidiff  per-pixel T1 fit, difference column showing the scored ROI error
With --best-grid, the log TI grid row is best_log_grid_schedule (fig4_replay.jl --best-grid)
and the held-out figures are saved with a _v2 suffix.

    .venv/bin/python ismrm_latex/vancouver/scripts/fig4_t1maps.py [--scan K]
    .venv/bin/python ismrm_latex/vancouver/scripts/fig4_t1maps.py --heldout 7 [--best-grid]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize
from matplotlib.patches import Circle, Rectangle

import figstyle

DATA = figstyle.FIGURES / "fig4_data"
RUNS = figstyle.ROOT / "runs" / "e2"
VALIDATION_METHODS = ["hist", "nomem", "log_grid", "cr_optimal"]
HELDOUT_METHODS = ["hist", "distilled", "log_grid", "cr_optimal"]
RL_METHODS = ["hist", "nomem"]
# The distilled protocol is not in figstyle yet; it borrows the other-RL token because it is the
# policy's schedule frozen, and the label used by the Fig 5 v2 table.
LABELS = {**figstyle.METHOD_LABELS, "distilled": "Fixed, distilled\nfrom RL policy"}
COLOURS = {**figstyle.METHOD_COLOURS, "distilled": figstyle.RL_OTHER}
N_SCANS = 24
TILE_HALF_MM = 13.0
# A pixel is shown as sphere when at least half of its spins belong to that sphere;
# mixed pixels hold water (T1 2.8 s), so their "true T1" has no single value.
SPHERE_PIXEL_FRACTION = 0.5
WATER_GREY = "#C4C4C4"
CM = 1 / 2.54


@dataclass(frozen=True)
class Scales:
    t1_range_s: tuple[float, float]
    t1_log: bool
    t1_ticks: list[float]
    t1_extend: str
    diff_lim: float
    diff_tick_step: float


@dataclass(frozen=True)
class Variant:
    maps: str  # "pixel": per-pixel fit; "roi": sphere painted with the ROI fit
    diffs: str
    diff_title: str
    number_key: str  # {roi} names the scored ROI fit


VARIANTS = {
    "pixel": Variant("pixel", "pixel", "Relative difference from truth",
                     "Number: {roi}'s T1 error (the scored value)."),
    "roi": Variant("roi", "roi", "Relative difference from truth",
                   "Disc colour and number: {roi} (the scored value)."),
    # Single-pixel noise (5-10%) hides the scored per-sphere differences in the pixel column.
    "pixel_roidiff": Variant("pixel", "roi", "Scored error (ROI fit) vs truth",
                             "Right disc colour and number: {roi}'s T1 error."),
}


LINEAR_T1 = {"t1_range_s": figstyle.T1_NORM, "t1_log": False, "t1_ticks": [0, 0.5, 1.0, 1.5, 2.0],
             "t1_extend": "max"}
# Validation scan: pixel noise and partial-volume edges reach 25%, ROI errors stay within 15%.
VALIDATION_SCALES = {
    "pixel": Scales(**LINEAR_T1, diff_lim=0.25, diff_tick_step=0.10),
    "roi": Scales(**LINEAR_T1, diff_lim=0.15, diff_tick_step=0.05),
}
# Held-out phantom 7 spans 41 ms to 1.8 s, and the plate's nominal T1s are log-spaced: on a linear
# axis every T1 under 0.1 s is the same purple, so the distilled protocol's 41 ms miss is invisible.
# At +/-10% single-pixel noise already saturates most pixels; +/-15% keeps those pale.
HELDOUT_SCALES = Scales(t1_range_s=(0.03, 3.0), t1_log=True, t1_ticks=[0.03, 0.1, 0.3, 1.0, 3.0],
                        t1_extend="both", diff_lim=0.15, diff_tick_step=0.05)


def load(tag: str) -> tuple[dict, dict]:
    with np.load(DATA / f"fig4_replay_{tag}.npz") as f:
        r = {key: f[key] for key in f.files}
    with np.load(DATA / f"fig4_maps_{tag}.npz") as f:
        mp = {key: f[key] for key in f.files}
    return r, mp


def replay_check(replays: list[dict]) -> dict:
    """Replayed (CPU) against recorded (GPU) diagnostics for the RL arms, all blocks of all scans."""
    out = {}
    for m in RL_METHODS:
        d_mape = np.concatenate([np.abs(r[f"{m}_mape"] - r[f"{m}_mape_recorded"]) for r in replays])
        d_t1 = np.concatenate([np.abs(np.nanmean(r[f"{m}_T1_est"], axis=1) - r[f"{m}_T1_est_mean_recorded"])
                               for r in replays])
        d_snr = np.concatenate([np.abs(r[f"{m}_snr_nema_peak"] / r[f"{m}_snr_nema_peak_recorded"] - 1)
                                for r in replays])
        final_equal = sum(r[f"{m}_mape"][-1] == r[f"{m}_mape_recorded"][-1] for r in replays)
        out[m] = {
            "n_blocks": int(d_mape.size),
            "max_abs_diff_block_mape": float(d_mape.max()),
            "max_abs_diff_block_T1_est_mean_s": float(d_t1.max()),
            "max_rel_diff_block_snr": float(d_snr.max()),
            "scans_with_identical_final_mape": int(final_equal),
        }
    recorded = json.loads((RUNS / "baseline_5sphere_560s_gpu" / "baseline_summary.json").read_text())
    out["cr_optimal"] = {
        "replayed_mean_mape_pct": 100 * float(np.mean([r["cr_optimal_mape"][-1] for r in replays])),
        "recorded_mean_mape_pct": recorded["cr_optimal"]["mape_pct"],
        "recorded_file": "runs/e2/baseline_5sphere_560s_gpu/baseline_summary.json",
    }
    return out


def choose_scan(final: dict[str, np.ndarray]) -> tuple[int, dict]:
    """Among scans with the histogram policy's MAPE inside its interquartile range, the one whose
    four MAPEs sit closest to each method's own median (sum of |log ratio|)."""
    med = {m: float(np.median(v)) for m, v in final.items()}
    q1, q3 = np.percentile(final["hist"], [25, 75])
    dev = sum(np.abs(np.log(final[m] / med[m])) for m in VALIDATION_METHODS)
    eligible = np.flatnonzero((final["hist"] >= q1) & (final["hist"] <= q3))
    k = int(eligible[np.argmin(dev[eligible])])
    ranking = sorted(eligible.tolist(), key=lambda j: dev[j])
    return k, {"median_mape_pct": {m: 100 * v for m, v in med.items()},
               "hist_iqr_pct": [100 * q1, 100 * q3],
               "deviation_score": {int(j): float(dev[j]) for j in ranking}}


def per_method_summary(r: dict, mp: dict, methods: list[str]) -> dict:
    t1 = r["T1_true"]
    npe = r["truth_T1"].shape[0]
    return {m: {
        "mape_pct": 100 * float(r[f"{m}_mape"][-1]),
        "n_blocks": len(r[f"{m}_mape"]),
        "scan_time_s": float(npe * r[f"{m}_actions"][:, 2].sum()),
        "min_TI_s": float(r[f"{m}_actions"][:, 0].min()),
        "roi_T1_s": mp[f"{m}_roi_T1"].tolist(),
        "roi_error_pct": (100 * (mp[f"{m}_roi_T1"] - t1) / t1).tolist(),
    } for m in methods}


def scan_geometry(r: dict) -> dict:
    npe, nfe = r["truth_T1"].shape
    fov_mm = 1e3 * float(r["fov_m"])
    dx, dy = fov_mm / nfe, fov_mm / npe
    # Pixel centre (0-based j) at (j - N/2) * d, matching MRISystemPhantom.phys_to_pixel.
    extent = [(-nfe / 2 - 0.5) * dx, (nfe / 2 - 0.5) * dx, (-npe / 2 - 0.5) * dy, (npe / 2 - 0.5) * dy]
    rr = int(r["roi_radius"])
    roi = []
    for ipe, ife in r["sphere_px"]:
        xc, yc = (ife - 1 - nfe / 2) * dx, (ipe - 1 - npe / 2) * dy
        roi.append((xc - (rr + 0.5) * dx, yc - (rr + 0.5) * dy, (2 * rr + 1) * dx, (2 * rr + 1) * dy))
    return {"extent": extent, "roi": roi, "roi_size": 2 * rr + 1}


def sphere_masks(r: dict) -> np.ndarray:
    fr = r["sphere_fraction"]
    pure = fr >= 1.0
    for i, t1 in enumerate(r["T1_true"]):
        # On pixels holding only sphere i, phantom_parameter_map's truth must be that sphere's T1.
        assert np.allclose(r["truth_T1"][pure[i]], t1), f"truth map disagrees with sphere {i}"
    return fr >= SPHERE_PIXEL_FRACTION


def signed_pct(x: float, decimals: int = 1) -> str:
    return f"{100 * x:+.{decimals}f}%".replace("-", "−")


def t1_label(t1_s: float) -> str:
    return f"{1e3 * t1_s:.0f} ms" if t1_s < 0.1 else f"{t1_s:.2f} s"


def draw(r: dict, mp: dict, methods: list[str], variant: Variant, scales: Scales,
         roi_box: bool = True, labels: dict[str, str] = LABELS) -> plt.Figure:
    geo = scan_geometry(r)
    masks = sphere_masks(r)
    t1_true = r["T1_true"]
    order = np.argsort(t1_true)
    centres = r["sphere_centre_mm"]
    radius = float(r["sphere_radius_mm"])
    t1_cmap = plt.get_cmap(figstyle.T1_CMAP).copy()
    diff_cmap = plt.get_cmap(figstyle.DIFF_CMAP).copy()
    for c in (t1_cmap, diff_cmap):
        c.set_bad(WATER_GREY)
    # A fresh norm per figure: a shared one collects colourbar callbacks from closed figures.
    t1_norm = (LogNorm if scales.t1_log else Normalize)(*scales.t1_range_s)
    diff_norm = Normalize(-scales.diff_lim, scales.diff_lim)

    W = 17.0 * CM
    label_w, tile, gap, block_gap = 2.75 * CM, 1.31 * CM, 0.1 * CM, 0.55 * CM
    top, row_gap, num_h, bottom = 0.95 * CM, 0.08 * CM, 0.36 * CM, 1.25 * CM
    row_h = tile + num_h + row_gap
    n_rows = 1 + len(methods)
    H = top + n_rows * row_h + bottom
    fig = plt.figure(figsize=(W, H))

    def ax_at(x_in: float, y_top_in: float, w_in: float, h_in: float) -> plt.Axes:
        return fig.add_axes([x_in / W, 1 - (y_top_in + h_in) / H, w_in / W, h_in / H])

    left0 = label_w
    right0 = label_w + 5 * tile + 4 * gap + block_gap

    def tile_ax(block_x: float, row: int, col: int) -> plt.Axes:
        ax = ax_at(block_x + col * (tile + gap), top + row * row_h, tile, tile)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_facecolor(WATER_GREY)
        return ax

    def show(ax: plt.Axes, i: int, kind: str, values, cmap, norm) -> None:
        x0, y0 = centres[i]
        if kind == "pixel":
            ax.imshow(np.ma.masked_invalid(np.where(masks[i], values, np.nan)), origin="lower",
                      extent=geo["extent"], cmap=cmap, norm=norm, interpolation="nearest")
        else:
            ax.add_patch(Circle((x0, y0), radius, facecolor=cmap(norm(values)), edgecolor="none"))
        ax.add_patch(Circle((x0, y0), radius, fill=False, ec=figstyle.INK, lw=0.6))
        ax.set_xlim(x0 - TILE_HALF_MM, x0 + TILE_HALF_MM)
        ax.set_ylim(y0 - TILE_HALF_MM, y0 + TILE_HALF_MM)
        ax.set_aspect("equal")

    rows = [("truth", "Ground truth")] + [(m, labels[m]) for m in methods]
    for row, (m, label) in enumerate(rows):
        y_mid = top + row * row_h + tile / 2
        if m != "truth":
            mape = r[f"{m}_mape"][-1]
            fig.text(0.32 * CM / W, 1 - y_mid / H, f"{label}\nMAPE {100 * mape:.1f}%", fontsize=7.5,
                     va="center", ha="left", linespacing=1.35)
            fig.add_artist(Rectangle((0.08 * CM / W, 1 - (y_mid + 0.32 * CM) / H), 0.11 * CM / W,
                                     0.64 * CM / H, transform=fig.transFigure, color=COLOURS[m], lw=0))
        else:
            fig.text(0.32 * CM / W, 1 - y_mid / H, label, fontsize=7.5, va="center", ha="left")

        for col, i in enumerate(order):
            ax = tile_ax(left0, row, col)
            if m == "truth":
                show(ax, i, variant.maps,
                     np.full(masks[i].shape, t1_true[i]) if variant.maps == "pixel" else t1_true[i],
                     t1_cmap, t1_norm)
                if roi_box:
                    ax.add_patch(Rectangle(geo["roi"][i][:2], *geo["roi"][i][2:], fill=False,
                                           ec=figstyle.INK, lw=0.6, ls=(0, (2, 1.5))))
                ax.set_title(t1_label(t1_true[i]), fontsize=7, pad=2.5, color=figstyle.INK_SECONDARY)
                continue
            roi_est = mp[f"{m}_roi_T1"][i]
            roi_err = (roi_est - t1_true[i]) / t1_true[i]
            pixel_map = mp[f"{m}_T1_map"]
            show(ax, i, variant.maps, pixel_map if variant.maps == "pixel" else roi_est, t1_cmap, t1_norm)
            dax = tile_ax(right0, row, col)
            if row == 1:
                dax.set_title(t1_label(t1_true[i]), fontsize=7, pad=2.5, color=figstyle.INK_SECONDARY)
            show(dax, i, variant.diffs,
                 (pixel_map - t1_true[i]) / t1_true[i] if variant.diffs == "pixel" else roi_err,
                 diff_cmap, diff_norm)
            dax.text(0.5, -0.06, signed_pct(roi_err), transform=dax.transAxes, ha="center", va="top",
                     fontsize=7, color=figstyle.INK)

    block_w = 5 * tile + 4 * gap
    left_title = "T1, per-pixel fit" if variant.maps == "pixel" else "T1, scored ROI estimate per sphere"
    fig.text((left0 + block_w / 2) / W, 1 - 0.22 * CM / H, left_title, ha="center", va="center",
             fontsize=8.5, fontweight="bold")
    fig.text((right0 + block_w / 2) / W, 1 - 0.22 * CM / H, variant.diff_title,
             ha="center", va="center", fontsize=8.5, fontweight="bold")

    key = ax_at(right0, top, block_w, tile)
    key.axis("off")
    n_roi = geo["roi_size"]
    roi_name = "that ROI fit" if roi_box else f"the {n_roi}×{n_roi}-pixel ROI fit"
    key_lines = [
        f"Circle: sphere edge ({2 * radius:g} mm); header: its true T1.",
        *([f"Dashed: {n_roi}×{n_roi}-pixel ROI whose mean signal is fitted."] if roi_box else []),
        variant.number_key.format(roi=roi_name),
        f"Grey: background water (T1 {float(r['water_T1_s']):g} s)"
        + (", masked." if variant.maps == "pixel" else "."),
    ]
    key.text(0.0, 0.62, "\n".join(key_lines), fontsize=7, va="center", ha="left", linespacing=1.4,
             color=figstyle.INK_SECONDARY, transform=key.transAxes)

    cb_y = top + n_rows * row_h + 0.12 * CM
    n_steps = int(round(scales.diff_lim / scales.diff_tick_step, 6))
    diff_ticks = scales.diff_tick_step * np.arange(-n_steps, n_steps + 1)
    for x0, cmap, norm, label, extend, ticks, tick_labels in (
        (left0, t1_cmap, t1_norm, "T1 (s)", scales.t1_extend, scales.t1_ticks,
         [f"{t:g}" for t in scales.t1_ticks]),
        (right0, diff_cmap, diff_norm, "T1 error", "both", diff_ticks,
         [signed_pct(t, 0) if abs(t) > 1e-9 else "0%" for t in diff_ticks]),
    ):
        cax = ax_at(x0 + 0.1 * block_w, cb_y, 0.8 * block_w, 0.2 * CM)
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal",
                          extend=extend)
        cb.minorticks_off()
        cb.set_ticks(ticks)
        cb.set_ticklabels(tick_labels)
        cb.ax.tick_params(labelsize=7, length=2, pad=1.5)
        cb.outline.set_linewidth(0.4)
        cb.set_label(label, fontsize=7.5, labelpad=2)
    return fig


def main_validation(scan: int | None) -> None:
    pairs = [load(f"k{k:02d}") for k in range(N_SCANS)]
    replays = [p[0] for p in pairs]
    final = {m: np.array([r[f"{m}_mape"][-1] for r in replays]) for m in VALIDATION_METHODS}
    k_auto, selection = choose_scan(final)
    k = k_auto if scan is None else scan
    r, mp = pairs[k]
    summary = {
        "scan": k,
        "seed": int(r["seed"]),
        "automatic_choice": k_auto,
        "selection": selection,
        "replay_check": replay_check(replays),
        "final_mape_pct_per_scan": {m: (100 * v).tolist() for m, v in final.items()},
        "chosen_scan": {
            "T1_true_s": r["T1_true"].tolist(),
            "sphere_indices": r["sphere_indices"].tolist(),
            "per_method": per_method_summary(r, mp, VALIDATION_METHODS),
        },
    }
    (DATA / "fig4_summary.json").write_text(json.dumps(summary, indent=1))
    for name, stem in (("pixel", "fig4_t1maps"), ("roi", "fig4_t1maps_roi")):
        fig = draw(r, mp, VALIDATION_METHODS, VARIANTS[name], VALIDATION_SCALES[name])
        figstyle.save(fig, stem)
        plt.close(fig)
    print(json.dumps({key: summary[key] for key in ("scan", "automatic_choice", "replay_check")}, indent=1))
    print(json.dumps(summary["chosen_scan"]["per_method"], indent=1))


def main_heldout(k: int, best_grid: bool) -> None:
    tag = f"heldout_k{k:02d}" + ("_bestgrid" if best_grid else "")
    r, mp = load(tag)
    labels = LABELS
    if best_grid:
        grid_tr = np.unique(r["log_grid_actions"][:, 2])
        assert grid_tr.size == 1, f"the log TI grid ran at several TRs: {grid_tr}"
        labels = {**LABELS, "log_grid": f"{LABELS['log_grid']}\nTR {grid_tr[0]:.1f} s"}
    # fig4_replay.jl stops unless each replay reproduces its recorded final T1 estimates.
    check = {m: {"replayed_T1_est_s": r[f"{m}_T1_est"][-1].tolist(),
                 "recorded_T1_est_s": r[f"{m}_final_T1_est_recorded"].tolist(),
                 "identical": bool(np.array_equal(r[f"{m}_T1_est"][-1], r[f"{m}_final_T1_est_recorded"]))}
             for m in HELDOUT_METHODS}
    summary = {
        "scan": k,
        "seed": int(r["seed"]),
        "T1_true_s": r["T1_true"].tolist(),
        "replay_check": check,
        "per_method": per_method_summary(r, mp, HELDOUT_METHODS),
    }
    (DATA / f"fig4_summary_{tag}.json").write_text(json.dumps(summary, indent=1))
    for name, suffix in (("pixel", ""), ("pixel_roidiff", "_roidiff")):
        fig = draw(r, mp, HELDOUT_METHODS, VARIANTS[name], HELDOUT_SCALES, roi_box=False, labels=labels)
        figstyle.save(fig, f"fig4_t1maps_p{k}{suffix}" + ("_v2" if best_grid else ""))
        plt.close(fig)
    print(json.dumps(summary, indent=1))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scan", type=int, default=None, help="override the automatic validation-scan choice")
    p.add_argument("--heldout", type=int, default=None, help="draw this held-out scan instead")
    p.add_argument("--best-grid", action="store_true",
                   help="with --heldout: the log TI grid row is best_log_grid_schedule (fig4_replay.jl --best-grid)")
    args = p.parse_args()
    if args.best_grid and args.heldout is None:
        p.error("--best-grid needs --heldout")
    figstyle.apply()
    if args.heldout is None:
        main_validation(args.scan)
    else:
        main_heldout(args.heldout, args.best_grid)


if __name__ == "__main__":
    main()

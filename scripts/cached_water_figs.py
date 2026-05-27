#!/usr/bin/env python3
"""Render report figures for a cached_water_validation run folder.

Reads scripts/runs/cached_water_e2/<label>/{config.json,arrays/relerr.csv,
arrays/t1fits.csv} and writes figures/ PNGs:
  relerr_vs_TI.png  cache fidelity (cached vs Bloch water-B0=0) + the water-B0
                    modelling cost (water-B0=0 vs water-B0=5), one curve per α.
  t1_fit.png        cached vs ground-truth T1 fits across α (1:1 scatter) +
                    per-sphere |ΔT1| bars; annotated with grid-floor agreement.
  speedup.png       per-step full-Bloch vs cached, from config timing.

Usage:
  python scripts/cached_water_figs.py --label npe32fe64_v1p0
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent / "runs" / "cached_water_e2"


def read_relerr(p):
    rows = defaultdict(list)  # (variant,target) -> list of (alpha,TI,ksp,img)
    with open(p) as f:
        for r in csv.DictReader(f):
            rows[(r["variant"], r["target"])].append(
                (float(r["alpha_deg"]), float(r["TI_s"]),
                 float(r["ksp_relerr"]), float(r["img_relerr"])))
    return rows


def read_fits(p):
    rows = []
    with open(p) as f:
        for r in csv.DictReader(f):
            rows.append({k: (float(v) if k not in ("variant", "label") else v)
                         for k, v in r.items()})
    return rows


def _curve(ax, data, alphas, title, ylab):
    for a in alphas:
        pts = sorted([(ti, kr) for (al, ti, kr, ir) in data if al == a])
        if pts:
            ti, kr = zip(*pts)
            ax.semilogy(ti, kr, "-o", ms=3, label=f"α={a:g}°")
    ax.set_xlabel("TI [s]"); ax.set_ylabel(ylab); ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=7)


def fig_relerr(rel, cfg, out):
    alphas = sorted(cfg["alphas_deg"])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    _curve(ax[0], rel[("water_cached", "water0")], alphas,
           "Cache fidelity: cached vs Bloch water (B0σ=0)", "k-space rel. error")
    _curve(ax[1], rel[("water0", "water5")], alphas,
           "Modelling cost: water B0σ=0 vs B0σ=5 Hz", "k-space rel. error")
    fig.suptitle(f"Cached-water validation — {cfg['label']}  "
                 f"({cfg['Npe']}×{cfg['Nfe']}, {cfg['voxel_size_mm']} mm, "
                 f"{cfg['n_grid']} α-bank)")
    fig.tight_layout()
    fig.savefig(out / "relerr_vs_TI.png", dpi=130); plt.close(fig)


def fig_t1(fits, cfg, out):
    truth = {(r["alpha_deg"], r["label"]): r["T1_fit_s"]
             for r in fits if r["variant"] == "koma_water0"}
    cached = [r for r in fits if r["variant"] == "cached"]
    alphas = sorted(cfg["alphas_deg"])
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(alphas)))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    devs = []
    for a, c in zip(alphas, cmap):
        xs = [truth[(a, r["label"])] for r in cached if r["alpha_deg"] == a]
        ys = [r["T1_fit_s"] for r in cached if r["alpha_deg"] == a]
        ax[0].scatter(xs, ys, s=18, color=c, label=f"α={a:g}°")
        devs += [r["rel_vs_truth_pct"] for r in cached if r["alpha_deg"] == a]
    lim = [0, max(max(truth.values()), 0.1) * 1.05]
    ax[0].plot(lim, lim, "k--", lw=1, alpha=0.6)
    ax[0].set_xlim(lim); ax[0].set_ylim(lim)
    ax[0].set_xlabel("ground-truth T1 fit [s] (Bloch, water B0σ=0)")
    ax[0].set_ylabel("cached T1 fit [s]")
    ax[0].set_title("Cached vs ground-truth T1 fit")
    ax[0].grid(True, alpha=0.3); ax[0].legend(fontsize=7)

    labels = sorted({r["label"] for r in cached},
                    key=lambda s: -truth[(alphas[0], s)])
    x = np.arange(len(labels)); w = 0.8 / len(alphas)
    for j, (a, c) in enumerate(zip(alphas, cmap)):
        d = {r["label"]: r["rel_vs_truth_pct"] for r in cached if r["alpha_deg"] == a}
        ax[1].bar(x + j * w, [d.get(l, 0) for l in labels], w, color=c, label=f"α={a:g}°")
    ax[1].set_xticks(x + 0.4); ax[1].set_xticklabels(labels, rotation=90, fontsize=6)
    ax[1].set_ylabel("|ΔT1| vs ground truth [%]")
    ax[1].set_title(f"Per-sphere deviation (mean {np.mean(devs):.3f}%, "
                    f"max {np.max(devs):.2f}% — grid floor)")
    ax[1].grid(True, axis="y", alpha=0.3); ax[1].legend(fontsize=7)
    fig.suptitle(f"Cached-water T1 fits — {cfg['label']}")
    fig.tight_layout()
    fig.savefig(out / "t1_fit.png", dpi=130); plt.close(fig)


def fig_speedup(cfg, out):
    t = cfg["timing_s"]
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    bars = ax.bar(["full Bloch\n(dry+water)", "cached\n(dry+rescale)"],
                  [t["full"], t["cached"]], color=["#888", "#d77757"])
    for b, v in zip(bars, [t["full"], t["cached"]]):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}s",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("per-step sim time [s]")
    ax.set_title(f"Per-step speedup {t['per_step_speedup']:.1f}×\n"
                 f"(α-bank build {t['bank_build']:.0f}s once)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "speedup.png", dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    args = ap.parse_args()
    rd = ROOT / args.label
    cfg = json.loads((rd / "config.json").read_text())
    out = rd / "figures"; out.mkdir(exist_ok=True)
    rel = read_relerr(rd / "arrays" / "relerr.csv")
    fits = read_fits(rd / "arrays" / "t1fits.csv")
    fig_relerr(rel, cfg, out)
    fig_t1(fits, cfg, out)
    fig_speedup(cfg, out)
    print(f"wrote figures to {out}")


if __name__ == "__main__":
    main()

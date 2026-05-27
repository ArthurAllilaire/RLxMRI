#!/usr/bin/env python
"""Render figures for ONE hybrid-water run folder produced by hybrid_water_run.jl.

Reads  runs/hybrid_water/<label>/{config.json, arrays/*}
Writes runs/hybrid_water/<label>/figures/*.png   (no PDFs)

Figures:
  images_4up_TI<tag>.png     4 image variants + diffs vs koma (at one TI block)
  kspace_4up_TI<tag>.png     4 log|k-space| + signed |k| diffs vs koma
  water_theory_vs_cache.png  water image & k-space: theory | cache | diff
  t1_fit_4variants.png       T1 scatter (4 variants) + per-sphere |dT1|/koma bars
  relerr_vs_TI.png           per-block relerr(TI) per variant (this run)
  cost_speedup.png           per-step sim cost bar + speedup annotation

Usage:
  python scripts/hybrid_water_figs.py --label a90p0_b00p0_npe32fe64
  python scripts/hybrid_water_figs.py --label <label> --ti 0.5
"""
import argparse, csv, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "runs", "hybrid_water")
TITLES = {"koma_full": "Koma full (truth)", "hybrid_cached": "Koma sph + cached (scalar)",
          "hybrid_cached_perline": "Koma sph + cached (per-line)",
          "hybrid_analytic": "Koma sph + analytic water", "theory_full": "Theory full"}


def title(v):
    return TITLES.get(v, v)


def load(label):
    rd = os.path.join(ROOT, label)
    ad = os.path.join(rd, "arrays")
    with open(os.path.join(rd, "config.json")) as f:
        cfg = json.load(f)
    imgv = cfg.get("img_variants", ["koma_full", "hybrid_cached", "hybrid_analytic", "theory_full"])
    watv = cfg.get("water_variants", ["water_koma", "water_cache", "water_theory"])
    A = lambda n: np.load(os.path.join(ad, n))
    arr = {v: {"ksp": A(f"{v}_ksp.npy"), "img": A(f"{v}_img.npy")} for v in imgv + watv}
    rel = list(csv.DictReader(open(os.path.join(ad, "relerr.csv"))))
    t1 = list(csv.DictReader(open(os.path.join(ad, "t1fits.csv"))))
    TIs = A("TIs.npy")
    return rd, cfg, arr, rel, t1, TIs, imgv, watv


def figdir(rd):
    d = os.path.join(rd, "figures"); os.makedirs(d, exist_ok=True); return d


def logmag(z, floor=1e-4):
    a = np.abs(z)
    return np.log10(np.maximum(a, a.max() * floor))


def images_4up(rd, arr, TIs, bi, tag, variants):
    n = len(variants)
    fig, ax = plt.subplots(2, n, figsize=(4 * n, 8))
    vmax = max(np.abs(arr[v]["img"][bi]).max() for v in variants)
    for j, v in enumerate(variants):
        im = ax[0, j].imshow(np.abs(arr[v]["img"][bi]), vmin=0, vmax=vmax, cmap="magma")
        ax[0, j].set_title(title(v), fontsize=9); plt.colorbar(im, ax=ax[0, j], fraction=0.046)
    ref = np.abs(arr["koma_full"]["img"][bi])
    others = [v for v in variants if v != "koma_full"]
    ax[1, 0].axis("off"); ax[1, 0].text(0.5, 0.5, "diff vs Koma full\n(signed)", ha="center", va="center")
    dmax = max((np.abs(arr[v]["img"][bi]) - ref).__abs__().max() for v in others) or 1.0
    for j, v in enumerate(others, start=1):
        d = np.abs(arr[v]["img"][bi]) - ref
        im = ax[1, j].imshow(d, vmin=-dmax, vmax=dmax, cmap="bwr")
        ax[1, j].set_title(f"{v} − koma", fontsize=9); plt.colorbar(im, ax=ax[1, j], fraction=0.046)
    fig.suptitle(f"Image variants  TI={TIs[bi]:.3f} s")
    fig.tight_layout(); fig.savefig(os.path.join(figdir(rd), f"images_4up_TI{tag}.png"), dpi=110)
    plt.close(fig)


def kspace_4up(rd, arr, TIs, bi, tag, variants):
    n = len(variants)
    fig, ax = plt.subplots(2, n, figsize=(4 * n, 8))
    for j, v in enumerate(variants):
        im = ax[0, j].imshow(logmag(arr[v]["ksp"][bi]), cmap="magma")
        ax[0, j].set_title(f"log|k|  {v}", fontsize=9); plt.colorbar(im, ax=ax[0, j], fraction=0.046)
    ref = np.abs(arr["koma_full"]["ksp"][bi])
    others = [v for v in variants if v != "koma_full"]
    ax[1, 0].axis("off"); ax[1, 0].text(0.5, 0.5, "|k| diff vs Koma full\n(signed)", ha="center", va="center")
    dmax = max((np.abs(arr[v]["ksp"][bi]) - ref).__abs__().max() for v in others) or 1.0
    for j, v in enumerate(others, start=1):
        d = np.abs(arr[v]["ksp"][bi]) - ref
        im = ax[1, j].imshow(d, vmin=-dmax, vmax=dmax, cmap="bwr")
        ax[1, j].set_title(f"{v} − koma", fontsize=9); plt.colorbar(im, ax=ax[1, j], fraction=0.046)
    fig.suptitle(f"k-space variants  TI={TIs[bi]:.3f} s")
    fig.tight_layout(); fig.savefig(os.path.join(figdir(rd), f"kspace_4up_TI{tag}.png"), dpi=110)
    plt.close(fig)


def water_fig(rd, arr, TIs, bi, tag):
    th_i, ca_i = np.abs(arr["water_theory"]["img"][bi]), np.abs(arr["water_cache"]["img"][bi])
    th_k, ca_k = arr["water_theory"]["ksp"][bi], arr["water_cache"]["ksp"][bi]
    fig, ax = plt.subplots(2, 3, figsize=(13, 8))
    vmax = max(th_i.max(), ca_i.max())
    for j, (img, t) in enumerate([(th_i, "water image: theory"), (ca_i, "water image: cached")]):
        im = ax[0, j].imshow(img, vmin=0, vmax=vmax, cmap="magma"); ax[0, j].set_title(t)
        plt.colorbar(im, ax=ax[0, j], fraction=0.046)
    d = ca_i - th_i; dm = np.abs(d).max() or 1.0
    im = ax[0, 2].imshow(d, vmin=-dm, vmax=dm, cmap="bwr"); ax[0, 2].set_title("image diff (cache−theory)")
    plt.colorbar(im, ax=ax[0, 2], fraction=0.046)
    for j, (k, t) in enumerate([(th_k, "water log|k|: theory"), (ca_k, "water log|k|: cached")]):
        im = ax[1, j].imshow(logmag(k), cmap="magma"); ax[1, j].set_title(t)
        plt.colorbar(im, ax=ax[1, j], fraction=0.046)
    dk = np.abs(ca_k) - np.abs(th_k); dkm = np.abs(dk).max() or 1.0
    im = ax[1, 2].imshow(dk, vmin=-dkm, vmax=dkm, cmap="bwr"); ax[1, 2].set_title("|k| diff (cache−theory)")
    plt.colorbar(im, ax=ax[1, 2], fraction=0.046)
    fig.suptitle(f"Water only: theory vs cached  TI={TIs[bi]:.3f} s")
    fig.tight_layout(); fig.savefig(os.path.join(figdir(rd), "water_theory_vs_cache.png"), dpi=110)
    plt.close(fig)


def t1_fig(rd, t1, variants):
    by = {}
    for r in t1:
        by.setdefault(r["variant"], []).append(r)
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    for v in variants:
        rows = by[v]
        tt = [float(r["T1_true_s"]) for r in rows]; tf = [float(r["T1_fit_s"]) for r in rows]
        ax[0].scatter(tt, tf, label=title(v), s=30, alpha=0.8)
    lim = [min(float(r["T1_true_s"]) for r in t1) * 0.8, max(float(r["T1_true_s"]) for r in t1) * 1.2]
    ax[0].plot(lim, lim, "k--", lw=1); ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("T1 true [s]"); ax[0].set_ylabel("T1 fit [s]"); ax[0].legend(fontsize=8)
    ax[0].set_title("T1 fit vs true")
    labels = [r["label"] for r in by["koma_full"]]
    others = [v for v in variants if v != "koma_full"]
    x = np.arange(len(labels)); w = 0.8 / max(len(others), 1)
    for k, v in enumerate(others):
        rel = [float(r["rel_vs_koma_pct"]) for r in by[v]]
        ax[1].bar(x + (k - (len(others) - 1) / 2) * w, rel, w, label=v)
    ax[1].axhline(1.0, color="g", ls=":", label="1% target")
    ax[1].set_xticks(x); ax[1].set_xticklabels(labels, rotation=90, fontsize=7)
    ax[1].set_ylabel("|ΔT1| / T1_koma [%]"); ax[1].set_yscale("log"); ax[1].legend(fontsize=8)
    ax[1].set_title("Per-sphere T1 deviation from Koma full")
    fig.tight_layout(); fig.savefig(os.path.join(figdir(rd), "t1_fit_4variants.png"), dpi=110)
    plt.close(fig)


def relerr_fig(rd, rel):
    full = [r for r in rel if r["target"] == "koma_full"]
    water = [r for r in rel if r["target"] == "water_koma"]
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for v in sorted({r["variant"] for r in full}):
        rows = sorted([r for r in full if r["variant"] == v], key=lambda r: float(r["TI_s"]))
        ti = [float(r["TI_s"]) for r in rows]
        ax[0].plot(ti, [float(r["ksp_relerr"]) for r in rows], "-o", label=f"{v} ksp")
        ax[0].plot(ti, [float(r["img_relerr"]) for r in rows], "--s", label=f"{v} img", alpha=0.6)
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("TI [s]")
    ax[0].set_ylabel("relerr vs Koma full"); ax[0].legend(fontsize=7); ax[0].set_title("Image-variant relerr vs TI")
    for v in sorted({r["variant"] for r in water}):
        rows = sorted([r for r in water if r["variant"] == v], key=lambda r: float(r["TI_s"]))
        ti = [float(r["TI_s"]) for r in rows]
        ax[1].plot(ti, [float(r["ksp_relerr"]) for r in rows], "-o", label=f"{v} ksp")
    ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlabel("TI [s]")
    ax[1].set_ylabel("relerr vs Koma water"); ax[1].legend(fontsize=7); ax[1].set_title("Water relerr vs TI")
    fig.tight_layout(); fig.savefig(os.path.join(figdir(rd), "relerr_vs_TI.png"), dpi=110)
    plt.close(fig)


def cost_fig(rd, cfg):
    t = cfg["timing_s"]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    bars = {"full sim": t["full"], "spheres-only": t["spheres"],
            "sph + scalar rescale": t["spheres"] + t["cached_rescale_scalar"],
            "sph + per-line rescale": t["spheres"] + t["cached_rescale_perline"]}
    ax.bar(list(bars.keys()), list(bars.values()), color=["#c43", "#37a", "#3a7", "#5a3"])
    ax.set_ylabel("per-step sim time [s]")
    ax.set_title(f"Per-step cost — speedup scalar {t['per_step_speedup_scalar']:.2f}× / "
                 f"per-line {t['per_step_speedup_perline']:.2f}×  (spin ratio {t['spin_ratio']:.2f}×)\n"
                 f"rescale: scalar {t['cached_rescale_scalar']:.1e}s, per-line {t['cached_rescale_perline']:.1e}s; "
                 f"one {t['full']:.2f}s water ref sim amortised/episode")
    for i, val in enumerate(bars.values()):
        ax.text(i, val, f"{val:.2f}s", ha="center", va="bottom")
    plt.xticks(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(rd, "figures", "cost_speedup.png"), dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--ti", type=float, default=0.5, help="representative TI [s] for the 4-up panels")
    a = ap.parse_args()
    rd, cfg, arr, rel, t1, TIs, imgv, watv = load(a.label)
    bi = int(np.argmin(np.abs(TIs - a.ti)))
    tag = str(round(float(TIs[bi]), 3)).replace(".", "p")
    images_4up(rd, arr, TIs, bi, tag, imgv)
    kspace_4up(rd, arr, TIs, bi, tag, imgv)
    water_fig(rd, arr, TIs, bi, tag)
    t1_fig(rd, t1, imgv)
    relerr_fig(rd, rel)
    cost_fig(rd, cfg)
    print(f"wrote figures to {os.path.join(rd, 'figures')}")


if __name__ == "__main__":
    main()

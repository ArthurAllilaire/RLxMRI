#!/usr/bin/env python
"""Stitch TI-vs-relative-error across several hybrid-water runs.

Each run folder (runs/hybrid_water/<label>/) holds arrays/relerr.csv + config.json,
written by hybrid_water_run.jl. This overlays relerr(TI) from multiple runs — e.g.
to compare how the cached-water (or analytic-water) error grows with TI across
different α or B0 settings.

Writes runs/hybrid_water/_stitched/relerr_vs_TI_<variant>.png

Usage:
  # explicit runs
  python scripts/hybrid_water_stitch.py --runs a90p0_b00p0_npe32fe64 a30p0_b00p0_npe32fe64
  # all runs, pick which model + which error metric
  python scripts/hybrid_water_stitch.py --all --variant hybrid_cached --metric ksp_relerr
"""
import argparse, csv, glob, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "runs", "hybrid_water")


def run_label(cfg):
    return f"α={cfg['alpha_deg']:g}° B0={cfg['b0_sigma_Hz']:g}Hz {cfg['Npe']}×{cfg['Nfe']}"


def load_run(label):
    rd = os.path.join(ROOT, label)
    with open(os.path.join(rd, "config.json")) as f:
        cfg = json.load(f)
    rel = list(csv.DictReader(open(os.path.join(rd, "arrays", "relerr.csv"))))
    return cfg, rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None, help="run labels under runs/hybrid_water/")
    ap.add_argument("--all", action="store_true", help="use every run folder found")
    ap.add_argument("--variant", default="hybrid_cached",
                    help="which model's error to plot (hybrid_cached/hybrid_analytic/theory_full/water_cache/water_theory)")
    ap.add_argument("--metric", default="ksp_relerr", choices=["ksp_relerr", "img_relerr"])
    a = ap.parse_args()

    labels = a.runs
    if a.all or not labels:
        labels = sorted(os.path.basename(os.path.dirname(p))
                        for p in glob.glob(os.path.join(ROOT, "*", "config.json")))
    if not labels:
        raise SystemExit(f"no runs found under {ROOT}")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for label in labels:
        cfg, rel = load_run(label)
        rows = sorted([r for r in rel if r["variant"] == a.variant], key=lambda r: float(r["TI_s"]))
        if not rows:
            print(f"  {label}: no rows for variant {a.variant}; skipping"); continue
        ti = [float(r["TI_s"]) for r in rows]
        y = [float(r[a.metric]) for r in rows]
        ax.plot(ti, y, "-o", label=run_label(cfg))
    null = None
    try:
        null = json.load(open(os.path.join(ROOT, labels[0], "config.json")))["T1_water_s"] * 0.693
    except Exception:
        pass
    if null:
        ax.axvline(null, color="grey", ls=":", lw=1, label=f"water null ≈ {null:.2f}s")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("TI [s]"); ax.set_ylabel(f"{a.metric} ({a.variant} vs ground truth)")
    ax.set_title(f"Water-model error vs TI — {a.variant}")
    ax.legend(fontsize=8)
    out = os.path.join(ROOT, "_stitched"); os.makedirs(out, exist_ok=True)
    p = os.path.join(out, f"relerr_vs_TI_{a.variant}_{a.metric}.png")
    fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()

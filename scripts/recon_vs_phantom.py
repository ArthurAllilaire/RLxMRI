#!/usr/bin/env python
"""Render the side-by-side phantom-vs-recon comparison.

Run scripts/recon_vs_phantom.jl first to produce the .txt inputs in
scripts/accurate/.  Writes a .png alongside the data.

Usage:
    python recon_vs_phantom.py                    # accurate (default)
    python recon_vs_phantom.py --mode inaccurate
    python recon_vs_phantom.py --mode both        # 2-row comparison
"""
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))


def load_data(subdir):
    d = os.path.join(here, subdir)
    occ = np.loadtxt(os.path.join(d, "recon_vs_phantom_occupancy.txt"))
    img = np.loadtxt(os.path.join(d, "recon_vs_phantom_image.txt"))
    ps  = np.loadtxt(os.path.join(d, "recon_vs_phantom_pershot.txt"))
    meta_path = os.path.join(d, "recon_vs_phantom_meta.txt")
    if os.path.exists(meta_path):
        TI, TR, Nfe, Npe = np.loadtxt(meta_path)
    else:
        TI, TR, Nfe, Npe = 3.0, 4.0, 64, 16
    return occ, img, ps, TI, TR, Nfe, Npe


def metrics(occ, img):
    occ_n = occ / occ.sum()
    img_n = img / img.sum()
    wape     = np.abs(img_n - occ_n).sum() / occ_n.sum() * 100
    nrmse    = np.sqrt(np.mean((img_n - occ_n) ** 2)) / occ_n.mean() * 100
    in_supp  = img[occ > 0].sum() / img.sum() * 100
    r        = np.corrcoef(occ.ravel(), img.ravel())[0, 1]
    return r, wape, nrmse, in_supp


def plot_row(axes, occ, img, ps, TI, TR, Nfe, Npe, row_label):
    r, wape, nrmse, in_supp = metrics(occ, img)
    extent = [-0.1, 0.1, -0.1, 0.1]
    shots  = np.arange(1, len(ps) + 1)

    im0 = axes[0].imshow(occ, cmap="viridis", origin="lower", extent=extent)
    axes[0].set_title(f"[{row_label}]  phantom occupancy\n{int(occ.sum())} spins on T1 plate")
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, shrink=0.9, label="spins / pixel")

    im1 = axes[1].imshow(img, cmap="viridis", origin="lower", extent=extent)
    axes[1].set_title(
        f"reconstructed |image|\n"
        f"Npe={int(Npe)}, Nfe={int(Nfe)}, TI={TI:g} s, TR={TR:g} s, "
        f"sim time = {TR*Npe:g} s\n"
        f"r={r:.3f}  WAPE={wape:.1f}%  NRMSE={nrmse:.1f}%  on-supp={in_supp:.1f}%",
        fontsize=9,
    )
    axes[1].set_xlabel("x [m]")
    axes[1].set_ylabel("y [m]")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, shrink=0.9)

    dc_label = "roughly constant (fixed)" if row_label == "accurate" else "drift = KomaMRI bug"
    axes[2].bar(shots, ps, color="steelblue", edgecolor="black")
    axes[2].set_title(f"per-shot DC sample\n{dc_label}")
    axes[2].set_xlabel("shot index k")
    axes[2].set_ylabel("|raw.profiles[k].data[Nfe/2, 1]|")
    axes[2].axhline(np.median(ps), color="red", linestyle="--", linewidth=1,
                    label=f"median = {np.median(ps):.0f}")
    axes[2].legend(loc="upper right", fontsize=9)
    axes[2].set_xticks(shots)

    return r, wape, nrmse, in_supp


parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["accurate", "inaccurate", "both"],
                    default="accurate")
args = parser.parse_args()

if args.mode == "both":
    fig, all_axes = plt.subplots(2, 3, figsize=(15, 9),
                                  gridspec_kw=dict(width_ratios=[1, 1, 1.2]))
    for row, subdir in enumerate(["accurate", "inaccurate"]):
        plot_row(all_axes[row], *load_data(subdir), row_label=subdir)
    fig.suptitle(
        "Recon-vs-phantom: 14-sphere T1 plate — KomaMRI bug fix comparison\n"
        "top: fixed KomaMRI   bottom: buggy KomaMRI",
        fontsize=12, y=1.01,
    )
    plt.tight_layout()
    out = os.path.join(here, "recon_vs_phantom_comparison.png")
else:
    occ, img, ps, TI, TR, Nfe, Npe = load_data(args.mode)
    r, wape, nrmse, in_supp = metrics(occ, img)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5),
                              gridspec_kw=dict(width_ratios=[1, 1, 1.2]))
    plot_row(axes, occ, img, ps, TI, TR, Nfe, Npe, row_label=args.mode)
    fig.suptitle(
        f"Recon-vs-phantom: 14-sphere T1 plate  [{args.mode}]   "
        f"(Pearson r = {r:.3f}, WAPE = {wape:.1f}%, "
        f"NRMSE = {nrmse:.1f}%, energy-on-support = {in_supp:.1f}%)",
        fontsize=11, y=1.02,
    )
    print(f"Pearson r            = {r:.3f}")
    print(f"WAPE (L1-normalised) = {wape:.1f} %")
    print(f"NRMSE                = {nrmse:.1f} %")
    print(f"energy-on-support    = {in_supp:.1f} %")
    out = os.path.join(here, args.mode, "recon_vs_phantom.png")

plt.tight_layout()
plt.savefig(out, dpi=130, bbox_inches="tight")
print(f"wrote {out}")

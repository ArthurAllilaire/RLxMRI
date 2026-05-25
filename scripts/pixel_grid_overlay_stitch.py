#!/usr/bin/env python
"""Stitch per-TI overlay PNGs into a single comparison figure.

Picks up every <runs_root>/<base>_TI*_*/pixel_grid_overlay_image*.png produced
by scripts/pixel_grid_overlay.{jl,py} matching the chosen grid, and arranges
them with TIs as columns and (raw, clean) as rows.

Usage:
    python scripts/pixel_grid_overlay_stitch.py --npe 64 --nfe 128 --voxel-mm 1.0
"""
import argparse
import glob
import os
import re
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

here = os.path.dirname(os.path.abspath(__file__))
runs_root = os.path.join(here, "runs", "pixel_grid_overlay")

parser = argparse.ArgumentParser()
parser.add_argument("--npe", type=int, default=64)
parser.add_argument("--nfe", type=int, default=128)
parser.add_argument("--fov", type=float, default=0.2)
parser.add_argument("--voxel-mm", type=float, default=1.0)
parser.add_argument("--out", default=None,
                    help="output PNG path (default: runs_root/stitched_<base>.png)")
args = parser.parse_args()

fov_tag = str(args.fov).replace(".", "p")
vox_tag = str(args.voxel_mm).replace(".", "p")
base = f"npe{args.npe}_nfe{args.nfe}_fov{fov_tag}_vox{vox_tag}mm"

# Find all per-TI dirs. Sort by TI numerically.
dirs = sorted(glob.glob(os.path.join(runs_root, base + "_TI*")))
def ti_of(d):
    m = re.search(r"_TI([0-9p]+)", os.path.basename(d))
    return float(m.group(1).replace("p", ".")) if m else 0.0
dirs = sorted({d for d in dirs if os.path.isdir(d)}, key=ti_of)
if not dirs:
    raise SystemExit(f"no _TI* subdirs under {runs_root} matching {base}")

rows = [("raw FFT image",          "pixel_grid_overlay_image.png"),
        ("Hamming + 2× zero-pad",  "pixel_grid_overlay_image_clean.png"),
        ("k-space (meas vs theo)", "pixel_grid_overlay_kspace.png")]
# Drop rows whose file doesn't exist in any of the discovered dirs.
rows = [(lbl, fn) for (lbl, fn) in rows
        if any(os.path.isfile(os.path.join(d, fn)) for d in dirs)]
nrows = len(rows)
ncols = len(dirs)

# k-space figures are 13:6 → wider. Use 5.5 wide per col which keeps the
# image rows roughly square; k-space row sits comfortably below.
fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 5.0 * nrows),
                         squeeze=False)
for j, d in enumerate(dirs):
    ti = ti_of(d)
    for i, (row_label, fname) in enumerate(rows):
        ax = axes[i, j]
        path = os.path.join(d, fname)
        if os.path.isfile(path):
            ax.imshow(mpimg.imread(path))
        else:
            ax.text(0.5, 0.5, "(missing)", ha="center", va="center",
                    transform=ax.transAxes)
        ax.set_axis_off()
        if i == 0:
            ax.set_title(f"TI = {ti:g} s", fontsize=12)
        if j == 0:
            ax.text(-0.02, 0.5, row_label, rotation=90, va="center", ha="right",
                    transform=ax.transAxes, fontsize=12)

fig.suptitle(f"Pixel-grid overlay  [{base}]   image variants + k-space, across TIs",
             fontsize=14, y=0.995)

out = args.out or os.path.join(runs_root, f"stitched_{base}.png")
plt.tight_layout()
plt.savefig(out, dpi=120, bbox_inches="tight")
print(f"wrote {out}")

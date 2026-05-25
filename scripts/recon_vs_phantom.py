#!/usr/bin/env python
"""Render the phantom-vs-recon + k-space comparison.

Run scripts/recon_vs_phantom.jl first to produce the .npy inputs in
scripts/runs/recon_vs_phantom/<run_label>/.  Writes a .png alongside the data.

Specify the run either via --run <label> or by repeating the same
--nfe/--npe/--fov/--voxel-mm flags used for the Julia script.

Usage:
    python recon_vs_phantom.py --nfe 128 --npe 32 --fov 0.2 --voxel-mm 3.0
    python recon_vs_phantom.py --run npe32_nfe128_fov0p2_vox3p0mm
"""
import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))
runs_root = os.path.join(here, "runs", "recon_vs_phantom")


def build_run_label(npe, nfe, fov, voxel_mm, water=False):
    fov_tag = str(fov).replace(".", "p")
    vox_tag = str(voxel_mm).replace(".", "p")
    water_tag = "_water" if water else ""
    return f"npe{npe}_nfe{nfe}_fov{fov_tag}_vox{vox_tag}mm{water_tag}"


def load_data(run_dir):
    occ = np.load(os.path.join(run_dir, "occupancy.npy"))
    img = np.load(os.path.join(run_dir, "image.npy"))
    ksp = np.load(os.path.join(run_dir, "kspace.npy"))
    with open(os.path.join(run_dir, "config.json")) as f:
        cfg = json.load(f)
    return occ, img, ksp, cfg


def metrics(occ, img):
    occ_n = occ / occ.sum()
    img_n = img / img.sum()
    wape     = np.abs(img_n - occ_n).sum() / occ_n.sum() * 100
    nrmse    = np.sqrt(np.mean((img_n - occ_n) ** 2)) / occ_n.mean() * 100
    in_supp  = img[occ > 0].sum() / img.sum() * 100
    r        = np.corrcoef(occ.ravel(), img.ravel())[0, 1]
    return r, wape, nrmse, in_supp


def magnitude_symmetry_error(ksp):
    """Mean ||S(k)| - |S(-k)|| / mean|S(k)| under grid-B (half-cell-centered)
    pairing: index i ↔ index N-1-i. Magnitude-only because the recon has a
    global complex phase from the RF-axis convention (image is ~42% imaginary
    by energy), which breaks raw Hermitian symmetry without affecting |·|."""
    mag = np.abs(ksp)
    return float(np.abs(mag - mag[::-1, ::-1]).mean() / mag.mean())


parser = argparse.ArgumentParser()
parser.add_argument("--run", default=None,
                    help="run label folder under scripts/runs/recon_vs_phantom/")
parser.add_argument("--nfe", type=int, default=128)
parser.add_argument("--npe", type=int, default=32)
parser.add_argument("--fov", type=float, default=0.2)
parser.add_argument("--voxel-mm", type=float, default=3.0)
parser.add_argument("--water", action="store_true")
args = parser.parse_args()

run_label = args.run or build_run_label(args.npe, args.nfe, args.fov, args.voxel_mm, args.water)
run_dir = os.path.join(runs_root, run_label)
if not os.path.isdir(run_dir):
    raise SystemExit(f"run directory not found: {run_dir}\n"
                     f"Did you run scripts/recon_vs_phantom.jl with matching flags?")

occ, img, ksp_meas, cfg = load_data(run_dir)
r, wape, nrmse, in_supp = metrics(occ, img)

FOV = cfg["FOV_m"]
Nfe, Npe = int(cfg["Nfe"]), int(cfg["Npe"])
TI, TR   = cfg["TI_s"], cfg["TR_s"]
extent_img = [-FOV/2, FOV/2, -FOV/2, FOV/2]

# Theoretical k-space = FFT of occupancy (geometry only, no contrast).
ksp_theory = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(occ)))

# log-magnitude for display; clip floor relative to peak
def log_mag(z):
    m = np.abs(z)
    peak = m.max()
    floor = peak * 1e-4
    return np.log10(np.maximum(m, floor))

lm_meas   = log_mag(ksp_meas)
lm_theory = log_mag(ksp_theory)

# k-space axis extents in 1/m: kx ∈ [-Nfe/2, Nfe/2)/FOV, ky ∈ [-Npe/2, Npe/2)/FOV
kx_max = Nfe / (2 * FOV)
ky_max = Npe / (2 * FOV)
extent_k = [-kx_max, kx_max, -ky_max, ky_max]

peak_meas   = np.unravel_index(np.argmax(np.abs(ksp_meas)),   ksp_meas.shape)
peak_theory = np.unravel_index(np.argmax(np.abs(ksp_theory)), ksp_theory.shape)
expected_peak = (Npe // 2, Nfe // 2)
sym_err = magnitude_symmetry_error(ksp_meas)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Top-left: phantom occupancy
im0 = axes[0, 0].imshow(occ, cmap="viridis", origin="lower", extent=extent_img)
axes[0, 0].set_title(f"phantom occupancy\n{int(occ.sum())} spins on T1 plate")
axes[0, 0].set_xlabel("x [m]"); axes[0, 0].set_ylabel("y [m]")
plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, shrink=0.9, label="spins / pixel")

# Top-right: reconstructed image
im1 = axes[0, 1].imshow(img, cmap="viridis", origin="lower", extent=extent_img)
axes[0, 1].set_title(
    f"reconstructed |image|\n"
    f"Npe={Npe}, Nfe={Nfe}, FOV={FOV:g} m, vox={cfg['voxel_size_mm']:g} mm\n"
    f"TI={TI:g} s, TR={TR:g} s, sim time = {TR*Npe:g} s\n"
    f"r={r:.3f}  WAPE={wape:.1f}%  NRMSE={nrmse:.1f}%  on-supp={in_supp:.1f}%",
    fontsize=9,
)
axes[0, 1].set_xlabel("x [m]"); axes[0, 1].set_ylabel("y [m]")
plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, shrink=0.9)

# Bottom-left: theoretical k-space = FFT of occupancy (under theoretical phantom)
im2 = axes[1, 0].imshow(lm_theory, cmap="magma", origin="lower", extent=extent_k,
                        aspect="auto")
axes[1, 0].set_title(
    f"theoretical log|FFT(occupancy)|\n"
    f"peak at (ipe={peak_theory[0]}, ife={peak_theory[1]})\n"
    f"geometry only — no T1/TR contrast weighting",
    fontsize=9,
)
axes[1, 0].set_xlabel("kx [1/m]"); axes[1, 0].set_ylabel("ky [1/m]")
plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, shrink=0.9, label="log10 |S|")

# Bottom-right: measured k-space (under reconstructed image)
im3 = axes[1, 1].imshow(lm_meas, cmap="magma", origin="lower", extent=extent_k,
                        aspect="auto")
axes[1, 1].set_title(
    f"measured log|k-space|\n"
    f"peak at (ipe={peak_meas[0]}, ife={peak_meas[1]})  expected ({expected_peak[0]}, {expected_peak[1]})\n"
    f"magnitude-symmetry mean err = {sym_err:.2e}  (|S(k)| vs |S(-k)|)",
    fontsize=9,
)
axes[1, 1].set_xlabel("kx [1/m]"); axes[1, 1].set_ylabel("ky [1/m]")
plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, shrink=0.9, label="log10 |S|")

fig.suptitle(
    f"Recon-vs-phantom  [{run_label}]   "
    f"(r={r:.3f}, WAPE={wape:.1f}%, on-supp={in_supp:.1f}%)",
    fontsize=11, y=1.00,
)

print(f"Pearson r            = {r:.3f}")
print(f"WAPE (L1-normalised) = {wape:.1f} %")
print(f"NRMSE                = {nrmse:.1f} %")
print(f"energy-on-support    = {in_supp:.1f} %")
print(f"measured peak idx    = {peak_meas}  (expected {expected_peak})")
print(f"mag-symmetry err     = {sym_err:.3e}")

out = os.path.join(run_dir, "recon_vs_phantom.png")
plt.tight_layout()
plt.savefig(out, dpi=130, bbox_inches="tight")
print(f"wrote {out}")

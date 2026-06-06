#!/usr/bin/env python
"""Render pixel grid (Npe × Nfe over FOV) overlaid on phantom spins + sphere
outlines, to diagnose whether spheres land on pixel centres or pixel edges.

Run scripts/pixel_grid_overlay.jl first with matching flags to produce inputs
in scripts/runs/pixel_grid_overlay/<run_label>/.

Usage:
    python scripts/pixel_grid_overlay.py
    python scripts/pixel_grid_overlay.py --npe 32 --nfe 64 --voxel-mm 1.0
    python scripts/pixel_grid_overlay.py --run npe32_nfe64_fov0p2_vox1p0mm
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from matplotlib.patches import Circle

here = os.path.dirname(os.path.abspath(__file__))
# Output root. Override with RUNS_ROOT to target a version folder, e.g.
# RUNS_ROOT=scripts/runs/eae656a_current-koma. Inherited by child processes the
# .jl scripts spawn, so the Julia driver and these plotters always agree.
runs_root = os.path.join(os.environ.get("RUNS_ROOT", os.path.join(here, "runs")),
                         "pixel_grid_overlay")


def build_run_label(npe, nfe, fov, voxel_mm):
    fov_tag = str(fov).replace(".", "p")
    vox_tag = str(voxel_mm).replace(".", "p")
    return f"npe{npe}_nfe{nfe}_fov{fov_tag}_vox{vox_tag}mm"


# ── Per-run state ─────────────────────────────────────────────────────────────

@dataclass
class RunContext:
    """All the shared per-run state. Built once by `load_run_context`, then
    threaded into each render function instead of re-passing 10 args."""
    run_dir:    str
    run_label:  str
    cfg:        dict
    img_meta:   dict
    FOV:        float
    Npe:        int
    Nfe:        int
    dx:         float
    dy:         float
    extent_mm:  tuple[float, float, float, float]
    x_edges_mm: np.ndarray
    y_edges_mm: np.ndarray
    spins_mm:   np.ndarray
    centres_mm: np.ndarray
    meta:       list[dict]
    half_mm:    float

    def path(self, name: str) -> str:
        return os.path.join(self.run_dir, name)


def load_run_context(run_dir: str, zoom_mm: float | None) -> RunContext:
    spins   = np.load(os.path.join(run_dir, "spins_xy.npy"))
    centres = np.load(os.path.join(run_dir, "sphere_centres.npy"))
    with open(os.path.join(run_dir, "sphere_meta.json")) as f:
        meta = json.load(f)
    with open(os.path.join(run_dir, "config.json")) as f:
        cfg = json.load(f)

    FOV = cfg["FOV_m"]
    Npe, Nfe = int(cfg["Npe"]), int(cfg["Nfe"])
    dx, dy = cfg["dx_fe_m"], cfg["dy_pe_m"]

    return RunContext(
        run_dir   = run_dir,
        run_label = os.path.basename(run_dir),
        cfg       = cfg,
        img_meta  = cfg.get("image", {"rendered": False}),
        FOV       = FOV,
        Npe       = Npe,
        Nfe       = Nfe,
        dx        = dx,
        dy        = dy,
        extent_mm = (-FOV/2*1000, FOV/2*1000, -FOV/2*1000, FOV/2*1000),
        x_edges_mm = (np.arange(Nfe + 1) - Nfe / 2 - 0.5) * dx * 1000,
        y_edges_mm = (np.arange(Npe + 1) - Npe / 2 - 0.5) * dy * 1000,
        spins_mm   = spins   * 1000,
        centres_mm = centres * 1000,
        meta       = meta,
        half_mm    = zoom_mm if zoom_mm is not None else FOV * 1000 / 2,
    )


# ── Small drawing primitives ──────────────────────────────────────────────────

def _draw_pixel_grid(ax, ctx: RunContext, color="white", alpha=0.20):
    for xe in ctx.x_edges_mm:
        ax.axvline(xe, color=color, lw=0.3, alpha=alpha, zorder=2)
    for ye in ctx.y_edges_mm:
        ax.axhline(ye, color=color, lw=0.3, alpha=alpha, zorder=2)


def _draw_spheres(ax, ctx: RunContext, edge="cyan", with_labels=False,
                  annotate_t1=False):
    for c, m in zip(ctx.centres_mm, ctx.meta):
        r_mm = m["radius_m"] * 1000
        ax.add_patch(Circle((c[0], c[1]), r_mm, fill=False,
                            edgecolor=edge, lw=1.2, zorder=4))
        ax.plot(c[0], c[1], "o", color=edge, ms=3, zorder=5)
        if annotate_t1:
            ax.annotate(f"{m['T1_s']*1000:.0f}", (c[0], c[1]),
                        ha="center", va="center", fontsize=7,
                        color=edge, zorder=6)
        elif with_labels:
            ax.annotate(m["label"], (c[0], c[1]),
                        xytext=(4, 4), textcoords="offset points",
                        fontsize=7, color=edge, zorder=6)


def _zoom_axes(ax, ctx: RunContext, xlabel="x [mm]  (FE →)",
               ylabel="y [mm]  (PE ↑)"):
    ax.set_xlim(-ctx.half_mm, ctx.half_mm)
    ax.set_ylim(-ctx.half_mm, ctx.half_mm)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def _log_mag(z, floor_ratio=1e-6):
    m = np.abs(z)
    ref = m.max() if m.max() > 0 else 1.0
    return np.log10(m + ref * floor_ratio)


def _save(fig, ctx: RunContext, name: str, also_pdf=False, **kw):
    out = ctx.path(name + ".png")
    plt.savefig(out, dpi=130, bbox_inches="tight", **kw)
    if also_pdf:
        plt.savefig(ctx.path(name + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


# ── Render functions: each writes one PNG ────────────────────────────────────

def render_spin_overlay(ctx: RunContext, suffix: str, image: np.ndarray | None):
    """The original 10×10 in. per-image overlay — grid + spins + sphere
    outlines, optionally on top of a magnitude image. `suffix` distinguishes
    `_image` (raw recon), `_image_clean` (Hamming + pad), or "" (no image)."""
    is_clean = suffix.endswith("_clean")
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal")

    if image is not None:
        im = ax.imshow(image, cmap="magma", origin="lower",
                       extent=ctx.extent_mm, interpolation="nearest", zorder=0)
        plt.colorbar(im, ax=ax, fraction=0.046, shrink=0.85,
                     label=f"|image|  (TI={ctx.img_meta['TI_s']:g} s)")
        s, c, alpha = 1, "white", 0.10
    else:
        s, c, alpha = 2, "0.55", 0.35

    ax.scatter(ctx.spins_mm[:, 0], ctx.spins_mm[:, 1], s=s, c=c, alpha=alpha,
               linewidths=0, label=f"spins (N={len(ctx.spins_mm)})", zorder=1)

    grid_color = "white" if image is not None else "0.75"
    grid_alpha = 0.25 if image is not None else 1.0
    _draw_pixel_grid(ax, ctx, color=grid_color, alpha=grid_alpha)

    sphere_edge = "cyan" if image is not None else "tab:blue"
    _draw_spheres(ax, ctx, edge=sphere_edge, with_labels=True)

    _zoom_axes(ax, ctx,
               xlabel="x  [mm]  (frequency encode →)",
               ylabel="y  [mm]  (phase encode ↑)")

    sphere_d_mm = 2 * ctx.meta[0]["radius_m"] * 1000
    recon_label = "Hamming + 2× zero-pad" if is_clean else "raw FFT"
    img_line = (f"\n|image| from IR-SE: TI={ctx.img_meta['TI_s']:g} s, "
                f"TE={ctx.img_meta['TE_s']:g} s, TR={ctx.img_meta['TR_s']:g} s, "
                f"recon = {recon_label}"
                if image is not None else "")
    ax.set_title(
        f"Pixel grid vs spins  [{ctx.run_label}]\n"
        f"FOV={ctx.FOV*1000:g} mm, voxel={ctx.cfg['voxel_size_mm']:g} mm, "
        f"Npe={ctx.Npe} → dy_PE={ctx.dy*1000:.3f} mm   "
        f"Nfe={ctx.Nfe} → dx_FE={ctx.dx*1000:.3f} mm   "
        f"sphere ⌀ {sphere_d_mm:g} mm   "
        f"({sphere_d_mm/(ctx.dy*1000):.1f} PE px, {sphere_d_mm/(ctx.dx*1000):.1f} FE px)\n"
        f"slice z = {ctx.cfg['z_plate_m']*1000:.2f} ± {ctx.cfg['slice_half_m']*1000:.3f} mm   "
        f"({ctx.cfg['n_spins_in_slice']} of {ctx.cfg['n_spins_total']} spins)"
        f"{img_line}",
        fontsize=9,
    )
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    plt.tight_layout()
    _save(fig, ctx, f"pixel_grid_overlay{suffix}", also_pdf=True)


def render_theory_vs_sim(ctx: RunContext, theory_mag, sim_img, diff_img):
    """3-panel: theoretical forward model | KomaMRI sim | diff."""
    vmax     = max(theory_mag.max(), sim_img.max())
    diff_abs = float(np.abs(diff_img).max())
    ti_s     = ctx.img_meta["TI_s"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    panels = [
        (theory_mag, f"Theory  |forward model|  (TI={ti_s:g} s)", "magma",
         0.0, vmax),
        (sim_img,    f"Simulated  |KomaMRI|  (TI={ti_s:g} s)",     "magma",
         0.0, vmax),
        (diff_img,   "Difference  (sim − theory)",                   "RdBu_r",
         -diff_abs, diff_abs),
    ]
    for ax, (data, title, cmap, vmin, vmx) in zip(axes, panels):
        im = ax.imshow(data, cmap=cmap, origin="lower", extent=ctx.extent_mm,
                       vmin=vmin, vmax=vmx, interpolation="nearest")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046)
        _draw_pixel_grid(ax, ctx)
        _draw_spheres(ax, ctx)
        _zoom_axes(ax, ctx)

    fig.suptitle(f"Forward model vs KomaMRI  [{ctx.run_label}]", fontsize=11)
    plt.tight_layout()
    _save(fig, ctx, "pixel_grid_overlay_comparison")


def render_t1_map(ctx: RunContext, t1_map):
    masked = np.where(t1_map > 0, t1_map, np.nan)
    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(masked, cmap="plasma", origin="lower",
                   extent=ctx.extent_mm, interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.046, label="T1 [s]")
    _draw_pixel_grid(ax, ctx)
    _draw_spheres(ax, ctx, edge="white", annotate_t1=True)
    _zoom_axes(ax, ctx)
    ax.set_title(f"True T1 map  [{ctx.run_label}]")
    plt.tight_layout()
    _save(fig, ctx, "pixel_grid_overlay_t1map")


def render_kspace_baseline(ctx: RunContext, ksp_meas, ksp_theo, theo_label):
    """1- or 2-panel log|k-space|: measured (+ theoretical if available)."""
    kx_max = ctx.Nfe / (2 * ctx.FOV)
    ky_max = ctx.Npe / (2 * ctx.FOV)
    extent_k = (-kx_max, kx_max, -ky_max, ky_max)

    def lmf(z): return np.log10(np.maximum(np.abs(z), np.abs(z).max() * 1e-4))
    ti_s = ctx.img_meta["TI_s"]

    # Panel painters, reused across the figures below.
    def draw_mag(ax, z, title):
        im = ax.imshow(lmf(z), cmap="magma", origin="lower",
                       extent=extent_k, aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("kx [1/m]"); ax.set_ylabel("ky [1/m]")
        plt.colorbar(im, ax=ax, fraction=0.046, label="log10 |S|")

    # ── 2-panel measured | theory (used by the stitch) ───────────────────────
    ncols = 2 if ksp_theo is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 6))
    if ncols == 1:
        axes = [axes]
    draw_mag(axes[0], ksp_meas, f"measured log|k-space|  (TI={ti_s:g} s)")
    if ksp_theo is not None:
        draw_mag(axes[1], ksp_theo, theo_label)
    fig.suptitle(f"k-space comparison  [{ctx.run_label}]", fontsize=11)
    plt.tight_layout()
    _save(fig, ctx, "pixel_grid_overlay_kspace")

    if ksp_theo is None:
        return

    # Signed |k| difference (measured − theory). Symlog colour scale so the
    # residual structure shows — a linear scale is dominated by a couple of
    # huge centre points and hides everything else.
    diff = np.abs(ksp_meas) - np.abs(ksp_theo)
    dmax = float(np.abs(diff).max()) or 1.0
    nz = np.abs(diff)[np.abs(diff) > 0]
    linthresh = float(np.median(nz)) if nz.size else dmax * 1e-3
    relerr = np.sum(np.abs(ksp_meas - ksp_theo)) / max(np.sum(np.abs(ksp_meas)), 1e-30)

    def draw_diff(ax):
        im = ax.imshow(diff, cmap="bwr", origin="lower", extent=extent_k,
                       aspect="auto",
                       norm=SymLogNorm(linthresh=linthresh, vmin=-dmax, vmax=dmax, base=10))
        ax.set_title(f"|k| diff (meas − theory), symlog\nL1 relerr = {relerr:.2e}")
        ax.set_xlabel("kx [1/m]"); ax.set_ylabel("ky [1/m]")
        plt.colorbar(im, ax=ax, fraction=0.046, label="Δ|S| (symlog)")

    # ── Standalone single-panel diff ─────────────────────────────────────────
    figd, axd = plt.subplots(figsize=(7, 6))
    draw_diff(axd)
    axd.set_title(axd.get_title() + f"  (TI={ti_s:g} s)")
    figd.suptitle(f"k-space diff  [{ctx.run_label}]", fontsize=11)
    plt.tight_layout()
    _save(figd, ctx, "pixel_grid_overlay_kspace_diff")

    # ── 3-panel measured | theory | diff ─────────────────────────────────────
    fig3, ax3 = plt.subplots(1, 3, figsize=(21, 6))
    draw_mag(ax3[0], ksp_meas, f"measured log|k-space|  (TI={ti_s:g} s)")
    draw_mag(ax3[1], ksp_theo, theo_label)
    draw_diff(ax3[2])
    fig3.suptitle(f"k-space comparison + diff  [{ctx.run_label}]", fontsize=11)
    plt.tight_layout()
    _save(fig3, ctx, "pixel_grid_overlay_kspace_diff3")


def render_variants_grid(ctx: RunContext, imgs, ksps):
    """5×2 figure: 4 sim variants + theory, image row + k-space row.
    Image row uses per-panel max (Hamming attenuates ~3×); ksp row is
    shared log-magnitude."""
    titles = ["Normal", "Clean recon", "Spoiled", "Spoiled + clean", "Theory"]
    ksp_disp = [_log_mag(k) for k in ksps]
    kvmin = min(a.min() for a in ksp_disp)
    kvmax = max(a.max() for a in ksp_disp)

    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    for j, (title, img, k_disp) in enumerate(zip(titles, imgs, ksp_disp)):
        ax = axes[0, j]
        im = ax.imshow(img, cmap="magma", origin="lower", extent=ctx.extent_mm,
                       vmin=0, vmax=img.max(), interpolation="nearest")
        ax.set_title(f"{title}\nmax={img.max():.3g}", fontsize=10)
        if j == 0:
            ax.set_ylabel("|image|  (per-panel norm)\ny [mm]", fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046)

        ax = axes[1, j]
        im = ax.imshow(k_disp, cmap="viridis", origin="lower",
                       vmin=kvmin, vmax=kvmax, interpolation="nearest",
                       aspect="auto")
        if j == 0:
            ax.set_ylabel("log10 |k-space|\nky", fontsize=10)
        ax.set_xlabel("kx", fontsize=9)
        if j == len(titles) - 1:
            plt.colorbar(im, ax=ax, fraction=0.046)

    ti = ctx.img_meta.get("TI_s", float("nan"))
    te = ctx.img_meta.get("TE_s", float("nan"))
    tr = ctx.img_meta.get("TR_s", float("nan"))
    fig.suptitle(
        f"IR-SE recon variants  [{ctx.run_label}]\n"
        f"TI={ti:g} s, TE={te:g} s, TR={tr:g} s   "
        f"Npe={ctx.Npe}, Nfe={ctx.Nfe}, FOV={ctx.FOV*1000:g} mm, "
        f"voxel={ctx.cfg['voxel_size_mm']:g} mm",
        fontsize=11,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, ctx, "pixel_grid_overlay_variants")


def render_kspace_diff(ctx: RunContext, ksp_normal, ksp_spoiled):
    """2×3 figure localising where the spoiler acts in k-space.
    Row 1: log|normal|, log|spoiled|, log|normal−spoiled|.
    Row 2: ky profile, kx profile, per-PE-line diff stems."""
    ksp_d = ksp_normal - ksp_spoiled

    panels_top = [
        (_log_mag(ksp_normal),  "log10 |k-space|  Normal"),
        (_log_mag(ksp_spoiled), "log10 |k-space|  Spoiled"),
        (_log_mag(ksp_d),       "log10 |Normal − Spoiled|\n(what the spoiler removed)"),
    ]
    vmin = min(p[0].min() for p in panels_top)
    vmax = max(p[0].max() for p in panels_top)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, (data, title) in zip(axes[0], panels_top):
        im = ax.imshow(data, cmap="viridis", origin="lower",
                       vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("kx index"); ax.set_ylabel("ky index")
        plt.colorbar(im, ax=ax, fraction=0.046)

    Npe_k, Nfe_k = ksp_normal.shape
    ky_axis = np.arange(Npe_k) - Npe_k // 2
    kx_axis = np.arange(Nfe_k) - Nfe_k // 2

    axes[1, 0].plot(ky_axis, np.abs(ksp_normal).mean(axis=1),
                    label="Normal", color="tab:blue")
    axes[1, 0].plot(ky_axis, np.abs(ksp_spoiled).mean(axis=1),
                    label="Spoiled", color="tab:orange")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xlabel("ky index (centred)")
    axes[1, 0].set_ylabel("mean |k-space|  along kx")
    axes[1, 0].set_title("ky profile  (which PE lines are contaminated)")
    axes[1, 0].legend(fontsize=9); axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(kx_axis, np.abs(ksp_normal).mean(axis=0),
                    label="Normal", color="tab:blue")
    axes[1, 1].plot(kx_axis, np.abs(ksp_spoiled).mean(axis=0),
                    label="Spoiled", color="tab:orange")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel("kx index (centred)")
    axes[1, 1].set_ylabel("mean |k-space|  along ky")
    axes[1, 1].set_title("kx profile  (readout direction)")
    axes[1, 1].legend(fontsize=9); axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].stem(ky_axis, np.abs(ksp_d).max(axis=1),
                    basefmt=" ", linefmt="C2-", markerfmt="C2o")
    axes[1, 2].set_yscale("log")
    axes[1, 2].set_xlabel("ky index (centred)")
    axes[1, 2].set_ylabel("max |Normal − Spoiled|  in this PE line")
    axes[1, 2].set_title("Diff per PE line\n(which shots spoil-changed)")
    axes[1, 2].grid(True, alpha=0.3)

    ti = ctx.img_meta.get("TI_s", float("nan"))
    tr = ctx.img_meta.get("TR_s", float("nan"))
    fig.suptitle(
        f"k-space diff: normal vs spoiled  [{ctx.run_label}]\n"
        f"TI={ti:g} s, TR={tr:g} s, Npe={ctx.Npe}, Nfe={ctx.Nfe}",
        fontsize=11,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, ctx, "pixel_grid_overlay_kspace_diff")


# ── Per-run orchestration ─────────────────────────────────────────────────────

def process_run(run_dir: str, zoom_mm: float | None):
    ctx = load_run_context(run_dir, zoom_mm)
    print(f"\n=== {ctx.run_label} ===")

    # PE-pixel-centre offset diagnostic table (cheap; always print).
    print(f"  {'label':>6}  {'cy [mm]':>9}  {'PE pix ctr [mm]':>17}  {'offset [mm]':>13}")
    for c, m in zip(ctx.centres_mm, ctx.meta):
        pe_idx = round(c[1] / (ctx.dy * 1000))
        pe_ctr = pe_idx * ctx.dy * 1000
        print(f"  {m['label']:>6}  {c[1]:>9.3f}  {pe_ctr:>17.3f}  {c[1]-pe_ctr:>+13.3f}")

    # Per-image overlays: raw recon and clean recon when present.
    overlays = []
    if os.path.isfile(ctx.path("image.npy")):
        overlays.append(("_image",       np.load(ctx.path("image.npy"))))
    if os.path.isfile(ctx.path("image_clean.npy")):
        overlays.append(("_image_clean", np.load(ctx.path("image_clean.npy"))))
    if not overlays:
        overlays = [("", None)]  # geometry-only overlay
    for suffix, image in overlays:
        render_spin_overlay(ctx, suffix, image)

    # Theory vs sim vs diff (needs image.npy, theory_magnitude.npy, diff_image.npy).
    if all(os.path.isfile(ctx.path(f)) for f in
           ("image.npy", "theory_magnitude.npy", "diff_image.npy")):
        render_theory_vs_sim(
            ctx,
            theory_mag = np.load(ctx.path("theory_magnitude.npy")),
            sim_img    = np.load(ctx.path("image.npy")),
            diff_img   = np.load(ctx.path("diff_image.npy")),
        )

    # T1 map.
    if os.path.isfile(ctx.path("T1_map.npy")):
        render_t1_map(ctx, np.load(ctx.path("T1_map.npy")))

    # K-space baseline (measured + optional theory or occupancy fallback).
    if os.path.isfile(ctx.path("kspace.npy")):
        ksp_meas = np.load(ctx.path("kspace.npy"))
        if os.path.isfile(ctx.path("theory_kspace.npy")):
            ksp_theo   = np.load(ctx.path("theory_kspace.npy"))
            theo_label = "theoretical log|FFT(forward model)|"
        elif os.path.isfile(ctx.path("occupancy.npy")):
            occ = np.load(ctx.path("occupancy.npy"))
            ksp_theo = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(occ)))
            theo_label = "theoretical log|FFT(occupancy)|  (geometry only)"
        else:
            ksp_theo, theo_label = None, ""
        render_kspace_baseline(ctx, ksp_meas, ksp_theo, theo_label)

    # 5×2 variants grid (needs all four sim variants + theory).
    variant_files = ["image.npy", "image_clean.npy",
                     "image_spoil.npy", "image_spoil_clean.npy",
                     "theory_magnitude.npy",
                     "kspace.npy", "kspace_clean.npy",
                     "kspace_spoil.npy", "kspace_spoil_clean.npy",
                     "theory_kspace.npy"]
    if all(os.path.isfile(ctx.path(f)) for f in variant_files):
        imgs = [np.abs(np.load(ctx.path(f))) for f in variant_files[:5]]
        ksps = [np.load(ctx.path(f)) for f in variant_files[5:]]
        render_variants_grid(ctx, imgs, ksps)

    # K-space diff (needs kspace.npy and kspace_spoil.npy).
    if (os.path.isfile(ctx.path("kspace.npy"))
            and os.path.isfile(ctx.path("kspace_spoil.npy"))):
        render_kspace_diff(
            ctx,
            ksp_normal  = np.load(ctx.path("kspace.npy")),
            ksp_spoiled = np.load(ctx.path("kspace_spoil.npy")),
        )


# ── Entrypoint ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=None)
    parser.add_argument("--npe", type=int, default=32)
    parser.add_argument("--nfe", type=int, default=64)
    parser.add_argument("--fov", type=float, default=0.2)
    parser.add_argument("--voxel-mm", type=float, default=1.0)
    parser.add_argument("--zoom", type=float, default=None,
                        help="half-extent in mm to zoom into (default: full FOV)")
    args = parser.parse_args()

    if args.run:
        run_dirs = [os.path.join(runs_root, args.run)]
    else:
        base = build_run_label(args.npe, args.nfe, args.fov, args.voxel_mm)
        run_dirs = sorted(glob.glob(os.path.join(runs_root, base + "*")))
        run_dirs = [d for d in run_dirs if os.path.isdir(d)]
    if not run_dirs:
        raise SystemExit(
            f"no run directories under {runs_root} matching "
            f"{'--run ' + args.run if args.run else 'grid base label'}\n"
            f"Run scripts/pixel_grid_overlay.jl first."
        )

    for run_dir in run_dirs:
        process_run(run_dir, args.zoom)


if __name__ == "__main__":
    main()

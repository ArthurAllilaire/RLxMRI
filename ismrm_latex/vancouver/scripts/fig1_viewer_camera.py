"""Recover the camera of the report's 3D viewer screenshot for the Fig 1(A) re-export.

report_latex/imgs/mri_system_phantom/T1_voxel_phantom.png is a browser screenshot of
plot_phantom_html (MRISystemPhantom.jl, examples/plot_phantom_3d.jl) after an interactive
rotation, so its camera was never saved. This fits it from the scene box corners, using
the projection of plotly.js 2.35 gl3d (the version PlotlyJS bundles): the axis ranges map
to a cube of half-size 0.5, lookAt(eye, origin, +z), perspective with fovy = pi/4 over the
plot area. Turntable rotation keeps the eye at plotly's default distance 1.25 * sqrt(3),
which a free fit reproduces (2.18), so only azimuth, elevation and the plot area are fitted.
The scene fixes the plot area's height and horizontal centre but not its width (the
vertical field of view is fixed); the width follows from the colourbar, which plotly
places at x = 1.02 of the plot width plus a 10 px pad.

Writes figures/fig1_viewer_camera.json, read by fig1_twin3d_viewer.jl.

Run from the RLxMRI repo root:
  .venv/bin/python ismrm_latex/vancouver/scripts/fig1_viewer_camera.py
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

import figstyle

SHOT = figstyle.ROOT / "report_latex" / "imgs" / "mri_system_phantom" / "T1_voxel_phantom.png"
OUT = figstyle.FIGURES / "fig1_viewer_camera.json"
# The screenshot was taken on a 2x display.
DEVICE_PX_PER_CSS = 2.0
EYE_DISTANCE = 1.25 * np.sqrt(3.0)
COLORBAR_X, COLORBAR_XPAD_CSS = 1.02, 10.0
H = 0.5
# Scene box corners (normalised scene coordinates) and where they sit in the screenshot,
# read off a zoomed, gridded view of it (device px, +-3 px). The sixth visible corner,
# (-x, +y, +z), is hidden by the hover box.
CORNERS = {
    "top, +x -y": ((H, -H, H), (281, 278)),
    "top, -x -y": ((-H, -H, H), (622, 240)),
    "bottom, +x -y": ((H, -H, -H), (343, 828)),
    "bottom, +x +y": ((H, H, -H), (862, 1004)),
    "bottom, -x +y": ((-H, H, -H), (1022, 727)),
}


def look_at(eye: np.ndarray) -> np.ndarray:
    f = -eye / np.linalg.norm(eye)
    s = np.cross(f, [0.0, 0.0, 1.0])
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def project(pts: np.ndarray, eye: np.ndarray, cx: float, y0: float, h: float) -> np.ndarray:
    """Screen position in a plot area of height h, horizontal centre cx and top y0."""
    # Depth rows of the perspective matrix do not affect screen x and y, so they stay zero;
    # with the aspect folded into column 0, the plot width cancels out.
    f = 1 / np.tan(np.pi / 8)
    proj = np.zeros((4, 4))
    proj[0, 0], proj[1, 1], proj[3, 2] = f, f, -1.0
    clip = np.c_[pts, np.ones(len(pts))] @ (proj @ look_at(eye)).T
    ndc = clip[:, :2] / clip[:, 3:4]
    return np.c_[cx + ndc[:, 0] * h / 2, y0 + (1 - ndc[:, 1]) / 2 * h]


def eye_from(az: float, el: float) -> np.ndarray:
    return EYE_DISTANCE * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])


def fit() -> tuple[np.ndarray, float]:
    pts = np.array([c[0] for c in CORNERS.values()])
    obs = np.array([c[1] for c in CORNERS.values()], float)

    def resid(q):
        return (project(pts, eye_from(q[0], q[1]), *q[2:]) - obs).ravel()

    # Levenberg-Marquardt from a coarse grid of azimuths and elevations.
    best = None
    for az0 in np.radians([10, 30, 50]):
        for el0 in np.radians([5, 20, 35]):
            q, lam = np.array([az0, el0, 700.0, 60.0, 950.0]), 1e-2
            for _ in range(300):
                r = resid(q)
                jac = np.empty((len(r), len(q)))
                for j in range(len(q)):
                    dq = np.zeros_like(q)
                    dq[j] = 1e-6 * max(1.0, abs(q[j]))
                    jac[:, j] = (resid(q + dq) - r) / dq[j]
                a = jac.T @ jac
                step = np.linalg.solve(a + lam * np.diag(np.diag(a)), -jac.T @ r)
                if (resid(q + step) ** 2).sum() < (r ** 2).sum():
                    q, lam = q + step, lam * 0.5
                else:
                    lam *= 4
            cost = (resid(q) ** 2).mean()
            if best is None or cost < best[1]:
                best = (q, cost)
    return best[0], float(np.sqrt(best[1]))


def colourbar_left_device_px() -> int:
    """First strongly coloured column right of the scene, on a row through the colourbar."""
    img = plt.imread(SHOT)[..., :3]
    row = img[img.shape[0] // 2]
    coloured = np.nonzero(row.max(-1) - row.min(-1) > 0.25)[0]
    return int(coloured[coloured > 0.8 * img.shape[1]].min())


def main() -> None:
    q, rms = fit()
    eye = eye_from(q[0], q[1])
    cx, y0, h = q[2:] / DEVICE_PX_PER_CSS
    bar_left = colourbar_left_device_px() / DEVICE_PX_PER_CSS
    w = (bar_left - COLORBAR_XPAD_CSS - cx) / (0.5 + COLORBAR_X - 1.0)
    result = {
        "source": str(SHOT.relative_to(figstyle.ROOT)),
        "eye": dict(zip("xyz", map(float, eye))),
        "plot_area_css": {"left": float(cx - w / 2), "top": float(y0), "width": float(w), "height": float(h)},
        "colourbar_left_css": bar_left,
        "corner_rms_device_px": rms,
    }
    OUT.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()

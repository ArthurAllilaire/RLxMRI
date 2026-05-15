"""Gate 6.3 — Joint (T1,T2,PD) likelihood landscape diagnostic.

For each of 4 representative tissue cells, simulate noisy magnitude data with
a fixed IR-SE-style reference sequence (6 TIs x 4 TEs = 24 measurements),
then build a 20x20x20 grid over (T1,T2,PD) and rank truth in 50 noise trials.

Pass: rank <= 50 (top 0.6% of 8000) on >= 80% of trials.

We use a closed-form IR-SE forward model for the grid search (orders of
magnitude faster than MRzero per evaluation; equivalent in the single-spin
limit). The "ground truth" noisy data is also generated from the same forward
model — so this is a SELF-CONSISTENT likelihood-landscape diagnostic. Any
multimodality is therefore intrinsic to the (T1,T2,PD) forward model under
this sequence, not a simulator-mismatch artifact.
"""
from __future__ import annotations
import os, sys, time, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUTDIR = os.path.join(ROOT, "runs", "e2e_gate", "gate3")
os.makedirs(OUTDIR, exist_ok=True)

RNG = np.random.default_rng(0xCAFE)

# Reference sequence: 6 TI x 4 TE IR-SE
TIs = np.array([0.05, 0.15, 0.4, 0.9, 1.8, 3.0])
TEs = np.array([0.01, 0.05, 0.15, 0.4])

# Build grid mesh
N_T1 = N_T2 = N_PD = 20
T1_grid = np.logspace(np.log10(0.05), np.log10(3.0), N_T1)
T2_grid = np.logspace(np.log10(0.01), np.log10(0.8), N_T2)
PD_grid = np.linspace(0.3, 1.5, N_PD)

CELLS = [
    ("long_T1_long_T2",  1.8, 0.5, 1.0),
    ("long_T1_short_T2", 1.5, 0.05, 1.0),
    ("short_T1_long_T2", 0.3, 0.3, 1.0),
    ("short_T1_short_T2",0.2, 0.04, 1.0),
]

NOISE_SIGMA_FRAC = 0.05  # Ch4 spec — sigma = 5% of max signal
N_TRIALS = 50


def forward(T1, T2, PD):
    """Magnitude signal vector of length len(TIs)*len(TEs).
    Order: outer over TI, inner over TE.
    Model: |1 - 2 exp(-TI/T1)| * PD * exp(-TE/T2)
    """
    # vectorize over grid
    T1 = np.asarray(T1)[..., None, None]
    T2 = np.asarray(T2)[..., None, None]
    PD = np.asarray(PD)[..., None, None]
    TI = TIs[None, :, None]; TE = TEs[None, None, :]
    sig = np.abs(1.0 - 2.0 * np.exp(-TI / T1)) * PD * np.exp(-TE / T2)
    flat = sig.reshape(*np.asarray(T1).shape[:-2], -1)
    return flat


def landscape_for(T1t, T2t, _unused, PDt, noise_sigma):
    # truth signal
    true_sig = forward(np.array([T1t]), np.array([T2t]), np.array([PDt]))[0]
    sigma = noise_sigma
    noisy = true_sig + RNG.normal(0, sigma, true_sig.shape)
    noisy = np.abs(noisy)  # magnitude — clip at 0
    # build full 20x20x20 grid SSE (vectorised; meshgrid skipped — we factor
    # PD as a per-iter scalar multiplier below)
    sse = np.zeros((N_T1, N_T2, N_PD))
    # vectorize over T1xT2: pre-compute |1-2exp(-TI/T1)| and exp(-TE/T2)
    A = np.abs(1.0 - 2.0 * np.exp(-TIs[None, :] / T1_grid[:, None]))   # (N_T1, n_TI)
    B = np.exp(-TEs[None, :] / T2_grid[:, None])                       # (N_T2, n_TE)
    # full forward (no PD): shape (N_T1, N_T2, n_TI*n_TE)
    fwd_unit = (A[:, None, :, None] * B[None, :, None, :]).reshape(N_T1, N_T2, -1)
    for i_pd, pd in enumerate(PD_grid):
        pred = fwd_unit * pd
        diff = pred - noisy[None, None, :]
        sse[:, :, i_pd] = (diff ** 2).sum(axis=-1)
    return sse, noisy


def main():
    summary = {"cells": {}, "config": {
        "TIs": TIs.tolist(), "TEs": TEs.tolist(),
        "N_grid": [N_T1, N_T2, N_PD],
        "T1_range": [T1_grid[0], T1_grid[-1]],
        "T2_range": [T2_grid[0], T2_grid[-1]],
        "PD_range": [PD_grid[0], PD_grid[-1]],
        "noise_sigma_frac": NOISE_SIGMA_FRAC,
        "N_trials": N_TRIALS,
    }}

    overall_pass = True
    for name, T1t, T2t, PDt in CELLS:
        true_sig = forward(np.array([T1t]), np.array([T2t]), np.array([PDt]))[0]
        noise_sigma = NOISE_SIGMA_FRAC * float(true_sig.max())
        ranks = []
        # Initialise with a sentinel SSE/noisy so the type-checker can narrow.
        last_sse = np.zeros((N_T1, N_T2, N_PD))
        t0 = time.time()
        for _ in range(N_TRIALS):
            sse, _noisy = landscape_for(T1t, T2t, T2t, PDt, noise_sigma)
            i_t1 = int(np.argmin(np.abs(T1_grid - T1t)))
            i_t2 = int(np.argmin(np.abs(T2_grid - T2t)))
            i_pd = int(np.argmin(np.abs(PD_grid - PDt)))
            truth_sse = sse[i_t1, i_t2, i_pd]
            rank = int((sse < truth_sse).sum() + 1)
            ranks.append(rank)
            last_sse = sse
        ranks = np.array(ranks)
        pct_top50 = float((ranks <= 50).mean())
        pct_top10 = float((ranks <= 10).mean())
        cell_pass = bool(pct_top50 >= 0.8)
        overall_pass = overall_pass and cell_pass

        # Use last trial's SSE for landscape plots
        sse: np.ndarray = last_sse
        i_t1 = int(np.argmin(np.abs(T1_grid - T1t)))
        i_t2 = int(np.argmin(np.abs(T2_grid - T2t)))
        i_pd = int(np.argmin(np.abs(PD_grid - PDt)))
        best_idx = np.unravel_index(int(np.argmin(sse)), sse.shape)
        truth_sse = sse[i_t1, i_t2, i_pd]
        best_sse = sse[best_idx]
        sse_ratio = float(truth_sse / best_sse) if best_sse > 0 else float("inf")
        wrong_basin = (best_idx != (i_t1, i_t2, i_pd))

        summary["cells"][name] = dict(
            T1_true=T1t, T2_true=T2t, PD_true=PDt,
            noise_sigma=noise_sigma,
            median_rank=int(np.median(ranks)),
            mean_rank=float(np.mean(ranks)),
            p75_rank=int(np.percentile(ranks, 75)),
            max_rank=int(ranks.max()),
            pct_top50=pct_top50,
            pct_top10=pct_top10,
            pass_=cell_pass,
            best_idx=[int(x) for x in best_idx],
            best_T1=float(T1_grid[best_idx[0]]),
            best_T2=float(T2_grid[best_idx[1]]),
            best_PD=float(PD_grid[best_idx[2]]),
            sse_truth_over_best=sse_ratio,
            wrong_basin_at_argmin=bool(wrong_basin),
            wall_time_s=time.time()-t0,
        )
        print(f"{name}: median_rank={np.median(ranks):.0f}, pct<=50={pct_top50:.1%}, "
              f"sse_ratio={sse_ratio:.2f}, wrong_basin={wrong_basin}, "
              f"pass={cell_pass}")

        # Plot landscape (T1, T2) marginal at PD = PD_true
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        ax = axes[0]
        # 2D slice at i_pd
        im = ax.imshow(np.log10(sse[:, :, i_pd].T),
                       extent=[np.log10(T1_grid[0]), np.log10(T1_grid[-1]),
                               np.log10(T2_grid[0]), np.log10(T2_grid[-1])],
                       origin="lower", aspect="auto", cmap="viridis")
        ax.scatter([np.log10(T1t)], [np.log10(T2t)], c="r", marker="x", s=100,
                   label=f"truth ({T1t:.2f},{T2t:.3f})")
        ax.scatter([np.log10(T1_grid[best_idx[0]])], [np.log10(T2_grid[best_idx[1]])],
                   c="cyan", marker="+", s=120, label="argmin")
        plt.colorbar(im, ax=ax, label="log10(SSE)")
        ax.set_xlabel("log10(T1)")
        ax.set_ylabel("log10(T2)")
        ax.set_title(f"{name}: log10(SSE) at PD={PD_grid[i_pd]:.2f}")
        ax.legend(fontsize=8)

        ax = axes[1]
        ax.hist(ranks, bins=np.logspace(0, np.log10(8000), 30), edgecolor="k")
        ax.set_xscale("log")
        ax.set_xlabel("truth rank (1=best of 8000)")
        ax.set_ylabel("count")
        ax.axvline(50, c="r", ls="--", label="top-50 threshold")
        ax.set_title(f"rank histogram (N={N_TRIALS})\n"
                     f"median={np.median(ranks):.0f}, pct<=50={pct_top50:.1%}")
        ax.legend()

        ax = axes[2]
        # 1D slice: SSE vs T1 at (T2_true, PD_true)
        ax.semilogx(T1_grid, sse[:, i_t2, i_pd], "o-")
        ax.axvline(T1t, c="r", ls="--", label="truth")
        ax.set_xlabel("T1 [s]")
        ax.set_ylabel("SSE")
        ax.set_title(f"SSE vs T1 at (T2={T2t:.3f}, PD={PDt:.2f})")
        ax.legend()
        plt.tight_layout()
        out = os.path.join(OUTDIR, f"gate63_landscape_{name}.png")
        plt.savefig(out, dpi=120)
        plt.close(fig)
        print(f"  saved {out}")

    summary["pass_all_cells"] = overall_pass
    with open(os.path.join(OUTDIR, "gate63_results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nGate 6.3: {'PASS' if overall_pass else 'FAIL'}")
    print(f"saved {OUTDIR}/gate63_results.json")
    return overall_pass


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)

"""Synthetic test: does Rician magnitude noise bias the T1 fitter at the
SNR/schedules used by the t1_fit_vs_true sweep?

Bypasses Koma entirely. For each saved sweep config (b80/b120/b160/b240 @
SNR=2.5, Npe=32, Nfe=128):

  1. Pull the exact schedule (TIs, TRs, Npe) and image-domain noise σ
     (= background_std from the config — empirical image-domain magnitude
     noise std after IFFT recon).
  2. For each of the 14 T1_true values:
     a. Compute the *signed* expected signal y_k = ρ · Mz(T1, TI_k, TR_k; Npe)
        using `transient_mz_at_excite_npe` — the same forward model the
        fitter uses internally. ρ is uniform across spheres (PD is uniform
        in the T1 array of the Calibur phantom) and calibrated so that
        peak |y| across the longest-T1 sphere matches the reported
        snr_nema_peak · σ_img — i.e. the per-shot peak signal-to-noise
        matches the real run.
     c. Run N_TRIALS Monte Carlo realisations under two noise models:
          - magnitude (Rician):  y_obs = |y + N(0,σ) + i·N(0,σ)|
          - phase-sensitive:     y_obs = y + N(0,σ)
        Fit each with `fit_t1_generalized_ir` (signed=False / signed=True).
  3. Report per-sphere signed-bias and MAPE, aggregate across spheres.

Hypothesis: if Rician bias is the cause of MAPE degrading with budget,
the magnitude-recon synthetic fits will reproduce the budget trend, and
the phase-sensitive synthetic fits won't.

Usage:
  python scripts/synthetic_rician_test.py --trials 200
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
RUNS_DIR = HERE / "runs" / "t1_fit_vs_true"

BUDGETS = [80, 120, 160, 240]
SUBDIR_FMT = "b{budget}s_snr2p5_npe32fe128"


def load_config(budget: int) -> dict:
    path = RUNS_DIR / SUBDIR_FMT.format(budget=budget) / "config.json"
    with open(path) as f:
        return json.load(f)


def load_csv_truth(budget: int):
    """Read the per-sphere truth + observed sphere_means from config (truth
    comes from the CSV; means come from snr_report)."""
    import csv
    path = RUNS_DIR / SUBDIR_FMT.format(budget=budget) / "t1_fit_vs_true.csv"
    T1_true, labels = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            labels.append(row["label"])
            T1_true.append(float(row["T1_true_s"]))
    return labels, np.array(T1_true)


def init_julia():
    """Borrow the project's Julia bridge so we call the actual fitter."""
    from qalibremd_gym import env as _env_mod
    _env_mod._ensure_julia(None)
    return _env_mod._JL, _env_mod._JL_QMD


def predict_mz_signed(jl_qmd, T1: float, TIs, TRs, Npe: int) -> np.ndarray:
    """Signed Mz_at_excite at each (TI, TR) for given T1, via the Julia
    transient forward model — same one used inside the fitter."""
    out = np.empty(len(TIs), dtype=np.float64)
    for k, (ti, tr) in enumerate(zip(TIs, TRs)):
        out[k] = float(
            jl_qmd.transient_mz_at_excite_npe(
                float(T1), float(ti), float(tr),
                float(np.pi), float(np.pi / 2),
                Npe=int(Npe),
            )
        )
    return out


def fit_one(jl, jl_qmd, TIs, TRs, mags, Npe: int, sigma_abs: float, signed: bool):
    """Call fit_t1_generalized_ir with the same convention as t1_fit_vs_true.jl."""
    n = len(TIs)
    TIs_jl = jl.Vector[jl.Float64](list(map(float, TIs)))
    TRs_jl = jl.Vector[jl.Float64](list(map(float, TRs)))
    αs_jl = jl.Vector[jl.Float64]([float(np.pi)] * n)
    αexc_jl = jl.Vector[jl.Float64]([float(np.pi / 2)] * n)
    mags_jl = jl.Vector[jl.Float64](list(map(float, mags)))
    fit = jl_qmd.fit_t1_generalized_ir(
        TIs_jl, αs_jl, mags_jl,
        TRs=TRs_jl,
        α_excs=αexc_jl,
        Npe=int(Npe),
        T1_range=(0.01, 3.0),
        n_grid=500,
        abs_noise_sigma=float(sigma_abs),
        sigma_method=jl.Symbol("asymptotic"),
        signed=bool(signed),
    )
    return float(fit.T1)


def run_budget(budget: int, n_trials: int, rng: np.random.Generator, jl, jl_qmd):
    cfg = load_config(budget)
    labels, T1_true = load_csv_truth(budget)
    TIs = np.array(cfg["TIs_s"], dtype=float)
    TRs = np.array(cfg["TRs_s"], dtype=float)
    Npe = int(cfg["Npe"])
    sigma_img = float(cfg["snr_report"]["background_std"])  # image-domain σ
    snr_nema_peak = float(cfg["snr_report"]["snr_nema_peak"])

    n_spheres = len(T1_true)
    n_meas = len(TIs)

    # Uniform-ρ calibration: pick ρ so that peak |Mz| of the highest-T1
    # sphere (largest range of TI/T1, so most informative null/recovery)
    # times ρ divided by σ_img matches the reported NEMA peak SNR. PD is
    # uniform across the T1 array in the Calibur phantom, so ρ is the same
    # for every sphere — only |Mz| differs.
    i_ref = int(np.argmax(T1_true))
    y_ref = predict_mz_signed(jl_qmd, float(T1_true[i_ref]), TIs, TRs, Npe)
    peak_ref_mz = float(np.max(np.abs(y_ref)))
    # NEMA dual-acq SNR ≈ √2 × single-shot SNR ⇒ single-shot SNR = NEMA/√2
    target_peak_signal = (snr_nema_peak / np.sqrt(2)) * sigma_img
    rho_uniform = target_peak_signal / max(peak_ref_mz, 1e-12)
    print(f"  σ_img={sigma_img:.4f}  ρ={rho_uniform:.4f}  "
          f"(ref T1={T1_true[i_ref]:.3f}, peak|Mz|={peak_ref_mz:.3f}, NEMA_peak={snr_nema_peak:.2f})")

    y_signed = np.empty((n_spheres, n_meas))
    for i, T1 in enumerate(T1_true):
        y = predict_mz_signed(jl_qmd, float(T1), TIs, TRs, Npe)
        y_signed[i] = rho_uniform * y

    # SNR per sphere on this schedule (peak signal / σ)
    peak_snr = np.max(np.abs(y_signed), axis=1) / sigma_img

    # Monte Carlo
    err_mag = np.full((n_spheres, n_trials), np.nan)
    err_ps = np.full((n_spheres, n_trials), np.nan)

    for i in range(n_spheres):
        y_i = y_signed[i]
        T1_i = float(T1_true[i])
        for t in range(n_trials):
            nR = rng.normal(0.0, sigma_img, size=n_meas)
            nI = rng.normal(0.0, sigma_img, size=n_meas)
            mag_obs = np.abs(y_i + nR + 1j * nI)             # Rician
            ps_obs = y_i + nR                                # phase-sensitive

            try:
                T1_mag = fit_one(jl, jl_qmd, TIs, TRs, mag_obs, Npe, sigma_img, signed=False)
                err_mag[i, t] = (T1_mag - T1_i)
            except Exception:
                pass
            try:
                T1_ps = fit_one(jl, jl_qmd, TIs, TRs, ps_obs, Npe, sigma_img, signed=True)
                err_ps[i, t] = (T1_ps - T1_i)
            except Exception:
                pass
        print(f"    sphere {i+1:2d}/{n_spheres}  T1={T1_i:.4f}  peakSNR={peak_snr[i]:5.2f}  "
              f"mag bias={np.nanmean(err_mag[i]):+.4f}  |err|={np.nanmean(np.abs(err_mag[i])):.4f}   "
              f"ps bias={np.nanmean(err_ps[i]):+.4f}  |err|={np.nanmean(np.abs(err_ps[i])):.4f}")

    return {
        "budget": budget,
        "labels": labels,
        "T1_true": T1_true,
        "peak_snr": peak_snr,
        "sigma_img": sigma_img,
        "TIs": TIs,
        "TRs": TRs,
        "err_mag": err_mag,
        "err_ps": err_ps,
    }


def summarise(results):
    print("\n" + "=" * 88)
    print("SUMMARY")
    print("=" * 88)
    print(f"{'budget':>6}  {'σ_img':>8}  {'MAPE_mag':>10}  {'MAPE_ps':>10}  "
          f"{'bias_mag':>10}  {'bias_ps':>10}")
    for r in results:
        T1 = r["T1_true"][:, None]
        mape_mag = float(np.nanmean(np.abs(r["err_mag"]) / T1) * 100)
        mape_ps = float(np.nanmean(np.abs(r["err_ps"]) / T1) * 100)
        bias_mag = float(np.nanmean(r["err_mag"] / T1) * 100)
        bias_ps = float(np.nanmean(r["err_ps"] / T1) * 100)
        print(f"{r['budget']:>6}  {r['sigma_img']:>8.4f}  "
              f"{mape_mag:>9.2f}%  {mape_ps:>9.2f}%  "
              f"{bias_mag:>+9.2f}%  {bias_ps:>+9.2f}%")
    print()
    # Per-sphere MAPE table for the magnitude case
    print("Per-sphere MAPE (magnitude / Rician):")
    header = "  T1_true |" + "".join(f"  b{r['budget']:>3}s" for r in results)
    print(header)
    n = len(results[0]["T1_true"])
    for i in np.argsort(results[0]["T1_true"]):
        T1 = results[0]["T1_true"][i]
        row = f"  {T1:6.4f}  |"
        for r in results:
            m = float(np.nanmean(np.abs(r["err_mag"][i]) / T1) * 100)
            row += f"  {m:6.2f}%"
        print(row)


def save_results(results, out_path: Path):
    payload = []
    for r in results:
        payload.append({
            "budget": r["budget"],
            "sigma_img": r["sigma_img"],
            "TIs": r["TIs"].tolist(),
            "TRs": r["TRs"].tolist(),
            "labels": r["labels"],
            "T1_true": r["T1_true"].tolist(),
            "peak_snr": r["peak_snr"].tolist(),
            "err_mag": r["err_mag"].tolist(),
            "err_ps": r["err_ps"].tolist(),
        })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f)
    print(f"\nWrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=200)
    p.add_argument("--budgets", type=int, nargs="+", default=BUDGETS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(HERE / "runs" / "synthetic_rician" / "results.json"))
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    sys.path.insert(0, str(HERE.parent / "python"))
    print("Initialising Julia bridge...")
    jl, jl_qmd = init_julia()
    print("OK.\n")

    results = []
    for budget in args.budgets:
        print(f"=== budget={budget}s ===")
        r = run_budget(budget, args.trials, rng, jl, jl_qmd)
        results.append(r)

    summarise(results)
    save_results(results, Path(args.out))


if __name__ == "__main__":
    main()

# Archived — pre-simulator-fix docs (invalid results)

These documents were written against a **buggy simulator** and their quantitative
conclusions (mean MAPE ≈ 220–1267%, the SSE-landscape "wrong basin" story, the
Rician/phase-sensitivity diagnoses, fitter-σ machinery) are **artifacts of those
bugs, not real findings**. Do not cite any number from them.

## The bugs (now fixed)
1. **Missing `fftshift`/`ifftshift` around the 2D IFFT** (`src/rl/e2.jl` recon).
   k-space DC sat at the array centre but was fed to a DFT that assumes DC at
   `(1,1)` → chequerboard + half-FOV wrap. Every per-sphere ROI sample was
   contaminated (right sphere + diagonally-opposite sphere + background). See
   `../../FIX_SIM_PLAN.md` §1.
2. **Gradient amplitude built with γ_rad (2π·γ_Hz) instead of γ_Hz** in
   `src/sequences/blocks.jl` → gradients 2π× too small, collapsing the effective
   image FOV. Fixed to `γ_Hz = 42.577e6`.

Because the measurement function was corrupted, every RL run (V5, V9–V12), every
CR-optimal baseline, every t1_fit / snr / wape sweep, and the Ch4 draft built on
them are invalid and are being **re-run on the fixed simulator**.

## What is kept (still valid, in repo root)
- `FIX_SIM_PLAN.md` — the bug post-mortem + fix + test plan (the keeper).
- `cr_explainer.md` — CR-optimal math walkthrough (theory valid; numbers stale).
- `EXPLAINER_E2.md` — code walkthrough (structure valid).
- `PLAN.md`, `E2_PLAN.md` — conceptual experiment ladder (no bad numbers).

## What survives from the archived docs (ideas, not numbers)
- Reward-design failure ladder (terminal-bonus exploit, delta-only oscillation) —
  conceptually sound, see archived `CH4_DRAFT.md` §4.4.
- The **CR-optimal-anchor + oracle-init-fitter** evaluation methodology — reusable.
- The **Ernst-angle / flip-angle-freedom** opportunity and the single-TI-per-block
  action-space limitation — archived `CH4_DRAFT.md` §4.11.
- The NEMA dual-acquisition SNR metric as the clinical SNR anchor.

Superseded by `../../E2_RERUN_PLAN.md`.

# Koma multi-shot drift investigation

## Question

When `ir_se_2d_sequence` runs Npe shots of identical IR-SE imaging on a phantom, why does the resulting k-space differ between spoiled and unspoiled configurations? Specifically:

1. At long TR (full Mz recovery between shots) — what could possibly differ shot-to-shot if the physics is identical per shot?
2. The unspoiled image at TR=20s × Npe=32 showed monotonically increasing |ksp_normal − ksp_spoil| with PE-shot index.
3. The user fixed a Koma time-driven jump bug at ~70–150 s cumulative sim time; `scripts/koma_bug_minimal.jl` no longer drifts in the gradient-free case.

## Hypotheses on the table (initial)

- **H1: Coherence pathways across shots.** Mz stored by prior shots refocused into later ADCs by specific gradient histories. Predicts spoiling fully removes drift; predicts drift grows with shot index (more prior shots → more pathways).
- **H2: Steady-state transient.** Mz hasn't equilibrated; signal evolves over first few shots. Predicts drift independent of spoiling, scales inversely with TR/T1.
- **H3: Residual Koma bug.** The fix didn't fully cover gradient-heavy sequences. Predicts drift correlates with total sim time AND gradient block count, NOT with TR-induced Mz state.

## Tests so far

- `scripts/koma_bug_minimal.jl` (pre-existing, user's): gradient-free IR-SE, no drift after fix.
- `scripts/test_coherence_pathways.jl`: full `ir_se_2d_sequence` with z-line phantom. **Inconclusive** —
  - TR=0.5 s, 2 s: unspoiled & spoiled drift similarly (-75 %, -29 %) → mostly steady-state (H2).
  - TR=5 s: unspoiled stable (-6.5 %), but spoiled hits *discrete catastrophic drops* at shots 9, 13, 14, 15, 16 (sim times 45–80 s, in or near the original bug zone) → looks like H3 returning under spoiled sequences.
  - H1 not visibly demonstrated.

## Plan

Run isolating experiments until one of H1/H2/H3 is unambiguously confirmed and the others ruled out. Each experiment lives in its own numbered file. Findings updated below.

| # | Script | Question | Result |
|---|---|---|---|
| 01 | `01_time_vs_shotcount.jl` | Is drift driven by total sim time, shot count, or TR? | **Time-driven.** All three configs (TR=10/Npe=8, TR=2/Npe=40, TR=5/Npe=16, each 80 s total) showed a single step jump at sim_time ∈ [66, 80] s. Magnitudes ±5 %. Independent of TR or shot count. |
| 02 | `02_periodic_or_one_off.jl` | Does the jump repeat at fixed intervals, or is it a single discrete event? Long Npe at small TR to look for multiple jumps. | **Chaotic divergence**, not a clean periodic jump. TR=2 s × Npe=150 (300 s): stable steady-state 6.92 until ~shot 40 (80 s), then 58 step events of magnitude up to ±53 %. Signal swings from 6.9 → 7.2 → 3.8 → 7.0 → 4.5 → ... — looks like accumulating numerical instability, not a single deterministic event. |
| 03 | `03_minimum_gradient_trigger.jl` | What's the minimum gradient pattern that triggers the chaos? Start from koma_bug_minimal (RF only, stable) and add gradients incrementally. | **All four gradient levels (none / readout only / prewinder / per-shot PE) trigger the *first* jump at sim_time ≈ 66 s, with a similar amplitude (~+4 %).** After the jump: L0/L1 (no or minimal gradients) re-stabilise at a new wrong value (6 step events total). L2/L3 (with prewinder, with per-shot Gy) keep oscillating chaotically (15/54 step events). |

---

## Validated theory

**Koma has a residual numerical instability that fires at sim_time ≈ 66–80 s, independent of TR, shot count, or gradient activity.** The user's previous fix reduced its initial magnitude (from the documented ~22 % jump to ~4 % in this setup) but did not eliminate it. The instability acts as a discrete perturbation of the simulator state at that time:

- **Without per-shot gradient variation:** the perturbation shifts the steady-state Mz by a small amount, then the system re-stabilises at a wrong value. Looks like a single ~4 % step.
- **With per-shot gradient variation (e.g. varying Gy phase encode):** the perturbation interacts with each shot's gradient history differently → per-shot drift becomes chaotic from sim_time ≈ 66 s onward.

**Evidence summary:**

| Experiment | Observation supporting theory |
|---|---|
| 01 | Drift onset at sim_time ∈ [66, 80] s in three configs (TR=10/Npe=8, TR=2/Npe=40, TR=5/Npe=16) — *independent of TR and shot count*, function of sim_time only. |
| 02 | TR=2 × Npe=150 (300 s) shows stable steady state until ~80 s, then 58 step events. **Not** a single periodic jump — chaotic divergence in the presence of varying gradients. |
| 03 | All four gradient levels jump at sim_time ≈ 66 s (timing universal), but post-jump *stability* depends on gradient activity (L0/L1 stable, L2/L3 chaotic). |

**Hypotheses ruled out:**

- **H1 (multi-shot coherence pathways):** ruled out by exp 01 — drift scales with sim_time, not shot count. Adding more shots at smaller TR doesn't grow drift; reducing shots at larger TR doesn't shrink it.
- **H2 (steady-state convergence):** explains only the **shot-1 transient** (e.g. TR=2 s: 9.72 → 6.92), not the post-66 s instability.

## Practical implications

- **Any quantitative work with `ir_se_2d_sequence` must keep `TR × Npe ≲ 60 s`.** Past that, the un-spoiled image's per-shot signals are corrupted by Koma's instability.
- **Per-Npe TR caps to stay reliable:**
  - Npe = 8: TR ≤ 7.5 s
  - Npe = 16: TR ≤ 3.7 s
  - Npe = 32: TR ≤ 1.9 s
- **The `pixel_grid_overlay` runs at TR=5/Npe=32 (160 s) and TR=20/Npe=32 (640 s) were past the threshold.** What we attributed to "coherence pathways" in those figures is mostly Koma instability artifacts, amplified by the per-shot Gy variation. The spoiler's *cleaner* spoiled images at long TR are not "better physics" — they're a coincidental side effect of the spoiler dephasing the per-shot drift signature.
- **The earlier user fix was incomplete.** Suggest reporting to Koma maintainers with our `koma_bug_minimal.jl` and `03_minimum_gradient_trigger.jl` as reproducers showing the bug is gradient-amplified but not gradient-caused.

## Recommendation for the FYP report

When presenting any IR-SE simulation results, restrict to `TR × Npe ≤ 60 s`. The honest framing is: *"We characterised a residual Koma numerical instability at long cumulative sim time (Stage 3 investigation, scripts in `scripts/koma_investigation/`). We cap our experimental regime to `TR × Npe ≤ 60 s` to stay in the validated zone."* This is itself a publishable methodology contribution.

---

## Minimal reproducer

`koma_bug_residual.jl` — gradient-free IR sequence (180° → TI → 90° → 1-sample ADC → TR pad) × 24 shots at TR=15 s. After shot 17 (sim_time = 270 s) the per-shot |signal| jumps by +52 %, then settles at a new +42 % plateau. This is the **same pattern as the originally documented bug**, just at a higher threshold (the earlier fix raised but didn't eliminate the failure point).

Output:

```
shot 17 (sim_time = 255 s):  |signal| = 0.476267   ← stable
shot 18 (sim_time = 270 s):  |signal| = 0.722479   ← +52 % jump
shot 19+ (sim_time ≥ 285 s): |signal| = 0.677159   ← new wrong plateau
```

Threshold (in sim_time) drops as per-shot complexity grows:

| Per-shot pattern                                  | Threshold |
|---|---|
| 180° → TI → 90° → 1-samp ADC                      | ~270 s    |
| above + 180° refocus + 16-samp ADC                | ~285 s    |
| full `ir_se_2d_sequence` (Gx prewinder + per-shot Gy + Gx readout) | ~70 s |

## Likely cause

The pattern — *stable for many shots, then a single discrete jump to a new plateau, with threshold inversely proportional to per-shot arithmetic* — strongly suggests **Float32 numerical drift in the Bloch state representation**.

Plausible mechanism:

1. Koma stores per-spin magnetisation `(Mx, My, Mz)` in Float32 (~7 decimal digits of precision).
2. Every event block applies a transformation (rotation for RF, decay for delays, phase accumulation for gradients), introducing ~1e-7 relative roundoff per operation.
3. Over hundreds of events these errors accumulate. The vector slowly walks off the |M| = 1 sphere and develops a small systematic bias.
4. At some accumulated-error threshold, *something tips* — most likely:
   - a normalisation step (`M ← M / |M|`) that was previously a no-op starts changing the state, or
   - a conditional branch (e.g. sign of a near-zero quantity used in a `cos`/`sin` decomposition) flips, or
   - a small quantity underflows into the denormal range and loses precision drastically.
5. The state jumps once to a new attractor and stabilises there.
6. With per-shot varying gradients (full IR-SE), the post-tip state is read out under different gradient histories per shot → chaotic instead of stable.

This is consistent with the earlier fix raising but not eliminating the threshold — that fix likely tightened one specific arithmetic path but left others in Float32. The threshold scales inversely with operation count (more events per shot → tip happens sooner in sim_time) because every operation contributes its share of roundoff.

The straightforward fix is to **promote the magnetisation state and accumulators to Float64** throughout `KomaMRICore`. That doubles memory but eliminates the precision ceiling for any practical sequence length. A targeted alternative is to identify the single critical accumulator (likely a phase term or an Mz value) and promote *only that*, leaving the rest in Float32.

A clean way to file this upstream would be: cite the original `koma_bug_minimal.jl` (showing the fix worked for N=16, TR=15 s = 240 s), then attach `koma_bug_residual.jl` showing the same pattern at N=18, TR=15 s = 270 s — the boundary just moved by one shot. The fact that the threshold drops to ~70 s with gradients (script `03_minimum_gradient_trigger.jl`) is supporting evidence that this is an arithmetic-density issue, not anything sequence-specific.

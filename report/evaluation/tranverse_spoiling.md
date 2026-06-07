
<!-- ============================================================
RELOCATE TO THE EVALUATION CHAPTER.
Quantitative justification for the perfect-spoiling assumption in the
IR-SE forward model, measured on the FIXED simulator. Numbers from the
7ceced7_fixed t1_fit_vs_true CSVs; per-sphere deltas computed from them.
============================================================ -->

## Evaluation aside: does the transverse-spoiling assumption matter?

Our IR-SE forward model assumes **perfect spoiling** — no transverse magnetisation
survives between shots, so each shot is an independent inversion-recovery measurement
and the monoexponential model is exact. Having wrongly suspected imperfect spoiling
during debugging, it is worth quantifying — now that the simulator is trusted — how
much spoiling *actually* changes the result.

The test is direct: run the no-noise $T_1$ validation on the **fixed** simulator
twice, with and without an explicit per-shot gradient spoiler, and compare the
recovered maps. If the assumption holds at our repetition times ($\text{TR}\gg T_1$,
near-complete $M_z$ recovery), the two should be indistinguishable.

| Configuration (fixed Koma, no noise) | Mean $T_1$ MAPE | Spheres changed by spoiling |
|---|---|---|
| No spoiler | **0.48 %** | — |
| Gradient spoiler per shot | **0.43 %** | **1 of 14** (only the $T_1=0.38\,$s sphere, by $-1.1\%$) |

Spoiling changes the recovered $T_1$ by *nothing* on 13 of 14 spheres — bit-for-bit
identical fits — and by barely 1 % on the last. The mean MAPE moves 0.48 % → 0.43 %,
within the fit's own numerical noise. At our operating point ($\text{TR}\ge5\,$s,
$T_1\le1.9\,$s) there is simply no residual transverse coherence to spoil: $M_{xy}$
has fully dephased and decayed under $T_2/T_2^\*$ long before the next inversion. **The
perfect-spoiling assumption is therefore validated empirically, not merely asserted**
— we can drop the spoiler gradient (saving sequence time and gradient load) with no
measurable loss of $T_1$ accuracy.

This also explains the misleading clue during debugging. On the *buggy* simulator,
spoiling appeared to change the images and shifted the MAPE (39 % → 44 %) — but that
was never physical: the spoiler simply dephased the *bug's* corrupted per-shot
signature differently, so it looked like it was doing something. On the fixed
simulator it does essentially nothing, which is the correct physical behaviour and
confirms the earlier "spoiling matters" reading was an artefact of the bug, not the
physics.

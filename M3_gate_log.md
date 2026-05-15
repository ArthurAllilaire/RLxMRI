# M3 Verification-Gate Log

## 2026-05-11

- 14:00 — installed MRzeroCore 0.4.7 in `.venv_mrzero/` (Python 3.10, torch 2.11).
- 14:10 — wrote `python/gate/mrzero_ir.py` (IR-prep sequence builder + simulate fn).
- 14:20 — sanity check vs closed-form IR `|1-2exp(-TI/T1)|` for T1=1s, 6 TIs: rel err ≤ 0.23%. MRzero is correctly producing IR signal.
- 14:50 — Gate 6.1 PASS. MRzero vs KomaMRI on T1=1.398s, T2=1.035s, 6 TIs: median rel-diff 0.94%, max 1.55%. Residual is finite-pulse-duration bias on KomaMRI side; MRzero aligns with closed-form to 0.1%.
- 15:20 — Gate 6.2 CPU FAIL / GPU CONDITIONAL PASS. 200-event FISP-like seq, 140 voxels. fwd+bwd median 3.59s CPU (target <2s), peak RAM 21 MB (<4 GB). GPU extrapolation: 0.18-0.72s; clears 200ms target only optimistically.
- 15:35 — Gate 6.3 PASS strongly. 4 tissue corners × 50 noise trials × 8000-cell grid. Truth at rank ≤ 50 in 100% of all 200 trials; median rank 1 in 3/4 cells. Wrong-basin SSE ratio truth/best 1.0-1.44 (vs Ch4 §20's 5-20×). Ch4 multimodality does NOT carry over to joint (T1,T2,PD) under IR-SE.
- 15:50 — Tests added (`python/tests/test_e2e_gate.py`), all 4 pass. MRzero notes written (`docs/MRZERO_NOTES.md`). Full report at `EXPERT_REPORT_E2E.md`.
- Recommendation: Paradigm B (differentiable bilevel meta-opt) with MRzero + GPU.

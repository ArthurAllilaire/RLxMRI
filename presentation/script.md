# Full Presentation Script — Adaptive qMRI Sequence Design with RL

**18 min + 10 min Q&A.** Problem-driven narrative · RL headline · live phantom demo · Run A = dedicated limitation slide.

Speaker notes are bullets, not a verbatim read. **`[FIG]`** marks the figure to show; **`⚠FIG`** flags a figure that is weak for a *talk* and what to make instead (consolidated list at the end). Timings are cumulative targets.

---

## ASSESSMENT ALIGNMENT (FYP rubric — keep these signals visible)

The talk is judged on the same four dimensions as the report. ~70% of the marks (Execution 50% + Evaluation 20%) reward technical **rigour** and **critical self-evaluation**, not results alone — so the deck deliberately foregrounds *how* we evaluated and *what doesn't generalise*, not just the headline numbers.

| Dimension | Wt | Where it shows up | Signal to land |
|---|---|---|---|
| Framing of problem | 15% | Slides 2–3 | The gap stated as a **tension**: adaptive qMRI exists but is *model-based only*; RL in MRI exists but is *non-quantitative* — nobody has combined them. FIG-2 quadrant. |
| Execution & technical quality | **50%** | Parts 1–3 (esp. 6–13) | Validated pipeline (upstream PRs), multi-fidelity *insight* (cached-water linearity), **fair evaluation**: strict held-out seeds, 24 episodes, non-overlapping CIs, matched budget + same fitter for all arms. Reproducibility: open-source pkg, ~373 tests. |
| Evaluation & reflection | 20% | Slide 11 (Run A limitation), 12 (TR-matched control = ruling out an alternative explanation), 15 (broader impact + what doesn't generalise) | Honest negative result; threats to validity; clinical tie-back. |
| Communication | 15% | Whole deck | Problem-driven structure, signposted roadmap, figures that *enhance* understanding (FIG-2/3/8). |

**Backup deck = "anticipating objections"** (an explicit 70–85% Execution signal). Pre-load the likely examiner challenges and where they're answered: fitter confound, single training seed, simulator-only, LSTM compute caveat.

---

## PART 0 — FRAME THE PROBLEM (target 0:00 → 2:30)

### Slide 1 — Title (0:00–0:15)
- "Adaptive quantitative MRI sequence design with reinforcement learning."
- Name; supervisors Andreas Wetscherek (ICR) and Wayne Luk (Imperial); BEng FYP.
- One line: *"Can an agent learn to design MRI scans on the fly, better than a fixed protocol?"*

### Slide 2 — Why this matters: the scan-time bottleneck (0:15–1:30)
**`[FIG]`** clinical motivation — MR-Linac / MRgRT photo or simple icon. *(No existing figure; ⚠FIG-1: grab one clean MR-Linac image, or skip and use a qMRI map.)*
- MRI gives the soft-tissue contrast that guides radiotherapy (MR-Linac), but **acquisition time is the clinical bottleneck** — longer scans mean motion artefacts and less on-table adaptation.
- **Quantitative MRI (qMRI)** goes further than a pretty picture: it measures *physical tissue constants* — T1, T2 — by acquiring several images at different timings and fitting a relaxation curve. That's even more scan time.
- Today those timings are a **fixed protocol**: the same inversion times for every patient, chosen in advance.
- The catch: the *most informative* timing depends on the tissue you're measuring — **which is exactly what you don't know yet.**

### Slide 3 — The adaptive idea, the gap, and the thesis (1:30–2:30)
**`[FIG]`** positioning quadrant. **⚠FIG-2: the report has this only as a dense 6-row table (`tab:e2-related-position`). For the talk, make a 2×2 quadrant** — axes *fixed → adaptive* and *non-quantitative → quantitative* — with prior work in three corners and a star in the empty one ("learned + adaptive + quantitative").
- **Adaptive MRI:** choose the next acquisition from the current estimate. Shown to accelerate qMRI up to ~2.5× (Beracha et al.).
- **The tension (state it sharply):** adaptive qMRI exists — but only as a hand-derived **Bayesian model** (Beracha). RL *has* controlled MRI scanners — but only for **non-quantitative** goals (k-space sampling, shape classification — Pineda, Walker-Samuel). **Nobody has put a *learned* policy on the *quantitative* objective.** That empty corner is the project.
- **Thesis / novelty:** to our knowledge the **first RL agent for adaptive quantitative MRI** — it conditions each acquisition on the current *fitted T1*, and is trained and scored directly on T1 error.
- **Roadmap (say while pointing):** doing this in simulation needs three things — a trustworthy **phantom** (A1), a **validated simulator** (A2), and the **RL** itself (A3). The first two are enablers; the RL is the result.

---

## PART 1 — A1: THE DIGITAL TWIN (target 2:30 → 5:30)

### Slide 4 — A known phantom to learn from (2:30–3:30)
**`[FIG]`** `imgs/mri_system_phantom/Premium-System-Phantom.png` (left, hardware) + `imgs/mri_system_phantom/T1_voxel_phantom.png` (right, voxelised twin coloured by T1). *Good figure — keep.*
- The agent learns entirely in simulation, so it needs a **known** object with ground-truth T1/T2. We use the **QalibreMD Model 130** — the standard NIST/ISMRM calibration phantom: 14 spheres spanning the clinical T1 range (~24 ms to 1.9 s).
- I built **`MRISystemPhantom.jl`**: a configurable, open-source digital twin that returns standard KomaMRI objects, plus the ground-truth labels, the T1 fitter, and per-episode randomisation the RL loop needs.
- One design decision matters downstream: the background water (~**80% of all spins**) sits on its **own coarsenable grid**, separate from the spheres — this is the lever that later makes RL training affordable.
- **Reproducible by construction:** shipped as a tested open-source Julia package (~**373 tests**, API docs), built on one serialisable `PhantomConfig → build_phantom` contract with an embedded RNG seed — so any result in the talk regenerates from a config. *(This is a deliberate reproducibility contribution, not just code.)*

### Slide 5 — LIVE DEMO: build a phantom and look at it (3:30–5:30)
**`[DEMO]`** run a Julia script live (`examples/plot_phantom_3d.jl` / `t1_mapping.jl`). Show: (1) build from a `PhantomConfig`, (2) the 3-D voxelised plate coloured by T1, (3) one simulated IR signal / reconstructed image.
- Narrate: "this is a real `KomaMRI.Phantom` — I can hand it straight to the Bloch simulator."
- Show the T1 sweep / recovery curve if quick.
- **Fallback:** pre-captured screenshots already on the slide; if the build stalls, talk over the stills and move on. **Hard stop at 5:30** — do not let the demo overrun.
- ⚠ Demo risk note: rehearse the exact commands; have the Julia session pre-warmed (JIT) before the talk so the first call isn't a 30 s compile.

---

## PART 2 — A2: VALIDATING THE SIMULATOR (target 5:30 → 8:30)

### Slide 6 — A plausible image is not a correct measurement (5:30–6:45)
**`[FIG]`** `imgs/komaMRI/buggy_t1_fit_vs_true.png` — fitted vs true T1, points off the diagonal, "39.4% MAPE". *Punchy, keep.*
- If the agent learns from a simulator, the simulator must be **quantitatively** right, not just visually plausible. KomaMRI is peer-reviewed and benchmarked — but only on **short** reference sequences.
- I validated the way qMRI actually demands: **simulate a known phantom, fit T1, compare to ground truth, no noise.** It should be near-exact.
- It wasn't: **39.4% mean error**, some spheres near 100%. Something was systematically wrong — and crucially, **the images still looked fine.**
- The hard part was diagnostic: the symptom looked exactly like an MRI-physics effect (imperfect spoiling). **Adding spoilers made it *worse* (39.4 → 44.0%)** — ruling that out. Longer TR also made it worse → the error was driven by **total elapsed time**, not physics.

### Slide 7 — Two floating-point bugs in KomaMRI, found & fixed (6:45–8:30)
**`[FIG]`** the floating-point mechanism schematic. **⚠FIG-3: this is the "aha" of the whole chapter and has NO purpose-built figure.** Make a simple number-line: two RF-edge markers `t1` and `t1+ε` distinct at small t, then **merging** as Float64 spacing `eps(t)≈t·2⁻⁵²` grows past the fixed `ε=10⁻¹⁴` around **t≈128 s**. Optionally pair with the k-space-collapse panel from `buggy_pixel_grid_overlay.png` (cropped — the full 3-row figure is too dense for a slide).
- The decisive diagnostic was to look at **k-space directly**: it collapsed after a *fixed cumulative time*, regardless of shot count.
- **Mechanism:** KomaMRI marks a sharp RF edge with a *fixed absolute* offset `ε=10⁻¹⁴ s`. But Float64 precision is *relative* — the gap to the next representable number grows as `eps(t)≈t·2⁻⁵²`. Past **~128 s**, `t + ε == t` bit-for-bit: the two edge markers **collapse**, one is deduplicated, and the integrator applies the wrong RF for one sample — **5% of a 1 ms pulse → a 20–25% signal jump → biased T1.**
- Reduced each to a **minimal reproducer**, traced it, fixed it, and contributed upstream: **PR #780 (merged)** and **#789 (open)**. A second, subtler variant of the same bug appeared after time-rebasing.
- **Result: 39.4% → 0.48% mean error** (max 1.2%). To our knowledge this floating-point failure mode hasn't been reported for an MRI simulator — and it bites *exactly* in the long-sequence regime RL training produces. **This is what makes everything in Part 3 trustworthy.**

---

## PART 3 — A3: ADAPTIVE RL (the headline, target 8:30 → 17:00)

### Slide 8 — The RL formulation (8:30–10:00)
**`[FIG]`** the env-loop diagram (`fig:e2-env-loop`, the TikZ figure). *Already vector/clean — keep, but consider simplifying labels for a slide.*
- One episode = a 2-D inversion-recovery scan through the T1 plate, under a fixed **scan-time budget** (240 s).
- **State:** the running per-sphere T1 estimates + budget used. The agent never sees pixels — only the fitter's estimates, the interface a real qMRI pipeline exposes.
- **Action:** the next block's timings — inversion time TI (and TR).
- **Reward:** *dense* — the per-step improvement in log-error, `Δlog MAPE`. (Why dense: an earlier single-voxel version with a big terminal bonus **collapsed to a fixed policy** — it just collected the bonus. Lesson baked in.)
- **Stack:** Gymnasium + Stable-Baselines3 **PPO**, Python ↔ Julia via juliacall with the simulator held **in-process** across steps.
- Noise is calibrated, not arbitrary: σ=50 → **SNR ≈ 17–28**, squarely in clinical IR range.

### Slide 9 — C2: the cost wall and the multi-fidelity ladder (10:00–11:45)
**`[FIG]`** a fidelity-ladder staircase. **⚠FIG-4: the report has this as a table (`tab:e2-fidelity-ladder`).** For the talk, draw a **staircase** (analytic → cached3 → cached → full3 → full) with cost rising left-to-right and a little "water spins" bar shrinking. Pair with a small **decomposition cartoon**: `S_full = S_spheres + S_water`, water computed *once* and cached.
- The wall: a full-Bloch step is ~4 s on CPU → a 200k-step run is **~9 days** before any evaluation. RL normally trains on analytic environments that are millions of times cheaper.
- Key observation: **80% of the spins are background water** the agent doesn't even measure.
- Two levers, both on the water: **coarsen** it (its own grid, proton density rescaled to conserve magnetisation) or **cache** it. Caching exploits **simulator linearity** — `S_full = S_spheres + S_water`, and the homogeneous water factorises so a new timing only rescales a template computed once. **~8× per-step saving**, matching full-Bloch T1 fits to **0.12%**.
- Net: a **~30× cost range** ladder, warm-starting one policy up the rungs.

### Slide 10 — The trust problem: bias-aware promotion (11:45–12:45)
**`[FIG]`** a small controller schematic. **⚠FIG-5: the report's controller diagram is commented out as "not polished enough" — and your `plan.md` TODO #1 flags exactly this.** Make a clean version: cheap rung trains → **probe on held-out FULL simulator** → promote only when the cheap rung plateaus *or stops ranking policies like the full sim*. The one message: **"a cheap rung must never grade its own homework."**
- A cheap, biased rung can keep *raising its own score* by exploiting its bias. So **promotion and checkpoint selection are scored on a held-out full-simulator probe**, never the cheap rung's own number.
- Promote on signals like *target plateau* and *ranking breakdown* (Spearman vs the full sim drops). And keep a **global-best checkpoint** on the target fidelity — because, as we'll see, the *last* policy in a curriculum is often **not** the best one.
- (Honest note for Q&A: these thresholds are hand-tuned, not derived — flagged as future work.)

### Slide 11 — DEDICATED LIMITATION: when does adaptivity *not* help? (Run A) (12:45–14:00)
**`[FIG]`** `imgs/e2_rl/runA_moneyplot.png` + `imgs/e2_rl/runA_ti_per_episode.png`. *OK, but ⚠FIG-6: the left "money plot" axis labels are small; enlarge fonts for projection. The right TI-scatter is the useful one (shows the policy is non-collapsed).*
- Run A = the **full 14-sphere plate**: spheres deliberately span the *entire* clinical T1 range. This is the **hardest case for adaptivity** — one well-chosen fixed schedule already serves a near-fixed fleet.
- Result: the **fixed Cramér–Rao schedule wins** (4.70% vs the agent's 9.44–11.68%). An honest negative.
- **But the agent is genuinely adaptive, not collapsed:** its TI varies within and across episodes (most-used bin only 11.9%) and tracks the running T1 estimate. The longest sphere is simply beyond its TI reach — an action-coverage ceiling.
- Two lessons that carry forward: (1) global-best checkpointing is *necessary* (a later stage eroded a better earlier policy); (2) **to test adaptivity you need a task with something to adapt to.** → motivates Run B.

### Slide 12 — Run B: adaptivity helps where there's variation (14:00–15:15)
**`[FIG]`** `imgs/e2_rl/runB_mape_comparison.png` (bar chart, RL vs fixed). *Good for a talk — verify font size, ⚠FIG-7 minor.*
- *(Say once, here — it covers every result slide:)* **how we compare fairly** — strict held-out seeds never used in training/selection, 24 episodes, 95% CIs, the **same scan budget and the same fitter** for the RL policy and every fixed baseline. So a win is a win on the estimation objective, not a tuning artefact.
- Run B is engineered so adaptation can matter: **5 spheres, T1 resampled continuously every episode** — the variation a fixed grid can't anticipate.
- **RL 4.62% vs best fixed log-grid 6.86%** at 240 s, non-overlapping CIs. The Cramér–Rao schedule, solved for a *nominal* fleet, **collapses (15.96%)** under per-episode variation — static-optimal vs static-average.
- **Is the gain just more measurements?** No: re-solve the fixed grid at the policy's *own* shorter TR and it does **worse**, not better. **The gain is adaptive placement, not block count.**
- Behaviour is a **two-phase "probe-then-refine"**: open with a long high-TI inversion probe informative across the range, then shorten TI once the estimate stabilises. Advantage holds at 560 s too (4.16% vs 6.04%).

### Slide 13 — The strongest finding: what should the policy *remember*? (15:15–16:30)
**`[FIG]`** `imgs/e2_rl/runB_memory_mape.png` (bar + CIs, the money result). Optionally add the behavioural contrast. **⚠FIG-8: the most *interesting* point — LSTM tracks the estimate (r=+0.78) and loses; histogram tracks coverage (r=−0.12) and wins — has no single clean figure.** Consider a two-panel "TI vs current estimate" scatter (LSTM monotonic ramp vs histogram flat/coverage-driven) to make it land.
- This is a partially-observed problem: the estimate is a *lossy* summary of *which* TIs you've already bought. So I compared three ways to carry that history: fitter **uncertainty (σ-channel)**, a learned **LSTM**, and an order-invariant **histogram of executed TIs**.
- **Result — `histogram ≫ no-memory > σ-channel > LSTM`.** The TI-coverage histogram is the **best result in the project: 2.93% MAPE, 95.8% of episodes under 5%**, vs 6.04% for the best fixed schedule. CI doesn't overlap any other arm.
- The twist: the **LSTM learned the *intuitive* rule** — longer TI for longer estimated T1 (r=+0.78) — and it **lost**. The histogram conditions on *which TI bins are still unsampled* (r=−0.12 with the estimate) and won. *(LSTM caveat for Q&A: ~5× slower/step → undertrained under equal wall-clock; needs a step-matched rerun.)*
- Takeaway: **what you remember matters as much as whether you adapt** — and an explicit coverage statistic beats both a learned memory and the fitter's own uncertainty.

### Slide 14 — Positioning vs published results (16:30–17:00)
**`[FIG]`** `tab:e2-quant-anchors` distilled to **3–4 bars/rows**, not the full table. ⚠FIG-9: don't paste the LaTeX table; show only the comparable headline numbers.
- Not a like-for-like ranking — tasks differ — but: prior work has **stronger hardware validation**; this work contributes a **different object**: a closed-loop RL policy scored on fitted-T1 error in a *validated* 2-D pipeline.
- Using Beracha et al.'s own precision→time heuristic, the 6.04% → 2.93% improvement (~2.06× precision) ≈ **~4× acceleration** — top of their reported range, on a *harder* task (a whole fleet, not one voxel).

---

## PART 4 — CLOSE (target 17:00 → 18:00)

### Slide 15 — Contributions & future work (17:00–18:00)
**`[FIG]`** none, or a compact 3-icon recap (twin / bug-fix / RL).
- Three contributions: (1) an **open-source executable twin** of a calibrated phantom; (2) **validation-by-recovery** that found & fixed two upstream KomaMRI bugs (39.4 → 0.48%); (3) the **first RL agent for adaptive qMRI**, beating matched fixed schedules where variation exists — best result **2.93% vs 6.04%**, plus the research finding that *coverage memory beats estimate-tracking*.
- **Why it matters (close the loop with Slide 2):** less scan time at equal accuracy is exactly the bottleneck in time-critical qMRI like MR-Linac — the ~4× nominal acceleration sits at the top of the published adaptive-qMRI range, on a harder fleet task.
- **What I'm *not* claiming (reflection, not hedging):** this is simulator-only, one training seed per arm, and reward flows through a single fitter the policy could be co-adapting to — so the honest next step is a physical-phantom benchmark, not a stronger claim. The full 14-sphere plate still favours fixed coverage.
- Future work: widen the T1 range and action space; T2/PD plates; a greedy-CRLB Bayesian baseline (interpretable + calibrated uncertainty); benchmark the competing paper; step-matched LSTM.
- "Thank you — happy to take questions."

---

## BACKUP SLIDES (Q&A only — so nothing needs memorising)

- **Full results tables:** Run A (`tab:run-a-current-comparison`), Run B 240 s (`tab:run-b-240`), memory ablation (`tab:run-b-memory`), quant anchors (`tab:e2-quant-anchors`).
- **Reward screening** (`tab:e2-reward-screen`) — why `Δlog MAPE`; the degenerate-policy lesson.
- **Cached-water validation:** `imgs/e2_rl/water_t1_fit_4variants.png`, `water_cache_relerr_vs_TI.png` — 0.12% match, why analytic water (~40% error) is too crude.
- **Noise calibration:** `imgs/e2_rl/snr_mape_vs_sigma.png`, `snr_sweep_images.png` — σ=50 justification, graceful to σ≈85.
- **Water coarsening fidelity / Hamming:** `imgs/mri_system_phantom/water_voxel_fidelity.png`, `hamming_phantom_diff.png`.
- **Cramér–Rao baseline** — what the fixed comparator actually is (multi-start coordinate descent; static-optimal vs static-average).
- **Parallel envs / CPU speedup** (SubprocVecEnv-style rollout splitting) — *not in the report*; mention if asked about engineering.
- **Second KomaMRI bug** (closing-knot collapse after rebasing) + reproducer numbers (52% jump at shot 18).
- **Fitter grid floor** — ~1% quantisation at 300 points; why MAPE can't go below ~1%.
- **Run B policy behaviour** (`runB_240_ti_per_episode.png`, `runB_curriculum_probes.png`) — probe-then-refine, zero repairs.

---

## CONSOLIDATED FIGURE WORK (where to invest effort)

**General rule:** report figures are multi-panel and small-font; slides need *one message, big text, readable from the back*. Crop/enlarge most of them.

| # | Slide | Issue | Suggested new visual |
|---|---|---|---|
| FIG-1 | 2 | No clinical-motivation image | One clean MR-Linac/MRgRT image, or a qMRI T1 map. Low effort. |
| **FIG-2** | 3 | Novelty is a dense 6-row table | **2×2 positioning quadrant** (fixed↔adaptive × non-quant↔quant), star in empty corner. High payoff. |
| **FIG-3** | 7 | The float-collapse "aha" has no figure | **Number-line schematic**: two RF markers merging as `eps(t)` overruns fixed `ε` at ~128 s. Highest payoff in the talk. |
| FIG-4 | 9 | Fidelity ladder is a table | **Staircase** with rising cost + shrinking water-spin bar; + `S=S_spheres+S_water` cartoon. |
| ~~FIG-5~~ | 10 | **DONE** — your polished `presentation/controller_diagram.tex` (TODO #1) | Compiled to `figs/fig5_controller.png` (lualatex → gs), embedded in slide 10. |
| FIG-6 | 11 | Run A money-plot fonts too small | Enlarge axis labels; lead with the TI-scatter (shows non-collapsed). |
| FIG-7 | 12 | Run B bar chart | Check font size; otherwise fine. |
| **FIG-8** | 13 | LSTM-vs-histogram behaviour not shown | **Two-panel TI-vs-estimate scatter** (LSTM r=+0.78 monotonic vs histogram r=−0.12 coverage). Makes the headline finding land. |
| FIG-9 | 14 | Quant-anchor table too dense | Distil to 3–4 comparable bars/rows. |

**Done:** FIG-2 (capability matrix), FIG-3 (float-collapse), FIG-5 (controller diagram — from `controller_diagram.tex`).
**Remaining priority if time is short:** FIG-8 (memory finding: TI-vs-estimate scatter) > FIG-4 (fidelity ladder) > env-loop export (slide 8) > FIG-1 / FIG-9 (crops/imagery).

**Regenerating figures:** `python presentation/make_figs.py` (FIG-2, FIG-3). FIG-5: `lualatex controller_diagram.tex && gs -sDEVICE=pngalpha -r300 -o figs/fig5_controller.png controller_diagram.pdf`. Then `python presentation/build_deck.py`.

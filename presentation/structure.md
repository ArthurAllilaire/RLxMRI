# Presentation Structure — Adaptive qMRI Sequence Design with RL

**Format:** 18 min talk + 10 min Q&A. Live phantom demo included.
**Decisions:** RL (A3) is the headline · problem-driven narrative · live demo = phantom build + plots · 14-sphere limitation gets a dedicated framing slide.

**Open assumptions to confirm:**
- Audience = 2 examiners, mixed MRI background → keep the 2-min context genuinely introductory.
- Reuse existing `report_latex` figures wherever a slide needs a plot (noted per slide).

**Time budget (≈18:00):** Context 2:00 · A1 twin+demo 3:00 · A2 validation 3:00 · **A3 RL 8:30** · Conclusion 1:00 · buffer 0:30.

---

## Part 0 — Frame the problem (≈2:30, slides 1–3)

### Slide 1 — Title (0:15)
- Title, name, "Imperial BEng FYP", supervisors (Wetscherek/ICR, Luk/Imperial).

### Slide 2 — The problem: qMRI uses fixed, one-size-fits-all protocols (1:15)
- qMRI = measuring *quantitative* tissue parameters (T1, T2), not just pretty pictures.
- Today's sequences are **fixed protocols**: same inversion times / echo times for every patient and every tissue.
- But the *most informative* acquisition depends on the tissue you're trying to measure — which you don't know yet. → inherent inefficiency.
- *Figure:* IR signal recovery curves for two different T1 values, showing the best TI differs.

### Slide 3 — The vision + the gap + our thesis (1:00)
- **Adaptive idea:** choose the next acquisition from the current estimate of the tissue parameters.
- **Where prior work stops** (condensed Ch2): adaptive qMRI exists but derives its rule from a fixed Bayesian model; RL in MRI exists but targets *non-quantitative* goals (k-space sampling, shape classification).
- **Thesis / novelty claim:** the first RL agent for adaptive *quantitative* MRI — conditions each acquisition on the current fitted parameters, trained/evaluated directly on fitted-T1 error.
- **Roadmap (1 line):** to do this in simulation we needed three things → trustworthy phantom (A1), validated simulator (A2), the RL formulation (A3).

---

## Part 1 — A1: the digital twin (enabler) (≈3:00, slides 4–5 + demo)

### Slide 4 — Why a digital twin, and what it provides (1:00)
- RL learns entirely from simulation → need a *known* phantom with ground-truth T1/T2.
- `MRISystemPhantom.jl`: configurable open-source twin of the QalibreMD Model 130 system phantom; returns standard KomaMRI objects.
- Also ships: ground-truth parameter maps, the T1 fitting pipeline, and per-episode randomisation (pose + material jitter) the RL loop needs.

### Slide 5 — LIVE DEMO: build a phantom & look at it (2:00)
- Build the phantom live in Julia.
- Show geometry plot (spheres / plate layout) + a material map (T1 array).
- Run/show one simulated signal or image.
- **Fallback:** pre-captured screenshots on the slide in case the live build misbehaves.

---

## Part 2 — A2: validating the simulator (enabler) (≈3:00, slides 6–7)

### Slide 6 — A plausible image is not a correct measurement (1:15)
- If the agent learns from a simulator, the simulator must be *quantitatively* trustworthy, not just visually plausible.
- Validate by **parameter recovery**: simulate a known phantom, fit T1, compare to ground truth.
- The catch: errors only appeared at the **long cumulative sequence times** RL training produces — 39.4% mean error.

### Slide 7 — Two upstream KomaMRI bugs found & fixed (1:45)
- Reduced to minimal reproducers; traced to two floating-point **time-discretisation** bugs (RF edge-marker collapse; closing-knot collapse after time rebasing).
- Fixed upstream → validation error **39.4% → 0.48%**.
- Contributed back to a widely used simulator; this is what makes the downstream RL results trustworthy.
- *Figure:* before/after recovery plot or error table.

---

## Part 3 — A3: adaptive RL (the headline) (≈8:30, slides 8–14)

### Slide 8 — Environment formulation (MDP) (1:30)
- **State:** signals observed so far + current fitted parameter estimate.
- **Action:** next acquisition timing (TI / TE / flip angle).
- **Reward:** dense, driven by fitted-T1 error (note the E1 lesson — terminal bonus collapse — briefly, why dense).
- **Stack:** Gymnasium env, Stable-Baselines3 PPO, Python ↔ Julia (juliacall, Julia stays in-process).
- *Figure:* env loop diagram (agent ↔ KomaMRI sim ↔ fitter).

### Slide 9 — C2: the cost wall & the multi-fidelity curriculum (1:45)
- Bloch-in-the-loop is far slower than the analytic envs RL usually trains on → training is compute-bound ("cost wall").
- **Fidelity ladder:** cheap low-fidelity configs to learn, accurate sims reserved for validation.
- **Cached water:** exploit simulator linearity to avoid re-simulating background.
- **Bias-aware promotion (switch rule)** + global-best checkpointing across stages.

### Slide 10 — Headline result: Run B, 5-sphere task (1:30)
- Best learned policy: **2.93% mean fitted-T1 error**.
- Beats the matched fixed schedule by **~3 percentage points**; **95.8% of episodes under 5%**.
- → adaptive RL works and beats the fixed baseline over this T1 range. (addresses C3)
- *Figure:* RL vs fixed error distribution / per-sphere comparison.

### Slide 11 — Memory ablation: what should the policy remember? (1:00)
- Question: how much history does the agent need (memory architecture)?
- Headline finding only; full ablation table → backup.

### Slide 12 — DEDICATED LIMITATION: when does the full task break? (1:15)
- Run A, full 14-sphere plate: the policy *fails* — the T1 range is too wide for a single policy.
- Reframe as a research finding: why range/scale matters for adaptive policies; the trade-off between task breadth and adaptive gain.
- *Figure:* Run A vs Run B comparison.

### Slide 13 — Positioning vs published results (0:45)
- Quantitative comparison to the published adaptive-qMRI / baseline numbers; where we sit.
- (Can fold into Slide 10 if time is tight.)

### Slide 14 — (spare / flex) — reserved buffer

---

## Part 4 — Close (≈1:00, slide 15)

### Slide 15 — Contributions & future work (1:00)
- Three contributions: (1) open-source executable phantom twin; (2) validation-by-recovery that fixed two upstream KomaMRI bugs; (3) **first RL agent for adaptive qMRI**.
- Future work (1 line): widen the workable T1 range; T2 plate; benchmark the competing paper; LSTM memory.
- Thanks + questions.

---

## Backup slides (Q&A only — so results don't need memorising)

- Full Run A / Run B results tables (per-sphere errors).
- Full memory ablation table.
- Architecture / data-flow diagram (Python↔Julia, juliacall in-process).
- **Parallel envs / CPU speedup** (SubprocVecEnv-style rollout splitting; threads). Not in report.
- Cramér–Rao baselines (CR-optimal schedules) as the theoretical yardstick.
- Cached-water linearity derivation.
- Augmentation details (pose randomisation, material jitter).
- 14-sphere failure deep dive (what the policy actually does).
- E1 degenerate-policy post-mortem (why dense reward + noise + harder task).

---

## TODO before final slides (from plan.md)
1. Multi-fidelity "failed second RL" flow diagram — make into a clean flow diagram.
2. (Optional, post-working-deck) Implement competing-paper benchmark; possibly run on T2 plate.
3. (Optional) Keep LSTM run going to check step-bound parity.

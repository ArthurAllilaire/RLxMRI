# Project Overview — Adaptive qMRI Sequence Design via Reinforcement Learning

**Student:** Arthur Allilaire (aa8123@ic.ac.uk)  
**Supervisors:** Wayne Luk (Imperial, primary academic supervisor) · Andreas Wetscherek (ICR, domain supervisor)  
**Submission deadline:** Friday 12 June 2026  
**Research completion target:** Monday 1 June 2026 (per Andreas)  
**Recurring meetings:** Monday 1-on-1 with Andreas · Thursday standup with Andreas

---

## Report Structure

Six chapters (standard Imperial FYP format):

1. **Introduction** — motivation, challenges C1–C3, contributions summary
2. **Background and Related Work** — qMRI, MRF, RL for medical imaging, KomaMRI, existing sequence optimisation work; how existing approaches address C1–C3 and where the gaps are
3. **A1 — Digital Twin and Conventional Baseline** (addresses C1 setup)
4. **A2 — RL Agent for Adaptive Quantitative MRI** (addresses C1, C2)
5. **A3 — Spatial Localisation and Domain Generalisation** (addresses C3)
6. **Conclusion** — summary, limitations, future work

---

## Challenges, Novel Aspects, and Quantified Benefits

### C1 — Adaptive pulse sequence design
No single optimal MR sequence exists across patients/phantoms. The optimal sequence depends on unknown tissue properties (T1, T2, PD). Conventional approaches use fixed protocol libraries that cannot adapt.

**Novel aspect:** RL agent learns a policy that selects the next acquisition block (inversion time, flip angle, slice position) based on signal observed so far — producing a patient/instance-specific sequence rather than a fixed protocol.

**Quantified benefit:** agent achieves target T1/T2 accuracy (MAPE < 5%) in fewer total acquisitions / less total scan time than the conventional fixed-TI grid baseline (E0). Reported as a Pareto curve of (MAPE, relative scan time).

### C2 — Scalable simulation-in-the-loop RL
Running a full MRI physics simulator (KomaMRI Bloch equations) at every RL step is expensive — training requires 10⁵–10⁶ environment steps. The key tradeoff is simulator fidelity vs compute budget.

**Novel aspect:** coarse-voxel training config (3–4 mm, low matrix size) combined with live simulator calls at every step; fine-voxel evaluation (1 mm) to verify transfer. Demonstrates that a practically trainable RL loop can still produce generalisable policies.

**Quantified benefit:** per-step wallclock < 100 ms at training config; policy trained this way achieves comparable MAPE on fine-voxel evaluation (sim-to-sim transfer gap reported explicitly).

### C3 — Spatial localisation under pose uncertainty
The phantom/patient's exact position and orientation in the scanner is unknown prior to acquisition. Existing qMRI sequence optimisation work assumes a known, fixed geometry.

**Novel aspect:** domain randomisation (random rotation + translation per episode) forces the agent to localise as well as map. A fast scout acquisition at the start of each episode gives the agent spatial information to plan slice placement before spending scan budget on diagnostic TI choices.

**Quantified benefit:** MAPE as a function of phantom translation and rotation magnitude — showing the agent degrades gracefully compared to a non-localisation-aware baseline that assumes fixed pose.

---

## Experiments

| ID | Name | Status | Addresses |
|---|---|---|---|
| E0 | Conventional baseline (IR-TSE + multi-TE SE) | Done | C1 (yardstick) |
| E1 | Single-voxel RL (PPO, TI + α action space) | Done | C1, C2 |
| E2 | Full T1-plate RL with gradients, 2D imaging, localisation | In progress | C1, C2, C3 |
| E3 | MRF-style fingerprinting (learned FA/TR schedules) | Planned | C1, C2 |
| E4 | Adaptive k-space trajectories (radial spoke angle) | Stretch | C1, C2 |
| E5 | Full 6-DoF pose estimation + parameter mapping | Stretch | C3 |

---

## Detailed Timetable

### Week 4 — 28 Apr – 4 May 2026
*Focus: E2 implementation*
- [ ] Implement gradient-encoded IR-SE sequence block (Gz, Gx, Gy)
- [ ] Verify 2D k-space → FFT → recognisable phantom image (visual sanity check)
- [ ] Implement complex Gaussian noise on simulator output
- [ ] Implement per-sphere ROI segmentation + running Levenberg–Marquardt T1 estimator
- [ ] Update Gym env: 14-sphere phantom, new observation/action spaces
- [ ] Implement scout localiser block as episode-initial fixed step
- [ ] Benchmark per-step wallclock at 3–4 mm; confirm < 100 ms
- [ ] Begin PPO training; monitor convergence and reward signal sanity
- [ ] Send Progress Update 1 to Wayne ✓

### Week 5 — 5–11 May 2026
*Focus: E2 evaluation and ablations*
- [ ] Evaluate trained E2 agent on held-out phantom configs (50 seeds)
- [ ] Compute T1 MAPE across all 14 spheres; compare vs E0 baseline
- [ ] Generate Pareto curve: MAPE vs relative scan time
- [ ] Plot TI-choice histogram — confirm non-uniform adaptive policy
- [ ] Run noise robustness sweep: MAPE vs σ ∈ {0, 0.02, 0.05, 0.10, 0.20}
- [ ] Run pose robustness sweep: MAPE vs translation magnitude and rotation angle
- [ ] Ablation: no domain randomisation vs full randomisation (memorisation failure)
- [ ] Write up E2 results as Jupyter notebook
- [ ] Send Progress Update 2 to Wayne

### Week 6 — 12–18 May 2026
*Focus: E3 implementation (MRF-style fingerprinting)*
- [ ] Pre-compute Bloch dictionary for (T1, T2, FA, TR) grid
- [ ] Implement dictionary-discriminability reward (inner product with dict entry)
- [ ] Swap action space to variable FA + TR per block
- [ ] Train E3 agent; monitor convergence
- [ ] Pilot k-space line interleaving across TIs (stretch within E3)
- [ ] Send Progress Update 3 to Wayne

### Week 7 — 19–25 May 2026
*Focus: E3 evaluation + model-based reconstruction pilot*
- [ ] Evaluate E3 agent; compare to E2 and E0 on same scan-time budget
- [ ] Pilot model-based backprop reconstruction (MRzero PDG framework)
- [ ] Write up E3 results
- [ ] Draft bullet-point outlines for all 6 report chapters
- [ ] Send Progress Update 4 to Wayne

### Week 8 — 26 May – 1 Jun 2026 ← Research deadline
*Focus: final experiments + begin write-up*
- [ ] Run any stretch experiments (E4 radial k-space if E3 is solid)
- [ ] Complete all experimental results and plots
- [ ] **All research complete by 1 June**
- [ ] Chapters 1–2 (intro + background): full draft
- [ ] Chapter 3 (A1): full draft
- [ ] Send Progress Update 5 to Wayne

### Week 9 — 2–8 Jun 2026
*Focus: full write-up*
- [ ] Chapter 4 (A2): full draft
- [ ] Chapter 5 (A3 + evaluation): full draft
- [ ] Chapter 6 (conclusion): full draft
- [ ] Internal review pass: check C1–C3 / novel aspects / quantified benefits are explicit in every chapter
- [ ] Send Progress Update 6 to Wayne

### Week 10 — 9–12 Jun 2026
*Focus: final polish and submission*
- [ ] Address reviewer comments; check figures, captions, references
- [ ] Proofread; verify page count and formatting requirements
- [ ] **Submit by Friday 12 June 2026**

---

## Key Constraints and Risks

| Risk | Mitigation |
|---|---|
| Per-step simulator too slow for E2 training | Benchmark first; reduce matrix size / voxel count if needed |
| PPO fails to converge on 14-sphere reward | Check reward scale; add per-step T1-estimate reward to densify signal |
| Localisation too hard to learn jointly with mapping | Decouple: train localiser policy first, then mapping policy conditioned on localiser output |
| PDG vs EPG error in interim report | Correct throughout before submission; confirmed MRzero uses PDG not EPG |
| Parallel Julia runtimes exceed memory | Fall back to single-process async collection if n_envs × RAM > 16 GB |
| Fiducial calibration values are placeholders | Not blocking for E1–E3; replace before any sim-to-real comparison |

---

## Report Requirements (from Wayne)

Per Wayne's Email 2, the report must be explicit about:
- **(a) What challenges does this project address?** → C1–C3 in Introduction and each chapter opening
- **(b) What are the novel aspects compared to previous work?** → Novel aspect bullets in each technical chapter
- **(c) What are the quantified benefits vs published work?** → Pareto curves, MAPE tables, robustness plots in each evaluation section

Recommended by Wayne: aim to have basic drafts with bullet points for ALL chapters and sections by **Monday 1 June** — aligns with the research completion deadline.

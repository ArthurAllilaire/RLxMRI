# Missed / Late-Breaking Related Work

Context: original literature search was around December 2025. The final report is currently frozen, but these are useful for the presentation, viva answers, and a possible conference submission.

## Should Mention In The Presentation

These are the highest-value additions because they protect the novelty claim without distracting from the project.

### Walker-Samuel 2023: full deep-RL scanner-control preprint

- **Work:** *Control of a simulated MRI scanner with deep reinforcement learning*.
- **Why it matters:** The report cites Walker-Samuel's 2019 ISMRM abstract, but the 2023 preprint is the fuller closest comparator for deep RL controlling a simulated scanner.
- **How to position:** Shows RL can control a simulated MRI scanner for active sensing / classification. This project differs by targeting quantitative parameter estimation, using a qMRI phantom, fitted T1 error, spatial reconstruction, and adaptive timing decisions.
- **Presentation use:** One sentence on the related-work/novelty slide: "Closest prior deep-RL scanner-control work targets classification rather than quantitative T1 mapping."
- **Link:** https://arxiv.org/abs/2305.13979

### DeepRF / RL-designed RF pulses

- **Work:** Shin et al., DeepRF / deep-RL-designed RF pulses, 2021 *Nature Machine Intelligence* (PDF in `report/relatedwork/DeepRF.pdf`).
- **Why it matters:** Important RL-in-MRI control work. It shows RL has been applied to lower-level RF waveform design.
- **What it actually does (so I don't re-read it):** Designs a single static **RF pulse waveform** B₁(t) (256 complex samples), offline. Four pulse types: slice-selective excitation/inversion (vs SLR reference) and B₁-insensitive volume/selective inversion (vs hyperbolic-secant reference). Two-stage pipeline:
  1. **RF generation (the "RL" part):** a GRU+FC RNN autoregressively emits 32 (amplitude, phase) pairs, mirrored+interpolated to 256. Trained with **PPO** (actor-critic, ref Schulman 2017). They generate **38.4M pulses**, keep the **top 256** by reward as seeds.
  2. **RF refinement (the not-RL part):** those 256 seeds get **10,000 steps of plain gradient ascent** through the *differentiable* Bloch simulator (Adam, reverse-mode autodiff). Best final pulse = result.
- **The reward (hand-built scalar, terminal-only):** always *(profile-match term) − c·ENG*, where `ENG = ∫|B₁|²dt`. E.g. excitation: `min(mean transverse mag over BW, c₁) − max(stopband ripple − c₂, 0) − c₃·ENG`; inversion: `−c₁·MSE(M_z, M_z_reference) − c₂·ENG`. The profile is the Bloch-simulated magnetization over a frequency/B₁ grid. All `cᵢ` hand-tuned. **No per-step reward** — the scalar is computed once, after the full pulse is simulated.
- **How PPO updates with a terminal-only reward:** episode = generating one 32-step pulse; actions sampled from `N(μₜ,σ²)`; critic head outputs `Vₜ`. Advantage `Âₜ = R − Vₜ` (terminal reward minus learned baseline), credit-assigned across all 32 steps via the clipped surrogate, batched over 256 pulses. Because the reward is terminal-only and "transitions" are just deterministic appends of the agent's own output, **the MDP is degenerate: PPO collapses to a clipped REINFORCE-with-baseline black-box optimizer of E[R(pulse)]** over the RNN's output distribution — RL used as a *search/seed* step, not a closed-loop controller.
- **Why the two stages (and why RL alone isn't enough):** their ablation shows PPO alone gives imprecise pulses and gradient-ascent alone (on bad seeds) underperforms — RL does coarse global exploration to find good basins, gradient ascent does the fine optimization. So it's genuine RL but propped up by classical optimal-control-style gradient descent.
- **Why the −ENG term is the whole point:** the profile-match term is *already maximized by the conventional pulse itself* (it's the reference), so a profile-only reward would just rediscover SLR/HS. The energy penalty is what makes it non-trivial and pushes it to a novel solution. RF energy ∝ **SAR** (tissue-heating safety limit), so lower energy is the real clinical win.
- **Results:** energy reductions at matched profile — excitation **−17%** (−20% vs equiripple SLR), slice-selective inversion **−11%**, B₁-insensitive volume inversion **−9%**, selective inversion **−2%**. Beats classical optimal control (DeepRF −17%/−11% vs OC −4%/−4%). Headline science: inversion pulses **violate the adiabatic condition** (min adiabaticity 0.04–0.1 vs HS ~3) → a magnetization-manipulation mechanism conventional theory wouldn't produce. Cost: **~27–56 hrs/pulse** (GPU) vs <1 s (SLR) / ~5 hrs (HS grid search). Offline, one-time per pulse type.
- **How to position vs this project:** control object is the **RF waveform** (low-level), designed **offline**, **open-loop** (never conditions on observations), terminal reward, objective = profile fidelity + energy. This project is the opposite: a **closed-loop** policy choosing **acquisition blocks** online, conditioned on **fitted tissue estimates**, with a dense quantitative-error objective. Clean "yes RL has touched MRI, but not adaptive qMRI acquisition" citation.
- **Presentation use:** Mention only if asked "has RL been used in MRI before?"
- **Link:** https://arxiv.org/abs/1912.09015
- **BibTeX:**
```bibtex
@article{Shin_2021,
   title={Deep reinforcement learning-designed radiofrequency waveform in MRI},
   volume={3},
   ISSN={2522-5839},
   url={http://dx.doi.org/10.1038/s42256-021-00411-1},
   DOI={10.1038/s42256-021-00411-1},
   number={11},
   journal={Nature Machine Intelligence},
   publisher={Springer Science and Business Media LLC},
   author={Shin, Dongmyung and Kim, Younghoon and Oh, Chungseok and An, Hongjun and Park, Juhyung and Kim, Jiye and Lee, Jongho},
   year={2021},
   month=Nov, pages={985–994} }
```

### Pineda et al. 2020 — Active MR k-space Sampling with RL

- **Work:** Pineda, Basu, Romero, Calandra, Drozdzal (Facebook AI Research + McGill), MICCAI 2020 (PDF in `report/relatedwork/Active_MR_k-space_Sampling_with_Reinforcement_Lear.pdf`). Same orbit as the NYU fastMRI project; code released.
- **Why it matters:** The closest prior "adaptive acquisition RL" — and, unlike DeepRF, it is **genuine closed-loop sequential RL** (dense per-step reward, policy conditions on observations). Cited in the core related-work table (`fig2b`).
- **What it actually does (so I don't re-read it):** Accelerated MRI — undersample k-space, reconstruct from partial data. Most DL work fixes the trajectory and improves the reconstructor; Pineda flips it: **fix a pre-trained, frozen CNN reconstructor and learn which k-space lines to acquire, in what order**, adapting per subject and across the whole range of acceleration factors. Cartesian sampling, one action = acquire one vertical k-space column.
- **The MDP (a POMDP):**
  - **State** `sₜ = ⟨x, Mₜ⟩` — hidden ground-truth image `x` + visible sampling mask `Mₜ`.
  - **Observation** `oₜ = ⟨x̂ₜ, Mₜ⟩` — current de-aliased **reconstruction** `x̂ₜ` from the frozen reconstructor + mask. Whole history is captured by `oₜ`, so the policy is a function of `oₜ` alone.
  - **Action** `aₜ` = an unobserved column index (332 valid columns on the 640×368 knee data); already-sampled columns masked out (value set to −∞).
  - **Transition:** deterministic — add column to mask, `x` unchanged. **Emission:** deterministic recon `x̂ₜ₊₁ = r(F⁻¹(Mₜ₊₁⊙y))`.
- **The reward (dense, per-step — opposite of DeepRF's terminal-only):** `R(sₜ,aₜ) = C(x̂ₜ,x) − C(x̂ₜ₊₁,x)` = the decrease in reconstruction error from adding that line. Telescopes to the final metric `C(x̂_T,x)`; γ=0.5. Every measurement gets immediate credit for how much it improved the image.
- **What "reconstruction quality" actually means:** `C ∈ {MSE, NMSE, −PSNR, −SSIM}` — MSE/NMSE/PSNR are pixelwise (PSNR is just log-MSE in dB); **SSIM is structural/perceptual** (local luminance/contrast/structure over windows, not pixelwise — the SSIM-trained policy behaves differently, more low-freq biased, exploits k-space Hermitian symmetry). Crucially the reference `x` is **not external/diagnostic ground truth — it is the reconstruction from *fully-sampled* k-space** (`x = F⁻¹(y)`). So "quality" = *image-domain fidelity of the undersampled reconstruction to the fully-sampled image*, says nothing about diagnostic value or quantitative-parameter accuracy. This is exactly why it sits in the `Quantitative=✗` column.
- **How DDQN updates:** solved with **Double Deep Q-Network** (value-based, off-policy — NOT policy gradient). A value net predicts expected future cumulative reward per candidate column; policy greedily picks max-value valid action. Trained by minimizing TD error on a 20k replay buffer, ε-greedy exploration, 5M transition steps.
- **Two variants + the adaptivity twist (key viva point):** **ss-ddqn** (subject-specific, the *adaptive* one) sees `x̂ₜ`+mask, so it *can* choose different lines per patient based on the revealed partial image — the genuine closed-loop "active sensing" pitch. **ds-ddqn** (dataset-specific) sees only the timestep, so its output is a deterministic function of time → **one fixed learned ordering for every subject** (effectively a static learned mask, no adaptation). **Surprise: ds-ddqn slightly beat ss-ddqn on most metrics** (Tables 1–2 AUC). So the headline win over heuristics came from *learning a good ordering*, **not** from *per-subject adaptation* — the adaptive model bought nothing measurable. Their defense (fair): they don't conclude adaptivity is useless — the **oracle** policy (ground-truth access, greedy best-next-line) shows **wide per-subject variety** in optimal trajectories and a **large remaining gap** to their models, so the *potential* exists; they blame **optimization/learning-stability issues** (a value net on a high-dim image input is far harder to train than one on a scalar timestep), not the idea of adaptation.
- **Why this matters for positioning (use defensively):** this is a top-lab (FAIR) precedent where, in a closely related active-acquisition setting, **a learned *fixed* schedule matched/beat the learned *adaptive* policy**. So any "adaptive beats fixed" claim must benchmark against a *strong learned/optimized fixed baseline* (e.g. CR-optimal / CRLB-inspired schedules), not just naive fixed grids — otherwise a reviewer will say the adaptive part may not be what's helping. Demonstrating that per-subject adaptivity genuinely pays off (which Pineda did not cleanly show) is itself a contribution, not a given.
- **Results:** both DDQNs beat all heuristics (random, low-freq-biased random, low→high, [31]'s evaluator scoring) on fastMRI single-coil knee. Scenario-30L (~3–11× accel): MSE 3–7% below best baseline at 4–10×; AUC gains 0.55–2.9%. Scenario-2L (extreme, up to 166×): ≥10% (up to 35%) below best heuristic under 100×; AUC gains 2.68–11.6%; p<10⁻⁸. Gap to one-step oracle remains. Setup simplified (single-coil, ignores practical phase-encoding issues).
- **Pros / what they did well:** uses **real acquired data, not a simulator** — trains and evaluates on the large-scale NYU fastMRI single-coil knee dataset (~19.9k train / 1.8k val / 1.9k test 2D images), so results are on genuine clinical k-space rather than Bloch-simulated signals. Clean POMDP formulation, dense reward, strong heuristic baselines + oracle upper bound, released code. (fastMRI dataset: https://fastmri.med.nyu.edu/)
- **How to position vs this project:** genuine closed-loop RL, but the **control object is which k-space lines to sample** and the **objective is reconstruction image quality** (qualitative imaging). This project controls **relaxation-preparation timings (TI/TR/FA)** to minimize **fitted quantitative T1/T2 error** on a calibrated phantom. Hence `fig2b`: "Deep RL (DQN), K-Space Recon Quality, Quantitative=✗"; `fig2_capability`: Adaptive=✓, Quantitative=✗. Both accurate.
- **Presentation use:** the "closest adaptive-acquisition RL" comparator — adapts *what to sample in k-space for a better image*, not *what contrast timing for a better quantitative estimate*.
- **Link:** https://arxiv.org/abs/2007.10469  ·  code: https://github.com/facebookresearch/active-mri-acquisition
- **BibTeX:**
```bibtex
@inproceedings{pineda_active_mri_rl_2020,
  author    = {Pineda, Luis and Basu, Sumana and Romero, Adriana and Calandra, Roberto and Drozdzal, Michal},
  title     = {Active {MR} k-space Sampling with Reinforcement Learning},
  booktitle = {Medical Image Computing and Computer Assisted Intervention -- MICCAI 2020},
  year      = {2020},
  pages     = {23--33},
  publisher = {Springer},
  doi       = {10.1007/978-3-030-59713-9_3}
}
```

### Vinding and Lund 2026 review: AI-powered MRI controls

- **Work:** *A Review of AI-Powered Controls in the Field of Magnetic Resonance Imaging*.
- **Why it matters:** Recent umbrella review for AI-assisted scanner control, including RF pulses, B0/B1 control, gradients, SAR, and acquisition control.
- **How to position:** Useful for showing awareness of the wider field. It supports the claim that this project sits inside a growing AI-control-for-MRI direction while remaining distinct as adaptive qMRI sequence timing.
- **Presentation use:** Good backup citation in Q&A; probably too broad for a slide unless there is a related-work slide.
- **Link:** https://www.mdpi.com/2073-431X/15/1/27

### Agent4MR / LLM-guided MR sequence development

- **Work:** *Agent4MR: Agentic MR Sequence Development with Large Language Models*.
- **Why it matters:** Very recent automated sequence-development work. It uses LLM agents and PyPulseq-style validation rather than RL, but it is directly relevant to automated pulse-sequence generation.
- **How to position:** Offline / tool-assisted sequence generation, not closed-loop adaptive qMRI. Good contrast against this project's learned policy that conditions each next acquisition on current fitted estimates.
- **Presentation use:** Mention only if discussing "automated sequence design" broadly.
- **Link:** https://arxiv.org/abs/2604.13282

### Recent active k-space acquisition work

- **Candidate works:** CUTE-MRI 2025, ASMR 2024, SUNO/dSUNO 2026, adaptive radial k-space RL 2025.
- **Why it matters:** The report already cites Pineda 2020 for active k-space RL. These newer papers show the area is still active and increasingly patient-/scan-adaptive.
- **How to position:** They adapt sampling masks or scan duration for reconstruction/downstream tasks. This project adapts relaxation-preparation timings for quantitative T1 estimation.
- **Presentation use:** If there is one related-work slide, group them as "active k-space sampling is adjacent but optimises what to sample in k-space, not the quantitative contrast timing."
- **Links:**
  - ASMR: https://arxiv.org/abs/2406.04318
  - CUTE-MRI: https://arxiv.org/abs/2508.14952
  - Adaptive radial k-space RL: https://arxiv.org/abs/2508.04727

## Useful For Conference Submission

These are worth checking more carefully if the work becomes a paper. They are not all presentation-worthy.

### Sequence Search / neural architecture search for MRI sequence design

- **Work:** Automated MR sequence design with neural architecture search, 2026.
- **Why it matters:** Fresh offline automated sequence-design comparator.
- **How to position:** Searches over sequence designs before acquisition; does not implement closed-loop adaptation from observations during the scan.
- **Link:** https://arxiv.org/html/2604.14788v1

### SeqGPT

- **Work:** ISMRM 2025 abstract on LLM generation of MRI pulse sequences.
- **Why it matters:** Another automated sequence-generation direction, useful if making broader claims about automated sequence design.
- **How to position:** LLM sequence synthesis, not simulator-trained RL policy optimisation for qMRI accuracy.
- **Link:** https://archive.ismrm.org/2025/3381.html

### MIMOSA

- **Work:** *Multi-Parametric Imaging Using Multiple-Echoes With Optimized Simultaneous Acquisition*, 2026 *Magnetic Resonance in Medicine*.
- **Why it matters:** Strong practical qMRI comparator: efficient multiparametric acquisition with NIST phantom / in-vivo validation and large acceleration claims.
- **How to position:** Not RL and not adaptive in the same sense, but important for any conference paper claiming efficient qMRI sequence design.
- **Link:** https://pubmed.ncbi.nlm.nih.gov/41088533/
- **What it actually is:** Pure MRI-physics / sequence-engineering work — *no deep learning or RL in the acquisition design*. The authors hand-design a new multiparametric pulse sequence (a 3D-QALAS variant fusing turbo-FLASH and multi-echo GRE with a spiral-like Cartesian trajectory) and use offline Bloch simulations to *optimise its fixed timing parameters once*, before scanning. The only learned component is on the reconstruction side: a zero-shot self-supervised network that reconstructs the undersampled k-space (R up to 11.8). So it is the opposite of this project's contribution — MIMOSA optimises a single fixed sequence offline and learns the reconstruction, whereas this project learns a closed-loop policy that adapts the *acquisition* online from fitted estimates. It is a strong practical qMRI efficiency comparator (NIST phantom + in-vivo, whole-brain T1/T2/T2*/PD/QSM in ~3 min), not a competing adaptive-control method.
- **BibTeX:**
```bibtex
@article{chen2026mimosa,
  author  = {Chen, Y. and Jun, Yohan and Heydari, A. and Yong, X. and Kim, J. and Lee, J. and Liu, H. and Ye, H. and Gagoski, Borjan and Fujita, Shohei and Bilgic, Berkin},
  title   = {{MIMOSA}: Multi-Parametric Imaging Using Multiple-Echoes With Optimized Simultaneous Acquisition for Highly-Efficient Quantitative {MRI}},
  journal = {Magnetic Resonance in Medicine},
  volume  = {95},
  number  = {3},
  pages   = {1528--1544},
  year    = {2026},
  month   = mar,
  doi     = {10.1002/mrm.70143},
  pmid    = {41088533}
}
```

### Zero-DeepSub

- **Work:** Zero-shot deep subspace reconstruction for rapid multiparametric qMRI / 3D-QALAS.
- **Why it matters:** Reconstruction-side rapid qMRI method. Helps distinguish acquisition-policy adaptivity from learned reconstruction acceleration.
- **How to position:** Improves reconstruction/estimation from qMRI data; does not choose the next acquisition action online.
- **Link:** https://pubmed.ncbi.nlm.nih.gov/38282270/

### RELAX-MORE / self-supervised qMRI acceleration

- **Work:** Self-supervised deep learning with model reinforcement for rapid T1 mapping.
- **Why it matters:** Another reconstruction/model-fitting-side qMRI acceleration approach.
- **How to position:** Relevant practical baseline family, but not adaptive sequence control.
- **Link:** search terms: `"RELAX-MORE" "rapid T1 mapping" MRI`

## Suggested Novelty Wording

Short version for slides:

> Prior automated MRI work has optimised RF pulses, fixed pulse sequences, or k-space sampling. The gap here is closed-loop adaptive quantitative MRI: after each acquisition, the policy observes the current fitted T1 estimates and chooses the next contrast timing to reduce final parameter-mapping error.

More defensive version for Q&A:

> RL has been used in MRI before, especially for RF pulse design, scanner-control demonstrations, and active k-space sampling. The novelty here is narrower: a simulation-in-the-loop RL policy for adaptive qMRI timing, evaluated by fitted T1 error on a spatially encoded digital calibration phantom against fixed and CRLB-inspired schedules.

## Citation Priority

For a 1-slide related-work summary, use:

1. Beracha et al. 2023 adaptive model-based MR.
2. Walker-Samuel 2023 deep-RL scanner control.
3. MRzero / Loktyushin et al. 2021 differentiable sequence discovery.
4. DeepRF / Shin et al. 2019-2021 RL RF pulse design.
5. Pineda 2020 plus one newer active k-space paper.

For a conference submission, also add:

1. Vinding and Lund 2026 review.
2. Agent4MR 2026.
3. Sequence Search 2026.
4. MIMOSA 2026.
5. Zero-DeepSub / RELAX-MORE if discussing qMRI acceleration beyond acquisition control.

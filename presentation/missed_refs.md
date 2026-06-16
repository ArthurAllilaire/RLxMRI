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

- **Work:** Shin et al., DeepRF / deep-RL-designed RF pulses, including the broader 2021 *Nature Machine Intelligence* paper.
- **Why it matters:** Important RL-in-MRI control work that is not currently in the core table. It shows RL has been applied to lower-level RF waveform design.
- **How to position:** It optimises RF pulses, not adaptive acquisition policies. The control object is pulse waveform design, whereas this project chooses acquisition blocks conditioned on fitted tissue estimates.
- **Presentation use:** Mention only if asked "has RL been used in MRI before?"
- **Link:** https://arxiv.org/abs/1912.09015

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

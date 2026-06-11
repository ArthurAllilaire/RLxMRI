# Literature grounding brief for final report revision

Date: 11 June 2026

Purpose: source material for strengthening the report's background, related
work, novelty claims, and supervisor-feedback response. This is deliberately a
research brief, not polished report prose. Use it to write compact additions
into `report_latex/` while keeping the report under the 60-page content limit.

## Executive recommendations

Highest-value edits:

1. Add a compact "Digital twins and phantoms for qMRI" subsection to Chapter 2,
   before the current simulator/Pulseq subsection. This directly addresses
   Wayne feedback items 3, 4, 5, 6 and 12.
2. Replace the incomplete `digitaltwin_review_2026` BibTeX TODO with the
   verified Greggio et al. (2026) metadata extracted from
   `report/1-s2.0-S1078817426000891-main.pdf`. This is the strongest digital
   twin source for your report because it is MRI-specific.
3. Add a short related-work table in Chapter 3 that compares:
   generic MRI phantoms, qMRI phantoms, digital twins, and this project's
   executable KomaMRI-compatible phantom package.
4. Add two sentences at the start of Chapter 4 explaining why simulator
   validation is different from simulator benchmarking: the novelty is a
   qMRI-specific, end-to-end recovery test that caught a long-time numerical
   failure hidden by plausible images.
5. Keep Chapter 5's existing related-work table, but add one row or paragraph
   positioning active MRI/k-space RL separately from sequence-design RL. This
   stops a marker from thinking "RL for MRI already exists" without the report
   explaining the distinction.
6. Do not add long background prose. Page count is already tight. Prefer one
   concise subsection, one compact table, and one sentence in each technical
   chapter aim.
7. Rebalance the Introduction. The current opening is MR-Linac/cancer-heavy,
   but the delivered project is broader: executable qMRI phantom twin,
   simulator validation, and adaptive sequence design. Keep MR-Linac as the
   motivating application, but introduce the immediate technical bottleneck as
   MRI/qMRI acquisition efficiency under rising MRI demand.

## Feedback coverage map

| Feedback item | What is still needed | Suggested action |
|---|---|---|
| 3. Chapter 2 should cover related approaches, gaps, and how this approach addresses them | Chapter 2 currently has MRI/RL background and a summary gap paragraph, but the digital-twin/phantom/simulator-validation axis is still thin | Add Chapter 2 subsection "Digital twins, phantoms and simulator validation"; add a compact comparison table |
| 4. Include digital twins in Chapter 2; connect achievements to new generation of digital twins for adaptive MRI | `references.bib` has an incomplete TODO for the 2026 ScienceDirect paper | Replace it with Greggio et al. 2026; cite it for MRI-specific DT themes, barriers and opportunities |
| 5. Chapter 3 physical phantom context should be in Chapter 2; related work on phantoms | Chapter 3 starts with the QalibreMD phantom but lacks literature context beyond the manual | Move/condense the generic "why phantoms matter" argument into Chapter 2; cite Keenan et al. 2024 and QIBA/NIST phantom papers |
| 6. Related work in Chapters 4-5; novelty needs comparison | Chapter 5 has a good table; Chapter 4 less so | Add a short Chapter 4 positioning paragraph vs KomaMRI/MRiLab/JEMRIS validation; preserve Chapter 5 table and add active acquisition distinction |
| 7. Page bound | Current content reportedly 62 pages | Add only compact material; move background figures and long KomaMRI details to appendix if needed |
| 8. Chapter 4 contents/intro | Already started: Chapter 4 now has aim and minitoc | Keep, maybe add one sentence "Related-work gap" in the aim |
| 9. Chapters 3-5 chapter aims | Already started | Keep consistent phrasing: aim, challenge, novelty, hardest part, chapter map |
| 10. Chapter 2-5 summaries | Already started for Chapter 2, 3, 4, 5 | Keep summaries short; avoid adding more |
| 11. Chapter 3 two reasons for digital twin and section mapping | Already started | The new Chapter 2 subsection should make the "why a twin" argument before Chapter 3 |
| 12. Highlight hardest parts and novelty vs published work | Partly done in Chapter aims | Add specific novelty comparisons from this brief into aims and tables |

## Core argument to build

The report should frame the project as a novel integration of four lines of
work, not as a claim that any one component is new in isolation:

1. qMRI phantoms provide traceable ground truth for quantitative maps, but most
   phantom work stops at physical validation and repeatability studies.
2. MRI simulators such as KomaMRI, MRiLab and JEMRIS provide Bloch-equation
   forward models, but their usual validation emphasises signal/image agreement,
   speed and usability rather than long-duration adaptive qMRI recovery tests.
3. Automated sequence-design work such as MRzero, MRF optimisation and AUTOSEQ
   shows that acquisition schedules can be learned or optimised, but much of it
   is offline: the schedule is fixed before the scan.
4. Adaptive MR and scanner-control RL show closed-loop acquisition is possible,
   but prior work either uses model-based Bayesian selection rather than RL
   policy learning, simplified 1-D/simulation tasks, classification objectives,
   or k-space sampling/reconstruction objectives rather than fitted qMRI
   parameter accuracy on a calibrated phantom.

The project's gap claim should therefore be:

> This project tests closed-loop reinforcement learning for qMRI timing in an
> end-to-end, spatially encoded Bloch-simulation pipeline grounded in a
> calibrated phantom digital twin. The novelty is the combination: a reusable
> executable qMRI phantom, strict simulator validation by parameter recovery,
> and a multi-fidelity RL environment where the policy is evaluated against
> fixed qMRI schedules on fitted T1 error.

This is strong but not overstated. It avoids claiming "first RL for MRI" or
"first adaptive MRI".

## Digital-twin overclaim guardrail

Use "digital twin" carefully. Greggio et al. define DT-MRI in terms of dynamic
virtual replicas linked to real-world counterparts through data streams,
simulation and analytics. Your project does not build a live clinical digital
twin that updates from scanner or patient data during deployment.

Safe wording:

> This is not a live clinical digital twin bidirectionally linked to scanner or
> patient data. It is a calibration-phantom digital twin: an executable,
> simulator-facing representation of a physical qMRI reference object, used for
> controlled simulation, validation and policy training.

Alternative shorter wording:

> The term digital twin is used here in a bounded sense: the twin represents a
> calibration phantom for simulation and validation, not a patient or scanner
> service connected to live clinical data.

Avoid:

- "patient digital twin"
- "clinical digital twin"
- "real-time bidirectional twin"
- "scanner digital twin"

unless the sentence explicitly says this project does **not** reach that level.

## Chapter-Level Related Work Vs Contribution

Use one of these one-liners in each technical chapter aim/summary so the novelty
claim is visible without adding long literature-review prose.

Chapter 3 / MRISystemPhantom:

> Existing open phantom projects make calibration hardware reproducible, and
> tools such as PhantomViewer and MR-BIAS analyse acquired phantom images. This
> chapter contributes the complementary simulator-side layer: an executable
> KomaMRI-compatible phantom with ground-truth material labels, configurable
> voxelisation, pose/material randomisation and fitting hooks for adaptive
> sequence-design experiments.

Chapter 4 / KomaMRI validation:

> KomaMRI and related simulators have already been benchmarked as general MRI
> forward models. This chapter tests a narrower but necessary condition for this
> project: whether the long-duration simulation, reconstruction, ROI extraction
> and fitting pipeline preserves known qMRI T1 values under the acquisition
> patterns reached by adaptive RL.

Chapter 5 / Adaptive RL:

> Prior adaptive and learned MRI work shows that acquisition parameters can be
> optimised or selected automatically, but published tasks differ: model-based
> adaptive T2/MRS, offline learned sequences, shape classification, or k-space
> sampling. This chapter tests a different object: a closed-loop RL policy that
> chooses IR-SE timings from current fitted T1 estimates and is evaluated
> against matched fixed qMRI schedules in a validated 2-D phantom simulator.

## Introduction rewrite note

Current issue:

- The existing Chapter 1 opening motivates the work through cancer,
  radiotherapy and MR-Linac. That is relevant, but it makes the project sound
  as if the main deliverable is an MR-Linac clinical workflow. The actual report
  has drifted, productively, toward a more defensible computing contribution:
  validated simulation infrastructure for adaptive qMRI sequence design.
- The digital-twinning review gives a better high-level pressure point:
  MRI demand is rising, radiology workforce shortages contribute to long waits,
  and MRI digital twins/protocol optimisation are emerging as ways to test and
  improve MRI workflows before deployment. Use this to widen the intro beyond
  radiotherapy without abandoning the MR-Linac motivation.

Suggested replacement structure for Chapter 1 introduction:

1. Paragraph 1: MRI demand and acquisition-time pressure.
   Use Greggio et al.'s figures sparingly: NHS performed over 2.5 million MRI
   scans in 2024, 10.3% up on the previous year, and 16.3% of UK patients waited
   more than six weeks against a 1% target. This motivates faster and smarter
   MRI workflows generally.
2. Paragraph 2: why MR-Linac/adaptive radiotherapy makes the bottleneck acute.
   MRI gives soft-tissue contrast without ionising radiation, but online
   adaptation is time-constrained; long acquisitions increase patient burden and
   motion artefacts.
3. Paragraph 3: why qMRI and sequence design, not only reconstruction.
   Reconstruction acceleration helps after data are collected, but qMRI accuracy
   also depends on which measurements are acquired. Fixed protocols are designed
   to work for broad populations; adaptive protocols could choose the next
   measurement from the information already observed.
4. Paragraph 4: why simulation/digital twin is needed before RL.
   Learned acquisition policies need many trials and cannot be trained directly
   on patients/scanners. A calibrated phantom digital twin and validated Bloch
   simulator provide ground truth, reproducibility and a safe environment for
   testing adaptive sequence design.

Suggested compact prose seed:

> MRI is increasingly central to diagnosis, treatment planning and image-guided
> therapy, but demand is growing faster than clinical capacity. A recent
> systematic review of MRI digital twinning reports that the NHS performed over
> 2.5 million MRI scans in 2024, a 10.3% increase on the previous year, while
> 16.3% of UK patients waited more than six weeks for MRI, far above the 1%
> target. Reducing acquisition time is therefore not only an image-processing
> problem; it is a workflow and capacity problem.
>
> The pressure is especially clear in MR-guided radiotherapy. MRI improves
> soft-tissue contrast without ionising radiation and enables online adaptation
> of treatment plans, but each additional acquisition keeps the patient on the
> treatment couch and increases the opportunity for motion artefacts. Faster
> reconstruction can help, but it does not answer a more fundamental question:
> which measurements should be acquired in the first place?
>
> This project studies that acquisition question for quantitative MRI. Rather
> than applying a fixed inversion-recovery protocol, an agent observes the
> current fitted tissue-parameter estimates and chooses the next acquisition
> timing. Training such a policy directly on hardware would be impractical, so
> the project first builds and validates the simulation infrastructure: a
> KomaMRI-compatible digital twin of a calibrated qMRI phantom, an end-to-end
> recovery test for the Bloch simulator, and a multi-fidelity RL environment for
> adaptive sequence design.

Concluding-chapter tie-in:

- In the Summary/Conclusion, this should become a broad impact point rather than
  a new claim:

> In a healthcare setting where MRI demand and waiting times are increasing,
> adaptive acquisition methods are attractive because they attack the scan-time
> bottleneck at the measurement-design level. This project does not yet provide
> a clinical protocol, but it contributes the validated simulator and phantom
> infrastructure needed to test such protocols safely and reproducibly.

## Verified key sources and summaries

### Digital twins in MRI, precision medicine and imaging

#### Greggio et al. (2026) - Wayne's ScienceDirect link

Metadata extracted from `report/1-s2.0-S1078817426000891-main.pdf`:

- Title: Exploring digital twinning in MRI: A systematic review of current
  applications, barriers, and future opportunities
- Authors: J. Greggio, N. Stogiannos, K.L. Stewart, D. Srivastava, S.P. Hirani,
  S. Hilton, S.M. Weldon, C. Malamateniou
- Journal: Radiography
- Volume: 32
- Year: 2026
- Article number: 103413
- DOI: 10.1016/j.radi.2026.103413
- PII / local PDF: S1078817426000891
- Open access under CC BY according to the PDF.

Why this is the right Wayne citation:

- It is explicitly about digital twinning in MRI, not only healthcare digital
  twins in general.
- It defines DT as dynamic virtual replicas bidirectionally linked to real-world
  counterparts, with real-time data integration, simulation and analytics.
- It systematically reviewed original DT-MRI articles from January 2020 to June
  2025 using PRISMA-style methods and included 51 studies.
- It found the DT-MRI literature is concentrated in diagnosis/treatment planning
  and monitoring (63%) and hardware/protocol/infrastructure (20%), while
  safety/QA, operational efficiency, training/education, and sustainability are
  much smaller themes.
- It identifies MRI-specific barriers directly relevant to this report:
  heterogeneous MRI data across vendors, pulse sequences and field strengths;
  high computational demands for complex simulations; need for large labelled
  datasets; hardware variability; governance, privacy and interoperability.
- It argues that training, safety and operational applications remain
  understudied. That supports framing this project as contributing to the
  underdeveloped "hardware, protocol and infrastructure" side of MRI digital
  twins rather than the already dominant diagnostic/treatment-planning side.

Report interpretation:

- Your work is not a full clinical DT because it is not dynamically connected
  to a live scanner or patient data stream. Do not overclaim.
- It is defensible to call it an executable phantom digital twin or
  calibration-object twin: a structured virtual replica of a physical MRI
  phantom, with known geometry/materials and simulation/fitting interfaces.
- Greggio et al. help you explain why this matters: MRI DT translation is held
  back by protocol heterogeneity, computational cost and validation barriers.
  Your project addresses those in a bounded setting by standardising the phantom
  model, validating KomaMRI for qMRI recovery, and creating a multi-fidelity RL
  training loop.

Placement:

- Chapter 2, new subsection "Digital twins, phantoms and simulator validation".
- One sentence in Chapter 3 aim:
  "In the terminology of medical digital twins, this is a deliberately bounded
  calibration-object twin rather than a patient twin: its value is traceable
  ground truth and executable simulation for method development."
- Chapter 5 multi-fidelity discussion, when motivating why computational cost
  is not only an RL inconvenience but an MRI-DT translation barrier.

BibTeX replacement for `digitaltwin_review_2026`:

```bibtex
@article{greggio_digital_twinning_mri_2026,
  author  = {Greggio, J. and Stogiannos, N. and Stewart, K. L. and Srivastava, D. and Hirani, S. P. and Hilton, S. and Weldon, S. M. and Malamateniou, C.},
  title   = {Exploring digital twinning in {MRI}: A systematic review of current applications, barriers, and future opportunities},
  journal = {Radiography},
  year    = {2026},
  volume  = {32},
  pages   = {103413},
  doi     = {10.1016/j.radi.2026.103413},
  url     = {https://doi.org/10.1016/j.radi.2026.103413}
}
```

Optional broader DT reviews if you want an extra general definition source:

```bibtex
@article{kamel_boulos_digital_twins_2021,
  author  = {Kamel Boulos, Maged N. and Zhang, Peng},
  title   = {Digital Twins: From Personalised Medicine to Precision Public Health},
  journal = {Journal of Personalized Medicine},
  year    = {2021},
  volume  = {11},
  number  = {8},
  pages   = {745},
  doi     = {10.3390/jpm11080745}
}

@article{katsoulakis_digital_twins_health_2024,
  author  = {Katsoulakis, Evangelos and Wang, Qian and Wu, Hao and Shahriyari, Leili and Fletcher, Ryan and Liu, Jing and Achenie, Luke and Liu, Hongfang and Jackson, Paul and Xiao, Yilun and others},
  title   = {Digital twins for health: a scoping review},
  journal = {npj Digital Medicine},
  year    = {2024},
  volume  = {7},
  pages   = {77},
  doi     = {10.1038/s41746-024-01073-0}
}
```

### qMRI phantoms and ground truth

#### Keenan et al. (2024) - review suggested by Wayne

Metadata:

- Title: Phantoms for quantitative body MRI: a review and discussion of the
  phantom value
- Journal: MAGMA
- DOI: 10.1007/s10334-024-01181-8
- Existing key in repo: `keenan_phantoms_2024`

Useful points:

- The paper reviews the role of phantoms in quantitative body MRI.
- Main report use: phantoms provide known values and repeatable structures for
  validating quantitative measurements, sequence performance, scanner
  dependence, and cross-site reproducibility.
- It supports the "why a phantom twin matters" argument. qMRI cannot be judged
  only by visual image quality because the target is a numerical tissue
  parameter.

Placement:

- Chapter 2: cite in the new phantom subsection.
- Chapter 3: cite when introducing QalibreMD/NIST/ISMRM phantom as a
  calibration device.
- Chapter 4: cite in the validation aim to support why parameter recovery is a
  stricter validation than visual plausibility.

Suggested prose seed:

> Quantitative MRI requires validation objects with known parameter values:
> the image can look plausible while the fitted T1/T2 values are biased. Recent
> qMRI phantom reviews therefore emphasise phantoms as traceable reference
> systems for scanner comparison, protocol optimisation and repeatability
> studies, rather than merely image-quality test objects.

#### QIBA/NIST system phantom and ISMRM/NIST context

Likely useful source:

- Stupic et al. / NIST ISMRM system phantom work is the canonical context for
  standardising T1/T2 measurements. Search terms if you want to add exact
  bibliography from Zotero: "NIST ISMRM MRI system phantom Stupic T1 T2".
- Russek et al. (ISMRM 2020 abstract 3384), "MRI Scanner Characterization with
  the ISMRM/NIST System Phantom", is a concise source for the phantom's intended
  scanner-characterisation role.

Main report use:

- The QalibreMD Model 130 phantom is not an arbitrary object. It belongs to a
  broader standardisation effort for qMRI system performance.
- This strengthens Chapter 3 and makes the digital twin look like an
  implementation of a recognised calibration artefact, not a private toy
  phantom.
- Russek et al. explicitly describe the system phantom as a baseline for
  measuring geometric distortion, nonlinear gradient correction efficacy, slice
  profile, resolution, SNR, relaxation-time accuracy and proton density. They
  also emphasise SI traceability, precision, long-term stability and monitoring
  by a national metrology institute.
- Useful nuance: the standard phantom protocols are presented as common
  baselines, not necessarily optimal measurement methods. That aligns well with
  your report: fixed phantom protocols provide the benchmark, while the project
  asks whether adaptive sequence design can improve a narrower qMRI objective.

Suggested prose seed:

> The physical phantom used here is valuable because it is tied to the
> NIST/ISMRM system-phantom standardisation effort: the contrast spheres provide
> known T1, T2 and proton-density-like values across clinically relevant ranges.
> The digital twin therefore inherits a meaningful validation target.

Alternative/stronger prose:

> The ISMRM/NIST system phantom was designed as a common scanner-characterisation
> baseline, with SI-traceable reference materials and structures for geometric
> distortion, slice profile, resolution, SNR, proton density and relaxation-time
> assessment. This project reuses that calibration target in simulation: rather
> than proposing a new phantom, it builds an executable version of an established
> reference object for adaptive sequence-design experiments.

Suggested BibTeX:

```bibtex
@inproceedings{russek_system_phantom_characterization_2020,
  author    = {Russek, Stephen E. and Boss, Michael A. and Charles, H. Cecil and Dienstfrey, Andrew M. and Evelhoch, Jeffrey L. and Gunter, Jeffrey L. and Hill, Derek L. G. and Jackson, Edward F. and Keenan, Kathryn E. and Liu, Guoying and Martin, Michele and Rentz, Nikki S. and Stupic, Karl F. and Yuan, Chun and Gimbutas, Zydrunas},
  title     = {{MRI} Scanner Characterization with the {ISMRM}/{NIST} System Phantom},
  booktitle = {Proceedings of the International Society for Magnetic Resonance in Medicine},
  year      = {2020},
  number    = {3384},
  url       = {https://cds.ismrm.org/protected/20MProceedings/PDFfiles/3384.html}
}
```

#### Multi-centre material-value standardisation: Pasini et al. (2025)

Source:

- Pasini et al. (2025), "Multi-center and multi-vendor evaluation study across
  1.5 T and 3 T scanners (part 2): T1 and T2 standardization in the ISMRM/NIST
  MR phantom", MAGMA, DOI: `10.1007/s10334-025-01260-4`.

Why it matters:

- This is useful around the material-values table, but probably not inside the
  table itself.
- The paper is recent evidence that the ISMRM/NIST phantom values are actively
  used for multi-site and multi-vendor T1/T2 standardisation, not just listed in
  a manufacturer manual.
- It evaluated T1 and T2 measurements on 13 scanners at 1.5 T and 3 T, across
  3 vendors and 7 sites, comparing against room-temperature reference values.
- It reported excellent correlation with reference values, good short-term
  reproducibility and inter-scanner agreement, with median inter-scanner CV
  below 7% for both T1 and T2 and below 5% in the renal range.
- It also reinforces that temperature matters: 3 T reference values were
  available at 16, 18, 20, 22, 24 and 26 degrees C, while 1.5 T values were
  given at 20 degrees C; the phantom includes MR-readable thermometer vials.
- It used PhantomViewer for centralised image processing. A single operator
  manually placed circular ROIs on the central slice while avoiding vial edges.
  This is a useful concrete link between current multi-site system-phantom
  studies and the open PhantomViewer/MR-BIAS analysis-tool ecosystem.

How to use:

- Add one compact paragraph before or after the material table:

> The reference values in Table X are not merely nominal constants from the
> phantom manual. The ISMRM/NIST system phantom has been used in recent
> multi-site, multi-vendor T1/T2 standardisation studies across 1.5 T and 3 T
> scanners, where measured relaxation values showed excellent correlation with
> reference values and good inter-scanner agreement. This supports using the
> phantom tables as a quantitative validation target, while also explaining why
> this implementation records field strength and temperature assumptions
> explicitly.

- This strengthens the material-values section without adding a large new
  table. The actual table should remain your implementation values.
- If space is tight, cite this in a sentence in the Chapter 3 physical phantom
  section and keep the detailed point in the Summary/Future Work limitations:
  your current package fixes the values at 20 degrees C, while future sim-to-real
  validation should implement temperature correction.
- Figure TODO: consider including Pasini et al.'s system-phantom figure
  showing `(a)` coronal view of the central plate, `(b)` temperature-vial ROI
  size/positioning, `(c)` NiCl2 vial layout, and `(d)` MnCl2 vial layout as a
  clearer source-backed plate-construction figure. It may be too busy as a
  four-panel figure for the main report, so leave as TODO unless it replaces an
  existing figure rather than adding a new one. Use the figure unchanged with
  attribution; do not crop, relabel or modify it because the PMC page lists the
  article as CC BY-NC-ND.

Suggested BibTeX:

```bibtex
@article{pasini_multicenter_nist_t1t2_2025,
  author  = {Pasini, Siria and Ringgaard, Steffen and Vendelboe, Tau and Garcia-Ruiz, Leyre and Strittmatter, Anika and Villa, Giulia and Raj, Anish and Echeverria-Chasco, Rebeca and Bozzetto, Michela and Brambilla, Paolo and others},
  title   = {Multi-center and multi-vendor evaluation study across 1.5 {T} and 3 {T} scanners (part 2): {T1} and {T2} standardization in the {ISMRM}/{NIST} {MR} phantom},
  journal = {Magnetic Resonance Materials in Physics, Biology and Medicine},
  year    = {2025},
  volume  = {38},
  number  = {3},
  pages   = {611--627},
  doi     = {10.1007/s10334-025-01260-4}
}
```


#### Open phantom reproducibility: CAD/STL hardware projects

Useful repositories:

- `usnistgov/open-source-mri-phantom`
  (`https://github.com/usnistgov/open-source-mri-phantom`): an open-source MRI
  phantom project that includes CAD/STL-style design artefacts for fabricating
  phantom components by 3D printing.
- `MRIStandards/SystemPhantom`
  (`https://github.com/MRIStandards/SystemPhantom`): the ISMRM/NIST system
  phantom repository, also containing hardware design files including STL files.

How to frame:

- These repositories show active interest in reproducibility and open-source
  practice around MRI calibration phantoms. They are useful evidence that the
  qMRI community is not only publishing phantom papers, but also releasing
  hardware artefacts so others can manufacture or inspect reference objects.
- Do not overclaim this as "digital twinning coming to calibration phantoms".
  A CAD/STL model is a reproducible physical design, not automatically a digital
  twin in the Greggio et al. sense, because it is not dynamically linked to a
  physical scanner/phantom or used for predictive updating.
- The best wording is: "open calibration-phantom infrastructure". Your work
  sits next to it by adding an executable simulation layer rather than a
  manufacturable hardware layer.

Suggested prose seed:

> Open phantom repositories such as the NIST open-source MRI phantom and the
> ISMRM/NIST system phantom release CAD/STL design files for fabricating
> calibration hardware. This shows the same reproducibility pressure that
> motivates this project, but at a different layer: those projects make the
> physical reference object reproducible, whereas `MRISystemPhantom.jl` makes a
> KomaMRI-executable version of a reference object available for simulation,
> validation and RL episode generation.

Suggested citation handling:

- GitHub repositories can be cited as software/data artefacts if used. If page
  budget is tight, mention them in one sentence and cite the repository URLs in
  a footnote rather than adding full BibTeX entries.
- They are especially useful in Chapter 3, immediately after introducing the
  physical phantom, to show that open phantom artefacts are part of the broader
  reproducibility context.

#### MR-BIAS: open qMRI phantom-analysis software

Source:

- Predecessor/open NIST tool: PhantomViewer,
  `https://github.com/MRIStandards/PhantomViewer`
- GitHub: `https://github.com/JamesCKorte/mrbias`
- Paper: "MR-BIAS: An open-source software tool for automated image analysis of
  the ISMRM/NIST MRI system phantom", Physics in Medicine & Biology, DOI:
  `10.1088/1361-6560/acbcbb`.

Why it matters:

- PhantomViewer and MR-BIAS are the closest related artefacts to acknowledge
  because they are open-source tools for analysing acquired MRI phantom data.
  PhantomViewer is the older NIST Python GUI analysis package for MRI phantoms
  developed by NIST and collaborators; its README states that it includes a
  `VPhantom` class describing several virtual phantoms, including the
  NIST/ISMRM system phantom, and uses `lmfit` for nonlinear least-squares
  fitting.
- PhantomViewer is still relevant in current literature: Pasini et al.'s 2025
  multi-site/multi-vendor ISMRM/NIST T1/T2 standardisation study used
  PhantomViewer for centralised image processing with manually placed ROIs.
- MR-BIAS is the stronger direct comparator because it is newer,
  open-source, targets the ISMRM/NIST MRI system phantom, and automates
  phantom-based qMRI analysis. It explicitly addresses PhantomViewer's manual
  ROI-analysis burden and inter-observer variability.
- It does overlap with your goals at the level of reproducible qMRI phantom
  tooling.
- It also includes T1 fitting, so the comparison should be explicit rather than
  pretending MR-BIAS is only ROI software. The paper states that MR-BIAS fits
  T1 with:
  - variable inversion recovery (T1-VIR), using a magnitude inversion-recovery
    signal with equilibrium magnetisation `M0`, inversion factor `Finv`,
    repetition time `TR`, inversion time `TIR`, fitted `T1`, and an additive
    noise/background factor `n`. `Finv` is the inversion efficiency/factor:
    ideal 180 degree inversion corresponds to `Finv = 2`, because the
    long-TR signal becomes `M0(1 - 2 exp(-TI/T1))`; imperfect inversion can be
    absorbed by fitting `Finv`;
  - variable flip angle (T1-VFA), using the usual spoiled-gradient-echo
    variable-flip-angle signal model with `M0`, `TR`, flip angle `alpha`, and
    fitted `T1`.
  It fits these parametric models with non-linear least squares using the
  Levenberg-Marquardt method through `lmfit`.
- The distinction is still clear:
  - MR-BIAS is post-acquisition analysis/QA software for phantom images,
    focused on hardware/protocol validation and bias/repeatability analysis.
  - `MRISystemPhantom.jl` is a simulator-facing digital phantom package that
    builds KomaMRI phantoms/sequences, provides ground truth labels, supports
    pose/material randomisation, and feeds an RL environment.
  - MR-BIAS helps analyse real scanner data from a standard phantom; your
    package helps generate synthetic scanner data and rewards for adaptive
    sequence design.
  - Your T1 fitter is related to MR-BIAS's T1-VIR family, but not the same
    workflow. It does not fit a free `Finv`; it takes the preparation angle
    `theta_inv` from the simulated sequence, so for standard E2 inversion
    `theta_inv = pi` and the effective inversion factor is fixed at
    `1 - cos(pi) = 2`. It also accepts per-block `TR`, per-block excitation
    angle `alpha_exc`, and an optional finite-`Npe` transient model for the
    phase-encode shots in a 2D block.
  - MR-BIAS includes a flip angle in its T1-VFA model, but that is not the same
    object as your IR-SE excitation angle. In MR-BIAS T1-VFA, flip angle is the
    swept contrast variable of a spoiled-GRE-style T1 mapping protocol. In your
    E2 fitter, `alpha_exc` is the excitation pulse inside an IR-SE block; it
    affects signal projection and the longitudinal carryover between repeated
    short-TR shots.
  - Therefore MR-BIAS could not be dropped into E2 directly. To reuse it for
    this project, it would need a simulator-side interface, per-block adaptive
    acquisition parameters, the IR-SE excitation-angle/state-carryover model,
    finite-`Npe` transient handling, and repeated refitting after each RL action.
- Concrete examples:
  - Conventional MR-BIAS T1-VIR input looks like a fixed image series:
    `TI = [50, 100, 200, 400, 800, 1600] ms`, one protocol, usually one
    `TR`, one inversion-recovery signal model, and one final fit per ROI after
    all images have been acquired. The fitter can treat the series as a
    conventional T1 mapping scan and estimate `T1` plus nuisance parameters
    such as `M0` and `Finv`.
  - An E2 episode is a sequence of agent-selected blocks, for example:
    block 1 `(TI=0.80 s, TR=2.2 s, alpha_exc=90 deg, Npe=32)`,
    block 2 `(TI=0.12 s, TR=0.7 s, alpha_exc=90 deg, Npe=32)`,
    block 3 `(TI=0.45 s, TR=1.1 s, alpha_exc=90 deg, Npe=32)`.
    After each block, the environment reconstructs that block, extracts ROI
    means, refits T1 using the history so far, and feeds the updated estimate
    back to the policy. The fit therefore needs the executed `TR` and
    `alpha_exc` for each block, not just the list of TIs.
  - The finite-`Npe` issue is specific to the 2D simulator loop. One E2 block
    is not a single instantaneous measurement at `TI`; it is 32 phase-encode
    shots. If `TR` is short, shot 1 starts from thermal equilibrium but later
    shots start from the longitudinal magnetisation left by previous shots.
    Your fitter models the mean `Mz_at_excite` across those 32 shots. A
    conventional post-acquisition T1-VIR fit would treat the block more like
    one image sample at one `TI`, missing this transient ramp.
  - The excitation-angle distinction matters in runs where `alpha_exc` is not
    fixed at 90 degrees. Suppose the agent chooses `alpha_exc=30 deg` for a
    block. The transverse signal is scaled by `sin(30 deg)`, and the remaining
    longitudinal magnetisation after excitation is scaled by `cos(30 deg)`,
    affecting the next shot in the same phase-encode train. MR-BIAS's T1-VFA
    flip angle model is a different spoiled-GRE acquisition model; it does not
    describe this IR-SE shot-to-shot carryover.
- This comparison is worth adding because it makes the novelty more credible:
  you are not ignoring the nearest open-source phantom-analysis work, but
  explaining why a simulator-side twin is a different contribution.

Suggested Chapter 3 prose:

> Existing open-source tools already support analysis of acquired ISMRM/NIST
> phantom data. PhantomViewer provides a Python GUI and virtual-phantom
> descriptions for NIST phantoms, while MR-BIAS automates analysis of acquired
> system-phantom images, including Levenberg--Marquardt fits for T1-VIR,
> T1-VFA and T2-MSE models. Their focus is hardware and protocol validation
> after acquisition. They would not directly support the adaptive E2 setting,
> because the RL agent changes TR and timing block by block, and the fitter must
> know the IR-SE excitation angle and the finite-Npe transient across
> phase-encode shots. The contribution here is complementary: the phantom is
> represented before acquisition as a KomaMRI object with known material labels
> and randomisation hooks, and the fitting model is embedded inside a
> simulator/RL loop, so it can generate synthetic measurements and rewards for
> adaptive sequence-design experiments.

Suggested BibTeX:

```bibtex
@article{korte_mrbias_2023,
  author  = {Korte, James C. and others},
  title   = {{MR-BIAS}: An open-source software tool for automated image analysis of the {ISMRM}/{NIST} {MRI} system phantom},
  journal = {Physics in Medicine \& Biology},
  year    = {2023},
  doi     = {10.1088/1361-6560/acbcbb}
}

@misc{phantomviewer_repo,
  author = {{MRI Standards}},
  title  = {{PhantomViewer}: Python analysis package for {MRI} phantoms},
  year   = {2013},
  url    = {https://github.com/MRIStandards/PhantomViewer},
  note   = {Accessed 11 June 2026}
}
```

Note: fill the complete author list/pages from Zotero/IOP before final
submission if you cite it. Do not leave "and others" in a final Vancouver-style
bibliography if you can export full metadata.

### MRI simulators and simulator validation

#### KomaMRI.jl

Existing key: `komaMRI:23`

Useful points from the paper abstract already in `refs/komaMRI.bib`:

- Open-source Julia framework for general MRI simulations.
- Solves Bloch equations with CPU/GPU parallelisation.
- Pulseq-compatible sequence input.
- Raw data can be stored in ISMRMRD format.
- Compared against JEMRIS and MRiLab; mean absolute differences below 0.1% vs
  JEMRIS were reported in their tests.
- Koma was reported as easier/faster in user/usability experiments and used to
  simulate MRF acquisitions.

How to use:

- Chapter 4 should not sound like "KomaMRI was unvalidated before this project".
  It was peer-reviewed and benchmarked.
- Your contribution is more specific: an end-to-end qMRI recovery test at long
  cumulative sequence time, under the exact long adaptive acquisitions the RL
  environment needs.

Suggested prose seed:

> KomaMRI had already been benchmarked against established simulators such as
> JEMRIS and MRiLab. The validation gap for this project was narrower but
> important: whether a long sequence assembled from many adaptive blocks still
> preserves quantitative T1 recovery when passed through simulation,
> reconstruction, ROI extraction and fitting.

#### MRIReco.jl: surrounding reconstruction tooling

Source:

- Knopp and Grosser (2021), "MRIReco.jl: An MRI reconstruction framework written
  in Julia", Magnetic Resonance in Medicine, DOI: `10.1002/mrm.28792`.
- Local experiment: `scripts/archive/t1_fit_vs_true_MReco.jl`.

Why it matters:

- MRIReco is relevant surrounding work because KomaMRI itself sits in the Julia
  MRI ecosystem and can interface with reconstruction tooling. The KomaMRI paper
  also mentions MRIReco as its reconstruction path.
- Your validation pipeline mainly uses an in-house Cartesian `raw_to_kspace`
  plus `kspace_to_image` IFFT path, not MRIReco, because E2 uses simple fully
  sampled Cartesian acquisitions where a direct FFT is enough.
- You did experiment with MRIReco in `scripts/archive/t1_fit_vs_true_MReco.jl`.
  That script ported the T1-fit-vs-truth validation pipeline to MRIReco:
  `build_phantom -> cr_optimize -> ir_se_2d_sequence -> simulate ->
  AcquisitionData -> MRIReco reconstruction -> ROI -> fit_t1_generalized_ir`.
- The archived notes say MRIReco matched the in-house FFT reconstruction for
  the basic Cartesian path, but was around 60% slower for this simple use case
  because it builds encoding operators and handles trajectory metadata. That is
  a good engineering justification for keeping the lightweight in-house IFFT in
  the RL hot path. In normal offline analysis a 60% reconstruction overhead
  might be acceptable; in PPO it is paid at every environment step, so it
  directly reduces the number of simulated acquisitions the agent can experience
  within the wall-clock budget.
- The same notes also record integration gotchas: KomaMRI raw headers/sequence
  definitions had to expose `Nx/Ny`, the trajectory needed to be marked
  Cartesian, and MRIReco returns FE-first image dimensions, unlike the in-house
  PE-first convention.
- This is not a main contribution, but it is worth mentioning briefly in
  Chapter 4 or an appendix to show that the reconstruction component was
  considered and cross-checked against a standard Julia reconstruction package.

How to use:

- Chapter 4, near "Reconstruction explanations were tested":

> The reconstruction path was also checked against MRIReco.jl, a general Julia
> MRI reconstruction framework. For the fully sampled Cartesian data used here,
> MRIReco agreed with the direct FFT path, but added avoidable overhead and
> metadata handling. Because reconstruction sits inside every RL environment
> step, this overhead would directly reduce the number of policy rollouts
> possible under the project wall-clock budget. The report therefore uses the
> simpler in-house FFT reconstruction for the RL loop while reserving MRIReco
> for future non-Cartesian, parallel-imaging or iterative reconstructions.

- Do not overemphasise this in the main text. It is supporting validation, not
  an achievement. A footnote or appendix sentence may be enough.

Suggested BibTeX:

```bibtex
@article{knopp_mrireco_2021,
  author  = {Knopp, Tobias and Grosser, Martin},
  title   = {{MRIReco.jl}: An {MRI} reconstruction framework written in {Julia}},
  journal = {Magnetic Resonance in Medicine},
  year    = {2021},
  volume  = {86},
  number  = {3},
  pages   = {1633--1646},
  doi     = {10.1002/mrm.28792},
  pmid    = {33817833}
}
```

#### MRiLab

Suggested source:

- Liu et al. (2017), MRiLab numerical MRI simulation package / fast realistic
  MRI simulations. DOI: 10.1109/TMI.2016.2620961.

Main use:

- Mention as established simulator comparator, not as central related work.
- Supports "MRI simulation is a mature area; this project contributes
  validation in a specific adaptive qMRI regime".

#### JEMRIS

Suggested source:

- Stoecker et al./JEMRIS paper, open-source MRI sequence design/simulation
  framework.

Main use:

- Same as MRiLab: use in a compact simulator comparison sentence/table only.

#### Pulseq

Existing key: `Pulseq:17`

Use:

- Pulseq is a sequence-description interoperability standard, not a simulator.
- Useful to explain why returning standard KomaMRI `Sequence` objects and using
  Pulseq-compatible infrastructure matters for future scanner deployment.

### Automated and adaptive MRI sequence design

#### MRzero - supervised differentiable sequence discovery

Existing key: `MRISeqSupLearning:21`

Useful points:

- End-to-end differentiable simulator/reconstruction framework.
- Optimises RF events, gradient moments and delay times from scratch against
  target contrast maps.
- Includes SAR and scan-time terms.
- Demonstrated translation to a real 3T Siemens Prisma scanner with phantom and
  in-vivo brain measurements.

Critical comparison:

- Stronger hardware validation than this project.
- But sequence optimisation is offline/supervised: it learns a fixed sequence
  or reconstruction strategy, not an observation-conditioned policy that changes
  later acquisitions based on fitted tissue estimates.
- It uses differentiability through the sequence/reconstruction model; this
  project treats KomaMRI as a black-box simulator and uses policy gradients
  through action probabilities only.

Placement:

- Chapter 2 sequence-design subsection.
- Chapter 5 related-work table already has this row; keep it.

Suggested prose:

> MRzero shows that learned pulse-sequence design can reach hardware, but it
> optimises a sequence offline against a target contrast. The gap for this
> report is closed-loop quantitative acquisition: the policy must choose the
> next timing from the current fitted T1 estimates during the scan.

#### AUTOSEQ - Bayesian RL-style sequence generation

Existing key: `zhu2018autoseq`

Useful points:

- Early framing of pulse-sequence generation as a game/sequential-control
  problem.
- Bayesian RL/active search in a simplified MRI physics simulator.
- Demonstrated discovery of canonical sequence behaviours such as gradient
  echo-like encoding in preliminary 1-D experiments.

Critical comparison:

- Very relevant conceptually because it casts sequence design as sequential
  control.
- Much simpler validation task than this project: preliminary 1-D setting, not
  fitted qMRI on a spatially encoded calibration phantom.
- It is closer to autonomous pulse-sequence discovery, while this project uses
  higher-level IR-SE blocks and tests adaptive timing.

Placement:

- Chapter 2 sequence-design subsection.
- Chapter 5 table already has this row.

#### Walker-Samuel (2019) - RL control of simulated MRI scanner

Existing key: `Walk_RL_CONTROL_MRI:19`

Useful points:

- Deep RL agent controls a simulated MRI scanner.
- Task-specific active sensing: classify phantom shapes while balancing image
  quality and acquisition time.
- Reported 99.8% shape-classification accuracy in the abstract/proceedings
  material.
- Agent learned behaviour resembling edge detection and sparse k-space use.

Critical comparison:

- Closest prior RL scanner-control comparator.
- Strong evidence that RL can control MRI-like acquisition decisions.
- But objective is classification, not quantitative parameter recovery.
- It is simulation-only and a simpler task; your project is also simulation-only
  but uses qMRI fitting error on a calibrated phantom and 2-D reconstruction.

Placement:

- Chapter 2 adaptive/RL subsection.
- Chapter 5 table already includes; keep caveat "closest direct deep-RL
  comparator".

#### Adaptive model-based MR - Beracha, Seginer and Tal (2023)

Existing key: `adaptive_mri:23`

Useful points from existing abstract:

- Adaptive real-time multi-echo experiment for estimating T2s.
- Bayesian framework + model-based reconstruction.
- Maintains and updates prior distribution of tissue parameters during scan.
- Simulations predicted 1.7-3.3x acceleration over static sequences.
- Phantom experiments corroborated simulations.
- Healthy volunteers: T2 measurement of N-acetyl-aspartate accelerated by 2.5x.

Critical comparison:

- Stronger external validation than this project: includes phantom and human
  volunteer data.
- It is model-based adaptive selection, not RL policy learning.
- It targets multi-echo T2/MRS-style estimation rather than RL over a
  spatially encoded T1 phantom pipeline.
- It is the strongest source to justify that adaptive MR is a real direction,
  not just a speculative RL idea.

Placement:

- Chapter 2 "Adaptive MRI acquisition".
- Chapter 5 related-work table already has this row.

Suggested prose:

> Adaptive model-based MR is the clearest precedent for the central idea that
> incoming subject data should change later acquisition parameters. The
> difference is the controller: Beracha et al. use Bayesian model-based
> selection, whereas this project learns a reusable policy in a Gymnasium
> environment and evaluates it against fixed schedules.

#### Magnetic Resonance Fingerprinting (MRF)

Existing keys:

- `mehta_mrf_review:2019`
- `Jordan_MRF_AUTO:21`

Useful points:

- MRF varies acquisition parameters over time and fits tissue parameters from a
  signal trajectory/dictionary.
- Jordan et al. automate MRF schedule design with physics-inspired optimisation.
- Their optimisation is directly relevant to qMRI sequence scheduling and
  multi-parameter fingerprints.

Critical comparison:

- MRF is qMRI and schedule design, so it is highly relevant.
- But standard MRF and automated MRF design are still usually open-loop: the
  schedule is chosen before acquisition, then applied.
- Your project's differentiator is observation-conditioned adaptation during
  acquisition, even though the action space is narrower and the results are
  simulated.

Placement:

- Chapter 2 MRF subsection already exists.
- Add one transition sentence from MRF to adaptive RL:
  "MRF shows the value of time-varying contrast schedules; this project asks
  whether the schedule should also depend on measurements already observed."

### Active MRI acquisition and k-space RL

Why include:

- Markers may know RL has been used for MRI acquisition. You need to distinguish
  k-space sampling/reconstruction RL from pulse-sequence/qMRI adaptation.
- This can be a single paragraph or one extra row in the Chapter 5 table.

Useful sources:

1. Pineda et al., Active MR k-space Sampling with Reinforcement Learning
   (MICCAI 2020).
2. Related works on active acquisition use policies to select k-space lines or
   sampling masks to improve reconstruction quality.

Critical comparison:

- These methods optimise which data to sample in k-space, usually with a
  reconstruction/image-quality objective.
- They generally do not choose RF/relaxation timing parameters such as TI/TR/TE,
  and they do not optimise fitted T1/T2 error against a calibrated qMRI phantom.
- This makes them adjacent but not direct comparators.

Placement:

- Add one row to Chapter 5 related-work table:

| Work | What it demonstrates | Critical comparison |
|---|---|---|
| Active k-space RL | RL can choose future k-space measurements to improve reconstruction | Adjacent acquisition-control literature. It usually controls sampling masks/lines and optimises image reconstruction, whereas this project controls sequence timing and optimises fitted T1 error |

Suggested BibTeX:

```bibtex
@inproceedings{pineda_active_mri_rl_2020,
  author    = {Pineda, Luis and Basu, Sumana and Romero, Adriana and Calandra, Roberto and Drozdzal, Michal},
  title     = {Active MR k-space Sampling with Reinforcement Learning},
  booktitle = {Medical Image Computing and Computer Assisted Intervention -- MICCAI 2020},
  year      = {2020},
  pages     = {23--33},
  publisher = {Springer},
  doi       = {10.1007/978-3-030-59713-9_3}
}
```

## Suggested report insertions

### Chapter 2: new subsection after MRF or before simulators

Suggested title:

`Digital twins, phantoms and simulator validation`

Suggested compact prose:

> The adaptive acquisition loop in this report requires a simulator whose
> outputs are not only visually plausible but quantitatively correct. This is
> where phantoms and digital twins enter the project. In qMRI, the target is a
> fitted physical parameter such as T1 or T2, so validation requires reference
> objects with known parameter values. Recent qMRI phantom reviews therefore
> treat phantoms as traceable reference systems for repeatability, protocol
> comparison and scanner validation, not only as image-quality objects. Medical
> digital-twin reviews make the complementary point that a useful twin is an
> executable representation used for prediction, optimisation and testing. The
> phantom twin in this project is deliberately narrower than a patient digital
> twin: it represents a calibration object rather than a person. Its value is
> that it provides known geometry, known relaxation values, controlled
> randomisation and a direct route into the Bloch simulator.

> This framing leaves a gap between existing work and this project. Physical
> phantoms provide ground truth but do not by themselves generate unlimited
> training episodes. MRI simulators provide flexible forward models but require
> task-specific validation. Automated sequence-design systems can optimise
> schedules, but are often offline or evaluated on target contrast rather than
> fitted qMRI error. This project combines the pieces: an executable phantom
> twin drives a validated KomaMRI pipeline, and an RL agent is trained and
> evaluated on the quantitative recovery objective.

Possible compact table:

| Related approach | Strength | Remaining gap for this project |
|---|---|---|
| Physical qMRI phantoms | Known T1/T2/PD values; repeatable scanner/protocol validation | Do not create randomised simulation episodes or cheap RL rollouts |
| Medical digital twins | Executable models for prediction/optimisation | Usually patient/system-level; not a ready qMRI RL environment |
| MRI simulators (KomaMRI/MRiLab/JEMRIS) | Bloch-equation forward models; sequence testing before scanner execution | Need end-to-end qMRI validation under long adaptive acquisitions |
| Offline sequence optimisation (MRzero/MRF optimisation) | Learns or optimises non-trivial schedules | Schedule fixed before scan; not observation-conditioned RL |
| Adaptive MR/RL scanner control | Shows closed-loop acquisition can help | Prior tasks differ: Bayesian T2/MRS adaptation, classification, or k-space sampling rather than fitted T1 mapping |

References to cite in the table/prose:

- `keenan_phantoms_2024`
- `greggio_digital_twinning_mri_2026`
- optionally `kamel_boulos_digital_twins_2021` or
  `katsoulakis_digital_twins_health_2024` for broader digital-twin framing
- `komaMRI:23`
- `MRISeqSupLearning:21`
- `adaptive_mri:23`
- `Walk_RL_CONTROL_MRI:19`
- optionally `pineda_active_mri_rl_2020`

### Chapter 3: strengthen novelty/related-work positioning

Current Chapter 3 aim is good. Add one paragraph after the physical phantom
intro or in the chapter aim:

> Existing qMRI phantom work usually studies physical reference objects and
> scanner/protocol repeatability. The contribution here is not a new physical
> phantom, but an executable, package-tested twin of a recognised calibration
> phantom that can generate KomaMRI phantoms, return ground-truth labels, and
> vary pose/material values across RL episodes. This makes it a method-development
> tool rather than only a validation object.

Feedback addressed:

- 4: digital twin framing
- 5: related phantom work
- 6: novelty vs published work
- 12: hardest/novel parts

### Chapter 4: strengthen related-work positioning

Add to Chapter 4 aim after "KomaMRI is peer-reviewed...":

> KomaMRI had already been benchmarked as a general simulator, including
> comparisons against established tools. The validation gap here is different:
> adaptive qMRI needs the entire long-duration pipeline to preserve fitted T1
> values, because a small timing error can bias the quantitative estimate while
> still producing a plausible image.

Possible comparison table, if page space allows:

| Validation level | Example | What it checks | What it misses |
|---|---|---|---|
| Simulator benchmark | KomaMRI vs JEMRIS/MRiLab | Bloch solution agreement, speed, usability | Project-specific long-sequence qMRI recovery |
| Visual phantom image | Reconstructed phantom looks plausible | Geometry, rough contrast, reconstruction convention | Biased T1/T2 fits |
| This chapter | Noise-free IR-SE recovery on digital QalibreMD T1 plate | End-to-end quantitative correctness under long cumulative times | Hardware imperfections and other sequence families |

Feedback addressed:

- 3: related approaches and gap
- 6: chapter-specific related work
- 12: hardest parts and novelty

### Chapter 5: add active acquisition distinction

Add a short paragraph after the existing related-work table:

> A separate active-acquisition literature uses RL to choose k-space samples or
> masks for accelerated reconstruction. That work is relevant because it treats
> MRI acquisition as a sequential decision problem, but it controls a different
> object: sampling locations rather than relaxation-preparation timings. This
> chapter instead fixes Cartesian reconstruction and asks whether the policy can
> choose more informative inversion-recovery blocks for fitted T1 estimation.

Feedback addressed:

- 6: more complete related work in Chapter 5
- 12: novelty compared with adjacent work

Chapter 5 TODO for Wayne's "compare benefits" feedback:

- Add two compact comparison tables rather than a single leaderboard. Published
  results are not directly comparable because the tasks differ (adaptive T2/MRS,
  offline sequence optimisation, shape classification, k-space sampling, fitted
  T1 mapping). The report should still give quantitative anchors where possible,
  but must state that they are not cross-paper rankings.
- Table A: qualitative positioning by control problem, validation level and
  relation to this chapter.
  Suggested columns: `Work`, `Control problem`, `Validation`, `Relation to this
  chapter`.
  Suggested rows:
  - Adaptive model-based MR: Bayesian multi-echo/T2 or MRS parameter selection;
    simulation/phantom/healthy volunteers; stronger external validation, not RL.
  - MRzero: differentiable offline sequence optimisation; phantom and 3T in
    vivo demonstration; stronger hardware evidence, fixed learned sequence not
    closed-loop policy.
  - Walker-Samuel: deep RL scanner control for active sensing; simulation;
    closest RL comparator but classification objective.
  - Active k-space RL/Pineda: RL k-space sample selection; retrospective
    datasets; sequential acquisition, but sampling-mask/reconstruction objective
    rather than TI/TR timing.
  - This chapter: RL choice of IR-SE timing from fitted T1 state; validated
    KomaMRI phantom simulation; simulator-only but directly evaluates fitted T1
    error against fixed qMRI schedules.
- Table B: quantitative anchors with caveats.
  Suggested columns: `Work`, `Reported benefit`, `Metric`, `Comparability
  caveat`.
  Suggested rows:
  - Adaptive model-based MR: 1.7--3.3x simulated acceleration and 2.5x volunteer
    acceleration; T2/MRS acquisition time; real data but different model-based
    task.
  - MRzero: learned GRE-like/target-contrast sequences with scanner
    demonstration; target image/contrast loss; offline supervised optimisation.
  - Walker-Samuel: 99.8% shape-classification accuracy; classification accuracy;
    RL scanner control but not qMRI estimation.
  - This chapter Run A: negative full-plate result, e.g. 9.44--11.68% MAPE vs
    4.70% fixed CR; fitted T1 MAPE; important limit of adaptivity.
  - This chapter Run B: 4.62% vs 6.86% MAPE at 240 s; held-out fitted T1 MAPE;
    matched fixed baseline, simulator-only.
  - This chapter memory ablation: 2.93% vs 6.04% MAPE at 560 s; held-out fitted
    T1 MAPE; best controlled result, still five-sphere simulated task.
- Add a paragraph under the table:

> These numbers should not be read as a cross-paper ranking. They show that
> prior work has stronger hardware validation and, in some cases, larger
> demonstrated acceleration, while this chapter contributes a different
> experimental object: a closed-loop RL policy evaluated on fitted T1 error
> against matched fixed schedules in a validated 2-D Bloch-simulation phantom
> pipeline.

### Chapter 1/summary: sharpen achievement language

Chapter 1 already maps A1-A3 to chapters. Add only if space allows:

> The project is therefore not a claim that reinforcement learning is new to
> MRI, nor that digital twins are new to medicine. The contribution is the
> integration of these ideas into a validated qMRI sequence-design testbed.

This is useful because it pre-empts novelty overclaiming.

## Suggested reference-table for final report

This table can be adapted into Chapter 2 or Chapter 5. It is intentionally
compact.

| Work area | Representative work | What it contributes | Gap this project targets |
|---|---|---|---|
| qMRI phantoms | Keenan et al. 2024; NIST/ISMRM system phantom work | Ground-truth reference values and repeatability framework | Physical reference objects do not themselves create RL training environments |
| Open phantom hardware/software | NIST open-source MRI phantom; MRIStandards SystemPhantom; MR-BIAS | Reproducible phantom fabrication files and automated scanner-data analysis | Focused on hardware calibration or post-acquisition QA rather than simulator-side episode generation |
| MRI digital twins | Greggio et al. 2026 | Maps MRI-DT applications, barriers and underexplored operational/protocol opportunities | Existing MRI-DT work is mostly diagnosis/treatment planning; protocol optimisation and validation remain less mature |
| Medical digital twins | Kamel Boulos and Zhang 2021; Katsoulakis et al. 2024 | Executable, data-linked models for prediction and personalised optimisation | Broad patient/system-level framing; not a qMRI scanner-control testbed |
| MRI simulators | KomaMRI, MRiLab, JEMRIS | Bloch-equation simulation before scanner execution | Need task-specific validation under long adaptive qMRI sequences |
| Offline learned sequence design | MRzero; automated MRF optimisation | Learned/optimised static acquisition schedules | Schedule is set before acquisition; no policy conditioned on current fit |
| Adaptive model-based MR | Beracha et al. 2023 | Real-time Bayesian update and T2 acceleration on phantom/volunteers | Model-based selection rather than RL policy; different task |
| RL scanner control | Walker-Samuel 2019; AUTOSEQ | MRI as sequential-control problem | Simplified/classification/1-D tasks, not fitted qMRI on calibrated phantom |
| Active k-space acquisition | Pineda et al. 2020 | RL for choosing future k-space measurements | Controls sampling pattern, not TI/TR/TE sequence timing |
| This project | MRISystemPhantom + KomaMRI + PPO/multi-fidelity | Validated digital-phantom qMRI environment and adaptive timing policy | Still simulator-only and positive result is controlled five-sphere task |

## Novelty claims that are safe

Safe:

- "A reusable KomaMRI-compatible digital twin of the QalibreMD Model 130 phantom
  with ground-truth material labels, pose/material randomisation and qMRI fitting
  utilities."
- "A complementary simulator-side contribution to existing open phantom
  hardware and analysis tools: CAD/STL repositories make calibration phantoms
  physically reproducible, and MR-BIAS analyses acquired phantom images, while
  this project generates KomaMRI-compatible synthetic acquisitions for adaptive
  sequence-design experiments."
- "An end-to-end qMRI validation test that exposed long-cumulative-time
  floating-point failures not visible from qualitative image inspection."
- "A multi-fidelity Python/Julia Gymnasium environment for Bloch-in-the-loop
  adaptive qMRI sequence timing."
- "A controlled demonstration that observation-conditioned RL with explicit
  acquisition-history memory can beat matched fixed log-grid schedules on a
  continuous five-sphere T1 task."

Avoid or qualify:

- Avoid "first RL for MRI" - false/too broad.
- Avoid "first adaptive MRI" - false; adaptive model-based MR exists.
- Avoid "solves adaptive qMRI" - too broad; full 14-sphere result is negative.
- Avoid "clinically validated" - all learned-policy results are simulated.
- Avoid "digital twin of patient/clinical scanner" - it is a phantom twin, not
  a patient/scanner digital twin.
- Avoid implying that CAD/STL phantom repositories are already digital twins.
  They are open hardware/reproducibility artefacts unless dynamically linked to
  real-world data and simulation.

## Page-budget advice

The most mark-efficient additions are short and comparative. To make room:

- Move one or two Chapter 2 explanatory figures to appendix if they are not used
  later. The k-space/PSF material is useful but can be compressed.
- Keep the KomaMRI bug chapter focused on the diagnostic path and final
  quantified fix; move excessive minimal-reproducer detail to appendix if needed.
- Do not split multi-fidelity and RL into separate chapters at this stage unless
  it helps page flow. Splitting creates more chapter overhead and does not solve
  the page limit. A single Chapter 5 with strong internal sections is acceptable.
- Keep the new literature material as a table plus 2-3 paragraphs, not several
  pages.

## Additional BibTeX candidates

Only add these if cited in report text.

### Active k-space RL

```bibtex
@inproceedings{pineda_active_mri_rl_2020,
  author    = {Pineda, Luis and Basu, Sumana and Romero, Adriana and Calandra, Roberto and Drozdzal, Michal},
  title     = {Active MR k-space Sampling with Reinforcement Learning},
  booktitle = {Medical Image Computing and Computer Assisted Intervention -- MICCAI 2020},
  year      = {2020},
  pages     = {23--33},
  publisher = {Springer},
  doi       = {10.1007/978-3-030-59713-9_3}
}
```

### MRiLab simulator

```bibtex
@article{liu_mrilab_2017,
  author  = {Liu, Feng and Velikina, Julia V. and Block, Walter F. and Kijowski, Richard and Samsonov, Alexey A.},
  title   = {Fast realistic MRI simulations based on generalized multi-pool exchange tissue model},
  journal = {IEEE Transactions on Medical Imaging},
  year    = {2017},
  volume  = {36},
  number  = {2},
  pages   = {527--537},
  doi     = {10.1109/TMI.2016.2620961}
}
```

Note: MRiLab is also often cited via the toolbox/software website. Use a
publisher/IEEE source if adding it to the final bibliography.

### MRI digital twin review

```bibtex
@article{greggio_digital_twinning_mri_2026,
  author  = {Greggio, J. and Stogiannos, N. and Stewart, K. L. and Srivastava, D. and Hirani, S. P. and Hilton, S. and Weldon, S. M. and Malamateniou, C.},
  title   = {Exploring digital twinning in {MRI}: A systematic review of current applications, barriers, and future opportunities},
  journal = {Radiography},
  year    = {2026},
  volume  = {32},
  pages   = {103413},
  doi     = {10.1016/j.radi.2026.103413},
  url     = {https://doi.org/10.1016/j.radi.2026.103413}
}
```

### Broader digital twins reviews

```bibtex
@article{kamel_boulos_digital_twins_2021,
  author  = {Kamel Boulos, Maged N. and Zhang, Peng},
  title   = {Digital Twins: From Personalised Medicine to Precision Public Health},
  journal = {Journal of Personalized Medicine},
  year    = {2021},
  volume  = {11},
  number  = {8},
  pages   = {745},
  doi     = {10.3390/jpm11080745}
}

@article{katsoulakis_digital_twins_health_2024,
  author  = {Katsoulakis, Evangelos and Wang, Qian and Wu, Hao and Shahriyari, Leili and Fletcher, Ryan and Liu, Jing and Achenie, Luke and Liu, Hongfang and Jackson, Paul and Xiao, Yilun and others},
  title   = {Digital twins for health: a scoping review},
  journal = {npj Digital Medicine},
  year    = {2024},
  volume  = {7},
  pages   = {77},
  doi     = {10.1038/s41746-024-01073-0}
}
```

## Final writing priorities

When writing from this brief, optimise for marker perception:

1. Show you know the adjacent literature.
2. State exactly where your work sits relative to it.
3. Quantify your own benefit only where the experiment supports it.
4. Be explicit about limitations.
5. Keep every addition tied to a feedback item and a challenge/achievement.

The strongest final-report sentence is probably:

> The project does not claim that RL, digital twins, phantoms or MRI simulators
> are new individually; its contribution is to combine them into a validated,
> executable qMRI acquisition testbed and to show, under controlled conditions,
> when closed-loop timing improves over fixed protocols.

TODO for final Summary/Future Work:

- Add a forward-looking point that the `MRISystemPhantom.jl` architecture is not
  intrinsically tied to the QalibreMD Model 130. The useful reusable idea is the
  separation between:
  - geometry/material descriptors: sphere/plate positions, radii, labels,
    reference relaxation values, pose/material randomisation;
  - voxelisation/simulation backend: converting those descriptors into KomaMRI
    spin clouds at a chosen resolution, with independent water-grid fidelity.
- This means the package pattern could be extended to other qMRI phantoms,
  including open-hardware phantom designs with CAD/STL files, by writing new
  descriptor builders while reusing the voxelisation, randomisation,
  reconstruction, ROI extraction and fitting machinery.
- Suggested Summary/Future Work wording:

> A useful extension would be to make the phantom description layer explicitly
> multi-phantom. The current implementation already separates physical
> descriptors from voxelisation: the QalibreMD geometry is first represented as
> labelled material objects, then converted into KomaMRI spins at the requested
> fidelity. Other open calibration phantoms could therefore be supported by
> adding new descriptor builders while reusing the same voxelisation,
> randomisation, reconstruction and fitting pipeline. This would turn the
> library from a single-phantom twin into a small framework for executable qMRI
> phantom twins.

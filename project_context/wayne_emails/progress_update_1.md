Subject: Weekly Progress Update — Arthur Allilaire FYP

Hi Wayne,

I hope you're well. As you recommended, I'll be sending weekly updates from here on. Below is a summary of where I am, what I'm working on this week, and my revised timetable through to submission.


Tasks completed this week:

- Built a fully parameterised digital twin of the QalibreMD Model 130 phantom in KomaMRI (the MRI simulator). This includes all 14 T1-array spheres, 14 T2-array spheres, PD spheres, and fiducial markers — each with correct material properties and configurable spatial resolution.
- Completed a conventional (non-RL) baseline experiment (E0): used standard inversion-recovery and multi-echo spin-echo sequences to recover T1 and T2 values from the phantom, validating that the simulator and digital twin are working correctly.
- Completed a first RL experiment (E1): trained a PPO agent on a simplified single-voxel version of the phantom, where the agent chooses inversion time (TI) and flip angle (α) to estimate T1. The agent converges quickly, which confirms the pipeline works end-to-end but also confirms the task is too simple — motivating the scale-up this week.
- Established the Python–Julia bridge (juliacall) so the RL agent (Python / Stable-Baselines3) can call the physics simulator (Julia / KomaMRI) at every step.


Tasks to complete this week:

The main task is scaling the RL environment from a single voxel to the full 14-sphere T1-array plate, and adding the spatial realism that was missing in E1:

- Adding gradient encoding to the sequence blocks (slice-selection, readout, phase-encoding) so the simulator produces real 2D k-space data rather than a scalar signal.
- Reconstructing a 2D image from k-space at each step (FFT) and fitting per-sphere T1 maps as the agent's reward signal.
- Implementing complex Gaussian noise on the simulated raw data (both real and imaginary channels) for realistic evaluation.
- Adding domain randomisation: random phantom rotation and translation per episode so the agent can't memorise positions.

The success criterion for E2 is that the agent achieves < 5% mean T1 error across all 14 spheres with better scan-time efficiency than the fixed conventional baseline.


Revised challenges, achievements and chapter structure:

To pick up from your earlier email, here is my current thinking on C1–C3 / A1–A3:

Challenges:

  C1 — Adaptive sequence design: there is no single optimal MR pulse sequence across patients/phantoms; the optimal sequence depends on the unknown tissue properties. How can an agent learn to adaptively choose sequences to efficiently estimate quantitative MRI parameters?

  C2 — Scalable simulation-in-the-loop RL: running a physics simulator at every RL step is expensive. How can we design the RL environment so training is feasible while keeping fidelity high enough for results to be meaningful?

  C3 — Spatial localisation under pose uncertainty: the phantom/patient's exact position in the scanner is unknown. How can the agent localise the object of interest and map its tissue properties jointly?

Achievements:

  A1 (Chapter 3): QalibreMD digital twin + conventional baseline — establishes the simulator, validates the twin, and sets the quantitative yardstick the RL agent must beat.

  A2 (Chapter 4): RL agent for adaptive T1/T2 mapping — from single-voxel (E1) to full-phantom 2D imaging (E2), with domain randomisation and noise. Directly addresses C1 and C2.

  A3 (Chapter 5): Spatial localisation and generalisation — phantom pose randomisation, scout acquisitions, and robustness evaluation across noise levels and orientations. Addresses C3 and provides the quantified evaluation.


Timetable (12 JUNE):

  Week 4   28 Apr – 4 May    E2: full-phantom RL with gradients, 2D imaging, noise, localisation seed
  Week 5   5–11 May          E2 evaluation: Pareto curves (accuracy vs scan time), noise robustness sweep
  Week 6   12–18 May         E3: MRF-style fingerprinting with learned FA/TR schedules
  Week 7   19–25 May         E3 evaluation + model-based reconstruction pilot
  Week 8   26 May – 1 Jun    Final experiments / stretch goals; begin writing all chapter drafts
  Week 9   2–8 Jun           Full write-up (all research complete by 1 June as discussed with Andreas)
  Week 10  9–12 Jun          Final polish and submission

I'm aiming to have bullet-point drafts of all six chapters completed by 1 June, in line with your recommendation.

Please let me know if you'd like me to send a more detailed breakdown of any of the above, or if you have suggestions on the chapter structure.

Many thanks,
Arthur

# PLAN_E2E_RL.md — Chapter 5: End-to-End Differentiable Adaptive MRI

**Status:** draft v0.1, 2026-05-10. Replaces E3 (MRF fingerprinting) in the project timetable.
**Owner:** Arthur. Reviewed-by: pending Andreas M3 meeting (2026-05-11).
**Time-box:** Section 6 verification gate by 2026-05-17 (one week). If the gate fails, fall back to the Section 12 plan B.

---

## 0. One-paragraph summary

Move from "tune (TI, TR, α) on a fixed IR-SE-2D template" (Ch4) to "construct an MRI pulse sequence block from primitive RF/gradient/ADC events end-to-end, and use it to jointly estimate T1, T2, and PD maps of an xy slice of the QalibreMD phantom." The proposed substrate is MRzero (a Phase Distribution Graph-based **differentiable** Bloch simulator), which exposes the full sequence-design degrees of freedom. The agent's job is twofold: design the sequence (action sequence over primitive events) and estimate the (T1, T2, PD) maps from the resulting images. Two estimation paradigms are on the table: (a) policy-gradient RL with a learned per-pixel parameter head, (b) gradient-based meta-optimisation of the sequence directly (no RL, exploiting differentiability). This plan deliberately defers the choice between (a) and (b) until after the Section 6 verification gate, because the right answer depends on whether the Ch4 SSE-landscape failure mode reappears under the new estimation setup.

**This plan is risky.** Section 9 lists the failure modes, ranked. The risk-mitigation strategy is to front-load three cheap verification experiments (Section 6) and only commit to the full end-to-end pipeline if they pass. The total budget is three weeks of research time + one week of writing.

---

## 1. Why this, and why now

### 1.1 Lessons from Ch4 that constrain Ch5's design

Ch4 produced a set of negative findings that any Ch5 plan must respect:

1. **The likelihood matters more than the policy.** The §20 SSE-landscape diagnostic on V12's failing spheres showed that the magnitude-reconstruction + 5% Gaussian noise likelihood has the true T1 at the 76th percentile of the SSE landscape on the median failing sphere. No fitter that scores candidates by SSE can recover truth in this regime. **Action: in Ch5, characterise the likelihood landscape before training anything.** This is the single biggest mistake from this week — it should have been done before V9.
2. **Fitter-side effects can fully mask schedule-side effects.** V12's 99 pp baseline-fitter win over CR-opt evaporated under oracle-init. **Action: any Ch5 claim of "the agent learns X" requires both a Cramér–Rao-style theoretical anchor and an oracle-init eval to be credible.**
3. **Adaptive within-episode behaviour is not the same as more information.** V12 was measurably adaptive (Pearson r = −0.26, KS-significant) but information-equivalent to a fixed schedule. **Action: distinguish in advance between "behavioural" and "informational" adaptivity claims and report both.**
4. **The agent will find any bias the fitter has and exploit it.** E2.3's "floor exploit" (TI=10 ms spam) and V4's reward-shape collapse are both instances of this. **Action: validate the estimator under random, fixed, and learned policies separately before training.**
5. **Don't switch simulator stacks without a feasibility spike first.** Mid-project. We do not have the runway to debug a new simulator + new estimator + new RL setup all at once. Section 6 is the spike.
6. **Time pressure is real.** 1 June bullet-point draft, 12 June submission. Ch5 has at most three weeks of research time. The plan must be feasible in that window or it must have a documented fallback.

### 1.2 Why end-to-end is interesting (the positive case)

The Ch4 chapter ends on a fitter-side bottleneck. The four ways to move that bottleneck (Section 4.10.5) are: phase-sensitive reconstruction, joint multi-sphere fit, Bayesian prior, or change the action space so that the data is fundamentally more identifying. End-to-end RL with primitive sequence-design actions plus joint (T1, T2, PD) estimation hits all four:

- **Phase-sensitive recon comes for free** if the sequence is designed in primitive events (the signal model is naturally signed; abs() is a downstream choice).
- **Joint estimation of T1 + T2 + PD per pixel** adds 3× more constraints per sphere than T1-alone. The wrong-basin SSE-ratios in Ch4 §20 are computed *given* a (T1, A) profile; under a (T1, T2, PD) profile the joint likelihood lives in a higher-dimensional space and the wrong basins may not be coherent across all three parameters simultaneously.
- **A learned slice-selective excitation and per-pixel fit** sidesteps the "one TI per block shared across all spheres" limitation of Ch4 (Section 4.11). Different spheres can be queried differently *within* one block.
- **The Ernst-angle DoF** (flagged by Andreas in M2 feedback, formalised in Ch4 §4.11) is automatically in the action space. If the agent emerges with $\cos\alpha^* \approx \exp(-\text{TR}/T_1)$ at convergence, that is a clean Ch5 "RL discovered a textbook result" figure.

### 1.3 Why differentiable simulation is the right substrate (probably)

MRzero (Loktyushin et al., 2021, *Magn. Reson. Med.*; codebase at `MRzero-Core`) implements a **Phase Distribution Graph (PDG)** simulator that is fully differentiable in PyTorch. (Note: the interim report incorrectly called it an EPG simulator — PDG is a strict generalisation and is what MRzero actually uses. Fix everywhere before final submission, per CLAUDE.md.)

PDG vs Bloch (what KomaMRI does today):

- **Bloch:** per-spin ODE integration. Exact for arbitrary tissue + sequence, slow, not differentiable in PyTorch.
- **PDG:** tracks a tree of magnetisation states (each node = a coherence pathway), each with phase, amplitude, T1, T2 dependencies as analytic expressions. Differentiable w.r.t. sequence parameters and tissue parameters. Faster than per-spin Bloch for typical sequences.

Differentiability lets us:

- Compute gradients of any loss (MAPE, Fisher information determinant, sequence time, SAR proxy) with respect to sequence parameters directly.
- Train the estimator network with backprop through the simulator.
- Optionally skip RL entirely and use gradient-based bilevel optimisation of the sequence (Section 5.3).

The hazard: PDG assumes a small number of coherence pathways. If the sequence drives a dense pathway tree (many refocusing pulses with varying flip angles), simulation cost grows quickly. Most published MRzero work is on relatively short sequence templates (~10–30 RF events). We will verify Section 6.2 that the QalibreMD phantom under our target sequence length is tractable before committing.

---

## 2. Concrete goal for Ch5

In one sentence: **train an end-to-end agent that, given the QalibreMD phantom, designs a pulse sequence from primitive events and produces (T1, T2, PD) maps of an xy slice — and demonstrates a quantified per-pixel benefit over (a) a CR-optimal fixed multi-parameter sequence, (b) a published multi-parameter mapping protocol (DESPOT1/DESPOT2 or a published MRF schedule), and (c) the Ch4 IR-SE-2D V12 policy retrained to estimate T2 and PD alongside T1.**

The headline metric is **per-pixel mean MAPE over all three parameters** at a fixed scan-time budget:

$$
\text{MAPE} = \frac{1}{3} \left( \overline{|\hat T_1 - T_1| / T_1} + \overline{|\hat T_2 - T_2| / T_2} + \overline{|\hat{\text{PD}} - \text{PD}|/\text{PD}} \right)
$$

averaged over the slice's voxels. Reporting also includes per-parameter MAPE and per-tissue MAPE so the chapter can pick apart where the win lives.

---

## 3. The C3 challenge that Ch5 addresses

The original C3 was "spatial localisation under pose uncertainty". Ch5 narrows this to "joint multi-parameter mapping with end-to-end learned sequence design". The pose-uncertainty thread is demoted to a robustness sub-experiment (Section 11.5) since the Ch4 work already establishes phantom-pose randomisation as a working invariance.

**Re-stated C3 for Ch5:** *No single MRI sequence template optimises jointly across (T1, T2, PD) for arbitrary tissue distributions. Can an agent learn — from scratch, with no template — to construct a sequence that produces identifying data for all three parameters simultaneously, while respecting hardware and time constraints?*

**Novelty vs prior work:**

- MRzero (Loktyushin et al.) optimises *fixed* sequence parameters end-to-end through PDG. Not adaptive: one sequence for all phantoms.
- AUTOSEQ (Zhu et al.) RL-optimises flip-angle and TR schedules but on isolated voxels and a CR-bound reward, not image-domain.
- MRF (Ma et al.) uses fixed pseudorandom schedules with dictionary matching. No learned sequence design, no adaptivity.
- Our combination — end-to-end RL or differentiable bilevel optimisation **with adaptive within-episode sequence construction and joint parameter mapping** — is, as far as we have found, unpublished.

The credibility of the novelty claim depends on Section 7's literature review being thorough. **Do not write the novelty claim into the report until Section 7 is complete.**

---

## 4. Architecture proposal — two paradigms, decided at the gate

Two ways to do "end-to-end" are on the table. Pre-commit to the gate experiment (Section 6) deciding between them, not to one in advance.

### 4.1 Paradigm A — PPO with primitive-action sequence design

State $s_t$:

- Slice-domain image stack from blocks so far (downsampled to e.g. 32×32×n_blocks)
- Per-pixel running (T1, T2, PD) estimates from a learned read-out network
- Per-pixel σ estimates
- Time-budget remaining, block index

Action $a_t \in \mathbb R^D$ — primitive sequence-event tokens:

- RF event: $(\alpha, \phi, \text{duration})$
- Gradient event: $(G_x, G_y, G_z, \text{duration})$
- ADC event: $(N_{\text{samples}}, \text{duration})$
- Wait: $(\text{duration})$

Each "block" is a learned-length sequence of event tokens (a small autoregressive policy outputs the next event until an "end-of-block" token). Then PDG simulates the block, the env reconstructs the image, the read-out network updates (T1, T2, PD) per pixel, reward is computed.

Reward: $-$ per-pixel mean MAPE + sparse incentives (Section 8.2).

Advantages: closest to "RL designs sequences", honours the user's "end-to-end RL" framing.
Disadvantages: enormous action space, very hard exploration, autoregressive policy is fragile, training likely unstable.

### 4.2 Paradigm B — Differentiable bilevel optimisation (RL outer, gradient inner)

Outer loop: a small RL or evolutionary outer optimiser proposes high-level sequence parameters (e.g. "FISP-like with 200 readouts, learned flip-angle schedule").
Inner loop: PyTorch autodiff through PDG gives the gradient of the per-pixel parameter MAPE w.r.t. the schedule directly. Use Adam over the schedule for a fixed phantom-distribution.
Estimator: a learned U-Net or per-pixel MLP that maps image stack → (T1, T2, PD) maps. Trained jointly with the schedule via the same backprop.

Advantages: vastly fewer effective degrees of freedom; gradient signal is dense; literature precedent (MRzero, AUTOSEQ).
Disadvantages: not strictly "RL" in the policy-gradient sense — closer to meta-learning / hyperparameter optimisation. May not satisfy the C1 narrative if the supervisor expects an adaptive, observation-conditional policy.

### 4.3 Paradigm C — Hybrid (recommended pre-gate hypothesis)

Outer: small PPO that picks one of K pre-trained "sequence skills" (each a Paradigm B schedule optimised for a different tissue regime) and decides when to switch.
Inner: each skill is a differentiable schedule, learned offline.
Estimator: shared learned read-out network.

Advantages: keeps RL in the picture for C1 narrative, but avoids the giant-action-space exploration problem of Paradigm A. Adaptivity is "pick the right schedule for this episode" rather than "construct events from scratch per block".
Disadvantages: more moving parts; the "from scratch" framing is weaker.

**Decision rule (set at the gate, Section 6):** Paradigm A only if the Section 6.2 PDG-simulation cost stays under 50 ms per block on the test phantom *and* Section 6.3 shows that the per-pixel estimator can recover (T1, T2, PD) from a known sequence to <30% MAPE. Otherwise Paradigm B or C.

---

## 5. Joint (T1, T2, PD) estimation

The estimator is the single most important component, given the Ch4 lesson. Two options, evaluated in parallel during the gate:

### 5.1 Per-pixel MAP estimator with learned prior

For each pixel $p$, given image stack $\mathbf m_p$ and sequence $\sigma$:

$$
(\hat T_1, \hat T_2, \hat{\text{PD}})_p = \arg\max_{T_1, T_2, \text{PD}} \log p(\mathbf m_p \mid T_1, T_2, \text{PD}, \sigma) + \log p_{\text{prior}}(T_1, T_2, \text{PD})
$$

Likelihood: phase-sensitive (signed) PDG forward model + Gaussian noise. Prior: a flow-based or mixture-Gaussian density trained on the QalibreMD phantom's known (T1, T2, PD) distribution.

Justification: the Ch4 §20 SSE-landscape problem is *exactly* the case where a well-calibrated prior collapses the wrong basin's posterior mass. Bayesian estimation is the recommended fitter swap from Ch4 §20.6.

### 5.2 Learned regression network (U-Net or per-pixel MLP)

Image stack → (T1, T2, PD) maps via a CNN, trained on simulated data with random tissue distributions and a chosen sequence. Used at inference without further optimisation.

Justification: avoids the SSE landscape entirely (the network learns the inverse map directly). Risk: out-of-distribution behaviour on tissue values outside the training distribution. The QalibreMD phantom has known T1/T2/PD values, so in-distribution coverage is trivial to verify, but real-tissue generalisation is not free.

### 5.3 Decision rule

Train both on a fixed Paradigm-B schedule during the gate (Section 6.4). Whichever achieves <30% per-pixel MAPE *and* shows no wrong-basin behaviour under the SSE-style diagnostic adapted to (T1, T2, PD) space wins.

If both fail: Ch5 reverts to the fallback plan (Section 12) — stay on KomaMRI, add T2 mapping to the Ch4 setup, do not attempt end-to-end.

---

## 6. Verification gate — three experiments, one week

Before committing to Paradigms A/B/C and to either estimator, run three cheap experiments. All three must pass before the rest of the plan is executed. **Hard deadline: 2026-05-17.** Status updates daily on `M3_gate_log.md`.

### 6.1 Gate-1 — MRzero installs, runs, agrees with KomaMRI on a reference sequence

Install MRzero in a separate Python venv (do not pollute the existing one until validated). Simulate the existing E0 IR-TSE conventional baseline sequence on the QalibreMD phantom using both KomaMRI and MRzero. Compare image-domain outputs.

Pass condition: per-pixel relative difference < 5% over the spheres. If not, MRzero is mis-configured or PDG doesn't represent the sequence faithfully. Diagnosis needed before continuing.

**Effort:** half a day. **Risk if fails:** medium — likely an MRzero config issue, fixable. If unfixable in 1 day, abandon MRzero and use KomaMRI with finite-difference gradients (slower, less elegant, but a working fallback).

### 6.2 Gate-2 — PDG simulation cost on the target sequence length

Time a single PDG simulation of a 200-event sequence (typical MRF / multi-parameter mapping length) on the full QalibreMD phantom. Check whether the PyTorch backprop through it stays under 1 GB GPU memory.

Pass condition: forward + backward < 200 ms per simulation; memory < 4 GB. If not, the differentiable approach is infeasible at the target sequence length — fall back to Paradigm C with shorter skills, or Paradigm A with finite-difference gradients.

**Effort:** half a day. **Risk if fails:** medium-high — PDG pathway explosion is a known failure mode for long sequences. The mitigation (shorter sequences, pruning low-amplitude pathways) is documented in the MRzero paper.

### 6.3 Gate-3 — SSE landscape diagnostic on the proposed estimator

**This is the most important gate.** It directly addresses the Ch4 §20 lesson.

For a fixed reference sequence (the CR-optimal schedule generalised to joint (T1, T2, PD) — see Section 7.1), simulate noisy images of the phantom under MRzero, then plot the joint likelihood landscape over (T1, T2, PD) for each of a representative set of voxels (long-T1+long-T2, short-T1+long-T2, short-T1+short-T2, long-T1+short-T2 — the four corners).

Pass condition: truth is at rank ≤ 50 / 8000 (top 0.6%) in the joint likelihood landscape on at least 80% of test voxels. (This is the analogue of Ch4 §20's "76th percentile" statistic — translated to a 3D parameter space with 20³ = 8000 grid points.)

If truth is *not* at top-0.6% rank on most voxels, the multimodal-SSE problem from Ch4 reappears in the (T1, T2, PD) likelihood. In that case, no estimator-side gain over Ch4 is possible without phase-sensitive recon AND a calibrated prior, both of which need separate validation.

**Effort:** one day. **Risk if fails:** high — would force a return to either KomaMRI + Bayesian prior (Section 12) or to a fundamentally different parameter-mapping setup (e.g. fingerprinting with a learned matched filter, not an MLE).

### 6.4 Gate-4 (parallel) — estimator feasibility

Use the same fixed reference sequence and synthetic phantom data. Train both candidate estimators (Section 5.1 MAP, Section 5.2 learned regressor) to convergence on simulated data. Report:

- Per-pixel mean MAPE on held-out simulated phantoms
- Per-pixel σ-calibration (median σ/|err|) — same diagnostic as Ch4 §2.2
- Performance on out-of-distribution tissue values (real ICR phantom samples if Andreas can supply; otherwise synthetic perturbations of the training distribution)

Pass condition: at least one estimator reaches <30% per-pixel mean MAPE with median σ/|err| in [0.5, 2.0]. If neither does, the end-to-end paradigm is held back by the estimator regardless of the sequence — Ch4's bottleneck recurs.

**Effort:** two days. **Risk if fails:** high — analogous to a Ch4 redo. The fallback is Paradigm C with the Ch4 LM fitter on T1 alone.

### 6.5 Gate decision matrix

|  | 6.1 | 6.2 | 6.3 | 6.4 | recommended paradigm |
|---|:---:|:---:|:---:|:---:|---|
| all pass | ✓ | ✓ | ✓ | ✓ | Paradigm A or B; choose by Section 8.5 |
| 6.2 fails | ✓ | ✗ | — | — | Paradigm C with KomaMRI, finite-diff gradients |
| 6.3 fails | ✓ | ✓ | ✗ | — | fallback to Section 12: T2 + PD added to Ch4 KomaMRI setup, no end-to-end |
| 6.4 fails | ✓ | ✓ | ✓ | ✗ | Bayesian-prior fitter on Ch4 setup; reframe Ch5 as fitter-side novelty |
| 6.1 fails | ✗ | — | — | — | KomaMRI + finite-diff; demote PDG to "future work" |

---

## 7. Sequence design — what the agent's action space actually contains

### 7.1 The (T1, T2, PD) Cramér–Rao anchor

Before the agent does anything, solve the analogue of Ch4 §6 in joint (T1, T2, PD) space:

$$
\text{Var}\left[\hat\theta_p\right] \succeq \left[J^\top J\right]^{-1}, \quad \theta_p = (T_1, T_2, \text{PD})_p
$$

at each phantom voxel, with $J$ the Jacobian of the PDG forward model w.r.t. $\theta_p$. Minimise the fleet-weighted trace of the inverse Fisher information matrix over the schedule. This gives a CR-optimal **fixed multi-parameter sequence** — the strongest non-adaptive baseline.

**Effort:** ~2 days after the gate. Reuses much of `src/baselines/cr_optimal.jl`.

### 7.2 Published baselines to include

Pick two from the multi-parameter mapping literature:

- **DESPOT1/DESPOT2** (Deoni et al.) — closed-form SPGR + SSFP, three flip angles each. Industry-standard.
- **Fixed MRF schedule** (Ma et al., 2013) — 1000-readout pseudorandom FA/TR. The MRF community's de facto baseline.

These need to be implementable in MRzero/PDG (Gate 6.1 verifies the forward model handles them).

### 7.3 The agent's action space

Three concentric options of increasing freedom, decided at the gate:

- **Tight (Paradigm C):** pick one of $K=4$ pre-trained sequence skills, plus a "stop here / continue" flag.
- **Medium (Paradigm B + outer RL):** schedule the next $(\alpha, \phi, \text{TR}_{\text{block}})$ triplet, with fixed block structure (RF → wait → readout → spoil).
- **Wide (Paradigm A):** primitive RF/gradient/ADC tokens as in Section 4.1.

Default to **Medium** unless the gate clearly clears Wide. The Wide option is what the user asked for, but it has the highest training-instability risk and the smallest literature precedent.

---

## 8. Training plan, assuming the gate clears

### 8.1 Phantom and fleet

QalibreMD T1-array + T2-array spheres plus PD spheres in the same xy slice. ~30 spheres per slice. Episode = single phantom realisation with random pose, T1/T2/PD jitter (±5% on each), and slice position. Time budget per episode: 250 s (matched to Ch4 tractability).

### 8.2 Reward (RL paradigm)

$$
r_t = -\bar{\text{MAPE}}_t + \lambda_{\Delta} \cdot (\bar{\text{MAPE}}_{t-1} - \bar{\text{MAPE}}_t) - \lambda_{\text{SAR}} \cdot \text{SAR}_t - \lambda_{\text{time}} \cdot \Delta t
$$

with $\bar{\text{MAPE}}$ the per-pixel mean over all three parameters; $\lambda_\Delta = 1.0$ (from Ch4 ablation); $\lambda_{\text{SAR}}, \lambda_{\text{time}}$ small. **Critical:** unlike Ch4, the SAR term forces an absolute physical constraint, not just a budget — this prevents the agent from spamming high-flip-angle pulses for SNR. SAR proxy is $\sum_k \alpha_k^2 / \text{TR}_k$, normalised to unit reasonable.

### 8.3 Action and policy

For Paradigm B (the default): autoregressive policy over $(\alpha, \phi, \text{TR}_{\text{block}})$ with continuous heads. PPO with 200k–500k steps. Discrete "stop block" token for variable-length sequences.

For Paradigm A (if Wide clears the gate): a small Transformer-style policy emitting event tokens. Vocabulary size ≈ 20 (event types × duration buckets). Maximum sequence length 100 events per block.

### 8.4 Estimator training

Both the sequence policy and the estimator are trained on the same simulated data. For Paradigm B, the estimator is trained offline first on a distribution of random fixed sequences (~24 hours, one-off), then frozen during RL training. For Paradigm B end-to-end (no RL outer), the estimator is co-trained via backprop through the PDG simulator.

### 8.5 Training cost budget

| stage | wall time | gate |
|---|---:|---|
| Gate 6.1–6.4 | 4 days | week 1 |
| CR-optimal multi-param baseline | 2 days | week 1 |
| Estimator offline training (Paradigm B) | 1 day overnight | week 2 |
| RL training run 1 (Paradigm B Medium) | ~8 h compute | week 2 |
| RL evaluation + diagnostics | 1 day | week 2 |
| RL training run 2 (Paradigm A Wide, if gate clears) | ~12 h compute | week 3 |
| Comparison, ablation, write-up | 3–4 days | week 3 |

Total research time: 3 weeks. Total compute: ~30 GPU-hours. Both within budget.

---

## 9. Risk assessment — be brutal

Ranked by likelihood × impact.

### 9.1 The SSE-landscape problem recurs in joint space (high probability, high impact)

If the magnitude-recon multimodality from Ch4 §20 carries over into the joint (T1, T2, PD) likelihood, no estimator-side fix is automatic. Mitigation: Gate 6.3 is exactly this test. If it fails, fall back to Section 12. **The probability is genuinely 30–50% given the Ch4 finding;** the joint likelihood has more constraints but also more wrong-basin opportunities.

### 9.2 MRzero / PDG can't simulate the QalibreMD phantom at usable speed (medium-high probability, medium impact)

PDG pathway explosion on long sequences. Mitigation: Gate 6.2 directly tests this. Fallback: KomaMRI + finite-difference gradients (10× slower per gradient eval but still tractable for Paradigm C). The probability is 20–30% — most MRzero papers use shorter sequences than ours.

### 9.3 Training instability under the Wide action space (high probability if attempted, medium impact)

Primitive-event action spaces are notoriously hard to RL. Even MuZero-style agents on game-tree problems with similar action-space cardinality require massive compute. Mitigation: default to Medium; only attempt Wide if Gate 6.3 clears decisively and there is week-3 time. Probability of Wide-paradigm failure: 60–80%; impact contained because Medium is a strict subset.

### 9.4 Estimator OOD generalisation collapses (medium probability, high impact)

The learned regressor (Section 5.2) trains on synthetic data; real phantom values are likely in-distribution but tissue variations are not. Mitigation: Section 6.4 includes a synthetic OOD test; Section 11.5 sweeps tissue-distribution perturbations. Probability: 20–40%; impact contained if the MAP estimator (Section 5.1) is a working fallback.

### 9.5 Time runs out before Ch5 has a publishable result (medium probability, very high impact)

Three weeks is tight for new simulator + new estimator + new RL setup. The gate (Section 6) and the fallback plan (Section 12) are designed to bound this risk. **Hard rule: if Gate 6 is not clear by 2026-05-17, switch to plan B immediately.** No "just a few more days".

### 9.6 The "novelty" claim doesn't survive literature review (low-medium probability, high impact)

If MRzero+adaptive+joint-mapping has been published in the last 12 months (we did the survey before March), Ch5 loses its novel-aspects bullet. Mitigation: Section 7 literature review must complete before week 2 training starts. If novelty is gone, reframe Ch5 around the CR-optimal-multi-param baseline + estimator-comparison angle — both still defensible.

### 9.7 Ernst-angle DoF doesn't emerge cleanly (low probability, low-medium impact)

The user wants the agent to discover Ernst-angle behaviour. If it doesn't emerge clearly in the trained policy, the "RL found a textbook result" figure is weaker. Mitigation: include $\alpha_{\text{exc}}$ as a learned action; report empirical $\cos\alpha$ vs $\exp(-\text{TR}/T_1)$ regardless of agreement; if no emergence, frame the negative result honestly (as Ch4 does for adaptivity-vs-information). Probability of clean emergence: 30–50%. Probability of *any* interpretable flip-angle behaviour: 80%.

### 9.8 The differentiable bilevel approach is no longer "RL" in a sense Andreas/Wayne accept (low probability, medium impact)

Paradigm B is closer to meta-learning than to RL. Wayne's email feedback emphasised the C1–C3 / A1–A3 structure but did not pin down RL specifically. Mitigation: keep an RL outer loop in Paradigm B (skill-selection or step-budget allocation) so the chapter can credibly say "RL outer, gradient inner". Probability of issue: low; covered by the Section 11.4 contingency.

---

## 10. Decision gates — when to stop and re-plan

- **2026-05-13 (Wed):** Gate 6.1 done. If MRzero is not running on the test phantom, switch to finite-diff KomaMRI track.
- **2026-05-14 (Thu):** Gate 6.2 done. If PDG cost is unworkable on a 200-event sequence, drop to Paradigm C scope.
- **2026-05-16 (Sat):** Gate 6.3 done. If SSE landscape is bad in joint space, **stop and switch to Section 12 plan B**. Do not start RL training.
- **2026-05-17 (Sun):** Gate 6.4 done + paradigm decided. Wayne gets a milestone update.
- **2026-05-24 (Sun):** Training run 1 evaluated. If MAPE > 50% above CR-multi-param baseline, do not attempt Paradigm A — write up Paradigm B as final.
- **2026-05-31 (Sun):** All experiments complete. Begin Ch5 prose draft.
- **2026-06-01:** Wayne deadline — bullet-point draft of all chapters.

---

## 11. Quantified benefit and report structure

Ch5 structure mirrors Ch4 for consistency:

- 5.1 Challenges + novelty
- 5.2 Differentiable Bloch simulation (PDG / MRzero primer)
- 5.3 Joint (T1, T2, PD) Cramér–Rao optimal anchor
- 5.4 Estimator design (Bayesian MAP vs learned regressor — Section 5.4 of this plan)
- 5.5 Sequence-design action space (Section 7 of this plan)
- 5.6 Training and convergence
- 5.7 Headline per-pixel MAPE table — RL vs CR-multi-param vs DESPOT vs MRF
- 5.8 Adaptivity diagnostics — does the agent's sequence vary across episode types?
- 5.9 Ernst-angle emergence figure
- 5.10 Limitations
- 5.11 Summary — C3 quantified benefit

The quantified-benefit anchor table (Section 5.7 in the eventual chapter):

| Policy | per-pixel mean MAPE | scan time | adaptive? |
|---|:---:|---:|:---:|
| DESPOT1/2 | TBD | ~120 s | no |
| MRF (fixed schedule) | TBD | ~12 s | no |
| CR-optimal joint sequence | TBD | ~120 s | no |
| Ch4 V12 + post-hoc T2/PD fit | TBD | 250 s | partial |
| **End-to-end Ch5 policy** | **TBD** | **TBD** | **yes** |

Pre-committed reading rules:

- "RL beats X" requires both raw-fitter MAPE difference > 1 σ at paired N=200 *and* oracle-init MAPE difference > 1 σ. (Ch4 §9 lesson.)
- "Adaptivity is real" requires Pearson r and KS-significant subset shift, both at paired N=200.
- "Ernst angle emerged" requires plotted $\cos\alpha$ vs $\exp(-\text{TR}/T_1)$ with $R^2 > 0.5$.

---

## 12. Plan B — what we do if the gate fails

The fallback assumes Gate 6.3 or 6.4 fails (the more likely failure modes). Total scope ~1.5 weeks.

### 12.1 Keep KomaMRI, add T2 and PD to Ch4

Extend the Ch4 env (`src/rl/e2.jl`) to:

- Add T2-array sphere materials (already in `src/materials/t2_array.jl`).
- Add PD-sphere materials.
- Extend the action space to also choose TE (controls T2 weighting).
- Extend the fitter to jointly recover (T1, T2, PD) per sphere.
- Reuse the CR-optimal solver to compute the multi-parameter analytic anchor.

The chapter becomes: "Ch5 extends the Ch4 RL framework to joint T1/T2/PD mapping on the same KomaMRI substrate, with the Bayesian-prior estimator from Ch4 §20.6." Less ambitious but a real result with a known toolchain.

### 12.2 Why this is still a credible C3 contribution

It demonstrates joint multi-parameter mapping (a real C3 question), uses the Bayesian-prior fitter (Ch4's recommended fitter swap), and gives the CR-optimal multi-parameter anchor as a baseline. The novelty bullet is "joint Bayesian-prior estimation of T1/T2/PD with adaptive sequence design over (TI, TE, TR, α)". Smaller scope, lower risk, still defensible at FYP level.

### 12.3 If even plan B is too risky after the gate

Final fallback: write Ch5 as a *methodological* chapter — full statistical analysis of the Ch4 results, with the Bayesian-prior fitter applied to V12's existing rollouts as the headline contribution. No new training, no new simulator. Effort: 1 week. This is the "minimum viable Ch5" and should not be the goal, but is the safety net.

---

## 13. What this plan deliberately does NOT do

- **No multi-slice / 3D imaging.** Out of scope; xy slice only.
- **No B0/B1 inhomogeneity handling beyond Ch4's 5 Hz B0 noise.** PDG handles it but adding it expands the test surface.
- **No real-scanner sim-to-real comparison.** The QalibreMD phantom values are manufacturer-specs; no real-data calibration in scope.
- **No multi-tissue compartment modelling.** Single (T1, T2, PD) per voxel.
- **No SAR-realistic constraints.** Use the proxy in Section 8.2 only; do not claim clinical realism.
- **No comparison to the Ch4 phase-sensitive recon attempt (V10/V11).** Those are documented as failures in Ch4 §15; we will not revisit them in Ch5 unless Section 6 specifically motivates it.

These are not in scope; do not write them into the Ch5 contributions list.

---

## 14. Open questions for Andreas (M3, 2026-05-11)

1. Is MRzero/PDG familiar enough that Andreas can review the Gate 6.1 sanity-check, or do we need an external sanity check?
2. What multi-parameter baselines (DESPOT/MRF/other) does the ICR lab consider canonical? This determines Section 7.2.
3. Is the time-budget (250 s episode) appropriate for joint T1+T2+PD mapping, or should it be increased (this would change CR-optimal directly)?
4. If Gate 6.3 fails — i.e. the joint likelihood landscape is also multimodal — is the Bayesian-prior fallback (Section 12.1) acceptable as Ch5 scope, or does Ch5 need a different "novel" angle?
5. Sanity check on the "Ernst-angle emergence" figure idea — is "trained agent's flip-angle behaviour matches Ernst-angle prediction" a useful headline result, or is it expected enough that it would not impress an examiner?

---

## 15. Critical self-assessment

To be explicit, even where it makes the plan less appealing:

- **The probability that Ch5 produces a clean headline win is ~40%**, not 80%. The Ch4 fitter-side bottleneck might just move to a higher-dimensional likelihood.
- **The probability that Ch5 produces a defensible chapter** (positive or negative result, with the right controls) **is ~85%.** That gap is what the gate (Section 6) and the fallback (Section 12) buy.
- **The user's stated preference for end-to-end RL with from-scratch sequence design is the highest-risk option** in this plan. Honouring it fully means Paradigm A, which has a 60–80% probability of failing to converge in the available time. Paradigm B (the recommended default) is closer to "differentiable meta-optimisation" than to RL in the policy-gradient sense; this might be a mismatch with the user's mental model and Wayne's expectations.
- **The gate-clearing experiments (Section 6) are the load-bearing risk-management tool.** They consume one week of three. If the gate is skipped or rushed, the rest of the plan inherits the Ch4 failure mode of "train first, diagnose later" — which this entire plan was written to avoid.
- **The Ernst-angle DoF restoration alone is a low-effort, near-guaranteed Ch5 contribution** (it's the addition Andreas already flagged). Even if the rest of the plan fails entirely, adding flip-angle as a learned action to the Ch4 setup, retraining V13, and reporting whether Ernst-angle behaviour emerges — that's a one-week chapter on its own. This should be the **week-1 backbone** of the plan regardless of which Paradigm clears the gate.

**Bottom line.** This plan is ambitious. It buys ambition by front-loading the failure modes into a one-week verification gate. If the gate clears, Ch5 is genuinely novel. If it fails, the fallback (Section 12) still produces a defensible chapter. The single biggest risk is time pressure — there is no margin for surprises beyond what the gate already buys. Read the gate decisions as binding, not advisory.

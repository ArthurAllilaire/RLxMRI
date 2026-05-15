# PARADIGM_A_PLAN.md — Schedule-conditioned regressor for Paradigm A

**Status:** draft v0.1, 2026-05-11. Sub-plan of `PLAN_E2E_RL.md` §4.1 and §5.
**Scope:** the estimator that turns (event timeline, MRzero signal buffer) → (T1, T2, PD, σ). Replaces the `_update_estimate` placeholder in `python/qalibremd_gym/env_paradigm_a.py` (the IR-SE closed-form `least_squares` against fake `(TI = 0.1·k, TE = 0.05)` pairs).

---

## 0. One-paragraph summary

The current Paradigm A env uses a hand-coded forward model (`closed_form_irse`) with fabricated timings to fit `(T1, T2, PD)` from the signal buffer. This makes the estimator decoupled from what the agent actually did — different schedules can produce identical fits, and the agent has no gradient through the estimator to "design more informative sequences". We replace it with a **schedule-conditioned learned regressor**: a neural network that consumes the agent's full event timeline plus the resulting MRzero signal samples as a single token sequence, and emits per-parameter means and log-σ. The network is pretrained offline on ~500k (random-schedule, voxel) pairs simulated through MRzero, validated by a rank-of-truth diagnostic before any RL run, then frozen for the first round of PPO. This is structurally the deep-learning MRF reconstruction pipeline (Cohen 2018, Fang 2017, Loktyushin 2021), extended to handle a per-episode-varying schedule emitted by an RL policy.

---

## 1. Input representation — tokenise the episode

Every event the agent emits, plus every ADC sample MRzero returns, is one token. The token carries both *what happened* and *what came out*, so the network never has to guess which signal sample belongs to which RF pulse.

### 1.1 Token schema

Each token is a fixed-width vector $\tau_i \in \mathbb{R}^{d_{\text{tok}}}$ with $d_{\text{tok}} \approx 14$:

| field | dim | meaning |
|---|:---:|---|
| `evtype_onehot` | 5 | RF / GRAD / ADC / WAIT / END_BLOCK |
| `param_value_normed` | 1 | event parameter (α in rad, G in T/m, dur in s, …), rescaled per type to ~unit variance |
| `time_since_last_RF` | 1 | seconds, log1p-normed (proxy for TI) |
| `time_since_block_start` | 1 | seconds, log1p-normed |
| `time_since_episode_start` | 1 | seconds, log1p-normed |
| `block_idx_norm` | 1 | `block_idx / max_blocks` |
| `signal_real` | 1 | if `evtype == ADC`, else 0 |
| `signal_imag` | 1 | if `evtype == ADC`, else 0 |
| `signal_mag` | 1 | redundant but stabilises early training |
| `is_adc` | 1 | mask flag |

Key design choices:

- **Phase is preserved** by emitting `signal_real` + `signal_imag` separately. This makes the input phase-sensitive — exactly the recommendation from Ch4 §20.6 / `PLAN_E2E_RL.md` §1.2.
- **Timing is encoded relative to the last RF**, not absolutely. This is the contrast mechanism that lets the network behave like a generalised IR/SE fitter without being told the sequence is IR or SE.
- **No fixed-grid assumption.** Signals arrive at whatever times the agent's `ADC` events fired. The token's `time_since_last_RF` field is the *real* TI.

### 1.2 Sequence length

`max_events = 100` per episode. Pad to 100 with a zero token + mask. This is small enough that a Transformer is cheap.

---

## 2. Network architecture

### 2.1 Backbone — small Transformer encoder

```
tokens  (B, 100, 14)
  → Linear(14 → 128)           token embedding
  → + sinusoidal position(100, 128)
  → TransformerEncoder(d=128, heads=4, layers=4, ff=256, dropout=0.1)
  → masked-mean-pool over valid tokens
  → MLP[128 → 128 → 6]         output head
```

Output 6-vector: `(μ_logT1, μ_logT2, μ_logPD, log σ_logT1, log σ_logT2, log σ_logPD)`.

Total parameters: ~400k. Forward pass: < 1 ms on CPU at batch=1.

### 2.2 Why Transformer over GRU/MLP

- **Attention naturally pairs ADC tokens with their causal RF context.** An MLP on flattened tokens would have to learn this from scratch.
- **Variable-length episodes via masking** — same network handles 5-event and 100-event episodes.
- **Permutation only of equal-time events** — positional embeddings keep the causal order intact.

A GRU is the cheap fallback if the Transformer turns out to overfit the offline dataset.

### 2.3 Output parameterisation

- Predict in log-space (parameters are positive and span 2+ decades).
- Predict $\log \sigma$ directly (numerical stability).
- Convert at inference: $\hat\theta = \exp(\mu)$, $\hat\sigma_\theta = \hat\theta \cdot \sigma_{\log\theta}$ (delta-method).

### 2.4 Loss

Gaussian NLL in log-space per parameter, mean over the three:

$$
\mathcal{L} = \tfrac{1}{3} \sum_{k \in \{T_1, T_2, \text{PD}\}} \left[ \tfrac{(\log\theta_k - \mu_k)^2}{2 \sigma_k^2} + \log \sigma_k \right]
$$

This trains $\mu$ and $\sigma$ jointly — same approach used by every deep-MRF paper with calibrated uncertainty.

---

## 3. Offline pretraining recipe

### 3.1 Dataset generation

Generator script `python/paradigm_a/gen_offline_dataset.py` (to be written):

```
for i in 1..500_000:
    θ_true = sample_from_prior(...)                  # QalibreMD prior
    schedule = sample_random_schedule(rng)            # see §3.2
    signals  = mrzero_simulate(schedule, θ_true)      # same path as env._execute_block
    signals  = add_rician_noise(signals, σ=5%)
    tokens   = tokenise(schedule, signals)            # §1.1
    write((tokens, θ_true)) to shard
```

Sharded HDF5 (~2 GB total at 500k samples).

### 3.2 Behaviour-policy distribution for schedules

The agent during RL will visit a non-uniform distribution over schedules. We need the pretraining data to **cover** that distribution — including the degenerate ones the agent might emit early in training.

Mixture (default weights):

- 50% uniform-random action sequences from the 40-token action space — broad coverage.
- 25% "structured random": forced to start with `RF`, alternating `WAIT/ADC`, plausible IR/SE-like skeletons.
- 15% degenerate: `END_BLOCK` early, blocks with no RF, very short sequences — teaches the net to output high σ on uninformative inputs.
- 10% on-policy refresh: after the first RL run, append rollouts from the trained policy to handle distribution shift.

The 10% on-policy slot is filled after Iteration 1 (see §6).

### 3.3 Training hyperparameters

| | value |
|---|---|
| Optimiser | AdamW |
| LR | 3e-4, cosine to 1e-5 |
| Batch | 256 |
| Epochs | 30 |
| Weight decay | 0.01 |
| Grad clip | 1.0 |
| Hardware | 1× GPU (overnight) |

### 3.4 Held-out validation

10% of shards held out by `θ_true` decile (so the test set covers the full prior). Track per-parameter MAPE and σ-calibration every epoch.

---

## 4. Pre-RL validation gates

**Do not run PPO until both pass.** This is the lesson from Ch4 §20 / `PLAN_E2E_RL.md` §1.1.

### 4.1 Rank-of-truth diagnostic

For each test sample, grid the network's implied likelihood over $20^3 = 8000$ points in $(T_1, T_2, \text{PD})$ space (treat $\mu, \sigma$ as a diagonal Gaussian). Compute the rank of the true parameter. **Pass condition:** truth is in the top 1% (rank ≤ 80 / 8000) on ≥ 80% of test samples.

This is the joint-space analogue of the Ch4 §20 SSE-percentile test.

### 4.2 σ-calibration

Compute median $\sigma / |\hat\theta - \theta_{\text{true}}|$ across the held-out set. **Pass condition:** in $[0.5, 2.0]$ for every parameter. Outside that band → the σ head is uninformative and reward shaping cannot rely on it.

### 4.3 Schedule-coverage stress test

Sample 1000 schedules from the IR-SE-like and degenerate slices separately. Confirm MAPE doesn't collapse on the IR-SE slice while exploding on degenerates — that would mean the network has memorised one mode.

---

## 5. Integration into `ParadigmAEnv`

Concrete edits to `python/qalibremd_gym/env_paradigm_a.py`:

1. Add `self.token_log: list[np.ndarray]` to `reset()`. Append a token in every `step()` after decoding the action.
2. After `_execute_block`, attach the returned signal samples to the **ADC tokens** of the current block (not to a separate buffer). The existing `self.signal_buf` becomes derivable from the tokens; keep it only for visualisation.
3. Replace `_update_estimate` with:
   ```python
   tokens_padded = pad(self.token_log, 100)
   mu, log_sigma = self.regressor(tokens_padded)
   self.theta_est   = np.exp(mu.detach().cpu().numpy())
   self.theta_sigma = (self.theta_est * np.exp(log_sigma).detach().cpu().numpy())
   ```
4. Load the regressor in `__init__` from a `regressor_ckpt` argument; `eval()` mode, `torch.no_grad()` for inference, frozen.
5. The 96-d signal-buffer slice of the obs is **kept for backwards compatibility** but optionally zeroed — the obs is dominated by `log θ_est`, `log σ`, and the budget/block scalars; the agent already has the relevant compression of the signal via `θ_est`.

Important: `_update_estimate` was called only on block-close. Move it to **every step** (cheap, < 1 ms) so the obs reflects the network's current best read after each token, not just each block.

---

## 6. Iteration plan with the RL loop

Iteration 1 — **frozen regressor**:

1. Pretrain on 500k random schedules (§3).
2. Pass §4 gates.
3. Run PPO with the regressor frozen. ~8 h compute.
4. Save 10k rollouts from the trained policy.

Iteration 2 — **on-policy refresh** (optional, only if Iteration 1's MAPE plateaus above the CR-optimal baseline):

5. Mix 10% on-policy rollouts into the dataset and retrain the regressor.
6. Re-validate §4.
7. Re-run PPO from the Iteration-1 checkpoint.

Iteration 3 — **co-training** (stretch, only if Iteration 2 is solid):

8. Unfreeze the regressor during PPO; add the NLL loss as an auxiliary loss with weight 0.1.
9. Watch for the E1 failure mode (agent + estimator collude into a degenerate equilibrium). If it appears, revert to frozen.

---

## 7. Previous and similar work

This section is the literature anchor for the Ch5 "novelty" claim and must complete before any prose is written into the report.

### 7.1 Magnetic Resonance Fingerprinting (MRF) — the closest classical relative

**Ma et al., 2013, *Nature*** introduced MRF: a long pseudorandom schedule of flip angles and TRs, with reconstruction by dictionary matching. Each tissue $(T_1, T_2)$ pair has a precomputed signal evolution; the observed signal is matched to its nearest dictionary atom. **The dictionary match is exactly a signal→parameter regression**, performed by a $k$-NN on a fixed schedule.

Limitations vs our setup:

- One fixed schedule chosen offline. No per-episode adaptation.
- Dictionary size grows multiplicatively in parameter dimensions; adding PD or B0 inflates storage rapidly.
- Reconstruction time is dictionary-bound (seconds to minutes per slice).

Why it matters: the *shape* of our pipeline (long schedule → signal evolution → invert to parameters) is MRF. The differences are (a) the schedule is RL-controlled per episode, and (b) the inversion is a schedule-conditioned neural network instead of a fixed dictionary.

### 7.2 Deep-learning MRF reconstruction

A line of work replaces dictionary matching with a neural network for speed and noise robustness. Key entries:

- **Cohen, Zhu, Rosen, 2018, *MRM* — "MR fingerprinting Deep RecOnstruction Network (DRONE)"**: 4-layer MLP, signal-evolution input → $(T_1, T_2)$ output. ~1000× faster than dictionary matching with comparable accuracy. **This is the canonical "learned regression head over the signal buffer" for one fixed schedule.**
- **Hoppe et al., 2017–2019**: CNN on the temporal signal evolution. Demonstrates that 1D convolutions over time pick up the same features the dictionary matching uses.
- **Fang et al., 2017–2019** (U-Net-MRF, RCA-U-Net): treats reconstruction as image-domain regression, exploits spatial context across pixels.
- **Streamlined MRF (Hamilton et al., 2021, *NeuroImage*)**: deep-learning recon enables whole-brain MRF at clinical scan times.
- **Self-attention pyramidal CNN (2022)**: denoising + attention front-end for undersampled MRF.

Common pattern: **fixed schedule, network trained for that schedule, network trained on simulated dictionary data with added noise/artifacts**. Our pretraining recipe (§3) is the natural generalisation when the schedule is no longer fixed.

### 7.3 MRzero — sequence-design via differentiable simulation

**Loktyushin et al., 2021, *MRM* — "MRzero: Automated discovery of MRI sequences using supervised learning"** (and the `MRzero-Core` codebase we already use). PDG-based differentiable Bloch simulation. Optimises (a) the schedule parameters and (b) a reconstruction operator jointly by backprop through the simulator. Demonstrated on T1 mapping, T2 mapping, and image-domain contrast.

How it differs from Paradigm A:

- The MRzero schedule **converges to a single fixed schedule** during training. At inference it does not adapt per phantom.
- The reconstruction operator is paired one-to-one with the converged schedule, not conditioned on a varying schedule.
- No RL — pure gradient descent on schedule parameters.

Paradigm A reuses MRzero's PDG simulator (`compute_graph` / `execute_graph`) but the schedule comes from an RL policy and the reconstruction must generalise across schedules.

### 7.4 AUTOSEQ and related RL-for-MRI

**Zhu et al., 2018 — AUTOSEQ**: RL over flip-angle / TR schedules with a Cramér–Rao-bound reward. Single-voxel, no image-domain estimation, no learned reconstruction (the reward is analytic from the Fisher information). Demonstrates that RL can find informative schedules in principle.

**Shin et al., 2022, *MRM***: RL for adaptive view ordering in 3D MRI — adaptive in *k*-space but the contrast (T1/T2) parameters are fixed inputs, not estimation targets.

Gap: none of the RL-for-MRI papers we have found pair the RL policy with a learned, schedule-conditioned inverse for joint $(T_1, T_2, \text{PD})$ recovery.

### 7.5 Generative forward MRF networks

**Yang et al., 2020, and follow-ups**: GAN where the generator takes $(α, \text{TR}, T_1, T_2)$ → signal evolution. This is the **forward** direction of the schedule-conditioned mapping. Inverting it (the direction we need) by amortised inference or by gradient inversion has been demonstrated for fixed schedule sets but not, to our knowledge, in an RL-adaptive loop.

### 7.6 Bayesian and MAP fitters for qMRI

**Sbrizzi et al., 2018; Bouhrara et al., 2016**: MAP estimation of relaxation parameters with informative priors. The Ch4 §20.6 fitter-swap recommendation lives here. Our regressor implicitly learns a prior (the training distribution from the QalibreMD model), so the comparison in the chapter is "learned implicit prior vs hand-specified prior", not "learned vs hand-coded forward model" — that's an interesting framing for the discussion.

### 7.7 Schedule-conditioned reconstruction — the gap

Combining the above:

- Deep-MRF gives us the signal→parameter regression for one fixed schedule.
- MRzero gives us differentiable simulation and joint schedule + recon training, but for one fixed converged schedule.
- AUTOSEQ gives us adaptive RL schedules but no learned image-domain estimator.

The Paradigm A regressor sits in the union — **schedule-conditioned signal→parameter regression where the schedule is supplied at inference time by an RL agent**. We have not found a published paper with that exact combination as of the May 2026 literature sweep. The novelty claim for Ch5 hinges on this gap surviving a thorough 2024–2026 search (see `PLAN_E2E_RL.md` §9.6) — do not write the claim into the report until that sweep is logged.

### 7.8 What the chapter should cite explicitly

Minimum citation set for the Ch5 §2 background and the Ch5 §5 estimator section:

- Ma 2013 (MRF)
- Cohen 2018 (DRONE — the canonical deep-MRF regressor)
- Hamilton 2021 (streamlined MRF — clinical relevance)
- Loktyushin 2021 (MRzero — our simulator)
- Zhu 2018 (AUTOSEQ — closest RL precedent)
- Sbrizzi 2018 (Bayesian fitter — comparison baseline mentioned in `PLAN_E2E_RL.md` §12.1)

Optional, depending on space:

- Hoppe 2017 (CNN-MRF)
- Fang 2019 (U-Net MRF)
- The 2022 self-attention MRF paper

## More prior work:
I searched the literature/web for this, and the short answer is: **yes, the estimator idea already exists in nearby forms, and the closest umbrella is absolutely MRF**—but I **did not find a primary-source paper that exactly matches** “arbitrary primitive event timeline + complex readouts → online T1/T2/PD(+uncertainty) decoder used as the state estimator inside RL for pulse-sequence design.” The pieces exist separately; your exact composition still looks less standard. ([nature.com](https://www.nature.com/articles/nature11971))

## Closest prior work I found

### 1) Classical MRF: fixed varying schedule + fingerprint inversion
The original **Magnetic Resonance Fingerprinting** paper by Ma et al. introduced exactly the high-level idea of using a deliberately varying acquisition and a pattern-recognition step to infer multiple tissue properties from the resulting signal evolution. That is the main reason your proposal feels “MRF-like.” ([nature.com](https://www.nature.com/articles/nature11971))

### 2) Learned decoders for MRF signals
This definitely exists. **DRONE** replaces dictionary matching with a neural network trained on Bloch-simulated MRF data and reports fast, accurate T1/T2 reconstruction; the abstract explicitly describes training on simulated MRF signals and regressing tissue parameters. ([arxiv.org](https://arxiv.org/abs/1710.05267))

There are also sequence-model variants: Öksüz et al. proposed **RNN-based MRF reconstruction**, arguing that the network should exploit the time-dependent structure of the fingerprint signal evolution rather than treat it as a static vector. More recently, transformer-based deep learning has also been reported for highly accelerated MRF reconstruction. ([arxiv.org](https://arxiv.org/abs/1812.08155))

### 3) “Dictionary-free learned regression from simulated qMRI signals” outside strict MRF
If your concern is specifically “has anyone already done learned regression from simulated MR signals to T1/T2/etc. without closed-form fitting?”, then yes: **PERK** is an explicit example. It uses prior distributions plus the nonlinear MR signal model to simulate parameter/measurement pairs, then learns a regression map for quantitative MRI parameter estimation. That is not a neural net, but conceptually it is very close to your “learned inverse head over simulated data” idea. ([arxiv.org](https://arxiv.org/abs/1710.02441))

### 4) Protocol-/acquisition-conditioned estimators
There is also prior work that starts to address your key objection—namely that the same signal value means different things under different acquisition histories. In cardiac MRF, Hamilton’s work includes a **fingerprint generator network** whose input includes **T1, T2, and the patient’s RR-interval timing vector**, i.e. acquisition-context information is part of the model rather than assumed fixed. ([pubmed.ncbi.nlm.nih.gov](https://pubmed.ncbi.nlm.nih.gov/35811730/))

Even closer in spirit, Kuppens et al. 2026 present **acquisition-independent qMRI parameter estimation with neural controlled differential equations**, where the model takes the signal sequence together with the corresponding independent-variable values and is designed to work across **different sequence lengths and acquisition configurations**. That is not arbitrary RF/gradient tokenization, but it is very much in the direction of “don’t assume one fixed protocol; feed the acquisition context to the model.” ([sciencedirect.com](https://www.sciencedirect.com/science/article/pii/S1361841525003147))

### 5) Sequence design/optimization is already a major literature
On the acquisition-design side, this is also not empty territory. Zhao et al. formulated **optimal experiment design for MRF** using Cramér–Rao ideas, and Jordan et al. later reported **automated pulse-sequence design for MRF** using physics-inspired optimization, claiming large scan-time gains over prior human-designed schedules. So if you are designing schedules, reviewers will absolutely compare you to CRB/Fisher/OED-style baselines, not just to hand-built sequences. ([pmc.ncbi.nlm.nih.gov](https://pmc.ncbi.nlm.nih.gov/articles/PMC6447464/?utm_source=openai))

### 6) RL in MRI design/control exists, but mostly not in your exact form
There is at least one clear primary-source example of RL for MRI control/design: **DeepRF** uses deep reinforcement learning to design RF waveforms. That paper also cites earlier MRI RL efforts including **AUTOSEQ** (Bayesian RL for pulse-sequence generation in an MRI physics simulation environment), simulated MRI-scanner control, and RL for sampling-pattern optimization. However, these works are about RF or acquisition-control design—not the full “learned tissue decoder over arbitrary event-history tokens” stack you described. ([nature.com](https://www.nature.com/articles/s42256-021-00411-1))

### 7) MR-STAT is another adjacent “don’t ignore this” family
A nearby family worth reviewing is **MR-STAT**, which performs parameter estimation directly from time-domain data via large-scale nonlinear inversion rather than dictionary matching. One MR-STAT paper explicitly says it reconstructs multiple quantitative parameter maps from a single short scan by performing spatial localization and parameter estimation on the time-domain data simultaneously, and notes that for some short robust pulse sequences conventional MRF reconstructions fail. This is not your amortized neural decoder, but it is another serious “generic transient-state qMRI inversion” line you do not want to overlook. ([arxiv.org](https://arxiv.org/abs/1904.13244))

## So: is your idea “just MRF”?
**Partly yes, partly no.** If you freeze the acquisition schedule and simply train a neural net from simulated signal evolutions to T1/T2/PD, then that is basically **deep-MRF / learned qMRI inversion**, not a new concept. ([nature.com](https://www.nature.com/articles/nature11971))

What seems **less already-done**, based on what I found, is the combination of: **(i)** arbitrary primitive event sequences rather than a fixed schedule, **(ii)** explicit conditioning on the full event history, **(iii)** online intermediate estimation during sequence generation, and **(iv)** using that estimator as the state for an RL policy that itself designs the next events. I’m phrasing that as an inference from the search, not a proof of nonexistence, because the literature is broad and some relevant work may be in ISMRM abstracts or niche conference proceedings. ([sciencedirect.com](https://www.sciencedirect.com/science/article/pii/S1361841525003147))

## What I think is already “claimed” vs what may still be novel

**Already claimed / crowded:**
- learned inversion from simulated MR signals to tissue parameters; ([arxiv.org](https://arxiv.org/abs/1710.05267))
- MRF-style varying schedules that create fingerprints; ([nature.com](https://www.nature.com/articles/nature11971))
- schedule optimization for qMRI/MRF; ([pmc.ncbi.nlm.nih.gov](https://pmc.ncbi.nlm.nih.gov/articles/PMC6447464/?utm_source=openai))
- some RL-based MRI design/control. ([nature.com](https://www.nature.com/articles/s42256-021-00411-1))

**Potentially differentiating for you:**
- decoder input is the **full primitive event timeline**, not just a fixed fingerprint index or a short vector of schedule metadata; ([sciencedirect.com](https://www.sciencedirect.com/science/article/pii/S1361841525003147))
- decoder is used **online during acquisition**, not just after a fixed scan; ([pubmed.ncbi.nlm.nih.gov](https://pubmed.ncbi.nlm.nih.gov/35811730/))
- policy outputs **pulse-sequence actions** and decoder feeds the RL state, i.e. **policy-conditioned qMRI inversion**. I did not find a clean canonical paper with exactly that stack in primary sources. ([nature.com](https://www.nature.com/articles/s42256-021-00411-1))

## My practical recommendation so you don’t repeat work

If you proceed, I would position the readout network as **borrowed infrastructure**, not as the paper’s novelty. In paper/proposal language, the safe claim is something like: **“We build on MRF/deep-qMRI ideas to learn an acquisition-history-conditioned inverse model, and use it inside an RL loop for adaptive pulse-sequence design.”** That clearly acknowledges prior art while preserving what is actually distinctive in your setup. ([nature.com](https://www.nature.com/articles/nature11971))

I would also benchmark against at least:
1. fixed-schedule MRF + learned decoder,  
2. CRB/Fisher or physics-inspired schedule optimization, and  
3. a simpler acquisition-conditioned regressor that uses per-readout metadata rather than full primitive tokens.  
Otherwise, it will be hard to show that the RL/event-token formulation is buying anything beyond established MRF/qMRI design methods. ([pmc.ncbi.nlm.nih.gov](https://pmc.ncbi.nlm.nih.gov/articles/PMC6447464/?utm_source=openai))

## Bottom line
**Yes: this is close to MRF.**  
**Yes: learned regression from simulated MR signals to tissue parameters already exists.**  
**Probably no: your exact “arbitrary event-timeline encoder + online RL state estimator” formulation does not look like a standard already-solved method from the primary sources I found.** ([nature.com](https://www.nature.com/articles/nature11971))

If you want, next I can turn this into a **prior-art matrix** with columns like:

- fixed schedule vs variable schedule  
- decoder sees schedule or not  
- online vs offline estimation  
- optimized by CRB / gradient / RL  
- closest baseline for your Paradigm A.

---

## 8. Risk and fallbacks specific to the regressor

| risk | likelihood | mitigation |
|---|---|---|
| Network fails §4.1 rank-of-truth gate | medium | Increase token dimensionality (add per-event B1, phase channel); switch to a larger Transformer; richer schedule sampler. |
| Network fails §4.2 σ-calibration | medium | Replace Gaussian NLL with a quantile loss or a small ensemble for predictive variance. |
| Distribution shift between offline schedules and on-policy RL rollouts | high once PPO starts converging | Iteration 2 on-policy refresh; if persistent, switch to online learning of the regressor during PPO with a stop-gradient on the policy gradient. |
| MRzero simulation too slow for 500k samples | low-medium | Cache deterministic sub-computations; parallelise generation across CPU workers; if needed, reduce to 200k and add stronger augmentation. |
| Regressor co-trains into the degenerate equilibrium of E1 | medium if Iteration 3 attempted | Keep Iteration 3 as a stretch goal only; document the E1 collapse as the prior failure mode and abandon co-training on the first sign of it. |

---

## 9. Concrete next steps

1. Write `python/paradigm_a/tokenise.py` — `tokenise(events, signals) → np.ndarray[100, 14]`.
2. Write `python/paradigm_a/regressor.py` — Transformer module, NLL loss, train loop.
3. Write `python/paradigm_a/gen_offline_dataset.py` — schedule sampler + MRzero call + HDF5 writer.
4. Generate 10k-sample smoke dataset; train regressor for 5 epochs; spot-check §4 gates.
5. If smoke clears, scale to 500k and full training overnight.
6. Wire the trained checkpoint into `ParadigmAEnv` (§5).
7. Re-run the existing `python/train_paradigm_a.py` smoke test; confirm reward signal is non-degenerate.
8. Only then commit to a full PPO run.

**Budget:** 3–4 working days for steps 1–6, 1 day for step 7, then the PPO run is on the `PLAN_E2E_RL.md` §8.5 schedule.


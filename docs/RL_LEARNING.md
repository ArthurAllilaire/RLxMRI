# RL Learning Plan

Target: understand and use PPO and SAC in a real project (this one).
Assumed background: neural nets, attention, backprop, basic ML, evolutionary algorithms.

---

## The core idea (read this first)

RL is just supervised learning where you don't have labels — instead you get a scalar reward signal after acting. The agent must figure out which past actions caused good outcomes (the **credit assignment problem**). Everything else is machinery to solve that problem efficiently.

Three things every RL algorithm needs:
- A **policy** π(a|s) — what action to take in state s
- A way to **evaluate** how good a state or action is (value functions)
- A way to **improve** the policy using that evaluation

---

## Stage 1 — Foundations (2–3 days)

### Markov Decision Processes (MDPs)
The formal framework everything is built on:
- **State** s, **action** a, **reward** r, **next state** s'
- **Transition** P(s'|s,a) — environment dynamics (usually unknown)
- **Discount factor** γ ∈ [0,1) — future rewards worth less than immediate ones
- **Episode** vs **continuing** tasks

### Value functions — the key objects
- **Return** Gₜ = rₜ + γrₜ₊₁ + γ²rₜ₊₂ + ... (discounted sum of future rewards)
- **V^π(s)** = expected return starting from state s, following policy π
- **Q^π(s,a)** = expected return starting from state s, taking action a, then following π
- **Advantage** A(s,a) = Q(s,a) − V(s) — how much better is action a than average?
- **Bellman equation**: V(s) = Σₐ π(a|s) [r + γ V(s')] — recursive definition, this is everything

### Exploration vs exploitation
- Epsilon-greedy, softmax policies
- Why this is hard (and why evolutionary algorithms don't solve it well at scale)

**Resource**: Sutton & Barto "Reinforcement Learning: An Introduction" chapters 1–4 (free PDF, just these chapters). Dense but worth it — everything else refers back to it.

---

## Stage 2 — Tabular → Deep (1–2 days)

### Q-learning (tabular)
```
Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') − Q(s,a)]
```
Update Q towards the Bellman target. Works when s and a are discrete and small.

### DQN — your bridge from DL to RL
Q-learning + neural net as Q function approximator. Two tricks that make it stable:
- **Experience replay** — store (s,a,r,s') tuples, sample random minibatches (breaks temporal correlation, just like shuffling training data)
- **Target network** — a frozen copy of Q updated slowly (stops the target moving while you chase it — like a loss surface that keeps shifting)

This is the point where your DL knowledge directly applies. Loss is just MSE between Q(s,a) and the Bellman target.

**Resource**: original DQN paper (Mnih et al. 2015) — short, readable.

---

## Stage 3 — Policy Gradients → PPO (3–4 days)

DQN only works with discrete actions. Your project has continuous actions, so you need policy gradient methods.

### REINFORCE (vanilla policy gradient)
Instead of learning Q, directly optimise the policy parameters θ:
```
∇θ J(θ) = E[∇θ log π(a|s) · Gₜ]
```
Intuition: increase the log-probability of actions that led to high return, decrease for low return. This is the policy gradient theorem — derivation is worth reading once.

Problems: extremely high variance (Gₜ is noisy), sample inefficient.

### Actor-Critic
Fix variance by replacing Gₜ with the advantage A(s,a) = r + γV(s') − V(s):
- **Actor** = policy network π(a|s) — decides actions
- **Critic** = value network V(s) — evaluates states, used only during training

Now you're training two networks simultaneously. The critic gives lower-variance signal than raw returns.

### PPO (Proximal Policy Optimization)
The problem with vanilla policy gradients: a too-large gradient step can collapse the policy catastrophically (unlike supervised learning where you can just reduce the learning rate).

PPO fixes this with a **clipped objective**:
```
L = E[ min(r(θ) · A, clip(r(θ), 1-ε, 1+ε) · A) ]
```
where r(θ) = π_new(a|s) / π_old(a|s) is the probability ratio between new and old policy.

- If the new policy diverges too far from old (r outside [1-ε, 1+ε]), the gradient is clipped to zero — hard stop on how much you can update
- ε is typically 0.1–0.2
- Simple to implement, very stable, works well on-policy

**On-policy**: you collect fresh experience with the current policy, update, throw it away. Wasteful but stable.

PPO is the default starting point for most new RL problems. If something works at all, PPO probably works.

**Resource**: PPO paper (Schulman et al. 2017) — 8 pages, very readable. Also OpenAI Spinning Up "Introduction to RL" notes (search "spinning up openai").

---

## Stage 4 — SAC (2–3 days)

PPO wastes data (on-policy). SAC fixes this.

### Off-policy learning
Store all experience in a **replay buffer**, sample from it at any time. Much more sample-efficient. The challenge: the data was collected under old policies, so you need corrections.

### Maximum entropy RL
SAC adds an entropy bonus to the reward:
```
J(π) = E[ Σ r(sₜ,aₜ) + α H(π(·|sₜ)) ]
```
H is entropy — the agent is rewarded for being uncertain/diverse in its actions. This means:
- Natural exploration without epsilon-greedy hacks
- Learns multiple ways to solve a problem (robust policy)
- α (temperature) controls the exploration/exploitation trade-off and can be tuned automatically

### SAC components
- **Two Q-networks** (to reduce overestimation bias — take the min)
- **One policy network** that outputs a Gaussian distribution over actions (mean + std)
- **Replay buffer** (like DQN)
- No separate target actor — policy is updated to maximise Q + entropy

SAC is the go-to for **continuous action spaces** (like MRI acquisition parameters). More sample-efficient than PPO but slightly trickier to tune.

**Resource**: SAC paper (Haarnoja et al. 2018) — read the intro and algorithm box. Also Spinning Up SAC page.

---

## Stage 5 — Applied to this project (ongoing)

Your E1 environment (`src/rl/e1.jl`) defines:
- `e1_reset!` — start a new episode (new phantom config)
- `e1_step!` — take an action (choose next sequence params), get observation + reward
- `e1_obs_dim`, `e1_action_table` — state and action space

Questions worth understanding before training:
1. What is the observation space? (What does the agent see after each acquisition?)
2. What is the action space? (What can it choose — TI values, TE, number of readouts?)
3. What is the reward? (Probably related to T1/T2 estimation error)
4. Is the action space discrete or continuous? (Determines PPO vs SAC preference)

**Start with PPO** — easier to debug, less hyperparameter-sensitive. Move to SAC if sample efficiency matters (i.e. each `e1_step!` is expensive to simulate).

---

## Concept map

```
Bellman equation
    └── Q-learning (tabular)
            └── DQN (+ neural net, replay buffer, target net)
                    └── discrete actions only

Policy gradient theorem
    └── REINFORCE (high variance)
            └── Actor-Critic (+ value baseline)
                    └── PPO (+ clipping, on-policy)     ← start here
                    └── SAC (+ entropy, off-policy, continuous)  ← use for MRI
```

---

## What to skip for now
- Model-based RL (you don't have a differentiable simulator)
- Multi-agent RL
- Distributional RL (C51, QR-DQN)
- The evolutionary algorithm connection — NES/CMA-ES exist but are outclassed by PPO/SAC on anything with a gradient signal

---

## Stage 6 — Gaps after Spinning Up (project-specific)

Spinning Up covers the algorithms well. These are the things it doesn't cover that matter for this project.

### Reward shaping (caused E1 to fail — read this first)

Spinning Up barely touches reward design. The formal result you need is **potential-based reward shaping** (Ng, Harada & Russell 1999): you can add `F(s,s') = γΦ(s') − Φ(s)` to any reward without changing the optimal policy. This justifies dense rewards derived from a "progress" signal.

E1's terminal bonus `+1.0` was additive and unrelated to the per-step cost structure — so the agent only needed to collect the bonus, not improve the sequence. The E2 dense `−MAPE_t` reward is approximately potential-based (Φ(s) = −MAPE). Keep the terminal bonus small or zero.

**Read**: Ng, Harada & Russell "Policy Invariance Under Reward Transformations" (1999) — 4 pages, §3 is the key theorem.

---

### Partial observability — this task is a POMDP

Spinning Up assumes full observability. This task is a **POMDP**: the agent can't see `T1_true`, only accumulated signals. Standard PPO/SAC treat the observation as Markovian, which works if the observation is rich enough. For this project the running LM estimate in the observation carries most of the needed state, so an MLP policy is probably fine — but flag the POMDP framing in the report (it's the honest description). If the MLP fails on E2, try `RecurrentPPO` from `sb3-contrib`, which gives the policy an LSTM over history.

---

### Entropy regularisation in PPO (directly relevant to E1 collapse)

Spinning Up mentions entropy but doesn't emphasise `ent_coef`. In SB3:

```python
PPO(..., ent_coef=0.01)  # adds α·H(π) to the training loss
```

A higher `ent_coef` (0.01–0.1) prevents the policy collapsing to a single repeated action. E1 almost certainly had this too low or zero. Set it and monitor `H(π)` during training.

---

### Diagnosing degenerate policies

Spinning Up doesn't cover diagnostics. Key checks to add to your eval loop:

- **Action entropy over training**: if `H(π(·|s))` → 0, the policy has collapsed
- **Action histogram across episodes**: plot distribution of chosen TIs — a spike on one value is bad
- **Episode return vs training step**: immediate plateau = trivial solution found
- **Correlation of actions with T1_true**: an adaptive policy should show `TI_chosen` varying with `T1_true`; a degenerate policy shows no correlation

---

### Practical SB3 details (Spinning Up is from-scratch)

- **`VecEnv`**: SB3 expects vectorised environments. `make_vec_env(env_fn, n_envs=4)` runs 4 envs in parallel — more data per PPO update. Each env needs its own Julia runtime; test whether 4 parallel runtimes fits in memory before committing.
- **`EvalCallback`**: auto-evaluates on a held-out set during training, saves best model checkpoint
- **Custom logging**: add MAPE and action entropy as custom scalars with `logger.record()` inside a callback
- **Hyperparameters that matter most**: `n_steps` (increase if episodes are long — should cover several full episodes), `batch_size`, `ent_coef`, `clip_range`

---

### Active sensing / optimal experiment design (for the report novelty claim)

Spinning Up has nothing on this, but it's the academic framing that makes the project publishable.

- **Fisher information**: the information a measurement provides about an unknown parameter is `I(θ) = E[(∂ log p(x|θ)/∂θ)²]`. For T1 mapping, an IR acquisition at `TI ≈ T1·ln(2)` is the maximally informative single measurement — the RL agent should learn to do this.
- **Bayesian experimental design (BED)**: choose the next experiment to maximise expected information gain. The RL agent is implicitly doing this; stating it explicitly is the novelty claim for Ch4/A2.
- **Connection to MRF** (Jordan et al. 2021): MRF maximises distinguishability of the signal manifold. The E3 dictionary-discriminability reward is this. Framing: "the agent learns to approximate the Fisher-optimal acquisition schedule" ties E2 and E3 together.

**Read**: skim the Jordan et al. 2021 intro for how they connect Fisher information to MRF sequence design.

---

### Curriculum learning (for scaling E2 → E3)

Not in Spinning Up. If the E2 agent doesn't converge on the full task:
1. Start training with no noise (`σ=0`), add noise after the policy stabilises
2. Start with 1 sphere, expand to 14 once the single-sphere task is solved
3. SB3 supports mid-training env parameter changes via callbacks

Not required for E2 but know it exists before concluding the task is "too hard".

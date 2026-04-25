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

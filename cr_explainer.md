# Cramér–Rao optimal schedule — math walkthrough

A line-by-line companion to `src/baselines/cr_optimal.jl`. Assumes you know basic matrix calculus and have seen Cramér–Rao before. Each section connects to specific lines in the code.

---

## 1. The forward model and what we're estimating

For sphere `j` with true relaxation time `T1_j` and amplitude `A_j`, an Npe-shot IR-SE block at `(TI, TR)` produces a measurement (image-domain magnitude after IFFT):

$$
S_k(T1_j, A_j) \;=\; A_j \cdot \bigl| \; \tilde m(T1_j; \mathrm{TI}_k, \mathrm{TR}_k) \; \bigr|
$$

where $\tilde m$ is the F1+ closed-form transient (`transient_mz_at_excite_npe` in `src/fitting/fits.jl`):

$$
\tilde m(T1; \mathrm{TI}, \mathrm{TR}) \;=\; \frac{1}{N_{\mathrm{pe}}} \sum_{s=1}^{N_{\mathrm{pe}}} M_z^{(s)}(T1, \mathrm{TI}, \mathrm{TR})
$$

with the per-shot recurrence in `E2_4_PLAN.md` §2.2. Take this as a black-box function; we only need that it is differentiable w.r.t. $T1$ and that $\partial S / \partial A = \tilde m$ is non-zero away from the null point.

The fitter, given measurements $\{m_k\}_{k=1..n}$ (one per block, one per sphere), solves

$$
(T1_j^*, A_j^*) \;=\; \arg\min_{T1, A} \;\sum_{k=1}^{n} \bigl(m_k \;-\; S_k(T1, A)\bigr)^2.
$$

This is the LM fit in `fit_t1_generalized_ir`. The Cramér–Rao bound below tells us the *minimum possible variance* of $T1_j^*$ given the schedule, *before we run anything*.

---

## 2. The Cramér–Rao lower bound

Assume measurement noise is iid Gaussian with variance $\sigma^2$ (line 260 of `src/rl/e2.jl` adds k-space noise; after IFFT it's approximately Gaussian per pixel — see §12.1 of `EXPERT_REPORT_E2_4.md` for the caveat).

The **Fisher information matrix** $I(T1, A) \in \mathbb{R}^{2 \times 2}$ for one sphere given a schedule of $n$ blocks is

$$
I_{ab}(T1, A) \;=\; \frac{1}{\sigma^2}\;\sum_{k=1}^{n}\;
\frac{\partial S_k}{\partial \theta_a}\;\frac{\partial S_k}{\partial \theta_b}
\quad \text{with}\quad \theta = (T1, A).
$$

Define the **Jacobian** $J \in \mathbb{R}^{n \times 2}$:

$$
J_{k,1} \;=\; \frac{\partial S_k}{\partial T1}\bigg|_{T1, A}, \qquad
J_{k,2} \;=\; \frac{\partial S_k}{\partial A}\bigg|_{T1, A}.
$$

Then $I = J^\top J / \sigma^2$, and the Cramér–Rao bound (CRB) says: for any unbiased estimator $\hat T1$,

$$
\boxed{\;\;\mathrm{Var}(\hat T1) \;\geq\; \sigma^2 \cdot \bigl[(J^\top J)^{-1}\bigr]_{T1, T1}\;\;}
$$

Code: `cr_T1_variance` (`cr_optimal.jl:60-95`).

The 2×2 matrix inverse has the closed form $\bigl[(J^\top J)^{-1}\bigr]_{T1, T1} = J_{:,2}^\top J_{:,2} / \det(J^\top J)$ where $\det(J^\top J) = (J_{:,1}^\top J_{:,1})(J_{:,2}^\top J_{:,2}) - (J_{:,1}^\top J_{:,2})^2$. Lines 80–88 of `cr_optimal.jl` implement this directly. If $\det \leq 0$ the matrix is singular (T1 is unidentified given this schedule) and we return `Inf`.

**Why the LM minimum minus this lower bound is informative.** The LM fit's actual variance equals the CRB asymptotically (in the large-data, no-multimodality regime). For multimodal SSE (§9.9.4, `EXPERT_REPORT_E2_4.md`) the CRB is *optimistic* — it underestimates the true variance. So the CRB is a *lower bound on what any fixed schedule can achieve* under perfect-modelling assumptions. CR-optimal is therefore a *theoretical anchor*, not a guarantee.

---

## 3. Why we use finite differences, not autodiff

The Jacobian has two columns. The amplitude column is trivial — $\partial S / \partial A = |\tilde m|$, exact. But $\partial S / \partial T1$ involves differentiating $|\sum_s M_z^{(s)}(T1)|$ where the absolute value is non-smooth at the null and the recurrence has $T1$ inside multiple `exp(−·/T1)` factors via $E_1$ and $E_2$.

We use central finite differences with a *relative* step (line 56 of `cr_optimal.jl`):

$$
\frac{\partial S}{\partial T1}\bigg|_{T1} \;\approx\; \frac{S(T1 + \varepsilon) - S(T1 - \varepsilon)}{2 \varepsilon},
\qquad \varepsilon = \max(10^{-4},\; 10^{-4} \cdot |T1|).
$$

The relative step matters because for $T1 = 0.023$ s an absolute step of $10^{-4}$ is 0.4% of the parameter (fine), but at $T1 = 1.84$ s it's $5.4 \times 10^{-5}$ relative (still fine). The step floor at $10^{-4}$ avoids loss-of-precision issues for very small $T1$.

Autodiff via `ForwardDiff.jl` would give exact derivatives at no compute cost overhead — but adds a project dep, and the finite-difference error here is well below the optimisation noise (the multi-start has stochastic seed-dependent variance ~1 % in $L$, which dominates).

---

## 4. Fleet objective: weighting per-sphere variances by $1/T1_j^2$

For a 14-sphere fleet under uniform sampling ($p_j = 1/14$ for each $j$), the natural objective is the **expected MAPE-squared**:

$$
\mathbb{E}_j\;\left[\;\left(\frac{\hat T1_j - T1_j}{T1_j}\right)^2\right]
\;\geq\; \mathbb{E}_j\;\left[\frac{\sigma^2 \cdot [(J_j^\top J_j)^{-1}]_{T1,T1}}{T1_j^2}\right]
\;\propto\; \sum_{j=1}^{14}\;\frac{[(J_j^\top J_j)^{-1}]_{T1,T1}}{T1_j^2}.
$$

The proportionality constant ($\sigma^2 / 14$) is independent of the schedule, so we drop it. Code: `cr_fleet_objective` (`cr_optimal.jl:97-117`).

**Why divide by $T1_j^2$?** Without it, the objective $\sum_j \mathrm{Var}(\hat T1_j)$ is dominated by the long-T1 spheres (their absolute variance is large), and the optimiser ignores short-T1 spheres entirely. Dividing by $T1_j^2$ makes the per-sphere term proportional to relative-error-squared — equivalent to weighting each sphere by its inverse T1 squared. That's the same weighting MAPE uses, so the CR-optimal schedule is the analytic lower bound for *MAPE-style* fleet errors. It also matches the env's reward function (mean MAPE) so the comparison V5-vs-CR-opt is apples-to-apples.

If you wanted to weight some spheres more (e.g. give 2× weight to short-T1), you'd pass `weights = [1, 1, ..., 2, 2, 2]` to `cr_fleet_objective`. This is the lever to test "what if CR-opt prioritised short-T1 the way V5's accidental floor exploit does?" — see §8 below.

---

## 5. Constrained optimisation problem

Putting it together, the CR-optimal schedule solves:

$$
\begin{aligned}
\text{minimise}_{(\mathrm{TI}_k,\, \mathrm{TR}_k)_{k=1..n}} \quad
& L \;=\; \sum_{j=1}^{14}\;\frac{[(J_j^\top J_j)^{-1}]_{T1,T1}}{T1_j^2} \\[4pt]
\text{subject to}\quad
& \sum_{k=1}^{n} \bigl(N_{\mathrm{pe}} \cdot \mathrm{TR}_k + \tau_{\mathrm{overhead}}\bigr) \;\leq\; 120\,\text{s} \\
& 0.01 \;\leq\; \mathrm{TI}_k \;\leq\; 3.0 \\
& \max(\mathrm{TI}_k + 0.05,\; 0.5) \;\leq\; \mathrm{TR}_k \;\leq\; 5.0 \\
& n \in \{4, 6, 8, 10, 12, 14, 16, 18\} \quad\text{(swept)}
\end{aligned}
$$

The schedule also implicitly fixes $\theta_{\mathrm{inv}} = \pi$, $\alpha_{\mathrm{exc}} = \pi/2$, $T_E = 20\,\text{ms}$ (matching what the env actually runs in simplified-action mode). $\tau_{\mathrm{overhead}} = 0.05$ s per block accounts for spoiler gradients, ADC delays, etc. (`block_time_s`, line 16).

This is a non-convex, non-smooth, mixed-integer (in $n$) problem. We solve it heuristically — see §6.

---

## 6. Why multi-start + coordinate descent (not LBFGS, not gradient descent)

Three properties of $L$ make naïve gradient descent fail:

1. **Multimodality.** The forward $\tilde m$ has the abs() in front, creating a non-smooth surface. Two distant schedules (e.g., one with all TIs near 0.5 s, one with all TIs spread log-uniformly) can give similar $L$ but be in different basins.
2. **Schedule symmetry.** If $(\mathrm{TI}_1, \mathrm{TI}_2, \ldots) = (a, b, \ldots)$ achieves $L^*$, then so does $(b, a, \ldots)$. Gradient methods get confused at these flat-region transitions.
3. **Non-smoothness at the null.** $\tilde m$ flips sign at $\mathrm{TI} \approx T1 \cdot \ln 2$; the abs() makes $|\tilde m|$ non-differentiable there. Finite-difference Jacobian smooths past this in expectation, but a gradient-method's step rule sees a kink.

**Multi-start** (`cr_optimize`, `cr_optimal.jl:225-251`):
- Sample $K = 1000$ random schedules log-uniformly: $\log \mathrm{TI}_k \sim U(\log 0.01, \log 3.0)$, $\log \mathrm{TR}_k \sim U(\log \max(\mathrm{TI}_k + 0.05, 0.5), \log 5.0)$.
- Reject any that exceed budget; resample up to 100× per slot.
- Score each by $L$.
- Take the top 10.

**Coordinate descent** (`refine_coordinate_descent`, `cr_optimal.jl:155-218`):
- For each of the top 10, iterate: for each block $k$, try multiplying $\mathrm{TI}_k$ by $\{1.3, 1/1.3, 1.1, 1/1.1\}$, then same for $\mathrm{TR}_k$.
- Accept any move that lowers $L$ and keeps the budget feasible.
- Stop when no move in a full pass improves $L$.

Both phases are deterministic given a seed (`MersenneTwister(0)` by default). Coordinate descent doesn't need gradients and handles the non-smoothness gracefully. The trade-off is *no convergence guarantee* — we report the best of 10 refined starts, with an "all" dictionary so you can see seed-spread.

A clean upgrade path: replace coordinate descent with `Optim.jl`'s `NelderMead` or `ParticleSwarm` (both gradient-free, no new dep beyond Optim). For E2.5 / §13 the multi-start spread is < 1 % across 10 starts, so the heuristic is sufficient.

---

## 7. The `n_blocks` sweep and why $n^* = 14$ for our fleet

The objective $L$ has two competing pressures as $n$ grows:

- **More blocks → more measurements per sphere → lower $\mathrm{Var}(\hat T1_j)$**. The Fisher information adds linearly: $I = \sum_k J_k^\top J_k$, so doubling $n$ (with the same per-block info) halves the variance.
- **More blocks → shorter TR per block** (to fit in the budget) → less per-block T1 information. F1+'s gradient-in-TR is steepest at TR ≈ 1–3 s; below 0.5 s the per-shot magnetisation drops and Jacobian rows shrink.

The crossover is fleet-dependent. For our 14-sphere T3 fleet at 120 s budget, the sweep gave (`cr_optimize_sweep`):

| $n$ | $L$ |
|---:|---:|
| 4 | 21.33 |
| 6 | 15.66 |
| 8 | 13.64 |
| 10 | 12.98 |
| 14 | 13.01 |

So $n^* = 10$ on the first run (500 starts) and $n^* = 14$ on the higher-resolution run (1000 starts) — the optimum is broad in the 10–14 region. Anything outside this band is dominated. **The right interpretation: ~10–14 blocks is the information-optimal block count for a 120 s budget on the 14-sphere fleet**, regardless of whether RL or analytic optimisation finds it.

---

## 8. What `TI_lo = 0.01` checks (and what it doesn't)

The solver's existing `TI_lo = 0.01` parameter (line 121 of `cr_optimal.jl`) sets the **lower bound on each $\mathrm{TI}_k$ during random sampling and refinement**. Two things to disentangle:

### 8.1 What the parameter currently does

The random sampling at line 122 draws $\log \mathrm{TI}_k \sim U(\log 0.01, \log 3.0)$. This *does* sample TIs near 0.01 s — about $\log(0.05/0.01)/\log(3.0/0.01) \approx 28\%$ of samples are below 0.05 s, and $\log(0.02/0.01)/\log(3.0/0.01) \approx 12\%$ are below 0.02 s. So the optimiser *sees* very-low TIs in its candidate pool.

But the global optimum the multi-start found has its smallest TI at **0.035 s**. None of the 14 TIs in the converged schedule sit below 0.03 s. Why? Because **the fleet objective $L = \sum_j \mathrm{Var}(\hat T1_j)/T1_j^2$ doesn't reward putting a TI at 0.015 s for the T1 = 0.023 s sphere** — that one sphere's variance reduction is tiny (its $T1_j^2$ in the denominator makes its weight smaller in absolute units than longer-T1 spheres benefit from a TI cluster at ~0.5 s). The optimiser's marginal gain from "use one block at TI = 0.02 s for sphere 13" is less than the marginal gain from "use that block at TI ≈ 0.5 s for spheres 4, 5, 6 simultaneously".

This is a feature, not a bug — the CR optimum balances information across the fleet, not per sphere. **It's also why V5 wins on idx 11–12**: V5's policy doesn't optimise the fleet objective; it spams TI = 0.01 s for delta_mape reward, which by happy accident is informative for the very-short-T1 spheres CR-opt under-weights.

### 8.2 What a "TI_lo experiment" would actually probe

There are three different things you could change, all sometimes called "lowering the TI floor":

**(a) Force inclusion of low-TI shots in the schedule.** Add a constraint $\min_k \mathrm{TI}_k \leq 0.02$. This *forces* the optimiser to spend at least one shot at very low TI, then optimises the rest around it. Tests "does CR-opt match V5 on idx 11–12 when forced to include sub-50 ms shots?"

**(b) Reweight the fleet objective toward short-T1 spheres.** Pass `weights = [1, 1, ..., w, w, w, w]` with $w = 5$ for indices 10–13. Tests "does CR-opt naturally use sub-50 ms TIs when the short-T1 spheres are weighted more in the objective?" The MAPE-style objective ($1/T1^2$) already weights short-T1 more than long-T1, but apparently not enough to pull TIs below 0.03 s.

**(c) Tighten the box constraint.** Change `TI_lo = 0.01` to `TI_lo = 0.005` (closer to the physics floor d180/2 ≈ 0.3 ms). This widens the search space but doesn't directly bias the optimum — only useful if the current floor was the binding constraint, which it isn't (the optimum sits at 0.035, well above 0.01).

**(d) Tighten the box constraint upward.** Change `TI_lo = 0.05` (matching the env's "non-floor" regime). This *removes* the very-low TI option entirely. Tests "what happens to CR-opt's idx 11–12 performance if the floor exploit is structurally unavailable?" — likely degrades, confirming that (a) and (b) above would close the V5 gap.

§13.6 of `EXPERT_REPORT_E2_4.md` mentions "CR-opt with TI_lo = 0.01 s" — that line was misnamed by me. The actual experiment worth running is **(a) or (b)**, not (c). Both of those test whether V5's idx 11–12 advantage is *capturable by an analytic schedule that knows about short-T1 spheres*, or whether RL is doing something the CR formalism can't see.

### 8.3 Predicted outcome of the "weighted CR-opt" experiment

If (b) is run with $w = 5$ on idx 10–13, the optimiser will likely:
- Move 1–2 of the current "0.04 s cluster" TIs down to the 0.015–0.025 s range (matching short-T1 optimum)
- At cost of ~1 fewer TI in the 0.5–0.7 s long-T1 cluster
- Net L (with original equal weights) probably *higher* than the unweighted optimum, but per-sphere MAPE on idx 11–12 lower

That predicts: **weighted CR-opt would narrow or eliminate V5's idx 11–12 advantage, while marginally worsening long-T1 performance**. If true, it confirms V5 has *no* per-sphere structural advantage over fleet-optimal fixed schedules — the floor exploit isn't a clever discovery, just an unintended consequence of equal sphere weighting in the analytic case.

If false (V5 still wins on idx 11–12 even when CR-opt is tuned to attack them), then V5 is doing *something* on these spheres that fixed schedules can't replicate — possibly the within-episode adaptivity claim still has support there.

This is the cheapest single experiment to run after E2.5: ~5 minutes (re-solve with weights, re-eval 30 eps).

---

## 9. Summary table of the math at a glance

| Symbol | Meaning | Code |
|---|---|---|
| $S_k(T1, A)$ | Forward signal at block $k$ | `f_signal` |
| $J \in \mathbb{R}^{n \times 2}$ | Jacobian, columns $\partial S/\partial T1$, $\partial S/\partial A$ | `jacobian_row` builds rows |
| $J^\top J$ | Gram matrix; Fisher info up to factor $\sigma^{-2}$ | inline, lines 80–88 |
| $[(J^\top J)^{-1}]_{T1,T1}$ | CR lower bound on $\mathrm{Var}(\hat T1)$ (up to $\sigma^2$) | `cr_T1_variance` |
| $L = \sum_j \mathrm{Var}_j / T1_j^2$ | Fleet objective ≈ $\mathbb{E}[\mathrm{MAPE}^2]$ | `cr_fleet_objective` |
| Multi-start search | $K$ random schedules + top-$N$ refinement | `cr_optimize` |
| Coordinate descent | Per-block $\mathrm{TI}, \mathrm{TR}$ perturbations | `refine_coordinate_descent` |
| `n_blocks` sweep | Outer loop over schedule length | `cr_optimize_sweep` |

---

## 10. What the CR-optimal *cannot* tell you

Three things outside its scope, important not to overclaim:

1. **The CR bound is a *lower* bound.** The actual fitter can do worse (multimodal SSE, optimiser convergence, finite-noise effects). So CR-opt's reported MAPE in §13 (220.8 %) is *the actual fitter's MAPE on the optimal schedule* — usually above the CR bound. The bound itself is what the optimiser minimises; the env eval is what we report.
2. **CR doesn't see multimodal SSE.** The Jacobian is local at the truth; a far-away SSE basin has no effect on $J^\top J$ at $T1_j$. So the CR objective treats short-T1 spheres as well-determined when they aren't (§9.9.4). This is why we need profile-likelihood σ for the *reported* uncertainties (E2_5_PLAN.md §3) — but it doesn't affect CR's *schedule*, only the σ values we'd quote afterwards.
3. **CR is a *fixed-schedule* optimum.** It has no observation channel — it cannot adapt to running T1_est within an episode. So a strictly-better policy must either (i) condition on observations (RL), or (ii) include sphere-identity information at decision time (oracle CR-opt, §3.1 Formulation B of `E2_TRACTABILITY_PLAN.md`). The fact that V5 ties CR-opt suggests V5's adaptive use of observations is small-or-nil on the 14-sphere fleet.

The cleanest way to read §13: **CR-opt is the ceiling for non-adaptive policies under perfect-modelling assumptions; V5 is at that ceiling within 6 %; the gap to RL adaptivity hasn't been demonstrated yet on this fleet.**

---

## 11. The optimisation algorithm itself — multi-start + coordinate descent

§6 mentioned "multi-start + coordinate descent" without explaining what those are. Each is a standard technique on its own; the combination handles the awkwardness of our objective.

### 11.1 The problem we're solving, restated abstractly

We have a function $L : \mathbb{R}^{2n} \to \mathbb{R}_{\geq 0}$ that takes a schedule $\mathbf{s} = (\mathrm{TI}_1, \mathrm{TR}_1, \mathrm{TI}_2, \mathrm{TR}_2, \ldots, \mathrm{TI}_n, \mathrm{TR}_n)$ and returns the fleet objective. We want $\mathbf{s}^* = \arg\min_{\mathbf{s} \in \mathcal{F}} L(\mathbf{s})$ where $\mathcal{F}$ is the feasible set (box constraints + budget).

Three things make this hard, repeating §6:
- **Non-convex.** $L$ has many local minima — schedules with different "structure" (e.g. one with all TIs in [0.05, 0.5] s vs one with all in [0.3, 1.5] s) can be local minima of comparable depth.
- **Non-smooth.** $|\tilde m|$ has a kink at the null point, so $\nabla L$ is undefined at points where any block sits exactly at $\mathrm{TI}_k = T1_j \ln 2$.
- **Constrained.** The budget constraint $\sum_k N_{\mathrm{pe}} \cdot \mathrm{TR}_k + \tau \leq B$ is a hyperplane in $\mathbb{R}^{2n}$ that can cross the unconstrained minimum.

Standard gradient descent fails on all three. We need something gradient-free and global-aware.

### 11.2 Multi-start: brute force against non-convexity

**The idea.** Pick $K$ random points in $\mathcal{F}$, evaluate $L$ at each, keep the best.

**Why it works (probabilistically).** If the global minimum lies in a basin of "volume" $V_{\min}$ and the feasible set has volume $V_{\mathcal{F}}$, the probability that one random sample lands in the global basin is $V_{\min}/V_{\mathcal{F}}$. With $K$ independent samples, the probability of *missing* the global basin entirely is $(1 - V_{\min}/V_{\mathcal{F}})^K$. For $V_{\min}/V_{\mathcal{F}} = 0.001$ and $K = 1000$, that's $\approx 0.37$ — i.e., you find it ~63 % of the time. For $K = 5000$ it's >99 %.

**What it doesn't give you.** Multi-start *finds the basin*, but the random sample within the basin probably isn't the basin's minimum. You still need a local-search step to drive each candidate down to its basin floor. That's what coordinate descent does.

**Pseudocode** (ours, `cr_optimize:225-251`):
```
for k = 1..K:
    s_k = sample_random_feasible()
    score[k] = L(s_k)
sort schedules by score
return top N for refinement
```

**Standard alternatives:** Latin hypercube sampling (more uniform than uniform), Sobol sequences (quasi-random, deterministic). For our use we use plain log-uniform — the randomness is fine because we score 1000 samples and only refine 10.

### 11.3 Coordinate descent: greedy local search without gradients

**The idea.** Given a current point $\mathbf{s}$, try perturbing **one coordinate at a time**. Accept any perturbation that decreases $L$. Repeat until no single-coordinate perturbation helps.

For a smooth convex $L$, coordinate descent converges to the global minimum (provably for separable functions). For non-smooth or non-convex $L$, it converges to a *coordinate-wise local minimum* — a point where no single coordinate move improves things, but a *joint* move of two coordinates might.

**Why this works for us specifically.** Our $L$'s non-smoothness is *along* coordinates (each block's $\mathrm{TI}_k$ has a kink at the null of one sphere). But the kinks for different spheres are at different $\mathrm{TI}$ values, so changing $\mathrm{TI}_k$ smoothly moves $L$ between kink crossings. Coordinate descent handles this gracefully — it just sees a piecewise-smooth 1D function in each direction and walks downhill.

**Pseudocode** (ours, `refine_coordinate_descent:155-218`):
```
repeat until no improvement in a full pass:
    for k = 1..n_blocks:
        for each parameter (TI_k, TR_k):
            for each step factor f in {1.3, 1/1.3, 1.1, 1/1.1}:
                propose new_value = current * f, clamped to feasible
                if L(new_schedule) < L_best AND budget_ok:
                    accept
                    L_best = L(new_schedule)
```

**Why multiplicative steps, not additive?** TIs span 2.5 decades (0.01 to 3 s) — an additive step of 0.05 s is a large move at TI=0.05 s but tiny at TI=2 s. Multiplicative steps respect the log-scale geometry of the parameter.

**Why four factors?** $\{1.3, 1/1.3\}$ does coarse exploration; $\{1.1, 1/1.1\}$ refines once you're near a minimum. Two-scale schedule is a poor man's line search — a real line search would scan a continuous range of step sizes per direction.

**Standard alternatives:**
- **Nelder–Mead** (downhill simplex). Maintains a simplex of $2n+1$ points and reflects, expands, contracts. Handles non-smoothness, no gradients. Generally ~2× faster than coordinate descent in our regime. `Optim.jl::NelderMead()`.
- **Pattern search** (Hooke–Jeeves). More sophisticated coordinate-descent variant with pattern moves between coordinate moves. `Optim.jl::ParticleSwarm()` is in this family.
- **CMA-ES** (Covariance Matrix Adaptation Evolution Strategy). State-of-the-art for non-smooth + non-convex, but heavier. Available via `Optim.jl::CMAEvolutionStrategy()` plugin or `Evolutionary.jl`.

### 11.4 Why the combination converges

For our 14-sphere problem with $n = 14$ blocks (28 dimensions):

- **1000 random samples** find the rough region of the global minimum with $K \approx 1000$ probability ≈ 99 % (the basin appears geometrically large — there's a wide range of TI-cluster locations that work, the optimum is broad).
- **Top 10 → coordinate descent** drives each into its local floor. Spread across the 10 final $L$ values is < 2 %.
- **Best of 10** is reported.

This is a heuristic — no convergence proof — but for E2.5's purposes (single global optimum reported, used as a baseline in §13) it's enough. The seed-spread of 2 % is small relative to the V5 vs CR-opt gap of 6 %, so the comparison is statistically meaningful.

### 11.5 What better methods would buy us

| Method | Pros | Cost |
|---|---|---|
| Nelder–Mead refinement | Faster local convergence (~2×); already in `Optim.jl` | 30 min to integrate |
| CMA-ES (single-stage, no multi-start needed) | Handles ~50-dim problems well; published convergence rate | 1 h to integrate |
| Bayesian optimisation (`BayesianOptimization.jl`) | Sample-efficient — far fewer L evaluations needed | More dependencies; setup heavier |
| LBFGS on a smoothed surrogate (replace abs with $\sqrt{x^2 + \epsilon^2}$) | Fast quasi-Newton convergence; gradient-aware | Smoothing changes the problem; need to retest CR claim |

For our scale (28 parameters, ~30 s total wall-time in the optimiser) the gain from any of these is modest — the bottleneck isn't optimiser efficiency, it's the env eval afterwards (~30 min for 30 episodes). I wouldn't replace coordinate descent unless we move to ~100-dim problems or want strict convergence guarantees for a publication.

---

## 12. Migration plan — moving from in-house to existing tooling

Three tiers, in increasing scope. Each is independent of the others — you can do tier 1 without tier 2, etc.

### 12.1 Tier 1 — Replace the optimiser only (~1 hour)

Keep `cr_T1_variance` and `cr_fleet_objective` unchanged (these encode our F1+ forward model and are correct). Replace `cr_optimize` and `refine_coordinate_descent` with calls to `Optim.jl`:

**What changes:**
- Add `Optim` to `Project.toml`.
- New `cr_optimize_optim` (or replace existing): builds a closure `f(s) = cr_fleet_objective(...)`, defines a constraint via `Optim.IPNewton()` with box constraints, optimises with `LBFGS()` if smoothness is OK or `NelderMead()` / `ParticleSwarm()` otherwise.
- Multi-start: keep the random-init loop, just call Optim instead of coordinate descent for refinement.

**Why this is worth it:** convergence guarantees, faster local refinement, smaller code surface, fewer "is the optimiser converged?" questions in the dissertation. **Why it's not urgent:** our seed-spread is 2 % across 10 starts; tightening to <1 % doesn't change the §13 conclusions.

**Concrete diff sketch** (just to give you the shape — not committed):
```julia
using Optim
function cr_optimize_optim(T1s; n_blocks, budget_s, Npe = 8)
    # Pack schedule into a single vector: [TI_1, TR_1, TI_2, TR_2, ..., TI_n, TR_n]
    function f(x::Vector{Float64})
        TIs = x[1:2:end]
        TRs = x[2:2:end]
        return cr_fleet_objective(T1s, TIs, TRs; Npe = Npe)
    end
    lo = repeat([0.01, 0.5],  n_blocks)
    hi = repeat([3.0,  5.0],  n_blocks)
    # Multi-start with 100 random initial schedules
    best = (x = nothing, L = Inf)
    for _ in 1:100
        x0 = lo .+ rand(2n_blocks) .* (hi .- lo)
        # Project to budget
        # ...
        result = optimize(f, lo, hi, x0, Fminbox(NelderMead()),
                            Optim.Options(iterations = 200))
        if result.minimum < best.L
            best = (x = result.minimizer, L = result.minimum)
        end
    end
    return ...
end
```

### 12.2 Tier 2 — Replace the forward model (~1–2 days)

The bigger gain. F1+ is correct *under perfect spoiling* — for `T2 = 20 ms ≪ TR − TI`, fine; for real tissue T2, breaks (`E2_4_PLAN.md` §2.5). Replacement options:

**Option A — In-house EPG.** Implement the EPG forward model per `E2_4_PLAN.md` §2.5.2 (~70 LOC). Use `transient_epg_at_excite_npe` instead of `transient_mz_at_excite_npe` in `cr_T1_variance`. Keep our solver. Probably the right move regardless — F1+ has known limits even on the simulator.

**Option B — Use `MRIReco.jl`'s EPG.** Comes with a tested EPG implementation. Imports cleanly into Julia. May be slower per call than a hand-rolled inner loop. ~30 min to integrate, *if* we accept their conventions on RF phase, T2*, etc.

**Option C — `MRzero` (Python).** Differentiable Bloch / PDG simulator. Forward model is more accurate than EPG (PDG handles arbitrary RF, slice profile, etc.). Integration: PyTorch ↔ Julia bridge gets messy; CR-opt would need to call Python from Julia. Heavier infrastructure.

**Recommendation: Option A.** EPG is 70 LOC, well-documented, no new deps. Worth doing. **Option C is overkill for this baseline** — MRzero shines for end-to-end-differentiable sequence design (where you backprop into RF waveforms), not for static CR-opt.

### 12.3 Tier 3 — Replace the whole CR-opt pipeline with a published reference (~3–5 days)

The most thorough version. **Asslander et al. (Magn Reson Med 2019)** "Optimized quantification of spin relaxation times in the hybrid state" published a CRB-driven sequence design with a Julia / MATLAB reference implementation. The conceptual mapping:

| Our component | Asslander equivalent |
|---|---|
| `cr_T1_variance` | `crb_t1_t2(...)` (their Fisher-info matrix builder) |
| F1+ forward | EPG forward (their `epg_simulate`) |
| `cr_optimize` | Their gradient-descent + Adam + projection |
| Schedule param | Their RF-train (FA per TR) plus TI / TR per block |

Their formulation also handles **T2 estimation jointly with T1**, and supports MR-fingerprinting-style sequences (varying FA across the schedule, which our env's simplified-action mode doesn't expose). To get a like-for-like comparison:

1. Lift our action space to expose FA per block (currently fixed at 90° in simplified-action).
2. Use their CRB optimiser (open-source, published).
3. Run the resulting schedule through our env (or theirs).

That's a real chunk of work but produces a publication-quality CR-opt anchor that's directly comparable to the MR-fingerprinting literature. Worth doing if we want to claim "RL beats SOTA fixed schedules" in Ch4 — currently we can only claim "RL ties our in-house CR-opt".

**Recommendation:** Defer to a stretch goal. The §13 result already gives us "V5 ties CR-opt within 6 %" which is publishable; trading 5 days of integration to refine that to "V5 ties Asslander-style CR-opt" is a polish move, not a structural change.

---

## 13. Can we fully replace our fitter? — Yes, but with a different cost-benefit than CR-opt

**Important distinction.** "CR-opt solver" runs *once per experiment* (computes a schedule, used as a baseline). The **fitter** (`fit_t1_generalized_ir` in `src/fitting/fits.jl`) runs *every block of every training episode* — ~14 spheres × ~8 blocks × ~25 000 episodes ≈ 2.8 million calls per V5 training run. The constraints are very different.

### 13.1 What "the fitter" is doing

For each sphere, after the $k$-th block, the fitter solves:

$$
(T1_j^*, A_j^*) \;=\; \arg\min_{T1, A}\;\sum_{k'=1}^{k} \bigl(m_{k'} \;-\; A \cdot \tilde m(T1; \mathrm{TI}_{k'}, \mathrm{TR}_{k'})\bigr)^2
$$

via Levenberg–Marquardt with a 200-point T1 grid as initialiser. Returns $(T1^*, A^*)$ plus a $\sigma_{T1}$ from the Jacobian (asymptotic CRB at the LM minimum). Currently called from `_e2_update_t1_estimates!` in `src/rl/e2.jl:273`.

Replacing it means swapping out:
1. The forward model (F1+ → EPG / PDG)
2. The estimator (LM + grid init → ?)
3. The σ method (asymptotic → profile-likelihood, per `E2_5_PLAN.md` §3)

These are independent. Profile-likelihood σ is already planned for E2.5; the other two are bigger.

### 13.2 Tier-by-tier, what replaces what

**Tier 1 — Fitter optimiser only.** Replace LM with `Optim.jl`'s `LevenbergMarquardt` (essentially identical) or `LsqFit.jl`'s `curve_fit` (LM with better stopping criteria). No quality change; ~10 min code reduction. Not worth it.

**Tier 2 — Forward model in the fitter (the right move).** Replace `transient_mz_at_excite_npe` with EPG (in-house Option A from §12.2). Mechanical change in `fit_t1_generalized_ir`: just call the new function. The fit quality improves on real-T2 tissue *if* we're sim-to-real testing; on the env at T2 = 20 ms it's effectively unchanged because spoiling holds.

**Estimated effort:** ~1 day if we do it carefully. Same EPG code that goes into Tier 2 of §12.2 can be reused here.

**Tier 3 — Different estimator entirely.** Replace LM with **dictionary matching** (the standard MRF approach):

1. Pre-compute a dictionary $\{S_k(T1_d, A_d = 1) : T1_d \in \text{grid}, k = 1..n\}$ for all blocks.
2. After each block, normalise the measured signal vector and match against the dictionary by inner product or correlation.
3. Estimated $T1^* = T1_{d^*}$ where $d^*$ is the best-matching dictionary entry.

**Pros:** No iterative optimisation, sub-microsecond per fit, robust to multimodal SSE (the dictionary trivially enumerates basins). Used in all production MRF systems (Ma et al. Nature 2013, Asslander follow-ups).

**Cons:** Estimate quantised by dictionary grid (mitigated by post-match interpolation). Requires dictionary regeneration if the schedule's $\mathrm{TI}/\mathrm{TR}$ pattern changes — fine for fixed schedules, awkward for RL where the agent picks $\mathrm{TI}$ each block (the dictionary depends on the schedule, so it'd need on-the-fly reconstruction; sub-second per block but a non-trivial engineering change).

**For our env: probably keep LM, swap forward model.** The "agent picks TI per block" use case maps poorly to dictionary matching. Stick with LM, replace F1+ with EPG, add profile-likelihood σ. That's the principled fitter upgrade.

### 13.3 What we can't easily replace from off-the-shelf code

- **Per-sphere fits driven by spatial ROI extraction.** The env extracts each sphere's signal from the IFFT'd magnitude image at the sphere's centre pixel (`src/rl/e2.jl:283-284`). External fitters (MRzero, Asslander) typically operate on already-extracted signal vectors, not on raw images. Our pipeline `simulate → IFFT → ROI → fit` is bespoke. Replacing the fitter doesn't break this — it slots in at the last step.
- **Real-time performance constraint.** Our LM call is ~1 ms per sphere per block. A PyTorch / MRzero call has ~10 ms overhead per inference even before the model runs. For 2.8M calls per training run, that's a ~10× wall-clock cost. Real-time RL training basically requires the fitter to be Julia-native and fast.
- **σ method's effect on policy obs channel.** The fitter's σ feeds the policy's observation. Changing σ method changes obs distribution, which changes what the policy learns. Profile-likelihood σ specifically (E2_5_PLAN.md §3) is the right replacement; bootstrap σ is too slow per call.

### 13.4 What I'd actually recommend, ranked by impact

| Change | Cost | Benefit | Recommendation |
|---|---|---|---|
| **Profile-likelihood σ** (`E2_5_PLAN.md` §3) | 1 day | Honest σ on multimodal-SSE failures (idx 11–12); enables σ-channel claim in Ch4 | **Do** |
| **In-house EPG forward model** (replace F1+) | 1–2 days | More accurate physics; sets up sim-to-real conversation | **Do for E3+** |
| **Dictionary-matching fitter (MRF-style)** | 2–3 days | Robust to multimodal SSE; standard in field | Skip — incompatible with RL action space |
| **Replace CR-opt solver with Optim.jl** | 1 hour | Tighter optimiser convergence | **Do during E2.5 Phase 2** |
| **Use Asslander's CRB tool** | 3–5 days | Publication-quality CR baseline | Stretch |
| **Use MRzero forward model** | 3–5 days | EPG-equivalent or better physics | Skip — overkill for sim-only |

The strongest move-fast-and-keep-quality plan: **profile-likelihood σ + in-house EPG + Optim.jl for CR-opt**. ~3 days total. Replacements where they pay off, in-house where the integration cost dominates the gain. This gets us a publishable Ch4 without a deep dive into the broader MR-fingerprinting toolchain.

The full-replacement plan (Asslander + MRzero) is a 1–2 week investment that turns this from "FYP with an in-house simulator" into "a benchmark-quality contribution to the MRF / RL-MR community". Out of scope for the FYP timeline but the right move if continuing post-thesis.

---

## 14. Why use `abs()` at all? — clinical convention vs information

The previous section (multimodal SSE) starts from the signal model

$$
S(T1, A;\, \mathrm{TI}) \;=\; A \cdot \bigl|\;1 \;-\; 2\,e^{-\mathrm{TI}/T1}\;\bigr|
$$

and shows the `|·|` is what creates the two-basin ambiguity. So **why are we taking the magnitude in the first place?** The sign of `1 − 2·exp(−TI/T1)` carries genuine T1 information — it tells you whether you're pre-null (still inverted, sign negative) or post-null (recovered, sign positive). Throwing it away costs us a useful discriminator.

### 14.1 Where the abs() comes from

Two layers of "magnitude" in the pipeline (`src/rl/e2.jl:255-265`):

1. **The reconstruction.** After `simulate(phantom, seq, scanner)` returns a complex k-space `ksp::Matrix{ComplexF32}`, we do
   ```julia
   ksp .+= σ × (randn() + im × randn())          # complex Gaussian noise
   image = abs.(ifft(ksp))                        # magnitude image
   ```
   The IFFT produces a complex image (real and imaginary parts both meaningful for a single inverted signal); `abs.(…)` collapses it to a non-negative real. **This is the abs() that creates the multimodal SSE.**

2. **The signal model in the fitter.** `transient_mz_at_excite_npe` returns the analytic predicted Mz which can be negative (signed). We compare `abs(prediction)` to the (already-magnitude) measurement. So the fitter's `|·|` is downstream of and matched to the reconstruction's `|·|` — both come from the same convention.

### 14.2 Why the convention exists in real MRI

Magnitude reconstruction is the **clinical default** for three converging reasons:

- **Arbitrary receive-phase offsets.** Receive coils on real scanners have phase offsets that depend on B0 inhomogeneity, eddy currents, the patient's geometry, gradient warm-up, and other things you can't predict. The complex IFFT image has a baseline phase that's *not* zero everywhere — it's a smoothly-varying field across the FOV.
- **Phase is unstable across acquisitions.** Even the same voxel in two consecutive blocks can have different phase because of small drifts in B0 or motion-induced phase. Phase-sensitive reconstruction would require either a reference scan or a phase-unwrapping algorithm to stitch together signed values across blocks.
- **Display convention.** Radiologists read magnitude images. `|S|` is what shows up on the console, on the PACS server, in the published images.

So clinical pipelines reconstruct magnitude images, fit T1 to magnitude data, and accept the multimodal SSE penalty in exchange for robustness to phase artefacts.

### 14.3 But this is a digital twin — we don't have real-scanner phase noise

In our simulator, **we know the ground-truth phase exactly**. KomaMRI returns the complex magnetisation deterministically; the only phase noise is the complex Gaussian we add ourselves at line 262, which has known statistics. There is no B0 inhomogeneity, no eddy currents, no patient motion — it's a clean digital phantom.

So we *could* preserve the sign and use a phase-sensitive reconstruction. Concretely:

- Replace `image = abs.(ifft(ksp))` with `image = real.(ifft(ksp))` (assuming the simulation's reference phase aligns the inverted signal along the real axis — which it does for our IR-SE setup with α_exc = π/2).
- Replace the fitter's forward model `abs(transient_mz_at_excite_npe(...))` with the signed `transient_mz_at_excite_npe(...)`.
- The signed signal model `S = A · (1 − 2·exp(−TI/T1))` is **monotonic in T1** (it goes from −A as T1 → 0 to +A as T1 → ∞). One measurement uniquely identifies T1 (modulo noise).

### 14.4 What changes if we drop abs() in the digital twin

| Property | With abs() (current) | Without abs() (phase-sensitive sim) |
|---|---|---|
| SSE landscape | Multimodal (multiple basins) | Unimodal (single basin per sphere) |
| LM convergence | Can land in wrong basin | Converges to truth |
| Asymptotic σ | Over-confident on multimodal failures | Honestly tight |
| Profile-likelihood σ | Wide on multimodal cells | Same as asymptotic (no second basin) |
| Short-T1 fit quality | 400–800 % MAPE for T1 < 0.05 s | Probably 50–150 % MAPE — the signal is informative once you can see the sign |
| Sim-to-real fidelity | Direct match to clinical magnitude pipeline | **Gap** — no real scanner does this without a phase-calibration step |

**The crucial line**: dropping abs() in simulation is *almost certainly* a 5–10× MAPE improvement on short-T1 spheres in §10–§13. It's the single biggest physics lever we haven't pulled.

### 14.5 What the sim-to-real story would say

The honest reframing if we keep abs() is: *"We model the standard clinical magnitude pipeline. The multimodal SSE artefact is a property of magnitude T1 mapping, not of our simulator."* This is true and defensible — published clinical IR-FLASH protocols all suffer from this on short-T1 fluids and rely on either (a) avoiding short-T1 species in the imaging volume, (b) using phase-sensitive reconstruction with a phase reference scan, or (c) accepting wider per-voxel uncertainty in fast-relaxing tissues.

If we drop abs() in simulation, we should **also** add a phase-sensitive reconstruction option as a future-work hook. Real-scanner work would need a B0 map + phase-unwrap pre-step before the fitter, which is its own infrastructure.

### 14.6 Phase-sensitive IR — what real systems actually do

When clinical IR sequences need short-T1 accuracy (e.g. fat suppression, contrast-agent T1 mapping), they use **phase-sensitive inversion recovery (PSIR)**:

1. Acquire a reference scan (long TR, no inversion) to map the receive-coil phase per voxel.
2. Subtract that phase from each subsequent block's complex image.
3. Take the real part of the phase-corrected image.

Result: signed signal that crosses zero at the null, no abs() ambiguity. Roughly doubles scan time (the reference adds overhead) but resolves the short-T1 multimodal problem on real hardware.

In our simulator, the equivalent is trivial — we know the reference phase analytically, no scan needed. So **sim-PSIR is roughly free**; we just haven't done it.

### 14.7 Recommendation

The honest engineering plan:

| Tier | Change | Effort | Effect |
|---|---|---|---|
| **A** | Add `phase_sensitive::Bool = false` env kwarg; when true, use `real.(ifft(ksp))` and signed fitter | 1 h code + 1 h test | 5–10× MAPE improvement on short-T1; no second basin; honest σ |
| **B** | Run V5's policy under phase-sensitive env (no retrain) — sanity-check the headline | 30 min compute | Confirms most of §13's short-T1 failure is the abs(), not the policy |
| **C** | Retrain a V10 under phase-sensitive env | ~5 h compute | Establishes the achievable MAPE under cleaner physics; sets a new lower bound |
| **D** | Document phase-sensitive vs magnitude as a sim-vs-clinical sensitivity in §12 | 30 min | Honest sim-to-real caveat |

For Ch4, doing (A)+(B)+(D) is high-value and cheap (~3 h total). It frames the §13 result as "RL achieves CR-opt-comparable performance under the clinical magnitude convention; under phase-sensitive simulation the floor is much lower for everyone, including non-RL baselines, and short-T1 multimodal-SSE failures vanish". That's a much cleaner Ch4 limitations paragraph than "the action set is too small for short-T1" — because the action set was never the problem; the recon convention was.

(C) is optional — it's an *additional* result, not a corrective one. The key insight is that the abs() in our model is a **simulation choice**, not a physics requirement, and we've been silently importing a real-world limitation that doesn't apply in the simulator.

### 14.8 Why we haven't already done this

Honest answer: I didn't think to question it. The abs() is in the standard MR-fitting recipe (`src/fitting/fits.jl::generalized_ir_signal:73-95` predates this work), and the env's reconstruction follows the convention without comment. The §9.9.4 multimodal-SSE failure mode is documented as a fitter problem; profile-likelihood σ is the planned fix — *for the magnitude regime*. But the deeper fix is "don't go into the magnitude regime in the first place when you don't have to". Worth doing in E2.5 alongside profile-likelihood σ; the two changes are complementary (one stops bad basins, the other reports honestly when they exist).

---

## 15. Multimodal SSE in detail — what "the wrong basin" actually looks like

The previous sections refer to "multimodal SSE" repeatedly. This section walks through it concretely, with worked numbers. The whole problem comes from **one measurement of `|S|` not uniquely identifying `T1`** — exactly the abs() consequence §14 just discussed, but from the fitter's side rather than the recon's.

### 15.1 The signal model and why two T1s give the same `|S|`

For amplitude $A$ and tissue $T1$, the IR-SE measurement at inversion time TI is (simplified F1+ form):

$$
S(T1;\, \mathrm{TI}) \;=\; A \cdot \bigl|\;1 \;-\; 2\,e^{-\mathrm{TI}/T1}\;\bigr|
$$

The argument `1 − 2·exp(−TI/T1)` is **negative for short TI** (magnetisation still inverted), **zero at TI = T1·ln 2** (null point), and **positive for long TI** (recovered). The `|·|` discards the sign, giving the magnitude image.

Pick TI = 0.05 s and compute |S|/A for several T1 values:

| T1 (s) | TI/T1 | exp(−TI/T1) | 1 − 2·exp | **\|S\|/A** |
|---:|---:|---:|---:|---:|
| 0.005 | 10.0 | 0.000045 | 0.9999 | **0.9999** |
| 0.01 | 5.0 | 0.0067 | 0.9866 | 0.987 |
| 0.05 | 1.0 | 0.368 | 0.264 | 0.264 |
| 0.072 | 0.69 | 0.500 | 0.000 | **0.000** ← null |
| 0.10 | 0.5 | 0.607 | −0.213 | 0.213 |
| 0.50 | 0.1 | 0.905 | −0.810 | 0.810 |
| 5.0 | 0.01 | 0.990 | −0.980 | 0.980 |

Notice **|S|/A is not monotonic in T1**. It hits zero at the null (T1 ≈ 0.072 s) and rises on *both sides*. So if you measure `|S|/A = 0.81` at TI = 0.05 s, the data is consistent with **two different T1 values**: T1 ≈ 0.5 s (post-null, "long arm") *or* T1 ≈ 0.013 s (pre-null, "short arm"). One measurement cannot distinguish them.

### 15.2 From signal ambiguity to SSE multimodality

The fitter minimises:

$$
\mathrm{SSE}(T1, A) \;=\; \sum_{k=1}^n \bigl(m_k \;-\; S(T1, A;\, \mathrm{TI}_k)\bigr)^2
$$

For a fixed candidate $T1$, the optimal $A$ is closed-form (least squares with one variable), so define the **profile SSE**:

$$
\mathrm{SSE}_{\text{prof}}(T1) \;=\; \min_A \mathrm{SSE}(T1, A)
$$

This is a 1D function of $T1$. **Multimodal SSE** means this curve has **multiple local minima** (multiple "basins"). Each basin is a candidate value of $T1$ that fits the data nearly as well as the truth.

### 15.3 Worked example — short-T1 sphere with all-saturated TIs

Suppose true $T1 = 0.023$ s, $A = 1$, and the agent picked TIs = [0.5, 1.0, 1.5, 2.0] s (all very long compared to $T1$). The true signals:

```
TI = 0.5:  TI/T1 = 21.7,  |S|/A = |1 − 2·exp(−21.7)| ≈ 1.000
TI = 1.0:  ≈ 1.000
TI = 1.5:  ≈ 1.000
TI = 2.0:  ≈ 1.000
```

All four measurements are essentially $A$. With small noise, we measure roughly `m ≈ [1.00, 1.00, 1.00, 1.00] · A`.

Now scan candidate $T1 \in [0.001, 5.0]$ s and compute SSE_prof:

| Candidate $T1$ | Predicted \|S\|/A at the 4 TIs | SSE_prof (with $A^*$ optimised) |
|---:|---|---:|
| 0.001 s | [1, 1, 1, 1] | ≈ 0 |
| 0.01 s | [1, 1, 1, 1] | ≈ 0 |
| 0.023 s (truth) | [1, 1, 1, 1] | ≈ 0 |
| 0.05 s | [1, 1, 1, 1] | ≈ 0 |
| 0.1 s | [0.99, 1.0, 1.0, 1.0] | tiny |
| 0.5 s | [0.81, 0.96, 0.99, 0.998] | small |
| 1.0 s | [0.39, 0.81, 0.93, 0.97] | larger |
| 2.0 s | [0.22, 0.39, 0.53, 0.63] | very large |

**The SSE is roughly zero across an entire range of short T1s** (anything ≪ 0.5 s) because they all predict $|S| \approx A$. So:
- The "basin" isn't a sharp valley — it's a wide flat plateau spanning all $T1 \ll 0.5$ s.
- Any LM init in [0.001, 0.1] s finds a fit with SSE ≈ 0.
- The reported $T1^*$ might be 0.001 s (very wrong), 0.05 s (close-ish), or 0.023 s (truth) — all fit the data essentially equally well.

This is the **trivial flat-basin failure**: the data doesn't constrain $T1$ at all in the short-T1 region. No fitter can fix this; the action set didn't put any TI in the informative window. *This is the §11.4 "structural unreachability" version of the problem — it's a budget/action problem, not really a multimodality problem.*

### 15.4 Genuinely multimodal example — abs() flip + sparse data

Now suppose the agent picked TIs = [0.05, 0.5, 1.5] s and true $T1 = 0.1$ s. True signals:

```
TI = 0.05, T1 = 0.1: |S|/A = |1 − 2·exp(−0.5)| = |−0.213| = 0.213
TI = 0.5,  T1 = 0.1: |S|/A = |1 − 2·exp(−5.0)|   ≈ 0.987
TI = 1.5,  T1 = 0.1: |S|/A ≈ 1.000
```

Measured `m ≈ [0.213, 0.987, 1.000]·A`. Now scan candidate $T1$:

| Candidate $T1$ | Predicted \|S\|/A at the 3 TIs | residuals (m − pred) |
|---:|---:|---|
| 0.013 s ("short arm twin") | [0.022, 1.000, 1.000] | [+0.19, −0.01, 0.00] |
| 0.030 s | [0.81, 1.00, 1.00] | [−0.60, −0.01, 0.00] |
| 0.072 s (null at TI = 0.05) | [0.000, 0.999, 1.00] | [+0.21, −0.01, 0.00] |
| **0.10 s (truth)** | [0.213, 0.987, 1.000] | [0, 0, 0] ← min |
| 0.20 s | [0.61, 0.92, 0.99] | [−0.40, +0.07, +0.01] |
| 0.50 s | [0.81, 0.27, 0.95] | [−0.60, +0.72, +0.05] |

The residual at TI = 0.05 s for $T1 = 0.013$ s is `0.213 − 0.022 = +0.19` — same magnitude as for $T1 = 0.072$ s (the null hypothesis). Both produce a TI = 0.05 residual ~0.2, vs the truth at $T1 = 0.10$ with residual 0.

So the SSE_prof landscape over $T1$ has roughly:
- A sharp minimum at $T1 = 0.10$ s (truth) with SSE ≈ 0
- A shallower local minimum near the null point $T1 \approx 0.072$ s
- A second shallow minimum at the "short arm twin" $T1 \approx 0.013$ s

**Three minima.** Adding noise to the measurements can flatten the truth's advantage, leaving the LM's grid-init scan to pick whichever it lands in. **That's a multimodal SSE basin.** The §13 short-T1 failure is V5 spheres landing in the 0.07 / 0.013 s wrong basin instead of the 0.10 s truth basin.

### 15.5 What LM does (and what goes wrong)

Levenberg–Marquardt is a **local, gradient-following optimiser**. It needs an initialisation, then walks downhill until the gradient is small.

Our `fit_t1_generalized_ir` initialises by:

1. Evaluating SSE_prof on a 200-point T1 grid spanning [0.01, 3.0] s.
2. Picking the grid point with lowest SSE.
3. Running LM from that point until convergence.

If the grid samples the right basin (truth at $T1 = 0.10$ s) and the grid SSE there is the lowest, LM converges to truth. ✓

But if noise has perturbed measurements such that the wrong basin's SSE is slightly lower at the grid point closest to it, LM converges to *that* basin instead. ✗ — and now $T1^*$ is e.g. 0.072 s when truth is 0.10 s.

The asymptotic σ from $(J^T J)^{-1}$ evaluated *at the wrong basin's minimum* is small (the basin is narrow there, gradient is well-defined, $J^T J$ is non-singular). **σ confidently says the fit is good — but it's good *relative to that basin*, not relative to the global landscape.** This is the over-confidence we see in §9.9.4 of `EXPERT_REPORT_E2_4.md`.

### 15.6 What profile-likelihood σ does (the §3 fix in `E2_5_PLAN.md`)

Profile-likelihood doesn't trust the local quadratic approximation. It computes SSE_prof($T1$) on a candidate grid and asks: **"what's the set of $T1$ values whose SSE is within a statistical threshold of the minimum?"**

For our 3-basin example, suppose SSE_min = 0.001 (at truth $T1 = 0.10$) and the F-test threshold is `SSE_min · 1.5 = 0.0015`. The set of $T1$s with SSE_prof ≤ 0.0015 is:
- A narrow interval around $T1 = 0.10$ s
- *And possibly* a second narrow interval around $T1 = 0.072$ s (if its basin is shallow enough)
- *And possibly* a third around $T1 = 0.013$ s

The 1σ width is the **total extent** from leftmost to rightmost $T1$ in the union — including across disconnected basins. Concretely if the three basins are at $T1 \in \{0.013, 0.072, 0.10\}$, profile-likelihood σ would report $\sigma_{T1} \approx (0.10 − 0.013)/2 = 0.044$ s — i.e., honestly large, spanning all the basins.

Asymptotic σ at the truth basin alone might be 0.005 s — saying "I'm 5 % uncertain" when the actual ambiguity across basins is 50 %.

**Crucially: profile-likelihood σ doesn't fix the wrong-basin problem (LM still picks one basin); it just *reports the ambiguity honestly* via wider σ.** That's the §13.5 / §9.9.4 finding: σ is the messenger, not the cure. The cure for *fitting* is more informative TIs at sub-50 ms range; the cure for *reporting* is profile-likelihood σ. The cure for *both* is dropping abs() (§14).

### 15.7 Why this dominates short-T1 spheres specifically

For long-T1 spheres ($T1 \approx 1$ s) and TIs in [0.1, 3.0] s, the action set straddles the null (TI = 0.69 s) and the informative window (TI ∈ [0.3, 2] s). The data has multiple strongly-discriminating measurements → SSE_prof has one sharp minimum → no multimodality → asymptotic σ is honest.

For short-T1 spheres ($T1 \approx 0.05$ s):
- Most action-set TIs are ≫ $T1$ → saturated → uninformative
- A few TIs near $T1$ (0.01–0.1 s) are informative *individually* but combined with many saturated points the SSE landscape develops the multi-basin shape above
- The abs()'s pre/post-null ambiguity is most visible here because TI = 0.05 s sits *near* the null for several short-T1 candidates simultaneously

So the multimodal-SSE failure mode is concentrated on short-T1 spheres in our env. **It's also the artefact §14's phase-sensitive reconstruction would resolve at the source** — the signed signal model is monotonic in $T1$, so a unique $T1$ produces each measurement, no two-basin ambiguity, no multimodal SSE.

### 15.8 Mental picture

The clearest way to think about it: imagine plotting SSE on the y-axis against $T1$ (log-scale x-axis) from 0.001 to 3 s.

- **Long-T1 sphere with diverse TIs**: one sharp dip at truth — like a clean parabola. LM falls into it from anywhere reasonable.
- **Short-T1 sphere with mostly-saturated data plus one near-null TI**: a wide flat plateau across short T1s with maybe a dip near the null and one near the short-arm twin — multiple "valleys" of comparable depth.

LM is dropped from a height (the grid scan's lowest point) and falls into whichever valley is nearest its drop point. Profile-likelihood is a hiker with a tape measure who walks the whole landscape and reports "the data is consistent with anywhere in this whole region". Phase-sensitive reconstruction (§14) flattens the abs()-induced valleys and leaves only the truth basin standing.

# Parameter fits: `src/fitting/fits.jl`

Every experiment in this report — the E0 conventional baseline, the E1 single-voxel RL agent, and the E2 spatially-resolved agent — ultimately reports a number: an estimate of T1 (or T2) extracted from a small set of magnitudes. `fits.jl` is the module that turns measured signal magnitudes into those quantitative estimates. It is deliberately self-contained (no external optimiser dependency) and is shared between the baselines and the RL environment, so that any apparent gain by the agent is measured on the **same estimator** as the baseline.

The module exposes four estimators / forward models:

| Name | What it fits | Where used |
|---|---|---|
| `fit_t2_se`                  | T2 from a multi-TE spin-echo magnitude curve, closed-form in log space | E0 T2 baseline |
| `fit_t1_ir`                  | T1 from a 3-parameter magnitude-IR model `|A − B·exp(−TI/T1)|` via a log grid + closed-form (A,B) | E0 T1 baseline |
| `steady_state_mz_at_excite`  | Closed-form steady-state Mz for an IR-prepped block at arbitrary (θ_inv, α_exc, TI, TR) | Forward model for "infinitely many shots" |
| `transient_mz_at_excite_npe` | Closed-form **mean** Mz over the first `Npe` shots of an IR block starting from M0=1 | Forward model that matches what the E2 sequence actually runs |
| `fit_t1_generalized_ir`      | T1 from (TI, α_inv, α_exc, TR, |S|) tuples under either of the two forward models above, with a choice of uncertainty estimator | E1 and E2 RL environments; later E0 variants |

The rest of this section first derives every formula from the Bloch equations, then walks through the three non-trivial pieces in turn — the E0 T1 fit, the closed-form forward models, and the generalised IR fit used by the RL environment — and finally describes the test suite that pins them.

---

## 0. Derivation from the Bloch equations

Every fit formula in this module is a solution of the Bloch equations under two simplifying assumptions — instantaneous ("hard") RF pulses and perfect transverse spoiling between blocks. This section derives them from scratch so that the closed forms in §2 are not magic.

### 0.1 The Bloch equations and their free-precession solution

In a frame rotating at the Larmor frequency, with relaxation times T1 (longitudinal) and T2 (transverse) and equilibrium magnetisation M0 along +z, the Bloch equations are

$$
\frac{dM_z}{dt} = -\frac{M_z - M_0}{T_1}
\qquad\qquad
\frac{dM_{xy}}{dt} = -\frac{M_{xy}}{T_2}\;\;(+\, i\,\Delta\omega\, M_{xy}\ \text{off-resonance})
$$

Between RF pulses (free precession) the two components decouple and each is a first-order linear ODE with a constant solution:

$$
M_z(t) = M_0 + \bigl(M_z(0) - M_0\bigr)\,e^{-t/T_1}
\qquad\text{(longitudinal recovery)}
$$

$$
|M_{xy}(t)| = |M_{xy}(0)|\,e^{-t/T_2}
\qquad\text{(transverse decay)}
$$

These two exponentials are the entire physical content of the fits. Everything else is bookkeeping: what each RF pulse does to the magnetisation vector, and which component the ADC samples.

### 0.2 RF pulses as rotations; the spoiling assumption

A hard RF pulse of flip angle `α` is modelled as an instantaneous rotation of the magnetisation vector by `α` about an axis in the transverse plane. Two consequences are all we need:

- It maps the longitudinal component to `Mz → cos(α)·Mz` (the part that stays longitudinal), and tips `sin(α)·Mz` into the transverse plane.
- The transverse signal the receiver coil measures is proportional to the transverse magnitude immediately after excitation, i.e. `S ∝ sin(α_exc)·|Mz⁻|` where `Mz⁻` is the longitudinal magnetisation *just before* the excitation pulse.

**Spoiling assumption.** All sequences here either crush residual transverse magnetisation with a gradient spoiler or operate at a TE/TR where it has decayed (short T2). We therefore assume the transverse component is zero at the start of every block, so only the longitudinal scalar `Mz` is carried from one pulse to the next. This collapses the full 3-vector Bloch dynamics to a scalar recurrence in `Mz` — the reason the forward models are one-liners rather than matrix exponentials.

### 0.3 The T2 fit (`fit_t2_se`)

A spin echo is `90° → TE/2 → 180° → TE/2 → echo`. The 90° tips M0 fully into the transverse plane; over the first TE/2 the magnetisation dephases under both irreversible T2 and reversible static dephasing (T2′ — field inhomogeneity, susceptibility). The 180° conjugates the accumulated phase, so the reversible part **refocuses** at the echo and only the irreversible T2 decay survives:

$$
|S(\mathrm{TE})| = M_0\,e^{-\mathrm{TE}/T_2}
$$

Taking logs gives $\log|S| = \log M_0 - \mathrm{TE}/T_2$, a straight line in TE — hence the closed-form log-linear regression in §1. The 180° refocus is what lets us fit the clean T2 rather than the messy T2*.

### 0.4 The IR fit (`fit_t1_ir` and the TR → ∞ generalised model)

Inversion recovery starts from equilibrium `Mz = M0` and applies a preparation pulse of angle `θ_inv`, leaving `Mz = M0·cos(θ_inv)` (= −M0 for the canonical 180° inversion). The magnetisation then recovers freely for the inversion time TI; substituting `Mz(0) = M0·cos(θ_inv)` into the longitudinal solution of §0.1:

$$
M_z(\mathrm{TI}) = M_0\,\bigl[\,1 - (1 - \cos\theta_{\text{inv}})\,e^{-\mathrm{TI}/T_1}\,\bigr]
$$

The excitation pulse then reads this out: $S \propto \sin(\alpha_{\text{exc}})\,|M_z(\mathrm{TI})|$. For the canonical $\theta_{\text{inv}} = \pi$ this is the textbook $M_0\,(1 - 2\,e^{-\mathrm{TI}/T_1})$, with the magnitude null at $\mathrm{TI}^* = T_1\ln 2$. This is exactly `generalized_ir_signal` in `blocks.jl` (which sets `Mz_after_prep = cos α`) and the TR → ∞ branch of `steady_state_mz_at_excite`. It is valid only when TR ≫ T1 so the spin is back at M0 before the next inversion — the assumption the E0 baseline enforces.

### 0.5 Finite TR: the two-segment recurrence (the §2 forward models)

When TR is comparable to T1, the spin does **not** return to M0 between blocks, so `Mz(TI)` depends on the magnetisation left over from the previous shot. Track `Mz_pre`, the longitudinal magnetisation just before the inversion pulse, and apply §0.1–§0.2 segment by segment over one block (taking M0 = 1):

$$
\begin{aligned}
\text{inversion:} \quad & M_z^{+} = \cos(\theta_{\text{inv}})\,M_z^{\text{pre}} \\
\text{recover over TI:} \quad & M_z^{\mathrm{TI}} = (1 - E_1) + \cos(\theta_{\text{inv}})\,E_1\,M_z^{\text{pre}}
   && E_1 = e^{-\mathrm{TI}/T_1} \\
\text{excitation:} \quad & M_z^{+} = \cos(\alpha_{\text{exc}})\,M_z^{\mathrm{TI}} \\
\text{recover over TR}-\text{TI:} \quad & M_z^{\text{pre}\,\prime} = (1 - E_2) + \cos(\alpha_{\text{exc}})\,E_2\,M_z^{\mathrm{TI}}
   && E_2 = e^{-(\mathrm{TR}-\mathrm{TI})/T_1}
\end{aligned}
$$

This pair of update rules is the whole forward model. Two regimes solve it:

- **Steady state** (`steady_state_mz_at_excite`): the sequence is repeated indefinitely, so $M_z^{\text{pre}\,\prime} = M_z^{\text{pre}}$. Setting the two equal and solving the resulting linear equation for $M_z^{\text{pre}}$ gives the closed form in §2,

  $$
  M_z^{\text{pre}} = \frac{(1 - E_2) + \cos(\alpha_{\text{exc}})\,E_2\,(1 - E_1)}{1 - \cos(\theta_{\text{inv}})\cos(\alpha_{\text{exc}})\,E_1 E_2}
  $$

  valid in the $N_{\text{pe}} \to \infty$ limit.
- **Transient** (`transient_mz_at_excite_npe`): the agent only runs `Npe` shots from a cold start `Mz_pre[1] = 1`, and the IFFT reconstruction averages the shot magnetisations. So the relevant quantity is the *mean* of `Mz_at_TI` over the first `Npe` shots — computed by iterating the recurrence `Npe` times rather than solving the fixed point. The two regimes meet as `Npe → ∞`, and both reduce to §0.4 as `TR → ∞` (where `E2 → 0`, decoupling the shots).

### 0.6 From `Mz` to measured magnitude

The receiver sees $S_{\text{obs}} = \sin(\alpha_{\text{exc}})\,A\,M_z^{\text{exc}}$, where $A$ lumps proton density, coil gain, and the constant T2/T2* echo attenuation (constant because TE is fixed within a fit). The generalised fit therefore (i) expects the caller to divide out `sin(α_exc)` per block — see §3 — and (ii) fits the single scalar `A` in closed form, leaving T1 as the only non-linear parameter to grid-search. The amplitude `A` absorbing the TE-dependent decay is why the T1 fits never need to know T2.

The "constant because TE is fixed" caveat is load-bearing. The moment a sequence **varies TE** between samples — a multi-echo / IR-TSE readout — the factor

$$
S_{\text{obs}} = \sin(\alpha_{\text{exc}})\,A\,M_z^{\text{exc}}(T_1)\,e^{-\mathrm{TE}/T_2}
$$

can no longer be folded into a single $A$, and T2 becomes identifiable from the data. That is exactly the regime the joint fit in §3 (`fit_t1_t2_generalized_ir`) targets, and the reason the T1-only fit must hold TE constant.

---

## 1. The E0 fits: `fit_t2_se` and `fit_t1_ir`

The two original baseline fits assume canonical IR / SE acquisitions with α_inv = π, α_exc = π/2, and TR ≫ T1, so the steady-state machinery in §2 is not needed.

### `fit_t2_se(TEs, mags)`

The spin-echo magnitude is `|S| = S0·exp(−TE/T2)`. Taking the logarithm linearises it: `log|S| = log S0 − TE/T2`. That is a single linear regression, so the fit is closed-form — we stack the equations as `X β = y` with `X = [1, −TE]` and recover `S0 = exp(β₁)`, `T2 = 1/β₂`. The implementation drops non-positive samples (their log is undefined) and rejects "non-decaying" data, where the recovered slope is the wrong sign, with a hard error. The reported residual is the RMS of the **log-space** residuals — small and dimensionless, suitable for asserting in tests but not directly comparable to the magnitude noise.

### `fit_t1_ir(TIs, mags)`

The inversion-recovery model is `|S| = |A − B·exp(−TI/T1)|`. The absolute value makes this **not** linearisable: the magnitude operator hides the sign flip at the null time (`TI* = T1·ln(B/A)`), so we don't know a-priori which samples have crossed it. The implementation handles this with a two-level search:

1. **Outer loop** — a log-spaced grid of `T1` candidates (default 400 points over 5 ms to 5 s). T1 enters the model only through `exp(−TI/T1)`, so once T1 is fixed the model is linear in (A, B).
2. **Sign search** — for each candidate T1 we sort the TIs in ascending order and try every "prefix flip": negate the first `k = 0, 1, …, n` measurements before fitting. Because `|A − B·e^{−TI/T1}|` is monotone for sensible (A, B) > 0, the true sign flips at most once and that flip lies at some prefix `k*`. The brute-force scan over `k` is `O(n)` per T1, which is negligible.
3. **Inner regression** — at fixed T1 and a fixed sign assignment, the model is `y = A − B·e`, i.e. a linear regression on the basis `[1, −e]`. Closed-form via `X \ y`.

Whichever `(T1, k)` combination minimises the sum-of-squared residuals wins. The returned `residual` is the RMS in magnitude units.

The estimator is intentionally simple: it has no notion of finite TR, varying α, or noise weighting. Those generalisations live in `fit_t1_generalized_ir` and were introduced when E1 / E2 needed them. The cost is that `fit_t1_ir` is only correct when the acquisition matches its assumed model — the E0 baseline pipeline ensures this by using a long-TR inversion-recovery turbo-spin-echo at α = 180° / 90°.

---

## 2. The forward models: `steady_state_mz_at_excite` and `transient_mz_at_excite_npe`

The E1 and E2 environments do not run canonical IR. They (i) repeat each IR block many times to fill k-space, (ii) let the agent pick **arbitrary** inversion and excitation flip angles, and (iii) operate at TR values comparable to T1 so the magnetisation never fully recovers. The fit therefore needs a forward model that matches what the simulator actually sees.

Two closed-form models cover the two regimes of interest.

### `steady_state_mz_at_excite(T1, TI, TR, θ_inv, α_exc)`

Solves the periodic fixed point of the recurrence

$$
\begin{aligned}
M_z^{\mathrm{TI}} &= (1 - E_1) + \cos(\theta_{\text{inv}})\,E_1\,M_z^{\text{pre}} \\
M_z^{\text{pre}}  &= (1 - E_2) + \cos(\alpha_{\text{exc}})\,E_2\,M_z^{\mathrm{TI}}
\end{aligned}
$$

with $E_1 = e^{-\mathrm{TI}/T_1}$, $E_2 = e^{-(\mathrm{TR}-\mathrm{TI})/T_1}$. Two equations, two unknowns, closed-form solution:

$$
M_z^{\text{pre}} = \frac{(1-E_2) + \cos(\alpha_{\text{exc}})\,E_2\,(1-E_1)}{1 - \cos(\theta_{\text{inv}})\cos(\alpha_{\text{exc}})\,E_1 E_2}
\qquad
M_z^{\mathrm{TI}} = (1-E_1) + \cos(\theta_{\text{inv}})\,E_1\,M_z^{\text{pre}}
$$

The function returns `Mz_at_TI` because that is the longitudinal magnetisation present at the moment the excitation pulse fires; multiplying by `sin(α_exc)` recovers the transverse signal the ADC samples. Two textbook limits are sanity checks that the formula is right:

- **θ_inv = π, α_exc = π/2** (canonical IR, finite TR) reduces to $1 - 2\,e^{-\mathrm{TI}/T_1} + e^{-\mathrm{TR}/T_1}$.
- **TR → ∞** reduces to $1 - (1 - \cos\theta_{\text{inv}})\,e^{-\mathrm{TI}/T_1}$ — the transient form used by `fit_t1_ir`.

A `TR ≤ 0` or `TR = Inf` argument is treated as TR → ∞ to make this the formal generalisation of the E0 model.

### `transient_mz_at_excite_npe(T1, TI, TR, θ_inv, α_exc; Npe)`

The steady-state model is **wrong** for the E2 sequence: the agent only acquires `Npe` shots (one per k-space row), starting from thermal equilibrium `Mz_pre[1] = 1`. The first few shots are nowhere near the fixed point, and the IFFT reconstruction averages the shot magnetisations together — so the relevant quantity is the **mean Mz_at_excite over the first Npe shots from M0 = 1**, not the steady state.

The recurrence is the same as above; we run it forward for `Npe` iterations and average:

```
Mz_pre[1]  = 1
for k = 1..Npe:
    Mz_at_TI[k]  = (1 − E1) + cos(θ_inv)·E1·Mz_pre[k]
    Mz_pre[k+1]  = (1 − E2) + cos(α_exc)·E2·Mz_at_TI[k]
return mean(Mz_at_TI[1..Npe])
```

This is `O(Npe)` arithmetic — comparable in cost to the steady-state form. Three limits double as algebraic sanity checks (and are pinned by tests; see §4):

| Limit | Reduces to |
|---|---|
| `Npe = 1`     | `1 − (1 − cos θ_inv)·exp(−TI/T1)` (single transient shot, TR-blind)         |
| `Npe → ∞`     | `steady_state_mz_at_excite(T1, TI, TR, θ_inv, α_exc)`                        |
| `TR → ∞`      | Every shot resets from M0 = 1, so the mean equals the single-shot transient |

The distinction between the two forward models is not a small effect. With Npe = 8 and TR ≈ T1, the steady-state model is **provably biased** on the data the agent generates — the E2 plan and §19.5.1 of the report quantify this — and the test suite includes a regression test that the steady-state fit misses the truth by ≥10 % on synthetic Npe-shot data while the transient fit lands within 5 %.

---

## 3. The generalised IR fit: `fit_t1_generalized_ir`

This is the workhorse fit for E1 and E2. Inputs are vectors of `(TI, α_inv, |S|)` plus optional per-block `(TR, α_exc, Npe)`. The forward model is dispatched by the `Npe` argument: `nothing` → steady-state, `::Int ≥ 1` → transient ramp from M0 = 1. The fit minimises the sum of squared residuals between observed magnitudes `m` and predicted `|A · Mz_at_excite|` over (T1, A).

### Why a grid scan and not Levenberg–Marquardt

`fit_t1_generalized_ir` does **not** run a non-linear optimiser. It scans a log-spaced grid over T1 (default 300 points) and at each candidate solves for the optimal `A` in closed form. The reason is the same magnitude-ambiguity that bites `fit_t1_ir`: `|A · Mz|` is not differentiable at the null time, the SSE surface is multimodal across T1, and a local optimiser (LM, BFGS) trivially gets stuck in a wrong basin. The grid scan is brute-force but `O(n_grid · n)` and the inner step is a closed-form ratio:

$$
\mathbf{ay} = |M_z^{\text{predicted}}|
\qquad
A = \frac{\mathbf{m}\cdot\mathbf{ay}}{\mathbf{ay}\cdot\mathbf{ay}}
\qquad
\mathrm{SSE} = \lVert A\,\mathbf{ay} - \mathbf{m}\rVert^2
$$

(here $\mathbf{ay}$ is the length-$n$ predicted-magnitude vector; the `signed = true` path drops the $|\cdot|$ — see below.) The grid step costs perhaps 100 μs per fit at default sizes, which is invisible next to the cost of a single Bloch simulation.


### Caller convention: α_exc scaling

The forward models return Mz at the moment of excitation. The transverse signal the scanner produces is `S_obs = sin(α_exc) · A · Mz`. With a single amplitude `A` in the fit, the per-block `sin(α_exc)` factor would alias into a per-block scaling of `Mz` — biasing T1 whenever `α_exc` varies between blocks. The fit therefore expects the caller to **pre-divide** the observed magnitudes by `sin(α_exc)` before passing them in (E2's `_e2_update_t1_estimates!` does this). The α-correction regression test in §4 pins both halves: an uncorrected fit on a mixed-α schedule biases T1, the corrected fit recovers it.

### Phase-sensitive (signed) fit

By default the fit is the magnitude-IR one: it predicts `|Mz|` and compares to `|S|`. Set `signed = true` and the fit predicts the signed `Mz` and compares to a signed measurement — appropriate when phase-sensitive reconstruction has already unwrapped the null-time sign flip. The signed model is monotone in T1, the SSE surface has a single basin, and the asymptotic σ estimator (below) is honest rather than over-confident. The unit tests confirm that on consistently-signed data (all TIs post-null) the magnitude and signed fits agree to 1 %.

### Uncertainty: `T1_sigma`

The RL agents observe not just `T1*` but a per-sphere σ_T1 channel: the running estimator's confidence. Three estimators are available via `sigma_method`:

- `:asymptotic` (default). Cramér–Rao from the local Jacobian at the LM minimum, `Σ ≈ σ²_eff · (J^T J)^{-1}`. Closed form via central-difference Jacobian of `predict` at `(T1*, A*)`. Cheap and smooth but **over-confident on multimodal SSE** — it only sees the basin LM landed in.
- `:profile_likelihood`. The span of T1 values whose SSE is within `σ²_eff` of the minimum (asymptotic χ²₁ test, 68.3 % confidence). Cost: one extra pass over the grid that was already scanned. Captures multimodal basins honestly — wide σ when several T1 candidates fit the data, tight σ when only one does. This is the σ-channel observation the E2.5 agent receives.
- `:bootstrap`. Resamples residuals at the LM optimum, refits each synthetic dataset by re-running the cached grid scan, and returns the standard deviation of the resampled T1 estimates. ~100× the cost of asymptotic, but the only method that captures **wrong-basin** convergence: different bootstrap samples can land in different basins, and that spread is the honest σ. Used as a diagnostic, not in the agent observation.

The "effective noise variance" `σ²_eff` is the larger of the residual-based estimate and a user-supplied noise floor:

- `σ²_resid = best_sse / (n − 2)` for `n > 4`. At `n ≤ 4` with two free parameters, the residual variance has no statistical power, so it is set to `+∞` and `σ²_eff` collapses to the floor (the "n-gate", §7.1 of `T1_FIT_AND_KOMA_TESTS.md`).
- The floor is either `abs_noise_sigma²` (preferred — absolute magnitude noise) or `(noise_sigma · rms(m))²` (legacy — relative-to-data noise). The absolute path is the one the agent sees, because the relative path couples σ_T1 to the agent's TI choice via `rms(m)` — picking saturated TIs would *reduce* σ_T1 even though the underlying noise hasn't changed. The "Fix B" test (§4) pins this decoupling.

### Oracle initialisation

`T1_oracle` narrows the grid to a `±oracle_band`-octave log-band around a supplied truth. This is a diagnostic-only knob: if the fit's MAPE collapses when given the oracle band, the bottleneck is wrong-basin convergence, not information content. It is never enabled in deployed code paths.

### Joint T1–T2 fit: `fit_t1_t2_generalized_ir`

When a sequence varies TE between samples — a multi-echo / IR-TSE readout — the echo attenuation `exp(−TE/T2)` is no longer a constant the amplitude can absorb (§0.6), and T2 becomes recoverable. This is the multiparametric analogue of the T1-only fit, over the model

$$
|S| = \bigl|\,A \cdot M_z^{\text{exc}}(T_1;\ \mathrm{TI}, \mathrm{TR}, \theta_{\text{inv}}, \alpha_{\text{exc}})\cdot e^{-\mathrm{TE}/T_2}\,\bigr|.
$$

The estimator reuses every piece of the T1-only fit. The only new structure is a **second grid axis**: the scan is now a 2-D log-grid over `(T1, T2)`, and at each grid point the predicted vector is the cached T1 forward model times the cached decay,

$$
\mathbf{y}_{ij} = M_z(T_{1,i};\ \cdot)\ \odot\ e^{-\mathrm{TE}/T_{2,j}},
$$

after which the optimal amplitude and SSE are the **same closed-form** `A = Σ m·|y| / Σ|y|²` ratio as before. Because the per-T1 `Mz` vectors and per-T2 decay vectors are each cached once, the inner step is a vector product plus the closed-form `A` — the cost is the grid area, not a non-linear solve. The point estimate is `argmin` over the 2-D SSE grid; `A` stays a profiled-out linear parameter, never gridded.

**Uncertainty** is profile-likelihood on each parameter: the 2-D SSE grid is marginalised onto each axis (minimise over the *other* parameter), then `_sigma_profile` returns the χ²₁ span within `σ²_eff` of the global minimum — the same machinery as the T1-only profile σ, applied twice. `σ²_eff` now uses the **3-parameter** residual variance `best_sse/(n−3)` (gate `n > 5`) since the fit spends T1, T2, and A. The profile construction makes the degenerate cases self-documenting: with a **constant TE** the SSE grid is flat along T2 (any T2 is absorbed by A), so the T2 profile passes the whole grid and `T2_sigma` spans the range — the fit reports "T2 unidentifiable" rather than a confident wrong number, and `T1` collapses to exactly the T1-only estimate. This is the bridge to the single-echo fits: hold TE constant and the joint fit reduces to `fit_t1_generalized_ir`; the §4 tests pin both that reduction and noiseless joint recovery.

The fit does **not** ship a σ-channel into any agent yet — it is the offline/library estimator for the IR-TSE multi-echo experiments; wiring an adaptive single-vs-multi-echo action into the RL environment is deferred.

---

### 4.7 Joint T1–T2 fit

`fit_t1_t2_generalized_ir` (`test_e2.jl` §"joint recovery + reduction limits") is pinned on three claims:

- **Noiseless joint recovery**: data from `|A·Mz(T1)·exp(−TE/T2)|` on a `(TI, TE)` grid recovers both T1 and T2 within 5 % across `(T1, T2) ∈ {(0.5, 0.1), (0.2, 0.05), (1.0, 0.2)} s`, with finite profile σ under 30 % of each truth.
- **Reduction to the T1-only fit**: with a **constant TE** the joint fit's `T1*` equals `fit_t1_generalized_ir`'s to within 5 %, and `T2_sigma` goes wide (`> 0.5·T2`) — the estimator reports T2 as unidentifiable rather than guessing, exactly as the flat-along-T2 SSE grid predicts.
- **Input validation**: TE-length mismatch, the `< 2` sample floor, and `Npe < 1` all throw.

Together with the closed-form, recovery, regression, σ-consistency, α-aware, calibration, and joint-fit tests, this gives well over forty independent assertions on the fitter — enough that any future edit to `fits.jl` that breaks an assumption used elsewhere in the stack will produce a named, failing test rather than a silent T1/T2 bias in a downstream run.

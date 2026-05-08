# E2.4 — Fix the fitter, then re-run A + C + delta

**Status:** planning · **Owner:** Arthur · **Date written:** 2026-05-07 · **Predecessor:** `EXPERT_REPORT.md` §19 (E2.3 catastrophic regression)

---

## 0. One-paragraph summary

E2.3 showed that combining all three §16.4 interventions (worst-case-weighted reward, σ-obs, delta-MAPE per-step reward) produces *worse* eval MAPE (1267 %) than every previous E2 run, despite PPO's internal diagnostics being healthier than ever (`explained_variance = +0.53`, `ep_len_mean = 8.17`, no entropy collapse). EXPERT_REPORT §19.5 traces the cause to a forward-model mismatch: `fit_t1_generalized_ir` assumes **steady-state IR repetition** but the simulator's `ir_se_2d_sequence` actually executes **`Npe` finite-length transient shots from `M0 = 1`** within each block. While the policy used a fixed action schedule (E2 §15, E2.2) the per-shot bias was constant and absorbed by the fitter's amplitude `A`; once the policy is forced to use *adaptive* TIs, the bias becomes per-shot, the fitter compensates by moving `T1`, and PPO faithfully exploits the closed-loop feedback that creates. **E2.4 is therefore not an RL experiment** — it is a forward-model fix (F1+ — a finite-Npe transient closed form), a KomaMRI cross-check on adaptive-action sequences, and a re-derivation of σ on the corrected model. Only after that lands does the A + C + delta retraining make sense. **An EPG-based forward model is sketched as the next sophistication step (§2.5)** if F1+ leaves residual bias; **F3 (cross-block magnetisation history) is deferred (§2.6)** as a separate experiment with its own MDP semantics.

---

## 1. Hypothesis and acceptance criteria

### 1.1 Hypothesis

**H1 (forward-model bias).** The dominant source of the E2.3 regression is steady-state-vs-transient mismatch between the fitter's forward model and the simulator. After replacing the forward model with a transient-aware closed form, eval MAPE on the *existing* E2.3 policy drops by ≥10× when re-fitting the same `(TI, TR, α_exc, |S|)` tuples.

**H2 (uncertainty calibration).** With the transient-aware forward model, `σ_T1 / |T1_est − T1_true|` (i.e. the σ-channel calibration ratio) on the same E2.2 / E2.3 rollouts becomes ≥0.3 (current value: ≤0.05 — three orders of magnitude over-confident).

**H3 (closing the RL loop).** Re-running A + C + delta-MAPE on top of the corrected fitter recovers monotone training (no late regression past 160k) and reaches eval MAPE < 30 % within 200k steps.

### 1.2 Acceptance criteria, in priority order

| Criterion | Target | Measured by |
|---|---|---|
| C1 — Cross-check passes for adaptive sequences | `rtol < 0.05` between transient closed-form and KomaMRI on N=20 randomised `(TI, TR)` schedules | new test in `test/test_e2.jl` |
| C2 — Fitter unbiased on simulator data | re-fitting E2.3 rollouts with new model yields MAPE < 50 % | offline script `python/refit_e2_3.py` (new) |
| C3 — σ calibrated | calibration ratio (def §1.1 H2) ≥ 0.3 on same data | extension of `python/diagnose_uncertainty.py` |
| C4 — RL loop re-runs cleanly | A+C+delta retraining: monotone, <30 % MAPE | new run `runs/e2/e2_4_A_C_delta` |

C1–C3 are the engineering deliverables; C4 is the *if-time-permits* RL re-run that turns this into a positive result for Ch4.

---

## 2. The forward-model fix — F1+ (finite-Npe transient closed form)

### 2.1 What's broken — and a critical correction to the §19 framing

`src/fitting/fits.jl:113-130` (`steady_state_mz_at_excite`) solves the fixed-point recurrence

```
Mz_pre   = (1 − E2) + cos(α_exc) · E2 · Mz_at_TI
Mz_at_TI = (1 − E1) + cos(θ_inv) · E1 · Mz_pre
```

i.e. assumes the IR block has been repeated *infinitely* until `Mz_pre` is at steady state — correct for clinical IR-TSE protocols shot many times consecutively.

`src/rl/e2.jl:236-271` (`_e2_simulate_step`) builds a sequence via `ir_se_2d_sequence(TI, TE, TR; α_exc, Nfe, Npe)` and calls `simulate()` once. **`ir_se_2d_sequence` (`src/sequences/blocks.jl:120-209`) is itself a multi-shot sequence — `Npe` IR-SE shots concatenated**, each separated by TR, with one Cartesian k-space row per shot. Shot 1 starts from `Mz = M0 = 1` (fresh phantom); shot k > 1 starts from whatever Mz survives the previous shot's `(TR − TI)` recovery. The image-domain pixel for each sphere comes from `abs.(ifft(ksp))` over the full Npe rows — so it integrates contributions from *every* shot's Mz_at_TI, **not** just the last one and **not** a steady-state value.

This means the original §19.5.1 framing was right in spirit but wrong in detail. Correct version:

- Shot 1 ≈ pure transient: `Mz_at_TI[1] = 1 − (1 − cos θ_inv) · E1`.
- Shot k → ∞ ≈ steady state: `Mz_at_TI[∞] ≈ steady_state_mz_at_excite(...)`.
- Shots 2…Npe interpolate between the two via the recurrence.
- The simulator output is roughly `(1/Npe) · Σ_k Mz_at_TI[k]` (sequential PE ordering, sphere centre pixel).

The fitter's steady-state model is too-aggressively-converged for early shots and approximately right for late shots, so it ends up biased toward whichever T1 minimises a *mixture* error — hence the closed-loop feedback to `T1_est` under adaptive TIs.

This also corrects the F1 prescription I gave originally. F1 (`n_rep = 1`, M0-start, TR ignored) is wrong in the *opposite* direction — too-aggressively-transient, ignoring the partial recovery the simulator does see between PE rows. The **correct minimum-change fix is F1+ — a finite-Npe transient closed form** that matches the simulator exactly.

### 2.2 F1+ — the finite-Npe transient closed form

Given `(T1, TI, TR, θ_inv, α_exc, Npe)` and starting from `Mz_pre[1] = M0 = 1`:

```
E1 = exp(−TI / T1)
E2 = exp(−(TR − TI) / T1)

Mz_pre[1]    = 1                                    # fresh phantom
for k = 1..Npe:
    Mz_at_TI[k]  = (1 − E1) + cos(θ_inv) · E1 · Mz_pre[k]    # post-TI, pre-excite
    Mz_pre[k+1]  = (1 − E2) + cos(α_exc) · E2 · Mz_at_TI[k]  # post-(TR−TI), pre-next-inversion

forward[T1; TI, TR, θ_inv, α_exc, Npe] = (1/Npe) · Σ_{k=1..Npe} Mz_at_TI[k]
```

The image-domain pixel from a uniformly-PE-encoded acquisition is the DC term of the IFFT, which is exactly `Σ_k ksp[k, central_freq] / Npe`. For a small homogeneous sphere occupying a few pixels, that DC term dominates the centre-pixel magnitude — giving the average-over-shots formula above. For non-trivial PE orderings (e.g. centric), the pixel is a Fourier-weighted sum of `Mz_at_TI[k]` instead; documenting this as a known approximation is fine for E2.4. The current code uses sequential ordering, so the average is the right approximation.

#### Why TR now matters in F1+

Inspect `Mz_pre[k+1]`: with short `TR − TI`, `E2 → 1` so `Mz_pre[k+1] → cos(α_exc) · Mz_at_TI[k]` — for the typical `α_exc = π/2` this is **0**, killing all Mz before the next inversion. With long `TR − TI`, `E2 → 0` so `Mz_pre[k+1] → 1` and each shot looks like shot 1 (pure transient from M0). In between, `forward` depends smoothly on TR. **The agent gets a clean, monotonic-in-TR gradient signal** — the loss of expressive power that pure F1 would have introduced is gone.

#### Limit checks (sanity, also pin in tests)

- `Npe = 1`: collapses to `1 − (1 − cos θ_inv) · E1` — the legacy TR→∞ E1 form.
- `Npe → ∞`: `Mz_pre[k]` reaches the fixed point of `Mz_pre = (1 − E2) + cos(α_exc) E2 [(1 − E1) + cos(θ_inv) E1 Mz_pre]`, which is exactly `steady_state_mz_at_excite`. So `forward → steady_state_mz_at_excite` (the existing test `test_e2.jl:122-155` continues to pass when called with large `Npe`).
- `TR → ∞`: `E2 → 0`, recurrence collapses to `Mz_pre[k+1] = 1` for all k > 0, so every shot is `1 − (1 − cos θ_inv) E1` (same as Npe=1). Agreement with E1's original assumption.

These three limits are independently checkable and serve as regression anchors.

### 2.3 Concrete code change

In `src/fitting/fits.jl`, add the new forward function (keep `steady_state_mz_at_excite` for clinical baselines):

```julia
"""
    transient_mz_at_excite_npe(T1, TI, TR, θ_inv, α_exc; Npe::Int)

Average over the `Npe` Mz_at_TI values produced by an Npe-shot IR block
starting from thermal equilibrium (`Mz_pre[1] = 1`). Matches what
`ir_se_2d_sequence` actually feeds the simulator. Limits:
  - Npe = 1     ⇒ legacy TR→∞ transient form, `1 − (1 − cos θ_inv) e^{−TI/T1}`.
  - Npe → ∞    ⇒ `steady_state_mz_at_excite`.
  - TR → ∞     ⇒ pure transient (every shot from M0).
"""
function transient_mz_at_excite_npe(T1::Real, TI::Real, TR::Real,
                                     θ_inv::Real, α_exc::Real; Npe::Int)
    @assert Npe ≥ 1 "Npe must be ≥ 1"
    E1 = exp(-TI / T1)
    E2 = exp(-(TR - TI) / T1)
    Mz_pre = 1.0
    accum  = 0.0
    @inbounds for k in 1:Npe
        Mz_at_TI = (1 - E1) + cos(θ_inv) * E1 * Mz_pre
        accum   += Mz_at_TI
        Mz_pre   = (1 - E2) + cos(α_exc) * E2 * Mz_at_TI
    end
    return accum / Npe
end
```

Then `fit_t1_generalized_ir` gains an `Npe::Int = 8` keyword (matching the env's default) and calls `transient_mz_at_excite_npe` instead of `steady_state_mz_at_excite`. Wire `Npe` through `_e2_update_t1_estimates!` (`src/rl/e2.jl:273-308`) so the fit's Npe always matches the simulator's Npe — read it from `env.Npe`. The Julia↔Python bridge (`src/rl/e2.jl` aliases at the bottom of the file) needs no change because Npe is passed inside the fit kwargs, not at the env level.

Cost: O(Npe) per grid point per fit, where Npe is currently 8. Each grid point goes from ~5 floating-point ops to ~25 — measurable, but the fit is sub-millisecond and dominated by the `Npe = 8` simulator anyway.

### 2.4 Estimated effort

- Code change: ~30 minutes (one new function in `fits.jl`, one keyword wired through, one `env.Npe` read in `_e2_update_t1_estimates!`).
- Test additions: ~1.5 hours (new `test/test_e2.jl` block — see §3 below).
- Refit script for E2.3 rollouts: ~1 hour (read existing rollouts, call the new fitter offline).

---

## 2.5 Next sophistication step — EPG forward model (sketch)

F1+ removes the **steady-state assumption**, but still rests on three idealisations:

1. **Perfect transverse spoiling** between PE rows (`Mxy = 0` at the start of every shot's TR-recovery window). The current sequence achieves this implicitly via short `T2 = 20 ms` ≪ `(TR − TI)`; on real tissue (T2 ≈ 80–2200 ms, see CLAUDE.md table) and longer-T2 phantom regimes this assumption breaks.
2. **Instantaneous, ideal RF pulses** (no slice-profile, no B1 inhomogeneity).
3. **Single-compartment relaxation** (one T1, one T2 per voxel — no MT, no off-resonance distribution).

**Extended Phase Graphs (EPG)** relax (1) directly; the framework also extends naturally to (3) with multi-compartment EPG and to slice-profile via integration over the slab. Reference: Weigel 2015, "Extended phase graphs: dephasing, RF pulses, and echoes — pure and simple", *J. Magn. Reson. Imaging* 41(2):266-295. Established Julia options: `MRIReco.jl` ships an EPG implementation; for full control we can also write our own — it's <100 LOC.

### 2.5.1 EPG in 90 seconds

EPG represents the magnetisation as a **discrete sum of configuration states** indexed by accumulated dephasing under a fixed gradient quantum `Δk`. After each spoiler gradient, every spin's accumulated phase increments by an integer multiple of `Δk·voxel`. So the entire transverse pool can be tracked as a vector of complex amplitudes:

```
F⁺ = (F⁺_0, F⁺_1, F⁺_2, …)        # transverse, "+ phase" states
F⁻ = (F⁻_0, F⁻_1, F⁻_2, …)        # transverse, "− phase" (conjugate-symmetric)
Z  = (Z_0,  Z_1,  Z_2,  …)         # longitudinal states
```

with `F⁺_0` the observable signal (at zero net dephasing), `Z_0` the bulk Mz, and higher-index states tracking how much magnetisation is dephased by `n·Δk` worth of gradient — i.e. waiting around to be rephased by a future gradient and produce an echo.

**Three operators act on (F⁺, F⁻, Z):**

| Op | Trigger | Effect |
|---|---|---|
| `R(α, φ)` | RF pulse, flip α, phase φ | 3×3 matrix mixing `(F⁺_n, F⁻_n, Z_n)` for every n. Standard Bloch rotation in state-vector form. |
| `E(τ; T1, T2)` | free precession of duration τ | `F⁺_n ← F⁺_n · exp(−τ/T2)`, `F⁻_n ← F⁻_n · exp(−τ/T2)`, `Z_n ← Z_n · exp(−τ/T1) + δ_{n,0}·(1 − exp(−τ/T1))` |
| `S(±1)` | gradient spoiler of magnitude `Δk` | shift index: `F⁺_n ← F⁺_{n−1}`, `F⁻_n ← F⁻_{n+1}`. `Z` is unaffected (longitudinal spins don't dephase). |

That's the entire formalism. **No spoiling assumption needed**: surviving Mxy that hasn't been fully rephased simply lives in higher-index states, and any subsequent RF pulse can move it back to `F⁺_0` (producing a stimulated echo) or into `Z_n` (producing a subsequent spin echo). This is what `steady_state_mz_at_excite` cannot represent — the steady-state recurrence collapses everything onto `Z_0` after each TR, by assumption.

### 2.5.2 EPG implementation sketch in Julia

```julia
"""
    EPGState(N::Int)

Configuration-state vector of length N. F[n+1] indexes F⁺_n; Fminus[n+1] is F⁻_n;
Z[n+1] is Z_n. N = ceil((TR − TI) / shot_time) is a safe truncation order
for the Npe-shot IR-SE sequence — each spoiler advances at most one state.
"""
mutable struct EPGState
    F::Vector{ComplexF64}        # F⁺_0, F⁺_1, …
    Fminus::Vector{ComplexF64}   # F⁻_0, F⁻_1, … (= conj(F⁻_{−n}) by symmetry)
    Z::Vector{ComplexF64}        # Z_0, Z_1, …
end

EPGState(N::Int) = EPGState(zeros(ComplexF64, N), zeros(ComplexF64, N),
                              ComplexF64[1; zeros(N-1)])    # equilibrium: Z_0=1

"""Apply RF pulse with flip α (rad), phase φ (rad). In-place on s."""
function rf!(s::EPGState, α::Real, φ::Real)
    c2, s2  = cos(α/2)^2, sin(α/2)^2
    sinα    = sin(α)
    eiφ, e_iφ, ei2φ, e_i2φ = cis(φ), cis(-φ), cis(2φ), cis(-2φ)
    @inbounds for n in eachindex(s.F)
        Fp, Fm, Zn = s.F[n], s.Fminus[n], s.Z[n]
        s.F[n]      = c2*Fp + e_i2φ*s2*Fm - im*e_iφ*sinα*Zn
        s.Fminus[n] = ei2φ*s2*Fp + c2*Fm + im*eiφ*sinα*Zn
        s.Z[n]      = -im/2*(eiφ*Fp - e_iφ*Fm)*sinα + cos(α)*Zn
    end
end

"""Free precession for time τ with relaxation (T1, T2). In-place on s."""
function relax!(s::EPGState, τ::Real, T1::Real, T2::Real)
    e1, e2 = exp(-τ/T1), exp(-τ/T2)
    @inbounds for n in eachindex(s.F)
        s.F[n]      *= e2
        s.Fminus[n] *= e2
        s.Z[n]      *= e1
    end
    s.Z[1] += (1 - e1)        # longitudinal regrowth toward M0=1 only on Z_0
end

"""Gradient spoiler: shift F⁺ up, F⁻ down by one state."""
function spoil!(s::EPGState)
    @inbounds for n in length(s.F):-1:2
        s.F[n] = s.F[n-1]
    end
    s.F[1] = conj(s.Fminus[1])              # the F⁻_1 → F⁺_0 reflection at zero
    @inbounds for n in 1:length(s.Fminus)-1
        s.Fminus[n] = s.Fminus[n+1]
    end
    s.Fminus[end] = 0
end
```

For an Npe-shot IR-SE block, the simulation loop is:

```julia
function epg_mz_at_excite_npe(T1::Real, T2::Real, TI::Real, TR::Real,
                                θ_inv::Real, α_exc::Real;
                                Npe::Int, N::Int = 16)
    s = EPGState(N)
    accum = ComplexF64(0)
    for k in 1:Npe
        rf!(s, θ_inv, 0)              # 1. inversion
        relax!(s, TI, T1, T2)         # 2. TI delay
        # Magnetisation at excitation moment is in Z (transverse not yet
        # generated by exc pulse). Accumulate Z_0 — the analogue of
        # Mz_at_excite for the EPG model.
        accum += s.Z[1]
        rf!(s, α_exc, π/2)            # 3. excitation (phase π/2 = standard)
        # Implicit: SE refocus, ADC, and TR-TI recovery omitted for the
        # T1-fit forward model — only Mz_at_excite matters for fitting T1.
        # A full EPG sim would include refocus + TR-TI recovery via spoil! +
        # relax! to track stimulated-echo pathways into the next shot.
        spoil!(s)                     # crude end-of-shot dephasing
        relax!(s, TR - TI, T1, T2)    # recovery to next inversion
    end
    return real(accum) / Npe          # Z is real at the inversion-pulse moment
end
```

That's ~70 LOC of new code. Drop-in replaces `transient_mz_at_excite_npe` in the fitter when called with an explicit T2 estimate.

### 2.5.3 What changes in the fit

The fitter currently has two unknowns (`T1`, `A`). EPG introduces T2 dependence — which is *information* (T2 is a real tissue property the project will eventually want to estimate, see E2.5 / E3) but also adds a degree of freedom. Two strategies:

- **EPG-T2-fixed**: pass the phantom's nominal T2 per sphere into the fit; still a 2-parameter (T1, A) fit. T2 is known from the QalibreMD calibration sheet, so this is fine for the digital-twin work. Loses the chance to *fit* T2.
- **EPG-T2-joint**: fit (T1, T2, A) jointly. Three unknowns; needs more / better-conditioned data. Worth attempting once F1+ is shown to leave residual bias.

The σ formula generalises directly: `J` becomes the n×3 Jacobian, the [T1, T1] entry of `(JᵀJ)⁻¹ · σ²_eff` is still the variance of `T1*`. No new derivation needed — see `docs/T1_FIT_AND_KOMA_TESTS.md` §4.

### 2.5.4 When to switch from F1+ to EPG

Acceptance criterion C2 in §1.2 is "MAPE < 50 % when re-fitting E2.3 rollouts". If F1+ achieves this, EPG is overkill for E2.4. If F1+ stalls at 50–200 % MAPE (i.e. better than the 1267 % catastrophe but not good enough for the C4 retraining pass condition of <30 %), the residual bias is from the spoiling assumption or finite-RF effects — both of which EPG addresses. Estimated effort if needed: 2–3 days (the LOC count is small but EPG validation against KomaMRI for varying actions is finicky).

### 2.5.5 Why not jump straight to EPG?

Three honest reasons:

1. **F1+ is correct for the spoiling regime the simulator currently runs in** (`T2 = 20 ms` in the test phantom, `T2 ≪ TR − TI` in env evaluation). Spending effort on EPG before F1+ has been shown insufficient is premature optimisation.
2. **Comparison ladder integrity.** F1+ is a single-axis change to the forward model from the steady-state baseline. EPG changes both the forward model *and* the assumption set. A cleaner ablation reads "steady-state → F1+ → EPG" — three rows, three controlled changes — than "steady-state → EPG".
3. **Time budget.** F1+ is half a day; EPG is 2–3 days. The FYP timeline in `project_context/PROJECT.md` only has E2.4 budgeted at ~2 working days.

So the priority is: F1+ first, EPG only if F1+ doesn't clear C2.

---

## 2.6 F3 deferred — magnetisation history across blocks (NOT in E2.4)

F3 (env carries per-voxel Mz between blocks; agent gets a real "rest vs. measure" choice) is appealing because it makes inter-block timing physically meaningful — a learnable axis the agent currently has no signal on. After thinking through it I'm deferring it to a separate experiment (likely E2.5 or absorbed into E3) for these reasons:

1. **Orthogonal to the E2.3 regression.** The current regression is caused by the *fitter's* model mismatch, not by missing inter-block coupling. F1+ resolves it; F3 doesn't (the env still resets between blocks under F1+ unless F3 lands too — but that's a separate change).
2. **MDP semantics change.** F3 introduces a new state variable (per-voxel Mz) and likely a new action axis (inter-block delay). All four prior E2 ablation rows (§18.7) would become non-comparable to E2.4 — the controlled-comparison story breaks.
3. **KomaMRI doesn't natively accept non-equilibrium Mz at simulator entry** — `Phantom` has no `Mz0` field and `simulate()` initialises every spin to thermal equilibrium. Plumbing Mz state across `simulate()` calls requires either patching `Phantom` (upstream PR) or prepending a "warmup" pseudo-sequence that re-creates the right pre-state (extra simulator cost, fiddly to verify).
4. **Self-degeneracy risk.** If the time budget isn't tight enough, the optimal F3 policy is to pad inter-block delays to ∞, recovering the current "fresh M0 each block" regime — but with extra wall-clock spent doing nothing. Avoiding this requires a binding total time budget *and* the agent solving a credit-assignment problem across orders-of-magnitude longer time horizons.
5. **The fit becomes non-causal.** Under F3, the analytic forward model needs to know the *starting* Mz of each block to predict its measurements. In the simulator we have that ground truth, but **on a real scanner it is unobservable** — so any F3 result is sim-only and doesn't transfer. That makes F3 a less compelling sim-to-real story than F1+ + EPG, both of which are causal in the same way a real reconstruction would be.

What F3 *is* good for, when we come back to it:

- **C3 (spatial localisation under pose uncertainty)**: revisiting a slice without resetting Mz is realistic and gives the agent a reason to model "I just hit this region 200 ms ago — Mz is still suppressed".
- A separate publishable angle: "adaptive sequence design **with magnetisation history**" extends the C2 narrative into a regime no prior work covers, and it composes naturally with E5 pose-tracking.

For now: noted, deferred, will revisit once F1+ ± EPG is closed out and Ch4's C2 chapter is drafted. The honest paragraph for the limitations chapter is "the env resets to thermal equilibrium between acquisitions; cross-acquisition magnetisation history is left to E2.5".

---

## 3. Cross-checks against KomaMRI

### 3.1 Test gap to plug (also `EXPERT_REPORT.md` §8.8 priority 1)

`test/test_e2.jl:122-155` uses `n_rep = 4` and one TI/TR pair per case. It does **not** validate the forward model under (a) `n_rep = 1` and (b) action sequences that vary TI / TR within the same fit. Both of those are exactly what E2 actually does.

### 3.2 New test cases to add

Append to `test/test_e2.jl`. The first three pin the closed-form *limits* of `transient_mz_at_excite_npe`; the next two cross-check it against KomaMRI under realistic and adaptive-TI conditions; the last is the negative regression that pins the steady-state model's bias.

```julia
@testset "F1+ closed-form limits" begin
    T1, TI, TR = 1.0, 0.5, 2.0
    # Limit 1: Npe = 1 ⇒ TR-blind transient form.
    @test isapprox(transient_mz_at_excite_npe(T1, TI, TR, π, π/2; Npe = 1),
                    1 - (1 - cos(π)) * exp(-TI/T1); rtol = 1e-12)
    # Limit 2: Npe → ∞ ⇒ steady-state form.
    @test isapprox(transient_mz_at_excite_npe(T1, TI, TR, π, π/2; Npe = 10_000),
                    steady_state_mz_at_excite(T1, TI, TR, π, π/2);
                    rtol = 1e-3)
    # Limit 3: TR → ∞ ⇒ Mz_pre[k] ≡ 1 for k > 0 ⇒ pure transient.
    @test isapprox(transient_mz_at_excite_npe(T1, TI, 1e6, π, π/2; Npe = 8),
                    1 - (1 - cos(π)) * exp(-TI/T1); rtol = 1e-6)
end

@testset "F1+ matches KomaMRI on Npe-shot IR-SE (fixed action)" begin
    # Use the actual ir_se_2d_sequence and pull the centre pixel out of the
    # IFFT. Short T2 enforces perfect spoiling, the only assumption F1+ shares
    # with the simulator.
    for (T1, TI, TR, Npe) in [(1.0, 0.5, 2.0, 8), (0.3, 0.1, 1.0, 8),
                                (0.05, 0.02, 0.5, 4)]
        obj = single_spin_phantom(; T1 = T1, T2 = 0.02)
        seq = ir_se_2d_sequence(TI, 0.02, TR;
                                 α_exc = π/2, FOV = 0.2,
                                 Nfe = 16, Npe = Npe, amp_T = 100e-6)
        raw = simulate(obj, seq, Scanner())
        ksp = ComplexF32[raw.profiles[k].data[1, 1] for k in 1:Npe]
        # DC-of-IFFT for a single-pixel-bright sphere ≈ mean(ksp)
        pix_sim = abs(sum(ksp) / Npe)
        pix_ana = abs(transient_mz_at_excite_npe(T1, TI, TR, π, π/2; Npe = Npe))
        @test isapprox(pix_sim, pix_ana; rtol = 0.05, atol = 5e-3)
    end
end

@testset "F1+ fitter recovers T1 from adaptive-TI sequences" begin
    # The regime E2 actually operates in: 8 distinct TIs/TRs, each block is
    # an Npe-shot IR-SE from M0=1. Generate from the (now-validated) closed
    # form to test the fit; the closed-form-vs-sim agreement is pinned above.
    rng = MersenneTwister(42)
    Npe = 8
    for T1 in [0.05, 0.1, 0.5, 1.0, 1.8]
        n   = 8
        TIs = sort!(exp.(log(0.01) .+ (log(2.5) − log(0.01)) .* rand(rng, n)))
        TRs = TIs .+ exp.(log(0.5) .+ log(3.0) .* rand(rng, n))   # TR > TI
        αes = fill(π/2, n);   αs = fill(π, n)
        mags = [abs(transient_mz_at_excite_npe(T1, TIs[i], TRs[i],
                                                  π, αes[i]; Npe = Npe))
                  for i in 1:n]
        fit = fit_t1_generalized_ir(TIs, αs, mags;
                    TRs = TRs, α_excs = αes, Npe = Npe)
        @test isapprox(fit.T1, T1; rtol = 0.05)
    end
end

@testset "Steady-state fitter is provably biased on Npe-shot data (regression for §19.5.1)" begin
    # Pin the bias direction: steady-state model on Npe=8 data must miss truth
    # by ≥10%, while F1+ recovers within 5%. Should fail loudly if anyone
    # silently re-introduces the steady-state assumption.
    T1, Npe   = 1.0, 8
    TIs       = [0.05, 0.3, 1.0, 2.0]
    αs        = fill(π, 4); αes = fill(π/2, 4)
    TRs       = fill(0.5, 4)         # short TR ⇒ pre*-vs-1 gap is large
    # "True" data from the Npe-shot transient model (ground truth).
    mags = [abs(transient_mz_at_excite_npe(T1, TIs[i], TRs[i], π, π/2; Npe = Npe))
              for i in 1:4]
    f_npe    = fit_t1_generalized_ir(TIs, αs, mags; TRs = TRs, α_excs = αes,
                                       Npe = Npe)
    f_steady = fit_t1_generalized_ir(TIs, αs, mags; TRs = TRs, α_excs = αes,
                                       Npe = 10_000)        # ≈ steady state
    @test abs(f_npe.T1    - T1) / T1 < 0.05      # F1+ unbiased
    @test abs(f_steady.T1 - T1) / T1 > 0.10      # steady-state biased
    @test abs(f_npe.T1 - T1) < abs(f_steady.T1 - T1)  # F1+ strictly better
end
```

The fourth test is the *negative regression* that pins §19.5.1 as a testable claim. The second test is the most expensive (it calls KomaMRI 3 times for fairly long sequences), so consider tagging it with `@testset.skip_long_running` if CI time matters.

### 3.3 Estimated effort

~1.5 hours including running the test suite end-to-end.

---

## 4. Re-derive σ on the corrected model

### 4.1 Why σ is still problematic even after F1+

The asymptotic σ formula (`σ²_eff · (JᵀJ)⁻¹`, `fits.jl:217-248`) doesn't depend on which forward model is used — it's a generic Cramér–Rao approximation. So once F1+ lands, the σ formula will use the *correct* Jacobian of the new closed form and will produce an unbiased σ *for that model*. But the four pre-existing issues from `docs/T1_FIT_AND_KOMA_TESTS.md` §7 still apply:

- §7.1 small-`n` DOF
- §7.2 signal-relative noise floor
- §7.3 floor coupling to action choice
- §7.4 multimodal SSE

§7.5 (the new one introduced by E2.3 — steady-state-vs-transient bias) is fully resolved by F1+. The other four are not. Note also that the §19.4 diagnostic showed σ on E2.3 is *honestly large* (median 81 %, 41 % of cells > 100 %) — not silently low as I'd predicted. The σ-channel's E2.3 problem is **saturation** ("it's all uncertain") not **calibration** ("looks confident, isn't"); §7.1's n-gate fix is therefore higher-priority than I credited. Concretely, with `n ≤ 4` and σ²_resid set to Inf (Fix A), the floor (Fix B) becomes the only signal — and if Fix B is decoupled from action choice, σ stops varying with the agent's noise floor and becomes the per-fit noise floor it was meant to be.

### 4.2 Minimum recommended σ fixes for E2.4 (one-line each)

Apply two of the four fixes proposed in `docs/T1_FIT_AND_KOMA_TESTS.md` §7.5:

**Fix A — n ≥ 5 gate on σ²_resid.** Replace `fits.jl:235`:
```julia
σ²_resid = n > 4 ? best_sse / (n - 2) : Inf
```
Forces the floor to dominate at small `n`. No risk to the math; just acknowledges that at n=3, the residual variance has no statistical power.

**Fix B — Absolute noise floor.** In `_e2_update_t1_estimates!` (`src/rl/e2.jl:281`), compute the noise floor *once* from the first block's signal RMS and pass an *absolute* `noise_sigma` to `fit_t1_generalized_ir` instead of the relative `env.noise_sigma_rel = 0.05`. Decouples σ from the agent's TI choice (resolves §7.3).

Implementation sketch:
```julia
# At episode reset (or first call to _e2_update_t1_estimates! per sphere):
if !isfinite(env.noise_floor_abs[i])
    env.noise_floor_abs[i] = env.noise_sigma_rel * mean(abs.(env.block_mags[i]))
end
fit = fit_t1_generalized_ir(...; noise_sigma = env.noise_floor_abs[i] /
                                                max(rms_m, 1e-9))
# (the ratio re-relativises so the existing σ²_floor formula gives the right
#  absolute number — equivalent to passing an absolute floor directly if the
#  fitter is extended to accept one)
```

Cleaner: extend `fit_t1_generalized_ir` to accept an `abs_noise_sigma` keyword and use that directly when set, falling back to the existing relative behaviour.

**Skip for now** — profile-likelihood σ (§7.5 fix 3) and bootstrap σ (fix 4). Both are useful but neither is on the critical path to fixing E2.3, and both are 1-day implementations. Document them as future work.

### 4.3 Estimated effort

~1 hour for fixes A + B + the corresponding regression test.

---

## 5. Validation runs (in order, each gated on the previous passing)

Each of these is an offline (no-RL-training) check using the *existing* E2.3 rollouts. Total ~3 hours of compute and analysis.

### 5.1 V1 — Re-fit E2.3 rollouts with the corrected model

`runs/e2/e2_3_A_C_delta/diagnostics/sigma_summary.json` already exists from the §19.4 diagnostic run, with per-block per-sphere `(TIs, TRs, α_excs, mags)`. Script `python/refit_e2_3.py` (new) reads it, calls `fit_t1_generalized_ir(...; Npe = env.Npe)` with the F1+ model, and reports:

- New per-sphere MAPE (predicted: <50 %, vs current 1267 %)
- New median σ_T1 / |T1_est − T1_true| (predicted: ≥0.3, vs current ≤0.05 once Fixes A + B land; without them the σ saturation observed in §19.4 will persist)
- Per-sphere bar plot in `report_plots/E2.4/refit_per_sphere_mape.png`

**Pass condition (acceptance criteria C2 + C3):** if per-sphere mean MAPE drops to <50 % and σ-calibration ratio ≥0.3, H1 + H2 are confirmed; proceed to V2.

### 5.2 V2 — Re-fit E2 §15 and E2.2 rollouts with corrected model (sanity)

Same script over the prior runs' rollouts. Predicted outcome: MAPE drops modestly (10–30 %) but not catastrophically — those policies were exploiting the bias-absorbed-by-A regime, so the fitter was *less* wrong on their data than on E2.3's.

If MAPE *increases* on E2 §15, F1+ is overcorrecting and we'd need to investigate (e.g. the simulator might do partial repetition we missed, the IFFT-DC approximation in §2.2 is inadequate, or the sphere centre pixel is being read at a different k-space weighting). If it stays flat or improves, F1+ is the right model for both regimes — the §19.5.2 explanation is correct.

### 5.3 V3 — Train-time test: simulator vs new analytic model under varying actions

Run the §3.2 second testset (the adaptive-TI fit) on N=50 random T1 values; report mean and p90 of `|T1_fit − T1_true| / T1_true`. **Pass condition (C1):** rtol ≤ 0.05 mean, ≤ 0.10 p90.

### 5.4 V4 — A + C + delta retraining (the headline run)

If V1–V3 pass:

```bash
bash run_e2.sh --reward-mode delta_mape --simplified-action \
               --terminal-bonus 0.0 --mape-alpha 0.5 \
               --timesteps 200000 \
               --out runs/e2/e2_4_A_C_delta
```

Same flags as E2.3 — the *only* code-level difference is the corrected fitter (and σ fixes). This is the controlled comparison: identical RL setup, single-axis change in the underlying physics model. **Pass condition (C4):** monotone descent, no late regression, eval MAPE < 30 % at 200k.

If V4 passes, E2.4 is the first E2 run that delivers a genuinely-adaptive policy, and it does so via a *physics-modelling fix* rather than an RL-side fix — exactly the "physics is the bottleneck" narrative §19.6 sets up.

---

## 6. Risks and what to do if each fires

| Risk | Likelihood | Diagnostic | Mitigation |
|---|---|---|---|
| **R1.** F1+ leaves residual bias because the perfect-spoiling assumption is the binding one (e.g. on real-T2 phantom regimes) | low–medium — `T2 = 20 ms ≪ TR − TI` in current sim, so spoiling holds | V2 has MAPE > 100 % on E2 §15 rollouts, OR V3 has rtol > 0.05 | Switch to EPG (§2.5). Cost: 2–3 days. F1+ is preserved as the "no-spoiling-assumption-needed" branch via the `Npe` keyword. |
| **R2.** F1+ fixes T1 bias but σ stays saturated/over-confident | medium — §7.1–7.4 unaddressed by the forward-model change | C3 (calibration ratio) < 0.3 even after Fixes A + B | Add profile-likelihood σ (§7.5 fix 3 of the docs). ~1 day. (Downgraded from before §19.4: σ is currently *over-saturated* not *over-confident*, so Fix A's n-gate is the dominant lever.) |
| **R3.** Even with corrected fitter + σ, A+C+delta still regresses past 160k | medium — §15.4 items 1, 4 still active under the new MDP | Look at PPO eval-MAPE curve; if monotone-then-flat we have a different ceiling | Drop α to 0.7 (less aggressive max-weight), or revert to Option B (the step-size penalty in §16.4) which is the only §16.4 ingredient never yet tested. |
| **R4.** Refit script discovers the saved E2.3 rollouts can't reproduce the per-block fit because of seed/RNG noise mismatch | low — `diagnose_uncertainty.py` already records mags | refit MAPE differs from training-time MAPE on the same checkpoints | Re-evaluate the policy with `eval_e2.py --episodes 30` to regenerate fresh rollouts under known seeds. |
| **R5.** F1+ over-corrects — i.e. the IFFT-DC ≈ mean approximation in §2.2 is too crude and the right per-shot weighting is centric or otherwise non-uniform | low — sequence uses sequential PE ordering (`ir_se_2d_sequence` builds shots 1…Npe in order, ky from −kmax to +kmax) | V2: MAPE on prior runs *increases* | Replace the `Σ Mz_at_TI[k] / Npe` average with a Fourier-weighted sum, with weights derived from the actual PE ordering. ~2 hours. |

R3 is the residual real-world risk — even a perfect fitter still leaves the §15.4 reward-sparsity and multimodality issues. If R3 fires, we have a paper-shaped result anyway: "a transient-aware fitter is necessary but not sufficient; further reward shaping is needed", which is also a publishable finding and slots into Ch4 cleanly.

---

## 7. Effort + timeline

| Stage | Hours | Wall time |
|---|---:|---|
| §2 forward-model code change (F1+) | 0.5 | 0.5h |
| §3 KomaMRI cross-checks (4 new tests including limit checks) | 1.5 | 2h |
| §4 σ fixes A + B | 1.0 | 1.5h |
| §5.1 V1 — refit E2.3 from existing `sigma_summary.json` | 1.5 | 2h |
| §5.2 V2 — refit prior runs | 0.5 | 1h |
| §5.3 V3 — adaptive-action validation | 0.5 | 1h |
| §5.4 V4 — A+C+delta retraining (200k PPO) | — | ~5h compute, mostly unattended |
| Writeup → `EXPERT_REPORT.md` §20 | 1.5 | 2h |
| **Total (F1+ path)** | **7 h hands-on + 5h compute** | **~2 working days** |
| (If R1 fires) §2.5 EPG forward model | 12–18 | 2–3 working days |

This fits comfortably inside one weekly Wayne-update cycle. If R1 fires we have a longer-but-still-finite EPG branch that still produces a clean, single-axis change to the physics model — and a *better* report angle (EPG is a direct connection to the published-MR-fingerprinting literature).

---

## 8. Why this is good for the report

The ablation now reads:

| Layer | What we changed | What it isolated |
|---|---|---|
| Reward (mean → delta) | E2 → E2.1 | dense per-step signal breaks early termination |
| Observation (+σ) | E2 → E2.2 | σ-channel is policy-inert without reward pressure |
| Reward + obs combined (A+C+delta) | E2 → E2.3 | exposes a *physics-model* failure invisible to fixed schedules |
| **Forward model (steady → finite-Npe transient)** | **E2.3 → E2.4** | **resolves the physics-model failure; A+C+delta now works** |
| (Stretch: forward model → EPG) | E2.4 → E2.4-EPG | removes the perfect-spoiling assumption; sets up E3 |

Four (or five) layers, each a single-axis ablation with a hypothesis, a diagnostic, and a result. That's the structure Wayne's interim feedback called for under "engagement and originality"; it's also exactly the C2 (scalable simulation-in-the-loop RL) chapter's punchline: **the simulator–fitter fidelity gap is the operational bottleneck, not RL training scale**.

Three pieces of MR-physics work fall out of E2.4 and are independently report-worthy regardless of how the RL retrain lands:

1. **F1+ closed form** — "transient finite-Npe steady-state-ramp model for adaptive multi-shot IR-T1 mapping" is a clean methods contribution. The three-limit (Npe=1 / Npe→∞ / TR→∞) verification is independently checkable and reads well in a methods section.
2. **The negative-regression test** for the steady-state model on transient data (§3.2 fourth testset) makes "fast-forward-model fidelity matters for sim-in-the-loop RL" empirically concrete — a one-paragraph result table in the limitations chapter.
3. **F3 limitations paragraph** — explicitly noting that cross-acquisition magnetisation history is left to E2.5 frames the project's current scope honestly and signposts a next-step research direction. This is exactly the kind of "future work that's specific, not vague" Wayne asks for.

If F1+ alone clears C4, the dissertation Ch4 has a clean four-layer ablation with a single physics-model change as the keystone. If EPG is required, Ch4 gains a fifth row but also a stronger external-validity argument (EPG is the standard tool in the field; F1+ is an in-house derivation).

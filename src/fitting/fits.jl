# Pure-Julia monoexponential fits used by the E0 baseline and the
# running-estimate feature of the later RL environments. No external
# optimiser dependency — T2 is closed-form in log space, T1 is a log-grid
# search over T1 with closed-form (A, B) at each T1.

"""
    fit_t2_se(TEs, magnitudes) -> (T2, S0, residual)

Log-linear fit of `|S| = S0 · exp(−TE/T2)` to magnitude-only SE data.
Returns the T2 estimate (s), amplitude S0, and the RMS of the residuals
in log space. Points with non-positive magnitude are dropped.
"""
function fit_t2_se(TEs::AbstractVector{<:Real},
                   mags::AbstractVector{<:Real})
    length(TEs) == length(mags) || error("TEs and mags length mismatch")
    keep = mags .> 0
    count(keep) >= 2 || error("Need ≥2 positive samples for T2 fit")
    t = Float64.(TEs[keep])
    y = log.(Float64.(mags[keep]))
    X = hcat(ones(length(t)), -t)
    β = X \ y
    S0 = exp(β[1])
    slope = β[2]
    slope > 0 || error("Non-decaying signal; slope = $slope")
    T2 = 1 / slope
    resid = sqrt(sum((X * β .- y).^2) / length(y))
    (T2 = T2, S0 = S0, residual = resid)
end

"""
    fit_t1_ir(TIs, mags; T1_range = (5e-3, 5.0), n_grid = 400)

3-parameter magnitude-IR fit: `|S| = |A − B · exp(−TI/T1)|`. A log-spaced
grid over `T1_range` is scanned; at each candidate T1 the optimal (A, B)
are closed-form from a linear regression of signed-sign · mag on
(1, exp(−TI/T1)). The best SSE wins.

The signed-sign is recovered heuristically: samples with TI below the
null time (where the sign flips) are negated before fitting. With a
magnitude acquisition we can only tell by iterating — we try both sign
patterns per candidate T1 and keep the better one.

Returns `(T1, A, B, residual)`.
"""
function fit_t1_ir(TIs::AbstractVector{<:Real},
                   mags::AbstractVector{<:Real};
                   T1_range::NTuple{2,<:Real} = (5e-3, 5.0),
                   n_grid::Int = 400)
    length(TIs) == length(mags) || error("TIs and mags length mismatch")
    length(TIs) >= 3 || error("Need ≥3 samples for 3-param IR fit")

    ti = Float64.(TIs)
    y  = Float64.(mags)

    T1_candidates = exp.(range(log(T1_range[1]), log(T1_range[2]);
                               length = n_grid))

    best_sse = Inf
    best = (T1 = NaN, A = NaN, B = NaN)

    # We don't know the true sign pattern (magnitude data). For a monotone
    # |A − B·e^{-TI/T1}|, the sign flips at most once at the null
    # TI*. For each candidate T1, try every prefix-flip of the sorted-TI
    # data and keep the best.
    order = sortperm(ti)
    ti_sorted = ti[order]
    y_sorted  = y[order]
    n = length(ti)

    for T1 in T1_candidates
        e = exp.(-ti_sorted ./ T1)
        for k in 0:n               # flip the first k points
            signs = ones(n)
            signs[1:k] .= -1.0
            y_signed = signs .* y_sorted
            # y = A - B·e → linear regression on [1, -e]
            X = hcat(ones(n), -e)
            β = X \ y_signed
            A, B = β[1], β[2]
            resid = X * β .- y_signed
            sse = sum(abs2, resid)
            if sse < best_sse
                best_sse = sse
                best = (T1 = T1, A = A, B = B)
            end
        end
    end

    (T1 = best.T1, A = best.A, B = best.B,
     residual = sqrt(best_sse / length(ti)))
end

"""
    steady_state_mz_at_excite(T1, TI, TR, θ_inv, α_exc) -> Mz

Steady-state longitudinal magnetisation (in units of M0) at the moment of
the excitation pulse, for an IR-prep block of the form

    [θ_inv inversion] → TI → [α_exc excitation, ADC] → (TR − TI recovery)

repeated indefinitely with **perfect transverse spoiling between TRs**.
Derivation: solve Mz_pre (just before inversion) from the recurrence

    Mz_at_TI = (1 − E1) + cos(θ_inv) · E1 · Mz_pre
    Mz_pre   = (1 − E2) + cos(α_exc) · E2 · Mz_at_TI

with E1 = exp(−TI/T1), E2 = exp(−(TR−TI)/T1). For TR → ∞ this collapses
to the conventional non-steady-state form `1 − (1 − cos θ_inv)·exp(−TI/T1)`.
Special cases:
- θ_inv = π,  α_exc = π/2:  Mz = 1 − 2·exp(−TI/T1) + exp(−TR/T1)  (textbook IR)
- TR = ∞:                    Mz = 1 − (1 − cos θ_inv)·exp(−TI/T1)
"""
@inline function steady_state_mz_at_excite(T1::Real, TI::Real, TR::Real,
                                            θ_inv::Real, α_exc::Real)
    T1f = Float64(T1)
    E1  = exp(-Float64(TI) / T1f)
    if !isfinite(TR) || TR <= 0
        # TR → ∞ : full recovery between repetitions (Mz_pre = M0)
        # α_exc has no effect
        # on Mz_at_excite (it only matters via the sin(α_exc) projection
        # into Mxy, which the caller handles).
        return 1.0 - (1.0 - cos(θ_inv)) * E1
    end
    a   = cos(θ_inv)
    b   = cos(α_exc)
    E2  = exp(-(Float64(TR) - Float64(TI)) / T1f)
    num = (1 - E2) + b * E2 * (1 - E1)
    den = 1 - a * b * E1 * E2
    Mz_pre = num / den
    # returns Mz_at_TI
    return 1 - E1 + a * E1 * Mz_pre
end

"""
    transient_mz_at_excite_npe(T1, TI, TR, θ_inv, α_exc; Npe::Int) -> Mz

Mean Mz_at_excite across the `Npe` shots of an IR block that starts from
thermal equilibrium (`Mz_pre[1] = M0 = 1`). This is the closed form that
matches what `ir_se_2d_sequence` actually feeds the simulator: a sequence
of `Npe` IR-SE shots, each separated by TR, with k-space row k acquired
on shot k, reconstructed via IFFT (whose DC term ≈ mean over shots for a
small-pixel sphere). Discussion + derivation: `docs/T1_FIT_AND_KOMA_TESTS.md`
§7.5 / §19.5.1 of `EXPERT_REPORT.md`.

Recurrence:

    Mz_pre[1]    = 1
    for k = 1..Npe:
        Mz_at_TI[k]  = (1 − E1) + cos(θ_inv) · E1 · Mz_pre[k]
        Mz_pre[k+1]  = (1 − E2) + cos(α_exc) · E2 · Mz_at_TI[k]
    return mean(Mz_at_TI[1:Npe])

with E1 = exp(−TI/T1), E2 = exp(−(TR−TI)/T1). Limits:
- `Npe = 1`     ⇒ legacy TR-blind transient form `1 − (1 − cos θ_inv) e^{−TI/T1}`.
- `Npe → ∞`     ⇒ `steady_state_mz_at_excite(T1, TI, TR, θ_inv, α_exc)`.
- `TR → ∞`      ⇒ `Mz_pre[k] ≡ 1` ⇒ pure transient (every shot from M0).
"""
@inline function transient_mz_at_excite_npe(T1::Real, TI::Real, TR::Real,
                                              θ_inv::Real, α_exc::Real;
                                              Npe::Int)
    Npe ≥ 1 || error("Npe must be ≥ 1")
    T1f = Float64(T1)
    E1  = exp(-Float64(TI) / T1f)
    a   = cos(θ_inv)
    if !isfinite(TR) || TR <= 0
        # TR → ∞: Mz_pre is reset to 1 at the start of every shot.
        return 1.0 - (1.0 - a) * E1
    end
    b   = cos(α_exc)
    E2  = exp(-(Float64(TR) - Float64(TI)) / T1f)
    Mz_pre = 1.0
    accum  = 0.0
    @inbounds for _ in 1:Npe
        Mz_at_TI = (1 - E1) + a * E1 * Mz_pre
        accum   += Mz_at_TI
        Mz_pre   = (1 - E2) + b * E2 * Mz_at_TI
    end
    return accum / Npe
end

"""
    fit_t1_generalized_ir(TIs, αs, mags; TRs, α_excs, Npe, T1_range, n_grid,
                          noise_sigma, abs_noise_sigma)

Fit T1 to (TI, α_inv, |S|) triples under an IR forward model

    |S| = |A · Mz_at_excite(T1; TI, TR, θ_inv, α_exc)|

`αs` is the **inversion / preparation** flip angle (e.g. π for canonical
IR, π/2 for SR). The optional `α_excs` gives the **excitation** flip
angle per sample; defaults to π/2 (so post-excitation Mz = 0). The
optional `TRs` gives the per-block repetition time; if omitted, each TR
is taken to be ∞.

`Npe` selects the forward model:
- `Npe === nothing` (default): steady-state IR (`steady_state_mz_at_excite`).
  Back-compat with the original generalized-IR fit.
- `Npe::Int`: finite-Npe transient ramp from M0=1
  (`transient_mz_at_excite_npe`). This matches what
  `ir_se_2d_sequence` actually does in E2 — see `EXPERT_REPORT.md` §19.5.1
  for why this matters under adaptive-action sequences.

A log-spaced T1 grid is scanned; at each candidate T1, the optimal |A|
is the closed-form ratio `sum(m · |y|) / sum(|y|²)`. Returns
`(T1, A, residual, T1_sigma)`.

`sigma_method` selects how `T1_sigma` is computed:
- `:asymptotic` (default): Cramér–Rao from Jacobian local at the LM minimum,
  `Σ ≈ σ²_eff · (J^T J)^-1`. Fast, smooth, but **over-confident on multimodal
  SSE** (the LM minimum is in one basin; asymptotic σ doesn't see other
  basins). Used by all pre-E2.5 callers.
- `:profile_likelihood`: span of T1s whose SSE is within `σ²_eff` of the
  minimum (asymptotic χ²₁ test, conf 0.683). Captures multimodal-SSE basins
  honestly — wide σ on degenerate fits, tight σ on well-determined ones. Use
  this for the σ-channel obs and any reported uncertainty after E2.5. See
  §15 of `cr_explainer.md` for the full derivation and §3 of `E2_5_PLAN.md`.

`σ²_eff = max(σ²_resid, σ²_floor)` where:
- σ²_resid = `best_sse / (n − 2)` for `n > 4`; **`Inf` for `n ≤ 4`**
  (Fix A: the residual variance has no statistical power below 5 samples
  with 2 free parameters, see `docs/T1_FIT_AND_KOMA_TESTS.md` §7.1).
- σ²_floor: pick **one** of the two paths.
  - `abs_noise_sigma` (Fix B): absolute noise σ in magnitude units. Used
    directly as the floor: `σ²_floor = abs_noise_sigma²`. Recommended —
    decouples σ_T1 from the agent's data-RMS-via-action choice
    (`docs/T1_FIT_AND_KOMA_TESTS.md` §7.3).
  - `noise_sigma`: legacy *relative* noise level (fraction of data RMS).
    `σ²_floor = (noise_sigma · rms(m))²`. Kept for back-compat.
  If both are passed, `abs_noise_sigma` wins.

Caller convention: `mags` should already be divided by sin(α_exc) per
sample (see `_e2_update_t1_estimates!`).

Degenerate at length < 2 — throws.
"""
function fit_t1_generalized_ir(TIs::AbstractVector{<:Real},
                               αs::AbstractVector{<:Real},
                               mags::AbstractVector{<:Real};
                               TRs::Union{Nothing,AbstractVector{<:Real}} = nothing,
                               α_excs::Union{Nothing,AbstractVector{<:Real}} = nothing,
                               Npe::Union{Nothing,Int} = nothing,
                               T1_range::NTuple{2,<:Real} = (5e-3, 5.0),
                               n_grid::Int = 300,
                               noise_sigma::Union{Nothing,Real} = nothing,
                               abs_noise_sigma::Union{Nothing,Real} = nothing,
                               sigma_method::Symbol = :asymptotic,
                               n_bootstrap::Int = 100,
                               bootstrap_seed::Int = 0,
                               signed::Bool = false)
    n = length(TIs)
    n == length(αs) == length(mags) ||
        error("TIs, αs, mags must be same length")
    n >= 2 || error("Need ≥2 samples for generalized IR fit")
    TRs    === nothing || length(TRs)    == n || error("TRs length mismatch")
    α_excs === nothing || length(α_excs) == n || error("α_excs length mismatch")
    Npe    === nothing || Npe ≥ 1                 || error("Npe must be ≥ 1")

    ti = Float64.(TIs)
    al = Float64.(αs)                                     # inversion angle
    m  = Float64.(mags)
    tr = TRs    === nothing ? fill(Inf, n) : Float64.(TRs)
    ae = α_excs === nothing ? fill(π/2, n) : Float64.(α_excs)

    # Forward model dispatch — steady-state by default for back-compat.
    Npe_i = Npe === nothing ? 0 : Int(Npe)
    use_transient = Npe_i ≥ 1
    @inline function predict(T1)
        out = Vector{Float64}(undef, n)
        if use_transient
            @inbounds for k in 1:n
                out[k] = transient_mz_at_excite_npe(T1, ti[k], tr[k],
                                                     al[k], ae[k];
                                                     Npe = Npe_i)
            end
        else
            @inbounds for k in 1:n
                out[k] = steady_state_mz_at_excite(T1, ti[k], tr[k],
                                                    al[k], ae[k])
            end
        end
        return out
    end

    T1_candidates = exp.(range(log(T1_range[1]), log(T1_range[2]);
                               length = n_grid))

    # Cache forward predictions per grid point — reused for the LM scan, the
    # asymptotic Jacobian, and (if requested) the bootstrap σ. Each entry is
    # the signed transient Mz vector; abs() is applied on read for the
    # magnitude path.
    predict_cache = Vector{Vector{Float64}}(undef, n_grid)
    @inbounds for (i, T1) in enumerate(T1_candidates)
        predict_cache[i] = predict(T1)
    end

    # Helper: closed-form (A, SSE) at one grid index for a given data vector.
    # Same formula for signed and magnitude paths — only the |·| changes.
    @inline function _A_and_sse(i::Int, m_vec::AbstractVector{Float64})
        y  = predict_cache[i]
        ay = signed ? y : abs.(y)
        den = sum(abs2, ay)
        den > 0 || return (NaN, Inf)
        A = sum(m_vec .* ay) / den
        r = A .* ay .- m_vec
        return (A, sum(abs2, r))
    end

    sse_grid = fill(Inf, n_grid)         # for profile-likelihood σ (§15 of cr_explainer.md)
    best_sse = Inf
    best_T1  = NaN
    best_A   = NaN
    best_idx = 0
    @inbounds for i in 1:n_grid
        A, sse = _A_and_sse(i, m)
        sse_grid[i] = sse
        if sse < best_sse
            best_sse = sse
            best_T1  = T1_candidates[i]
            best_A   = A
            best_idx = i
        end
    end

    # ── Asymptotic σ_T1 via central-difference Jacobian at the optimum ──
    # Magnitude model: model_k(T1, A) = A · |y_k(T1)|.
    #   ∂model_k/∂A  = |y_k|;  ∂model_k/∂T1 ≈ A · sign(y_k) · (y_p − y_m) / (2h).
    # Signed model:    model_k(T1, A) = A · y_k(T1).
    #   ∂model_k/∂A  = y_k;    ∂model_k/∂T1 ≈ A ·              (y_p − y_m) / (2h).
    T1_sigma = NaN
    if isfinite(best_T1) && best_T1 > 0 && isfinite(best_A)
        h      = max(1e-6, 1e-4 * best_T1)
        y0     = predict(best_T1)
        y_p    = predict(best_T1 + h)
        y_m    = predict(best_T1 - h)
        if signed
            dA  = y0
            dT1 = best_A .* (y_p .- y_m) ./ (2h)
        else
            sgn = sign.(y0)
            dA  = abs.(y0)
            dT1 = best_A .* sgn .* (y_p .- y_m) ./ (2h)
        end
        a11    = sum(abs2, dA)
        a22    = sum(abs2, dT1)
        a12    = sum(dA .* dT1)
        det_JtJ = a11 * a22 - a12^2

        # Fix A: n-gate. At n ≤ 4 with p=2 free parameters the residual
        # variance has no statistical power, so we *replace* σ²_eff with
        # the floor instead of `max`-ing — `max(Inf, floor) = Inf` would
        # collapse the σ to ∞ even when an honest floor was provided.
        # Fix B: prefer absolute floor when given (decouples from data RMS).
        σ²_floor = if abs_noise_sigma !== nothing
            Float64(abs_noise_sigma)^2
        elseif noise_sigma !== nothing
            rms_m = sqrt(sum(abs2, m) / n)
            (Float64(noise_sigma) * rms_m)^2
        else
            0.0
        end
        σ²_eff = if n > 4
            σ²_resid = best_sse / (n - 2)
            max(σ²_resid, σ²_floor)
        else
            σ²_floor             # only honest source at small n
        end

        if sigma_method == :profile_likelihood
            # Profile-likelihood σ: span of T1s whose SSE is within the
            # asymptotic-LR threshold of SSE_min. Captures multimodal-SSE
            # basins that asymptotic σ misses *if* both basins lie within
            # `σ²_eff` of SSE_min (§15.6 of `cr_explainer.md`). When LM
            # lands cleanly in a wrong basin, profile-σ stays narrow —
            # that's the case bootstrap σ catches and profile σ doesn't.
            if σ²_eff > 0
                threshold = best_sse + σ²_eff
                pass_mask = sse_grid .≤ threshold
                if any(pass_mask)
                    T1_lo = minimum(T1_candidates[pass_mask])
                    T1_hi = maximum(T1_candidates[pass_mask])
                    T1_sigma = (T1_hi - T1_lo) / 2
                end
            end
        elseif sigma_method == :asymptotic
            if det_JtJ > 0 && σ²_eff > 0
                var_T1 = σ²_eff * a11 / det_JtJ
                T1_sigma = var_T1 > 0 ? sqrt(var_T1) : NaN
            end
        elseif sigma_method == :bootstrap
            # Bootstrap σ — resample residuals, refit each synthetic dataset
            # by re-running the (cached) grid scan, take std of T1*_b. The
            # right tool for wrong-basin multimodal-SSE failures: different
            # bootstrap samples can land in different basins under noise,
            # and that spread *is* the honest σ. Cost: ~n_bootstrap × n_grid
            # closed-form evals, dominated by the closed-form A formula
            # (sub-millisecond per σ estimate at default sizes).
            if isfinite(best_T1) && best_T1 > 0 && n >= 2
                # Predicted signal at the LM optimum (signed or magnitude)
                y_at_best = predict_cache[best_idx]
                ay_at_best = signed ? y_at_best : abs.(y_at_best)
                fitted    = best_A .* ay_at_best
                residuals = m .- fitted

                rng_b = MersenneTwister(bootstrap_seed)
                T1_boots = Vector{Float64}(undef, n_bootstrap)
                @inbounds for b in 1:n_bootstrap
                    # Resample residuals with replacement
                    idxs = rand(rng_b, 1:n, n)
                    m_b  = fitted .+ residuals[idxs]
                    # Refit using cached predict_cache
                    best_sse_b = Inf
                    best_T1_b  = NaN
                    for i in 1:n_grid
                        _, sse_b = _A_and_sse(i, m_b)
                        if sse_b < best_sse_b
                            best_sse_b = sse_b
                            best_T1_b  = T1_candidates[i]
                        end
                    end
                    T1_boots[b] = best_T1_b
                end
                # std of bootstrap T1 estimates — corrected (n-1 denominator)
                valid = filter(isfinite, T1_boots)
                if length(valid) >= 2
                    μ      = sum(valid) / length(valid)
                    var_T1 = sum((x - μ)^2 for x in valid) / (length(valid) - 1)
                    T1_sigma = sqrt(var_T1)
                end
            end
        else
            error("sigma_method must be :asymptotic, :profile_likelihood, " *
                  "or :bootstrap, got :$sigma_method")
        end
    end

    (T1 = best_T1, A = best_A,
     residual = sqrt(best_sse / n),
     T1_sigma = T1_sigma)
end

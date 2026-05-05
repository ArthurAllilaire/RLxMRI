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
        # TR → ∞ : full recovery between repetitions, α_exc has no effect
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
    return 1 - E1 + a * E1 * Mz_pre
end

"""
    fit_t1_generalized_ir(TIs, αs, mags; TRs, α_excs, T1_range, n_grid, noise_sigma)

Fit T1 to (TI, α_inv, |S|) triples under the steady-state IR model

    |S| = |A · Mz_at_excite(T1; TI, TR, θ_inv, α_exc)|

where `Mz_at_excite` is the steady-state longitudinal magnetisation given
by `steady_state_mz_at_excite`. `αs` is the **inversion / preparation**
flip angle (e.g. π for canonical IR, π/2 for SR). The optional `α_excs`
gives the **excitation** flip angle per sample; defaults to π/2 (so
post-excitation Mz = 0, matching most IR implementations). The optional
`TRs` gives the per-block repetition time; if omitted, each TR is taken
to be ∞ (i.e. the legacy "TR ≫ T1, full recovery" assumption — kept for
back-compat with single-shot E1).

When TR is finite and varies between blocks, ignoring it biases the fit
because the magnetisation has not relaxed to M0 before each inversion.
The E2 action space lets the agent pick TR ∈ [0.5, 5] s, comparable to
plate T1s, so passing `TRs` is essential there.

A log-spaced T1 grid is scanned; at each candidate T1, the optimal |A|
is the closed-form ratio `sum(m · |y|) / sum(|y|²)`. Returns
`(T1, A, residual, T1_sigma)`. `T1_sigma` is the asymptotic 1-σ
uncertainty from the model Jacobian (computed by central differences in
T1 to avoid hand-deriving the steady-state derivative):

    Σ ≈ σ²_eff · (J^T J)^-1,    T1_sigma = sqrt(Σ[T1, T1])

`noise_sigma`, if given, is a *relative* noise level (fraction of data
RMS) used as a floor on the residual-based σ² estimate.

Caller convention: `mags` should already be divided by sin(α_exc) per
sample (see `_e2_update_t1_estimates!`), since the model lumps the
sin(α_exc) projection into A. With finite TR the Mz_pre still depends on
cos(α_exc), so `α_excs` must still be supplied even when mags are
sin-corrected.

Degenerate at length < 2 — throws.
"""
function fit_t1_generalized_ir(TIs::AbstractVector{<:Real},
                               αs::AbstractVector{<:Real},
                               mags::AbstractVector{<:Real};
                               TRs::Union{Nothing,AbstractVector{<:Real}} = nothing,
                               α_excs::Union{Nothing,AbstractVector{<:Real}} = nothing,
                               T1_range::NTuple{2,<:Real} = (5e-3, 5.0),
                               n_grid::Int = 300,
                               noise_sigma::Union{Nothing,Real} = nothing)
    n = length(TIs)
    n == length(αs) == length(mags) ||
        error("TIs, αs, mags must be same length")
    n >= 2 || error("Need ≥2 samples for generalized IR fit")
    TRs    === nothing || length(TRs)    == n || error("TRs length mismatch")
    α_excs === nothing || length(α_excs) == n || error("α_excs length mismatch")

    ti = Float64.(TIs)
    al = Float64.(αs)                                     # inversion angle
    m  = Float64.(mags)
    tr = TRs    === nothing ? fill(Inf, n) : Float64.(TRs)
    ae = α_excs === nothing ? fill(π/2, n) : Float64.(α_excs)

    @inline predict(T1) = [steady_state_mz_at_excite(T1, ti[k], tr[k], al[k], ae[k])
                           for k in 1:n]

    T1_candidates = exp.(range(log(T1_range[1]), log(T1_range[2]);
                               length = n_grid))

    best_sse = Inf
    best_T1  = NaN
    best_A   = NaN
    for T1 in T1_candidates
        y  = predict(T1)
        ay = abs.(y)
        den = sum(abs2, ay)
        den > 0 || continue
        A = sum(m .* ay) / den           # closed-form |A|
        r = A .* ay .- m
        sse = sum(abs2, r)
        if sse < best_sse
            best_sse = sse
            best_T1  = T1
            best_A   = A
        end
    end

    # ── Asymptotic σ_T1 via central-difference Jacobian at the optimum ──
    # model_k(T1, A) = A · |y_k(T1)|, with y_k from steady_state_mz_at_excite.
    # ∂model_k/∂A  = |y_k|
    # ∂model_k/∂T1 ≈ A · sign(y_k) · (y_k(T1+h) − y_k(T1−h)) / (2h)
    T1_sigma = NaN
    if isfinite(best_T1) && best_T1 > 0 && isfinite(best_A)
        h      = max(1e-6, 1e-4 * best_T1)
        y0     = predict(best_T1)
        y_p    = predict(best_T1 + h)
        y_m    = predict(best_T1 - h)
        sgn    = sign.(y0)
        dA     = abs.(y0)
        dT1    = best_A .* sgn .* (y_p .- y_m) ./ (2h)
        a11    = sum(abs2, dA)
        a22    = sum(abs2, dT1)
        a12    = sum(dA .* dT1)
        det_JtJ = a11 * a22 - a12^2

        σ²_resid = n > 2 ? best_sse / (n - 2) : 0.0
        σ²_floor = if noise_sigma === nothing
            0.0
        else
            rms_m = sqrt(sum(abs2, m) / n)
            (Float64(noise_sigma) * rms_m)^2
        end
        σ²_eff = max(σ²_resid, σ²_floor)

        if det_JtJ > 0 && σ²_eff > 0
            var_T1 = σ²_eff * a11 / det_JtJ
            T1_sigma = var_T1 > 0 ? sqrt(var_T1) : NaN
        end
    end

    (T1 = best_T1, A = best_A,
     residual = sqrt(best_sse / n),
     T1_sigma = T1_sigma)
end

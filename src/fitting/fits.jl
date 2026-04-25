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
    fit_t1_generalized_ir(TIs, αs, mags; T1_range, n_grid)

Fit T1 to (TI, α, |S|) triples under the generalized-IR model

    |S| = |A · (1 − (1 − cos α) · exp(−TI / T1))|

A log-spaced T1 grid is scanned; at each candidate T1, the optimal |A| is
the closed-form ratio `sum(m · |y|) / sum(|y|²)`. Returns `(T1, A,
residual)`. Works for any mix of α values (e.g. IR at 180°, SR at 90°,
small-tip at 10°) because the sign pattern of `y` is fully determined by
T1 and α — no magnitude-null ambiguity like the pure-IR case.

Degenerate at length < 2 — throws.
"""
function fit_t1_generalized_ir(TIs::AbstractVector{<:Real},
                               αs::AbstractVector{<:Real},
                               mags::AbstractVector{<:Real};
                               T1_range::NTuple{2,<:Real} = (5e-3, 5.0),
                               n_grid::Int = 300)
    length(TIs) == length(αs) == length(mags) ||
        error("TIs, αs, mags must be same length")
    length(TIs) >= 2 || error("Need ≥2 samples for generalized IR fit")

    ti = Float64.(TIs)
    al = Float64.(αs)
    m  = Float64.(mags)

    T1_candidates = exp.(range(log(T1_range[1]), log(T1_range[2]);
                               length = n_grid))

    best_sse = Inf
    best_T1  = NaN
    best_A   = NaN
    for T1 in T1_candidates
        y = @. 1 - (1 - cos(al)) * exp(-ti / T1)
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

    (T1 = best_T1, A = best_A, residual = sqrt(best_sse / length(ti)))
end

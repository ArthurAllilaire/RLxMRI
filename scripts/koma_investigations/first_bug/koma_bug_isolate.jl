# Isolate the time-axis bug: does `points_from_key_times` produce a different
# number of samples for a 1 ms RF pulse depending on its absolute start time?

const MIN_RISE_TIME = 1e-14

function points_from_key_times(times; dt)
    t = Float64[]
    for i = 1:length(times)-1
        if dt < times[i+1] - times[i]
            taux = collect(range(times[i], times[i+1]; step=dt))
        else
            taux = [times[i], times[i+1]]
        end
        append!(t, taux)
    end
    return t
end

# A 1ms RF pulse at start time t0. delay=0, T=1ms.
function rf_samples(t0; T_rf = 1e-3, Δt_rf = 5e-5)
    ϵ = MIN_RISE_TIME
    t1 = t0
    t2 = t1 + T_rf
    tc = t1 + T_rf / 2
    raw = sort([t1, t1 + ϵ, tc, t2 - ϵ, t2])
    points = points_from_key_times(raw; dt=Δt_rf)
    return length(points), length(unique(points))
end

println("Effect of absolute start time on RF discretization (1ms pulse, Δt_rf=5e-5)")
println("expected: should always be the same")
println()
println("t0 [s]         | nraw   nunique   eps(t0)")
println("---------------+----------------------------------")
for t0 in (0.0, 1.0, 3.0, 8.0, 11.0, 50.0, 67.0, 67.011, 75.020, 99.020, 200.0, 1000.0)
    nraw, nu = rf_samples(t0)
    println("  $(rpad(t0,12))  | $(lpad(nraw,4))   $(lpad(nu,7))   $(eps(t0))")
end

# Now also test the issue with ϵ getting absorbed by fp.
println()
println("MIN_RISE_TIME ϵ visibility at t0:")
println("t0       | t0+ϵ == t0?   | t0+ϵ - t0")
for t0 in (0.0, 1.0, 10.0, 60.0, 100.0, 1000.0)
    println("  $(rpad(t0,7))| $(t0 + MIN_RISE_TIME == t0) | $(t0 + MIN_RISE_TIME - t0)")
end

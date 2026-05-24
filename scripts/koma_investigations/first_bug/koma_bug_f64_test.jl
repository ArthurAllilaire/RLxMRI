# Decisive test: rerun the multi-shot bug with sim_params=Dict("precision"=>"f64").
# Hypothesis: the per-shot drift is from Float32 precision in the discretised
# sequence (seq.t cumulative time + Δt) losing resolution once total sim time
# exceeds ~70 s. If f64 fixes it, that's the root cause.

using QalibreMDPhantom, KomaMRI, Suppressor
using KomaMRI: Grad

const FOV = 0.2
const Nfe = 64
const Npe = 32

function zero_gy!(seq)
    for i in 1:length(seq)
        if abs(seq[i].GR[2, 1].A) > 1e-15
            seq[i].GR[2, 1] = Grad(0.0, seq[i].GR[2, 1].T)
        end
    end
    seq
end

println("KomaMRI version: ", pkgversion(KomaMRI))

phantom_one = KomaMRI.Phantom(
    name = "ball",
    x = [0.02], y = [0.0], z = [0.0],
    T1 = [1.0], T2 = [1.0], T2s = [1.0], ρ = [1.0], Δw = [0.0],
)

for precision in ("f32", "f64")
    println("\n── precision = $precision, TR = 8 s, 16 shots ──")
    println("  k    |d[mid]|     ratio vs k=1")
    seq = Suppressor.@suppress ir_se_2d_sequence(
        3.0, 0.02, 8.0; α_exc = π/2, FOV = FOV, Nfe = Nfe, Npe = 16,
    )
    zero_gy!(seq)
    raw = Suppressor.@suppress simulate(
        phantom_one, seq, Scanner();
        sim_params = Dict{String,Any}("precision" => precision),
    )
    ref = abs(raw.profiles[1].data[Nfe ÷ 2, 1])
    for k in 1:16
        dm = abs(raw.profiles[k].data[Nfe ÷ 2, 1])
        marker = abs(dm/ref - 1) > 0.05 ? "  ← DRIFT" : ""
        println("  $(lpad(k,2))   $(rpad(round(dm, digits=4),11)) $(round(dm/max(ref,1e-9), digits=3))$marker")
    end
end

# Also test the long-TR case where drift starts very early in f32.
for precision in ("f32", "f64")
    println("\n── precision = $precision, TR = 30 s, 8 shots (sim time 240 s) ──")
    println("  k    |d[mid]|     ratio vs k=1")
    seq = Suppressor.@suppress ir_se_2d_sequence(
        3.0, 0.02, 30.0; α_exc = π/2, FOV = FOV, Nfe = Nfe, Npe = 8,
    )
    zero_gy!(seq)
    raw = Suppressor.@suppress simulate(
        phantom_one, seq, Scanner();
        sim_params = Dict{String,Any}("precision" => precision),
    )
    ref = abs(raw.profiles[1].data[Nfe ÷ 2, 1])
    for k in 1:8
        dm = abs(raw.profiles[k].data[Nfe ÷ 2, 1])
        marker = abs(dm/ref - 1) > 0.05 ? "  ← DRIFT" : ""
        println("  $(lpad(k,2))   $(rpad(round(dm, digits=4),11)) $(round(dm/max(ref,1e-9), digits=3))$marker")
    end
end

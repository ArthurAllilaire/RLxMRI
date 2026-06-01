# Does the bug survive at Koma's default Δt_rf = 1e-5, or is it just an
# artefact of our coarser 5e-5 stepping? Single-voxel, identical shots,
# scan TR and Δt_rf jointly.

using MRISystemPhantom, KomaMRI, Suppressor
using KomaMRI: Grad

const FOV = 0.2
const Nfe = 64

function zero_gy!(seq)
    for i in 1:length(seq)
        if abs(seq[i].GR[2, 1].A) > 1e-15
            seq[i].GR[2, 1] = Grad(0.0, seq[i].GR[2, 1].T)
        end
    end
    seq
end

phantom_one = KomaMRI.Phantom(
    name="ball", x=[0.02], y=[0.0], z=[0.0],
    T1=[1.0], T2=[1.0], T2s=[1.0], ρ=[1.0], Δw=[0.0],
)

function run_case(TR, Npe, Δt_rf)
    seq = Suppressor.@suppress ir_se_2d_sequence(
        3.0, 0.02, TR; α_exc=π/2, FOV=FOV, Nfe=Nfe, Npe=Npe,
    )
    zero_gy!(seq)
    raw = Suppressor.@suppress simulate(
        phantom_one, seq, Scanner();
        sim_params=Dict{String,Any}("Δt_rf" => Δt_rf),
    )
    vals = [abs(raw.profiles[k].data[Nfe÷2, 1]) for k in 1:Npe]
    ref = vals[1]
    # first shot where the value diverges from the running median by > 1%
    first_bad = nothing
    for k in 2:Npe
        if abs(vals[k] - vals[k-1]) / ref > 0.01
            first_bad = k; break
        end
    end
    max_dev = maximum(abs.(vals .- ref) ./ ref) * 100  # %
    return vals, first_bad, max_dev
end

println("Δt_rf sweep on single-voxel identical-shot test")
println("(bug present ⇔ first_bad ≠ nothing, max_dev > ~1%)")
println()
println("Δt_rf [s] | TR [s] | Npe | sim time | first wrong | max dev [%]")
println("──────────┼────────┼─────┼──────────┼─────────────┼────────────")
for Δt_rf in (1e-5, 2e-5, 5e-5)
    for (TR, Npe) in ((4.0, 16), (8.0, 16), (15.0, 16), (30.0, 8))
        vals, fb, dev = run_case(TR, Npe, Δt_rf)
        fb_str = fb === nothing ? "  none  " : "shot $fb"
        println("  $(rpad(Δt_rf,7))| $(rpad(TR,6))| $(rpad(Npe,3))| $(rpad(TR*Npe,8))s| $(rpad(fb_str,12))| $(round(dev, digits=2))")
    end
    println()
end

# Show the full shot trace for the most informative case at default Δt_rf.
println("Full trace at Δt_rf=1e-5 (default), TR=15 s, Npe=16 (sim time 240 s):")
vals, fb, dev = run_case(15.0, 16, 1e-5)
for (k, v) in enumerate(vals)
    flag = k > 1 && abs(v - vals[k-1]) / vals[1] > 0.01 ? "  ← jump" : ""
    println("  shot $(lpad(k,2)): $(round(v, digits=5))$flag")
end

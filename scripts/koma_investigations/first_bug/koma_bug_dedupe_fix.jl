# Candidate root-cause fix for the multi-shot drift.
#
# Root cause (confirmed by koma_bug_b1leak.jl): in `get_variable_times`, the
# `sort!(unique!(t))` step uses bit-exact equality. Time points that are
# nominally identical (e.g. end-of-RF-block = start-of-next-block) are
# computed via two different fp paths (one via cumsum(DUR), one via
# t0 + delay + T), and at non-zero absolute times they differ by 1-2 ULPs.
# These near-duplicates survive `unique!`. After linear interpolation, both
# end up just-inside the RF support with full envelope amplitude, so
# `get_sim_ranges` (which uses `sum(abs.(B1)) > 0.0` to classify RF blocks)
# pulls the extra sample into the RF block. One extra ~5e-5 s rotation step
# over-rotates the refocus pulse and corrupts the subsequent state.
#
# Candidate fix: collapse near-duplicate times after sort with a tolerance.

using QalibreMDPhantom, KomaMRI, Suppressor
const KomaMRIBase = KomaMRI.KomaMRIBase
using KomaMRI: Grad

# Override discretize to insert a near-duplicate-collapse pass after
# get_variable_times. We rebuild seqd from scratch using the deduped time
# vector — same path as KomaMRIBase.discretize.
const KMB = KomaMRIBase
function patched_discretize(seq::KMB.Sequence;
                            sampling_params=KMB.default_sampling_params(),
                            motion=KMB.NoMotion())
    t, _ = KMB.get_variable_times(seq;
        Δt=sampling_params["Δt"], Δt_rf=sampling_params["Δt_rf"], motion=motion)
    # === FIX: collapse near-duplicate time points (within 1 ns) ===
    tol = 1e-12
    keep = trues(length(t))
    for i in 2:length(t)
        if t[i] - t[i-1] < tol
            keep[i] = false
        end
    end
    t = t[keep]
    Δt = t[2:end] .- t[1:end-1]
    # === end FIX ===
    B1, Δf, ψ  = KMB.get_rfs(seq, t)
    Gx, Gy, Gz = KMB.get_grads(seq, t)
    tadc = KMB.get_adc_sampling_times(seq)
    tadc_set = Set(tadc)
    ADCflag = [tt in tadc_set for tt in t]
    return KMB.DiscreteSequence(Gx, Gy, Gz, complex.(B1), Δf, ψ, ADCflag, t, Δt)
end
KMB.discretize(seq::KMB.Sequence; kw...) = patched_discretize(seq; kw...)

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

for (TR, Npe) in ((8.0, 16), (30.0, 8))
    println("\n── TR=$TR s, Npe=$Npe (sim time $(TR*Npe) s) ──")
    seq = Suppressor.@suppress ir_se_2d_sequence(3.0, 0.02, TR; α_exc=π/2, FOV=0.2, Nfe=64, Npe=Npe)
    zero_gy!(seq)
    raw = Suppressor.@suppress simulate(phantom_one, seq, Scanner())
    ref = abs(raw.profiles[1].data[32, 1])
    println("  k    |d[mid]|     ratio vs k=1")
    for k in 1:Npe
        dm = abs(raw.profiles[k].data[32, 1])
        mark = abs(dm/ref - 1) > 0.05 ? "  ← DRIFT" : ""
        println("  $(lpad(k,2))   $(rpad(round(dm, digits=4),11)) $(round(dm/max(ref,1e-9), digits=3))$mark")
    end
end

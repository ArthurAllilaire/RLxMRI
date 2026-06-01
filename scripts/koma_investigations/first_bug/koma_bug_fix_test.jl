# Test a candidate fix: detect RF blocks with a non-trivial threshold on B1,
# instead of `> 0.0`. Hypothesis: linear-interpolated B1 at time-axis samples
# that should be exactly at the RF support boundary picks up tiny non-zero
# values once the absolute time gets large enough, due to floating-point
# rounding in the time vector. Those "leaked" RF samples get added to the
# excitation block by `get_sim_ranges`, corrupting subsequent magnetisation.

using MRISystemPhantom, KomaMRI, Suppressor
const KomaMRICore = KomaMRI.KomaMRICore
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

# Patched: use a threshold rather than `> 0.0`. The RF envelope peak is on the
# order of 1 (units of T), so any threshold from 1e-12 to 1e-6 should be safe.
function patched_get_sim_ranges(seqd; max_block_length=Inf, max_rf_block_length=Inf)
    ranges = UnitRange{Int}[]
    ranges_bool = Bool[]
    start_idx_rf_block = 0
    start_idx_gr_block = 0
    N = length(seqd.Δt)
    rf_threshold = 1e-9  # was: 0.0
    for i in eachindex(seqd.Δt)
        is_rf = sum(abs.(seqd.B1[i:i+1])) > rf_threshold
        if is_rf
            if start_idx_rf_block == 0
                start_idx_rf_block = i
            end
            if start_idx_gr_block > 0
                split_ranges = KomaMRICore.split_range(start_idx_gr_block:i, max_block_length)
                append!(ranges, split_ranges)
                append!(ranges_bool, fill(false, length(split_ranges)))
                start_idx_gr_block = 0
            end
        else
            if start_idx_gr_block == 0
                start_idx_gr_block = i
            end
            if start_idx_rf_block > 0
                split_ranges = KomaMRICore.split_range(start_idx_rf_block:i, max_rf_block_length)
                append!(ranges, split_ranges)
                append!(ranges_bool, fill(true, length(split_ranges)))
                start_idx_rf_block = 0
            end
        end
    end
    if start_idx_rf_block > 0
        split_ranges = KomaMRICore.split_range(start_idx_rf_block:N, max_rf_block_length)
        append!(ranges, split_ranges)
        append!(ranges_bool, fill(true, length(split_ranges)))
    end
    if start_idx_gr_block > 0
        split_ranges = KomaMRICore.split_range(start_idx_gr_block:N, max_block_length)
        append!(ranges, split_ranges)
        append!(ranges_bool, fill(false, length(split_ranges)))
    end
    return ranges, ranges_bool
end

# Monkey-patch
KomaMRICore.get_sim_ranges(seqd::KomaMRICore.DiscreteSequence;
                          max_block_length=Inf, max_rf_block_length=Inf) =
    patched_get_sim_ranges(seqd; max_block_length, max_rf_block_length)

phantom_one = KomaMRI.Phantom(
    name = "ball", x = [0.02], y = [0.0], z = [0.0],
    T1 = [1.0], T2 = [1.0], T2s = [1.0], ρ = [1.0], Δw = [0.0],
)

println("─── With patched get_sim_ranges (threshold 1e-9) ───")
println()

for (TR, label) in ((8.0, "TR=8s, 16 shots (128s sim time)"), (30.0, "TR=30s, 8 shots (240s)"))
    println("── $label ──")
    Npe_test = TR == 8.0 ? 16 : 8
    seq = Suppressor.@suppress ir_se_2d_sequence(
        3.0, 0.02, TR; α_exc = π/2, FOV = FOV, Nfe = Nfe, Npe = Npe_test,
    )
    zero_gy!(seq)
    raw = Suppressor.@suppress simulate(phantom_one, seq, Scanner())
    ref = abs(raw.profiles[1].data[Nfe ÷ 2, 1])
    println("  k    |d[mid]|     ratio vs k=1")
    for k in 1:Npe_test
        dm = abs(raw.profiles[k].data[Nfe ÷ 2, 1])
        marker = abs(dm/ref - 1) > 0.05 ? "  ← DRIFT" : ""
        println("  $(lpad(k,2))   $(rpad(round(dm, digits=4),11)) $(round(dm/max(ref,1e-9), digits=3))$marker")
    end
    println()
end

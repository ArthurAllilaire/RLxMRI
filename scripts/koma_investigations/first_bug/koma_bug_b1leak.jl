# Locate the exact leaked-B1 sample at shot 9's refocus pulse.

using QalibreMDPhantom, KomaMRI, Suppressor
const KomaMRIBase = KomaMRI.KomaMRIBase
using KomaMRI: Grad

function zero_gy!(seq)
    for i in 1:length(seq)
        if abs(seq[i].GR[2, 1].A) > 1e-15
            seq[i].GR[2, 1] = Grad(0.0, seq[i].GR[2, 1].T)
        end
    end
    seq
end

seq = Suppressor.@suppress ir_se_2d_sequence(3.0, 0.02, 8.0; α_exc=π/2, FOV=0.2, Nfe=64, Npe=16)
zero_gy!(seq)
seqd = KomaMRIBase.discretize(seq)

# Shot 1 refoc is around t≈3.011, shot 9 refoc around t≈67.02.
function dump_window(t_centre; halfwin=0.001)
    idx = findall(t -> abs(t - t_centre) < halfwin, seqd.t)
    println("\nWindow around t=$t_centre s ($(length(idx)) samples):")
    println("  idx     |   t (s)               |   B1 (T)               |   Gx       Gy")
    for i in idx
        b1 = abs(seqd.B1[i])
        gx = seqd.Gx[i]; gy = seqd.Gy[i]
        b1_marker = b1 > 0 ? " ←RF" : ""
        println("  $(lpad(i,5))   | $(rpad(seqd.t[i], 22))| $(rpad(b1, 22))| $(round(gx, digits=5))   $(round(gy, digits=5))$b1_marker")
    end
end

println("=== Shot 1 refocus (~t=3.011) ===")
dump_window(3.011)
println("\n=== Shot 9 refocus (~t=67.010) ===")
dump_window(67.010)

# Probe leftover transverse magnetisation between phase-encode shots.
#
# Builds the IR-SE 2D sequence truncated to K phase-encode shots (with the
# worst-case manual-schedule block: TI = 2.8 s, TR = 5 s) and reads back the
# final Mag state from KomaMRI (sim_params["return_type"] = "state"). For each
# sphere we report mean |Mxy| / ρ — the residual transverse fraction at the
# end of the readout window of shot K.
#
# Rough analytical estimate (decay-only, ignoring RF rotations) for sphere k:
#   |Mxy_residual| ≈ exp(-TR / T2_k)
# Worst case sphere (T1-array :T15, T2 = 1.542 s, TR = 5 s) ≈ 3.9 %.
#
# Usage:
#   julia --project=. scripts/probe_residual_mxy.jl
#   julia --project=. scripts/probe_residual_mxy.jl --TI 2.8 --TR 5.0 --Ks 1,4,16,32
# TODO: add this check whenever I do run ir se 2d as will want to know per sphere what the error is saying
using MRISystemPhantom, KomaMRI, Suppressor
using Random, Statistics, Printf

const FOV      = 0.2
const VOXEL_MM = 1.0
const TE       = 0.020
const ALPHA    = π / 2

# CLI
TI_block = 2.8
TR_block = 5.0
Ks       = [1, 4, 16, 32]
Npe_full = 32
Nfe      = 64
let i = 1
    while i <= length(ARGS)
        a = ARGS[i]
        if a == "--TI"; global TI_block = parse(Float64, ARGS[i+1]); i += 2
        elseif a == "--TR"; global TR_block = parse(Float64, ARGS[i+1]); i += 2
        elseif a == "--Ks"; global Ks = parse.(Int, split(ARGS[i+1], ",")); i += 2
        elseif a == "--npe"; global Npe_full = parse(Int, ARGS[i+1]); i += 2
        elseif a == "--nfe"; global Nfe = parse(Int, ARGS[i+1]); i += 2
        else; i += 1; end
    end
end

println("="^70)
println("Residual |Mxy| probe — TI=$TI_block s  TR=$TR_block s  Nfe=$Nfe  Ks=$Ks")
println("="^70)

# Phantom
cfg     = PhantomConfig(field = :T15, voxel_size_mm = VOXEL_MM, include_plates = [:T1])
phantom = build_phantom(cfg)
descs   = sphere_descriptors(:T1, cfg; rng = MersenneTwister(0))
n_sph   = length(descs)
println("Phantom: $(length(phantom.x)) spins  /  $n_sph spheres (1.5 T)")

# Map each phantom spin to its sphere (0 = background / unassigned)
sphere_of = zeros(Int, length(phantom.x))
for (i, d) in enumerate(descs)
    cx, cy, cz = d.centre
    r2 = d.radius^2
    @inbounds for j in eachindex(phantom.x)
        sphere_of[j] != 0 && continue
        dx = phantom.x[j] - cx
        dy = phantom.y[j] - cy
        dz = phantom.z[j] - cz
        if dx*dx + dy*dy + dz*dz <= r2
            sphere_of[j] = i
        end
    end
end
sphere_counts = [count(==(i), sphere_of) for i in 1:n_sph]
println("Spins per sphere: ", sphere_counts)

# T2 table for analytical comparison
T2_table = [1.542, 1.196, 0.8717, 0.6461, 0.4575, 0.3353, 0.2384, 0.1706,
            0.1216, 0.0837, 0.0592, 0.0426, 0.0304, 0.0213]

# Helper: build IR-SE sequence with K phase-encode shots (full Npe_full per ky
# grid; we just keep the first K shots of the sequence so the gradients are
# representative of what e2 actually fires).
function k_shot_sequence(K::Int)
    # Easiest: build the full sequence then drop trailing blocks. ir_se_2d_sequence
    # builds one shot at a time concatenated with +. We rebuild a shorter version.
    seq = Suppressor.@suppress ir_se_2d_sequence(
        TI_block, TE, TR_block;
        α_exc = ALPHA, FOV = FOV, Nfe = Nfe, Npe = K,
    )
    return seq
end

# Run simulate with return_type = "state" → returns the final Mag vector
function final_mag(seq)
    sim_params = Dict{String,Any}("return_type" => "state")
    Suppressor.@suppress simulate(phantom, seq, scanner_for_field(cfg); sim_params)
end

t_decay = TR_block - TI_block - TE   # free-decay window between echo K and inversion K+1
println("\nFree-decay window between last echo and next inversion ≈ $(round(t_decay, digits=3)) s")
println("\n  K   sphere  T1      T2      ⟨|Mxy|⟩/ρ   exp(-(TR-TI-TE)/T2)")
println("  " * "─"^70)

for K in Ks
    seq = k_shot_sequence(K)
    Xt  = final_mag(seq)
    mxy_abs = abs.(Xt.xy)

    for i in 1:n_sph
        mask = sphere_of .== i
        any(mask) || continue
        ρ_i   = mean(phantom.ρ[mask])
        mxy_i = mean(mxy_abs[mask]) / ρ_i
        T2_i  = T2_table[i]
        rough = exp(-t_decay / T2_i)
        @printf("  %2d   T1-%-2d   %.4f  %.4f  %9.3e   %9.3e\n",
                K, i, descs[i].T1, T2_i, mxy_i, rough)
    end
    println("  " * "─"^70)
end

println("\nState is sampled at the end of shot K's TR delay (= just before the next")
println("inversion). The rough estimate exp(-(TR-TI-TE)/T2) is upper-bounded by")
println("the Mxy that existed at the echo decaying freely — Koma includes RF.")

# Gate 6.1 — KomaMRI side: simulate IR signal at user-supplied TIs for a single
# (T1, T2) cell. Outputs JSON: {"TIs": [...], "mag": [...]}.
#
# Usage: julia --project=. python/gate/run_koma_ir.jl <T1> <T2> <out_json> <TI1,TI2,...>

using JSON
include(joinpath(@__DIR__, "..", "..", "src", "QalibreMDPhantom.jl"))
using .QalibreMDPhantom

T1   = parse(Float64, ARGS[1])
T2   = parse(Float64, ARGS[2])
out  = ARGS[3]
TIs  = parse.(Float64, split(ARGS[4], ","))

mags = Float64[]
for TI in TIs
    # Use small amp_T (default 2e-6) hard pulses; matches MRzero's instantaneous pulse
    m = QalibreMDPhantom.measure_ir_signal(; T1 = T1, T2 = T2, TI = TI,
                                            amp_T = 2e-6,
                                            n_adc = 1, dur_adc = 1e-4)
    push!(mags, m)
end

open(out, "w") do io
    JSON.print(io, Dict("T1" => T1, "T2" => T2, "TIs" => TIs, "mag" => mags))
end
println("wrote ", out)

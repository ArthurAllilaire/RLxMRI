# Dump one shot of the E2 IR-SE block (the sequence the RL policy controls) to
# JSON for fig3_figure.py. The waveforms are sampled from the real KomaMRI
# Sequence that `_e2_simulate_step` (julia/rl/e2.jl) simulates, at the TI and TR
# of block 2 of the hist run's validation scans (the fixed opening probe).
#
# Run from the RLxMRI repo root:
#   julia --project=. ismrm_latex/vancouver/scripts/fig3_sequence_dump.jl

using JSON
using KomaMRI
using MRISystemPhantom

const ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
const RUN = joinpath(ROOT, "runs", "e2", "mf_runB_5sphere_hist_560s_gpu")
const OUT = joinpath(@__DIR__, "..", "figures", "fig3_sequence_waveforms.json")
const BLOCK = 2

env_kwargs = JSON.parsefile(joinpath(RUN, "run_config.json"))["base_env_kwargs"]
Nfe = Int(env_kwargs["Nfe"])
Npe = Int(env_kwargs["Npe"])
# run_config.json carries no FOV, so E2Env used its keyword default (julia/rl/e2.jl:241).
FOV = Float64(get(env_kwargs, "FOV", 0.2))

episodes = JSON.parsefile(joinpath(RUN, "global_best", "diagnostics", "episodes.json"))
TI, TE, TR = (Float64(episodes[1][k][BLOCK]) for k in ("TI", "TE", "TR"))
alpha_deg = Float64(episodes[1]["alpha_deg"][BLOCK])
same_in_all = all(isapprox(Float64(ep[k][BLOCK]), v; rtol = 1e-6)
                  for ep in episodes for (k, v) in (("TI", TI), ("TE", TE), ("TR", TR)))

seq = ir_se_2d_sequence(TI, TE, TR; α_exc = deg2rad(alpha_deg), FOV = FOV, Nfe = Nfe, Npe = Npe)
length(seq) % Npe == 0 || error("sequence length $(length(seq)) is not a multiple of Npe = $Npe")
nb = length(seq) ÷ Npe
T0 = KomaMRIBase.get_block_start_times(seq)

# Shot 1 plus the first block of shot 2, so the next inversion marks the end of TR.
s = KomaMRIBase.get_samples(seq, 1:(nb + 1))

centre(i) = T0[i] + seq.DUR[i] / 2
rf_blocks = [i for i in 1:nb if KomaMRIBase.is_RF_on(seq[i])]
adc_blocks = [i for i in 1:nb if KomaMRIBase.is_ADC_on(seq[i])]
length(rf_blocks) == 3 && length(adc_blocks) == 1 ||
    error("expected 3 RF blocks and 1 ADC block per shot, got $(length(rf_blocks)) and $(length(adc_blocks))")

flip_deg(i) = 360 * KomaMRIBase.γ * abs(seq.RF[1, i].A) * seq.RF[1, i].T
# Shot 1 encodes the outermost ky line, so its phase-encode lobe is nonzero.
pe_block = findfirst(i -> !iszero(seq.GR[2, i].A), 1:nb)
gy_per_shot = [Float64(seq.GR[2, (k - 1) * nb + pe_block].A) for k in 1:Npe]

wave(e) = Dict("t_s" => collect(Float64, e.t), "A" => collect(Float64, abs.(e.A)))
wave_signed(e) = Dict("t_s" => collect(Float64, e.t), "A" => collect(Float64, e.A))

out = Dict(
    "source" => Dict(
        "run" => relpath(RUN, ROOT),
        "episodes_block_index_1based" => BLOCK,
        "same_TI_TE_TR_in_all_validation_scans" => same_in_all,
        "builder" => "MRISystemPhantom.ir_se_2d_sequence",
    ),
    "TI_s" => TI, "TE_s" => TE, "TR_s" => TR, "alpha_deg" => alpha_deg,
    "Nfe" => Nfe, "Npe" => Npe, "FOV_m" => FOV,
    "shot_duration_s" => T0[nb + 1],
    "rf" => wave(s.rf),
    "gx" => wave_signed(s.gx),
    "gy" => wave_signed(s.gy),
    "adc_t_s" => collect(Float64, s.adc.t),
    "rf_events" => [Dict("centre_s" => centre(i), "dur_s" => seq.DUR[i], "flip_deg" => flip_deg(i))
                    for i in rf_blocks],
    "next_inversion" => Dict("centre_s" => centre(nb + 1), "dur_s" => seq.DUR[nb + 1]),
    "adc" => Dict("centre_s" => centre(only(adc_blocks)), "dur_s" => seq.DUR[only(adc_blocks)],
                  "n_samples" => length(s.adc.t)),
    "gy_pe_amplitude_per_shot_T_per_m" => gy_per_shot,
    "pe_block" => Dict("start_s" => T0[pe_block], "dur_s" => seq.DUR[pe_block]),
)

mkpath(dirname(OUT))
open(OUT, "w") do io
    JSON.print(io, out, 1)
end
println("TI = $(round(TI; digits = 4)) s, TE = $TE s, TR = $(round(TR; digits = 4)) s, ",
        "identical in all scans: $same_in_all, $nb blocks per shot")
println("RF flips: ", [round(flip_deg(i); digits = 1) for i in rf_blocks])
println("wrote $(normpath(OUT))")

# # Live demo — random phantom → quantitative T1 recovery
#
# Sample a phantom with random tissue T1 values, image it with a short inversion-
# recovery sweep, fit T1 per sphere, and show the digital twin recovers the
# right quantitative values.
#
#   julia --project=. examples/demo_presentation.jl       # random
#   julia --project=. examples/demo_presentation.jl 7     # fixed seed 7
#
# ⚠ Run once before the talk to warm the JIT, then re-run live.

using MRISystemPhantom, KomaMRI, Distributions, Printf, Statistics, Suppressor, Random

seed = isempty(ARGS) ? rand(1:10_000) : parse(Int, ARGS[1])
nTI, Npe, FOV, TR = 7, 64, 0.2, 3.0     # IR-SE: TR finite, so fitter must know it

# 1 — Sample a phantom: each T1-plate sphere gets a fresh uniform-random T1, with
#     T2 kept at its nominal T2/T1 ratio; random in-plane pose. Save it as HTML.
rpcfg = RandomPhantomConfig(
    base = PhantomConfig(field = :T3, voxel_size_mm = 2.0,
        include_plates = [:T1, :water], slice_thickness_mm = 10.0,
        slice_center_mm = (0.0, 0.0, PLATE_Z_MM.T1)),
    material_sampler = MaterialDistributionSampler(
        T1 = PerPlate(:T1 => Uniform(0.2, 2.5)),         # random T1 ∈ [0.2, 2.5] s
        T2 = PerPlate(:T1 => PreserveNominalRatio(:T2, :T1))),
    pose_sampler = InPlanePoseSampler(rotation_sigma_rad = 0.20, translation_sigma_mm = 5.0))

samp = sample_phantom(rpcfg; rng_seed = seed)
obj, cfg, truth = samp.phantom, samp.cfg, samp.truth
descs = imaged_descriptors(truth.descriptors_sampled[:T1], cfg)   # pose + units applied by the lib
px = sphere_descriptor_pixels(descs, Npe, Npe, FOV)

html = joinpath(@__DIR__, "demo_phantom_seed$(seed).html")
plot_phantom_html(cfg; color_by = :T1, file = html)
println("\nRandom phantom: seed $seed — $(length(descs)) T1 spheres → $html")

# 2 — Inversion-recovery sweep: live Bloch sim at each (log-spaced) TI.
TIs = exp.(range(log(0.02), log(2.8); length = nTI))
@printf "Simulating %d inversion times (live Bloch sim)...\n" nTI
images = map(enumerate(TIs)) do (k, TI)
    seq = ir_se_2d_sequence(TI, 0.012, TR; FOV = FOV, Nfe = Npe, Npe = Npe,
                            spoiler = SpoilerConfig(enabled = true))
    raw = Suppressor.@suppress simulate(obj, seq, scanner_for_field(cfg))
    @printf "  [%d/%d] TI = %4.0f ms\n" k nTI 1000TI
    abs.(kspace_to_image(raw_to_kspace(raw, Npe, Npe)))
end

# 3 — Fit T1 per sphere and report.
T1_true = [d.T1 for d in descs]
T1_fit  = [fit_t1_generalized_ir(TIs, fill(π, nTI),
               [roi_mean(img, p[1], p[2]; r = 1) for img in images];
               TRs = fill(TR, nTI), Npe = Npe, sigma_method = :profile_likelihood).T1 for p in px]

println("\n  sphere   T1 true     T1 fit      error\n  ---------------------------------------")
for (i, (t, f)) in enumerate(zip(T1_true, T1_fit))
    @printf "   %2d     %5.0f ms    %5.0f ms    %+5.1f %%\n" i 1000t 1000f 100*(f-t)/t
end
@printf "  ---------------------------------------\n  T1 MAPE over %d spheres:  %.2f %%\n\n" length(descs) 100mean(abs.(T1_fit .- T1_true) ./ T1_true)

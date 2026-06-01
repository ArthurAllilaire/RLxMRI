# Report-ready validation of the E2 cached-water model (src/water_cache.jl) under
# E2 conditions. Writes ONE run folder of arrays + metadata; figures are rendered
# separately by scripts/cached_water_figs.py. Julia writes data only.
#
# It exercises the SAME library functions E2 uses (build_dry_and_water,
# build_cached_water_model with the α-bank, cached_water_ksp), so it validates the
# exact path the env takes. Three image-forming variants per (α, TI) block:
#   koma_water5   full Bloch with B0σ on the water too  (E2's :bloch — full physics)
#   koma_water0   Bloch spheres (B0σ) + Bloch water at B0σ=0  (the scene cached approximates)  ← GROUND TRUTH
#   cached        Bloch spheres (B0σ) + cached water template (B0σ=0)
# cached vs koma_water0 = cache fidelity (expect grid floor); koma_water0 vs
# koma_water5 = the documented water-B0 modelling cost. α is swept to test the
# bank + interpolation (include an OFF-grid α).
#
# ── RESULTS (2026-05-27, field=:T15, 32×64, 1.0 mm, dry=4816 / water=37822 spins,
#    5° α-bank = 36 templates, noiseless, schedule TI 0.05–2.2 s) ──────────────
# cached vs ground truth (Bloch spheres B0σ=5 + Bloch water B0σ=0), T1 fit |ΔT1|:
#     α=90°  mean 0.244%  max 1.150%      (== the hybrid-water harness grid floor)
#     α=30°  mean 0.000%  max 0.000%      (on-grid, exact)
#     α=17°  mean 0.163%  max 1.150%      (OFF-grid → bank interpolation; grid floor)
#     α= 5°  mean 0.082%  max 1.150%
#   ⇒ cached reproduces Bloch to the fitter's log-T1 grid resolution at every α.
# Per-step speedup ≈ 8.0× (full dry+water 4.47 s → dry+rescale 0.56 s; rescale
#   ~5e-5 s). α-bank build ≈ 153 s ONCE (global scope, amortised over all episodes).
# Water-only k-space relerr (figures/relerr_vs_TI.png): V-notch ~1e-8 at the ref TI,
#   rising to ~3–15% near the water null (~1.94 s) where |water|→0 so it barely
#   affects the combined k-space / fits. Water-B0 modelling cost (B0σ=0 vs 5 Hz) is
#   ~10–40% on the water alone — that is the realism traded for the cache.
#
# DESIGN RATIONALE (why a bank + B0σ=0 water — see scripts/cached_water_diagnostics.jl
# for the measured sweeps and src/water_cache.jl header for the physics):
#   • single fixed-α reference FAILS to generalise across α (90°-ref on a 30°
#     schedule → 9% mean / 61% max T1 error) ⇒ α-matched template bank + interp.
#   • per-spin B0σ=5 Hz makes the water k-space TI-phase-dependent (~5% at α=90°,
#     ~7–13% at α=30° even matched) ⇒ build the water template at B0σ=0; spheres
#     (Bloch-simulated) keep B0σ=5.
#
# Usage:
#   julia --project=. scripts/cached_water_validation.jl
#   julia --project=. scripts/cached_water_validation.jl --alphas 90,60,30,17,5 --label e2val

using MRISystemPhantom, KomaMRI, Suppressor
using Random, Statistics, Printf, JSON, NPZ
include(joinpath(@__DIR__, "t1_fit_lib.jl"))

const FOV=0.2; const TE=0.020; const TR=5.0
const B0_SPHERES = 5.0          # E2 augment on the Bloch-simulated spheres
const GRID_STEP_DEG = 5.0       # α-bank spacing (matches E2: e2_action_lo:5:hi)

let
    global Npe=32; global Nfe=64; global voxel=1.0; global field=:T15
    global alphas=[90.0,60.0,30.0,17.0,5.0]   # 17° is OFF the 5° grid → interpolated
    global label=""
    i=1
    while i<=length(ARGS)
        a=ARGS[i]
        if     a=="--npe"    && i<length(ARGS); global Npe=parse(Int,ARGS[i+1]); i+=2
        elseif a=="--nfe"    && i<length(ARGS); global Nfe=parse(Int,ARGS[i+1]); i+=2
        elseif a=="--voxel"  && i<length(ARGS); global voxel=parse(Float64,ARGS[i+1]); i+=2
        elseif a=="--alphas" && i<length(ARGS); global alphas=parse.(Float64,split(ARGS[i+1],",")); i+=2
        elseif a=="--label"  && i<length(ARGS); global label=ARGS[i+1]; i+=2
        else i+=1 end
    end
end
isempty(label) && (label="npe$(Npe)fe$(Nfe)_v$(replace(string(voxel),"."=>"p"))")

run_dir=joinpath(@__DIR__,"runs","cached_water_e2",label)
arr=joinpath(run_dir,"arrays"); mkpath(arr); mkpath(joinpath(run_dir,"figures"))
println("="^64); println("cached_water_validation  label=$label  Npe=$Npe Nfe=$Nfe voxel=$voxel  α=$alphas"); println("="^64)

slice_c=MRISystemPhantom.PLATE_Z_MM.T1
cfg5=PhantomConfig(field=field, voxel_size_mm=voxel, include_plates=[:T1,:water],
                   augment=AugmentConfig(B0_sigma_Hz=B0_SPHERES),
                   slice_thickness_mm=voxel, slice_center_mm=slice_c)
dry,   water5 = build_dry_and_water(cfg5)                          # spheres B0=5, water B0=5
_,     water0 = build_dry_and_water(cfg5; water_B0_sigma_Hz=0.0)   # water B0=0 (cache scene)
scanner=scanner_for_field(field)
@printf("spins: dry=%d water=%d\n", length(dry.x), length(water5.x))

α_grid=deg2rad.(GRID_STEP_DEG:GRID_STEP_DEG:180.0)
t_bank=@elapsed model=build_cached_water_model(water0,scanner; FOV=FOV,Nfe=Nfe,Npe=Npe, α_grid=α_grid)
@printf("built α-bank: %d templates in %.1fs\n", length(α_grid), t_bank)

descs=sphere_descriptors(:T1,cfg5; rng=MersenneTwister(0)); nsph=length(descs)
T1_true=[d.T1 for d in descs]
sphere_px=[(phys_to_pixel(d.centre[2],Npe,FOV),phys_to_pixel(d.centre[1],Nfe,FOV)) for d in descs]
seqf(TI,α)=Suppressor.@suppress ir_se_2d_sequence(TI,TE,TR; α_exc=α,FOV=FOV,Nfe=Nfe,Npe=Npe)
simk(ph,TI,α)=raw_to_kspace(Suppressor.@suppress(simulate(ph,seqf(TI,α),scanner)),Npe,Nfe)
relerr(a,b)=sum(abs.(a.-b))/max(sum(abs.(b)),eps())

TIs=[0.05,0.10,0.20,0.40,0.70,1.10,1.60,2.20]; nb=length(TIs)
img_variants=["koma_water5","koma_water0","cached"]
water_variants=["water5","water0","water_cached"]

relf=open(joinpath(arr,"relerr.csv"),"w"); println(relf,"alpha_deg,block,TI_s,variant,target,ksp_relerr,img_relerr")
fitf=open(joinpath(arr,"t1fits.csv"),"w"); println(fitf,"alpha_deg,variant,label,T1_true_s,T1_fit_s,mape_vs_true_pct,rel_vs_truth_pct")

for αdeg in alphas
    α=deg2rad(αdeg)
    K=Dict(v=>Vector{Matrix{ComplexF32}}(undef,nb) for v in vcat(img_variants,water_variants))
    for b in 1:nb
        TI=TIs[b]
        kdry=simk(dry,TI,α); kw5=simk(water5,TI,α); kw0=simk(water0,TI,α)
        kwc=cached_water_ksp(model,TI,TR,α,TE)
        K["koma_water5"][b]=kdry.+kw5; K["koma_water0"][b]=kdry.+kw0; K["cached"][b]=kdry.+kwc
        K["water5"][b]=kw5; K["water0"][b]=kw0; K["water_cached"][b]=kwc
    end
    img=Dict(v=>[kspace_to_image(K[v][b]) for b in 1:nb] for v in keys(K))
    # relerr: cache fidelity (vs koma_water0) and total incl. B0 cost (vs koma_water5)
    for b in 1:nb
        for (v,tgt) in (("cached","koma_water0"),("cached","koma_water5"),("koma_water0","koma_water5"))
            println(relf,"$αdeg,$b,$(TIs[b]),$v,$tgt,$(relerr(K[v][b],K[tgt][b])),$(relerr(img[v][b],img[tgt][b]))")
        end
        for (v,tgt) in (("water_cached","water0"),("water0","water5"))
            println(relf,"$αdeg,$b,$(TIs[b]),$v,$tgt,$(relerr(K[v][b],K[tgt][b])),$(relerr(img[v][b],img[tgt][b]))")
        end
    end
    # T1 fits
    bTI=[copy(TIs) for _ in 1:nsph]; bTR=[fill(TR,nb) for _ in 1:nsph]; bα=[fill(α,nb) for _ in 1:nsph]
    αinv=fill(π,nb); sinα=abs(sin(α)); fits=Dict{String,Any}()
    for v in img_variants
        fits[v]=fit_fleet(bTI,αinv,accumulate_block_mags(K[v],sphere_px,sinα),T1_true,Npe; block_TRs=bTR, block_α_excs=bα)
    end
    truth=fits["koma_water0"]
    for v in img_variants, i in 1:nsph
        rel=abs(fits[v].T1_fit[i]-truth.T1_fit[i])/truth.T1_fit[i]*100
        println(fitf,"$αdeg,$v,$(descs[i].label),$(T1_true[i]),$(fits[v].T1_fit[i]),$(fits[v].mapes[i]),$rel")
    end
    dev=[abs(fits["cached"].T1_fit[i]-truth.T1_fit[i])/truth.T1_fit[i]*100 for i in 1:nsph]
    @printf("  α=%5.1f° MAPE koma0=%.2f%% cached=%.2f%%  cached-vs-truth |Δ| mean=%.3f%% max=%.3f%%\n",
            αdeg, mean(truth.mapes), mean(fits["cached"].mapes), mean(dev), maximum(dev))
    # stacked arrays
    stk(v)=(A=Array{ComplexF32,3}(undef,nb,Npe,Nfe); for b in 1:nb; A[b,:,:]=K[v][b]; end; A)
    sti(v)=(A=Array{Float32,3}(undef,nb,Npe,Nfe); for b in 1:nb; A[b,:,:]=img[v][b]; end; A)
    tag=replace(@sprintf("a%05.1f",αdeg),"."=>"p")
    for v in keys(K); npzwrite(joinpath(arr,"$(v)_$(tag)_ksp.npy"),stk(v)); npzwrite(joinpath(arr,"$(v)_$(tag)_img.npy"),sti(v)); end
end
close(relf); close(fitf)
npzwrite(joinpath(arr,"TIs.npy"),TIs)
npzwrite(joinpath(arr,"sphere_px.npy"),[sphere_px[i][j] for i in 1:nsph, j in 1:2])
npzwrite(joinpath(arr,"T1_true.npy"),T1_true)
println("wrote relerr.csv, t1fits.csv, arrays")

# timing: full Bloch (dry+water5) vs cached (dry + rescale)
simk(dry,0.5,deg2rad(45.0)); simk(water5,0.5,deg2rad(45.0)); cached_water_ksp(model,0.5,TR,deg2rad(45.0),TE)  # warmup
t_dry=@elapsed simk(dry,0.5,deg2rad(45.0)); t_w5=@elapsed simk(water5,0.5,deg2rad(45.0))
t_resc=@elapsed cached_water_ksp(model,0.5,TR,deg2rad(45.0),TE)
t_full=t_dry+t_w5; t_cached=t_dry+t_resc
@printf("timing: full(dry+water)=%.3fs cached(dry+rescale)=%.3fs  per-step speedup=%.2f×  (rescale=%.2es, bank build=%.1fs)\n",
        t_full,t_cached,t_full/t_cached,t_resc,t_bank)

open(joinpath(run_dir,"config.json"),"w") do io
    JSON.print(io, Dict{String,Any}(
        "label"=>label,"field"=>String(field),"FOV_m"=>FOV,"voxel_size_mm"=>voxel,
        "Nfe"=>Nfe,"Npe"=>Npe,"TE_s"=>TE,"TR_s"=>TR,"alphas_deg"=>alphas,"TIs_s"=>TIs,
        "B0_spheres_Hz"=>B0_SPHERES,"B0_water_Hz"=>0.0,"grid_step_deg"=>GRID_STEP_DEG,
        "n_grid"=>length(α_grid),"img_variants"=>img_variants,"water_variants"=>water_variants,
        "ground_truth"=>"koma_water0","n_spins"=>Dict("dry"=>length(dry.x),"water"=>length(water5.x)),
        "timing_s"=>Dict("full"=>t_full,"cached"=>t_cached,"rescale"=>t_resc,"bank_build"=>t_bank,
                         "per_step_speedup"=>t_full/t_cached),
    ),2); println(io)
end
println("wrote $run_dir/config.json\nDone.")

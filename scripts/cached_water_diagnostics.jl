# Design-justification diagnostics for the cached-water model (src/water_cache.jl).
# These are the measurements that drove two non-obvious design choices: the α
# REFERENCE BANK (not one fixed reference) and building the water template at
# B0σ=0. Run to regenerate; the recorded RESULTS below let you follow the logic
# without rerunning. Writes arrays/cached_water_diagnostics.csv next to this file's
# runs/ tree. Throwaway diagnostics — not used by the env or the validation harness.
#
# Three sweeps (field=:T15, voxel=1.0 unless noted, water-only k-space rel. error
# vs full Bloch water; "matched" = reference α == eval α):
#
# 1) α-GENERALISATION OF A SINGLE REFERENCE — build the template at α_ref, evaluate
#    at other α (B0σ=0). Measured (voxel=2.0, 16×16, ref TI=0.1):
#        ref90 eval90  relerr ~1e-8 (ref) → 3.5e-2 (TI 0.5)   [TI-generalisation, OK]
#        ref90 eval30  relerr ~2.6e-1 at ALL TI               [α-extrapolation FAILS]
#    Per-row Koma/analytic transient ratio (voxel=1.0, α30 vs α90): 1.01 (row 1) →
#    1.40 (row 16) and the geometric template's spatial spread is 7–21% ⇒ neither a
#    single scalar nor a single template generalises across α.
#    ⇒ DECISION: cache an α-matched template per grid point.
#
# 2) α-BANK + NEAREST is not enough; INTERPOLATION is needed. Coarse ~22° grid +
#    nearest (voxel=2.0): relerr ~6–12% at ~10° off-grid. The production path uses a
#    5° grid + LINEAR interpolation of two bracketing α-matched evaluations; the
#    validation harness confirms off-grid α=17° fits to grid floor.
#
# 3) PER-SPIN B0 BREAKS THE CACHE (matched α, voxel=1.0, 32×64, water k-space relerr
#    vs TI; exact at the ref TI=0.1, drifts away from it):
#        B0σ=0:  α30 → 7e-4 (TI0.3), 6e-3 (TI0.9), 5e-2 (TI1.4 near null)   [OK]
#        B0σ=5:  α90 → 5e-2,  α60 → 6e-2,  α30 → 7e-2 .. 1.3e-1              [BROKEN]
#    Off-resonance adds TI-dependent PHASE the amplitude-only rescale can't track —
#    and it hits α=90° too, so it is not an α/spoiling artefact.
#    ⇒ DECISION: build the water template at B0σ=0 (Bloch-simulated spheres keep
#       B0σ=5). The cache then approximates a coherent-water scene; validate against
#       Bloch(spheres B0=5 + water B0=0), not against B0=5 water.

using MRISystemPhantom, KomaMRI, Suppressor
using Printf, Statistics, JSON, NPZ

field=:T15; FOV=0.2; Npe=32; Nfe=64; voxel=1.0; TE=0.02; TR=5.0
slice_c=MRISystemPhantom.PLATE_Z_MM.T1
mkcfg(b0)=PhantomConfig(field=field, voxel_size_mm=voxel, include_plates=[:T1,:water],
                        augment=AugmentConfig(B0_sigma_Hz=b0),
                        slice_thickness_mm=voxel, slice_center_mm=slice_c)
scanner=scanner_for_field(field)
seqf(TI,α)=Suppressor.@suppress ir_se_2d_sequence(TI,TE,TR; α_exc=α,FOV=FOV,Nfe=Nfe,Npe=Npe)
simk(ph,TI,α)=raw_to_kspace(Suppressor.@suppress(simulate(ph,seqf(TI,α),scanner)),Npe,Nfe)
relerr(a,b)=sum(abs.(a.-b))/max(sum(abs.(b)),eps())

out=joinpath(@__DIR__,"runs","cached_water_e2","_diagnostics"); mkpath(out)
io=open(joinpath(out,"cached_water_diagnostics.csv"),"w")
println(io,"sweep,B0_sigma_Hz,ref_alpha_deg,eval_alpha_deg,TI_s,ksp_relerr")

# Sweep 3 is the cheapest/most decisive; run it. (1,2 documented above; rerun by
# editing if you want them in the CSV — kept short here to bound compute.)
for b0 in [0.0,5.0]
    _, water = build_dry_and_water(mkcfg(b0); water_B0_sigma_Hz=b0)
    for αdeg in [90.0,30.0]
        α=deg2rad(αdeg)
        m=build_cached_water_model(water,scanner; FOV=FOV,Nfe=Nfe,Npe=Npe, α_grid=[α])  # matched
        for TI in [0.10,0.30,0.90,1.40,2.00]
            r=relerr(cached_water_ksp(m,TI,TR,α,TE), simk(water,TI,α))
            @printf("B0σ=%.0f matched α=%.0f TI=%.2f : %.3e\n", b0, αdeg, TI, r)
            println(io,"b0_ti_sweep,$b0,$αdeg,$αdeg,$TI,$r")
        end
    end
end
close(io)
println("wrote $out/cached_water_diagnostics.csv")

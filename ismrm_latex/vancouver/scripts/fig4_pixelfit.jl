# Fig 4 per-pixel T1 fit: apply the environment's own per-sphere fitter
# (fit_t1_generalized_ir with the settings of julia/rl/e2.jl:_e2_update_t1_estimates!)
# to every phantom pixel of the block images saved by fig4_replay.jl.
#
#   julia -t 8 --project=<RLxMRI> fig4_pixelfit.jl                # every replayed validation scan
#   julia -t 8 --project=<RLxMRI> fig4_pixelfit.jl 6              # selected validation scans
#   julia -t 8 --project=<RLxMRI> fig4_pixelfit.jl --heldout 7    # selected held-out scans
#   julia -t 8 --project=<RLxMRI> fig4_pixelfit.jl --heldout --best-grid 7   # fig4_replay.jl --best-grid
#
# Writes figures/fig4_data/fig4_maps_<tag>.npz per scan (tags as in fig4_common.jl).

include(joinpath(@__DIR__, "fig4_common.jl"))

function fit_series(env::E2Env, mags_raw::AbstractVector{<:Real}, TIs, TRs, α_excs)
    sin_α = max.(abs.(sin.(α_excs)), 1e-3)
    fit_t1_generalized_ir(TIs, fill(π, length(TIs)), Float64.(mags_raw) ./ sin_α;
                          TRs = TRs, α_excs = α_excs, Npe = env.Npe,
                          T1_range = env.T1_sample_range, n_grid = env.fitter_n_grid,
                          abs_noise_sigma = _sin_corrected_abs_noise(env.noise_sigma_abs, α_excs),
                          sigma_method = env.sigma_method, signed = env.phase_sensitive)
end

function fit_scan(env::E2Env, tag::AbstractString)
    d = npzread(replay_path(tag))
    occ = d["occupancy"]
    sphere_px = d["sphere_px"]
    out = Dict{String,Any}("scan" => d["scan"])
    for m in sort([chop(key; tail = length("_images")) for key in keys(d) if endswith(key, "_images")])
        imgs = d["$(m)_images"]                       # (n_blocks, Npe, Nfe)
        act = d["$(m)_actions"]                       # (n_blocks, 4): TI, TE, TR, alpha_deg
        TIs, TRs, α_excs = act[:, 1], act[:, 3], deg2rad.(act[:, 4])
        _, Npe, Nfe = size(imgs)
        T1_map = fill(NaN, Npe, Nfe)
        pix = [CartesianIndex(i, j) for i in 1:Npe, j in 1:Nfe if occ[i, j] > 0]
        Threads.@threads for p in pix
            T1_map[p] = fit_series(env, imgs[:, p[1], p[2]], TIs, TRs, α_excs).T1
        end
        # Re-derive the scored per-sphere ROI estimates from the saved images: they must
        # equal the env's own final T1_est, proving the saved series is what was fitted.
        roi_T1 = [fit_series(env, [roi_mean(imgs[b, :, :], sphere_px[i, 1], sphere_px[i, 2];
                                            r = env.roi_radius) for b in axes(imgs, 1)],
                             TIs, TRs, α_excs).T1
                  for i in axes(sphere_px, 1)]
        env_T1 = d["$(m)_T1_est"][end, :]
        roi_T1 == env_T1 || error("$tag $m: ROI refit $(roi_T1) != env estimate $(env_T1)")
        out["$(m)_T1_map"] = T1_map
        out["$(m)_roi_T1"] = roi_T1
        @printf("%s %-10s %4d pixels fitted, ROI refit matches env\n", tag, m, length(pix))
    end
    npzwrite(maps_path(tag), out)
end

function main(args)
    a = scan_args(args)
    tags = isempty(a.nums) ?
        sort([m[1] for m in (match(r"^fig4_replay_(k\d+)\.npz$", f) for f in readdir(DATA_DIR))
              if m !== nothing]) :
        [scan_tag(k; heldout = a.heldout, best_grid = a.best_grid) for k in a.nums]
    env = env_from_run(BASELINE_RUN)
    for tag in tags
        fit_scan(env, tag)
    end
end

main(ARGS)

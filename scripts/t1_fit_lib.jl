# Shared T1-fit pipeline helpers, extracted from scripts/t1_fit_vs_true.jl so the
# same per-sphere fit + run-dir + figure machinery can be reused (e.g. by
# scripts/hybrid_water_validation.jl) without duplicating it.
#
# Assumes the including script has already done:
#   using QalibreMDPhantom, KomaMRI
# (for raw_to_kspace, kspace_to_image, add_noise!, roi_mean,
#  fit_t1_generalized_ir, transient_mz_at_excite_npe, snr_report_to_dict).

using DelimitedFiles, Random, Statistics, Printf, JSON, NPZ

"""
    accumulate_block_mags(ksp_clean, sphere_px, sin_α; σ, rng, clean_recon,
                          img_pad, roi_radius, phase_sensitive) → Vector{Vector{Float64}}

Per-block: optionally add complex Gaussian noise σ, reconstruct (magnitude or
phase-sensitive; optional Hamming/zero-pad clean recon), and sample each sphere's
ROI pixel — divided by `sin_α` so the fitter sees a single amplitude across blocks.
Returns `block_mags[i]` = the magnitude time-series the fitter sees for sphere `i`.
"""
function accumulate_block_mags(ksp_clean::AbstractVector, sphere_px, sin_α;
                               σ::Real = 0.0, rng::AbstractRNG = Random.GLOBAL_RNG,
                               clean_recon::Bool = false, img_pad::Int = 1,
                               roi_radius::Int = 0, phase_sensitive::Bool = false)
    n_spheres = length(sphere_px)
    block_mags = [Float64[] for _ in 1:n_spheres]
    for blk in eachindex(ksp_clean)
        ksp = copy(ksp_clean[blk])
        add_noise!(ksp, σ; rng = rng)
        img = clean_recon ?
            kspace_to_image(ksp; pad_factor = img_pad, hamming = true,
                                  phase_sensitive = phase_sensitive) :
            kspace_to_image(ksp; phase_sensitive = phase_sensitive)
        for i in 1:n_spheres
            ipe, ife = sphere_px[i]
            push!(block_mags[i], roi_mean(img, ipe, ife; r = roi_radius) / sin_α)
        end
    end
    block_mags
end

"""
    fit_fleet(block_TIs, α_inv_vec, block_mags, T1_true, Npe; block_TRs,
              block_α_excs, abs_noise, phase_sensitive, T1_range, n_grid,
              sigma_method) → NamedTuple

Fit `fit_t1_generalized_ir` per sphere. `α_inv_vec` is the per-block inversion
flip-angle vector (typically `fill(π, n_blocks)`), shared across spheres — matching
the original positional call. Returns `(T1_fit, T1_sigma, M0_fit, mapes)`
(MAPE in percent, vs `T1_true`).
"""
function fit_fleet(block_TIs, α_inv_vec, block_mags, T1_true, Npe::Int;
                   block_TRs, block_α_excs,
                   abs_noise = nothing, phase_sensitive::Bool = false,
                   T1_range = (0.01, 3.0), n_grid::Int = 500,
                   sigma_method::Symbol = :profile_likelihood)
    n_spheres = length(block_mags)
    T1_fit   = zeros(n_spheres)
    T1_sigma = fill(NaN, n_spheres)
    M0_fit   = zeros(n_spheres)
    mapes    = zeros(n_spheres)
    for i in 1:n_spheres
        fit = fit_t1_generalized_ir(
            block_TIs[i], α_inv_vec, block_mags[i];
            TRs             = block_TRs[i],
            α_excs          = block_α_excs[i],
            Npe             = Npe,
            T1_range        = T1_range,
            n_grid          = n_grid,
            abs_noise_sigma = abs_noise,
            sigma_method    = sigma_method,
            signed          = phase_sensitive,
        )
        T1_fit[i]   = fit.T1
        T1_sigma[i] = fit.T1_sigma
        M0_fit[i]   = fit.A
        mapes[i]    = abs(fit.T1 - T1_true[i]) / T1_true[i] * 100
    end
    (T1_fit = T1_fit, T1_sigma = T1_sigma, M0_fit = M0_fit, mapes = mapes)
end

"""
    print_fit_table(descs, T1_true, fit)

Print the per-sphere `T1_true / T1_fit / σ / MAPE` table plus MEAN/MAX rows.
`fit` is the NamedTuple from `fit_fleet`.
"""
function print_fit_table(descs, T1_true, fit)
    n_spheres = length(descs)
    println()
    println("  sphere   T1_true [s]   T1_fit [s]   T1_σ [s]   MAPE [%]")
    println("  " * "─"^58)
    for i in 1:n_spheres
        @printf("  %6s   %10.4f   %10.4f   %8.4f   %7.2f\n",
                descs[i].label, T1_true[i], fit.T1_fit[i],
                isnan(fit.T1_sigma[i]) ? 0.0 : fit.T1_sigma[i], fit.mapes[i])
    end
    println("  " * "─"^58)
    @printf("  %6s   %10s   %10s   %8s   %7.2f\n", "MEAN", "", "", "", mean(fit.mapes))
    @printf("  %6s   %10s   %10s   %8s   %7.2f\n", "MAX", "", "", "", maximum(fit.mapes))
end

"""
    write_rundir(outdir, descs, T1_true, fit, centres, block_TIs, block_TRs,
                 block_mags, cfg_dict, TI_dense, TR_eff, Npe, α_exc; snr_rep)

Write `config.json`, `t1_fit_vs_true.csv`, `block_signals.csv`, and
`recovery_curves.npz` into `outdir` — the exact layout the Python figure scripts
(`t1_fit_vs_true.py`, `plot_recovery_curves_koma.py`) consume. `cfg_dict` is the
run config to serialise; if `snr_rep` is given it's attached under "snr_report".
"""
function write_rundir(outdir, descs, T1_true, fit, centres,
                      block_TIs, block_TRs, block_mags,
                      cfg_dict::AbstractDict, TI_dense, TR_eff, Npe::Int, α_exc;
                      snr_rep = nothing)
    n_spheres = length(descs)
    mkpath(outdir)

    open(joinpath(outdir, "config.json"), "w") do io
        d = Dict{String,Any}(cfg_dict)
        if snr_rep !== nothing
            d["snr_report"] = snr_report_to_dict(snr_rep)
        end
        JSON.print(io, d, 2)
        println(io)
    end

    csv_path = joinpath(outdir, "t1_fit_vs_true.csv")
    open(csv_path, "w") do io
        println(io, "label,T1_true_s,T1_fit_s,T1_sigma_s,M0_fit,mape_pct,cx_m,cy_m")
        for i in 1:n_spheres
            cx, cy = centres[i][1], centres[i][2]
            sig = isnan(fit.T1_sigma[i]) ? 0.0 : fit.T1_sigma[i]
            println(io, "$(descs[i].label),$(T1_true[i]),$(fit.T1_fit[i]),$sig,$(fit.M0_fit[i]),$(fit.mapes[i]),$cx,$cy")
        end
    end
    println("\nWrote $csv_path")

    signals_path = joinpath(outdir, "block_signals.csv")
    open(signals_path, "w") do io
        println(io, "label,block,TI_s,TR_s,mag")
        for i in 1:n_spheres
            for k in 1:length(block_TIs[i])
                println(io, "$(descs[i].label),$k,$(block_TIs[i][k]),$(block_TRs[i][k]),$(block_mags[i][k])")
            end
        end
    end
    println("Wrote $signals_path")

    y_true = Matrix{Float64}(undef, n_spheres, length(TI_dense))
    y_fit  = Matrix{Float64}(undef, n_spheres, length(TI_dense))
    for i in 1:n_spheres
        for (k, ti) in enumerate(TI_dense)
            y_true[i, k] = fit.M0_fit[i] * abs(transient_mz_at_excite_npe(
                T1_true[i], ti, TR_eff, π, α_exc; Npe = Npe))
            y_fit[i, k]  = fit.M0_fit[i] * abs(transient_mz_at_excite_npe(
                fit.T1_fit[i], ti, TR_eff, π, α_exc; Npe = Npe))
        end
    end
    npzwrite(joinpath(outdir, "recovery_curves.npz"), Dict(
        "TI_dense" => collect(TI_dense),
        "y_true"   => y_true,
        "y_fit"    => y_fit,
    ))
    println("Wrote $(joinpath(outdir, "recovery_curves.npz"))")
    csv_path
end

"""
    try_render(script_name, args)

Run a Python figure script in scripts/ (same dir as this file), fail-soft. Set
`PYTHON=.venv/bin/python` to pick the venv interpreter.
"""
function try_render(script_name, args)
    python = get(ENV, "PYTHON", "python")
    cmd = `$python $(joinpath(@__DIR__, script_name)) $args`
    try
        run(cmd)
        println("rendered via $(script_name)")
    catch e
        println("skipped $(script_name) ($e) — to render manually: $(cmd)")
    end
end

"""
    render_t1_figures(run_label)

Render the two T1-fit report figures for a run-dir labelled `run_label`.
"""
function render_t1_figures(run_label)
    try_render("t1_fit_vs_true.py", ["--subdir", run_label])
    try_render("plot_recovery_curves_koma.py", ["--run", run_label])
end

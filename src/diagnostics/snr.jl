# Shared SNR diagnostics — used by both `src/rl/e2.jl` (via diagnose_e2)
# and `scripts/t1_fit_vs_true.jl`. Pure functions over already-reconstructed
# magnitude images; no environment coupling.
#
# Three SNR metrics, all reported together so the figure-of-merit number in
# the FYP report is rooted in a standard MRI method (NEMA MS-1) and not the
# internal `target_snr` knob (which is `ksp_rms / σ` — non-standard, see plan).
#
# Conventions
# -----------
# * NEMA single-image (clinical default):
#     SNR = mean(signal in sphere ROI) / (std(background) / 0.6551)
#   The /0.6551 corrects for Rayleigh bias on magnitude images of
#   zero-mean complex Gaussian noise (Henkelman 1985; Gudbjartsson-Patz 1995).
# * NEMA MS-1 dual-acquisition (the gold standard for reproducible SNR):
#     SNR = mean(signal in ROI on (A+B)/2)  /  ( std(A−B in background) / √2 )
#   Two identical acquisitions with independent noise realisations. The
#   difference image is zero-mean Gaussian (no Rayleigh correction needed)
#   and removes any structured background. Standard in NEMA MS-1 (2014).
# * Internal k-space SNR (calibration knob):
#     SNR_ksp = sqrt(mean(|ksp|²)) / σ
#   Non-standard; reported here for cross-check against the `target_snr`
#   argument to E2Env.

"""
    SNRReport

Aggregated SNR metrics from one (or two) acquisition(s) of the same sequence.

Fields:
- `ksp_rms`              — √mean(|ksp_A|²)
- `sigma_used`           — noise σ injected on k-space
- `snr_ksp`              — `ksp_rms / sigma_used` (internal calibration metric)
- `background_std`       — std of magnitude image A over background pixels
- `diff_roi_std`         — std of (A − B) pooled over signal ROIs; dual-acq noise
- `sphere_means`         — per-sphere mean signal on (A+B)/2 in ROI
- `snr_nema_per_sphere`  — `sphere_means ./ (background_std / 0.6551)`
- `snr_nema_peak`        — `max(sphere_means) / (background_std / 0.6551)`
- `snr_dual_per_sphere`  — `sphere_means ./ (diff_background_std / √2)`
- `snr_dual_peak`        — `max(sphere_means) / (diff_background_std / √2)`
"""
struct SNRReport
    ksp_rms::Float64
    sigma_used::Float64
    snr_ksp::Float64
    background_std::Float64
    diff_roi_std::Float64
    sphere_means::Vector{Float64}
    snr_nema_per_sphere::Vector{Float64}
    snr_nema_peak::Float64
    snr_dual_per_sphere::Vector{Float64}
    snr_dual_peak::Float64
end

const RAYLEIGH_FACTOR = 0.6551  # √((4 − π)/2)

"""
    background_mask(phantom, Npe, Nfe, FOV; erosion_px=1) → BitMatrix

Build a per-pixel mask of background regions (phantom-free pixels) by
projecting phantom spins onto the image grid via `phantom_occupancy` and
selecting zero-occupancy pixels. Optional `erosion_px` removes a ring of
background pixels adjacent to the phantom, to avoid Gibbs / partial-volume
leakage from sphere edges contaminating the noise estimate.
"""
function background_mask(phantom, Npe::Int, Nfe::Int, FOV::Real; erosion_px::Int = 1)
    occ = phantom_occupancy(phantom, Npe, Nfe, FOV)
    bg = occ .== 0.0
    if erosion_px > 0
        # Shrink bg by `erosion_px`: any bg pixel within `erosion_px` of a
        # non-bg pixel (Chebyshev distance) is removed. Cheap O(N·k²) loop.
        bg_in = copy(bg)
        for i in 1:Npe, j in 1:Nfe
            bg_in[i, j] || continue
            adjacent = false
            for di in -erosion_px:erosion_px, dj in -erosion_px:erosion_px
                (di == 0 && dj == 0) && continue
                ii, jj = i + di, j + dj
                (1 <= ii <= Npe && 1 <= jj <= Nfe) || continue
                if !bg[ii, jj]
                    adjacent = true; break
                end
            end
            adjacent && (bg_in[i, j] = false)
        end
        return bg_in
    end
    bg
end

"""
    _roi_pixels(ipe, ife, roi_radius, Npe, Nfe) → Vector{CartesianIndex{2}}

Pixel indices in a `(2·roi_radius+1)²` square ROI clamped to image bounds.
`roi_radius=0` returns the single centre pixel.
"""
@inline function _roi_pixels(ipe::Int, ife::Int, roi_radius::Int, Npe::Int, Nfe::Int)
    pe_lo = clamp(ipe - roi_radius, 1, Npe)
    pe_hi = clamp(ipe + roi_radius, 1, Npe)
    fe_lo = clamp(ife - roi_radius, 1, Nfe)
    fe_hi = clamp(ife + roi_radius, 1, Nfe)
    [CartesianIndex(i, j) for i in pe_lo:pe_hi for j in fe_lo:fe_hi]
end

"""
    nema_stats(img, sphere_px, bg_mask; roi_radius=0) → NamedTuple

Single-acquisition NEMA SNR. Returns
`(; sphere_means, background_std, snr_per_sphere, snr_peak)`. Background std is
the raw std over `bg_mask` pixels; the `/0.6551` Rayleigh correction is applied
inside `snr_per_sphere`.
"""
function nema_stats(img::AbstractMatrix{<:Real},
                    sphere_px::Vector{NTuple{2,Int}},
                    bg_mask::AbstractMatrix{Bool}; roi_radius::Int = 0)
    Npe, Nfe = size(img)
    size(bg_mask) == (Npe, Nfe) ||
        error("bg_mask size $(size(bg_mask)) ≠ img size $((Npe, Nfe))")
    bg_vals = Float64.(img[bg_mask])
    isempty(bg_vals) && error("background mask is empty (no zero-occupancy pixels)")
    bg_std = Statistics.std(bg_vals; corrected = true)
    sphere_means = Float64[
        Statistics.mean(Float64.(img[_roi_pixels(p[1], p[2], roi_radius, Npe, Nfe)]))
        for p in sphere_px
    ]
    noise = bg_std / RAYLEIGH_FACTOR
    snr_per_sphere = sphere_means ./ max(noise, eps(Float64))
    (; sphere_means, background_std = bg_std,
       snr_per_sphere, snr_peak = maximum(snr_per_sphere))
end

"""
    dual_acq_stats(img_a, img_b, sphere_px; roi_radius=0) → NamedTuple

NEMA MS-1 dual-acquisition SNR. `img_a` and `img_b` are magnitude images from
two acquisitions of the same sequence with independent noise realisations.

Method (NEMA MS-1):
  noise = std( (A − B) restricted to **signal ROIs**, pooled across all
              spheres ) / √2
  signal_per_sphere = mean( (A + B)/2 in ROI )
  SNR_per_sphere    = signal_per_sphere / noise

Evaluating the difference std **inside the signal ROI** (not the background)
is what makes this method robust: at high SNR the magnitude image is locally
Gaussian around |S|, so A − B is zero-mean Gaussian with variance 2σ_img², and
dividing by √2 recovers σ_img. This eliminates both Rayleigh bias (which
would apply if we used the background) and any structured background.

Pooling the difference values across all sphere ROIs gives enough samples for
a reliable std even when each individual ROI is small (single-pixel default).
"""
function dual_acq_stats(img_a::AbstractMatrix{<:Real},
                        img_b::AbstractMatrix{<:Real},
                        sphere_px::Vector{NTuple{2,Int}}; roi_radius::Int = 0)
    Npe, Nfe = size(img_a)
    size(img_b) == size(img_a) ||
        error("img_a and img_b must have the same size")
    diff = Float64.(img_a) .- Float64.(img_b)
    half_sum = (Float64.(img_a) .+ Float64.(img_b)) ./ 2.0

    # Pool diff values across all sphere ROIs for the noise estimate.
    diff_pool = Float64[]
    sphere_means = Float64[]
    for p in sphere_px
        pix = _roi_pixels(p[1], p[2], roi_radius, Npe, Nfe)
        append!(diff_pool, diff[pix])
        push!(sphere_means, Statistics.mean(half_sum[pix]))
    end
    length(diff_pool) >= 2 || error("dual_acq_stats: pooled diff has < 2 samples")
    diff_roi_std = Statistics.std(diff_pool; corrected = true)
    noise = diff_roi_std / sqrt(2.0)

    snr_per_sphere = sphere_means ./ max(noise, eps(Float64))
    (; sphere_means, diff_roi_std,
       snr_per_sphere, snr_peak = maximum(snr_per_sphere))
end

"""
    snr_report(phantom, seq, scanner; σ, sphere_px, bg_mask, ksp_a, img_a,
                rng, phase_sensitive=false, roi_radius=0) → SNRReport

Build a full `SNRReport`. Caller must already have:
- `ksp_a` — first acquisition's k-space (post-noise)
- `img_a` — first acquisition's reconstructed magnitude image

Runs **one** additional `simulate(phantom, seq, scanner)` to produce
acquisition B, adds independent noise (`add_noise!` with σ), reconstructs via
`kspace_to_image`, and returns all three SNR metrics. This is the entry point
both call sites (`E2Env`, `t1_fit_vs_true.jl`) use.
"""
function snr_report(phantom, seq, scanner;
                    σ::Real,
                    sphere_px::Vector{NTuple{2,Int}},
                    bg_mask::AbstractMatrix{Bool},
                    ksp_a::AbstractMatrix{<:Complex},
                    img_a::AbstractMatrix{<:Real},
                    rng::AbstractRNG,
                    phase_sensitive::Bool = false,
                    roi_radius::Int = 0)
    Npe, Nfe = size(ksp_a)
    raw_b = Suppressor.@suppress simulate(phantom, seq, scanner)
    ksp_b = raw_to_kspace(raw_b, Npe, Nfe)
    add_noise!(ksp_b, Float64(σ); rng = rng)
    img_b = kspace_to_image(ksp_b; phase_sensitive = phase_sensitive)

    ksp_rms = sqrt(sum(abs2, ksp_a) / length(ksp_a))
    snr_ksp = σ > 0 ? ksp_rms / Float64(σ) : 0.0

    s = nema_stats(img_a, sphere_px, bg_mask; roi_radius = roi_radius)
    d = dual_acq_stats(img_a, img_b, sphere_px; roi_radius = roi_radius)

    SNRReport(
        Float64(ksp_rms),
        Float64(σ),
        Float64(snr_ksp),
        Float64(s.background_std),
        Float64(d.diff_roi_std),
        Float64.(d.sphere_means),                # use (A+B)/2 means as the canonical signal
        Float64.(s.snr_per_sphere),
        Float64(s.snr_peak),
        Float64.(d.snr_per_sphere),
        Float64(d.snr_peak),
    )
end

"""
    print_snr_report(io, rep::SNRReport; label="SNR report")

One-screen pretty-print for stdout / log files.
"""
function print_snr_report(io::IO, rep::SNRReport; label::AbstractString = "SNR report")
    println(io, label, "  (σ = ", round(rep.sigma_used, sigdigits = 4), ")")
    println(io, "  ksp_rms           = ", round(rep.ksp_rms, sigdigits = 5),
                "    snr_ksp        = ", round(rep.snr_ksp, sigdigits = 4),
                "   (calibration knob, non-standard)")
    println(io, "  background_std    = ", round(rep.background_std, sigdigits = 5),
                "    snr_nema_peak  = ", round(rep.snr_nema_peak, sigdigits = 4),
                "   (NEMA single-image)")
    println(io, "  diff_roi_std      = ", round(rep.diff_roi_std, sigdigits = 5),
                "    snr_dual_peak  = ", round(rep.snr_dual_peak, sigdigits = 4),
                "   (NEMA MS-1 dual-acq, REPORT FIGURE)")
    println(io, "  per-sphere snr_dual = ", round.(rep.snr_dual_per_sphere, digits = 2))
end
print_snr_report(rep::SNRReport; kwargs...) = print_snr_report(stdout, rep; kwargs...)

"""
    snr_report_to_dict(rep::SNRReport) → Dict{String,Any}

Serialisation helper for JSON output (e.g. `config.json` in
`scripts/t1_fit_vs_true.jl`).
"""
function snr_report_to_dict(rep::SNRReport)
    Dict{String,Any}(
        "ksp_rms"             => rep.ksp_rms,
        "sigma_used"          => rep.sigma_used,
        "snr_ksp"             => rep.snr_ksp,
        "background_std"      => rep.background_std,
        "diff_roi_std"        => rep.diff_roi_std,
        "sphere_means"        => rep.sphere_means,
        "snr_nema_per_sphere" => rep.snr_nema_per_sphere,
        "snr_nema_peak"       => rep.snr_nema_peak,
        "snr_dual_per_sphere" => rep.snr_dual_per_sphere,
        "snr_dual_peak"       => rep.snr_dual_peak,
    )
end

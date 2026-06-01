Contents:

- everything to do with my noise investigations
- snr_sweep.py

---

## SNR metric notes (from src/diagnostics/snr.jl)

### NEMA single-image (Method 4)

    SNR = mean(signal in sphere ROI) / (std(background) / 0.6551)

This is NEMA MS-1 Method 4:
- S = mean pixel value within MROI (Magnetic Region Of Interest)
- baseline pixel offset = average pixel value in background ROI
- calculate SD of background ROI — it will be Rayleigh distributed, so convert
  to image noise with correction factor SD/0.6551

The /0.6551 corrects for Rayleigh bias on magnitude images of zero-mean complex
Gaussian noise (Henkelman 1985; Gudbjartsson-Patz 1995). 0.6551 = √((4−π)/2).

Alternative method (not implemented): calculate mean of background ROI / 1.25
and use that as image noise (std is noisy thanks to squared term × artefacts).
The ratio mean/std should be ≈ 1.91 for a pure Rayleigh distribution
(see Appendix A of the NEMA MS-1 guide).

### NEMA MS-1 dual-acquisition (gold standard)

    SNR = mean(signal in ROI on (A+B)/2) / ( std(A−B in background) / √2 )

Two identical acquisitions with independent noise realisations. The difference
image is zero-mean Gaussian (no Rayleigh correction needed) and removes any
structured background. Divide by √2 because of the subtraction of both images.
Standard in NEMA MS-1 (2014):
https://www.nema.org/docs/default-source/standards-document-library/ms1-2008-r2014-watermarked.pdf

Could also use a different formula to test temporal instability — haven't
implemented that yet.

### Internal k-space SNR

    SNR_ksp = sqrt(mean(|ksp|²)) / σ

Non-standard; reported only as a cross-check against the image-domain metrics.

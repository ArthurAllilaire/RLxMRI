Trying to understand gibbs ringing - have got it down to the k-space we have gotten is missing the edge of k-space. by the convolution theorem that's the same as convolving with the Fourier transform of the box in image space - which is a sin() thing. so every pixel is getting convolved by this thing.

## The convolution theorem

It's a single statement: **multiplication in one domain = convolution in the other.** For a Fourier pair (image space $\leftrightarrow$ k-space):

$$\mathcal{F}\{f \cdot g\} = \mathcal{F}\{f\} * \mathcal{F}\{g\} \qquad\text{and}\qquad \mathcal{F}\{f * g\} = \mathcal{F}\{f\}\cdot\mathcal{F}\{g\}$$

That's the whole engine behind everything we've said:

- In k-space you **multiply** the true signal by a sampling box.
- So in image space the true image gets **convolved** by (FT of the box) = the PSF.

Why it's true in one line: a convolution is "slide, multiply, integrate." Under a Fourier transform each shift becomes a phase factor $e^{-i k x}$, the integral collapses the slide, and what's left is a plain product. The transform turns the expensive sliding-overlap operation into pointwise multiplication. (This is also why FFT-based convolution is fast: transform, multiply, transform back.)

Convolution itself, concretely:
$$(\rho * \text{PSF})(x) = \int \rho(x')\,\text{PSF}(x - x')\,dx'$$
"Put a copy of the PSF at every point of the true object, scaled by the object's value there, and sum." Every sharp feature gets replaced by a little sinc — hence ringing.

## The PSF and how it scales with $N$

$$\text{PSF}(x) = \frac{\sin(\pi N x/\text{FOV})}{N\,\sin(\pi x/\text{FOV})}$$

This is the **Dirichlet kernel** (periodic sinc). Read it as numerator-over-denominator and almost everything falls out. Recall pixel spacing $\Delta = \text{FOV}/N$.

**1. Peak at $x=0$ equals 1.** Both numerator and denominator $\to 0$; L'Hôpital gives the ratio $\to 1$. The center pixel passes through at full weight. ✓ (This is why $\rho$ isn't rescaled — the kernel is normalized.)

**2. Zeros are spaced one pixel apart.** Numerator $\sin(\pi N x/\text{FOV}) = 0$ when
$$\frac{\pi N x}{\text{FOV}} = m\pi \;\Rightarrow\; x = m\,\frac{\text{FOV}}{N} = m\,\Delta.$$
So zero-crossings sit at $\pm\Delta, \pm2\Delta, \dots$ — **the main lobe is always one pixel wide**, by construction. This is the key invariant: *in pixel units the PSF shape barely changes with $N$.*

**3. So what does $N$ actually change?** The main lobe width is $\Delta = \text{FOV}/N$ in **physical** units. Bigger $N$ → finer $\Delta$ → narrower main lobe → **better spatial resolution**. The ringing doesn't get worse or better in relative terms — it just gets packed into a finer physical scale.

| Quantity                            | Scaling with $N$                         |
| ----------------------------------- | ---------------------------------------- |
| Main-lobe width (physical)          | $\text{FOV}/N$ — shrinks ∝ $1/N$         |
| Main-lobe width (pixels)            | **1 pixel, always**                      |
| First side-lobe height              | **≈ −21%, always** (fixed by sinc shape) |
| Side-lobe decay                     | $1/x$, always                            |
| Number of side-lobes across the FOV | grows ∝ $N$                              |

**4. The two limits.**
- $N \to \infty$: $\Delta \to 0$, the main lobe collapses toward a **delta function** — perfect point, ringing squeezed to infinitesimal scale. (Matches the infinite-k-space picture.)
- Small $N$: wide main lobe, fat physical ringing — coarse, visibly blurry/ringy image.

The crucial takeaway for your report: **increasing $N$ does not reduce Gibbs ringing — it only rescales it.** The −21% first side-lobe is intrinsic to the rectangular window. To actually suppress ringing you must change the *window shape* (apodisation), trading the −21% side-lobe for a wider main lobe — i.e. resolution for smoothness. That's why your two grids both ring: they differ in $N$, but both use a hard rectangular window, and that's what sets the side-lobe structure.

A small notational note on your line 162–164: the formula as written is the **1-D** kernel and is a *normalized* form (peak = 1). The honest version has $|x| \le \text{FOV}/2$ and the 2-D PSF in your text is the separable product $\text{PSF}(x)\,\text{PSF}(y)$ with possibly different $N_{pe}, N_{fe}$.

# FFT SHIFT NOTES:
Notes: Fix SIM-PLAN 1 - 1.4
### 1.1 What KomaMRI returns

`simulate(phantom, seq, Scanner())` returns ADC profiles in **acquisition
order**. For a Cartesian readout, that is one row of k-space per PE shot, with
samples ordered from `-k_max` to `+k_max` along the readout direction. After
stacking PE rows in shot order, the resulting `ksp` matrix has DC (k=0) at the
**centre** of the array — index `(Npe÷2+1, Nfe÷2+1)` — not at index `(1,1)`.

### 1.2 What `ifft` expects

Julia's `FFTW.ifft` (and NumPy's, and MATLAB's) uses the standard DFT
convention: input index `(1,1)` is DC; the corners of the array are low
frequency; the centre of the array is the Nyquist frequency. To reconstruct
an image correctly from acquired k-space, the pipeline must be:

```
ksp_acquired                       (DC at centre)
        │ ifftshift               ← move DC from centre to (1,1)
        ▼
ksp_dftorder                       (DC at (1,1))
        │ ifft                    ← reconstruct image, DC at (1,1)
        ▼
img_dftorder                       (image centre at (1,1), corners at edges)
        │ fftshift                ← move image centre from (1,1) to centre
        ▼
img_centred                        (image centred — phantom-aligned)
```

The current code is `img_complex = ifft(ksp, (1, 2))` with no shifts at all.
That is equivalent to feeding DC-at-centre data into a DFT routine that
believes DC is at `(1,1)`. The result is the true image multiplied pointwise
by `(-1)^(i+j)` (a chequerboard) **and** wrapped half-FOV in both directions.

### 1.3 The visible consequence in E2

`_e2_build_episode_phantom` (`e2.jl:313–320`) computes ROI pixels from sphere
positions in metres using `mod(round(cx · Nfe / FOV), Nfe) + 1`. That formula
assumes the **phantom centre lands at image pixel `(1,1)`** — i.e. an
unshifted reconstruction. Today, the phantom is at the origin of physical
space (`x=y=0` after the random translation), so `round(0 · Nfe / FOV) + 1 = 1`,
and the ROI for a centre-of-FOV sphere is pixel `(1,1)`.

Two facts then collide:

1. With the buggy recon (no shifts), pixel `(1,1)` happens to receive the
   half-FOV-wrapped, chequerboard-modulated DC signal. For an isotropic
   phantom this is **roughly the right amplitude** for the centre sphere —
   which is exactly why a smoke-test "the agent learns something" never
   tripped on it.
2. For **off-centre spheres**, the wrap collapses the alias: a sphere at
   `(+FOV/4, 0)` lands at image-domain pixel `(Nfe÷4+Nfe÷2, ·)` after the
   correct shift, but at `(Nfe÷4, ·)` under the current ROI formula — i.e.
   half-FOV away from where the magnitude actually peaks. The ROI samples
   either zero, noise, or another sphere's tail.

The agent has been training against a corrupted measurement function where
the per-sphere signal is partly the right sphere, partly the diagonally-opposite
sphere, and partly background. That this still produced a plausible-looking
TI-vs-T1 correlation (§2.4 of M3.md) is a (small) miracle and probably a
testament to the chequerboard-modulated centre pixel still being roughly
correct for the highest-T1 sphere on the plate.

### 1.4 The fix

Replace `e2.jl:399–400` with:

```julia
img_complex = fftshift(ifft(ifftshift(ksp, (1, 2)), (1, 2)), (1, 2))
```

And change the ROI mapping in `_e2_build_episode_phantom` to centred indexing:

```julia
ife = mod(round(Int, cx * env.Nfe / env.FOV) + env.Nfe ÷ 2, env.Nfe) + 1
ipe = mod(round(Int, cy * env.Npe / env.FOV) + env.Npe ÷ 2, env.Npe) + 1
```

`ifftshift` and `fftshift` differ only for odd `N`; since defaults are even
(`Nfe=64`, `Npe=32` after item §3), either is fine in the most common path,
but `ifftshift`-then-`fftshift` is the safe pairing for arbitrary `N`.


# `fit_t1_generalized_ir` and the KomaMRI cross-checks — explainer

This is a guided tour of `src/fitting/fits.jl::fit_t1_generalized_ir` and the
test suite that pins it (and its forward model) against KomaMRI's Bloch
simulator. It answers two questions specifically:

1. How does the reconstruction take the acquisition parameters
   (`TI`, `α_inv`, `TR`, `α_exc`) into account?
2. How is the reported T1 uncertainty (`T1_sigma`) computed?

References: code lives in `src/fitting/fits.jl:113-253`; the tests are in
`test/test_e2.jl:45-155` and `test/test_e1.jl:27-49`.

---

## 0. MRI primer (just the bits this file needs)

Skip if you've already done the MRI reading. This section is the
minimum vocabulary you need for §1 onward.

### Magnetisation, Mz and Mxy

Inside the scanner's strong static field B0 (along ẑ), each voxel of
tissue has a **net magnetisation vector** `M = (Mx, My, Mz)`:
- `Mz` (longitudinal) — component along B0. At thermal equilibrium
  `Mz = M0 ∝ proton density`. **Mz alone produces no signal** — the
  receive coil only sees magnetisation rotating in the transverse plane.
- `Mxy = Mx + i·My` (transverse) — the rotating component the coil
  picks up as the MR signal. After an RF pulse tips Mz into the
  transverse plane, `|Mxy|` is what the ADC samples.

Two relaxation processes regrow / decay these components:
- **T1 (longitudinal)**: how fast `Mz` recovers toward `M0` after being
  disturbed. `Mz(t) = M0 + (Mz(0) − M0)·exp(−t/T1)`. T1 is the thing we
  are trying to estimate.
- **T2 (transverse)**: how fast `|Mxy|` decays toward zero.
  `|Mxy(t)| = |Mxy(0)|·exp(−t/T2)`. T2 governs how long the signal
  lasts during readout, but doesn't directly enter the T1 fit (it gets
  lumped into the amplitude `A`).

### Flip angles, RF pulses, and what `α` / `θ` mean

An **RF pulse** is a brief oscillating B1 field at the proton's
resonance frequency. In the rotating frame, it just rotates `M` about
some transverse axis by an angle equal to the **flip angle** —
conventionally written `α` or `θ`. The flip angle is determined by the
pulse's amplitude and duration:

```
α = 2π · γ · B1 · duration
```

After a pulse of flip angle `α`, an Mz that was at value `Mz⁻` becomes:
- new `Mz⁺ = cos(α) · Mz⁻`
- new `|Mxy|⁺ = sin(α) · |Mz⁻|` (added to whatever was already there)

So:
- `α = π/2` (90°) tips Mz fully into the transverse plane:
  `cos(π/2) = 0`, `sin(π/2) = 1` — all signal, no Mz left.
- `α = π` (180°) flips Mz to −Mz: `cos(π) = −1`, `sin(π) = 0` — no
  signal generated, just inversion. This is what the **inversion**
  pulse in IR does.
- `α = π/2` is the standard **excitation** pulse.
- Small `α` (e.g. 10°) tips a *little* into Mxy and leaves most of Mz
  intact — useful for fast imaging where you want repeated excitations
  without destroying the longitudinal pool.

Two flip angles appear in this fit, and they have different jobs:
- **`θ_inv` (or `α_inv`, `αs` in code)** — the **prep** / inversion
  flip angle. Applied first, knocks Mz down. Canonical IR uses
  `θ_inv = π` (full flip to −Mz). Saturation recovery uses `π/2`
  (Mz → 0). Small-tip preps use small angles. The fit handles all
  three uniformly through the `cos(θ_inv)` term.
- **`α_exc`** — the **excitation** flip angle, applied at the end of
  the TI delay to read out Mz_at_excite. Almost always `π/2` so
  `sin(α_exc) = 1` (all of Mz becomes signal) and `cos(α_exc) = 0`
  (no Mz survives — kills the steady-state transient).

### Inversion recovery, TI, TR — the timing

The inversion-recovery (IR) block has a fixed shape:

```
[θ_inv prep]  →  TI  →  [α_exc + ADC]  →  (rest of TR)  →  next prep
└──────────────┘     └────────────────┘ └────────────┘
       ↑                   ↑                  ↑
   inversion pulse    excitation+readout   recovery delay
```

- **TI (inversion time)** — delay between the prep pulse and the
  excitation pulse. During TI, Mz is recovering toward M0 from
  whatever the prep left it at. Choosing different TIs samples this
  recovery curve at different points — that's how we estimate T1.
- **TR (repetition time)** — period between consecutive prep pulses.
  TR sets how long the spins have to recover *between blocks* before
  being inverted again.

> **Why `TR − TI` for the recovery, not `TR`?** It's just the timing
> arithmetic of the block, not a convention. TR is the **whole**
> period from one prep to the next.  The block spends `TI` between
> prep and readout, then the remaining `TR − TI` between readout and
> the next prep. That trailing piece is when Mz recovers freely (no
> RF) before the next inversion. So:
> - During `TI`: Mz starts at `cos(θ_inv)·Mz_pre` and recovers for
>   time `TI` → `E1 = exp(−TI/T1)` controls how much.
> - During `TR − TI`: Mz starts at `cos(α_exc)·Mz_at_TI` (whatever
>   the excitation left behind) and recovers for time `TR − TI` →
>   `E2 = exp(−(TR − TI)/T1)` controls how much.

### "Perfect transverse spoiling" — what it is and why we assume it

After the excitation pulse, some magnetisation is in Mxy. If any of
that survives until the *next* inversion pulse (TR later), it gets
folded back into Mz by the inversion and contaminates the next block's
starting state. The forward model does **not** track Mxy across
blocks — it assumes it's all gone by the time the next prep arrives.

"**Perfect transverse spoiling**" is the assumption that
`Mxy = 0` at the end of every TR period. There are two ways this
happens in practice:

1. **Natural T2 decay** — if `T2 ≪ (TR − TI)`, the transverse
   component decays to ~0 on its own. The KomaMRI cross-check at
   `test_e2.jl:122-155` deliberately uses a very short `T2 = 20 ms`
   to enforce exactly this regime so that the analytic
   "no-Mxy-survives" model matches the Bloch simulation.
2. **Spoiler gradients / RF spoiling** — real MRI sequences add
   strong, randomised gradient pulses (or cycle the RF phase) at the
   end of each TR to dephase any residual Mxy. The signal averages to
   zero across the voxel. This is the "spoiling" you'd actually use
   on a scanner. The forward model is agnostic about *how* you
   achieve `Mxy = 0`; it just assumes you do.

The recurrence in §1 quietly relies on this: when we write
`Mz_pre = (1 − E2) + cos(α_exc)·E2·Mz_at_TI`, the only thing carried
over from the previous block is `cos(α_exc)·Mz_at_TI` — i.e. the
post-excitation Mz, with no Mxy contribution. Drop perfect spoiling
and you'd need a second coupled equation tracking Mxy.

---

## 1. The forward model

The fitter assumes a steady-state inversion-recovery block of the shape

```
[θ_inv prep]  →  TI delay  →  [α_exc excite + ADC]  →  (TR − TI) recovery
```

repeated indefinitely with **perfect transverse spoiling** between TRs.
At the moment of the excitation pulse the longitudinal magnetisation
(in units of M0) is

```
Mz_at_excite(T1; TI, TR, θ_inv, α_exc)
```

implemented in `steady_state_mz_at_excite` (`fits.jl:113-130`). The
derivation solves the two-equation recurrence

```
Mz_at_TI = (1 − E1) + cos(θ_inv) · E1 · Mz_pre
Mz_pre   = (1 − E2) + cos(α_exc) · E2 · Mz_at_TI
```

with `E1 = exp(−TI/T1)` and `E2 = exp(−(TR−TI)/T1)`.

### What the two equations mean, walked through in time

Each equation is a single application of the **T1 recovery law**

```
Mz(t) = M0 + (Mz_start − M0) · exp(−t/T1)
      = (1 − E)  +  Mz_start · E              (with M0 = 1, E = exp(−t/T1))
```

with one specific `Mz_start` (the value left by the most recent RF
pulse) and one specific elapsed time. The two equations are just
"recover for time TI" and "recover for time TR − TI", glued together at
the RF pulses that bracket each interval.

#### Why the recovery law has that form — solving the Bloch ODE for Mz

A common stumble: the recurrences contain a `(1 − E)` term *added* to
`Mz_start · E`, not a single `Mz_start · E` factor. That extra
constant is *not* a fudge — it's there because **T1 recovery is toward
equilibrium `M0`, not toward zero**. T2 decays to zero (single
exponential, proportional to where you started); T1 doesn't.

The Bloch equation for the longitudinal component is

```
dMz/dt = −(Mz − M0) / T1
```

i.e. `Mz` is pulled toward `M0` at a rate proportional to how far away
it currently is. Solve this with initial value `Mz(0) = Mz_start`.

**Method 1 (change of variable).** Shift the origin to the equilibrium
point by defining

```
u(t) := Mz(t) − M0
```

`M0` is a constant so `du/dt = dMz/dt`, and the ODE becomes the
textbook exponential-decay equation

```
du/dt = − u / T1     ⇒     u(t) = u(0) · exp(−t/T1)
```

Translate back: `u(t) = Mz(t) − M0` and `u(0) = Mz_start − M0`, so

```
Mz(t) − M0  =  (Mz_start − M0) · exp(−t/T1)
Mz(t)       =  M0 + (Mz_start − M0) · exp(−t/T1)
```

The trick is **shifting the origin to the equilibrium point**. The
*deviation from equilibrium* follows a clean single-exponential decay;
sliding `M0` back in gives the full Mz.

**Method 2 (separation of variables).** Mechanical, no trick:

```
dMz / (Mz − M0) = − dt / T1
ln |Mz − M0|     = −t/T1 + C
Mz − M0          = K · exp(−t/T1)              (K = ±exp(C))
```

Pin `K` from the initial condition: at `t = 0` we have `exp(0) = 1` so
`K = Mz_start − M0`, giving the same answer.

**Reading the result.** With `M0 = 1` and `E = exp(−t/T1)` the formula
splits into two terms:

```
Mz(t) =      M0       +    (Mz_start − M0)   ·     E
          └────┘            └────────────┘         └─┘
       where Mz heads      how far from         fraction of the
       (equilibrium)       equilibrium you      gap not yet closed
                           started

      = (1 − E)       +    Mz_start · E
        └─────┘            └─────────┘
       regrowth toward     fading memory
       M0 (independent     of Mz_start
       of Mz_start)
```

Two sanity checks:
- `t = 0`: `E = 1`, so `Mz = (1 − 1) + Mz_start·1 = Mz_start` — haven't
  moved yet. ✓
- `t → ∞`: `E = 0`, so `Mz = 1 + 0 = M0` — fully recovered, all memory
  of `Mz_start` erased. ✓
- At `t = T1`, `E = 1/e ≈ 0.37`: exactly 37 % of the original
  equilibrium gap remains. At `t = 5·T1` only ~0.7 % is left.

The `(1 − E)` "constant" you see in the IR recurrence is exactly the
*regrowth-toward-M0* piece — present even if `Mz_start = 0`, because
the spins are still being pulled toward equilibrium.

#### "But the leftover `cos(α_exc)·Mz_at_TI` is already on z — why does it have a `· E` factor?"

Tempting to read `cos(α_exc)·Mz_at_TI · E2` as "the leftover Mz
shrinking with time". It is **not**. Mz never decays on its own under
T1 relaxation — it always heads toward M0. The `· E2` factor is the
*fading-memory-of-Mz_start* piece of the recovery formula, paired
inseparably with the `(1 − E2)` regrowth piece.

Concrete numbers — `Mz_start = cos(α_exc)·Mz_at_TI = 0.3`, `M0 = 1`, let
`t = TR − TI` run from 0 to ∞:

| `t`       | `E2`  | `(1 − E2)` | `0.3 · E2` | sum = Mz |
|---|---|---|---|---|
| 0          | 1.00 | 0.00 | 0.30  | **0.30** |
| 0.5·T1     | 0.61 | 0.39 | 0.18  | **0.57** |
| T1         | 0.37 | 0.63 | 0.11  | **0.74** |
| 3·T1       | 0.05 | 0.95 | 0.015 | **0.965** |
| ∞          | 0.00 | 1.00 | 0.00  | **1.00** |

Mz **grows** from 0.30 to 1.0 — exactly recovery toward M0. The
`0.3·E2` column shrinks, the `(1 − E2)` column grows, and the sum
(actual Mz) climbs smoothly. Same formula handles `Mz_start > M0`:
e.g. `Mz_start = 1.5` gives `Mz(t) = 1 + 0.5·E2`, which comes *down*
from 1.5 to 1.0. The `· E` factor describes the closing of the gap to
equilibrium, which can be approached from either side.

Equivalently you can write the recovery as a single term

```
Mz_pre = 1 + (cos(α_exc)·Mz_at_TI − 1) · E2          ← deviation form
```

— the "deviation from equilibrium decays exponentially" view. The
linear-in-`Mz_at_TI` form

```
Mz_pre = (1 − E2) + cos(α_exc)·Mz_at_TI · E2          ← recurrence form
```

is the same curve rearranged so the steady-state fixed-point algebra
`Mz_pre = f(Mz_pre)` is one substitution away.

#### Where the regrowth actually comes from — *not* from the `sin(α_exc)` Mxy piece

A common follow-up: "the excitation tipped `sin(α_exc)·Mz_at_TI` away
from z; doesn't that magnetisation flow back during recovery?" **No.**
Mxy and Mz are decoupled. The `sin(α_exc)` piece **dephases**
(T2 / T2*) — individual spin moments fan out around the transverse
plane until they sum to zero — and the energy dissipates into the
local environment. It does not return to z.

The regrowth `(1 − E)` is **thermal re-equilibration with the
lattice**. At equilibrium in B0, a tiny Boltzmann-set excess of spins
sit aligned with B0; that excess is M0. Knock Mz away from M0 and
energy flows between spins and lattice (rotational tumbling of nearby
molecules, mostly) to restore the equilibrium population. The
magnetisation comes from the lattice, not from your earlier pulses.

Why we know this: if Mxy flowed back into Mz, T1 would have to equal
T2. In real tissue T1 is typically 5–20× longer than T2:

| Tissue (1.5T) | T1   | T2   |
|---|---|---|
| Grey matter   | 1.0 s | 95 ms |
| White matter  | 0.6 s | 80 ms |
| CSF           | 4.5 s | 2.2 s |
| Fat           | 0.3 s | 85 ms |

T1 and T2 are independent physical processes:
- **T2** — loss of phase coherence among spins (entropy-like, fast).
- **T1** — energy exchange with the lattice to restore the Boltzmann
  population (energetic, slower).

`|M|` is **not** conserved during relaxation. Right after a 90° pulse
`Mz = 0, |Mxy| = M0` so `|M| = M0`. A few T2 later `|Mxy| ≈ 0` while
Mz is still mostly recovered — `|M| ≪ M0`. Over several T1 the lattice
refills Mz back to M0 and `|M| = M0` again. The total magnitude
shrinks and grows; it doesn't "circulate" around the sphere.

So in the recurrence:

```
Mz_pre = (1 − E2)                +    cos(α_exc) · Mz_at_TI · E2
         └────────┘                    └────────────────────────┘
       energy refilled from           the part of Mz that wasn't
       lattice toward M0              tipped away — its gap to M0
       (NOT recycled Mxy)             still partially open by E2
```

Two separate sources. The `sin(α_exc)·Mz_at_TI` that was tipped into
Mxy is gone (dephased, dissipated); the regrowth is fresh thermal
relaxation pulling Mz toward M0 from whatever it currently is.

Pick a reference moment: **just before an inversion pulse**, with Mz
equal to some value `Mz_pre`. Walk one full TR period forward:

| Time mark | Event | Mz value | Why |
|---|---|---|---|
| `t = 0⁻` | just before inversion | `Mz_pre` | the reference |
| `t = 0⁺` | just after inversion (flip by `θ_inv`) | `cos(θ_inv) · Mz_pre` | RF rotates Mz; for `θ_inv = π` this is `−Mz_pre` |
| `t = TI⁻` | just before excitation | `(1 − E1) + cos(θ_inv)·E1·Mz_pre` | T1 recovery for time TI from `cos(θ_inv)·Mz_pre` toward 1; this is **`Mz_at_TI`** |
| `t = TI⁺` | just after excitation (flip by `α_exc`) | `cos(α_exc) · Mz_at_TI` | RF rotates again; for `α_exc = π/2` this is **0** |
| `t = TR⁻` | just before the *next* inversion | `(1 − E2) + cos(α_exc)·E2·Mz_at_TI` | T1 recovery for time `TR − TI` from `cos(α_exc)·Mz_at_TI` toward 1; this is the **next block's `Mz_pre`** |

So the two equations are literally:

- **Eq. 1** = "the value of Mz at the readout instant, after recovering
  for `TI` from the post-inversion state." This is what the ADC sees
  (times `sin(α_exc)` for the projection into Mxy).
- **Eq. 2** = "the value of Mz one full period later, after recovering
  for `TR − TI` from the post-excitation state." Closing the loop.

**Steady state** is the demand that the `Mz_pre` you start the cycle
with equals the `Mz_pre` you end the cycle with — repeating the
sequence forever doesn't change the values. Substituting Eq. 1 into
Eq. 2 and solving the fixed-point equation `Mz_pre = f(Mz_pre)` gives

```
Mz_pre        = [(1 − E2) + cos(α_exc) · E2 · (1 − E1)]
                / [1 − cos(θ_inv) · cos(α_exc) · E1 · E2]
Mz_at_excite  = 1 − E1 + cos(θ_inv) · E1 · Mz_pre
```

Two limits are baked in and used as sanity checks:
- `TR → ∞`  ⇒  `Mz = 1 − (1 − cos θ_inv) · exp(−TI/T1)` (legacy form
  E1 was using).
- `θ_inv = π, α_exc = π/2`  ⇒  `Mz = 1 − 2·exp(−TI/T1) + exp(−TR/T1)`
  (textbook IR with finite TR).

The observed magnitude is modelled as

```
|S_k| = |A · Mz_at_excite(T1; TI_k, TR_k, θ_inv_k, α_exc_k)|
```

`A` lumps together the proton density, receiver gain, the `sin(α_exc)`
projection of Mz into Mxy, and any T2 weighting at echo time — anything
that does not depend on `T1` or on `Mz` directly.

### How the four acquisition parameters enter the fit

| Parameter | Role | Where it enters |
|---|---|---|
| `TI` (per-sample) | Inversion delay | `E1 = exp(−TI/T1)` |
| `α_inv` / `αs` | Prep flip angle | `cos(θ_inv)` term — this is what makes IR (`π`) and SR (`π/2`) and small-tip preps the *same* fit |
| `TR` (per-sample, optional) | Repetition time | `E2 = exp(−(TR−TI)/T1)` — corrects for incomplete recovery between blocks |
| `α_exc` (per-sample, optional) | Excitation flip | Enters Mz_pre via `cos(α_exc)`; the `sin(α_exc)` projection into Mxy must be removed by the caller (see §3) |

If `TRs` is omitted, every per-sample TR is treated as `Inf` (full
recovery — the old E1 assumption). If `α_excs` is omitted, every
excitation defaults to `π/2`, which kills the `cos(α_exc)` transient and
matches most "real" IR implementations.

---

## 2. The fit itself — grid search + closed-form amplitude

Two unknowns: `T1` and `A`. Two-stage strategy (`fits.jl:196-215`):

1. **Outer loop**: log-spaced grid of `n_grid` (default 300) candidate
   T1 values across `T1_range` (default `5 ms … 5 s`).
2. **Inner closed form**: for each candidate T1, compute the predicted
   shape `y_k = Mz_at_excite(T1; …)`, take `|y_k|`, then the amplitude
   that minimises `Σ (A·|y_k| − m_k)²` is the closed-form ratio

   ```
   A = Σ m_k · |y_k|  /  Σ |y_k|²
   ```

   **Derivation.** With T1 fixed, `|y_k|` is fully determined and `A` is
   the only free parameter. The SSE

   ```
   L(A) = Σ_k (A·|y_k| − m_k)²
   ```

   is a 1-D quadratic in `A`. Setting `dL/dA = 0`:

   ```
   dL/dA = 2 Σ_k |y_k| · (A·|y_k| − m_k) = 0
        ⇒  A · Σ_k |y_k|²  =  Σ_k m_k · |y_k|
        ⇒  A = Σ m_k · |y_k|  /  Σ |y_k|²
   ```

   The second derivative `2 Σ |y_k|² > 0`, so this is the unique
   minimum. Equivalently it's the ordinary least-squares solution
   `A = (XᵀX)⁻¹ Xᵀm` for the one-column design matrix
   `X = [|y_1|, …, |y_n|]ᵀ`, collapsing to a scalar ratio.

3. Score by SSE; keep the (T1, A) pair with the lowest SSE.

There is **no gradient descent** — log-grid + per-grid linear regression
is robust against the magnitude-IR sign ambiguity (the inner `|·|`
already collapses both signs of `y_k`) and avoids needing a starting
guess.

---

## 3. Caller convention — `α_exc` and the `sin(α_exc)` correction

A subtlety: `A` is one scalar per fit. If `α_exc` varies across samples
within a single fit, the per-sample `sin(α_exc)` projection cannot be
absorbed into a single `A`. The fitter therefore expects the caller to
**pre-divide each magnitude by `sin(α_exc_k)` before passing it in**.

`_e2_update_t1_estimates!` in `python/qalibremd_gym/env_e2.py` does this.
The regression test at `test_e2.jl:45-72` pins it: feeding raw
`sin(α)·|1 − 2 e^{−TI/T1}|` produces a biased fit; feeding `mags / sin(α)`
recovers `T1_true` to within 5 %.

Note that `α_exc` still has to be passed **separately** via `α_excs`,
because `cos(α_exc)` enters `Mz_pre` whenever TR is finite. Only the
`sin(α_exc)` projection is the caller's responsibility.

---

## 4. Uncertainty — the asymptotic `T1_sigma`

`fits.jl:217-248`. The headline formula is

```
Σ_θ ≈ σ²_eff · (JᵀJ)⁻¹,    T1_sigma = √Σ[T1, T1]
```

with `θ = (A, T1)` and `J` the 2-column Jacobian of the model
`f_k(θ) = A · |y_k(T1)|` evaluated at the optimum. The rest of this
section derives that formula from scratch (A-level Further Maths is
enough — partial derivatives, 2×2 matrix inverses, Taylor series, and
the normal distribution).

### 4.0 What "uncertainty" means here

You measured `n` magnitudes `m_k`, each with some random noise. The
fitter returns the parameters `θ* = (A*, T1*)` that minimise

```
L(θ) = Σ_k (f_k(θ) − m_k)²
```

If you re-ran the *same* experiment with fresh noise, you'd get a
slightly different `T1*` each time. **`T1_sigma` is the standard
deviation of that distribution of `T1*` values** — how much your
estimate would jiggle under repeated sampling. It is *not* a measure
of the bias from a wrong model.

### 4.1 Why minimising SSE is the same as fitting a Gaussian

Assume each measurement has independent Gaussian noise:
`m_k = f_k(θ_true) + ε_k` with `ε_k ~ N(0, σ²)`. The probability
density of the data given parameters is

```
p(m | θ) = ∏_k  (1 / √(2π)·σ) · exp(−(f_k(θ) − m_k)² / 2σ²)
```

Take `−log` and drop terms that don't depend on `θ`:

```
−log p(m | θ) = (1 / 2σ²) · Σ_k (f_k(θ) − m_k)² = L(θ) / 2σ²
```

So **minimising SSE = maximising the Gaussian likelihood**. Useful
because we now have a probability distribution to work with, not just
an optimisation target.

### 4.2 Taylor-expand `L` around its minimum

Near `θ*`, expand `L` to second order. The gradient vanishes at the
minimum, so the linear term drops out:

```
L(θ) ≈ L(θ*) + ½ (θ − θ*)ᵀ H (θ − θ*)
```

where `H` is the 2×2 Hessian (matrix of second partials) at `θ*`.
Plug into the likelihood:

```
p(θ | m) ∝ exp(−(θ − θ*)ᵀ H (θ − θ*) / (4σ²))
```

That's a **2-D Gaussian in θ** centred on `θ*` with covariance matrix
`Σ = 2σ² H⁻¹`. The variance of `T1` (the `[T1, T1]` entry of `Σ`) is
what we want.

### 4.3 Approximating `H` by `JᵀJ`

Computing `H = ∂²L/∂θ²` directly is annoying because `f_k` is
non-linear. There's a clean shortcut. Let `r_k(θ) = f_k(θ) − m_k` so
`L = Σ r_k²`. First derivative:

```
∂L/∂θ_i = 2 Σ_k r_k · ∂f_k/∂θ_i
```

Second derivative (product rule):

```
∂²L/∂θ_i ∂θ_j = 2 Σ_k (∂f_k/∂θ_i)(∂f_k/∂θ_j) + 2 Σ_k r_k · ∂²f_k/∂θ_i ∂θ_j
```

At the optimum the residuals `r_k` are small and roughly mean-zero, so
the second sum approximately cancels. Drop it:

```
H ≈ 2 JᵀJ,     where  J_{ki} = ∂f_k/∂θ_i
```

`J` is the **Jacobian** — `n × 2` matrix of model-derivatives w.r.t.
each parameter. Plug back: the factor of 2 cancels the 2 in `Σ = 2σ²H⁻¹`,
leaving

```
Σ ≈ σ² (JᵀJ)⁻¹
```

This is the famous formula. ("Cramér–Rao bound" / "Fisher information"
in textbook language — names you don't need.)

### 4.4 Building `J` for our model

Two columns: `∂f_k/∂A` and `∂f_k/∂T1`, with `f_k = A · |y_k(T1)|`.

- **`∂f_k/∂A = |y_k|`** — easy, `A` enters linearly.
- **`∂f_k/∂T1`** — differentiate `|y_k(T1)|`. Two tricks:
  1. *Chain rule through `|·|`*: `d|y|/dT1 = sign(y) · dy/dT1`.
     Valid as long as `y ≠ 0` (i.e. away from the IR null).
  2. *Central finite difference for `dy/dT1`*: instead of
     differentiating the messy `steady_state_mz_at_excite` by hand,

     ```
     dy_k/dT1 ≈ (y_k(T1 + h) − y_k(T1 − h)) / (2h)
     ```

     This is the symmetric difference quotient — error `O(h²)`, much
     better than the one-sided version. Code uses
     `h = max(1e-6, 1e-4 · T1*)`.

  Combining: `∂f_k/∂T1 ≈ A · sign(y_k) · (y_k(T1+h) − y_k(T1−h)) / (2h)`.

### 4.5 Inverting the 2×2 `JᵀJ`

Write

```
JᵀJ = [[a11, a12],
       [a12, a22]]
```

with (sums over `k`):
- `a11 = Σ (∂f_k/∂A)²       = Σ |y_k|²`
- `a22 = Σ (∂f_k/∂T1)²`
- `a12 = Σ (∂f_k/∂A)(∂f_k/∂T1)`

Standard 2×2 inverse:

```
(JᵀJ)⁻¹ = (1 / det) · [[ a22, −a12],
                       [−a12,  a11]],     det = a11·a22 − a12²
```

We only want the `[T1, T1]` entry — bottom-right — so

```
Var(T1) = σ² · a11 / det(JᵀJ)
```

That's exactly the line in the code. `T1_sigma = √Var(T1)`. The
`det(JᵀJ) > 0` guard avoids dividing by zero at degenerate fits.

> **Why `a11` on top and not `a22`?** Counter-intuitive at first.
> Reason: when parameters are correlated, the uncertainty in `T1`
> depends on how well you can pin down `T1` *after* `A` has eaten up
> some of the data's wiggle room. The off-diagonal `a12` measures that
> coupling, and the algebra of the 2×2 inverse puts `a11` — the
> *A*-direction stiffness — into the `T1` variance. Loosely: a stiffer
> `A` direction means less data variance is "spent" absorbing `A`, so
> more is left to constrain `T1`.

### 4.6 Estimating `σ²_eff`

The derivation assumed `σ²` known. It usually isn't. Two estimates,
the larger wins:

- **Residual-based**: if the model is right, leftover SSE at the
  optimum is roughly `(n − p)·σ²` with `p = 2` parameters fitted (two
  degrees of freedom "spent"). So
  `σ²_resid = SSE* / (n − 2)` — the reduced χ². Undefined for `n ≤ 2`
  (set to 0).
- **Floor from declared noise**: if the caller passes `noise_sigma` as
  a *relative* level, multiply by the data RMS for an absolute floor:
  `σ²_floor = (noise_sigma · rms(m))²`.

Then `σ²_eff = max(σ²_resid, σ²_floor)`. The floor stops `T1_sigma`
from collapsing to zero on noise-free synthetic data while still
letting noisy fits report their actual residual scale.

### 4.7 What `T1_sigma` means in plain words

If the assumptions hold, **re-running the experiment many times with
fresh noise would produce `T1*` values roughly Gaussian-distributed
with standard deviation `T1_sigma` around the true T1**. A 1σ
interval covers ~68%, 2σ covers ~95%. The "asymptotic" in
"asymptotic σ" means this is exact only as `n → ∞`; at finite `n` it
is an approximation that breaks where the assumptions do (see §7).

### What `T1_sigma` does *not* capture

- Bias from a misspecified model (e.g. ignoring TR when `TR/T1` is
  small — see §5).
- Grid resolution: the reported optimum is snapped to the nearest grid
  point, so `T1_sigma < (grid spacing)/2` is meaningless. Default
  spacing is `≈ exp((log 5 − log 0.005)/300) ≈ 1.023×` per grid step,
  i.e. ~2 % — comparable to the test tolerances.
- Sign-ambiguity flips between adjacent local optima in T1 — magnitude
  IR has multimodal SSE in T1 with too-few well-placed TIs.

---

## 5. The KomaMRI cross-checks — what they pin down

`test/test_e2.jl:122-155` ("steady-state IR formula matches KomaMRI
simulation") is the headline cross-check. It validates the **forward
model** (`steady_state_mz_at_excite`) against an actual Bloch simulation:

1. Build a single-spin phantom with short T2 (`T2 ≪ TR − TI`) so
   transverse magnetisation has decayed before the next inversion —
   this enforces the "perfect spoiling" assumption analytically baked
   into the forward model.
2. Construct a sequence of `n_rep = 4` IR shots
   (`180°  → TI delay → 90° → 1-sample ADC → recovery`) and call
   `simulate(obj, seq, Scanner())`.
3. Take `abs(raw.profiles[end].data[1, 1])` — the magnitude of the last
   ADC sample, which has reached steady state.
4. Compare to `abs(steady_state_mz_at_excite(T1, TI, TR, π, π/2))`.
5. Tolerance: `rtol = 0.05, atol = 5e-3`. Across four (TI, TR) pairs.

Why short pulses (`amp_T = 100 µT`) matter: the analytic model treats
RF as instantaneous. Long pulses bias the simulator by ~0.5 % per pulse
(see `test_e1.jl:147` — "simulate backend is biased by RF duration").
Short hard pulses reduce that to within tolerance.

### Companion tests (still in `test_e2.jl`)

- **`steady_state_mz_at_excite: analytic identities`** (`:74-96`) —
  closed-form sanity: `TR → ∞` collapses to the legacy form;
  `θ_inv=π, α_exc=π/2` matches `1 − 2·e^{−TI/T1} + e^{−TR/T1}`; and the
  `TR/T1 = 0.5` regime differs from `TR=∞` by > 0.3 (so the correction
  matters).
- **`TR-aware fit unbiased; TR-blind biased`** (`:98-120`) — generates
  textbook IR data with finite TRs comparable to T1, then shows
  `f_aware` (called with `TRs=`) recovers T1 to `rtol = 0.02` while
  `f_blind` (legacy `TR=∞` assumption) is provably worse. This pins the
  reason the E2 fit must accept `TRs` even though E1 didn't.
- **`α_exc-scaled magnitudes need correction`** (`:45-72`) — the
  caller-side `mags / sin(α_exc)` rule from §3. Regression test for the
  fix in `_e2_update_t1_estimates!`.

### And in `test_e1.jl`

- `fit_t1_generalized_ir recovers T1 from clean data` (`:27-49`) —
  spans 4 ground-truth T1 values across the plate range and a mixed
  IR + SR (`α ∈ {π, π/2}`) acquisition, verifying that the
  `cos(α_inv)` term genuinely allows mixing prep schemes in one fit.
- `simulate backend matches analytical within a few percent` (`:134-150`)
  — runs the E1 environment with the analytic forward model and the
  simulator backend on identical seeds and checks the terminal MAPE
  agrees within 5 % absolute. Establishes that the analytical hot path
  is a safe stand-in during RL training.

---

## 6. Summary diagram

```
                  acquisition params per shot
            (TI, α_inv, TR, α_exc)   m_k = |observed magnitude|
                       │                        │
                       ▼                        ▼
              steady_state_mz_at_excite     mags / sin(α_exc)   ← caller
                       │                        │
                       └────────┬───────────────┘
                                ▼
                  for each T1 in log-grid:
                     y_k = Mz_at_excite(T1; …)
                     A   = Σ m·|y| / Σ |y|²        (closed form)
                     SSE = Σ (A·|y| − m)²
                                │
                       argmin → (T1*, A*, SSE*)
                                │
                       central-difference Jacobian at (T1*, A*)
                                │
                       σ²_eff = max(SSE/(n−2), (noise·rms)²)
                                │
                       Var(T1) = σ²_eff · a11 / det(JᵀJ)
                                │
                                ▼
                    (T1, A, residual, T1_sigma)
```

---

## 7. Known limitations of `T1_sigma` (the σ-channel)

The asymptotic σ_T1 produced by `fit_t1_generalized_ir` (lines 217–247 of `src/fitting/fits.jl`) is wired into the E2 observation as `log10(σ_T1[i] / T1_est[i])` per sphere — this is "Option C" of the §16.4 plan in `EXPERT_REPORT.md`. After running the E2.2 policy through `python/diagnose_uncertainty.py`, the reported σ_T1 is much smaller than the true error: median 5.2 %, mean 16.4 %, p90 27.7 %, while the *true* MAPE on the same rollouts is **97.8 %**. The σ-channel is therefore feeding the policy a misleadingly confident signal.

**The math is not wrong.** `var_T1 = σ²_eff · a11 / det(JᵀJ)` is the correct [T1, T1] entry of the 2×2 inverse JᵀJ for parameters [A, T1]; the central-difference Jacobian is correct (`d|y|/dT1 = sign(y) · dy/dT1` with A treated as an independent partial); and `σ²_eff = max(σ²_resid, σ²_floor)` is the standard guard. **Three structural issues** explain the gap between reported σ and reality:

### 7.1 n − p = 1 DOF when an episode terminates at 3 blocks

```julia
σ²_resid = n > 2 ? best_sse / (n - 2) : 0.0     # line 235
```

A 3-point fit with 2 free parameters can almost always interpolate the data — *especially* when the agent clusters TIs (E2.2 puts ≈64 % of all blocks at TI ≈ 10 ms), so the effective informative TIs per fit is closer to 2 than 3. `best_sse ≈ 0` ⇒ `σ²_resid ≈ 0`, and the asymptotic-Gaussian assumption that justifies σ²·(JᵀJ)⁻¹ requires `n ≫ p`. At n=3 the residual variance estimate has no statistical power.

### 7.2 The noise floor is the only thing keeping σ non-zero — and it scales with *signal*

```julia
rms_m    = sqrt(sum(abs2, m) / n)               # RMS of observed magnitudes
σ²_floor = (Float64(noise_sigma) * rms_m)^2     # line 239–240
σ²_eff   = max(σ²_resid, σ²_floor)              # line 242
```

With `noise_sigma_rel = 0.05` and a typical mid-signal `rms_m ≈ 0.5`, `σ²_floor ≈ 6.25 × 10⁻⁴`. For E2.2's near-zero `σ²_resid`, the floor wins almost every time, giving a consistent ~5 % σ — not a measurement of fit quality, just `0.05 × signal`.

### 7.3 The floor couples σ to the agent's own action choice

`rms_m` is the RMS of the *observed signal* given the TIs the agent picked. So the agent can implicitly game its own confidence reading: high-signal TIs (TI ≈ TI_min where `|1 − 2 e^{−TI/T1}| ≈ 1`) → larger `rms_m` → larger σ; null-region TIs → smaller `rms_m` → smaller σ. The σ-channel isn't measuring the same thing across action choices, which would already make it hard for a policy to learn a stable mapping `σ ↦ TI`.

### 7.4 Magnitude-IR null-crossing makes SSE multi-modal

The asymptotic σ is *local* curvature only. Magnitude IR fits have multiple SSE basins (one per sign-flip pattern of the data, see §3 of this doc — the `2^n` enumeration is precisely there to handle this). The fitter picks the global-best basin, but `var_T1 = σ²·a11/det(JᵀJ)` reflects only the width of *that* basin. The fact that there are several other shallow basins of comparable depth is invisible to the asymptotic σ. This is the regime where likelihood-based or bootstrap σ is needed.

### 7.5 Recommended fixes (single-file, in `fits.jl`)

1. **Don't trust σ²_resid until n ≥ 5.** Replace line 235:
   ```julia
   σ²_resid = n > 4 ? best_sse / (n - 2) : Inf   # untrusted at small n
   ```
   Forces `σ²_eff = σ²_floor` in the small-n regime — at least makes the floor's role explicit instead of accidental.

2. **Use an absolute noise floor, not signal-relative.** Pass `noise_sigma` as the absolute σ measured from the imaging chain (e.g. `env.noise_sigma_rel × rms_initial_signal` computed once at episode reset), not recomputed per fit. Decouples confidence from the agent's TI choice.

3. **Profile-likelihood σ instead of asymptotic.** Sweep T1 around `best_T1` until SSE rises by χ²₁(0.68) = 1; the half-width is the 1σ. Captures the multi-modal SSE landscape that breaks the asymptotic formula at small n. Cost: O(grid_size) extra evaluations per fit, negligible vs. the Bloch sim.

4. **Bootstrap σ (most robust, most expensive).** Resample residuals, refit, take std of T1 across refits. No Gaussianity assumption. Recommended only if (1)–(3) prove insufficient.

### 7.6 Quick experiment to confirm

Apply fix (1) alone (one-line change), re-run `python/diagnose_uncertainty.py` on the existing E2.2 policy. If the median σ jumps from 5 % to something realistic (50–100 %), the σ-channel becomes a much more honest signal for the planned E2.3 (A + C + delta-MAPE) run to actually condition on.

### 7.7 Where this came from

Discovered while writing the §18 E2.2 ablation in `EXPERT_REPORT.md`. The full analysis is in §18.11 there (§18.11 also discusses why σ-obs alone — i.e. Option C without Option A — was empirically inert on the E2.2 200k run). This `docs/` entry is the algorithmic-correctness companion: the σ formula is right but its assumptions break in the regime the agent is actually operating in.

---

## 8. Better reconstruction models — what's beyond this fitter

The current fitter is a deliberate compromise: cheap enough to call inside the RL inner loop, strong enough to estimate T1 from a handful of well-chosen IR shots. Every speed gain came at the cost of an idealising assumption. This section maps the upgrade landscape and recommends an ordering for the FYP.

### 8.1 Hierarchy of assumptions and what relaxes each one

| Assumption baked into `fit_t1_generalized_ir` | Method that relaxes it | Cost vs. analytic |
|---|---|---|
| Perfect transverse spoiling (Mxy = 0 between blocks) | **EPG** — Extended Phase Graphs | ~10–100× |
| Steady state reached per shot under varying actions | **Transient Bloch / EPG forward simulation** | ~10–100× |
| Single-compartment relaxation | **EPG with sub-states / PDG** | ~10× EPG |
| Instantaneous, ideal RF pulses | **Full Bloch simulator-in-the-loop** (KomaMRI / MRzero) | seconds-per-fit |
| Nominal flip angle = actual flip angle (no B1 inhomogeneity) | **Joint T1 + B1 fit** | ~2× |
| Perfect on-resonance (no B0 variation) | **Joint T1 + B0 fit** | ~2× |
| Rectangular slice profile | **Slice-profile-aware fit** (integrate over slab) | 5–20× |
| Scalar grid search over T1 | **MRF-style dictionary matching** (multi-parameter) | offline-heavy, fast inference |
| Asymptotic point estimate of T1 with σ from JᵀJ | **Bayesian / MCMC posterior** | 10–1000× |

### 8.2 EPG (Extended Phase Graphs) — the standard upgrade

Tracks the magnetisation as a sum of *configuration states* indexed by accumulated phase from gradients. Each RF pulse is a small matrix that mixes states; each delay applies T1/T2 and shifts states. **No spoiling assumption** — you compute exactly how much Mxy survives a TR and how much gets folded back into Mz at the next pulse. Standard reference: Weigel 2015, "Extended phase graphs: dephasing, RF pulses, and echoes — pure and simple" (J. Magn. Reson. Imaging). Open implementations: `epyg` (Python), `MRIReco.jl` (Julia). Drop-in replacement for `steady_state_mz_at_excite`; the rest of the fit is unchanged.

### 8.3 PDG (Phase Distribution Graphs) — what MRzero uses

Strict generalisation of EPG with a continuous phase distribution; more accurate for irregular or non-spoiled gradients. **MRzero** (Loktyushin et al., MRM 2021) is the differentiable reference implementation — gradients flow through the simulator, so you can fit T1 (and B1, B0, T2…) by gradient descent. Same cost ballpark as EPG. Note the project's existing CLAUDE.md flags that the interim report incorrectly calls MRzero an EPG simulator — it's PDG.

### 8.4 Magnetic Resonance Fingerprinting (MRF)

Pre-simulate a large dictionary of `(T1, T2, B1, B0)` → signal trajectories (using EPG / PDG / Bloch, with the actual transient — no steady-state assumption needed), then match the observed trajectory to the closest dictionary entry. Originally Ma et al., Nature 2013. **Decouples** expensive simulation (offline) from fast reconstruction (cosine similarity at inference). Natural fit for this project because:

- The agent's action sequence already defines the trajectory shape.
- A dictionary built once over the action space + tissue parameter range gives O(N) lookup at inference.
- Generalises trivially to multi-parameter joint estimation (T1, T2, ρ, B1).
- Removes the steady-state-per-shot failure mode flagged in §1 — dictionary entries simulate the actual transient under varying actions.

This is the single biggest upgrade if a strong report angle is the goal. Already on the roadmap as E3.

### 8.5 Bloch-simulator-in-the-loop fit

Brute-force option: use KomaMRI (or MRzero) directly as the forward model and fit T1 with `scipy.optimize.least_squares` or gradient descent. **Removes essentially every assumption** — finite RF, off-resonance, real T2, slice profile, anything baked into the phantom. Used as a **reference truth** rather than for production RL because it's seconds per fit. Useful for **validation**: run a few episodes' worth of fits with the simulator-in-the-loop and quantify the bias of the fast analytic fit. That number goes straight into the limitations chapter.

### 8.6 Bayesian inference — proper uncertainty

The current `T1_sigma` is Cramér-Rao-style asymptotic. A real Bayesian fit (MCMC over the same forward model, or variational with a normalising flow) gives the **full posterior** `p(T1 | data)`. Multimodal magnitude-IR curves (§4 "What `T1_sigma` does not capture") are exactly the case where the asymptotic σ lies and the posterior tells the truth. Worth the ~10× cost only if uncertainty calibration matters to the agent's reward — which it might in E2 if σ is added to the observation channel.

### 8.7 Joint multi-parameter fits

The single biggest *practical* improvement for sim-to-real work: fit `(T1, B1, M0)` jointly instead of T1 alone. Real B1 inhomogeneity is ±10–20 % across a phantom, directly biasing the `cos/sin(α_exc)` terms. Inversion efficiency `η = (1 − cos θ_inv)/2` is rarely the nominal 1.0 — typically 0.85–0.95 depending on the inversion pulse. Both are easy to add as extra fit parameters at the cost of needing more / better-conditioned data.

### 8.8 Recommended upgrade path for the FYP

Three filters: (a) does it fit in the RL inner loop? (b) does it remove the bias §1 flagged? (c) does it give a write-up angle?

Priority order:

1. **Add the varying-action KomaMRI cross-check** (§5 companion tests). Quantifies the actual steady-state bias the fitter has under E2-realistic action sequences. ~30 minutes; produces a figure for the limitations chapter regardless of what else is done.
2. **EPG-based forward model**. Drop-in replacement for `steady_state_mz_at_excite`. Removes the spoiling and steady-state-per-shot assumptions in one shot. Still fast enough for the inner loop. Paper-shaped contribution: "EPG-based reward modelling for sequence-design RL".
3. **MRF-style dictionary reconstruction**. Bigger swing — restructures the inner loop. Already aligned with E3 in the master plan.
4. **Joint T1 + B1 fit**. Doesn't matter for digital-twin work but matters for any future sim-to-real. Frame as a forward-looking limitations item rather than building it.

Realistic plan given timeline: option 1 immediately, option 2 if option 1 surfaces >5 % T1 bias, option 3 inside E3 where it's the natural fit, option 4 as a limitations-chapter bullet.

### 8.9 The honest limitations paragraph for the report

> The reconstructor assumes perfect transverse spoiling and per-shot steady state. We validate the analytic forward model against KomaMRI under matching idealisations (`test_e2.jl:122-155`); under varying-action sequences the empirical bias is *X %* (Figure *Y*). EPG-based and MRF-style reconstructions remove these assumptions at the cost of inner-loop compute and are evaluated in §[E3]. Joint estimation of B1 and inversion efficiency is left for sim-to-real work.

# EXPLAINER — walk through the code

A reading guide for the repo. Goes top-down through the Julia package,
then E0 (non-RL baseline), then E1 (RL). Along the way it calls out
Julia syntax that's non-obvious coming from Python, and flags the
design decisions that were tradeoffs rather than "one right answer".

Cross-refs to PLAN.md sections use `§`; cross-refs to the README use
the section name.

---

## 1. What this project is

A **digital twin of the QalibreMD System Standard Model 130 phantom**
(the NIST/ISMRM quantitative-MRI calibration sphere) + the scaffolding
to train RL agents against it in [KomaMRI](https://github.com/JuliaHealth/KomaMRI.jl).

The twin is intentionally parameterised: every knob an RL loop might
want to vary — field strength, voxel size, rotation, translation, T1/T2
jitter, sphere drop-out — is a field on one `PhantomConfig` struct. An
RL rollout samples a fresh config per episode, calls `build_phantom(cfg)`,
and gets a `KomaMRI.Phantom` it can feed to `simulate`.

---

## 2. Repo map

```
src/                       — the MRISystemPhantom Julia package
├── MRISystemPhantom.jl    — module, `include`s everything else, exports API
├── config.jl              — `PhantomConfig`, `AugmentConfig`  (RL contract)
├── builder.jl             — `SphereDescriptor`, `build_*`, `build_phantom`
├── augment.jl             — rotation, translation, per-spin jitter
├── materials/             — plain-data tables (T1, T2, PD, fiducial, water)
├── geometry/              — `voxelise_sphere`, plate/fiducial layouts
├── sequences/blocks.jl    — `ir_sequence`, `se_sequence`, `generalized_ir_signal`
├── fitting/fits.jl        — `fit_t1_ir`, `fit_t2_se`, `fit_t1_generalized_ir`
├── baselines/e0.jl        — non-RL E0: per-sphere IR/SE sweep + MAPE report
└── rl/e1.jl               — `E1Env`: stateful RL env for single-sphere T1

test/                      — Julia test suite (`] test`). 606 tests.
├── runtests.jl            — includes everything below
├── test_materials.jl      — tables vs manual
├── test_geometry.jl       — voxel volume, 57-point grid, ring radii
├── test_builder.jl        — descriptor counts, water exclusion
├── test_augment.jl        — SO(3), translation, jitter σ
├── test_determinism.jl    — same rng_seed → identical phantom
├── test_simulation.jl     — FID on one sphere → T2 fit within 10 %
├── test_baseline.jl       — E0 MAPE < 3 %
└── test_e1.jl             — fits, env reset/step/terminate

examples/                  — runnable Julia scripts
├── plot_phantom.jl        — render interactive HTML phantom maps
└── e0_baseline.jl         — run E0 and print the per-sphere MAPE tables

python/                    — RL/Python side
├── julia_runtime/         — Julia 1.11 sub-project for juliacall
├── qalibremd_gym/         — Gymnasium wrapper
│   ├── env.py             — `QalibreMDE1Env` (juliacall → E1Env)
│   └── juliapkg.json      — Julia-version pin for juliapkg
├── baseline_e1.py         — fixed 8-block IR grid baseline
├── train_e1.py            — PPO trainer
├── eval_e1.py             — head-to-head baseline vs trained policy
└── test_wrapper.py        — 4 Python-side sanity tests

Project.toml, Manifest.toml — Julia deps (main package on Julia 1.12)
PLAN.md                    — RL experiment plan (what we're building toward)
README.md                  — feature-level usage documentation
EXPLAINER.md               — this document
```

---

## 3. Julia cheat-sheet for Python readers

Only the idioms that actually appear in this codebase.

| Julia | Meaning / Python equivalent |
|---|---|
| `module Foo … end` | top-level namespace; one-per-file convention not required. |
| `include("file.jl")` | splice the file's code into the current scope (like Python's `exec(open(…).read())` but lexical). Modules use this to compose. |
| `using Foo` / `import Foo` | `using` brings exported names into scope; `import` only makes `Foo.bar` accessible. |
| `export foo, bar!` | control what `using` pulls in. |
| `const X = …` | module-level constant. |
| `struct Foo; x::Int; end` | immutable struct (default). |
| `mutable struct` | fields can be reassigned. RL env must be mutable — episodes mutate. |
| `Base.@kwdef struct` | gives the struct a keyword-only constructor. Used for `PhantomConfig`, `AugmentConfig`. |
| `function foo(x; a = 1)` | `;` separates positional from keyword args. |
| `foo! = mutates its argument` | convention (enforced by humans, not the compiler). `e1_reset!` mutates `env`. |
| `a .+ b`, `@. a + b` | broadcasted add; `@.` broadcasts every op in the expression. |
| `[f(x) for x in xs]` | list comprehension (returns `Vector`). |
| `T where T<:Real` | parametric type constraint. |
| `NTuple{3,Float64}` | fixed-length tuple `(Float64,Float64,Float64)`. |
| `@inbounds` | skip bounds-check inside hot loops. |
| `@info "…" field = value` | logger macro; prints key/value pairs. |
| `abs2(x)` = `x*x` (avoids sqrt in `abs`) |
| `error("…")` | raise an exception; `@test_throws ErrorException` asserts it. |
| `Base.@kwdef struct … end` | kwarg-only constructor (like `@dataclass(kw_only=True)` in Python). |

The single most counterintuitive bit: **arrays are column-major, 1-indexed**.
`a[1]` is the first element, not `a[0]`. `a[:, 1]` is the first column.

---

## 4. Reading the package top-to-bottom

### 4.1 `src/MRISystemPhantom.jl`

```julia
module MRISystemPhantom

using KomaMRI
using Random
using LinearAlgebra
import Suppressor

# --- materials (pure data) ------------------------------------------------
include("materials/fiducial.jl")    # defines Relax, FIDUCIAL_PROPS
include("materials/background.jl")  # uses Relax; defines BACKGROUND_WATER
…

export PhantomConfig, AugmentConfig, SphereDescriptor, build_phantom, …

end # module
```

The order of `include` matters: `materials/fiducial.jl` defines the
`Relax` struct, and `background.jl` uses it, so fiducial must come
first. Later includes can depend on any earlier ones. This is just
lexical splicing.

The module exports a public API via `export`. Anything not exported
lives in `MRISystemPhantom.foo` and is fair game internally but marks
a boundary with downstream code.

### 4.2 `config.jl` — the RL contract

```julia
Base.@kwdef struct AugmentConfig
    T1_sigma_rel::Float64       = 0.0
    T2_sigma_rel::Float64       = 0.0
    PD_sigma_abs::Float64       = 0.0
    position_sigma_mm::Float64  = 0.0
    B0_sigma_Hz::Float64        = 0.0
    drop_sphere_p::Float64      = 0.0
end

Base.@kwdef struct PhantomConfig
    field::Symbol                     = :T3
    voxel_size_mm::Float64            = 2.0
    include_plates::Vector{Symbol}    = [:T1, :T2, :PD, :fiducials, :water]
    …
    augment::AugmentConfig            = AugmentConfig()
    rng_seed::Int                     = 0
    custom_sphere_map::Dict{Symbol,Any} = Dict{Symbol,Any}()
    keep_sphere_labels::Union{Nothing,Vector{Symbol}} = nothing
    drop_sphere_labels::Vector{Symbol} = Symbol[]
    custom_sphere_descriptors::Vector{SphereDescriptor} = SphereDescriptor[]
end
```

`Base.@kwdef` gives us keyword-only constructors with defaults, the
closest equivalent to Python `@dataclass(kw_only=True)`. **Why a
single config struct?** So there's exactly one place an RL sampler
can call `PhantomConfig(rotation = …, augment = AugmentConfig(…))`.
PLAN §4 supervisor comment: "How do you avoid that the agent simply
'learns' the manual values?" — answer: by sampling a new
`PhantomConfig` per episode, which plumbs randomness into every layer
below.

`:T3` is a **Symbol** — like a Python string interned as an
identifier. Symbols are the idiomatic Julia key for small dispatch
sets (`:T3` vs `:T15` vs `:new` vs `:legacy`). Cheaper than strings
and signal "this is a tag, not a name".

### 4.3 `materials/*.jl` — plain data

```julia
const T1_ARRAY = Dict(
    :T15 => [1.879, 1.432, 1.027, 0.7513, 0.527, 0.3841, 0.2723, 0.1945,
             0.1378, 0.0947, 0.067, 0.04814, 0.03435, 0.02416],
    :T3  => [1.838, 1.362, 0.9983, 0.7258, 0.5091, 0.367, 0.2587, 0.1847,
             0.1308, 0.0909, 0.0642, 0.04628, 0.03265, 0.02295],
)
```

**Design decision — PLAN §1:** "data tables and geometry kept in plain
Julia source files (no magic numbers buried inside functions)". All
material values live here. If QalibreMD ships an updated
calibration, updating this file is a one-line edit and the whole
stack re-derives from it — no search-and-replace across the codebase.

`Dict{Symbol,Vector{Float64}}` is the key-value table. `:T15` /
`:T3` = 1.5 T / 3 T — the `.` in 1.5 is dropped because Symbols can't
contain dots. See §13 below.

### 4.4 `geometry/sphere.jl`

```julia
function voxelise_sphere(centre::NTuple{3,<:Real}, radius::Real, Δx::Real)
    cx, cy, cz = Float64.(centre)
    r = Float64(radius)
    h = Float64(Δx)
    xs = cx - r : h : cx + r          # a UnitRange
    ys = cy - r : h : cy + r
    zs = cz - r : h : cz + r
    X = reshape(collect(xs), :, 1, 1) # reshape to (Nx, 1, 1)
    Y = reshape(collect(ys), 1, :, 1)
    Z = reshape(collect(zs), 1, 1, :)
    mask = @. (X - cx)^2 + (Y - cy)^2 + (Z - cz)^2 <= r^2
    O = zero(X) .+ zero(Y) .+ zero(Z) # broadcast-compatible zero array
    X3 = X .+ O
    Y3 = Y .+ O
    Z3 = Z .+ O
    return X3[mask], Y3[mask], Z3[mask]
end
```

The trick: **3-D broadcasting via reshape**. `X` is shape `(Nx,1,1)`,
`Y` is `(1,Ny,1)`, `Z` is `(1,1,Nz)`. Broadcasting them together
produces a `(Nx,Ny,Nz)` array without ever materialising the full
Cartesian product explicitly (until the final mask application).

`@.` at line 5 turns the *whole* expression into broadcasted ops —
equivalent to writing `.^2`, `.+`, `.<=` everywhere, but readable.

**Design decision — PLAN §7:** "single-sphere phantom = single voxel
at the limit". For RL training we're mostly using *closed-form* signal
(not voxels), so this voxeliser is invoked mainly for the full-twin
render and for the `:simulate` fidelity-validation backend.

### 4.5 `geometry/plate_layouts.jl`

Provides the two concentric-ring layout for the 14 contrast spheres
per plate (outer 10 at r = 65 mm, inner 4 at r = 28 mm) and the 5×5×5
cubic grid clipped to a 95 mm sphere for the fiducials (which gives
exactly 57 points — tested in `test_geometry.jl`).

**Design decision — PLAN §2.5**: the manual doesn't tabulate exact
per-sphere (x,y) coordinates. We use a parametric ring layout that
matches the photos; when actual coordinates arrive they slot into
this file via a loader, no changes elsewhere.

### 4.6 `builder.jl` — assembling the twin

```julia
struct SphereDescriptor
    centre::NTuple{3,Float64}
    radius::Float64
    ρ::Float64
    T1::Float64
    T2::Float64
    T2s::Float64
    Δw::Float64
    label::Symbol
end
```

A *descriptor* is "everything needed to voxelise and tag one sphere".
It's immutable and serialisable — small enough that drop-out and
override are cheap. The pipeline:

1. Material tables + plate layouts → list of `SphereDescriptor`s.
2. Apply generated-sphere overrides, keep/drop label filters, and random dropout.
3. Voxelise each descriptor → `Phantom` per sphere.
4. Concatenate via `+` (KomaMRI `Phantom` supports `+`).
5. Add background water with final built sphere volumes cut out.
6. Rotate/translate.
7. Per-spin Gaussian jitter.

The key insight: **steps 1–2 are data only, no KomaMRI calls**. This
makes them fast, testable (without needing the simulator), and
decoupled from whether we're using the analytical or the simulated
backend downstream.

### 4.7 `augment.jl`

Rotation uses XYZ-Euler + a standard 3×3 matrix multiply:

```julia
function rotation_matrix(α::Real, β::Real, γ::Real)
    Rx = [1 0 0; 0 cos(α) -sin(α); 0 sin(α) cos(α)]
    Ry = [cos(β) 0 sin(β); 0 1 0; -sin(β) 0 cos(β)]
    Rz = [cos(γ) -sin(γ) 0; sin(γ) cos(γ) 0; 0 0 1]
    Rx * Ry * Rz
end
```

The tests assert:
* `R * R' == I` (orthogonal; `R'` is the transpose — like NumPy's
  `.T`).
* `det(R) == 1` (proper rotation, not a reflection).
* `|r|` is invariant after rotating the phantom.

Per-spin noise is vanilla Gaussian on (T1, T2, ρ, x, y, z, Δw) with
stddevs drawn from `AugmentConfig`. Zero-sigma is a no-op so by
default this function is a pass-through.

---

## 5. The sequence + fitting layer

### 5.1 `sequences/blocks.jl`

Two kinds of entry points:

1. **Pulseq-style sequence constructors** (`ir_sequence`,
   `se_sequence`) — return a KomaMRI `Sequence` you can pass to
   `simulate`. RF durations are computed from `α = 2π·γ·B1·t` (see
   `rf_duration`). For SE we harden the B1 amplitude to 20 μT so the
   180° pulse fits inside the shortest TE we care about — otherwise
   the 180° pulse's duration would exceed TE/2 and the sequence
   constructor would `error`.
2. **Closed-form signal** (`generalized_ir_signal`) — no simulator
   call at all. Under the single-spin, hard-pulse, perfect-spoiling
   idealisations, the ADC magnitude after an IR-prep block is
   analytically `|Mz(TI)| · exp(−t / T2)`. This is ~10,000× faster
   than `simulate` and is the training hot path for E1 (PLAN §7).

`single_spin_phantom(T1, T2)` returns a `Phantom` with one spin at
the origin. That's all you need for non-spatial measurements (E0,
E1). No voxels, no gradients, no spatial encoding.

### 5.2 `fitting/fits.jl`

Three fits, all pure Julia (no external optimisation lib):

| Function | Model | Method |
|---|---|---|
| `fit_t2_se(TEs, mags)` | `S = S0·exp(−TE/T2)` | log-linear closed-form on `(TE, log|S|)`. |
| `fit_t1_ir(TIs, mags)` | `S = |A − B·exp(−TI/T1)|` | log-spaced T1 grid; at each T1 the optimal (A, B) is a 2-eq linear regression. Magnitude sign-ambiguity resolved by trying every "flip the first k points" pattern — only one is right for each candidate T1. |
| `fit_t1_generalized_ir(TIs, αs, mags)` | `S = |A · (1 − (1 − cos α) · exp(−TI/T1))|` | log-spaced T1 grid; at each T1 the sign of `y` is fully determined, so `|A|` is closed-form. No magnitude-ambiguity search needed. |

**Why hand-rolled?** The fits are fast (< 1 ms), deterministic, and
don't drag in a Levenberg-Marquardt dependency. The running fit in
the RL env runs on every step so pure-Julia closed-form-per-candidate
is the right call.

**Design decision — PLAN §4 E1 "running Levenberg-Marquardt
estimate":** the plan calls for LM; in practice a grid over T1 + a
closed-form secondary fit is equivalent and numerically robust
against magnitude-null issues, which LM struggles with.

---

## 6. E0 — the non-RL baseline (`baselines/e0.jl`)

Goal (PLAN §4): verify the twin + simulate + fit pipeline recovers
the manual T1/T2 values on every sphere, MAPE < 3 %.

Orchestrator `run_e0(; field)`:

1. For each T1-array sphere i in 1..14:
   * build an **adaptive TI schedule** — 10 log-spaced points from
     `T1_i/20` to `5·T1_i`. (A single fixed schedule can't cover 23
     ms–1.88 s at 3T; adaptive sizing is the only way to get good
     fits across the whole array.)
   * for each TI, run one IR on a single-spin phantom, grab the
     first ADC sample's magnitude.
   * `fit_t1_ir` → T1 estimate.
2. For each T2-array sphere, analogous with SE and adaptive TE
   schedule.
3. Return per-sphere error + aggregate MAPE.

Observed: **T1 MAPE = 0.51 %, T2 MAPE = 0.014 %** at both 1.5 T and
3 T. Acceptable: it's the *yardstick* the RL agent has to match;
systematic bias on both sides is a wash.

**What the numbers mean and why T1 has a 0.5 % offset**

MAPE (Mean Absolute Percentage Error) measures how far estimated
values are from ground truth on average. T2 at 0.014 % is essentially
perfect. T1 at 0.51 % is also very small, but the error is
*systematic* — a consistent bias rather than random noise.

The culprit is **RF-duration bias**. The inversion-recovery fitting
equation assumes an instantaneous 180° pulse: magnetisation is flipped
at time 0 and then recovers for exactly TI seconds. In reality the
inversion pulse has finite duration (order 1–10 ms), and T1 relaxation
proceeds throughout the pulse. By the time the pulse ends, the
magnetisation hasn't been perfectly inverted — it has already partially
recovered. The *effective* TI is therefore slightly different from the
nominal TI written into the sequence. Because the fit uses the nominal
TI, this produces a small but consistent error in the recovered T1.

Why the bias is field-strength independent: pulse duration is a
hardware/sequence property, not a function of B₀, so the offset is the
same at 1.5 T and 3 T.

Why T2 is unaffected: spin-echo T2 estimation does not depend on
inversion-time precision in the same way, so it is immune to this
particular artefact.

Running it yourself:

```bash
julia --project=. examples/e0_baseline.jl
```

---

## 7. E1 — the first RL experiment

This is where the project earns its name.

### 7.1 MDP formulation (PLAN §3, §4 E1)

* **State** (not all observable to the agent): the simulated spin
  (T1_true, T2_true) drawn fresh per episode from a log-uniform
  range `[20 ms, 2 s]`, plus the history of blocks played so far and
  their signals.
* **Action**: one of `|TI_set| × |α_set| = 6 × 3 = 18` parameterised
  IR-prep blocks. Each block = (inversion-ish prep at flip-angle α,
  delay TI, 90° excitation, ADC). α = 180° is canonical inversion
  recovery; α = 90° is saturation recovery; α = 10° is a small-tip
  prep (barely perturbs Mz). Supervisor suggested this
  parameterisation verbatim in the interim-report annotations.
* **Observation** (float32, length 85): `[last-block ADC magnitudes
  (64) ; log10(running T1 estimate) ; blocks_frac ; time_frac ;
  per-action play counts / max_blocks]`. The per-action-count tail
  is what lets a memoryless MLP policy know what it's already tried.
* **Reward** per step:
  ```
  r = −|T̂₁ − T₁_true| / T₁_true      (dense accuracy error)
     − λ · block_time / budget        (dense time cost)
     + terminal_bonus   iff episode is done and error < 3 %
  ```
* **Episode termination**: hit `max_blocks` (default 12) OR scan-time
  budget exceeded.

### 7.2 `rl/e1.jl` — the stateful environment

```julia
mutable struct E1Env
    # --- static config ---
    TI_set::Vector{Float64}
    α_set::Vector{Float64}
    action_table::Vector{Tuple{Float64,Float64}}
    …
    backend::Symbol                 # :analytical (fast) or :simulate (slow, fidelity)

    # --- episode state ---
    rng::MersenneTwister
    T1_true::Float64
    T2_true::Float64
    TIs_used::Vector{Float64}
    αs_used::Vector{Float64}
    mags_used::Vector{Float64}
    last_signal::Vector{Float64}
    action_counts::Vector{Int}
    n_blocks::Int
    time_used_s::Float64
    T1_est::Float64
    done::Bool
end
```

It's `mutable struct` because `e1_reset!` and `e1_step!` both mutate
it. Python drives the env by calling `e1_reset_b` / `e1_step_b` (see
§8 on the `_b` alias — Python can't spell `!`).

`_block_signal` dispatches on `env.backend`:
* `:analytical` — calls `generalized_ir_signal` (closed-form, ~μs).
* `:simulate` — builds a `Sequence` and calls `KomaMRI.simulate`
  (~ms). Used only for the fidelity cross-check in `test_e1.jl`
  — training uses `:analytical`.

**Anti-memorisation** (PLAN's biggest supervisor concern):
`e1_reset!` samples a *fresh* T1 from a continuous distribution.
The agent never sees the manual 14 values as a finite set, so there's
nothing to memorise.

Reward design is explicit — each term is a config field (`λ_time`,
`terminal_bonus`, `success_tol`) so tuning is trivial.

### 7.3 The `_b` alias at the bottom of the file

```julia
const e1_reset_b = e1_reset!
const e1_step_b  = e1_step!
```

Python identifiers can't contain `!`, and juliacall doesn't
auto-translate bang names. So we expose an alias without `!`. The
`!` convention still applies Julia-side — `e1_reset!` is the canonical
name and what tests use.

---

## 8. Python side (`python/`)

### 8.1 Why a separate Julia sub-project?

`juliacall` 0.9.31 requires Julia **≤ 1.11** (the bundled
`PythonCall.jl` doesn't load on 1.12 yet). Our main project's
Manifest is built with 1.12 (because that's the current juliaup
release channel). So `python/julia_runtime/` is a second, parallel
Julia environment pinned to 1.11, with:

```toml
[deps]
PythonCall = "6099a3de-0909-46bc-b1f4-468b9a2dfc0d"
MRISystemPhantom = "6229b35e-1f84-4e2c-829e-3cca79e866b7"
```

The `MRISystemPhantom` entry is **dev-pathed** to the repo root
(`Pkg.develop(path=".")` when bootstrapping), so any edit to
`src/*.jl` is picked up by the Python layer without a reinstall.

When `juliacall` 0.9.32+ drops Julia 1.12 support, this sub-project
can go away and we can point `juliacall` at the main project
directly.

### 8.2 `python/qalibremd_gym/env.py`

```python
def _ensure_julia(project_dir=None):
    …
    j11 = Path.home() / ".julia" / "juliaup" / "julia-1.11.9+…" / "bin" / "julia"
    if j11.exists():
        os.environ.setdefault("PYTHON_JULIAPKG_EXE", str(j11))
    os.environ.setdefault("PYTHON_JULIAPKG_OFFLINE", "yes")
    os.environ.setdefault("PYTHON_JULIAPKG_PROJECT", runtime_proj)

    from juliacall import Main as jl
    jl.seval("using MRISystemPhantom")
```

The three env vars:
* `PYTHON_JULIAPKG_EXE` — force juliacall to use Julia 1.11 from
  juliaup (otherwise it'd download its own Julia on first run).
* `PYTHON_JULIAPKG_OFFLINE=yes` — don't let juliapkg touch
  Project.toml or try to resolve anything. The sub-project is
  already set up; we don't want juliapkg second-guessing.
* `PYTHON_JULIAPKG_PROJECT` — point at `python/julia_runtime/`.

The `QalibreMDE1Env` class is a thin Gymnasium shell:

```python
def step(self, action):
    a_julia = int(action) + 1          # Python 0-based → Julia 1-based
    obs, reward, done, info_dict = _JL_QMD.e1_step_b(self._env, a_julia)
    obs = np.asarray(obs, dtype=np.float32)
    info = {str(k): _to_py(v) for k, v in info_dict.items()}
    return obs, float(reward), bool(done), False, info
```

The `False` is Gymnasium's `truncated` flag — we don't truncate for
wall-clock reasons, only for scan-time budget (handled inside the
Julia env). Observation is auto-converted to `np.float32` because
Stable-Baselines3 expects that for Box spaces.

### 8.3 Baseline, trainer, evaluator

`baseline_e1.py`: plays a fixed 8-block IR grid at α = 180°. No
learning. Acts as both a simulator sanity check and the
PLAN-specified yardstick (MAPE ≈ 0.54 %, success rate 100 %).

`train_e1.py`: PPO with Stable-Baselines3, MLP policy `[128, 128]`.
The `EvalAgainstBaselineCallback` rolls a deterministic version of
the current policy on a held-out seed offset every `eval_interval`
steps and logs `{step, mape_pct, p90_pct, success_rate}` to
`runs/e1/<name>/eval_history.json`.

`eval_e1.py`: final head-to-head. Reports MAPE, median error, p90,
mean scan time, mean reward, and success rate for both baseline and
trained agent on the same seed set.

### 8.4 `python/test_wrapper.py`

Four assertions exercising the full Python → Julia → env loop:
spaces & reset, step & terminate, determinism on seed, and an IR
ladder that should drop MAPE < 10 %. Running

```bash
python python/test_wrapper.py
```

is the quickest way to check the plumbing after any change.

---

## 9. How a training run unfolds

```
python/train_e1.py
    │
    ▼
DummyVecEnv  ─ [QalibreMDE1Env worker]
    │                │
    │                ▼
    │         juliacall → Julia 1.11 runtime
    │                │
    │                ▼
    │         MRISystemPhantom.e1_step_b(env, action)
    │                │
    │                ▼
    │         generalized_ir_signal (closed-form, ~μs)
    │                +
    │         fit_t1_generalized_ir (grid search, ~μs)
    │                │
    │   (obs, reward, done, info) → back across juliacall
    ▼
PPO collect rollout (n_steps = 256)
    │
    ▼
PPO gradient step on MLP policy (CPU, fast)
    │
    ▼
EvalAgainstBaselineCallback (every 5k steps):
    roll deterministic policy on held-out seeds → log MAPE to JSON
    │
    ▼
Repeat until total_timesteps reached.
```

At ~4 ms/step on the default settings, 50k steps is ~3 minutes of
wallclock. The training hot path does **no** KomaMRI calls — that
side lies dormant until you flip `backend="simulate"` for a fidelity
spot-check.

---

## 10. Design decisions — the whys behind the whats

A compact list of the calls that were tradeoffs rather than the only
option, with their reasoning.

| Decision | Alternative | Why this way |
|---|---|---|
| Single `PhantomConfig` struct as the RL contract | expose many free functions | Exactly one place to sample randomness per episode — anti-memorisation hinges on this. |
| Data tables in `.jl` files, not YAML/JSON | config files | A Julia file can also hold `const` constants and helper lookups; code already reads it; one fewer format to parse. |
| Closed-form signal for E1 training | call `KomaMRI.simulate` every step | RL wants 10⁵–10⁶ env steps. `simulate` is ~ms; closed-form is ~μs. PLAN §7 explicitly allows this for single-sphere/non-spatial measurements. Fidelity-verified against `:simulate` in tests. |
| Grid-search T1 fit + closed-form |A| | LevMarq fit via `LsqFit.jl` | Fast enough, deterministic, no extra dep, robust to magnitude nulls (important for IR). |
| Separate Julia 1.11 runtime for Python | bump everything to 1.11 | `juliacall`'s Julia-version cap is its problem, not ours. Main project stays modern; Python gets its own compatible runtime. |
| 18-action discrete space for E1 | continuous | PPO converges faster on discrete; PLAN §4 E1 specifies discrete. Continuous is E2. |
| Per-action play-count in observation | pure last-signal | Without memory (MLP policy), the agent can't avoid replaying the same action. Play-counts are the minimum sufficient statistic. |
| Hand-rolled Gymnasium API (not SB3-included) | brax / rllib wrappers | Gym is the lingua franca of RL libraries. One wrapper, many downstream options. |

---

## 11. Key files to read, in order, if you want to *really* grok it

1. **`PLAN.md`** — the "what we're building and why", including the
   supervisor's annotations.
2. **`src/config.jl`** — the RL contract (3 minutes).
3. **`src/builder.jl`** — top-down pipeline from config to Phantom.
4. **`src/rl/e1.jl`** — the stateful env.
5. **`python/qalibremd_gym/env.py`** — the thinnest possible Python
   shell over (4).
6. **`python/train_e1.py`** — PPO integration and what the eval
   callback actually measures.
7. **`examples/e0_baseline.jl`** — non-RL yardstick. If E0 breaks,
   RL will too; this script is a 30-second sanity probe.

---

## 12. Common-failure debug guide

* **"juliacall failed to start"** — almost always a Julia-version
  mismatch. `juliaup status` should list 1.11.9; if not, `juliaup add
  1.11`. The runtime sub-project is rebuilt with `julia +1.11
  --project=python/julia_runtime -e 'using Pkg;
  Pkg.develop(path="."); Pkg.instantiate()'`.
* **"Error: Missing source file for QalibreMDJuliaRuntime"** — a
  stale `Project.toml` in `python/julia_runtime/` had `name/uuid` set,
  making Julia expect a `src/QalibreMDJuliaRuntime.jl`. Fix: remove
  the `name`/`uuid`/`version` keys (a plain env has none).
* **"PPO policy collapses to a single action, MAPE 50 %+"** — reward
  scale is the usual cause. Log episode-mean reward and make sure
  the accuracy term (~ −0.001 → −1.0) and the time term
  (~ −0.01/step) are in roughly the same order. Alternative: clip the
  per-step error at 1.0 and use log-error on the terminal. PLAN §5
  discusses this.
* **"Tests pass but the full `simulate` path hangs"** — Kaleido (the
  Plotly renderer used by `plot_phantom_map`) can dead-lock in
  headless containers. Kill `kaleido` processes; set
  `JULIA_DEBUG=all` to confirm.

---

## 13. Glossary

| Term | Meaning |
|---|---|
| **QalibreMD Model 130** | NIST/ISMRM-traceable phantom, 57 fiducial + 42 contrast spheres. |
| **T1 / T2 / PD** | longitudinal / transverse relaxation times; proton density. |
| **IR** | inversion-recovery; canonical T1 measurement technique. |
| **SE** | spin-echo; canonical T2 measurement. |
| **SR** | saturation-recovery; T1 via a 90° prep instead of 180°. |
| **FID** | free-induction decay; the bare signal after a single excitation. |
| **MAPE** | mean absolute percentage error. |
| **TE / TI / TR** | echo time / inversion time / repetition time (milliseconds). |
| **B1** | RF field amplitude (tesla). |
| **Pulseq** | vendor-neutral pulse-sequence format. Used here abstractly. |
| **`:T3`, `:T15`** | Julia Symbol tags for 3 T and 1.5 T (the "5" is the decimal). |
| **`α`** | RF flip angle, radians. |
| **`ρ`** | spin density. |
| **`Δw`** | off-resonance angular frequency (rad/s). |
| **MDP** | Markov decision process — state, action, transition, reward. |
| **PPO** | proximal policy optimisation; the default modern on-policy RL algo. |
| **juliacall** | Python → Julia bridge. |

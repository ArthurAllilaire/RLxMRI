# Things to understand:

1. Gradient spoiling
2. pixel_grid_overlay.jl
- blocks.jl - the new gre_2d_sequence
- test_simulation.jl - why is it there? what does it actually test?
- understand e two code
- send email to andreas plus wayne
- 

check why this is so slow:

Turns out its 5 seconds per simulation

run cr optimal alpha benchlines
merge cr optimal alpha and cr optimal together once understood
understand cr optimal - at least high level

seems like TR is assumed to be 0.5 seconds? turns out thats just hardcoded lowest TR val of E2

# Next steps:

julia +1.11 --project=python/julia_runtime_gpu -e 'import Pkg; Pkg.instantiate(); Pkg.precompile(); using CUDA; CUDA.versioninfo()'
julia +1.11 --project=python/julia_runtime_gpu \
-e 'import Pkg; Pkg.add("CUDA"); Pkg.resolve(); Pkg.instantiate(); Pkg.precompile(); using CUDA; CUDA.versioninfo()'

PYTHON_JULIAPKG_PROJECT="$PWD/python/julia_runtime_gpu" \
PYTHON_JULIAPKG_OFFLINE=yes PYTHON_JULIACALL_HANDLE_SIGNALS=yes \
PYTHON_JULIACALL_THREADS=3 JULIA_NUM_THREADS=3 \
PYTHONPATH=python python -c '
import time
from qalibremd_gym.env_e2 import QalibreMDE2Env

env = QalibreMDE2Env(
    use_gpu=True,
    subset_size=5,
    forced_sphere_indices=[1,3,6,8,14],
    t1_sampler="linear_uniform_range",
    pose_mode="inplane_jitter",
    translation_sigma_mm=2.0,
    rotation_sigma_rad=0.05,
    roi_radius=1,
    fix_te=True,
    learn_alpha=True,
    log_ti_action=True,
    Nfe=64,
    Npe=64,
    max_blocks=8,
    time_budget_s=240.0,
    forward_model="bloch",
    water_model="bloch",
)
obs, info = env.reset(seed=1)
for i in range(8):
    t0 = time.time()
    obs, reward, terminated, truncated, info = env.step([0.0, 0.0, 0.0])
    print("step", i + 1, "dt", round(time.time() - t0, 2), "mape", info.get("mape"), flush=True)
    if terminated or truncated:
        break
'

PYTHON_JULIAPKG_PROJECT="$PWD/python/julia_runtime_gpu" \
PYTHON_JULIAPKG_OFFLINE=yes \
python -c 'from juliacall import Main as jl; jl.seval("using CUDA"); print(jl.seval("CUDA.functional()")); print(jl.seval("CUDA.device()"))'

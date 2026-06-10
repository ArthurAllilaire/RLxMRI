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


# R1 — TI-coverage-histogram run (E2_HISTORY_ABLATION.md §4)
mkdir -p runs/e2/mf_runB_5sphere_hist_560s_gpu
PYTHON_JULIAPKG_PROJECT="$PWD/python/julia_runtime_gpu" \
PYTHON_JULIAPKG_OFFLINE=yes PYTHON_JULIACALL_HANDLE_SIGNALS=yes \
PYTHON_JULIACALL_THREADS=3 JULIA_NUM_THREADS=3 \
PYTHONUNBUFFERED=1 python -u python/train_e2_mf.py \
--out runs/e2/mf_runB_5sphere_hist_560s_gpu \
--multi-fidelity --mf-plan analytic,cached3,cached,full3,full \
--reward-mode delta_log_mape --mape-alpha 1.0 \
--fix-te --log-ti-action --include-ti-history \
--n-envs 1 --field T15 --time-budget 560 --max-blocks 20 \
--subset-size 5 --forced-sphere-indices 1,3,6,8,14 \
--t1-sampler linear_uniform_range \
--pose-mode inplane_jitter --translation-sigma-mm 2.0 \
--rotation-sigma-rad 0.05 --roi-radius 1 \
--use-gpu \
--train-seed 0 --eval-seed 500000 \
--mf-budget-hours 9 --mf-full-reserve-frac 0.20 \
--mf-min-steps 4096,8192,8192,8192,0 \
--mf-max-steps 20000,160000,160000,80000,300000 \
--n-steps 512 --batch-size 64 \
--eval-interval 10000 --eval-episodes 20 \
--mf-decision-rollouts 4 --mf-probe-episodes-full 4 \
--mf-global-best-episodes 12 \
--mf-use-lookahead --mf-lookahead-rollouts 1 \
--mf-lookahead-margin 1.15 --mf-slope-collapse-frac 0.25 \
2>&1 | tee runs/e2/mf_runB_5sphere_hist_560s_gpu/run.log

# R2 — RecurrentPPO (LSTM, base obs) run (E2_HISTORY_ABLATION.md §4)
mkdir -p runs/e2/mf_runB_5sphere_lstm_560s_gpu
PYTHON_JULIAPKG_PROJECT="$PWD/python/julia_runtime_gpu" \
PYTHON_JULIAPKG_OFFLINE=yes PYTHON_JULIACALL_HANDLE_SIGNALS=yes \
PYTHON_JULIACALL_THREADS=3 JULIA_NUM_THREADS=3 \
PYTHONUNBUFFERED=1 python -u python/train_e2_mf.py \
--out runs/e2/mf_runB_5sphere_lstm_560s_gpu \
--multi-fidelity --mf-plan analytic,cached3,cached,full3,full \
--reward-mode delta_log_mape --mape-alpha 1.0 \
--fix-te --log-ti-action --recurrent \
--n-envs 1 --field T15 --time-budget 560 --max-blocks 20 \
--subset-size 5 --forced-sphere-indices 1,3,6,8,14 \
--t1-sampler linear_uniform_range \
--pose-mode inplane_jitter --translation-sigma-mm 2.0 \
--rotation-sigma-rad 0.05 --roi-radius 1 \
--use-gpu \
--train-seed 0 --eval-seed 500000 \
--mf-budget-hours 9 --mf-full-reserve-frac 0.20 \
--mf-min-steps 4096,8192,8192,8192,0 \
--mf-max-steps 20000,160000,160000,80000,300000 \
--n-steps 512 --batch-size 64 \
--eval-interval 10000 --eval-episodes 20 \
--mf-decision-rollouts 4 --mf-probe-episodes-full 4 \
--mf-global-best-episodes 12 \
--mf-use-lookahead --mf-lookahead-rollouts 1 \
--mf-lookahead-margin 1.15 --mf-slope-collapse-frac 0.25 \
2>&1 | tee runs/e2/mf_runB_5sphere_lstm_560s_gpu/run.log
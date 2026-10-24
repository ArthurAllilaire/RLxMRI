# ISMRM 2027 abstract - figure plan

Five figures agreed with Andreas on 22 Oct 2026 (`meeting_with_supervisor.md`).
Deadline: upload by 28 Oct; Andreas wants the figures and captions in a Word document first.
Each figure section below is written to be handed to one subagent as a self-contained brief.
Paths are relative to the RLxMRI repo root; `../MRISystemPhantom.jl/` is the sibling package.

## Ground rules for every figure

- **Sources of truth:** the run outputs under `runs/e2/`, the report sources in `report_latex/`, and the presentation sources in `presentation/`.
- **Do not trust `ismrm_latex/british/`.** It is unchecked LLM work from an earlier session (flagged in its README); its figures have visible defects and its numbers are unverified.
- **`submission.md` is partly wrong** (see "Corrections" below); re-derive numbers from the JSON files rather than copying them.
- **Seeds.** Held-out = seeds 600000-600023 (`global_best/eval_summary.json`, `global_best/eval_t1err_heldout.json`, `baselines_heldout_*`).
  The per-block schedules in `global_best/diagnostics/episodes.json` are seed 500000, the validation stream used during training.
  Never call seed-500000 data "held-out"; call it "validation scans".
- **Runtime.** Pure Julia runs locally (`julia --project=.`, Julia 1.12).
  Python-to-juliacall scripts do not run on macOS, so anything that needs a trained policy `.zip` (e.g. a new `diagnose_e2.py` run) needs the Linux/GPU VM.
  `python/diagnose_e2.py` imports juliacall at module load, so write standalone plotting readers instead of importing it.
- **Outputs.** Scripts in `ismrm_latex/vancouver/scripts/`, figures in `ismrm_latex/vancouver/figures/` as PNG (300 dpi) and PDF.
  Every number in a figure comes from a file read by its script, never typed in by hand.
- **Shared style: `scripts/figstyle.py` (done).** Import it in every Python figure script and call `figstyle.apply()`.
  The scheme follows the phantom viewer: viridis encodes T1 everywhere (Figs 1, 3B, 4).
  Teal `ACCENT` marks the TI-histogram policy and anything the policy controls (TI and TR in the sequence diagram).
  Slate `RL_OTHER` marks the other RL policies, told apart by direct labels rather than colour.
  Greys mark the fixed protocols (log grid dark grey dashed, Cramer-Rao light grey dotted).
  `RUNG_COLOURS` is a single-hue light-to-dark ramp for the five fidelity rungs (Figs 1C, 2B, 3A).
  `DIFF_CMAP` (RdBu_r, neutral at zero) is for Fig 4's difference maps.
  The accent and secondary pair passes the dataviz palette validator; pure viridis colours failed it (too muted).
  For the TikZ figure, use the same hex values and a sans font.
- **Chart rules** (load the `dataviz` skill before plotting): never two y-axes, linear axes unless the data spans orders of magnitude, direct labels over legends where possible, and render the output and look at it before calling it done.
- **Captions:** at most 500 characters including spaces.
  Methods detail goes in captions, because the body has only 750 words.
- **Preview figure:** ISMRM also needs one caption-less preview figure, legible at phone width (see the end of this file).

## Corrections found while planning

1. **Success rate is per scan, not per sphere.** It is the share of the 24 scans whose 5-sphere MAPE is under 5% (`python/eval_e2.py:168`, `python/baseline_e2.py:324-325`).
   Arthur described it to Andreas as per-sphere; the per-sphere rate for the histogram arm would be 85.0%.
2. **P90** is the 90th percentile of the 24 per-scan MAPEs (`eval_e2.py:166`); Andreas suggested dropping it.
3. **Blocks and times in `submission.md` are wrong.** `diagnose_e2.py` records a final proposed block that the budget guard rejected (`julia/rl/e2.jl:888-893`), and those values are seed 500000, not held-out.
   For example, "LSTM 577 s" exceeds the 560 s budget.
   Held-out mean scan times from the eval logs: histogram 541 s, no memory 548 s, sigma 544 s, LSTM 525 s, RL 240 s 227 s.
4. **The fixed log grid at 560 s ran 9 blocks (504 s), not 10**; its TI = 3.0 s block never ran.
   At 240 s it ran 3 blocks (163 s), not 4.
5. **The Cramer-Rao schedule minimises the expected bound over the 14 nominal plate T1s** (`"formulation": "expected_loss_all_14_pool"` in `cr_optimal_schedule.json`), not the 5 randomised spheres.
   This is a fair fixed design: the 14 values span the range the scan T1s are drawn from (0.024-1.88 s), and a fixed protocol cannot know a scan's T1s.
   Evidence that the pool choice is not what holds it back: a Cramer-Rao "oracle" solved per scan with that scan's true T1s (`evaluate_cr_oracle`, `python/baseline_e2.py:346`) reached 7.05% on the validation seeds, against 7.34% for the pool schedule (`runs/e2/baseline_5sphere_560s_gpu/baseline_summary.json`, a June run).
   The 7.34% reproduces bitwise with today's code (Fig 4 replay).
   The same file's log grid (5.71%, 10 blocks) predates the change that stops the grid at 9 blocks; today's code gives 5.17% on those seeds, so do not quote 5.71%.
   The Cramer-Rao criterion (variance of an unbiased estimator) is what underperforms at this SNR and block count.
   Describe it in captions as "optimised for the plate's T1 range".
6. **The action is (TI, TR) only.**
   TE is fixed at 20 ms and the flip angle at 90° (`run_config.json`: `fix_te: true, learn_alpha: false`).
   The deck and report loop diagram label the action "(TI, TE, TR, alpha)", and the meeting notes say "TI/TE"; both must be fixed in Fig 2.
   Exception: the 240 s no-memory run (`mf_runB_cached3_cached_full3_full_gpu`) also learned the flip angle (`learn_alpha: true`; 55-90 degrees, mean 89.8, on validation scans), so its Fig 5 row carries a footnote.
11. **The LSTM's scored checkpoint never trained at full Bloch.**
    Its `global_best/best_meta.json` is from the `cached` rung at step 16,384, before the full stage began.
    That, more than the 7.3x fewer full-Bloch steps, is why it lags; say "checkpoint selected before any full-Bloch training".
7. **Spheres are 15 mm in diameter** (`CONTRAST_RADIUS_M = 7.5e-3`, `../MRISystemPhantom.jl/src/builder.jl:4`).
   The report says "15 mm radius" (`report_latex/chapters/MRISystemPhantom.tex:85,93,101,299`); do not copy that.
8. **Inner and outer loop are swapped in the Granola summary.**
   The report calls one environment step the inner RL loop and the fidelity curriculum the outer loop; use the report's naming.
9. **The image Granola attached under Figure 1 belongs to Figure 4.**
   It is the 32x64 pixel-grid overlay (IR-SE, TI 0.1 s, FOV 200 mm, 15 mm spheres) that Andreas pointed at when describing a T1 map.
10. **The "~30x" is the cost range of the fidelity ladder, not the water cache's speed-up.**
    It is wall-clock per RL step (simulate one 32-shot IR-SE block, reconstruct, fit): 0.034 s for the analytic model against 1.03 s for full Bloch at 1 mm, on the 14-sphere plate on CPU (Run A).
    The water cache alone is about 3x per step (0.34 s against 1.03 s), or 8x on the simulation part.
    On the GPU 5-sphere histogram run the whole range is about 5.6x (0.019 s/step analytic, 0.092 s cached, 0.108 s full, including PPO overhead), because the GPU simulates spins in parallel.
    Any figure or caption that quotes 30x must say "14-sphere plate, CPU".

## Figure 1 - Digital twin, augmentation and fidelity ladder

**Shows:** what MRISystemPhantom.jl is and why Bloch-in-the-loop RL is affordable.

**Panels:**
- **(A) 3D render of the twin**, spheres coloured by T1, water translucent, plates labelled.
  Field strength does not matter for this panel (Arthur, 22 Oct); use whichever renders best, and 1.5 T if equal, since all runs used 1.5 T.
- **(B) Build two candidates and choose by eye:**
  - **B1, randomisation:** the T1-plate slab from three random training phantoms.
    Each scan redraws the five active spheres' T1 (uniform over 0.024-1.88 s, T2/T1 ratio kept), shifts the phantom in-plane (sigma 2 mm) and rotates it (sigma 0.05 rad, about 3 degrees); the other nine sphere positions are removed and filled with water (verified from the slab data by the Fig 1 agent).
    Pose jitter applies on the analytic and full-Bloch rungs only; the cached rungs fix the pose.
    Visually, the five outlined spheres change colour and the pose shift is subtle.
    It shows the agent cannot memorise one phantom, which is why fixed protocols struggle.
  - **B2, slices:** the nominal twin sliced through the T1, T2 and PD plates, showing the twin's structure.
- **(C) Fidelity ladder inset:** cost per RL step for analytic, cached3, cached, full3, full on a **single linear axis** (Arthur dislikes the log axis, and the dual-axis design breaks the one-axis rule).
  Bars filled with `RUNG_COLOURS`, value printed on each bar, and the spin count (sphere + water) as a small label under each rung instead of a second axis.

**Existing sources:**
- 3D viewer: `plot_phantom_html` in `../MRISystemPhantom.jl/ext/MRISystemPhantomPlotlyJSExt.jl:55-210`, driven by `../MRISystemPhantom.jl/examples/plot_phantom_3d.jl:24-28`.
  The report and deck images (`report_latex/imgs/mri_system_phantom/T1_voxel_phantom.png`, `presentation/figs/phantomdocs.png`) are browser screenshots, with the dropdown, slider and hover tooltip visible, and at 3 T.
  Do not reuse them.
- Random phantoms: `RandomPhantomConfig` / `sample_phantom_config` (used by `e2_reset!`, `julia/rl/e2.jl:789-792`); ground-truth maps via `phantom_parameter_map` (`../MRISystemPhantom.jl/src/geometry/projection.jl:42-57`).
- Ladder: `fig4_ladder()` in `presentation/make_figs.py:300-377`, numbers hard-coded from Table `tab:e2-fidelity-ladder` (`report_latex/chapters/AdaptiveRLMRI.tex:692-716`).
  Spin counts come from `scripts/e2_spin_counts.json`; costs are Run A CPU stage averages (raw: `sec_per_step` in `runs/e2/mf_v2_runA_cpu_9h/fidelity_history.json`).
  GPU 5-sphere costs: `sec_per_step` in `runs/e2/mf_runB_5sphere_hist_560s_gpu/fidelity_history.json`.

**Build:**
1. (A) Write a Julia script that builds the PlotlyJS figure for `PhantomConfig(field = :T15, ...)`, removes `updatemenus` and `sliders`, sets the camera, font sizes and a colourbar in seconds, and exports a static PNG.
   `plot_phantom_html` only saves HTML, so either add a function to the extension that returns the figure, or copy its trace-building code into the script.
   Static export needs Kaleido (PlotlyKaleido for PlotlyJS 0.18).
   Fallback: dump spin positions, T1 and plate labels from Julia to `.npz` and render with Python plotly + kaleido (neither is installed in `.venv` yet).
2. (B1) In Julia, build `E2Env` with the hist run's `base_env_kwargs` and call `e2_reset!(env; rng_seed = s)` for three seeds, so the phantoms come from the exact training distribution.
   Export the T1 slab map at high resolution (truth, not an image, so 1 mm is fine) and plot with matplotlib, outlining the five randomised spheres (`sphere_indices` [1, 3, 6, 8, 14]).
   (B2) Same tooling with `phantom_parameter_map` at the three plate heights of the nominal twin.
3. (C) Rewrite `fig4_ladder()` in the shared style as described in panel (C): no title, linear seconds axis, no second axis.
   Use the CPU 14-sphere numbers (they isolate simulation cost and match the report) and say "14-sphere plate, CPU" on the panel or in the caption.

**Risks:** Kaleido install; legibility of a dense 3D sphere cloud at abstract size (consider showing only the T1 plate in colour and the other plates in grey).

**Draft caption (check length):**
MRISystemPhantom.jl: open-source digital twin of the ISMRM/NIST system phantom (Calibur QalibreMD Model 130), returning KomaMRI phantoms. (A) Twin at 1.5 T, spheres coloured by T1 within the water-filled housing. (B) Three random training phantoms: five T1-plate spheres drawn uniformly over 0.024-1.88 s, pose jittered in-plane. (C) Fidelity ladder: Bloch-simulating only sphere spins and adding cached water cuts cost per RL step (14-sphere plate, CPU).

**Effort:** half a day; Kaleido is the unknown.

## Figure 2 - Inner and outer training loops

**Shows:** enough of the training setup for someone to reproduce it.

**Panels:**
- **(A) Inner loop, one environment step:** policy -> (TI, TR) -> budget guard -> forward model at the current rung -> noisy k-space -> image -> ROI signals -> generalised-IR T1 fit over all blocks so far -> observation and reward -> policy; plus episode reset and the PPO update.
- **(B) Outer loop, fidelity curriculum:** the ladder analytic -> cached3 -> cached -> full3 -> full -> best policy at full, with the promotion logic at one rung (dual probe every 4 rollouts; plateau with lookahead, ranking breakdown, bias intolerance, budget reserve; promotion copies weights and observation statistics and resets reward normalisation).

**Existing sources (both TikZ):**
- Env loop: inline in `report_latex/chapters/AdaptiveRLMRI.tex:197-296` (`fig:e2-env-loop`); standalone copy `presentation/envloop_diagram.tex`, built by `presentation/make_envloop.sh` into `presentation/figs/fig_envloop.png`.
- Controller: inline in `report_latex/chapters/AdaptiveRLMRI.tex:541-655` (`fig:e2-mf-controller`); standalone copy `presentation/controller_diagram.tex` (built with lualatex + gs into `presentation/figs/fig5_controller.png`).

**Build:**
1. Copy both standalone `.tex` files into one `figures/fig2_loops.tex` with panels (A) and (B) side by side (A about 60% of the width).
2. Edits to (A):
   - Action label "(TI, TE, TR, alpha)" -> "(TI, TR)".
   - Observation: add "12-bin histogram of executed log TI" (the winning arm's memory) next to "log10 T1 est, budget".
   - Reset box: replace the code identifiers `RandomPhantomConfig -> sample_phantom_config` with plain language ("random phantom: 5 sphere T1s, pose jitter").
   - Forward model: "IR-SE block: 32 shots, one k-space line each, + noise", and mark that its fidelity is set by panel (B).
   - Fit box: "per-sphere generalised-IR T1 fit, all blocks so far".
3. Edits to (B): keep the ladder row and the four promotion triggers with their thresholds (Andreas: show how it was done, in enough detail to replicate), but compress the layout; say "full-Bloch probe phantoms not used for training" rather than "held-out", since the probes use the seed-500000 stream.
4. Switch TikZ to a sans font so it matches the matplotlib figures; compile with `lualatex`, convert with `pdftoppm -r 300`.

**Risks:** legibility at abstract width; (B) is dense and may need its own row instead of sitting beside (A).

**Draft caption (check length):**
Training loops. (A) One environment step: the PPO policy chooses TI and TR for the next IR-SE block (32 shots, TE 20 ms); the block is simulated with k-space noise and reconstructed, and T1 is refitted per sphere from all blocks so far. Observation: log T1 estimates, budget used, 12-bin histogram of executed log-TIs. Reward: drop in log MAPE. (B) One policy is warm-started up the fidelity ladder; promotion is judged on full-Bloch probe scans.

**Effort:** half a day (TikZ editing).

## Figure 3 - Training curve, TI choices, sequence block

**Shows:** the best run (TI histogram, 560 s) improving during training, the TIs it then picks scan by scan, and the pulse sequence whose TI and TR it controls.

**Panels:**
- **(A) Full-Bloch MAPE versus training wall-clock** for the hist run, with the fidelity rungs shaded and the final held-out score marked.
- **(B) TI versus block index**, one line per scan, opening probe shaded.
- **(C) Inset in (B): one IR-SE shot** (180 inversion, TI, 90 excitation, 180 refocus, readout of one k-space line, recovery to TR), with TI and TR highlighted as the policy's choices.

**Data (all in `runs/e2/mf_runB_5sphere_hist_560s_gpu/`):**
- `fidelity_history.json`: 4-episode full-Bloch probes (`mape_H_pct`) and `stage_start`/`stage_end` events; `wall_s` is absolute Unix time.
  Rung switches at 0.010 h (analytic -> cached3), 0.196 h (-> cached), 0.410 h (-> full3), 0.739 h (-> full); end at 8.967 h.
- `stage4_full/eval_history.json`: the only curve inside the 8 h full stage, 28 points of 20-episode full-Bloch evals (`mape_pct`); `wall_s` is relative to the full stage start, so add 0.739 h.
- `global_best/best_meta.json`: selected checkpoint at step 242,769, about 6.98 h.
- `global_best/eval_summary.json`: held-out MAPE 2.93% [2.58, 3.30].
- `global_best/diagnostics/episodes.json`: 24 validation scans (seed 500000), per block `TI`, `TR`, `TE`, `mape`, `T1_est_mean`, `T1_est_at_decision`, plus `T1_true` per sphere.
  It contains literal `NaN` (Python's `json` accepts it).
  In 10 of 24 scans the last entry is a proposal the budget guard rejected: drop it when `32 * sum(TR) > 560`.
- Sequence: `ir_se_2d_sequence` in `../MRISystemPhantom.jl/src/sequences/blocks.jl:217-277`, called from `julia/rl/e2.jl:580-586` with FOV 0.2 m, Nfe 64, Npe 32.
  One shot: hard 180 inversion (0.587 ms), TI centre-to-centre, hard 90 (0.294 ms), Gx prewinder and Gy phase encode (0.5 ms), TE/2, hard 180 refocus, readout `ADC(64, 1 ms)`, delay to TR.
  Non-selective pulses on a 1 mm slab, no crushers, one ky line per shot, 32 shots per block.

**Existing sources:**
- `report_latex/imgs/e2_rl/runB_curriculum_probes.png` (240 s / no-memory / sigma runs; no generating script exists anywhere) and `report_latex/imgs/e2_rl/runA_moneyplot.png` (Run A, with stage lines; made by `python/plot_mf_curriculum.py`).
  Both show a straight line from about 1 h to 9 h because they ignore `stage4_full/eval_history.json`.
- `runs/e2/mf_runB_5sphere_hist_560s_gpu/global_best/diagnostics/ti_per_episode.png` already exists for the hist run (made by `diagnose_e2.py:plot_ti_per_episode`, `:206-220`), but it keeps the rejected final proposals and has the debugging title.
- Sequence plots: only interactive KomaMRI HTML (`../MRISystemPhantom.jl/examples/plot_seqs.jl:33-63`) at TI/TE/TR 400/80/1500 ms and a tiny matrix; no static diagram exists.

**Build:**
1. (A) New matplotlib script.
   Use a two-segment x axis (0-0.75 h stretched for the four cheap rungs, 0.75-9 h for full), shade each rung in the ladder colours, plot the 4-episode probes as small markers and the 20-episode evals as a line, mark the selected checkpoint, and put the held-out 2.93% as a star.
   Optional: dashed reference lines for the fixed grid (6.04%) and Cramer-Rao (7.74%) held-out MAPE, labelled as held-out.
2. (B) Standalone matplotlib reader of `episodes.json`: log TI axis (0.01-3 s), drop the rejected proposals, shade blocks 1-2 ("identical in every scan": TI 1.06 s then 0.49 s), colour lines by the scan's mean true T1 (viridis, matching Fig 4).
   Recompute the TI-versus-estimate correlation from the same file if the text quotes it (the report gives r = -0.12 for the histogram arm and +0.78 for the LSTM, `AdaptiveRLMRI.tex:1290-1291`).
   **Done (`scripts/fig3_ti_correlation.py`): r = -0.12 reproduces but is misleading.**
   At block 2 no fit exists, so every scan's "estimate" is the fitter's neutral prior (0.173 s, `julia/rl/e2.jl:723`); those 24 identical points set the sign.
   Counting only decisions made on a real fit: histogram r = +0.23 (n = 268), LSTM r = +0.62.
   In the abstract, say "weakly coupled to the running estimate (r = +0.23, LSTM +0.62)", not "nearly uncorrelated (r = -0.12)".
3. (C) Julia script that builds one shot with `ir_se_2d_sequence(TI, 0.020, TR; ...)` at the E2 settings (use a real pair from `episodes.json`, e.g. TI 0.49 s and TR 1.87 s from block 2) and dumps RF, Gx, Gy and ADC waveforms to JSON.
   Draw it in matplotlib with a broken time axis (inversion | TI gap | 90-180-readout zoomed to about 25 ms | recovery to TR), TI and TR in the accent colour, and "x32 shots per block" in a label.

**Risks:** the full-stage evals are noisy (3.0-9.6%), so the curve will look jagged; showing raw points rather than smoothing is the honest choice.

**Draft caption (check length):**
TI-histogram policy, 560 s budget. (A) Full-Bloch T1 MAPE on validation phantoms during training (one GPU, 9 h); shading marks the fidelity rung, the star the selected checkpoint's held-out MAPE (2.93%). (B) TI chosen at each block of 24 validation scans, coloured by mean sphere T1: a fixed two-block opening probe (shaded), then per-scan choices. Inset: one IR-SE shot; the policy sets TI and TR, and each of 32 shots acquires one k-space line.

**Effort:** (A) and (B) 2-3 hours; (C) half a day.

### Fig 3 follow-up (future work, after Fig 3 is done): show the policy improving over training

Idea (Arthur, 22 Oct): compare the policy early in training with the policy at the end.
Show the distribution of TIs chosen in the first scans against those at the end, each beside the resulting fitted recovery curves and the ground-truth curves.
This shows what "getting better" means physically, which a MAPE curve cannot.

What exists and what is missing:
- No TI choices are logged during training: tensorboard holds only rewards and losses, and the stage evals in `stage*/eval_history.json` only store aggregates.
- Checkpoints exist: `stage{0_analytic,1_cached3,2_cached,3_full3,4_full}/policy.zip` (end of each rung), `global_best/best_policy.zip`, and the final `policy.zip`.
  Check which `vecnorm.pkl` (observation normalisation) goes with each checkpoint.
- So each checkpoint must be rolled out again on the same validation seeds at full fidelity, saving per block the TIs, TRs and signals, plus the per-sphere T1 estimates (`diagnose_e2.py:347-350` currently drops them).

Two ways to roll out:
- **VM route:** `python/diagnose_e2.py` per checkpoint on the Linux/GPU VM (juliacall does not run on macOS), patched to keep per-sphere estimates and signals.
- **Local route, investigate first:** the policies are small MLPs.
  Export each checkpoint's weights and VecNormalize statistics to `.npz` with stable-baselines3 (no Julia needed), reimplement the deterministic forward pass and the action mapping of `python/qalibremd_gym/env_e2.py` in Julia, and roll out with `e2_reset!`/`e2_step!` in pure Julia.
  Validate by reproducing the recorded `global_best/diagnostics/episodes.json` exactly (the opening probe must give TI 1.0621 s then 0.4948 s).
  If it works, it also allows strictly held-out (seed 600000) scans for Fig 4 without the VM.
  Report feasibility before building the figure.

## Figure 4 - T1 maps against ground truth

**Shows:** what one scan's T1 map looks like under each protocol, beside the ground truth.
Andreas asked for one representative scan, not an average.
He wants the ground-truth T1 map on top and each method's map below, with difference-to-truth maps on the right, because a 3% versus 6% difference is invisible in colour alone.

**This figure needs new computation; no T1 map exists anywhere.**
- The pipeline fits T1 per sphere from 3x3-pixel ROI means (`julia/rl/e2.jl:623-636`, `roi_radius = 1`), not per pixel.
- Per-sphere estimates are not saved either: `diagnose_e2.py:347-350` drops them, and the held-out file `eval_t1err_heldout.json` only stores `t1_true_s` and `ape`.
- Baselines store only aggregates (`baseline_e2.py:279-343`).
So every method must be replayed through the simulator.

**Replay (pure Julia, runs locally):**
1. Pick one scan index `k`: a hist scan whose final MAPE in `episodes.json` is near that arm's median.
   Every arm's `episodes.json` uses seed 500000 + k, so all methods see the same phantom (same T1s and pose); only the schedules and noise differ.
2. `using MRISystemPhantom, KomaMRI`, `include("julia/rl_boot.jl")`, then build `E2Env` with the run's `base_env_kwargs` from `run_config.json` and `use_gpu = false`, mirroring `python/qalibremd_gym/env_e2.py:186-231`.
3. For each method: `e2_reset!(env; rng_seed = 500000 + k)`, check that the phantom's T1s match `T1_true` in `episodes.json`, then call `e2_step!(env, [TI, TE, TR, alpha_deg])` for each executed block.
   After each step, `copy(reshape(env.last_image_mag, env.Npe, env.Nfe))`, because only the last image is cached (`e2.jl:629`).
   Alternatively call `_e2_simulate_step(env, TI, TE, TR, alpha)` (`e2.jl:577-612`) directly.
   - RL arms: the recorded actions from each arm's `global_best/diagnostics/episodes.json`, minus the rejected final proposal.
   - Fixed grid: `make_log_grid_trmatched` (`python/baseline_e2.py:86-101`): TR 1.7 s, log-spaced TI over 0.05-3.0 s, only the 9 blocks that fit.
   - Cramer-Rao: `runs/e2/baselines_heldout_560s/cr_optimal_schedule.json`.
4. Record the env's own per-sphere ROI estimates (the numbers that were scored) and check that the replayed running MAPE roughly matches `mape` in `episodes.json`.
   Expect small CPU/GPU float32 differences; the noise stream (`MersenneTwister(seed)` in `e2.jl:789-792`, `add_noise!` with `rng = env.rng`, `e2.jl:603`) is only guaranteed identical on the same Julia and KomaMRI versions.
5. Per-pixel map (new code): for each pixel, call `fit_t1_generalized_ir` (`../MRISystemPhantom.jl/src/fitting/fits.jl:434-499`) on that pixel's block series with the same TIs, TRs, flip angles, `Npe`, `T1_range = (0.01, 3.0)` and noise sigma as `e2.jl:707-718`.
   Ground truth: `phantom_parameter_map(env.phantom, env.phantom.T1, 32, 64, 0.2)`; mask with `background_mask` / `phantom_occupancy`.
6. Save maps, truth, mask, sphere pixel centres and per-sphere estimates to `.npz`, and plot in matplotlib: viridis T1 (0-2 s) for maps, diverging relative error (e.g. +/-25%) for differences, the five randomised spheres outlined, each method's per-sphere ROI error printed beside its circle.

**Decisions and risks:**
- **Resolution.** Images are Nfe 64 x Npe 32 over 200 mm, so pixels are 3.125 x 6.25 mm and a 15 mm sphere covers about 2.4 x 4.8 pixels.
  Recommendation: keep the training resolution (it is what the agent saw) and zoom to the plate, with sphere outlines.
- **Water.** Water T1 = 2.8 s sits near the fit's 3.0 s upper bound, so water pixels will be biased; mask the background or grey it out.
- **Pixel-level artefacts.** The fitter models the centre k-space line, so expect phase-encode ringing in per-pixel maps.
  Fallback: paint each sphere disc with the method's ROI estimate, labelled as such, which is exactly what was scored.
- **Which methods.** Default: histogram, no memory, fixed grid, Cramer-Rao (4 rows plus the truth row); sigma and LSTM would make it too tall.
- **Seeds: a validation scan (seed 500000) is fine** (Arthur, 22 Oct); say "validation scan" in the caption, not "held-out".
- **Water relaxation values from Magali.** `BACKGROUND_WATER` in `../MRISystemPhantom.jl/src/materials/background.jl:5-8` (1.5 T: T1 2.8 s, T2 2.2 s; no YAML or config knob).
  Do not change it for these figures: every trained policy and every number in Fig 5 used these values, so changing them would make Fig 4 inconsistent with the table.
  Treat Magali's values as future work.

**Draft caption (check length):**
T1 maps from one simulated scan of the same random phantom under each 560 s protocol. Top: ground truth. Rows: TI-histogram policy, memoryless policy, fixed log-spaced TI grid, Cramer-Rao-optimal schedule; left, pixel-wise T1 fit to all acquired blocks; right, difference from ground truth. Circles: the five randomised spheres, with each method's ROI error (%). 64x32 matrix, 3.1x6.3 mm pixels, NEMA SNR ~17.

**Effort:** a full day; most of it is the replay script and the per-pixel fit.

## Figure 5 - Main results table

**Shows:** MAPE and success rate on the 24 held-out scans for every arm and comparator.
Andreas prefers a table to the bar chart (more information); drop P90.
ISMRM figures are images, so render the table as an image (LaTeX standalone with booktabs, then `pdftoppm -r 300`), generated from the JSON files by a script; paste it as a native Word table in Andreas's document so it can be edited.

**Proposed table (held-out seeds 600000-600023, values from the files below):**

| Method | Memory | Budget | Mean scan time | MAPE % [95% CI] | Scans < 5% |
|---|---|---|---|---|---|
| RL | TI histogram | 560 s | 541 s | **2.93** [2.58, 3.30] | **95.8%** |
| RL | none | 560 s | 548 s | 4.16 [3.41, 4.99] | 66.7% |
| RL | fitter uncertainty (sigma) | 560 s | 544 s | 5.25 [4.32, 6.22] | 58.3% |
| RL | LSTM (dagger) | 560 s | 525 s | 7.31 [6.10, 8.65] | 25.0% |
| Fixed log grid, TR 1.7 s | - | 560 s | 504 s | 6.04 [5.36, 6.70] | 25.0% |
| Cramer-Rao optimal | - | 560 s | 558 s | 7.74 [7.08, 8.42] | 8.3% |
| RL (optional rows) | none | 240 s | 227 s | 4.62 [4.06, 5.19] | 62.5% |
| Fixed log grid, TR 1.7 s | - | 240 s | 163 s | 6.86 [5.83, 7.90] | 20.8% |
| Cramer-Rao optimal | - | 240 s | 240 s | 15.96 [12.26, 19.81] | 0.0% |

**Data:**
- MAPE, CI and success: `runs/e2/mf_runB_5sphere_{hist_,,sigma_,lstm_}560s_gpu/global_best/eval_summary.json` (the no-memory directory is `mf_runB_5sphere_560s_gpu`); baselines `runs/e2/baselines_heldout_560s/baseline_summary.json`.
- 240 s rows: `runs/e2/mf_runB_cached3_cached_full3_full_gpu/global_best/eval_summary.json` and `.../baselines_heldout/baseline_summary.json`.
- Success and mean time for RL arms are not in `eval_summary.json`; they are printed in each run's `eval_globalbest_24ep_heldout.log` (`eval_e2.py:361-363`).
  Success can also be recomputed exactly from `eval_t1err_heldout.json` (reshape `ape` to 24 x 5, mean per row, share < 0.05).
- Blocks per scan on held-out seeds are not saved for RL arms, so the table uses scan time instead of block count.
- Metric definitions: `julia/rl/e2.jl:729-739` (per-scan MAPE, clipped to [0, 1]), `eval_e2.py:165-168`, bootstrap CI in `baseline_e2.py:41-58` (5000 resamples, percentile 95%).
- LSTM footnote: the LSTM ran 37,862 full-Bloch steps against 274,537 for the histogram arm (7.3x fewer), from the `full` stage start and end in each `fidelity_history.json`.

**Decided:** keep the LSTM row with the dagger; drop P90.
**Still open:** the 240 s rows.
They carry the "more scan time did not help the fixed protocols" result, which the current title in `submission.md` depends on now that its time-versus-error figure has been dropped.
Build the table with and without them.

**Draft caption (check length):**
T1 error on 24 held-out random phantoms, identical for every method and never used for training or checkpoint selection. MAPE: mean absolute percentage error over five spheres, averaged over scans, with bootstrap 95% CI. <5%: share of scans with MAPE under 5%. Fixed grid: log-spaced TIs, TR 1.7 s. Cramer-Rao: (TI, TR) schedule optimised for the plate's T1 range. All use the same T1 fitter. (dagger) LSTM had ~7x fewer full-Bloch training steps.

**Effort:** 1-2 hours.

## Preview figure (no caption)

Required for the programme and must be legible at phone width.
The table cannot serve; a simplified bar chart from the Fig 5 data works (histogram policy, no memory, fixed grid, Cramer-Rao; large text; 3:2).
Build it in the Fig 5 script.

## Suggested split into subagents

1. **Shared style + Fig 5 + preview** (pure Python, quickest; settles method colours and names for the rest).
2. **Fig 3** (Python for A and B; a small Julia waveform dump for C).
3. **Fig 1** (Julia for A and B, Python for C; Kaleido setup).
4. **Fig 2** (TikZ editing and compilation).
5. **Fig 4** (Julia replay and per-pixel fit, then Python plotting; the longest and riskiest, so start it early).

Final step: assemble the five figures and captions into one Word document for Andreas, with a character count beside each caption.

## Decisions (22 Oct, Arthur)

- Fidelity ladder: linear cost axis, no second axis; CPU 14-sphere numbers.
- One colour scheme across all figures, in the family of the phantom viewer (`scripts/figstyle.py`).
- 3D phantom: 1.5 T or 3 T both fine.
- P90 dropped; LSTM row kept.
- Fig 2 edits as listed.
- Figs 3 and 4 may use validation seeds (500000).
- Fig 3: build the static sequence diagram; the training-evolution idea is a follow-up for the Fig 3 agent.
- Cramer-Rao schedule over the 14-plate pool is fine (see correction 5).
- Simulator water T1/T2 unchanged for now.

## Control experiment: adaptation versus schedule shape (22 Oct, `figures/control_summary.md`)

Pure-Julia rollout of the TI-histogram `global_best` on the 24 held-out phantoms reproduces the official evaluation bit for bit (2.93%).
On the same phantoms:
- Adaptive policy: 2.93% [2.58, 3.30], worst scan 5.1%.
- Each phantom given another scan's schedule: 3.61% [3.01, 4.45] (paired +0.67 pp [-0.01, +1.55]).
- One fixed protocol from the policy's per-block median TI and TR, built from validation scans only: 3.34% [2.30, 4.86], worst scan 17.9%; built from the held-out scans themselves: 2.99%.
- Fixed log grid 6.04%, Cramer-Rao 7.74%.

Reading: most of the gain over the fixed comparators comes from the schedule shape RL discovered (short TIs paired with short TRs, about 13 blocks in 560 s), and a fixed protocol distilled from the policy keeps most of it.
Per-scan adaptation shows as robustness: the policy's shortest TI tracks the phantom's shortest T1 (r = +0.57), which rescues the rare very-short-T1 spheres (41 and 46 ms) where the distilled protocol fails (up to 74% error on one sphere).
n = 24, and the two-phantom explanation is post hoc, so state it as an observation, not a quantified effect.
Consequence for the abstract: frame RL as discovering a better protocol (with adaptation as a worst-case safeguard), not as "adaptation buys precision".

## Library bug found (not affecting any result)

`T2_ARRAY` in `../MRISystemPhantom.jl/src/materials/t2_array.jl` holds T1-like values for the T2-plate spheres.
At 1.5 T it gives T2 = 2.640 s for T2-1 and 0.0797 s for T2-14; the manual's Table 1 for serial >= 0042 (page 23 of `QalibreMD_NIST_ISMRM_System_Standard_Model_130_T1_T2_PD_User_Manual.pdf`) gives 1044 ms and 7.79 ms.
`T1_OF_T2_ARRAY` and the T1-plate values (`T1_ARRAY`, `T2_OF_T1_ARRAY`) match the manual, so training, Figs 3-5 and the T1 results are unaffected.
Anyone simulating T2 with the twin (e.g. the eddy-current student's work) gets T2-plate T2s up to 10x too long.
Fig 1 B2 shows these wrong values, so use B1.
Fix in the package (both field strengths, with a test against the manual table) before the abstract advertises the library.

## Still open

- Fig 5: include the 240 s rows?
- Fig 4: confirm the four methods (histogram, no memory, fixed grid, Cramer-Rao).
- Optional caption or text line: the per-scan Cramer-Rao oracle (7.05% on validation seeds) to pre-empt "the comparator was not tuned to these T1s".
- `submission.md` needs updating to match: five figures, corrected blocks and times, "validation" versus "held-out" wording, action (TI, TR).

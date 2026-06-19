Presentation format: 18 minutes + 10 minutes Q&A afterwards.
We are strongly encouraged to include a live demo. The MRISystemPhantom can definitely have a live demo - including showing the plots.
Not too much live demo possible in the RL stuff i don't think - could come up with a visualiser of the RL policy decisions live but might be too much effort.

See report_latex for the pdf and the latex files that generated it - including the figures will be good to use them.

Immediate deadlines - need to send slides for feedback at 6pm 16 June.

Structure:

- 2 minutes max context: Includes background research + related works - Chapter 2 condensed
- Method
- Implementation
- Results
- Evaluation

Or could follow the report structure


To include:

Extra slides that don't make the presentation but are instead there in case people ask about them in the Q&A - e.g. result slides so i don't have to memorise them.

Things not included in report_latex that could be added to the script:

- The parallel envs & threads used to speed up RL training (e.g. the SubProcVec or smth like that to split rollout collection) - not used for GPU but used for CPU runs (maybe not important enough to make presentation)
- 


TODO items:
1. The failed second RL diagram for the multi-fidelity chapter - could make that into a flow diagram
2. Could implement the benchmark of the other paper to better compare my results. Could even run on T2 plate to see results of that. Need a time estimate for these - and will be done at the end post a working presentation.
3. Could keep running the LSTM to see if step bound they're the same
4. Pixel-grid overlay panels (slides 6/7/7b) use mismatched TIs: buggy run is TI=0.065 (npe64/nfe128) and the fixed/clean run is TI=0.1 (npe32/nfe64). For a clean before/after, regenerate both at the same TI and matrix size via `scripts/pixel_grid_overlay.jl`. Modest effort (re-run the script twice with matched config) but needs the Julia/KomaMRI env on Linux/WSL/GPU VM — juliacall does not run on local macOS. Not worth it for the talk; do it if the figure goes in the report.
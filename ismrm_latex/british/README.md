# BIC-ISMRM 2026 abstract

> **Unchecked LLM-generated work.**
> The text, numbers and figures in this folder were drafted by an earlier LLM session and have not been reviewed by Arthur.
> Do not reuse anything from here without checking it against the run outputs in `runs/` and the final report.
> Known visual defects in `figures/method_overview.png`: panel labels overlap titles, panel C is cropped, and panel D has no arrows.

Generate the figures from the saved report and evaluation data:

```bash
.venv/bin/python ismrm_latex/british/make_figures.py
```

Compile the two-page submission:

```bash
cd ismrm_latex/british
latexmk -lualatex main.tex
```

The programme synopsis is stored separately in `synopsis.txt`.

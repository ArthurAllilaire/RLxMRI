#!/usr/bin/env bash
# Build the env-loop diagram PNG for the deck from the standalone TikZ wrapper
# of the report's fig:e2-env-loop. Outputs figs/fig_envloop.png.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
mkdir -p figs
lualatex -interaction=nonstopmode -halt-on-error envloop_diagram.tex >/dev/null
# 300 dpi raster, white background, for slide embedding.
gs -dSAFER -dBATCH -dNOPAUSE -sDEVICE=png16m -r300 \
   -dTextAlphaBits=4 -dGraphicsAlphaBits=4 \
   -sOutputFile=figs/fig_envloop.png envloop_diagram.pdf
echo "wrote figs/fig_envloop.png"

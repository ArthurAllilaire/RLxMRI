#!/usr/bin/env bash
# Build the minimal-edit Figure 2 in both variants: the original lookahead logic
# (fig2_loops_minimal) and the code-accurate one (fig2_loops_minimal_logicfix).
# Usage: scripts/fig2_loops_minimal.sh [BUILD_DIR]   (default: a temp dir)
set -euo pipefail
FIGURES="$(cd "$(dirname "$0")/../figures" && pwd)"
BUILD="${1:-$(mktemp -d)}"
mkdir -p "$BUILD"
cd "$FIGURES"
for variant in fig2_loops_minimal fig2_loops_minimal_logicfix; do
  defs=""
  [ "$variant" = fig2_loops_minimal_logicfix ] && defs='\def\logicfix{}'
  lualatex -interaction=nonstopmode -halt-on-error -output-directory="$BUILD" \
    -jobname="$variant" "${defs}\input{fig2_loops_minimal.tex}" >/dev/null
  cp "$BUILD/$variant.pdf" "$FIGURES/$variant.pdf"
  pdftoppm -r 300 -png -singlefile "$FIGURES/$variant.pdf" "$FIGURES/$variant"
  echo "wrote $FIGURES/$variant.pdf and $FIGURES/$variant.png"
done

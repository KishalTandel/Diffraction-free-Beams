#!/usr/bin/env bash
# Regenerates every figure in the paper (figures/*.pdf) and prints the numerical checks.
set -e
cd "$(dirname "$0")/src"
python fig_schematics.py   # Fig. 1 (generation schematic) and Fig. 8 (application sketches)
python fig_numeric.py      # Figs. 2,4,5,6,7 + figures/results.json
python fig_tradeoff.py     # Fig. 3 (f vs G) and Fig. 9 (light-sheet profiles)
python checks.py | tee ../figures/checks_output.txt

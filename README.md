# Diffraction-free beams: simulation and figure code

Code accompanying the PH4107 term paper *Diffraction-free beams in optics: production and uses*
(IISER Kolkata). It regenerates every numerical figure in the paper and checks the analytic
formulas against simulation.

## Run

```bash
pip install -r requirements.txt
./run_all.sh          # about 1 minute; writes figures/*.pdf, figures/results.json, figures/checks_output.txt
```

Tested with Python 3.13, NumPy 2.5, SciPy 1.18, Matplotlib 3.11. Needs about 1 GB of RAM
(2048 x 2048 complex FFTs). A LaTeX installation is not needed.

## Files

| File | Purpose | Output |
|---|---|---|
| `src/sim.py` | Shared tools: exact 2D angular-spectrum propagation (FFT) and 1D paraxial split-step propagation | |
| `src/style.py` | Matplotlib style and colours | |
| `src/fig_schematics.py` | Schematic drawings | Fig. 1 `fig_schematic_generation.pdf`, Fig. 9 `fig_schematic_apps.pdf` |
| `src/fig_numeric.py` | Bessel, Bessel-Gauss, axicon and Airy simulations, self-healing | Fig. 2 `fig_bessel_profile.pdf`, Fig. 3 `fig_bg_check.pdf`, Fig. 5 `fig_propagation.pdf`, Fig. 6 `fig_airy.pdf`, Fig. 7 `fig_selfheal_bessel.pdf`, Fig. 8 `fig_airy_heal.pdf`, `results.json` (extra, not in the paper: `fig_tradeoff.pdf`) |
| `src/fig_tradeoff.py` | Range-efficiency trade-off and light-sheet profile | Fig. 4 `fig_tradeoff_a.pdf`, Fig. 10 `fig_tradeoff_b.pdf` |
| `src/checks.py` | Prints every number quoted in the paper (Appendix B table, Secs. III and VII). Run after `fig_numeric.py` | `checks_output.txt` |

## Parameters used throughout

- Running example: lambda = 632.8 nm, input Gaussian w0 = 1 mm, sin(theta) = 8e-3 (kt = 7.94e4 1/m), grid 2048 x 2048 with 4 um spacing, 100 steps of 2 mm.
- Airy beams: 1D paraxial equation in the dimensionless variables s = x/x0, xi = z/(k x0^2); split-step on 2^15 points, ds = 0.02, dxi = 0.05; truncation a = 0.02 (a = 0.1 for the closed-form check).
- Light sheet: NA 0.40 in water (n = 1.33), lambda0 = 488 nm, z_max = 50 um.

## Notes

- `fig_numeric.py` prints and saves in `results.json` the self-healing numbers quoted in the paper (heal distance 18.8 mm, 24% at z_heal, Airy recovery 27/41/60% at xi = 4, 6, 9).
- The on-axis overshoot of about 30% near z = 65 mm after the obstacle (Fig. 7b) is a real feature of the simulation that the paper does not analyse.

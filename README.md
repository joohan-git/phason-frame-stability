# Pressure stability of quasicrystalline frames

Code and data for the article by J. Nam on the pressure stability of quasicrystalline frames.

Repository: <https://github.com/joohan-git/phason-frame-stability>

Archive: <https://doi.org/10.5281/zenodo.24090906>

## Requirements

Python 3.12 with the packages in `requirements.txt`:

```text
python -m pip install -r requirements.txt
```

`pypardiso` is optional and speeds up the icosahedral solver. No external data are needed.

## Quick test

```text
python smoke_test.py
```

This recomputes, on one CPU in about two minutes, the fourfold splitting of the 5/7 Ammann–Beenker cells on the B1 and E2a paths, one Penrose cell, the forbidden-patch fraction and threshold of one held-out cell, the long-wave threshold of the ordered 12/17 cell, the square-lattice post-buckling path, the penalized sampler at zero and nonzero penalty, and the analysis of the design test, and compares them with the stored results. It exits with a nonzero code if a check fails.

## Running calculations

Scripts write their results next to themselves and replace the stored file of the same name. Examples:

```text
python symmetry/representation.py
python symmetry/point_groups.py
python phason_paths/assemble_paths.py
python phason_paths/assemble_families.py
python isotropic_loss/assemble_isotropic.py
python heldout_test/heldout_stats.py
python slope_survey/slope_table.py
python nonlinear/exact_references.py
python nonlinear/run_perfect.py square 2
python twelvefold/analyze.py
python icosahedral/icosa_statistics.py
python icosahedral/assemble_icosa.py
python icosahedral/compare_operators.py
python design_intervention/analyse_intervention.py
python heldout_test/analyse_heldout.py ab
python disorder/ensemble.py --p 5 --q 7 --mode random --seed 1 --outdir ensemble_test
```

Production runs (the ensembles of up to 189,284 vertices, the design and defect plans and the icosahedral series) use the same scripts with a plan file and a task index, for example `python design/design_run.py design/plan_part2.json 0 design/out_part2`, and take hours to days of CPU time.

## Contents

| Folder | Content |
|---|---|
| `frames/` | Ammann–Beenker and Penrose tilings by cut and project, beam frames, critical pressure |
| `slope_survey/` | Periodic tilings of slope space (Fig. 1) |
| `symmetry/` | Phason and elastic representations, selection tables (Fig. 2, Supplementary Section 1) |
| `phason_paths/` | Approximant paths, fourfold splitting, window offsets, flips (Figs. 2 and 3) |
| `penrose/` | Penrose paths and offsets (Fig. 3) |
| `isotropic_loss/` | Isotropic loss at fixed size and patch census (Fig. 3b, c) |
| `heldout_test/` | Held-out test of the isotropic loss with an independent frame and the main codes (Fig. 3d) |
| `independent_code/` | Second implementation: paths, Penrose check, frozen prediction law, Penrose ensembles (Figs. 3 and 5a) |
| `forbidden_patches/` | Registered forbidden-patch test (Supplementary Section 3) |
| `disorder/` | Random-tiling ensembles, size dependence and the critical mode of a random tiling (Figs. 4 and 5b) |
| `objectivity/` | Corotational shear term of the geometric stiffness (Methods) |
| `robustness/` | Mesh and slenderness (Fig. 6b, Supplementary Section 4) |
| `nonlinear/` | Equilibrium paths and imperfections (Fig. 6a, Supplementary Section 5) |
| `nonlinear_check/` | Second nonlinear solver (Supplementary Section 5) |
| `design/` | Reference lattices, strength and stiffness map, load states, scatter (design figure, Supplementary Section 7) |
| `defects/` | Missing and bowed struts (Supplementary Section 8) |
| `twelvefold/` | Twelvefold frame (Supplementary Section 6) |
| `icosahedral/` | Icosahedral frames (Supplementary Section 3) |
| `design_intervention/` | Patch content changed by design: penalized sampler, periodic pairs, analysis and checks (Supplementary Section 9) |

JSON files hold the computed results and NPZ files hold stored tilings. `SHA256SUMS.json` lists file checksums.

## Citation

See `CITATION.cff`.

## License

MIT (`LICENSE`).

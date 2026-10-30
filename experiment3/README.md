# Experiment 3: linear time-varying electronic filter

This directory contains Section 4.4 and Appendix D.4 of the final *Transfer
Learning for Operators* manuscript: eight target regimes, three train/validation/
test rotations per regime, and seven estimators. The 24 held-out evaluations use
one target training record and twelve candidate source tasks (30 source records).

| Paper item | Retained numerical evidence |
| --- | --- |
| Table 1: overall seven-method errors | `outputs/results/literature/method_table.csv` |
| Table 2: selected representation dimensions | `outputs/results/target_classes.json` |
| Tables 3–4: errors by target regime | `outputs/results/literature/per_target_summary.csv` |
| Source and weight selection, Appendix D.4.3–4 | `outputs/results/source_selection.csv` |
| Spectral and penalty diagnostics, Appendix D.4.4 | `diagnostics.py` (requires raw data) |

The four original methods have 96 score rows in `outputs/results/all_test_scores.csv`;
the three literature methods have 72 rows in `outputs/results/literature/all_test_scores.csv`.
The literature table combines these into the seven methods in the paper:
Target-only, Pooling, Unprojected, Projected, Jiao–McKeague (`jm_ts`),
Lin–Reimherr (`lr_atl`), and Hu–Lin (`hl_agg`).

The representation and validation searches are part of the reported estimators
and are retained. Unreported standalone transfer variants, plotting programs,
notebooks, and generated images are omitted. Mathematical details and data-split
rules are in [EXPERIMENT_SETUP.md](EXPERIMENT_SETUP.md).

## Reproduce the numerical experiment

From the repository root, with the dependencies installed:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python experiment3/run.py all --archive /path/to/data_LPV.zip --jobs 4
python experiment3/literature.py --archive /path/to/data_LPV.zip --jobs 4
```

`run.py` executes preparation, leave-one-target-out representation selection,
source calibration/selection, held-out evaluation, and comparison with the frozen
four-method results. Its phases can also run individually: `prepare`, `meta`,
`select`, `evaluate`, and `compare`.
`literature.py` runs the three final-paper comparison methods and checks its refitted
Target-only and Pooling errors against the reference scores (relative tolerance
`1e-9`). `--collect-only` rebuilds its numerical tables from completed run records.
New results go under `.runs/experiment3`, including `results/literature`, so the
retained reference data stay intact. Use `--workdir`, `--raw-dir`, or `--archive`
to select other locations; `LTV_RAW_DIR` and `LTV_ARCHIVE` are also supported.

The source package did not retain raw outputs for the supplementary spectral,
expanded-LR-grid, and zero-lasso-HL checks described in Appendix D.4.4.
`diagnostics.py` provides explicit reruns of those checks using the same estimators.
These diagnostics have not been run on the benchmark while preparing this package;
no supplementary numeric result is claimed as newly reproduced.

```bash
python experiment3/diagnostics.py --kind all --archive /path/to/data_LPV.zip --jobs 4
```

The raw [Lataire et al. benchmark archive](https://www.kth.se/social/files/558d84b7f276541077ddc833/data_LPV.zip)
is not redistributed. `prepare` verifies its SHA-256 before extraction:

```text
cbc58ad9541f073326f61ce463ca0be0b41d62f450f13ed9000eb2b492d2dc2d
```

The expected archive size is 227,072,311 bytes and it contains 49 MATLAB files.
The reference is Lataire, Louarroudi, Pintelon and Rolain (2015),
“Benchmark data on a linear time- and parameter-varying system”,
*IFAC-PapersOnLine* 48(28), 1477–1482.

## Check the retained package

```bash
python -m pytest -q tests/test_experiment3_ltv.py
```

These tests check the numerical solvers, frozen design, representation selection,
all seven aggregate table values, and per-target tables. A raw-data test runs only
when `LTV_RAW_DIR` is set and the preparation phase has completed.

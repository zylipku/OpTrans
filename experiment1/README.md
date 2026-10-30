# Experiment 1: functional regression

This directory contains the numerical experiment in final-paper Section 4.2,
Figure 1 and Appendix D.2. It retains all 13 target sample sizes, including the
small-sample comparisons discussed in the appendix, and all seven reported
methods: Target-only, Pooling, Unprojected, Projected, JM, LR and HL.

The model and frozen simulation protocol are described in
[EXPERIMENT_SETUP.md](EXPERIMENT_SETUP.md). No external data are needed.
Commands below run from the repository root with the environment described in
[the package README](../README.md).

```bash
# Check retained numbers and frozen selection IDs.
python tests/validate_release.py

# Refit three complete trajectories and rebuild retained statistical tables.
python tests/validate_release.py --recompute

# Regenerate the seven-method numerical summary into a separate run directory.
python experiment1/literature.py summarize

# Refit the registered base estimators, matched-parameter ablation and
# fixed-support invariance checks in a separate run directory.
python -m transfer_operator.runner confirmation \
  --config experiment1/configs --output .runs/functional --workers 1 --limit 3
python tests/validate_refits.py \
  --output-directory .runs/functional --expected-trajectories 3
```

Remove `--limit 3` for the full 1,000-trajectory base experiment. To recompute the
literature baselines with their supplied frozen parameters, run
`python experiment1/literature.py confirmation --workers 8`, followed by
`summarize`. New literature outputs are written under
`.runs/experiment1-literature/outputs/`; pass `--output PATH` to use another
destination. Published files in `experiment1/outputs/` are protected.
`summarize` reads the regenerated risk archive at the selected destination when
present, or the published archive otherwise.
The full study is substantially more expensive than the validation checks.

The independent development protocol is part of the paper's estimator definition
and remains available when a complete rerun is required:

```bash
python -m transfer_operator.runner development \
  --config experiment1/configs --output .runs/functional-development --workers 8
python experiment1/literature.py development --workers 8
python experiment1/literature.py select

# Explicitly evaluate the selections produced above.
python experiment1/literature.py confirmation --workers 8 \
  --parameters .runs/experiment1-literature/outputs/results/literature/frozen_parameters.csv
python experiment1/literature.py summarize
```

The base runner checks its new selections against the supplied freeze. The
literature `select` command writes a new frozen-parameter CSV under `--output`;
it uses only the 500 independent development datasets per target size, never
confirmation risks. Confirmation always uses the published frozen parameters
unless `--parameters` explicitly selects another CSV; a pre-existing run's
selections are never adopted automatically. This rule also applies to `all`.

Retained numerical files:

- `outputs/results/literature/summary_by_method_n0.csv` and
  `paired_comparisons.csv`: the seven methods and paired risk ratios underlying
  final Figure 1. The figure displays `n0 >= 50`; data retain all 13 sizes.
- `outputs/results/projection_ablation.csv`: matched-Unprojected comparison in
  Appendix D.2.3; `projection_vs_unprojected.csv`: independently tuned comparison.
- `outputs/results/iteration_vs_first.csv` and compact confirmation diagnostics:
  the fixed-support invariance check in Appendix D.2.2, through refit 10.
- `outputs/data/`: individual risks, model/data fingerprints, numerical
  diagnostics and the development aggregates that verify frozen selections.

Stored scientific values are inherited from the source release. The standalone
`lr_tl`/`hl_tl` reference comparisons, `target_c1` reference result, exploratory
selection-stability/boundary reports, plotting programs, notebooks and images
are excluded. Primitive source-fit routines remain where used inside the
reported adaptive methods. This package does not claim a new full Monte Carlo run.

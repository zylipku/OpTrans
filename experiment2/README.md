# Advection-diffusion experiments

This directory contains the numerical experiments in Section 4.3 and Appendix D.3 of
*Transfer Learning for Operators*:

| Paper item | Configuration | Retained results |
| --- | --- | --- |
| Figure 2: noise levels | `configs/advection_sigma.yaml` | `outputs/results/literature/advection_sigma/` |
| Figure 3: target sample sizes and two recalibration rounds | `configs/advection_n0.yaml` | `outputs/results/literature/advection_n0/` |
| Appendix D.3.1: constant-mode risk, source contrast and spectral diagnostics | both configurations above | `outputs/results/literature/DIAGNOSTICS.json` |

The seven compared methods are Target-only, Pooling, Unprojected, Projected,
Jiao-McKeague (`jm_ts`), Lin-Reimherr (`lr_atl`) and Hu-Lin (`hl_agg`). Figure 3
also compares Projected with one and two additional recalibration rounds.
`projected_k0` is the pilot, `projected_k1` is Projected, and `projected_k2`/`k3`
are the two additional rounds. The independently tuned Unprojected estimator is
never replaced with Projected's projection-off control.

## Reproduce

Run from the package root after installing its dependencies, with one numerical
thread per worker:

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python experiment2/run.py --workers 4
python experiment2/literature.py evaluation --workers 4 --base-outputs .runs/experiment2-base
python experiment2/literature.py summarize --base-outputs .runs/experiment2-base
python experiment2/literature.py diagnostics --workers 4
```

The first command generates both base-estimator studies with the frozen
multipliers. The second evaluates the three literature methods using their
frozen development selections. It also regenerates Target-only and Pooling to
check that the methods use the same observations. `summarize` computes all
paper means, paired comparisons and percentile intervals. New results go under
`.runs/`; retained inputs and numerical results remain unchanged. To rebuild
summaries from retained risks alone, run `python experiment2/literature.py summarize`.

For a short execution check, run one repetition of one cell:

```sh
python experiment2/run.py --scenario advection_sigma --cell 0 --replications 1 --workers 1 --outputs .runs/pde-smoke
```

Such output is marked `complete: false` and cannot reproduce Monte Carlo intervals.
The complete run uses 300 repetitions per cell. The 625-dimensional empirical
source covariance decompositions make full runs materially more expensive than
the short checks.

Appendix D.1/D.3 specifies independent development tuning for the three
literature methods. To repeat that part of the reported protocol, run these
before `evaluation` (the result directory is shared between stages):

```sh
python experiment2/literature.py development --workers 4
python experiment2/literature.py select
```

Development uses 300 independent repetitions per setting and the grids in
`configs/literature.yaml`. Selection never uses evaluation observations.

## Retained data and verification

`outputs/data/advection_sigma/cell*.npz` contains the eight retained, packed
300-repetition archives. They preserve the reported risk, support, identity and frozen-parameter fields
from the original per-repetition records. Unused tuning-window, geometry and
extra-round diagnostics have been removed. `outputs/results/advection_n0/risks.npz` contains the nine sample-size
cells, restricted to the pilot, reported estimators and reported recalibration
rounds. Literature risks and frozen parameters contain only the three reported
literature methods. All retained methods use the original seeds and numerical
operation order.

```sh
python -m pytest -q tests/test_experiment2_models.py tests/test_experiment2_replications.py tests/test_experiment2_records.py tests/test_experiment2_advection_n0.py tests/test_pde_rules.py tests/test_pde_records.py
```

These checks cover the operator equations, calibration and projection, frozen
scaling, paired bootstrap, retained means and representative simulation replays.
Floating-point archive comparisons use absolute tolerance `1e-12` with zero
relative tolerance; supports, seeds and candidate identities remain exact.
The slim package contains no plotting scripts, images or notebooks.

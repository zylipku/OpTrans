# OpTrans

Standalone numerical experiments for the paper *Instrumental Transfer Learning for Solution Operators*.

| Final manuscript | Directory | Retained experiments |
| --- | --- | --- |
| Section 4.2, Figure 1, Appendix D.2 | [experiment1](experiment1/README.md) | Functional regression, all 13 target sizes, matched-parameter projection ablation and fixed-support invariance |
| Section 4.3, Figures 2–3, Appendix D.3 | [experiment2](experiment2/README.md) | Advection–diffusion noise and target-size studies, two additional recalibration rounds, spectral diagnostics |
| Section 4.4, Tables 1–4, Appendix D.4 | [experiment3](experiment3/README.md) | Electronic LTV filter, 8 targets × 3 held-out rotations, representation/source selection and diagnostic entry points |

The seven compared estimators are Target-only, Pooling, Unprojected, Projected,
Jiao–McKeague (`jm_ts`), Lin–Reimherr (`lr_atl`), and Hu–Lin (`hl_agg`).
See [the baseline adaptation](LITERATURE_BASELINES.md) and each experiment's setup.

## Install and check

Use Python 3.12 and run from this repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pip install --no-deps -e .
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m pytest -q tests
python tests/validate_release.py --recompute
```

`requirements.txt` pins the numerical dependencies. No plotting or notebook
packages are required. Full Monte Carlo runs can take substantially longer than
checks of the retained archives; the experiment READMEs distinguish these commands.
The real-data experiment requires the external benchmark archive described in
[experiment3](experiment3/README.md); it is not redistributed here.

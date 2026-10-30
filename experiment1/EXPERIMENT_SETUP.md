# Functional-regression protocol

This is the final paper's Section 4.2 / Appendix D.2 block model. For
`V = W = R ⊕ R^512`, the solution operator is
`A_a(x, f) = (x + <a,f>, Λf)`, with `Λ_kk = k^-1`, input covariance
`ν_k = k^-2`, and the normalized target slope proportional to `k^-0.75`.
The two governing-operator perturbations use odd/even coordinate directions
through coordinate 16. Exact model construction, perturbation balancing,
operation order and SHA-256 checks are in `configs/config.yaml` and the shared
model implementation.

The sample sizes are
`n0 = [10,20,30,40,50,70,100,200,300,400,500,700,1000]`,
`n1 = n2 = 9.5*n0`, and `N = 20*n0`. Each setting tunes on 500 independent
development datasets. Evaluation uses 1,000 independent trajectories with
nested task-data prefixes of `(1000,9500,9500)`. Prediction risk is the exact
noise-free target population risk; observation-noise variance is not added.

The seven reported methods are Target-only, Pooling, Unprojected, Projected,
Jiao–McKeague (`jm_ts`), adaptive Lin–Reimherr (`lr_atl`) and aggregated Hu–Lin
(`hl_agg`). The literature estimators use `c=1`, identified by the shared
noise-free output block. Their grids, candidate order, target-data splits and
source-selection rules are fixed in `configs/literature.yaml` and
`transfer_operator/literature/`. Unprojected and Projected have independently
selected parameters. The matched-Unprojected ablation uses Projected's exact
pilot, calibration and refit parameters.

Frozen selections are supplied in `outputs/results/frozen_parameters.csv` and
`outputs/results/literature/frozen_parameters.csv`. Candidate registries and
model checksums protect their identities. Development seeds start at
`2026101410`; confirmation uses `2026101490`. Paired percentile confidence
intervals resample entire trajectories jointly across all methods and sizes,
using 5,000 draws with seed `2026101491`.

Appendix D.2.2 explains why the selected calibration span is invariant under
fixed-support refitting in this model. The compact archive therefore retains
refits 1–10 as a numerical invariance check; all their stored risks are identical.
These are diagnostics of the stated identity, not extra plotted comparisons.
All numerical tests use absolute tolerance `1e-12` and zero relative tolerance
for archived floating-point risks and diagnostics, with exact discrete identities
and raw-observation fingerprints. Model, dictionary and grid checksums remain
exact. Hashes of derived empirical covariances, response moments and fitted
coefficients are reported separately: floating reduction roundoff can change
their bytes even when the input observations are identical. Original-engine
bitwise fixtures are checked only when the
recorded numerical-library versions match the execution environment.

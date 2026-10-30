# Numerical validation

Run from the OpTrans repository root after installing `requirements-dev.txt`:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m pytest -q tests
python tests/validate_release.py --recompute
python tests/validate_release.py --manifest-only --reference experiment2/outputs
python tests/validate_release.py --manifest-only --reference experiment3/outputs
```

The suite checks independent least-squares constructions, lasso optimality,
aggregation, frozen selections, summary means, retained result hashes, selected
recomputed simulation observations, and LTV tables. It does not rerun every
Monte Carlo trajectory. Real-data refitting requires the external benchmark
archive; the corresponding test is skipped when it is unavailable.

The functional checks recompute trajectories 0–2 at all 13 sizes and verify
prediction risks and diagnostics against the archived values. Floating risks and
numeric diagnostics use `atol=1e-12, rtol=0`; supports, candidate IDs,
raw-observation fingerprints, shapes and discrete diagnostics remain exact.
Hashes of derived floating-point moments and coefficients are reported separately,
since last-bit BLAS differences change their digests. Unmodified source and packaged
engines were independently replayed on the same three trajectories: all 30 arrays
(585 risk entries) were bitwise equal; both showed the same covariance-hash differences
from the historical archive. The maximum archive risk difference was `1.06e-14`,
within the unchanged `1e-12` tolerance. Rebuilt statistical
CSV files and metadata must match byte for byte. Separate development/confirmation
fixtures also compare with the original engine; those bitwise checks report a
skip when their recorded Python/NumPy/SciPy environment differs.

The PDE tests cover only the final advection–diffusion studies: operators and
adjoints, empirical moments, calibration and projection, packed records, selected
registered replications, fixed-support recalibration, and literature-baseline
summaries. Removed diffusion/population-perturbation studies have no tests here.

The LTV tests check the frozen design and target classes, numerical solvers,
Table 1's seven-method values, and per-target results for Tables 3–4. Diagnostic
entry points for Appendix D.4.4 require raw data; historical supplementary output
was not retained by the source release.

Each `RELEASE_MANIFEST.json` covers its own experiment tree and explicitly listed
shared dependencies. Validators reject changed, missing, extra, or escaping files.
After an intentional modification, update the relevant provenance and reseal:

```bash
python tests/seal_manifests.py
python -m pytest -q tests
```

New simulations use `.runs/`; validation uses temporary directories and preserves
retained observations and numerical results.

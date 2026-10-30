# Final-paper protocol: Section 4.3 and Appendix D.3

The target PDE on the periodic two-dimensional domain has Fourier symbol
`ell_0(k) = 1 + |k|^2`. The two source symbols are
`ell_j(k) = ell_0(k) + i theta_j . k`, with `theta_1=(1,0)` and
`theta_2=(0,1)`. Restriction to `max(|k_1|,|k_2|) <= 12` gives 625 real
coordinates (one constant and 312 conjugate pairs). Forcing and noise use
independent Rademacher signs with envelopes `ell_0^-1` and
`sigma * ell_0^-5/2`. Risk is the analytical prediction error on this retained
space, equation (75).

Both studies use `n1=n2=16384`, full uncentered empirical source covariance
whitening, and no sample splitting for the proposed estimators. Initial Lasso
calibration enumerates active sets and signs exactly. Recalibration uses
unpenalized least squares on the initial selected supports and rebuilds the
projection at every round.

| Study | Target sizes | Noise | Evaluation seed |
| --- | --- | --- | --- |
| Figure 2 | 256, 1024 | 0.05, 0.1, 0.25, 0.5 | 202610160501 |
| Figure 3 | 16, 32, 64, 128, 256, 512, 1024, 2048, 4096 | 0.25 | 202610250601 |

Each cell retains all 300 repetitions, including support-selection errors.
Streams follow `SeedSequence([seed, replication, task]).spawn(2)`, separately
for forcing and noise. Paired percentile intervals use 20,000 resamples with
bootstrap seeds 202610160599 and 202610250699, respectively. Random integer
indices are converted to counts and normalized weights before averaging.

The frozen multipliers are Projected `(c0,cbar,ctheta)=(.003,.003,.02)`,
Unprojected `(.003,.001,.005)`, Target-only `cT=.001` and Pooling `cPool=0`.
With `N=n0+n1+n2`, the penalties are:

- `lambda0 = c0 * n0^(-2/3)`;
- `eta = cbar * N^(-2/3)` and `lambda = (N/n0)*eta`;
- `lambda_j = ctheta * [sqrt(neff(eta)/nj) + 1/(nj*sqrt(eta)) + n0^(-1/3)]`.

The three literature methods are tuned separately at every `(sigma,n0)` on
300 independent development repetitions, then frozen. Development seeds are
202610260501 (noise study) and 202610260601 (sample-size study). Truncation
retains complete conjugate pairs; the largest 313-unit choice uses all 625
real operator coordinates. The exact grids and stable candidate indices are
in `configs/literature.yaml` and each scenario's `frozen_parameters.csv`.

Appendix D.3.1's spectral analysis is reproduced by the `diagnostics` stage:
the target least-squares risk fraction in the constant mode, source contrast
energy in low-frequency units, and the angle between source and target
operators. No hindsight-tuned or oracle estimator comparisons are retained.

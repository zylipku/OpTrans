# Literature baselines

The final manuscript compares three adaptive transfer methods in the same operator
space as Target-only, Pooling, Unprojected, and Projected.

| Key | Method | Reference |
| --- | --- | --- |
| `jm_ts` | Coefficient-shape transfer | Jiao and McKeague (2025), *Coefficient shape transfer learning for functional linear regression*, arXiv:2506.11367v4 |
| `lr_atl` | Adaptive hypothesis transfer (ATL-FLR) | Lin and Reimherr (2024), *On hypothesis transfer learning of functional linear models*, ICML, PMLR 235, 30252–30285 |
| `hl_agg` | Adaptive FPCA transfer with lasso correction | Hu and Lin (2025), *Transfer learning meets functional linear regression: no negative transfer under posterior drift*, AAAI 39(16), 17351–17358 |

## Operator-space adaptation (Appendix D.1)

Each task supplies the empirical normal matrix and predictor–response cross-moment
of the quadratic prediction loss. The regression coefficient is the operator's
coordinate vector: 512 functional-regression coefficients, 625 real Fourier
multiplier coordinates (313 conjugate-frequency units), or complex B-spline
harmonic-transfer coefficients. Moments are uncentered and no intercept is added.

JM fits task coefficients in target empirical principal directions, aligns their
signs, averages by sample size, and fits the resulting shape's amplitude on target
data. Its truncation and source subset are selected on independent development
data in simulations; LTV uses validation to select from source subsets ranked by
shape misalignment.

LR fits a source ridge estimate and target ridge offset over nested source sets
ranked on the first target half. The source penalty scales with selected source
sample size to power −6/7. Star-shaped aggregation uses held-out target data.
LTV uses its training and validation records and the class's spline smoothing
penalty. The estimator is not refitted on the held-out portion.

HL uses source principal directions, pooled least squares, and a lasso target
offset. Cross-moment discrepancy ranks sources; each nested candidate has its own
selected truncation and lasso fraction. Sparse aggregation uses held-out target
data. Complex LTV coefficients use a fixed eigenvector phase and separate real
and imaginary lasso coordinates.

The standalone `lr_tl` and `hl_tl` experiment variants are excluded. Their numerical
building blocks remain because the reported adaptive methods use them. Candidate
IDs for the three retained methods preserve their original registry order.

## Selection and pairing

The simulation configurations retain the final Appendix D.1 grids: 500 development
datasets per functional target size and 300 per advection setting. Selection is
frozen before 1000 functional trajectories or 300 advection replications per cell.
LTV selects on validation using the same 24 target rotations and source records.
All numerical grids and seed IDs are in `experimentN/configs/`.

Functional literature methods use the noise-free block coefficient `c=1` specified
in Appendix D.1; their errors are paired with the archived original estimators.
Runners verify pairing by independently recomputing Target-only and Pooling.
Tests cover least-squares constructions, lasso optimality, aggregation, and stored
means. Retained baseline results contain only the three methods in the final paper.

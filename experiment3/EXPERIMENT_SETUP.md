# Final-paper protocol: Section 4.4 and Appendix D.4

Each record averages three steady-state periods of 15,640 samples at 156.25 kHz,
then applies the Fourier transform normalized by 15,640. Evaluation uses frequency
lines 51–2001 (0.5–20 kHz): 73 forced lines and 1,878 sideband lines. Measured
input spectra enter all fits; reference inputs label the forced lines only.

Eight targets combine one- or three-harmonic scheduling, 350 or 700 mV peak-to-peak
amplitudes, and two scheduling realizations. Each has three independent-input
records, rotated through training, validation, and testing. Source shortlists use
only scheduling and input channels. They contain six constant and six time-varying
regimes per target, drawn from 46 available regimes; each contributes two or three
records respectively. Target trajectories and test records are excluded.

## Representation and four original methods

The operator class uses complex B-spline harmonic-transfer coefficients with shift
radius `R` and spline dimension `B`, predicting the output via the integrated
state representation. Leave-one-target-out selection considers
`R = [7, 10, 15, 20]`, `B = [20, 30, 40]`, and smoothing multipliers
`[1e-8, 1e-6, 1e-4]`. Each target's selected class minimizes validation error on the
other seven targets' 21 rotations. Ties favor smaller B, smaller R, then larger
smoothing. The selected classes have 930 or 1,640 complex coefficients and
smoothing `1e-4`, shared across that target's three rotations.

Target-only fits one training record. Pooling fits all target and source training
records together and selects its smoothing on validation. Unprojected and Projected
calibrate each source using whitened instrumental moments and the scheduling
contrast dictionary. Their independent forward selection accepts a source only
when validation loss improves by at least 2%, up to four sources, jointly searching
common source weights `[0, .01, .03, .1, .3, 1, 3]`. Calibration gates, elastic-net
penalties, normalization, and tie rules are specified in `configs/design.json`.
The class and calibration constants are validated when loading this frozen design.

## Literature comparison methods

All methods use the same operator class, observations, and prediction loss.
JM ranks sources by coefficient-shape misalignment, selects a spectral truncation
and nested source set on validation, then estimates an amplitude on training.
LR ranks sources by the spline penalty norm, constructs source-ridge plus
target-offset fits for nested sets, and aggregates predictions on validation.
HL ranks sources by cross-moment discrepancy, constructs source principal-component
fits and lasso target offsets, tunes each nested-set candidate on validation,
and applies sparse aggregation. The exact final grids are in
`configs/literature.yaml`; only JM (`jm_ts`), adaptive LR (`lr_atl`), and adaptive
HL (`hl_agg`) are evaluated. Internal nonadaptive fitting routines remain as
building blocks and mathematical regression checks.

The target validation record chooses all parameters and aggregation weights.
Predictions on the test input are stored before reading the test output. There is
no refitting on combined training and validation records. The error for each band
is the sum of squared prediction residuals divided by the test output energy in
that band, averaged over the 24 evaluations for Table 1 and over three rotations
per target for Tables 3–4.

## Supplementary checks discussed in Appendix D.4.4

`diagnostics.py` evaluates only the three checks discussed in the final paper:
normal-matrix eigenvalues relative to the largest eigenvalue, LR's source-smoothing
grid extended by `[1, 100]`, and zero-lasso endpoints for HL's lower-grid-bound
candidates. These checks require the raw benchmark. The new diagnostic runner
writes separate results under the run directory; the source package contains no
archived numeric results from the manuscript's original supplementary check runs.

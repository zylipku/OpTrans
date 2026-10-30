"""Literature transfer estimators for functional linear regression, on the paper's operator classes.

Jiao and McKeague (2025, coefficient-shape transfer TS), Lin and Reimherr (2024,
TL-FLR and ATL-FLR) and Hu and Lin (2025, TL-FLR and Agg TL-FLR), adapted to the
three experiments through their least-squares geometry (``LITERATURE_BASELINES.md``):

- :mod:`core`         method keys and labels, shape transfer, development selection
- :mod:`lasso`        exact lasso homotopy path with segment-exact evaluation
- :mod:`aggregation`  held-out parts, nested source sets, sparse and star-shaped aggregation
- :mod:`functional`   experiment1: the functional block model (Section 4.2)
- :mod:`advection`    experiment2: Fourier-multiplier advection-diffusion operators (Section 4.3)
- :mod:`ltv`          experiment3: harmonic-transfer classes of the LTV benchmark (Section 4.4)
"""

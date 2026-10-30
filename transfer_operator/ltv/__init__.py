"""Real-data transfer for linear time-varying targets (Section 4.4, experiment3).

Ported from the frozen `LTV_target_representation_first_02` and Pooling runs of
the real-data audit (2026-09-24/25) with the numerical routines unchanged:

- :mod:`records`    streaming MATLAB v5 column reader, design channels, Fourier records
- :mod:`catalog`    inventory of the benchmark archive: groups, input spectra, forced lines
- :mod:`harmonic`   banded harmonic-transfer conventions, dictionary, calibration, metrics
- :mod:`banded`     complex banded normal equations, source moments, projection, selection
- :mod:`calibration` per-source calibration with gate folds
- :mod:`pooling`    one common operator fitted to the target and all source records
"""

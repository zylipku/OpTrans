"""Pooling: one common operator fitted to the target train record and every source record.

Only the smoothing penalty is validated; there is no calibration, projection,
source selection or weighting, and no Target-only fallback.
"""
from __future__ import annotations

import numpy as np

from .banded import fit_quadratic, target_stats
from .harmonic import metrics, unpack


def choose_smooth(curves):
    """Minimum validation error; machine-precision ties favour the larger smooth."""
    if len(curves) != 3:
        raise ValueError('Exactly three smoothing candidates are required')
    minimum = min(float(row['validation_full_pe']) for row in curves)
    eps = np.finfo(np.float64).eps
    eligible = [row for row in curves
                if abs(float(row['validation_full_pe'])-minimum)
                <= eps*max(abs(float(row['validation_full_pe'])), abs(minimum),
                           np.finfo(np.float64).tiny)]
    return float(max(eligible, key=lambda row: row['smooth'])['smooth'])


def band_product(ab, vector):
    """Hermitian band multiplication."""
    result = ab[0]*vector
    for offset in range(1, len(ab)):
        result[offset:] += ab[offset, :-offset]*vector[:-offset]
        result[:-offset] += ab[offset, :-offset].conj()*vector[offset:]
    return result


def normal_equation_residual(model, h, b, regularizer, coefficients):
    beta = unpack(coefficients).reshape(2*model.R+1, model.B).T.reshape(-1)
    lhs = band_product(h.ab+regularizer.ab, beta)
    return float(np.linalg.norm(lhs-b)/max(float(np.linalg.norm(b)), 1e-300))


def fit_pooling(model, target_train, source_records, validation, smoothing_grid):
    """Fit each smoothing penalty on the identical complete record bank."""
    grid = [float(value) for value in smoothing_grid]
    if len(grid) != 3 or set(grid) != {1e-8, 1e-6, 1e-4}:
        raise ValueError('Pooling smoothing grid must be exactly 1e-8,1e-6,1e-4')
    if not source_records:
        raise ValueError('Pooling requires every frozen source record')
    fitting_records = [target_train] + list(source_records)
    h, b = target_stats(model, fitting_records)
    regularizers, coefficients, curves = {}, {}, []
    Y = validation['Y'][model.rows]
    forced = np.isin(model.rows, validation.get('ref', {}).get('forced_lines', []))
    for smooth in grid:
        regularizer = model.regularizer(h, smooth)
        z = fit_quadratic(h, b, regularizer)
        full_pe = float(metrics(model.predict(validation['F'], z), Y, forced)['full_band'])
        residual = normal_equation_residual(model, h, b, regularizer, z)
        if not np.isfinite(full_pe) or not np.isfinite(residual):
            raise FloatingPointError('Nonfinite Pooling fit or validation PE')
        if residual > 1e-8:
            raise ArithmeticError('Pooling coefficients do not solve the pooled normal equations')
        regularizers[smooth] = regularizer
        coefficients[smooth] = z
        curves.append(dict(smooth=smooth, validation_full_pe=full_pe,
                           normal_equation_relative_residual=residual))
    return dict(h=h, b=b, regularizers=regularizers, coefficients=coefficients,
                curves=curves, selected_smooth=choose_smooth(curves),
                fitting_record_count=len(fitting_records))

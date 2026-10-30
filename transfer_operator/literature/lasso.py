"""Exact lasso paths for the contrast step of Hu and Lin (2025).

The objective is ``1/2 d'Gd - c'd + tau ||d||_1``, the sufficient-statistic form
of ``(2n)^{-1} ||Y - Xw - Xd||^2 + tau ||d||_1``. Homotopy (LARS with the lasso
modification, Efron et al. 2004) follows the exact piecewise-linear path, so
every grid value of tau is solved exactly rather than to an iteration
tolerance. The active Gram is factorised incrementally (Cholesky append on a
join, Givens downdate on a drop). A diagonal Gram has the closed-form
soft-thresholding solution.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_triangular


def _append(L, column, diagonal):
    """Cholesky factor of ``[[A, a], [a', g]]`` from that of ``A``."""
    if L.size == 0:
        return np.array([[np.sqrt(max(diagonal, 1e-300))]])
    row = solve_triangular(L, column, lower=True, check_finite=False)
    square = max(diagonal - row @ row, 1e-14 * max(diagonal, 1e-300))
    size = L.shape[0]
    grown = np.zeros((size + 1, size + 1))
    grown[:size, :size] = L
    grown[size, :size] = row
    grown[size, size] = np.sqrt(square)
    return grown


def _delete(L, k):
    """Cholesky factor after deleting variable ``k`` (Givens on the Hessenberg rest)."""
    H = np.delete(L, k, axis=0)
    for j in range(k, H.shape[0]):
        a, b = H[j, j], H[j, j + 1]
        r = np.hypot(a, b)
        if r == 0.0:
            continue
        c, s = a / r, b / r
        left, right = H[j:, j].copy(), H[j:, j + 1].copy()
        H[j:, j] = c * left + s * right
        H[j:, j + 1] = -s * left + c * right
    return H[:, :-1]


def lasso_path(G, c, taus, *, ridge=0.0, max_steps=None):
    """Return ``(len(taus), M)`` solutions of ``1/2 d'(G+ridge I)d - c'd + tau|d|_1``.

    ``G`` is a symmetric positive semidefinite matrix or the diagonal of one.
    A positive ``ridge`` makes every active Gram nonsingular; it is the only
    change to the objective. Taus may be given in any order and are clipped to
    the path: values above ``max|c|`` give zero.
    """
    c = np.asarray(c, dtype=float)
    taus = np.asarray(taus, dtype=float)
    if np.any(taus < 0) or not np.all(np.isfinite(taus)):
        raise ValueError('Lasso penalties must be finite and nonnegative')
    G = np.asarray(G, dtype=float)
    M = c.size
    if G.ndim == 1:
        diagonal = G + ridge
        if np.any(diagonal <= 0):
            raise np.linalg.LinAlgError('Nonpositive diagonal lasso curvature')
        return (np.sign(c)[None] * np.maximum(np.abs(c)[None] - taus[:, None], 0.0)
                / diagonal[None])
    if G.shape != (M, M):
        raise ValueError('Gram and linear term dimensions differ')
    if ridge:
        G = G + float(ridge) * np.eye(M)
    out = np.zeros((taus.size, M))
    tau = float(np.max(np.abs(c), initial=0.0))
    if tau == 0.0 or not taus.size:
        return out
    tau_min = float(taus.min())
    first = int(np.argmax(np.abs(c)))
    order, signs = [first], [float(np.sign(c[first]))]
    active = np.zeros(M, dtype=bool)
    active[first] = True
    L = _append(np.zeros((0, 0)), None, G[first, first])
    columns = np.empty((M, M), order='F')     # G[:, order], kept in step with the active set
    columns[:, 0] = G[:, first]
    # On a segment with active set A and signs s, the solution is exactly
    # d_A(t) = u - t v with u = G_AA^{-1} c_A and v = G_AA^{-1} s, so every knot
    # and every requested penalty is evaluated afresh, without accumulated drift.
    segments = []
    joined = first   # a variable that has just joined cannot leave at the same knot
    limit = 50 * M + 10 if max_steps is None else int(max_steps)
    for _ in range(limit):
        index = np.asarray(order)
        size = index.size
        rhs = np.column_stack((c[index], np.asarray(signs)))
        uv = solve_triangular(L.T, solve_triangular(L, rhs, lower=True, check_finite=False),
                              lower=False, check_finite=False)
        u, v = uv[:, 0], uv[:, 1]
        product = columns[:, :size] @ uv
        offset = c - product[:, 0]            # correlations: offset + t * slope
        slope = product[:, 1]
        ceiling = tau * (1.0 - 1e-12)
        floor, event, which = tau_min, 'end', -1
        inactive = np.flatnonzero(~active)
        if inactive.size:
            # A correlation reaches +t (-t) from inside only if 1 - slope (1 + slope) > 0.
            a, b = offset[inactive], slope[inactive]
            with np.errstate(divide='ignore', invalid='ignore'):
                plus = np.where(1.0 - b > 1e-12, a / (1.0 - b), -np.inf)
                minus = np.where(1.0 + b > 1e-12, -a / (1.0 + b), -np.inf)
            times = np.maximum(np.where((plus < ceiling) & (plus > floor), plus, -np.inf),
                               np.where((minus < ceiling) & (minus > floor), minus, -np.inf))
            k = int(np.argmax(times))
            if times[k] > floor:
                floor, event, which = float(times[k]), 'join', int(inactive[k])
        with np.errstate(divide='ignore', invalid='ignore'):
            zero = np.where(v != 0.0, u / v, -np.inf)
        zero = np.where((zero < ceiling) & (zero > floor) & (index != joined), zero, -np.inf)
        if zero.size:
            k = int(np.argmax(zero))
            if zero[k] > floor:
                floor, event, which = float(zero[k]), 'drop', k
        segments.append((tau, floor, index.copy(), u, v))
        tau = floor
        if event == 'drop':
            variable = order.pop(which)
            signs.pop(which)
            active[variable] = False
            L = _delete(L, which)
            columns[:, which:size - 1] = columns[:, which + 1:size]
            joined = -1
        elif event == 'join':
            L = _append(L, G[index, which], G[which, which])
            columns[:, size] = G[:, which]
            order.append(which)
            signs.append(float(np.sign(offset[which] + tau * slope[which])))
            active[which] = True
            joined = which
        else:
            break
    else:
        raise RuntimeError('Lasso homotopy did not reach the smallest requested penalty')
    top = segments[0][0]
    for i, value in enumerate(taus):
        if value >= top:
            continue          # at or above max|c| the solution is exactly zero
        for high, low, index, u, v in segments:
            if low <= value <= high:
                out[i, index] = u - value * v
                break
    return out

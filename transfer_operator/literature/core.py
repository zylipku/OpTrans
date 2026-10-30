"""Shared pieces of the three literature transfer estimators.

All three papers study scalar-on-function regression ``y = <X, beta> + eps``.
Here every experiment supplies the least-squares geometry of its own operator
class: task ``j`` contributes ``n_j`` observations through the empirical Gram
``G_j = n_j^{-1} sum X^*X`` (the empirical target curvature for j = 0) and the
cross moment ``g_j = n_j^{-1} sum X^*Y``. FPCA of the covariate becomes the
eigendecomposition of a Gram, the slope function becomes the operator, and the
L2/RKHS inner products become those of the experiment's hypothesis space.

Method keys (display names in :data:`LABELS`):

``jm_ts``   Jiao and McKeague (2025), coefficient-shape transfer (TS).
``lr_tl``   Lin and Reimherr (2024), offset transfer TL-FLR (Algorithm 1).
``lr_atl``  Lin and Reimherr (2024), aggregation-based ATL-FLR (Algorithm 2).
``hl_tl``   Hu and Lin (2025), FPCA transfer with lasso bias correction (Algorithm 1).
``hl_agg``  Hu and Lin (2025), adaptive aggregated version (Algorithm 2).
"""
from __future__ import annotations

import numpy as np

REPORTED = ('jm_ts', 'lr_atl', 'hl_agg')
# Only the final manuscript's three comparisons are registered as experiments.
# TL-FLR primitives remain available inside the adaptive estimators.
METHODS = REPORTED
LABELS = {
    'jm_ts': 'Jiao & McKeague (2025), TS',
    'lr_tl': 'Lin & Reimherr (2024), TL-FLR',
    'lr_atl': 'Lin & Reimherr (2024), ATL-FLR',
    'hl_tl': 'Hu & Lin (2025), TL-FLR',
    'hl_agg': 'Hu & Lin (2025), Agg TL-FLR',
}
PAPERS = {
    'jm_ts': 'Jiao, S. and McKeague, I. W. (2025). Coefficient shape transfer learning for '
             'functional linear regression. arXiv:2506.11367v4.',
    'lr_tl': 'Lin, H. and Reimherr, M. (2024). On hypothesis transfer learning of functional '
             'linear models. ICML, PMLR 235, 30252-30285.',
    'hl_tl': 'Hu, X. and Lin, Z. (2025). Transfer learning meets functional linear regression: '
             'no negative transfer under posterior drift. AAAI 39(16), 17351-17358.',
}
PAPERS['lr_atl'], PAPERS['hl_agg'] = PAPERS['lr_tl'], PAPERS['hl_tl']


def descending_eigh(G):
    """Eigenpairs of a symmetric matrix with eigenvalues in decreasing order."""
    values, vectors = np.linalg.eigh(np.asarray(G, dtype=float))
    return values[::-1].copy(), vectors[:, ::-1].copy()


def numerical_rank(values, rtol=1e-12):
    values = np.asarray(values, dtype=float)
    top = float(values.max(initial=0.0))
    return int(np.sum(values > rtol * top)) if top > 0 else 0


def shape_combination(coefficients, counts):
    """Jiao and McKeague's shape estimate ``sum_j omega_j beta_j``.

    ``coefficients[0]`` is the target pre-estimate; the remaining entries are
    the informative sources. ``omega_j = sign(<beta_j, beta_0>) n_j / n_total``
    with the sign of the target itself equal to one.
    """
    target = coefficients[0]
    total = float(sum(counts))
    shape = np.zeros_like(target, dtype=float)
    for index, (beta, count) in enumerate(zip(coefficients, counts)):
        sign = 1.0 if index == 0 else (1.0 if float(beta @ target) >= 0.0 else -1.0)
        shape = shape + sign * (count / total) * beta
    return shape


def amplitude(shape, gram_shape, cross):
    """Least-squares scalar ``a_0`` for ``a_0 * shape`` on the target sample.

    ``gram_shape`` is ``G_0 shape`` and ``cross`` is ``g_0``; a zero shape
    returns zero amplitude.
    """
    curvature = float(shape @ gram_shape)
    return float(shape @ cross) / curvature if curvature > 0.0 else 0.0


ATL_SOURCE_EXPONENT = 6 / 7


def atl_source_scale(size, first):
    """Factor on ATL-FLR's transfer-step penalty for a candidate with ``size`` source observations.

    Lin and Reimherr's code sets the transfer-step penalty of candidate ``T_k`` to
    ``c (|T_k| N_A)^(-6/7)`` (their rate ``n_S^(-2r/(2r+1))`` with r = 3), with one
    constant ``c`` for all candidates. The adapters keep the penalty of the first
    nested candidate (``first`` observations) at the tuned fraction and scale the others.
    """
    return (float(size) / float(first)) ** (-ATL_SOURCE_EXPONENT)


def select_development(risks):
    """Smallest mean development risk among candidates finite in every replication.

    ``risks`` is (replications, rows), or (replications, sets, rows) for
    ``hl_agg``, whose Algorithm 2 candidates are tuned one by one. Ties keep the
    first row. Returns ``(indices, means, mcses)`` with one entry per candidate
    set (a single entry for a two-dimensional bank).
    """
    risks = np.asarray(risks, dtype=float)
    cube = risks[:, None, :] if risks.ndim == 2 else risks
    indices, means, mcses = [], [], []
    for block in np.moveaxis(cube, 1, 0):
        finite = np.isfinite(block).all(axis=0)
        if not finite.any():
            raise ValueError('No candidate is finite in every development replication')
        average = np.full(block.shape[1], np.inf)
        average[finite] = block[:, finite].mean(axis=0)
        index = int(np.argmin(average))
        indices.append(index)
        means.append(float(average[index]))
        mcses.append(float(np.std(block[:, index], ddof=1) / np.sqrt(block.shape[0])))
    return indices, means, mcses


def frozen_row(method, rows, indices):
    """The frozen parameters: a registry row, or one row per hl_agg candidate."""
    if method == 'hl_agg':
        return dict(candidates=[dict(truncation=rows[i]['truncation'], lasso=rows[i]['lasso']) for i in indices])
    return rows[indices[0]]


def subsets(m):
    """All subsets of the sources 1..m, smallest first (the empty set first)."""
    from itertools import combinations
    return [tuple(c) for size in range(m + 1) for c in combinations(range(1, m + 1), size)]


def registry(grid, m):
    """Stable zero-based candidate lists for every literature method.

    ``grid`` supplies ``truncation`` (FPCA dimensions), ``target_ridge`` and
    ``source_ridge`` fractions (Lin and Reimherr's lambda_2 and lambda_1) and
    ``lasso`` fractions of the zero-solution threshold (Hu and Lin's tau).
    Source sets are explicit tuning values for ``jm_ts`` and data-driven for
    ``lr_atl`` and ``hl_agg``.
    """
    sets = subsets(m)
    rows = {method: [] for method in METHODS}
    for informative in sets:
        for D in grid['truncation']:
            rows['jm_ts'].append(dict(sources=list(informative), truncation=int(D)))
    for r1 in grid['source_ridge']:
        for r2 in grid['target_ridge']:
            rows['lr_atl'].append(dict(source_ridge=r1, target_ridge=r2))
    for M in grid['truncation']:
        for tau in grid['lasso']:
            rows['hl_agg'].append(dict(truncation=int(M), lasso=tau))
    for method, values in rows.items():
        for index, row in enumerate(values):
            row['index'] = index
    return rows

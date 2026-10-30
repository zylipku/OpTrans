"""Held-out quadratic risks, nested source sets and sparse aggregation.

Both adaptive literature estimators split the target sample, fit candidate
estimators on one part, and aggregate them on the other part:

* Lin and Reimherr (2024, Algorithms 2-3): candidate sets ordered by the RKHS
  distance between target-only and source-only fits; star-shaped sparse
  aggregation of Gaiffas and Lecue (2011) with a preselection step. The
  constants ``c = 4(1+9b)`` and ``phi = b sqrt((log M + 0.001)/n)`` follow the
  authors' released code (github.com/haotianlin/HTL-FLM, ``Staragg_func``).
* Hu and Lin (2025, Algorithm 2): candidate sets ordered by the discrepancy of
  cross-covariances; sparse aggregation without preselection.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HeldOut:
    """Empirical squared-error risk ``b'Gb - 2g'b + yy`` of one data part.

    ``G`` is a dense symmetric matrix or its diagonal. ``yy`` is the mean
    squared response, so ``risk`` equals the held-out mean squared error.
    """
    n: int
    G: np.ndarray
    g: np.ndarray
    yy: float

    def gram(self, beta):
        return self.G * beta if np.ndim(self.G) == 1 else self.G @ beta

    def risk(self, beta):
        beta = np.asarray(beta, dtype=float)
        return float(beta @ self.gram(beta) - 2.0 * self.g @ beta + self.yy)

    def norm(self, beta):
        """Empirical prediction norm ``(n^{-1} sum <X_i, beta>^2)^{1/2}``."""
        beta = np.asarray(beta, dtype=float)
        return float(np.sqrt(max(beta @ self.gram(beta), 0.0)))


def nested_sets(statistic):
    """``[set of the t smallest values]`` for t = 0..m (stable order for ties)."""
    order = np.argsort(np.asarray(statistic, dtype=float), kind='stable')
    return [tuple(sorted(int(j) for j in order[:t])) for t in range(len(order) + 1)]


def _segment(part, left, right):
    """Minimise ``part.risk(lam*left + (1-lam)*right)`` over lam in [0, 1]."""
    direction = left - right
    curvature = direction @ part.gram(direction)
    if curvature <= 0.0:
        return 1.0, part.risk(left)
    slope = direction @ (part.gram(right) - part.g)
    lam = float(np.clip(-slope / curvature, 0.0, 1.0))
    return lam, part.risk(lam * left + (1.0 - lam) * right)


def sparse_aggregation(candidates, part):
    """Hu and Lin (2025): best candidate plus the best segment to another one.

    Returns ``(beta, info)``; ``info`` records the two indices and the weight
    ``lam`` of ``lam*beta_{l1} + (1-lam)*beta_{l2}``.
    """
    candidates = [np.asarray(b, dtype=float) for b in candidates]
    risks = [part.risk(b) for b in candidates]
    first = int(np.argmin(risks))
    best = (risks[first], 1.0, first)
    for index, beta in enumerate(candidates):
        if index == first:
            continue
        lam, value = _segment(part, candidates[first], beta)
        if value < best[0]:
            best = (value, lam, index)
    value, lam, second = best
    beta = lam * candidates[first] + (1.0 - lam) * candidates[second]
    return beta, dict(first=first, second=second, weight=lam, risk=value)


def star_aggregation(candidates, part1, part2, bound):
    """Lin and Reimherr (2024): preselection on part 1, star segment on part 2.

    ``bound`` is the constant ``b`` of their code: the largest response or
    candidate prediction on the target data. ``phi`` uses the size of the
    whole aggregation sample, as in the released code.
    """
    candidates = [np.asarray(b, dtype=float) for b in candidates]
    count = len(candidates)
    risk1 = np.array([part1.risk(b) for b in candidates])
    risk2 = np.array([part2.risk(b) for b in candidates])
    first = int(np.argmin(risk1))
    phi = bound * np.sqrt((np.log(count) + 0.001) / (part1.n + part2.n))
    constant = 4.0 * (1.0 + 9.0 * bound)
    norms1 = np.array([part1.norm(b - candidates[first]) for b in candidates])
    kept = [k for k in range(count)
            if risk1[k] <= risk1[first] + constant * max(phi * norms1[k], phi * phi)]
    best = None
    for k in kept:
        squared = part2.norm(candidates[k] - candidates[first]) ** 2
        if k == first or squared <= 0.0:
            lam = 1.0
        else:
            lam = float(np.clip(0.5 * ((risk2[k] - risk2[first]) / squared + 1.0), 0.0, 1.0))
        beta = lam * candidates[first] + (1.0 - lam) * candidates[k]
        value = part2.risk(beta)
        if best is None or value < best[0]:
            best = (value, k, lam, beta)
    value, second, lam, beta = best
    return beta, dict(first=first, second=second, weight=lam, risk=value,
                      preselected=len(kept))

"""Experiment 2 adapter: Fourier-multiplier operators of advection-diffusion.

The hypothesis space H is the real Hilbert space of conjugate-symmetric Fourier
multipliers (Section 4.3) in its orthonormal real coordinates: the constant mode
and one cosine/sine pair per retained frequency. The least-squares Gram of
``h -> A(h)F`` is exactly diagonal (cross terms cancel within every pair), so
``G_j`` and ``g_j`` are ``model.target_terms`` of the task moments, FPCA orders
the constant mode and the frequency pairs by their Gram value (a pair is one
degenerate eigenspace and is kept or dropped as a unit), ridge solves are
elementwise and the Hu-Lin lasso is soft-thresholding. The slope inner product
and RKHS norm are those of H (the Hilbert-Schmidt norm), and the risk is (91).
"""
from __future__ import annotations

from itertools import groupby

import numpy as np

from ..pde_moments import empirical_task
from .aggregation import HeldOut, nested_sets, sparse_aggregation, star_aggregation
from .core import amplitude, atl_source_scale, shape_combination
from .lasso import lasso_path


def task_moments(model, stats, U=None):
    G, g = model.target_terms(stats)
    yy = float(np.mean(np.sum(U * U, axis=0))) if U is not None else np.nan
    return HeldOut(int(stats.n), G, g, yy)


def units(model):
    """Constant mode, then one (cos, sin) coordinate pair per retained frequency."""
    return [np.array([0])] + [np.array([c, s]) for c, s in zip(model.cos, model.sin)]


def fpca_order(model, G):
    """Units by decreasing Gram value (stable for ties): the FPCA basis of H."""
    blocks = units(model)
    values = np.array([G[b[0]] for b in blocks])
    return [blocks[i] for i in np.argsort(-values, kind='stable')]


def leading(order, count):
    return np.concatenate(order[:count])


class AdvectionWorld:
    """Target moments and samples with two source moments for one replication."""

    def __init__(self, model, target_stats, F, U, source_stats):
        self.model = model
        self.F, self.U = np.asarray(F, float), np.asarray(U, float)
        self.n0 = int(target_stats.n)
        self.target = task_moments(model, target_stats, self.U)
        self.sources = [task_moments(model, st) for st in source_stats]
        half = self.n0 // 2
        quarter = half + (self.n0 - half) // 2
        part = lambda a, b: task_moments(model, empirical_task(self.F[:, a:b], self.U[:, a:b]), self.U[:, a:b])
        self.first, self.second = part(0, half), part(half, self.n0)
        self.second_a, self.second_b = part(half, quarter), part(quarter, self.n0)
        self._orders = {}

    def order(self, key):
        if key not in self._orders:
            if key == 'target':
                G = self.target.G
            elif key == 'first':
                G = self.first.G
            else:
                total = sum(self.sources[j - 1].n for j in key[1])
                G = sum(self.sources[j - 1].n / total * self.sources[j - 1].G for j in key[1])
            self._orders[key] = fpca_order(self.model, G)
        return self._orders[key]

    def risks(self, columns):
        h = np.asarray(columns, float)
        if h.ndim == 1:
            return self.model.risk(h)
        return np.sum(self.model.q[:, None] * (h - self.model.truth[:, None]) ** 2, axis=0)

    # --- Jiao and McKeague -----------------------------------------------------
    def jm_ts(self, rows, coefficients=False):
        out, coef = np.full(len(rows), np.nan), {}
        order = self.order('target')
        for i, row in enumerate(rows):
            kept = leading(order, row['truncation'])
            chosen = [self.target] + [self.sources[j - 1] for j in row['sources']]
            shape = shape_combination([m.g[kept] / m.G[kept] for m in chosen], [m.n for m in chosen])
            scale = amplitude(shape, self.target.G[kept] * shape, self.target.g[kept])
            h = np.zeros(self.model.d)
            h[kept] = scale * shape
            out[i] = self.risks(h)
            if coefficients:
                coef[i] = dict(h=h, amplitude=scale)
        return (out, coef) if coefficients else out

    # --- Lin and Reimherr --------------------------------------------------------
    def _transfer_step(self, informative, fraction):
        G = sum(self.sources[j - 1].G for j in informative)
        g = sum(self.sources[j - 1].g for j in informative)
        return g / (G + fraction * G.max())

    @staticmethod
    def _calibration_step(part, start, fractions):
        residual = part.g - part.G * start
        penalties = np.asarray(fractions, float) * part.G.max()
        return start[:, None] + residual[:, None] / (part.G[:, None] + penalties[None, :])

    def lr_tl(self, rows, coefficients=False):
        out, coef = np.full(len(rows), np.nan), {}
        key = lambda i: (tuple(rows[i]['sources']), -1.0 if rows[i]['source_ridge'] is None
                         else rows[i]['source_ridge'])
        for (informative, r1), group in groupby(sorted(range(len(rows)), key=key), key=key):
            group = list(group)
            start = np.zeros(self.model.d) if not informative else self._transfer_step(informative, r1)
            h = self._calibration_step(self.target, start, [rows[i]['target_ridge'] for i in group])
            out[group] = self.risks(h)
            if coefficients:
                coef.update({i: dict(h=h[:, k]) for k, i in enumerate(group)})
        return (out, coef) if coefficients else out

    def _norms(self, h):
        """``||A(h)F_i||`` for every target sample; a pair acts as a scaled rotation."""
        if not hasattr(self, '_power'):
            square = self.F ** 2
            self._power = np.vstack([square[0], square[self.model.cos] + square[self.model.sin]])
        weights = np.r_[h[0] ** 2, (h[self.model.cos] ** 2 + h[self.model.sin] ** 2) / 2.0]
        return np.sqrt(weights @ self._power)

    def lr_atl(self, rows, coefficients=False):
        out, coef = np.full(len(rows), np.nan), {}
        m = len(self.sources)
        response = float(np.sqrt(np.sum(self.U ** 2, axis=0)).max())
        for i, row in enumerate(rows):
            r1, r2 = row['source_ridge'], row['target_ridge']
            target_only = self._calibration_step(self.first, np.zeros(self.model.d), [r2])[:, 0]
            distances = [float(np.sum((target_only - src.g / (src.G + r1 * src.G.max())) ** 2))
                         for src in self.sources]
            sets = [tuple(j + 1 for j in s) for s in nested_sets(distances)]
            size = lambda informative: sum(self.sources[j - 1].n for j in informative)
            candidates = [target_only] + [
                self._calibration_step(self.first, self._transfer_step(s, r1 * atl_source_scale(size(s), size(sets[1]))),
                                       [r2])[:, 0] for s in sets[1:]]
            bound = max(response, max(float(self._norms(b).max()) for b in candidates))
            h, info = star_aggregation(candidates, self.second_a, self.second_b, bound)
            out[i] = self.risks(h)
            if coefficients:
                coef[i] = dict(h=h, sets=sets, **info)
        return (out, coef) if coefficients else out

    # --- Hu and Lin ------------------------------------------------------------------
    def _hl_candidate(self, part, informative, M, fractions):
        kept = leading(self.order(('pooled', tuple(informative))), M)
        total = sum(self.sources[j - 1].n for j in informative)
        G = sum(self.sources[j - 1].n / total * self.sources[j - 1].G for j in informative)
        g = sum(self.sources[j - 1].n / total * self.sources[j - 1].g for j in informative)
        w = g[kept] / G[kept]
        linear = part.g[kept] - part.G[kept] * w
        taus = np.asarray(fractions, float) * float(np.max(np.abs(linear)))
        h = np.zeros((self.model.d, len(fractions)))
        h[kept] = w[:, None] + lasso_path(part.G[kept], linear, taus).T
        return h

    def _fpca_ols(self, key, part, M):
        kept = leading(self.order(key), M)
        h = np.zeros(self.model.d)
        h[kept] = part.g[kept] / part.G[kept]
        return h

    def hl_tl(self, rows, coefficients=False):
        out, coef = np.full(len(rows), np.nan), {}
        key = lambda i: (tuple(rows[i]['sources']), rows[i]['truncation'])
        for (informative, M), group in groupby(sorted(range(len(rows)), key=key), key=key):
            group = list(group)
            if not informative:
                h = self._fpca_ols('target', self.target, M)[:, None].repeat(len(group), axis=1)
            else:
                h = self._hl_candidate(self.target, informative, M, [rows[i]['lasso'] for i in group])
            out[group] = self.risks(h)
            if coefficients:
                coef.update({i: dict(h=h[:, k]) for k, i in enumerate(group)})
        return (out, coef) if coefficients else out

    def _hl_sets(self):
        discrepancy = [float(np.sum((src.g - self.first.g) ** 2)) for src in self.sources]
        return [tuple(j + 1 for j in s) for s in nested_sets(discrepancy)]

    def hl_agg_candidates(self, rows):
        """Risk of every Algorithm 2 candidate on every (m, tau) row, shape (sets, rows).

        Candidate 0 is target FPCA least squares on D1 (``truncation`` only);
        candidate l is Algorithm 1 on D1 with the l sources of smallest
        cross-moment discrepancy. Each candidate is tuned separately.
        """
        sets = self._hl_sets()
        out = np.full((len(sets), len(rows)), np.nan)
        for M, group in groupby(sorted(range(len(rows)), key=lambda i: rows[i]['truncation']),
                                key=lambda i: rows[i]['truncation']):
            group = list(group)
            out[0, group] = self.risks(self._fpca_ols('first', self.first, M))
            for l, informative in enumerate(sets[1:], 1):
                out[l, group] = self.risks(self._hl_candidate(self.first, informative, M,
                                                              [rows[i]['lasso'] for i in group]))
        return out

    def hl_agg(self, rows, coefficients=False):
        """Sparse aggregation on D2 of the candidates, each with its frozen (m, tau)."""
        out, coef = np.full(len(rows), np.nan), {}
        sets = self._hl_sets()
        for i, row in enumerate(rows):
            tuned = row['candidates']
            candidates = [self._fpca_ols('first', self.first, tuned[0]['truncation'])]
            candidates += [self._hl_candidate(self.first, informative, params['truncation'], [params['lasso']])[:, 0]
                           for informative, params in zip(sets[1:], tuned[1:])]
            h, info = sparse_aggregation(candidates, self.second)
            out[i] = self.risks(h)
            if coefficients:
                coef[i] = dict(h=h, sets=sets, **info)
        return (out, coef) if coefficients else out

    def evaluate(self, method, rows, coefficients=False):
        return getattr(self, method)(rows, coefficients=coefficients)

    def development(self, method, rows):
        """Development risks: one row per candidate, per Algorithm 2 candidate for hl_agg."""
        return self.hl_agg_candidates(rows) if method == 'hl_agg' else self.evaluate(method, rows)

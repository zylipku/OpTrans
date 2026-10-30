"""Experiment 1 adapter: the scalar response of the coupled functional model.

Every task observes ``F = (0, f)`` and ``U = (y, Lambda f)`` with
``y = <a_j, f> + epsilon`` (Section 4.2). The block output ``Lambda f`` is
observed without noise and is the same in every task, so its least-squares
coefficient is exactly ``c = 1``, the value Pooling fixes. The literature
estimators are therefore applied to the scalar functional linear model
``y = <a, f> + epsilon``; its covariates have one distribution in all tasks
(the posterior-drift setting of Lin and Reimherr and of Hu and Lin). The
hypothesis-space norm of ``a`` is the Euclidean norm of (25), the target FPCA is
the eigendecomposition of ``S_0 = n0^{-1} sum f f'``, and the risk is (82) at
``c = 1``.

Moments are uncentered, as for every estimator of the experiment (all inputs
and responses have mean zero). Sample splits of the adaptive methods take the
first and second halves of the i.i.d. target sample.
"""
from __future__ import annotations

from itertools import groupby

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .aggregation import HeldOut, nested_sets, sparse_aggregation, star_aggregation
from .core import amplitude, atl_source_scale, descending_eigh, numerical_rank, shape_combination
from .lasso import lasso_path


def sample_moments(f, y):
    n = len(y)
    return HeldOut(n, f.T @ f / n, y @ f / n, float(y @ y) / n)


def _ridge_columns(vectors, values, rhs, penalties):
    """``(G + lam I)^{-1} rhs`` for every lam, from ``G = V diag(values) V'``."""
    coordinates = vectors.T @ rhs
    values = np.clip(values, 0.0, None)
    return vectors @ (coordinates[:, None] / (values[:, None] + np.asarray(penalties)[None, :]))


class FunctionalWorld:
    """One target sample and its two source moments, with cached spectra."""

    def __init__(self, model, f0, y0, sources):
        self.model = model
        self.f0, self.y0 = np.asarray(f0, float), np.asarray(y0, float)
        self.n0 = len(self.y0)
        self.target = sample_moments(self.f0, self.y0)
        self.sources = [HeldOut(int(st['n']), st['Sf'], st['yf'], float(st['yy'])) for st in sources]
        half = self.n0 // 2
        self.first = sample_moments(self.f0[:half], self.y0[:half])
        self.second = sample_moments(self.f0[half:], self.y0[half:])
        quarter = half + (self.n0 - half) // 2
        self.second_a = sample_moments(self.f0[half:quarter], self.y0[half:quarter])
        self.second_b = sample_moments(self.f0[quarter:], self.y0[quarter:])
        self._spectra = {}

    # --- cached spectra -------------------------------------------------------
    def spectrum(self, key):
        if key not in self._spectra:
            if key == 'target':
                G = self.target.G
            elif key == 'first':
                G = self.first.G
            elif key[0] == 'source':
                G = self.sources[key[1] - 1].G
            elif key[0] == 'sum':
                G = sum(self.sources[j - 1].G for j in key[1])
            elif key[0] == 'pooled':
                total = sum(self.sources[j - 1].n for j in key[1])
                G = sum(self.sources[j - 1].n / total * self.sources[j - 1].G for j in key[1])
            else:
                raise KeyError(key)
            self._spectra[key] = descending_eigh(G)
        return self._spectra[key]

    def risks(self, columns):
        a = np.asarray(columns, float)
        if a.ndim == 1:
            return self.model.risk(1.0, a)
        return np.sum(self.model.nu[:, None] * (a - self.model.a0[:, None]) ** 2, axis=0)

    # --- Jiao and McKeague ------------------------------------------------------
    def jm_ts(self, rows, coefficients=False):
        values, vectors = self.spectrum('target')
        cross = {0: vectors.T @ self.target.g}
        grams = {j: vectors.T @ src.G @ vectors for j, src in enumerate(self.sources, 1)}
        cross.update({j: vectors.T @ src.g for j, src in enumerate(self.sources, 1)})
        out, coef = np.full(len(rows), np.nan), {}
        for D in sorted({r['truncation'] for r in rows}):
            if D > self.n0 - 1 or D > numerical_rank(values):
                continue
            estimates = {0: cross[0][:D] / values[:D]}
            for j in grams:
                estimates[j] = cho_solve(cho_factor(grams[j][:D, :D], lower=True), cross[j][:D])
            for i, row in enumerate(rows):
                if row['truncation'] != D:
                    continue
                chosen = [0] + list(row['sources'])
                shape = shape_combination([estimates[j] for j in chosen],
                                          [self.n0] + [self.sources[j - 1].n for j in chosen[1:]])
                scale = amplitude(shape, values[:D] * shape, cross[0][:D])
                a = vectors[:, :D] @ (scale * shape)
                out[i] = self.risks(a)
                if coefficients:
                    coef[i] = dict(a=a, amplitude=scale)
        return (out, coef) if coefficients else out

    # --- Lin and Reimherr ---------------------------------------------------------
    def _transfer_step(self, informative, fractions):
        """Algorithm 1 transfer step: ridge on sum_{t in S} n_t^{-1} sum loss."""
        values, vectors = self.spectrum(('sum', tuple(informative)))
        rhs = sum(self.sources[j - 1].g for j in informative)
        return _ridge_columns(vectors, values, rhs, np.asarray(fractions) * values[0])

    def _calibration_step(self, part, key, start, fractions):
        """Algorithm 1 calibration: ridge offset of ``start`` on a target part."""
        values, vectors = self.spectrum(key)
        rhs = part.g - part.G @ start
        return start[:, None] + _ridge_columns(vectors, values, rhs, np.asarray(fractions) * values[0])

    def lr_tl(self, rows, coefficients=False):
        out, coef = np.full(len(rows), np.nan), {}
        key = lambda i: (tuple(rows[i]['sources']), rows[i]['source_ridge'] is None and -1 or rows[i]['source_ridge'])
        order = sorted(range(len(rows)), key=key)
        for (informative, _), group in groupby(order, key=key):
            group = list(group)
            fractions = [rows[i]['target_ridge'] for i in group]
            if not informative:
                values, vectors = self.spectrum('target')
                beta = _ridge_columns(vectors, values, self.target.g, np.asarray(fractions) * values[0])
            else:
                start = self._transfer_step(informative, [rows[group[0]]['source_ridge']])[:, 0]
                beta = self._calibration_step(self.target, 'target', start, fractions)
            out[group] = self.risks(beta)
            if coefficients:
                coef.update({i: dict(a=beta[:, k]) for k, i in enumerate(group)})
        return (out, coef) if coefficients else out

    def lr_atl(self, rows, coefficients=False):
        out, coef = np.full(len(rows), np.nan), {}
        values, vectors = self.spectrum('first')
        m = len(self.sources)
        for i, row in enumerate(rows):
            r1, r2 = row['source_ridge'], row['target_ridge']
            target_only = _ridge_columns(vectors, values, self.first.g, [r2 * values[0]])[:, 0]
            distances = []
            for j in range(1, m + 1):
                sv, svec = self.spectrum(('source', j))
                fit = _ridge_columns(svec, sv, self.sources[j - 1].g, [r1 * sv[0]])[:, 0]
                distances.append(float(np.sum((target_only - fit) ** 2)))
            sets = [tuple(j + 1 for j in s) for s in nested_sets(distances)]
            size = lambda informative: sum(self.sources[j - 1].n for j in informative)
            candidates = [target_only]
            for informative in sets[1:]:
                scale = atl_source_scale(size(informative), size(sets[1]))
                start = self._transfer_step(informative, [r1 * scale])[:, 0]
                candidates.append(self._calibration_step(self.first, 'first', start, [r2])[:, 0])
            bound = max(float(self.y0.max()), max(float((self.f0 @ b).max()) for b in candidates))
            beta, info = star_aggregation(candidates, self.second_a, self.second_b, bound)
            out[i] = self.risks(beta)
            if coefficients:
                coef[i] = dict(a=beta, sets=sets, **info)
        return (out, coef) if coefficients else out

    # --- Hu and Lin -------------------------------------------------------------------
    def _hl_candidate(self, part, informative, M, fractions):
        """Algorithm 1 on a target part: pooled-source FPCA, LS, lasso contrast."""
        values, vectors = self.spectrum(('pooled', tuple(informative)))
        if M > numerical_rank(values):
            return None
        total = sum(self.sources[j - 1].n for j in informative)
        pooled_cross = sum(self.sources[j - 1].n / total * self.sources[j - 1].g for j in informative)
        basis = vectors[:, :M]
        w = (basis.T @ pooled_cross) / values[:M]
        gram = basis.T @ part.G @ basis
        linear = basis.T @ part.g - gram @ w
        taus = np.asarray(fractions, float) * float(np.max(np.abs(linear)))
        ridge = 1e-12 * float(np.trace(gram)) / M
        contrast = lasso_path(gram, linear, taus, ridge=ridge)
        return basis @ (w[:, None] + contrast.T)

    def _fpca_ols(self, key, part, M):
        values, vectors = self.spectrum(key)
        return vectors[:, :M] @ ((vectors[:, :M].T @ part.g) / values[:M])

    def hl_tl(self, rows, coefficients=False):
        out, coef = np.full(len(rows), np.nan), {}
        key = lambda i: (tuple(rows[i]['sources']), rows[i]['truncation'])
        for (informative, M), group in groupby(sorted(range(len(rows)), key=key), key=key):
            group = list(group)
            if not informative:
                if M > self.n0 - 1:
                    continue
                beta = self._fpca_ols('target', self.target, M)[:, None].repeat(len(group), axis=1)
            else:
                beta = self._hl_candidate(self.target, informative, M, [rows[i]['lasso'] for i in group])
                if beta is None:
                    continue
            out[group] = self.risks(beta)
            if coefficients:
                coef.update({i: dict(a=beta[:, k]) for k, i in enumerate(group)})
        return (out, coef) if coefficients else out

    def _hl_sets(self):
        discrepancy = [float(np.sum((src.g - self.first.g) ** 2)) for src in self.sources]
        return [tuple(j + 1 for j in s) for s in nested_sets(discrepancy)]

    def hl_agg_candidates(self, rows):
        """Risk of every Algorithm 2 candidate on every (m, tau) row, shape (sets, rows).

        Candidate 0 is target FPCA least squares on D1 (it uses ``truncation``
        only); candidate l is Algorithm 1 on D1 with the l sources of smallest
        cross-moment discrepancy. Each candidate is tuned separately, as each is
        fitted by its own Algorithm 1 in Hu and Lin's Algorithm 2.
        """
        sets = self._hl_sets()
        out = np.full((len(sets), len(rows)), np.nan)
        rank = self.first.n - 1
        for M, group in groupby(sorted(range(len(rows)), key=lambda i: rows[i]['truncation']),
                                key=lambda i: rows[i]['truncation']):
            group = list(group)
            if rank >= 1:
                out[0, group] = self.risks(self._fpca_ols('first', self.first, min(M, rank)))
            for l, informative in enumerate(sets[1:], 1):
                fit = self._hl_candidate(self.first, informative, M, [rows[i]['lasso'] for i in group])
                if fit is not None:
                    out[l, group] = self.risks(fit)
        return out

    def hl_agg(self, rows, coefficients=False):
        """Sparse aggregation on D2 of the candidates, each with its frozen (m, tau)."""
        out, coef = np.full(len(rows), np.nan), {}
        sets = self._hl_sets()
        rank = self.first.n - 1
        for i, row in enumerate(rows):
            tuned = row['candidates']
            if rank < 1:
                continue
            candidates = [self._fpca_ols('first', self.first, min(tuned[0]['truncation'], rank))]
            for informative, params in zip(sets[1:], tuned[1:]):
                fit = self._hl_candidate(self.first, informative, params['truncation'], [params['lasso']])
                candidates.append(None if fit is None else fit[:, 0])
            if any(c is None for c in candidates):
                continue
            beta, info = sparse_aggregation(candidates, self.second)
            out[i] = self.risks(beta)
            if coefficients:
                coef[i] = dict(a=beta, sets=sets, **info)
        return (out, coef) if coefficients else out

    def evaluate(self, method, rows, coefficients=False):
        return getattr(self, method)(rows, coefficients=coefficients)

    def development(self, method, rows):
        """Development risks: one row per candidate, per Algorithm 2 candidate for hl_agg."""
        return self.hl_agg_candidates(rows) if method == 'hl_agg' else self.evaluate(method, rows)

"""Experiment 3 adapter: the banded harmonic-transfer class on the benchmark records.

Tasks are scheduling regimes and one record is one functional observation
(Section 4.4). The target trains on one record, the 12 candidate source tasks of
the frozen design contribute their 30 records, the validation record of the
rotation makes every data-driven choice (it plays the part of the second target
sample of the adaptive methods), and the test record only scores stored
predictions. Task ``j`` enters through the normal equations ``(h_j, b_j)`` of its
records (``banded.target_stats``); ``h_j / n_j`` and ``b_j / n_j`` are its
record-averaged Gram and cross moment. The hypothesis space is the frozen class
(R, B) of the target, its norm is the class's second-difference smoothing
penalty (the regulariser of every fit), coefficients are complex in b-major
order, and FPCA is the eigendecomposition of a record-averaged Gram. Real
parameters (amplitudes, lasso coordinates) use the real and imaginary parts.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigh

from ..ltv.banded import fit_quadratic, pack_complex, target_stats
from ..ltv.harmonic import unpack
from ..ltv.pooling import band_product
from .aggregation import HeldOut, nested_sets, sparse_aggregation
from .core import amplitude, atl_source_scale, shape_combination
from .lasso import lasso_path


def dense(h):
    """Dense Hermitian matrix of a pure band Quadratic (b-major coordinates)."""
    if h.lowrank:
        raise ValueError('A projected quadratic has no complex dense Gram')
    p = h.p
    H = np.zeros((p, p), dtype=complex)
    for offset in range(h.ab.shape[0]):
        index = np.arange(p - offset)
        H[index + offset, index] = h.ab[offset, index]
        if offset:
            H[index, index + offset] = np.conj(h.ab[offset, index])
    return H


def to_packed(beta, model):
    return pack_complex(np.asarray(beta).reshape(model.B, model.d).T.reshape(-1))


def from_packed(z, model):
    return unpack(z).reshape(model.d, model.B).T.reshape(-1)


def smoothing_norm(model):
    """Gram of the class norm: second differences along b for every shift, plus 1e-12."""
    diff = np.diff(np.eye(model.B), 2, axis=0)
    return np.kron(diff.T @ diff, np.eye(model.d)) + 1e-12 * np.eye(model.p)


def canonical_phase(vectors):
    """Rotate each column so that its largest-modulus entry (the first such) is real and positive.

    A complex eigenvector is defined up to a unit factor. The Hu-Lin lasso
    penalises real and imaginary FPC coordinates separately, which depends on it.
    """
    vectors = np.array(vectors, dtype=complex)
    lead = vectors[np.argmax(np.abs(vectors), axis=0), np.arange(vectors.shape[1])]
    return vectors * (np.abs(lead) / np.where(lead == 0, 1.0, lead))[None, :]


def descending_heigh(H):
    """Eigenpairs in decreasing order, each eigenvector with its canonical phase."""
    values, vectors = eigh(H, check_finite=False)
    return values[::-1].copy(), canonical_phase(vectors[:, ::-1])


def realify(G, c):
    return np.block([[G.real, -G.imag], [G.imag, G.real]]), np.r_[c.real, c.imag]


class LTVScope:
    """One target rotation: train/validation records, test input and source tasks."""

    def __init__(self, model, smooth, train, validation, test_F, source_tasks):
        self.model, self.smooth = model, float(smooth)
        self.h0, self.b0 = target_stats(model, [train])
        self.sources = []
        for sid, records in source_tasks:
            h, b = target_stats(model, records)
            self.sources.append(dict(id=sid, h=h, b=b, n=len(records)))
        self.val_F, self.test_F = validation['F'], test_F
        self.val_Y = validation['Y'][model.rows]
        self.energy = float(np.sum(np.abs(self.val_Y) ** 2))
        # Validation risk in prediction space: sum |Y - P|^2 is a quadratic in P.
        self.val_part = HeldOut(1, np.ones(2 * len(self.val_Y)), np.r_[self.val_Y.real, self.val_Y.imag],
                                self.energy)
        self._cache = {}

    # --- shared pieces --------------------------------------------------------
    def fit(self, h, b, smooth):
        return fit_quadratic(h, b, self.model.regularizer(h, smooth))

    def predictions(self, z):
        return self.model.predict(self.val_F, z), self.model.predict(self.test_F, z)

    def loss(self, prediction):
        return float(np.sum(np.abs(self.val_Y - prediction) ** 2) / self.energy)

    def cached(self, key, build):
        if key not in self._cache:
            self._cache[key] = build()
        return self._cache[key]

    def target_dense(self):
        return self.cached('H0', lambda: dense(self.h0))

    def target_spectrum(self):
        return self.cached('target', lambda: descending_heigh(self.target_dense()))

    def pooled_spectrum(self, members):
        def build():
            total = sum(self.sources[k]['n'] for k in members)
            H = sum(dense(self.sources[k]['h']) for k in members) / total
            values, vectors = descending_heigh(H)
            cross = sum(self.sources[k]['b'] for k in members) / total
            return values, vectors, cross
        return self.cached(('pooled', tuple(members)), build)

    def aggregate(self, candidates):
        """Hu-Lin sparse aggregation of candidate predictions on the validation record."""
        val = [np.r_[v.real, v.imag] for v, _ in candidates]
        beta, info = sparse_aggregation(val, self.val_part)
        w, a, b = info['weight'], info['first'], info['second']
        test = w * candidates[a][1] + (1.0 - w) * candidates[b][1]
        valid = w * candidates[a][0] + (1.0 - w) * candidates[b][0]
        return valid, test, info

    # --- Jiao and McKeague -------------------------------------------------------
    def jm_ts(self, truncations):
        values, vectors = self.target_spectrum()
        top = min(max(truncations), self.model.p)
        basis = vectors[:, :top]
        cross0 = basis.conj().T @ self.b0
        grams = [basis.conj().T @ dense(s['h']) @ basis for s in self.sources]
        crosses = [basis.conj().T @ s['b'] for s in self.sources]
        rows = []
        for D in sorted({min(int(D), self.model.p) for D in truncations}):
            e0 = cross0[:D] / values[:D]
            estimates = [cho_solve(cho_factor(G[:D, :D], lower=True), c[:D]) for G, c in zip(grams, crosses)]
            real0 = np.r_[e0.real, e0.imag]
            misalignment = []
            for e in estimates:
                real = np.r_[e.real, e.imag]
                cosine = float(real @ real0) / max(np.linalg.norm(real) * np.linalg.norm(real0), 1e-300)
                misalignment.append(np.sqrt(max(0.0, 1.0 - cosine * cosine)))
            for informative in nested_sets(misalignment):
                coefficients = [np.r_[e0.real, e0.imag]] + [np.r_[estimates[k].real, estimates[k].imag]
                                                           for k in informative]
                shape_real = shape_combination(coefficients, [1] + [self.sources[k]['n'] for k in informative])
                shape = shape_real[:D] + 1j * shape_real[D:]
                scale = amplitude(shape_real, np.r_[(values[:D] * shape).real, (values[:D] * shape).imag],
                                  np.r_[cross0[:D].real, cross0[:D].imag])
                z = to_packed(basis[:, :D] @ (scale * shape), self.model)
                valid, test = self.predictions(z)
                rows.append(dict(truncation=D, sources=[self.sources[k]['id'] for k in informative],
                                 amplitude=scale, validation_loss=self.loss(valid), test=test))
        return rows

    # --- Lin and Reimherr ----------------------------------------------------------
    def _transfer(self, members, s1):
        H = sum(self.sources[k]['h'] * (1.0 / self.sources[k]['n']) for k in members)
        b = sum(self.sources[k]['b'] / self.sources[k]['n'] for k in members)
        return self.fit(H, b, s1)

    def _calibrate(self, start, s2):
        rhs = self.b0 - band_product(self.h0.ab, from_packed(start, self.model))
        return start + self.fit(self.h0, rhs, s2)

    def lr_candidates(self, source_smooth, target_smooth, scaled=False):
        """TL-FLR on the training record for every (lambda_1, lambda_2) and nested set.

        Sources are ranked by the class-norm distance between their own fits and
        the target-only fit (Lin and Reimherr's Algorithm 2); set 0 is target only.
        ``scaled`` gives ATL-FLR's candidates, whose transfer-step smoothing shrinks
        with the candidate's number of source records (:func:`atl_source_scale`).
        """
        def build():
            norm = smoothing_norm(self.model)
            records = lambda members: sum(self.sources[k]['n'] for k in members)
            grid = []
            for s1 in source_smooth:
                own = [from_packed(self.fit(s['h'], s['b'], s1), self.model) for s in self.sources]
                for s2 in target_smooth:
                    z0 = self.fit(self.h0, self.b0, s2)
                    beta0 = from_packed(z0, self.model)
                    distance = [float(np.real((beta0 - b).conj() @ (norm @ (beta0 - b)))) for b in own]
                    sets = nested_sets(distance)
                    candidates = [self.predictions(z0)]
                    for members in sets[1:]:
                        smooth = s1 * atl_source_scale(records(members), records(sets[1])) if scaled else s1
                        candidates.append(self.predictions(self._calibrate(self._transfer(members, smooth), s2)))
                    grid.append(dict(source_smooth=s1, target_smooth=s2, sets=sets, candidates=candidates,
                                     ranking=[self.sources[k]['id'] for k in np.argsort(distance, kind='stable')]))
            return grid
        return self.cached(('lr', scaled, tuple(source_smooth), tuple(target_smooth)), build)

    def lr_tl(self, source_smooth, target_smooth):
        """Algorithm 1 with the source set chosen among the nested sets on validation."""
        rows = []
        for entry in self.lr_candidates(source_smooth, target_smooth):
            for k, (valid, test) in enumerate(entry['candidates']):
                rows.append(dict(sources=[self.sources[j]['id'] for j in entry['sets'][k]],
                                 source_smooth=entry['source_smooth'] if k else None,
                                 target_smooth=entry['target_smooth'], validation_loss=self.loss(valid), test=test))
        return rows

    def lr_atl(self, source_smooth, target_smooth):
        """Algorithm 2: the nested candidates aggregated on the validation record."""
        rows = []
        for entry in self.lr_candidates(source_smooth, target_smooth, scaled=True):
            valid, test, info = self.aggregate(entry['candidates'])
            rows.append(dict(source_smooth=entry['source_smooth'], target_smooth=entry['target_smooth'],
                             validation_loss=self.loss(valid), test=test, ranking=entry['ranking'],
                             first=info['first'], second=info['second'], weight=info['weight']))
        return rows

    # --- Hu and Lin ---------------------------------------------------------------
    def _hl_paths(self, members, M, fractions):
        """Algorithm 1 on the train record for every lasso fraction (one exact path)."""
        values, vectors, cross = self.pooled_spectrum(members)
        basis = vectors[:, :M]
        w = (basis.conj().T @ cross) / values[:M]
        H0 = self.target_dense()
        G = basis.conj().T @ H0 @ basis
        c = basis.conj().T @ self.b0 - G @ w
        Gr, cr = realify(G, c)
        taus = np.asarray(fractions, float) * float(np.max(np.abs(cr)))
        path = lasso_path(Gr, cr, taus, ridge=1e-12 * float(np.trace(Gr)) / len(cr))
        return [to_packed(basis @ (w + d[:M] + 1j * d[M:]), self.model) for d in path]

    def _fpca_ols(self, M):
        values, vectors = self.target_spectrum()
        return to_packed(vectors[:, :M] @ ((vectors[:, :M].conj().T @ self.b0) / values[:M]), self.model)

    def hl_candidates(self, truncations, fractions):
        """Every Algorithm 1 fit on the training record, one row per (set, m, tau).

        Sources are ranked by the cross-moment discrepancy (Hu and Lin's
        Algorithm 2); set 0 is target FPCA least squares, which uses m only.
        """
        def build():
            discrepancy = [float(np.sum(np.abs(s['b'] / s['n'] - self.b0) ** 2)) for s in self.sources]
            sets = nested_sets(discrepancy)
            rows = []
            for M in sorted({min(int(M), self.model.p) for M in truncations}):
                valid, test = self.predictions(self._fpca_ols(M))
                rows.append(dict(set=0, truncation=M, lasso=None, valid=valid, test=test))
                for k, members in enumerate(sets[1:], 1):
                    for tau, z in zip(fractions, self._hl_paths(members, M, fractions)):
                        valid, test = self.predictions(z)
                        rows.append(dict(set=k, truncation=M, lasso=tau, valid=valid, test=test))
            ranking = [self.sources[j]['id'] for j in np.argsort(discrepancy, kind='stable')]
            return sets, ranking, rows
        return self.cached(('hl', tuple(truncations), tuple(fractions)), build)

    def hl_tl(self, truncations, fractions):
        """Algorithm 1 with the source set, m and tau chosen on validation."""
        sets, _, table = self.hl_candidates(truncations, fractions)
        return [dict(sources=[self.sources[j]['id'] for j in sets[r['set']]], truncation=r['truncation'],
                     lasso=r['lasso'], validation_loss=self.loss(r['valid']), test=r['test']) for r in table]

    def hl_agg(self, truncations, fractions):
        """Algorithm 2: each candidate tuned on validation, then sparse aggregation there."""
        sets, ranking, table = self.hl_candidates(truncations, fractions)
        tuned = []
        for k in range(len(sets)):
            rows = [r for r in table if r['set'] == k]
            tuned.append(min(rows, key=lambda r: self.loss(r['valid'])))
        valid, test, info = self.aggregate([(r['valid'], r['test']) for r in tuned])
        return [dict(validation_loss=self.loss(valid), test=test, ranking=ranking,
                     candidates=[dict(truncation=r['truncation'], lasso=r['lasso']) for r in tuned],
                     first=info['first'], second=info['second'], weight=info['weight'])]

"""Complex banded normal equations for the harmonic-transfer class.

Coefficients are stored b-major internally to exploit the cubic B-spline Gram
bandwidth; public packed coefficients keep the r-major ordering (real parts
then imaginary parts). The projection is a real low-rank downdate solved by
Woodbury, because the real nuisance span is not a complex-linear span.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import cholesky_banded, cho_solve_banded, solve, svd

from .harmonic import calibration_data, momentize, realvec, unpack


def pack_complex(beta):
    beta = np.asarray(beta).reshape(-1)
    return np.r_[beta.real, beta.imag]


@dataclass
class Quadratic:
    """Hermitian complex band minus L.T @ L in its real representation.

    ``lowrank`` rows use real/imag b-major ordering. ``ab`` follows SciPy's
    lower Hermitian-band convention, ab[i-j,j] == H[i,j] for i >= j.
    """
    ab: np.ndarray
    B: int
    d: int
    lowrank: tuple = ()

    @property
    def p(self):
        return self.B * self.d

    def copy(self):
        return Quadratic(self.ab.copy(), self.B, self.d,
                         tuple(x.copy() for x in self.lowrank))

    def __add__(self, other):
        if np.isscalar(other) and other == 0:
            return self.copy()
        if not isinstance(other, Quadratic):
            return NotImplemented
        if (self.B, self.d, self.ab.shape) != (other.B, other.d, other.ab.shape):
            raise ValueError('Cannot mix harmonic parameter spaces')
        return Quadratic(self.ab + other.ab, self.B, self.d,
                         self.lowrank + other.lowrank)

    __radd__ = __add__

    def __mul__(self, weight):
        weight = float(weight)
        if weight < 0:
            raise ValueError('Quadratic weights must be nonnegative')
        return Quadratic(self.ab * weight, self.B, self.d,
                         tuple(np.sqrt(weight) * x for x in self.lowrank)
                         if weight else ())

    __rmul__ = __mul__

    def trace(self):
        # A real projection downdate has no unique complex-linear Gram. Its
        # real trace divided by two gives the compatible regularizer scale.
        return float(self.ab[0].real.sum() -
                     sum(float(np.sum(x*x))/2 for x in self.lowrank))


class Harmonic:
    """(A_beta F)_k = sum_{r,b} beta[r,b] B_b(omega_k) F[k-r] on the output band.

    ``R`` neighbouring harmonic shifts on each side, ``B`` cubic B-spline basis
    functions on [lo, hi] Hz; the model output is i*omega*A F (the measured
    output is the time derivative of the state).
    """
    def __init__(self, n, fs, R, B, lo=500., hi=20000., wc=2*np.pi*7500.):
        self.n, self.fs, self.R, self.B, self.wc = n, fs, R, B, wc
        self.freq = np.fft.rfftfreq(n, 1/fs)
        self.omega = 2*np.pi*self.freq
        self.rows = np.flatnonzero((self.freq >= lo) & (self.freq <= hi))
        if not len(self.rows) or self.rows[0] <= R+3 or self.rows[-1]+R+3 >= len(self.freq)-1:
            raise ValueError('Required dictionary/transfer halo must avoid DC and Nyquist')
        self.shifts = np.arange(-R, R+1)
        self.indices = self.rows[:, None] - self.shifts[None, :]
        knots = np.r_[[lo]*3, np.linspace(lo, hi, B-2), [hi]*3]
        self.phi = BSpline.design_matrix(self.freq[self.rows], knots, 3).toarray()
        self.factor = 1j*self.omega[self.rows]/wc
        self.d = 2*R+1
        self.p = self.d*B
        self.bandwidth = min(4*self.d-1, self.p-1)
        self._supports = {(b, c): np.flatnonzero(self.phi[:, b]*self.phi[:, c])
                          for b in range(B) for c in range(max(0, b-3), b+1)}

    def predict(self, F, z):
        beta = unpack(z).reshape(self.d, self.B)
        bands = self.phi @ beta.T
        return self.factor * np.einsum('kr,kr->k', bands, np.asarray(F)[self.indices], optimize=True)

    def apply_fields(self, fields, z):
        beta = unpack(z).reshape(self.d, self.B)
        bands = self.phi @ beta.T
        return self.factor[:, None]*np.einsum('kr,krm->km', bands, fields[self.indices, :], optimize=True)

    def regularizer(self, h, smooth):
        """smooth * trace(H)/p * second-difference penalty + 1e-12 * trace(H)/p * I."""
        scale = h.trace()/h.p
        ab = np.zeros_like(h.ab)
        diff = np.diff(np.eye(self.B), 2, axis=0)
        penalty = diff.T @ diff
        for off in range(min(2, self.B-1)+1):
            for b in range(self.B-off):
                j = b*self.d + np.arange(self.d)
                ab[off*self.d, j] = scale*smooth*penalty[b+off, b]
        ab[0] += scale*1e-12
        return Quadratic(ab, self.B, self.d)


def _structured_stats(model, values, outcomes):
    """Gram for X[k,mode,b,r] = phi[k,b] * values[k,mode,r]."""
    values = np.asarray(values)
    outcomes = np.asarray(outcomes)
    if values.ndim != 3 or outcomes.shape != values.shape[:2]:
        raise ValueError('Expected rows x modes x shifts and rows x modes')
    ab = np.zeros((model.bandwidth+1, model.p), dtype=complex)
    rhs = np.empty((model.B, model.d), dtype=complex)
    row_idx, col_idx = np.indices((model.d, model.d))
    inner_rhs = np.einsum('kmr,km->kr', values.conj(), outcomes, optimize=True)
    rhs[:] = model.phi.T @ inner_rhs
    for (b, c), support in model._supports.items():
        if not len(support):
            continue
        weight = np.sqrt(model.phi[support, b]*model.phi[support, c])
        local = (values[support] * weight[:, None, None]).reshape(-1, model.d)
        block = local.conj().T @ local
        delta = (b-c)*model.d + row_idx-col_idx
        columns = c*model.d + col_idx
        keep = delta >= 0
        ab[delta[keep], columns[keep]] = block[keep]
    ab[0] = ab[0].real
    return Quadratic(ab, model.B, model.d), rhs.reshape(-1)


def target_stats(model, records):
    """Normal equations of the least-squares fit to one or more records."""
    values = np.stack([model.factor[:, None]*r['F'][model.indices] for r in records], axis=1)
    outcomes = np.stack([r['Y'][model.rows] for r in records], axis=1)
    return _structured_stats(model, values, outcomes)


def fit_quadratic(h, b, regularizer):
    """Solve the regularised normal equations: banded Cholesky plus Woodbury."""
    a = h + regularizer
    factor = cholesky_banded(a.ab, lower=True, check_finite=False)
    solution = cho_solve_banded((factor, True), b, check_finite=False)
    if a.lowrank:
        lowrank = np.concatenate(a.lowrank, axis=0)
        real_rhs = np.r_[solution.real, solution.imag]
        complex_columns = lowrank[:, :a.p].T + 1j*lowrank[:, a.p:].T
        columns = cho_solve_banded((factor, True), complex_columns, check_finite=False)
        real_columns = np.r_[columns.real, columns.imag]
        middle = np.eye(len(lowrank)) - lowrank @ real_columns
        middle = (middle+middle.T)/2
        adjustment = solve(middle, lowrank @ real_rhs, assume_a='pos', check_finite=False)
        solution += columns @ adjustment
    beta_parent = solution.reshape(a.B, a.d).T.reshape(-1)
    return pack_complex(beta_parent)


def _projection_map(model, values, Q):
    """Q.T @ realmap(X) without materialising the large real design matrix."""
    nmodes = values.shape[1]
    m = len(model.rows)*nmodes
    q = (Q[:m] + 1j*Q[m:]).reshape(len(model.rows), nmodes, -1)
    local = np.einsum('kmq,kmr->kqr', q.conj(), values, optimize=True)
    cross = np.einsum('kb,kqr->qbr', model.phi, local, optimize=True).reshape(Q.shape[1], model.p)
    return np.c_[cross.real, -cross.imag]


def source_stats(model, records, z, base_atoms, theta):
    """Whitened source moment quadratics after calibration: Unprojected and Projected.

    The projection removes the real span of the calibrated active generators
    A_z D_l (frozen at the pilot z) from the source moments.
    """
    data = calibration_data(model, records, z, base_atoms)
    corrected = [r['F']-D@theta for r, D in zip(records, data['atoms'])]
    raw = np.stack([model.factor[:, None]*f[model.indices] for f in corrected])
    values = np.einsum('nkr,kna->kar', raw, data['transform'], optimize=True)
    y = momentize(np.stack([r['Y'][model.rows] for r in records]), data['transform'])
    h, b = _structured_stats(model, values, y.reshape(values.shape[:2]))
    active = np.flatnonzero(np.abs(theta) > 1e-12*max(np.max(np.abs(theta)), 1e-30))
    g = data['G'][:, active]
    gr = np.r_[g.real, g.imag]
    if gr.shape[1]:
        u, s, _ = svd(gr, full_matrices=False, check_finite=False)
        Q = u[:, s > 1e-10*s[0]]
        qa = _projection_map(model, values, Q)
        qy = Q.T @ realvec(y)
        correction = qa.T @ qy
        hp = Quadratic(h.ab.copy(), h.B, h.d, (qa,))
        bp = b-correction[:model.p]-1j*correction[model.p:]
    else:
        hp, bp = h.copy(), b.copy()
        Q = np.zeros((2*len(y), 0))
    return dict(Unprojected=(h, b), Projected=(hp, bp), n=len(records),
                active_atoms=[data['names'][i] for i in active], projection_rank=Q.shape[1])


def forward_select(h, b, reg, pilot, source_quadratics, scales, score, margin=.02, max_sources=4):
    """Deterministic greedy source selection on the validation score.

    Every forward step and the final choice must improve the validation loss by
    at least ``margin``; one global source scale per candidate set, retuned at
    every step; ties favour smaller scale, then source id.
    """
    current = float(score(pilot))
    selected, best_z, alpha = [], pilot.copy(), 0.
    curve = [dict(step=0, proposed_source=None, sources=[], scale=0., validation_loss=current, accepted=True)]
    for step in range(1, max_sources+1):
        choices = []
        for sid in sorted(source_quadratics):
            if sid in selected:
                continue
            trial = selected+[sid]
            sh = sum(source_quadratics[s][0]*source_quadratics[s][2] for s in trial)
            sb = sum(source_quadratics[s][1]*source_quadratics[s][2] for s in trial)
            for weight in scales:
                z = pilot if weight == 0 else fit_quadratic(h+weight*sh, b+weight*sb, reg)
                loss = float(score(z))
                index = len(curve)
                curve.append(dict(step=step, proposed_source=sid, sources=trial,
                                  scale=weight, validation_loss=loss, accepted=False))
                choices.append((loss, weight, sid, index, z))
        if not choices:
            break
        loss, weight, sid, index, z = min(choices, key=lambda c: (c[0], c[1], c[2]))
        if weight <= 0 or loss > current*(1-margin):
            break
        curve[index]['accepted'] = True
        selected.append(sid)
        current, alpha, best_z = loss, weight, z.copy()
    return dict(sources=selected, scale=alpha, validation_loss=current, z=best_z, curves=curve)

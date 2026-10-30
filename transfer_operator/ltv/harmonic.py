"""Harmonic-transfer conventions, perturbation dictionary, calibration and scoring.

All routines are data-agnostic. The operator class couples each output
frequency to ``2R+1`` neighbouring input harmonic shifts with cubic B-spline
frequency dependence (see :class:`banded.Harmonic`). States are scaled by
``wc`` (``V = wc * W``) and derivatives by ``1/wc`` for conditioning; a
calibrated coefficient of the atom ``name:d`` converts to physical units as
``theta_physical = theta_scaled * wc**(1-d)``.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve

# Local instruments span the model's shifts plus this halo of neighbouring lines;
# their empirical covariance is whitened with this relative ridge.
INSTRUMENT_HALO = 3
WHITENING_RIDGE_RELATIVE = 1e-4


def realvec(a):
    a = np.asarray(a).reshape(-1)
    return np.r_[a.real, a.imag]


def realmap(a):
    return np.block([[a.real, -a.imag], [a.imag, a.real]])


def unpack(z):
    p = len(z)//2
    return z[:p]+1j*z[p:]


def metrics(pred, Y, forced_mask):
    """Normalised squared output error on the full band, forced lines, sidebands."""
    out = {}
    for name, mask in [('full_band', np.ones(len(Y), bool)), ('forced_lines', forced_mask),
                       ('sideband_only', ~forced_mask)]:
        den = float(np.sum(np.abs(Y[mask])**2))
        num = float(np.sum(np.abs(Y[mask]-pred[mask])**2))
        out[name] = num/max(den, 1e-300)
        out[name+'_sse'] = num
        out[name+'_energy'] = den
        out[name+'_bins'] = int(mask.sum())
    return out


def dictionary(V, omega, base_atoms, wc):
    """Perturbation atoms q_h(t) * d^d V/dt^d in the Fourier domain.

    ``base_atoms`` are 'dc', 'cos<h>', 'sin<h>' (scheduling harmonics); each is
    applied after derivative order d = 0, 1, 2, giving at most 21 atoms named
    ``<atom>:d<d>``. Only low positive interior lines are used downstream, so
    zero padding at the spectrum boundaries cannot influence a model row.
    """
    columns, names = [], []

    def shift(a, h):
        out = np.zeros_like(a)
        if h > 0:
            out[h:] = a[:-h]
        elif h < 0:
            out[:h] = a[-h:]
        else:
            out = a.copy()
        return out
    for d in (0, 1, 2):
        v = V*(1j*omega/wc)**d
        for name in base_atoms:
            if name == 'dc':
                a = v
            else:
                h = int(name[-1])
                minus, plus = shift(v, h), shift(v, -h)
                a = (minus+plus)/2 if name.startswith('cos') else (minus-plus)/(2j)
            columns.append(a)
            names.append(f'{name}:d{d}')
    return np.stack(columns, axis=1), names


def instrument_transform(model, records, eta_rel=WHITENING_RIDGE_RELATIVE):
    """Reduced whitening of the local input instruments; rows x records x modes.

    For n records and d local input lines the moment is R^T conj(U)
    diag(sqrt(lambda/(lambda+eta)))/sqrt(n), where U, lambda diagonalise
    FF^H/n: equal to the full ridge-whitened moments up to a right unitary map,
    so norms, calibration and projection are preserved.
    """
    offsets = np.arange(-model.R-INSTRUMENT_HALO, model.R+INSTRUMENT_HALO+1)
    idx = model.rows[:, None]+offsets[None, :]
    f = np.stack([r['F'][idx] for r in records], axis=1)
    n = len(records)
    g = np.einsum('knd,kmd->knm', f, f.conj(), optimize=True)/n
    ev, u = np.linalg.eigh(g)
    ev = np.maximum(ev, 0)
    eta = eta_rel*np.sum(ev, axis=1)/len(offsets)
    weights = np.sqrt(ev/np.maximum(ev+eta[:, None], 1e-300))/np.sqrt(n)
    return u.conj()*weights[:, None, :]


def momentize(data, transform):
    """data: records x rows [x features]; result: rows*modes [x features]."""
    if data.ndim == 2:
        return np.einsum('nk,kna->ka', data, transform, optimize=True).reshape(-1)
    out = np.einsum('nkp,kna->kap', data, transform, optimize=True)
    return out.reshape(-1, data.shape[-1])


def calibration_data(model, records, z, base_atoms):
    """Whitened source moments of the generators A_z D_l, the pilot residual and the output."""
    transform = instrument_transform(model, records)
    generators, base, ys, atoms = [], [], [], []
    for r in records:
        D, names = dictionary(r['V'], model.omega, base_atoms, model.wc)
        atoms.append(D)
        generators.append(model.apply_fields(D, z))
        base.append(model.predict(r['F'], z)-r['Y'][model.rows])
        ys.append(r['Y'][model.rows])
    G = momentize(np.stack(generators), transform)
    b = momentize(np.stack(base), transform)
    y = momentize(np.stack(ys), transform)
    return dict(G=G, b=b, y=y, transform=transform, atoms=atoms, names=names)


def elastic_calibrate(data, l1_rel, ridge=1e-6):
    """Real normalised-column elastic net: l1 relative to lambda_max, ridge 1e-6."""
    A = np.r_[data['G'].real, data['G'].imag]
    b = realvec(data['b'])
    norm = np.maximum(np.linalg.norm(A, axis=0), 1e-30)
    a = A/norm
    H = a.T@a+ridge*np.eye(a.shape[1])
    c = a.T@b
    lam = l1_rel*float(np.max(np.abs(c)))
    if lam == 0:
        return solve(H, c, assume_a='pos')/norm
    x = np.zeros(len(c))
    for _ in range(3000):
        old = x.copy()
        for j in range(len(c)):
            q = c[j]-H[j]@x+H[j, j]*x[j]
            x[j] = np.sign(q)*max(abs(q)-lam, 0.)/H[j, j]
        if np.max(np.abs(x-old)) < 1e-10*max(1e-12, np.max(np.abs(x))):
            break
    return x/norm


def calibration_residual(data, theta):
    den = max(float(np.linalg.norm(data['y'])), 1e-300)
    zero = float(np.linalg.norm(data['b'])/den)
    residual = float(np.linalg.norm(data['b']-data['G']@theta)/den)
    return dict(residual=residual, theta_zero_residual=zero,
                improvement_factor=zero/max(residual, 1e-300))

"""Periodic advection-diffusion model of Section 4.3 and Appendix D.3."""
from typing import Protocol, runtime_checkable
import numpy as np

SQRT2 = np.sqrt(2.0)

@runtime_checkable
class PDEModel(Protocol):
    """Finite operator coordinates and their real Frobenius adjoint.

    ``apply_adjoint(X, Z)`` is the adjoint of the parameter-to-output map
    ``h -> apply(h, X)``, not the state-space adjoint of a fixed operator.
    ``derivative(X,l)`` applies the l-th PDE dictionary operator.
    ``map_hessian(X)`` must return the diagonal of the parameter-map Hessian;
    general dense parameterizations require a separate numerical backend.
    """
    family: str
    d: int
    p: int
    s: float
    q: np.ndarray
    truth: np.ndarray
    thetas: np.ndarray

    def apply(self, h, X): ...
    def apply_adjoint(self, X, Z): ...
    def derivative(self, X, l): ...
    def target_terms(self, stats): ...
    def map_hessian(self, X): ...
    def source_h(self, theta): ...
    def risk(self, h): ...

class AdvectionDiffusionModel:
    family = "advection_diffusion"
    p = 2

    def __init__(self, R=12, s=2.0, tau=0.25, sigma=1.0):
        self.R, self.s, self.tau, self.sigma = int(R), float(s), float(tau), float(sigma)
        if self.R < 1 or self.s <= 1:
            raise ValueError("Require R >= 1 and s > 1")
        # Shell ordering makes the mode set nested when R changes.
        modes = [(a, b) for a in range(-self.R, self.R + 1)
                 for b in range(-self.R, self.R + 1)
                 if a > 0 or (a == 0 and b > 0)]
        modes.sort(key=lambda k: (max(abs(k[0]), abs(k[1])), k[0], k[1]))
        self.pairs = np.asarray(modes, dtype=int)
        self.d = 1 + 2 * len(modes)
        self.cos = np.arange(1, self.d, 2)
        self.sin = np.arange(2, self.d, 2)
        self.ell_pairs = 1.0 + np.sum(self.pairs ** 2, axis=1)
        self.ell = np.r_[1.0, np.repeat(self.ell_pairs, 2)]
        self.q = self.ell ** (-self.s)
        self.sqrt_q = np.sqrt(self.q)
        self.noise_envelope = self.sigma * self.ell ** (-(self.s + 3.0) / 2.0)
        self.beta = self.s / 2.0
        self.thetas = np.array([[self.tau, 0.0], [0.0, self.tau]])
        self.truth = self.source_h(np.zeros(2))

    def h_from_symbols(self, symbols, constant=1.0):
        """Convert one complex symbol per representative pair to real H coordinates.

        An array with 1+number_of_pairs entries may include the constant first.
        """
        symbols = np.asarray(symbols)
        if symbols.size == len(self.pairs) + 1:
            constant, symbols = symbols[0], symbols[1:]
        if symbols.size != len(self.pairs):
            raise ValueError("Incorrect number of Fourier pair symbols")
        h = np.empty(self.d, dtype=float)
        h[0] = float(np.real(constant))
        h[self.cos] = SQRT2 * np.real(symbols)
        h[self.sin] = SQRT2 * np.imag(symbols)
        return h

    def symbols_from_h(self, h):
        h = np.asarray(h)
        return h[self.cos] / SQRT2 + 1j * h[self.sin] / SQRT2

    def source_h(self, theta):
        theta = np.asarray(theta, dtype=float)
        return self.h_from_symbols(1.0 / (self.ell_pairs + 1j * (self.pairs @ theta)))

    def apply(self, h, X):
        """Apply multiplier h to vectors (d,) or column batches (d,r)."""
        X, h = np.asarray(X), np.asarray(h)
        out = np.empty_like(X, dtype=float)
        x, y = h[self.cos] / SQRT2, h[self.sin] / SQRT2
        if X.ndim == 2:
            x, y = x[:, None], y[:, None]
        out[0] = h[0] * X[0]
        out[self.cos] = x * X[self.cos] + y * X[self.sin]
        out[self.sin] = -y * X[self.cos] + x * X[self.sin]
        return out

    def derivative(self, X, l):
        """Apply partial_l, with zero-based l in {0,1}."""
        X = np.asarray(X)
        out = np.empty_like(X, dtype=float)
        k = self.pairs[:, int(l)]
        if X.ndim == 2:
            k = k[:, None]
        out[0] = 0.0
        out[self.cos] = k * X[self.sin]
        out[self.sin] = -k * X[self.cos]
        return out

    def apply_adjoint(self, X, Z):
        """Adjoint of h -> apply(h,X), for the real Frobenius product."""
        X, Z = np.asarray(X), np.asarray(Z)
        if X.ndim == 1:
            X, Z = X[:, None], Z[:, None]
        h = np.empty(self.d)
        h[0] = np.dot(X[0], Z[0])
        h[self.cos] = (np.einsum('ij,ij->i', X[self.cos], Z[self.cos])
                       + np.einsum('ij,ij->i', X[self.sin], Z[self.sin])) / SQRT2
        h[self.sin] = (np.einsum('ij,ij->i', X[self.sin], Z[self.cos])
                       - np.einsum('ij,ij->i', X[self.cos], Z[self.sin])) / SQRT2
        return h

    def map_hessian(self, X):
        """Diagonal of T_X^* T_X; cross terms cancel within every real pair."""
        row_norms = np.einsum('ij,ij->i', X, X)
        return np.r_[row_norms[0], np.repeat(
            (row_norms[self.cos] + row_norms[self.sin]) / 2.0, 2)]

    def target_terms(self, stats):
        Cff, Cuf = stats.Cff, stats.Cuf
        diagonal = np.r_[Cff[0, 0], np.repeat(
            (Cff[self.cos, self.cos] + Cff[self.sin, self.sin]) / 2.0, 2)]
        b = np.empty(self.d)
        b[0] = Cuf[0, 0]
        b[self.cos] = (Cuf[self.cos, self.cos] + Cuf[self.sin, self.sin]) / SQRT2
        b[self.sin] = (Cuf[self.cos, self.sin] - Cuf[self.sin, self.cos]) / SQRT2
        return diagonal, b

    def risk(self, h):
        err = np.asarray(h) - self.truth
        return float(np.dot(self.q, err * err))


FourierModel = AdvectionDiffusionModel

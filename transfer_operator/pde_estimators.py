"""Advection calibration by active/sign enumeration, projection and low-rank refits."""
from dataclasses import dataclass
from itertools import combinations, product
import numpy as np
from scipy.linalg import eigh, solve

@dataclass
class SourceMoments:
    X: np.ndarray
    Y: np.ndarray
    DY: np.ndarray
    eta: float
    n: int
    W: np.ndarray = None
    covariance_eigenvalues: np.ndarray = None

    def corrected_X(self, theta):
        return self.X - np.einsum('l,lij->ij', theta, self.DY)

    def moment(self, model, h, theta):
        return self.Y - model.apply(h, self.corrected_X(theta))

@dataclass
class Calibration:
    theta: np.ndarray
    support: tuple
    gram: object
    kkt: float
    objective: float
    restricted_min_eigenvalue: float
    cross: object = None

@dataclass
class Projector:
    """Q[rank,d,d] is an orthonormal basis in real Frobenius geometry."""
    Q: np.ndarray

    @property
    def rank(self):
        return self.Q.shape[0]

    def coordinates(self, matrix):
        return np.einsum('rij,ij->r', self.Q, matrix)

    def project(self, matrix):
        return np.einsum('r,rij->ij', self.coordinates(matrix), self.Q)

    def remove(self, matrix):
        return matrix - self.project(matrix)

    def change(self, other):
        if self.rank != other.rank:
            return 1.0
        if self.rank == 0:
            return 0.0
        new = self.Q.reshape(self.rank, -1)
        old = other.Q.reshape(other.rank, -1)
        residual = new - (new @ old.T) @ old
        # sin(theta_max) evaluated from the orthogonal residual avoids the
        # sqrt(machine-epsilon) floor of sqrt(1-cos(theta_min)^2).
        square = float(np.linalg.eigvalsh(residual @ residual.T)[-1])
        return float(min(1.0, np.sqrt(max(0.0, square))))



def lasso_from_quadratic(gram, c, constant, penalty=0.0, support=None):
    """Exact active/sign enumeration, with the PDF's no-1/2 objective.

Explicit ``support`` is used for restricted recalibration on a previously
Lasso-selected support; it is never inferred from a true coefficient here.
"""
    gram, c = np.asarray(gram), np.asarray(c)
    p = len(c)
    if support is not None:
        active = tuple(int(l) for l in support)
        theta = np.zeros(p)
        if active:
            ix = np.asarray(active)
            theta[ix] = np.linalg.lstsq(gram[np.ix_(ix, ix)], -c[ix], rcond=1e-12)[0]
            kkt = float(np.max(np.abs(2 * (gram @ theta + c)[ix])))
        else:
            kkt = 0.0
        objective = constant + 2 * c @ theta + theta @ gram @ theta
    else:
        penalty = float(penalty)
        if penalty < 0:
            raise ValueError("Lasso penalty must be nonnegative")
        candidates = []
        for count in range(p + 1):
            for candidate in combinations(range(p), count):
                ix = np.asarray(candidate, dtype=int)
                for signs in product((-1.0, 1.0), repeat=count):
                    theta_candidate = np.zeros(p)
                    if count:
                        ss = np.asarray(signs)
                        theta_candidate[ix] = np.linalg.lstsq(
                            gram[np.ix_(ix, ix)], -c[ix] - 0.5 * penalty * ss,
                            rcond=1e-12)[0]
                        if np.any(theta_candidate[ix] * ss < -1e-12):
                            continue
                    gradient = 2.0 * (gram @ theta_candidate + c)
                    actual = np.flatnonzero(np.abs(theta_candidate) > 1e-12)
                    inactive = np.flatnonzero(np.abs(theta_candidate) <= 1e-12)
                    violation = max(
                        float(np.max(np.abs(gradient[actual] + penalty * np.sign(theta_candidate[actual])))) if len(actual) else 0.0,
                        float(np.max(np.maximum(np.abs(gradient[inactive]) - penalty, 0.0))) if len(inactive) else 0.0)
                    objective_candidate = (constant + 2 * c @ theta_candidate
                                           + theta_candidate @ gram @ theta_candidate
                                           + penalty * np.sum(np.abs(theta_candidate)))
                    candidates.append((violation, objective_candidate, theta_candidate))
        tolerance = 1e-9 * (1.0 + np.max(np.abs(c)) + penalty)
        feasible = [row for row in candidates if row[0] <= tolerance]
        selected = min(feasible, key=lambda row: row[1]) if feasible else min(candidates, key=lambda row: (row[0], row[1]))
        kkt, objective, theta = selected
        theta[np.abs(theta) <= 1e-12] = 0.0
        active = tuple(map(int, np.flatnonzero(theta)))
    minimum = float(np.linalg.eigvalsh(gram[np.ix_(active, active)])[0]) if active else np.nan
    return Calibration(theta, active, gram, float(kkt), float(objective), minimum)



def prepare_source(model, stats, eta, keep_whitener=False, eigencache=None):
    """Full same-sample regularized whitening, including off-diagonal moments."""
    if eta <= 0:
        raise ValueError("eta must be positive")
    if eigencache is None:
        values, vectors = eigh(stats.Cff, check_finite=False, driver='evr')
    else:
        values, vectors = eigencache
    values = np.maximum(values, 0.0)
    W = (vectors * (values + eta) ** -0.5) @ vectors.T
    X, Y = stats.Cff @ W, stats.Cuf @ W
    DY = np.stack([model.derivative(Y, l) for l in range(model.p)])
    return SourceMoments(X, Y, DY, float(eta), stats.n,
                         W if keep_whitener else None, values)


def ridge_fit(model, stats, ridge):
    diagonal, b = model.target_terms(stats)
    return b / (diagonal + float(ridge))


def pooled_fit(model, stats_list, ridge):
    """Sample-weighted target/source pooling, with registered reduction order."""
    n = sum(stats.n for stats in stats_list)
    diagonal, b = np.zeros(model.d), np.zeros(model.d)
    for stats in stats_list:
        d, bb = model.target_terms(stats)
        diagonal += stats.n * d / n
        b += stats.n * bb / n
    return b / (diagonal + float(ridge))


def calibration(model, source, h, penalty=0.0, support=None):
    """Sparse initial calibration or LS restricted to a fixed explicit support.

    The loss is ||M||² + penalty ||theta||₁, with no hidden factor 1/2.
    A restricted zero coefficient retains its selected support. A zero-rank
    problem returns the minimum-norm coefficient zero.
    """
    if penalty < 0:
        raise ValueError("Lasso penalty must be nonnegative")
    if support is not None and (len(set(support)) != len(support) or
                               any(int(l) != l or l < 0 or l >= model.p for l in support)):
        raise ValueError("support must contain distinct dictionary indices")
    base = source.Y - model.apply(h, source.X)
    generators = np.stack([model.apply(h, source.DY[l]) for l in range(model.p)])
    flat = generators.reshape(model.p, -1)
    gram, c = flat @ flat.T, flat @ base.ravel()
    result = lasso_from_quadratic(gram, c, float(np.sum(base * base)), penalty, support)
    result.cross = c
    return result


def source_projector(model, source, h, support):
    """Rebuild the Frobenius projector at the current operator h."""
    if len(support) == 0:
        return Projector(np.empty((0, model.d, model.d)))
    generators = np.column_stack([
        model.apply(h, source.DY[int(l)]).ravel() for l in support])
    Q, R = np.linalg.qr(generators, mode='reduced')
    U, values, _ = np.linalg.svd(R, full_matrices=False)
    rank = int(np.sum(values > (1e-12 * values[0]))) if len(values) and values[0] > 0 else 0
    basis = (Q @ U[:, :rank]).T.reshape(rank, model.d, model.d)
    return Projector(basis)


def refit_system(model, target, sources, thetas, projectors, rho, ridge):
    """Return D,V,b for [diag(D) - VV.T] h = b."""
    D, b = model.target_terms(target)
    D = D + float(ridge)
    b = b.copy()
    lowrank = []
    for source, theta, projector, weight in zip(sources, thetas, projectors, rho):
        X = source.corrected_X(theta)
        D += weight * model.map_hessian(X)
        b += weight * model.apply_adjoint(X, source.Y)
        if projector is None:
            continue
        for Q in projector.Q:
            v = model.apply_adjoint(X, Q)
            coefficient = np.sum(Q * source.Y)
            lowrank.append(np.sqrt(weight) * v)
            b -= weight * coefficient * v
    V = np.column_stack(lowrank) if lowrank else np.empty((model.d, 0))
    return D, V, b


def refit(model, target, sources, thetas, projectors, rho, ridge, dense=False):
    """Structured Woodbury refit; return (operator, numerical diagnostics).

    Advection retains its registered Schur threshold 1e-11 for dense fallback.
    """
    D, V, b = refit_system(model, target, sources, thetas, projectors, rho, ridge)
    if np.any(D <= 0):
        raise np.linalg.LinAlgError("Nonpositive diagonal in refit")
    schur = np.eye(V.shape[1]) - V.T @ (V / D[:, None])
    minimum = float(np.linalg.eigvalsh(schur)[0]) if V.shape[1] else 1.0
    use_dense = bool(dense or minimum <= 1e-11)
    if use_dense:
        h = solve(np.diag(D) - V @ V.T, b, assume_a='pos', check_finite=False)
    else:
        h = b / D
        if V.shape[1]:
            h += (V / D[:, None]) @ solve(schur, V.T @ h, assume_a='pos', check_finite=False)
    residual = D * h - V @ (V.T @ h) - b
    info = {"solve_residual": float(np.linalg.norm(residual) / max(np.linalg.norm(b), 1e-30)),
            "schur_min_eigenvalue": minimum, "projector_rank": int(V.shape[1]),
            "used_dense": use_dense, "diagonal_min": float(D.min())}
    return h, info

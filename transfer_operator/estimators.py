"""Four independently tuned estimators and Section 2.3 fixed-support iteration.

Pilot: target ridge. Pooling: shared-operator ridge. Unprojected: calibrated
source moment refit. Projected: remove the pilot-selected generator span before
refitting. The projection-off ablation uses Projected's candidate unchanged.
The cached refits and independent dense validator retain the original arithmetic.
"""
from __future__ import annotations
from dataclasses import dataclass
import copy, itertools, hashlib, json
import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigh
from .model import BlockModel

PRIMARY=("pilot","pooling","unprojected","projected")
RANK_RTOL=1e-10

@dataclass
class Candidate:
    lam0_frac: float
    lambar_frac: float
    eta_rule: str
    lamj_frac: float           # +inf = no transfer (pilot only)
    idx: int = -1

    def key(self):
        return (self.lam0_frac, self.lambar_frac, self.eta_rule, self.lamj_frac)

    @property
    def no_transfer(self):
        return not np.isfinite(self.lamj_frac)


def lam_max(b) -> float:
    """lambda_{j,max} = 2 max_l |b_l|: the exact zero-solution threshold of ||M_hat(A~, theta)||^2 + lambda ||theta||_1."""
    b = np.asarray(b, float)
    return float(2.0 * np.abs(b).max()) if b.size else 0.0


def eta_from_rule(rule: str, lambar: float) -> float:
    return {"eq": lambar, "x4": 4 * lambar, "d4": lambar / 4}[rule]


def candidate_record(candidate):
    return dict(index=int(candidate.idx), lam0_frac=float(candidate.lam0_frac),
                lambar_frac=float(candidate.lambar_frac), eta_rule=candidate.eta_rule,
                lamj_frac=None if candidate.no_transfer else float(candidate.lamj_frac),
                no_transfer=bool(candidate.no_transfer))

def candidate_object(record):
    candidate=Candidate(float(record["lam0_frac"]),float(record["lambar_frac"]),
        str(record["eta_rule"]),np.inf if record["lamj_frac"] is None else float(record["lamj_frac"]),int(record["index"]))
    if candidate_record(candidate) != record:
        raise ValueError("Candidate serialization mismatch")
    return candidate

def candidate_registry(design):
    """Exact stable Cartesian IDs, including target-only transfer exits.

    All numerical grid values and optional expected hashes belong to the
    supplied experiment design, not to the estimator package.
    """
    g=design["grid"]
    transfer=[Candidate(a,b,e,c) for a,b,e,c in itertools.product(
        g["lam0_frac"],g["lambar_frac"],g["eta_rule"],g["lamj_frac"])]
    transfer += [Candidate(a,0.,"eq",np.inf) for a in g["lam0_frac"]]
    candidates=dict(pilot=[Candidate(a,0.,"eq",np.inf) for a in g["lam0_frac"]],
        pooling=[Candidate(g["lam0_frac"][0],b,"eq",np.inf) for b in g["lambar_frac"]],
        unprojected=transfer,projected=transfer)
    registry={}
    for method,rows in candidates.items():
        for i,candidate in enumerate(rows): candidate.idx=i
        registry[method]=[candidate_record(c) for c in rows]
    digest=hashlib.sha256(json.dumps(registry,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    if "candidate_registry_sha256" in design and digest!=design["candidate_registry_sha256"]:
        raise ValueError("Candidate grid/ID checksum mismatch")
    return registry

def lasso_cd(Gram: np.ndarray, b: np.ndarray, lam: float, tol: float = 1e-12, maxit: int = 100_000,
             theta0=None) -> np.ndarray:
    """argmin_theta  theta^T Gram theta + 2 b^T theta + lam ||theta||_1   (exact CD, real theta)."""
    p = b.size
    theta = np.zeros(p) if theta0 is None else np.array(theta0, float)
    diag = np.diag(Gram).copy()
    diag[diag <= 0] = np.inf
    half = 0.5 * lam
    for _ in range(maxit):
        maxdelta = 0.0
        for l in range(p):
            r = b[l] + Gram[l] @ theta - Gram[l, l] * theta[l]
            new = -np.sign(r) * max(abs(r) - half, 0.0) / diag[l] if np.isfinite(diag[l]) else 0.0
            maxdelta = max(maxdelta, abs(new - theta[l]))
            theta[l] = new
        if maxdelta < tol * max(1.0, np.abs(theta).max()):
            break
    return theta


class BlockView:
    def __init__(self, model: BlockModel, j: int, st: dict, eta: float, rho: float, whitening: str = "empirical", *, rank_rtol=RANK_RTOL):
        self.model, self.j, self.eta, self.rho = model, j, eta, float(rho)
        self.rank_rtol = float(rank_rtol)
        Sf = st["Sf"]
        if whitening == "empirical":
            w, V = eigh(Sf)
            W = (V * (np.clip(w, 0, None) + eta) ** -0.5) @ V.T
        else:
            W = np.diag((model.nu + eta) ** -0.5)
        self.W = W
        self.WS = W @ Sf                              # (K,K)
        self.Wyf = W @ st["yf"]                       # (K,)
        self.WSLh = (self.WS * model.mu[None, :]) @ model.h.T   # (K, p): W S_f Lambda h_l
        self.blockF2 = float(np.sum(((Sf * model.mu[:, None]) @ W) ** 2))   # ||Lambda S_f W||_F^2

    def xrow(self, c, a, theta):
        return self.Wyf - self.WS @ a - c * (self.WSLh @ np.asarray(theta, float))

    def moment_norm2(self, c, a, theta):
        return float(np.sum(self.xrow(c, a, theta) ** 2) + (1 - c) ** 2 * self.blockF2)

    def generators(self, c_tilde):
        return -c_tilde * self.WSLh                   # (K, p) columns g_l

    def gram_b(self, c_tilde, a_tilde):
        G = self.generators(c_tilde)
        M0 = self.xrow(c_tilde, a_tilde, np.zeros(self.model.p))
        return G.T @ G, G.T @ M0

    def basis(self, c_tilde, support, rtol=None):
        """Rank-revealing orthonormal basis of span{Phi_hat_l(eta) T~ : l in support} (K x r), or None if empty.

        A QR basis on a rank-deficient generator matrix (duplicated or
        collinear generators, or c_tilde = 0) returns s orthonormal columns spanning MORE than the generator span and
        would project out arbitrary extra directions.  SVD with singular values > rtol * s_max keeps exactly the span.
        Identical output space on every full-rank support (this model: p = 2 disjoint dictionary)."""
        support = np.asarray(support, int)
        rtol = self.rank_rtol if rtol is None else float(rtol)
        if support.size == 0:
            return None
        G = self.generators(c_tilde)[:, support]
        U, s, _ = np.linalg.svd(G, full_matrices=False)
        if s.size == 0 or s[0] <= 0:
            return None
        keep = s > rtol * s[0]
        return U[:, keep] if keep.any() else None     # (K, r) orthonormal, r = rank


class SourceBank:
    """Source eigensystems and whitening matrices reused across all folds."""

    def __init__(self, model, sst, *, rank_rtol=RANK_RTOL):
        self.model, self.sst = model, sst
        self.rank_rtol = float(rank_rtol)
        self.N_S = sum(st["n"] for st in sst)
        pooled = sum(st["n"] * st["Sf"] for st in sst) / self.N_S
        self.s_max = float(eigh(pooled, eigvals_only=True).max())
        self.eigen = [eigh(st["Sf"]) for st in sst]
        self._views = {}

    def views(self, eta):
        eta = float(eta)
        if eta not in self._views:
            views = []
            for j, (st, (w, V)) in enumerate(zip(self.sst, self.eigen)):
                # Same expression as BlockView.__init__, with a reused eigh.
                W = (V * (np.clip(w, 0, None) + eta) ** -0.5) @ V.T
                v = BlockView.__new__(BlockView)
                v.model, v.j, v.eta, v.rho = self.model, j, eta, st["n"]
                v.rank_rtol = self.rank_rtol
                v.WS = W @ st["Sf"]
                v.Wyf = W @ st["yf"]
                v.WSLh = (v.WS * self.model.mu[None, :]) @ self.model.h.T
                v.blockF2 = float(np.sum(((st["Sf"] * self.model.mu[:, None]) @ W) ** 2))
                v.XtX, v.Xty = v.WS.T @ v.WS, v.WS.T @ v.Wyf
                v.XU, v.UU = v.WS.T @ v.WSLh, v.WSLh.T @ v.WSLh
                v.Uy = v.WSLh.T @ v.Wyf
                v._bases = {}
                views.append(v)
            self._views[eta] = views
        return self._views[eta]


class FoldFit:
    """One target training sample and cached source-conditioned factorizations."""

    def __init__(self, model, tst, sources):
        self.model, self.tst, self.sources = model, tst, sources
        self.w0, self.V0 = eigh(tst["Sf"])
        self.s0_max = float(self.w0.max())
        self.vy0 = self.V0.T @ tst["yf"]
        self.rhos = np.array([st["n"] / tst["n"] for st in sources.sst])
        self.N = tst["n"] + sources.N_S
        self._pilots, self._systems, self._calibrations, self._pooling = {}, {}, {}, {}
        self._calibration_grams = {}

    def pilot(self, lam0):
        lam0 = float(lam0)
        if lam0 not in self._pilots:
            a = self.V0 @ (self.vy0 / (self.w0 + lam0))
            self._pilots[lam0] = (self.tst["mL"] / (self.tst["mL"] + lam0), a)
        return self._pilots[lam0]

    def system(self, eta, lam):
        key = float(eta), float(lam)
        if key not in self._systems:
            views = self.sources.views(eta)
            A = self.tst["Sf"] + lam * np.eye(self.model.K)
            g = self.tst["yf"].copy()
            B = self.tst["mL"]
            du = []
            for rho, v in zip(self.rhos, views):
                A += rho * v.XtX
                g += rho * v.Xty
                B += rho * v.blockF2
                du.append(rho * v.XU)
            fac = cho_factor(A, lower=True, check_finite=False)
            D = np.concatenate(du, axis=1)
            solved = cho_solve(fac, np.column_stack([g, D]), check_finite=False)
            self._systems[key] = dict(views=views, fac=fac, a_base=solved[:, 0],
                                     D=D, Dinv=solved[:, 1:], B=B, lam=lam,
                                     projected={})
        return self._systems[key]

    def calibration(self, c0, a0, lam0, eta, frac):
        key = float(lam0), float(eta), float(frac)
        if key not in self._calibrations:
            gram_key = float(lam0), float(eta)
            if gram_key not in self._calibration_grams:
                views = self.sources.views(eta)
                Gb = [v.gram_b(c0, a0) for v in views]
                lmax = [lam_max(b) for _, b in Gb]
                self._calibration_grams[gram_key] = Gb, lmax
            Gb, lmax = self._calibration_grams[gram_key]
            lams = [frac * x for x in lmax]
            th = [lasso_cd(G, b, float(lam)) for (G, b), lam in zip(Gb, lams)]
            supports = [np.flatnonzero(t != 0) for t in th]
            self._calibrations[key] = th, supports, lmax, lams
        return self._calibrations[key]

    def projected(self, sys, c0, supports):
        # For a nonzero scalar pilot, its magnitude does not change a span.
        key = tuple(tuple(int(x) for x in s) for s in supports), bool(c0 != 0)
        if key not in sys["projected"]:
            U, q, ranks = [], [], []
            for rho, v, support in zip(self.rhos, sys["views"], supports):
                skey = tuple(int(x) for x in support), bool(c0 != 0)
                if skey not in v._bases:
                    v._bases[skey] = v.basis(1.0 if c0 != 0 else 0.0, support)
                Q = v._bases[skey]
                ranks.append(0 if Q is None else Q.shape[1])
                if Q is not None:
                    U.append(np.sqrt(rho) * v.WS.T @ Q)
                    q.append(np.sqrt(rho) * Q.T @ v.Wyf)
            a = sys["a_base"].copy()
            if U:
                U = np.concatenate(U, axis=1)
                q = np.concatenate(q)
                AiU = cho_solve(sys["fac"], U, check_finite=False)
                small = np.eye(U.shape[1]) - U.T @ AiU
                small = (small + small.T) * 0.5
                rhs = U.T @ a - q
                a += AiU @ np.linalg.solve(small, rhs)
            c = sys["B"] / (sys["B"] + sys["lam"])
            sys["projected"][key] = float(c), a, ranks
        return sys["projected"][key]

    def unprojected(self, sys, thetas):
        th = np.concatenate(thetas)
        d, Ai_d = sys["D"] @ th, sys["Dinv"] @ th
        h, gc = sys["B"] + sys["lam"], sys["B"]
        for rho, v, t in zip(self.rhos, sys["views"], thetas):
            h += rho * float(t @ v.UU @ t)
            gc += rho * float(t @ v.Uy)
        denominator = h - d @ Ai_d
        if not np.isfinite(denominator) or denominator <= 0:
            raise np.linalg.LinAlgError("nonpositive unprojected Schur complement")
        c = (gc - d @ sys["a_base"]) / denominator
        return float(c), sys["a_base"] - Ai_d * c

    def fit(self, cand, method):
        lam0 = cand.lam0_frac * self.s0_max
        c0, a0 = self.pilot(lam0)
        info = dict(scales=dict(s0_max=self.s0_max, s_max=self.sources.s_max), lam0=lam0)
        if method == "pilot" or (method in ("projected", "unprojected") and cand.no_transfer):
            return c0, a0, info
        lambar = cand.lambar_frac * self.sources.s_max
        lam = self.N / self.tst["n"] * lambar
        info.update(lambar=lambar, lam=lam, N_train=self.N, n0_train=self.tst["n"])
        if method == "pooling":
            if lam not in self._pooling:
                A = self.tst["Sf"] + lam * np.eye(self.model.K)
                g = self.tst["yf"].copy()
                for rho, st in zip(self.rhos, self.sources.sst):
                    A += rho * st["Sf"]
                    g += rho * st["yf"]
                self._pooling[lam] = cho_solve(cho_factor(A, lower=True, check_finite=False), g, check_finite=False)
            return 1.0, self._pooling[lam], info
        eta = eta_from_rule(cand.eta_rule, lambar)
        th, supports, lmax, lams = self.calibration(c0, a0, lam0, eta, cand.lamj_frac)
        info.update(eta=eta, lam_max=lmax, lam_js=[float(x) for x in lams],
                    supports=[s.tolist() for s in supports], thetas=[t.tolist() for t in th])
        sys = self.system(eta, lam)
        if method == "projected":
            c, a, ranks = self.projected(sys, c0, supports)
            info["projection_ranks"] = ranks
        elif method == "unprojected":
            c, a = self.unprojected(sys, th)
        else:
            raise ValueError(method)
        return c, a, info


def _support(support, p):
    result = np.asarray(support, dtype=int)
    if result.ndim != 1 or len(np.unique(result)) != result.size:
        raise ValueError("Support must contain distinct indices")
    if np.any(result < 0) or np.any(result >= p):
        raise ValueError("Support index outside dictionary")
    return result


def projection_basis(view, c, support):
    """Literal SVD basis of Phi_l(A[c,a]), retaining its actual numerical rank."""
    support = _support(support, view.model.p)
    generators = -float(c) * view.WSLh[:, support]
    if not support.size:
        return np.empty((view.model.K, 0)), np.empty(0)
    u, singular, _ = np.linalg.svd(generators, full_matrices=False)
    keep = singular > view.rank_rtol * singular[0] if singular[0] > 0 else np.zeros_like(singular, dtype=bool)
    return u[:, keep], singular


def restricted_recalibration(view, c, a, support):
    """Eq. (11) using the rectangular design, without a penalty or reselection.

    The block part of the moment is constant in theta.  Thus minimizing its full
    norm amounts to solving c*WSLh[:, support] theta = Wyf - WS*a.  A rank-deficient
    design receives its minimum-norm solution at the registered SVD tolerance.
    The original support is returned even if a recalibrated coefficient is zero.
    """
    support = _support(support, view.model.p)
    residual0 = view.Wyf - view.WS @ a
    design = float(c) * view.WSLh[:, support]
    theta = np.zeros(view.model.p)
    rank, singular = 0, np.empty(0)
    if support.size:
        coefficient, _, rank, singular = np.linalg.lstsq(design, residual0, rcond=view.rank_rtol)
        theta[support] = coefficient
    residual = residual0 - float(c) * view.WSLh @ theta
    score = design.T @ residual
    basis, _ = projection_basis(view, c, support)
    design_norm = float(np.linalg.norm(design))
    leakage = float(np.linalg.norm(design - basis @ (basis.T @ design))) / design_norm if design_norm else 0.
    diagnostics = dict(fixed_support=support.tolist(), numerical_rank=int(rank),
                       singular_values=singular.tolist(),
                       singular_min_max_ratio=float(singular[-1] / singular[0]) if singular.size and singular[0] > 0 else None,
                       normalized_generator_projection_leakage=leakage,
                       score_l2=float(np.linalg.norm(score)),
                       moment_norm2=float(residual @ residual + (1 - c) ** 2 * view.blockF2),
                       unpenalized=True, support_reselected=False)
    return theta, diagnostics


def dense_refit(fold, views, thetas, lam, c_reference, supports):
    """Independent literal Eq. (12) joint solve in z=(c,a), for validation.

    No call to FoldFit.projected, the Woodbury cache, or core.refit is made.
    The K-dimensional projected source rows and the scalar block-loss curvature
    are assembled directly from their defining objective.
    """
    k = fold.model.K
    hessian = np.eye(k + 1) * float(lam)
    right = np.zeros(k + 1)
    hessian[0, 0] += fold.tst["mL"]
    hessian[1:, 1:] += fold.tst["Sf"]
    right[0] += fold.tst["mL"]
    right[1:] += fold.tst["yf"]
    ranks = []
    for rho, view, theta, support in zip(fold.rhos, views, thetas, supports):
        design = np.column_stack((view.WSLh @ theta, view.WS))
        response = view.Wyf.copy()
        basis, _ = projection_basis(view, c_reference, support)
        ranks.append(int(basis.shape[1]))
        if basis.shape[1]:
            design -= basis @ (basis.T @ design)
            response -= basis @ (basis.T @ response)
        hessian += rho * design.T @ design
        right += rho * design.T @ response
        hessian[0, 0] += rho * view.blockF2
        right[0] += rho * view.blockF2
    fitted = np.linalg.solve(hessian, right)
    if not np.isfinite(fitted).all():
        raise FloatingPointError("Nonfinite dense iterative projected refit")
    return float(fitted[0]), fitted[1:], ranks


def iteration_path(fold, candidate, max_step=10, *, backend="cached"):
    """Return coefficients, true risk, calibration and diagnostics for A[0..K].

    ``steps[k]['theta']`` is theta[k], namely the recalibration AT A[k] that
    would be used for A[k+1].  At k=0 it is the original Lasso calibration.
    Therefore the initial Lasso theta is also kept in ``initial_info``.
    ``projection_ranks`` refers to the updated projection AT the given A[k].
    ``refit_projection_ranks`` refers to the projection used to produce A[k].

    Results contain numpy coefficient arrays deliberately; callers may archive
    them directly or serialize them with the existing scientific JSON cleaner.
    """
    if isinstance(max_step, bool) or not isinstance(max_step, (int, np.integer)) or max_step < 0:
        raise ValueError("max_step must be a nonnegative integer")
    if backend not in {"cached", "dense"}:
        raise ValueError("backend must be cached or dense")
    lam0 = float(candidate.lam0_frac * fold.s0_max)
    c0, a0 = fold.pilot(lam0)
    c1, a1, initial_info = fold.fit(candidate, "projected")
    no_transfer = bool(candidate.no_transfer)
    supports = ([np.asarray(s, dtype=int) for s in initial_info["supports"]]
                if not no_transfer else [np.empty(0, dtype=int) for _ in range(fold.model.m)])
    fixed_supports = [s.tolist() for s in supports]
    theta0 = ([np.asarray(t, dtype=float) for t in initial_info["thetas"]]
              if not no_transfer else [np.zeros(fold.model.p) for _ in range(fold.model.m)])
    views = [] if no_transfer else fold.sources.views(initial_info["eta"])
    system = None if no_transfer else fold.system(initial_info["eta"], initial_info["lam"])
    steps = []
    c, a, previous_theta = c0, a0.copy(), theta0
    refit_ranks = None
    for step in range(max_step + 1):
        if step:
            if no_transfer:
                c, a, refit_ranks = c0, a0.copy(), [0] * fold.model.m
            elif backend == "dense":
                c, a, refit_ranks = dense_refit(fold, views, previous_theta,
                                               initial_info["lam"], c, supports)
            elif step == 1:
                c, a = c1, a1.copy()
                refit_ranks = list(initial_info["projection_ranks"])
            else:
                c, a, refit_ranks = fold.projected(system, c, supports)
                a, refit_ranks = a.copy(), list(refit_ranks)
        risk = float(fold.model.risk(c, a))
        if not np.isfinite(c) or not np.isfinite(a).all() or not np.isfinite(risk):
            raise FloatingPointError("Nonfinite iterate or true risk")
        if no_transfer:
            theta = [t.copy() for t in theta0]
            calibration_diagnostics = []
            ranks = [0] * fold.model.m
        else:
            ranks = [int(projection_basis(v, c, s)[0].shape[1]) for v, s in zip(views, supports)]
            if step == 0:
                theta = [t.copy() for t in theta0]
                calibration_diagnostics = []
            else:
                calibrated = [restricted_recalibration(v, c, a, s) for v, s in zip(views, supports)]
                theta = [x[0] for x in calibrated]
                calibration_diagnostics = [x[1] for x in calibrated]
                if ranks != [x["numerical_rank"] for x in calibration_diagnostics]:
                    raise AssertionError("Recalibration rank and projection rank disagree")
        if any(np.any(t[np.setdiff1d(np.arange(fold.model.p), s)] != 0)
               for t, s in zip(theta, supports)):
            raise AssertionError("A recalibration escaped its initial support")
        previous = steps[-1] if steps else None
        steps.append(dict(step=step, c=float(c), a=a.copy(), risk=risk,
                          theta=np.asarray(theta), fixed_supports=copy.deepcopy(fixed_supports),
                          projection_ranks=ranks, refit_projection_ranks=refit_ranks,
                          calibration_kind="target_only_exit" if no_transfer else ("lasso" if step == 0 else "restricted_unpenalized_ls"),
                          calibration_diagnostics=calibration_diagnostics,
                          coefficient_change_norm=None if previous is None else float(np.hypot(c - previous["c"], np.linalg.norm(a - previous["a"]))),
                          risk_change=None if previous is None else risk - previous["risk"]))
        previous_theta = theta
    first = steps[1] if len(steps) > 1 else steps[0]
    later = steps[2:]
    compact_diagnostics = dict(
        max_postfirst_coefficient_drift=max((float(np.hypot(s["c"] - first["c"],
            np.linalg.norm(s["a"] - first["a"]))) for s in later), default=0.),
        max_postfirst_absolute_risk_drift=max((abs(s["risk"] - first["risk"]) for s in later), default=0.),
        first_recalibration_theta_change=float(np.linalg.norm(steps[1]["theta"] - steps[0]["theta"])) if len(steps) > 1 else None,
        max_postfirst_theta_drift=max((float(np.linalg.norm(s["theta"] - first["theta"])) for s in later), default=0.),
        max_normalized_generator_projection_leakage=max((d["normalized_generator_projection_leakage"] for s in steps for d in s["calibration_diagnostics"]), default=0.),
        max_recalibration_score_l2=max((d["score_l2"] for s in steps for d in s["calibration_diagnostics"]), default=0.))
    return dict(steps=steps, fixed_supports=fixed_supports, iteration_diagnostics=compact_diagnostics,
                initial_info=copy.deepcopy(initial_info), no_transfer=no_transfer,
                backend=backend, rank_rtol=fold.sources.rank_rtol,
                supports_fixed=True, same_observations=True,
                penalties_fixed=True, recalibration_penalized=False)

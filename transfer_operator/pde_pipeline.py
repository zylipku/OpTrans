"""One pilot/calibration/refit/recalibration pipeline for structured PDE models.

``additional_rounds=0`` means Projected P. Values 1 and 2 append RC[1] and
RC[2]. An externally supplied initial_operator is B, never an RC round.
"""
from dataclasses import dataclass
import numpy as np
from .pde_estimators import (
    calibration, prepare_source, ridge_fit, source_projector, refit,
)


@dataclass
class Tuning:
    lam0: float
    eta: float
    ridge: float
    penalties: np.ndarray
    effective_dimension: float


def _positive_integer(value):
    return (not isinstance(value, (bool, np.bool_)) and
            isinstance(value, (int, np.integer)) and value > 0)


def frozen_tuning(model, n0, ns, candidate, allocation=None):
    """Evaluate frozen multipliers; no tuning search or data selection occurs.

    candidate is (c0,cbar,ctheta), or a mapping with these keys. The vector
    evaluation order is retained from the registered advection implementation.
    """
    if candidate is None:
        raise ValueError('A frozen candidate or explicit Tuning is required')
    if isinstance(candidate, dict):
        c0, cbar, ctheta = (candidate[key] for key in ('c0', 'cbar', 'ctheta'))
    else:
        c0, cbar, ctheta = candidate
    if not _positive_integer(n0) or not _positive_integer(ns):
        raise ValueError('n0 and ns must be positive integers')
    if not all(np.isfinite(c) for c in (c0, cbar, ctheta)) or c0 < 0 or cbar <= 0 or ctheta < 0:
        raise ValueError('Frozen multipliers require c0 >= 0, cbar > 0, ctheta >= 0')
    if allocation is None:
        if ns % 2:
            raise ValueError('The default two-source allocation requires an even ns')
    elif (not len(allocation) or not all(_positive_integer(n) for n in allocation)
          or sum(allocation) != ns):
        raise ValueError('Source allocation must contain positive integers summing to ns')
    exponent = model.s / (model.s + 1)
    N = n0 + ns
    lam0 = c0 * n0 ** (-exponent)
    eta = cbar * N ** (-exponent)
    neff = float(np.sum(model.q / (model.q + eta)))
    allocation = np.asarray(allocation if allocation is not None else [ns//2]*2)
    rm = np.sqrt(neff / allocation) + 1 / (allocation * np.sqrt(eta))
    penalties = ctheta * (rm + n0 ** (-exponent / 2))
    return Tuning(lam0, eta, N/n0*eta, penalties, neff)


def pilot(model, target, ridge):
    """Target-only ridge pilot, before support selection."""
    return ridge_fit(model, target, ridge)


@dataclass
class RefitStep:
    reference: np.ndarray
    calibrations: list
    projectors: list
    operator: np.ndarray
    info: dict

    @property
    def thetas(self):
        return [fit.theta for fit in self.calibrations]


def recalibration_round(model, target, sources, current, supports, rho, ridge):
    """Restricted LS on the initial supports, then updated projection/refit."""
    if not len(sources) or len(supports) != len(sources) or len(rho) != len(sources):
        raise ValueError('Every source needs one fixed support and one weight')
    fits = [calibration(model, src, current, support=support)
            for src, support in zip(sources, supports)]
    projectors = [source_projector(model, src, current, support)
                  for src, support in zip(sources, supports)]
    h, info = refit(model, target, sources, [c.theta for c in fits],
                    projectors, rho, ridge)
    return RefitStep(current, fits, projectors, h, info)


@dataclass
class TransferFit:
    pilot: np.ndarray
    sources: list
    supports: list
    initial_calibrations: list
    unprojected: np.ndarray
    steps: list
    tuning: Tuning
    rho: object

    @property
    def projected(self):
        """Operators [P, RC[1], ...]; the pilot is deliberately separate."""
        return [step.operator for step in self.steps]

    @property
    def operator_path(self):
        return [self.pilot] + self.projected


def fit_transfer(model, target, source_stats, candidate=None, *, tuning=None,
                 additional_rounds=2, projected=True, initial_operator=None,
                 supports=None, eigencache=None, sources=None, rho=None):
    """Run the shared stages with same-sample moments and fixed supports.

    Exact TaskStats and explicit supports support independent equation checks.
    The paper experiments learn supports from Lasso.
    ``projected=False`` computes independently tuned U without extra refits.
    """
    if (isinstance(additional_rounds, (bool, np.bool_)) or
        not isinstance(additional_rounds, (int, np.integer)) or additional_rounds < 0):
        raise ValueError('additional_rounds must be a nonnegative integer')
    count = len(source_stats)
    if not count:
        raise ValueError('At least one source task is required')
    if not _positive_integer(target.n) or not all(_positive_integer(st.n) for st in source_stats):
        raise ValueError('Task sample counts must be positive integers')
    for stats in [target] + list(source_stats):
        if np.shape(stats.Cff) != (model.d, model.d) or np.shape(stats.Cuf) != (model.d, model.d):
            raise ValueError('Task moments must have shape (model.d, model.d)')
    for name, values in (('sources', sources), ('supports', supports),
                         ('eigencache', eigencache), ('rho', rho)):
        if values is not None and len(values) != count:
            raise ValueError(f'{name} must have one entry per source task')
    allocation = [stats.n for stats in source_stats]
    if tuning is None:
        tuning = frozen_tuning(model, target.n, sum(allocation), candidate, allocation)
    if (np.shape(tuning.penalties) != (count,) or
        not np.all(np.isfinite(tuning.penalties)) or np.any(np.asarray(tuning.penalties) < 0)):
        raise ValueError('Tuning needs one finite nonnegative penalty per source')
    if (not all(np.isfinite(x) for x in (tuning.lam0, tuning.eta, tuning.ridge)) or
        tuning.lam0 < 0 or tuning.eta <= 0 or tuning.ridge < 0):
        raise ValueError('Tuning requires lam0 >= 0, eta > 0, ridge >= 0')
    if rho is None:
        rho = np.asarray(allocation) / target.n
    if not np.all(np.isfinite(rho)) or np.any(np.asarray(rho) < 0):
        raise ValueError('Source weights must be finite and nonnegative')
    if sources is None:
        eigenpairs = [None]*len(source_stats) if eigencache is None else eigencache
        sources = [prepare_source(model, stats, tuning.eta, eigencache=eigen)
                   for stats, eigen in zip(source_stats, eigenpairs)]
    elif any(source.eta != tuning.eta or source.n != stats.n
             for source, stats in zip(sources, source_stats)):
        raise ValueError('Prepared moments must match the current eta and source sample counts')
    B = pilot(model, target, tuning.lam0) if initial_operator is None else initial_operator
    if np.shape(B) != (model.d,) or not np.all(np.isfinite(B)):
        raise ValueError('The initial operator must be a finite model.d vector')
    if supports is None:
        initial = [calibration(model, source, B, penalty)
                   for source, penalty in zip(sources, tuning.penalties)]
        supports = [fit.support for fit in initial]
    else:
        supports = [tuple(support) for support in supports]
        initial = [calibration(model, source, B, support=support)
                   for source, support in zip(sources, supports)]
    theta = [fit.theta for fit in initial]
    projectors = [source_projector(model, source, B, support)
                  for source, support in zip(sources, supports)]
    U, _ = refit(model, target, sources, theta, [None]*len(sources), rho, tuning.ridge)
    steps = []
    if projected:
        h, info = refit(model, target, sources, theta, projectors, rho, tuning.ridge)
        steps.append(RefitStep(B, initial, projectors, h, info))
        for _ in range(int(additional_rounds)):
            step = recalibration_round(model, target, sources, h, supports, rho, tuning.ridge)
            steps.append(step)
            h = step.operator
    return TransferFit(B, sources, supports, initial, U, steps, tuning, rho)

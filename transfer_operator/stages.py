"""Common stage/result interface with representation-specific numerical backends.

The coupled adapter delegates to the unchanged experiment1 backend. PDE models
share the moment/calibration/refit implementation in pde_pipeline; a new PDE
representation implements PDEModel instead of copying an estimator.
"""
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .estimators import iteration_path
from .pde_pipeline import fit_transfer


@dataclass
class CoupledProblem:
    fold: Any
    candidate: Any


@dataclass
class PDEProblem:
    model: Any
    target: Any
    sources: list
    candidate: Any = None
    options: dict = field(default_factory=dict)


@dataclass
class StagePath:
    """Pilot is separate; refits[0] is P (or U), refits[k] is RC[k].

    Coupled coefficients are (c,a); PDE coefficients are real H vectors.
    ``raw`` preserves backend diagnostics without changing their conventions.
    """
    pilot: Any
    refits: list
    risks: list
    supports: list
    method: str
    raw: Any


def fit_stages(problem, *, projected=True, additional_rounds=2):
    """Fit frozen parameters on supplied moments, without selection or sampling.

    Call separately with each method's independently frozen candidate. Additional
    rounds apply to Projected; Unprojected returns its single refit. Exact PDE
    population moments use PDEProblem.options (initial_operator, supports,
    tuning and rho) and the same implementation as empirical moments.
    """
    if (isinstance(additional_rounds, bool) or
        not isinstance(additional_rounds, (int, np.integer)) or additional_rounds < 0):
        raise ValueError('additional_rounds must be a nonnegative integer')
    method = 'Projected' if projected else 'Unprojected'
    if isinstance(problem, PDEProblem):
        result = fit_transfer(problem.model, problem.target, problem.sources,
                              problem.candidate, projected=projected,
                              additional_rounds=additional_rounds, **problem.options)
        refits = result.projected if projected else [result.unprojected]
        return StagePath(result.pilot, refits,
                         [problem.model.risk(h) for h in refits], result.supports,
                         method, result)
    if not isinstance(problem, CoupledProblem):
        raise TypeError('problem must be CoupledProblem or PDEProblem')
    fold, candidate = problem.fold, problem.candidate
    if projected:
        result = iteration_path(fold, candidate, max_step=additional_rounds+1)
        pilot = (result['steps'][0]['c'], result['steps'][0]['a'])
        refits = [(step['c'], step['a']) for step in result['steps'][1:]]
        return StagePath(pilot, refits, [step['risk'] for step in result['steps'][1:]],
                         result['fixed_supports'], method, result)
    pilot = fold.pilot(float(candidate.lam0_frac*fold.s0_max))
    c, a, info = fold.fit(candidate, 'unprojected')
    return StagePath(pilot, [(c, a)], [fold.model.risk(c, a)],
                     info.get('supports', []), method, info)

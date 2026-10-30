"""Registered advection replications and paired percentile-bootstrap means."""
import numpy as np
from scipy.linalg import eigh
from .pde_models import AdvectionDiffusionModel
from .pde_moments import generate_task
from .pde_estimators import ridge_fit, pooled_fit, prepare_source
from .pde_pipeline import frozen_tuning, fit_transfer


def advection_replicate(cell, candidates, seed, replicate=0, *, additional_rounds=2):
    """Replay P and independently tuned U from one original advection record.

    Returns candidate dictionaries with risk0/risk1... and projection_off_risk,
    supports, and theta_path. U is candidate 1's projection_off_risk, never P's
    matched projection-off control. Baselines use the cell's frozen multipliers.
    """
    model = AdvectionDiffusionModel(**{key: cell[key] for key in ('R','s','sigma','tau')})
    if 'thetas' in cell:
        model.thetas = np.asarray(cell['thetas'], dtype=float)
    n0, ns = cell['n0'], cell['nS']
    allocation = cell.get('allocation', [ns//len(model.thetas)]*len(model.thetas))
    profile = cell.get('profile', 'full')
    target = generate_task(model, n0, [0.,0.], [seed,replicate,0], profile=profile)
    stats = [generate_task(model, nj, theta, [seed,replicate,j+1], profile=profile)
             for j,(nj,theta) in enumerate(zip(allocation,model.thetas))]
    eigenpairs = [eigh(st.Cff, check_finite=False) for st in stats]
    source_cache, pilot_cache, records = {}, {}, []
    for candidate in candidates:
        tuning = frozen_tuning(model, n0, ns, candidate, allocation)
        if tuning.eta not in source_cache:
            source_cache[tuning.eta] = [prepare_source(model, st, tuning.eta, eigencache=eig)
                                        for st,eig in zip(stats,eigenpairs)]
        if tuning.lam0 not in pilot_cache:
            pilot_cache[tuning.lam0] = ridge_fit(model, target, tuning.lam0)
        fit = fit_transfer(model, target, stats, tuning=tuning,
                           additional_rounds=additional_rounds,
                           projected=candidate.get('role') != 'unprojected',
                           sources=source_cache[tuning.eta], initial_operator=pilot_cache[tuning.lam0])
        row = {f'risk{k}': model.risk(h) for k,h in enumerate(fit.operator_path)}
        row.update(projection_off_risk=model.risk(fit.unprojected),
                   supports=fit.supports,
                   theta_path=np.asarray([np.concatenate(step.thetas) for step in fit.steps]),
                   kkt=max([c.kkt for c in fit.initial_calibrations]
                           + [c.kkt for step in fit.steps for c in step.calibrations]),
                   lam0=tuning.lam0, eta=tuning.eta, ridge=tuning.ridge,
                   neff=tuning.effective_dimension, penalties=tuning.penalties)
        records.append(row)
    baselines = {}
    for name, default, size in (('target-only', .001, n0), ('pooling', 0., n0+ns)):
        mults = cell.get('baseline_multipliers', {}).get(name, [default])
        for mult in mults:
            penalty = mult*size**(-model.s/(model.s+1))
            h = (ridge_fit(model,target,penalty) if name == 'target-only' else
                 pooled_fit(model,[target]+stats,penalty))
            baselines[(name,mult)] = model.risk(h)
    return dict(candidates=records, baselines=baselines)



def bootstrap_resample_means(risks, draws, seed, *, algorithm):
    """Use the registered integer indices -> counts -> normalized weights algorithm."""
    risks = np.asarray(risks)
    if risks.ndim not in (1, 2) or not len(risks) or not np.all(np.isfinite(risks)):
        raise ValueError('risks must be a nonempty finite replication vector or matrix')
    if (isinstance(draws, (bool, np.bool_)) or
        not isinstance(draws, (int, np.integer)) or draws <= 0):
        raise ValueError('draws must be a positive integer')
    n = len(risks)
    rng = np.random.default_rng(seed)
    if algorithm == 'indices':
        indices = rng.integers(n, size=(draws,n))
        counts = np.zeros((draws,n), dtype=np.int16)
        np.add.at(counts, (np.arange(draws)[:,None],indices), 1)
        weights = counts.astype(float)/n
        return weights@risks
    raise ValueError("algorithm must be 'indices'")

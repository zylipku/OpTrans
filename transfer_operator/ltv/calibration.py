"""Per-source calibration of the perturbation coefficients with held-out gate folds."""
from __future__ import annotations

import numpy as np

from .harmonic import calibration_data, calibration_residual, elastic_calibrate


def calibrate_pair(model, records, pilot, pair, design):
    """Calibrate theta^(j) of one source at the pilot and gate it on held-out records.

    LTV sources (three distinct realisations): three 2-fit/1-gate rotations,
    l1 chosen within each fold by leave-one-record-out CV on the two fit
    records; every held-out gate must pass. Constant sources: PN records
    calibrate (fixed ``const_l1``), the MS record gates. The gate requires a
    relative moment residual <= max_relative_residual and an improvement
    factor >= min_improvement_factor over theta = 0. After passing, theta is
    refitted on all records with the median selected l1.
    """
    cfg = design['dictionary']
    base = pair['base_atoms']
    grid = cfg['l1_grid']
    ridge = cfg['ridge']
    const_l1 = cfg.get('const_l1', .001)
    const = all('Const' in r['ref']['file'] for r in records)
    folds, lambdas = [], []
    if const:
        pn = [i for i, r in enumerate(records) if r['ref']['file'].startswith('PN_')]
        ms = [i for i, r in enumerate(records) if r['ref']['file'].startswith('MS_')]
        if not pn or not ms or records[pn[0]]['ref']['u_ref_sha256'] == records[ms[0]]['ref']['u_ref_sha256']:
            return dict(passed=False, reason='missing_independent_PN_MS', folds=[]), None
        splits = [(pn, [ms[0]])]
    elif len(records) == 3 and len({r['ref']['u_ref_sha256'] for r in records}) == 3:
        splits = [([i for i in range(3) if i != g], [g]) for g in range(3)]
    else:
        return dict(passed=False, reason='not_three_source_realizations', folds=[]), None
    for fit_idx, gate_idx in splits:
        inner = []
        if const:
            lam = const_l1
        else:
            ds = {i: calibration_data(model, [records[i]], pilot, base) for i in fit_idx}
            for l in grid:
                scores = []
                for a, b in [fit_idx, fit_idx[::-1]]:
                    theta = elastic_calibrate(ds[a], l, ridge)
                    scores.append(calibration_residual(ds[b], theta)['residual']**2)
                inner.append(dict(l1=l, loss=float(np.mean(scores)), fold_losses=scores))
            lam = min(inner, key=lambda x: (x['loss'], -x['l1']))['l1']
        fit = calibration_data(model, [records[i] for i in fit_idx], pilot, base)
        th = elastic_calibrate(fit, lam, ridge)
        held = calibration_data(model, [records[i] for i in gate_idx], pilot, base)
        res = calibration_residual(held, th)
        passed = (res['residual'] <= design['gate']['max_relative_residual']
                  and res['improvement_factor'] >= design['gate']['min_improvement_factor'])
        folds.append(dict(fit_records=[records[i]['ref'] for i in fit_idx],
                          gate_records=[records[i]['ref'] for i in gate_idx], l1=lam,
                          inner_curves=inner, theta_scaled=th.tolist(), atom_names=fit['names'],
                          passed=bool(passed), **res))
        lambdas.append(lam)
    passed = all(f['passed'] for f in folds)
    selected_lam = float(np.median(lambdas))
    final = calibration_data(model, records, pilot, base)
    theta = elastic_calibrate(final, selected_lam, ridge) if passed else None
    row = dict(passed=passed, folds=folds, l1_final=selected_lam,
               theta_final_scaled=theta.tolist() if theta is not None else None,
               atom_names=final['names'], source_records=len(records))
    if theta is not None:
        row['theta_final_physical'] = [float(th*model.wc**(1-int(name.split(':d')[1])))
                                       for th, name in zip(theta, final['names'])]
    return row, theta

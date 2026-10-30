"""Stages of the real-data benchmark: target class, source selection, pooling, held-out scoring.

Every stage receives the frozen design and a :class:`records.RecordBank`. The
held-out test response of a rotation is read only by :func:`evaluate_scope`;
selection and pooling see the train and validation records, the source bank,
and the *input* spectrum of the test record.
"""
from __future__ import annotations

import numpy as np

from .banded import Harmonic, fit_quadratic, forward_select, source_stats, target_stats
from .calibration import calibrate_pair
from .design import class_grid
from .harmonic import INSTRUMENT_HALO, WHITENING_RIDGE_RELATIVE, metrics
from .pooling import fit_pooling

METHODS = ('Target only', 'Pooling', 'Unprojected', 'Projected')
TRANSFER_METHODS = ('Unprojected', 'Projected')
BANDS = ('full_band', 'forced_lines', 'sideband_only')
BAND_LABELS = {'full_band': 'Full band', 'forced_lines': 'Forced lines', 'sideband_only': 'Sidebands'}


def make_model(design, hp):
    f = design['frequency']
    return Harmonic(f['N'], f['fs'], hp['R'], hp['B'], *f['output_band_Hz'], wc=f['wc'])


def check_constants(design):
    moments = design['moments']
    if (moments['instrument_halo'], moments['whitening_ridge_relative']) != (INSTRUMENT_HALO, WHITENING_RIDGE_RELATIVE):
        raise ValueError('The frozen moment constants differ from the implementation')


def relative_loss(pred, y):
    return float(np.sum(np.abs(pred-y)**2)/np.sum(np.abs(y)**2))


def scope_name(target, rotation):
    return f"{target['trajectory_hash'][:12]}_r{rotation}"


def source_records(bank, system):
    """All records of a source system (PN and MS for constant scheduling; three realisations for LTV)."""
    return [bank.record(ref) for ref in system['records']]


# --- target class --------------------------------------------------------------

def meta_losses(design, bank, target, rotation):
    """Train/validation losses of the 36 target-only configurations on one rotation."""
    split = target['rotations'][rotation]
    train, val = bank.record(split['train']), bank.record(split['validation'])
    grid = class_grid(design)
    rows = []
    for R, B in sorted({(c['R'], c['B']) for c in grid}):
        model = make_model(design, dict(R=R, B=B))
        h, b = target_stats(model, [train])
        for hp in [c for c in grid if (c['R'], c['B']) == (R, B)]:
            beta = fit_quadratic(h, b, model.regularizer(h, hp['smooth']))
            rows.append(dict(target=target['id'], trajectory_hash=target['trajectory_hash'], rotation=rotation, **hp,
                             train_loss=relative_loss(model.predict(train['F'], beta), train['Y'][model.rows]),
                             validation_loss=relative_loss(model.predict(val['F'], beta), val['Y'][model.rows])))
    return rows


def _same(row, hp):
    return (int(row['R']), int(row['B']), float(row['smooth'])) == (hp['R'], hp['B'], hp['smooth'])


def choose_classes(design, rows):
    """Leave-one-target-out class choice: mean validation loss over the other seven targets' 21 rotations."""
    grid = class_grid(design)
    n_targets = len(design['targets'])
    choices = []
    for t in design['targets']:
        aux = [a['id'] for a in design['targets'] if a['id'] != t['id']]
        comparisons = []
        for hp in grid:
            vals = [float(row['validation_loss']) for row in rows if row['target'] in aux and _same(row, hp)]
            if len(vals) != 3*(n_targets-1):
                raise ValueError('Incomplete meta cache')
            comparisons.append(dict(**hp, mean_loss=float(np.mean(vals)), median_loss=float(np.median(vals))))
        best = min(comparisons, key=lambda x: (x['mean_loss'], x['B'], x['R'], -x['smooth']))
        choices.append(dict(held_out_target=t['id'], trajectory_hash=t['trajectory_hash'], auxiliary_targets=aux,
                            hyperparameters={k: best[k] for k in ('R', 'B', 'smooth')},
                            mean_loss=best['mean_loss'], median_loss=best['median_loss']))
    return choices


# --- transfer estimators -------------------------------------------------------

def select_sources(design, bank, target, rotation, hp):
    """Target-only pilot, per-source calibration gates, Unprojected and Projected forward selection."""
    check_constants(design)
    split = target['rotations'][rotation]
    train, val = bank.record(split['train']), bank.record(split['validation'])
    model = make_model(design, hp)
    h, b = target_stats(model, [train])
    reg = model.regularizer(h, hp['smooth'])
    pilot = fit_quadratic(h, b, reg)
    yv = val['Y'][model.rows]

    def score(z):
        return relative_loss(model.predict(val['F'], z), yv)
    baseline = score(pilot)
    systems = {s['id']: s for s in design['sources']}
    gates, stats = [], {}
    for pair in target['candidates']:
        sid = pair['source_id']
        system = systems[sid]
        if system['trajectory_hash'] == target['trajectory_hash']:
            raise AssertionError('Target operator cannot be a source')
        records = source_records(bank, system)
        gate, theta = calibrate_pair(model, records, pilot, pair, design)
        row = dict(source=sid, source_kind=system['kind'], base_atoms=pair['base_atoms'], passed=gate['passed'],
                   reason=gate.get('reason'), l1_final=gate.get('l1_final'),
                   fold_residuals=[f['residual'] for f in gate['folds']],
                   fold_improvement_factors=[f['improvement_factor'] for f in gate['folds']],
                   theta_final_physical=gate.get('theta_final_physical'), atom_names=gate.get('atom_names'))
        if gate['passed']:
            stats[sid] = source_stats(model, records, pilot, pair['base_atoms'], theta)
            row.update(active_atoms=stats[sid]['active_atoms'], projection_rank=stats[sid]['projection_rank'])
        gates.append(row)
    rules = design['selection']
    fits = {'Target only': pilot}
    selections = {}
    for method in TRANSFER_METHODS:
        quadratics = {sid: (q[method][0], q[method][1], q['n']) for sid, q in stats.items()}
        sel = forward_select(h, b, reg, pilot, quadratics, rules['source_scale_grid'], score,
                             rules['step_relative_improvement'], rules['max_sources'])
        fits[method] = sel.pop('z')
        sel.pop('curves')
        selections[method] = sel
    test_F = bank.input_spectrum(split['test'])
    return dict(hyperparameters=hp, rows=model.rows, target_only_validation_loss=baseline, gates=gates,
                selections=selections, passing_sources=sorted(stats), coefficients=fits,
                predictions={m: model.predict(test_F, z) for m, z in fits.items()})


def pooling_scope(design, bank, target, rotation, hp):
    """One common operator on the train record and every record of the 12 candidate source tasks."""
    split = target['rotations'][rotation]
    train, val = bank.record(split['train']), bank.record(split['validation'])
    systems = {s['id']: s for s in design['sources']}
    records = [r for pair in target['candidates'] for r in source_records(bank, systems[pair['source_id']])]
    keys = [(r['ref']['file'], r['ref']['column']) for r in records]
    if len(set(keys)) != len(keys) or (split['test']['file'], split['test']['column']) in keys:
        raise AssertionError('Invalid pooled record bank')
    model = make_model(design, hp)
    fitted = fit_pooling(model, train, records, val, design['pooling']['smoothing_grid'])
    smooth = fitted['selected_smooth']
    z = fitted['coefficients'][smooth]
    test_F = bank.input_spectrum(split['test'])
    return dict(hyperparameters=dict(R=hp['R'], B=hp['B'], smooth=smooth), rows=model.rows,
                n_source_tasks=len(target['candidates']), n_source_records=len(records),
                curves=fitted['curves'], coefficients=z, prediction=model.predict(test_F, z))


# --- held-out scoring ----------------------------------------------------------

def evaluate_scope(design, bank, target, rotation, rows, predictions):
    """Score frozen predictions on the held-out test record of the rotation."""
    split = target['rotations'][rotation]
    test = bank.record(split['test'])
    Y = test['Y'][rows]
    forced = np.isin(rows, split['test']['forced_lines'])
    return [dict(target=target['id'], rotation=rotation, method=method, **metrics(pred, Y, forced))
            for method, pred in predictions.items()]


def method_table(scores, methods=METHODS):
    """Mean of each band over the 24 held-out evaluations, one row per method."""
    table = []
    for method in methods:
        rows = [row for row in scores if row['method'] == method]
        if len(rows) != 24:
            raise ValueError(f'{method}: expected 24 evaluations, found {len(rows)}')
        table.append({'method': method, **{band: float(np.mean([float(r[band]) for r in rows])) for band in BANDS}})
    return table


def per_target_table(design, scores, methods=METHODS):
    """Mean of the three rotations for every target and method."""
    table = []
    for index, target in enumerate(design['targets']):
        entry = dict(index=index, target=target['id'], kind=target['kind'], P2P=target['P2P'])
        for method in methods:
            rows = [row for row in scores if row['method'] == method and row['target'] == target['id']]
            if len(rows) != 3:
                raise ValueError('Expected three rotations per target and method')
            for band in BANDS:
                entry[f'{method}:{band}'] = float(np.mean([float(r[band]) for r in rows]))
        table.append(entry)
    return table

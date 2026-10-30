#!/usr/bin/env python3
"""Literature baselines in Experiment 3 (Section 4.4, real-data LTV benchmark).

Adds the estimators of Jiao & McKeague (2025), Lin & Reimherr (2024) and Hu & Lin
(2025) (``transfer_operator.literature``) to the frozen benchmark: the same 24
target rotations, frozen target classes, 12 candidate source tasks per target and
held-out scores. Every choice is made on the validation record and the test-input
predictions are stored before any test response is read. Target only is refitted
and must reproduce its frozen held-out scores.

    python experiment3/literature.py --archive /path/to/data_LPV.zip [--jobs 6]

The raw archive is extracted into the experiment3 work directory by run.py's prepare
phase (it is not redistributed). New results go to workdir/results/literature;
the retained outputs are used only as reference data.
"""
from __future__ import annotations

import os

for _variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_variable, '1')

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import yaml  # noqa: E402

EXPERIMENT_ROOT = Path(__file__).resolve().parent
RELEASE_ROOT = EXPERIMENT_ROOT.parent
if str(RELEASE_ROOT) not in sys.path:
    sys.path.insert(0, str(RELEASE_ROOT))

import run  # noqa: E402  (experiment3/run.py: paths, design and table helpers)
from transfer_operator.literature.core import LABELS, REPORTED as METHODS  # noqa: E402
from transfer_operator.literature.ltv import LTVScope  # noqa: E402
from transfer_operator.ltv import pipeline  # noqa: E402
from transfer_operator.ltv.records import dump  # noqa: E402

SPEC = yaml.safe_load((EXPERIMENT_ROOT / 'configs/literature.yaml').read_text())
RESULTS = EXPERIMENT_ROOT / 'outputs/results/literature'
FROZEN = EXPERIMENT_ROOT / 'outputs/results'
METHOD_NAMES = {m: LABELS[m] for m in METHODS}
SCORES = ('full_band', 'forced_lines', 'sideband_only')


def grids():
    floats = lambda key: [float(x) for x in SPEC[key]]
    return dict(jm_truncation=[int(x) for x in SPEC['jm_truncation']], lr_source=floats('lr_source_smooth'),
                lr_target=floats('lr_target_smooth'), hl_truncation=[int(x) for x in SPEC['hl_truncation']],
                hl_lasso=floats('hl_lasso'))


def build_scope(paths, design, target, rotation, hp):
    bank = paths.bank(design)
    split = target['rotations'][rotation]
    model = pipeline.make_model(design, hp)
    systems = {s['id']: s for s in design['sources']}
    tasks = [(pair['source_id'], pipeline.source_records(bank, systems[pair['source_id']]))
             for pair in target['candidates']]
    scope = LTVScope(model, hp['smooth'], bank.record(split['train']), bank.record(split['validation']),
                     bank.input_spectrum(split['test']), tasks)
    return bank, model, scope


def _select(row_lists):
    """Smallest validation loss per method; ties keep the first candidate."""
    chosen = {}
    for method, rows in row_lists.items():
        losses = np.array([r['validation_loss'] for r in rows])
        if not np.isfinite(losses).any():
            raise FloatingPointError(f'No finite validation loss for {method}')
        losses[~np.isfinite(losses)] = np.inf
        chosen[method] = rows[int(np.argmin(losses))]
    return chosen


def _job(args):
    workdir, raw, ti, rotation = args
    paths = run._paths_from(workdir, raw)
    design = run.load_design()
    target = design['targets'][ti]
    hp = next(c for c in json.loads((FROZEN / 'target_classes.json').read_text())
              if c['held_out_target'] == target['id'])['hyperparameters']
    bank, model, scope = build_scope(paths, design, target, rotation, hp)
    g = grids()
    started = time.time()
    candidates = dict(jm_ts=scope.jm_ts(g['jm_truncation']),
                      lr_atl=scope.lr_atl(g['lr_source'], g['lr_target']),
                      hl_agg=scope.hl_agg(g['hl_truncation'], g['hl_lasso']))
    chosen = _select(candidates)
    # Pairing: refit Target only and Pooling, which must reproduce their frozen held-out scores.
    _, target_only = scope.predictions(scope.fit(scope.h0, scope.b0, hp['smooth']))
    pooling = pipeline.pooling_scope(design, bank, target, rotation, hp)['prediction']
    folder = Path(workdir) / 'literature' / pipeline.scope_name(target, rotation)
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / 'predictions_before_test.npz', rows=model.rows, target_only=target_only,
                        pooling=pooling, **{m: chosen[m]['test'] for m in METHODS})
    # A non-finite candidate loss (never selected) is stored as null: the records are strict JSON.
    finite = lambda v: None if isinstance(v, float) and not np.isfinite(v) else v
    selection = {m: {k: v for k, v in chosen[m].items() if k != 'test'} for m in METHODS}
    curves = {m: [{k: finite(v) for k, v in r.items() if k != 'test'} for r in rows] for m, rows in candidates.items()}
    dump(folder / 'selection.json', dict(target=target['id'], rotation=rotation, hyperparameters=hp,
                                         selected=selection, validation_curves=curves,
                                         seconds=round(time.time() - started, 1)))
    # Test responses are read only now, for stored predictions.
    predictions = {m: chosen[m]['test'] for m in METHODS}
    predictions['Target only (refit)'] = target_only
    predictions['Pooling (refit)'] = pooling
    scores = pipeline.evaluate_scope(design, bank, target, rotation, model.rows, predictions)
    dump(folder / 'test_scores.json', scores)
    return ti, rotation, {m: f"{chosen[m]['validation_loss']:.3e}" for m in METHODS}




def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--workdir')
    parser.add_argument('--raw-dir')
    parser.add_argument('--archive')
    parser.add_argument('--jobs', type=int)
    parser.add_argument('--collect-only', action='store_true',
                        help='rebuild workdir/results/literature from the stored records of a finished run')
    args = parser.parse_args(argv)
    paths = run.Paths(args)
    if not (paths.workdir / 'catalog.json').exists():
        run.prepare(paths)
    design = run.load_design()
    if args.collect_only:
        collect(paths, design)
        return
    jobs = [(paths.workdir, paths.raw, ti, r) for ti, r in run.scopes(design)]
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.jobs or SPEC['jobs']) as pool:
        for ti, r, info in pool.map(_job, jobs):
            print(f'literature target {ti} rotation {r}: {info} ({time.time()-started:.0f}s)', flush=True)
    collect(paths, design)


def collect(paths, design):
    scores, selections, difference = [], [], {m: [] for m in METHODS}
    for ti, r in run.scopes(design):
        target = design['targets'][ti]
        folder = paths.workdir / 'literature' / pipeline.scope_name(target, r)
        scores.extend(json.loads((folder / 'test_scores.json').read_text()))
        info = json.loads((folder / 'selection.json').read_text())
        for m in METHODS:
            selections.append(dict(target=target['id'], rotation=r, method=m, **info['hyperparameters'],
                                   **{k: v for k, v in info['selected'][m].items()}))
        # How far each test-input prediction is from Target only's (both stored before scoring).
        with np.load(folder / 'predictions_before_test.npz') as stored:
            for m in METHODS:
                difference[m].append(float(np.linalg.norm(stored[m] - stored['target_only'])
                                           / np.linalg.norm(stored['target_only'])))
    frozen = {(row['target'], int(row['rotation']), row['method']): row
              for row in run.read_csv(FROZEN / 'all_test_scores.csv')}
    worst = {'Target only': 0.0, 'Pooling': 0.0}
    for row in scores:
        if row['method'] in ('Target only (refit)', 'Pooling (refit)'):
            method = row['method'].replace(' (refit)', '')
            reference = frozen[(row['target'], int(row['rotation']), method)]
            for band in SCORES:
                worst[method] = max(worst[method], abs(row[band] - float(reference[band])) / float(reference[band]))
    if max(worst.values()) > float(SPEC['pairing_check_rtol']):
        raise AssertionError(f'Refitted Target only/Pooling differ from the frozen scores: {worst}')
    literature = [dict(row, method=METHOD_NAMES[row['method']]) for row in scores if row['method'] in METHODS]
    registered = [row for row in run.read_csv(FROZEN / 'all_test_scores.csv')]
    for row in registered:
        for band in SCORES:
            row[band] = float(row[band])
    every = registered + literature
    methods = list(pipeline.METHODS) + [METHOD_NAMES[m] for m in METHODS]
    output = paths.workdir / 'results/literature'
    output.mkdir(parents=True, exist_ok=True)
    run.write_csv(output / 'all_test_scores.csv', literature)
    run.write_csv(output / 'literature_selection.csv', selections)
    table = pipeline.method_table(every, methods)
    run.write_csv(output / 'method_table.csv', table)
    run.write_csv(output / 'per_target_summary.csv', pipeline.per_target_table(design, every, methods))
    dump(output / 'SUMMARY.json', dict(evaluations=len(literature) // len(METHODS), methods=methods,
                                        bands=list(SCORES),
                                        refit_max_relative_difference=worst,
                                        method_means={row['method']: {b: row[b] for b in SCORES} for row in table},
                                        prediction_difference_to_target_only={
                                            METHOD_NAMES[m]: dict(median=float(np.median(v)), max=float(np.max(v)))
                                            for m, v in difference.items()}))
    for row in table:
        print(f"{row['method']:<36}" + ''.join(f"{row[b]:>12.3e}" for b in SCORES), flush=True)
    print(f'Target only refit vs frozen: max relative difference {worst["Target only"]:.2e}; '
          f'Pooling refit: {worst["Pooling"]:.2e}', flush=True)


if __name__ == '__main__':
    main()

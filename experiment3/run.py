#!/usr/bin/env python3
"""Re-run the real-data benchmark from the raw archive and compare with the frozen results.

Phases (``all`` runs the first five in order):

  prepare    verify/extract data_LPV.zip, build the catalogue, check the frozen design
  meta       864 target-only fits (8 targets x 3 rotations x 36 classes) and the LOTO classes
  select     per rotation: pilot, calibration gates, Unprojected/Projected selection, Pooling;
             predictions on the test input are stored before any test response is read
  evaluate   score the stored predictions on the held-out test responses, write the tables
  compare    compare the re-run with outputs/results

Everything is written under ``--workdir`` (default ``.runs/experiment3``); the
frozen ``outputs/`` are never modified. Run from the repository root with one BLAS thread.
"""
from __future__ import annotations

import os

for _variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_variable, '1')

import argparse  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
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

from transfer_operator.ltv import pipeline  # noqa: E402
from transfer_operator.ltv.catalog import build_catalog, extract_archive  # noqa: E402
from transfer_operator.ltv.design import derive_design  # noqa: E402
from transfer_operator.ltv.records import RecordBank, dump, sha_file  # noqa: E402

CONFIG = yaml.safe_load((EXPERIMENT_ROOT / 'configs/experiment.yaml').read_text())
DESIGN_PATH = EXPERIMENT_ROOT / 'configs/design.json'
OUTPUTS = EXPERIMENT_ROOT / 'outputs/results'
RESULT_FILES = ('representation_meta_cv.csv', 'target_classes.json', 'source_selection.csv',
                'pooling_selection.csv', 'all_test_scores.csv', 'per_target_summary.csv',
                'method_table.csv', 'SUMMARY.json')


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def read_csv(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def load_design():
    if sha_file(DESIGN_PATH) != CONFIG['design_sha256']:
        raise RuntimeError('configs/design.json differs from the frozen design')
    return json.loads(DESIGN_PATH.read_text())


class Paths:
    def __init__(self, args):
        self.workdir = Path(args.workdir or (RELEASE_ROOT / CONFIG['workdir'])).resolve()
        raw = args.raw_dir or os.environ.get('LTV_RAW_DIR')
        self.raw = Path(raw).resolve() if raw else self.workdir / 'raw'
        archive = args.archive or os.environ.get('LTV_ARCHIVE')
        self.archive = Path(archive).resolve() if archive else self.workdir / CONFIG['archive']['name']
        self.jobs = args.jobs or CONFIG['jobs']
        self.workdir.mkdir(parents=True, exist_ok=True)

    def bank(self, design):
        catalog = json.loads((self.workdir / 'catalog.json').read_text())
        arrays = np.load(self.workdir / 'design_arrays.npz')
        return RecordBank(self.raw, catalog, arrays, design['frequency'])


def scopes(design):
    return [(ti, r) for ti in range(len(design['targets'])) for r in range(3)]


# --- prepare ---------------------------------------------------------------------

def prepare(paths):
    design = load_design()
    expected = CONFIG['archive']
    mats = sorted(paths.raw.glob('*.mat'))
    if len(mats) != expected['mat_files']:
        if not paths.archive.is_file():
            raise FileNotFoundError(
                f'Raw .mat files are missing under {paths.raw} and {paths.archive} does not exist. '
                f'Download {expected["url"]} (sha256 {expected["sha256"]}) and pass --archive.')
        digest = sha_file(paths.archive)
        if digest != expected['sha256']:
            raise RuntimeError(f'Archive hash {digest} differs from the frozen {expected["sha256"]}')
        names = extract_archive(paths.archive, paths.raw)
        print(f'extracted {len(names)} containers to {paths.raw}', flush=True)
    catalog = build_catalog(paths.raw, paths.workdir)
    files = {f['file']: f['sha256'] for f in catalog['files']}
    if len(files) != expected['mat_files']:
        raise RuntimeError(f'Expected {expected["mat_files"]} containers, catalogued {len(files)}')
    rules = {k: v for k, v in design.items() if k not in ('targets', 'sources')}
    derived = derive_design(catalog, np.load(paths.workdir / 'design_arrays.npz'), rules)
    same = json.dumps(derived, sort_keys=True) == json.dumps(design, sort_keys=True)
    report = dict(raw_dir=str(paths.raw), containers=len(files), container_sha256=files,
                  design_sha256=sha_file(DESIGN_PATH), derived_design_matches_frozen=same)
    dump(paths.workdir / 'PREPARE.json', report)
    if not same:
        raise RuntimeError('The design derived from the archive differs from configs/design.json')
    print(f'catalogued {len(files)} containers, {len(catalog["groups"])} groups; frozen design reproduced', flush=True)


# --- meta ------------------------------------------------------------------------

def _meta_job(args):
    workdir, raw, ti, r = args
    paths = _paths_from(workdir, raw)
    design = load_design()
    target = design['targets'][ti]
    rows = pipeline.meta_losses(design, paths.bank(design), target, r)
    folder = paths.workdir / 'meta'
    folder.mkdir(exist_ok=True)
    dump(folder / f'{pipeline.scope_name(target, r)}.json', rows)
    return ti, r, len(rows)


def _paths_from(workdir, raw):
    namespace = argparse.Namespace(workdir=str(workdir), raw_dir=str(raw), archive=None, jobs=None)
    return Paths(namespace)


def run_parallel(paths, function, design, label):
    jobs = [(paths.workdir, paths.raw, ti, r) for ti, r in scopes(design)]
    started = time.time()
    with ProcessPoolExecutor(max_workers=paths.jobs) as pool:
        for ti, r, info in pool.map(function, jobs):
            print(f'{label} target {ti} rotation {r}: {info} ({time.time()-started:.0f}s)', flush=True)


def meta(paths):
    design = load_design()
    run_parallel(paths, _meta_job, design, 'meta')
    rows = []
    for ti, r in scopes(design):
        rows.extend(json.loads((paths.workdir / 'meta' / f'{pipeline.scope_name(design["targets"][ti], r)}.json').read_text()))
    if len(rows) != 36*len(scopes(design)):
        raise RuntimeError('Incomplete meta cache')
    write_csv(paths.workdir / 'representation_meta_cv.csv', rows)
    choices = pipeline.choose_classes(design, rows)
    dump(paths.workdir / 'target_classes.json', choices)
    print(json.dumps([dict(target=c['held_out_target'], **c['hyperparameters']) for c in choices], indent=1), flush=True)


# --- select ----------------------------------------------------------------------

def _select_job(args):
    workdir, raw, ti, r = args
    paths = _paths_from(workdir, raw)
    design = load_design()
    target = design['targets'][ti]
    hp = next(c for c in json.loads((paths.workdir / 'target_classes.json').read_text())
              if c['held_out_target'] == target['id'])['hyperparameters']
    bank = paths.bank(design)
    folder = paths.workdir / 'tasks' / pipeline.scope_name(target, r)
    folder.mkdir(parents=True, exist_ok=True)
    selected = pipeline.select_sources(design, bank, target, r, hp)
    pooled = pipeline.pooling_scope(design, bank, target, r, hp)
    if not np.array_equal(selected['rows'], pooled['rows']):
        raise AssertionError('Inconsistent output rows')
    predictions = dict(selected['predictions'], Pooling=pooled['prediction'])
    np.savez_compressed(folder / 'predictions_before_test.npz', rows=selected['rows'],
                        **{m.replace(' ', '_'): p for m, p in predictions.items()})
    np.savez_compressed(folder / 'coefficients.npz', **{m.replace(' ', '_'): z for m, z in selected['coefficients'].items()},
                        Pooling=pooled['coefficients'])
    dump(folder / 'selection.json', dict(
        target=target['id'], rotation=r, hyperparameters=hp,
        target_only_validation_loss=selected['target_only_validation_loss'], gates=selected['gates'],
        passing_sources=selected['passing_sources'], selections=selected['selections'],
        pooling=dict(selected_smooth=pooled['hyperparameters']['smooth'], curves=pooled['curves'],
                     n_source_tasks=pooled['n_source_tasks'], n_source_records=pooled['n_source_records'])))
    summary = {m: (s['sources'], s['scale']) for m, s in selected['selections'].items()}
    return ti, r, f"passing {len(selected['passing_sources'])}, {summary}, pooling smooth {pooled['hyperparameters']['smooth']}"


def select(paths):
    design = load_design()
    run_parallel(paths, _select_job, design, 'select')
    source_rows, pooling_rows = [], []
    for ti, r in scopes(design):
        target = design['targets'][ti]
        info = json.loads((paths.workdir / 'tasks' / pipeline.scope_name(target, r) / 'selection.json').read_text())
        for method in pipeline.TRANSFER_METHODS:
            sel = info['selections'][method]
            source_rows.append(dict(target=target['id'], rotation=r, method=method, **info['hyperparameters'],
                                    target_only_validation_loss=info['target_only_validation_loss'],
                                    passing_sources=info['passing_sources'], sources=sel['sources'],
                                    scale=sel['scale'], validation_loss=sel['validation_loss']))
        pooling_rows.append(dict(target=target['id'], rotation=r, R=info['hyperparameters']['R'],
                                 B=info['hyperparameters']['B'], selected_smooth=info['pooling']['selected_smooth'],
                                 n_source_tasks=info['pooling']['n_source_tasks'],
                                 n_source_records=info['pooling']['n_source_records'],
                                 **{f"validation_full_band_smooth_{c['smooth']:g}": c['validation_full_pe']
                                    for c in info['pooling']['curves']}))
    write_csv(paths.workdir / 'source_selection.csv', source_rows)
    write_csv(paths.workdir / 'pooling_selection.csv', pooling_rows)


# --- evaluate --------------------------------------------------------------------

def _evaluate_job(args):
    workdir, raw, ti, r = args
    paths = _paths_from(workdir, raw)
    design = load_design()
    target = design['targets'][ti]
    folder = paths.workdir / 'tasks' / pipeline.scope_name(target, r)
    with np.load(folder / 'predictions_before_test.npz') as saved:
        rows = saved['rows']
        predictions = {m: saved[m.replace(' ', '_')] for m in pipeline.METHODS}
    scores = pipeline.evaluate_scope(design, paths.bank(design), target, r, rows, predictions)
    dump(folder / 'test_scores.json', scores)
    return ti, r, {s['method']: f"{s['full_band']:.3e}" for s in scores}


def evaluate(paths):
    design = load_design()
    run_parallel(paths, _evaluate_job, design, 'evaluate')
    scores = []
    for ti, r in scopes(design):
        scores.extend(json.loads((paths.workdir / 'tasks' / pipeline.scope_name(design['targets'][ti], r)
                                  / 'test_scores.json').read_text()))
    write_tables(paths.workdir, design, scores)


def write_tables(folder, design, scores):
    folder = Path(folder)
    write_csv(folder / 'all_test_scores.csv', scores)
    table = pipeline.method_table(scores)
    write_csv(folder / 'method_table.csv', table)
    write_csv(folder / 'per_target_summary.csv', pipeline.per_target_table(design, scores))
    classes = json.loads((folder / 'target_classes.json').read_text())
    summary = dict(targets=len(design['targets']), rotations=3, evaluations=len(scores)//len(pipeline.METHODS),
                   methods=list(pipeline.METHODS), bands=list(pipeline.BANDS),
                   target_classes={c['held_out_target']: c['hyperparameters'] for c in classes},
                   method_means={row['method']: {band: row[band] for band in pipeline.BANDS} for row in table})
    dump(folder / 'SUMMARY.json', summary)
    for row in table:
        print(f"{row['method']:<12}" + ''.join(f"{row[band]:>12.3e}" for band in pipeline.BANDS), flush=True)
    return table


# --- compare ---------------------------------------------------------------------

def compare(paths, reference=OUTPUTS):
    rtol = CONFIG['comparison']['rtol']
    report = dict(workdir=str(paths.workdir), reference=str(reference), rtol=rtol, checks={})
    ok = True

    def check(name, passed, detail=None):
        nonlocal ok
        ok &= bool(passed)
        report['checks'][name] = dict(passed=bool(passed), **({'detail': detail} if detail is not None else {}))

    new_classes = json.loads((paths.workdir / 'target_classes.json').read_text())
    old_classes = json.loads((Path(reference) / 'target_classes.json').read_text())
    check('target_classes', [c['hyperparameters'] for c in new_classes] == [c['hyperparameters'] for c in old_classes])
    for name, keys, exact in (('source_selection.csv', ('target', 'rotation', 'method'), ('sources', 'scale')),
                              ('pooling_selection.csv', ('target', 'rotation'), ('selected_smooth',))):
        new = {tuple(r[k] for k in keys): r for r in read_csv(paths.workdir / name)}
        old = {tuple(r[k] for k in keys): r for r in read_csv(Path(reference) / name)}
        check(name, new.keys() == old.keys() and all(
            all(float(new[k][f]) == float(old[k][f]) if f != 'sources' else new[k][f] == old[k][f] for f in exact)
            for k in new))
    new = {(r['target'], r['rotation'], r['method']): r for r in read_csv(paths.workdir / 'all_test_scores.csv')}
    old = {(r['target'], r['rotation'], r['method']): r for r in read_csv(Path(reference) / 'all_test_scores.csv')}
    worst = 0.
    if new.keys() == old.keys():
        for key in new:
            for band in pipeline.BANDS:
                a, b = float(new[key][band]), float(old[key][band])
                worst = max(worst, abs(a-b)/max(abs(b), 1e-300))
    check('all_test_scores.csv', new.keys() == old.keys() and worst <= rtol, dict(max_relative_difference=worst))
    new_table = {r['method']: r for r in read_csv(paths.workdir / 'method_table.csv')}
    old_table = {r['method']: r for r in read_csv(Path(reference) / 'method_table.csv')}
    worst_table = max(abs(float(new_table[m][b])-float(old_table[m][b]))/float(old_table[m][b])
                      for m in old_table for b in pipeline.BANDS) if new_table.keys() == old_table.keys() else float('inf')
    check('method_table.csv', worst_table <= rtol, dict(max_relative_difference=worst_table))
    report['passed'] = ok
    dump(paths.workdir / 'COMPARISON.json', report)
    print(json.dumps(report, indent=1), flush=True)
    return ok


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('phase', choices=['prepare', 'meta', 'select', 'evaluate', 'compare', 'all'])
    parser.add_argument('--workdir')
    parser.add_argument('--raw-dir')
    parser.add_argument('--archive')
    parser.add_argument('--jobs', type=int)
    args = parser.parse_args(argv)
    paths = Paths(args)
    phases = ['prepare', 'meta', 'select', 'evaluate', 'compare'] if args.phase == 'all' else [args.phase]
    for phase in phases:
        started = time.time()
        result = globals()[phase](paths)
        print(f'[{phase}] finished in {time.time()-started:.0f}s', flush=True)
        if phase == 'compare' and not result:
            sys.exit(1)


if __name__ == '__main__':
    main()

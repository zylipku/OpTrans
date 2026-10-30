#!/usr/bin/env python3
"""Audit the relocated experiment, retained tables, and registered bootstrap.

--recompute refits three complete confirmation trajectories using the documented
archive tolerance, and rebuilds four statistical CSVs plus metadata byte for byte.
Manifest checks cover the selected experiment and its listed shared dependencies.
--manifest-only supports any experiment via --reference experimentN/outputs;
the statistical and frozen-fixture checks apply to experiment1.
"""
from pathlib import Path
import argparse
import hashlib
import json
import tempfile

import numpy as np
from _paths import RELEASE_ROOT, DEFAULT_CONFIG, DEFAULT_OUTPUTS
from transfer_operator import ExperimentConfig
from transfer_operator.estimators import candidate_registry
from transfer_operator.reporting import (load_data, methods_for_iterations, mcse,
                                         read_csv, summarize_cube)
from validate_refits import recompute_records, validate_records
from validate_manifests import validate_manifest

STATISTICAL_TABLES = (
    'summary_by_method_n0.csv', 'iteration_vs_first.csv',
    'projection_ablation.csv', 'projection_vs_unprojected.csv',
)


def validate(root=DEFAULT_OUTPUTS, recompute=False, config_directory=DEFAULT_CONFIG,
             check_manifest=True, confirmation_records=None):
    root = Path(root).expanduser().resolve()
    config = ExperimentConfig.from_directory(config_directory)
    design = config.design
    methods = methods_for_iterations(design['iteration_steps'])
    manifest_checks = 0
    if check_manifest:
        manifest_checks = validate_manifest(root/'RELEASE_MANIFEST.json')
    retained = json.loads((Path(__file__).parent/'fixtures/retained_results_sha256.json').read_text())
    for name, digest in retained['files'].items():
        assert hashlib.sha256((root/'results'/name).read_bytes()).hexdigest() == digest, f'Changed retained table: {name}'
    metadata, risks = load_data(root)
    assert risks.shape == (design['confirmation_reps'], len(design['n0_values']), len(methods))
    assert np.isfinite(risks).all() and (risks > 0).all()
    assert tuple(metadata['methods']) == methods
    sizes = metadata['sizes']
    assert sizes == design['n0_values']
    index = {n: i for i, n in enumerate(sizes)}
    summaries = read_csv(root/'results/summary_by_method_n0.csv')
    assert len(summaries) == len(sizes)*len(methods)
    for row in summaries:
        values = risks[:, index[int(row['n0'])], methods.index(row['method_key'])]
        assert float(row['mean']) == float(values.mean())
        assert float(row['mcse']) == mcse(values)
        assert row['complete'] == 'True' and int(row['finite_fits']) == design['confirmation_reps']
    pair_count = 0
    for name in ('iteration_vs_first.csv', 'projection_ablation.csv', 'projection_vs_unprojected.csv'):
        for row in read_csv(root/'results'/name):
            si = index[int(row['n0'])]
            left = risks[:, si, methods.index(row['method_key'])]
            right = risks[:, si, methods.index(row['comparator_key'])]
            delta = left - right
            assert float(row['mean_difference']) == float(delta.mean())
            assert float(row['mcse_difference']) == mcse(delta)
            assert float(row['percentage_improvement']) == float(-100*delta.mean()/right.mean())
            pair_count += 1
    registry = candidate_registry(design)
    dev_schema = json.loads((root/'data/development_schema.json').read_text())
    frozen_rows = read_csv(root/'results/frozen_parameters.csv')
    frozen_index = {(int(r['n0']), r['method']): int(r['index']) for r in frozen_rows}
    dev_checks = 0
    with np.load(root/'data/development_summary.npz', allow_pickle=False) as data:
        for row in frozen_rows:
            n, method = int(row['n0']), row['method']
            key = f'n{n:03d}__{method}'
            mean, se, finite = (data[key+'__'+suffix] for suffix in ('mean','mcse','finite_counts'))
            assert mean.shape == (len(registry[method]),)
            assert np.isfinite(mean).all() and np.isfinite(se).all()
            assert np.all(finite == design['development_reps'])
            def preference(i):
                candidate = registry[method][i]
                lamj = np.inf if candidate['lamj_frac'] is None else candidate['lamj_frac']
                return (mean[i], -lamj, -candidate['lambar_frac'], -candidate['lam0_frac'])
            selected = min(range(len(mean)), key=preference)
            assert selected == int(row['index']) == frozen_index[n,method]
            assert selected == dev_schema['selected_ids'][str(n)][method]
            assert registry[method][selected] == design['selected_candidates'][str(n)][method]
            dev_checks += 1
    assert dev_checks == len(sizes)*len(registry)
    post_first_equal = bool(np.array_equal(risks[:,:,5:], np.repeat(risks[:,:,5:6],len(design['iteration_steps'])-1,axis=2)))
    assert post_first_equal
    gains = read_csv(root/'results/projection_vs_unprojected.csv')
    unfavorable_sizes = [int(r['n0']) for r in gains if float(r['percentage_improvement']) < 0]
    exact_files = []
    confirmation = None
    if recompute:
        records = recompute_records(config, trajectories=3) if confirmation_records is None else confirmation_records
        assert len(records) == 3, "Recompute checks three full confirmation trajectories"
        confirmation = validate_records(records, config, root)
        # Keep even transient work within the user-authorized release directory.
        with tempfile.TemporaryDirectory(prefix='.statistics-', dir=RELEASE_ROOT) as temporary:
            temp = Path(temporary)
            summarize_cube(risks, temp, bootstrap_seed=metadata['bootstrap']['seed'],
                           draws=metadata['bootstrap']['draws'], sizes=sizes,
                           iterations=design['iteration_steps'], settings=design['settings'])
            for name in STATISTICAL_TABLES + ('STATISTICS_METADATA.json',):
                if (temp/name).read_bytes() != (root/'results'/name).read_bytes():
                    raise AssertionError(f'Recomputed file differs: {name}')
                exact_files.append(name)
    return dict(status='PASS', confirmation_shape=list(risks.shape),
                manifest_files_checked=manifest_checks,
                retained_result_files_hash_checked=len(retained['files']),
                method_summary_rows=len(summaries), paired_rows_checked=pair_count,
                development_minimizers_checked=dev_checks,
                projected_steps_1_to_10_exactly_equal=post_first_equal,
                unfavorable_projected_vs_unprojected_sizes=unfavorable_sizes,
                bootstrap_recomputed=recompute, byte_equal_recomputed_files=exact_files,
                confirmation_recomputation=confirmation)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--reference', type=Path, default=DEFAULT_OUTPUTS)
    parser.add_argument('--recompute', action='store_true')
    manifest_mode = parser.add_mutually_exclusive_group()
    manifest_mode.add_argument('--skip-manifest', action='store_true', help='Check content while updating the experiment manifest.')
    manifest_mode.add_argument('--manifest-only', action='store_true', help='Validate only the manifest in --reference, for any experiment.')
    args = parser.parse_args()
    if args.manifest_only:
        if args.recompute:
            parser.error('--manifest-only cannot be combined with --recompute')
        checked = validate_manifest(args.reference.expanduser()/'RELEASE_MANIFEST.json')
        result = dict(status='PASS', manifest_files_checked=checked)
    else:
        result = validate(args.reference, args.recompute, args.config, not args.skip_manifest)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()

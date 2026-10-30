#!/usr/bin/env python3
"""Check reconstructed inputs, original-engine fixtures, and archive tolerance.

The development archive retains aggregate mean/MCSE, so per-world checks use
three fixtures from the unchanged original engine. Original-engine fixtures are
bitwise checks in their recorded Python/NumPy/SciPy environment. The published
confirmation archive uses the documented absolute tolerance of 1e-12. Validation
runs in memory and prints its result; no JSON records are created. Raw-observation
hashes remain exact across environments; hashes of derived floating empirical
moments are informational in the published-archive comparison. The compatible
original-engine fixture comparison remains bitwise for every array.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import sys

import numpy as np
import scipy
from _paths import DEFAULT_CONFIG, DEFAULT_OUTPUTS
from transfer_operator import ExperimentConfig, candidate_registry, run_development
from validate_refits import confirmation_arrays, recompute_records, validate_records

FIXTURES = Path(__file__).resolve().parent/'fixtures'


def numeric_environment():
    return dict(numpy=np.__version__, scipy=scipy.__version__,
                python='.'.join(map(str, sys.version_info[:3])))


def fixture_compatible(metadata):
    expected = metadata['numeric_environment']
    return all(expected[key] == value for key,value in numeric_environment().items())


def validate(config_directory=DEFAULT_CONFIG, reference_directory=DEFAULT_OUTPUTS,
             trajectories=3, confirmation_records=None):
    start = time.monotonic()
    if trajectories != 3:
        raise ValueError('The original-engine fixture contains exactly three confirmation trajectories')
    config = ExperimentConfig.from_directory(config_directory)
    design, model = config.design, config.model
    arrays = [getattr(model, key) for key in ('nu', 'mu', 'a0', 'h', 'thetas')]
    model_digest = hashlib.sha256(b''.join(np.ascontiguousarray(a, dtype='<f8').tobytes() for a in arrays)).hexdigest()
    dictionary_digest = hashlib.sha256(np.ascontiguousarray(model.h, dtype='<f8').tobytes()).hexdigest()
    assert model_digest == design['model_arrays_sha256']
    assert dictionary_digest == design['dictionary_sha256']
    registry = candidate_registry(design)
    grid_digest = hashlib.sha256(json.dumps(registry, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    assert grid_digest == design['candidate_registry_sha256']
    fixture_metadata = json.loads((FIXTURES/'development_baseline.json').read_text())
    baseline_metadata = json.loads((FIXTURES/'confirmation_baseline.json').read_text())
    assert fixture_metadata['source_sha256'] == baseline_metadata['source_sha256']
    # Preserve the original model/dictionary anchor independently of the new YAML.
    for world in fixture_metadata['worlds']:
        original_setting = world['record']['identity']['setting']
        assert model_digest == original_setting['model_arrays_sha256']
        assert dictionary_digest == original_setting['dictionary_sha256']
    development_checks = []
    development_compatible = fixture_compatible(fixture_metadata)
    for world in fixture_metadata['worlds']:
        baseline_path = FIXTURES/world['file']
        assert hashlib.sha256(baseline_path.read_bytes()).hexdigest() == world['sha256']
        if not development_compatible:
            development_checks.append(dict(n0=world['n0'], rep=world['rep'],
                status='SKIPPED_INCOMPATIBLE_FIXTURE_ENVIRONMENT'))
            continue
        values, record = run_development(world['n0'], world['rep'], config)
        assert record == world['record'], f'Development identity/failures differ: {world["n0"]}'
        comparisons = {}
        with np.load(baseline_path, allow_pickle=False) as baseline:
            assert set(values) == set(baseline.files)
            for name, value in values.items():
                expected = baseline[name]
                exact = value.dtype == expected.dtype and value.shape == expected.shape and value.tobytes() == expected.tobytes()
                assert exact, f'Development values differ: {world["n0"]}, {name}'
                comparisons[name] = dict(shape=list(value.shape), bitwise_equal=exact,
                                         max_absolute_difference=float(np.max(np.abs(value-expected))))
        development_checks.append(dict(status='PASS', n0=world['n0'], rep=world['rep'],
            candidate_risks=sum(len(values[m]) for m in registry),
            data_fingerprints_exact=True, arrays=comparisons,
            max_absolute_difference=max(v['max_absolute_difference'] for v in comparisons.values())))
        print(json.dumps(dict(stage='development', n0=world['n0'], max_absolute_difference=0.0)), flush=True)
    records = recompute_records(config, trajectories) if confirmation_records is None else confirmation_records
    assert len(records) == trajectories
    archive = validate_records(records, config, reference_directory)
    original_engine = _original_engine(records, config, reference_directory, baseline_metadata)
    return dict(status='PASS',
        refactor_status='PASS' if development_compatible and original_engine['status']=='PASS' else 'SKIPPED_INCOMPATIBLE_FIXTURE_ENVIRONMENT',
        numeric_environment=numeric_environment(),
        development_fixture_environment=fixture_metadata['numeric_environment'],
        model_arrays_sha256=model_digest,
        dictionary_sha256=dictionary_digest, candidate_registry_sha256=grid_digest,
        development_checks=development_checks,
        development_reference_scope=fixture_metadata['scope'],
        development_reference_source='release/tests/fixtures/development_baseline.json',
        confirmation=dict(archive=archive, original_engine=original_engine),
        elapsed_seconds=time.monotonic()-start)


def _original_engine(records, config, reference_directory, metadata):
    assert len(records) == metadata['trajectories']
    baseline_path = FIXTURES/metadata['file']
    assert hashlib.sha256(baseline_path.read_bytes()).hexdigest() == metadata['sha256']
    if not fixture_compatible(metadata):
        return dict(status='SKIPPED_INCOMPATIBLE_FIXTURE_ENVIRONMENT',
                    numeric_environment=numeric_environment(),
                    fixture_environment=metadata['numeric_environment'])
    arrays, _, _ = confirmation_arrays(records, config, reference_directory)
    comparisons = {}
    with np.load(baseline_path, allow_pickle=False) as baseline:
        assert set(arrays) == set(baseline.files)
        for key, value in arrays.items():
            expected = baseline[key]
            exact = value.dtype == expected.dtype and value.shape == expected.shape and value.tobytes() == expected.tobytes()
            assert exact, f'Reconstructed engine differs from original engine: {key}'
            comparisons[key] = dict(bitwise_equal=True, shape=list(value.shape))
    return dict(status='PASS', source='release/tests/fixtures/confirmation_baseline.npz',
        numeric_environment=numeric_environment(), fixture_environment=metadata['numeric_environment'],
        trajectories=len(records), setting_datasets=len(records)*len(config.design['n0_values']),
        max_absolute_risk_difference=0.0, arrays=comparisons)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--reference', type=Path, default=DEFAULT_OUTPUTS)
    parser.add_argument('--trajectories', type=int, default=3, choices=[3])
    args = parser.parse_args()
    result = validate(args.config, args.reference, args.trajectories)
    archive = result['confirmation']['archive']
    print(json.dumps(dict(status=result['status'],
        development_worlds=len(result['development_checks']), refactor_status=result['refactor_status'],
        trajectories=archive['trajectories'], archive_atol=archive['archive_atol'],
        max_absolute_risk_difference=archive['max_absolute_risk_difference'],
        max_absolute_diagnostic_difference=archive['max_absolute_diagnostic_difference'],
        original_engine_risk_difference=result['confirmation']['original_engine'].get('max_absolute_risk_difference')), indent=2))


if __name__ == '__main__':
    main()

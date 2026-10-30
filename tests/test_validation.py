"""Runnable release checks; share three trajectories across expensive validations."""
from __future__ import annotations

import json

import numpy as np
import pytest
from _paths import DEFAULT_CONFIG, DEFAULT_OUTPUTS
from transfer_operator import ExperimentConfig
from validate_equivalence import validate as validate_equivalence
from validate_estimators import validate as validate_estimators
from validate_refits import ARCHIVE_ATOL, compare_array, recompute_records
from validate_release import validate as validate_release


@pytest.fixture(scope='module')
def confirmation_records():
    return recompute_records(ExperimentConfig.from_directory(DEFAULT_CONFIG), trajectories=3)


def test_archive_tolerance_preserves_strict_schema_and_discrete_values():
    reference = np.array([0., 1., np.nan, np.inf])
    within = reference.copy()
    within[0] = ARCHIVE_ATOL / 2
    comparison = compare_array(within, reference)
    assert comparison['passed'] and not comparison['exact_equal']
    assert comparison['max_absolute_difference'] == ARCHIVE_ATOL / 2
    outside = reference.copy()
    outside[0] = 2 * ARCHIVE_ATOL
    assert not compare_array(outside, reference)['passed']
    # A relative tolerance must not allow an error at a large absolute scale.
    assert not compare_array(np.array([1e6 + 1e-7]), np.array([1e6]))['passed']
    different_nan = reference.copy()
    different_nan[2] = 0.
    assert not compare_array(different_nan, reference)['passed']
    different_inf = reference.copy()
    different_inf[3] = -np.inf
    assert not compare_array(different_inf, reference)['passed']
    assert not compare_array(np.array([2]), np.array([1]))['passed']
    assert not compare_array(np.array([b'changed']), np.array([b'initial']))['passed']
    assert not compare_array(np.ones(2), np.ones((1, 2)))['passed']
    assert not compare_array(np.ones(2, dtype=np.float32), np.ones(2))['passed']
    print('PASS: archive tolerance, exact NaN masks, zero rtol, schemas and discrete values')


def test_archive_fingerprint_policy_preserves_raw_data_and_support_checks(monkeypatch):
    """Derived moment hashes may differ; changed observations/supports still fail."""
    import validate_refits as refits

    with np.load(DEFAULT_OUTPUTS / 'data/confirmation_diagnostics.npz') as source:
        archive = {key: source[key].copy() if key in {'n0', 'step'} else source[key][:1].copy()
                   for key in source.files}
    with np.load(DEFAULT_OUTPUTS / 'data/confirmation_risks.npz') as source:
        archive['risks'] = source['risks'][:1].copy()
    arrays = {key: value.copy() for key, value in archive.items()}
    monkeypatch.setattr(refits, 'confirmation_arrays',
                        lambda *args: ({key: value.copy() for key, value in arrays.items()}, True, 65))
    config = ExperimentConfig.from_directory(DEFAULT_CONFIG)
    for key in refits.MOMENT_FINGERPRINTS:
        arrays[key].flat[0] = b'0' * 64
    result = refits.validate_records([{}], config)
    assert result['status'] == 'PASS'
    assert all(result['raw_data_fingerprints_exact'].values())
    assert not any(result['moment_fingerprints_bitwise_equal'].values())
    for key in refits.RAW_DATA_FINGERPRINTS:
        arrays[key].flat[0] = b'0' * 64
        with pytest.raises(AssertionError, match='archive comparison policy'):
            refits.validate_records([{}], config)
        arrays[key] = archive[key].copy()
    arrays['fixed_support'].flat[0] = ~arrays['fixed_support'].flat[0]
    with pytest.raises(AssertionError, match='archive comparison policy'):
        refits.validate_records([{}], config)
    arrays['fixed_support'] = archive['fixed_support'].copy()
    arrays['risks'].flat[0] = np.nan
    with pytest.raises(AssertionError, match='archive comparison policy'):
        refits.validate_records([{}], config)


def test_independent_estimator_equations():
    result = validate_estimators()
    assert result['status'] == 'PASS'
    print(json.dumps(dict(status=result['status'], dense_baselines=sum(
        len(s['baseline']) for s in result['settings']), iteration_comparisons=sum(
        len(s['iterations']) for s in result['settings']))))


def test_original_engine_equivalence_and_archive(confirmation_records):
    result = validate_equivalence(confirmation_records=confirmation_records)
    assert result['status'] == 'PASS'
    archive = result['confirmation']['archive']
    print(json.dumps(dict(status=result['status'], refactor_status=result['refactor_status'],
        trajectories=archive['trajectories'],
        max_absolute_risk_difference=archive['max_absolute_risk_difference'],
        max_absolute_diagnostic_difference=archive['max_absolute_diagnostic_difference'])))


def test_release_manifest_selections_and_rebuilt_tables(confirmation_records):
    result = validate_release(recompute=True, confirmation_records=confirmation_records)
    assert result['status'] == 'PASS'
    print(json.dumps(dict(status=result['status'],
        manifest_files_checked=result['manifest_files_checked'],
        frozen_selections=result['development_minimizers_checked'],
        byte_equal_recomputed_files=result['byte_equal_recomputed_files'])))

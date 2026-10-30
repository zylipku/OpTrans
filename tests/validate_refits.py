"""Compare regenerated confirmation fits with the immutable published reference.

Floating risks and diagnostics use atol=1e-12, rtol=0, with exact NaN masks.
Discrete diagnostics, raw-observation fingerprints, random identities, and
selections are exact. Operator and empirical-moment fingerprints describe
floating calculations and are reported without requiring cross-environment byte
equality. No validation files are written.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from _paths import DEFAULT_CONFIG, DEFAULT_OUTPUTS
from transfer_operator import ExperimentConfig, run_trajectory
from transfer_operator.diagnostics import compact_diagnostics, records_from

ARCHIVE_ATOL = 1e-12
OPERATOR_FINGERPRINTS = {'fit_operator_sha256', 'iteration_operator_sha256'}
MOMENT_FINGERPRINTS = {'source_covariance_sha256', 'source_response_moment_sha256'}
RAW_DATA_FINGERPRINTS = {'full_task_data_sha256', 'task_prefix_data_sha256', 'target_data_sha256'}


def compare_array(values, target, *, floating_tolerance=True):
    """Return an auditable comparison; do not silently broadcast or cast arrays."""
    values, target = np.asarray(values), np.asarray(target)
    schema_equal = values.shape == target.shape and values.dtype == target.dtype
    if not schema_equal:
        return dict(shape=list(values.shape), passed=False, schema_equal=False,
                    exact_equal=False)
    exact = values.tobytes() == target.tobytes()
    row = dict(shape=list(values.shape), schema_equal=True, exact_equal=exact)
    if values.dtype.kind == 'f' and floating_tolerance:
        nan_mask_equal = np.array_equal(np.isnan(values), np.isnan(target))
        finite = np.isfinite(values) & np.isfinite(target)
        difference = float(np.max(np.abs(values[finite] - target[finite]))) if finite.any() else 0.0
        row.update(passed=bool(nan_mask_equal and np.allclose(
            values, target, atol=ARCHIVE_ATOL, rtol=0.0, equal_nan=True)),
            nan_mask_equal=nan_mask_equal, max_absolute_difference=difference,
            atol=ARCHIVE_ATOL, rtol=0.0)
    else:
        row['passed'] = exact
    return row


def confirmation_arrays(records, config, reference_directory=DEFAULT_OUTPUTS):
    """Check complete registered records and assemble archive-compatible arrays."""
    expected = len(records)
    if not 1 <= expected <= config.design['confirmation_reps']:
        raise ValueError('Invalid trajectory count')
    if sorted(r['trajectory'] for r in records) != list(range(expected)):
        raise ValueError('Missing, extra, or duplicate trajectory identities')
    if any(r['status'] != 'ok' for r in records):
        raise ValueError('Failed fits cannot be dropped from validation')
    schema = json.loads((Path(reference_directory)/'data/confirmation_schema.json').read_text())
    assert schema['sizes'] == config.design['n0_values']
    risks = np.empty((expected, len(schema['sizes']), len(schema['methods'])))
    primary_agrees = True
    candidate_checks = 0
    for record in records:
        if record['exp_id'] != config.design['confirmation_exp_id']:
            raise ValueError('Incorrect confirmation random identity')
        if record['per_replication_selection'] or record['support_reselection']:
            raise ValueError('Confirmation changed selection rules')
        settings = {s['n0']: s for s in record['settings']}
        if len(settings) != len(record['settings']) or set(settings) != set(schema['sizes']):
            raise ValueError('Each trajectory must contain every registered setting exactly once')
        for ni, n0 in enumerate(schema['sizes']):
            setting = settings[n0]
            chosen = config.design['selected_candidates'][str(n0)]
            for method, fit in setting['fits'].items():
                candidate = chosen['projected' if method == 'projection_removed' else method]
                assert fit['candidate'] == candidate, f'Frozen candidate differs: {n0}, {method}'
                candidate_checks += 1
            step = {v['step']: v for v in setting['iterations']}
            if 1 in step:
                primary_agrees &= (setting['fits']['projected']['risk'] == step[1]['risk'] and
                                   setting['fits']['projected']['operator_sha256'] == step[1]['operator_sha256'])
            for mi, method in enumerate(schema['methods']):
                fit = step[int(method.split('_')[1])] if method.startswith('projected_') else setting['fits'][method]
                risks[record['trajectory'], ni, mi] = fit['risk']
    assert np.isfinite(risks).all(), 'Nonfinite recomputed risk'
    arrays = compact_diagnostics(records, config, repetitions=expected)
    arrays['risks'] = risks
    return arrays, primary_agrees, candidate_checks


def validate_records(records, config, reference_directory=DEFAULT_OUTPUTS):
    reference_directory = Path(reference_directory).expanduser().resolve()
    arrays, primary_agrees, candidate_checks = confirmation_arrays(records, config, reference_directory)
    expected = len(records)
    risks = arrays.pop('risks')
    with np.load(reference_directory/'data/confirmation_risks.npz', allow_pickle=False) as reference:
        risk_comparison = compare_array(risks, reference['risks'][:expected])
    comparisons = {}
    with np.load(reference_directory/'data/confirmation_diagnostics.npz', allow_pickle=False) as reference:
        if set(reference.files) != set(arrays):
            raise ValueError('Diagnostic schema changed')
        for key, values in arrays.items():
            target = reference[key] if key in {'n0', 'step'} else reference[key][:expected]
            row = compare_array(values, target)
            if key in OPERATOR_FINGERPRINTS:
                row['comparison'] = 'informational_float_coefficient_fingerprint'
                row['passed'] = row['schema_equal']
            elif key in MOMENT_FINGERPRINTS:
                # These hash f.T @ f / n and y @ f / n, not the sampled data.
                # BLAS reduction roundoff changes their bytes across environments.
                # The full and prefix raw-observation hashes remain exact checks.
                row['comparison'] = 'informational_float_moment_fingerprint'
                row['passed'] = row['schema_equal']
            comparisons[key] = row
    passed = risk_comparison['passed'] and primary_agrees and all(r['passed'] for r in comparisons.values())
    result = dict(status='PASS' if passed else 'MISMATCH',
        trajectories=expected, setting_datasets=expected*len(config.design['n0_values']),
        risk_cells=int(risks.size), risk_bitwise_equal=risk_comparison['exact_equal'],
        max_absolute_risk_difference=risk_comparison.get('max_absolute_difference'),
        max_absolute_diagnostic_difference=max(
            (r.get('max_absolute_difference', 0.0) for r in comparisons.values()), default=0.0),
        archive_atol=ARCHIVE_ATOL, archive_rtol=0.0,
        frozen_candidates_exact_checked=candidate_checks,
        projected_baseline_equals_step1=bool(primary_agrees), diagnostics=comparisons,
        operator_fingerprints_bitwise_equal={k:comparisons[k]['exact_equal'] for k in sorted(OPERATOR_FINGERPRINTS)},
        moment_fingerprints_bitwise_equal={k:comparisons[k]['exact_equal'] for k in sorted(MOMENT_FINGERPRINTS)},
        raw_data_fingerprints_exact={k:comparisons[k]['exact_equal'] for k in sorted(RAW_DATA_FINGERPRINTS)},
        scope='Full registered confirmation' if expected == config.design['confirmation_reps'] else 'Partial confirmation check',
        numeric_comparison='Floating arrays: atol=1e-12, rtol=0, exact NaN masks. Discrete values, selections and raw-observation fingerprints: exact. Floating operator and empirical-moment fingerprints: informational.')
    assert passed, 'Regenerated fits exceed the archive comparison policy: ' + json.dumps(result)
    return result


def recompute_records(config, trajectories=3):
    records = []
    for trajectory in range(trajectories):
        record = run_trajectory(trajectory, config)
        records.append(record)
        print(json.dumps(dict(stage='confirmation', trajectory=trajectory,
                              settings=len(record['settings']))), flush=True)
    return records


def validate(output_directory, expected=1000, config_directory=DEFAULT_CONFIG,
             reference_directory=DEFAULT_OUTPUTS):
    output = Path(output_directory).expanduser().resolve()
    reference = Path(reference_directory).expanduser().resolve()
    if output == reference or reference in output.parents:
        raise ValueError('Regenerated records must be outside the retained reference outputs')
    records = list(records_from(output/'records'))
    if len(records) != expected:
        raise ValueError('Unexpected trajectory count')
    return validate_records(records, ExperimentConfig.from_directory(config_directory), reference)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-directory', type=Path, required=True)
    parser.add_argument('--expected-trajectories', type=int, default=1000)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--reference', type=Path, default=DEFAULT_OUTPUTS)
    args = parser.parse_args()
    result = validate(args.output_directory, args.expected_trajectories, args.config, args.reference)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

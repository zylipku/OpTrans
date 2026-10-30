"""Compact, lossless numeric diagnostics from the registered confirmation runs.

Undefined initial-step changes and non-LS diagnostics have a zero placeholder;
the explicit applicability masks distinguish these from observed zeros.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np

FIT_METHODS = ('pilot', 'pooling', 'unprojected', 'projected', 'projection_removed')


def penalty_labels(sources):
    return ('lam0', 'lambar', 'eta', 'lam') + tuple(f'lam_j{j}' for j in range(1, sources + 1))


def compact_diagnostics(records, config, repetitions=None):
    """Consume original-like trajectory records in any order; reject gaps."""
    repetitions = config.design['confirmation_reps'] if repetitions is None else repetitions
    sizes, steps = config.design['n0_values'], config.design['iteration_steps']
    sources, dictionary = config.model.m, config.model.p
    nt, nm = len(steps), len(FIT_METHODS)
    penalties = penalty_labels(sources)
    rank_dtype = np.uint8 if dictionary <= np.iinfo(np.uint8).max else np.uint64
    shape = (repetitions, len(sizes))
    arrays = dict(
        trajectory=np.arange(repetitions), n0=np.asarray(sizes), step=np.asarray(steps),
        full_task_data_sha256=np.empty((repetitions, sources + 1), dtype='S64'),
        task_prefix_data_sha256=np.empty(shape + (sources + 1,), dtype='S64'),
        source_covariance_sha256=np.empty(shape + (sources,), dtype='S64'),
        source_response_moment_sha256=np.empty(shape + (sources,), dtype='S64'),
        target_data_sha256=np.empty(shape, dtype='S64'),
        fit_c=np.zeros(shape + (nm,)),
        fit_operator_sha256=np.empty(shape + (nm,), dtype='S64'),
        fit_penalties=np.zeros(shape + (nm, len(penalties))),
        fit_penalty_applicable=np.zeros(shape + (nm, len(penalties)), dtype=bool),
        fixed_support=np.zeros(shape + (sources, dictionary), dtype=bool),
        no_transfer=np.zeros(shape, dtype=bool),
        iteration_c=np.zeros(shape + (nt,)),
        iteration_operator_sha256=np.empty(shape + (nt,), dtype='S64'),
        iteration_theta=np.zeros(shape + (nt, sources, dictionary)),
        projection_ranks=np.zeros(shape + (nt, sources), dtype=rank_dtype),
        refit_projection_ranks=np.zeros(shape + (nt, sources), dtype=rank_dtype),
        has_refit_projection_ranks=np.zeros(shape + (nt,), dtype=bool),
        coefficient_change_norm=np.zeros(shape + (nt,)),
        risk_change=np.zeros(shape + (nt,)),
        has_predecessor=np.zeros(shape + (nt,), dtype=bool),
        has_ls_diagnostics=np.zeros(shape + (nt, sources), dtype=bool),
        calibration_score_l2=np.zeros(shape + (nt, sources)),
        normalized_projection_leakage=np.zeros(shape + (nt, sources)),
        calibration_moment_norm2=np.zeros(shape + (nt, sources)),
        calibration_numerical_rank=np.zeros(shape + (nt, sources), dtype=rank_dtype),
        singular_min_max_ratio=np.zeros(shape + (nt, sources)),
    )
    seen = set()
    for record in records:
        i = int(record['trajectory'])
        if i in seen or not 0 <= i < repetitions or record['status'] != 'ok':
            raise ValueError('Duplicate, out-of-range, or failed confirmation identity')
        seen.add(i)
        arrays['full_task_data_sha256'][i] = record['full_task_data_sha256']
        settings = {s['n0']: s for s in record['settings']}
        if set(settings) != set(sizes):
            raise ValueError('Each trajectory must contain all registered settings')
        for ni, n0 in enumerate(sizes):
            setting = settings[n0]
            for key in ('task_prefix_data_sha256', 'source_covariance_sha256',
                        'source_response_moment_sha256', 'target_data_sha256'):
                arrays[key][i, ni] = setting[key]
            for mi, method in enumerate(FIT_METHODS):
                fit = setting['fits'][method]
                arrays['fit_c'][i, ni, mi] = fit['c']
                arrays['fit_operator_sha256'][i, ni, mi] = fit['operator_sha256']
                info = fit['info']
                penalty = [info.get(k) for k in penalties[:4]]
                penalty += info.get('lam_js', [None] * sources)
                for pi, value in enumerate(penalty):
                    if value is not None:
                        arrays['fit_penalties'][i, ni, mi, pi] = value
                        arrays['fit_penalty_applicable'][i, ni, mi, pi] = True
            diag = setting['iteration_diagnostics']
            arrays['no_transfer'][i, ni] = diag['no_transfer']
            for j, support in enumerate(diag['fixed_supports']):
                arrays['fixed_support'][i, ni, j, support] = True
            if [x['step'] for x in setting['iterations']] != list(steps):
                raise ValueError('Missing or reordered iteration states')
            for state in setting['iterations']:
                k = state['step']
                index = (i, ni, k)
                arrays['iteration_c'][index] = state['c']
                arrays['iteration_operator_sha256'][index] = state['operator_sha256']
                arrays['iteration_theta'][index] = state['theta']
                arrays['projection_ranks'][index] = state['projection_ranks']
                if state['refit_projection_ranks'] is not None:
                    arrays['refit_projection_ranks'][index] = state['refit_projection_ranks']
                    arrays['has_refit_projection_ranks'][index] = True
                if state['coefficient_change_norm'] is not None:
                    arrays['coefficient_change_norm'][index] = state['coefficient_change_norm']
                    arrays['risk_change'][index] = state['risk_change']
                    arrays['has_predecessor'][index] = True
                for j, d in enumerate(state['calibration_diagnostics']):
                    q = index + (j,)
                    arrays['has_ls_diagnostics'][q] = True
                    for out, key in [('calibration_score_l2', 'score_l2'),
                                     ('normalized_projection_leakage', 'normalized_generator_projection_leakage'),
                                     ('calibration_moment_norm2', 'moment_norm2'),
                                     ('calibration_numerical_rank', 'numerical_rank'),
                                     ('singular_min_max_ratio', 'singular_min_max_ratio')]:
                        arrays[out][q] = d[key]
    if seen != set(range(repetitions)):
        raise ValueError('Incomplete confirmation trajectories')
    return arrays


def records_from(directory):
    for path in sorted(Path(directory).glob('trajectory_*.json')):
        yield json.loads(path.read_text())


def schema(arrays):
    sources, dictionary = arrays['fixed_support'].shape[-2:]
    return dict(
        axes=dict(trajectory='independent confirmation replication', n0=arrays['n0'].tolist(),
                  step=arrays['step'].tolist(), fit_method=list(FIT_METHODS),
                  task=['target'] + [f'source_{j}' for j in range(1, sources + 1)],
                  source=list(range(1, sources + 1)),
                  dictionary_column=list(range(dictionary)), penalty=list(penalty_labels(sources))),
        arrays={k: dict(shape=list(v.shape), dtype=str(v.dtype)) for k, v in arrays.items()},
        fingerprints='SHA-256 of contiguous float64 byte representations, same order as original samplers',
        undefined_values='Zero placeholders only where the corresponding applicability mask is false.',
        theta_semantics='theta[0] is initial Lasso; theta[k>=1] is restricted unpenalized LS at A[k].',
        initial_support='Boolean source-by-dictionary masks, held fixed for every iteration.',
        observations='Raw observation matrices are regenerated from registered streams; they are not duplicated here.',
    )


def array_fingerprint(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()

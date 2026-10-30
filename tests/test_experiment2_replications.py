"""Replay selected paper replications, including an actual support-selection failure."""
import numpy as np
import pytest
import yaml

from _paths import RELEASE_ROOT
from transfer_operator.pde_records import load_pde_record
from transfer_operator.pde_replication import advection_replicate

OUTPUTS = RELEASE_ROOT/'experiment2/outputs'


def compare(actual, expected, *, risk=False):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    if risk:
        # Unchanged reference source shows the same BLAS-level archive drift.
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0, equal_nan=False)
    else:
        np.testing.assert_array_equal(actual, expected)
    return float(np.max(np.abs(actual-expected), initial=0.))


@pytest.mark.parametrize('cell,rep', [(0, 0), (2, 6), (9, 0)])
def test_registered_advection_including_support_failure(cell, rep):
    config = yaml.safe_load((RELEASE_ROOT/'experiment2/configs/advection_sigma.yaml').read_text())
    n0, sigma = next((n, sigma) for n in config['n0_grid']
                     for index, sigma in zip(config['archive_cells'][n], config['sigma_grid'])
                     if index == cell)
    setting = dict(R=config['radius'], s=config['smoothness'], sigma=sigma, tau=config['tau'],
                   n0=n0, nS=config['source_total'], profile=config['noise_profile'])
    candidates = [dict(config['frozen_multipliers'][method], role=role)
                  for method, role in [('Projected', 'projected'), ('Unprojected', 'unprojected')]]
    result = advection_replicate(setting, candidates, config['seed'], rep, additional_rounds=2)
    differences = []
    saved = load_pde_record(OUTPUTS/'data/advection_sigma'/f'cell{cell:03d}.npz', replicate=rep)
    for candidate, row in enumerate(result['candidates']):
        indices, = np.nonzero((saved['method'] == 'projected') & (saved['candidate'] == candidate))
        index, = indices
        for key in ['lam0', 'eta', 'ridge', 'neff']:
            compare(row[key], saved[key][index])
        for key in ['projection_off_risk', 'risk0']:
            differences.append(compare(row[key], saved[key][index], risk=True))
        assert row['kkt'] < 1e-10
        for k in range(1, 4):
            if f'risk{k}' in row:
                differences.append(compare(row[f'risk{k}'], saved[f'risk{k}'][index], risk=True))
        for j, support in enumerate(row['supports'], 1):
            encoded = ''.join(str(k+1) for k in support) or 'empty'
            original = str(saved[f'j{j}_support'][index])
            assert encoded == (original if original == 'empty' else str(int(float(original))))
    for (method, multiplier), risk in result['baselines'].items():
        index, = np.flatnonzero((saved['method'] == method) & (saved['multiplier'] == multiplier))
        differences.append(compare(risk, saved['risk'][index], risk=True))
    if cell == 2 and rep == 6:
        assert result['candidates'][0]['supports'] == [(0,), (1,)]
        assert result['candidates'][1]['supports'] == [(0, 1), (1,)]
    print(f'advection cell={cell} rep={rep}: {len(differences)} risks within 1e-12; tuning/supports exact, '
          f'max diff={max(differences)}')

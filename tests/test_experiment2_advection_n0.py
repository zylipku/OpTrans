"""Advection-diffusion n0 sweep: records, summary, selected numerical replays."""
import json

import numpy as np
import pytest
import yaml

from _paths import RELEASE_ROOT
from experiment2 import advection_n0 as sweep

OUTPUTS = RELEASE_ROOT / 'experiment2/outputs'
CONFIG = yaml.safe_load((RELEASE_ROOT / 'experiment2/configs/advection_n0.yaml').read_text())
N0 = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]



def records():
    with np.load(OUTPUTS / sweep.RESULTS / 'risks.npz') as saved:
        return ({n0: saved[f'n0_{n0}'] for n0 in N0}, {n0: saved[f'exact_{n0}'] for n0 in N0},
                list(saved['columns']))


def summary():
    return json.loads((OUTPUTS / sweep.RESULTS / 'SUMMARY.json').read_text())


def test_summary_recomputes_from_the_records():
    risks, exact, columns = records()
    assert columns == list(sweep.COLUMNS) and CONFIG['n0_grid'] == N0
    assert all(risks[n0].shape == (300, len(sweep.COLUMNS)) and np.all(np.isfinite(risks[n0])) and np.all(risks[n0] > 0) for n0 in N0)
    stored = summary()
    expected = sweep.summarize(CONFIG, risks, exact)
    assert [c['setting']['n0'] for c in stored['cells']] == N0
    for cell, exp in zip(stored['cells'], expected['cells']):
        assert cell['setting'] == exp['setting'] and cell['support'] == exp['support'] and cell['replicates'] == 300
        for column in sweep.COLUMNS:
            assert cell['risk'][column]['mean'] == exp['risk'][column]['mean'], (cell['setting']['n0'], column)
            assert cell['risk'][column]['ci95'] == exp['risk'][column]['ci95'], (cell['setting']['n0'], column)
    assert stored['config']['frozen_multipliers'] == CONFIG['frozen_multipliers']
    assert stored['config']['seed'] == CONFIG['seed'] and stored['config']['bootstrap_seed'] == CONFIG['bootstrap']['seed']


@pytest.mark.parametrize('n0', [16, 4096])
def test_registered_replication_recomputes_with_archive_tolerance(n0):
    risks, exact, _ = records()
    row, ex = sweep.replicate(CONFIG, n0, 0)
    # The unchanged source has the same last-bit drift in the current BLAS.
    np.testing.assert_allclose(row, risks[n0][0], atol=1e-12, rtol=0)
    print(f"n0={n0}: max archive risk difference {np.max(np.abs(row-risks[n0][0])):.3g}")
    np.testing.assert_array_equal(ex, exact[n0][0])


def test_recalibration_and_support_pattern():
    s = summary()
    p = np.array([c['risk']['projected_k1']['mean'] for c in s['cells']])
    rc1 = np.array([c['risk']['projected_k2']['mean'] for c in s['cells']])
    rc2 = np.array([c['risk']['projected_k3']['mean'] for c in s['cells']])
    u = np.array([c['risk']['unprojected']['mean'] for c in s['cells']])
    assert np.all(rc1 < p) and np.all(u > p)
    reduction = 100 * (1 - rc1 / p)
    assert np.all(np.diff(reduction) < 0)                  # shrinks with n0
    assert [round(x, 1) for x in reduction[:4]] == [26.3, 11.0, 5.0, 2.3]
    assert np.all(reduction[4:] < 1)
    np.testing.assert_allclose(rc2, rc1, rtol=2e-3, atol=0)
    assert all(c['support']['projected_exact'] == 300 for c in s['cells'])
    assert [c['support']['unprojected_exact'] for c in s['cells']] == [108, 140, 184, 233, 278, 294, 299, 300, 300]

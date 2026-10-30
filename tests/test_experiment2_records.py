"""Retained advection records, paired confidence intervals, and final-paper grids."""
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path

import numpy as np
import yaml

import _paths
from transfer_operator.pde_replication import bootstrap_resample_means
from transfer_operator.pde_records import PackedRecords

RELEASE = Path(__file__).resolve().parents[1]
OUT = RELEASE / 'experiment2/outputs'
CONFIGS = RELEASE / 'experiment2/configs'
ADVECTION_CELLS = (0, 1, 2, 3, 6, 7, 8, 9)
COLUMNS = ('projected_k0', 'projected_k1', 'projected_k2', 'projected_k3',
           'unprojected', 'target-only', 'pooling')
SIGMA_FIELDS = {
    'candidate', 'cell', 'replicate', 'method', 'multiplier', 'success',
    'lam0', 'eta', 'ridge', 'neff', 'risk', 'risk0', 'risk1', 'risk2', 'risk3',
    'projection_off_risk', 'j1_support', 'j2_support', 'exact_all',
}


@lru_cache(None)
def advection_records(cell):
    """Resolve each stored candidate identity, without discarding support failures."""
    rows, exact = [], []
    packed = PackedRecords(OUT / 'data/advection_sigma' / f'cell{cell:03d}.npz')
    assert len(packed) == 300
    for rep in range(len(packed)):
        archive = packed.record(rep)
        methods, candidates = archive['method'], archive['candidate']
        p = np.flatnonzero((methods == 'projected') & (candidates == 0)).item()
        u = np.flatnonzero((methods == 'projected') & (candidates == 1)).item()
        t = np.flatnonzero(methods == 'target-only').item()
        pool = np.flatnonzero(methods == 'pooling').item()
        assert np.all(archive['success'] == 1)
        assert np.all(archive['replicate'] == rep)
        assert np.all(archive['cell'] == cell)
        row = [archive[f'risk{k}'][p] for k in range(4)]
        row += [archive['projection_off_risk'][u], archive['risk'][t], archive['risk'][pool]]
        rows.append(row)
        exact.append([archive['exact_all'][p], archive['exact_all'][u]])
    risks = np.asarray(rows)
    config = yaml.safe_load((CONFIGS / 'advection_sigma.yaml').read_text())
    boot = bootstrap_resample_means(risks, config['bootstrap']['draws'],
                                    config['bootstrap']['seed'], algorithm='indices')
    document = json.loads((OUT / 'results/advection_sigma/SIGMA_SUMMARY.json').read_text())
    summary = next(entry for entry in document['cells'] if entry['cell'] == cell)
    return summary, risks, boot, np.asarray(exact)


def test_advection_means_intervals_and_actual_support_counts():
    counts = []
    for cell in ADVECTION_CELLS:
        summary, risks, boot, exact = advection_records(cell)
        # Preserve original per-column reductions from the registered analysis.
        means = np.array([risks[:, i].mean() for i in range(len(COLUMNS))])
        intervals = np.quantile(boot, [.025, .975], axis=0).T
        expected_means = np.array([summary['risk'][name]['mean'] for name in COLUMNS])
        expected_intervals = np.array([[summary['risk'][name]['lo'], summary['risk'][name]['hi']]
                                      for name in COLUMNS])
        np.testing.assert_allclose(means, expected_means, atol=1e-17, rtol=0)
        np.testing.assert_allclose(intervals, expected_intervals, atol=1e-16, rtol=0)
        counts.append(exact.sum(axis=0).astype(int).tolist())
    assert counts == [[300, 300], [300, 300], [300, 281], [300, 155],
                      [300, 300], [300, 300], [300, 300], [300, 264]]


def test_packed_records_cover_exactly_the_paper_cells_without_pickle():
    directory = OUT / 'data/advection_sigma'
    assert sorted(p.name for p in directory.glob('*.npz')) == [f'cell{c:03d}.npz' for c in ADVECTION_CELLS]
    for cell in ADVECTION_CELLS:
        path = directory / f'cell{cell:03d}.npz'
        packed = PackedRecords(path)
        assert len(packed) == 300
        assert set(packed.fields) == SIGMA_FIELDS
        assert len(packed.fields) == len(SIGMA_FIELDS)
        with np.load(path, allow_pickle=False) as archive:
            assert set(archive.files) == SIGMA_FIELDS | {'__packed_records__'}
            assert all(not archive[field].dtype.hasobject for field in archive.files)
            assert all(archive[field].shape[0] == 300 for field in packed.fields)


def test_configs_cover_only_final_paper_grids_and_frozen_values():
    assert sorted(p.name for p in CONFIGS.glob('*.yaml')) == [
        'advection_n0.yaml', 'advection_sigma.yaml', 'literature.yaml']
    sigma = yaml.safe_load((CONFIGS / 'advection_sigma.yaml').read_text())
    n0 = yaml.safe_load((CONFIGS / 'advection_n0.yaml').read_text())
    assert sigma['sigma_grid'] == [.05, .1, .25, .5]
    assert sigma['n0_grid'] == [256, 1024]
    assert tuple(c for n in sigma['n0_grid'] for c in sigma['archive_cells'][n]) == ADVECTION_CELLS
    assert n0['n0_grid'] == [16,32,64,128,256,512,1024,2048,4096]
    assert n0['sigma'] == .25
    for config in (sigma, n0):
        assert config['radius'] == 12 and config['smoothness'] == 2
        assert config['source_total'] == 32768 and config['replications'] == 300
        assert config['bootstrap']['draws'] == 20000
        assert config['risk'] == 'retained_galerkin' and config['noise_profile'] == 'full'
        expected = {'Projected': dict(c0=.003, cbar=.003, ctheta=.02),
                    'Unprojected': dict(c0=.003, cbar=.001, ctheta=.005)}
        for method, values in expected.items():
            assert {k: config['frozen_multipliers'][method][k] for k in values} == values
        assert config['frozen_multipliers']['Target_only'] == .001
        assert config['frozen_multipliers']['Pooling'] == 0.

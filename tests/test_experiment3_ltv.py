"""Experiment 3 (real-data LTV benchmark): numerics, frozen design, stored tables."""
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.io import loadmat, savemat

from _paths import RELEASE_ROOT
from transfer_operator.ltv import banded, harmonic, pipeline, pooling
from transfer_operator.ltv.design import class_grid
from transfer_operator.ltv.records import read_mat_column, period_spectrum

EXPERIMENT = RELEASE_ROOT / 'experiment3'
RESULTS = EXPERIMENT / 'outputs/results'
CONFIG = yaml.safe_load((EXPERIMENT / 'configs/experiment.yaml').read_text())
DESIGN = json.loads((EXPERIMENT / 'configs/design.json').read_text())
# Table 1 of the final paper (mean over 24 held-out evaluations), as printed.
PAPER_TABLE = {
    'Target only': (4.17e-5, 1.02e-6, 3.47e-4),
    'Pooling': (1.12e-1, 1.87e-2, 7.01e-1),
    'Unprojected': (2.67e-5, 6.42e-7, 2.06e-4),
    'Projected': (2.65e-5, 5.98e-7, 2.06e-4),
}


def read_csv(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def small_model(R=3, B=8):
    return banded.Harmonic(2048, 2048., R, B, lo=30., hi=970., wc=300.)


def synthetic_records(model, seed, count):
    rng = np.random.default_rng(seed)
    records = []
    for _ in range(count):
        F = rng.normal(size=model.n//2+1) + 1j*rng.normal(size=model.n//2+1)
        V = .5*F + .02*np.roll(F, 1) + .03*np.roll(F, -3)
        Y = 1j*model.omega*V/model.wc
        records.append(dict(F=F, V=V, Y=Y, ref=dict(file='MS_synthetic.mat', column=0, forced_lines=[40, 50])))
    return records


def dense_design(model, F):
    """X[k, (r, b)] = factor_k * phi[k, b] * F[k - r], matching Harmonic.predict."""
    return (model.factor[:, None, None] * F[model.indices][:, :, None] * model.phi[:, None, :]).reshape(len(model.rows), -1)


def dense_regularizer(model, h, smooth):
    scale = h.trace()/h.p
    diff = np.diff(np.eye(model.B), 2, axis=0)
    penalty = np.kron(np.eye(model.d), diff.T @ diff)
    return scale*smooth*penalty + scale*1e-12*np.eye(model.p)


# --- numerics ------------------------------------------------------------------

def test_banded_solver_matches_dense_normal_equations():
    model = small_model()
    train = synthetic_records(model, 1, 1)[0]
    h, b = banded.target_stats(model, [train])
    X = dense_design(model, train['F'])
    y = train['Y'][model.rows]
    for smooth in (1e-8, 1e-4):
        z = banded.fit_quadratic(h, b, model.regularizer(h, smooth))
        beta = np.linalg.solve(X.conj().T @ X + dense_regularizer(model, h, smooth), X.conj().T @ y)
        np.testing.assert_allclose(harmonic.unpack(z), beta, rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(model.predict(train['F'], z), X @ beta, rtol=1e-9, atol=1e-12)
    assert abs(h.trace() - np.trace(X.conj().T @ X).real) <= 1e-9*h.trace()


def test_projection_downdate_matches_dense_real_solve():
    model = small_model()
    records = synthetic_records(model, 2, 3)
    pilot = banded.fit_quadratic(*banded.target_stats(model, records[:1]), model.regularizer(banded.target_stats(model, records[:1])[0], 1e-6))
    stats = banded.source_stats(model, records, pilot, ['dc', 'cos1', 'sin1'], theta=np.r_[.01, .02, -.01, np.zeros(6)])
    assert stats['projection_rank'] == 3 and stats['active_atoms'] == ['dc:d0', 'cos1:d0', 'sin1:d0']
    h, b = stats['Unprojected']
    hp, bp = stats['Projected']
    reg = model.regularizer(h, 1e-6)
    # Dense real representation of the unprojected quadratic (complex Hermitian band -> real symmetric).
    dense = np.zeros((h.p, h.p), dtype=complex)
    for offset in range(len(h.ab)):
        for j in range(h.p-offset):
            dense[j+offset, j] = h.ab[offset, j]
            dense[j, j+offset] = np.conj(h.ab[offset, j])
    dense_reg = np.zeros_like(dense)
    for offset in range(len(reg.ab)):
        for j in range(h.p-offset):
            dense_reg[j+offset, j] = reg.ab[offset, j]
            dense_reg[j, j+offset] = np.conj(reg.ab[offset, j])
    A = harmonic.realmap(dense + dense_reg)
    L = np.concatenate(hp.lowrank, axis=0)
    expected = np.linalg.solve(A - L.T @ L, harmonic.realvec(bp))
    z = banded.fit_quadratic(hp, bp, reg)
    solution = harmonic.unpack(np.r_[expected[:h.p], expected[h.p:]])
    np.testing.assert_allclose(harmonic.unpack(z), solution.reshape(h.B, h.d).T.reshape(-1), rtol=1e-8, atol=1e-12)
    # Without the projection the Woodbury branch is bypassed and both quadratics agree.
    zu = banded.fit_quadratic(h, b, reg)
    np.testing.assert_allclose(harmonic.unpack(zu), np.linalg.solve(dense+dense_reg, b).reshape(h.B, h.d).T.reshape(-1), rtol=1e-8, atol=1e-12)


def test_forward_selection_margin_and_determinism():
    model = small_model()
    target, source = synthetic_records(model, 3, 1)[0], synthetic_records(model, 4, 3)
    h, b = banded.target_stats(model, [target])
    reg = model.regularizer(h, 1e-6)
    pilot = banded.fit_quadratic(h, b, reg)
    sh, sb = banded.target_stats(model, source)
    validation = synthetic_records(model, 5, 1)[0]
    yv = validation['Y'][model.rows]

    def score(z):
        return pipeline.relative_loss(model.predict(validation['F'], z), yv)
    quadratics = {'a': (sh, sb, 3), 'b': (sh*.5, sb*.5, 3)}
    scales = [0., .1, 1., 3.]
    first = banded.forward_select(h, b, reg, pilot, quadratics, scales, score, margin=.02, max_sources=2)
    second = banded.forward_select(h, b, reg, pilot, quadratics, scales, score, margin=.02, max_sources=2)
    assert first['sources'] == second['sources'] and first['scale'] == second['scale']
    assert first['validation_loss'] == second['validation_loss'] <= score(pilot)
    assert len(first['curves']) == 1 + sum(len(scales)*(2-step) for step in range(len(first['sources'])+1) if step < 2)
    never = banded.forward_select(h, b, reg, pilot, quadratics, scales, score, margin=1., max_sources=2)
    assert never['sources'] == [] and never['scale'] == 0. and np.array_equal(never['z'], pilot)


def test_pooling_smooth_choice_and_fit():
    curves = [dict(smooth=1e-8, validation_full_pe=.5), dict(smooth=1e-6, validation_full_pe=.5*(1+1e-17)),
              dict(smooth=1e-4, validation_full_pe=.6)]
    assert pooling.choose_smooth(curves) == 1e-6
    with pytest.raises(ValueError):
        pooling.choose_smooth(curves[:2])
    model = small_model()
    train, *sources = synthetic_records(model, 6, 4)
    validation = synthetic_records(model, 7, 1)[0]
    fitted = pooling.fit_pooling(model, train, sources, validation, [1e-8, 1e-6, 1e-4])
    assert fitted['fitting_record_count'] == 4 and [c['smooth'] for c in fitted['curves']] == [1e-8, 1e-6, 1e-4]
    assert all(c['normal_equation_relative_residual'] <= 1e-8 for c in fitted['curves'])
    assert fitted['selected_smooth'] in (1e-8, 1e-6, 1e-4)
    with pytest.raises(ValueError):
        pooling.fit_pooling(model, train, sources, validation, [1e-8, 1e-6])


def test_metrics_and_dictionary():
    y = np.array([1+1j, 2., 3j, .5])
    forced = np.array([True, False, True, False])
    exact = harmonic.metrics(y, y, forced)
    assert exact['full_band'] == exact['forced_lines'] == exact['sideband_only'] == 0.
    assert (exact['full_band_bins'], exact['forced_lines_bins'], exact['sideband_only_bins']) == (4, 2, 2)
    off = harmonic.metrics(np.zeros(4), y, forced)
    assert off['full_band'] == 1. and off['forced_lines_energy'] == 11.
    model = small_model()
    V = np.arange(model.n//2+1, dtype=complex)
    D, names = harmonic.dictionary(V, model.omega, ['dc', 'cos2', 'sin2'], model.wc)
    assert names == ['dc:d0', 'cos2:d0', 'sin2:d0', 'dc:d1', 'cos2:d1', 'sin2:d1', 'dc:d2', 'cos2:d2', 'sin2:d2']
    np.testing.assert_array_equal(D[:, 0], V)
    np.testing.assert_allclose(D[10, 1], (V[8]+V[12])/2)
    np.testing.assert_allclose(D[10, 2], (V[8]-V[12])/(2j))
    np.testing.assert_allclose(D[:, 3], V*1j*model.omega/model.wc)


def test_streaming_mat_reader_matches_scipy(tmp_path):
    rng = np.random.default_rng(8)
    y = rng.normal(size=(3*64, 5))
    for compress in (False, True):
        path = tmp_path / f'MS_Const_N64_RMS70_P2P700_{int(compress)}.mat'
        savemat(path, dict(fs=np.array([[64.]]), y_m=y, u_ref=rng.normal(size=(64, 5))), do_compression=compress)
        reference = loadmat(path)['y_m']
        for column in (0, 3, 4):
            np.testing.assert_array_equal(read_mat_column(path, 'y_m', column), reference[:, column])
        with pytest.raises(ValueError):
            read_mat_column(path, 'y_m', 5)
        with pytest.raises(KeyError):
            read_mat_column(path, 'absent', 0)
    spectrum = period_spectrum(y[:, 0], 64)
    np.testing.assert_allclose(spectrum, np.fft.rfft(y[:, 0].reshape(3, 64).mean(0))/64)
    with pytest.raises(ValueError):
        period_spectrum(y[:10, 0], 64)


# --- frozen design -------------------------------------------------------------

def test_frozen_design_is_consistent():
    text = (EXPERIMENT / 'configs/design.json').read_bytes()
    assert hashlib.sha256(text).hexdigest() == CONFIG['design_sha256']
    assert len(class_grid(DESIGN)) == 36 and len(DESIGN['targets']) == 8 and len(DESIGN['sources']) == 46
    assert sum(s['kind'] == 'Const' for s in DESIGN['sources']) == 30
    systems = {s['id']: s for s in DESIGN['sources']}
    for target in DESIGN['targets']:
        assert target['kind'] in ('Harm_h1', 'Harm_h3') and len(target['candidates']) == 12
        roles = [c['shortlist_role'] for c in target['candidates']]
        assert roles == ['nearest_const']*6 + ['nearest_ltv']*4 + ['direction_diverse_ltv']*2
        own = {(r[k]['file'], r[k]['column']) for r in target['rotations'] for k in ('train', 'validation', 'test')}
        assert len(own) == 3
        for rotation in target['rotations']:
            assert {rotation[k]['u_ref_sha256'] for k in ('train', 'validation', 'test')}.__len__() == 3
            assert all(rotation[k]['forced_lines'] for k in ('train', 'validation', 'test'))
        bank = []
        for candidate in target['candidates']:
            system = systems[candidate['source_id']]
            assert system['trajectory_hash'] != target['trajectory_hash']
            assert len(candidate['base_atoms']) == 1 + 2*len(candidate['active_harmonics'])
            bank += [(r['file'], r['column']) for r in system['records']]
        assert len(bank) == len(set(bank)) == 30 and not own & set(bank)
    pipeline.check_constants(DESIGN)
    assert CONFIG['archive']['sha256'] == 'cbc58ad9541f073326f61ce463ca0be0b41d62f450f13ed9000eb2b492d2dc2d'


# --- stored results ------------------------------------------------------------

def test_target_classes_recompute_from_the_meta_cache():
    rows = read_csv(RESULTS / 'representation_meta_cv.csv')
    assert len(rows) == 864
    stored = json.loads((RESULTS / 'target_classes.json').read_text())
    recomputed = pipeline.choose_classes(DESIGN, rows)
    assert [c['hyperparameters'] for c in recomputed] == [c['hyperparameters'] for c in stored]
    assert [c['held_out_target'] for c in stored] == [t['id'] for t in DESIGN['targets']]
    for new, old in zip(recomputed, stored):
        assert new['mean_loss'] == old['mean_loss'] and new['auxiliary_targets'] == old['auxiliary_targets']
    selections = read_csv(RESULTS / 'source_selection.csv')
    classes = {c['held_out_target']: c['hyperparameters'] for c in stored}
    assert len(selections) == 48
    for row in selections:
        hp = classes[row['target']]
        assert (int(row['R']), int(row['B']), float(row['smooth'])) == (hp['R'], hp['B'], hp['smooth'])
        assert float(row['scale']) in DESIGN['selection']['source_scale_grid']
        assert len(json.loads(row['sources'])) <= DESIGN['selection']['max_sources']
        assert set(json.loads(row['sources'])) <= set(json.loads(row['passing_sources']))
    pooled = read_csv(RESULTS / 'pooling_selection.csv')
    assert len(pooled) == 24 and all((int(r['n_source_tasks']), int(r['n_source_records'])) == (12, 30) for r in pooled)
    for row in pooled:
        curves = [dict(smooth=s, validation_full_pe=float(row[f'validation_full_band_smooth_{s:g}']))
                  for s in DESIGN['pooling']['smoothing_grid']]
        assert float(row['selected_smooth']) == pooling.choose_smooth(curves)


def test_method_table_recomputes_and_matches_the_paper():
    scores = read_csv(RESULTS / 'all_test_scores.csv')
    assert len(scores) == 96 and {r['method'] for r in scores} == set(pipeline.METHODS)
    table = pipeline.method_table(scores)
    stored = read_csv(RESULTS / 'method_table.csv')
    assert [r['method'] for r in stored] == list(pipeline.METHODS)
    for new, old in zip(table, stored):
        assert all(new[band] == float(old[band]) for band in pipeline.BANDS)
        for band, printed in zip(pipeline.BANDS, PAPER_TABLE[new['method']]):
            assert float(f'{new[band]:.2e}') == printed, (new['method'], band, new[band], printed)
    per_target = pipeline.per_target_table(DESIGN, scores)
    stored_targets = read_csv(RESULTS / 'per_target_summary.csv')
    assert len(stored_targets) == 8
    for new, old in zip(per_target, stored_targets):
        assert new['target'] == old['target']
        assert all(new[f'{m}:{b}'] == float(old[f'{m}:{b}']) for m in pipeline.METHODS for b in pipeline.BANDS)
    summary = json.loads((RESULTS / 'SUMMARY.json').read_text())
    assert summary['evaluations'] == 24 and summary['method_means'] == {r['method']: {b: r[b] for b in pipeline.BANDS} for r in table}
    for row in scores:
        for band in pipeline.BANDS:
            assert abs(float(row[band]) - float(row[band+'_sse'])/float(row[band+'_energy'])) <= 1e-15
        assert int(row['full_band_bins']) == int(row['forced_lines_bins']) + int(row['sideband_only_bins']) == 1951


def test_final_seven_method_tables_match_scores_and_paper():
    from transfer_operator.literature.core import LABELS, REPORTED
    folder = RESULTS / 'literature'
    literature = read_csv(folder / 'all_test_scores.csv')
    methods = list(pipeline.METHODS) + [LABELS[m] for m in REPORTED]
    assert len(literature) == 72
    assert {r['method'] for r in literature} == {LABELS[m] for m in REPORTED}
    every = read_csv(RESULTS / 'all_test_scores.csv') + literature
    table = pipeline.method_table(every, methods)
    stored = read_csv(folder / 'method_table.csv')
    printed = dict(PAPER_TABLE, **{
        LABELS['jm_ts']: (1.93e-3, 1.71e-3, 4.83e-3),
        LABELS['lr_atl']: (4.15e-5, 8.06e-7, 3.48e-4),
        LABELS['hl_agg']: (5.53e-5, 1.27e-5, 3.83e-4),
    })
    assert len(stored) == 7
    for new, old in zip(table, stored):
        assert new['method'] == old['method']
        for band, value in zip(pipeline.BANDS, printed[new['method']]):
            assert new[band] == float(old[band])
            assert float(f'{new[band]:.2e}') == value
    per_target = pipeline.per_target_table(DESIGN, every, methods)
    retained = read_csv(folder / 'per_target_summary.csv')
    assert len(per_target) == len(retained) == 8
    for new, old in zip(per_target, retained):
        assert new['target'] == old['target']
        for method in methods:
            for band in pipeline.BANDS:
                assert new[f'{method}:{band}'] == float(old[f'{method}:{band}'])
    selections = read_csv(folder / 'literature_selection.csv')
    assert len(selections) == 72 and {r['method'] for r in selections} == set(REPORTED)


# --- optional raw-data checks ---------------------------------------------------

RAW = os.environ.get('LTV_RAW_DIR')
WORKDIR = RELEASE_ROOT / CONFIG['workdir']


@pytest.mark.skipif(not RAW or not (WORKDIR / 'catalog.json').is_file(),
                    reason='set LTV_RAW_DIR and run `python experiment3/run.py prepare` first')
def test_one_rotation_recomputes_from_raw_data():
    from transfer_operator.ltv.records import RecordBank
    catalog = json.loads((WORKDIR / 'catalog.json').read_text())
    bank = RecordBank(RAW, catalog, np.load(WORKDIR / 'design_arrays.npz'), DESIGN['frequency'])
    target = DESIGN['targets'][1]
    rows = pipeline.meta_losses(DESIGN, bank, target, 2)
    stored = [r for r in read_csv(RESULTS / 'representation_meta_cv.csv') if r['target'] == target['id'] and r['rotation'] == '2']
    assert len(rows) == len(stored) == 36
    for new, old in zip(rows, stored):
        assert (new['R'], new['B'], new['smooth']) == (int(old['R']), int(old['B']), float(old['smooth']))
        np.testing.assert_allclose(new['validation_loss'], float(old['validation_loss']), rtol=1e-9)
    forced = np.flatnonzero(np.abs(np.fft.rfft(read_mat_column(Path(RAW) / target['rotations'][2]['test']['file'], 'u_ref',
                                                           target['rotations'][2]['test']['column']))) > 0)
    assert set(target['rotations'][2]['test']['forced_lines']) <= set(forced.tolist())

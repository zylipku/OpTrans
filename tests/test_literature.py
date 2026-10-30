"""Literature baselines: exact solvers, independent recomputation and stored-result checks."""
import json

import numpy as np
import pytest

from _paths import RELEASE_ROOT
from transfer_operator.estimators import lasso_cd
from transfer_operator.literature.aggregation import HeldOut, nested_sets, sparse_aggregation, star_aggregation
from transfer_operator.literature.core import METHODS, amplitude, registry, shape_combination
from transfer_operator.literature.functional import FunctionalWorld
from transfer_operator.literature.lasso import lasso_path
from transfer_operator.model import BlockModel


def kkt(G, c, tau, d):
    gradient = c - G @ d
    active = d != 0
    inside = np.max(np.abs(gradient[active] - tau * np.sign(d[active])), initial=0.0)
    outside = np.max(np.maximum(np.abs(gradient[~active]) - tau, 0.0), initial=0.0)
    return max(inside, outside) / tau


def test_lasso_path_exact_against_kkt_and_coordinate_descent():
    rng = np.random.default_rng(11)
    worst_kkt = worst_objective = 0.0
    for trial in range(400):
        M, n = int(rng.integers(1, 40)), int(rng.integers(3, 60))
        X = rng.standard_normal((n, M)) @ (np.eye(M) + 0.5 * rng.standard_normal((M, M)))
        G, c = X.T @ X / n, rng.standard_normal(M)
        ridge = 1e-3 if n <= M else 0.0
        Gr = G + ridge * np.eye(M)
        taus = np.max(np.abs(c)) * np.array([1.5, 1.0, .6, .3, .1, .03, .01])
        for tau, d in zip(taus, lasso_path(G, c, taus, ridge=ridge)):
            worst_kkt = max(worst_kkt, kkt(Gr, c, tau, d))
            if trial < 40 and M <= 12:   # pure-Python coordinate descent is the slow reference
                reference = lasso_cd(Gr, -c, 2 * tau, tol=1e-12, maxit=20000)
                objective = lambda x: .5 * x @ Gr @ x - c @ x + tau * np.abs(x).sum()
                worst_objective = max(worst_objective,
                                      (objective(d) - objective(reference)) / (1 + abs(objective(reference))))
    # Residuals are round-off of the active-set solves (condition number times
    # machine precision, relative to tau); optimality is checked by objective.
    assert worst_kkt < 1e-4 and worst_objective < 1e-10
    diagonal = rng.uniform(.1, 2, 9)
    c = rng.standard_normal(9)
    np.testing.assert_allclose(lasso_path(diagonal, c, [.2, .05]), lasso_path(np.diag(diagonal), c, [.2, .05]),
                               atol=1e-14, rtol=0)
    print(f'lasso homotopy: max relative KKT violation {worst_kkt:.1e}, objective excess vs CD {worst_objective:.1e}')


def test_aggregations_match_brute_force():
    rng = np.random.default_rng(5)
    X = rng.standard_normal((60, 5))
    y = X @ rng.standard_normal(5) + rng.standard_normal(60)
    part = lambda s: HeldOut(len(s), X[s].T @ X[s] / len(s), y[s] @ X[s] / len(s), float(y[s] @ y[s]) / len(s))
    candidates = [rng.standard_normal(5) for _ in range(4)]
    whole = part(np.arange(60))
    beta, info = sparse_aggregation(candidates, whole)
    grid = np.linspace(0, 1, 4001)
    brute = min(whole.risk(w * candidates[info['first']] + (1 - w) * b) for b in candidates for w in grid)
    assert info['first'] == int(np.argmin([whole.risk(b) for b in candidates]))
    assert whole.risk(beta) <= brute + 1e-12
    beta, info = star_aggregation(candidates, part(np.arange(30)), part(np.arange(30, 60)), bound=float(np.abs(y).max()))
    second = part(np.arange(30, 60))
    brute = min(second.risk(w * candidates[info['first']] + (1 - w) * b) for b in candidates for w in grid)
    assert second.risk(beta) <= brute + 1e-12 and info['preselected'] == 4
    assert nested_sets([3., 1., 2.]) == [(), (1,), (1, 2), (0, 1, 2)]


def _small_world(seed=3, K=24, n0=40, n1=400):
    """A small coupled model with the experiment's structure, and raw samples."""
    rng = np.random.default_rng(seed)
    k = np.arange(1, K + 1, dtype=float)
    h = np.zeros((2, K))
    h[0, ::2], h[1, 1::2] = 1.0, 1.0
    h /= np.linalg.norm(h, axis=1, keepdims=True)
    model = BlockModel('small', k ** -2.0, 1 / k, k ** -.75 / np.linalg.norm(k ** -.75), h,
                       np.array([[.5, 0.], [0., 3.]]))
    def draw(n, a):
        f = np.where(rng.random((n, K)) < .5, -1., 1.) * np.sqrt(model.nu)
        return f, f @ a + rng.standard_normal(n)
    f0, y0 = draw(n0, model.a0)
    raw = [draw(n1, model.a_j(j)) for j in range(2)]
    sources = [dict(n=n1, Sf=f.T @ f / n1, yf=y @ f / n1, yy=float(y @ y) / n1) for f, y in raw]
    return model, f0, y0, raw, FunctionalWorld(model, f0, y0, sources)


def test_functional_estimators_against_raw_sample_constructions():
    model, f0, y0, raw, world = _small_world()
    risk = lambda a: model.risk(1.0, a)
    # Jiao-McKeague: FPCA of S0, per-domain least squares on projected scores.
    w, V = np.linalg.eigh(f0.T @ f0 / len(y0))
    V = V[:, ::-1]
    for D, informative in ((5, [1]), (8, [1, 2]), (6, [])):
        basis = V[:, :D]
        fits = [np.linalg.lstsq(f0 @ basis, y0, rcond=None)[0]]
        fits += [np.linalg.lstsq(raw[j - 1][0] @ basis, raw[j - 1][1], rcond=None)[0] for j in informative]
        counts = [len(y0)] + [len(raw[j - 1][1]) for j in informative]
        shape = shape_combination(fits, counts)
        scores = f0 @ basis @ shape
        expected = basis @ (float(scores @ y0) / float(scores @ scores) * shape)
        got = world.jm_ts([dict(sources=informative, truncation=D)])[0]
        assert np.isclose(got, risk(expected), rtol=1e-9, atol=1e-14)
    # Lin-Reimherr TL-FLR: per-source normalised ridge, then ridge offset on the target.
    lam1, lam2 = .3, .05
    G = sum(f.T @ f / len(y) for f, y in raw)
    beta_s = np.linalg.solve(G + lam1 * np.linalg.eigvalsh(G).max() * np.eye(G.shape[0]),
                             sum(y @ f / len(y) for f, y in raw))
    S0 = f0.T @ f0 / len(y0)
    delta = np.linalg.solve(S0 + lam2 * np.linalg.eigvalsh(S0).max() * np.eye(S0.shape[0]),
                            (y0 - f0 @ beta_s) @ f0 / len(y0))
    got = world.lr_tl([dict(sources=[1, 2], source_ridge=lam1, target_ridge=lam2)])[0]
    assert np.isclose(got, risk(beta_s + delta), rtol=1e-9)
    # Hu-Lin TL-FLR: pooled-source FPCA, pooled LS, lasso contrast (coordinate descent reference).
    M, fraction = 6, .1
    F = np.vstack([f for f, _ in raw])
    Y = np.concatenate([y for _, y in raw])
    values, Phi = np.linalg.eigh(F.T @ F / len(Y))
    Phi = Phi[:, ::-1][:, :M]
    w_hat = np.linalg.lstsq(F @ Phi, Y, rcond=None)[0]
    Xi = f0 @ Phi
    gram, linear = Xi.T @ Xi / len(y0), Xi.T @ (y0 - Xi @ w_hat) / len(y0)
    tau = fraction * np.max(np.abs(linear))
    d = lasso_cd(gram + 1e-12 * np.trace(gram) / M * np.eye(M), -linear, 2 * tau, tol=1e-14)
    got = world.hl_tl([dict(sources=[1, 2], truncation=M, lasso=fraction)])[0]
    assert np.isclose(got, risk(Phi @ (w_hat + d)), rtol=1e-7)


def test_functional_target_only_exits_equal_target_fits():
    model, f0, y0, raw, world = _small_world(seed=9)
    S0, b0 = f0.T @ f0 / len(y0), y0 @ f0 / len(y0)
    lam = .01 * np.linalg.eigvalsh(S0).max()
    ridge = np.linalg.solve(S0 + lam * np.eye(S0.shape[0]), b0)
    assert np.isclose(world.lr_tl([dict(sources=[], source_ridge=None, target_ridge=.01)])[0], model.risk(1., ridge), rtol=1e-9)
    w, V = np.linalg.eigh(S0)
    V = V[:, ::-1][:, :7]
    ols = V @ np.linalg.lstsq(f0 @ V, y0, rcond=None)[0]
    for method, row in (('jm_ts', dict(sources=[], truncation=7)), ('hl_tl', dict(sources=[], truncation=7, lasso=None))):
        values, coefficients = world.evaluate(method, [row], coefficients=True)
        assert np.isclose(values[0], model.risk(1., ols), rtol=1e-9)
    _, coefficients = world.jm_ts([dict(sources=[], truncation=7)], coefficients=True)
    assert np.isclose(coefficients[0]['amplitude'], 1.0, rtol=1e-12)


def test_shape_transfer_recovers_proportional_slopes():
    """With an exactly proportional source the TS amplitude rescales the shape."""
    shape = shape_combination([np.array([1., 2.]), np.array([-3., -6.])], [10, 30])
    np.testing.assert_allclose(shape, np.array([.25 * 1 + .75 * 3, .25 * 2 + .75 * 6]))
    assert amplitude(np.array([1., 0.]), np.array([2., 0.]), np.array([4., 1.])) == 2.0


def test_registry_counts_and_stable_ids():
    grid = dict(truncation=[1, 2], target_ridge=[.1, 1.], source_ridge=[.01], lasso=[1., .5])
    rows = registry(grid, 2)
    assert {m: len(rows[m]) for m in METHODS} == dict(jm_ts=8, lr_atl=2, hl_agg=4)
    assert all(row['index'] == i for m in METHODS for i, row in enumerate(rows[m]))


def test_advection_gram_is_the_least_squares_hessian():
    from transfer_operator.pde_models import AdvectionDiffusionModel
    from transfer_operator.pde_moments import generate_task
    from transfer_operator.literature.advection import task_moments
    model = AdvectionDiffusionModel(R=2, s=2.0, sigma=.25, tau=1.0)
    stats, F, U = generate_task(model, 50, [0., 0.], [1, 2, 3], return_samples=True)
    moments = task_moments(model, stats, U)
    hessian = np.zeros((model.d, model.d))
    linear = np.zeros(model.d)
    for i in range(model.d):
        e = np.zeros(model.d)
        e[i] = 1.0
        column = model.apply(e, F)
        linear[i] = np.sum(column * U) / 50
        for j in range(model.d):
            f = np.zeros(model.d)
            f[j] = 1.0
            hessian[i, j] = np.sum(column * model.apply(f, F)) / 50
    np.testing.assert_allclose(hessian, np.diag(moments.G), atol=1e-14)
    np.testing.assert_allclose(linear, moments.g, atol=1e-14)
    h = np.random.default_rng(0).standard_normal(model.d)
    np.testing.assert_allclose(moments.risk(h), np.mean(np.sum((U - model.apply(h, F)) ** 2, axis=0)), rtol=1e-12)


def test_ltv_dense_gram_matches_band_storage():
    from transfer_operator.ltv.banded import Quadratic
    from transfer_operator.ltv.pooling import band_product
    from transfer_operator.literature.ltv import dense
    rng = np.random.default_rng(2)
    B, d = 6, 3
    p = B * d
    A = rng.standard_normal((40, p)) + 1j * rng.standard_normal((40, p))
    H = A.conj().T @ A
    bandwidth = p - 1
    ab = np.zeros((bandwidth + 1, p), dtype=complex)
    for offset in range(bandwidth + 1):
        index = np.arange(p - offset)
        ab[offset, index] = H[index + offset, index]
    h = Quadratic(ab, B, d)
    np.testing.assert_allclose(dense(h), H, atol=1e-12)
    x = rng.standard_normal(p) + 1j * rng.standard_normal(p)
    np.testing.assert_allclose(band_product(ab, x), H @ x, atol=1e-11)


@pytest.mark.skipif(not (RELEASE_ROOT / 'experiment1/outputs/data/literature_confirmation.npz').exists(),
                    reason='experiment1 literature confirmation not present')
def test_experiment1_stored_literature_results():
    import csv
    data = np.load(RELEASE_ROOT / 'experiment1/outputs/data/literature_confirmation.npz')
    assert data['risks'].shape == (1000, 13, len(METHODS)) and np.isfinite(data['risks']).all()
    assert float(data['pairing_max_abs_difference'].max()) <= 1e-12
    with (RELEASE_ROOT / 'experiment1/outputs/results/literature/summary_by_method_n0.csv').open() as stream:
        rows = {(int(r['n0']), r['method_key']): r for r in csv.DictReader(stream)}
    for si, n0 in enumerate(data['n0']):
        for mi, method in enumerate(METHODS):
            assert float(rows[int(n0), method]['mean']) == pytest.approx(float(data['risks'][:, si, mi].mean()), rel=1e-12)
    assert set(data['methods'].tolist()) == set(METHODS)
    print(f"experiment1 literature: 1000 x 13 x 3 finite risks; pairing max diff {data['pairing_max_abs_difference'].max():.2e}")


@pytest.mark.skipif(not (RELEASE_ROOT / 'experiment2/outputs/results/literature/advection_n0/risks.npz').exists(),
                    reason='experiment2 literature evaluation not present')
def test_experiment2_stored_literature_results():
    for name in ('advection_n0', 'advection_sigma'):
        folder = RELEASE_ROOT / 'experiment2/outputs/results/literature' / name
        summary = json.loads((folder / 'SUMMARY.json').read_text())
        with np.load(folder / 'risks.npz') as data:
            for cell in summary['cells']:
                risks = data[f"{cell['cell']}__risks"]
                assert risks.shape == (300, len(METHODS)) and np.isfinite(risks).all()
                for mi, method in enumerate(METHODS):
                    assert cell['risk'][method]['mean'] == pytest.approx(float(risks[:, mi].mean()), rel=1e-12)
    print('experiment2 literature: stored risks reproduce every summary mean')


@pytest.mark.skipif(not (RELEASE_ROOT / 'experiment2/outputs/results/literature/DIAGNOSTICS.json').exists(),
                    reason='experiment2 literature diagnostics not present')
def test_experiment2_diagnostics_agree_with_the_evaluation():
    import csv
    folder = RELEASE_ROOT / 'experiment2/outputs/results/literature'
    diagnostics = json.loads((folder / 'DIAGNOSTICS.json').read_text())
    summary = json.loads((folder / 'advection_n0/SUMMARY.json').read_text())
    with (folder / 'advection_n0/frozen_parameters.csv').open(newline='') as stream:
        frozen = {r['cell']: json.loads(r['parameters']) for r in csv.DictReader(stream) if r['method'] == 'jm_ts'}
    means = {c['cell']: c['risk']['jm_ts']['mean'] for c in summary['cells']}
    checked = 0
    for cell in diagnostics['cells']:
        assert 0 < cell['constant_mode_share'] < 1 and cell['target_least_squares'] > 0
        if frozen[cell['cell']]['sources'] == [] and frozen[cell['cell']]['truncation'] == diagnostics['units']:
            # Jiao-McKeague's target-only exit on every unit is target least squares, on the same replications.
            assert cell['target_least_squares'] == pytest.approx(means[cell['cell']], rel=1e-9)
            checked += 1
    assert checked > 0
    cells = {c['cell']: c for c in summary['cells']}
    for cell in diagnostics['cells']:
        for method in ('lr_atl', 'hl_agg'):
            # Both aggregation methods reduce to target least squares on the half of the sample they fit on.
            assert abs(cells[cell['cell']]['risk'][method]['mean'] / cell['target_least_squares_first_half'] - 1) < .01
    for contrast in diagnostics['contrast']:
        assert contrast['constant_mode'] == 0.0   # every source shares the constant Fourier mode
    print(f'experiment2 diagnostics: target least squares equals Jiao-McKeague in {checked} cells')


def _ltv_problem(seed=4):
    """A small harmonic-transfer class with synthetic records, and its dense design."""
    from transfer_operator.ltv.banded import Harmonic
    from transfer_operator.literature.ltv import LTVScope
    rng = np.random.default_rng(seed)
    model = Harmonic(400, 400.0, 2, 5, lo=20.0, hi=60.0, wc=2 * np.pi * 40)
    size = len(model.freq)
    rows = len(model.rows)
    def record(z):
        F = rng.standard_normal(size) + 1j * rng.standard_normal(size)
        Y = np.zeros(size, dtype=complex)
        Y[model.rows] = model.predict(F, z) + 0.05 * (rng.standard_normal(rows) + 1j * rng.standard_normal(rows))
        return dict(F=F, Y=Y)
    z0 = rng.standard_normal(2 * model.p)
    tasks = [(f's{j}', [record(z0 + 0.3 * rng.standard_normal(2 * model.p)) for _ in range(2 + j % 2)])
             for j in range(4)]
    train, validation = record(z0), record(z0)
    test_F = rng.standard_normal(size) + 1j * rng.standard_normal(size)
    scope = LTVScope(model, 1e-4, train, validation, test_F, tasks)

    def design(F):
        X = np.zeros((rows, model.p), dtype=complex)
        for b in range(model.B):
            for r in range(model.d):
                X[:, b * model.d + r] = model.factor * model.phi[:, b] * F[model.indices[:, r]]
        return X
    return model, scope, tasks, train, test_F, design


def test_ltv_literature_estimators_against_dense_constructions():
    from transfer_operator.ltv.banded import target_stats
    from transfer_operator.literature.ltv import canonical_phase, dense, descending_heigh, realify, to_packed
    model, scope, tasks, train, test_F, design = _ltv_problem()
    X0 = design(train['F'])
    H0, b0 = X0.conj().T @ X0, X0.conj().T @ train['Y'][model.rows]
    np.testing.assert_allclose(dense(scope.h0), H0, rtol=1e-12, atol=1e-12 * np.abs(H0).max())
    np.testing.assert_allclose(scope.b0, b0, rtol=1e-12, atol=1e-12 * np.abs(b0).max())
    predict = lambda beta: model.predict(test_F, to_packed(beta, model))
    # Jiao-McKeague with no source and every direction is target least squares.
    jm = [r for r in scope.jm_ts([model.p]) if not r['sources']]
    np.testing.assert_allclose(jm[0]['test'], predict(np.linalg.solve(H0, b0)), rtol=1e-7, atol=1e-10)
    # Lin-Reimherr TL-FLR: per-task normalised ridge transfer, ridge offset on the target.
    ids = [sid for sid, _ in tasks]
    row = next(r for r in scope.lr_tl([1e-4], [1e-2]) if len(r['sources']) == 2)
    members = [ids.index(s) for s in row['sources']]
    quadratic = sum(scope.sources[k]['h'] * (1.0 / scope.sources[k]['n']) for k in members)
    H_S = dense(quadratic)
    b_S = sum(scope.sources[k]['b'] / scope.sources[k]['n'] for k in members)
    beta_S = np.linalg.solve(H_S + dense(model.regularizer(quadratic, 1e-4)), b_S)
    delta = np.linalg.solve(H0 + dense(model.regularizer(scope.h0, 1e-2)), b0 - H0 @ beta_S)
    np.testing.assert_allclose(row['test'], predict(beta_S + delta), rtol=1e-8, atol=1e-11)
    # Hu-Lin TL-FLR: pooled-source FPCA, pooled LS, lasso on real and imaginary coordinates.
    M, fraction = 6, .1
    row = next(r for r in scope.hl_tl([M], [1., fraction]) if len(r['sources']) == 3 and r['lasso'] == fraction)
    members = [ids.index(s) for s in row['sources']]
    total = sum(scope.sources[k]['n'] for k in members)
    values, vectors = descending_heigh(sum(dense(scope.sources[k]['h']) for k in members) / total)
    basis = vectors[:, :M]
    w = basis.conj().T @ (sum(scope.sources[k]['b'] for k in members) / total) / values[:M]
    G = basis.conj().T @ H0 @ basis
    Gr, cr = realify(G, basis.conj().T @ b0 - G @ w)
    tau = fraction * np.max(np.abs(cr))
    d = lasso_cd(Gr + 1e-12 * np.trace(Gr) / len(cr) * np.eye(len(cr)), -cr, 2 * tau, tol=1e-14, maxit=200000)
    np.testing.assert_allclose(row['test'], predict(basis @ (w + d[:M] + 1j * d[M:])), rtol=1e-6, atol=1e-9)
    # Canonical phases make the basis independent of the eigensolver's unit factors.
    rotation = np.exp(1j * np.random.default_rng(1).uniform(0, 2 * np.pi, vectors.shape[1]))
    np.testing.assert_allclose(canonical_phase(vectors * rotation[None, :]), vectors, atol=1e-12)
    # Hu-Lin Agg tunes every candidate on validation and aggregates on validation.
    agg = scope.hl_agg([M], [1., fraction])[0]
    assert len(agg['candidates']) == len(tasks) + 1 and agg['candidates'][0]['lasso'] is None
    print('LTV literature estimators: dense constructions reproduced (JM, LR, HL), phase invariance checked')


def test_hu_lin_aggregation_uses_each_candidates_own_tuning():
    model, f0, y0, raw, world = _small_world(seed=12, n0=60)
    tuned = [dict(truncation=4, lasso=None), dict(truncation=3, lasso=.2), dict(truncation=6, lasso=.05)]
    value, coefficients = world.hl_agg([dict(candidates=tuned)], coefficients=True)
    sets = coefficients[0]['sets']
    candidates = [world._fpca_ols('first', world.first, 4)]
    candidates += [world._hl_candidate(world.first, s, p['truncation'], [p['lasso']])[:, 0]
                   for s, p in zip(sets[1:], tuned[1:])]
    beta, _ = sparse_aggregation(candidates, world.second)
    assert np.isclose(value[0], model.risk(1.0, beta), rtol=1e-12)
    grid = [dict(truncation=t, lasso=l) for t in (3, 4, 6) for l in (.2, .05)]
    risks = world.hl_agg_candidates(grid)
    assert risks.shape == (3, len(grid))
    assert np.allclose(risks[0], [model.risk(1.0, world._fpca_ols('first', world.first, r['truncation'])) for r in grid])


def test_lin_reimherr_atl_scales_the_transfer_penalty_with_the_candidate_size():
    """ATL-FLR from raw samples: halves, ranking, nested TL-FLR candidates, star aggregation."""
    from transfer_operator.literature.core import atl_source_scale
    from transfer_operator.literature.functional import sample_moments
    model, f0, y0, raw, _ = _small_world(seed=21, n0=80, n1=600)
    raw = [raw[0], (raw[1][0][:250], raw[1][1][:250])]          # unequal source sizes
    sources = [dict(n=len(y), Sf=f.T @ f / len(y), yf=y @ f / len(y), yy=float(y @ y) / len(y)) for f, y in raw]
    world = FunctionalWorld(model, f0, y0, sources)
    r1, r2 = .02, .05
    half, quarter = 40, 60
    ridge = lambda G, b, r: np.linalg.solve(G + r * np.linalg.eigvalsh(G).max() * np.eye(len(b)), b)
    S1, g1 = f0[:half].T @ f0[:half] / half, y0[:half] @ f0[:half] / half
    target_only = ridge(S1, g1, r2)
    grams = [(f.T @ f / len(y), y @ f / len(y)) for f, y in raw]
    distances = [float(np.sum((target_only - ridge(G, g, r1)) ** 2)) for G, g in grams]
    order = list(np.argsort(distances, kind='stable'))
    candidates = [target_only]
    for size in (1, 2):
        members = order[:size]
        scale = atl_source_scale(sum(len(raw[k][1]) for k in members), len(raw[order[0]][1]))
        start = ridge(sum(grams[k][0] for k in members), sum(grams[k][1] for k in members), r1 * scale)
        candidates.append(start + ridge(S1, g1 - S1 @ start, r2))
    bound = max(float(y0.max()), max(float((f0 @ b).max()) for b in candidates))
    beta, _ = star_aggregation(candidates, sample_moments(f0[half:quarter], y0[half:quarter]),
                               sample_moments(f0[quarter:], y0[quarter:]), bound)
    got, coefficients = world.lr_atl([dict(source_ridge=r1, target_ridge=r2)], coefficients=True)
    assert np.isclose(got[0], model.risk(1.0, beta), rtol=1e-9)
    assert coefficients[0]['sets'][1] == (order[0] + 1,)
    assert atl_source_scale(850, 600) < 1 and atl_source_scale(600, 600) == 1.0

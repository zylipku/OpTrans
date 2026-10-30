"""Independent PDE equations, moments and least-squares criteria (small dimensions)."""
from itertools import product

import numpy as np
import pytest

from _paths import RELEASE_ROOT
from transfer_operator.pde_models import AdvectionDiffusionModel
from transfer_operator.pde_moments import TaskStats, generate_task, population_task
from transfer_operator.pde_estimators import (
    calibration, prepare_source, refit, ridge_fit, source_projector,
)


@pytest.fixture
def model():
    return AdvectionDiffusionModel(R=1, sigma=.2, tau=.25)


def sample(model, n, theta, seed):
    return generate_task(model, n, theta, seed, return_samples=True)


def test_model_adjoint_hessian_resolvent_and_risk(model):
    rng = np.random.default_rng(719)
    h = rng.normal(size=model.d)
    X, Z = rng.normal(size=(2, model.d, 17))
    eye = np.eye(model.d)
    A = model.apply(h, eye)
    assert np.sum(A*A) == pytest.approx(h@h, abs=2e-13)
    assert np.sum(model.apply(h, X)*Z) == pytest.approx(
        h@model.apply_adjoint(X, Z), abs=2e-13)
    design = np.column_stack([model.apply(e, X).ravel() for e in eye])
    np.testing.assert_allclose(design.T@design, np.diag(model.map_hessian(X)),
                               rtol=0, atol=2e-13)
    target = model.apply(model.truth, eye)
    for theta in model.thetas:
        source = model.apply(model.source_h(theta), eye)
        shift = sum(theta[l]*model.derivative(eye, l) for l in range(model.p))
        np.testing.assert_allclose(source-target+target@shift@source, 0,
                                   rtol=0, atol=2e-13)
    exact = np.sum((model.apply(h-model.truth, eye)*model.sqrt_q)**2)
    assert model.risk(h) == pytest.approx(exact, abs=2e-13)
    for l in range(model.p):
        derivative = model.derivative(eye, l)
        np.testing.assert_array_equal(derivative.T, -derivative)


def test_exact_population_and_empirical_moments(model):
    eta = .013
    # Exhaust all signs: the empirical forcing covariance equals population.
    F = np.array(list(product((-1., 1.), repeat=model.d))).T*model.sqrt_q[:, None]
    for j, theta in enumerate(model.thetas):
        U = model.apply(model.source_h(theta), F)
        empirical = TaskStats(F.shape[1], F@F.T/F.shape[1], U@F.T/F.shape[1])
        population = population_task(model, theta)
        np.testing.assert_allclose(empirical.Cff, population.Cff, rtol=0, atol=2e-14)
        np.testing.assert_allclose(empirical.Cuf, population.Cuf, rtol=0, atol=2e-14)
        exact = prepare_source(model, population, eta)
        np.testing.assert_allclose(exact.Y-model.apply(model.truth, exact.corrected_X(theta)),
                                   0, rtol=0, atol=2e-13)
        stats, Fs, Us = sample(model, 37, theta, [719, j])
        np.testing.assert_allclose(stats.Cff, Fs@Fs.T/37, rtol=0, atol=2e-14)
        np.testing.assert_allclose(stats.Cuf, Us@Fs.T/37, rtol=0, atol=2e-14)
        source = prepare_source(model, stats, eta, keep_whitener=True)
        np.testing.assert_allclose(source.W@(stats.Cff+eta*np.eye(model.d))@source.W,
                                   np.eye(model.d), rtol=0, atol=2e-13)
        h = .9*model.truth
        shifted_F = Fs-sum(theta[l]*model.derivative(Us, l) for l in range(model.p))
        direct = (Us-model.apply(h, shifted_F))@(source.W@Fs).T/37
        np.testing.assert_allclose(source.Y-model.apply(h, source.corrected_X(theta)),
                                   direct, rtol=0, atol=2e-13)


def dense_criterion(model, F, U, sources, theta, projectors, rho, ridge):
    """Materialize the rectangular criterion; do not use refit normal equations."""
    eye = np.eye(model.d)
    designs = [np.column_stack([model.apply(e, F).ravel() for e in eye])/np.sqrt(F.shape[1])]
    responses = [U.ravel()/np.sqrt(F.shape[1])]
    for source, th, projection, weight in zip(sources, theta, projectors, rho):
        X = source.corrected_X(th)
        remove = (lambda v: v) if projection is None else projection.remove
        designs.append(np.sqrt(weight)*np.column_stack([
            remove(model.apply(e, X)).ravel() for e in eye]))
        responses.append(np.sqrt(weight)*remove(source.Y).ravel())
    designs.append(np.sqrt(ridge)*eye)
    responses.append(np.zeros(model.d))
    return np.linalg.lstsq(np.vstack(designs), np.concatenate(responses), rcond=None)[0]


def test_calibration_kkt_empty_rank_zero_and_dense_refit(model):
    target, F, U = sample(model, 31, np.zeros(model.p), 122)
    sources = [prepare_source(model, sample(model, n, theta, 123+j)[0], .013)
               for j, (n, theta) in enumerate(zip((37, 19), model.thetas))]
    pilot = ridge_fit(model, target, .01)
    rho = [37/31, 19/31]
    max_difference = 0.
    for source in sources:
        for penalty in (0., .0001, .01, 10.):
            fit = calibration(model, source, pilot, penalty)
            generators = np.stack([model.apply(pilot, d).ravel() for d in source.DY])
            residual = source.Y-model.apply(pilot, source.corrected_X(fit.theta))
            gradient = 2*generators@residual.ravel()
            active = fit.theta != 0
            np.testing.assert_allclose(gradient[active]+penalty*np.sign(fit.theta[active]),
                                       0, rtol=0, atol=2e-10)
            assert np.all(np.abs(gradient[~active]) <= penalty+2e-10)
        zero = np.zeros(model.d)
        empty = calibration(model, source, pilot, support=())
        np.testing.assert_array_equal(empty.theta, zero[:model.p])
        full_support = tuple(range(model.p))
        degenerate = calibration(model, source, zero, support=full_support)
        assert degenerate.support == full_support
        np.testing.assert_array_equal(degenerate.theta, zero[:model.p])
        assert source_projector(model, source, zero, full_support).rank == 0
    for supports in ([(), ()], [tuple(range(model.p))]*2,
                     [tuple(np.flatnonzero(t)) for t in model.thetas]):
        fits = [calibration(model, source, pilot, support=support)
                for source, support in zip(sources, supports)]
        theta = [fit.theta for fit in fits]
        projections = [source_projector(model, source, pilot, support)
                       for source, support in zip(sources, supports)]
        for project in (False, True):
            projectors = projections if project else [None]*2
            fast, info = refit(model, target, sources, theta, projectors, rho, .007)
            direct = dense_criterion(model, F, U, sources, theta, projectors, rho, .007)
            difference = float(np.max(np.abs(fast-direct)))
            max_difference = max(max_difference, difference)
            np.testing.assert_allclose(fast, direct, rtol=0, atol=2e-12)
    print(f'{type(model).__name__}: independent dense refit max diff={max_difference:.3g}')


def test_fixed_support_round_rebuilds_projection_and_common_interface(model):
    from transfer_operator.pde_pipeline import fit_transfer, recalibration_round
    from transfer_operator.stages import PDEProblem, fit_stages
    target = sample(model, 31, np.zeros(model.p), 392)[0]
    source_stats = [sample(model, 47, theta, 393+j)[0]
                    for j, theta in enumerate(model.thetas)]
    candidate = dict(c0=.03, cbar=.01, ctheta=.001)
    fit = fit_transfer(model, target, source_stats, candidate, additional_rounds=2)
    assert len(fit.steps) == 3
    changes = []
    for previous, step in zip(fit.steps, fit.steps[1:]):
        np.testing.assert_array_equal(step.reference, previous.operator)
        assert [cal.support for cal in step.calibrations] == fit.supports
        direct = recalibration_round(model, target, fit.sources, previous.operator,
                                     fit.supports, fit.rho, fit.tuning.ridge)
        np.testing.assert_array_equal(step.operator, direct.operator)
        for source, support, projection, old in zip(fit.sources, fit.supports,
                                                   step.projectors, previous.projectors):
            rebuilt = source_projector(model, source, previous.operator, support)
            np.testing.assert_array_equal(projection.Q, rebuilt.Q)
            changes.append(projection.change(old))
    assert max(changes) > 1e-6  # a stale fixed projector would fail this fixture
    common = fit_stages(PDEProblem(model, target, source_stats, candidate))
    np.testing.assert_array_equal(common.pilot, fit.pilot)
    np.testing.assert_array_equal(common.refits, fit.projected)
    assert common.method == 'Projected'
    common_u = fit_stages(PDEProblem(model, target, source_stats, candidate), projected=False)
    np.testing.assert_array_equal(common_u.refits[0], fit.unprojected)


def test_duplicate_generator_rank_and_minimum_norm():
    from dataclasses import replace
    model = AdvectionDiffusionModel(R=1)
    source = prepare_source(model, population_task(model, model.thetas[0]), .01)
    duplicate = replace(source, DY=np.stack([source.DY[0]]*2))
    projection = source_projector(model, duplicate, model.truth, (0, 1))
    assert projection.rank == 1
    fit = calibration(model, duplicate, model.truth, support=(0, 1))
    assert fit.support == (0, 1)
    assert fit.theta[0] == pytest.approx(fit.theta[1], abs=2e-14)
    assert fit.kkt < 1e-12


def test_coupled_adapter_preserves_original_backend_bitwise():
    from test_shared_api import small_config
    from transfer_operator.simulation import development_world
    from transfer_operator.estimators import FoldFit, SourceBank, candidate_object, iteration_path
    from transfer_operator.stages import CoupledProblem, fit_stages
    config = small_config(sources=3, steps=3)
    model, target, sources, _ = development_world(7, 0, config)
    fold = FoldFit(model, target, SourceBank(model, sources))
    candidate = candidate_object(config.design['selected_candidates']['7']['projected'])
    original = iteration_path(fold, candidate, max_step=3)
    common = fit_stages(CoupledProblem(fold, candidate), additional_rounds=2)
    assert common.supports == original['fixed_supports']
    for (c, a), step in zip([common.pilot]+common.refits, original['steps']):
        assert c == step['c']
        np.testing.assert_array_equal(a, step['a'])
    direct = fold.fit(candidate, 'unprojected')
    adapted = fit_stages(CoupledProblem(fold, candidate), projected=False)
    assert adapted.refits[0][0] == direct[0]
    np.testing.assert_array_equal(adapted.refits[0][1], direct[1])

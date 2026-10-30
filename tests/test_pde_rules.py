"""Frozen rules, bootstrap streams, live scalar configs, and source-count safety."""
import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

import _paths
from transfer_operator import PDEStudy
from transfer_operator.pde_models import AdvectionDiffusionModel
from transfer_operator.pde_moments import population_task
from transfer_operator.pde_pipeline import Tuning, fit_transfer, frozen_tuning
from transfer_operator.pde_replication import bootstrap_resample_means

RELEASE = Path(__file__).resolve().parents[1]
CONFIGS = RELEASE/'experiment2/configs'
RESULTS = RELEASE/'experiment2/outputs/results'


def read(path):
    return json.loads(path.read_text())


@pytest.mark.parametrize('allocation', [[32, 32], [17, 47]])
def test_frozen_rule_matches_paper_formula(allocation):
    model = AdvectionDiffusionModel(R=1, s=2)
    c0, cbar, ctheta = .03, .001, .02
    result = frozen_tuning(model, 24, 64, (c0, cbar, ctheta), allocation)
    eta = cbar/(88**(2/3))
    neff = math.fsum(float(q/(q+eta)) for q in model.q)
    expected = [ctheta*(math.sqrt(neff/n)+1/(n*math.sqrt(eta))+24**(-1/3))
                for n in allocation]
    assert result.lam0 == pytest.approx(c0/24**(2/3), rel=2e-15)
    assert result.eta == pytest.approx(eta, rel=2e-15)
    assert result.ridge == pytest.approx((88/24)*eta, rel=2e-15)
    assert result.effective_dimension == pytest.approx(neff, rel=2e-15)
    np.testing.assert_allclose(result.penalties, expected, rtol=2e-15, atol=0)


def test_bootstrap_registered_indices_remain_paired():
    # Fixed dyadic risks expose a changed RNG sampler or unpaired resampling.
    risks = np.array([1., 2., 4., 8.])
    actual = bootstrap_resample_means(np.column_stack([risks, risks+7, 2*risks]),
                                     6, 719, algorithm='indices')
    np.testing.assert_array_equal(actual[:, 0], [2.75, 4., 5.25, 3.25, 3.25, 4.5])
    np.testing.assert_array_equal(actual[:, 1]-actual[:, 0], 7)
    np.testing.assert_array_equal(actual[:, 2], 2*actual[:, 0])
    with pytest.raises(ValueError):
        bootstrap_resample_means([], 6, 719, algorithm='indices')


def test_expanded_yaml_cells_match_retained_paper_results():
    advection = PDEStudy.from_yaml(CONFIGS/'advection_sigma.yaml')
    config = yaml.safe_load((CONFIGS/'advection_sigma.yaml').read_text())
    summary = read(RESULTS/'advection_sigma/SIGMA_SUMMARY.json')
    archived = {cell['cell']: cell for cell in summary['cells']}
    assert len(advection.cells) == len(archived) == 8
    for cell in advection.cells:
        original = archived[cell.archive_cell]['setting']
        assert cell.setting == {k: original[k] for k in ('R','s','tau','sigma','n0','nS')}
        assert cell.seed == config['seed']
        assert cell.bootstrap_seed == config['bootstrap']['seed']
        for method in ('Projected', 'Unprojected'):
            assert cell.multipliers[method] == {k: config['frozen_multipliers'][method][k]
                                               for k in ('c0','cbar','ctheta')}
    cell = advection.cells[0]
    assert cell.make_model().d == 625
    assert cell.tuning().eta > 0
    with pytest.raises(KeyError):
        advection.cell('advection_sigma', 4)  # sigma=1 is absent from the final paper.
    n0_study = PDEStudy.from_yaml(CONFIGS/'advection_n0.yaml')
    assert [cell.setting['n0'] for cell in n0_study.cells] == [16,32,64,128,256,512,1024,2048,4096]
    assert all(cell.setting['sigma'] == .25 and cell.setting['nS'] == 32768 for cell in n0_study.cells)


def test_config_rejects_nonpaper_model(tmp_path):
    spec = yaml.safe_load((CONFIGS/'advection_sigma.yaml').read_text())
    spec['model'] = 'unsupported'
    path = tmp_path/'changed.yaml'
    path.write_text(yaml.safe_dump(spec))
    with pytest.raises(ValueError):
        PDEStudy.from_yaml(path)


def test_explicit_tuning_supports_three_unequally_weighted_sources():
    model = AdvectionDiffusionModel(R=1)
    target = population_task(model, [0., 0.], n=3)
    sources = [population_task(model, theta, n=n)
               for theta,n in zip(([.1,0.],[0.,.2],[.4,.1]), (5,7,11))]
    tuning = Tuning(.01, .003, .01, np.zeros(3), 0.)
    fit = fit_transfer(model, target, sources, tuning=tuning, supports=[()]*3,
                       projected=False)
    np.testing.assert_array_equal(fit.rho, np.array([5,7,11])/3)
    assert len(fit.sources) == len(fit.supports) == 3
    # With empty supports, the exact-moment criterion separates coordinatewise.
    w = model.q/np.sqrt(model.q+tuning.eta)
    numerator = model.q*model.truth
    denominator = model.q+tuning.ridge
    for theta, weight in zip(([.1,0.],[0.,.2],[.4,.1]), (5/3,7/3,11/3)):
        numerator = numerator+weight*w*w*model.source_h(theta)
        denominator = denominator+weight*w*w
    np.testing.assert_allclose(fit.unprojected, numerator/denominator, rtol=0, atol=2e-15)
    for options in ({'rho':[1.,1.]}, {'supports':[(),()]},
                    {'eigencache':[None,None]}, {'additional_rounds':True},
                    {'tuning':Tuning(.01,.003,.01,np.zeros(2),0.)}):
        kwargs = dict(tuning=tuning, projected=False)
        kwargs.update(options)
        with pytest.raises(ValueError):
            fit_transfer(model, target, sources, **kwargs)


@pytest.mark.parametrize('n0,ns,candidate,allocation', [
    (0,64,(.03,.001,.02),None), (24,63,(.03,.001,.02),None),
    (24,64,(.03,0.,.02),None), (24,64,(.03,.001,-.02),None),
    (24,64,(.03,.001,.02),[32,31])])
def test_invalid_frozen_rules_fail_before_estimation(n0,ns,candidate,allocation):
    with pytest.raises(ValueError):
        frozen_tuning(AdvectionDiffusionModel(R=1),n0,ns,candidate,allocation)

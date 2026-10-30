"""Independently check cached estimator equations using release inputs only.

Reconstruct trajectory 0, compare all four fitted operators with dense normal
equations, and compare k=0..10 with literal Section 2.3 recalibration/refits.
No data or tuning choices are changed. Results are printed to stdout only.
"""
from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
from _paths import RELEASE_ROOT, DEFAULT_CONFIG
from transfer_operator import ExperimentConfig
from transfer_operator.simulation import nested_samples, prefix_statistics
from transfer_operator.estimators import (PRIMARY,FoldFit,SourceBank,candidate_object,iteration_path,
                            projection_basis,restricted_recalibration,eta_from_rule)
import numpy as np


def independent_dense_fit(fold,candidate,method):
    """Build the full quadratic normal equations, without cached Schur/Woodbury."""
    n=fold.model.K;k=n+1
    lam0=candidate.lam0_frac*fold.s0_max
    if method=='pilot' or method in ('unprojected','projected') and candidate.no_transfer:
        return fold.tst['mL']/(fold.tst['mL']+lam0),np.linalg.solve(fold.tst['Sf']+lam0*np.eye(n),fold.tst['yf'])
    lambar=candidate.lambar_frac*fold.sources.s_max
    lam=fold.N/fold.tst['n']*lambar
    if method=='pooling':
        h=fold.tst['Sf'].copy()+lam*np.eye(n);rhs=fold.tst['yf'].copy()
        for rho,st in zip(fold.rhos,fold.sources.sst):
            h+=rho*st['Sf'];rhs+=rho*st['yf']
        return 1.,np.linalg.solve(h,rhs)
    c0,a0=fold.pilot(lam0)
    eta=eta_from_rule(candidate.eta_rule,lambar)
    thetas,supports,_,_=fold.calibration(c0,a0,lam0,eta,candidate.lamj_frac)
    h=np.eye(k)*lam;rhs=np.zeros(k)
    h[0,0]+=fold.tst['mL'];rhs[0]+=fold.tst['mL']
    h[1:,1:]+=fold.tst['Sf'];rhs[1:]+=fold.tst['yf']
    for rho,v,theta,support in zip(fold.rhos,fold.sources.views(eta),thetas,supports):
        x=np.column_stack((v.WSLh@theta,v.WS));y=v.Wyf.copy()
        if method=='projected':
            q,_=projection_basis(v,c0,support)
            x-=q@(q.T@x);y-=q@(q.T@y)
        h+=rho*x.T@x;rhs+=rho*x.T@y
        h[0,0]+=rho*v.blockF2;rhs[0]+=rho*v.blockF2
    z=np.linalg.solve(h,rhs)
    return float(z[0]),z[1:]


def validate(config_directory=DEFAULT_CONFIG):
    start=time.monotonic();config=ExperimentConfig.from_directory(config_directory)
    design=config.design;model=config.model
    samples,_=nested_samples(model,design['confirmation_exp_id'],0,design['settings'])
    rows=[]
    for setting in design['settings']:
        tst,sst,_=prefix_statistics(model,samples,setting)
        fold=FoldFit(model,tst,SourceBank(model,sst))
        chosen=design['selected_candidates'][str(setting['n0'])]
        baseline=[]
        for method in PRIMARY+('projection_removed',):
            candidate=candidate_object(chosen['projected' if method=='projection_removed' else method])
            effective='unprojected' if method=='projection_removed' else method
            c,a,_=fold.fit(candidate,effective)
            cd,ad=independent_dense_fit(fold,candidate,effective)
            np.testing.assert_allclose(c,cd,atol=1e-11,rtol=1e-10)
            np.testing.assert_allclose(a,ad,atol=2e-9,rtol=2e-8)
            np.testing.assert_allclose(model.risk(c,a),model.risk(cd,ad),atol=1e-10,rtol=1e-8)
            baseline.append(dict(method=method,max_a_abs_error=float(np.max(np.abs(a-ad))),c_abs_error=abs(c-cd),risk_abs_error=abs(model.risk(c,a)-model.risk(cd,ad))))
        candidate=candidate_object(chosen['projected'])
        cached=iteration_path(fold,candidate,max_step=max(design['iteration_steps']),backend='cached')
        dense=iteration_path(fold,candidate,max_step=max(design['iteration_steps']),backend='dense')
        iteration=[]
        for a,b in zip(cached['steps'],dense['steps']):
            assert a['fixed_supports']==b['fixed_supports']
            assert a['projection_ranks']==b['projection_ranks']
            for key,atol,rtol in [('c',1e-11,1e-10),('a',2e-9,2e-8),('risk',1e-10,1e-8),('theta',2e-9,2e-8)]:
                np.testing.assert_allclose(a[key],b[key],atol=atol,rtol=rtol)
            iteration.append(dict(step=a['step'],max_a_abs_error=float(np.max(np.abs(a['a']-b['a']))),risk_abs_error=abs(a['risk']-b['risk'])))
        rows.append(dict(n0=setting['n0'],baseline=baseline,iterations=iteration,
            cached_postfirst_risk_drift=cached['iteration_diagnostics']['max_postfirst_absolute_risk_drift'],
            dense_postfirst_risk_drift=dense['iteration_diagnostics']['max_postfirst_absolute_risk_drift']))
    # Boundary geometry checks use one view, without drawing a new dataset.
    view=fold.sources.views(fold.fit(candidate,'projected')[2]['eta'])[0]
    q,_=projection_basis(view,0.,[0,1]);assert q.shape[1]==0
    q,_=projection_basis(view,1.,[]);assert q.shape[1]==0
    import copy
    duplicate=copy.copy(view);duplicate.WSLh=view.WSLh.copy();duplicate.WSLh[:,1]=duplicate.WSLh[:,0]
    q,_=projection_basis(duplicate,1.,[0,1]);assert q.shape[1]==1
    theta,diag=restricted_recalibration(duplicate,1.,np.zeros(model.K),[0,1])
    assert diag['numerical_rank']==1 and np.isfinite(theta).all()
    files=list((RELEASE_ROOT/'transfer_operator').glob('*.py'))+list(config.directory.glob('*.yaml'))+[Path(__file__)]
    return dict(status='PASS',trajectory=0,settings=rows,
        edge_cases=['zero c','empty support','duplicated rank-one generators','minimum-norm restricted LS'],
        elapsed_seconds=time.monotonic()-start,
        source_sha256={(str(p.relative_to(RELEASE_ROOT)) if p.is_relative_to(RELEASE_ROOT) else str(p)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    args=parser.parse_args()
    result=validate(args.config)
    print(json.dumps(dict(status=result['status'],settings=len(result['settings']),dense_baselines=sum(len(s['baseline']) for s in result['settings']),iteration_comparisons=sum(len(s['iterations']) for s in result['settings']),elapsed_seconds=result['elapsed_seconds'])),flush=True)


if __name__=='__main__': main()

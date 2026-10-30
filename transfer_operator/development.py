"""Evaluate the complete candidate grid and reproduce development-only choices.

True risk is used only in independent simulation development. Confirmation uses
the frozen selected_candidates in the supplied design and never calls selection.
"""
from __future__ import annotations
import numpy as np
from .estimators import PRIMARY, candidate_object, candidate_registry, SourceBank, FoldFit
from .simulation import development_world
from .config import ExperimentConfig

def evaluate_candidates(model,tst,sst,registry,*,rank_rtol=1e-10):
    full=FoldFit(model,tst,SourceBank(model,sst,rank_rtol=rank_rtol))
    arrays,failures={},[]
    for method in PRIMARY:
        indices=list(range(len(registry[method])))
        values=np.full(len(indices),np.nan)
        for offset,index in enumerate(indices):
            candidate=candidate_object(registry[method][index])
            try:
                c,a,info=full.fit(candidate,method)
                risk=model.risk(c,a)
                if not np.isfinite(c) or not np.isfinite(a).all() or not np.isfinite(risk):
                    raise FloatingPointError("Nonfinite candidate fit or risk")
                values[offset]=risk
            except Exception as exc:
                failures.append(dict(method=method,candidate_id=index,error=repr(exc)))
        arrays[method]=values
        arrays[method+"__candidate_ids"]=np.asarray(indices,dtype=np.int64)
    arrays["observable_scales"]=np.asarray([full.s0_max,full.sources.s_max])
    return arrays,failures

def run_development(n0,index,config: ExperimentConfig):
    model,tst,sst,identity=development_world(n0,index,config)
    arrays,failures=evaluate_candidates(model,tst,sst,candidate_registry(config.design),
        rank_rtol=config.design.get("numerics",{}).get("rank_rtol",1e-10))
    return arrays,dict(identity=identity,failures=failures,status="failed" if failures else "ok")

def select_candidates(banks,registry,*,expected_reps=None):
    """Select once from candidate-by-repetition development risk matrices.

    An exact mean tie prefers larger lamj, lambar, lam0 fractions in that order,
    followed by stable registry order. No approximate-tie tolerance is used.
    Pass ``expected_reps`` to require the complete registered development bank.
    """
    selected={}
    for n in banks:
        selected[str(n)]={}
        for method in PRIMARY:
            losses=np.asarray(banks[n][method],dtype=float)
            if (losses.ndim!=2 or losses.shape[0]!=len(registry[method]) or losses.shape[1]<1
                    or expected_reps is not None and losses.shape[1]!=expected_reps
                    or not np.isfinite(losses).all()):
                raise ValueError("Selection requires every finite candidate and the requested development repetitions")
            means=losses.mean(axis=1)
            def key(i):
                c=candidate_object(registry[method][i])
                return (means[i],(-c.lamj_frac,-c.lambar_frac,-c.lam0_frac))
            index=min(range(len(means)),key=key)
            selected[str(n)][method]=registry[method][index]
    return selected

"""Independent development data and paired nested confirmation trajectories.

Sampling uses the exact stream splitting, 8192-row chunks, floating operations,
and full-prefix statistics of the released experiment. Every estimator within
a setting sees identical data. Confirmation never chooses candidates.
"""
from __future__ import annotations
import hashlib
import numpy as np
from .model import BlockModel
from .config import ExperimentConfig
from .estimators import PRIMARY, SourceBank, FoldFit, candidate_object, candidate_registry, iteration_path

def array_sha(*arrays):
    digest=hashlib.sha256()
    for value in arrays: digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()

def _streams(exp_id: int, rep: int, task: int):
    ss = np.random.SeedSequence([int(exp_id), int(rep), int(task)])
    c1, c2 = ss.spawn(2)
    return np.random.default_rng(c1), np.random.default_rng(c2)


def _draw_v2(model, n, rng_xi, rng_z, chunk):
    sq = np.sqrt(model.nu)
    done = 0
    while done < n:
        c = min(chunk, n - done)
        xi = np.where(rng_xi.random((c, model.K)) < 0.5, -1.0, 1.0)      # one 64-bit draw per entry -> prefix-consistent
        yield xi * sq, rng_z.standard_normal(c)
        done += c


def stats_v2(model: BlockModel, n: int, exp_id: int, rep: int, task: int, chunk: int = 8192, explicit: bool = False):
    """Sufficient statistics of task `task` (0 = target, j+1 = source j) with n copies.  The same (exp_id, rep, task)
    with a larger n extends the random streams. The chunk size and matrix
    reduction order must remain fixed for bitwise-identical sufficient statistics.
    explicit=True additionally returns the samples (tests only)."""
    rng_xi, rng_z = _streams(exp_id, rep, task)
    a = model.a0 if task == 0 else model.a_j(task - 1)
    K = model.K
    Sf = np.zeros((K, K))
    yf = np.zeros(K)
    yy = 0.0
    mL = 0.0
    fs, ys = [], []
    for f, z in _draw_v2(model, n, rng_xi, rng_z, chunk):
        y = f @ a + model.sigma * z
        Sf += f.T @ f
        yf += y @ f
        yy += float(y @ y)
        mL += float(np.sum((f * model.mu) ** 2))
        if explicit:
            fs.append(f)
            ys.append(y)
    out = dict(n=n, Sf=Sf / n, yf=yf / n, yy=yy / n, mL=mL / n)
    if explicit:
        out["f"], out["y"] = np.vstack(fs), np.concatenate(ys)
    return out


def target_stats_from(model, f, y):
    n = f.shape[0]
    return dict(n=n, Sf=f.T @ f / n, yf=y @ f / n, yy=float(y @ y) / n, mL=float(np.sum((f * model.mu) ** 2)) / n)


def nested_samples(model, exp_id, trajectory, settings, *, chunk=8192):
    """Draw each task once at its largest registered allocation.

    The archived sampler uses SeedSequence([exp_id,trajectory,task]).spawn(2)
    for task-independent forcing and scalar response noise. Prefixes therefore
    have the archived law and are nested exactly, without new random streams at
    smaller sample sizes.
    """
    max_counts = [max(s[f'n{j}'] for s in settings) for j in range(model.m + 1)]
    streams = [stats_v2(model, n, int(exp_id), int(trajectory),
                                           task, chunk=chunk, explicit=True)
               for task, n in enumerate(max_counts)]
    samples = [(st['f'], st['y']) for st in streams]
    identities = dict(trajectory=int(trajectory), exp_id=int(exp_id),
                      task_max_counts=max_counts,
                      full_task_data_sha256=[array_sha(f, y) for f, y in samples],
                      seed_rule='SeedSequence([exp_id, trajectory, task]).spawn(2)',
                      forcing_and_noise_streams_separate=True,
                      task_streams_independent=True)
    return samples, identities


def prefix_statistics(model, samples, setting):
    stats, hashes = [], []
    for j, (f, y) in enumerate(samples):
        n = int(setting[f'n{j}'])
        if n > len(y):
            raise AssertionError('Registered prefix exceeds drawn task sample')
        fs, ys = f[:n], y[:n]
        stats.append(target_stats_from(model, fs, ys))
        hashes.append(array_sha(fs, ys))
    return stats[0], stats[1:], dict(task_prefix_data_sha256=hashes,
                  target_data_sha256=hashes[0],
                  source_covariance_sha256=[array_sha(st['Sf']) for st in stats[1:]],
                  source_response_moment_sha256=[array_sha(st['yf']) for st in stats[1:]])


def development_world(n0,index,config: ExperimentConfig):
    design=config.design
    if n0 not in design["n0_values"] or not 0<=index<design["development_reps"]:
        raise ValueError("Unregistered development identity")
    model=config.model; setting=next(s for s in design["settings"] if s["n0"]==n0)
    exp_id=design["development_exp_ids"][str(n0)]
    chunk=design.get("numerics",{}).get("chunk",8192)
    target=stats_v2(model,n0,exp_id,index,0,chunk=chunk,explicit=True)
    f0,y0=target["f"],target["y"]
    tst=target_stats_from(model,f0,y0)
    sst=[stats_v2(model,setting[f"n{j}"],exp_id,index,j,chunk=chunk) for j in range(1,model.m + 1)]
    identity=dict(exp_id=exp_id,rep=index,n0=n0,setting=setting,
        target_data_sha256=array_sha(f0,y0),source_covariance_sha256=[array_sha(t["Sf"]) for t in sst],
        source_response_moment_sha256=[array_sha(t["yf"]) for t in sst])
    return model,tst,sst,identity

def run_trajectory(trajectory,config: ExperimentConfig,sizes=None):
    """Reconstruct frozen fits on one registered trajectory (optionally subset n0).

    All maximum task samples are generated even when sizes is a subset, so the
    response arithmetic and prefixes match the original full study exactly.
    Coefficients are represented by SHA-256 hashes; c, risks, calibration, and
    diagnostic quantities are returned explicitly.
    """
    design=config.design; registry=candidate_registry(design)
    if isinstance(trajectory,bool) or int(trajectory)!=trajectory or not 0<=trajectory<design["confirmation_reps"]:
        raise ValueError("Unregistered confirmation trajectory")
    trajectory=int(trajectory)
    sizes=design["n0_values"] if sizes is None else list(sizes)
    if len(sizes)!=len(set(sizes)) or any(n not in design["n0_values"] for n in sizes):
        raise ValueError("Sample sizes must be distinct registered values")
    for n in sizes:
        for method in PRIMARY:
            candidate=design["selected_candidates"][str(n)][method]
            if registry[method][candidate["index"]]!=candidate:
                raise ValueError("Frozen selection is not in the exact candidate registry")
    model=config.model; exp_id=design["confirmation_exp_id"]
    numerics=design.get("numerics",{})
    samples,identities=nested_samples(model,exp_id,trajectory,design["settings"],chunk=numerics.get("chunk",8192))
    result=dict(trajectory=trajectory,exp_id=exp_id,settings=[],
        per_replication_selection=False,support_reselection=False,**{k:v for k,v in identities.items() if k not in ("trajectory","exp_id")})
    for setting in design["settings"]:
        if setting["n0"] not in sizes: continue
        tst,sst,identity=prefix_statistics(model,samples,setting)
        full=FoldFit(model,tst,SourceBank(model,sst,rank_rtol=numerics.get("rank_rtol",1e-10)))
        chosen=design["selected_candidates"][str(setting["n0"])]
        fits={}
        for method in PRIMARY+("projection_removed",):
            rec=chosen["projected" if method=="projection_removed" else method]
            c,a,info=full.fit(candidate_object(rec),"unprojected" if method=="projection_removed" else method)
            risk=model.risk(c,a)
            if not np.isfinite(risk) or not np.isfinite(c) or not np.isfinite(a).all():
                raise FloatingPointError("Nonfinite confirmation fit")
            fits[method]=dict(risk=float(risk),c=float(c),info=info,candidate=rec,operator_sha256=array_sha(np.asarray([c]),a))
        path=iteration_path(full,candidate_object(chosen["projected"]),max_step=max(design["iteration_steps"]),backend="cached")
        steps=[]
        for step in path["steps"]:
            step=dict(step); a=np.asarray(step.pop("a"))
            step["operator_sha256"]=array_sha(np.asarray([step["c"]]),a)
            steps.append(step)
        if len(steps)>1 and (steps[1]["risk"]!=fits["projected"]["risk"] or steps[1]["operator_sha256"]!=fits["projected"]["operator_sha256"]):
            raise AssertionError("First iteration differs from Projected")
        post=steps[2:]
        diag=dict(**path["iteration_diagnostics"],fixed_supports=path["fixed_supports"],no_transfer=path["no_transfer"],
            max_coefficient_change_after_first=max((s["coefficient_change_norm"] for s in post),default=0.),
            max_risk_change_after_first=max((abs(s["risk"]-steps[1]["risk"]) for s in post),default=0.),
            all_operator_hashes_equal_after_first=len({s["operator_sha256"] for s in steps[1:]})<=1)
        allocations={f"n{j}":setting[f"n{j}"] for j in range(model.m + 1)}
        result["settings"].append(dict(setting=setting,**allocations,N=setting.get("N",sum(allocations.values())),
            fits=fits,iterations=steps,iteration_diagnostics=diag,failures=[],**identity))
    result["status"]="ok"
    return result

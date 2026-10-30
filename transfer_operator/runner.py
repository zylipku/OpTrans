"""Recreate confirmation fits or the independent development selection.

Outputs use a new directory outside the supplied configs and retained outputs.
Confirmation never changes its YAML config. Development selections are emitted
separately and compared with that freeze, without feeding back into confirmation.
"""
from __future__ import annotations
import argparse, hashlib, json, os, time, traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from .config import ExperimentConfig
from .estimators import PRIMARY, candidate_registry
from .simulation import run_trajectory
from .development import run_development, select_candidates
import numpy as np


def clean(value):
    if isinstance(value,dict): return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [clean(v) for v in value]
    if isinstance(value,np.ndarray): return clean(value.tolist())
    if isinstance(value,np.generic): return value.item()
    return value


def save_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(clean(value),indent=2,sort_keys=True,allow_nan=False)+'\n')
    os.replace(temporary,path)


def input_fingerprint(config):
    """Fingerprint code and explicit config bytes, independent of their location."""
    h=hashlib.sha256()
    for label, directory, pattern in [('package',Path(__file__).resolve().parent,'*.py'),
                                      ('config',config.directory,'*.yaml')]:
        for path in sorted(directory.glob(pattern)):
            h.update(f'{label}/{path.name}'.encode());h.update(path.read_bytes())
    if config.spec.get('frozen_selections'):
        h.update(b'frozen_selections')
        h.update((config.directory/config.spec['frozen_selections']).read_bytes())
    return h.hexdigest()


def confirmation_worker(task):
    trajectory,outdir,compact,configdir=task;outdir=Path(outdir)
    start=time.monotonic()
    try:
        config=ExperimentConfig.from_directory(configdir)
        record=run_trajectory(trajectory,config)
        if compact:
            labels=list(PRIMARY)+['projection_removed']+[f'projected_k{k}' for k in config.design['iteration_steps']]
            fits=[[s['fits'][m] for m in list(PRIMARY)+['projection_removed']]+s['iterations'] for s in record['settings']]
            np.savez_compressed(outdir/'records'/f'trajectory_{trajectory:04d}.npz',
                trajectory=np.asarray(trajectory),n0_values=np.asarray([s['n0'] for s in record['settings']]),
                method_labels=np.asarray(labels),risk=np.asarray([[f['risk'] for f in row] for row in fits]),
                c=np.asarray([[f['c'] for f in row] for row in fits]),
                operator_sha256=np.asarray([[f['operator_sha256'] for f in row] for row in fits]),
                task_prefix_data_sha256=np.asarray([s['task_prefix_data_sha256'] for s in record['settings']]),
                full_task_data_sha256=np.asarray(record['full_task_data_sha256']))
        else:
            save_json(outdir/'records'/f'trajectory_{trajectory:04d}.json',record)
        return dict(trajectory=trajectory,status='ok',elapsed_seconds=time.monotonic()-start)
    except Exception as exc:
        failure=dict(trajectory=trajectory,status='failed',error=repr(exc),traceback=traceback.format_exc())
        save_json(outdir/'failures'/f'trajectory_{trajectory:04d}.json',failure)
        return failure


def development_worker(task):
    n,index,outdir,configdir=task;outdir=Path(outdir)
    try:
        arrays,record=run_development(n,index,ExperimentConfig.from_directory(configdir))
        np.savez_compressed(outdir/'records'/f'n{n:04d}_d{index:04d}.npz',**arrays)
        save_json(outdir/'records'/f'n{n:04d}_d{index:04d}.json',record)
        return dict(n0=n,index=index,status=record['status'])
    except Exception as exc:
        failure=dict(n0=n,index=index,status='failed',error=repr(exc),traceback=traceback.format_exc())
        save_json(outdir/'failures'/f'n{n:04d}_d{index:04d}.json',failure)
        return failure


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['confirmation','development'])
    parser.add_argument('--config',type=Path,required=True,help='Experiment config.yaml or its containing directory')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=12)
    parser.add_argument('--limit',type=int,help='Optional smoke-check count; does not declare full reproduction')
    parser.add_argument('--compact',action='store_true',help='Confirmation only: emit risk/coefficient/data hashes in compact NPZ files')
    args=parser.parse_args()
    if args.workers<1 or args.limit is not None and args.limit<1: parser.error('workers/limit must be positive')
    config=ExperimentConfig.from_directory(args.config)
    output=args.output.expanduser().resolve()
    protected=(config.directory,config.directory.parent/'outputs',Path(__file__).resolve().parent)
    if any(output==p or p in output.parents or output in p.parents for p in protected):
        parser.error('Output overlaps supplied configs, retained experiment outputs, or package code; use a separate run directory')
    if args.compact and args.stage!='confirmation': parser.error('--compact applies to confirmation only')
    output.mkdir(parents=True,exist_ok=True)
    marker=output/'RUN_INPUTS.json'
    spec=dict(stage=args.stage,input_fingerprint=input_fingerprint(config),compact=args.compact)
    if marker.exists():
        if json.loads(marker.read_text())!=spec: parser.error('Output belongs to different inputs/stage')
    elif any(output.iterdir()): parser.error('Use a new or empty output directory')
    else: save_json(marker,spec)
    (output/'records').mkdir(exist_ok=True)
    design=config.design
    if args.stage=='confirmation':
        suffix='npz' if args.compact else 'json'
        all_tasks=[(i,str(output),args.compact,str(config.directory)) for i in range(design['confirmation_reps'])]
        tasks=[t for t in all_tasks if not (output/'records'/f'trajectory_{t[0]:04d}.{suffix}').exists()]
        worker=confirmation_worker
    else:
        all_tasks=[(n,i,str(output),str(config.directory)) for n in design['n0_values'] for i in range(design['development_reps'])]
        tasks=[t for t in all_tasks if not (output/'records'/f'n{t[0]:04d}_d{t[1]:04d}.json').exists()]
        worker=development_worker
    if args.limit is not None: tasks=tasks[:args.limit]
    failures=0;start=time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(worker,t) for t in tasks]
        for done,future in enumerate(as_completed(futures),1):
            record=future.result();failures+=record['status']!='ok'
            if done==1 or done%25==0 or done==len(tasks):
                progress=dict(stage=args.stage,completed_this_run=done,total_this_run=len(tasks),failures=failures,elapsed_seconds=time.monotonic()-start)
                save_json(output/'PROGRESS.json',progress);print(json.dumps(progress),flush=True)
    if failures: raise RuntimeError(f'{failures} failures retained; no success-conditioned summary')
    if args.stage=='confirmation':
        complete=all((output/'records'/f'trajectory_{i:04d}.{suffix}').exists() for i in range(design['confirmation_reps']))
        if complete and not args.compact:
            complete=all(json.loads((output/'records'/f'trajectory_{i:04d}.json').read_text())['status']=='ok' for i in range(design['confirmation_reps']))
    else:
        complete=all((output/'records'/f'n{n:04d}_d{i:04d}.npz').exists() for n,i,_,_ in all_tasks)
        if complete:
            registry=candidate_registry(design);banks={n:{m:np.empty((len(registry[m]),design['development_reps'])) for m in PRIMARY} for n in design['n0_values']}
            for n,i,_,_ in all_tasks:
                record=json.loads((output/'records'/f'n{n:04d}_d{i:04d}.json').read_text())
                if record['status']!='ok': raise RuntimeError('Failed development row retained')
                with np.load(output/'records'/f'n{n:04d}_d{i:04d}.npz',allow_pickle=False) as z:
                    for m in PRIMARY: banks[n][m][:,i]=z[m]
            selected=select_candidates(banks,registry,expected_reps=design['development_reps'])
            frozen=design.get('selected_candidates')
            save_json(output/'DEVELOPMENT_SELECTION.json',dict(selected_candidates=selected,
                exact_match_to_frozen=None if frozen is None else selected==frozen))
            if frozen is not None and selected!=frozen: raise AssertionError('Recreated development selections differ from published freeze')
    save_json(output/'COMPLETION.json',dict(stage=args.stage,complete=complete,requested_datasets=len(all_tasks),input_fingerprint=spec['input_fingerprint']))
    print(json.dumps(dict(stage=args.stage,complete=complete)),flush=True)


if __name__=='__main__': main()

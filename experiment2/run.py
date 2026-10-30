#!/usr/bin/env python3
"""Regenerate the paper's frozen-rule advection studies (Figures 2 and 3)."""
from __future__ import annotations

import argparse
import json
import os
from multiprocessing import get_context
from pathlib import Path
import sys

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from experiment2.advection_n0 import COLUMNS, replicate
from transfer_operator.pde_config import PDEStudy
from transfer_operator.pde_replication import bootstrap_resample_means

EXPERIMENT = Path(__file__).resolve().parent


def _worker(task):
    key, config, n0, repetition = task
    risks, supports = replicate(config, n0, repetition)
    return key, repetition, risks, supports


def run(name, outputs, workers=1, replications=None, cell_index=None):
    """Run complete cells; optional bounds are smoke checks, never full reproduction."""
    study = PDEStudy.from_yaml(EXPERIMENT / "configs" / f"{name}.yaml")
    spec = study.specification
    count = int(spec["replications"] if replications is None else replications)
    if workers < 1 or not 1 <= count <= int(spec["replications"]):
        raise ValueError("workers must be positive and replications within the registered range")
    cells = list(study.cells)
    if cell_index is not None:
        cells = [study.cell(name, cell_index)]
    outputs = Path(outputs).resolve()
    retained = (EXPERIMENT / "outputs").resolve()
    if outputs == retained or retained in outputs.parents or outputs in retained.parents:
        raise ValueError("outputs must not overlap the retained paper results")
    keys = {c.archive_cell: (f'n0_{c.setting["n0"]}' if name == "advection_n0"
                             else f'cell{c.archive_cell:03d}') for c in cells}
    risks = {k: np.empty((count, len(COLUMNS))) for k in keys.values()}
    supports = {k: np.empty((count, 2), dtype=int) for k in keys.values()}
    tasks = []
    for c in cells:
        config = dict(spec, sigma=c.setting["sigma"], archived_refits=3)
        tasks.extend((keys[c.archive_cell], config, c.setting["n0"], rep) for rep in range(count))
    with get_context("spawn").Pool(workers) as pool:
        for done, (key, rep, row, exact) in enumerate(pool.imap_unordered(_worker, tasks), 1):
            risks[key][rep], supports[key][rep] = row, exact
            if done == 1 or done % 100 == 0 or done == len(tasks):
                print(json.dumps(dict(study=name, completed=done, total=len(tasks))), flush=True)
    folder = outputs / "results" / name
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / "risks.npz", columns=np.asarray(COLUMNS),
                        **risks, **{f"exact_{key}": values for key, values in supports.items()})
    boot = spec["bootstrap"]
    summaries = []
    for c in cells:
        key = keys[c.archive_cell]
        values = risks[key]
        if not np.isfinite(values).all():
            raise FloatingPointError(f"Nonfinite risk in {name}/{key}")
        means = bootstrap_resample_means(values, int(boot["draws"]), int(boot["seed"]), algorithm="indices")
        lo, hi = np.quantile(means, [.025, .975], axis=0)
        summaries.append(dict(cell=c.archive_cell, key=key, setting=c.setting, replications=count,
                              risk={method: dict(mean=float(values[:, i].mean()),
                                                ci95=[float(lo[i]), float(hi[i])])
                                    for i, method in enumerate(COLUMNS)},
                              exact_supports=dict(projected=int(supports[key][:, 0].sum()),
                                                  unprojected=int(supports[key][:, 1].sum()))))
    summary = dict(study=name, complete=(count == spec["replications"] and cell_index is None),
                   columns=list(COLUMNS), cells=summaries, bootstrap=boot,
                   risk="Equation (75), retained 625-dimensional Fourier space")
    (folder / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("all", "advection_n0", "advection_sigma"), default="all")
    parser.add_argument("--outputs", type=Path, default=ROOT / ".runs/experiment2-base")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--replications", type=int, help="Bound each cell for a smoke check; paper default is 300")
    parser.add_argument("--cell", type=int, help="Run just one original cell id for a smoke check")
    args = parser.parse_args()
    names = ("advection_sigma", "advection_n0") if args.scenario == "all" else (args.scenario,)
    for name in names:
        result = run(name, args.outputs, args.workers, args.replications, args.cell)
        print(json.dumps(dict(study=name, complete=result["complete"], output=str(args.outputs / "results" / name))))


if __name__ == "__main__":
    main()

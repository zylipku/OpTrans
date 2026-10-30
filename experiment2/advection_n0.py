#!/usr/bin/env python3
"""Frozen-rule sample-size study for Figure 3.

The target sizes, sample allocations, random seed and multipliers are specified
in configs/advection_n0.yaml. Each repetition records the pilot, Projected,
two additional recalibration rounds, independently tuned Unprojected,
Target-only and Pooling, together with exact-support indicators.

Run from the package root with ``python experiment2/advection_n0.py --workers 4``.
New records and 20,000-draw paired percentile summaries are written to
.runs/experiment2-base/results/advection_n0. The unified ``experiment2/run.py``
entry point also supports the noise study and single-cell execution checks.
"""
from __future__ import annotations

import argparse
import json
import os
from multiprocessing import get_context
from pathlib import Path
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

RELEASE_ROOT = Path(__file__).resolve().parents[1]
if str(RELEASE_ROOT) not in sys.path:
    sys.path.insert(0, str(RELEASE_ROOT))

import numpy as np
import yaml

from transfer_operator.pde_replication import advection_replicate, bootstrap_resample_means

EXPERIMENT_ROOT = Path(__file__).resolve().parent
CONFIG = EXPERIMENT_ROOT / "configs/advection_n0.yaml"
RESULTS = "results/advection_n0"
COLUMNS = ("projected_k0", "projected_k1", "projected_k2", "projected_k3",
           "unprojected", "target-only", "pooling")


def load_config(path=CONFIG):
    return yaml.safe_load(Path(path).read_text())


def candidates(config):
    m = config["frozen_multipliers"]
    return [dict(m["Projected"], role="projected"), dict(m["Unprojected"], role="unprojected")]


def cell(config, n0):
    return dict(R=config["radius"], s=float(config["smoothness"]), sigma=float(config["sigma"]),
                tau=float(config["tau"]), n0=int(n0), nS=int(config["source_total"]),
                profile=config["noise_profile"],
                baseline_multipliers={"target-only": [float(config["frozen_multipliers"]["Target_only"])],
                                      "pooling": [float(config["frozen_multipliers"]["Pooling"])]})


def replicate(config, n0, index):
    """(risks[COLUMNS], exact_supports[P, U]) for one replication of one size."""
    result = advection_replicate(cell(config, n0), candidates(config), int(config["seed"]), int(index),
                                 additional_rounds=int(config["archived_refits"]) - 1)
    p, u = result["candidates"]
    t_mult = float(config["frozen_multipliers"]["Target_only"])
    pool_mult = float(config["frozen_multipliers"]["Pooling"])
    risks = np.array([p[f"risk{k}"] for k in range(4)]
                     + [u["projection_off_risk"], result["baselines"][("target-only", t_mult)],
                        result["baselines"][("pooling", pool_mult)]])
    exact = np.array([int(tuple(s) == ((0,), (1,))) for s in (p["supports"], u["supports"])])
    return risks, exact


def _worker(args):
    config, n0, index = args
    risks, exact = replicate(config, n0, index)
    return n0, index, risks, exact


def summarize(config, risks, exact):
    """SUMMARY.json content from the per-replication arrays (dicts keyed by n0)."""
    boot = config["bootstrap"]
    cells = []
    for n0 in config["n0_grid"]:
        means = bootstrap_resample_means(risks[n0], int(boot["draws"]), int(boot["seed"]), algorithm="indices")
        lo, hi = np.quantile(means, [.025, .975], axis=0)
        cells.append(dict(
            setting=dict(R=config["radius"], s=float(config["smoothness"]), tau=float(config["tau"]),
                         sigma=float(config["sigma"]), nS=int(config["source_total"]), n0=int(n0)),
            N=int(config["source_total"]) + int(n0), replicates=int(risks[n0].shape[0]),
            risk={c: dict(mean=float(risks[n0][:, i].mean()), ci95=[float(lo[i]), float(hi[i])])
                  for i, c in enumerate(COLUMNS)},
            support=dict(projected_exact=int(exact[n0][:, 0].sum()), unprojected_exact=int(exact[n0][:, 1].sum()))))
    return dict(config=dict(name="advection_n0_sweep", source="experiment2/configs/advection_n0.yaml",
                            setting=dict(R=config["radius"], s=float(config["smoothness"]), tau=float(config["tau"]),
                                         sigma=float(config["sigma"]), nS=int(config["source_total"])),
                            n0_grid=[int(n) for n in config["n0_grid"]], replicates=int(config["replications"]),
                            seed=int(config["seed"]), bootstrap_seed=int(boot["seed"]), bootstrap_reps=int(boot["draws"]),
                            K=int(config["archived_refits"]), frozen_multipliers=config["frozen_multipliers"]),
                risk_columns=list(COLUMNS), cells=cells,
                inference="20,000 paired whole-trajectory bootstrap draws (integer indices -> counts -> weights, "
                          "as the advection archive); pointwise 95% percentile intervals.")


def run(config, outputs, workers, replications=None):
    replications = int(config["replications"] if replications is None else replications)
    if workers < 1 or not 1 <= replications <= int(config["replications"]):
        raise ValueError("workers must be positive; replications must be in the registered range")
    outputs = Path(outputs).resolve()
    retained = (EXPERIMENT_ROOT / "outputs").resolve()
    if outputs == retained or retained in outputs.parents or outputs in retained.parents:
        raise ValueError("outputs must not overlap retained paper results")
    directory = outputs / RESULTS
    directory.mkdir(parents=True, exist_ok=True)
    risks = {n0: np.empty((replications, len(COLUMNS))) for n0 in config["n0_grid"]}
    exact = {n0: np.empty((replications, 2), int) for n0 in config["n0_grid"]}
    tasks = [(config, n0, i) for n0 in config["n0_grid"] for i in range(replications)]
    start = time.time()
    with get_context("fork").Pool(workers) as pool:
        for n0, index, row, ex in pool.imap_unordered(_worker, tasks, chunksize=2):
            risks[n0][index] = row
            exact[n0][index] = ex
    summary = summarize(config, risks, exact)
    summary["config"]["replicates"] = replications
    summary["complete"] = replications == int(config["replications"])
    summary["config"]["wall_minutes"] = round((time.time() - start) / 60, 2)
    summary["config"]["workers"] = workers
    (directory / "SUMMARY.json").write_text(json.dumps(summary, indent=1) + "\n")
    np.savez_compressed(directory / "risks.npz", columns=np.asarray(COLUMNS),
                        **{f"n0_{n0}": risks[n0] for n0 in config["n0_grid"]},
                        **{f"exact_{n0}": exact[n0] for n0 in config["n0_grid"]})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--outputs", type=Path, default=RELEASE_ROOT / ".runs/experiment2-base")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--replications", type=int, help="override (smoke runs); default from the config")
    args = parser.parse_args()
    summary = run(load_config(), args.outputs, args.workers, args.replications)
    print(f"{summary['config']['wall_minutes']} min -> {args.outputs / RESULTS}")
    for c in summary["cells"]:
        r = c["risk"]
        print(f"n0={c['setting']['n0']:5d} T={r['target-only']['mean']:.3e} Pool={r['pooling']['mean']:.3e} "
              f"U={r['unprojected']['mean']:.3e} P={r['projected_k1']['mean']:.3e} "
              f"RC1={r['projected_k2']['mean']:.3e} RC2={r['projected_k3']['mean']:.3e} "
              f"exact P/U={c['support']['projected_exact']}/{c['support']['unprojected_exact']}")


if __name__ == "__main__":
    main()

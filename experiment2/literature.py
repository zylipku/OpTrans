#!/usr/bin/env python3
"""Literature baselines in Experiment 2 (Section 4.3, advection-diffusion).

Adds the estimators of Jiao & McKeague (2025), Lin & Reimherr (2024) and Hu & Lin
(2025) (``transfer_operator.literature``) to the registered studies of Figure 3
(target-sample-size sweep, configs/advection_n0.yaml) and Figure 2 (noise sweep,
configs/advection_sigma.yaml) without changing them. Each new method is tuned on
300 development replications drawn with independent seeds (configs/literature.yaml)
and frozen before evaluation; evaluation regenerates the registered observations
replication by replication, and recomputes Target only and Pooling with the package
estimators, which must match retained risks within absolute tolerance 1e-12 (zero relative tolerance).

    python experiment2/literature.py development --workers 14
    python experiment2/literature.py select
    python experiment2/literature.py evaluation --workers 14
    python experiment2/literature.py summarize
    python experiment2/literature.py diagnostics --workers 14

Development matrices stay in --run-dir (default release/.runs/experiment2-literature);
new results go to --results-dir (default .runs/experiment2-literature/results).
Frozen inputs are read from outputs/results/literature unless reselected in --results-dir.
The diagnostics stage reproduces the spectral analysis in Appendix D.3.1.
"""
from __future__ import annotations

import argparse
import csv
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

from transfer_operator.literature.advection import AdvectionWorld, units
from transfer_operator.literature.core import LABELS, REPORTED as METHODS, frozen_row, registry, select_development
from transfer_operator.pde_estimators import pooled_fit, ridge_fit
from transfer_operator.pde_models import AdvectionDiffusionModel
from transfer_operator.pde_moments import generate_task
from transfer_operator.pde_records import PackedRecords
from transfer_operator.pde_replication import bootstrap_resample_means

EXPERIMENT_ROOT = Path(__file__).resolve().parent
CONFIGS = EXPERIMENT_ROOT / "configs"
OUTPUTS = EXPERIMENT_ROOT / "outputs"
BASE_OUTPUTS = OUTPUTS
REFERENCE_RESULTS = OUTPUTS / "results/literature"
RESULTS = RELEASE_ROOT / ".runs/experiment2-literature/results"
DEFAULT_RUN = RELEASE_ROOT / ".runs/experiment2-literature"
PARAMETERS = ("sources", "truncation", "source_ridge", "target_ridge", "lasso")
REGISTERED = ("target-only", "pooling", "unprojected", "projected", "projected_rc1", "projected_rc2")
NAMES = {"target-only": "Target only", "pooling": "Pooling", "unprojected": "Unprojected",
         "projected": "Projected", "projected_rc1": "Projected, one additional round",
         "projected_rc2": "Projected, two additional rounds", **LABELS}


def load_spec():
    spec = yaml.safe_load((CONFIGS / "literature.yaml").read_text())
    if spec.get("version") != 1:
        raise ValueError("Expected version: 1 in literature.yaml")
    grid = dict(truncation=[int(x) for x in spec["truncation"]],
                target_ridge=[float(x) for x in spec["target_ridge"]],
                source_ridge=[float(x) for x in spec["source_ridge"]],
                lasso=[float(x) for x in spec["lasso"]])
    return spec, grid


def scenario(name):
    """Cells (key, sigma, n0) and the registered settings of one scenario."""
    spec, _ = load_spec()
    entry = spec["scenarios"][name]
    config = yaml.safe_load((CONFIGS / entry["config"]).read_text())
    if name == "advection_n0":
        cells = [(f"n0_{n0}", float(config["sigma"]), int(n0)) for n0 in config["n0_grid"]]
        multipliers = config["frozen_multipliers"]
        bootstrap = config["bootstrap"]
        boot = dict(seed=int(bootstrap["seed"]), draws=int(bootstrap["draws"]))
    else:
        cells = [(f"cell{index:03d}", float(sigma), int(n0))
                 for n0 in config["n0_grid"] for index, sigma in zip(config["archive_cells"][n0], config["sigma_grid"])]
        multipliers = config["frozen_multipliers"]
        boot = dict(seed=int(config["bootstrap"]["seed"]), draws=int(config["bootstrap"]["draws"]))
    return dict(name=name, cells=cells, radius=int(config["radius"]), s=float(config["smoothness"]),
                tau=float(config["tau"]), ns=int(config["source_total"]), seed=int(config["seed"]),
                development_seed=int(entry["development_seed"]), replications=int(config["replications"]),
                target_multiplier=float(multipliers["Target_only"]), pooling_multiplier=float(multipliers["Pooling"]),
                bootstrap=boot)


_STATE = {}


def _state():
    if not _STATE:
        spec, grid = load_spec()
        _STATE.update(spec=spec, registry=registry(grid, 2),
                      scenarios={name: scenario(name) for name in spec["scenarios"]})
    return _STATE


def _replicate(task):
    """One replication of one noise level: shared sources, every target size."""
    name, seed, rep, sigma, mode = task
    state = _state()
    sc = state["scenarios"][name]
    model = AdvectionDiffusionModel(R=sc["radius"], s=sc["s"], sigma=sigma, tau=sc["tau"])
    allocation = [sc["ns"] // 2] * 2
    sources = [generate_task(model, nj, theta, [seed, rep, j + 1], profile="full")
               for j, (nj, theta) in enumerate(zip(allocation, model.thetas))]
    exponent = model.s / (model.s + 1)
    results = {}
    for key, cell_sigma, n0 in sc["cells"]:
        if cell_sigma != sigma:
            continue
        target, F, U = generate_task(model, n0, [0., 0.], [seed, rep, 0], profile="full", return_samples=True)
        world = AdvectionWorld(model, target, F, U, sources)
        if mode == "development":
            results[key] = {m: world.development(m, state["registry"][m]) for m in METHODS}
            continue
        frozen = state["frozen"][name][key]
        risks, diag = np.full(len(METHODS), np.nan), np.full((2, 5), np.nan)
        for mi, m in enumerate(METHODS):
            value, coef = world.evaluate(m, [frozen[m]], coefficients=True)
            risks[mi] = value[0]
            if m in ("lr_atl", "hl_agg"):
                info = coef[0]
                diag[0 if m == "lr_atl" else 1] = (info["sets"][1][0], info["first"], info["second"], info["weight"],
                                                   info.get("preselected", len(info["sets"])))
        target_only = model.risk(ridge_fit(model, target, sc["target_multiplier"] * n0 ** (-exponent)))
        pooling = model.risk(pooled_fit(model, [target] + sources, sc["pooling_multiplier"] * (n0 + sc["ns"]) ** (-exponent)))
        results[key] = dict(risks=risks, aggregation=diag, pairing=np.array([target_only, pooling]))
    return name, rep, results


def _tasks(mode):
    state = _state()
    tasks = []
    for name, sc in state["scenarios"].items():
        seed = sc["development_seed"] if mode == "development" else sc["seed"]
        reps = int(state["spec"]["development_replications"]) if mode == "development" else sc["replications"]
        for sigma in sorted({c[1] for c in sc["cells"]}):
            tasks += [(name, seed, rep, sigma, mode) for rep in range(reps)]
    return tasks


def development(run_dir, workers):
    state = _state()
    reps = int(state["spec"]["development_replications"])
    banks = {name: {key: {} for key, _, _ in sc["cells"]} for name, sc in state["scenarios"].items()}
    tasks = _tasks("development")
    start = time.time()
    with get_context("fork").Pool(workers) as pool:
        for done, (name, rep, results) in enumerate(pool.imap_unordered(_replicate, tasks), 1):
            for key, risks in results.items():
                for m in METHODS:
                    if m not in banks[name][key]:
                        banks[name][key][m] = np.full((reps,) + np.shape(risks[m]), np.nan)
                    banks[name][key][m][rep] = risks[m]
            if done % 100 == 0 or done == len(tasks):
                print(json.dumps(dict(stage="development", done=done, total=len(tasks),
                                      minutes=round((time.time() - start) / 60, 2))), flush=True)
    folder = Path(run_dir) / "development"
    folder.mkdir(parents=True, exist_ok=True)
    for name, cells in banks.items():
        np.savez_compressed(folder / f"{name}.npz", **{f"{key}__{m}": arrays[m]
                                                       for key, arrays in cells.items() for m in METHODS})


def select(run_dir):
    state = _state()
    for name, sc in state["scenarios"].items():
        frozen, summary = [], {}
        with np.load(Path(run_dir) / "development" / f"{name}.npz") as bank:
            for key, sigma, n0 in sc["cells"]:
                for m in METHODS:
                    risks = bank[f"{key}__{m}"]
                    indices, means, mcses = select_development(risks)
                    cube = risks[:, None, :] if risks.ndim == 2 else risks
                    finite = np.isfinite(cube).all(axis=0)
                    frozen.append(dict(cell=key, sigma=sigma, n0=n0, method=m, index=json.dumps(indices),
                                       parameters=json.dumps(frozen_row(m, state["registry"][m], indices)),
                                       development_mean=json.dumps(means), development_mcse=json.dumps(mcses),
                                       finite_candidates=json.dumps(finite.sum(axis=1).tolist()),
                                       candidates=len(state["registry"][m])))
                    summary[f"{key}__{m}__mean"] = np.where(finite, cube.mean(axis=0), np.nan)
        folder = RESULTS / name
        folder.mkdir(parents=True, exist_ok=True)
        keys = list(frozen[0])
        with (folder / "frozen_parameters.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=keys, lineterminator="\n")
            writer.writeheader()
            writer.writerows(frozen)
        np.savez_compressed(folder / "development_summary.npz", **summary)
        print(json.dumps(dict(scenario=name, frozen=[(r["cell"], r["method"], r["index"]) for r in frozen])), flush=True)


def _input(name, filename):
    """Read a newly produced stage input, or its frozen paper counterpart."""
    candidate = RESULTS / name / filename
    return candidate if candidate.is_file() else REFERENCE_RESULTS / name / filename


def frozen_rows(name):
    state = _state()
    rows = {}
    with _input(name, "frozen_parameters.csv").open(newline="") as stream:
        for record in csv.DictReader(stream):
            method, indices = record["method"], json.loads(record["index"])
            parameters = json.loads(record["parameters"])
            if parameters != frozen_row(method, state["registry"][method], indices):
                raise ValueError("Frozen literature parameters differ from the candidate registry")
            rows.setdefault(record["cell"], {})[method] = parameters
    return rows


def archived(name, key):
    """Archived per-replication risks of the registered estimators, keyed as REGISTERED."""
    compact = BASE_OUTPUTS / "results" / name / "risks.npz"
    if compact.is_file():
        with np.load(compact) as data:
            columns = list(data["columns"])
            risks = data[key]
        pick = dict(zip(REGISTERED, ("target-only", "pooling", "unprojected", "projected_k1",
                                     "projected_k2", "projected_k3")))
        return {k: risks[:, columns.index(v)] for k, v in pick.items()}
    packed = PackedRecords(BASE_OUTPUTS / "data/advection_sigma" / f"{key}.npz")
    rows = []
    for rep in range(len(packed)):
        record = packed.record(rep)
        methods, candidates = record["method"], record["candidate"]
        p = np.flatnonzero((methods == "projected") & (candidates == 0)).item()
        u = np.flatnonzero((methods == "projected") & (candidates == 1)).item()
        t = np.flatnonzero(methods == "target-only").item()
        pool = np.flatnonzero(methods == "pooling").item()
        rows.append([record["risk"][t], record["risk"][pool], record["projection_off_risk"][u],
                     record["risk1"][p], record["risk2"][p], record["risk3"][p]])
    rows = np.asarray(rows, dtype=float)
    return {k: rows[:, i] for i, k in enumerate(REGISTERED)}


def evaluation(workers):
    state = _state()
    state["frozen"] = {name: frozen_rows(name) for name in state["scenarios"]}
    store = {name: {key: dict(risks=np.full((sc["replications"], len(METHODS)), np.nan),
                              aggregation=np.full((sc["replications"], 2, 5), np.nan),
                              pairing=np.full((sc["replications"], 2), np.nan))
                    for key, _, _ in sc["cells"]} for name, sc in state["scenarios"].items()}
    tasks = _tasks("evaluation")
    start = time.time()
    with get_context("fork").Pool(workers) as pool:
        for done, (name, rep, results) in enumerate(pool.imap_unordered(_replicate, tasks), 1):
            for key, values in results.items():
                for field, value in values.items():
                    store[name][key][field][rep] = value
            if done % 100 == 0 or done == len(tasks):
                print(json.dumps(dict(stage="evaluation", done=done, total=len(tasks),
                                      minutes=round((time.time() - start) / 60, 2))), flush=True)
    for name, cells in store.items():
        worst = 0.0
        for key, values in cells.items():
            reference = archived(name, key)
            difference = np.abs(values["pairing"] - np.column_stack([reference["target-only"], reference["pooling"]]))
            if not np.isfinite(difference).all():
                raise FloatingPointError(f"Nonfinite pairing risk in {name}/{key}")
            worst = max(worst, float(difference.max()))
            if not np.isfinite(values["risks"]).all():
                raise FloatingPointError(f"Nonfinite literature risk in {name}/{key}")
        if worst > 1e-12:
            raise AssertionError(f"{name}: recomputed Target only/Pooling differ from the archive by {worst}")
        folder = RESULTS / name
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder / "risks.npz", methods=np.asarray(METHODS),
                            **{f"{key}__{field}": value for key, values in cells.items() for field, value in values.items()})
        print(json.dumps(dict(scenario=name, pairing_max_abs_difference=worst)), flush=True)


def summarize():
    state = _state()
    for name, sc in state["scenarios"].items():
        (RESULTS / name).mkdir(parents=True, exist_ok=True)
        boot = sc["bootstrap"]
        cells, paired, preselected = [], [], []
        with np.load(_input(name, "risks.npz")) as data:
            for key, sigma, n0 in sc["cells"]:
                preselected.append(data[f"{key}__aggregation"][:, 0, 4])   # Lin-Reimherr candidates kept
                reference = archived(name, key)
                matrix = np.column_stack([reference[k] for k in REGISTERED] + [data[f"{key}__risks"]])
                keys = list(REGISTERED) + list(METHODS)
                means = bootstrap_resample_means(matrix, boot["draws"], boot["seed"], algorithm="indices")
                lo, hi = np.quantile(means, [.025, .975], axis=0)
                cells.append(dict(cell=key, sigma=sigma, n0=n0, replications=int(matrix.shape[0]),
                                  risk={k: dict(mean=float(matrix[:, i].mean()), ci95=[float(lo[i]), float(hi[i])])
                                        for i, k in enumerate(keys)}))
                for left in ("projected", "unprojected", "target-only"):
                    for right in METHODS:
                        li, ri = keys.index(left), keys.index(right)
                        ratio = means[:, ri] / means[:, li]
                        reduction = 100 * (1 - means[:, li] / means[:, ri])
                        paired.append(dict(cell=key, sigma=sigma, n0=n0, method_key=left, comparator_key=right,
                                           ratio_of_means=float(matrix[:, ri].mean() / matrix[:, li].mean()),
                                           ratio_ci_low=float(np.quantile(ratio, .025)),
                                           ratio_ci_high=float(np.quantile(ratio, .975)),
                                           percentage_reduction=float(100 * (1 - matrix[:, li].mean() / matrix[:, ri].mean())),
                                           reduction_ci_low=float(np.quantile(reduction, .025)),
                                           reduction_ci_high=float(np.quantile(reduction, .975)),
                                           fraction_left_better=float(np.mean(matrix[:, li] < matrix[:, ri]))))
        preselected = np.concatenate(preselected)
        summary = dict(scenario=name, methods=keys, labels={k: NAMES[k] for k in keys}, cells=cells,
                       bootstrap=dict(seed=boot["seed"], draws=boot["draws"], algorithm="indices",
                                      unit="complete replication, jointly across seven methods and two recalibration rounds"),
                       atl_preselection=dict(cell_replications=int(preselected.size),
                                             kept_every_candidate=int(np.sum(preselected == 3))),
                       registered_methods="archived records; not recomputed")
        (RESULTS / name / "SUMMARY.json").write_text(json.dumps(summary, indent=1) + "\n")
        keys_out = list(paired[0])
        with (RESULTS / name / "paired_comparisons.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=keys_out, lineterminator="\n")
            writer.writeheader()
            writer.writerows(paired)
        print(json.dumps(dict(scenario=name, cells=len(cells))), flush=True)


TOP_UNITS = (6,)


def _diagnostic(rep):
    """Unit-wise risks of one registered replication of the n0 sweep (sigma of Figure 3)."""
    state = _state()
    sc = state["scenarios"]["advection_n0"]
    model = AdvectionDiffusionModel(R=sc["radius"], s=sc["s"], sigma=sc["cells"][0][1], tau=sc["tau"])
    blocks = units(model)
    unit_risk = lambda h: np.array([np.sum(model.q[b] * (h[b] - model.truth[b]) ** 2) for b in blocks])
    sources = [generate_task(model, nj, theta, [sc["seed"], rep, j + 1], profile="full")
               for j, (nj, theta) in enumerate(zip([sc["ns"] // 2] * 2, model.thetas))]
    out = {}
    for key, _, n0 in sc["cells"]:
        target, F, U = generate_task(model, n0, [0., 0.], [sc["seed"], rep, 0], profile="full",
                                     return_samples=True)
        world = AdvectionWorld(model, target, F, U, sources)
        out[key] = dict(target=unit_risk(world.target.g / world.target.G),
                        first_half=world.risks(world.first.g / world.first.G))
    return out


def diagnostics(workers):
    """Reproduce Appendix D.3.1's constant-mode risk and low-frequency source contrast."""
    state = _state()
    sc = state["scenarios"]["advection_n0"]
    sigma, reps = sc["cells"][0][1], sc["replications"]
    model = AdvectionDiffusionModel(R=sc["radius"], s=sc["s"], sigma=sigma, tau=sc["tau"])
    blocks = units(model)
    with get_context("fork").Pool(workers) as pool:
        results = pool.map(_diagnostic, range(reps))
    contrast = []
    for j, theta in enumerate(model.thetas, 1):
        source = model.source_h(theta)
        energy = np.array([np.sum(model.q[b] * (source - model.truth)[b] ** 2) for b in blocks])
        sine = lambda w: float(np.sqrt(max(0.0, 1 - np.sum(w * source * model.truth) ** 2
                                           / (np.sum(w * source ** 2) * np.sum(w * model.truth ** 2)))))
        share = np.cumsum(np.sort(energy)[::-1]) / energy.sum()
        # |sin angle(A(j), A(0))| in H (Hilbert-Schmidt, the inner product of shape transfer) and in the risk.
        contrast.append(dict(source=j, risk=float(energy.sum()), constant_mode=float(energy[0]),
                             abs_sin_angle=sine(np.ones_like(model.q)),
                             share_in_top_units={str(k): float(share[k - 1]) for k in TOP_UNITS}))
    cells = []
    for key, _, n0 in sc["cells"]:
        target = np.mean([r[key]["target"] for r in results], axis=0)
        cells.append(dict(cell=key, n0=n0, target_least_squares=float(target.sum()),
                          constant_mode_share=float(target[0] / target.sum()),
                          target_least_squares_first_half=float(np.mean([r[key]["first_half"] for r in results]))))
        print(json.dumps(dict(stage="diagnostics", **cells[-1])), flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "DIAGNOSTICS.json").write_text(json.dumps(dict(
        scenario="advection_n0", sigma=sigma, replications=reps, seed=sc["seed"], units=len(blocks),
        contrast=contrast, cells=cells), indent=1) + "\n")


def main():
    global RESULTS, BASE_OUTPUTS
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("stage", choices=["development", "select", "evaluation", "summarize", "diagnostics",
                                          "all"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--results-dir", type=Path, default=RESULTS,
                        help="New result directory; frozen package results remain read-only")
    parser.add_argument("--base-outputs", type=Path, default=BASE_OUTPUTS,
                        help="Retained base outputs or fresh experiment2/run.py output directory")
    args = parser.parse_args()
    BASE_OUTPUTS = args.base_outputs.resolve()
    if args.workers < 1:
        parser.error("workers must be positive")
    RESULTS = args.results_dir.resolve()
    retained = OUTPUTS.resolve()
    if RESULTS == retained or retained in RESULTS.parents or RESULTS in retained.parents:
        parser.error("results-dir must not overlap the retained outputs")
    if args.stage in ("development", "all"):
        development(args.run_dir, args.workers)
    if args.stage in ("select", "all"):
        select(args.run_dir)
    if args.stage in ("evaluation", "all"):
        evaluation(args.workers)
    if args.stage in ("summarize", "all"):
        summarize()
    if args.stage in ("diagnostics", "all"):
        diagnostics(args.workers)


if __name__ == "__main__":
    main()

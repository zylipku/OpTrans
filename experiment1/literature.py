#!/usr/bin/env python3
"""Literature baselines in Experiment 1 (Section 4.2), on the registered study.

Adds the estimators of Jiao & McKeague (2025), Lin & Reimherr (2024) and Hu & Lin
(2025) (``transfer_operator.literature``) to the registered functional-regression
experiment without changing it: the same 500 development datasets per n0 select
each new method's candidate by the smallest mean true risk (73), and the same 1000
nested confirmation trajectories evaluate the frozen choices. Every confirmation
setting also recomputes Target only and Pooling with the package estimators and
checks them against the archived risk cube, so the new risks are paired with the
archived ones observation for observation. The literature estimators use the exact
least-squares block coefficient c = 1. Only the three methods reported in the final
paper (``literature.core.REPORTED``) are evaluated and retained.

    python experiment1/literature.py development --workers 14   # 13 x 500 worlds
    python experiment1/literature.py select
    python experiment1/literature.py confirmation --workers 14  # 1000 trajectories
    python experiment1/literature.py summarize

Development matrices stay in --run-dir (default .runs/experiment1-literature).
New numerical outputs go to --output (default .runs/experiment1-literature/outputs).
Published reference outputs are read-only. Confirmation uses their frozen
parameters unless --parameters explicitly names a newly selected CSV.
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

from transfer_operator import ExperimentConfig
from transfer_operator.estimators import FoldFit, SourceBank, candidate_object
from transfer_operator.literature.core import LABELS, REPORTED as METHODS, frozen_row, registry, select_development
from transfer_operator.literature.functional import FunctionalWorld
from transfer_operator.reporting import (
    bootstrap_means, interval, load_data, mcse, paired_statistics, save_json, write_csv,
)
from transfer_operator.simulation import array_sha, nested_samples, prefix_statistics, stats_v2

EXPERIMENT_ROOT = Path(__file__).resolve().parent
CONFIG = EXPERIMENT_ROOT / "configs"
OUTPUTS = EXPERIMENT_ROOT / "outputs"
FROZEN_RESULTS = OUTPUTS / "results/literature"
REFERENCE_DATA = OUTPUTS / "data/literature_confirmation.npz"
DEFAULT_RUN = RELEASE_ROOT / ".runs/experiment1-literature"
DEFAULT_OUTPUT = DEFAULT_RUN / "outputs"
RESULTS = DEFAULT_OUTPUT / "results/literature"
DATA = DEFAULT_OUTPUT / "data/literature_confirmation.npz"
REGISTERED = ("pilot", "pooling", "unprojected", "projected_1")
NAMES = dict(pilot="Target only", pooling="Pooling", unprojected="Unprojected", projected_1="Projected",
             **LABELS)


def load_literature(config):
    spec = yaml.safe_load((CONFIG / "literature.yaml").read_text())
    if spec.get("version") != 1:
        raise ValueError("Expected version: 1 in literature.yaml")
    grid = dict(truncation=[int(x) for x in spec["truncation"]],
                target_ridge=[float(x) for x in spec["target_ridge"]],
                source_ridge=[float(x) for x in spec["source_ridge"]],
                lasso=[float(x) for x in spec["lasso"]])
    registered = config.design["grid"]
    for key, name in (("target_ridge", "lam0_frac"), ("source_ridge", "lambar_frac"), ("lasso", "lamj_frac")):
        if grid[key][:len(registered[name])] != registered[name]:
            raise ValueError(f"literature.yaml {key} must start with the registered {name} grid")
    return spec, grid


_STATE = {}


def _state():
    if not _STATE:
        config = ExperimentConfig.from_directory(CONFIG)
        spec, grid = load_literature(config)
        _STATE.update(config=config, spec=spec, registry=registry(grid, config.model.m))
    return _STATE


def development_world(n0, index):
    """The registered development dataset with the target sample itself."""
    config = _state()["config"]
    design, model = config.design, config.model
    setting = next(s for s in design["settings"] if s["n0"] == n0)
    exp_id = design["development_exp_ids"][str(n0)]
    chunk = design["numerics"]["chunk"]
    target = stats_v2(model, n0, exp_id, index, 0, chunk=chunk, explicit=True)
    sources = [stats_v2(model, setting[f"n{j}"], exp_id, index, j, chunk=chunk)
               for j in range(1, model.m + 1)]
    return target["f"], target["y"], sources


def _development_task(task):
    n0, index = task
    state = _state()
    f, y, sources = development_world(n0, index)
    world = FunctionalWorld(state["config"].model, f, y, sources)
    return n0, index, {m: world.development(m, state["registry"][m]) for m in METHODS}, array_sha(f, y)


def development(run_dir, workers):
    state = _state()
    design = state["config"].design
    reps = design["development_reps"]
    folder = Path(run_dir) / "development"
    folder.mkdir(parents=True, exist_ok=True)
    sizes = [n for n in design["n0_values"] if not (folder / f"n{n:04d}.npz").exists()]
    banks = {n: {} for n in sizes}
    hashes = {n: [""] * reps for n in sizes}
    remaining = {n: reps for n in sizes}
    tasks = [(n, i) for n in sizes for i in range(reps)]
    start = time.time()
    with get_context("fork").Pool(workers) as pool:
        for done, (n0, index, risks, sha) in enumerate(pool.imap_unordered(_development_task, tasks, chunksize=4), 1):
            for m in METHODS:
                if m not in banks[n0]:
                    banks[n0][m] = np.full((reps,) + np.shape(risks[m]), np.nan)
                banks[n0][m][index] = risks[m]
            hashes[n0][index] = sha
            remaining[n0] -= 1
            if not remaining[n0]:
                np.savez_compressed(folder / f"n{n0:04d}.npz", target_sha256=np.asarray(hashes[n0]), **banks[n0])
                del banks[n0]
            if done % 250 == 0 or done == len(tasks):
                print(json.dumps(dict(stage="development", done=done, total=len(tasks),
                                      minutes=round((time.time() - start) / 60, 2))), flush=True)


def select(run_dir):
    """Freeze each method's smallest mean development risk (finite in every world)."""
    state = _state()
    design = state["config"].design
    folder = Path(run_dir) / "development"
    frozen, summary = [], {}
    for n0 in design["n0_values"]:
        with np.load(folder / f"n{n0:04d}.npz") as bank:
            for m in METHODS:
                risks = bank[m]
                if risks.shape[0] != design["development_reps"] or risks.shape[-1] != len(state["registry"][m]):
                    raise ValueError(f"Incomplete development bank for {m} at n0={n0}")
                indices, means, mcses = select_development(risks)
                cube = risks[:, None, :] if risks.ndim == 2 else risks
                finite = np.isfinite(cube).all(axis=0)
                frozen.append(dict(n0=n0, method=m, index=json.dumps(indices),
                                   parameters=json.dumps(frozen_row(m, state["registry"][m], indices)),
                                   development_mean=json.dumps(means), development_mcse=json.dumps(mcses),
                                   finite_candidates=json.dumps(finite.sum(axis=1).tolist()),
                                   candidates=len(state["registry"][m])))
                summary[f"n{n0:04d}_{m}_mean"] = np.where(finite, cube.mean(axis=0), np.nan)
    RESULTS.mkdir(parents=True, exist_ok=True)
    keys = list(frozen[0])
    with (RESULTS / "frozen_parameters.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(frozen)
    np.savez_compressed(RESULTS / "development_summary.npz", **summary)
    print(json.dumps([dict(n0=r["n0"], method=r["method"], index=r["index"], mean=r["development_mean"])
                      for r in frozen]), flush=True)


def frozen_rows(parameters=None):
    """Load the published freeze, or an explicitly supplied selection CSV."""
    state = _state()
    rows = {}
    path = FROZEN_RESULTS / "frozen_parameters.csv" if parameters is None else Path(parameters).expanduser().resolve()
    with path.open(newline="") as stream:
        for record in csv.DictReader(stream):
            method, indices = record["method"], json.loads(record["index"])
            n0 = int(record["n0"])
            if method not in METHODS or method in rows.get(n0, {}):
                raise ValueError("Unknown or duplicate frozen literature method")
            parameters = json.loads(record["parameters"])
            if parameters != frozen_row(method, state["registry"][method], indices):
                raise ValueError("Frozen literature parameters differ from the candidate registry")
            rows.setdefault(n0, {})[method] = parameters
    if set(rows) != set(state["config"].design["n0_values"]) or any(set(r) != set(METHODS) for r in rows.values()):
        raise ValueError("Frozen parameters must include every target size and reported method")
    return rows


def _confirmation_task(trajectory):
    state = _state()
    config = state["config"]
    design, model = config.design, config.model
    frozen, archive = state["frozen"], state["archive"]
    samples, identities = nested_samples(model, design["confirmation_exp_id"], trajectory,
                                         design["settings"], chunk=design["numerics"]["chunk"])
    sizes = len(design["settings"])
    risks = np.full((sizes, len(METHODS)), np.nan)
    agg = np.full((sizes, 2, 5), np.nan)   # [lr_atl, hl_agg] x [first-ranked source, first, second, weight, preselected]
    amplitude = np.full(sizes, np.nan)
    pairing = 0.0
    for si, setting in enumerate(design["settings"]):
        tst, sst, _ = prefix_statistics(model, samples, setting)
        n0 = setting["n0"]
        world = FunctionalWorld(model, samples[0][0][:n0], samples[0][1][:n0], sst)
        for mi, m in enumerate(METHODS):
            value, coef = world.evaluate(m, [frozen[n0][m]], coefficients=True)
            risks[si, mi] = value[0]
            info = coef.get(0, {})
            if m == "jm_ts" and info:
                amplitude[si] = info["amplitude"]
            if m in ("lr_atl", "hl_agg") and info:
                k = 0 if m == "lr_atl" else 1
                agg[si, k] = (info["sets"][1][0], info["first"], info["second"], info["weight"],
                              info.get("preselected", len(info["sets"])))
        fold = FoldFit(model, tst, SourceBank(model, sst, rank_rtol=design["numerics"]["rank_rtol"]))
        for k, method in enumerate(("pilot", "pooling")):
            c, a, _ = fold.fit(candidate_object(design["selected_candidates"][str(n0)][method]), method)
            pairing = max(pairing, abs(model.risk(c, a) - archive[trajectory, si, k]))
    return trajectory, risks, agg, amplitude, pairing, identities["full_task_data_sha256"]


def confirmation(workers, parameters=None):
    state = _state()
    design = state["config"].design
    _, cube = load_data(OUTPUTS)
    state.update(frozen=frozen_rows(parameters), archive=cube)
    reps, sizes = design["confirmation_reps"], len(design["settings"])
    risks = np.full((reps, sizes, len(METHODS)), np.nan)
    agg = np.full((reps, sizes, 2, 5), np.nan)
    amplitude = np.full((reps, sizes), np.nan)
    pairing = np.full(reps, np.nan)
    hashes = [None] * reps
    start = time.time()
    with get_context("fork").Pool(workers) as pool:
        for done, result in enumerate(pool.imap_unordered(_confirmation_task, range(reps)), 1):
            t, r, g, amp, pair, sha = result
            risks[t], agg[t], amplitude[t], pairing[t], hashes[t] = r, g, amp, pair, sha
            if done % 50 == 0 or done == reps:
                print(json.dumps(dict(stage="confirmation", done=done, total=reps,
                                      max_pairing_difference=float(np.nanmax(pairing)),
                                      minutes=round((time.time() - start) / 60, 2))), flush=True)
    tolerance = float(state["spec"]["pairing_check_atol"])
    if not np.isfinite(risks).all():
        raise FloatingPointError("Nonfinite literature confirmation risk")
    if float(pairing.max()) > tolerance:
        raise AssertionError(f"Recomputed Target only/Pooling differ from the archive by {pairing.max()}")
    DATA.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DATA, risks=risks, methods=np.asarray(METHODS), n0=np.asarray(design["n0_values"]),
                        trajectory=np.arange(reps), aggregation=agg, jm_amplitude=amplitude,
                        pairing_max_abs_difference=pairing, task_data_sha256=np.asarray(hashes))
    print(json.dumps(dict(saved=str(DATA), max_pairing_difference=float(pairing.max()))), flush=True)


def summarize():
    """Means, MCSEs and paired whole-trajectory bootstrap intervals (registered index matrix)."""
    state = _state()
    design = state["config"].design
    schema, cube = load_data(OUTPUTS)
    source = DATA if DATA.exists() else REFERENCE_DATA
    with np.load(source) as data:
        literature = data["risks"]
        preselected = data["aggregation"][:, :, 0, 4]   # Lin-Reimherr candidates kept by the preselection
        if list(data["methods"]) != list(METHODS):
            raise ValueError("Unexpected literature method axis")
        if list(data["n0"]) != design["n0_values"] or list(data["trajectory"]) != list(range(design["confirmation_reps"])):
            raise ValueError("Literature risks must contain the complete registered trajectories and target sizes")
    registered = np.stack([cube[:, :, schema["methods"].index(m)] for m in REGISTERED], axis=-1)
    keys = list(REGISTERED) + list(METHODS)
    combined = np.concatenate([registered, literature], axis=-1)
    reps = combined.shape[0]
    indices = np.random.default_rng(design["bootstrap_seed"]).integers(
        0, reps, size=(design["bootstrap_draws"], reps), dtype=np.int64)
    boot = bootstrap_means(combined, indices)
    summary, paired = [], []
    comparisons = [(left, right) for left in ("projected_1", "unprojected") for right in METHODS]
    # Every registered method against Unprojected, for the figure's ratio panel.
    comparisons += [("unprojected", k) for k in ("pilot", "pooling", "projected_1")]
    differences = np.stack([combined[:, :, keys.index(l)] - combined[:, :, keys.index(r)] for l, r in comparisons], axis=-1)
    difference_boot = bootstrap_means(differences, indices)
    for si, n0 in enumerate(design["n0_values"]):
        for mi, key in enumerate(keys):
            risk = combined[:, si, mi]
            lo, hi = interval(boot[:, si, mi])
            summary.append(dict(n0=n0, method_key=key, method=NAMES[key], mean=float(risk.mean()), mcse=mcse(risk),
                                median=float(np.median(risk)), q90=float(np.quantile(risk, .9)),
                                q99=float(np.quantile(risk, .99)), mean_ci_low=lo, mean_ci_high=hi))
        for ci, (left, right) in enumerate(comparisons):
            li, ri = keys.index(left), keys.index(right)
            lo, hi = interval(boot[:, si, ri] / boot[:, si, li])
            paired.append(dict(n0=n0, method_key=left, comparator_key=right,
                               **paired_statistics(combined[:, si, li], combined[:, si, ri], boot[:, si, li],
                                                   boot[:, si, ri], difference_boot[:, si, ci]),
                               comparator_over_method=float(combined[:, si, ri].mean() / combined[:, si, li].mean()),
                               comparator_over_method_ci_low=lo, comparator_over_method_ci_high=hi))
    RESULTS.mkdir(parents=True, exist_ok=True)
    write_csv(RESULTS / "summary_by_method_n0.csv", summary)
    write_csv(RESULTS / "paired_comparisons.csv", paired)
    save_json(RESULTS / "STATISTICS_METADATA.json", dict(
        methods=keys, labels={k: NAMES[k] for k in keys}, trajectories=reps, seed=design["bootstrap_seed"],
        draws=design["bootstrap_draws"],
        resampling_unit="Complete confirmation trajectory, jointly across sizes and all seven methods "
                        "(the registered index matrix of STATISTICS_METADATA.json)",
        intervals="Pointwise 95% percentile paired trajectory bootstrap",
        percentage_improvement="100 * (1 - mean(left) / mean(right)); positive favours left",
        comparator_over_method="mean(right) / mean(left), with the bootstrap interval of the ratio of means",
        atl_preselection=dict(settings=int(preselected.size), kept_every_candidate=int(np.sum(preselected == 3)),
                              note="Lin-Reimherr star aggregation: settings (trajectory x n0) in which the "
                                   "preselection kept all three candidates"),
        registered_methods="Archived risks of outputs/data/confirmation_risks.npz; not recomputed here"))
    print(json.dumps(dict(summary=str(RESULTS / "summary_by_method_n0.csv"), rows=len(summary),
                          risk_source=str(source))), flush=True)


def main():
    global RESULTS, DATA
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("stage", choices=["development", "select", "confirmation", "summarize", "all"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="New numerical outputs; published reference outputs are protected")
    parser.add_argument("--parameters", type=Path,
                        help="Explicit frozen-parameter CSV; defaults to the published selections")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    for destination in (args.output, args.run_dir):
        resolved = destination.expanduser().resolve()
        for protected in (OUTPUTS, CONFIG, RELEASE_ROOT / "transfer_operator"):
            if resolved == protected or protected in resolved.parents or resolved in protected.parents:
                parser.error("Run/output directory overlaps supplied reference outputs, configuration or code")
    output = args.output.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve()
    RESULTS = output / "results/literature"
    DATA = output / "data/literature_confirmation.npz"
    if args.stage in ("development", "all"):
        development(run_dir, args.workers)
    if args.stage in ("select", "all"):
        select(run_dir)
    if args.stage in ("confirmation", "all"):
        confirmation(args.workers, args.parameters)
    if args.stage in ("summarize", "all"):
        summarize()


if __name__ == "__main__":
    main()

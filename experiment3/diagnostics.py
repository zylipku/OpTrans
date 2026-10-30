#!/usr/bin/env python3
"""Recompute the raw-data diagnostics discussed in Appendix D.4.4.

    python experiment3/diagnostics.py --archive /path/to/data_LPV.zip --kind all

The checks use the frozen representation and validation choices. Outputs are
written to ``<workdir>/diagnostics/<kind>``; no archived results are overwritten.
This runner was added while packaging the final paper. Its outputs are fresh
computations, not the unavailable historical diagnostic records.
"""
from __future__ import annotations

import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

EXPERIMENT_ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = EXPERIMENT_ROOT.parent
for _path in (PACKAGE_ROOT, EXPERIMENT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import numpy as np

import literature
import run
from transfer_operator.literature.aggregation import nested_sets
from transfer_operator.literature.core import LABELS
from transfer_operator.ltv import pipeline
from transfer_operator.ltv.records import dump

SPECTRAL_TARGET = "MS_Harm_h3_N15640_RMS70_P2P350__g1"
SPECTRAL_THRESHOLD = 1e-12


def spectral_diagnostic(scope):
    """Count poorly excited modes using the same eigensolver as the estimators."""
    values, _ = scope.target_spectrum()
    threshold = SPECTRAL_THRESHOLD * float(values[0])
    return dict(parameters=scope.model.p, relative_threshold=SPECTRAL_THRESHOLD,
                absolute_threshold=threshold, below_threshold=int(np.sum(values < threshold)),
                largest_eigenvalue=float(values[0]), smallest_eigenvalue=float(values[-1]))


def lr_grid_diagnostic(scope, grid):
    """Select original and extended LR candidates solely by validation loss."""
    original = grid["lr_source"]
    extended = list(dict.fromkeys(original + [1.0, 100.0]))
    rows = scope.lr_atl(extended, grid["lr_target"])
    selected = literature._select(dict(
        original=[row for row in rows if row["source_smooth"] in original],
        extended=rows,
    ))
    return selected


def hl_zero_penalty(scope, selection, lower_fraction):
    """Check tau=0 for the frozen per-set candidates at the lasso grid boundary.

    Each candidate retains its selected truncation and source set. Only its
    offset penalty changes; validation responses, never test responses, score
    this check. The zero endpoint uses the same stabilization ridge as the
    registered Hu--Lin solver.
    """
    discrepancy = [float(np.sum(np.abs(source["b"] / source["n"] - scope.b0) ** 2))
                   for source in scope.sources]
    sets = nested_sets(discrepancy)
    ranking = [scope.sources[j]["id"] for j in np.argsort(discrepancy, kind="stable")]
    if ranking != json.loads(selection["ranking"]):
        raise AssertionError("Hu--Lin source ranking differs from the frozen selection")
    candidates = json.loads(selection["candidates"])
    if len(candidates) != len(sets):
        raise AssertionError("Hu--Lin frozen candidate count differs from the source bank")
    rows = []
    for index, candidate in enumerate(candidates[1:], 1):
        if candidate["lasso"] != lower_fraction:
            continue
        dimension = int(candidate["truncation"])
        original, zero = scope._hl_paths(sets[index], dimension, [lower_fraction, 0.0])
        original_loss, zero_loss = [scope.loss(scope.model.predict(scope.val_F, z))
                                    for z in (original, zero)]
        rows.append(dict(parameters=scope.model.p, source_set=index,
                         sources=[scope.sources[j]["id"] for j in sets[index]],
                         truncation=dimension, original_lasso=lower_fraction,
                         original_validation_loss=original_loss,
                         zero_validation_loss=zero_loss,
                         validation_loss_ratio=zero_loss / original_loss))
    return rows


def _job(arguments):
    workdir, raw, ti, rotation, kind = arguments
    paths = run._paths_from(workdir, raw)
    design = run.load_design()
    target = design["targets"][ti]
    classes = json.loads((literature.FROZEN / "target_classes.json").read_text())
    hp = next(row["hyperparameters"] for row in classes if row["held_out_target"] == target["id"])
    bank, model, scope = literature.build_scope(paths, design, target, rotation, hp)
    grid = literature.grids()
    folder = paths.workdir / "diagnostics" / kind / pipeline.scope_name(target, rotation)
    folder.mkdir(parents=True, exist_ok=True)
    result = dict(target=target["id"], rotation=rotation, hyperparameters=hp)
    if kind in ("all", "spectrum"):
        result["spectrum"] = spectral_diagnostic(scope)
    if kind in ("all", "hl"):
        selections = run.read_csv(literature.FROZEN / "literature/literature_selection.csv")
        selected = next(row for row in selections if row["target"] == target["id"]
                        and int(row["rotation"]) == rotation and row["method"] == "hl_agg")
        result["hl_zero_lasso"] = hl_zero_penalty(scope, selected, min(grid["hl_lasso"]))
        result["hl_penalized_candidate_count"] = len(scope.sources)
    if kind in ("all", "lr"):
        selected = lr_grid_diagnostic(scope, grid)
        predictions = {name: row["test"] for name, row in selected.items()}
        # Freeze predictions and all selection metadata before reading the test response.
        np.savez_compressed(folder / "predictions_before_test.npz", rows=model.rows, **predictions)
        result["lr_selection"] = {name: {key: value for key, value in row.items() if key != "test"}
                                  for name, row in selected.items()}
        dump(folder / "selection_before_test.json", result)
        result["lr_scores"] = pipeline.evaluate_scope(
            design, bank, target, rotation, model.rows, predictions)
        reference = next(row for row in run.read_csv(literature.FROZEN / "literature/all_test_scores.csv")
                         if row["target"] == target["id"] and int(row["rotation"]) == rotation
                         and row["method"] == LABELS["lr_atl"])
        original = next(row for row in result["lr_scores"] if row["method"] == "original")
        difference = max(abs(original[band] - float(reference[band])) / float(reference[band])
                         for band in literature.SCORES)
        result["lr_original_max_relative_difference"] = difference
        if difference > float(literature.SPEC["pairing_check_rtol"]):
            raise AssertionError(f"Original LR does not reproduce frozen scores: {difference:g}")
    dump(folder / "diagnostics.json", result)
    return result


def summarize(results, kind):
    summary = dict(kind=kind, rotations=len(results),
                   provenance="Recomputed diagnostics; historical diagnostic records were not archived.")
    if kind in ("all", "spectrum"):
        summary["spectra"] = [dict(target=row["target"], rotation=row["rotation"], **row["spectrum"])
                              for row in results]
    if kind in ("all", "lr"):
        scores = [score for row in results for score in row["lr_scores"]]
        means = pipeline.method_table(scores, ("original", "extended"))
        summary["lr_method_means"] = means
        summary["lr_percent_change"] = {band: 100 * (means[1][band] / means[0][band] - 1)
                                        for band in literature.SCORES}
        summary["lr_original_max_relative_difference"] = max(
            row["lr_original_max_relative_difference"] for row in results)
    if kind in ("all", "hl"):
        rows = [entry for result in results for entry in result["hl_zero_lasso"]]
        summary["hl_penalized_candidate_count"] = sum(row["hl_penalized_candidate_count"] for row in results)
        summary["hl_lower_boundary_count"] = len(rows)
        summary["hl_by_class_dimension"] = {
            str(dimension): dict(count=len(ratios), min_ratio=min(ratios), median_ratio=float(np.median(ratios)),
                                 max_ratio=max(ratios))
            for dimension in sorted({row["parameters"] for row in rows})
            for ratios in [[row["validation_loss_ratio"] for row in rows if row["parameters"] == dimension]]
        }
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kind", choices=("all", "spectrum", "lr", "hl"), default="all")
    parser.add_argument("--workdir")
    parser.add_argument("--raw-dir")
    parser.add_argument("--archive")
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    paths = run.Paths(args)
    if not (paths.workdir / "catalog.json").exists():
        run.prepare(paths)
    design = run.load_design()
    scopes = run.scopes(design)
    if args.kind == "spectrum":
        ti = next(i for i, target in enumerate(design["targets"]) if target["id"] == SPECTRAL_TARGET)
        scopes = [(ti, 0)]
    jobs = [(paths.workdir, paths.raw, ti, rotation, args.kind) for ti, rotation in scopes]
    if args.jobs == 1:
        results = [_job(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            results = list(pool.map(_job, jobs))
    output = paths.workdir / "diagnostics" / args.kind / "SUMMARY.json"
    dump(output, summarize(results, args.kind))
    print(output)


if __name__ == "__main__":
    main()

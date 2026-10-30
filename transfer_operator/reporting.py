"""Reusable paired statistics for archived transfer-operator experiments.

This module only reads archived risks: it performs no fitting or selection.
Bootstrap indices resample complete trajectories jointly across every size and
method. Failed/missing fits remain in the risk cube and invalidate the affected
unconditional summary. Iteration zero is the Projected candidate's pilot; it is
not conflated with the independently tuned target-only baseline.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


BASELINES = ("pilot", "pooling", "unprojected")


def methods_for_iterations(iterations):
    """Method-axis order for the common estimator comparison protocol."""
    return BASELINES + ("projection_removed",) + tuple(f"projected_{k}" for k in iterations)


def method_labels(iterations):
    """Stable labels for retained numerical tables."""
    return {"pilot": "Target only", "pooling": "Naive pooling",
            "unprojected": "Unprojected transfer",
            "projection_removed": "Projection removed at Projected parameters",
            **{f"projected_{k}": ("Projected candidate's initial pilot (k=0)"
                                    if k == 0 else f"Projected, k={k}")
               for k in iterations}}




def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), indent=2, sort_keys=True,
                               allow_nan=False) + "\n")


def write_csv(path, rows):
    rows = clean(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True)
                             if isinstance(v, (dict, list)) else v
                             for k, v in row.items()})


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def bootstrap_means(matrix, indices, batch=32):
    matrix = np.asarray(matrix, dtype=float)
    answer = np.empty((len(indices),) + matrix.shape[1:], dtype=float)
    for start in range(0, len(indices), batch):
        answer[start:start + batch] = matrix[indices[start:start + batch]].mean(axis=1)
    return answer


def interval(values):
    values = np.asarray(values, dtype=float)
    if not values.size or not np.isfinite(values).all():
        return np.nan, np.nan
    return tuple(float(x) for x in np.quantile(values, [.025, .975]))


def mcse(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or not np.isfinite(values).all():
        return np.nan
    return 0.0 if np.ptp(values) == 0 else float(np.std(values, ddof=1) / np.sqrt(len(values)))


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if np.isfinite(number) else np.nan


def load_data(root):
    """Return ``(schema, risks[trajectory, sample_size, method])``.

    Iteration zero uses Projected's pilot penalty. It is different from the
    independently tuned ``pilot`` baseline. Missing values must never be dropped.
    """
    root = Path(root)
    metadata = json.loads((root / "data/confirmation_schema.json").read_text())
    with np.load(root / "data/confirmation_risks.npz", allow_pickle=False) as data:
        risks = data["risks"].copy()
        n0 = data["n0"].tolist()
        trajectory = data["trajectory"].tolist()
    methods = metadata["methods"]
    if len(set(methods)) != len(methods) or n0 != metadata["sizes"]:
        raise ValueError("Unexpected stored axis labels")
    if risks.ndim != 3 or risks.shape[2] != len(methods):
        raise ValueError("Method labels do not match the risk axis")
    if trajectory != list(range(metadata["trajectories"])):
        raise ValueError("Missing or reordered confirmation trajectories")
    if list(risks.shape) != metadata["shape"]:
        raise ValueError("Risk cube shape does not match schema")
    if hashlib.sha256(risks.tobytes()).hexdigest() != metadata["risks_array_sha256"]:
        raise ValueError("Risk array content hash differs from released data")
    return metadata, risks


def paired_statistics(left, right, boot_left, boot_right, boot_difference,
                      complete=True):
    """Left minus right and percent mean-risk reduction relative to right."""
    left, right = np.asarray(left), np.asarray(right)
    valid = bool(complete and len(left) and np.isfinite(left).all()
                 and np.isfinite(right).all())
    result = dict(complete=valid, mean_difference=np.nan, mcse_difference=np.nan,
                  difference_ci_low=np.nan, difference_ci_high=np.nan,
                  ratio_of_means=np.nan, percentage_improvement=np.nan,
                  improvement_ci_low=np.nan, improvement_ci_high=np.nan,
                  fraction_better=np.nan, fraction_worse=np.nan,
                  fraction_equal=np.nan, max_absolute_paired_difference=np.nan,
                  observed_constant_difference=False)
    if not valid:
        return result
    difference = left - right
    result.update(mean_difference=float(difference.mean()),
                  mcse_difference=mcse(difference),
                  fraction_better=float(np.mean(difference < 0)),
                  fraction_worse=float(np.mean(difference > 0)),
                  fraction_equal=float(np.mean(difference == 0)),
                  max_absolute_paired_difference=float(np.abs(difference).max()),
                  observed_constant_difference=bool(np.ptp(difference) == 0))
    result["difference_ci_low"], result["difference_ci_high"] = interval(boot_difference)
    if right.mean() > 0 and np.all(boot_right > 0):
        result["ratio_of_means"] = float(left.mean() / right.mean())
        # Using the paired difference avoids cancellation when the risks agree
        # to machine precision, and is algebraically 100*(1-meanL/meanR).
        result["percentage_improvement"] = float(-100 * difference.mean() / right.mean())
        percent_boot = -100 * boot_difference / boot_right
        result["improvement_ci_low"], result["improvement_ci_high"] = interval(percent_boot)
    return result




def summarize_cube(cube, outdir, *, bootstrap_seed, draws, sizes, iterations, settings):
    """Write unconditional means and paired pointwise bootstrap intervals.

    All experiment choices are explicit: ``settings`` supplies the source counts
    and total N for every n0; ``iterations`` determines the method-axis order.
    This comparison protocol requires the initial pilot (0) and first refit (1).
    Use :func:`paired_statistics` or :func:`bootstrap_means` for other contrasts.
    """
    iterations = tuple(int(k) for k in iterations)
    if not {0, 1}.issubset(iterations) or len(set(iterations)) != len(iterations):
        raise ValueError("Iterations must contain distinct values including 0 and 1")
    methods = methods_for_iterations(iterations)
    labels = method_labels(iterations)
    setting_by_size = {int(row["n0"]): row for row in settings}
    if set(setting_by_size) != set(sizes):
        raise ValueError("Settings must supply every requested target size")
    if bootstrap_seed is None:
        raise ValueError("The bootstrap seed must be registered explicitly")
    cube = np.asarray(cube, dtype=float)
    expected_reps = cube.shape[0]
    if cube.shape != (expected_reps, len(sizes), len(methods)):
        raise ValueError("Risk cube shape must be trajectories × sizes × methods")
    if expected_reps < 2 or draws < 2:
        raise ValueError("At least two replications and bootstrap draws are required")
    sizes = tuple(int(n) for n in sizes)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    complete_records = bool(np.isfinite(cube).all())
    seen = set(range(expected_reps))
    indices = np.random.default_rng(int(bootstrap_seed)).integers(
        0, expected_reps, size=(int(draws), expected_reps), dtype=np.int64)
    boot = bootstrap_means(cube, indices)
    iteration_pairs = tuple((f"projected_{k}", "projected_1") for k in iterations if k >= 1)
    comparisons = iteration_pairs + (("projected_1", "unprojected"),
                                     ("projected_1", "projection_removed"))
    differences = np.stack([cube[:, :, methods.index(left)] - cube[:, :, methods.index(right)]
                            for left, right in comparisons], axis=-1)
    difference_boot = bootstrap_means(differences, indices)
    summaries, paired = [], []
    for si, n0 in enumerate(sizes):
        setting = setting_by_size[n0]
        source_keys = sorted((key for key in setting if key.startswith("n")
                              and key[1:].isdigit() and key != "n0"),
                             key=lambda key: int(key[1:]))
        context = dict(n0=n0, **{key: int(setting[key]) for key in source_keys},
                       N=int(setting["N"]), expected_replications=expected_reps,
                       observed_trajectories=len(seen))
        for mi, method in enumerate(methods):
            risk = cube[:, si, mi]
            valid = bool(complete_records and np.isfinite(risk).all())
            lo, hi = interval(boot[:, si, mi]) if valid else (np.nan, np.nan)
            row = dict(**context, method_key=method, method=labels[method],
                       iteration=int(method.split("_")[-1]) if method.startswith("projected_") else None,
                       complete=valid, finite_fits=int(np.isfinite(risk).sum()),
                       missing_or_failed_fits=int((~np.isfinite(risk)).sum()),
                       mean=float(risk.mean()) if valid else np.nan,
                       mcse=mcse(risk) if valid else np.nan,
                       median=float(np.median(risk)) if valid else np.nan,
                       q90=float(np.quantile(risk, .9)) if valid else np.nan,
                       q95=float(np.quantile(risk, .95)) if valid else np.nan,
                       q99=float(np.quantile(risk, .99)) if valid else np.nan,
                       mean_ci_low=lo, mean_ci_high=hi)
            summaries.append(row)
        for ci_, (left, right) in enumerate(comparisons):
            li, ri = methods.index(left), methods.index(right)
            paired.append(dict(**context, method_key=left, comparator_key=right,
                comparison_type=("iteration_vs_first" if ci_ < len(iteration_pairs)
                                 else "baseline" if ci_ == len(iteration_pairs)
                                 else "projection_ablation"),
                iteration=int(left.split("_")[-1]) if left.startswith("projected_") else None,
                **paired_statistics(cube[:, si, li], cube[:, si, ri], boot[:, si, li],
                                    boot[:, si, ri], difference_boot[:, si, ci_], complete_records)))
    tables = {"summary_by_method_n0.csv": summaries,
              "iteration_vs_first.csv": [r for r in paired if r["comparison_type"] == "iteration_vs_first"],
              "projection_ablation.csv": [r for r in paired if r["comparison_type"] == "projection_ablation"],
              "projection_vs_unprojected.csv": [r for r in paired if r["method_key"] == "projected_1"
                                                  and r["comparator_key"] == "unprojected"],
              }
    for name, rows in tables.items():
        write_csv(outdir / name, rows)
    metadata = dict(seed=int(bootstrap_seed), draws=int(draws), trajectories=expected_reps,
        sizes=sizes, iterations=iterations,
        index_matrix_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),
        resampling_unit="Complete independent trajectory jointly across every size, baseline and iteration",
        intervals="Pointwise 95% percentile paired trajectory bootstrap; not simultaneous intervals",
        percentage_improvement="-100 * mean(left-right) / mean(right), algebraically 100*(1-mean(left)/mean(right))",
        difference_sign="Left minus right; negative favors left",
        improvement_sign="Positive favors left",
        zero_variance="An observed constant paired difference has a point interval. No population variance claim.",
        iteration_zero="Pilot at the frozen Projected candidate lambda0; independent Target-only tuning is a separate baseline",
        tuning="Frozen before confirmation. No tuning, iteration selection, or minimum-risk envelope in this module.")
    save_json(outdir / "STATISTICS_METADATA.json", metadata)
    return clean(dict(summary=summaries, comparisons=paired, metadata=metadata))

"""Small YAML experiment inputs expanded into the shared estimator API."""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from .model import BlockModel, build_model, model_checksums


def _positive_integer(value, name):
    if isinstance(value, bool) or int(value) != value or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _frozen_candidates(path, design, registry):
    """Resolve CSV candidate IDs; reject stale duplicate numerical columns."""
    selected = {str(n): {} for n in design['n0_values']}
    with path.open(newline='') as stream:
        for row in csv.DictReader(stream):
            n, method, index = row['n0'], row['method'], int(row['index'])
            if (n not in selected or method not in registry or method in selected[n]
                    or not 0 <= index < len(registry[method])):
                raise ValueError('Frozen selection has an unknown/duplicate identity or candidate ID')
            candidate = registry[method][index]
            for key, expected in candidate.items():
                if key == 'index' or key not in row:
                    continue
                value = row[key]
                if isinstance(expected, bool):
                    if value not in ('True', 'False'):
                        raise ValueError('Invalid frozen selection boolean')
                    value = value == 'True'
                elif expected is None:
                    value = None if value == '' else value
                elif isinstance(expected, float):
                    value = float(value)
                if value != expected:
                    raise ValueError(f'Frozen selection {key} differs from candidate ID {index}')
            selected[n][method] = candidate
    if any(set(methods) != set(registry) for methods in selected.values()):
        raise ValueError('Frozen selections must contain every registered size and method')
    return selected


@dataclass(frozen=True)
class ExperimentConfig:
    """Reusable model and study design, loaded from one experiment config.yaml.

    Experiment2/3 may use their own scalar inputs, grids, seeds, allocations and
    frozen-selection CSV. Programmatic callers can also supply any BlockModel.
    """

    directory: Path
    design: dict
    model: BlockModel
    spec: dict = field(default_factory=dict)

    @classmethod
    def from_directory(cls, directory):
        path = Path(directory).expanduser().resolve()
        path = path / 'config.yaml' if path.is_dir() else path
        spec = yaml.safe_load(path.read_text())
        if not isinstance(spec, dict) or spec.get('version') != 1:
            raise ValueError('Expected version: 1 in config.yaml')
        model = build_model(spec)
        sampling, numerics = spec['sampling'], dict(spec['numerics'])
        sizes = [_positive_integer(n, 'n0') for n in sampling['n0_values']]
        if len(sizes) != len(set(sizes)) or not sizes:
            raise ValueError('n0_values must be nonempty and distinct')
        if len(sampling['source_target_ratios']) != model.m:
            raise ValueError('Source allocation count must match model sources')
        numerics['chunk'] = _positive_integer(numerics['chunk'], 'chunk')
        steps = numerics['iterations']
        if isinstance(steps, bool) or int(steps) != steps or steps < 0:
            raise ValueError('iterations must be a nonnegative integer')
        numerics['rank_rtol'] = float(numerics['rank_rtol'])
        if not 0 < numerics['rank_rtol'] < 1:
            raise ValueError('rank_rtol must be between zero and one')
        if numerics['prefix_rule'] != 'maximum_then_prefix' or numerics['fixed_supports'] is not True:
            raise ValueError('Only maximum_then_prefix with fixed supports is supported')
        checksums = model_checksums(model)
        settings = []
        ratio = float(sampling['total_target_ratio'])
        for n in sizes:
            counts = dict(n0=n, **{f'n{j}': _positive_integer(n * float(r), 'source allocation')
                          for j, r in enumerate(sampling['source_target_ratios'], 1)})
            total = _positive_integer(n * ratio, 'total allocation')
            if total != sum(counts.values()):
                raise ValueError('Total allocation N does not equal the task allocations')
            settings.append(dict(**counts, N=total, ratio=ratio,
                                 id=f'n{n:04d}_r{ratio:g}', K_D=model.meta['K_D'], **checksums))
        # PyYAML 1.1 treats repr floats such as 1e-08 as strings. Converting
        # these exact decimal tokens to float preserves all archived grid bits.
        grid = {key: [float(x) for x in spec['grid'][key]]
                for key in ('lam0_frac', 'lambar_frac', 'lamj_frac')}
        if any(not values or any(not np.isfinite(x) or x <= 0 for x in values)
               for values in grid.values()):
            raise ValueError('Candidate grids must contain finite positive fractions')
        grid['eta_rule'] = list(spec['grid']['eta_rule'])
        if not grid['eta_rule'] or any(rule not in ('eq', 'x4', 'd4') for rule in grid['eta_rule']):
            raise ValueError('Unsupported eta rule')
        development, confirmation, bootstrap = (spec[key] for key in ('development', 'confirmation', 'bootstrap'))
        design = dict(n0_values=sizes, settings=settings, grid=grid,
                      development_reps=_positive_integer(development['repetitions'], 'development_reps'),
                      development_exp_ids={str(n): int(development['first_exp_id']) + i for i, n in enumerate(sizes)},
                      confirmation_reps=_positive_integer(confirmation['repetitions'], 'confirmation_reps'),
                      confirmation_exp_id=int(confirmation['exp_id']),
                      bootstrap_draws=_positive_integer(bootstrap['draws'], 'bootstrap_draws'),
                      bootstrap_seed=int(bootstrap['seed']), iteration_steps=list(range(int(steps) + 1)),
                      confirmation_max_counts=[max(s[f'n{j}'] for s in settings) for j in range(model.m + 1)],
                      seed_rule='SeedSequence([experiment_id, replication, task]).spawn(2)',
                      confirmation_retuning=False, grid_extensions_allowed=False, initial_supports_fixed=True,
                      numerics=numerics, **checksums)
        if 'candidate_registry_sha256' in spec.get('checksums', {}):
            design['candidate_registry_sha256'] = spec['checksums']['candidate_registry_sha256']
        from .estimators import candidate_registry
        registry = candidate_registry(design)
        design['candidate_counts'] = {method: len(rows) for method, rows in registry.items()}
        if spec.get('frozen_selections'):
            frozen_path = (path.parent / spec['frozen_selections']).resolve()
            design['selected_candidates'] = _frozen_candidates(frozen_path, design, registry)
        return cls(path.parent, design, model, spec)

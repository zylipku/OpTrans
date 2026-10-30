"""Validated scalar designs for the paper's two advection-diffusion studies."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .pde_models import AdvectionDiffusionModel
from .pde_pipeline import frozen_tuning


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


def _number(value, name, minimum=0., strict=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    if value < minimum or (strict and value == minimum):
        raise ValueError(f'{name} is outside its supported range')
    return value


def _multipliers(values):
    result = {}
    for method in ('Projected', 'Unprojected'):
        result[method] = {key: _number(values[method][key], f'{method}.{key}', strict=key == 'cbar')
                          for key in ('c0', 'cbar', 'ctheta')}
    for method in ('Target_only', 'Pooling'):
        result[method] = _number(values[method], method)
    return result


@dataclass(frozen=True)
class PDECell:
    group: str
    archive_cell: int
    model_family: str
    setting: dict
    seed: int
    bootstrap_seed: int
    multipliers: dict

    def make_model(self):
        return AdvectionDiffusionModel(**{key: self.setting[key] for key in ('R', 's', 'tau', 'sigma')})

    def tuning(self, method='Projected'):
        """Evaluate frozen scaling; source supports are still estimated from data."""
        if method not in ('Projected', 'Unprojected'):
            raise ValueError('method must be Projected or Unprojected')
        return frozen_tuning(self.make_model(), self.setting['n0'], self.setting['nS'], self.multipliers[method])


@dataclass(frozen=True)
class PDEStudy:
    path: Path
    cells: tuple[PDECell, ...]
    specification: dict

    @classmethod
    def from_yaml(cls, path):
        path = Path(path)
        spec = yaml.safe_load(path.read_text())
        if not isinstance(spec, dict) or spec.get('model') != 'advection_diffusion':
            raise ValueError('Expected the paper advection_diffusion model')
        group = spec.get('group')
        if group not in ('advection_n0', 'advection_sigma'):
            raise ValueError('Expected advection_n0 or advection_sigma study')
        if spec.get('source_allocation') != 'equal_halves':
            raise ValueError('The paper uses equal_halves allocation')
        common = dict(R=_integer(spec['radius'], 'radius', 1),
                      s=_number(spec['smoothness'], 'smoothness', 1., strict=True),
                      tau=_number(spec['tau'], 'tau'),
                      nS=_integer(spec['source_total'], 'source_total', 2))
        if common['nS'] % 2:
            raise ValueError('equal_halves requires an even source_total')
        seed = _integer(spec['seed'], 'seed')
        bootstrap_seed = _integer(spec['bootstrap']['seed'], 'bootstrap.seed')
        multipliers = _multipliers(spec['frozen_multipliers'])
        cells = []
        for position, n0 in enumerate(spec['n0_grid']):
            _integer(n0, 'n0', 1)
            if group == 'advection_sigma':
                sigmas, indices = spec['sigma_grid'], spec['archive_cells'][n0]
                if len(sigmas) != len(indices):
                    raise ValueError('Every sigma needs its original archive cell')
            else:
                sigmas, indices = [spec['sigma']], [position]
            for sigma, index in zip(sigmas, indices):
                setting = dict(common, n0=n0, sigma=_number(sigma, 'sigma'))
                cells.append(PDECell(group, _integer(index, 'archive_cell'), spec['model'], setting,
                                     seed, bootstrap_seed, multipliers))
        identities = [(cell.group, cell.archive_cell) for cell in cells]
        if not cells or len(set(identities)) != len(identities):
            raise ValueError('Study cells must have unique group/archive identities')
        return cls(path, tuple(cells), spec)

    def cell(self, group, archive_cell):
        for cell in self.cells:
            if (cell.group, cell.archive_cell) == (group, archive_cell):
                return cell
        raise KeyError((group, archive_cell))

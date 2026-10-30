"""Exercise reuse with different source counts, dimensions, sizes and steps."""
from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import yaml
from _paths import RELEASE_ROOT
from transfer_operator import (ExperimentConfig, BlockModel, candidate_registry,
                               run_development, run_trajectory, select_candidates)
from transfer_operator.runner import clean
from transfer_operator.diagnostics import compact_diagnostics, schema
from transfer_operator.reporting import methods_for_iterations, summarize_cube, read_csv


def small_config(sources=3, steps=2):
    model = BlockModel('alternate', nu=np.array([1., .5, .25, .125]),
        mu=np.array([.5, .25, .125, .0625]), a0=np.array([.2, -.1, .1, .05]),
        h=np.eye(4)[:3], thetas=np.arange(sources*3).reshape(sources, 3)*.04,
        sigma=.2, beta=1.)
    sizes = [7, 11]
    settings = []
    for n in sizes:
        counts = {f'n{j}': n*(j+1) for j in range(sources+1)}
        settings.append(dict(**counts, N=sum(counts.values())))
    design = dict(n0_values=sizes, settings=settings, iteration_steps=list(range(steps+1)),
        development_reps=2, confirmation_reps=2, confirmation_exp_id=902,
        development_exp_ids={str(n):900+n for n in sizes},
        grid=dict(lam0_frac=[.1,.5], lambar_frac=[.05], eta_rule=['eq'], lamj_frac=[.1]))
    registry = candidate_registry(design)
    design['selected_candidates'] = {str(n): {m:rows[0] for m,rows in registry.items()} for n in sizes}
    return ExperimentConfig(RELEASE_ROOT, design, model)


class SharedAPITest(unittest.TestCase):
    def test_one_and_three_sources_and_short_iteration_paths(self):
        for sources, steps in ((1,0), (3,2)):
            with self.subTest(sources=sources,steps=steps):
                config = small_config(sources, steps)
                records = [run_trajectory(i,config) for i in range(2)]
                arrays = compact_diagnostics(records, config)
                self.assertEqual(arrays['iteration_theta'].shape,(2,2,steps+1,sources,3))
                self.assertEqual(arrays['fit_penalties'].shape,(2,2,5,4+sources))
                self.assertEqual(schema(arrays)['axes']['n0'],[7,11])
                self.assertEqual(schema(arrays)['axes']['source'],list(range(1,sources+1)))
                self.assertEqual(records[0]['task_max_counts'],[11*(j+1) for j in range(sources+1)])
                subset = run_trajectory(0,config,sizes=[7])
                self.assertEqual(clean(subset['settings'][0]),clean(records[0]['settings'][0]))

    def test_alternate_development_selection_is_complete(self):
        config = small_config()
        registry = candidate_registry(config.design)
        banks = {}
        for n in config.design['n0_values']:
            worlds = [run_development(n,i,config) for i in range(2)]
            self.assertTrue(all(record['status']=='ok' for _,record in worlds))
            banks[n] = {method:np.stack([values[method] for values,_ in worlds],axis=1) for method in registry}
        selected = select_candidates(banks, registry, expected_reps=2)
        self.assertEqual(set(selected),{'7','11'})
        invalid = copy.deepcopy(banks)
        invalid[7]['projected'][0,0] = np.nan
        with self.assertRaisesRegex(ValueError,'every finite candidate'):
            select_candidates(invalid,registry,expected_reps=2)
        with self.assertRaises(ValueError):
            run_trajectory(2,config)
        with self.assertRaises(ValueError):
            run_trajectory(0,config,sizes=[7,7])

    def test_load_and_report_alternate_config(self):
        with tempfile.TemporaryDirectory(prefix='.test-shared-',dir=RELEASE_ROOT) as temporary:
            root = Path(temporary)
            spec = yaml.safe_load((RELEASE_ROOT/'experiment1/configs/config.yaml').read_text())
            spec['model'].update(K=8, sigma=.2, R=.5, target_exponent=1.)
            spec['dictionary'].update(cutoff=4, exponent=.7)
            spec['sampling'] = dict(n0_values=[7,11],total_target_ratio=6,source_target_ratios=[2,3])
            spec['development'] = dict(repetitions=2,first_exp_id=900)
            spec['confirmation'] = dict(repetitions=2,exp_id=902)
            spec['grid'] = dict(lam0_frac=[.1,.5],lambar_frac=[.05],eta_rule=['eq'],lamj_frac=[.1])
            spec['numerics'].update(iterations=2,chunk=4,rank_rtol=1e-8)
            spec['frozen_selections'] = 'selections.csv'
            spec.pop('checksums')
            (root/'selections.csv').write_text('n0,method,index\n'+''.join(
                f'{n},{method},0\n' for n in (7,11) for method in ('pilot','pooling','projected','unprojected')))
            (root/'config.yaml').write_text(yaml.safe_dump(spec))
            config = ExperimentConfig.from_directory(root)
            self.assertEqual((config.model.K,config.model.m,config.model.p),(8,2,2))
            records = [run_trajectory(i,config) for i in range(2)]
            methods = methods_for_iterations(config.design['iteration_steps'])
            cube = np.asarray([[[setting['iterations'][int(method.split('_')[1])]['risk']
                                if method.startswith('projected_') else setting['fits'][method]['risk']
                                for method in methods] for setting in r['settings']] for r in records])
            summarize_cube(cube,root/'tables',bootstrap_seed=12,draws=25,
                sizes=config.design['n0_values'],iterations=config.design['iteration_steps'],settings=config.design['settings'])
            rows = read_csv(root/'tables/summary_by_method_n0.csv')
            self.assertEqual(len(rows),14)
            self.assertEqual({int(r['n0']) for r in rows},{7,11})
            self.assertEqual({int(r['n2']) for r in rows},{21,33})
            spec['sampling']['total_target_ratio'] += 1
            (root/'config.yaml').write_text(yaml.safe_dump(spec))
            with self.assertRaisesRegex(ValueError,'Total allocation'):
                ExperimentConfig.from_directory(root)


if __name__ == '__main__':
    unittest.main()

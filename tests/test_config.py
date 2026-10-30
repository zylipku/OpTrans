"""Closed-form reconstruction and immutable candidate identities from compact YAML."""
import csv
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import pytest
import yaml

from _paths import DEFAULT_CONFIG, RELEASE_ROOT
from transfer_operator import ExperimentConfig, candidate_registry
from transfer_operator.estimators import BlockView, projection_basis, restricted_recalibration
from transfer_operator.model import ARRAY_NAMES, model_checksums
from transfer_operator.runner import input_fingerprint

# Computed from the original dgp.json before removing the expanded arrays.
ORIGINAL_ARRAY_HASHES = {
    'nu': '1b4d9ac3b8e4394c3f942d09efb4ef55c202bc1389e99627b58418acd76ce367',
    'mu': '0c266681f5dbbea1517fd552112407f11db996e58affaf96b417f08cfd418172',
    'a0': 'c538336d772f9b9ece74a605742b4e080a8c6d21a75a868a9ce2fe858361024e',
    'h': '79af19c9f64c15856e6078c9f702742757c6300a01e3f8a2b9b51ff50b25c98a',
    'thetas': '9a9a33da2df85aaf399fd55b603e05b605f3faf48178f55304a1cb7e466635b6',
}
ORIGINAL_MODEL_SHA256 = '2274eab5d606d75adc41765d4a7efd904a3d33061608c366cc47e22557e3380d'
ORIGINAL_REGISTRY_SHA256 = '7452e40570b84c639419b031b616fb2f74cfddf7e3920ba11c3b830dd29c6901'


def test_rebuilt_arrays_match_each_original_dgp_array_bitwise():
    config = ExperimentConfig.from_directory(DEFAULT_CONFIG)
    for key in ARRAY_NAMES:
        actual = np.ascontiguousarray(getattr(config.model, key), dtype='<f8')
        assert hashlib.sha256(actual.tobytes()).hexdigest() == ORIGINAL_ARRAY_HASHES[key], key
    assert model_checksums(config.model)['model_arrays_sha256'] == ORIGINAL_MODEL_SHA256
    assert config.model.thetas[1, 1] == 10.580160221155685
    print('5 rebuilt arrays / 2,564 float64 values match original DGP SHA-256 exactly')


def test_exact_grids_and_csv_candidate_ids():
    config = ExperimentConfig.from_directory(DEFAULT_CONFIG/'config.yaml')
    registry = candidate_registry(config.design)
    digest = hashlib.sha256(json.dumps(registry, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    assert digest == ORIGINAL_REGISTRY_SHA256
    assert {method: len(rows) for method, rows in registry.items()} == dict(
        pilot=19, pooling=19, unprojected=5073, projected=5073)
    with (DEFAULT_CONFIG/config.spec['frozen_selections']).open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 52
    for row in rows:
        assert config.design['selected_candidates'][row['n0']][row['method']] == registry[row['method']][int(row['index'])]
    assert config.design['development_exp_ids'] == {
        str(n): 2026101410+i for i,n in enumerate(config.design['n0_values'])}
    assert config.design['confirmation_max_counts'] == [1000,9500,9500]
    print('Exact candidate grid hash and 52 frozen CSV candidate IDs verified')


def test_checksums_and_frozen_csv_reject_changed_inputs():
    spec = yaml.safe_load((DEFAULT_CONFIG/'config.yaml').read_text())
    with tempfile.TemporaryDirectory(prefix='.test-config-', dir=RELEASE_ROOT) as temporary:
        root = Path(temporary)
        spec['frozen_selections'] = 'selections.csv'
        original_csv = (DEFAULT_CONFIG/'../outputs/results/frozen_parameters.csv').read_text()
        (root/'selections.csv').write_text(original_csv)
        config_path = root/'config.yaml'
        config_path.write_text(yaml.safe_dump(spec))
        config = ExperimentConfig.from_directory(root)
        old_fingerprint = input_fingerprint(config)
        spec['model']['target_exponent'] = .76
        config_path.write_text(yaml.safe_dump(spec))
        with pytest.raises(ValueError, match='checksum mismatch'):
            ExperimentConfig.from_directory(root)
        spec['model']['target_exponent'] = .75
        spec['grid']['lam0_frac'][0] = .001
        config_path.write_text(yaml.safe_dump(spec))
        with pytest.raises(ValueError, match='grid/ID checksum mismatch'):
            ExperimentConfig.from_directory(root)
        spec['grid']['lam0_frac'][0] = 3.16227766016838e-09
        config_path.write_text(yaml.safe_dump(spec))
        (root/'selections.csv').write_text(original_csv.replace('10,pilot,eq,15,', '10,pilot,eq,14,', 1))
        with pytest.raises(ValueError, match='differs from candidate ID'):
            ExperimentConfig.from_directory(root)
        assert input_fingerprint(config) != old_fingerprint


def test_rank_tolerance_controls_projection_and_recalibration():
    # A nearly collinear two-column generator distinguishes configured rank
    # thresholds in both the projection and the rectangular least-squares solve.
    config = ExperimentConfig.from_directory(DEFAULT_CONFIG)
    view = BlockView.__new__(BlockView)
    view.model = config.model
    view.WSLh = np.zeros((config.model.K, 2))
    view.WSLh[0, 0], view.WSLh[1, 1] = 1., 1e-6
    view.WS = np.zeros((config.model.K, config.model.K))
    view.Wyf = np.ones(config.model.K)
    view.blockF2 = 0.
    for tolerance, rank in ((1e-10, 2), (1e-4, 1)):
        view.rank_rtol = tolerance
        assert view.basis(1., [0,1]).shape[1] == rank
        assert projection_basis(view, 1., [0,1])[0].shape[1] == rank
        _, diagnostic = restricted_recalibration(view, 1., np.zeros(config.model.K), [0,1])
        assert diagnostic['numerical_rank'] == rank

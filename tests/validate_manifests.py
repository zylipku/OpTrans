"""Validate one experiment tree and its explicitly registered shared dependencies.

Paths stay relative to the package root. The manifest's ``release`` identifies the only
directory scanned for coverage; shared dependencies are individual ``files``
entries. Adding another experiment or an unrelated shared file needs no reseal.
"""
from pathlib import Path
import hashlib
import json
import re

from _paths import RELEASE_ROOT


SHARED_ROOT_FILES = {'.gitattributes', '.gitignore', 'README.md', 'LITERATURE_BASELINES.md', 'pyproject.toml'}


def validate_manifest(manifest_path, release_root=RELEASE_ROOT):
    """Check complete experiment coverage and every registered dependency hash."""
    release_root = Path(release_root).resolve()
    manifest_path = Path(manifest_path).resolve()
    assert manifest_path.is_file(), 'Release manifest missing'
    manifest = json.loads(manifest_path.read_text())
    assert manifest['path_root'] == '.', 'Unsupported manifest root'
    experiment = manifest['release']
    assert re.fullmatch(r'experiment[1-9][0-9]*', experiment), 'Invalid manifest owner'
    experiment_root = release_root / experiment
    assert manifest_path == experiment_root / 'outputs/RELEASE_MANIFEST.json', (
        'Manifest location does not match its owner')
    excluded = set(manifest['excluded_directories'])

    def transient(relative):
        return (relative.suffix in {'.pyc', '.pyo'} or any(
            part in excluded or part.endswith('.egg-info') for part in relative.parts))

    registered = set(manifest['files'])
    shared = set()
    for relative in registered:
        path = Path(relative)
        assert (not path.is_absolute() and '..' not in path.parts and
                path.as_posix() == relative and not transient(path)), (
            f'Invalid manifest path: {relative}')
        target = release_root / path
        assert target.resolve() == target, f'Invalid manifest path: {relative}'
        is_shared = (
            len(path.parts) > 1 and path.parts[0] in {'transfer_operator', 'tests'}
        ) or (len(path.parts) == 1 and (
            relative in SHARED_ROOT_FILES or relative.startswith('requirements')))
        assert path.parts[0] == experiment or is_shared, (
            f'File outside experiment/shared scope: {relative}')
        assert target != manifest_path, 'Manifest cannot hash itself'
        if is_shared:
            shared.add(relative)

    actual = {
        path.relative_to(release_root).as_posix()
        for path in experiment_root.rglob('*')
        if path.is_file() and path != manifest_path
        and not transient(path.relative_to(release_root))
    }
    expected = registered - shared
    assert actual == expected, (
        f'Manifest coverage differs; unlisted={sorted(actual - expected)}, '
        f'missing={sorted(expected - actual)}')
    for relative, expected_hash in manifest['files'].items():
        path = release_root / relative
        assert path.is_file(), f'Missing release file: {relative}'
        raw = path.read_bytes()
        assert len(raw) == expected_hash['bytes'], f'Changed size: {relative}'
        assert hashlib.sha256(raw).hexdigest() == expected_hash['sha256'], (
            f'Changed file: {relative}')
    return len(registered)

#!/usr/bin/env python3
"""Rewrite every experimentN/outputs/RELEASE_MANIFEST.json after an intended change.

Each manifest lists every file under its own experiment folder (except itself)
plus the shared files it depends on (transfer_operator/, tests/, top-level
packaging), with byte size and SHA-256; transient caches are excluded.  Where
the experiment's PROVENANCE.json has a record for a file, its provenance
fields are kept in the manifest entry, and its size and hash must still match.
Run from anywhere; validate afterwards with ``python -m pytest tests``.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from _paths import RELEASE_ROOT

SHARED_ROOT_FILES = {'.gitattributes', '.gitignore', 'README.md', 'LITERATURE_BASELINES.md', 'pyproject.toml'}


def _entry(path):
    raw = path.read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def seal(release_root=RELEASE_ROOT):
    release_root = Path(release_root).resolve()
    counts = {}
    for experiment in sorted(p.name for p in release_root.glob('experiment[0-9]*') if p.is_dir()):
        manifest_path = release_root / experiment / 'outputs/RELEASE_MANIFEST.json'
        old = json.loads(manifest_path.read_text())
        excluded = set(old['excluded_directories'])
        provenance_path = release_root / experiment / 'outputs/PROVENANCE.json'
        provenance = json.loads(provenance_path.read_text()) if provenance_path.is_file() else {}
        records = provenance.get('files') if isinstance(provenance.get('files'), dict) else {}

        def transient(relative):
            return relative.suffix in {'.pyc', '.pyo'} or any(
                part in excluded or part.endswith('.egg-info') for part in relative.parts)

        files = {}
        for path in sorted(release_root.rglob('*')):
            if not path.is_file() or path == manifest_path:
                continue
            relative = path.relative_to(release_root)
            if transient(relative):
                continue
            top = relative.parts[0]
            shared = (len(relative.parts) > 1 and top in {'transfer_operator', 'tests'}) or (
                len(relative.parts) == 1 and (relative.name in SHARED_ROOT_FILES
                                              or relative.name.startswith('requirements')))
            if top != experiment and not shared:
                continue
            entry = _entry(path)
            record = records.get(relative.as_posix())
            if record is not None:
                assert (record['bytes'], record['sha256']) == (entry['bytes'], entry['sha256']), relative
                entry = {**record, **entry}
            files[relative.as_posix()] = entry
        manifest = {key: value for key, value in old.items() if key not in {'files', 'release'}}
        manifest.update(release=experiment, files=files)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
        counts[experiment] = len(files)
    return counts


if __name__ == '__main__':
    print(json.dumps(seal(), indent=2))

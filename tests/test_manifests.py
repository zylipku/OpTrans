"""Experiment manifests enforce their own contents and explicit shared dependencies."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile

import pytest

from _paths import RELEASE_ROOT
from validate_manifests import validate_manifest


def write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def entry(path):
    raw = path.read_bytes()
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def update_manifest(path, update):
    manifest = json.loads(path.read_text())
    update(manifest)
    path.write_text(json.dumps(manifest))


@pytest.fixture
def manifest_tree():
    runs = RELEASE_ROOT / ".runs"
    runs.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="manifest-tests-", dir=runs) as temporary:
        root = Path(temporary) / "release"
        shared = ("transfer_operator/core.py", "tests/check_common.py", "requirements.txt")
        for relative in shared:
            write(root / relative, b"shared dependency\n")
        manifests = {}
        for owner in ("experiment1", "experiment2"):
            own = (f"{owner}/configs/config.yaml", f"{owner}/outputs/data.bin")
            for relative in own:
                write(root / relative, b"owned data\n")
            manifest_path = root / owner / "outputs/RELEASE_MANIFEST.json"
            manifest = {
                "path_root": ".",
                "release": owner,
                "excluded_directories": [".runs", "__pycache__", ".pytest_cache"],
                "files": {relative: entry(root / relative) for relative in (*own, *shared)},
            }
            manifest_path.write_text(json.dumps(manifest))
            manifests[owner] = manifest_path
        yield root, manifests


@pytest.mark.parametrize("owner", ["experiment1", "experiment2"])
def test_each_experiment_validates_its_own_manifest(manifest_tree, owner):
    root, manifests = manifest_tree
    assert validate_manifest(manifests[owner], release_root=root) == 5


@pytest.mark.parametrize("owner", ["experiment1", "experiment2"])
@pytest.mark.parametrize("operation", ["add", "change", "delete"])
def test_other_experiments_and_unlisted_shared_files_are_independent(manifest_tree, owner, operation):
    root, manifests = manifest_tree
    other = "experiment2" if owner == "experiment1" else "experiment1"
    unrelated = [root / other / "outputs/data.bin", root / "tests/check_unrelated.py",
                 root / "transfer_operator/unrelated.py", root / "README.md"]
    for path in unrelated:
        write(path, b"unrelated\n")
    if operation == "add":
        for path in unrelated:
            write(path.with_name("new_" + path.name), b"new unrelated file\n")
    elif operation == "change":
        for path in unrelated:
            path.write_bytes(b"changed unrelated file\n")
    else:
        for path in unrelated:
            path.unlink()
    # Even an invalid or missing sibling manifest is outside this owner's scope.
    if operation == "delete":
        manifests[other].unlink()
    else:
        manifests[other].write_text("not JSON")
    write(root / "experiment3/outputs/new.bin", b"future experiment\n")
    assert validate_manifest(manifests[owner], release_root=root) == 5


@pytest.mark.parametrize("operation", ["add", "change", "delete"])
def test_own_file_changes_fail(manifest_tree, operation):
    root, manifests = manifest_tree
    path = root / "experiment1/outputs/data.bin"
    if operation == "add":
        write(path.with_name("unregistered.bin"), b"extra data\n")
    elif operation == "change":
        path.write_bytes(b"X" + path.read_bytes()[1:])
    else:
        path.unlink()
    with pytest.raises(AssertionError):
        validate_manifest(manifests["experiment1"], release_root=root)


@pytest.mark.parametrize("relative", ["transfer_operator/core.py", "tests/check_common.py", "requirements.txt"])
@pytest.mark.parametrize("operation", ["change", "delete"])
def test_registered_shared_dependencies_are_checked(manifest_tree, relative, operation):
    root, manifests = manifest_tree
    path = root / relative
    if operation == "change":
        path.write_bytes(b"X" + path.read_bytes()[1:])
    else:
        path.unlink()
    for manifest in manifests.values():
        with pytest.raises(AssertionError):
            validate_manifest(manifest, release_root=root)


def test_own_transient_files_are_ignored(manifest_tree):
    root, manifests = manifest_tree
    for relative in ("__pycache__/cached.pyc", ".runs/temporary.bin", ".pytest_cache/state",
                     "package.egg-info/metadata", "outputs/temporary.pyo"):
        write(root / "experiment1" / relative, b"transient\n")
    assert validate_manifest(manifests["experiment1"], release_root=root) == 5


@pytest.mark.parametrize("owner", ["experiment2", "experiment99", "../experiment1", "unknown"])
def test_owner_must_match_manifest_location(manifest_tree, owner):
    root, manifests = manifest_tree
    path = manifests["experiment1"]
    update_manifest(path, lambda manifest: manifest.update(release=owner))
    with pytest.raises(AssertionError):
        validate_manifest(path, release_root=root)


def test_cross_experiment_entry_is_rejected(manifest_tree):
    root, manifests = manifest_tree
    relative = "experiment2/outputs/data.bin"
    path = manifests["experiment1"]
    update_manifest(path, lambda manifest: manifest["files"].update({relative: entry(root / relative)}))
    with pytest.raises(AssertionError):
        validate_manifest(path, release_root=root)


@pytest.mark.parametrize("kind", ["parent", "absolute", "symlink"])
def test_paths_cannot_escape_release(manifest_tree, kind):
    root, manifests = manifest_tree
    outside = root.parent / "outside.bin"
    write(outside, b"outside release\n")
    if kind == "parent":
        relative = "../outside.bin"
    elif kind == "absolute":
        relative = str(outside)
    else:
        relative = "experiment1/outputs/escaped.bin"
        (root / relative).symlink_to(outside)
    path = manifests["experiment1"]
    update_manifest(path, lambda manifest: manifest["files"].update({relative: entry(outside)}))
    with pytest.raises(AssertionError):
        validate_manifest(path, release_root=root)

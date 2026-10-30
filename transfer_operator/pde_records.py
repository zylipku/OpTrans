"""Lossless, object-free storage of per-replication PDE records.

Original fields keep their names, with replication as the first axis. Only
fields with varying shape or string width need extra metadata. Loading a
record restores the original shape and dtype, including Unicode widths.
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np


_META = "__packed_records__"
_PREFIX = "__packed_"
_FORMAT = "pde-records-v1"


class PackedRecords:
    """Load one packed cell once and expose its original replication records."""

    def __init__(self, path):
        self.path = Path(path)
        with np.load(self.path, allow_pickle=False) as archive:
            self._arrays = {key: archive[key] for key in archive.files}
        self._meta = json.loads(self._arrays.pop(_META).tobytes().decode("utf-8"))
        if self._meta["format"] != _FORMAT:
            raise ValueError(f"Unsupported packed record format in {self.path}")
        self.fields = tuple(self._meta["fields"])

    def __len__(self):
        return self._meta["count"]

    def record(self, index):
        """Return a new dictionary of arrays with original dtypes and shapes."""
        if not isinstance(index, (int, np.integer)) or not 0 <= index < len(self):
            raise IndexError(f"Replication index {index} outside [0, {len(self)})")
        result = {}
        for field in self.fields:
            spec = self._meta["special"].get(field, {})
            if spec.get("per_record"):
                result[field] = self._arrays[f"{_PREFIX}rep_{field}_{index}"].copy()
                continue
            stacked = self._arrays[field]
            value = stacked[index:index + 1].reshape(stacked.shape[1:])
            if "shape_key" in spec:
                rank = int(self._arrays[spec["rank_key"]][index])
                shape = tuple(self._arrays[spec["shape_key"]][index, :rank])
                length = int(self._arrays[spec["length_key"]][index])
                value = value[:length].reshape(shape)
            if "dtype_key" in spec:
                dtype = spec["dtypes"][int(self._arrays[spec["dtype_key"]][index])]
                value = np.asarray(value).astype(np.dtype(dtype), copy=True)
            else:
                # np.asarray keeps zero-dimensional originals as arrays.
                value = np.asarray(value).copy()
            result[field] = value
        return result

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def load_pde_record(path, replicate=None):
    """Read an ordinary NPZ, or one replication from a packed cell NPZ."""
    if replicate is not None:
        return PackedRecords(path).record(replicate)
    with np.load(path, allow_pickle=False) as archive:
        if _META in archive.files:
            raise ValueError("A packed cell requires an explicit replication index")
        return {key: archive[key] for key in archive.files}


def _verify(original, restored, *, path, field):
    if original.dtype != restored.dtype or original.shape != restored.shape:
        raise ValueError(f"Packing changed dtype or shape: {path}:{field}")
    equal_nan = original.dtype.kind in "fc"
    if not np.array_equal(original, restored, equal_nan=equal_nan):
        raise ValueError(f"Packing changed values: {path}:{field}")
    if original.tobytes(order="C") != restored.tobytes(order="C"):
        raise ValueError(f"Packing changed array bits: {path}:{field}")


def pack_records(paths, output_path):
    """Compress records in the supplied order and verify every original array.

    No inputs are removed. Uniform fields are stacked; ragged fields are padded
    and carry lengths and shapes. Varying string widths use the widest dtype
    plus per-record dtype codes. Other mixed dtypes use separate per-record
    arrays to avoid lossy numeric casts. All arrays are object-free.
    """
    paths = [Path(path) for path in paths]
    if not paths:
        raise ValueError("Cannot pack an empty record group")
    records = [load_pde_record(path) for path in paths]
    fields = tuple(records[0])
    if any(key.startswith(_PREFIX) for key in fields):
        raise ValueError(f"Original fields must not use reserved prefix {_PREFIX}")
    if any(set(record) != set(fields) for record in records):
        raise ValueError("All packed records must contain the same fields")
    arrays, special = {}, {}
    ragged_fields, varying_dtype_fields, fallback_fields = [], [], []
    for field in fields:
        values = [record[field] for record in records]
        if any(value.dtype.hasobject for value in values):
            raise ValueError(f"Object arrays cannot be packed safely: {field}")
        if any(value.dtype.fields or value.dtype.subdtype for value in values):
            raise ValueError(f"Structured/subarray dtypes are not supported: {field}")
        dtypes = list(dict.fromkeys(value.dtype.str for value in values))
        spec = {}
        if len(dtypes) > 1:
            varying_dtype_fields.append(field)
            if len({value.dtype.kind for value in values}) != 1 or values[0].dtype.kind not in "SU":
                # Distinct numeric dtypes can change NaN payloads or precision
                # when cast; keep these arrays as-is instead.
                for index, value in enumerate(values):
                    arrays[f"{_PREFIX}rep_{field}_{index}"] = value
                special[field] = {"per_record": True}
                fallback_fields.append(field)
                continue
            spec["dtypes"] = dtypes
            spec["dtype_key"] = f"{_PREFIX}dtype_{field}"
            code_dtype = np.min_scalar_type(len(dtypes) - 1)
            arrays[spec["dtype_key"]] = np.asarray(
                [dtypes.index(value.dtype.str) for value in values], dtype=code_dtype)
            dtype = np.result_type(*(value.dtype for value in values))
        else:
            dtype = values[0].dtype
        if len({value.shape for value in values}) == 1:
            arrays[field] = np.stack(values, axis=0, dtype=dtype)
        else:
            ragged_fields.append(field)
            length_key = f"{_PREFIX}length_{field}"
            rank_key = f"{_PREFIX}rank_{field}"
            shape_key = f"{_PREFIX}shape_{field}"
            spec.update(length_key=length_key, rank_key=rank_key, shape_key=shape_key)
            arrays[length_key] = np.asarray([value.size for value in values], dtype=np.int64)
            arrays[rank_key] = np.asarray([value.ndim for value in values], dtype=np.int64)
            shapes = np.zeros((len(values), max(value.ndim for value in values)), dtype=np.int64)
            padded = np.zeros((len(values), max(value.size for value in values)), dtype=dtype)
            for index, value in enumerate(values):
                shapes[index, :value.ndim] = value.shape
                padded[index, :value.size] = value.reshape(-1)
            arrays[shape_key], arrays[field] = shapes, padded
        if spec:
            special[field] = spec
    meta = dict(format=_FORMAT, count=len(paths), fields=fields, special=special)
    arrays[_META] = np.frombuffer(json.dumps(meta, separators=(",", ":")).encode("utf-8"), dtype=np.uint8)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output_path.parent, suffix=".npz", delete=False) as handle:
        temporary_path = Path(handle.name)
    try:
        np.savez_compressed(temporary_path, **arrays)
        packed = PackedRecords(temporary_path)
        checked = 0
        for index, (path, original) in enumerate(zip(paths, records)):
            restored = packed.record(index)
            if set(restored) != set(original):
                raise ValueError(f"Packing changed field names: {path}")
            for field, value in original.items():
                _verify(value, restored[field], path=path, field=field)
                checked += 1
        temporary_path.chmod(0o644)
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return {
        "records_verified": len(paths), "fields_per_record": len(fields),
        "arrays_verified": checked, "ragged_fields": ragged_fields,
        "varying_dtype_fields": varying_dtype_fields, "per_record_fallback_fields": fallback_fields,
        "checks": ["all_original_fields", "dtype", "shape", "numpy_array_equal_equal_nan",
                   "exact_array_bytes_including_nan_payloads_and_signed_zero"],
    }

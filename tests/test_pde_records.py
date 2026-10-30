"""Check lossless archive decoding beyond the regular production record shapes."""
from pathlib import Path
import tempfile
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pytest

from _paths import RELEASE_ROOT
from transfer_operator.pde_records import PackedRecords, load_pde_record, pack_records


@pytest.fixture
def record_directory():
    cache = RELEASE_ROOT / '.cache'
    cache.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='packed-records-', dir=cache) as directory:
        yield Path(directory)


def assert_record_bits(actual, expected):
    assert set(actual) == set(expected)
    for field, original in expected.items():
        restored = actual[field]
        assert restored.dtype == original.dtype, field
        assert restored.shape == original.shape, field
        assert np.array_equal(restored, original, equal_nan=original.dtype.kind in 'fc'), field
        assert restored.tobytes() == original.tobytes(), field


def test_round_trip_all_fields_ragged_shapes_string_widths_and_special_float_bits(record_directory):
    records, paths = [], []
    for rep, width in enumerate((1, 7, 3)):
        # Distinct NaN payloads and signed zero must survive, not just compare equal.
        floats = np.array([0, 1 << 63, 0x7FF8000000000001 + rep, 0xFFF0000000000000],
                          dtype=np.uint64).view(np.float64)
        complex_values = np.array([0x7FF8000000000001 + rep, 1 << 63,
                                   0x3FF0000000000000, 0x7FF8000000000042],
                                  dtype=np.uint64).view(np.complex128)
        records.append({
            'scalar': np.array(rep, dtype='>i8'),
            'matrix': np.arange(6, dtype=np.int16).reshape(2, 3),
            'float_bits': floats,
            'complex_bits': complex_values,
            'names': np.array(['π', 'x'], dtype=f'U{width}'),
            'bytes': np.array([b'x', b''], dtype=f'S{width}'),
            'ragged': (np.empty((0, 2), dtype=np.float32),
                       np.array([1, -0., 3], dtype=np.float32),
                       np.array(2.5, dtype=np.float32))[rep],
            'empty': np.empty(((0,), (1, 0), (0, 3))[rep], dtype=np.uint8),
            'mixed_numeric': (np.array([1], dtype=np.float32),
                              np.array([1 + 2**-40], dtype=np.float64),
                              np.array([2**60 + 1], dtype=np.uint64))[rep],
        })
        path = record_directory / f'rep{rep:04d}.npz'
        np.savez(path, **records[-1])
        paths.append(path)
    output = record_directory / 'cell.npz'
    audit = pack_records(paths, output)
    packed = PackedRecords(output)
    assert len(packed) == 3
    assert set(packed.fields) == set(records[0])
    assert audit['records_verified'] == 3
    assert audit['arrays_verified'] == 3 * len(records[0])
    assert set(audit['ragged_fields']) == {'ragged', 'empty'}
    assert set(audit['varying_dtype_fields']) == {'names', 'bytes', 'mixed_numeric'}
    assert audit['per_record_fallback_fields'] == ['mixed_numeric']
    for rep, original in enumerate(records):
        assert_record_bits(load_pde_record(paths[rep]), original)
        assert_record_bits(packed.record(rep), original)
        assert_record_bits(load_pde_record(output, replicate=rep), original)
    with np.load(output, allow_pickle=False) as archive:
        assert all(not archive[field].dtype.hasobject for field in archive.files)
        assert archive['matrix'].shape == (3, 2, 3)
        assert archive['scalar'].shape == (3,)
        assert archive['names'].shape == (3, 2)
        assert 'ragged' in archive.files and 'empty' in archive.files
    with ZipFile(output) as archive:
        assert all(entry.compress_type == ZIP_DEFLATED for entry in archive.infolist())
    packed.record(0)['matrix'][0, 0] = 123
    assert_record_bits(packed.record(0), records[0])
    with pytest.raises(ValueError, match='replication index'):
        load_pde_record(output)
    for index in (-1, 3, 0.5, '0'):
        with pytest.raises(IndexError):
            packed.record(index)


def test_invalid_sources_are_rejected_without_overwriting_destination(record_directory):
    output = record_directory / 'cell.npz'
    output.write_bytes(b'existing destination')
    with pytest.raises(ValueError, match='empty'):
        pack_records([], output)
    first, second = (record_directory / f'rep{i}.npz' for i in range(2))
    np.savez(first, value=np.array([1.]))
    np.savez(second, other=np.array([1.]))
    with pytest.raises(ValueError, match='same fields'):
        pack_records([first, second], output)
    np.savez(first, __packed_collision=np.array([1]))
    with pytest.raises(ValueError, match='reserved prefix'):
        pack_records([first], output)
    np.savez(first, value=np.array([{'not': 'safe'}], dtype=object))
    with pytest.raises(ValueError, match='Object arrays'):
        pack_records([first], output)
    assert output.read_bytes() == b'existing destination'

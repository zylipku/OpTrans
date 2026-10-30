"""Column-selective MATLAB v5 access and Fourier-domain records.

The benchmark archive stores every record of a file as one column of the
matrices ``u_ref``, ``u_m``, ``p_ref``, ``p_m`` and ``y_m``. The design
channels (inputs and scheduling signals) are read with SciPy; the measured
output ``y_m`` is read one column at a time by streaming the container, so a
run only materialises the responses it is entitled to (train, validation, the
source bank, and the held-out test response only in the evaluation phase).
"""
from __future__ import annotations

import hashlib
import json
import struct
import zlib
from pathlib import Path

import numpy as np
import scipy.io as sio

DESIGN_KEYS = ['fs', 'p_ref', 'p_m', 'u_ref', 'u_m']


def sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def sha_array(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def read_design_channels(path):
    """fs and the input/scheduling matrices of one file (no measured output)."""
    return sio.loadmat(path, variable_names=DESIGN_KEYS)


class InflateReader:
    """Bounded-memory sequential decompressor; skip never constructs arrays."""
    def __init__(self, raw, size):
        self.raw, self.left = raw, size
        self.decoder = zlib.decompressobj()
        self.buf = bytearray()

    def read(self, n):
        while len(self.buf) < n and self.left:
            chunk = self.raw.read(min(65536, self.left))
            if not chunk:
                raise EOFError('Truncated MAT compression stream')
            self.left -= len(chunk)
            self.buf.extend(self.decoder.decompress(chunk))
        out = bytes(self.buf[:n])
        del self.buf[:n]
        return out


def exact(f, n):
    a = f.read(n)
    if len(a) != n:
        raise EOFError(f'Expected {n} bytes, got {len(a)}')
    return a


def skip(f, n):
    while n:
        m = min(n, 65536)
        exact(f, m)
        n -= m


def element(f):
    tag = exact(f, 8)
    a, b = struct.unpack('<II', tag)
    if a >> 16:
        return a & 65535, (a >> 16), tag[4:4+(a >> 16)]
    return a, b, None


def payload(f):
    dtype, size, small = element(f)
    if small is not None:
        return dtype, small
    b = exact(f, size)
    skip(f, (-size) % 8)
    return dtype, b


def read_mat_column(path, variable, column):
    """Read one real double column without loading any other column."""
    with open(path, 'rb') as raw:
        header = exact(raw, 128)
        if header[126:128] != b'IM':
            raise ValueError('Only little-endian MATLAB v5 files are supported')
        while tag := raw.read(8):
            dtype, size = struct.unpack('<II', tag)
            end = raw.tell() + size
            stream = InflateReader(raw, size) if dtype == 15 else raw
            if dtype == 15:
                matrix_type, _, small = element(stream)
                if matrix_type != 14 or small is not None:
                    raise ValueError('Expected miMATRIX inside compressed block')
            elif dtype != 14:
                raw.seek(end + (-size) % 8)
                continue
            _, flags = payload(stream)
            _, dims = payload(stream)
            _, name = payload(stream)
            if name.decode() != variable:
                # Compressed elements are not padded in scipy/MATLAB v5 files.
                raw.seek(end if dtype == 15 else end + (-size) % 8)
                continue
            shape = struct.unpack('<' + 'i'*(len(dims)//4), dims)
            if len(shape) != 2 or not 0 <= column < shape[1]:
                raise ValueError(f'Invalid column {column} for {shape}')
            if struct.unpack('<I', flags[:4])[0] & 0x800:
                raise ValueError('Complex response data unsupported')
            real_type, real_size, small = element(stream)
            if real_type != 9 or small is not None or real_size != 8*np.prod(shape):
                raise ValueError('Expected dense miDOUBLE response')
            skip(stream, column * shape[0] * 8)
            return np.frombuffer(exact(stream, shape[0]*8), dtype='<f8').copy()
    raise KeyError(variable)


def period_spectrum(signal, n):
    """FFT/N of the average over the complete measured periods of a signal."""
    signal = np.asarray(signal)
    if len(signal) % n:
        raise ValueError('Noninteger number of measured periods')
    return np.fft.rfft(signal.reshape(-1, n).mean(0)) / n


class RecordBank:
    """Fourier-domain records of the catalogued inputs and the measured outputs.

    ``arrays`` holds the input spectra F of every record (``<group>__F<col>``)
    and the mean scheduling signal of every group; measured outputs are read on
    demand and cached per (file, column).
    """
    def __init__(self, raw_dir, catalog, arrays, frequency):
        self.raw_dir = Path(raw_dir)
        self.record_group = {(r['file'], r['column']): g['id']
                             for g in catalog['groups'] for r in g['records']}
        self.groups = {g['id']: g for g in catalog['groups']}
        self.arrays = arrays
        self.n, self.fs, self.wc = frequency['N'], frequency['fs'], frequency['wc']
        self._cache = {}

    def input_spectrum(self, ref):
        return self.arrays[self.record_group[(ref['file'], ref['column'])] + f'__F{ref["column"]}']

    def record(self, ref):
        """dict(F, Y, V, ref): input spectrum, output spectrum, state spectrum."""
        key = (ref['file'], ref['column'])
        if key not in self._cache:
            y = read_mat_column(self.raw_dir / ref['file'], 'y_m', ref['column'])
            Y = period_spectrum(y, self.n)
            omega = 2*np.pi*np.fft.rfftfreq(self.n, 1/self.fs)
            V = np.zeros_like(Y)
            V[1:] = self.wc*Y[1:]/(1j*omega[1:])
            self._cache[key] = dict(F=self.input_spectrum(ref), Y=Y, V=V, ref=ref)
        return self._cache[key]

"""Inventory the benchmark archive from its five design channels.

Every column of a file is a record; columns sharing the same reference
scheduling signal ``p_ref`` form a *group* (one operating regime, up to three
input realisations). The catalogue stores, per record, the SHA-256 of its
reference input, the forced lines (bins whose reference-input amplitude exceeds
1e-3 of the maximum) and the input RMS; per group, the measured scheduling
signal's descriptor (DC and first three harmonics). ``design_arrays`` holds the
input spectrum F of every record and the mean scheduling signal of every group.
No measured output is read here.
"""
from __future__ import annotations

import itertools
import re
import zipfile
from pathlib import Path

import numpy as np

from .records import dump, read_design_channels, sha_array, sha_file

FILE_PATTERN = r'(MS|PN)_(Const|Harm_h\d+|Ramp)_N(\d+)_RMS(\d+)_P2P(\d+)\.mat'


def describe(p):
    c = np.fft.rfft(p)/len(p)
    # Euclidean distance is the RMS distance of the DC+h1..h3 reconstruction.
    return np.r_[c[0].real, np.sqrt(2)*c[1:4].real,
                 -np.sqrt(2)*c[1:4].imag].tolist()


def extract_archive(archive, raw_dir):
    """Extract every .mat container of the archive into raw_dir (if missing)."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    names = []
    with zipfile.ZipFile(archive) as z:
        for info in sorted(z.infolist(), key=lambda i: i.filename):
            if not info.filename.endswith('.mat'):
                continue
            name = Path(info.filename).name
            path = raw_dir/name
            if not path.exists():
                with z.open(info) as src, path.open('wb') as dst:
                    while chunk := src.read(1 << 20):
                        dst.write(chunk)
            names.append(name)
    return names


def build_catalog(raw_dir, out_dir):
    """Write catalog.json and design_arrays.npz for every .mat file in raw_dir."""
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    groups, files, design_arrays = [], [], {}
    for path in sorted(raw_dir.glob('*.mat')):
        name = path.name
        match = re.fullmatch(FILE_PATTERN, name)
        if not match:
            continue
        d = read_design_channels(path)
        excitation, kind, n, rms, p2p = match.groups()
        n, rms, p2p = int(n), int(rms), int(p2p)
        fs = float(d['fs'].squeeze())
        pr, pm, ur, um = (np.asarray(d[k]) for k in ('p_ref', 'p_m', 'u_ref', 'u_m'))
        assert pr.shape == ur.shape and pr.shape[0] == n
        assert pm.shape == um.shape and len(pm) % n == 0
        file_hash = sha_file(path)
        files.append(dict(file=name, sha256=file_hash, N=n, RMS=rms,
                          P2P=p2p, excitation=excitation, kind=kind,
                          fs=fs, measured_periods=len(pm)//n, columns=pr.shape[1]))
        by_hash = {}
        for c in range(pr.shape[1]):
            by_hash.setdefault(sha_array(pr[:, c]), []).append(c)
        for gi, (ph, cols) in enumerate(by_hash.items()):
            key = f'{path.stem}__g{gi}'
            p = np.stack([pm[:, c].reshape(-1, n).mean(0) for c in cols]).mean(0)
            uhash = [sha_array(ur[:, c]) for c in cols]
            pair_corr = []
            for a, b in itertools.combinations(cols, 2):
                ua, ub = ur[:, a]-ur[:, a].mean(), ur[:, b]-ur[:, b].mean()
                cc = np.fft.ifft(np.fft.fft(ua)*np.conj(np.fft.fft(ub))).real
                pair_corr.append(float(np.max(np.abs(cc))/(np.linalg.norm(ua)*np.linalg.norm(ub))))
            different = len(cols) == 3 and len(set(uhash)) == 3 and max(pair_corr, default=1) < 0.999
            records = []
            for c, h in zip(cols, uhash):
                amp = np.abs(np.fft.rfft(ur[:, c]))
                forced = np.flatnonzero(amp > 1e-3*amp.max()).tolist()
                records.append(dict(file=name, column=c, u_ref_sha256=h,
                                    u_m_sha256=sha_array(um[:, c]), forced_lines=forced,
                                    input_rms=float(np.sqrt(np.mean(ur[:, c]**2)))))
                design_arrays[f'{key}__F{c}'] = np.fft.rfft(um[:, c].reshape(-1, n).mean(0))/n
            reasons = []
            if excitation != 'MS':
                reasons.append('not_MS')
            if n != 15640:
                reasons.append('not_slow_N15640')
            if rms != 70:
                reasons.append('not_RMS70')
            if kind not in ('Harm_h1', 'Harm_h3'):
                reasons.append('not_h1_h3_Harm')
            if not different:
                reasons.append('not_three_distinct_nonshift_input_realizations')
            co = np.fft.rfft(p)/n
            groups.append(dict(id=key, file=name, file_sha256=file_hash, columns=cols,
                               excitation=excitation, kind=kind, N=n, RMS=rms, P2P=p2p, fs=fs,
                               trajectory_hash=ph, measured_trajectory_hash=sha_array(p),
                               descriptor=describe(p), mean_p=float(p.mean()), ptp_p=float(np.ptp(p)),
                               harmonic_amplitudes=(2*np.abs(co[1:21])).tolist(),
                               records=records, n_unique_inputs=len(set(uhash)),
                               input_max_circular_correlations=pair_corr, three_input_realizations=different,
                               primary_target=not reasons, target_exclusion_reasons=reasons))
            design_arrays[key+'__p'] = p
    catalog = dict(files=files, groups=groups,
                   independence_note='Distinct input hashes and no signed circular-shift duplicates establish distinct realizations; statistical independence relies on acquisition design, not provable from waveforms. Repeated periods are not independent records.',
                   channels_read=['fs', 'p_ref', 'p_m', 'u_ref', 'u_m'])
    dump(out_dir/'catalog.json', catalog)
    np.savez_compressed(out_dir/'design_arrays.npz', **design_arrays)
    return catalog

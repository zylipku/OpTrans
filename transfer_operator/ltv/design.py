"""Derive the frozen experimental design from the catalogue (no response data).

Targets: every group of slowly varying (N = 15640) RMS-70 multisine records with
a harmonic scheduling signal (h1 or h3) and three distinct input realisations.
Sources: every system of the same N/RMS with a Const, h1 or h3 scheduling
signal that differs from the target's; PN and MS records of one identical
scheduling trajectory form one system. Each target receives a shortlist of 12
candidate source tasks (six nearest Const, four nearest LTV, two
direction-diverse LTV) measured by the descriptor distance of the measured
scheduling signals. Rotations cycle the three target records through
train/validation/test.
"""
from __future__ import annotations

import itertools

import numpy as np

TARGET_RECORD_KEYS = ('file', 'column', 'u_ref_sha256', 'forced_lines')
SOURCE_RECORD_KEYS = ('file', 'column', 'u_ref_sha256')


def _record(record, keys):
    return {key: record[key] for key in keys}


def derive_design(catalog, arrays, rules):
    """Targets, sources and candidate shortlists; ``rules`` are the frozen constants."""
    groups = catalog['groups']
    eligible = [g for g in groups if g['N'] == 15640 and g['RMS'] == 70
                and g['kind'] in ('Const', 'Harm_h1', 'Harm_h3')]
    systems = []
    for h in sorted({g['trajectory_hash'] for g in eligible}):
        gs = sorted([g for g in eligible if g['trajectory_hash'] == h],
                    key=lambda g: (g['excitation'] != 'PN', g['id']))
        primary = gs[0]
        systems.append(dict(id=primary['id'], trajectory_hash=h, kind=primary['kind'],
                            groups=[g['id'] for g in gs], primary_group=primary['id'],
                            descriptor=primary['descriptor'],
                            records=[_record(r, SOURCE_RECORD_KEYS) for g in gs for r in g['records']]))
    shortlist = rules['shortlist']
    activation = rules['dictionary']['harmonic_activation']
    targets = []
    for t in sorted([g for g in groups if g['primary_target']], key=lambda g: g['trajectory_hash']):
        desc = np.array(t['descriptor'])
        candidates = [s for s in systems if s['trajectory_hash'] != t['trajectory_hash']]

        def distance(s):
            return float(np.linalg.norm(np.array(s['descriptor'])-desc))
        constants = sorted([s for s in candidates if s['kind'] == 'Const'],
                           key=lambda s: (distance(s), s['id']))[:shortlist['nearest_const']]
        ltvs = sorted([s for s in candidates if s['kind'] != 'Const'], key=lambda s: (distance(s), s['id']))
        nearest = shortlist['nearest_ltv']
        near = ltvs[:nearest+2*shortlist['direction_diverse_ltv']]
        selected = ltvs[:nearest]

        def direction(s):
            d = np.array(s['descriptor'])-desc
            return d/max(np.linalg.norm(d), 1e-30)
        for _ in range(shortlist['direction_diverse_ltv']):
            left = [s for s in near if s not in selected]
            if not left:
                break
            s = min(left, key=lambda s: (max(float(direction(s)@direction(a)) for a in selected),
                                         distance(s), s['id']))
            selected.append(s)
        cs = []
        for s in constants+selected:
            q = arrays[s['primary_group']+'__p']-arrays[t['id']+'__p']
            coeff = np.fft.rfft(q)/len(q)
            amp = 2*np.abs(coeff[1:4])
            active = [h for h in (1, 2, 3)
                      if amp[h-1] >= max(activation['absolute'], activation['relative']*float(amp.max()))]
            atom_names = (['dc'] if abs(coeff[0].real) >= activation['absolute'] else [])
            atom_names += [f'{mode}{h}' for h in active for mode in ('cos', 'sin')]
            role = ('nearest_const' if s in constants
                    else 'nearest_ltv' if s in selected[:nearest] else 'direction_diverse_ltv')
            cs.append(dict(source_id=s['id'], distance=distance(s), shortlist_role=role,
                           active_harmonics=active, base_atoms=atom_names))
        recs = sorted(t['records'], key=lambda r: (r['u_ref_sha256'], r['column']))
        recs = [_record(r, TARGET_RECORD_KEYS) for r in recs]
        rotations = [dict(rotation=r, train=recs[r], validation=recs[(r+1) % 3], test=recs[(r+2) % 3])
                     for r in range(3)]
        targets.append(dict(id=t['id'], trajectory_hash=t['trajectory_hash'], kind=t['kind'], P2P=t['P2P'],
                            descriptor=t['descriptor'], rotations=rotations, candidates=cs))
    if len(targets) != 8:
        raise RuntimeError('The benchmark archive must yield exactly eight target regimes')
    return dict(rules, targets=targets, sources=systems)


def class_grid(rules):
    """The 36 target-class configurations (R shifts, B splines, smoothing)."""
    grid = rules['target_class']
    return [dict(R=r, B=b, smooth=l) for r, b, l in itertools.product(grid['R'], grid['B'], grid['smooth'])]

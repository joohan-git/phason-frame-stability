"""atlas-floor check. new1/new2 (radius-1/2 patches absent from the QC atlas) on the approximant
SEQUENCES down to |delta| ~ 1.5e-4, with two atlas sizes (R = 75 and R = 110) to expose a finite-atlas floor.
Compares new2/|delta| with the c0 deficit/|delta| along each line (B1 line small?).  -> patch_census_sequences.json (P)"""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames', 'phason_paths', 'penrose', 'symmetry')]  # shared modules
import os
os.environ.update(OMP_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='-1')
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).parent; ROOT = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / 'phason_paths')); sys.path.insert(0, str(ROOT / 'penrose'))
import ab_frame as AB  # noqa
import penrose_frame as PN   # noqa
from patch_census_cells import qc_atlas, patches_periodic  # noqa
sys.path.insert(0, str(ROOT / 'phason_paths'))
from ab_families import family_lifts  # noqa
LAM = {'ab': 0.79984, 'pen': 0.8106}

ab_rows = {(r['family'], r['p'], r['q']): r for r in json.loads((ROOT / 'phason_paths' / 'ab_families.json').read_text())['rows']}
pen = json.loads((ROOT / 'penrose' / 'penrose_results.json').read_text())
PCELLS = pen['cells']
A_LINE = [([6, 5, 2, -2, -5], [0, 2, 3, 3, 2]), ([10, 8, 3, -3, -8], [0, 3, 5, 5, 3]), ([16, 13, 5, -5, -13], [0, 5, 8, 8, 5]),
          ([26, 21, 8, -8, -21], [0, 8, 13, 13, 8]), ([42, 34, 13, -13, -34], [0, 13, 21, 21, 13])]


def pen_c0(name):
    return PCELLS[name]['c0_mean']


def main():
    out = {}
    for cls in ('ab', 'pen'):
        M = AB if cls == 'ab' else PN; nd = 4 if cls == 'ab' else 5
        atl = {R: qc_atlas(cls, R) for R in (75.0, 110.0)}
        jobs = []
        if cls == 'ab':
            for fam in ('B1', 'B2', 'E2a', 'E2diag'):
                for p, q in ((2, 3), (5, 7), (12, 17)) + (((29, 41),) if fam in ('B1', 'E2a') else ()):
                    L = family_lifts(p, q, fam); r = ab_rows[(fam, p, q)]
                    jobs.append((f'{fam}_{p}/{q}', L[0], L[1], r['amp'], r['harm']['c0']))
        else:
            for i, (L1, L2) in enumerate(A_LINE):
                jobs.append((f'a_{i + 2}', L1, L2, None, pen_c0(f'a_{i + 2}')))
            for n in range(2, 7):
                L1, L2 = PN.fib_approx(n); jobs.append((f'fib{n}', L1, L2, None, pen_c0(f'b_fib{n}')))
        rows = []
        for name, L1, L2, d, c0 in jobs:
            tl = M.canonical(L1, L2)
            Mt = M.phason_matrix(tl['L1'], tl['L2'])
            if cls == 'ab':
                a, b = M.harmonic_parts(Mt)
            else:
                a, b, v = M.phason_parts(Mt)
            d = float(np.hypot(abs(a), abs(b)))
            st, p2 = patches_periodic(tl['lifts'], tl['L1'], tl['L2'], M, nd)
            row = dict(name=name, V=len(st), delta=d, c0=c0, deficit=(LAM[cls] - c0) if c0 is not None else None)
            for R, (a1, a2) in atl.items():
                row[f'new1_R{int(R)}'] = sum(s not in a1 for s in st) / len(st)
                row[f'new2_R{int(R)}'] = sum(p not in a2 for p in p2) / len(p2)
            row['new2_over_delta'] = row['new2_R110'] / d
            row['deficit_over_delta'] = row['deficit'] / d if row['deficit'] is not None else None
            rows.append(row)
            print(cls, json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
        out[cls] = dict(atlas_sizes={str(int(R)): [len(a1), len(a2)] for R, (a1, a2) in atl.items()}, rows=rows)
    (HERE / 'patch_census_sequences.json').write_text(json.dumps(dict(status='P', lambda_QC=LAM, **out), indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()

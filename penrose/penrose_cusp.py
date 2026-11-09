"""Penrose cusp sweep (stress protocol, 2 el./edge, nphi = 8 long-wave directions, Richardson
DQ = 0.002/0.001 -> 0).  python penrose_cusp.py LINE   (LINE = aline | bline | vline | controls)
-> penrose_cusp_LINE.json (P).  Phason parts (a: E4, b: E2, v: E4 diagonal tilt) from penrose_frame."""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames',)]  # shared modules
import os
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='-1')
import json, math, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import penrose_frame as C

A = [([2, 2, 1, -1, -2], [0, 1, 1, 1, 1]), ([4, 3, 1, -1, -3], [0, 1, 2, 2, 1]), ([6, 5, 2, -2, -5], [0, 2, 3, 3, 2]),
     ([10, 8, 3, -3, -8], [0, 3, 5, 5, 3]), ([16, 13, 5, -5, -13], [0, 5, 8, 8, 5]), ([26, 21, 8, -8, -21], [0, 8, 13, 13, 8]),
     ([42, 34, 13, -13, -34], [0, 13, 21, 21, 13])]
B = [C.fib_approx(n) for n in range(1, 7)]
Vl = [([5, 3, 2, -2, -3], [0, 2, 3, 3, 2]), ([3, 2, 1, -1, -2], [0, 3, 5, 5, 3]), ([7, 6, 2, -2, -6], [0, 3, 5, 5, 3]),
      ([12, 9, 4, -4, -9], [0, 3, 5, 5, 3]), ([19, 15, 6, -6, -15], [0, 5, 8, 8, 5]), ([23, 19, 7, -7, -19], [0, 8, 13, 13, 8]),
      ([45, 36, 14, -14, -36], [0, 8, 13, 13, 8])]


def evaluate(name, L1, L2, offset=None, dqs=(0.002, 0.001), kfull=None):
    t0 = time.time()
    tl = C.canonical(L1, L2, offset=offset)
    V = len(tl['lifts'])
    assert C.check_tiling(tl['lifts'], tl['L1'], tl['L2']) == 0
    f = C.Frame(tl['lifts'], tl['L1'], tl['L2'], sub=2); f.set_load('stress')
    phis, ext, tab = f.longwave(nphi=8, dqs=dqs)
    a, b, v = C.phason_parts(C.phason_matrix(tl['L1'], tl['L2']))
    h = C.harmonics(phis, ext)
    row = dict(name=name, L1=tl['L1'].tolist(), L2=tl['L2'].tolist(), V=V, offset=tl['offset'].tolist(), dqs=list(dqs),
               a=[a.real, a.imag], b=[b.real, b.imag], v=[v.real, v.imag],
               delta=float(math.sqrt(abs(a) ** 2 + abs(b) ** 2 + abs(v) ** 2)),
               thin_thick=C.tile_counts(tl['lifts'], tl['L1'], tl['L2']),
               pc_phi=ext.tolist(), pc_phi_dq_small=tab[dqs[1]].tolist(), phis_deg=np.degrees(phis).tolist(),
               pc_lw_min_sampled=float(ext.min()), harm=h, abs_h4=float(math.hypot(*h['h4'])), abs_h2=float(math.hypot(*h['h2'])),
               prestress_resid=f.resid)
    ks = [(0, 0), (0.5, 0.5)] + ([(0.5, 0), (0, 0.5)] if (kfull if kfull is not None else V < 2000) else [])
    row['pc_k'] = {str(k): float(f.lam(k)) for k in ks}
    row['pc_min_all'] = float(min(row['pc_lw_min_sampled'], h['min_model'], *row['pc_k'].values()))
    row['seconds'] = time.time() - t0
    print(json.dumps({k: row[k] for k in ['name', 'V', 'delta', 'a', 'b', 'v', 'abs_h4', 'abs_h2', 'pc_lw_min_sampled', 'pc_k', 'seconds']} | {'c0': h['c0']}), flush=True)
    return row


def main(line):
    out = Path(__file__).with_name(f'penrose_cusp_{line}.json'); rows = []
    if line == 'aline':
        jobs = [(f'a_{i}', *L) for i, L in enumerate(A)]
    elif line == 'bline':
        jobs = [(f'b_fib{i+1}', *L) for i, L in enumerate(B)]
    elif line == 'vline':
        jobs = [(f'v_{i}', *L) for i, L in enumerate(Vl)]
    else:  # offset and DQ controls
        rng = np.random.default_rng(5)
        jobs = []
        for nm, L in (('b_fib5', B[4]), ('a_4', A[4])):
            for s in range(3):
                o = rng.uniform(-.5, .5, 5); o += (0.5 - np.array([1, -1, 1, -1, 1]) @ o) / 5 * np.array([1, -1, 1, -1, 1])
                jobs.append((f'{nm}_offset{s}', *L, o))
        jobs.append(('b_fib6_dqhalf', *B[5], None, (0.001, 0.0005), False))
    for j in jobs:
        rows.append(evaluate(*j))
        out.write_text(json.dumps(dict(status='P', units='p_c/(EA lambda_E), stress protocol, sub=2', line=line, rows=rows), indent=1), encoding='utf-8')


if __name__ == '__main__':
    main(sys.argv[1])

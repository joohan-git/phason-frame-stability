"""Isotropic part c0 vs |delta| on the in-plane lines (offset means from penrose_results.json), fit
c0 = lam_QC - c |delta|^beta with lam_QC free; plus a structural count: fraction of vertex stars of
the canonical cut that are not among the 8 Penrose (de Bruijn) vertex types, vs |delta|.
-> penrose_isotropic.json (P; the vertex-type count is exact for the default offset)"""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames',)]  # shared modules
import json, math, sys
from pathlib import Path
import numpy as np
from scipy.optimize import curve_fit
sys.path.insert(0, str(Path(__file__).parent))
import penrose_frame as C
from penrose_cusp import A, B
H = Path(__file__).parent
cells = json.loads((H / 'penrose_results.json').read_text(encoding='utf-8'))['cells']
pts = [(n, math.hypot(t['a'], t['b']), t['c0_mean'], t['V']) for n, t in cells.items()
       if (n.startswith('a_') or n.startswith('b_fib')) and t['V'] >= 94]
d = np.array([p[1] for p in pts]); c0 = np.array([p[2] for p in pts])
f = lambda x, l, c, b: l - c * x ** b
popt, pcov = curve_fit(f, d, c0, p0=(0.811, 0.2, 1.0))
lin = curve_fit(lambda x, l, c: l - c * x, d, c0, p0=(0.811, 0.2))[0]
quad = curve_fit(lambda x, l, c: l - c * x ** 2, d, c0, p0=(0.811, 10))[0]
res = dict(status='P', fit_free=dict(lam_QC=popt[0], c=popt[1], beta=popt[2], se=np.sqrt(np.diag(pcov)).tolist()),
           fit_linear=dict(lam_QC=lin[0], c=lin[1], rms=float(np.sqrt(np.mean((c0 - (lin[0] - lin[1] * d)) ** 2)))),
           fit_quadratic=dict(lam_QC=quad[0], c=quad[1], rms=float(np.sqrt(np.mean((c0 - (quad[0] - quad[1] * d ** 2)) ** 2)))),
           points=[dict(cell=p[0], delta=p[1], c0=p[2], V=p[3], deficit_over_delta=(popt[0] - p[2]) / p[1]) for p in pts])


def star_sig(lifts, L1, L2):
    """Vertex star as the sorted tuple of occupied physical directions (10 possible, multiples of 36 deg),
    up to rotation and reflection."""
    red = C.Reducer(L1, L2); look = {tuple(r) for r in lifts}; E = np.eye(5, dtype=np.int64)
    sigs = []
    for r in lifts:
        occ = np.zeros(10, int)
        for k in range(5):
            if tuple(red(r + E[k])[0]) in look: occ[k] = 1
            if tuple(red(r - E[k])[0]) in look: occ[k + 5] = 1
        forms = []
        for s in range(10):
            o = np.roll(occ, s); forms += [tuple(o), tuple(o[::-1])]
        sigs.append(min(forms))
    return sigs


# reference: the 8 Penrose vertex stars, from the largest canonical cell (Fibonacci n = 6)
ref_tl = C.canonical(*C.fib_approx(6)); ref = set(star_sig(ref_tl['lifts'], ref_tl['L1'], ref_tl['L2']))
res['n_star_types_fib6'] = len(ref)
rows = []
for nm, L in [(f'a_{i}', A[i]) for i in range(2, 7)] + [(f'b_fib{i+1}', B[i]) for i in range(1, 6)]:
    tl = C.canonical(*L); s = star_sig(tl['lifts'], tl['L1'], tl['L2'])
    a, b, v = C.phason_parts(C.phason_matrix(tl['L1'], tl['L2']))
    from collections import Counter
    cnt = Counter(s)
    rows.append(dict(cell=nm, V=len(s), delta=math.hypot(abs(a), abs(b)), n_types=len(cnt),
                     frac_nonref=sum(v_ for k, v_ in cnt.items() if k not in ref) / len(s)))
    print(rows[-1], flush=True)
res['star_types'] = rows
(H / 'penrose_isotropic.json').write_text(json.dumps(res, indent=1, default=float), encoding='utf-8')
print(json.dumps({k: res[k] for k in ['fit_free', 'fit_linear', 'fit_quadratic', 'n_star_types_fib6']}, default=float, indent=1))
for p in res['points']: print(p)

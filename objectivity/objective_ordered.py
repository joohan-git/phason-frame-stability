import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames', 'phason_paths', 'penrose', 'disorder', 'slope_survey')]  # shared modules
import sys, json, time
from pathlib import Path
import numpy as np
from objective_frame import *
src = ''
ns = {'np': np}; ns.update(vars(__import__('ab_families'))); fl = ns['family_lifts']
out = HERE / 'objective_ordered.jsonl'
PLAN = [(f, 5, 7, s) for f in ('B1','B2','E2a','E2diag') for s in (7, 11, 23)] + [(f, 12, 17, s) for f in ('B1','B2','E2a','E2diag') for s in (7, 11)] + [(f, 2, 3, 7) for f in ('B1','B2','E2a','E2diag')]
for fam, p, q, seed in PLAN:
    L = fl(p, q, fam); tl = C.canonical(L[0], L[1], seed=seed)
    a, b = C.harmonic_parts(C.phason_matrix(tl['L1'], tl['L2']))
    rec = dict(fam=fam, p=p, q=q, seed=seed, V=len(tl['lifts']), a=float(abs(a)), b=float(abs(b)))
    F = ObjFrame(tl['lifts'], tl['L1'], tl['L2'], sub=2); F.set_load('stress')
    rec['Mrel'] = float(np.abs(F.Msum).max()*2/np.abs(F.N).max())
    for nm, obj in (('old', False), ('obj', True)):
        F.objective = obj
        t0 = time.time(); ph, ext, _ = F.longwave(nphi=8); h = C.harmonics(ph, ext)
        rec[nm] = dict(c0=h['c0'], h2=h['h2'], h4=h['h4'], pcmin=float(ext.min()), rms=h['fit_rms'], sec=time.time() - t0)
    with out.open('a') as o: o.write(json.dumps(rec) + '\n')
    n, o_ = rec['obj'], rec['old']
    print(fam, f'{p}/{q}', seed, rec['V'], 'c0 %.6f->%.6f |h2| %.2e->%.2e |h4| %.3e->%.3e' % (o_['c0'], n['c0'], abs(complex(*o_['h2'])), abs(complex(*n['h2'])), abs(complex(*o_['h4'])), abs(complex(*n['h4']))), flush=True)

"""Follow-up to the narrow G7 and G8 misses at sub 16: compare with the exact polygon integral and refine to sub 32. Registered thresholds are not changed.
The polygon through nodes on sin(2 pi x) has int w^2 = (2 + cos(2 pi/sub))/3 * 1/2 (C), so c(sub) = 6 (2 + cos(2 pi/sub))/3."""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames', 'design', 'nonlinear', 'phason_paths')]  # shared modules
import json
from defects_nonlinear import *
r = 0.03; t = math.sqrt(3) * r
nodes, a1, a2, E = X10.ref_def('hexagonal')
vecs = []
for e in E:
    v = tuple(np.round(e[2], 12))
    if v not in vecs:
        vecs.append(v)
Ph = np.array(vecs, float).T
edges_h = np.array([(e[0], e[1], vecs.index(tuple(np.round(e[2], 12))), e[3], e[4]) for e in E], np.int64)
R = dict(G7=[], G8=[])
for sub in (4, 8, 16, 32):
    mh = from_edges(nodes, edges_h, Ph, a1, a2, sub, r, 'hex'); dU = bow_shape(mh, None, 'full'); K00, _, _ = schur_moduli(mh)
    for w0 in (0.01, 0.03):
        e = w0 / t; Kb, _, _ = schur_moduli(mh.with_imperfection(w0 * dU, None))
        cf = (K00 / Kb - 1) / e ** 2; cp = 6 * (2 + math.cos(2 * math.pi / sub)) / 3
        R['G7'].append(dict(sub=sub, w0=w0, c_fit=cf, c_polygon_C=cp, rel_to_polygon=cf / cp - 1, rel_to_6=cf / 6 - 1))
        print(R['G7'][-1], flush=True)
sub = 32; w0 = 0.01
mh = from_edges(nodes, edges_h, Ph, a1, a2, sub, r, 'hex'); mb = mh.with_imperfection(w0 * bow_shape(mh, None, 'full'), None)
ms = Model(mh); mo = Model(mb); lam_h = float(ms.linear_critical(macro=True, nev=2)[0][0])
for frac in (0.1, 0.25, 0.5):
    p = frac * lam_h; zs = primary_to(ms, p, nstep=6); zb = primary_to(mo, p, nstep=6)
    As = ms.area(zs) / mh.A0 - 1; Ab = mo.area(zb) / mb.A0 - 1
    n = math.sqrt(3) * p * mh.lamE / (4 * math.pi ** 2 * mh.EI); a = 1 / (1 - n)
    pred = -2 * math.pi ** 2 * w0 ** 2 * (a ** 2 - 1) * (1 + As / 2)
    R['G8'].append(dict(sub=sub, frac=frac, rel=float((Ab - As) / pred - 1)))
    print(R['G8'][-1], flush=True)
(HERE_defects / 'checks_nonlinear_extra.json').write_text(json.dumps(R, indent=1), encoding='utf-8')

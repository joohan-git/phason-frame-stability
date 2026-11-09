"""gates G5-G8. Writes checks_nonlinear.json."""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames', 'design', 'nonlinear', 'phason_paths')]  # shared modules
import json, time
from defects_nonlinear import *
R = {}
out = HERE_defects / 'checks_nonlinear.json'


def rec(k, val, ok, **kw):
    R[k] = dict(value=val, passed=bool(ok), **kw)
    print(k, val, 'PASS' if ok else 'FAIL', str(kw)[:600], flush=True)
    out.write_text(json.dumps(R, indent=1, default=float), encoding='utf-8')


# G5a zero bow = unbowed mesh
m = get_mesh('B1_5_7', 2)
lam0, vec, st = Model(m).linear_critical(macro=True, nev=2)
sg = edge_signs(m, 'gate')
m0 = m.with_imperfection(0.0 * bow_shape(m, sg), None)
lam1, _, _ = Model(m0).linear_critical(macro=True, nev=2)
K0, G0, _ = schur_moduli(m); K1, G1, _ = schur_moduli(m0)
rec('G5a', abs(lam1[0] - lam0[0]), abs(lam1[0] - lam0[0]) <= 1e-10 and abs(K1 - K0) <= 1e-10, K=K0)
# G5b nonlinear stored run
du, e0 = mode_shape(m, vec[:, 0])
res = RI.run_one(m, du, e0, 1e-4, float(lam0[0]))
rec('G5b', [res['p_05'], res['p_lin']], abs(res['p_05'] - 0.7969) <= 2e-4 and abs(res['p_lin'] - 0.7972) <= 2e-4, target=[0.7969, 0.7972], nsteps=res['nsteps'])
# G5c K from Schur block
m12 = get_mesh('B1_12_17', 2); K, G, mm = schur_moduli(m12)
rec('G5c', K, abs(K / 0.5144846106176608 - 1) <= 1e-6, target=0.5144846106176608, G=G, Gstored=0.003027896950922193)
# G6 clamped half-sine, square lattice, horizontal bars only, same sign (C)
r = 0.03; i2 = r ** 2 / 4
g6 = {}
for sub in (2, 4, 8, 16):
    ms = square_mesh(sub, r)
    nE = ms.ne // sub
    s = np.array([1.0 if abs(ms.d0[e * sub, 1]) < 1e-12 else 0.0 for e in range(nE)])
    dU = bow_shape(ms, s)
    K00, _, _ = schur_moduli(ms)
    row = {}
    for w0 in (0.01, 0.03):
        Kb, _, _ = schur_moduli(ms.with_imperfection(w0 * dU, None))
        S11 = 1 + w0 ** 2 * (0.5 - 4 / math.pi ** 2) / i2
        exact = 1 / (S11 + 1)
        row[str(w0)] = dict(K=Kb, exact_C=exact, rel=Kb / exact - 1, bow_part_err=(1 / Kb - 1 / K00) / (S11 - 1) - 1)
    g6[sub] = row
    print('G6', sub, row, flush=True)
rec('G6', max(abs(g6[16][k]['rel']) for k in g6[16]), max(abs(g6[16][k]['rel']) for k in g6[16]) <= 0.01, table=g6)
# G7 Symons and Fleck on the hexagonal lattice with full-wave bars, same handedness (C)
nodes, a1, a2, E = X10.ref_def('hexagonal')
vecs = []
for e in E:
    v = tuple(np.round(e[2], 12))
    if v not in vecs:
        vecs.append(v)
Ph = np.array(vecs, float).T
edges_h = np.array([(e[0], e[1], vecs.index(tuple(np.round(e[2], 12))), e[3], e[4]) for e in E], np.int64)
t = math.sqrt(3) * r
g7 = {}
for sub in (4, 8, 16):
    mh = from_edges(nodes, edges_h, Ph, a1, a2, sub, r, 'hex')
    dU = bow_shape(mh, None, 'full')
    K00, _, _ = schur_moduli(mh)
    row = {}
    for w0 in (0.01, 0.03):
        e = w0 / t
        Kb, _, _ = schur_moduli(mh.with_imperfection(w0 * dU, None))
        row[str(w0)] = dict(ratio=Kb / K00, exact_C=1 / (1 + 6 * e ** 2), rel=(Kb / K00) * (1 + 6 * e ** 2) - 1, c_fit=(K00 / Kb - 1) / e ** 2)
    g7[sub] = row
    print('G7', sub, row, flush=True)
rec('G7', max(abs(g7[16][k]['rel']) for k in g7[16]), max(abs(g7[16][k]['rel']) for k in g7[16]) <= 0.01, table=g7, K0=K00, K0_exact_note='0.28868 at r=0.03 (design)')
# G8 bow growth under load (C), sub 16, w0 = 1 %. N = sqrt3 p; shortening by bow = pi^2 w0^2 (a^2 - 1), a = 1/(1 - n), n = N/N2
sub = 16; w0 = 0.01
mh = from_edges(nodes, edges_h, Ph, a1, a2, sub, r, 'hex')
mb = mh.with_imperfection(w0 * bow_shape(mh, None, 'full'), None)
ms = Model(mh); mo = Model(mb)
lam_h = float(ms.linear_critical(macro=True, nev=2)[0][0])
rows = []
for frac in (0.1, 0.25, 0.5):
    p = frac * lam_h
    zs = primary_to(ms, p, nstep=6); zb = primary_to(mo, p, nstep=6)
    As = ms.area(zs) / mh.A0 - 1; Ab = mo.area(zb) / mb.A0 - 1
    n = math.sqrt(3) * p * mh.lamE / (4 * math.pi ** 2 * mh.EI)
    a = 1 / (1 - n)
    d_bow = math.pi ** 2 * w0 ** 2 * (a ** 2 - 1)
    pred = -2 * d_bow * (1 + As / 2)
    rows.append(dict(frac=frac, p=p, n=n, dA_bow=Ab - As, pred_C=pred, rel=(Ab - As) / pred - 1))
    print('G8', rows[-1], flush=True)
rec('G8', max(abs(x['rel']) for x in rows), max(abs(x['rel']) for x in rows) <= 0.01, rows=rows, lam_hex=lam_h)
print('nl gates done')

"""gates G0-G4. Writes checks_linear_extra.json."""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames', 'design', 'nonlinear', 'phason_paths')]  # shared modules
import json, time
from defects_core import *
R = {}
out = HERE / 'checks_linear_extra.json'


def rec(k, val, ok, **kw):
    R[k] = dict(value=val, passed=bool(ok), **kw)
    print(k, val, 'PASS' if ok else 'FAIL', str(kw)[:400], flush=True)
    out.write_text(json.dumps(R, indent=1, default=str), encoding='utf-8')


D = json.loads((M / 'design' / 'design_results.json').read_text())['D']['cells']


def cellT(cls, name, L, **kw):
    return dict(cls=cls, name=name, L1=L[0], L2=L[1], off=7, **kw)


# G1 supercell vs primitive
for lat, N1, N2 in (('triangular', 12, 12),):
    t0 = time.time()
    fp = X.make_ref(lat, 0.03, 2); fp.set_stress(-np.eye(2)); bz = X.bz_scan(fp, 24); ext = fp.lw_phi(np.arange(12) * np.pi / 12)
    lam_p = min(bz[0], float(ext.min())); K_p = X.moduli(fp.C3)['K']
    mod, lifts, L1, L2, edges = ref_super(lat, N1, N2, 0.03); f = frame(mod, lifts, L1, L2, edges, 2); s = sample(f, mod)
    rec(f'G1_{lat}', s['lam_c'], abs(s['lam_c'] / lam_p - 1) <= 1e-5 and abs(s['K'] / K_p - 1) <= 1e-5, primitive=lam_p, K=s['K'], K_p=K_p, kbz=bz[1], mode=s['mode'], V=len(lifts), E=len(edges), sec=time.time() - t0)
# G2 deletion: triangular minus the c120 family = 60 deg rhomb lattice built directly
S3 = math.sqrt(3); N = 6
mod, lifts, L1, L2, edges = ref_super('triangular', N, N, 0.03)
ek = edges[edges[:, 2] != 2]
f = frame(mod, lifts, L1, L2, ek, 2); f.set_stress(-np.eye(2))
modr = X.load('pen'); a1 = np.array([1., 0.]); a2 = np.array([.5, S3 / 2]); modr.P = np.array([a1, a2]).T; modr.NDIM = 2; X.set_r(modr, 0.03)
Pi = np.linalg.pinv(modr.P)
Xn = np.array([a * a1 + b * a2 for b in range(N) for a in range(N)]); lf = (Pi @ Xn.T).T
ed = []
for b in range(N):
    for a in range(N):
        i = a + N * b
        ed.append((i, (a + 1) % N + N * b, 0, (a + 1) // N, 0)); ed.append((i, a + N * ((b + 1) % N), 1, 0, (b + 1) // N))
f2 = frame(modr, lf, Pi @ (N * a1), Pi @ (N * a2), np.array(ed, np.int64), 2); f2.set_stress(-np.eye(2))
errs = [abs(f.lam_N(k) / f2.lam_N(k) - 1) for k in ((0, 0), (.5, .5), (.23, .11))]
Kerr = abs(X.moduli(f.C3)['K'] / X.moduli(f2.C3)['K'] - 1)
rec('G2', max(errs), max(errs) <= 1e-6 and Kerr <= 1e-9, Kerr=Kerr, errs=errs)
# G3 exact K: square N x N minus a whole row of horizontal edges
N = 8
mod, lifts, L1, L2, edges = ref_super('square', N, N, 0.03)
row = np.array([(e[2] == 0) and (e[0] // N == 3) for e in edges])
ek = edges[~row]; f = frame(mod, lifts, L1, L2, ek, 2); f.set_stress(-np.eye(2)); K = X.moduli(f.C3)['K']
exact = 1 / (N / (N - 1) + 1)
rec('G3', K, abs(K / exact - 1) <= 1e-9, exact_C=exact, rel=abs(K / exact - 1), removed=int(row.sum()))
# G4 connectivity checker
mod, lifts, L1, L2, edges = ref_super('square', 8, 8, 0.03); V = len(lifts)
col = np.array([(e[2] == 0) and (e[0] % 8 == 3) for e in edges])    # every horizontal edge crossing one vertical line: x period lost
a = spans(V, edges); b = spans(V, edges[~col])
vert = (edges[:, 2] == 1); c = spans(V, edges[~vert]); d = connected(V, edges[~vert])
hc = (edges[:, 0] == 0) | (edges[:, 1] == 0)
e_ = spans(V, edges[~hc])
rec('G4', [bool(a), bool(b), bool(c), bool(d), bool(e_)], a and (not b) and (not c) and (not d) and (not e_), meaning='intact spans; line-cut loses x period; all verticals removed disconnects; isolated vertex rejected')
print('gates done')

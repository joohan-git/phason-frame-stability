import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames', 'phason_paths', 'penrose', 'disorder', 'slope_survey')]  # shared modules
import json, sys, time, math
from objective_frame import *
sys.path.insert(0, str(HERE.parent / 'penrose'))
import penrose_frame as Pn
class PObj(Pn.Frame):
    objective = True
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        Le = 1.0 / self.sub
        self.Ve = np.array([vmat(Pn.P[0, q], Pn.P[1, q], Le) for q in range(Pn.NDIM)])
    def prestress(self, strain=-np.eye(2)):
        out = super().prestress(strain)
        K0 = self._assemble(np.ones(len(self.ei)), self.Ke, (0, 0)).real.tocsc()
        xb = self.xa + Pn.P[:, self.dirk].T / self.sub
        dimp = np.zeros((len(self.ei), 6))
        dimp[:, 0:2] = self.xa @ strain.T; dimp[:, 3:5] = xb @ strain.T
        fe = np.einsum('eab,eb->ea', self.Ke[self.dirk], dimp)
        f = np.zeros(self.n)
        np.add.at(f, (3 * self.ei[:, None] + np.arange(3)).ravel(), fe[:, :3].ravel())
        np.add.at(f, (3 * self.ej[:, None] + np.arange(3)).ravel(), fe[:, 3:].ravel())
        w = np.zeros(self.n); fr = self.free_gamma
        w[fr] = self._lu0.solve(-f[fr])
        d = dimp.copy()
        d[:, :3] += w[(3 * self.ei[:, None] + np.arange(3))]
        d[:, 3:] += w[(3 * self.ej[:, None] + np.arange(3))]
        fe2 = np.einsum('eab,eb->ea', self.Ke[self.dirk], d)
        self.Msum = fe2[:, 2] + fe2[:, 5]
        return out
    def lam(self, k, nev=4, vectors=False):
        k = np.asarray(k, float)
        K = self._assemble(np.ones(len(self.ei)), self.Ke, k).tocsc()
        H = -self._assemble(self.N, self.Ge, k).tocsc()
        if self.objective:
            H = (H - self._assemble(self.Msum, self.Ve, k)).tocsc()
        gamma = np.allclose(k, np.rint(k))
        fr = self.free_gamma if gamma else np.arange(self.n)
        K = K[fr][:, fr]; H = H[fr][:, fr]
        lu = spla.splu(K.tocsc(), permc_spec='COLAMD')
        op = spla.LinearOperator(K.shape, matvec=lambda x: lu.solve(H @ x), dtype=complex)
        vals, vecs = spla.eigs(op, k=nev, which='LR', tol=1e-10, maxiter=50000)
        mu = vals.real.max()
        return 1 / mu / Pn.LAM_E if mu > 0 else np.inf
out = HERE / 'objective_mixed.jsonl'
def go(name, F):
    F.set_load('stress'); rec = dict(name=name, V=int(F.n / 3))
    for nm, obj in (('old', False), ('obj', True)):
        F.objective = obj; ph, ext, _ = F.longwave(nphi=4)
        v = dict(lw=float(ext.min()), gamma=float(F.lam((0, 0))), hh=float(F.lam((.5, .5)))); v['min'] = min(v.values()); rec[nm] = v
    with out.open('a') as o: o.write(json.dumps(rec) + '\n')
    print(name, rec['V'], 'min %.6f -> %.6f (%.2e) gamma %.6f -> %.6f' % (rec['old']['min'], rec['obj']['min'], rec['obj']['min'] / rec['old']['min'] - 1, rec['old']['gamma'], rec['obj']['gamma']), flush=True)
d4 = json.loads((HERE.parent / 'disorder' / 'random_mode.json').read_text())
go('AB fig4 seed22 1393', ObjFrame(np.array(d4['lifts']), d4['L1'], d4['L2'], sub=2))
L1, L2 = C.ab_approx(5, 7)
for fl_, seed in ((239 * 4, 1), (239 * 40, 2), (239 * 400, 3)):
    tl = C.canonical(L1, L2, seed=7); r = C.flip_mc(tl['lifts'], tl['L1'], tl['L2'], fl_, seed)
    lifts = r['lifts'] if isinstance(r, dict) else (r[0] if isinstance(r, tuple) else r)
    go(f'AB 5/7 flips{fl_} s{seed}', ObjFrame(lifts, tl['L1'], tl['L2'], sub=2))
ST = json.loads((HERE.parent / 'slope_survey' / 'slope_table.json').read_text(encoding='utf-8'))
print(list(ST.keys())[:8] if isinstance(ST, dict) else type(ST), flush=True)
A = [([4, 3, 1, -1, -3], [0, 1, 2, 2, 1]), ([10, 8, 3, -3, -8], [0, 3, 5, 5, 3])]
for i, (a1, a2) in enumerate(A):
    tl = Pn.canonical(a1, a2)
    go(f'Pen a{i}', PObj(tl['lifts'], tl['L1'], tl['L2'], sub=2))
    r = Pn.flip_mc(tl['lifts'], tl['L1'], tl['L2'], len(tl['lifts']) * 100, 5)
    lifts = r['lifts'] if isinstance(r, dict) else (r[0] if isinstance(r, tuple) else r)
    go(f'Pen a{i} random', PObj(lifts, tl['L1'], tl['L2'], sub=2))
for n in (2, 3):
    b1, b2 = Pn.fib_approx(n); tl = Pn.canonical(b1, b2)
    go(f'Pen fib{n}', PObj(tl['lifts'], tl['L1'], tl['L2'], sub=2))

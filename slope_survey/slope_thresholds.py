"""lambda(s): hydrostatic buckling limit of periodic rhombus tilings on the slope space of Z^4.

Every periodic tiling here uses the SAME four unit edge vectors as Ammann-Beenker
(directions 0, 45, 90, 135 deg), hence the same tile set (unit squares, unit 45-deg rhombi).
A tiling is the cut of Z^4 by a rational plane E = span(L1, L2), L1, L2 in Z^4,
window = projection of the unit hypercube along E (canonical cut), generic offset.
Tile counts per cell = Pluecker minors p_ij(L1, L2) (all >= 0 <=> valid tiling).

Mechanics: planar Euler-Bernoulli frame, rigid joints, 3 dof/node, each edge split in
`SUB` elements, EA = 1, EI = (r^2/4) EA (circular strut, r = 0.03), L = 1.
Prestress: periodic linear solve under macroscopic hydrostatic strain -I (strain protocol)
or hydrostatic stress (stress protocol, rescaled to mean strain -1).
lambda = smallest critical multiplier over a Bloch k grid; reported / lambda_E,
lambda_E = pi^2 EI / (EA L^2) (pinned Euler strain of one edge).
Linear buckling only; not certified; small cells only.
"""
from __future__ import annotations
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames',)]  # shared modules
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import itertools, json, math, sys, time
from fractions import Fraction
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as la
from scipy.spatial import ConvexHull

T = 1 / math.sqrt(2)
P = np.array([[1, T, 0, -T], [0, T, 1, T]])  # AB physical projection (columns = edge vectors)
R = 0.03
EA = 1.0
EI = R * R / 4
LAM_E = math.pi ** 2 * EI / EA
SUB = 2
DQ = 0.004
PAIRS = list(itertools.combinations(range(4), 2))


def minors(L1, L2):
    return {f"{i+1}{j+1}": int(L1[i] * L2[j] - L1[j] * L2[i]) for i, j in PAIRS}


def tiling(L1, L2, offset_seed=7):
    L1 = np.array(L1, int); L2 = np.array(L2, int)
    m = minors(L1, L2)
    if all(v <= 0 for v in m.values()):
        L1, L2 = L2, L1; m = minors(L1, L2)
    assert all(v >= 0 for v in m.values()), f"slope not in closed positive Grassmannian: {m}"
    Lm = np.array([L1, L2], float)
    # orthonormal basis of E-perp
    q, _ = np.linalg.qr(np.c_[Lm.T, np.eye(4)])
    perp = q[:, 2:4].T
    cube = np.array(list(itertools.product([-.5, .5], repeat=4)))
    hull = ConvexHull(cube @ perp.T)
    eq = hull.equations
    rng = np.random.default_rng(offset_seed)
    c = rng.uniform(-.5, .5, 4) * 0.37 + np.array([0.1173, 0.2711, -0.0618, 0.3307])
    cperp = perp @ c

    def inside(n):
        y = perp @ n - cperp
        return np.all(eq[:, :2] @ y + eq[:, 2] < -1e-9)

    G = np.array([[L1 @ L1, L1 @ L2], [L2 @ L1, L2 @ L2]], dtype=object)
    det = int(G[0, 0] * G[1, 1] - G[0, 1] * G[1, 0])
    adj = np.array([[G[1, 1], -G[0, 1]], [-G[1, 0], G[0, 0]]], dtype=object)

    def reduce(n):
        rhs = np.array([int(n @ L1), int(n @ L2)], dtype=object)
        num = adj @ rhs
        a, b = int(num[0]) // det, int(num[1]) // det
        return tuple(int(v) for v in (n - a * L1 - b * L2)), (a, b)

    # seed point: search near origin
    start = None
    for v in itertools.product(range(-2, 3), repeat=4):
        n = np.array(v)
        if inside(n.astype(float)):
            start = n; break
    assert start is not None
    s0, _ = reduce(start)
    reps = {s0: 0}; order = [s0]; edges = []
    queue = [s0]
    while queue:
        cur = queue.pop()
        nc = np.array(cur)
        for k in range(4):
            nb = nc.copy(); nb[k] += 1
            if inside(nb.astype(float)):
                r, wrap = reduce(nb)
                if r not in reps:
                    reps[r] = len(order); order.append(r); queue.append(r)
                edges.append((reps[cur], reps[r], k, wrap))
            nb = nc.copy(); nb[k] -= 1
            if inside(nb.astype(float)):
                r, _ = reduce(nb)
                if r not in reps:
                    reps[r] = len(order); order.append(r); queue.append(r)
    nV = len(order)
    edges = list({(i, j, k, w) for i, j, k, w in edges})
    expected = sum(m.values())
    assert nV == expected, (nV, expected, m)
    assert len(edges) == 2 * nV, (len(edges), nV)
    X = np.array(order, float) @ P.T
    cell = np.array([P @ L1, P @ L2])
    deg = np.zeros(nV, int)
    for i, j, k, w in edges:
        deg[i] += 1; deg[j] += 1
    return dict(L1=L1.tolist(), L2=L2.tolist(), minors=m, X=X, edges=edges, cell=cell, deg=deg)


def beam_K(L):
    k = np.zeros((6, 6))
    a = EA / L
    k[np.ix_([0, 3], [0, 3])] = a * np.array([[1, -1], [-1, 1]])
    b = EI / L ** 3 * np.array([[12, 6 * L, -12, 6 * L], [6 * L, 4 * L * L, -6 * L, 2 * L * L],
                               [-12, -6 * L, 12, -6 * L], [6 * L, 2 * L * L, -6 * L, 4 * L * L]])
    k[np.ix_([1, 2, 4, 5], [1, 2, 4, 5])] = b
    return k


def beam_G(L):
    g = np.zeros((6, 6))
    g[np.ix_([1, 2, 4, 5], [1, 2, 4, 5])] = np.array(
        [[36, 3 * L, -36, 3 * L], [3 * L, 4 * L * L, -3 * L, -L * L],
         [-36, -3 * L, 36, -3 * L], [3 * L, -L * L, -3 * L, 4 * L * L]]) / (30 * L)
    # axial-only string term omitted (as in K08 bloch_r3)
    return g


def rot(e):
    c, s = e
    r = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
    return np.kron(np.eye(2), r)


class Frame:
    def __init__(self, t):
        self.t = t
        X = t['X']; cell = t['cell']
        nodes = [x for x in X]
        els = []
        for i, j, k, w in t['edges']:
            a = X[i]; d = P[:, k]
            ids = [i]
            for h in range(1, SUB):
                ids.append(len(nodes)); nodes.append(a + d * h / SUB)
            ids.append(j)
            for h in range(SUB):
                wrap = np.array(w) if h == SUB - 1 else np.zeros(2, int)
                xa = a + d * h / SUB; xb = a + d * (h + 1) / SUB
                T_ = rot(d)
                L = 1 / SUB
                els.append(dict(i=ids[h], j=ids[h + 1], wrap=wrap, xa=xa, xb=xb, e=d,
                                K=T_.T @ beam_K(L) @ T_, G=T_.T @ beam_G(L) @ T_, edge=len(els) // SUB))
        self.nodes = np.array(nodes); self.els = els; self.n = 3 * len(nodes)
        self.cell = cell

    def assemble(self, mats, k):
        rows, cols, vals = [], [], []
        for e, m in zip(self.els, mats):
            di = 3 * e['i'] + np.arange(3); dj = 3 * e['j'] + np.arange(3)
            ph = np.exp(2j * np.pi * np.dot(k, e['wrap']))
            for (A, B, blk) in [(di, di, m[:3, :3]), (dj, dj, m[3:, 3:]),
                                (di, dj, ph * m[:3, 3:]), (dj, di, np.conj(ph) * m[3:, :3])]:
                rr, cc = np.meshgrid(A, B, indexing='ij')
                rows.append(rr.ravel()); cols.append(cc.ravel()); vals.append(np.asarray(blk, complex).ravel())
        M = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(self.n, self.n))
        return (M + M.conj().T) / 2

    def prestress(self, strain):
        K0 = self.assemble([e['K'] for e in self.els], np.zeros(2)).real.tocsc()
        f = np.zeros(self.n); imposed = []
        for e in self.els:
            d = np.r_[strain @ e['xa'], 0, strain @ e['xb'], 0]
            fe = e['K'] @ d
            f[3 * e['i']:3 * e['i'] + 3] += fe[:3]; f[3 * e['j']:3 * e['j'] + 3] += fe[3:]
            imposed.append(d)
        free = np.arange(2, self.n)
        w = np.zeros(self.n)
        w[free] = spla.spsolve(K0[free][:, free].tocsc(), -f[free])
        N = []; stress = np.zeros((2, 2))
        for e, d in zip(self.els, imposed):
            d = d + np.r_[w[3 * e['i']:3 * e['i'] + 3], w[3 * e['j']:3 * e['j'] + 3]]
            fe = e['K'] @ d
            delta = e['xb'] - e['xa']
            N.append(EA * SUB * np.dot(e['e'], d[3:5] - d[0:2]))
            stress += .5 * (np.outer(fe[3:5], delta) + np.outer(delta, fe[3:5]))
        area = abs(np.linalg.det(self.cell))
        return np.array(N), stress / area

    def set_load(self, protocol):
        if protocol == 'strain':
            eps = -np.eye(2)
        else:
            basis = [np.array([[1., 0], [0, 0]]), np.array([[0., 0], [0, 1]]), np.array([[0, 1.], [1, 0]]) / math.sqrt(2)]
            C = np.array([[np.sum(b * self.prestress(a)[1]) for a in basis] for b in basis])
            c = la.solve(C, np.array([-1., -1., 0.]))
            eps = sum(ci * b for ci, b in zip(c, basis))
            eps = eps / (-np.trace(eps) / 2)
        self.eps = eps
        self.N, self.stress = self.prestress(eps)
        self.Gm = [Ni * e['G'] for Ni, e in zip(self.N, self.els)]
        return eps

    def lam(self, k):
        K = self.assemble([e['K'] for e in self.els], k).tocsc()
        H = -self.assemble(self.Gm, k).tocsc()
        gamma = np.allclose(k, np.rint(k))
        free = np.arange(2, self.n) if gamma else np.arange(self.n)
        K = K[free][:, free]; H = H[free][:, free]
        if self.n < 1500:
            vals = la.eigh(H.toarray(), K.toarray(), eigvals_only=True)
            mu = vals.max()
        else:
            # largest mu of H v = mu K v  <=> shift-invert-free: use Lanczos with M=K
            Kf = spla.splu(K.tocsc())
            op = spla.LinearOperator(K.shape, matvec=lambda x: Kf.solve(H @ x), dtype=complex)
            vals = spla.eigs(op, k=6, which='LR', return_eigenvectors=False, tol=1e-9, maxiter=20000)
            mu = vals.real.max()
        return 1 / mu if mu > 0 else np.inf


def kgrid(nk):
    pts = set()
    for a in range(nk):
        for b in range(nk):
            ka, kb = Fraction(a, nk), Fraction(b, nk)
            m = ((-ka) % 1, (-kb) % 1)
            if m in pts:
                continue
            pts.add((ka, kb))
    return [np.array([float(a), float(b)]) for a, b in sorted(pts)]


def evaluate(name, L1, L2, protocols=('strain', 'stress'), nk=None):
    t0 = time.time()
    t = tiling(L1, L2)
    V = len(t['X'])
    fr = Frame(t)
    if nk is None:
        nk = 8 if V <= 12 else 4 if V <= 120 else 2
    ks = kgrid(nk)
    # long-wave probes (strong ellipticity limit): physical |q| = 2*pi*DQ along 12 directions
    small = []
    for phi in np.arange(12) * np.pi / 12:
        u = np.array([np.cos(phi), np.sin(phi)])
        small.append(DQ * (t['cell'] @ u))
    m = t['minors']; tot = sum(m.values())
    sq = (m['13'] + m['24']) / tot
    out = dict(name=name, L1=t['L1'], L2=t['L2'], minors=m, V=V, square_fraction=sq,
               freq={k: v / tot for k, v in m.items()},
               degree_hist={str(d): int(np.sum(t['deg'] == d)) for d in np.unique(t['deg'])},
               density=V / abs(np.linalg.det(t['cell'])), nk=nk, results={})
    for prot in protocols:
        eps = fr.set_load(prot)
        Nedge = fr.N.reshape(-1, SUB).mean(axis=1)
        lam = {}
        for k in ks:
            lam[tuple(np.round(k, 4))] = fr.lam(k)
        lw = [fr.lam(k) for k in small]
        kmin = min(lam, key=lam.get)
        lfin = lam[kmin]
        lmin = min(lfin, min(lw))
        p = -np.trace(fr.stress) / 2
        out['results'][prot] = dict(
            lam_over_lamE=lmin / LAM_E, lam_grid_over_lamE=lfin / LAM_E, k_crit=list(kmin),
            lam_longwave_over_lamE=min(lw) / LAM_E, longwave_dir_deg=float(15 * int(np.argmin(lw))),
            longwave_all=[v / LAM_E for v in lw], eps=eps.tolist(),
            pressure_per_unit_strain=p, pc_over_EAlamE=lmin * p / LAM_E,
            N_over_affine=dict(min=float(-Nedge.max()), max=float(-Nedge.min()), mean=float(-Nedge.mean())),
            bipartite_rayleigh_bound=float(1 / max(-Nedge.mean(), 1e-30)),
            lam_by_k={str(k): v / LAM_E for k, v in lam.items()})
    out['seconds'] = time.time() - t0
    print(json.dumps({'name': name, 'V': V, 'sq': round(sq, 4),
                      **{p: round(out['results'][p]['lam_over_lamE'], 4) for p in protocols},
                      'grid': round(out['results'][protocols[0]]['lam_grid_over_lamE'], 4),
                      'lw': round(out['results'][protocols[0]]['lam_longwave_over_lamE'], 4),
                      'Nmax_strain': round(out['results']['strain']['N_over_affine']['max'], 3) if 'strain' in protocols else None,
                      's': round(out['seconds'], 1)}), flush=True)
    return out


def ab_approx(p, q):
    return [q, p, 0, -p], [0, p, q, p]


SLOPES = [
    # pure lattices (closed boundary of Gr+)
    ('square_13', [1, 0, 0, 0], [0, 0, 1, 0]),
    ('rhomb_12', [1, 0, 0, 0], [0, 1, 0, 0]),
    # symmetric AB line: p/q, QC at 1/sqrt2
    *[(f'AB_{p}/{q}', *ab_approx(p, q)) for p, q in [(1, 4), (1, 3), (1, 2), (2, 3), (5, 7), (12, 17), (3, 4), (1, 1), (3, 2), (2, 1)]],
    # asymmetric slopes (all minors > 0)
    ('asym_A', [2, 1, 0, -1], [0, 1, 1, 1]),
    ('asym_B', [1, 1, 0, -1], [0, 1, 2, 1]),
    ('asym_C', [3, 1, 0, 0], [0, 1, 2, 1]),
    ('asym_D', [2, 2, 1, -1], [0, 1, 2, 2]),
    ('asym_E', [1, 0, 0, -1], [0, 1, 1, 0]),
    ('asym_F', [2, 1, 0, -1], [1, 1, 1, 0]),
]


if __name__ == '__main__':
    sel = sys.argv[1:]
    rows = []
    for name, L1, L2 in SLOPES:
        if sel and name not in sel:
            continue
        try:
            rows.append(evaluate(name, L1, L2))
        except AssertionError as e:
            print(name, 'FAILED', e, flush=True)
            rows.append(dict(name=name, error=str(e)))
    out = Path(__file__).with_name('slope_thresholds.json' if not sel else 'slope_thresholds_partial.json')
    out.write_text(json.dumps(dict(lamE=LAM_E, r=R, sub=SUB, rows=rows), indent=1, default=str), encoding='utf-8')

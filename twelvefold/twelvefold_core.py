"""D12 core: canonical (Oguey-Duneau-Katz / Voronoi-window) cut-and-project tilings from the
rank-4 dodecagonal lattice Z[zeta12] = A2 (+) A2, their rational approximants, and the same
2D Euler-Bernoulli rigid-joint frame as main/ab_frame.py (EA = 1, EI = r^2/4, r = 0.03, unit edges,
`sub` elements per edge, stress (pressure) protocol, long-wave Richardson DQ -> 0).

Conventions
  * lattice coordinates n in Z^4 on basis v0..v3, v_k = (zeta^k ; zeta^{5k}) in R^2 (phys) x R^2 (int),
    zeta = e^{i pi/6}.  Roots (unit physical edges): v0..v3, v4 = v2 - v0, v5 = v3 - v1 (and negatives).
    Gram = [[2,0,1,0],[0,2,0,1],[1,0,2,0],[0,1,0,2]] (A2 + A2), det B = 3.
  * 12-fold generator g = multiplication by zeta: n -> (-n3, n0, n1 + n3, n2); physical R(30), internal R(150).
  * slope plane E' = span(x(L1), x(L2)); internal coordinate along E': y'(z) = z_int - M z_phys,
    M = Q L (P L)^-1 = uniform phason strain (internal = M physical); S = aI + bJ + cZ + dX.
    alpha = a + i b (conformal, m = 4, E4), beta = c + i d (anticonformal, m = 6, B1 + B2).
  * vertex n accepted iff y'(n - c) in y'(Vor) (Voronoi cell = hexagon x hexagon, symmetric);
    edge [n, n + v] iff y'(c - n) in y'(F_v), F_v = facet of Vor dual to root v (exact canonical rule).
  * physical vertex position = P n (fixed physical plane, as in ab_frame).
"""
from __future__ import annotations
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames', 'phason_paths')]  # shared modules
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import itertools, math
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as la
from scipy.spatial import ConvexHull

K = np.arange(4)
P = np.array([np.cos(K * np.pi / 6), np.sin(K * np.pi / 6)])
Q = np.array([np.cos(5 * K * np.pi / 6), np.sin(5 * K * np.pi / 6)])
B = np.vstack([P, Q])
ROOTS = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, 0, 1, 0], [0, -1, 0, 1]], np.int64)
DIRS = (ROOTS @ P.T).T  # 2 x 6 unit physical directions at 0,30,...,150 deg
assert np.allclose(np.linalg.norm(DIRS, axis=0), 1)
GEN = np.array([[0, 0, 0, -1], [1, 0, 0, 0], [0, 1, 0, 1], [0, 0, 1, 0]], np.int64)  # zeta *
UNIT = 2 * np.eye(4, dtype=np.int64) + 2 * GEN - np.linalg.matrix_power(GEN, 3)  # u = 2 + sqrt3
R_STRUT = 0.03
EA = 1.0
EI = R_STRUT ** 2 / 4
LAM_E = math.pi ** 2 * EI / EA


def _voronoi():
    xr = (ROOTS @ B.T)  # 6 x 4 embedded roots
    allr = np.vstack([xr, -xr])
    ev = [allr[i] for i in (0, 2, 4)]; ev = ev + [-e for e in ev]  # even roots
    od = [allr[i] for i in (1, 3, 5)]; od = od + [-e for e in od]
    def hexv(rs):
        ang = [math.atan2(np.dot(r, rs[1]), np.dot(r, rs[0])) for r in rs]
        rs = [r for _, r in sorted(zip(ang, rs), key=lambda t: t[0])]
        return [(rs[j] + rs[(j + 1) % 6]) / 3 for j in range(6)]
    H1, H2 = hexv(ev), hexv(od)
    verts = np.array([h1 + h2 for h1 in H1 for h2 in H2])
    for r in allr:  # check: Vor = {x : x.r <= 1}
        assert np.all(verts @ r <= 1 + 1e-12)
    facets = [verts[np.abs(verts @ xr[k] - 1) < 1e-9] for k in range(6)]
    assert all(len(f) == 12 for f in facets)
    return verts, facets


VOR, FACETS = _voronoi()


def phason_matrix(L1, L2):
    L = np.array([L1, L2], float).T
    return (Q @ L) @ np.linalg.inv(P @ L)


def strain_parts(M):
    """alpha = a + i b (conformal, aI + bJ), beta = c + i d (cZ + dX)."""
    a = (M[0, 0] + M[1, 1]) / 2; b = (M[1, 0] - M[0, 1]) / 2
    c = (M[0, 0] - M[1, 1]) / 2; d = (M[1, 0] + M[0, 1]) / 2
    return complex(a, b), complex(c, d)


class Reducer:
    def __init__(self, L1, L2):
        self.L1 = np.array(L1, np.int64); self.L2 = np.array(L2, np.int64)
        g11, g12, g22 = int(self.L1 @ self.L1), int(self.L1 @ self.L2), int(self.L2 @ self.L2)
        self.det = g11 * g22 - g12 * g12
        self.adj = (g22, -g12, -g12, g11)

    def __call__(self, n):
        r1, r2 = int(n @ self.L1), int(n @ self.L2)
        a = (self.adj[0] * r1 + self.adj[1] * r2) // self.det
        b = (self.adj[2] * r1 + self.adj[3] * r2) // self.det
        return n - a * self.L1 - b * self.L2, (a, b)


def _hull_eq(pts):
    h = ConvexHull(pts)
    return h.equations, h.volume


def tiling(L1, L2, offset=None, seed=7):
    """Canonical tiling for slope span(x(L1), x(L2)).  offset: 4-vector c in embedding space."""
    L1 = np.array(L1, np.int64); L2 = np.array(L2, np.int64)
    if np.linalg.det(np.array([P @ L1, P @ L2])) < 0:
        L1, L2 = L2, L1
    M = phason_matrix(L1, L2)
    ymap = lambda Z: Z[..., 2:] - Z[..., :2] @ M.T  # embedding -> y'
    Ybasis = Q - M @ P  # y'(n) = Ybasis n
    W_eq, W_area = _hull_eq(ymap(VOR))
    F_eq = [_hull_eq(ymap(F))[0] for F in FACETS]
    if offset is None:
        rng = np.random.default_rng(seed)
        offset = rng.uniform(-1, 1, 4) * np.array([0.37, 0.41, 0.29, 0.33]) + np.array([0.1173, 0.2711, -0.0618, 0.3307])
    yc = ymap(np.asarray(offset, float))
    cell = np.array([P @ L1, P @ L2])
    area = abs(np.linalg.det(cell))
    nexp = area * W_area / 3.0
    red = Reducer(L1, L2)

    def inside(eq, y, tol=-1e-9):
        return bool(np.all(eq[:, :2] @ y + eq[:, 2] < tol))

    def vok(n):
        return inside(W_eq, Ybasis @ n - yc)

    start = None
    for v in itertools.product(range(-3, 4), repeat=4):
        if vok(np.array(v, float)):
            start = np.array(v, np.int64); break
    s0, _ = red(start)
    reps = {tuple(s0): 0}; order = [tuple(s0)]; stack = [tuple(s0)]
    allroots = np.vstack([ROOTS, -ROOTS])
    margin = []
    while stack:
        cur = np.array(stack.pop(), np.int64)
        for d in allroots:
            nb = cur + d
            if vok(nb.astype(float)):
                r, _ = red(nb); r = tuple(r)
                if r not in reps:
                    reps[r] = len(order); order.append(r); stack.append(r)
    lifts = np.array(order, np.int64)
    # minimal distance of internal coords to window boundary (genericity diagnostic)
    yy = lifts @ Ybasis.T - yc
    wmargin = float(np.min(-(yy @ W_eq[:, :2].T + W_eq[:, 2]).max(1)))
    # edges by the facet criterion
    look = {tuple(r): i for i, r in enumerate(lifts)}
    edges = []; nonroot_pairs = 0; fmargin = np.inf
    for i, r in enumerate(lifts):
        yr = Ybasis @ r - yc  # y'(n - c);  need y'(c - n) = -yr in y'(F_v)
        for k in range(6):
            s = -(F_eq[k][:, :2] @ (-yr) + F_eq[k][:, 2]).max()
            fmargin = min(fmargin, abs(s))
            rr, w = red(r + ROOTS[k])
            j = look.get(tuple(rr))
            if s > 0:
                assert j is not None, 'edge to non-vertex'
                edges.append((i, j, k, w[0], w[1]))
            elif j is not None:
                nonroot_pairs += 1  # both endpoints accepted but facet not cut (non-edge)
    edges = np.array(edges, np.int64)
    return dict(L1=L1, L2=L2, M=M, lifts=lifts, edges=edges, cell=cell, area=area, n_expected=nexp,
                W_area=W_area, W_eq=W_eq, Ybasis=Ybasis, yc=yc, offset=np.asarray(offset, float),
                window_margin=wmargin, facet_margin=float(fmargin), unit_pairs_not_edges=nonroot_pairs)


def faces(tl):
    """Trace faces of the periodic planar graph. Returns list of (n_sides, sorted interior angles deg,
    signed area) and vertex configurations."""
    lifts, edges, cell = tl['lifts'], tl['edges'], tl['cell']
    V = len(lifts)
    nbr = [[] for _ in range(V)]  # (angle, j, wrap(2), dvec)
    for (i, j, k, w1, w2) in edges:
        d = DIRS[:, k]
        nbr[i].append((math.atan2(d[1], d[0]) % (2 * math.pi), j, (w1, w2), k, 1))
        nbr[j].append((math.atan2(-d[1], -d[0]) % (2 * math.pi), i, (-w1, -w2), k, -1))
    for l in nbr:
        l.sort()
    pos = {}
    for i in range(V):
        for idx, (_, j, w, k, s) in enumerate(nbr[i]):
            pos[(i, j, w, k, s)] = idx
    used = set(); out = []
    for i in range(V):
        for idx in range(len(nbr[i])):
            if (i, idx) in used:
                continue
            cur, ci = i, idx; poly = [np.zeros(2)]; dirs = []; n = 0
            while (cur, ci) not in used:
                used.add((cur, ci))
                ang, j, w, k, s = nbr[cur][ci]
                dv = s * DIRS[:, k]; dirs.append(dv); poly.append(poly[-1] + dv)
                # at j, reverse half-edge is (j -> cur, -w, k, -s); next = previous in CCW order (face on left)
                ridx = pos[(j, cur, (-w[0], -w[1]), k, -s)]
                ci = (ridx - 1) % len(nbr[j]); cur = j; n += 1
                if n > 50:
                    break
            pts = np.array(poly[:-1])
            A = 0.5 * np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1])
            angs = []
            for t in range(len(dirs)):
                a_, b_ = -dirs[t - 1], dirs[t]
                angs.append(round(math.degrees(math.acos(np.clip(a_ @ b_, -1, 1)))))
            out.append((len(dirs), tuple(sorted(angs)), A, bool(np.linalg.norm(poly[-1]) < 1e-9)))
    vconf = {}
    for i in range(V):
        gaps = []
        a = [t[0] for t in nbr[i]]
        for t in range(len(a)):
            g = (a[(t + 1) % len(a)] - a[t]) % (2 * math.pi)
            gaps.append(int(round(math.degrees(g))))
        # canonical cyclic word
        word = min(tuple(gaps[t:] + gaps[:t]) for t in range(len(gaps)))
        word = min(word, min(tuple((gaps[::-1])[t:] + (gaps[::-1])[:t]) for t in range(len(gaps))))
        vconf[word] = vconf.get(word, 0) + 1
    return out, vconf


def validate(tl):
    fs, vconf = faces(tl)
    kinds = {}
    for n, angs, A, closed in fs:
        if n == 3 and angs == (60, 60, 60):
            key = 'triangle'
        elif n == 4 and angs == (90, 90, 90, 90):
            key = 'square'
        elif n == 4 and angs == (30, 30, 150, 150):
            key = 'rhomb30'
        else:
            key = f'other{n}{angs}'
        kinds[key] = kinds.get(key, 0) + 1
    V, E, F = len(tl['lifts']), len(tl['edges']), len(fs)
    tot = sum(abs(A) for *_, A, _c in fs)
    neg = sum(1 for *_, A, _c in fs if A < 0)
    return dict(V=V, E=E, F=F, euler=V - E + F, n_expected=tl['n_expected'], area=tl['area'],
                face_area_sum=tot, neg_area_faces=neg, kinds=kinds, vconf={'.'.join(map(str, k)): v for k, v in sorted(vconf.items(), key=lambda t: -t[1])},
                all_closed=all(c for *_, c in fs), window_margin=tl['window_margin'],
                facet_margin=tl['facet_margin'], unit_pairs_not_edges=tl['unit_pairs_not_edges'])


def approximant(n_pow, w, kind):
    """L1 = u^n * w (w in lattice coords), L2 = zeta^3 L1 (kind 'C4', pure alpha) or zeta^2 L1 ('C6', pure beta)."""
    L1 = np.linalg.matrix_power(UNIT, n_pow) @ np.array(w, np.int64)
    L2 = np.linalg.matrix_power(GEN, 3 if kind == 'C4' else 2) @ L1
    return L1, L2


# ----------------------------------------------------------------- frame mechanics (ab_frame, 6 directions)
def _rot6(c, s):
    r = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1.]])
    return np.kron(np.eye(2), r)


def _beam_KG(L):
    k = np.zeros((6, 6)); g = np.zeros((6, 6))
    k[np.ix_([0, 3], [0, 3])] = EA / L * np.array([[1, -1], [-1, 1]])
    k[np.ix_([1, 2, 4, 5], [1, 2, 4, 5])] = EI / L ** 3 * np.array(
        [[12, 6 * L, -12, 6 * L], [6 * L, 4 * L * L, -6 * L, 2 * L * L],
         [-12, -6 * L, 12, -6 * L], [6 * L, 2 * L * L, -6 * L, 4 * L * L]])
    g[np.ix_([1, 2, 4, 5], [1, 2, 4, 5])] = np.array(
        [[36, 3 * L, -36, 3 * L], [3 * L, 4 * L * L, -3 * L, -L * L],
         [-36, -3 * L, 36, -3 * L], [3 * L, -L * L, -3 * L, 4 * L * L]]) / (30 * L)
    return k, g


class Frame:
    def __init__(self, tl, sub=2):
        lifts, edges = tl['lifts'], tl['edges']
        self.sub = sub
        X = tl['X'] if 'X' in tl else lifts @ P.T  # a six-dimensional variant passes physical positions of Z^6 lifts
        V = len(X)
        self.cell = tl['cell']
        nodes = [X]
        ei, ej, wr, xa, dirs = [], [], [], [], []
        mid0 = V
        for e, (i, j, k, w1, w2) in enumerate(edges):
            d = DIRS[:, k]
            ids = [i] + [mid0 + e * (sub - 1) + h for h in range(sub - 1)] + [j]
            for h in range(sub):
                ei.append(ids[h]); ej.append(ids[h + 1])
                wr.append((w1, w2) if h == sub - 1 else (0, 0))
                xa.append(X[i] + d * h / sub); dirs.append(k)
        if sub > 1:
            hh = np.arange(1, sub) / sub
            mids = (X[edges[:, 0]][:, None, :] + hh[None, :, None] * DIRS[:, edges[:, 2]].T[:, None, :]).reshape(-1, 2)
            nodes.append(mids)
        self.X = np.vstack(nodes)
        self.ei = np.array(ei); self.ej = np.array(ej); self.wrap = np.array(wr, float)
        self.xa = np.array(xa); self.dirk = np.array(dirs)
        self.n = 3 * len(self.X)
        kl, gl = _beam_KG(1.0 / sub)
        self.Ke = np.zeros((6, 6, 6)); self.Ge = np.zeros((6, 6, 6))
        for k in range(6):
            Tm = _rot6(DIRS[0, k], DIRS[1, k])
            self.Ke[k] = Tm.T @ kl @ Tm; self.Ge[k] = Tm.T @ gl @ Tm
        di = 3 * self.ei[:, None] + np.arange(3)[None, :]
        dj = 3 * self.ej[:, None] + np.arange(3)[None, :]
        D = np.concatenate([di, dj], 1)
        self.rows = np.repeat(D, 6, axis=1).ravel()
        self.cols = np.tile(D, (1, 6)).ravel()
        blk = np.zeros((6, 6), int); blk[:3, 3:] = 1; blk[3:, :3] = -1
        self.blk = np.tile(blk.ravel(), len(self.ei))
        self.free_gamma = np.arange(2, self.n)

    def _assemble(self, coef, mats, k):
        ph = np.exp(2j * np.pi * (self.wrap @ np.asarray(k, float)))
        vals = (coef[:, None, None] * mats[self.dirk]).reshape(len(self.ei), 36).astype(complex)
        b = self.blk.reshape(-1, 36)
        phase = np.where(b == 1, ph[:, None], np.where(b == -1, np.conj(ph)[:, None], 1.0))
        M = sp.csr_matrix(((vals * phase).ravel(), (self.rows, self.cols)), shape=(self.n, self.n))
        return (M + M.conj().T) * 0.5

    def prestress(self, strain):
        ones = np.ones(len(self.ei))
        if not hasattr(self, '_lu0'):
            K0 = self._assemble(ones, self.Ke, (0, 0)).real.tocsc()
            self._K0 = K0
            fr = self.free_gamma
            self._lu0 = spla.splu(K0[fr][:, fr].tocsc(), permc_spec='COLAMD')
        K0 = self._K0
        xb = self.xa + DIRS[:, self.dirk].T / self.sub
        dimp = np.zeros((len(self.ei), 6))
        dimp[:, 0:2] = self.xa @ strain.T; dimp[:, 3:5] = xb @ strain.T
        fe = np.einsum('eab,eb->ea', self.Ke[self.dirk], dimp)
        f = np.zeros(self.n)
        np.add.at(f, (3 * self.ei[:, None] + np.arange(3)).ravel(), fe[:, :3].ravel())
        np.add.at(f, (3 * self.ej[:, None] + np.arange(3)).ravel(), fe[:, 3:].ravel())
        fr = self.free_gamma
        w = np.zeros(self.n)
        w[fr] = self._lu0.solve(-f[fr])
        d = dimp.copy()
        d[:, :3] += w[(3 * self.ei[:, None] + np.arange(3))]
        d[:, 3:] += w[(3 * self.ej[:, None] + np.arange(3))]
        e = DIRS[:, self.dirk].T
        self.N = EA * self.sub * np.einsum('ea,ea->e', e, d[:, 3:5] - d[:, 0:2])
        self.resid = float(np.linalg.norm(K0 @ w + f) / max(np.linalg.norm(f), 1e-30))
        fb = np.einsum('eab,eb->ea', self.Ke[self.dirk], d)[:, 3:5]
        delta = e / self.sub
        sig = 0.5 * (np.einsum('ea,eb->ab', fb, delta) + np.einsum('ea,eb->ab', delta, fb))
        self.stress = sig / abs(np.linalg.det(self.cell))
        return self.N

    def set_load(self):
        """macroscopic stress -I (unit pressure); multiplier = p_c / (EA lambda_E)."""
        basis = [np.array([[1., 0], [0, 0]]), np.array([[0., 0], [0, 1]]), np.array([[0, 1.], [1, 0]]) / math.sqrt(2)]
        Cm = np.zeros((3, 3))
        for a_, Bm in enumerate(basis):
            self.prestress(Bm)
            for b_, Bb in enumerate(basis):
                Cm[b_, a_] = np.sum(Bb * self.stress)
        c = la.solve(Cm, np.array([-1., -1., 0.]))
        eps = sum(ci * Bm for ci, Bm in zip(c, basis))
        self.C3 = Cm; self.eps = eps
        self.prestress(eps)
        self.pressure = float(-np.trace(self.stress) / 2)
        return eps

    def lam(self, k, nev=4):
        k = np.asarray(k, float)
        Km = self._assemble(np.ones(len(self.ei)), self.Ke, k).tocsc()
        H = -self._assemble(self.N, self.Ge, k).tocsc()
        gamma = np.allclose(k, np.rint(k))
        fr = self.free_gamma if gamma else np.arange(self.n)
        Km = Km[fr][:, fr]; H = H[fr][:, fr]
        if Km.shape[0] <= 600:
            vals, vec = la.eigh(H.toarray(), Km.toarray(), subset_by_index=(Km.shape[0] - 1, Km.shape[0] - 1), driver='gvx')
            mu = vals[-1]
        else:
            lu = spla.splu(Km.tocsc(), permc_spec='COLAMD')
            op = spla.LinearOperator(Km.shape, matvec=lambda x: lu.solve(H @ x), dtype=complex)
            vals = spla.eigs(op, k=nev, which='LR', tol=1e-10, maxiter=50000, return_eigenvectors=False)
            mu = float(np.max(vals.real))
        return 1 / mu / LAM_E if mu > 0 else np.inf

    def longwave(self, nphi=12, dqs=(0.002, 0.001)):
        phis = np.arange(nphi) * np.pi / nphi
        tab = {}
        for dq in dqs:
            tab[dq] = np.array([self.lam(dq * (self.cell @ np.array([math.cos(p), math.sin(p)]))) for p in phis])
        a, b = dqs[0], dqs[1]
        r = (a / b) ** 2
        ext = (r * tab[b] - tab[a]) / (r - 1)
        return phis, ext, tab


def harmonics(phis, lam, ms=(2, 4, 6)):
    """lam(phi) = c0 + sum_m Re[h_m e^{-i m phi}]."""
    cols = [np.ones_like(phis)]
    for m in ms:
        cols += [np.cos(m * phis), np.sin(m * phis)]
    A = np.array(cols).T
    c, *_ = np.linalg.lstsq(A, lam, rcond=None)
    out = dict(c0=float(c[0]), fit_rms=float(np.sqrt(np.mean((A @ c - lam) ** 2))))
    for t, m in enumerate(ms):
        out[f'h{m}'] = [float(c[1 + 2 * t]), float(c[2 + 2 * t])]
    return out

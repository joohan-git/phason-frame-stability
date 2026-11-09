"""Penrose (Z^5, C10) version of main/ab_frame.py: rational cuts of Z^5, hexagon-flip MC,
window excess, and the same 2D Euler-Bernoulli frame (rigid joints, unit edges, r = 0.03,
stress protocol, long-wave Richardson DQ -> 0).

Conventions
  * basis f_k of Z^5 with physical direction at angle k*pi/5 (k = 0..4); this is the usual
    Penrose star e_j (angles 2 pi j/5) after f = (e0, -e3, e1, -e4, e2), so every pair has a
    positive physical determinant (same convention as the AB code: all minors >= 0).
  * physical map P (2x5), in-plane internal map Q (angles 7 k pi/5) and diagonal row
    D = (-1)^k / sqrt2; P P^T = Qi Qi^T = 5/2 I, P Qi^T = 0.  C10 generator g (signed
    permutation f_k -> f_{k+1}, f_4 -> -f_0): P g = R(36) P, Q g = R(252) Q, D g = -D.
  * uniform phason strain M (3x2, internal = M physical); w = a z + b zbar (in-plane),
    v = M[2,0] + i M[2,1] (diagonal tilt).  Under g: a ~ e^{6 i th} (|m| = 4, E4),
    b ~ e^{8 i th} (|m| = 2, E2), v ~ -e^{i th} (|m| = 4, E4).
  * canonical window = projection of the unit 5-cube on the 3D complement of the slope plane;
    offset with sum (-1)^k c_k = 1/2 (Penrose layering: pentagonal sections) for slopes with
    D.L = 0.
"""
from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import itertools, math
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as la
from scipy.spatial import ConvexHull
import numba

NDIM = 5
TH = np.arange(NDIM) * math.pi / 5
P = np.array([np.cos(TH), np.sin(TH)])
Q = np.array([np.cos(7 * TH), np.sin(7 * TH)])
D = np.array([(-1.0) ** np.arange(NDIM)]) * math.sqrt(0.5)
QI = np.vstack([Q, D])
TAU = (1 + math.sqrt(5)) / 2
R_STRUT = 0.03
EA = 1.0
EI = R_STRUT ** 2 / 4
LAM_E = math.pi ** 2 * EI / EA
PAIRS = list(itertools.combinations(range(NDIM), 2))
G = np.zeros((5, 5), np.int64)
for _j in range(4):
    G[_j + 1, _j] = 1
G[0, 4] = -1


def minors(L1, L2):
    return {f"{i}{j}": int(L1[i] * L2[j] - L1[j] * L2[i]) for i, j in PAIRS}


def phason_matrix(L1, L2):
    L = np.array([L1, L2], float).T
    return (QI @ L) @ np.linalg.inv(P @ L)


def phason_parts(M):
    a = ((M[0, 0] + M[1, 1]) + 1j * (M[1, 0] - M[0, 1])) / 2
    b = ((M[0, 0] - M[1, 1]) + 1j * (M[1, 0] + M[0, 1])) / 2
    v = M[2, 0] + 1j * M[2, 1]
    return complex(a), complex(b), complex(v)


def M_from_parts(a, b, v):
    A = np.array([[a.real, -a.imag], [a.imag, a.real]]) + np.array([[b.real, b.imag], [b.imag, -b.real]])
    return np.vstack([A, [[v.real, v.imag]]])


def fib_approx(n):
    F = [1, 1]
    while len(F) < n + 3:
        F.append(F[-1] + F[-2])
    return [2 * F[n], F[n + 1], F[n - 1], -F[n - 1], -F[n + 1]], [0, F[n], F[n + 1], F[n + 1], F[n]]


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


OFFSET = np.array([0.1173, -0.2711, 0.0618 + 0.5, 0.3307, -0.0119])  # sum (-1)^k c_k = 0.5 + generic


def canonical(L1, L2, offset=None):
    L1 = np.array(L1, np.int64); L2 = np.array(L2, np.int64)
    m = minors(L1, L2)
    if all(v <= 0 for v in m.values()):
        L1, L2 = L2, L1; m = minors(L1, L2)
    assert all(v > 0 for v in m.values()), m
    Lm = np.array([L1, L2], float)
    q_, _ = np.linalg.qr(np.c_[Lm.T, np.eye(NDIM)])
    perp = q_[:, 2:NDIM].T  # 3 x 5 orthonormal complement
    cube = np.array(list(itertools.product([-.5, .5], repeat=NDIM)))
    eq = ConvexHull(cube @ perp.T).equations
    c = OFFSET if offset is None else np.asarray(offset, float)
    cs = np.array([1, -1, 1, -1, 1]) @ c
    c = c + (0.5 - (cs % 1.0)) / 5 * np.array([1, -1, 1, -1, 1]) if offset is None else c
    cperp = perp @ c
    red = Reducer(L1, L2)

    def inside(n):
        y = perp @ n - cperp
        return bool(np.all(eq[:, :3] @ y + eq[:, 3] < -1e-9))

    start = None
    for v in itertools.product(range(-2, 3), repeat=NDIM):
        if inside(np.array(v, float)):
            start = np.array(v, np.int64); break
    s0, _ = red(start)
    reps = {tuple(s0): 0}; order = [tuple(s0)]; stack = [tuple(s0)]
    E = np.eye(NDIM, dtype=np.int64)
    while stack:
        cur = np.array(stack.pop(), np.int64)
        for k in range(NDIM):
            for sgn in (1, -1):
                nb = cur + sgn * E[k]
                if inside(nb.astype(float)):
                    r, _ = red(nb); r = tuple(r)
                    if r not in reps:
                        reps[r] = len(order); order.append(r); stack.append(r)
    lifts = np.array(order, np.int64)
    assert len(lifts) == sum(m.values()), (len(lifts), sum(m.values()))
    return dict(L1=L1, L2=L2, minors=m, lifts=lifts, perp=perp, window_eq=eq, cperp=cperp, offset=c)


def edges_from_lifts(lifts, L1, L2):
    red = Reducer(L1, L2)
    look = {tuple(r): i for i, r in enumerate(lifts)}
    E = np.eye(NDIM, dtype=np.int64)
    edges = []
    for i, r in enumerate(lifts):
        for k in range(NDIM):
            rr, w = red(r + E[k])
            j = look.get(tuple(rr))
            if j is not None:
                edges.append((i, j, k, w[0], w[1]))
    edges = np.array(edges, np.int64)
    assert len(edges) == 2 * len(lifts), (len(edges), len(lifts))
    return edges


def check_tiling(lifts, L1, L2):
    red = Reducer(L1, L2)
    look = {tuple(r) for r in lifts}
    E = np.eye(NDIM, dtype=np.int64)
    bad = 0
    for r in lifts:
        ds = []
        for k in range(NDIM):
            for s in (1, -1):
                d = s * E[k]
                if tuple(red(r + d)[0]) in look:
                    ds.append(d)
        ang = sorted((math.atan2(*(P @ d)[::-1]) % (2 * math.pi), tuple(d)) for d in ds)
        if len(ang) < 3:
            bad += 1; continue
        for (a1, d1), (a2, d2) in zip(ang, ang[1:] + [(ang[0][0] + 2 * math.pi, ang[0][1])]):
            if a2 - a1 >= math.pi - 1e-9:
                bad += 1; break
            if tuple(red(r + np.array(d1) + np.array(d2))[0]) not in look:
                bad += 1; break
    return bad


def tile_counts(lifts, L1, L2):
    """Number of thin (36 deg) and thick (72 deg) rhombi from the edge-pair faces."""
    red = Reducer(L1, L2); look = {tuple(r) for r in lifts}
    E = np.eye(NDIM, dtype=np.int64); cnt = {}
    for r in lifts:
        for i, j in PAIRS:
            if all(tuple(red(r + d)[0]) in look for d in (E[i], E[j], E[i] + E[j])):
                cnt[(i, j)] = cnt.get((i, j), 0) + 1
    thin = sum(v for (i, j), v in cnt.items() if (j - i) in (1, 4))
    thick = sum(v for (i, j), v in cnt.items() if (j - i) in (2, 3))
    return thin, thick


# ----------------------------------------------------------------- flip MC (numba)
KB = 11
OFF = 1 << (KB - 1)


@numba.njit(cache=True)
def _key(r):
    k = 0
    for c in range(5):
        k = (k << KB) | (r[c] + OFF)
    return k


@numba.njit(cache=True)
def _reduce(n, L1, L2, adj, det, out):
    r1 = 0; r2 = 0
    for c in range(5):
        r1 += n[c] * L1[c]; r2 += n[c] * L2[c]
    a = (adj[0] * r1 + adj[1] * r2) // det
    b = (adj[2] * r1 + adj[3] * r2) // det
    for c in range(5):
        out[c] = n[c] - a * L1[c] - b * L2[c]


@numba.njit(cache=True)
def _hfind(keys, k):
    mask = keys.shape[0] - 1
    h = ((k * 0x9E3779B97F4A7C15) >> 20) & mask
    while True:
        kk = keys[h]
        if kk == k:
            return h
        if kk == -1:
            return -1
        h = (h + 1) & mask


@numba.njit(cache=True)
def _hins(keys, vals, k, v):
    mask = keys.shape[0] - 1
    h = ((k * 0x9E3779B97F4A7C15) >> 20) & mask
    while keys[h] != -1 and keys[h] != -2:
        h = (h + 1) & mask
    keys[h] = k; vals[h] = v


@numba.njit(cache=True)
def _present(keys, n, L1, L2, adj, det, tmp):
    _reduce(n, L1, L2, adj, det, tmp)
    return _hfind(keys, _key(tmp)) >= 0


@numba.njit(cache=True)
def _mc(lifts, keys, vals, L1, L2, adj, det, Px, Py, nattempt, seed, maxacc, cx, cy, cellinv, cell, rmove):
    """Uniform hexagon-flip MC: a vertex with exactly three neighbours whose edge vectors
    surround it (convex hexagon of three rhombi) moves to r + d0 + d1 + d2."""
    np.random.seed(seed)
    V = lifts.shape[0]
    tmp = np.zeros(5, np.int64); t2 = np.zeros(5, np.int64); r = np.zeros(5, np.int64)
    ds = np.zeros((3, 5), np.int64); new = np.zeros(5, np.int64)
    acc = 0
    for it in range(nattempt):
        v = np.random.randint(V)
        for c in range(5):
            r[c] = lifts[v, c]
        if rmove > 0.0:
            vx0 = 0.0; vy0 = 0.0
            for c in range(5):
                vx0 += Px[c] * r[c]; vy0 += Py[c] * r[c]
            ddx = vx0 - cx; ddy = vy0 - cy
            f0 = cellinv[0, 0] * ddx + cellinv[1, 0] * ddy
            f1 = cellinv[0, 1] * ddx + cellinv[1, 1] * ddy
            f0 -= np.floor(f0 + 0.5); f1 -= np.floor(f1 + 0.5)
            mx = f0 * cell[0, 0] + f1 * cell[1, 0]; my = f0 * cell[0, 1] + f1 * cell[1, 1]
            if mx * mx + my * my > rmove * rmove:
                continue
        nd = 0
        for k in range(5):
            for s in (1, -1):
                for c in range(5):
                    tmp[c] = r[c]
                tmp[k] += s
                if _present(keys, tmp, L1, L2, adj, det, t2):
                    if nd < 3:
                        for c in range(5):
                            ds[nd, c] = 0
                        ds[nd, k] = s
                    nd += 1
        if nd != 3:
            continue
        cr = np.zeros(3)
        for a in range(3):
            b = (a + 1) % 3
            ax = 0.0; ay = 0.0; bx = 0.0; by = 0.0
            for c in range(5):
                ax += Px[c] * ds[a, c]; ay += Py[c] * ds[a, c]
                bx += Px[c] * ds[b, c]; by += Py[c] * ds[b, c]
            cr[a] = ax * by - ay * bx
        if not ((cr[0] > 1e-9 and cr[1] > 1e-9 and cr[2] > 1e-9) or (cr[0] < -1e-9 and cr[1] < -1e-9 and cr[2] < -1e-9)):
            continue
        ok = True
        for a in range(3):
            for b in range(a + 1, 3):
                for c in range(5):
                    tmp[c] = r[c] + ds[a, c] + ds[b, c]
                if not _present(keys, tmp, L1, L2, adj, det, t2):
                    ok = False
        if not ok:
            continue
        for c in range(5):
            tmp[c] = r[c] + ds[0, c] + ds[1, c] + ds[2, c]
        _reduce(tmp, L1, L2, adj, det, new)
        if _hfind(keys, _key(new)) >= 0:
            continue
        cnt = 0
        for k in range(5):
            for s in (1, -1):
                for c in range(5):
                    tmp[c] = new[c]
                tmp[k] += s
                if _present(keys, tmp, L1, L2, adj, det, t2):
                    cnt += 1
        if cnt != 3:
            continue
        h = _hfind(keys, _key(r))
        keys[h] = -2
        _hins(keys, vals, _key(new), v)
        for c in range(5):
            lifts[v, c] = new[c]
        acc += 1
        if maxacc > 0 and acc >= maxacc:
            break
    return acc


def flip_mc(lifts, L1, L2, nattempt, seed, maxacc=0, center=(0.0, 0.0), rmove=0.0):
    """rmove > 0 restricts attempted moves to vertices within physical distance rmove of center."""
    lifts = np.array(lifts, np.int64).copy()
    V = len(lifts)
    size = 1 << int(math.ceil(math.log2(8 * V)))
    keys = -np.ones(size, np.int64); vals = np.zeros(size, np.int64)
    for i, r in enumerate(lifts):
        _hins(keys, vals, _key(np.array(r, np.int64)), i)
    red = Reducer(L1, L2)
    cell = np.array([P @ np.array(L1, float), P @ np.array(L2, float)])
    acc = _mc(lifts, keys, vals, np.array(L1, np.int64), np.array(L2, np.int64),
              np.array(red.adj, np.int64), np.int64(red.det), P[0].copy(), P[1].copy(),
              int(nattempt), int(seed), int(maxacc), float(center[0]), float(center[1]),
              np.linalg.inv(cell), cell, float(rmove))
    return lifts, int(acc)


def window_excess(lifts, tl):
    """Signed Euclidean distance (3D internal space) outside the canonical window (<= 0 inside)."""
    y = lifts @ tl['perp'].T - tl['cperp']
    eq = tl['window_eq']
    return np.max(y @ eq[:, :3].T + eq[:, 3], axis=1)


# ----------------------------------------------------------------- frame mechanics
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
    """Identical element/prestress/eigen machinery to ab_frame.Frame, with NDIM = 5 edge directions."""

    def __init__(self, lifts, L1, L2, sub=2, edges=None):
        self.L1, self.L2, self.sub = np.array(L1), np.array(L2), sub
        if edges is None:
            edges = edges_from_lifts(lifts, L1, L2)
        X = lifts @ P.T
        V = len(X)
        self.cell = np.array([P @ np.array(L1, float), P @ np.array(L2, float)])
        nodes = [X]
        ei, ej, wr, xa, dirs = [], [], [], [], []
        mid0 = V
        for e, (i, j, k, w1, w2) in enumerate(edges):
            d = P[:, k]
            ids = [i] + [mid0 + e * (sub - 1) + h for h in range(sub - 1)] + [j]
            for h in range(sub):
                ei.append(ids[h]); ej.append(ids[h + 1])
                wr.append((w1, w2) if h == sub - 1 else (0, 0))
                xa.append(X[i] + d * h / sub); dirs.append(k)
        if sub > 1:
            hh = np.arange(1, sub) / sub
            mids = (X[edges[:, 0]][:, None, :] + hh[None, :, None] * P[:, edges[:, 2]].T[:, None, :]).reshape(-1, 2)
            nodes.append(mids)
        self.X = np.vstack(nodes)
        self.ei = np.array(ei); self.ej = np.array(ej); self.wrap = np.array(wr, float)
        self.xa = np.array(xa); self.dirk = np.array(dirs)
        self.n = 3 * len(self.X)
        Le = 1.0 / sub
        kl, gl = _beam_KG(Le)
        self.Ke = np.zeros((NDIM, 6, 6)); self.Ge = np.zeros((NDIM, 6, 6))
        for k in range(NDIM):
            Tm = _rot6(P[0, k], P[1, k])
            self.Ke[k] = Tm.T @ kl @ Tm; self.Ge[k] = Tm.T @ gl @ Tm
        di = 3 * self.ei[:, None] + np.arange(3)[None, :]
        dj = 3 * self.ej[:, None] + np.arange(3)[None, :]
        Dd = np.concatenate([di, dj], 1)
        self.rows = np.repeat(Dd, 6, axis=1).ravel()
        self.cols = np.tile(Dd, (1, 6)).ravel()
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

    def prestress(self, strain=-np.eye(2)):
        ones = np.ones(len(self.ei))
        K0 = self._assemble(ones, self.Ke, (0, 0)).real.tocsc()
        xb = self.xa + P[:, self.dirk].T / self.sub
        dimp = np.zeros((len(self.ei), 6))
        dimp[:, 0:2] = self.xa @ strain.T; dimp[:, 3:5] = xb @ strain.T
        fe = np.einsum('eab,eb->ea', self.Ke[self.dirk], dimp)
        f = np.zeros(self.n)
        np.add.at(f, (3 * self.ei[:, None] + np.arange(3)).ravel(), fe[:, :3].ravel())
        np.add.at(f, (3 * self.ej[:, None] + np.arange(3)).ravel(), fe[:, 3:].ravel())
        fr = self.free_gamma
        w = np.zeros(self.n)
        if not hasattr(self, '_lu0'):
            self._lu0 = spla.splu(K0[fr][:, fr].tocsc(), permc_spec='COLAMD')
        w[fr] = self._lu0.solve(-f[fr])
        d = dimp.copy()
        d[:, :3] += w[(3 * self.ei[:, None] + np.arange(3))]
        d[:, 3:] += w[(3 * self.ej[:, None] + np.arange(3))]
        e = P[:, self.dirk].T
        self.N = EA * self.sub * np.einsum('ea,ea->e', e, d[:, 3:5] - d[:, 0:2])
        self.resid = float(np.linalg.norm(K0 @ w + f) / max(np.linalg.norm(f), 1e-30))
        fb = np.einsum('eab,eb->ea', self.Ke[self.dirk], d)[:, 3:5]
        delta = e / self.sub
        sig = 0.5 * (np.einsum('ea,eb->ab', fb, delta) + np.einsum('ea,eb->ab', delta, fb))
        self.stress = sig / abs(np.linalg.det(self.cell))
        return self.N

    def set_load(self, protocol='stress'):
        if protocol == 'strain':
            eps = -np.eye(2)
        else:
            basis = [np.array([[1., 0], [0, 0]]), np.array([[0., 0], [0, 1]]), np.array([[0, 1.], [1, 0]]) / math.sqrt(2)]
            Cm = np.zeros((3, 3))
            for a, B in enumerate(basis):
                self.prestress(B)
                for b, Bb in enumerate(basis):
                    Cm[b, a] = np.sum(Bb * self.stress)
            c = la.solve(Cm, np.array([-1., -1., 0.]))
            eps = sum(ci * B for ci, B in zip(c, basis))
            self.C3 = Cm
        self.eps = eps
        self.prestress(eps)
        self.pressure = float(-np.trace(self.stress) / 2)
        return eps

    def lam(self, k, nev=4, vectors=False):
        k = np.asarray(k, float)
        K = self._assemble(np.ones(len(self.ei)), self.Ke, k).tocsc()
        H = -self._assemble(self.N, self.Ge, k).tocsc()
        gamma = np.allclose(k, np.rint(k))
        fr = self.free_gamma if gamma else np.arange(self.n)
        K = K[fr][:, fr]; H = H[fr][:, fr]
        if K.shape[0] <= 1200:
            vals, vec = la.eigh(H.toarray(), K.toarray(), subset_by_index=(K.shape[0] - 1, K.shape[0] - 1), driver='gvx')
            mu = vals[-1]; v = vec[:, -1]
        else:
            lu = spla.splu(K.tocsc(), permc_spec='COLAMD')
            op = spla.LinearOperator(K.shape, matvec=lambda x: lu.solve(H @ x), dtype=complex)
            vals, vecs = spla.eigs(op, k=nev, which='LR', tol=1e-10, maxiter=50000)
            i = int(np.argmax(vals.real)); mu = vals[i].real; v = vecs[:, i]
        lamv = 1 / mu / LAM_E if mu > 0 else np.inf
        if vectors:
            full = np.zeros(self.n, complex); full[fr] = v
            return lamv, full
        return lamv

    def longwave(self, nphi=8, dqs=(0.002, 0.001)):
        phis = np.arange(nphi) * np.pi / nphi
        tab = {}
        for dq in dqs:
            tab[dq] = np.array([self.lam(dq * (self.cell @ np.array([math.cos(p), math.sin(p)]))) for p in phis])
        a, b = dqs
        r = (a / b) ** 2
        ext = (r * tab[b] - tab[a]) / (r - 1)
        return phis, ext, tab


def harmonics(phis, lam):
    """lam(phi) = c0 + Re[h2 e^{-2i phi}] + Re[h4 e^{-4i phi}] (period pi), least squares."""
    cols = [np.ones_like(phis)]
    for m in (2, 4):
        cols += [np.cos(m * phis), np.sin(m * phis)]
    A = np.array(cols).T
    c, *_ = np.linalg.lstsq(A, lam, rcond=None)
    h2 = c[1] + 1j * c[2]; h4 = c[3] + 1j * c[4]
    fine = np.linspace(0, np.pi, 721)
    model = c[0] + np.real(h2 * np.exp(-2j * fine)) + np.real(h4 * np.exp(-4j * fine))
    return dict(c0=float(c[0]), h2=[float(h2.real), float(h2.imag)], h4=[float(h4.real), float(h4.imag)],
                fit_rms=float(np.sqrt(np.mean((A @ c - lam) ** 2))), min_model=float(model.min()),
                phi_min_model_deg=float(np.degrees(fine[np.argmin(model)])))

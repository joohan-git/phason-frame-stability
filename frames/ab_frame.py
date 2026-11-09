"""Core: AB-edge rhombus tilings on Z^4 slope space, periodic flip Monte Carlo,
phason diagnostics, and a 2D Euler-Bernoulli frame buckling solver (strain protocol).

Conventions (same as slope_thresholds.py, r1):
  * physical star P (AB, unit edges at 0/45/90/135 deg); AB internal map Q (P P^T = Q Q^T = 2 I, P Q^T = 0).
  * a periodic tiling = reduced integer lifts (mod Z L1 + Z L2); adjacency = unit lift steps.
  * frame: EA = 1, EI = r^2/4, r = 0.03, each unit edge split in `sub` elements, rigid joints.
  * prestress: periodic linear solve under macroscopic strain -I (strain protocol).
  * long-wave threshold lambda(phi): smallest critical multiplier at physical wavevector
    q = 2 pi DQ (cos phi, sin phi); Richardson DQ -> 0 from DQ = 0.002, 0.001 (error ~ DQ^2).
  * all thresholds reported / lambda_E, lambda_E = pi^2 EI / (EA L^2).
Optional numba (flip MC); falls back to pure Python.
"""
from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import itertools, math, time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as la
from scipy.spatial import ConvexHull

try:
    import numba
    njit = numba.njit(cache=True)
    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    HAVE_NUMBA = False
    def njit(f):
        return f

T = 1 / math.sqrt(2)
P = np.array([[1, T, 0, -T], [0, T, 1, T]])
Q = np.array([[1, -T, 0, T], [0, T, -1, T]])
R_STRUT = 0.03
EA = 1.0
EI = R_STRUT ** 2 / 4
LAM_E = math.pi ** 2 * EI / EA
PAIRS = list(itertools.combinations(range(4), 2))


# ----------------------------------------------------------------- geometry
def minors(L1, L2):
    return {f"{i+1}{j+1}": int(L1[i] * L2[j] - L1[j] * L2[i]) for i, j in PAIRS}


def phason_matrix(L1, L2):
    """Uniform phason strain M (internal = M physical), relative to the AB internal map Q."""
    L = np.array([L1, L2], float).T
    return (Q @ L) @ np.linalg.inv(P @ L)


def harmonic_parts(M):
    """w = a z + b zbar : a (m=2, E2) and b (m=4, B1+iB2) under C8 (phys m=1, int m=3)."""
    a = ((M[0, 0] + M[1, 1]) + 1j * (M[1, 0] - M[0, 1])) / 2
    b = ((M[0, 0] - M[1, 1]) + 1j * (M[1, 0] + M[0, 1])) / 2
    return complex(a), complex(b)


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


def canonical(L1, L2, seed=7):
    """Canonical cut tiling for slope span(L1, L2); returns dict with reduced lifts."""
    L1 = np.array(L1, np.int64); L2 = np.array(L2, np.int64)
    m = minors(L1, L2)
    if all(v <= 0 for v in m.values()):
        L1, L2 = L2, L1; m = minors(L1, L2)
    assert all(v >= 0 for v in m.values()), m
    Lm = np.array([L1, L2], float)
    q_, _ = np.linalg.qr(np.c_[Lm.T, np.eye(4)])
    perp = q_[:, 2:4].T
    cube = np.array(list(itertools.product([-.5, .5], repeat=4)))
    eq = ConvexHull(cube @ perp.T).equations
    rng = np.random.default_rng(seed)
    c = rng.uniform(-.5, .5, 4) * 0.37 + np.array([0.1173, 0.2711, -0.0618, 0.3307])
    cperp = perp @ c
    red = Reducer(L1, L2)

    def inside(n):
        y = perp @ n - cperp
        return bool(np.all(eq[:, :2] @ y + eq[:, 2] < -1e-9))

    start = None
    for v in itertools.product(range(-2, 3), repeat=4):
        if inside(np.array(v, float)):
            start = np.array(v, np.int64); break
    s0, _ = red(start)
    reps = {tuple(s0): 0}; order = [tuple(s0)]; stack = [tuple(s0)]
    E = np.eye(4, dtype=np.int64)
    while stack:
        cur = np.array(stack.pop(), np.int64)
        for k in range(4):
            for sgn in (1, -1):
                nb = cur + sgn * E[k]
                if inside(nb.astype(float)):
                    r, _ = red(nb); r = tuple(r)
                    if r not in reps:
                        reps[r] = len(order); order.append(r); stack.append(r)
    lifts = np.array(order, np.int64)
    assert len(lifts) == sum(m.values()), (len(lifts), m)
    return dict(L1=L1, L2=L2, minors=m, lifts=lifts, perp=perp, window_eq=eq, cperp=cperp)


def edges_from_lifts(lifts, L1, L2):
    red = Reducer(L1, L2)
    look = {tuple(r): i for i, r in enumerate(lifts)}
    E = np.eye(4, dtype=np.int64)
    edges = []
    for i, r in enumerate(lifts):
        for k in range(4):
            rr, w = red(r + E[k])
            j = look.get(tuple(rr))
            if j is not None:
                edges.append((i, j, k, w[0], w[1]))
    edges = np.array(edges, np.int64)
    assert len(edges) == 2 * len(lifts), (len(edges), len(lifts))
    return edges


def check_tiling(lifts, L1, L2):
    """Local validity: every vertex star has consecutive angle gaps < pi and every
    consecutive edge pair closes a rhombus; E = 2V."""
    red = Reducer(L1, L2)
    look = {tuple(r) for r in lifts}
    E = np.eye(4, dtype=np.int64)
    bad = 0
    for r in lifts:
        ds = []
        for k in range(4):
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


# ----------------------------------------------------------------- flip Monte Carlo (numba)
KB = 11
OFF = 1 << (KB - 1)


@njit
def _key(n0, n1, n2, n3):
    return ((((n0 + OFF) << KB | (n1 + OFF)) << KB | (n2 + OFF)) << KB) | (n3 + OFF)


@njit
def _reduce(n, L1, L2, adj, det, out):
    r1 = n[0] * L1[0] + n[1] * L1[1] + n[2] * L1[2] + n[3] * L1[3]
    r2 = n[0] * L2[0] + n[1] * L2[1] + n[2] * L2[2] + n[3] * L2[3]
    a = (adj[0] * r1 + adj[1] * r2) // det
    b = (adj[2] * r1 + adj[3] * r2) // det
    for c in range(4):
        out[c] = n[c] - a * L1[c] - b * L2[c]


@njit
def _hfind(keys, k):
    mask = keys.shape[0] - 1
    h = (k * 0x9E3779B97F4A7C15) >> 20
    h = h & mask
    while True:
        kk = keys[h]
        if kk == k:
            return h
        if kk == -1:
            return -1
        h = (h + 1) & mask


@njit
def _hins(keys, vals, k, v):
    mask = keys.shape[0] - 1
    h = (k * 0x9E3779B97F4A7C15) >> 20
    h = h & mask
    while keys[h] != -1 and keys[h] != -2:
        h = (h + 1) & mask
    keys[h] = k; vals[h] = v


@njit
def _lookup(keys, vals, r):
    h = _hfind(keys, _key(r[0], r[1], r[2], r[3]))
    return -1 if h < 0 else vals[h]


@njit
def _mc(lifts, keys, vals, L1, L2, adj, det, Px, Py, nattempt, seed,
        bias, cx, cy, cellinv, cell, radius, maxacc, rmove):
    """Uniform flip MC (bias = 0) or biased toward removing e2/e4 edges inside a disk
    (bias > 0, Metropolis weight exp(-bias * dE)). Returns accepted count."""
    np.random.seed(seed)
    V = lifts.shape[0]
    tmp = np.zeros(4, np.int64); r = np.zeros(4, np.int64)
    ds = np.zeros((3, 4), np.int64); new = np.zeros(4, np.int64)
    acc = 0
    for it in range(nattempt):
        v = np.random.randint(V)
        for c in range(4):
            r[c] = lifts[v, c]
        if rmove > 0.0:
            vx0 = 0.0; vy0 = 0.0
            for c in range(4):
                vx0 += Px[c] * r[c]; vy0 += Py[c] * r[c]
            ddx = vx0 - cx; ddy = vy0 - cy
            f0 = cellinv[0, 0] * ddx + cellinv[1, 0] * ddy
            f1 = cellinv[0, 1] * ddx + cellinv[1, 1] * ddy
            f0 -= np.floor(f0 + 0.5); f1 -= np.floor(f1 + 0.5)
            mx = f0 * cell[0, 0] + f1 * cell[1, 0]; my = f0 * cell[0, 1] + f1 * cell[1, 1]
            if mx * mx + my * my > rmove * rmove:
                continue
        nd = 0
        for k in range(4):
            for s in (1, -1):
                for c in range(4):
                    tmp[c] = r[c]
                tmp[k] += s
                _reduce(tmp.copy(), L1, L2, adj, det, tmp)
                if _lookup(keys, vals, tmp) >= 0:
                    if nd < 3:
                        for c in range(4):
                            ds[nd, c] = 0
                        ds[nd, k] = s
                    nd += 1
        if nd != 3:
            continue
        # convexity: origin strictly inside triangle of physical d's
        cr = np.zeros(3)
        for a in range(3):
            b = (a + 1) % 3
            ax = 0.0; ay = 0.0; bx = 0.0; by = 0.0
            for c in range(4):
                ax += Px[c] * ds[a, c]; ay += Py[c] * ds[a, c]
                bx += Px[c] * ds[b, c]; by += Py[c] * ds[b, c]
            cr[a] = ax * by - ay * bx
        if not ((cr[0] > 1e-9 and cr[1] > 1e-9 and cr[2] > 1e-9) or (cr[0] < -1e-9 and cr[1] < -1e-9 and cr[2] < -1e-9)):
            continue
        ok = True
        for a in range(3):
            for b in range(a + 1, 3):
                for c in range(4):
                    tmp[c] = r[c] + ds[a, c] + ds[b, c]
                _reduce(tmp.copy(), L1, L2, adj, det, tmp)
                if _lookup(keys, vals, tmp) < 0:
                    ok = False
        if not ok:
            continue
        for c in range(4):
            new[c] = r[c] + ds[0, c] + ds[1, c] + ds[2, c]
        _reduce(new.copy(), L1, L2, adj, det, new)
        if _lookup(keys, vals, new) >= 0:
            continue
        # new vertex must have exactly three unit-step neighbours (the corners)
        cnt = 0
        for k in range(4):
            for s in (1, -1):
                for c in range(4):
                    tmp[c] = new[c]
                tmp[k] += s
                _reduce(tmp.copy(), L1, L2, adj, det, tmp)
                if _lookup(keys, vals, tmp) >= 0:
                    cnt += 1
        if cnt != 3:
            continue
        if bias > 0.0:
            # energy = number of e2/e4 edge midpoints inside the disk; a flip moves the three
            # edges at v (same axes) from x_v + d/2 to x_new - d/2 = x_v + S - d/2.
            sx = 0.0; sy = 0.0; vx = 0.0; vy = 0.0
            for c in range(4):
                vx += Px[c] * r[c]; vy += Py[c] * r[c]
                for a in range(3):
                    sx += Px[c] * ds[a, c]; sy += Py[c] * ds[a, c]
            dE = 0.0
            for a in range(3):
                if ds[a, 1] == 0 and ds[a, 3] == 0:
                    continue
                dx_ = 0.0; dy_ = 0.0
                for c in range(4):
                    dx_ += Px[c] * ds[a, c]; dy_ += Py[c] * ds[a, c]
                for which in range(2):
                    if which == 0:
                        x = vx + 0.5 * dx_; y = vy + 0.5 * dy_
                    else:
                        x = vx + sx - 0.5 * dx_; y = vy + sy - 0.5 * dy_
                    ddx = x - cx; ddy = y - cy
                    f0 = cellinv[0, 0] * ddx + cellinv[1, 0] * ddy
                    f1 = cellinv[0, 1] * ddx + cellinv[1, 1] * ddy
                    f0 -= np.floor(f0 + 0.5); f1 -= np.floor(f1 + 0.5)
                    mx = f0 * cell[0, 0] + f1 * cell[1, 0]; my = f0 * cell[0, 1] + f1 * cell[1, 1]
                    if mx * mx + my * my < radius * radius:
                        dE += -1.0 if which == 0 else 1.0
            if dE > 0 and np.random.random() > math.exp(-bias * dE):
                continue
        # apply
        h = _hfind(keys, _key(r[0], r[1], r[2], r[3]))
        keys[h] = -2
        _hins(keys, vals, _key(new[0], new[1], new[2], new[3]), v)
        for c in range(4):
            lifts[v, c] = new[c]
        acc += 1
        if maxacc > 0 and acc >= maxacc:
            break
    return acc


def flip_mc(lifts, L1, L2, nattempt, seed, bias=0.0, center=(0.0, 0.0), radius=0.0, maxacc=0, rmove=0.0):
    lifts = np.array(lifts, np.int64).copy()
    V = len(lifts)
    size = 1 << int(math.ceil(math.log2(8 * V)))
    keys = -np.ones(size, np.int64); vals = np.zeros(size, np.int64)
    for i, r in enumerate(lifts):
        _hins(keys, vals, _key(*[int(x) for x in r]), i)
    red = Reducer(L1, L2)
    cell = np.array([P @ np.array(L1, float), P @ np.array(L2, float)])
    acc = _mc(lifts, keys, vals, np.array(L1, np.int64), np.array(L2, np.int64),
              np.array(red.adj, np.int64), np.int64(red.det), P[0].copy(), P[1].copy(),
              int(nattempt), int(seed), float(bias), float(center[0]), float(center[1]),
              np.linalg.inv(cell), cell, float(radius), int(maxacc), float(rmove))
    return lifts, int(acc)


# ----------------------------------------------------------------- phason diagnostics
def phason_stats(lifts, L1, L2, perp, ref_lifts=None, radii=(4.0,)):
    from scipy.spatial import cKDTree
    X = lifts @ P.T
    cell = np.array([P @ np.array(L1, float), P @ np.array(L2, float)])
    y = lifts @ perp.T  # internal coordinates relative to the slope plane (reduction-invariant)
    out = dict(perp_var=float(np.mean(np.sum((y - y.mean(0)) ** 2, 1))))
    if ref_lifts is not None:
        ref = {tuple(r) for r in ref_lifts}
        out['moved_fraction'] = float(np.mean([tuple(r) not in ref for r in lifts]))
    orth = abs(cell[0, 1]) < 1e-9 and abs(cell[1, 0]) < 1e-9
    if orth:
        box = np.array([cell[0, 0], cell[1, 1]])
        Xm = np.mod(X, box); Xm[Xm >= box] = 0.0
        tree = cKDTree(Xm, boxsize=box)
        for R in radii:
            G2 = []
            for i, nb in enumerate(tree.query_ball_point(Xm, R)):
                d = Xm[nb] - Xm[i]; d -= box * np.round(d / box)
                A = np.c_[d, np.ones(len(nb))]
                coef, *_ = np.linalg.lstsq(A, y[nb], rcond=None)
                G2.append(np.sum(coef[:2] ** 2))
            out[f'local_phason_strain_ms_R{R:g}'] = float(np.mean(G2))
    return out


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
    def __init__(self, lifts, L1, L2, sub=2, edges=None):
        self.L1, self.L2, self.sub = np.array(L1), np.array(L2), sub
        if edges is None:
            edges = edges_from_lifts(lifts, L1, L2)
        X = lifts @ P.T
        V = len(X); nE = len(edges)
        self.cell = np.array([P @ np.array(L1, float), P @ np.array(L2, float)])
        nodes = [X]
        # element connectivity
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
        self.Ke = np.zeros((4, 6, 6)); self.Ge = np.zeros((4, 6, 6))
        for k in range(4):
            Tm = _rot6(P[0, k], P[1, k])
            self.Ke[k] = Tm.T @ kl @ Tm; self.Ge[k] = Tm.T @ gl @ Tm
        di = 3 * self.ei[:, None] + np.arange(3)[None, :]
        dj = 3 * self.ej[:, None] + np.arange(3)[None, :]
        D = np.concatenate([di, dj], 1)  # (ne, 6)
        self.rows = np.repeat(D, 6, axis=1).ravel()
        self.cols = np.tile(D, (1, 6)).ravel()
        blk = np.zeros((6, 6), int); blk[:3, 3:] = 1; blk[3:, :3] = -1
        self.blk = np.tile(blk.ravel(), len(self.ei))
        self.free_gamma = np.arange(2, self.n)

    def _assemble(self, coef, mats, k):
        ph = np.exp(2j * np.pi * (self.wrap @ np.asarray(k, float)))
        vals = (coef[:, None, None] * mats[self.dirk]).reshape(len(self.ei), 36).astype(complex)
        phase = np.where(self.blk.reshape(-1, 36) == 1, ph[:, None], np.where(self.blk.reshape(-1, 36) == -1, np.conj(ph)[:, None], 1.0))
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
        """'strain': macroscopic strain -I.  'stress': macroscopic stress -I (unit pressure,
        EA = 1 per unit length), so the returned multiplier is p_c / (EA lambda_E)."""
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
        if K.shape[0] <= 2500:
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

    def longwave(self, nphi=12, dqs=(0.002, 0.001)):
        phis = np.arange(nphi) * np.pi / nphi
        tab = {}
        for dq in dqs:
            tab[dq] = np.array([self.lam(dq * (self.cell @ np.array([math.cos(p), math.sin(p)]))) for p in phis])
        if len(dqs) == 2:
            a, b = dqs  # Richardson for error ~ DQ^2
            r = (a / b) ** 2
            ext = (r * tab[b] - tab[a]) / (r - 1)
        else:
            ext = tab[dqs[0]]
        return phis, ext, tab


def harmonics(phis, lam):
    """Fit lam(phi) = c0 + sum_m Re[h_m e^{-i m phi}], m = 2, 4 (period pi)."""
    cols = [np.ones_like(phis)]
    for m in (2, 4):
        cols += [np.cos(m * phis), np.sin(m * phis)]
    A = np.array(cols).T
    c, res, *_ = np.linalg.lstsq(A, lam, rcond=None)
    h2 = c[1] + 1j * c[2]; h4 = c[3] + 1j * c[4]   # Re[h e^{-im phi}] = Re h cos + Im h sin
    fine = np.linspace(0, np.pi, 721)
    model = c[0] + np.real(h2 * np.exp(-2j * fine)) + np.real(h4 * np.exp(-4j * fine))
    return dict(c0=float(c[0]), h2=[float(h2.real), float(h2.imag)], h4=[float(h4.real), float(h4.imag)],
                fit_rms=float(np.sqrt(np.mean((A @ c - lam) ** 2))), min_model=float(model.min()),
                phi_min_model_deg=float(np.degrees(fine[np.argmin(model)])))


def ab_approx(p, q, m=1):
    return [m * q, m * p, 0, -m * p], [0, m * p, m * q, m * p]


def disk_square_fraction(lifts, L1, L2, center, radius):
    """Fraction of vertices within the disk whose star uses only axes e1, e3 (pure square-13 lattice)."""
    red = Reducer(L1, L2)
    look = {tuple(r) for r in lifts}
    cell = np.array([P @ np.array(L1, float), P @ np.array(L2, float)])
    ci = np.linalg.inv(cell)
    E = np.eye(4, dtype=np.int64)
    n_in = n_sq = 0
    for r in lifts:
        x = P @ r - np.asarray(center)
        f = x @ ci; f -= np.floor(f + 0.5); x = f @ cell
        if x @ x >= radius ** 2:
            continue
        n_in += 1
        axes = {k for k in range(4) for s in (1, -1) if tuple(red(r + s * E[k])[0]) in look}
        n_sq += axes <= {0, 2}
    return n_in, n_sq


def window_excess(lifts, tl):
    """Signed distance of each vertex's internal coordinate outside the canonical window
    (<= 0 inside). tl = canonical(...) dict of the same slope."""
    y = lifts @ tl['perp'].T - tl['cperp']
    eq = tl['window_eq']
    return np.max(y @ eq[:, :2].T + eq[:, 2], axis=1)

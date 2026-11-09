r"""3D icosahedral (Ammann-Kramer-Neri) rhombohedral tilings as cuts of Z^6,
rational (cubic / tetragonal) approximants, 3D rhombic-dodecahedron flip MC, and a 3D
rigid-joint Euler-Bernoulli frame (6 dof/node, torsion) with the stress protocol of
..\main\ab_frame.py (unit hydrostatic stress, long-wave Richardson DQ -> 0).

Conventions
  * star e_k (k = 0..5) = (1,t,0),(t,0,1),(0,1,t),(-1,t,0),(t,0,-1),(0,-1,t) / sqrt(2+t),
    t = golden ratio; P (3x6) has these unit columns, P P^T = 2 I.
  * Q (3x6): sqrt2 * orthonormal basis of the complement of the rows of P (Q Q^T = 2 I,
    P Q^T = 0), oriented so that it is the Galois-conjugate star (t -> -1/t) up to a rotation.
  * a periodic tiling: 6x3 integer period matrix L (columns = 6D periods); vertices are
    reduced integer lifts n (mod L Z^3) inside the canonical window = projection of the unit
    6-cube on the orthogonal complement of span(L) (offset c generic). Physical positions use
    the ideal P (ideal golden rhombohedra, unit edges); cell = (P L)^T rows.
  * uniform phason strain M = (Q L)(P L)^{-1} (3x3, internal = M physical).
  * frame: EA = 1, EI = r^2/4 (r = 0.03), GJ = EI/(1+nu) (circular, J = 2I, nu = 0.3), each
    unit edge split into `sub` elements, rigid joints. Thresholds / lambda_E,
    lambda_E = pi^2 EI/(EA L^2), i.e. p_c/(EA lambda_E) with p = force per unit area.
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

try:
    import numba
    njit = numba.njit(cache=True)
except Exception:  # pragma: no cover
    def njit(f):
        return f

TAU = (1 + math.sqrt(5)) / 2
_S = math.sqrt(2 + TAU)
STAR = np.array([[1, TAU, 0], [TAU, 0, 1], [0, 1, TAU], [-1, TAU, 0], [TAU, 0, -1], [0, -1, TAU]]) / _S
P = STAR.T.copy()                                   # 3x6
_STARc = np.array([[1, -1 / TAU, 0], [-1 / TAU, 0, 1], [0, 1, -1 / TAU],
                   [-1, -1 / TAU, 0], [-1 / TAU, 0, -1], [0, -1, -1 / TAU]])
_STARc /= np.linalg.norm(_STARc, axis=1)[:, None]
Q = _STARc.T.copy()
assert np.allclose(P @ P.T, 2 * np.eye(3)) and np.allclose(Q @ Q.T, 2 * np.eye(3)) and np.allclose(P @ Q.T, 0)
NDIM = 6
TRIPLES = list(itertools.combinations(range(NDIM), 3))

R_STRUT = 0.03
NU = 0.3
EA = 1.0
EI = R_STRUT ** 2 / 4
GJ = EI / (1 + NU)
JA = R_STRUT ** 2 / 2          # polar radius of gyration^2 (torsional geometric term)
LAM_E = math.pi ** 2 * EI / EA


# ----------------------------------------------------------------- symmetry
def icosahedral_group():
    """All signed 6x6 permutation matrices g with P g = R P (R orthogonal). Returns list of
    (g, R, Rp) with Q g = Rp Q."""
    out = []
    for perm in itertools.permutations(range(6)):
        for signs in itertools.product((1, -1), repeat=6):
            g = np.zeros((6, 6))
            for i, (j, s) in enumerate(zip(perm, signs)):
                g[j, i] = s           # g e_i = s e_j
            R = P @ g @ P.T / 2
            if np.allclose(R @ P, P @ g, atol=1e-9) and np.allclose(R @ R.T, np.eye(3), atol=1e-9):
                Rp = Q @ g @ Q.T / 2
                assert np.allclose(Rp @ Q, Q @ g, atol=1e-9)
                out.append((g, R, Rp))
    return out


def ih_class(R):
    """Class label of an element of I_h from its 3x3 (par) matrix: proper part and parity."""
    d = np.linalg.det(R)
    r = R * np.sign(d)
    tr = np.trace(r)
    if abs(tr - 3) < 1e-6:
        c = 'E'
    elif abs(tr - TAU) < 1e-6:
        c = 'C5'
    elif abs(tr - (1 - TAU)) < 1e-6:
        c = 'C5^2'
    elif abs(tr - 0) < 1e-6:
        c = 'C3'
    elif abs(tr + 1) < 1e-6:
        c = 'C2'
    else:
        raise ValueError(tr)
    return c, (1 if d > 0 else -1)


CHAR_I = {  # character table of I (proper rotations)
    'A': dict(E=1, C5=1, **{'C5^2': 1}, C3=1, C2=1),
    'T1': dict(E=3, C5=TAU, **{'C5^2': 1 - TAU}, C3=0, C2=-1),
    'T2': dict(E=3, C5=1 - TAU, **{'C5^2': TAU}, C3=0, C2=-1),
    'G': dict(E=4, C5=-1, **{'C5^2': -1}, C3=1, C2=0),
    'H': dict(E=5, C5=0, **{'C5^2': 0}, C3=-1, C2=1),
}


def irrep_char(name, R):
    """name like 'Hg' or 'T1u'."""
    base, par = name[:-1], name[-1]
    c, d = ih_class(R)
    return CHAR_I[base][c] * (1 if (par == 'g' or d > 0) else -1)


IRREPS = [b + p for b in CHAR_I for p in 'gu']


def decompose(chars, group):
    """chars: list of characters (same order as group) -> multiplicities."""
    out = {}
    for nm in IRREPS:
        m = sum(c * irrep_char(nm, R) for c, (g, R, Rp) in zip(chars, group)) / len(group)
        if abs(m) > 1e-6:
            out[nm] = round(m, 6)
    return out


def phason_projectors(group):
    """Projectors (9x9, acting on vec(M), M 3x3 row-major) onto the G and H isotypic parts of
    the phason-strain space; action M -> Rp M R^T."""
    projs = {}
    for nm in ('Gg', 'Hg'):
        Pm = np.zeros((9, 9))
        for g, R, Rp in group:
            Pm += irrep_char(nm, R) * np.kron(Rp, R)
        projs[nm] = Pm * int(nm[0] == 'G' and 4 or 5) / len(group)
    return projs


# ----------------------------------------------------------------- lattices / approximants
def cubic_periods(p, q):
    """6x3 period matrix of the p/q cubic approximant (p/q -> tau). P L = 2(q + p tau)/S * I."""
    Lx = [q, p, 0, -q, p, 0]
    Ly = [p, 0, q, p, 0, -q]
    Lz = [0, q, p, 0, -q, p]
    return np.array([Lx, Ly, Lz], np.int64).T


def tetra_periods(pq_xy, pq_z):
    (p, q), (pz, qz) = pq_xy, pq_z
    Lx = [q, p, 0, -q, p, 0]
    Ly = [p, 0, q, p, 0, -q]
    Lz = [0, qz, pz, 0, -qz, pz]
    return np.array([Lx, Ly, Lz], np.int64).T


def minors(L):
    return {I: int(round(np.linalg.det(L[list(I), :].astype(float)))) for I in TRIPLES}


def phason_matrix(L):
    L = np.asarray(L, float)
    return (Q @ L) @ np.linalg.inv(P @ L)


def check_minor_signs(L):
    """Valid tiling needs sign(minor_I(L)) == sign(det P_I) * sigma for all I (or 0)."""
    m = minors(L)
    s = {I: np.sign(np.linalg.det(P[:, list(I)])) for I in TRIPLES}
    a = [m[I] * s[I] for I in TRIPLES]
    return all(x >= 0 for x in a) or all(x <= 0 for x in a), sum(abs(v) for v in m.values())


class Reducer:
    def __init__(self, L):
        self.L = np.array(L, np.int64)
        G = self.L.T @ self.L
        self.G = G
        self.det = int(round(np.linalg.det(G.astype(float))))
        self.adj = np.array(np.round(np.linalg.inv(G.astype(float)) * self.det), np.int64)
        assert np.array_equal(self.adj @ G, self.det * np.eye(3, dtype=np.int64))

    def __call__(self, n):
        r = self.L.T @ n
        a = (self.adj @ r) // self.det
        return n - self.L @ a, a


def canonical(L, offset=None, seed=7):
    """Canonical cut tiling for slope span(L). Returns dict with reduced lifts etc."""
    L = np.array(L, np.int64)
    ok, nv = check_minor_signs(L)
    assert ok, minors(L)
    q_, _ = np.linalg.qr(np.c_[L.astype(float), np.eye(6)])
    perp = q_[:, 3:6].T
    cube = np.array(list(itertools.product([-.5, .5], repeat=6)))
    eq = ConvexHull(cube @ perp.T).equations
    if offset is None:
        rng = np.random.default_rng(seed)
        offset = rng.uniform(-.5, .5, 6) * 0.37 + np.array([0.1173, 0.2711, -0.0618, 0.3307, -0.2213, 0.0491])
    c = np.asarray(offset, float)
    cperp = perp @ c
    red = Reducer(L)

    def inside(n):
        y = perp @ n - cperp
        return bool(np.all(eq[:, :3] @ y + eq[:, 3] < -1e-9))

    start = None
    for v in itertools.product(range(-2, 3), repeat=6):
        if inside(np.array(v, float)):
            start = np.array(v, np.int64); break
    s0, _ = red(start)
    reps = {tuple(s0): 0}; order = [tuple(s0)]; stack = [tuple(s0)]
    E = np.eye(6, dtype=np.int64)
    while stack:
        cur = np.array(stack.pop(), np.int64)
        for k in range(6):
            for sgn in (1, -1):
                nb = cur + sgn * E[k]
                if inside(nb.astype(float)):
                    r = tuple(red(nb)[0])
                    if r not in reps:
                        reps[r] = len(order); order.append(r); stack.append(r)
    lifts = np.array(order, np.int64)
    assert len(lifts) == nv, (len(lifts), nv)
    return dict(L=L, lifts=lifts, perp=perp, window_eq=eq, cperp=cperp, offset=c, nv=nv)


def edges_from_lifts(lifts, L):
    red = Reducer(L)
    look = {tuple(r): i for i, r in enumerate(lifts)}
    E = np.eye(6, dtype=np.int64)
    edges = []
    for i, r in enumerate(lifts):
        for k in range(6):
            rr, w = red(r + E[k])
            j = look.get(tuple(rr))
            if j is not None:
                edges.append((i, j, k, w[0], w[1], w[2]))
    edges = np.array(edges, np.int64)
    return edges


def check_tiling(lifts, L):
    """3D validity: lower-corner parallelepipeds (n; a<b<c) with all 8 corners present.
    Valid tiling => count == V, sum |det P_abc| == cell volume, E == 3V."""
    red = Reducer(L)
    look = {tuple(r) for r in lifts}
    E = np.eye(6, dtype=np.int64)
    cnt = 0; vol = 0.0
    dets = {I: abs(np.linalg.det(P[:, list(I)])) for I in TRIPLES}
    for r in lifts:
        for I in TRIPLES:
            ok = True
            for sub in itertools.product((0, 1), repeat=3):
                if sub == (0, 0, 0):
                    continue
                n = r + sum(s * E[k] for s, k in zip(sub, I))
                if tuple(red(n)[0]) not in look:
                    ok = False; break
            if ok:
                cnt += 1; vol += dets[I]
    cellvol = abs(np.linalg.det(P @ L))
    ne = len(edges_from_lifts(lifts, L))
    return dict(V=len(lifts), tiles=cnt, vol=vol, cellvol=cellvol, E=ne,
                ok=bool(cnt == len(lifts) and abs(vol - cellvol) < 1e-6 * cellvol and ne == 3 * len(lifts)))


def window_excess(lifts, tl):
    y = lifts @ tl['perp'].T - tl['cperp']
    eq = tl['window_eq']
    return np.max(y @ eq[:, :3].T + eq[:, 3], axis=1)


def perp_var(lifts, tl):
    y = lifts @ tl['perp'].T
    return float(np.mean(np.sum((y - y.mean(0)) ** 2, 1)))


# ----------------------------------------------------------------- flip MC (numba)
KB = 10
OFF = 1 << (KB - 1)


@njit
def _key(r):
    k = 0
    for c in range(6):
        k = (k << KB) | (r[c] + OFF)
    return k


@njit
def _reduce(n, L, adj, det, out):
    r = np.zeros(3, np.int64)
    for a in range(3):
        s = 0
        for c in range(6):
            s += L[c, a] * n[c]
        r[a] = s
    aa = np.zeros(3, np.int64)
    for a in range(3):
        s = 0
        for b in range(3):
            s += adj[a, b] * r[b]
        aa[a] = s // det
    for c in range(6):
        s = n[c]
        for a in range(3):
            s -= L[c, a] * aa[a]
        out[c] = s


@njit
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


@njit
def _hins(keys, vals, k, v):
    mask = keys.shape[0] - 1
    h = ((k * 0x9E3779B97F4A7C15) >> 20) & mask
    while keys[h] != -1 and keys[h] != -2:
        h = (h + 1) & mask
    keys[h] = k; vals[h] = v


@njit
def _present(keys, n, L, adj, det, tmp):
    _reduce(n, L, adj, det, tmp)
    return _hfind(keys, _key(tmp)) >= 0


@njit
def _mc(lifts, keys, vals, L, adj, det, Pm, nattempt, seed, maxacc):
    np.random.seed(seed)
    V = lifts.shape[0]
    r = np.zeros(6, np.int64); tmp = np.zeros(6, np.int64); t2 = np.zeros(6, np.int64)
    ds = np.zeros((4, 6), np.int64); new = np.zeros(6, np.int64)
    acc = 0
    D = np.zeros((3, 4))
    for it in range(nattempt):
        v = np.random.randint(V)
        for c in range(6):
            r[c] = lifts[v, c]
        nd = 0
        for k in range(6):
            for s in (1, -1):
                for c in range(6):
                    tmp[c] = r[c]
                tmp[k] += s
                if _present(keys, tmp, L, adj, det, t2):
                    if nd < 4:
                        for c in range(6):
                            ds[nd, c] = 0
                        ds[nd, k] = s
                    nd += 1
        if nd != 4:
            continue
        for a in range(4):
            for i in range(3):
                x = 0.0
                for c in range(6):
                    x += Pm[i, c] * ds[a, c]
                D[i, a] = x
        # null vector of D (3x4): w_a = (-1)^a det(D without column a)
        w = np.zeros(4)
        for a in range(4):
            cols = [b for b in range(4) if b != a]
            m = np.empty((3, 3))
            for i in range(3):
                for j in range(3):
                    m[i, j] = D[i, cols[j]]
            dd = (m[0, 0] * (m[1, 1] * m[2, 2] - m[1, 2] * m[2, 1])
                  - m[0, 1] * (m[1, 0] * m[2, 2] - m[1, 2] * m[2, 0])
                  + m[0, 2] * (m[1, 0] * m[2, 1] - m[1, 1] * m[2, 0]))
            w[a] = dd * (1.0 if a % 2 == 0 else -1.0)
        if not ((w[0] > 1e-9 and w[1] > 1e-9 and w[2] > 1e-9 and w[3] > 1e-9) or
                (w[0] < -1e-9 and w[1] < -1e-9 and w[2] < -1e-9 and w[3] < -1e-9)):
            continue
        ok = True
        # pair sums and triple sums present
        for a in range(4):
            for b in range(a + 1, 4):
                for c in range(6):
                    tmp[c] = r[c] + ds[a, c] + ds[b, c]
                if not _present(keys, tmp, L, adj, det, t2):
                    ok = False
        if not ok:
            continue
        for a in range(4):
            for c in range(6):
                tmp[c] = r[c] + ds[0, c] + ds[1, c] + ds[2, c] + ds[3, c] - ds[a, c]
            if not _present(keys, tmp, L, adj, det, t2):
                ok = False
        if not ok:
            continue
        for c in range(6):
            tmp[c] = r[c] + ds[0, c] + ds[1, c] + ds[2, c] + ds[3, c]
        _reduce(tmp, L, adj, det, new)
        if _hfind(keys, _key(new)) >= 0:
            continue
        cnt = 0
        for k in range(6):
            for s in (1, -1):
                for c in range(6):
                    tmp[c] = new[c]
                tmp[k] += s
                if _present(keys, tmp, L, adj, det, t2):
                    cnt += 1
        if cnt != 4:
            continue
        h = _hfind(keys, _key(r))
        keys[h] = -2
        _hins(keys, vals, _key(new), v)
        for c in range(6):
            lifts[v, c] = new[c]
        acc += 1
        if maxacc > 0 and acc >= maxacc:
            break
    return acc


def flip_mc(lifts, L, nattempt, seed, maxacc=0):
    lifts = np.array(lifts, np.int64).copy()
    V = len(lifts)
    size = 1 << int(math.ceil(math.log2(8 * V)))
    keys = -np.ones(size, np.int64); vals = np.zeros(size, np.int64)
    for i, r in enumerate(lifts):
        _hins(keys, vals, _key(r), i)
    red = Reducer(L)
    acc = _mc(lifts, keys, vals, np.array(L, np.int64), red.adj, np.int64(red.det), P.copy(),
              int(nattempt), int(seed), int(maxacc))
    return lifts, int(acc)


# ----------------------------------------------------------------- 3D beam element
def beam12(L):
    """Local 12x12 elastic K and geometric G (per unit axial force, tension positive) for a 3D
    Euler-Bernoulli beam; dofs (u,v,w,tx,ty,tz) at node 1 then node 2, local x = axis."""
    k = np.zeros((12, 12)); g = np.zeros((12, 12))
    k[np.ix_([0, 6], [0, 6])] = EA / L * np.array([[1, -1], [-1, 1]])
    k[np.ix_([3, 9], [3, 9])] = GJ / L * np.array([[1, -1], [-1, 1]])
    kb = EI / L ** 3 * np.array([[12, 6 * L, -12, 6 * L], [6 * L, 4 * L * L, -6 * L, 2 * L * L],
                                 [-12, -6 * L, 12, -6 * L], [6 * L, 2 * L * L, -6 * L, 4 * L * L]])
    gb = np.array([[36, 3 * L, -36, 3 * L], [3 * L, 4 * L * L, -3 * L, -L * L],
                   [-36, -3 * L, 36, -3 * L], [3 * L, -L * L, -3 * L, 4 * L * L]]) / (30 * L)
    iv = [1, 5, 7, 11]                 # v, theta_z (theta_z = +dv/dx)
    iw = [2, 4, 8, 10]                 # w, theta_y (theta_y = -dw/dx)
    sgn = np.diag([1, -1, 1, -1])
    k[np.ix_(iv, iv)] = kb; g[np.ix_(iv, iv)] = gb
    k[np.ix_(iw, iw)] = sgn @ kb @ sgn; g[np.ix_(iw, iw)] = sgn @ gb @ sgn
    g[np.ix_([3, 9], [3, 9])] = JA / L * np.array([[1, -1], [-1, 1]])
    return k, g


def rot12(e):
    e = np.asarray(e, float); e = e / np.linalg.norm(e)
    a = np.array([0, 0, 1.]) if abs(e[2]) < 0.9 else np.array([1., 0, 0])
    y = np.cross(a, e); y /= np.linalg.norm(y)
    z = np.cross(e, y)
    Rf = np.array([e, y, z])
    return np.kron(np.eye(4), Rf)


class Frame3D:
    """Periodic rigid-joint 3D frame. nodes X (n,3), elements (ei, ej, wrap(3), dir k, xa)."""

    def __init__(self, X, cell, elems_i, elems_j, wraps, dirs, dirvecs, Le, xa):
        self.X = np.asarray(X, float); self.cell = np.asarray(cell, float)
        self.ei = np.asarray(elems_i); self.ej = np.asarray(elems_j)
        self.wrap = np.asarray(wraps, float); self.dirk = np.asarray(dirs)
        self.dirvecs = np.asarray(dirvecs, float); self.Le = float(Le); self.xa = np.asarray(xa, float)
        self.nn = len(self.X); self.n = 6 * self.nn
        kl, gl = beam12(self.Le)
        nd = len(self.dirvecs)
        self.Ke = np.zeros((nd, 12, 12)); self.Ge = np.zeros((nd, 12, 12))
        for k in range(nd):
            Tm = rot12(self.dirvecs[k])
            self.Ke[k] = Tm.T @ kl @ Tm; self.Ge[k] = Tm.T @ gl @ Tm
        di = 6 * self.ei[:, None] + np.arange(6)[None, :]
        dj = 6 * self.ej[:, None] + np.arange(6)[None, :]
        Dm = np.concatenate([di, dj], 1)
        self.rows = np.repeat(Dm, 12, axis=1).ravel()
        self.cols = np.tile(Dm, (1, 12)).ravel()
        blk = np.zeros((12, 12), int); blk[:6, 6:] = 1; blk[6:, :6] = -1
        self.blk = np.tile(blk.ravel(), len(self.ei)).reshape(-1, 144)
        self.free_gamma = np.arange(3, self.n)
        self.vol = abs(np.linalg.det(self.cell))

    @classmethod
    def from_tiling(cls, lifts, L, sub=2, edges=None):
        if edges is None:
            edges = edges_from_lifts(lifts, L)
        X = lifts @ P.T
        V = len(X)
        ei, ej, wr, xa, dirs = [], [], [], [], []
        for e, (i, j, k, w1, w2, w3) in enumerate(edges):
            d = P[:, k]
            ids = [i] + [V + e * (sub - 1) + h for h in range(sub - 1)] + [j]
            for h in range(sub):
                ei.append(ids[h]); ej.append(ids[h + 1])
                wr.append((w1, w2, w3) if h == sub - 1 else (0, 0, 0))
                xa.append(X[i] + d * h / sub); dirs.append(k)
        nodes = [X]
        if sub > 1:
            hh = np.arange(1, sub) / sub
            mids = (X[edges[:, 0]][:, None, :] + hh[None, :, None] * P[:, edges[:, 2]].T[:, None, :]).reshape(-1, 3)
            nodes.append(mids)
        cell = (P @ np.asarray(L, float)).T
        f = cls(np.vstack(nodes), cell, ei, ej, wr, dirs, P.T, 1.0 / sub, xa)
        f.V = V; f.sub = sub
        return f

    def _assemble(self, coef, mats, k):
        ph = np.exp(2j * np.pi * (self.wrap @ np.asarray(k, float)))
        vals = (coef[:, None, None] * mats[self.dirk]).reshape(len(self.ei), 144).astype(complex)
        phase = np.where(self.blk == 1, ph[:, None], np.where(self.blk == -1, np.conj(ph)[:, None], 1.0))
        M = sp.csr_matrix(((vals * phase).ravel(), (self.rows, self.cols)), shape=(self.n, self.n))
        return (M + M.conj().T) * 0.5

    def prestress(self, strain):
        ones = np.ones(len(self.ei))
        if not hasattr(self, '_lu0'):
            K0 = self._assemble(ones, self.Ke, (0, 0, 0)).real.tocsc()
            fr = self.free_gamma
            self._K0 = K0
            self._lu0 = spla.splu(K0[fr][:, fr].tocsc(), permc_spec='COLAMD')
        e = self.dirvecs[self.dirk]
        xb = self.xa + e * self.Le
        dimp = np.zeros((len(self.ei), 12))
        dimp[:, 0:3] = self.xa @ strain.T; dimp[:, 6:9] = xb @ strain.T
        fe = np.einsum('eab,eb->ea', self.Ke[self.dirk], dimp)
        f = np.zeros(self.n)
        np.add.at(f, (6 * self.ei[:, None] + np.arange(6)).ravel(), fe[:, :6].ravel())
        np.add.at(f, (6 * self.ej[:, None] + np.arange(6)).ravel(), fe[:, 6:].ravel())
        fr = self.free_gamma
        w = np.zeros(self.n)
        w[fr] = self._lu0.solve(-f[fr])
        d = dimp.copy()
        d[:, :6] += w[(6 * self.ei[:, None] + np.arange(6))]
        d[:, 6:] += w[(6 * self.ej[:, None] + np.arange(6))]
        self.N = EA / self.Le * np.einsum('ea,ea->e', e, d[:, 6:9] - d[:, 0:3])
        self.resid = float(np.linalg.norm(self._K0 @ w + f) / max(np.linalg.norm(f), 1e-30))
        fb = np.einsum('eab,eb->ea', self.Ke[self.dirk], d)[:, 6:9]
        delta = e * self.Le
        sig = 0.5 * (np.einsum('ea,eb->ab', fb, delta) + np.einsum('ea,eb->ab', delta, fb))
        self.stress = sig / self.vol
        return self.N

    def set_load(self, protocol='stress'):
        if protocol == 'strain':
            eps = -np.eye(3)
        else:
            basis = []
            for i in range(3):
                B = np.zeros((3, 3)); B[i, i] = 1; basis.append(B)
            for i, j in ((1, 2), (0, 2), (0, 1)):
                B = np.zeros((3, 3)); B[i, j] = B[j, i] = 1 / math.sqrt(2); basis.append(B)
            Cm = np.zeros((6, 6))
            for a, B in enumerate(basis):
                self.prestress(B)
                for b, Bb in enumerate(basis):
                    Cm[b, a] = np.sum(Bb * self.stress)
            c = la.solve(Cm, np.array([-1., -1., -1., 0, 0, 0]))
            eps = sum(ci * B for ci, B in zip(c, basis))
            self.C6 = Cm
        self.eps = eps
        self.prestress(eps)
        self.pressure = float(-np.trace(self.stress) / 3)
        return eps

    def lam_modes(self, k, nev=8):
        """Largest mu of K v = mu H v (lambda = 1/mu / LAM_E). Returns sorted list of
        (lambda, vector) for positive mu."""
        k = np.asarray(k, float)
        K = self._assemble(np.ones(len(self.ei)), self.Ke, k).tocsc()
        H = -self._assemble(self.N, self.Ge, k).tocsc()
        gamma = np.allclose(k, np.rint(k))
        fr = self.free_gamma if gamma else np.arange(self.n)
        K = K[fr][:, fr]; H = H[fr][:, fr]
        if K.shape[0] <= 1500:
            mu, vec = la.eigh(H.toarray(), K.toarray())
            idx = np.argsort(mu)[::-1][:nev]
            mu = mu[idx]; vec = vec[:, idx]
        else:
            lu = spla.splu(K, permc_spec='COLAMD')
            op = spla.LinearOperator(K.shape, matvec=lambda x: lu.solve(H @ x), dtype=complex)
            mu, vec = spla.eigs(op, k=nev, which='LR', tol=1e-10, maxiter=100000)
            mu = mu.real
            idx = np.argsort(mu)[::-1]; mu = mu[idx]; vec = vec[:, idx]
        out = []
        for m, v in zip(mu, vec.T):
            if m <= 0:
                continue
            full = np.zeros(self.n, complex); full[fr] = v
            out.append((1 / m / LAM_E, full))
        return out

    def lam(self, k, nev=4):
        m = self.lam_modes(k, nev)
        return m[0][0] if m else np.inf

    def classify(self, v, q):
        """Translation content: mean Bloch-demodulated displacement U, translational fraction,
        longitudinal fraction |U.qhat|^2/|U|^2, participation ratio of node translations."""
        u = v.reshape(-1, 6)[:, :3]
        ph = np.exp(-1j * (self.X @ q))
        U = (u * ph[:, None]).mean(0)
        tf = float(np.sum(np.abs(U) ** 2) * len(u) / max(np.sum(np.abs(u) ** 2), 1e-300))
        qh = q / max(np.linalg.norm(q), 1e-300)
        lf = float(abs(U @ qh) ** 2 / max(np.sum(np.abs(U) ** 2), 1e-300))
        a2 = np.sum(np.abs(u) ** 2, 1)
        pr = float(a2.sum() ** 2 / (len(a2) * np.sum(a2 ** 2) + 1e-300))
        return dict(trans=tf, longi=lf, pr=pr, U=U)

    def acoustic(self, n, dq, nev=8):
        """Acoustic thresholds at physical wavevector q = 2 pi dq n: returns dict with
        T (sorted list of transverse lambdas), L (or None), local (lowest non-acoustic)."""
        n = np.asarray(n, float); n = n / np.linalg.norm(n)
        q = 2 * np.pi * dq * n
        kred = dq * (self.cell @ n)
        modes = self.lam_modes(kred, nev)
        T, Lg, loc = [], [], []
        for lamv, v in modes:
            c = self.classify(v, q)
            if c['trans'] > 0.5:
                (Lg if c['longi'] > 0.5 else T).append(lamv)
            else:
                loc.append(lamv)
        return dict(T=sorted(T), L=(min(Lg) if Lg else None), local=(min(loc) if loc else None),
                    all=[m[0] for m in modes])

    def longwave(self, dirs, dqs=(0.002, 0.001), nev=8):
        """Richardson (error ~ dq^2) on polarization-tracked acoustic thresholds."""
        a, b = dqs; r = (a / b) ** 2
        res = []
        for n in dirs:
            A = self.acoustic(n, a, nev); B = self.acoustic(n, b, nev)
            row = dict(n=list(map(float, n)))
            for key, f in (('T1', lambda d: d['T'][0] if len(d['T']) > 0 else None),
                           ('T2', lambda d: d['T'][1] if len(d['T']) > 1 else None),
                           ('L', lambda d: d['L'])):
                xa, xb = f(A), f(B)
                row[key] = None if (xa is None or xb is None) else float((r * xb - xa) / (r - 1))
                row[key + '_raw'] = [xa, xb]
            row['local'] = [A['local'], B['local']]
            res.append(row)
        return res


def sc_frame(sub=2):
    """Simple cubic rigid-joint frame, a = 1, one node per cell (+ midnodes)."""
    X0 = np.zeros((1, 3))
    dirvecs = np.eye(3)
    ei, ej, wr, xa, dirs = [], [], [], [], []
    nodes = [X0]
    nid = 1
    for k in range(3):
        ids = [0] + list(range(nid, nid + sub - 1)) + [0]
        for h in range(sub - 1):
            nodes.append((dirvecs[k] * (h + 1) / sub)[None, :])
        nid += sub - 1
        for h in range(sub):
            ei.append(ids[h]); ej.append(ids[h + 1])
            w = dirvecs[k].astype(int) if h == sub - 1 else np.zeros(3, int)
            wr.append(tuple(w)); xa.append(dirvecs[k] * h / sub); dirs.append(k)
    f = Frame3D(np.vstack(nodes), np.eye(3), ei, ej, wr, dirs, dirvecs, 1.0 / sub, xa)
    f.V = 1; f.sub = sub
    return f


def supercell_sc(m, sub=2):
    """m x m x m supercell of the SC frame (for zone-boundary modes at Gamma of the supercell)."""
    V = m ** 3
    idx = {c: i for i, c in enumerate(itertools.product(range(m), repeat=3))}
    X = [np.array(c, float) for c in idx]
    ei, ej, wr, xa, dirs = [], [], [], [], []
    nid = V
    for c, i in idx.items():
        for k in range(3):
            c2 = list(c); c2[k] += 1
            w = [0, 0, 0]
            if c2[k] == m:
                c2[k] = 0; w[k] = 1
            j = idx[tuple(c2)]
            ids = [i] + list(range(nid, nid + sub - 1)) + [j]
            for h in range(sub - 1):
                X.append(np.array(c, float) + np.eye(3)[k] * (h + 1) / sub)
            nid += sub - 1
            for h in range(sub):
                ei.append(ids[h]); ej.append(ids[h + 1])
                wr.append(tuple(w) if h == sub - 1 else (0, 0, 0))
                xa.append(np.array(c, float) + np.eye(3)[k] * h / sub); dirs.append(k)
    f = Frame3D(np.array(X), m * np.eye(3), ei, ej, wr, dirs, np.eye(3), 1.0 / sub, xa)
    f.V = V; f.sub = sub
    return f


# ----------------------------------------------------------------- directions / harmonics
def dirs13():
    d = [(1, 0, 0), (0, 1, 0), (0, 0, 1),
         (1, 1, 1), (1, 1, -1), (1, -1, 1), (-1, 1, 1),
         (1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1), (0, 1, 1), (0, 1, -1)]
    return [np.array(x, float) / np.linalg.norm(x) for x in d]


def K4(n):
    n = np.asarray(n, float); n = n / np.linalg.norm(n)
    return float(np.sum(n ** 4) - 0.6)


def fit_cubic(dirs, vals):
    """vals(n) = c0 + c4 K4(n) [+ l=2 terms]; returns c0, c4, rms, and l=2 (xx-yy, 2zz-xx-yy)."""
    ok = [i for i, v in enumerate(vals) if v is not None]
    A = []
    for i in ok:
        n = dirs[i] / np.linalg.norm(dirs[i])
        A.append([1, K4(n), n[0] ** 2 - n[1] ** 2, 2 * n[2] ** 2 - n[0] ** 2 - n[1] ** 2,
                  n[0] * n[1], n[1] * n[2], n[0] * n[2]])
    A = np.array(A); y = np.array([vals[i] for i in ok])
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    c2 = np.linalg.lstsq(A[:, :2], y, rcond=None)[0]
    return dict(c0=float(c[0]), c4=float(c[1]), l2=[float(x) for x in c[2:]],
                rms_full=float(np.sqrt(np.mean((A @ c - y) ** 2))),
                c0_iso=float(c2[0]), c4_iso=float(c2[1]),
                rms_c0c4=float(np.sqrt(np.mean((A[:, :2] @ c2 - y) ** 2))))

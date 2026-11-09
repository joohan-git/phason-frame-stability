"""cell_tiling.py -- independent cut-and-project implementation for heldout.

Ammann-Beenker (Z^4 -> 2D) and Penrose (Z^5 -> 2D, 3D internal) periodic
approximants as canonical cuts of the hypercubic lattice by a rational plane
spanned by two integer vectors L1, L2. Written without reference to the
project's main codes (ab_frame/penrose_frame/design_core).
"""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames',)]  # shared modules
import itertools
import math
from math import gcd

import numpy as np


# ----------------------------------------------------------------------------
# integer linear algebra
# ----------------------------------------------------------------------------
def smith_complete(L):
    """Integer d x 2 matrix L spanning a primitive rank-2 sublattice.
    Returns unimodular M (d x d, int) whose first two columns are a basis of
    the same sublattice and whose other columns complete a basis of Z^d, and
    its integer inverse. Raises ValueError if the lattice is not primitive."""
    A = [list(map(int, row)) for row in np.asarray(L).T.tolist()]  # 2 x d, rows = L1^T, L2^T
    d = len(A[0])
    V = [[int(i == j) for j in range(d)] for i in range(d)]

    def col_op(j, k, q):  # col_j -= q * col_k
        for r in range(2):
            A[r][j] -= q * A[r][k]
        for r in range(d):
            V[r][j] -= q * V[r][k]

    def col_swap(j, k):
        for r in range(2):
            A[r][j], A[r][k] = A[r][k], A[r][j]
        for r in range(d):
            V[r][j], V[r][k] = V[r][k], V[r][j]

    def col_neg(j):
        for r in range(2):
            A[r][j] = -A[r][j]
        for r in range(d):
            V[r][j] = -V[r][j]

    for piv in range(2):
        while True:
            cols = [j for j in range(piv, d) if A[piv][j] != 0]
            if not cols:
                raise ValueError("rank deficient")
            jmin = min(cols, key=lambda j: abs(A[piv][j]))
            if jmin != piv:
                col_swap(piv, jmin)
            rest = [j for j in range(piv + 1, d) if A[piv][j] != 0]
            if not rest:
                break
            for j in rest:
                col_op(j, piv, A[piv][j] // A[piv][piv])
        if A[piv][piv] < 0:
            col_neg(piv)
        if A[piv][piv] != 1:
            raise ValueError("lattice not primitive (pivot %d)" % A[piv][piv])
        if piv == 0:
            # row op: row1 -= A[1][0] * row0 (new basis of the same sublattice)
            q = A[1][0]
            for j in range(d):
                A[1][j] -= q * A[0][j]
    Vm = np.array(V, dtype=np.int64)
    M = np.rint(np.linalg.inv(Vm.T)).astype(np.int64)  # L^T V = [I|0] -> M = V^{-T}
    Minv = Vm.T.copy()
    assert np.array_equal(M @ Minv, np.eye(d, dtype=np.int64))
    # the first two columns of M span the lattice of L
    Lf = np.asarray(L, dtype=np.int64)
    coef = Minv[:2, :] @ Lf
    assert np.array_equal(M[:, :2] @ coef, Lf)
    return M, Minv


def is_primitive(L):
    L = np.asarray(L, dtype=np.int64)
    d = L.shape[0]
    g = 0
    for i in range(d):
        for j in range(i + 1, d):
            g = gcd(g, int(L[i, 0] * L[j, 1] - L[j, 0] * L[i, 1]))
    return g == 1



def lll_reduce(Bf, delta=0.75):
    """Floating-point LLL on the columns of Bf (m x n). Returns integer
    unimodular U (n x n) such that Bf @ U is LLL-reduced."""
    B = np.array(Bf, float)
    n = B.shape[1]
    U = np.eye(n, dtype=np.int64)

    def gso(B):
        Q = np.zeros_like(B)
        mu = np.zeros((n, n))
        for i in range(n):
            v = B[:, i].copy()
            for j in range(i):
                mu[i, j] = (B[:, i] @ Q[:, j]) / (Q[:, j] @ Q[:, j])
                v -= mu[i, j] * Q[:, j]
            Q[:, i] = v
        return Q, mu

    k = 1
    Q, mu = gso(B)
    it = 0
    while k < n and it < 10000:
        it += 1
        for j in range(k - 1, -1, -1):
            q = int(np.rint(mu[k, j]))
            if q != 0:
                B[:, k] -= q * B[:, j]
                U[:, k] -= q * U[:, j]
                Q, mu = gso(B)
        if Q[:, k] @ Q[:, k] >= (delta - mu[k, k - 1] ** 2) * (Q[:, k - 1] @ Q[:, k - 1]):
            k += 1
        else:
            B[:, [k - 1, k]] = B[:, [k, k - 1]]
            U[:, [k - 1, k]] = U[:, [k, k - 1]]
            Q, mu = gso(B)
            k = max(k - 1, 1)
    return U


# ----------------------------------------------------------------------------
# stars
# ----------------------------------------------------------------------------
class Star:
    def __init__(self, cls):
        self.cls = cls
        if cls == "ab":
            k = np.arange(4)
            self.d = 4
            self.E = np.stack([np.cos(k * np.pi / 4), np.sin(k * np.pi / 4)])
            self.I = np.stack([np.cos(3 * k * np.pi / 4), np.sin(3 * k * np.pi / 4)])
            self.ndir = 8
        elif cls == "pen":
            k = np.arange(5)
            self.d = 5
            self.E = np.stack([np.cos(2 * np.pi * k / 5), np.sin(2 * np.pi * k / 5)])
            self.I = np.vstack([np.cos(4 * np.pi * k / 5), np.sin(4 * np.pi * k / 5),
                                np.ones(5) / np.sqrt(5)])
            self.ndir = 10
        else:
            raise ValueError(cls)
        # orthonormal basis of internal space (rows)
        q, _ = np.linalg.qr(self.I.T)
        self.N0 = q.T  # (d-2) x d
        self.group = self._point_group()

    def _point_group(self):
        d = self.d
        mats = []
        if self.cls == "ab":
            rot = np.zeros((4, 4), int)
            for k in range(3):
                rot[k + 1, k] = 1
            rot[0, 3] = -1  # e3 -> -e0
            mir = np.zeros((4, 4), int)
            mir[0, 0] = 1
            mir[3, 1] = -1
            mir[2, 2] = -1
            mir[1, 3] = -1
            gens = [rot, mir]
        else:
            rot = np.zeros((5, 5), int)
            for k in range(5):
                rot[(k + 1) % 5, k] = 1
            mir = np.zeros((5, 5), int)
            for k in range(5):
                mir[(-k) % 5, k] = 1
            gens = [rot, mir, -np.eye(5, dtype=int)]
        seen = {tuple(np.eye(d, dtype=int).ravel())}
        frontier = [np.eye(d, dtype=int)]
        while frontier:
            new = []
            for g in frontier:
                for h in gens:
                    m = h @ g
                    key = tuple(m.ravel())
                    if key not in seen:
                        seen.add(key)
                        new.append(m)
            frontier = new
        mats = [np.array(k).reshape(d, d) for k in seen]
        # check that the group preserves the physical star up to sign
        for m in mats:
            Em = self.E @ m
            for k in range(d):
                v = Em[:, k]
                ok = any(np.allclose(v, s * self.E[:, j]) for j in range(d) for s in (1, -1))
                assert ok
        return mats

    def phason(self, L):
        """S (dI x 2) with I L = S E L; returns S, delta, sector dict."""
        L = np.asarray(L, float)
        EL = self.E @ L
        S = (self.I @ L) @ np.linalg.inv(EL)
        delta = np.linalg.norm(S) / np.sqrt(2)
        a = (S[0, 0] + S[1, 1]) / 2
        b = (S[1, 0] - S[0, 1]) / 2
        c = (S[0, 0] - S[1, 1]) / 2
        dd = (S[0, 1] + S[1, 0]) / 2
        sec = {}
        if self.cls == "ab":
            sec = {"E2": (a * a + b * b) / delta**2, "B1": c * c / delta**2, "B2": dd * dd / delta**2}
            sec["delta2"] = delta
        else:
            diag = (S[2, 0] ** 2 + S[2, 1] ** 2) / 2
            sec = {"E4a": (a * a + b * b) / delta**2, "E2": (c * c + dd * dd) / delta**2,
                   "E4diag": diag / delta**2}
            sec["delta2"] = math.sqrt(a * a + b * b + c * c + dd * dd)
        return S, delta, sec


# ----------------------------------------------------------------------------
# windows (zonotopes)
# ----------------------------------------------------------------------------
class Zonotope:
    """Centered zonotope sum_k t_k g_k, t_k in [-1/2, 1/2], generators g (m x dim)."""

    def __init__(self, gens):
        g = np.asarray(gens, float)
        self.g = g
        dim = g.shape[1]
        normals = []
        if dim == 2:
            for k in range(g.shape[0]):
                n = np.array([-g[k, 1], g[k, 0]])
                nn = np.linalg.norm(n)
                if nn > 1e-12:
                    normals.append(n / nn)
        elif dim == 3:
            for i, j in itertools.combinations(range(g.shape[0]), 2):
                n = np.cross(g[i], g[j])
                nn = np.linalg.norm(n)
                if nn > 1e-12:
                    normals.append(n / nn)
        else:
            raise ValueError(dim)
        self.normals = np.array(normals)
        self.h = 0.5 * np.abs(self.normals @ g.T).sum(axis=1)
        self.radius = 0.5 * np.linalg.norm(g, axis=1).sum()

    def contains(self, y):
        y = np.atleast_2d(y)
        proj = np.abs(y @ self.normals.T)
        return np.all(proj <= self.h[None, :] + 1e-12, axis=1)


# ----------------------------------------------------------------------------
# periodic approximant cell
# ----------------------------------------------------------------------------
class Cell:
    def __init__(self, star, L1, L2, gamma):
        self.star = star
        d = star.d
        L = np.array([L1, L2], dtype=np.int64).T  # d x 2
        if not is_primitive(L):
            raise ValueError("non-primitive lattice")
        self.M, self.Minv = smith_complete(L)
        self.L = self.M[:, :2].copy()  # basis of the same lattice adapted to M
        # Minv rows 0,1 give (a,b) coords; rows 2.. give coset coords
        self.T = star.E @ self.L  # 2 x 2 periods (basis adapted to M, consistent with the edge shifts)
        self.area = abs(np.linalg.det(self.T))
        q, _ = np.linalg.qr(L.astype(float), mode="complete")
        self.Q = q[:, :2]  # plane basis
        self.N = q[:, 2:].T  # complement basis rows (d-2) x d
        self.window = Zonotope(self.N.T)  # generators N e_k (rows = k)
        self.gamma = np.asarray(gamma, float)
        self.ygamma = self.N @ self.gamma
        self.S, self.delta, self.sectors = star.phason(L)
        self._build()

    def inside(self, n):
        n = np.atleast_2d(n)
        return self.window.contains(n @ self.N.T - self.ygamma)

    def _build(self):
        d = self.star.d
        B = self.M[:, 2:]  # d x (d-2) coset basis
        Y = self.N @ B  # (d-2)x(d-2)
        U = lll_reduce(Y)  # reduce the projected coset basis so the search box is small
        B = B @ U
        Y = self.N @ B
        Yinv = np.linalg.inv(Y)
        R = self.window.radius + np.linalg.norm(self.ygamma) + 1e-9
        bound = np.ceil(np.abs(Yinv).sum(axis=1) * R).astype(int)
        ranges = [range(-b, b + 1) for b in bound]
        C = np.array(list(itertools.product(*ranges)), dtype=np.int64)
        nlift = C @ B.T  # candidates (coset reps)
        ok = self.inside(nlift)
        nlift = nlift[ok]
        # bring into fundamental cell: coordinates along L1, L2 (real)
        coords = (self.Q.T @ nlift.T).T @ np.linalg.inv(self.Q.T @ self.L.astype(float)).T
        shift = -np.floor(coords).astype(np.int64)
        nlift = nlift + shift @ self.L.T
        self.n = nlift  # V x d
        self.V = len(nlift)
        self.x = (self.star.E @ nlift.T).T
        # index: coset coords -> vertex id
        cos = (self.Minv[2:, :] @ nlift.T).T
        self.index = {tuple(c): i for i, c in enumerate(cos.tolist())}
        # edges
        edges = []
        for k in range(d):
            ek = np.zeros(d, dtype=np.int64)
            ek[k] = 1
            nb = nlift + ek
            okk = self.inside(nb)
            for i in np.nonzero(okk)[0]:
                nn = nb[i]
                c = tuple((self.Minv[2:, :] @ nn).tolist())
                j = self.index[c]
                diff = nn - nlift[j]
                ab = self.Minv[:2, :] @ diff
                assert np.array_equal(self.L @ ab, diff)
                edges.append((int(i), int(j), int(ab[0]), int(ab[1]), k))
        self.edges = edges
        self.E = len(edges)

    def neighbours_lift(self, n):
        d = self.star.d
        out = []
        for k in range(d):
            for s in (1, -1):
                m = n.copy()
                m[k] += s
                if self.inside(m)[0]:
                    out.append(m)
        return out


# ----------------------------------------------------------------------------
# quasicrystal patch (for the atlas)
# ----------------------------------------------------------------------------
class QCPatch:
    def __init__(self, star, gamma, R):
        self.star = star
        d = star.d
        self.window = Zonotope(star.N0.T)
        self.gamma = np.asarray(gamma, float)
        self.yg = star.N0 @ self.gamma
        self.R = R
        # BFS from a point near origin
        start = None
        for c in itertools.product(range(-3, 4), repeat=d):
            n = np.array(c, dtype=np.int64)
            if self.inside(n)[0] and np.linalg.norm(star.E @ n) < 3:
                start = n
                break
        assert start is not None
        seen = {tuple(start.tolist())}
        frontier = [start]
        pts = [start]
        while frontier:
            new = []
            for n in frontier:
                for m in self.neighbours_lift(n):
                    t = tuple(m.tolist())
                    if t not in seen and np.linalg.norm(star.E @ m) <= R:
                        seen.add(t)
                        new.append(m)
                        pts.append(m)
            frontier = new
        self.n = np.array(pts)
        self.x = (star.E @ self.n.T).T
        self.set = seen

    def inside(self, n):
        n = np.atleast_2d(n)
        return self.window.contains(n @ self.star.N0.T - self.yg)

    def neighbours_lift(self, n):
        d = self.star.d
        out = []
        for k in range(d):
            for s in (1, -1):
                m = n.copy()
                m[k] += s
                if self.inside(m)[0]:
                    out.append(m)
        return out


# ----------------------------------------------------------------------------
# patch keys
# ----------------------------------------------------------------------------
def patch_lifts(obj, n0, radius):
    """Lift differences of all vertices within graph distance <= radius of n0."""
    seen = {tuple(n0.tolist())}
    frontier = [n0]
    diffs = []
    for _ in range(radius):
        new = []
        for n in frontier:
            for m in obj.neighbours_lift(n):
                t = tuple(m.tolist())
                if t not in seen:
                    seen.add(t)
                    new.append(m)
                    diffs.append(m - n0)
        frontier = new
    return np.array(diffs, dtype=np.int64)


def canonical_key(diffs, group):
    best = None
    for g in group:
        t = sorted(map(tuple, (diffs @ g.T).tolist()))
        t = tuple(t)
        if best is None or t < best:
            best = t
    return best


def raw_key(diffs):
    return tuple(sorted(map(tuple, diffs.tolist())))


def build_atlas(star, R, gamma, radii=(1, 2), reduce=True):
    patch = QCPatch(star, gamma, R)
    atlases = {}
    for r in radii:
        keys = set()
        for n in patch.n:
            if np.linalg.norm(star.E @ n) <= R - r - 1.5:
                dif = patch_lifts(patch, n, r)
                keys.add(canonical_key(dif, star.group) if reduce else raw_key(dif))
        atlases[r] = keys
    return atlases, patch


def forbidden_fraction(cell, atlases):
    out = {}
    for r, atlas in atlases.items():
        bad = 0
        for n in cell.n:
            dif = patch_lifts(cell, n, r)
            if canonical_key(dif, cell.star.group) not in atlas:
                bad += 1
        out[r] = bad / cell.V
    return out


def make_gamma(star, seed):
    rng = np.random.default_rng(seed)
    g = rng.uniform(0, 1, star.d)
    if star.cls == "pen":
        # centred cube: P3 levels are the half-integer sum levels -> sum(gamma) = 1/2 exactly
        # (any eps shifts the pentagon boundaries and creates non-P3 vertices at density ~ eps)
        g = g - g.mean() + 0.5 / 5
    return g

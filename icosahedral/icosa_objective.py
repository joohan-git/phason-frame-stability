"""complete objective (corotational) geometric stiffness of the 3D rigid-joint frame.

Element energy U(q) = 1/2 d(q)^T K_nat d(q), d = natural deformations: chord extension, twist phi2x - phi1x and the four end
bending rotations, measured in a corotated frame E. Node rotations are R_i = exp(theta_i) R_i0 (spin variables).
  m = (a + b)/2, M = exp(m), p = M ex, q = chord direction, S = minimal rotation p -> q, E = S M, Q_i = E^T R_i, phi_i = vee(Q_i)
is objective to the order needed (E -> exp(w) E under a rigid rotation exp(w) to second order).
Hessian of U at the prestressed reference: K_mat + K_geo with K_geo = sum_alpha S_alpha D^2 d_alpha, S = K_nat d.
The D^2 d_alpha are computed once with exact second-order jets, so no finite differences are involved.
The axial part keeps the existing beam-column matrix N*Ge (string plus consistent bending plus Wagner torsion term), identical to the
earlier code, so that setting the non-axial forces to zero recovers it.
Rigid rotation r_w: element K_geo r_w = (w x F1, 1/2 w x M1, w x F2, 1/2 w x M2). Equilibrated nodes then sum to zero."""
from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from icosa_fast import *

NV = 12


class Jet:
    __slots__ = ('c', 'g', 'H')

    def __init__(self, c=0.0, g=None, H=None):
        self.c = float(c)
        self.g = np.zeros(NV) if g is None else g
        self.H = np.zeros((NV, NV)) if H is None else H

    @staticmethod
    def var(i):
        g = np.zeros(NV); g[i] = 1.0
        return Jet(0.0, g)

    def __add__(self, o):
        o = o if isinstance(o, Jet) else Jet(o)
        return Jet(self.c + o.c, self.g + o.g, self.H + o.H)
    __radd__ = __add__

    def __neg__(self):
        return Jet(-self.c, -self.g, -self.H)

    def __sub__(self, o):
        return self + (-(o if isinstance(o, Jet) else Jet(o)))

    def __rsub__(self, o):
        return (-self) + o

    def __mul__(self, o):
        if not isinstance(o, Jet):
            return Jet(self.c * o, self.g * o, self.H * o)
        return Jet(self.c * o.c, self.c * o.g + o.c * self.g, self.c * o.H + o.c * self.H + np.outer(self.g, o.g) + np.outer(o.g, self.g))
    __rmul__ = __mul__

    def inv(self):
        c = self.c
        return Jet(1 / c, -self.g / c ** 2, -self.H / c ** 2 + 2 * np.outer(self.g, self.g) / c ** 3)

    def sqrt(self):
        s = self.c ** .5
        return Jet(s, self.g / (2 * s), self.H / (2 * s) - np.outer(self.g, self.g) / (4 * s ** 3))


def jmat_mul(A, B):
    return [[sum((A[i][k] * B[k][j] for k in range(3)), Jet()) for j in range(3)] for i in range(3)]


def jskew(v):
    z = Jet()
    return [[z, -v[2], v[1]], [v[2], z, -v[0]], [-v[1], v[0], z]]


def jexp2(v):
    """Rotation matrix to second order in the spin vector."""
    A = jskew(v); A2 = jmat_mul(A, A)
    return [[A[i][j] + 0.5 * A2[i][j] + (1.0 if i == j else 0.0) for j in range(3)] for i in range(3)]


def jtrans(A):
    return [[A[j][i] for j in range(3)] for i in range(3)]


def jvee(Q):
    return [0.5 * (Q[2][1] - Q[1][2]), 0.5 * (Q[0][2] - Q[2][0]), 0.5 * (Q[1][0] - Q[0][1])]


def natural_jets(L):
    """Jets (in local axes, variables u1, a, u2, b) of ext, phi1 (3), phi2 (3)."""
    v = [Jet.var(i) for i in range(12)]
    u1, a, u2, b = v[0:3], v[3:6], v[6:9], v[9:12]
    c = [u2[0] - u1[0] + L, u2[1] - u1[1], u2[2] - u1[2]]
    l2 = sum((x * x for x in c), Jet())
    l = l2.sqrt(); li = l.inv()
    q = [x * li for x in c]
    ext = l - L
    m = [0.5 * (a[i] + b[i]) for i in range(3)]
    Mm = jexp2(m)
    p = [Mm[i][0] for i in range(3)]
    w = [p[1] * q[2] - p[2] * q[1], p[2] * q[0] - p[0] * q[2], p[0] * q[1] - p[1] * q[0]]
    dot = sum((p[i] * q[i] for i in range(3)), Jet())
    W = jskew(w); W2 = jmat_mul(W, W); f = (dot + 1.0).inv()
    S = [[W[i][j] + W2[i][j] * f + (1.0 if i == j else 0.0) for j in range(3)] for i in range(3)]
    E = jmat_mul(S, Mm)
    phi1 = jvee(jmat_mul(jtrans(E), jexp2(a)))
    phi2 = jvee(jmat_mul(jtrans(E), jexp2(b)))
    return ext, phi1, phi2


def natural_model(L):
    """B (6x12, first derivatives of the 6 natural coordinates), Knat (6x6), D (6,12,12) second derivatives.
    Coordinates: ext, twist, phi1y, phi1z, phi2y, phi2z."""
    ext, p1, p2 = natural_jets(L)
    coords = [ext, p2[0] - p1[0], p1[1], p1[2], p2[1], p2[2]]
    B = np.array([c.g for c in coords]); D = np.array([c.H for c in coords])
    Kn = np.zeros((6, 6))
    Kn[0, 0] = EA / L; Kn[1, 1] = GJ / L
    bl = EI / L * np.array([[4., 2.], [2., 4.]])
    Kn[np.ix_([2, 4], [2, 4])] = bl; Kn[np.ix_([3, 5], [3, 5])] = bl
    return B, Kn, D


class ObjModel:
    """Per-element geometric matrices for a loaded Frame3D (f.set_load done)."""

    def __init__(self, f, axial_only=False):
        self.f = f
        L = f.Le
        B, Kn, D = natural_model(L)
        self.B, self.Kn, self.D = B, Kn, D
        kl, _ = beam12(L)
        self.Bcheck = float(np.max(np.abs(B.T @ Kn @ B - kl)) / np.max(np.abs(kl)))
        nd = len(f.dirvecs)
        Dg = np.zeros((nd, 6, 12, 12)); self.T = np.zeros((nd, 12, 12))
        for k in range(nd):
            T = rot12(f.dirvecs[k]); self.T[k] = T
            for al in range(6):
                Dg[k, al] = T.T @ D[al] @ T
        d = self.element_disp()
        self.d = d
        ql = np.einsum('eab,eb->ea', self.T[f.dirk], d)
        dn = ql @ B.T
        Sn = dn @ Kn
        self.Sn = Sn
        self.fe = np.einsum('eab,eb->ea', f.Ke[f.dirk], d)
        self.axial_only = axial_only
        GT = f.N[:, None, None] * f.Ge[f.dirk]
        if not axial_only:
            GT = GT + np.einsum('ea,eabc->ebc', Sn[:, 1:6], Dg[f.dirk][:, 1:6])
        self.GT = GT
        e = f.dirvecs[f.dirk]
        F = self.fe[:, 6:9]; Fp = F - np.sum(F * e, 1)[:, None] * e
        self.Fp_over_N = float(np.sqrt(np.mean(np.sum(Fp ** 2, 1))) / np.sqrt(np.mean(f.N ** 2)))
        self.Mrms_over_NL = float(np.sqrt(np.mean(Sn[:, 2:] ** 2)) / (np.sqrt(np.mean(f.N ** 2)) * L))
        self.Trms_over_NL = float(np.sqrt(np.mean(Sn[:, 1] ** 2)) / (np.sqrt(np.mean(f.N ** 2)) * L))
        vals = GT.reshape(len(f.ei), 144)
        M = sp.csr_matrix((vals.ravel(), (f.rows, f.cols)), shape=(f.n, f.n))
        self.Gs = ((M + M.T) * .5).tocsr()
        self.fr = f.free_gamma

    def element_disp(self):
        f = self.f
        e = f.dirvecs[f.dirk]; xb = f.xa + e * f.Le; eps = f.eps
        dimp = np.zeros((len(f.ei), 12)); dimp[:, 0:3] = f.xa @ eps.T; dimp[:, 6:9] = xb @ eps.T
        fe = np.einsum('eab,eb->ea', f.Ke[f.dirk], dimp)
        b = np.zeros(f.n)
        np.add.at(b, (6 * f.ei[:, None] + np.arange(6)).ravel(), fe[:, :6].ravel())
        np.add.at(b, (6 * f.ej[:, None] + np.arange(6)).ravel(), fe[:, 6:].ravel())
        w = np.zeros(f.n); w[f.free_gamma] = f._lu0.solve(-b[f.free_gamma])
        d = dimp.copy()
        d[:, :6] += w[6 * f.ei[:, None] + np.arange(6)]; d[:, 6:] += w[6 * f.ej[:, None] + np.arange(6)]
        return d

    def rigid_test(self):
        """Element residual of K_geo r_w against the rotated force, and node residual; w = 3 axes."""
        f = self.f
        e = f.dirvecs[f.dirk]; xb = f.xa + e * f.Le
        el, node, nf = [], [], []
        for w in np.eye(3):
            d = np.zeros((len(e), 12))
            d[:, 0:3] = np.cross(w, f.xa); d[:, 3:6] = w; d[:, 6:9] = np.cross(w, xb); d[:, 9:12] = w
            r = np.einsum('eab,eb->ea', self.GT, d)
            F = self.fe
            exp = np.zeros_like(r)
            exp[:, 0:3] = np.cross(w, F[:, 0:3]); exp[:, 3:6] = .5 * np.cross(w, F[:, 3:6])
            exp[:, 6:9] = np.cross(w, F[:, 6:9]); exp[:, 9:12] = .5 * np.cross(w, F[:, 9:12])
            el.append(float(np.max(np.linalg.norm(r - exp, axis=1)))); nf.append(float(np.max(np.linalg.norm(exp, axis=1))))
            b = np.zeros(f.n)
            np.add.at(b, (6 * f.ei[:, None] + np.arange(6)).ravel(), r[:, :6].ravel())
            np.add.at(b, (6 * f.ej[:, None] + np.arange(6)).ravel(), r[:, 6:].ravel())
            node.append(float(np.linalg.norm(b)))
        return dict(element_max=max(el), force_scale=max(nf), node_norm=max(node))

    def halfk(self, k):
        f = self.f
        ph = np.cos(2 * np.pi * (f.wrap @ np.asarray(k, float)))
        vals = self.GT.reshape(len(f.ei), 144) * np.where(f.blk != 0, ph[:, None], 1.0)
        M = sp.csr_matrix((vals.ravel(), (f.rows, f.cols)), shape=(f.n, f.n))
        return ((M + M.T) * .5).tocsr()


class HomogObj(Homog):
    """Exact k -> 0 acoustic tensor with the complete geometric matrix."""

    def __init__(self, f, om):
        self.f = f; self.om = om
        self.fr = f.free_gamma
        self.K0 = real_assemble(f, np.ones(len(f.ei)), f.Ke)
        self.Hm = -om.Gs
        self.Delta = f.dirvecs[f.dirk] * f.Le
        self.ri = (6 * f.ei[:, None] + np.arange(6)[None, :]).ravel()
        self.rj = (6 * f.ej[:, None] + np.arange(6)[None, :]).ravel()
        self.cache = {}

    def tensor(self, lam, full=False):
        key = round(float(lam), 13)
        if key in self.cache:
            return self.cache[key]
        f = self.f; lp = lam * LAM_E
        Me = f.Ke[f.dirk] + lp * self.om.GT
        A = (self.K0 + lp * self.om.Gs).tocsr()
        S = SymSolver(A[self.fr][:, self.fr])
        G = np.zeros((3, f.n, 3))
        for j in range(3):
            dj = self.Delta[:, j]
            ci = dj[:, None, None] * Me[:, 0:6, 6:9]; cj = -dj[:, None, None] * Me[:, 6:12, 0:3]
            for a in range(3):
                G[j][:, a] = np.bincount(self.ri, ci[:, :, a].ravel(), f.n) + np.bincount(self.rj, cj[:, :, a].ravel(), f.n)
        sym = .5 * (Me[:, 0:3, 6:9] + np.transpose(Me[:, 0:3, 6:9], (0, 2, 1)))
        D = -np.einsum('ej,el,eab->jlab', self.Delta, self.Delta, sym)
        Gw = G[:, self.fr, :].transpose(1, 0, 2).reshape(len(self.fr), 9)
        X = S.solve(Gw); neg = S.neg; S.free()
        corr = (Gw.T @ X).reshape(3, 3, 3, 3)
        Lam = D - corr.transpose(0, 2, 1, 3)
        Lam = .5 * (Lam + Lam.transpose(1, 0, 2, 3)); Lam = .5 * (Lam + Lam.transpose(0, 1, 3, 2))
        out = (Lam / f.vol, neg); self.cache[key] = out
        return out


def lowest_shift(K, H, sigma, nev=4):
    """Lowest eigenvalues lam' > sigma of K v = lam' H v, by shift-invert about sigma (lam' units). Returns lam (LAM_E units), vectors, inertia."""
    S = SymSolver((K - sigma * H).tocsr())
    neg = S.neg
    op = spla.LinearOperator(K.shape, matvec=lambda x: S.solve(H @ x, refine=1), dtype=float)
    th, vec = spla.eigs(op, k=nev, which='LR', tol=1e-9, ncv=max(2 * nev + 12, 24), maxiter=4000)
    S.free()
    th = th.real; idx = np.argsort(th)[::-1]; th = th[idx]; vec = vec[:, idx].real
    lam = [float((sigma + 1 / t) / LAM_E) if t > 0 else np.inf for t in th]
    return lam, vec, neg


def lowest_auto(K, H, guess=0.90, nev=4):
    """Choose a shift below the lowest eigenvalue with inertia (neg = 0), then shift-invert."""
    s = guess
    for _ in range(12):
        St = SymSolver((K - s * LAM_E * H).tocsr()); neg = St.neg; St.free()
        if neg == 0:
            break
        s *= 0.8
    lam, vec, _ = lowest_shift(K, H, s * LAM_E, nev)
    return lam, vec, s


def frame_gamma_obj(f, om, nev=4, guess=.90):
    fr = f.free_gamma
    K = real_assemble(f, np.ones(len(f.ei)), f.Ke)[fr][:, fr].tocsr()
    H = (-om.Gs)[fr][:, fr].tocsr()
    lam, vec, s = lowest_auto(K, H, guess, nev)
    full = np.zeros((f.n, vec.shape[1])); full[fr] = vec
    return lam, full


def frame_halfk_obj(f, om, k, nev=2, guess=.90):
    K = real_assemble(f, np.ones(len(f.ei)), f.Ke, k).tocsr()
    H = (-om.halfk(k)).tocsr()
    lam, vec, s = lowest_auto(K, H, guess, nev)
    return lam, vec


def odd_part(L):
    return float(((L[0, 0, 1, 1] + L[1, 1, 2, 2] + L[2, 2, 0, 0]) - (L[0, 0, 2, 2] + L[1, 1, 0, 0] + L[2, 2, 1, 1])) / 3)


def antis_ratio(L):
    return float(np.linalg.norm(L - L.transpose(2, 3, 0, 1)) / 2 / np.linalg.norm(L))

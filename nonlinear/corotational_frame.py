"""periodic corotational planar beam frame under prescribed macroscopic hydrostatic stress.

Unknowns z = (u_x, u_y, theta at all nodes except two fixed translations of node 0; ehat[3]).
Current chord of an element: d = (I+E) d0 + u_j - u_i, with E = [[e1, e3], [e3, e2]] / s,
s = sqrt(A0), and ehat = (e1, e2, e3).  E symmetric fixes the rigid-rotation gauge; joint rotations
stay free, so a simple shear appears as pure shear plus a uniform joint rotation.

Element energy (corotational, small local strain, shallow-arch membrane strain):
    tau_a = theta_a - (beta - beta0)
    Phi   = (L/15) (2 tau1^2 - tau1 tau2 + 2 tau2^2)         = int (f')^2 dx of the cubic bow
    E     = EA/(2L) (delta + Phi/2)^2 + (EI/L)(2 tau1^2 + 2 tau1 tau2 + 2 tau2^2)
Linearised about tau = 0 this gives exactly K_e + N G_e + (M1+M2) V_e of objectivity/objective_frame.py,
where G_e is the consistent beam-column form = N z z^T / L + the Phi chain term.

Total potential under true hydrostatic pressure p (Cauchy, conjugate to the current cell area):
    Pi(z, p) = sum E + p A0 det(I+E)
All pressures inside the solver are in EA/ell; reported values are p / (EA lambda_E), lambda_E =
pi^2 EI / (EA ell^2), i.e. units of pi^2 EI / ell^3, as in the article.
"""
from __future__ import annotations
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames', 'phason_paths', 'objectivity', 'disorder')]  # shared modules
import os, math, sys, json, time
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as la

HERE = Path(__file__).resolve().parent
ROUGH = HERE.parent / 'disorder'


# ------------------------------------------------------------------ mesh
class Mesh:
    """Periodic frame. ei, ej: element ends; wrap: integer cell offset of end j, in cell rows;
    d0: reference chord (imperfect geometry included); cell: rows are lattice vectors."""

    def __init__(self, X, ei, ej, wrap, d0, cell, sub, r=0.03, EA=1.0, name=''):
        self.X = np.asarray(X, float); self.ei = np.asarray(ei); self.ej = np.asarray(ej)
        self.wrap = np.asarray(wrap, np.int64); self.d0 = np.asarray(d0, float)
        self.cell = np.asarray(cell, float); self.sub = sub; self.r = r; self.EA = EA; self.name = name
        self.EI = EA * r ** 2 / 4; self.lamE = math.pi ** 2 * self.EI / EA
        self.nn = len(self.X); self.ne = len(self.ei)
        self.n = 3 * self.nn
        self.A0 = abs(np.linalg.det(self.cell)); self.s = math.sqrt(self.A0)
        self.nu = self.n - 2                # free node dofs
        self.nz = self.nu + 3
        self._geom()
        self._pattern()

    def _geom(self):
        self.L = np.linalg.norm(self.d0, axis=1)
        self.b0 = np.arctan2(self.d0[:, 1], self.d0[:, 0])
        s = self.s; d0 = self.d0; ne = self.ne
        # T: (ne, 4, 9) from y = (ui,ti,uj,tj,e1,e2,e3) to g = (dx,dy,t1,t2)
        T = np.zeros((ne, 4, 9))
        T[:, 0, 0] = -1; T[:, 0, 3] = 1; T[:, 1, 1] = -1; T[:, 1, 4] = 1
        T[:, 2, 2] = 1; T[:, 3, 5] = 1
        T[:, 0, 6] = d0[:, 0] / s; T[:, 0, 8] = d0[:, 1] / s
        T[:, 1, 7] = d0[:, 1] / s; T[:, 1, 8] = d0[:, 0] / s
        self.T = T

    def _pattern(self):
        di = 3 * self.ei[:, None] + np.arange(3)[None, :]
        dj = 3 * self.ej[:, None] + np.arange(3)[None, :]
        self.D = np.concatenate([di, dj], 1) - 2      # free-dof index, negative => fixed
        rows = np.repeat(self.D, 6, axis=1).ravel(); cols = np.tile(self.D, (1, 6)).ravel()
        self.mask = (rows >= 0) & (cols >= 0)
        self.rows = rows[self.mask]; self.cols = cols[self.mask]
        # border pattern
        self.brows = self.D.ravel()
        self.bmask = self.brows >= 0

    def with_imperfection(self, dU=None, E0=None, name=None):
        """New mesh with stress-free reference x = (I+E0) X + dU (polygonal members)."""
        I = np.eye(2) + (np.zeros((2, 2)) if E0 is None else E0)
        d = self.d0 @ I.T
        if dU is not None:
            d = d + dU[self.ej] - dU[self.ei]
        cell = self.cell @ I.T
        X = self.X @ I.T + (0 if dU is None else dU)
        return Mesh(X, self.ei, self.ej, self.wrap, d, cell, self.sub, self.r, self.EA, name or self.name + '+imp')


def from_lifts(lifts, L1, L2, sub=2, r=0.03, name=''):
    import ab_frame as C
    edges = C.edges_from_lifts(lifts, L1, L2)
    return from_edges(lifts @ C.P.T, edges, C.P, C.P @ np.array(L1, float), C.P @ np.array(L2, float), sub, r, name)


def from_edges(XV, edges, P, a1, a2, sub, r, name):
    V = len(XV); nE = len(edges)
    ei, ej, wr, d0 = [], [], [], []
    for e, (i, j, k, w1, w2) in enumerate(edges):
        d = P[:, k]
        ids = [i] + [V + e * (sub - 1) + h for h in range(sub - 1)] + [j]
        for h in range(sub):
            ei.append(ids[h]); ej.append(ids[h + 1])
            wr.append((w1, w2) if h == sub - 1 else (0, 0)); d0.append(d / sub)
    nodes = [XV]
    if sub > 1:
        hh = np.arange(1, sub) / sub
        nodes.append((XV[edges[:, 0]][:, None, :] + hh[None, :, None] * P[:, edges[:, 2]].T[:, None, :]).reshape(-1, 2))
    return Mesh(np.vstack(nodes), ei, ej, wr, np.array(d0), np.array([a1, a2]), sub, r, name=name)


def square_mesh(sub=2, r=0.03):
    """Unit square lattice, one vertex per cell."""
    P = np.array([[1., 0.], [0., 1.]])
    edges = np.array([[0, 0, 0, 1, 0], [0, 0, 1, 0, 1]], np.int64)
    return from_edges(np.zeros((1, 2)), edges, P, P[:, 0], P[:, 1], sub, r, 'square')


def supercell(m: Mesh, S, name=None):
    """S: 2x2 integer matrix, new lattice rows = S @ cell."""
    S = np.array(S, np.int64); det = int(round(np.linalg.det(S))); assert det > 0
    adj = np.array([[S[1, 1], -S[0, 1]], [-S[1, 0], S[0, 0]]], np.int64)

    def red(c):   # c row in base-cell coordinates; returns reduced c, wrap in S rows
        w = np.floor((c @ adj) / det + 1e-12).astype(np.int64)
        return c - w @ S, w
    reps = {}; order = []
    span = int(np.abs(S).sum()) + 1
    for a in range(-span, span + 1):
        for b in range(-span, span + 1):
            c, _ = red(np.array([a, b], np.int64))
            if tuple(c) not in reps:
                reps[tuple(c)] = len(order); order.append(tuple(c))
    assert len(order) == det, (len(order), det)
    nc = len(order); nn = m.nn
    X = np.vstack([m.X + np.array(c) @ m.cell for c in order])
    ei, ej, wr, d0 = [], [], [], []
    for ci, c in enumerate(order):
        tgt = np.array(c, np.int64)[None, :] + m.wrap
        for e in range(m.ne):
            r_, w = red(tgt[e])
            ei.append(ci * nn + m.ei[e]); ej.append(reps[tuple(r_)] * nn + m.ej[e])
            wr.append(w); d0.append(m.d0[e])
    return Mesh(X, ei, ej, wr, np.array(d0), S @ m.cell, m.sub, m.r, m.EA, name or m.name + '_sc')


# ------------------------------------------------------------------ element kinematics
def element_gh(d, t1, t2, L, b0, EA, EI, hess=True):
    """Energy, gradient and Hessian in g = (dx, dy, t1, t2). Vectorised over elements."""
    l = np.hypot(d[:, 0], d[:, 1])
    r = d / l[:, None]; z = np.c_[-r[:, 1], r[:, 0]]
    c0, s0 = np.cos(b0), np.sin(b0)
    psi = np.arctan2(c0 * d[:, 1] - s0 * d[:, 0], c0 * d[:, 0] + s0 * d[:, 1])    # beta - beta0
    tau1 = t1 - psi; tau2 = t2 - psi
    delta = l - L
    aL = EA / L; bL = EI / L; f = L / 15
    Phi = f * (2 * tau1 ** 2 - tau1 * tau2 + 2 * tau2 ** 2)
    S = delta + 0.5 * Phi
    Phi1 = f * (4 * tau1 - tau2); Phi2 = f * (4 * tau2 - tau1)
    N = aL * S
    m1 = N * 0.5 * Phi1 + bL * (4 * tau1 + 2 * tau2)
    m2 = N * 0.5 * Phi2 + bL * (2 * tau1 + 4 * tau2)
    En = 0.5 * aL * S ** 2 + bL * (2 * tau1 ** 2 + 2 * tau1 * tau2 + 2 * tau2 ** 2)
    ne = len(l)
    g = np.zeros((ne, 4))
    g[:, :2] = N[:, None] * r - ((m1 + m2) / l)[:, None] * z
    g[:, 2] = m1; g[:, 3] = m2
    if not hess:
        return En, g, N, m1 + m2
    # H3 in (delta, tau1, tau2)
    H3 = np.zeros((ne, 3, 3))
    H3[:, 0, 0] = aL
    H3[:, 0, 1] = H3[:, 1, 0] = aL * 0.5 * Phi1
    H3[:, 0, 2] = H3[:, 2, 0] = aL * 0.5 * Phi2
    H3[:, 1, 1] = aL * (0.5 * Phi1) ** 2 + N * 0.5 * f * 4 + bL * 4
    H3[:, 2, 2] = aL * (0.5 * Phi2) ** 2 + N * 0.5 * f * 4 + bL * 4
    H3[:, 1, 2] = H3[:, 2, 1] = aL * 0.25 * Phi1 * Phi2 - N * 0.5 * f + bL * 2
    J = np.zeros((ne, 3, 4))
    J[:, 0, :2] = r
    J[:, 1, :2] = -z / l[:, None]; J[:, 2, :2] = -z / l[:, None]
    J[:, 1, 2] = 1; J[:, 2, 3] = 1
    H = np.einsum('eia,eij,ejb->eab', J, H3, J, optimize=True)
    zz = np.einsum('ea,eb->eab', z, z); rz = np.einsum('ea,eb->eab', r, z)
    H[:, :2, :2] += (N / l)[:, None, None] * zz + ((m1 + m2) / l ** 2)[:, None, None] * (rz + rz.transpose(0, 2, 1))
    return En, g, H, N, m1 + m2


# ------------------------------------------------------------------ system
class Model:
    def __init__(self, mesh: Mesh):
        self.m = mesh; self.free_macro = True

    def unpack(self, z):
        m = self.m
        q = np.zeros(m.n); q[2:] = z[:m.nu]
        return q, z[m.nu:]

    def gvec(self, z):
        m = self.m; q, eh = self.unpack(z)
        yn = q[m.D + 2]    # (ne, 6)
        s = m.s
        d = m.d0 + yn[:, 3:5] - yn[:, 0:2]
        d[:, 0] += (eh[0] * m.d0[:, 0] + eh[2] * m.d0[:, 1]) / s
        d[:, 1] += (eh[2] * m.d0[:, 0] + eh[1] * m.d0[:, 1]) / s
        return d, yn[:, 2], yn[:, 5]

    def area(self, z):
        m = self.m; e = z[m.nu:] / m.s
        return m.A0 * ((1 + e[0]) * (1 + e[1]) - e[2] ** 2)

    def area_grad(self, z):
        m = self.m; e = z[m.nu:] / m.s
        h = np.zeros(m.nz)
        h[m.nu:] = m.s * np.array([1 + e[1], 1 + e[0], -2 * e[2]])   # A0 / s = s
        return h

    def energy(self, z, p=0.0):
        m = self.m; d, t1, t2 = self.gvec(z)
        En, _, _, _ = element_gh(d, t1, t2, m.L, m.b0, m.EA, m.EI, hess=False)
        return En.sum() + p * self.area(z)

    def residual(self, z, p):
        m = self.m; d, t1, t2 = self.gvec(z)
        _, g, N, Ms = element_gh(d, t1, t2, m.L, m.b0, m.EA, m.EI, hess=False)
        fy = np.einsum('eai,ea->ei', m.T, g)
        R = np.zeros(m.nz)
        idx = m.D.ravel(); v = fy[:, :6].ravel(); ok = idx >= 0
        np.add.at(R, idx[ok], v[ok])
        R[m.nu:] += fy[:, 6:].sum(0)
        return R + p * self.area_grad(z)

    def tangent(self, z, p):
        m = self.m; d, t1, t2 = self.gvec(z)
        _, g, H, N, Ms = element_gh(d, t1, t2, m.L, m.b0, m.EA, m.EI)
        Hy = np.einsum('eai,eab,ebj->eij', m.T, H, m.T, optimize=True)
        return self._blocks(Hy, p, z)

    def _blocks(self, Hy, p, z):
        m = self.m
        vals = Hy[:, :6, :6].reshape(-1)[m.mask]
        Kuu = sp.csc_matrix((vals, (m.rows, m.cols)), shape=(m.nu, m.nu))
        Kuu = (Kuu + Kuu.T) * 0.5
        B = np.zeros((m.nu, 3))
        for a in range(3):
            np.add.at(B[:, a], m.brows[m.bmask], Hy[:, :6, 6 + a].reshape(-1)[m.bmask])
        C = Hy[:, 6:, 6:].sum(0)
        C = C + p * np.array([[0, 1., 0], [1, 0, 0], [0, 0, -2.]])   # A0/s^2 = 1
        return Kuu, B, 0.5 * (C + C.T)

    # --- linear (prescribed-force) operator of gate (a)
    def linear_state(self):
        """Linear equilibrium under unit pressure at the reference geometry; element N, Msum.
        Returns K blocks, the solution and the geometric blocks H (K + lambda H singular)."""
        m = self.m; z0 = np.zeros(m.nz)
        Kuu, B, C = self.tangent(z0, 0.0)
        lu = spla.splu(Kuu, permc_spec='MMD_AT_PLUS_A', options=dict(SymmetricMode=True), diag_pivot_thresh=0.0)
        Y = lu.solve(B); S = C - B.T @ Y
        hF = np.array([1., 1., 0.]) * m.s
        eh = la.solve(S, -hF)
        u = -Y @ eh
        z = np.r_[u, eh]
        # element forces from the linearised energy
        d, t1, t2 = self.gvec(z)
        dd = d - m.d0
        r0 = m.d0 / m.L[:, None]; z0v = np.c_[-r0[:, 1], r0[:, 0]]
        psi = np.sum(dd * z0v, 1) / m.L
        N = m.EA / m.L * np.sum(dd * r0, 1)
        tau1 = t1 - psi; tau2 = t2 - psi
        Ms = m.EI / m.L * 6 * (tau1 + tau2)
        # geometric matrix in g
        ne = m.ne; L = m.L
        H = np.zeros((ne, 4, 4))
        zz = np.einsum('ea,eb->eab', z0v, z0v); rz = np.einsum('ea,eb->eab', r0, z0v)
        H[:, :2, :2] = (N / L)[:, None, None] * zz + (Ms / L ** 2)[:, None, None] * (rz + rz.transpose(0, 2, 1))
        # Phi chain: N/2 d2Phi in (t1, t2, psi)
        f = L / 15
        P2 = np.zeros((ne, 2, 2))
        P2[:, 0, 0] = P2[:, 1, 1] = N * 0.5 * f * 4; P2[:, 0, 1] = P2[:, 1, 0] = -N * 0.5 * f
        Jt = np.zeros((ne, 2, 4))
        Jt[:, 0, :2] = -z0v / L[:, None]; Jt[:, 1, :2] = -z0v / L[:, None]; Jt[:, 0, 2] = 1; Jt[:, 1, 3] = 1
        H += np.einsum('eia,eij,ejb->eab', Jt, P2, Jt, optimize=True)
        Hy = np.einsum('eai,eab,ebj->eij', m.T, H, m.T, optimize=True)
        Huu, HB, HC = self._blocks(Hy, 1.0, z0)    # includes unit pressure Hessian on macro block
        return dict(Kuu=Kuu, B=B, C=C, Huu=Huu, HB=HB, HC=HC, z=z, N=N, Ms=Ms, lu=lu, Y=Y, S=S)

    def linear_critical(self, macro=True, nev=4):
        """Smallest lambda>0 with K + lambda H singular, reported / lamE; returns value, mode."""
        m = self.m; st = self.linear_state()
        lu = st['lu']; B, C, HB, HC, Huu = st['B'], st['C'], st['HB'], st['HC'], st['Huu']
        if macro:
            S = st['S']; Y = st['Y']; lus = la.cho_factor(S)

            def solve(x):
                xu, xf = x[:m.nu], x[m.nu:]
                yu = lu.solve(xu); ef = la.cho_solve(lus, xf - B.T @ yu)
                return np.r_[yu - Y @ ef, ef]

            def Hv(x):
                xu, xf = x[:m.nu], x[m.nu:]
                return np.r_[Huu @ xu + HB @ xf, HB.T @ xu + HC @ xf]

            def Kv(x):
                xu, xf = x[:m.nu], x[m.nu:]
                return np.r_[st['Kuu'] @ xu + B @ xf, B.T @ xu + C @ xf]
            nn = m.nz
        else:
            solve = lu.solve; Hv = lambda x: Huu @ x; Kv = lambda x: st['Kuu'] @ x; nn = m.nu
        A = spla.LinearOperator((nn, nn), matvec=lambda x: -Hv(x), dtype=float)
        Kop = spla.LinearOperator((nn, nn), matvec=Kv, dtype=float)
        Minv = spla.LinearOperator((nn, nn), matvec=solve, dtype=float)
        ncv = min(nn - 1, max(40, 3 * nev))
        vals, vecs = spla.eigsh(A, k=min(nev, nn - 2), M=Kop, Minv=Minv, which='LA', ncv=ncv, tol=1e-10, maxiter=20000)
        order = np.argsort(-vals)
        lams = 1 / vals[order] / m.lamE
        return lams, vecs[:, order], st


class Tangent:
    """Factorised tangent; solve and inertia (no pivoting across the symmetric ordering)."""

    def __init__(self, Kuu, B, C, m, macro=True):
        self.m = m; self.macro = macro; self.Kuu = Kuu; self.B = B; self.C = C
        self.lu = spla.splu(Kuu, permc_spec='MMD_AT_PLUS_A', options=dict(SymmetricMode=True), diag_pivot_thresh=0.0)
        if macro:
            self.Y = self.lu.solve(B)
            self.S = C - B.T @ self.Y
            self.Slu = la.lu_factor(self.S)
        self.size = m.nz if macro else m.nu

    def inertia(self):
        """Number of negative eigenvalues (Sylvester on the L D L^T of the symmetric ordering)."""
        dg = self.lu.U.diagonal()
        nn = int(np.sum(dg < 0))
        if self.macro:
            nn += int(np.sum(np.linalg.eigvalsh(0.5 * (self.S + self.S.T)) < 0))
        return nn

    def solve(self, x):
        if not self.macro:
            return self.lu.solve(x)
        xu, xf = x[:self.m.nu], x[self.m.nu:]
        yu = self.lu.solve(xu)
        ef = la.lu_solve(self.Slu, xf - self.B.T @ yu)
        return np.r_[yu - self.Y @ ef, ef]

    def mv(self, x):
        if not self.macro:
            return self.Kuu @ x
        xu, xf = x[:self.m.nu], x[self.m.nu:]
        return np.r_[self.Kuu @ xu + self.B @ xf, self.B.T @ xu + self.C @ xf]

    def lowest(self, k=4, tol=1e-9):
        n = self.size
        A = spla.LinearOperator((n, n), matvec=self.mv, dtype=float)
        O = spla.LinearOperator((n, n), matvec=self.solve, dtype=float)
        vals, vecs = spla.eigsh(A, k=k, sigma=0.0, OPinv=O, which='LM', tol=tol, ncv=min(n - 1, max(30, 3 * k)), maxiter=5000)
        o = np.argsort(vals)
        return vals[o], vecs[:, o]


# ------------------------------------------------------------------ Newton / continuation
class Stop(Exception):
    pass


def tang(model, z, p):
    Kuu, B, C = model.tangent(z, p)
    return Tangent(Kuu, B, C, model.m)


def rtol(model):
    return 1e-9 * model.m.lamE * 1e0


def newton_p(model, z, p, maxit=12, tol=None):
    """Pressure control."""
    tol = rtol(model) if tol is None else tol
    for it in range(maxit):
        R = model.residual(z, p)
        nr = np.linalg.norm(R)
        if nr < tol:
            return z, it, True
        T = tang(model, z, p)
        dz = T.solve(-R)
        # simple cap on steps: joint rotation < 0.3 rad per iteration
        cap = np.abs(dz).max()
        if cap > 1.0:
            dz *= 1.0 / cap
        z = z + dz
    return z, maxit, np.linalg.norm(model.residual(z, p)) < 100 * tol


def newton_con(model, z, p, cz, cp, g0, maxit=12, tol=None):
    """Bordered Newton with a linear constraint cz.z + cp.p = g0."""
    tol = rtol(model) if tol is None else tol
    for it in range(maxit):
        R = model.residual(z, p)
        g = cz @ z + cp * p - g0
        if np.linalg.norm(R) < tol and abs(g) < 1e-11:
            return z, p, it, True
        T = tang(model, z, p)
        h = model.area_grad(z)
        a = T.solve(-R); b = T.solve(h)
        # refine once: the block solve loses digits close to a singular T
        a += T.solve(-R - T.mv(a)); b += T.solve(h - T.mv(b))
        dnm = cz @ b - cp
        dp = (cz @ a + g) / dnm
        dz = a - b * dp
        if not np.isfinite(dp) or np.abs(dz).max() > 5.0:
            return z, p, it, False
        z = z + dz; p = p + dp
    R = model.residual(z, p)
    return z, p, maxit, np.linalg.norm(R) < 100 * tol


def measures(model, z, p, modes=None):
    m = model.m
    d, t1, t2 = model.gvec(z)
    c0, s0 = np.cos(m.b0), np.sin(m.b0)
    psi = np.arctan2(c0 * d[:, 1] - s0 * d[:, 0], c0 * d[:, 0] + s0 * d[:, 1])
    eh = z[m.nu:] / m.s
    tm = t1 * 0  # relative rotation: remove uniform joint rotation (a rigid-body part of simple shear)
    q, _ = model.unpack(z)
    th = q[2::3]; thm = th.mean()
    psirel = psi - thm
    # edge-level chord (vertex to vertex over `sub` elements): mesh-independent amplitude measure, added for gate (d)
    sub = m.sub; ned = m.ne // sub
    D = d.reshape(ned, sub, 2).sum(1); D0 = m.d0.reshape(ned, sub, 2).sum(1)
    pe = np.arctan2(D0[:, 0] * D[:, 1] - D0[:, 1] * D[:, 0], D0[:, 0] * D[:, 0] + D0[:, 1] * D[:, 1]) - thm
    tau1 = t1 - psi; tau2 = t2 - psi
    _, g, N, Ms = element_gh(d, t1, t2, m.L, m.b0, m.EA, m.EI, hess=False)
    return dict(p=p / m.lamE, A_rel=model.area(z) / m.A0 - 1, e1=eh[0], e2=eh[1], e3=eh[2],
                edev=float(math.hypot((eh[0] - eh[1]) / 2, eh[2])),
                rms_chord=float(np.sqrt(np.mean(psirel ** 2))), max_chord=float(np.abs(psirel).max()),
                rms_edge=float(np.sqrt(np.mean(pe ** 2))), max_edge=float(np.abs(pe).max()),
                max_tau=float(max(np.abs(tau1).max(), np.abs(tau2).max())), max_th=float(np.abs(th - thm).max()),
                Nmin=float(N.min()), Nmax=float(N.max()))


def follow(model, z0, p0, t0, ds0, nsteps, stop=None, dsmax=None, dsmin=None, wp=None, eig=True, log=None, k_eig=3, guard_neg=None):
    """Pseudo-arclength from (z0, p0) with initial tangent t0 = (tz, tp) (not necessarily normalised).
    wp scales pressure in the arc metric.  Records measures, inertia, lowest eigenvalue per step."""
    m = model.m
    wp = wp if wp is not None else 1.0 * m.s / m.lamE
    z, p = z0.copy(), float(p0)
    tz, tp = t0[0].copy(), float(t0[1])
    nrm = math.sqrt(tz @ tz + (wp * tp) ** 2); tz /= nrm; tp /= nrm
    ds = ds0; dsmax = dsmax or 20 * ds0; dsmin = dsmin or 1e-5 * ds0
    rows = []; states = []
    fails = 0; zprev, pprev = z, p; lastneg = 0
    for step in range(nsteps):
        zp = z + ds * tz; pp = p + ds * tp
        cz = tz; cp = wp ** 2 * tp
        g0 = cz @ zp + cp * pp
        zn, pn, it, ok = newton_con(model, zp, pp, cz, cp, g0)
        if not ok:
            ds *= 0.5; fails += 1
            if ds < dsmin:
                rows.append(dict(status='stuck', step=step, ds=ds))
                break
            continue
        # branch-hop protection: reject a corrector that moved far from the predictor or a secant that turned
        dzs = zn - z; dps = pn - p
        sec = math.sqrt(dzs @ dzs + (wp * dps) ** 2)
        corr = math.sqrt((zn - zp) @ (zn - zp) + (wp * (pn - pp)) ** 2)
        cos_sec = (dzs @ tz + wp ** 2 * dps * tp) / max(sec, 1e-30)
        if (corr > 0.5 * ds or cos_sec < 0.8) and ds > dsmin * 4:
            ds *= 0.5; fails += 1
            continue
        # new tangent
        T = tang(model, zn, pn)
        h = model.area_grad(zn)
        b = T.solve(h); b += T.solve(h - T.mv(b))
        # T tz = -h tp; orthogonal constraint with the previous tangent
        dnm = cz @ b - cp
        tpn = 1.0 / (-dnm) if abs(dnm) > 0 else tp
        tzn = -b * tpn
        nrm = math.sqrt(tzn @ tzn + (wp * tpn) ** 2)
        tzn /= nrm; tpn /= nrm
        # guard_neg: resolve the turn of an unfolded bifurcation rather than step across it onto the unstable remnant
        negn = T.inertia()
        if guard_neg is not None and negn > lastneg and ds > guard_neg:
            ds *= 0.5; fails += 1
            continue
        lastneg = negn
        # orientation is fixed by the constraint row (c.t = +1), so the follower does not reverse
        cosang = tzn @ tz + wp ** 2 * tpn * tp
        if cosang < 0.5 and ds > dsmin * 10:
            ds *= 0.5; fails += 1
            continue
        z, p = zn, pn; tz, tp = tzn, tpn
        rec = measures(model, z, p); rec['step'] = step; rec['nit'] = it; rec['ds'] = ds
        rec['neg'] = negn
        if eig:
            try:
                v, _ = lowest_gen(model, T, k=k_eig)
                rec['mu'] = v.tolist()
            except Exception as ex:
                rec['mu'] = None
        rows.append(rec); states.append((z.copy(), p))
        if log:
            log(rec)
        if stop and stop(rec, rows):
            rows[-1]['status'] = 'stop'
            break
        if it <= 3:
            ds = min(ds * 1.4, dsmax)
        elif it >= 7:
            ds *= 0.7
    return rows, states


def primary_to(model, p_target_units, nstep=20, z0=None, p0=0.0):
    """Pressure-controlled fundamental path, target in units of lamE."""
    m = model.m; ptgt = p_target_units * m.lamE
    z = np.zeros(m.nz) if z0 is None else z0.copy()
    ps = np.linspace(p0, ptgt, nstep + 1)[1:]
    zprev = None; pprev = None
    pl = p0
    for pn in ps:
        zg = z if zprev is None else z + (z - zprev) * (pn - pl) / (pl - pprev)
        zn, it, ok = newton_p(model, zg, pn, maxit=30)
        if not ok:
            raise Stop('primary Newton failed at p=%.5f' % (pn / m.lamE))
        zprev, pprev = z, pl; z = zn; pl = pn
    return z


def base_op(model):
    """K0 = tangent at (z = 0, p = 0), metric for the generalised problem T v = mu K0 v; mu ~ 1 - p/p_i."""
    if not hasattr(model, '_K0'):
        Kuu, B, Cc = model.tangent(np.zeros(model.m.nz), 0.0)
        model._K0 = Tangent(Kuu, B, Cc, model.m)
    return model._K0


def lowest_gen(model, T, k=4, tol=1e-9):
    K0 = base_op(model); n = T.size
    A = spla.LinearOperator((n, n), matvec=T.mv, dtype=float)
    M = spla.LinearOperator((n, n), matvec=K0.mv, dtype=float)
    O = spla.LinearOperator((n, n), matvec=T.solve, dtype=float)
    vals, vecs = spla.eigsh(A, k=k, M=M, sigma=0.0, OPinv=O, which='LM', tol=tol, ncv=min(n - 1, max(30, 3 * k)), maxiter=5000)
    o = np.argsort(vals)
    return vals[o], vecs[:, o]


def find_critical(model, p_guess, mode0=None, z0=None, p0=None, k=4, tolmu=2e-6, frac=0.985, maxit=60, step_frac=0.004, pwin=None):
    """First singular or closest-approach point of the path followed by pressure control from below.
    Tracks the LOWEST eigenvalue of T v = mu K0 v (mu ~ 1 - p/p_c).  Three outcomes, reported in 'kind':
      'zero'   mu reaches 0 by secant (exact bifurcation, or a fold approached to tolerance),
      'closest' mu has a positive minimum and then rises (unfolded bifurcation; natural path stays stable),
      'fold'   Newton fails with mu small and falling (limit point in p)."""
    m = model.m
    p = p_guess * frac if p0 is None else p0
    z = primary_to(model, p, nstep=20) if z0 is None else newton_p(model, z0, p * m.lamE, maxit=40)[0]
    trace = []; best = None; zprev = None; pprev = None; h = step_frac * p; kind = None; rising = 0
    for it in range(maxit):
        T = tang(model, z, p * m.lamE)
        vals, vecs = lowest_gen(model, T, k=k)
        mu = float(vals[0]); v = vecs[:, 0]
        neg = T.inertia()
        trace.append([p, mu, neg])
        cur = dict(p=p, z=z.copy(), mu=mu, mode=v.copy(), T=T, vals=vals, vecs=vecs, neg=neg)
        if best is None or abs(mu) < abs(best['mu']):
            best = cur; rising = 0
        else:
            rising += 1
        if abs(mu) < tolmu:
            kind = 'zero'; break
        if pwin is not None and p > pwin and mu > 0:
            kind = 'closest_in_window'; break
        if len(trace) >= 3 and rising >= 2 and mu > 0:
            kind = 'closest'
        if kind == 'closest' and h < 2e-5 * p:
            break
        if mu < 0:
            # crossed: secant back between the last positive and this point
            pos = [t for t in trace if t[1] > 0]
            if not pos:
                kind = 'below_start'; break
            pa, ma = pos[-1][0], pos[-1][1]
            pn = pa + (p - pa) * ma / (ma - mu)
            if abs(pn - p) < 1e-7 * p:
                kind = 'zero'; break
        elif kind == 'closest':
            # refine around the best point with a shrinking symmetric probe
            h *= 0.5
            cand = [best['p'] - h, best['p'] + h]
            pn = cand[0] if not any(abs(t[0] - cand[0]) < 1e-9 for t in trace) else cand[1]
            z = best['z']; zprev = None
        else:
            pn = p + min(0.5 * mu * p, h) if mu > 0 else p
        zg = z if zprev is None or p == pprev else z + (z - zprev) * (pn - p) / (p - pprev)
        zn, nit, ok = newton_p(model, zg, pn * m.lamE, maxit=20)
        if not ok:
            zn, nit, ok = newton_p(model, z, pn * m.lamE, maxit=30)
        if not ok:
            h *= 0.5
            if h < 1e-5 * p:
                kind = 'fold'; break
            continue
        pprev, zprev = p, z
        p, z = pn, zn
    best['trace'] = trace; best['kind'] = kind or 'maxit'; best['zero_found'] = best['kind'] == 'zero'
    return best


def channels_extrapolated(model, lams, modes, fracs=(0.97, 0.98, 0.99), k=None):
    """Each mode i of the linear operator is tracked (K0 overlap) along the fundamental path at p <= 0.99 p_lin[0];
    mu_i(p) is extrapolated quadratically to zero.  This defines p_c on the fundamental path also when the first bifurcation
    unfolds or when a channel lies above the first crossing.  z is the quadratic extrapolation of the fundamental path."""
    m = model.m; K0 = base_op(model)
    n = len(lams); k = k or max(6, n + 3)
    Ps, Zs, Mus, Vs, Ovs = [], [], [], [], []
    z = None; p0 = 0.0
    for fr in fracs:
        p = float(lams[0]) * fr
        z = primary_to(model, p, nstep=12 if z is None else 3, z0=z, p0=p0); p0 = p * m.lamE
        T = tang(model, z, p * m.lamE)
        vals, vecs = lowest_gen(model, T, k=k)
        ov = np.abs(vecs.T @ K0.mv(modes)) if modes.ndim == 2 else None
        idx = [int(np.argmax(ov[:, i])) for i in range(n)]
        Ps.append(p); Zs.append(z.copy()); Mus.append([float(vals[i]) for i in idx]); Vs.append([vecs[:, i].copy() for i in idx]); Ovs.append([float(ov[idx[i], i]) for i in range(n)])
    Ps = np.array(Ps); Mus = np.array(Mus)
    out = []
    for i in range(n):
        c = np.polyfit(Mus[:, i], Ps, 2)   # p as a function of mu, evaluated at mu = 0
        pc = float(c[2])
        # Lagrange weights for z at pc
        L = [np.prod([(pc - Ps[b]) / (Ps[a] - Ps[b]) for b in range(3) if b != a]) for a in range(3)]
        zc = sum(L[a] * Zs[a] for a in range(3))
        out.append(dict(p=pc, z=zc, mu=Mus[-1, i], mode=Vs[-1][i], vals=Mus[:, i].copy(), neg=None, kind='extrapolated', zero_found=False, overlap=Ovs[-1][i], trace=[[float(a), float(b)] for a, b in zip(Ps, Mus[:, i])]))
    return out


def macro_content(model, mode):
    """Macro amplitude of a mode as strain: bulk, B1 (e1-e2), B2 (e3) parts, normalised to the nonaffine rest."""
    m = model.m
    eh = mode[m.nu:] / m.s
    return dict(bulk=float((eh[0] + eh[1]) / 2), B1=float((eh[0] - eh[1]) / 2), B2=float(eh[2]),
                ufrac=float(np.linalg.norm(mode[:m.nu]) / np.linalg.norm(mode)))


def participation(model, mode):
    m = model.m
    q = np.zeros(m.n); q[2:] = mode[:m.nu]
    w = q[0::3] ** 2 + q[1::3] ** 2 + q[2::3] ** 2
    return float(w.sum() ** 2 / (m.nn * (w ** 2).sum() + 1e-300))


# ------------------------------------------------------------------ branches
def default_wp(m):
    return math.sqrt(m.nz) / m.lamE


def seed_branch(model, zc, pc, phi, sign, eps):
    """Two points on the branch through (zc, pc) with phi.(z - zc) = sign eps, 2 sign eps."""
    m = model.m; phi = phi / np.linalg.norm(phi)
    g = phi @ zc + sign * eps
    z1, p1, it, ok = newton_con(model, zc + sign * eps * phi, pc, phi, 0.0, g)
    if not ok:
        raise Stop('seed 1 failed')
    z2, p2, it, ok = newton_con(model, z1 + sign * eps * phi, p1, phi, 0.0, g + sign * eps)
    if not ok:
        raise Stop('seed 2 failed')
    return z1, p1, z2, p2


def run_branch(model, zc, pc, phi, sign, eps, nsteps, stop, ds_factor=1.0, eig=True, k_eig=3):
    m = model.m
    z1, p1, z2, p2 = seed_branch(model, zc, pc, phi, sign, eps)
    wp = default_wp(m)
    rows0 = [measures(model, z1, p1), measures(model, z2, p2)]
    for r_, zz_, pp_ in zip(rows0, (z1, z2), (p1, p2)):
        r_['modal'] = float((phi / np.linalg.norm(phi)) @ (zz_ - zc)); T = tang(model, zz_, pp_); r_['neg'] = T.inertia()
    rows, states = follow(model, z2, p2, (z2 - z1, p2 - p1), ds_factor * eps, nsteps, stop=stop, wp=wp, dsmax=40 * eps * ds_factor, eig=eig, k_eig=k_eig)
    for r_, (zz_, pp_) in zip([r for r in rows if 'p' in r], states):
        r_['modal'] = float((phi / np.linalg.norm(phi)) @ (zz_ - zc))
    return rows0 + rows, states

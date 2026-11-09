"""Independent geometrically nonlinear planar beam-frame solver (nonlinear verification).

Written from physics only (no reading of nonlinear/corotational_frame.py).

Element: 2D corotational Euler-Bernoulli beam (Crisfield-type). Local element is a
Hermite cubic with the consistent bending-shortening term in the axial strain,
    eps = (Ln - L0)/L0 + (2 t1^2 - t1 t2 + 2 t2^2)/30 ,
so the local geometric stiffness of the cubic element is included and the Euler load of a
pinned column converges as O(h^4). Rigid rotation of the chord is removed exactly via the
corotational frame, so the element is exactly invariant under arbitrary rigid rotation.

Periodic cell: node displacements u (periodic) plus a symmetric macroscopic strain
E = (exx, eyy, exy); F = I + E. Element end b sees u_b + E W where W is the lattice wrap
vector of the element. Load: true (Cauchy) hydrostatic pressure p, conservative potential
Pi_ext = p A0 det(F) (work conjugate to area). Pressure unit: pi^2 EI / ell^3, ell = 1.

Units: EA = 1, EI = r^2/4, r = 0.03.
"""
from __future__ import annotations
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames', 'phason_paths', 'disorder')]  # shared modules
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import math
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

EA = 1.0
R_STRUT = 0.03
EI = R_STRUT ** 2 / 4
PUNIT = math.pi ** 2 * EI          # pressure unit, ell = 1
T = 1 / math.sqrt(2)
P = np.array([[1, T, 0, -T], [0, T, 1, T]])


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# ---------------------------------------------------------------------------------------------
class Frame:
    """Planar rigid-jointed beam frame, optionally periodic.

    X      : (nn, 2) reference node positions
    ea, eb : element end nodes
    W      : (nel, 2) wrap vectors (reference end b position = X[eb] + W); zeros if open
    periodic: if True, three extra DOFs (exx, eyy, exy) are appended after the 3*nn nodal DOFs.
    A0     : reference cell area (periodic only)
    """

    def __init__(self, X, ea, eb, W=None, periodic=False, A0=None, EA=EA, EI=EI):
        self.X = np.asarray(X, float)
        self.nn = len(self.X)
        self.ea = np.asarray(ea, np.int64)
        self.eb = np.asarray(eb, np.int64)
        self.nel = len(self.ea)
        self.W = np.zeros((self.nel, 2)) if W is None else np.asarray(W, float)
        self.periodic = periodic
        self.A0 = A0
        self.EA, self.EI = EA, EI
        d0 = self.X[self.eb] + self.W - self.X[self.ea]
        self.L0 = np.hypot(d0[:, 0], d0[:, 1])
        self.beta0 = np.arctan2(d0[:, 1], d0[:, 0])
        self.ndof = 3 * self.nn + (3 if periodic else 0)
        self.iE = 3 * self.nn
        # element dof map: 6 nodal dofs (+3 strain dofs)
        cols = [3 * self.ea, 3 * self.ea + 1, 3 * self.ea + 2, 3 * self.eb, 3 * self.eb + 1, 3 * self.eb + 2]
        if periodic:
            cols += [np.full(self.nel, self.iE), np.full(self.nel, self.iE + 1), np.full(self.nel, self.iE + 2)]
        self.edof = np.stack(cols, axis=1)              # (nel, 6 or 9)
        m = self.edof.shape[1]
        self.rows_K = np.repeat(self.edof, m, axis=1).ravel()
        self.cols_K = np.tile(self.edof, (1, m)).ravel()
        # A matrix (nel, 6, m): element 6-vector = A @ (u_a, u_b, E)
        A = np.zeros((self.nel, 6, m))
        for i in range(6):
            A[:, i, i] = 1.0
        if periodic:
            A[:, 3, 6] = self.W[:, 0]; A[:, 3, 8] = self.W[:, 1]
            A[:, 4, 7] = self.W[:, 1]; A[:, 4, 8] = self.W[:, 0]
        self.A = A
        self.fixed = np.zeros(self.ndof, bool)
        self.free = np.arange(self.ndof)

    # ---- boundary conditions
    def fix(self, dofs):
        self.fixed[np.asarray(dofs, int)] = True
        self.free = np.flatnonzero(~self.fixed)

    # ---- kinematics
    def strain(self, d):
        if not self.periodic:
            return np.zeros(3)
        return d[self.iE:self.iE + 3]

    def element_state(self, d):
        """Return current chord geometry and local deformations for all elements."""
        u = d[:3 * self.nn].reshape(self.nn, 3)
        exx, eyy, exy = self.strain(d)
        ua = u[self.ea, :2]
        ub = u[self.eb, :2].copy()
        ub[:, 0] += self.W[:, 0] * exx + self.W[:, 1] * exy
        ub[:, 1] += self.W[:, 0] * exy + self.W[:, 1] * eyy
        xa = self.X[self.ea] + ua
        xb = self.X[self.eb] + self.W + ub
        dx = xb - xa
        Ln = np.hypot(dx[:, 0], dx[:, 1])
        c = dx[:, 0] / Ln; s = dx[:, 1] / Ln
        beta = np.arctan2(dx[:, 1], dx[:, 0])
        alpha = wrap_angle(beta - self.beta0)
        ubar = (Ln ** 2 - self.L0 ** 2) / (Ln + self.L0)
        t1 = u[self.ea, 2] - alpha
        t2 = u[self.eb, 2] - alpha
        return Ln, c, s, alpha, ubar, t1, t2

    def energy(self, d, p=0.0):
        Ln, c, s, alpha, ubar, t1, t2 = self.element_state(d)
        L0 = self.L0
        eps = ubar / L0 + (2 * t1 ** 2 - t1 * t2 + 2 * t2 ** 2) / 30.0
        U = 0.5 * self.EA * L0 * eps ** 2 + (2 * self.EI / L0) * (t1 ** 2 + t1 * t2 + t2 ** 2)
        Ut = float(U.sum())
        if self.periodic:
            Ut += p * PUNIT * self.A0 * self.detF(d)
        return Ut

    def detF(self, d):
        exx, eyy, exy = self.strain(d)
        return (1 + exx) * (1 + eyy) - exy ** 2

    def internal(self, d, want_K=True):
        """Internal force vector and (optionally) consistent tangent (CSC) for the full dof set."""
        Ln, c, s, alpha, ubar, t1, t2 = self.element_state(d)
        L0 = self.L0; EA_, EI_ = self.EA, self.EI
        g1 = (4 * t1 - t2) / 30.0
        g2 = (-t1 + 4 * t2) / 30.0
        eps = ubar / L0 + (2 * t1 ** 2 - t1 * t2 + 2 * t2 ** 2) / 30.0
        N = EA_ * eps
        M1 = N * L0 * g1 + (2 * EI_ / L0) * (2 * t1 + t2)
        M2 = N * L0 * g2 + (2 * EI_ / L0) * (t1 + 2 * t2)
        z0 = np.zeros_like(c); o = np.ones_like(c)
        r = np.stack([-c, -s, z0, c, s, z0], axis=1)
        z = np.stack([s, -c, z0, -s, c, z0], axis=1)
        e3 = np.stack([z0, z0, o, z0, z0, z0], axis=1)
        e6 = np.stack([z0, z0, z0, z0, z0, o], axis=1)
        B = np.stack([r, e3 - z / Ln[:, None], e6 - z / Ln[:, None]], axis=1)   # (nel,3,6)
        fl = np.stack([N, M1, M2], axis=1)
        fg = np.einsum('eij,ei->ej', B, fl)                                       # (nel,6)
        f9 = np.einsum('eij,ei->ej', self.A, fg)
        f = np.zeros(self.ndof)
        np.add.at(f, self.edof.ravel(), f9.ravel())
        if not want_K:
            return f, None
        Kl = np.zeros((self.nel, 3, 3))
        Kl[:, 0, 0] = EA_ / L0
        Kl[:, 0, 1] = Kl[:, 1, 0] = EA_ * g1
        Kl[:, 0, 2] = Kl[:, 2, 0] = EA_ * g2
        Kl[:, 1, 1] = EA_ * L0 * g1 ** 2 + N * L0 * 4 / 30.0 + 4 * EI_ / L0
        Kl[:, 2, 2] = EA_ * L0 * g2 ** 2 + N * L0 * 4 / 30.0 + 4 * EI_ / L0
        Kl[:, 1, 2] = Kl[:, 2, 1] = EA_ * L0 * g1 * g2 - N * L0 / 30.0 + 2 * EI_ / L0
        Kg = np.einsum('eki,ekl,elj->eij', B, Kl, B)
        Kg += (N / Ln)[:, None, None] * np.einsum('ei,ej->eij', z, z)
        rz = np.einsum('ei,ej->eij', r, z)
        Kg += ((M1 + M2) / Ln ** 2)[:, None, None] * (rz + rz.transpose(0, 2, 1))
        K9 = np.einsum('eki,ekl,elj->eij', self.A, Kg, self.A)
        K = sp.coo_matrix((K9.ravel(), (self.rows_K, self.cols_K)), shape=(self.ndof, self.ndof)).tocsc()
        return f, K

    # ---- load (true hydrostatic pressure, conservative potential p*A0*det F)
    def load_grad(self, d):
        """d(Pi_ext)/d(dofs) per unit p_hat (only strain dofs nonzero)."""
        g = np.zeros(self.ndof)
        if self.periodic:
            exx, eyy, exy = self.strain(d)
            g[self.iE:self.iE + 3] = PUNIT * self.A0 * np.array([1 + eyy, 1 + exx, -2 * exy])
        return g

    def load_hess(self, p):
        if not self.periodic:
            return sp.csc_matrix((self.ndof, self.ndof))
        H = np.zeros((3, 3)); H[0, 1] = H[1, 0] = 1.0; H[2, 2] = -2.0
        H *= p * PUNIT * self.A0
        i = self.iE + np.arange(3)
        return sp.coo_matrix((H.ravel(), (np.repeat(i, 3), np.tile(i, 3))), shape=(self.ndof, self.ndof)).tocsc()

    def residual(self, d, p, want_K=True):
        f, K = self.internal(d, want_K)
        Rv = f + p * self.load_grad(d)
        if want_K:
            K = K + self.load_hess(p)
        return Rv, K

    def dR_dp(self, d):
        return self.load_grad(d)

    # ---- reduced helpers
    def Kff(self, K):
        fr = self.free
        return K[fr][:, fr].tocsc()


# ---------------------------------------------------------------------------------------------
def det_sign(lu):
    """Sign of det from a SuperLU factorisation (parity of the inertia)."""
    du = lu.U.diagonal(); dl = lu.L.diagonal()
    sgn = np.prod(np.sign(du)) * np.prod(np.sign(dl))

    def perm_parity(pm):
        pm = np.asarray(pm); n = len(pm); seen = np.zeros(n, bool); par = 1
        for i in range(n):
            if not seen[i]:
                j = i; cyc = 0
                while not seen[j]:
                    seen[j] = True; j = pm[j]; cyc += 1
                if cyc % 2 == 0:
                    par = -par
        return par
    return sgn * perm_parity(lu.perm_r) * perm_parity(lu.perm_c)


def lowest_eigs(Kff, k=4, lu=None, dense_limit=2500):
    """Lowest-magnitude eigenvalues of the symmetric reduced tangent (sorted ascending) and vectors."""
    n = Kff.shape[0]
    if n <= dense_limit:
        w, v = np.linalg.eigh(Kff.toarray())
        idx = np.argsort(w)[:k]
        return w[idx], v[:, idx]
    if lu is None:
        lu = spla.splu(Kff)
    op = spla.LinearOperator(Kff.shape, matvec=lu.solve, dtype=float)
    w, v = spla.eigsh(Kff, k=k, sigma=0.0, which='LM', OPinv=op, tol=1e-10, maxiter=5000)
    idx = np.argsort(w)
    return w[idx], v[:, idx]


def newton(frame, d, p, tol=1e-10, maxit=30, verbose=False):
    """Newton at fixed load p (reduced system). Returns (d, converged, iterations)."""
    fr = frame.free
    for it in range(maxit):
        Rv, K = frame.residual(d, p)
        Rf = Rv[fr]
        scale = max(1.0, np.abs(Rv).max()) if it == 0 else scale
        nr = np.abs(Rf).max()
        if verbose:
            print(f"   newton it {it}: |R|max={nr:.3e}")
        if nr < tol:
            return d, True, it
        lu = spla.splu(frame.Kff(K))
        dd = lu.solve(-Rf)
        d = d.copy(); d[fr] += dd
    Rv, _ = frame.residual(d, p, want_K=False)
    return d, bool(np.abs(Rv[fr]).max() < tol * 100), maxit


def linear_buckling(frame, d0=None, p_b=0.02, k=4):
    """Classical linearised threshold: K(p) ~ K0 + p K1, K1 by secant on the fundamental path."""
    fr = frame.free
    d0 = np.zeros(frame.ndof) if d0 is None else d0
    _, K0 = frame.residual(d0, 0.0)
    db, ok, _ = newton(frame, d0, p_b)
    assert ok
    _, Kb = frame.residual(db, p_b)
    K1 = (Kb - K0) / p_b
    K0f = frame.Kff(K0); K1f = frame.Kff(K1)
    n = K0f.shape[0]
    if n <= 2500:
        import scipy.linalg as la
        w = la.eigh(-K1f.toarray(), K0f.toarray(), eigvals_only=True)
        mu = np.sort(w)[::-1][:k]
    else:
        lu0 = spla.splu(K0f)
        Minv = spla.LinearOperator(K0f.shape, matvec=lu0.solve, dtype=float)
        mu, _ = spla.eigsh(-K1f, k=k, M=K0f, Minv=Minv, which='LA', tol=1e-10, maxiter=10000)
        mu = np.sort(mu)[::-1]
    with np.errstate(divide='ignore'):
        pcs = 1.0 / mu
    return pcs, db


def linear_buckling_classical(frame, k=4):
    """Classical linearised buckling: undeformed geometry, axial forces N from the linear solve at
    unit p, consistent beam-column geometric stiffness (cubic) + string term + load Hessian. No moments,
    no prebuckling rotation or shortening."""
    fr = frame.free
    d0 = np.zeros(frame.ndof)
    _, K0 = frame.residual(d0, 0.0)
    K0f = frame.Kff(K0)
    lu0 = spla.splu(K0f)
    q = frame.load_grad(d0)[fr]
    dlin = np.zeros(frame.ndof); dlin[fr] = lu0.solve(-q)       # linear response per unit p_hat
    eps_ = 1e-7
    Ln, c, s, alpha, ubar, t1, t2 = frame.element_state(eps_ * dlin)
    N = frame.EA * (ubar / frame.L0) / eps_                      # linear axial force per unit p_hat
    L0 = frame.L0; EI_ = frame.EI
    z0 = np.zeros_like(c); o = np.ones_like(c)
    c0 = np.cos(frame.beta0); s0 = np.sin(frame.beta0)
    r = np.stack([-c0, -s0, z0, c0, s0, z0], axis=1)
    z = np.stack([s0, -c0, z0, -s0, c0, z0], axis=1)
    e3 = np.stack([z0, z0, o, z0, z0, z0], axis=1); e6 = np.stack([z0, z0, z0, z0, z0, o], axis=1)
    B = np.stack([r, e3 - z / L0[:, None], e6 - z / L0[:, None]], axis=1)
    Kl = np.zeros((frame.nel, 3, 3))
    Kl[:, 1, 1] = Kl[:, 2, 2] = N * L0 * 4 / 30.0
    Kl[:, 1, 2] = Kl[:, 2, 1] = -N * L0 / 30.0
    Kg = np.einsum('eki,ekl,elj->eij', B, Kl, B) + (N / L0)[:, None, None] * np.einsum('ei,ej->eij', z, z)
    K9 = np.einsum('eki,ekl,elj->eij', frame.A, Kg, frame.A)
    KG = sp.coo_matrix((K9.ravel(), (frame.rows_K, frame.cols_K)), shape=(frame.ndof, frame.ndof)).tocsc()
    KG = KG + frame.load_hess(1.0)
    KGf = frame.Kff(KG)
    n = K0f.shape[0]
    if n <= 2500:
        import scipy.linalg as la
        w = la.eigh(-KGf.toarray(), K0f.toarray(), eigvals_only=True)
        mu = np.sort(w)[::-1][:k]
    else:
        Minv = spla.LinearOperator(K0f.shape, matvec=lu0.solve, dtype=float)
        mu, _ = spla.eigsh(-KGf, k=k, M=K0f, Minv=Minv, which='LA', tol=1e-10, maxiter=10000)
        mu = np.sort(mu)[::-1]
    with np.errstate(divide='ignore'):
        return 1.0 / mu


def element_measures(fr, d):
    """Element (sub-edge) chord rotation and local end rotations."""
    Ln, c, s, alpha, ubar, t1, t2 = fr.element_state(d)
    return dict(max_elem_chord=float(np.abs(alpha).max()), rms_elem_chord=float(np.sqrt(np.mean(alpha ** 2))),
                max_tau=float(max(np.abs(t1).max(), np.abs(t2).max())))


class ArcLength:
    """Riks normal-plane arc-length continuation in (d_free, p)."""

    def __init__(self, frame, psi=None, tol=1e-9, maxit=12, target_it=4, verbose=False):
        self.f = frame; self.psi = psi; self.tol = tol; self.maxit = maxit
        self.target_it = target_it; self.verbose = verbose

    def tangent(self, d, p, lu=None):
        fr = self.f.free
        if lu is None:
            Rv, K = self.f.residual(d, p)
            lu = spla.splu(self.f.Kff(K))
        q = self.f.dR_dp(d)[fr]
        dpdir = lu.solve(-q)
        return dpdir, lu

    def step(self, d, p, dl, prev, lu=None):
        """One arc-length step. prev = (Dd_prev, Dp_prev) or None. Returns (d, p, Dd, Dp, iters, lu, ok)."""
        f = self.f; fr = f.free
        dpdir, lu = self.tangent(d, p, lu)
        if self.psi is None:
            self.psi = float(np.linalg.norm(dpdir))
        nrm = math.sqrt(dpdir @ dpdir + self.psi ** 2)
        sgn = 1.0
        if prev is not None:
            sgn = 1.0 if (prev[0] @ dpdir + self.psi ** 2 * prev[1]) >= 0 else -1.0
        Dd = sgn * dl * dpdir / nrm; Dp = sgn * dl / nrm
        Dd0, Dp0 = Dd.copy(), Dp
        dn = d.copy(); dn[fr] += Dd; pn = p + Dp
        for it in range(self.maxit):
            Rv, K = f.residual(dn, pn)
            Rf = Rv[fr]
            nr = np.abs(Rf).max()
            if self.verbose:
                print(f"     it {it}: |R|={nr:.2e} p={pn:.6f}")
            if nr < self.tol:
                return dn, pn, Dd, Dp, it, None, True
            lu = spla.splu(f.Kff(K))
            d1 = lu.solve(-Rf)
            d2 = lu.solve(-f.dR_dp(dn)[fr])
            dp = -(Dd0 @ d1) / (Dd0 @ d2 + self.psi ** 2 * Dp0)
            dd = d1 + dp * d2
            Dd += dd; Dp += dp
            dn[fr] += dd; pn += dp
            if not np.isfinite(nr) or abs(Dp) > 5:
                break
        return d, p, None, None, self.maxit, None, False

    def run(self, d, p, dl, nmax, monitor, stop, dl_min=1e-7, dl_max=None, prev=None, accept=None):
        """Continue until stop(state) is True. monitor(state) records. Returns list of states.
        accept(d_old, d_new) -> bool may reject a converged step (too large a change); it is then redone with dl/2."""
        states = []
        for n in range(nmax):
            while True:
                dn, pn, Dd, Dp, it, _, ok = self.step(d, p, dl, prev)
                if ok and (accept is None or accept(d, dn)):
                    break
                dl *= 0.5
                if dl < dl_min:
                    raise RuntimeError("arc-length step failed (dl < dl_min)")
            d, p, prev = dn, pn, (Dd, Dp)
            st = monitor(d, p, Dp, dl, n)
            states.append(st)
            if stop(st, states):
                break
            if it <= 2:
                dl *= 1.5
            elif it > self.target_it + 1:
                dl *= 0.6
            if dl_max is not None:
                dl = min(dl, dl_max)
        return states, d, p, prev, dl


# ---------------------------------------------------------------------------------------------
def build_periodic(lifts, L1, L2, edges, sub):
    """Periodic frame from a rhombus tiling (lifts in Z^4, edges (i,j,k,w0,w1))."""
    lifts = np.asarray(lifts, np.int64)
    V = len(lifts)
    X = (P @ lifts.T).T
    a1 = P @ np.asarray(L1, float); a2 = P @ np.asarray(L2, float)
    A0 = abs(a1[0] * a2[1] - a1[1] * a2[0])
    cell = np.array([a1, a2])
    nodes = [X]
    ea, eb, W = [], [], []
    nxt = V
    edge_elems = []   # per edge: (i, j, Wedge)
    for (i, j, k, w0, w1) in edges:
        ek = P[:, k]
        Wedge = w0 * a1 + w1 * a2
        # check consistency of wrap convention
        chain = [i]
        mids = X[i] + np.outer(np.arange(1, sub) / sub, ek)
        for m in range(sub - 1):
            nodes.append(mids[m:m + 1]); chain.append(nxt); nxt += 1
        chain.append(j)
        for q in range(sub):
            ea.append(chain[q]); eb.append(chain[q + 1])
            W.append(Wedge if q == sub - 1 else np.zeros(2))
        edge_elems.append((i, j, Wedge))
    Xall = np.vstack(nodes)
    fr = Frame(Xall, ea, eb, np.array(W), periodic=True, A0=A0)
    # check edge geometry: reference chord must be unit and along e_k
    EI_ = np.array([e[0] for e in edge_elems]); EJ = np.array([e[1] for e in edge_elems])
    EW = np.array([e[2] for e in edge_elems])
    chord0 = X[EJ] + EW - X[EI_]
    assert np.allclose(np.hypot(chord0[:, 0], chord0[:, 1]), 1.0, atol=1e-9), "edge wrap convention wrong"
    fr.edge_i, fr.edge_j, fr.edge_W, fr.chord0 = EI_, EJ, EW, chord0
    fr.V = V; fr.cell = cell
    fr.fix([0, 1])     # remove rigid translations
    return fr


def chord_rotations(fr, d):
    """Rotation angle of every edge chord (vertex to vertex) relative to its reference direction."""
    u = d[:3 * fr.nn].reshape(fr.nn, 3)
    exx, eyy, exy = fr.strain(d)
    Em = np.array([[exx, exy], [exy, eyy]])
    xi = fr.X[fr.edge_i] + u[fr.edge_i, :2]
    xj = fr.X[fr.edge_j] + fr.edge_W + u[fr.edge_j, :2] + fr.edge_W @ Em
    ch = xj - xi
    c0 = fr.chord0
    return np.arctan2(c0[:, 0] * ch[:, 1] - c0[:, 1] * ch[:, 0], c0[:, 0] * ch[:, 0] + c0[:, 1] * ch[:, 1])


def square_lattice(nx, ny, sub):
    """nx x ny periodic square lattice of unit edges."""
    V = nx * ny
    idx = lambda i, j: (i % nx) * ny + (j % ny)
    X = np.array([[i, j] for i in range(nx) for j in range(ny)], float)
    edges = []
    for i in range(nx):
        for j in range(ny):
            edges.append((idx(i, j), idx(i + 1, j), 0, 1 if i + 1 == nx else 0, 0))
            edges.append((idx(i, j), idx(i, j + 1), 2, 0, 1 if j + 1 == ny else 0))
    # emulate lifts: use P columns 0 (x) and 2 (y): lifts (i, 0, j, 0) -> X = (i, j)
    lifts = np.array([[i, 0, j, 0] for i in range(nx) for j in range(ny)], np.int64)
    L1 = np.array([nx, 0, 0, 0]); L2 = np.array([0, 0, ny, 0])
    return build_periodic(lifts, L1, L2, edges, sub)

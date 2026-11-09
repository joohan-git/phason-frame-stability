"""cell_frame.py -- independent rigid-jointed Euler-Bernoulli beam-frame code.

Periodic frame from a Cell (vertices x, edges (i, j, a, b, k) with the image
of j displaced by a*T1 + b*T2). Two elements per edge. Static homogenization
with three macroscopic strain degrees of freedom (zero-load tangent tensor,
bulk modulus, equilibrium axial forces under hydrostatic stress), Bloch
eigen-buckling (K_E(k) + lambda K_G(k)) with shift-invert on the pencil,
long-wave Richardson extrapolation, zone-centre and zone-boundary checks.
Written without reference to the project's main codes.
"""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames',)]  # shared modules
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

TOVER = 0.03  # slenderness t / l on the edge length (l = 1)
EMOD = 1.0
AREA = TOVER
INERTIA = TOVER**3 / 12.0
PEULER = np.pi**2 * EMOD * INERTIA  # pi^2 EI / l^3 with l = 1: pressure unit
SECTION = "rect"


def set_section(kind):
    """'rect': rectangular strip of thickness t = 0.03 (I/A = t^2/12, my registered reading).
    'circ': circular strut of radius r = 0.03 (I/A = r^2/4, the main codes' reading)."""
    global AREA, INERTIA, PEULER, SECTION
    if kind == "rect":
        AREA, INERTIA = TOVER, TOVER**3 / 12.0
    elif kind == "circ":
        AREA, INERTIA = np.pi * TOVER**2, np.pi * TOVER**4 / 4
    else:
        raise ValueError(kind)
    PEULER = np.pi**2 * EMOD * INERTIA
    SECTION = kind


def beam_local(L, EA, EI):
    k = np.zeros((6, 6))
    a = EA / L
    b = 12 * EI / L**3
    c = 6 * EI / L**2
    d = 4 * EI / L
    e = 2 * EI / L
    k[0, 0] = k[3, 3] = a
    k[0, 3] = k[3, 0] = -a
    k[1, 1] = k[4, 4] = b
    k[1, 4] = k[4, 1] = -b
    k[1, 2] = k[2, 1] = k[1, 5] = k[5, 1] = c
    k[2, 4] = k[4, 2] = k[4, 5] = k[5, 4] = -c
    k[2, 2] = k[5, 5] = d
    k[2, 5] = k[5, 2] = e
    return k


def geom_local(L, N, axial_terms=True):
    """Consistent beam-column geometric stiffness for axial force N (tension > 0)."""
    g = np.zeros((6, 6))
    f = N / L
    if axial_terms:
        g[0, 0] = g[3, 3] = f
        g[0, 3] = g[3, 0] = -f
    g[1, 1] = g[4, 4] = 6 / 5 * f
    g[1, 4] = g[4, 1] = -6 / 5 * f
    g[1, 2] = g[2, 1] = g[1, 5] = g[5, 1] = L / 10 * f
    g[2, 4] = g[4, 2] = g[4, 5] = g[5, 4] = -L / 10 * f
    g[2, 2] = g[5, 5] = 2 * L**2 / 15 * f
    g[2, 5] = g[5, 2] = -L**2 / 30 * f
    return g


def rotation(t):
    c, s = t
    R = np.eye(6)
    R[0, 0] = R[1, 1] = R[3, 3] = R[4, 4] = c
    R[0, 1] = R[3, 4] = s
    R[1, 0] = R[4, 3] = -s
    return R


class Frame:
    def __init__(self, cell, nelem=2, axial_terms=True):
        self.cell = cell
        self.T = cell.T  # 2x2 columns T1, T2
        self.area = cell.area
        V = cell.V
        x = cell.x
        E = cell.star.E
        nodes = [x[i] for i in range(V)]
        elems = []  # (p, q, shift(2,), tvec, L)
        self.nelem = nelem
        for (i, j, a, b, k) in cell.edges:
            t = E[:, k]
            prev = i
            prev_pos = x[i]
            for m in range(1, nelem):
                pos = x[i] + t * m / nelem
                nodes.append(pos)
                q = len(nodes) - 1
                elems.append((prev, q, np.zeros(2), t, 1.0 / nelem))
                prev = q
            elems.append((prev, j, np.array([a, b], float), t, 1.0 / nelem))
        self.nodes = np.array(nodes)
        self.nn = len(nodes)
        self.V = V
        self.elems = elems
        self.ndof = 3 * self.nn
        # element matrices in global frame
        kes, dofs, shifts, tvecs = [], [], [], []
        for (p, q, s, t, L) in elems:
            R = rotation(t)
            ke = R.T @ beam_local(L, EMOD * AREA, EMOD * INERTIA) @ R
            kes.append(ke)
            dofs.append(np.array([3 * p, 3 * p + 1, 3 * p + 2, 3 * q, 3 * q + 1, 3 * q + 2]))
            shifts.append(s)
            tvecs.append(t)
        self.ke = np.array(kes)  # ne x 6 x 6
        self.dofs = np.array(dofs)
        self.shifts = np.array(shifts)  # ne x 2 (integer multiples of T1, T2)
        self.tvec = np.array(tvecs)
        self.Lel = np.array([e[4] for e in elems])
        self.shift_xy = self.shifts @ self.T.T  # ne x 2 physical shift vectors
        self.axial_terms = axial_terms
        self.rows = np.repeat(self.dofs, 6, axis=1).reshape(len(elems), 36)
        self.cols = np.tile(self.dofs, (1, 6)).reshape(len(elems), 36)
        self.N = None
        self.kg = None

    # ------------------------------------------------------------------
    def assemble(self, mats, kvec=None):
        """Bloch-assemble per-element 6x6 matrices (ne x 6 x 6)."""
        ne = len(self.elems)
        if kvec is None or not np.any(kvec):
            data = mats.reshape(ne, 36)
            A = sp.coo_matrix((data.ravel(), (self.rows.ravel(), self.cols.ravel())),
                              shape=(self.ndof, self.ndof)).tocsc()
            return A
        ph = np.exp(1j * (self.shift_xy @ kvec))  # phase of node q
        phase = np.ones((ne, 6), complex)
        phase[:, 3:] = ph[:, None]
        # K_pq picks phase(q)/phase(p): entry (r, c) gets conj(phase_r) * phase_c
        full = np.conj(phase)[:, :, None] * mats * phase[:, None, :]
        A = sp.coo_matrix((full.reshape(ne, 36).ravel(), (self.rows.ravel(), self.cols.ravel())),
                          shape=(self.ndof, self.ndof)).tocsc()
        return A

    # ------------------------------------------------------------------
    def homogenize(self, p0=None):
        if p0 is None:
            p0 = PEULER
        """Static periodic homogenization at zero load.
        Returns dict with tangent tensor components, K, shear sector, and sets
        axial forces N (per element) for macroscopic stress -p0 I."""
        ne = len(self.elems)
        K0 = self.assemble(self.ke)
        # B_e: affine part of element dofs from macro strain (e1, e2, e3) = (exx, eyy, exy)
        # u = eps . X with X the actual end position
        B = np.zeros((ne, 6, 3))
        for a, (p, q, s, t, L) in enumerate(self.elems):
            Xp = self.nodes[p]
            Xq = self.nodes[q] + self.shift_xy[a]
            for (row, X) in ((0, Xp), (3, Xq)):
                B[a, row, 0] = X[0]
                B[a, row, 2] = X[1]
                B[a, row + 1, 1] = X[1]
                B[a, row + 1, 2] = X[0]
        # coupling Kc = sum A_e^T ke B_e  (ndof x 3), Kee = sum B^T ke B
        keB = np.einsum("eij,ejk->eik", self.ke, B)  # ne x 6 x 3
        Kc = np.zeros((self.ndof, 3))
        np.add.at(Kc, self.dofs.ravel(), keB.reshape(-1, 3))
        Kee = np.einsum("eij,eik->jk", B, keB)
        # pin translations of node 0
        free = np.ones(self.ndof, bool)
        free[[0, 1]] = False
        idx = np.nonzero(free)[0]
        Kff = K0[idx][:, idx].tocsc()
        lu = spla.splu(Kff)
        X = lu.solve(Kc[idx])  # ndof_f x 3
        Cred = Kee - Kc[idx].T @ X  # energy = 1/2 eps^T Cred eps, with A eps:C:eps
        Cm = Cred / self.area  # [[C1111, C1122, 2C1112],[.,C2222,2C2212],[.,.,4C1212]]
        C1111, C1122, C2212 = Cm[0, 0], Cm[0, 1], Cm[1, 2] / 2
        C2222, C1112, C1212 = Cm[1, 1], Cm[0, 2] / 2, Cm[2, 2] / 4
        Kbulk = (C1111 + 2 * C1122 + C2222) / 4
        C11 = (C1111 - 2 * C1122 + C2222) / 2  # T1:C:T1
        C22 = 2 * C1212  # T2:C:T2
        C12 = C1112 - C2212  # T1:C:T2
        E2a = (C1111 - C2222) / 2  # trace-traceless couplings (E2 sector)
        E2b = C1112 + C2212
        # equilibrium under macroscopic stress -p0 I: generalized force A*(-p0,-p0,0)
        f = self.area * np.array([-p0, -p0, 0.0])
        eps = np.linalg.solve(Cred, f)
        ufl = np.zeros(self.ndof)
        ufl[idx] = -X @ eps
        # element axial forces
        d = ufl[self.dofs] + np.einsum("eik,k->ei", B, eps)  # ne x 6
        du = d[:, 3:5] - d[:, 0:2]
        elong = np.einsum("ei,ei->e", du, self.tvec)
        self.N = EMOD * AREA * elong / self.Lel
        self.eps_hydro = eps
        kg = []
        for a in range(ne):
            R = rotation(self.tvec[a])
            kg.append(R.T @ geom_local(self.Lel[a], self.N[a], self.axial_terms) @ R)
        self.kg = np.array(kg)
        # virial check: sum N L t t = p0 A I approx (sign: compression N<0)
        vir = -np.einsum("e,e,ei,ej->ij", self.N, self.Lel, self.tvec, self.tvec) / self.area
        self.C = dict(C1111=C1111, C1122=C1122, C2222=C2222, C1112=C1112, C2212=C2212, C1212=C1212,
                      K=Kbulk, C11=C11, C22=C22, C12=C12, E2a=E2a, E2b=E2b,
                      mu_mean=(C11 + C22) / 4, areal_strain=eps[0] + eps[1],
                      virial_trace=float(np.trace(vir)) / (2 * p0), N_mean=float(self.N.mean()),
                      N_cv=float(self.N.std() / abs(self.N.mean())))
        return self.C

    # ------------------------------------------------------------------
    def critical_factor(self, kvec, nev=4, lam_guess=0.4, gamma_penalty=False):
        """Smallest lambda > 0 with K_E(k) + lambda K_G(k) singular (units of p0)."""
        KE = self.assemble(self.ke, kvec)
        KG = self.assemble(self.kg, kvec)
        if gamma_penalty:
            # remove the two rigid translations at k = 0
            T = np.zeros((self.ndof, 2))
            T[0::3, 0] = 1
            T[1::3, 1] = 1
            T /= np.sqrt(self.nn)
            KE = (KE + sp.csc_matrix(T @ T.T) * 1.0).tocsc()
        sigma = -1.0 / lam_guess
        vals, vecs = spla.eigsh(KG, k=nev, M=KE, sigma=sigma, which="LM", tol=1e-10)
        lam = np.array([-1.0 / v if v < 0 else np.inf for v in vals])
        order = np.argsort(lam)
        return lam[order], vecs[:, order]

    def longwave_stiffness(self, kvec, p, nev=2):
        """Lowest eigenvalues of K_E(k) + p K_G(k) and the mean squared vertex
        translation amplitude of the lowest eigenvector (for normalization)."""
        H = self.assemble(self.ke + p * self.kg, kvec)
        vals, vecs = spla.eigsh(H, k=nev, sigma=0.0, which="LM", tol=1e-10)
        o = np.argsort(vals)
        vals, vecs = vals[o], vecs[:, o]
        v = vecs[:, 0]
        ux = v[0:3 * self.V:3]
        uy = v[1:3 * self.V:3]
        ubar2 = float(np.mean(np.abs(ux)**2 + np.abs(uy)**2))
        return vals, ubar2


    # ------------------------------------------------------------------
    # exact long-wave limit: homogenized tangent of K_E + p K_G over the full
    # displacement gradient H = (Hxx, Hyy, Hxy, Hyx); acoustic tensor
    # Q_ij(n, p) = n_k A_ikjl n_l; the long-wave threshold in direction theta is
    # the smallest p > 0 at which Q(n(theta), p) loses positive definiteness.
    def _Bfull(self):
        if getattr(self, "_Bf", None) is None:
            ne = len(self.elems)
            B = np.zeros((ne, 6, 4))
            for a, (p, q, s, t, L) in enumerate(self.elems):
                Xp = self.nodes[p]
                Xq = self.nodes[q] + self.shift_xy[a]
                for (row, X) in ((0, Xp), (3, Xq)):
                    B[a, row, 0] = X[0]      # Hxx
                    B[a, row, 2] = X[1]      # Hxy
                    B[a, row + 1, 1] = X[1]  # Hyy
                    B[a, row + 1, 3] = X[0]  # Hyx
            self._Bf = B
            free = np.ones(self.ndof, bool)
            free[[0, 1]] = False
            self._free = np.nonzero(free)[0]
        return self._Bf

    def tangent(self, p):
        """4x4 homogenized tangent A (energy density 1/2 H:A:H) of K_E + p K_G."""
        B = self._Bfull()
        mats = self.ke + p * self.kg if p != 0 else self.ke
        K0 = self.assemble(mats)
        keB = np.einsum("eij,ejk->eik", mats, B)
        Kc = np.zeros((self.ndof, 4))
        np.add.at(Kc, self.dofs.ravel(), keB.reshape(-1, 4))
        Kee = np.einsum("eij,eik->jk", B, keB)
        idx = self._free
        Kff = K0[idx][:, idx].tocsc()
        lu = spla.splu(Kff)
        X = lu.solve(Kc[idx])
        return (Kee - Kc[idx].T @ X) / self.area

    @staticmethod
    def acoustic(A, n):
        """Q_ij = n_k A_{ik,jl} n_l with H index order (xx, yy, xy, yx)."""
        # map (i,k) -> index: (0,0)->0, (1,1)->1, (0,1)->2, (1,0)->3
        idx = {(0, 0): 0, (1, 1): 1, (0, 1): 2, (1, 0): 3}
        Q = np.zeros((2, 2))
        for i in range(2):
            for j in range(2):
                sacc = 0.0
                for k in range(2):
                    for l in range(2):
                        sacc += n[k] * A[idx[(i, k)], idx[(j, l)]] * n[l]
                Q[i, j] = sacc
        return Q

    def longwave_exact(self, directions, p_hint=0.8, tol=1e-8, maxit=30):
        """Long-wave thresholds lambda(theta) from the homogenized acoustic
        tensor, by secant iteration in p (A(p) evaluations cached and shared)."""
        cache = {}

        def A_of(p):
            key = round(p, 12)
            if key not in cache:
                cache[key] = self.tangent(p)
            return cache[key]

        def f(p, n):
            Q = self.acoustic(A_of(p), n)
            w, v = np.linalg.eigh(Q)
            return w[0], v[:, 0]

        lam = []
        pol = []
        for th in directions:
            n = np.array([np.cos(th), np.sin(th)])
            p0, p1 = 0.0, p_hint
            f0, _ = f(p0, n)
            f1, _ = f(p1, n)
            for it in range(maxit):
                p2 = p1 - f1 * (p1 - p0) / (f1 - f0)
                f2, v2 = f(p2, n)
                p0, f0, p1, f1 = p1, f1, p2, f2
                if abs(p1 - p0) < tol:
                    break
            lam.append(p1)
            pol.append(float(abs(v2 @ n)))  # |m . n| : 0 transverse, 1 longitudinal
        lam = np.array(lam)
        th = np.array(directions)
        out = {"lw_exact": lam.tolist(), "c0": float(lam.mean()),
               "h4": [float(2 * np.mean(lam * np.cos(4 * th))), float(2 * np.mean(lam * np.sin(4 * th)))],
               "h2": [float(2 * np.mean(lam * np.cos(2 * th))), float(2 * np.mean(lam * np.sin(2 * th)))],
               "lw_min": float(lam.min()), "pol_long": pol, "n_tangent_evals": len(cache)}
        # transverse long-wave stiffness at p = 0 and its pressure derivative
        A0 = A_of(0.0)
        Ap = A_of(p_hint)
        cT, g, lam_lin = [], [], []
        for th in directions:
            n = np.array([np.cos(th), np.sin(th)])
            m = np.array([-np.sin(th), np.cos(th)])
            x0 = m @ self.acoustic(A0, n) @ m
            xp = m @ self.acoustic(Ap, n) @ m
            cT.append(float(x0))
            g.append(float(-(xp - x0) / p_hint))
            lam_lin.append(float(x0 / (-(xp - x0) / p_hint)))
        out["cT"] = cT
        out["g"] = g
        out["lam_lin"] = lam_lin
        out["cT_mean"] = float(np.mean(cT))
        out["g_mean"] = float(np.mean(g))
        out["A0"] = A0.tolist()
        out["Ap"] = Ap.tolist()
        out["p_hint"] = p_hint
        return out

    # ------------------------------------------------------------------
    def reciprocal(self):
        return 2 * np.pi * np.linalg.inv(self.T).T  # columns G1, G2

    def analyse(self, directions, kmags=(0.02, 0.04), zone=True, lam_guess=0.4):
        out = {}
        lw = np.zeros((len(directions), len(kmags)))
        for i, th in enumerate(directions):
            n = np.array([np.cos(th), np.sin(th)])
            for j, km in enumerate(kmags):
                lam, _ = self.critical_factor(km * n, lam_guess=lam_guess)
                lw[i, j] = lam[0]
        lam0 = (4 * lw[:, 0] - lw[:, 1]) / 3  # Richardson in k^2
        out["directions_deg"] = [float(np.degrees(t)) for t in directions]
        out["lw_k1"] = lw[:, 0].tolist()
        out["lw_k2"] = lw[:, 1].tolist()
        out["lw_extrap"] = lam0.tolist()
        out["c0"] = float(lam0.mean())
        th = np.array(directions)
        out["h4"] = [float(2 * np.mean(lam0 * np.cos(4 * th))), float(2 * np.mean(lam0 * np.sin(4 * th)))]
        out["h2"] = [float(2 * np.mean(lam0 * np.cos(2 * th))), float(2 * np.mean(lam0 * np.sin(2 * th)))]
        out["lw_min"] = float(lam0.min())
        if zone:
            G = self.reciprocal()
            lamG, _ = self.critical_factor(np.zeros(2), gamma_penalty=True, lam_guess=lam_guess)
            out["gamma"] = lamG[:3].tolist()
            zb = []
            for kv in (G[:, 0] / 2, G[:, 1] / 2, (G[:, 0] + G[:, 1]) / 2):
                lamz, _ = self.critical_factor(kv, lam_guess=lam_guess)
                zb.append(lamz[0])
            out["zone_boundary"] = [float(z) for z in zb]
            out["pc_min"] = float(min(out["lw_min"], lamG[0], min(zb)))
        return out

    def longwave_moduli(self, directions, p_ref, kmags=(0.02, 0.04)):
        """X(theta, p) = tau / (k^2 A ubar^2) at p = 0 and p = p_ref, Richardson in k^2."""
        res = {"X0": [], "Xp": []}
        for th in directions:
            n = np.array([np.cos(th), np.sin(th)])
            for key, p in (("X0", 0.0), ("Xp", p_ref)):
                xs = []
                for km in kmags:
                    vals, ub2 = self.longwave_stiffness(km * n, p)
                    xs.append(vals[0] / (km**2 * self.area * ub2))
                res[key].append(float((4 * xs[0] - xs[1]) / 3))
        X0 = np.array(res["X0"])
        Xp = np.array(res["Xp"])
        g = -(Xp - X0) / p_ref
        res["g"] = g.tolist()
        res["cT_mean"] = float(X0.mean())
        res["g_mean"] = float(g.mean())
        res["lam_lin"] = (X0 / g).tolist()  # first-order (linear in p) estimate of lambda(theta)
        return res

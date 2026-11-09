"""on top of the unmodified ab_frame / penrose_frame frames (imported, never edited).
 * load(name): private module copy, so r and the star can be set per task.
 * mk_refs: reference lattices through a penrose_frame copy with its star P replaced (unit edges only).
 * XFrame mixin: sparse route, general prescribed stress, loaded tangent, strut demand, scatter.
 * moduli(): 3x3 C -> K, E(th), G(th), harmonic split.
All physics equations are copied from Frame.prestress / Frame.lam; only per-element data differ.
"""
from __future__ import annotations
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames', 'penrose', 'phason_paths', 'robustness', 'heldout_test', 'isotropic_loss')]  # shared modules
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
if os.name == "nt":
    pass
import importlib, importlib.util, math, sys
from pathlib import Path
import numpy as np
import scipy.sparse.linalg as spla
import scipy.linalg as la

HERE = Path(__file__).resolve().parent
_SRC = {'ab': HERE.parent / 'frames' / 'ab_frame.py', 'pen': HERE.parent / 'frames' / 'penrose_frame.py'}
# numba cache and module names must stay stable: the two parents are imported once and copies get own globals
_count = 0


def load(cls):
    global _count
    _count += 1
    spec = importlib.util.spec_from_file_location(f"{'ab_frame' if cls == 'ab' else 'penrose_frame'}_design_{_count}", _SRC[cls])
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def flipper(cls):
    """flip_mc from the plainly imported parent module (numba needs an importable module name)."""
    sys.path.insert(0, str(_SRC[cls].parent))
    return importlib.import_module('ab_frame' if cls == 'ab' else 'penrose_frame').flip_mc


def set_r(mod, r):
    mod.R_STRUT = r; mod.EI = r ** 2 / 4; mod.LAM_E = math.pi ** 2 * mod.EI / mod.EA


# ------------------------------------------------------------------ frame extension
def make_xframe(mod):
    """Subclass of mod.Frame. Same K, G, N. Adds routes used by design."""
    P = mod.P
    class XFrame(mod.Frame):
        dense_max = 600

        def lam_N(self, k, N=None, nev=4, vectors=False):
            """Parent Frame.lam with the dense route limited to dense_max dofs (same operator)."""
            k = np.asarray(k, float)
            K = self._assemble(np.ones(len(self.ei)), self.Ke, k).tocsc()
            H = -self._assemble(self.N if N is None else N, self.Ge, k).tocsc()
            gamma = np.allclose(k, np.rint(k))
            fr = self.free_gamma if gamma else np.arange(self.n)
            K = K[fr][:, fr]; H = H[fr][:, fr]
            if K.shape[0] <= self.dense_max:
                vals, vec = la.eigh(H.toarray(), K.toarray(), subset_by_index=(K.shape[0] - 1, K.shape[0] - 1), driver='gvx')
                mu = vals[-1]; v = vec[:, -1]
            else:
                lu = spla.splu(K.tocsc(), permc_spec='COLAMD')
                op = spla.LinearOperator(K.shape, matvec=lambda x: lu.solve(H @ x), dtype=complex)
                vals, vecs = spla.eigs(op, k=nev, which='LR', tol=1e-10, maxiter=50000)
                i = int(np.argmax(vals.real)); mu = vals[i].real; v = vecs[:, i]
            lamv = 1 / mu / mod.LAM_E if mu > 0 else np.inf
            if vectors:
                full = np.zeros(self.n, complex); full[fr] = v
                return lamv, full
            return lamv

        # ---------- load
        def compute_C3(self):
            basis = [np.array([[1., 0], [0, 0]]), np.array([[0., 0], [0, 1]]), np.array([[0, 1.], [1, 0]]) / math.sqrt(2)]
            Cm = np.zeros((3, 3))
            for a, B in enumerate(basis):
                self.prestress(B)
                for b, Bb in enumerate(basis):
                    Cm[b, a] = np.sum(Bb * self.stress)
            self.C3 = Cm; self._basis = basis
            return Cm

        def set_stress(self, S):
            """Prescribe macroscopic stress S (2x2): same 3x3 solve as Frame.set_load('stress')."""
            if not hasattr(self, '_basis'):
                self.compute_C3()
            S = np.asarray(S, float)
            c = la.solve(self.C3, np.array([S[0, 0], S[1, 1], math.sqrt(2) * S[0, 1]]))
            eps = sum(ci * B for ci, B in zip(c, self._basis))
            self.eps = eps
            self.prestress(eps)
            self.stress_resid = float(np.linalg.norm(self.stress - S) / max(np.linalg.norm(S), 1e-30))
            return eps

        # ---------- buckling for the current self.N
        def lw_phi(self, phis, dqs=(0.002, 0.001), N=None):
            tab = {dq: np.array([self.lam_N(dq * (self.cell @ np.array([math.cos(p), math.sin(p)])), N=N) for p in phis]) for dq in dqs}
            a, b = dqs; r = (a / b) ** 2
            return (r * tab[b] - tab[a]) / (r - 1)

        def crit(self, phis=None, nphi=8, extra=((0, 0), (0.5, 0.5)), N=None):
            if phis is None:
                phis = np.arange(nphi) * np.pi / nphi
            ext = self.lw_phi(phis, N=N)
            ex = {str(tuple(k)): float(self.lam_N(k, N=N)) for k in extra}
            lmin = float(ext.min()); kmin = float(min(ex.values()))
            return dict(phis=[float(x) for x in phis], lw=[float(x) for x in ext], lw_min=lmin, extra=ex,
                        lam_c=min(lmin, kmin), mode='longwave' if lmin <= kmin else min(ex, key=ex.get))

        def crit_multi(self, Ns, phis, dqs=(0.002, 0.001), extra=((0, 0), (0.5, 0.5))):
            """crit for several prestress vectors, reusing the K factorization per wavevector (same operator)."""
            kp = [dq * (self.cell @ np.array([math.cos(p), math.sin(p)])) for dq in dqs for p in phis] + [np.array(k, float) for k in extra]
            out = np.zeros((len(Ns), len(kp)))
            ones = np.ones(len(self.ei))
            for ik, k in enumerate(kp):
                K = self._assemble(ones, self.Ke, k).tocsc()
                gamma = np.allclose(k, np.rint(k))
                fr = self.free_gamma if gamma else np.arange(self.n)
                K = K[fr][:, fr]
                if K.shape[0] <= self.dense_max:
                    Kd = K.toarray()
                    for iN, N in enumerate(Ns):
                        H = -self._assemble(N, self.Ge, k).tocsc()[fr][:, fr]
                        mu = la.eigh(H.toarray(), Kd, subset_by_index=(K.shape[0] - 1, K.shape[0] - 1), driver='gvx', eigvals_only=True)[-1]
                        out[iN, ik] = 1 / mu / mod.LAM_E if mu > 0 else np.inf
                else:
                    lu = spla.splu(K.tocsc(), permc_spec='COLAMD')
                    for iN, N in enumerate(Ns):
                        H = -self._assemble(N, self.Ge, k).tocsc()[fr][:, fr]
                        op = spla.LinearOperator(K.shape, matvec=lambda x: lu.solve(H @ x), dtype=complex)
                        vals = spla.eigs(op, k=4, which='LR', tol=1e-10, maxiter=50000, return_eigenvectors=False)
                        mu = vals.real.max()
                        out[iN, ik] = 1 / mu / mod.LAM_E if mu > 0 else np.inf
            n = len(phis); a, b = dqs; r = (a / b) ** 2
            rows = []
            for iN in range(len(Ns)):
                ext = (r * out[iN, n:2 * n] - out[iN, :n]) / (r - 1)
                ex = {str(tuple(k)): float(out[iN, 2 * n + i]) for i, k in enumerate(extra)}
                lmin = float(ext.min()); kmin = min(ex.values())
                rows.append(dict(lw=[float(x) for x in ext], lw_min=lmin, phi_min=float(phis[int(np.argmin(ext))]), extra=ex,
                                 lam_c=min(lmin, kmin), mode='longwave' if lmin <= kmin else min(ex, key=ex.get)))
            return rows

        # ---------- element data (uniform in the parent; per-element in ScatterFrame)
        def _evec(self):
            return P[:, self.dirk].T, np.full(len(self.ei), 1.0 / self.sub)

        def demand(self, S=-np.eye(2)):
            """Strut strain demand per unit stress (first-fibre, rectangular strut t = sqrt(3) r)."""
            self.set_stress(S)
            d = self._full_d(self.eps)
            fe = np.einsum('eab,eb->ea', self.Ke[self.dirk], d)
            M = np.abs(fe[:, [2, 5]]).max(1)
            r = mod.R_STRUT; EIe = getattr(self, 'EIe', mod.EI); EAe = getattr(self, 'EAe', mod.EA)
            t = math.sqrt(3) * r * getattr(self, 'tfac', 1.0)
            ax = np.abs(self.N) / EAe
            bend = M * (t / 2) / EIe
            return dict(ax_mean=float(ax.mean()), ax_max=float(ax.max()), bend_max=float(bend.max()),
                        total_max=float((ax + bend).max()), compress_max=float((-self.N / EAe + bend).max()))

        def _full_d(self, strain):
            evec, Le = self._evec()
            xb = self.xa + evec * Le[:, None]
            dimp = np.zeros((len(self.ei), 6))
            dimp[:, 0:2] = self.xa @ strain.T; dimp[:, 3:5] = xb @ strain.T
            fe = np.einsum('eab,eb->ea', self.Ke[self.dirk], dimp)
            f = np.zeros(self.n)
            np.add.at(f, (3 * self.ei[:, None] + np.arange(3)).ravel(), fe[:, :3].ravel())
            np.add.at(f, (3 * self.ej[:, None] + np.arange(3)).ravel(), fe[:, 3:].ravel())
            w = np.zeros(self.n); fr = self.free_gamma
            w[fr] = self._lu0.solve(-f[fr])
            d = dimp
            d[:, :3] += w[(3 * self.ei[:, None] + np.arange(3))]
            d[:, 3:] += w[(3 * self.ej[:, None] + np.arange(3))]
            return d

        # ---------- loaded tangent L_ijkl(lam) (Bloch-wave homogenization at Gamma)
        def tangent(self, lam):
            """4x4 tangent at multiplier lam (units LAM_E) of the current self.N, basis F11,F12,F21,F22.
            u = F x + periodic fluctuation; energy density 1/2 F:L:F for K + lam LAM_E G(N)."""
            lam_abs = lam * mod.LAM_E
            ones = np.ones(len(self.ei))
            Kt = (self._assemble(ones, self.Ke, (0, 0)) + lam_abs * self._assemble(self.N, self.Ge, (0, 0))).real.tocsc()
            fr = self.free_gamma
            lu = spla.splu(Kt[fr][:, fr].tocsc(), permc_spec='COLAMD')
            evec, Le = self._evec()
            xb = self.xa + evec * Le[:, None]
            Fb = [np.array([[1., 0], [0, 0]]), np.array([[0., 1], [0, 0]]), np.array([[0., 0], [1, 0]]), np.array([[0., 0], [0, 1]])]
            Ket = self.Ke[self.dirk] + lam_abs * self.N[:, None, None] * self.Ge[self.dirk]
            D, fs, ws = [], [], []
            for F in Fb:
                d = np.zeros((len(self.ei), 6))
                d[:, 0:2] = self.xa @ F.T; d[:, 3:5] = xb @ F.T
                fe = np.einsum('eab,eb->ea', Ket, d)
                f = np.zeros(self.n)
                np.add.at(f, (3 * self.ei[:, None] + np.arange(3)).ravel(), fe[:, :3].ravel())
                np.add.at(f, (3 * self.ej[:, None] + np.arange(3)).ravel(), fe[:, 3:].ravel())
                w = np.zeros(self.n); w[fr] = lu.solve(-f[fr])
                D.append(d); fs.append(f); ws.append(w)
            A = abs(np.linalg.det(self.cell))
            L = np.zeros((4, 4))
            for a in range(4):
                for b in range(4):
                    L[a, b] = (np.einsum('ea,eab,eb->', D[a], Ket, D[b]) + fs[a] @ ws[b]) / A
            return 0.5 * (L + L.T)

    return XFrame


def make_scatter_frame(mod):
    """Per-element geometry and stiffness on the same element formulas. Edges keep their straight line
    between (possibly moved) joints; sub elements per edge equal in length."""
    XFrame = make_xframe(mod)
    class ScatterFrame(XFrame):
        def __init__(self, lifts, L1, L2, sub=2, edges=None, g_EA=None, g_EI=None, dX=None, cellF=None, tfac=None):
            if edges is None:
                edges = mod.edges_from_lifts(lifts, L1, L2)
            super().__init__(lifts, L1, L2, sub=sub, edges=edges)
            nE = len(edges); V = len(lifts)
            X = lifts @ mod.P.T
            g_EA = np.ones(nE) if g_EA is None else np.asarray(g_EA, float)
            g_EI = np.ones(nE) if g_EI is None else np.asarray(g_EI, float)
            dX = np.zeros((V, 2)) if dX is None else np.asarray(dX, float)
            F = np.eye(2) if cellF is None else np.asarray(cellF, float)   # affine map on the whole frame
            self.cell = self.cell @ F.T
            d = mod.P[:, edges[:, 2]].T
            vec = (d + dX[edges[:, 1]] - dX[edges[:, 0]]) @ F.T
            x0 = (X[edges[:, 0]] + dX[edges[:, 0]]) @ F.T
            Ledge = np.linalg.norm(vec, axis=1)
            e = vec / Ledge[:, None]
            ne = nE * sub
            self.xa = np.repeat(x0, sub, axis=0) + np.repeat(vec, sub, axis=0) * (np.tile(np.arange(sub), nE) / sub)[:, None]
            self.ev = np.repeat(e, sub, axis=0)
            self.Le = np.repeat(Ledge / sub, sub)
            self.EAe = mod.EA * np.repeat(g_EA, sub)
            self.EIe = mod.EI * np.repeat(g_EI, sub)
            self.tfac = 1.0 if tfac is None else np.repeat(np.asarray(tfac, float), sub)
            # element matrices: same closed forms as _beam_KG, rotated as _rot6
            L = self.Le; a = self.EAe / L; b = self.EIe / L ** 3
            kl = np.zeros((ne, 6, 6)); gl = np.zeros((ne, 6, 6))
            idx = [1, 2, 4, 5]
            kb = np.zeros((ne, 4, 4))
            kb[:, 0, 0] = 12; kb[:, 0, 1] = 6 * L; kb[:, 0, 2] = -12; kb[:, 0, 3] = 6 * L
            kb[:, 1, 1] = 4 * L * L; kb[:, 1, 2] = -6 * L; kb[:, 1, 3] = 2 * L * L
            kb[:, 2, 2] = 12; kb[:, 2, 3] = -6 * L; kb[:, 3, 3] = 4 * L * L
            gb = np.zeros((ne, 4, 4))
            gb[:, 0, 0] = 36; gb[:, 0, 1] = 3 * L; gb[:, 0, 2] = -36; gb[:, 0, 3] = 3 * L
            gb[:, 1, 1] = 4 * L * L; gb[:, 1, 2] = -3 * L; gb[:, 1, 3] = -L * L
            gb[:, 2, 2] = 36; gb[:, 2, 3] = -3 * L; gb[:, 3, 3] = 4 * L * L
            for i in range(4):
                for j in range(i):
                    kb[:, i, j] = kb[:, j, i]; gb[:, i, j] = gb[:, j, i]
            kb *= b[:, None, None]; gb /= (30 * L)[:, None, None]
            for i, ii in enumerate(idx):
                for j, jj in enumerate(idx):
                    kl[:, ii, jj] = kb[:, i, j]; gl[:, ii, jj] = gb[:, i, j]
            kl[:, 0, 0] = a; kl[:, 0, 3] = -a; kl[:, 3, 0] = -a; kl[:, 3, 3] = a
            c, s = self.ev[:, 0], self.ev[:, 1]
            T = np.zeros((ne, 6, 6))
            for o in (0, 3):
                T[:, o, o] = c; T[:, o, o + 1] = s; T[:, o + 1, o] = -s; T[:, o + 1, o + 1] = c; T[:, o + 2, o + 2] = 1
            self.Ke = np.einsum('eba,ebc,ecd->ead', T, kl, T, optimize=True)
            self.Ge = np.einsum('eba,ebc,ecd->ead', T, gl, T, optimize=True)
            self.Rigid = T
            self.dirk = np.arange(ne)

        def _evec(self):
            return self.ev, self.Le

        def prestress(self, strain=-np.eye(2)):
            ones = np.ones(len(self.ei))
            K0 = self._assemble(ones, self.Ke, (0, 0)).real.tocsc()
            xb = self.xa + self.ev * self.Le[:, None]
            dimp = np.zeros((len(self.ei), 6))
            dimp[:, 0:2] = self.xa @ strain.T; dimp[:, 3:5] = xb @ strain.T
            fe = np.einsum('eab,eb->ea', self.Ke, dimp)
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
            self.N = self.EAe / self.Le * np.einsum('ea,ea->e', self.ev, d[:, 3:5] - d[:, 0:2])
            self.resid = float(np.linalg.norm(K0 @ w + f) / max(np.linalg.norm(f), 1e-30))
            fb = np.einsum('eab,eb->ea', self.Ke, d)[:, 3:5]
            delta = self.ev * self.Le[:, None]
            sig = 0.5 * (np.einsum('ea,eb->ab', fb, delta) + np.einsum('ea,eb->ab', delta, fb))
            self.stress = sig / abs(np.linalg.det(self.cell))
            return self.N
    return ScatterFrame


# ------------------------------------------------------------------ moduli
def rot_mandel(C, th):
    """Mandel 3x3 (11,22,sqrt2 12) rotated by th (components in axes rotated by th)."""
    c, s = math.cos(th), math.sin(th); r2 = math.sqrt(2)
    Q = np.array([[c * c, s * s, r2 * c * s], [s * s, c * c, -r2 * c * s], [-r2 * c * s, r2 * c * s, c * c - s * s]])
    return Q @ C @ Q.T


def moduli(C3):
    C3 = 0.5 * (C3 + C3.T)
    C11, C22, C12 = C3[0, 0], C3[1, 1], C3[0, 1]
    C16, C26, C66 = C3[0, 2] / math.sqrt(2), C3[1, 2] / math.sqrt(2), C3[2, 2] / 2
    T0 = (C11 - 2 * C12 + 4 * C66 + C22) / 8
    T1 = (C11 + 2 * C12 + C22) / 8
    R0 = complex(C11 - 2 * C12 - 4 * C66 + C22, 4 * (C16 - C26)) / 8   # |R0| e^{4 i Phi0}
    R1 = complex(C11 - C22, 2 * (C16 + C26)) / 8                        # |R1| e^{2 i Phi1}
    S = np.linalg.inv(C3)
    K = 1 / (S[0, 0] + S[1, 1] + 2 * S[0, 1])
    ths = np.linspace(0, np.pi, 180, endpoint=False)
    E, G, nu = [], [], []
    for th in ths:
        Sr = rot_mandel(S, th)
        E.append(1 / Sr[0, 0]); G.append(1 / (2 * Sr[2, 2])); nu.append(-Sr[0, 1] / Sr[0, 0])
    E, G, nu = map(np.array, (E, G, nu))
    return dict(C3=C3.tolist(), K=float(K), T0=float(T0), T1=float(T1), absR0=float(abs(R0)), R0=[R0.real, R0.imag],
                absR1=float(abs(R1)), R1=[R1.real, R1.imag], Emin=float(E.min()), Emax=float(E.max()), Emean=float(E.mean()),
                Gmin=float(G.min()), Gmax=float(G.max()), Gmean=float(G.mean()), numin=float(nu.min()), numax=float(nu.max()),
                num_mean=float(nu.mean()), K_iso=float(2 * T1), G_iso=float(T0), E_iso=float(4 * 2 * T1 * T0 / (2 * T1 + T0)))


def harmonics_general(th, y, ms=(2, 4, 8)):
    """y(th) on a period-pi grid: fit c0 + sum Re[h_m e^{-i m th}]; returns |h_m| and rms."""
    cols = [np.ones_like(th)]
    for m in ms:
        cols += [np.cos(m * th), np.sin(m * th)]
    A = np.array(cols).T
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    out = dict(c0=float(c[0]), rms=float(np.sqrt(np.mean((A @ c - y) ** 2))))
    for i, m in enumerate(ms):
        out[f'h{m}'] = float(math.hypot(c[1 + 2 * i], c[2 + 2 * i]))
    return out


def acoustic_lambda(lamgrid, Ls, phi):
    """Smallest lam with min eig A(n, lam) = 0 by linear interpolation of the root on a dense spline."""
    n = np.array([math.cos(phi), math.sin(phi)])
    ev = []
    for L in Ls:
        Lt = L.reshape(2, 2, 2, 2)  # i j k l  with F_ij
        A = np.einsum('ijkl,j,l->ik', Lt, n, n)
        ev.append(np.linalg.eigvalsh(0.5 * (A + A.T))[0])
    ev = np.array(ev)
    for i in range(len(ev) - 1):
        if ev[i] > 0 >= ev[i + 1]:
            # quadratic through the bracket and its neighbour
            j0 = min(max(i - 1, 0), len(ev) - 3)
            p = np.polyfit(lamgrid[j0:j0 + 3], ev[j0:j0 + 3], 2)
            rr = np.roots(p); rr = rr[np.isreal(rr)].real
            rr = rr[(rr >= lamgrid[i] - 1e-9) & (rr <= lamgrid[i + 1] + 1e-9)]
            if len(rr):
                return float(rr[0]), ev.tolist()
            return float(lamgrid[i] + (lamgrid[i + 1] - lamgrid[i]) * ev[i] / (ev[i] - ev[i + 1])), ev.tolist()
    return float('nan'), ev.tolist()


# ------------------------------------------------------------------ reference lattices (star-replaced penrose_frame copy)
S3 = math.sqrt(3)


def ref_def(name):
    """unit-edge lattices. returns nodes, a1, a2, edges (i, j, vec, w1, w2) with Xi + vec = Xj + w1 a1 + w2 a2."""
    c60 = (0.5, S3 / 2); c120 = (-0.5, S3 / 2)
    if name == 'square':
        a1, a2 = (1., 0.), (0., 1.)
        nodes = [(0., 0.)]; E = [(0, 0, (1., 0.), 1, 0), (0, 0, (0., 1.), 0, 1)]
    elif name == 'rhomb45':
        a1, a2 = (1., 0.), (1 / math.sqrt(2), 1 / math.sqrt(2))
        nodes = [(0., 0.)]; E = [(0, 0, a1, 1, 0), (0, 0, a2, 0, 1)]
    elif name == 'triangular':
        a1, a2 = (1., 0.), c60
        nodes = [(0., 0.)]; E = [(0, 0, (1., 0.), 1, 0), (0, 0, c60, 0, 1), (0, 0, c120, -1, 1)]
    elif name == 'kagome':
        a1, a2 = (2., 0.), (1., S3)
        A, B, C = (0., 0.), (1., 0.), (.5, S3 / 2)
        nodes = [A, B, C]
        E = [(0, 1, (1., 0.), 0, 0), (1, 0, (1., 0.), 1, 0), (0, 2, c60, 0, 0), (2, 0, c60, 0, 1), (1, 2, c120, 0, 0), (2, 1, c120, -1, 1)]
    elif name == 'hexagonal':
        a1, a2 = (1.5, S3 / 2), (1.5, -S3 / 2)
        nodes = [(0., 0.), (1., 0.)]
        E = [(0, 1, (1., 0.), 0, 0), (1, 0, c60, 1, 0), (1, 0, (.5, -S3 / 2), 0, 1)]
    elif name == 'rhombille':
        a1, a2 = (1.5, S3 / 2), (1.5, -S3 / 2)
        nodes = [(0., 0.), (1., 0.), (2., 0.)]  # A deg 6, B deg 3, C deg 3
        E = [(0, 1, (1., 0.), 0, 0), (1, 0, c60, 1, 0), (1, 0, (.5, -S3 / 2), 0, 1),
             (2, 0, (1., 0.), 1, 1), (2, 0, c120, 1, 0), (2, 0, (-.5, -S3 / 2), 0, 1)]
    else:
        raise KeyError(name)
    return np.array(nodes), np.array(a1), np.array(a2), E


def make_ref(name, r, sub):
    """Build a star-replaced penrose_frame copy and an XFrame on it. Unit-edge check enforced."""
    nodes, a1, a2, E = ref_def(name)
    vecs = []
    for e in E:
        v = tuple(np.round(e[2], 12))
        if v not in vecs:
            vecs.append(v)
    mod = load('pen')
    mod.P = np.array(vecs, float).T; mod.NDIM = len(vecs)
    assert np.allclose(np.linalg.norm(mod.P, axis=0), 1)
    set_r(mod, r)
    Pinv = np.linalg.pinv(mod.P)
    lifts = (Pinv @ nodes.T).T
    edges = np.array([(e[0], e[1], vecs.index(tuple(np.round(e[2], 12))), e[3], e[4]) for e in E], float)
    for (i, j, v, w1, w2) in E:
        res = nodes[i] + np.array(v) - nodes[j] - w1 * a1 - w2 * a2
        assert np.allclose(res, 0, atol=1e-10), (name, i, j, res)
    edges_i = edges.astype(np.int64)
    XF = make_xframe(mod)
    f = XF(lifts, Pinv @ a1, Pinv @ a2, sub=sub, edges=edges_i)
    assert np.allclose(f.cell, np.array([a1, a2]), atol=1e-12)
    f.mod = mod; f.ledge = len(E) / abs(np.linalg.det(f.cell))
    return f


def bz_scan(f, n=24):
    """Hydrostatic-like threshold of the current load on an n x n reduced Bloch grid (half zone by k -> -k)."""
    best = (np.inf, None)
    for i in range(n):
        for j in range(n):
            if (i, j) > ((n - i) % n, (n - j) % n):
                continue
            k = (i / n, j / n)
            v = f.lam_N(k)
            if v < best[0] - 1e-12:
                best = (float(v), k)
    return best


# ------------------------------------------------------------------ stress states
def state_tensor(kind, th=0.0, R=None):
    """Compressive principal -1 along th. R = sigma2/sigma1 (R=1 hydrostatic, R=0 uniaxial, R=-1 pure shear)."""
    if kind == 'uniaxial':
        R = 0.0
    c, s = math.cos(th), math.sin(th)
    Q = np.array([[c, -s], [s, c]])
    return Q @ np.diag([-1.0, -float(R)]) @ Q.T


def boot_ci(x, nb=4000, seed=0, stat=np.mean):
    x = np.asarray(x, float)
    if len(x) < 2:
        return [float(x.mean()), float(x.mean())]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (nb, len(x)))
    v = stat(x[idx], axis=1)
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

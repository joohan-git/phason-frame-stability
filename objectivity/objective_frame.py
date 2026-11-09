"""Objective (corotational, Crisfield) 2D geometric stiffness: existing N term + shear term.
Second variation of U(l-L, th1-beta, th2-beta): N zz^T/l + (M1+M2)(r z^T + z r^T)/l^2.
In 2D rotations commute, so no further moment/twist term exists at the joints."""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames', 'phason_paths', 'penrose', 'disorder', 'slope_survey')]  # shared modules
import os, sys, math
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from pathlib import Path
import numpy as np, scipy.sparse.linalg as spla, scipy.linalg as la
HERE = Path(__file__).resolve().parent
import ab_frame as C

def vmat(c, s, L):
    r = np.array([-c, -s, 0, c, s, 0.]); z = np.array([s, -c, 0, -s, c, 0.])
    return (np.outer(r, z) + np.outer(z, r)) / L ** 2

class ObjFrame(C.Frame):
    objective = True
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        Le = 1.0 / self.sub
        self.Ve = np.array([vmat(C.P[0, q], C.P[1, q], Le) for q in range(4)])
    def prestress(self, strain=-np.eye(2)):
        out = super().prestress(strain)
        # recompute local end moments from the same solved displacement field
        K0 = self._assemble(np.ones(len(self.ei)), self.Ke, (0, 0)).real.tocsc()
        xb = self.xa + C.P[:, self.dirk].T / self.sub
        dimp = np.zeros((len(self.ei), 6))
        dimp[:, 0:2] = self.xa @ strain.T; dimp[:, 3:5] = xb @ strain.T
        fe = np.einsum('eab,eb->ea', self.Ke[self.dirk], dimp)
        f = np.zeros(self.n)
        np.add.at(f, (3 * self.ei[:, None] + np.arange(3)).ravel(), fe[:, :3].ravel())
        np.add.at(f, (3 * self.ej[:, None] + np.arange(3)).ravel(), fe[:, 3:].ravel())
        w = np.zeros(self.n); fr = self.free_gamma
        w[fr] = self._lu0.solve(-f[fr])
        d = dimp.copy()
        d[:, :3] += w[(3 * self.ei[:, None] + np.arange(3))]
        d[:, 3:] += w[(3 * self.ej[:, None] + np.arange(3))]
        self.fe_full = np.einsum('eab,eb->ea', self.Ke[self.dirk], d)
        self.Msum = self.fe_full[:, 2] + self.fe_full[:, 5]
        return out
    def lam(self, k, nev=4, vectors=False):
        k = np.asarray(k, float)
        K = self._assemble(np.ones(len(self.ei)), self.Ke, k).tocsc()
        H = -self._assemble(self.N, self.Ge, k).tocsc()
        if self.objective:
            H = (H - self._assemble(self.Msum, self.Ve, k)).tocsc()
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
        lamv = 1 / mu / C.LAM_E if mu > 0 else np.inf
        return lamv

def element_check():
    """Rigid rotation of one prestressed element: (K_geo) dq must equal the rotated force, J f."""
    rng = np.random.default_rng(1); L = 0.5
    kl, gl = C._beam_KG(L)
    for q in range(4):
        c, s = C.P[0, q], C.P[1, q]; c, s = c / math.hypot(c, s), s / math.hypot(c, s)
    c, s = 0.6, 0.8
    T = C._rot6(c, s); Ke = T.T @ kl @ T
    d = rng.normal(size=6)
    f = Ke @ d          # self-equilibrated element force in global axes
    Fsum = f[:2] + f[3:5]
    x1 = np.zeros(2); x2 = L * np.array([c, s])
    dq = np.array([-x1[1], x1[0], 1, -x2[1], x2[0], 1.])       # small rotation about origin
    # chord force N
    N = np.dot(f[3:5], [c, s])
    Kg_string = N * np.outer(np.array([s, -c, 0, -s, c, 0.]), np.array([s, -c, 0, -s, c, 0.])) / L
    Kg = Kg_string + (f[2] + f[5]) * vmat(c, s, L)
    Jf = np.array([-f[1], f[0], 0, -f[4], f[3], 0.])
    return np.linalg.norm(Kg @ dq - Jf), np.linalg.norm(Kg_string @ dq - Jf), np.linalg.norm(Jf)
if __name__ == '__main__':
    print('element rigid-rotation residual: with shear %.2e, string only %.2e, |Jf| %.2e' % element_check())

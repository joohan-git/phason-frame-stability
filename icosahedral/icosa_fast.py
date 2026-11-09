"""MKL PARDISO symmetric-indefinite solves, Gamma / half-integer-k eigenvalues (eigsh),
and the exact k -> 0 acoustic (long-wave) tensor of the prestressed 3D frame (Schur complement onto
uniform translations, first/second order in q), so that long-wave thresholds for any number of
directions cost one factorization per load multiplier."""
from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
from icosa_core import *

try:
    import pypardiso
    HAVE_PARDISO = True
except Exception:  # pragma: no cover
    HAVE_PARDISO = False


_POOL = []


def _get_pardiso():
    # PyPardisoSolver() globs the file system for the MKL library (~4 s): create once, reuse.
    for s in _POOL:
        if not getattr(s, '_busy', False):
            s._busy = True
            return s
    s = pypardiso.PyPardisoSolver(mtype=-2)
    s._busy = True
    _POOL.append(s)
    return s


class SymSolver:
    """Real symmetric (indefinite) direct solve; PARDISO mtype -2 (inertia from iparm 23)."""

    def __init__(self, A):
        self.A = sp.csr_matrix(A)
        if HAVE_PARDISO:
            self.s = _get_pardiso()
            self.At = sp.triu(self.A).tocsr()
            self.At.sort_indices()
            self.s.factorize(self.At)
            self.neg = int(self.s.get_iparm(23))
        else:
            self.lu = spla.splu(self.A.tocsc(), permc_spec='MMD_AT_PLUS_A')
            self.neg = None

    def _s(self, b):
        if HAVE_PARDISO:
            return self.s.solve(self.At, np.ascontiguousarray(b))
        return self.lu.solve(b)

    def solve(self, b, refine=2):
        x = self._s(b)
        for _ in range(refine):
            x = x + self._s(b - self.A @ x)
        return x

    def free(self):
        if HAVE_PARDISO:
            self.s.free_memory(everything=True)
            self.s._busy = False


class _LU0:
    def __init__(self, A):
        self.S = SymSolver(A)

    def solve(self, b):
        return self.S.solve(b, refine=2)


def prep_load(f):
    """Use PARDISO for the Gamma prestress solve (instead of SuperLU in icosa_core)."""
    K0 = real_assemble(f, np.ones(len(f.ei)), f.Ke)
    fr = f.free_gamma
    f._K0 = K0
    f._lu0 = _LU0(K0[fr][:, fr])


def real_assemble(f, coef, mats, k=None):
    """Real Bloch matrix at Gamma (k None) or at half-integer k (phases +-1)."""
    vals = (coef[:, None, None] * mats[f.dirk]).reshape(len(f.ei), 144)
    if k is not None:
        ph = np.cos(2 * np.pi * (f.wrap @ np.asarray(k, float)))
        vals = vals * np.where(f.blk != 0, ph[:, None], 1.0)
    M = sp.csr_matrix((vals.ravel(), (f.rows, f.cols)), shape=(f.n, f.n))
    return ((M + M.T) * 0.5).tocsr()


def lowest_real(K, H, nev=6):
    """Largest mu of H v = mu K v (K SPD) -> lambda = 1/mu/LAM_E ascending, with vectors."""
    S = SymSolver(K)
    op = spla.LinearOperator(K.shape, matvec=lambda x: S.solve(x, refine=1), dtype=float)
    mu, vec = spla.eigsh(H, k=nev, M=K, Minv=op, which='LA', tol=1e-10, maxiter=20000)
    S.free()
    idx = np.argsort(mu)[::-1]
    lam = [float(1 / m / LAM_E) if m > 0 else np.inf for m in mu[idx]]
    return lam, vec[:, idx]


def frame_gamma(f, nev=6):
    fr = f.free_gamma
    K = real_assemble(f, np.ones(len(f.ei)), f.Ke)[fr][:, fr]
    H = -real_assemble(f, f.N, f.Ge)[fr][:, fr]
    lam, vec = lowest_real(K.tocsr(), H.tocsr(), nev)
    full = np.zeros((f.n, vec.shape[1])); full[fr] = vec
    return lam, full


def frame_halfk(f, k, nev=4):
    K = real_assemble(f, np.ones(len(f.ei)), f.Ke, k)
    H = -real_assemble(f, f.N, f.Ge, k)
    return lowest_real(K, H, nev)


def participation(v):
    u = v.reshape(-1, 6)[:, :3]
    a2 = np.sum(np.abs(u) ** 2, 1)
    return float(a2.sum() ** 2 / (len(a2) * np.sum(a2 ** 2) + 1e-300))


class Homog:
    """Lambda[j,l,a,b](lam): S(q) = q_j q_l Lambda[j,l,a,b] + O(q^3), Schur complement of
    K(q) - lam*LAM_E*H(q) (phase-modulated basis) onto uniform translations, per unit volume.
    Acoustic matrix for direction n: Lambda(n)_ab = n_j n_l Lambda[j,l,a,b]. Requires f.N."""

    def __init__(self, f):
        self.f = f
        self.fr = f.free_gamma
        self.K0 = real_assemble(f, np.ones(len(f.ei)), f.Ke)
        self.Hm = -real_assemble(f, f.N, f.Ge)
        self.Delta = f.dirvecs[f.dirk] * f.Le
        self.ri = (6 * f.ei[:, None] + np.arange(6)[None, :]).ravel()
        self.rj = (6 * f.ej[:, None] + np.arange(6)[None, :]).ravel()
        self.cache = {}

    def tensor(self, lam):
        key = round(float(lam), 13)
        if key in self.cache:
            return self.cache[key]
        f = self.f
        lp = lam * LAM_E
        Me = f.Ke[f.dirk] + lp * f.N[:, None, None] * f.Ge[f.dirk]      # A_e = K_e - lp H_e, H = -G(N)
        A = (self.K0 - lp * self.Hm).tocsr()
        S = SymSolver(A[self.fr][:, self.fr])
        G = np.zeros((3, f.n, 3))
        for j in range(3):
            dj = self.Delta[:, j]
            ci = dj[:, None, None] * Me[:, 0:6, 6:9]
            cj = -dj[:, None, None] * Me[:, 6:12, 0:3]
            for a in range(3):
                G[j][:, a] = np.bincount(self.ri, ci[:, :, a].ravel(), f.n) + np.bincount(self.rj, cj[:, :, a].ravel(), f.n)
        sym = 0.5 * (Me[:, 0:3, 6:9] + np.transpose(Me[:, 0:3, 6:9], (0, 2, 1)))
        D = -np.einsum('ej,el,eab->jlab', self.Delta, self.Delta, sym)
        Gw = G[:, self.fr, :].transpose(1, 0, 2).reshape(len(self.fr), 9)
        X = S.solve(Gw)
        neg = S.neg
        S.free()
        corr = (Gw.T @ X).reshape(3, 3, 3, 3)          # [j,a,l,b]
        Lam = D - corr.transpose(0, 2, 1, 3)
        Lam = 0.5 * (Lam + Lam.transpose(1, 0, 2, 3))
        Lam = 0.5 * (Lam + Lam.transpose(0, 1, 3, 2))
        out = (Lam / f.vol, neg)
        self.cache[key] = out
        return out

    @staticmethod
    def amat(Lam, n):
        n = np.asarray(n, float); n = n / np.linalg.norm(n)
        return np.einsum('jlab,j,l->ab', Lam, n, n)


def fib_sphere(m):
    """m quasi-uniform directions on the upper hemisphere (lambda(n) = lambda(-n))."""
    i = np.arange(m) + 0.5
    z = 1 - i / m
    r = np.sqrt(1 - z * z); ph = np.pi * (1 + 5 ** 0.5) * i
    return np.c_[r * np.cos(ph), r * np.sin(ph), z]


def find_lamW(H, lo=0.3, hi=3.0, nbis=16):
    """Lowest multiplier where A_WW (Gamma, node-0 translations fixed) loses positivity (inertia)."""
    if H.tensor(hi)[1] == 0:
        return hi
    if H.tensor(lo)[1] > 0:
        return lo
    for _ in range(nbis):
        mid = 0.5 * (lo + hi)
        if H.tensor(mid)[1] > 0:
            hi = mid
        else:
            lo = mid
    return lo


def longwave_roots(H, dirs, lamW, lam_lo=0.3, ngrid=12, exact=None, xtol=1e-8):
    """Branch thresholds lambda_b(n) (b = 0,1,2 ascending acoustic eigenvalues) below lamW.
    Coarse grid brackets the first sign change of each branch; directions with index in `exact`
    (default all) are refined by Brent on exact tensors (fresh factorizations), the others are
    returned as linear interpolation inside the bracket (sampling only). None = no root below lamW."""
    dirs = [np.asarray(d, float) / np.linalg.norm(d) for d in dirs]
    top = lamW * (1 - 1e-7)
    grid = np.unique(np.r_[np.linspace(lam_lo, top, ngrid), lamW * np.array([0.95, 0.98, 0.995, 0.999])])
    grid = grid[grid < lamW]
    E = np.array([[np.linalg.eigvalsh(Homog.amat(H.tensor(l)[0], n)) for n in dirs] for l in grid])
    exact = set(range(len(dirs))) if exact is None else set(exact)
    roots = []
    for d, n in enumerate(dirs):
        rr = []
        for b in range(3):
            s = E[:, d, b]
            idx = np.where((s[:-1] > 0) & (s[1:] <= 0))[0]
            if not len(idx):
                rr.append(None); continue
            x0, x1 = grid[idx[0]], grid[idx[0] + 1]
            if d in exact:
                g = lambda l, n=n, b=b: np.linalg.eigvalsh(Homog.amat(H.tensor(l)[0], n))[b]
                rr.append(float(brentq(g, x0, x1, xtol=xtol)))
            else:
                y0, y1 = s[idx[0]], s[idx[0] + 1]
                rr.append(float(x0 + (x1 - x0) * y0 / (y0 - y1)))
        roots.append(rr)
    return roots


def refine_min(H, dirs, lam_guess, lamW, tol=1e-9):
    """Global long-wave minimum: Brent on g(lam) = min over directions n (continuous, Nelder-Mead
    on the sphere started from the best sampled direction) of eig_0(Lambda(n, lam))."""
    from scipy.optimize import minimize
    n0 = np.asarray(dirs[0], float); n0 /= np.linalg.norm(n0)
    th0, ph0 = math.acos(max(-1, min(1, n0[2]))), math.atan2(n0[1], n0[0])
    best = {}

    def g(l):
        Lam = H.tensor(l)[0]
        fun = lambda x: np.linalg.eigvalsh(Homog.amat(Lam, [math.sin(x[0]) * math.cos(x[1]), math.sin(x[0]) * math.sin(x[1]), math.cos(x[0])]))[0]
        r = minimize(fun, [th0, ph0], method='Nelder-Mead', options=dict(xatol=1e-7, fatol=1e-14, maxiter=2000))
        best['x'] = r.x
        return r.fun
    a = lam_guess * 0.99; b = min(lam_guess * 1.01, lamW * (1 - 1e-7))
    k = 0
    while g(a) <= 0 and k < 20:
        a *= 0.98; k += 1
    if g(b) > 0:
        return None, None
    lam = float(brentq(g, a, b, xtol=tol))
    g(lam)
    x = best['x']
    return lam, [math.sin(x[0]) * math.cos(x[1]), math.sin(x[0]) * math.sin(x[1]), math.cos(x[0])]


def analyse(f, dirs_fit=None, nsphere=200, kpts=((0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5), (0.5, 0.5, 0.5)),
            nev_gamma=6, do_gamma=True):
    """Full threshold analysis of a loaded frame (f.set_load done)."""
    H = Homog(f)
    out = {}
    lg, vg = frame_gamma(f, nev_gamma)
    out['gamma'] = lg[:4]; out['gamma_pr'] = participation(vg[:, 0])
    lamW = lg[0] * (1 - 1e-9)
    out['lamW'] = lamW
    out['lamW_inertia_check'] = [H.tensor(lamW * 0.999)[1], H.tensor(lamW * 1.001)[1]]
    dirs_fit = dirs13() if dirs_fit is None else dirs_fit
    sph = fib_sphere(nsphere)
    alld = list(dirs_fit) + list(sph)
    roots = longwave_roots(H, alld, lamW, exact=range(len(dirs_fit)))
    nfit = len(dirs_fit)
    out['dirs_fit'] = [list(map(float, d)) for d in dirs_fit]
    out['roots_fit'] = roots[:nfit]
    b0 = [r[0] for r in roots if r[0] is not None]
    out['lw_min_sampled'] = float(min(b0)) if b0 else None
    if b0:
        i = int(np.argmin([r[0] if r[0] is not None else np.inf for r in roots]))
        out['lw_min_dir'] = list(map(float, alld[i] / np.linalg.norm(alld[i])))
        out['lw_min'], out['lw_min_dir_opt'] = refine_min(H, [alld[i]], out['lw_min_sampled'], lamW)
    else:
        out['lw_min'] = None
    b0s = [r[0] for r in roots[nfit:]]
    b1s = [r[1] for r in roots[nfit:]]
    out['sphere_T1_mean'] = float(np.mean([x for x in b0s if x is not None])) if any(x is not None for x in b0s) else None
    out['sphere_T2_mean'] = float(np.mean([x for x in b1s if x is not None])) if any(x is not None for x in b1s) else None
    out['sphere_T1'] = b0s; out['sphere_T2'] = b1s; out['sphere_L'] = [r[2] for r in roots[nfit:]]
    out['nfact_lw'] = len(H.cache)
    out['kpts'] = {}
    for k in kpts:
        lk, vk = frame_halfk(f, k, 2)
        out['kpts'][str(k)] = [lk[0], participation(vk[:, 0])]
    cands = [x for x in [out.get('lw_min'), lamW if out.get('lw_min') is None else None, (out.get('gamma') or [None])[0]] + [v[0] for v in out['kpts'].values()] if x is not None]
    out['p_c'] = float(min(cands))
    return out

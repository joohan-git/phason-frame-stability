"""Step 1 gates for the independent solver (hermite_frame.py).
A: rigid-rotation invariance (open frame, large rotation 2.5 rad)
B: finite-difference check of the consistent tangent (periodic B1 5/7, sub=2, incl. strain dofs, p)
C: pinned elastica vs exact elliptic-integral solution
D: square-lattice macroscopic threshold p_c = (2x)^2/pi^2, tan x = 2x = 0.55059, sub = 2,4,8, cells 1x1 and 2x2
"""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames', 'phason_paths', 'disorder')]  # shared modules
import sys, json, math
from pathlib import Path
import numpy as np
import scipy.sparse.linalg as spla
from scipy.special import ellipk, ellipe
from scipy.optimize import brentq
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import hermite_frame as nb
import ab_frame as C   # only canonical / edges_from_lifts used
from ab_families import family_lifts

out = {}
rng = np.random.default_rng(1)

# ------------------------------------------------------------------ A: rigid rotation
def open_square_patch(nx, ny, sub):
    X = np.array([[i, j] for i in range(nx + 1) for j in range(ny + 1)], float)
    idx = lambda i, j: i * (ny + 1) + j
    nodes = [X]; ea = []; eb = []; nxt = len(X)
    def add_edge(i, j):
        nonlocal nxt
        chain = [i]
        for m in range(1, sub):
            nodes.append(X[i:i + 1] + (X[j] - X[i]) * m / sub); chain.append(nxt); nxt += 1
        chain.append(j)
        for q in range(sub):
            ea.append(chain[q]); eb.append(chain[q + 1])
    for i in range(nx + 1):
        for j in range(ny + 1):
            if i < nx: add_edge(idx(i, j), idx(i + 1, j))
            if j < ny: add_edge(idx(i, j), idx(i, j + 1))
    return nb.Frame(np.vstack(nodes), ea, eb)

fr = open_square_patch(3, 2, 3)
d = np.zeros(fr.ndof)
d[0::3] = 0.05 * rng.standard_normal(fr.nn); d[1::3] = 0.05 * rng.standard_normal(fr.nn)
d[2::3] = 0.2 * rng.standard_normal(fr.nn)
U0 = fr.energy(d); f0, K0 = fr.internal(d)
phi = 2.5; Rm = np.array([[math.cos(phi), -math.sin(phi)], [math.sin(phi), math.cos(phi)]]); tr = np.array([3.0, -7.0])
x = fr.X + d.reshape(-1, 3)[:, :2]
xn = x @ Rm.T + tr
d2 = d.copy().reshape(-1, 3); d2[:, :2] = xn - fr.X; d2[:, 2] += phi; d2 = d2.ravel()
U1 = fr.energy(d2); f1, K1 = fr.internal(d2)
Tm = np.zeros((fr.ndof, fr.ndof))
for n in range(fr.nn):
    Tm[3 * n:3 * n + 2, 3 * n:3 * n + 2] = Rm; Tm[3 * n + 2, 3 * n + 2] = 1
f_rot = Tm @ f0
K_rot = Tm @ K0.toarray() @ Tm.T
out['A_rigid_rotation'] = dict(phi=phi, U_before=U0, U_after=U1, dU_rel=abs(U1 - U0) / U0,
                               f_err_rel=float(np.abs(f1 - f_rot).max() / np.abs(f0).max()),
                               K_err_rel=float(np.abs(K1.toarray() - K_rot).max() / np.abs(K0.toarray()).max()))
print('A rigid rotation:', out['A_rigid_rotation'])

# ------------------------------------------------------------------ B: FD tangent (periodic B1 5/7, sub 2)
L = family_lifts(5, 7, 'B1'); tl = C.canonical(L[0], L[1]); edges = C.edges_from_lifts(tl['lifts'], tl['L1'], tl['L2'])
fr = nb.build_periodic(tl['lifts'], tl['L1'], tl['L2'], edges, sub=2)
d = np.zeros(fr.ndof)
d[:3 * fr.nn:3] = 0.01 * rng.standard_normal(fr.nn); d[1:3 * fr.nn:3] = 0.01 * rng.standard_normal(fr.nn)
d[2:3 * fr.nn:3] = 0.05 * rng.standard_normal(fr.nn); d[fr.iE:] = [-0.003, -0.002, 0.004]
p = 0.3
Rv, K = fr.residual(d, p)
# energy gradient check
h = 1e-6; idx = rng.choice(fr.ndof, 60, replace=False); idx = np.unique(np.r_[idx, fr.iE, fr.iE + 1, fr.iE + 2])
gnum = np.zeros(len(idx)); Knum = np.zeros((len(idx), fr.ndof))
for a, i in enumerate(idx):
    dp = d.copy(); dp[i] += h; dm = d.copy(); dm[i] -= h
    gnum[a] = (fr.energy(dp, p) - fr.energy(dm, p)) / (2 * h)
    Knum[a] = (fr.residual(dp, p, want_K=False)[0] - fr.residual(dm, p, want_K=False)[0]) / (2 * h)
Kd = K.toarray()
out['B_fd_tangent'] = dict(grad_err_rel=float(np.abs(gnum - Rv[idx]).max() / np.abs(Rv).max()),
                           K_err_rel=float(np.abs(Knum - Kd[idx]).max() / np.abs(Kd).max()),
                           K_asym_rel=float(np.abs(Kd - Kd.T).max() / np.abs(Kd).max()),
                           dRdp_err=float(np.abs((fr.residual(d, p + 1e-4, want_K=False)[0] - fr.residual(d, p - 1e-4, want_K=False)[0]) / 2e-4 - fr.dR_dp(d)).max()))
print('B FD tangent:', out['B_fd_tangent'])

# ------------------------------------------------------------------ C: pinned elastica
def elastica_exact(delta, extensible=False):
    # delta = end shortening / L ; k = sin(alpha/2) ; delta = 2 - 2E(m)/K(m) ; P/PE = (2K/pi)^2
    # extensible: add the first-order axial shortening P L / EA of the compressed rod
    ax = (math.pi ** 2 * nb.EI / nb.EA) if extensible else 0.0
    g = lambda m: 2 - 2 * ellipe(m) / ellipk(m) + ax * (2 * ellipk(m) / math.pi) ** 2 - delta
    m = brentq(g, 1e-12, 0.999999)
    return (2 * ellipk(m) / math.pi) ** 2, 2 * math.asin(math.sqrt(m))

resC = []
for n, r in ((8, 0.03), (16, 0.03), (16, 0.003)):
    X = np.c_[np.linspace(0, 1, n + 1), np.zeros(n + 1)]
    fr = nb.Frame(X, np.arange(n), np.arange(1, n + 1), EI=r ** 2 / 4)
    fr.fix([0, 1, 3 * n, 3 * n + 1])   # left pinned, right roller (uy = 0), ux of right end prescribed (fixed dof)
    prev = None
    for delta in (0.05, 0.2, 0.4):
        if prev is None:
            d = np.zeros(fr.ndof); d[3 * n] = -delta
            s = X[:, 0]; amp = math.sqrt(delta) * 0.8   # buckled initial guess
            d[1::3] = amp * np.sin(math.pi * s); d[2::3] = amp * math.pi * np.cos(math.pi * s); d[0::3] = -delta * s
        else:   # scale previous converged shape
            dp_, d = prev; d = d.copy(); sc = math.sqrt(delta / dp_)
            d[1::3] *= sc; d[2::3] *= sc; d[0::3] *= delta / dp_; d[3 * n] = -delta
        d, ok, it = nb.newton(fr, d, 0.0, tol=1e-12, maxit=60)
        assert ok, (n, delta)
        prev = (delta, d)
        f, _ = fr.internal(d, want_K=False)
        Pn = -f[3 * n] / (math.pi ** 2 * fr.EI)
        alpha = d[2]
        Pe, ae = elastica_exact(delta)
        resC.append(dict(n=n, r=r, delta=delta, P_over_PE=Pn, exact_inext=Pe, rel_err=Pn / Pe - 1, alpha=alpha, alpha_exact=ae, alpha_err=alpha - ae,
                         note='exact = inextensible elastica; residual O(P/EA) is extensibility, vanishes with r'))
        print(f"C elastica n={n} r={r} delta={delta}: P/PE={Pn:.6f} exact={Pe:.6f} ({Pn/Pe-1:+.1e})  alpha={alpha:.5f} exact={ae:.5f} ({alpha-ae:+.1e})")
out['C_elastica'] = resC

# ------------------------------------------------------------------ D: square lattice threshold
x = brentq(lambda x: math.tan(x) - 2 * x, 1.0, 1.5)
pc_exact = (2 * x) ** 2 / math.pi ** 2
resD = []
for (nx, ny) in ((1, 1), (2, 2)):
    for sub in (2, 4, 8):
        fr = nb.square_lattice(nx, ny, sub)
        pcs, _ = nb.linear_buckling(fr, k=4)
        pcs_pos = np.sort(pcs[pcs > 0])
        # nonlinear path check: lowest tangent eigenvalue sign just below / above
        res = dict(cell=f"{nx}x{ny}", sub=sub, p_lin=float(pcs_pos[0]), rel_err=float(pcs_pos[0] / pc_exact - 1),
                   next=[float(v) for v in pcs_pos[1:3]])
        d0 = np.zeros(fr.ndof)
        lam = {}
        for pp in (0.95 * pcs_pos[0], 1.05 * pcs_pos[0]):
            dd, ok, _ = nb.newton(fr, d0, pp); assert ok
            _, K = fr.residual(dd, pp); w, _ = nb.lowest_eigs(fr.Kff(K), k=2)
            lam[f"{pp/pcs_pos[0]:.2f}"] = float(w[0])
        res['lambda_min_at_0.95_1.05'] = lam
        resD.append(res)
        print('D square', res)
# slenderness check: the analytic value is for an inextensible beam; prestrain N/EA ~ p pi^2 EI/EA
resR = []
for r in (0.03, 0.01, 0.003):
    fr = nb.square_lattice(1, 1, 8); fr.EI = r ** 2 / 4
    nb_punit = nb.PUNIT; nb.PUNIT = math.pi ** 2 * fr.EI
    pcs, _ = nb.linear_buckling(fr, k=2); pcs_pos = np.sort(pcs[pcs > 0])
    nb.PUNIT = nb_punit
    resR.append(dict(r=r, p_lin=float(pcs_pos[0]), rel_err=float(pcs_pos[0] / pc_exact - 1)))
    print('D slenderness', resR[-1])
out['D_square'] = dict(pc_exact=pc_exact, x=x, rows=resD, slenderness=resR)
(HERE / 'solver_checks.json').write_text(json.dumps(out, indent=1, default=float), encoding='utf-8')

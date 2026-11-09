"""Gate (c): analytic benchmarks. All references are derived here from the elastica (C), no literature value is needed.
c1 pinned column: P/Pe = (2K/pi)^2, chord/l = 2E/K - 1, k = sin(alpha/2).
c2 square lattice checkerboard (k=(1/2,1/2)): every joint moment balance forces M = 0 at member ends, so each member is a
   pinned elastica with force along its chord; p = P/chord -> p/pE = (2K/pi)^2 / (2E/K - 1).
c3 square lattice macroscopic shear (the critical long-wave mode): antisymmetric S member, force perpendicular to the other
   lattice vector. tan x = 2 x gives the linear threshold; quadrature gives the exact inextensible post-buckling path."""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames', 'phason_paths', 'objectivity', 'disorder')]  # shared modules
import json, math
import numpy as np
from scipy.special import ellipk, ellipe, ellipkinc, ellipeinc
from scipy.optimize import brentq
from corotational_frame import *
out = {}


def column(nel, alphas, r=1e-3):
    EA = 1.0; EI = EA * r ** 2 / 4; L = 1.0 / nel
    n = 3 * (nel + 1)
    Pe = math.pi ** 2 * EI
    ei = np.arange(nel); ej = ei + 1
    b0 = np.zeros(nel); LL = np.full(nel, L)

    def eval_(q):
        dx = L + q[3 * ej] - q[3 * ei]; dy = q[3 * ej + 1] - q[3 * ei + 1]
        d = np.c_[dx, dy]
        _, g, H, N, Ms = element_gh(d, q[3 * ei + 2], q[3 * ej + 2], LL, b0, EA, EI)
        R = np.zeros(n); K = np.zeros((n, n))
        Tm = np.zeros((4, 6)); Tm[0, 0] = -1; Tm[0, 3] = 1; Tm[1, 1] = -1; Tm[1, 4] = 1; Tm[2, 2] = 1; Tm[3, 5] = 1
        for e in range(nel):
            idx = np.r_[3 * e:3 * e + 6]
            R[idx] += Tm.T @ g[e]; K[np.ix_(idx, idx)] += Tm.T @ H[e] @ Tm
        return R, K
    res = []
    q = np.zeros(n); P = Pe
    xs = np.linspace(0, 1, nel + 1)
    al_all = np.arange(1, 121) * math.pi / 180
    targets = {round(math.degrees(a)) for a in alphas}
    for al in al_all:
        if not res and al == al_all[0]:
            q[1::3] = (al / math.pi) * np.sin(math.pi * xs); q[2::3] = al * np.cos(math.pi * xs); q[0::3] = -xs * al ** 2 / 8
        # constraints: ux0 = uy0 = 0, uyN = 0, th0 = al; unknown P closed by th_N = -al.
        fixed = [0, 1, 3 * nel + 1, 2]; free = [i for i in range(n) if i not in fixed]
        q[2] = al
        for it in range(40):
            R, K = eval_(q)
            R = R.copy(); R[3 * nel] += P
            rr = R[free]
            if np.linalg.norm(rr) < 1e-13:
                break
            A = np.zeros((len(free) + 1, len(free) + 1))
            A[:-1, :-1] = K[np.ix_(free, free)]
            col = np.zeros(n); col[3 * nel] = 1
            A[:-1, -1] = col[free]
            row = np.zeros(n); row[3 * nel + 2] = 1
            A[-1, :-1] = row[free]
            b = np.r_[-rr, -(q[3 * nel + 2] + al)]
            sol = np.linalg.solve(A, b)
            q[free] += sol[:-1]; P += sol[-1]
        if round(math.degrees(al)) not in targets:
            continue
        k = math.sin(al / 2); mm = k * k; KK = ellipk(mm); EE = ellipe(mm)
        ex = (2 * KK / math.pi) ** 2; chord = 2 * EE / KK - 1
        res.append(dict(alpha_deg=math.degrees(al), PPe=P / Pe, exact=ex, err=P / Pe / ex - 1, chord=1 + q[3 * nel], chord_exact=chord,
                        chord_err=(1 + q[3 * nel]) / chord - 1))
    return res


out['c1_column'] = {}
for nel in (4, 8, 16, 32):
    out['c1_column'][nel] = column(nel, np.radians([5, 15, 30, 60, 90, 120]))
    print('column nel', nel, ['%.0f: %.2e' % (a['alpha_deg'], a['err']) for a in out['c1_column'][nel]], flush=True)

# c3 exact quadrature
x_c = brentq(lambda x: math.tan(x) - 2 * x, 0.5, 1.5)
pc_exact = (2 * x_c) ** 2 / math.pi ** 2
out['c3_linear_exact'] = dict(x=x_c, pc=pc_exact, published_0p453_converted=0.453 * 12 / math.pi ** 2)
print('square exact linear: tan x = 2x, x=%.8f, pc = %.8f; 0.453 E(t/l)^3 -> %.4f' % (x_c, pc_exact, 0.453 * 12 / math.pi ** 2))


def sq_exact(th0):
    """th0 = angle between joint tangent and member force = half the lattice shear angle."""
    def geo(al):
        k = math.sin(al / 2); mm = k * k
        ph0 = math.asin(min(1.0, math.sin(th0 / 2) / k))
        I0 = ellipk(mm) - ellipkinc(ph0, mm)
        Ic = 2 * (ellipe(mm) - ellipeinc(ph0, mm)) - I0
        Is = 2 * k * math.cos(ph0)
        return I0, Ic, Is
    f = lambda al: (lambda g: g[2] / g[1] - math.tan(2 * th0))(geo(al))
    al0 = th0 / math.cos(x_c)
    a, b = th0 * 1.0001, min(al0 * 2, math.pi * 0.99)
    al = brentq(f, a, b, xtol=1e-13)
    I0, Ic, Is = geo(al)
    dl = math.hypot(Ic, Is) / I0
    p = (2 * I0) ** 2 / dl / math.pi ** 2
    return dict(th0=th0, alpha=al, p=p, chord=dl, e3=dl * math.sin(th0), e1=dl * math.cos(th0) - 1)


ths = np.radians([0.5, 1, 2, 3, 5, 8, 12, 16, 20])
ex = [sq_exact(t) for t in ths]
fit_t = np.radians([0.25, 0.5, 1, 1.5])
b2_exact = np.polyfit(fit_t ** 2, [sq_exact(t)['p'] / pc_exact - 1 for t in fit_t], 1)[0]
out['c3_exact'] = ex; out['c3_b2_exact_per_rad2'] = b2_exact
print('exact square b2 (p/pc-1 per psi^2) = %.4f' % b2_exact, [(round(math.degrees(e['th0']), 1), round(e['p'] / pc_exact, 5)) for e in ex], flush=True)


def sq_branch(sub, r, maxdeg=20, nsteps=250):
    model = Model(square_mesh(sub, r))
    lam, vec, st = model.linear_critical(macro=True)
    cr = find_critical(model, lam[0], frac=0.99)
    phi = cr['mode']
    eps = 0.02
    stop = lambda rec, rows: rec['edev'] > math.sin(math.radians(maxdeg)) * 1.05
    rows, states = run_branch(model, cr['z'], cr['p'] * model.m.lamE, phi, 1, eps, nsteps, stop, ds_factor=1.0, k_eig=2)
    return model, lam[0], cr, [r_ for r_ in rows if 'p' in r_]


out['c3_solver'] = []
for sub, r in ((4, 3e-3), (8, 3e-3), (16, 3e-3), (4, 3e-2), (8, 3e-2)):
    model, lam0, cr, rows = sq_branch(sub, r)
    pp = np.array([x['p'] for x in rows]); e3 = np.abs([x['e3'] for x in rows]); e1 = np.array([x['e1'] for x in rows])
    psi = np.arctan2(e3, 1 + e1)
    errs = []
    for ps_, val in zip(psi, pp):
        if ps_ > math.radians(0.3) and ps_ < math.radians(21):
            e = sq_exact(ps_)
            errs.append(dict(deg=math.degrees(ps_), solver=val / pc_exact, exact=e['p'] / pc_exact, rel_err=val / e['p'] - 1))
    errs = errs[:: max(1, len(errs) // 8)]
    rec = dict(sub=sub, r=r, lin_macro=float(lam0), pc_NL=cr['p'], pc_NL_over_lin_minus1=cr['p'] / float(lam0) - 1, lamE=model.m.lamE,
               ratio_shift_to_lamE=(cr['p'] / float(lam0) - 1) / model.m.lamE, errs=errs, nsteps=len(rows),
               neg_on_branch=[x['neg'] for x in rows][:6], trace=cr['trace'][-3:])
    out['c3_solver'].append(rec)
    print('sq sub', sub, 'r', r, 'lin %.6f NL %.6f (rel %.3e, /lamE %.3f)' % (lam0, cr['p'], cr['p'] / lam0 - 1, rec['ratio_shift_to_lamE']),
          ['%.0f:%.2e' % (x['deg'], x['rel_err']) for x in errs], 'neg', rec['neg_on_branch'][:3], flush=True)


# c2 checkerboard
def checker(sub, r, alphas=np.radians([1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 35, 40, 45])):
    mesh = supercell(square_mesh(sub, r), [[2, 0], [0, 2]])
    model = Model(mesh); m = mesh
    V = 4
    X = m.X[:V]
    sg = np.array([1 if (round(x) + round(y)) % 2 == 0 else -1 for x, y in X])
    th = np.zeros(m.nn); th[:V] = sg
    ux = np.zeros(m.nn); uy = np.zeros(m.nn)
    # base cell has 2 edges and `sub` elements per edge; chains are contiguous in element order per cell copy
    for e in range(m.ne // sub):
        chain = list(range(e * sub, (e + 1) * sub))
        i0 = m.ei[chain[0]]
        d = m.d0[chain[0]] / np.linalg.norm(m.d0[chain[0]]); nrm = np.array([-d[1], d[0]])
        for h, c in enumerate(chain[:-1]):
            s_ = (h + 1) / sub; nd = m.ej[c]
            sa = th[i0]
            w = sa * math.sin(math.pi * s_) / math.pi
            ux[nd] = w * nrm[0]; uy[nd] = w * nrm[1]; th[nd] = sa * math.cos(math.pi * s_)
    phi = np.zeros(m.n)
    phi[0::3] = ux; phi[1::3] = uy; phi[2::3] = th
    res = []
    zprev = None; p = 1.0 * m.lamE
    for al in alphas:
        z = zprev if zprev is not None else np.r_[(phi * al)[2:], np.array([1., 1., 0.]) * m.s * (-al ** 2 / 8 - m.lamE)]
        c = np.zeros(m.nz); c[0] = 1.0            # rotation of vertex 0 is free dof 0
        z2, p2, it, ok = newton_con(model, z.copy(), p, c, 0.0, sg[0] * al, maxit=30)
        k = math.sin(al / 2); mm = k * k; KK = ellipk(mm); EE = ellipe(mm)
        exact_ = (2 * KK / math.pi) ** 2 / (2 * EE / KK - 1)
        res.append(dict(alpha_deg=math.degrees(al), ok=bool(ok), p=p2 / m.lamE, exact=exact_, rel=p2 / m.lamE / exact_ - 1))
        if ok:
            zprev, p = z2, p2
    return res


out['c2_checker'] = {}
rr = checker(8, 0.03); out['c2_checker']['8_0.03'] = rr
print('checker 8 0.03 r-shift/lamE', [round(x['rel'] / (math.pi ** 2 * 0.03 ** 2 / 4), 3) for x in rr][:13], 'rel@45', rr[-1]['rel'], flush=True)
rr = checker(16, 0.03); out['c2_checker']['16_0.03'] = rr
print('checker 16 0.03 r-shift/lamE', [round(x['rel'] / (math.pi ** 2 * 0.03 ** 2 / 4), 3) for x in rr][:13], 'rel@45', rr[-1]['rel'], flush=True)
rr = checker(16, 0.015); out['c2_checker']['16_0.015'] = rr
print('checker 16 0.015 r-shift/lamE', [round(x['rel'] / (math.pi ** 2 * 0.015 ** 2 / 4), 3) for x in rr][:13], 'rel@45', rr[-1]['rel'], flush=True)
(HERE / 'exact_references.json').write_text(json.dumps(out, indent=1, default=float))

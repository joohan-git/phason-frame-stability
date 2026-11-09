"""r2: T_h-odd (major-antisymmetric) channel of the loaded acoustic tensor and objective-projected thresholds.
Theorem (objective W(F), symmetric prestress): Lambda(q,u) = C_sym(q,u) + sigma_jl q_j q_l |u|^2, hence Lambda[jl,ab] = Lambda[ab,jl].
The stored operator (axial-only geometric stiffness) violates this. Lambda_s = (Lambda[jl,ab] + Lambda[ab,jl])/2 removes the violation at first order.
Thresholds from stored tensors at lam = 0.85, 0.9, 0.92, root by cubic Lagrange interpolation in lam, calibrated by applying the same
procedure to the raw tensor and comparing with the stored exact lw_min. Writes axial_objective_estimate.json."""
import os, sys, json, glob
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
import numpy as np
from scipy.optimize import minimize
from icosa_fast import fib_sphere, dirs13
from load_axial_runs import load
R, _ = load()
SPH = np.vstack([fib_sphere(300), np.array(dirs13())])


def odd(L):
    return float(((L[0, 0, 1, 1] + L[1, 1, 2, 2] + L[2, 2, 0, 0]) - (L[0, 0, 2, 2] + L[1, 1, 0, 0] + L[2, 2, 1, 1])) / 3)


def antis_norm(L):
    return float(np.linalg.norm(L - L.transpose(2, 3, 0, 1)) / 2), float(np.linalg.norm(L))


def thresholds(ls, Ts, lamW):
    """Lowest two acoustic roots per direction and the global minimum, on the interpolant."""
    ls = np.array(ls); Ts = np.array(Ts)
    grid = np.linspace(ls.min(), min(lamW * (1 - 1e-6), ls.max() + .03), 90)
    # Lagrange weights
    W = np.array([[np.prod([(g - ls[m]) / (ls[k] - ls[m]) for m in range(len(ls)) if m != k]) for k in range(len(ls))] for g in grid])
    Lg = np.einsum('gk,kjlab->gjlab', W, Ts)
    A = np.einsum('gjlab,ej,el->geab', Lg, SPH, SPH)
    E = np.linalg.eigvalsh(A)  # (g, e, 3)
    out = []
    for b in (0, 1):
        r = np.full(len(SPH), np.nan)
        for e in range(len(SPH)):
            s = E[:, e, b]; i = np.where((s[:-1] > 0) & (s[1:] <= 0))[0]
            if len(i):
                i = i[0]; r[e] = grid[i] + (grid[i + 1] - grid[i]) * s[i] / (s[i] - s[i + 1])
        out.append(r)
    return out


fam = {}
for fn in sorted(glob.glob(os.path.join(here, 'runs_zero_load', '*.json'))):
    r = json.load(open(fn)); fam.setdefault(r['tag'], []).append(r)
res = {}
for tag, rs in sorted(fam.items(), key=lambda x: x[1][0]['V']):
    rows = []
    for r in rs:
        keys = [k for k in ('0.85', '0.9', '0.92', 'root') if k in r['tensors']]
        ls = [float(k) if k != 'root' else r['lam_iso_root'] for k in keys]
        if len(set(np.round(ls, 6))) < len(ls) or len(ls) < 3:
            keys = [k for k in ('0.85', '0.9', '0.92') if k in r['tensors']]; ls = [float(k) for k in keys]
        Ts = [np.array(r['tensors'][k]) for k in keys]
        Ss = [.5 * (T + T.transpose(2, 3, 0, 1)) for T in Ts]
        raw = thresholds(ls, Ts, r['lamW']); sym = thresholds(ls, Ss, r['lamW'])
        od = {k: odd(np.array(v)) for k, v in r['tensors'].items()}
        an = {k: antis_norm(np.array(v)) for k, v in r['tensors'].items()}
        s = R[f"{r['tag']}_s{r['seed']}.json"]
        rows.append(dict(seed=r['seed'], lw_stored=s['lw_min'], gamma=s['gamma'][0], raw_min=float(np.nanmin(raw[0])), sym_min=float(np.nanmin(sym[0])),
                         sym_T1_mean=float(np.nanmean(sym[0])), sym_T2_mean=float(np.nanmean(sym[1])), raw_T1_mean=float(np.nanmean(raw[0])), raw_T2_mean=float(np.nanmean(raw[1])),
                         sym_T1_max=float(np.nanmax(sym[0])), nan_T2_sym=int(np.isnan(sym[1]).sum()),
                         odd=od, antis_over_norm=an.get('0.92', [0, 1])[0] / an.get('0.92', [0, 1])[1], lam_iso=r['lam_iso_root']))
    M = np.diag(np.array(rs[0]['M']))
    e = dict(V=rs[0]['V'], n=len(rows), m=float(M.mean()), h=float(M[2] - .5 * (M[0] + M[1])), rows=rows)
    for k in ('lw_stored', 'raw_min', 'sym_min', 'gamma', 'sym_T1_mean', 'sym_T2_mean', 'raw_T1_mean', 'raw_T2_mean', 'sym_T1_max'):
        v = [x[k] for x in rows]; e[k] = [float(np.mean(v)), float(np.std(v, ddof=1) / len(v) ** .5) if len(v) > 1 else None]
    e['calib_abs_max'] = float(max(abs(x['raw_min'] - x['lw_stored']) for x in rows))
    lk = [x['lam_iso'] for x in rows if x['lam_iso']]
    e['lam_iso'] = [float(np.mean(lk)), float(np.std(lk, ddof=1) / len(lk) ** .5) if len(lk) > 1 else None]
    for l in ('0.5', '0.85', '0.92'):
        v = [x['odd'][l] for x in rows if l in x['odd']]
        if v:
            e[f'odd_{l}'] = [float(np.mean(v)), float(np.std(v, ddof=1) / len(v) ** .5) if len(v) > 1 else None]
    res[tag] = e
    print(tag.ljust(16), 'V', e['V'], 'm %+.4f h %+.4f' % (e['m'], e['h']), 'lw %.5f raw %.5f (cal %.1e) sym %.5f' % (e['lw_stored'][0], e['raw_min'][0], e['calib_abs_max'], e['sym_min'][0]),
          'iso %.5f' % e['lam_iso'][0], 'symT %.4f %.4f rawT %.4f %.4f' % (e['sym_T1_mean'][0], e['sym_T2_mean'][0], e['raw_T1_mean'][0], e['raw_T2_mean'][0]), 'G %.4f' % e['gamma'][0], 'antis/norm %.1e' % np.mean([x['antis_over_norm'] for x in rows]), 'odd.92', e.get('odd_0.92', [None])[0], flush=True)
json.dump(res, open(os.path.join(here, 'axial_objective_estimate.json'), 'w'), indent=1, default=float)

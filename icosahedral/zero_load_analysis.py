"""r2: harmonic analysis of the loaded acoustic tensor. tau(n) = tr[P_n Lambda(n) P_n]/2 (transverse), nu(n) = n.Lambda(n).n.
Offset-averaged coefficients per family; regressions of the K4 coefficient on m (G) and the Y20 coefficient on h (H)."""
import os, sys, json, glob
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
import numpy as np
from icosa_core import P
_z, _w = np.polynomial.legendre.leggauss(8)
_ph = np.arange(18) * 2 * np.pi / 18
N = np.array([[np.sqrt(1 - z * z) * np.cos(p), np.sqrt(1 - z * z) * np.sin(p), z] for z in _z for p in _ph])
W = np.array([w / 18 / 2 for w in _w for p in _ph])
K4 = np.sum(N ** 4, 1) - 0.6
Y20 = (3 * N[:, 2] ** 2 - 1) / 2
X2 = N[:, 0] ** 2 - N[:, 1] ** 2
BAS = dict(K4=K4, Y20=Y20, X2=X2)


def coef(f, b):
    return float(W @ (f * b) / (W @ (b * b)))


def funcs(Lam):
    A = np.einsum('jlab,ej,el->eab', np.array(Lam), N, N)
    nu = np.einsum('ea,eab,eb->e', N, A, N)
    tau = (np.trace(A, axis1=1, axis2=2) - nu) / 2
    # traceless transverse part: split(n) = sqrt(2 tr[(PAP)^2] - (tr PAP)^2)
    Pn = np.eye(3)[None] - N[:, :, None] * N[:, None, :]
    B = Pn @ A @ Pn
    tr2 = np.einsum('eab,eba->e', B, B)
    sp = np.sqrt(np.maximum(2 * tr2 - (2 * tau) ** 2, 0))
    return tau, nu, sp


fam = {}
for fn in glob.glob(os.path.join(here, 'runs_zero_load', '*.json')):
    r = json.load(open(fn)); fam.setdefault(r['tag'], []).append(r)
LS = ['0.0', '0.5', '0.85', '0.9', '0.92', '0.90']
out = {}
for tag, rs in fam.items():
    M = np.diag(np.array(rs[0]['M']))
    e = dict(V=rs[0]['V'], n=len(rs), a=rs[0]['a'], Mdiag=M.tolist(), m=float(M.mean()), h=float(M[2] - .5 * (M[0] + M[1])), repro_max=max(r['repro_C6'] for r in rs),
             lam_iso=[r['lam_iso_root'] for r in rs])
    for l in ('0.0', '0.5', '0.85', '0.9', '0.92', 'root'):
        vals = []
        for r in rs:
            if l not in r['tensors']:
                continue
            tau, nu, sp = funcs(r['tensors'][l])
            t0 = float(W @ tau); n0 = float(W @ nu)
            res = tau - t0 - sum(coef(tau, b) * b for b in BAS.values())
            vals.append(dict(t0=t0, n0=n0, **{'t' + k: coef(tau, b) for k, b in BAS.items()}, **{'n' + k: coef(nu, b) for k, b in BAS.items()},
                             split_rms=float(np.sqrt(W @ sp ** 2)), res_rms=float(np.sqrt(W @ res ** 2))))
        if vals:
            ks = vals[0].keys()
            e[l] = {k: [float(np.mean([v[k] for v in vals])), float(np.std([v[k] for v in vals], ddof=1) / len(vals) ** .5) if len(vals) > 1 else None] for k in ks}
            e[l]['nused'] = len(vals)
    # offset-averaged tensor: split and residual after averaging
    for l in ('0.0', '0.92'):
        Ls = [r['tensors'][l] for r in rs if l in r['tensors']]
        if Ls:
            tau, nu, sp = funcs(np.mean(np.array(Ls), 0))
            e[l]['avg_split_rms'] = float(np.sqrt(W @ sp ** 2))
    out[tag] = e
order = sorted(out, key=lambda k: out[k]['V'])


def fit(xs, ys, es, names):
    """Weighted LS y = X beta; errors from sd floor; returns beta, se, chi2/dof."""
    X = np.array(xs, float); y = np.array(ys, float); s = np.array(es, float)
    w = 1 / s
    b, *_ = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)
    cov = np.linalg.inv((X * w[:, None]).T @ (X * w[:, None]))
    chi = float(np.sum(((X @ b - y) * w) ** 2)) / max(len(y) - X.shape[1], 1)
    return {n: [float(a), float(c)] for n, a, c in zip(names, b, np.sqrt(np.diag(cov)))}, chi


fits = {}
for l, lk in (('0.0', '0.0'), ('0.5', '0.5'), ('0.85', '0.85'), ('0.9', '0.9'), ('0.92', '0.92')):
    fs = [k for k in order if lk in out[k] and out[k]['V'] >= 136]
    # conversion to threshold units near 0.92
    t_at = {k: out[k][lk]['t0'][0] for k in fs}
    for part in ('t', 'n'):
        yG = [out[k][lk][part + 'K4'][0] for k in fs]; eG = [max(out[k][lk][part + 'K4'][1] or 0, 1e-9) for k in fs]
        yH = [out[k][lk][part + 'Y20'][0] for k in fs]; eH = [max(out[k][lk][part + 'Y20'][1] or 0, 1e-9) for k in fs]
        ms = [out[k]['m'] for k in fs]; hs = [out[k]['h'] for k in fs]; a2 = [np.mean(np.array(out[k]['a']) ** -2) for k in fs]
        fits[f'{part}K4_{l}'] = dict(zip(('beta', 'chi2'), fit(list(zip(ms)), yG, eG, ['kG'])))
        fits[f'{part}K4_{l}_m_h_a2'] = dict(zip(('beta', 'chi2'), fit(list(zip(ms, hs, a2)), yG, eG, ['kG', 'kh', 'a^-2'])))
        fits[f'{part}Y20_{l}'] = dict(zip(('beta', 'chi2'), fit(list(zip(hs)), yH, eH, ['kH'])))
        fits[f'{part}Y20_{l}_h_m_a2'] = dict(zip(('beta', 'chi2'), fit(list(zip(hs, ms, a2)), yH, eH, ['kH', 'km', 'a^-2'])))
json.dump(dict(fam=out, fits=fits), open(os.path.join(here, 'zero_load_tensor.json'), 'w'), indent=1)
for l in ('0.0', '0.85', '0.92', 'root'):
    print('--- lam', l)
    for k in order:
        e = out[k]
        if l not in e:
            continue
        x = e[l]
        print(k.ljust(16), 'V%6d m %+.4f h %+.4f n%d' % (e['V'], e['m'], e['h'], x['nused']), 't0 %.4e tK4 %+.3e(%.1e) tY20 %+.3e(%.1e) | nK4 %+.3e nY20 %+.3e | split %.2e res %.2e' % (x['t0'][0], x['tK4'][0], x['tK4'][1] or 0, x['tY20'][0], x['tY20'][1] or 0, x['nK4'][0], x['nY20'][0], x['split_rms'][0], x['res_rms'][0]), 'avgsplit %.2e' % x.get('avg_split_rms', np.nan))
print('lam_iso', {k: (round(np.mean([v for v in out[k]['lam_iso'] if v]), 5), sum(1 for v in out[k]['lam_iso'] if v)) for k in order})
print(json.dumps(fits, indent=0)[:0])
for k, v in fits.items():
    if k[0] == 't':
        print(k, v)

# ---- r2 targeted regressions (family level, OLS, jackknife over families)
def ols(rows, ycol, xcols):
    X = np.array([[r[c] for c in xcols] for r in rows], float); y = np.array([r[ycol] for r in rows], float)
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    jk = np.array([np.linalg.lstsq(np.delete(X, i, 0), np.delete(y, i), rcond=None)[0] for i in range(len(y))])
    se = np.sqrt((len(y) - 1) / len(y) * np.sum((jk - jk.mean(0)) ** 2, 0))
    rs = y - X @ b
    return {c: [float(a), float(s)] for c, a, s in zip(xcols, b, se)}, dict(rms=float(np.sqrt(np.mean(rs ** 2))), y_rms=float(np.sqrt(np.mean(y ** 2))), n=len(y))


targ = {}
slopes = {}
for k in order:
    e = out[k]
    if '0.9' in e and '0.92' in e:
        slopes[k] = (e['0.92']['t0'][0] - e['0.9']['t0'][0]) / 0.02
tp = float(np.mean([slopes[k] for k in ('cubic_5-3', 'cubic_8-5') if k in slopes]))
targ['dtau0_dlam_near_threshold'] = tp
for l in ('0.0', '0.5', '0.85', '0.9', '0.92'):
    rows = []
    for k in order:
        e = out[k]
        if l not in e or e['V'] < 136:
            continue
        a2 = float(np.mean(np.array(e['a']) ** -2))
        rows.append(dict(tag=k, m=e['m'], h=e['h'], h2=e['h'] ** 2, m2=e['m'] ** 2, a2=a2, absm=abs(e['m']), mh=e['m'] * e['h'], tK4=e[l]['tK4'][0], tY20=e[l]['tY20'][0], nK4=e[l]['nK4'][0], nY20=e[l]['nY20'][0], t0=e[l]['t0'][0]))
    d = {}
    for nm, xc in (('m', ['m']), ('m_h2', ['m', 'h2']), ('m_h2_a2', ['m', 'h2', 'a2']), ('m_h2_absm', ['m', 'h2', 'absm'])):
        d['tK4~' + nm] = ols(rows, 'tK4', xc)
    tet = [r for r in rows if r['tag'].startswith('tetra')]
    for nm, xc in (('h', ['h']), ('h_h2', ['h', 'h2']), ('h_h2_mh', ['h', 'h2', 'mh'])):
        d['tY20~' + nm] = ols(tet, 'tY20', xc)
        d['nY20~' + nm] = ols(tet, 'nY20', xc)
    d['cubic_ctrl_tY20_rms'] = float(np.sqrt(np.mean([r['tY20'] ** 2 for r in rows if r['tag'].startswith('cubic')])))
    d['_rows'] = rows
    targ[l] = d
# antisymmetric pair estimates of kH (same |h|, sign reversed)
pairs = [('tetra_2-1x3-2', 'tetra_3-2x2-1'), ('tetra_5-3x3-2', 'tetra_3-2x5-3'), ('tetra_8-5x5-3', 'tetra_5-3x8-5')]
targ['pairs'] = {}
for a, b in pairs:
    if a in out and b in out:
        ha, hb = out[a]['h'], out[b]['h']
        rowp = dict(h=[ha, hb], m=[out[a]['m'], out[b]['m']])
        for l in ('0.0', '0.5', '0.85', '0.9', '0.92'):
            if l in out[a] and l in out[b]:
                ya, yb = out[a][l]['tY20'][0], out[b][l]['tY20'][0]
                na, nb = out[a][l]['nY20'][0], out[b][l]['nY20'][0]
                sa, sb = out[a][l]['tY20'][1], out[b][l]['tY20'][1]
                rowp[l] = dict(kH_t=(ya - yb) / (ha - hb), kH_t_se=float(np.hypot(sa, sb) / abs(ha - hb)), c_t=(ya + yb) / (ha ** 2 + hb ** 2), kH_n=(na - nb) / (ha - hb), split_a=out[a][l]['split_rms'][0], split_b=out[b][l]['split_rms'][0], tK4_a=out[a][l]['tK4'][0], tK4_b=out[b][l]['tK4'][0])
        targ['pairs'][a + '|' + b] = rowp
# split and iso bracket along the cubic series; [100] pair check (offset average vs per offset), T_h odd part
cub = [k for k in order if k.startswith('cubic')]
targ['cubic_series'] = {}
for k in cub:
    e = out[k]; rs = fam[k]
    t = dict(V=e['V'], m=e['m'], n=e['n'], lam_iso_mean=float(np.mean([v for v in e['lam_iso'] if v])), lam_iso_sem=float(np.std([v for v in e['lam_iso'] if v], ddof=1) / max(1, len(e['lam_iso'])) ** .5) if e['n'] > 1 else None,
             split_over_absm={l: e[l]['split_rms'][0] / abs(e['m']) for l in ('0.0', '0.85', '0.92') if l in e}, split_lam_units_0p92=(e['0.92']['split_rms'][0] / abs(tp)) if '0.92' in e else None,
             tK4_over_m={l: e[l]['tK4'][0] / e['m'] for l in ('0.0', '0.5', '0.85', '0.9', '0.92') if l in e},
             tK4_sem_over_m={l: (e[l]['tK4'][1] or 0) / abs(e['m']) for l in ('0.0', '0.5', '0.85', '0.9', '0.92') if l in e})
    Ls = [np.array(r['tensors']['0.92']) for r in rs if '0.92' in r['tensors']]
    if Ls:
        def p100(L):
            s = []
            for ax, (u, v) in zip(range(3), ((1, 2), (0, 2), (0, 1))):
                A = L[ax, ax]
                s.append(np.hypot(A[u, u] - A[v, v], 2 * A[u, v]))
            return np.mean(s)
        def odd(L):
            return float(((L[0, 0, 1, 1] + L[1, 1, 2, 2] + L[2, 2, 0, 0]) - (L[0, 0, 2, 2] + L[1, 1, 0, 0] + L[2, 2, 1, 1])) / 3)
        def k44(L):
            return float((L[0, 0, 1, 1] + L[1, 1, 2, 2] + L[2, 2, 0, 0] + L[0, 0, 2, 2] + L[1, 1, 0, 0] + L[2, 2, 1, 1]) / 6)
        t['split100_per_offset_lam'] = float(np.mean([p100(L) for L in Ls]) / abs(tp)); t['split100_avg_tensor_lam'] = float(p100(np.mean(Ls, 0)) / abs(tp))
        t['Th_odd_lam'] = float(np.mean([odd(L) for L in Ls]) / abs(tp)); t['Th_odd_sem_lam'] = float(np.std([odd(L) for L in Ls], ddof=1) / len(Ls) ** .5 / abs(tp)) if len(Ls) > 1 else None
    targ['cubic_series'][k] = t
json.dump(dict(fam=out, targeted=targ), open(os.path.join(here, 'zero_load_tensor.json'), 'w'), indent=1, default=float)
print('=== targeted'); print('dtau', tp)
for l in ('0.0', '0.85', '0.92'):
    print('lam', l, {k: v for k, v in targ[l].items() if k != '_rows'})
print(json.dumps(targ['pairs'], indent=0, default=float)); print(json.dumps(targ['cubic_series'], indent=0, default=float))

"""r3: assemble icosa_results.json from runs_objective/ against the axial-only runs (runs/, runs_axial_large/)."""
import os, sys, json, glob
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
import numpy as np
from scipy.stats import spearmanr
from load_axial_runs import load
OLD, _ = load()
NEW = {os.path.basename(f): json.load(open(f)) for f in glob.glob(os.path.join(here, 'runs_objective', '*.json'))}
_z, _w = np.polynomial.legendre.leggauss(8)
_ph = np.arange(18) * 2 * np.pi / 18
N = np.array([[np.sqrt(1 - z * z) * np.cos(p), np.sqrt(1 - z * z) * np.sin(p), z] for z in _z for p in _ph])
W = np.array([w / 18 / 2 for w in _w for p in _ph])
K4 = np.sum(N ** 4, 1) - 0.6; Y20 = (3 * N[:, 2] ** 2 - 1) / 2
coef = lambda f, b: float(W @ (f * b) / (W @ (b * b)))


def fn(L):
    A = np.einsum('jlab,ej,el->eab', np.array(L), N, N)
    nu = np.einsum('ea,eab,eb->e', N, A, N)
    tau = (np.trace(A, axis1=1, axis2=2) - nu) / 2
    Pn = np.eye(3)[None] - N[:, :, None] * N[:, None, :]
    B = Pn @ A @ Pn
    sp = np.sqrt(np.maximum(2 * np.einsum('eab,eba->e', B, B) - (2 * tau) ** 2, 0))
    return tau, nu, sp


def sem(v):
    v = [x for x in v if x is not None]
    return [float(np.mean(v)), float(np.std(v, ddof=1) / len(v) ** .5) if len(v) > 1 else None, len(v)]


fam = {}
for b, r in NEW.items():
    if r['flips'] or r['sub'] != 2:
        continue
    fam.setdefault(b.rsplit('_s', 1)[0], []).append(r)
out = dict(single_code=True, labels='C exact/derivation; P point estimate on finite cells', n_new=len(NEW))
# O1 O2
out['O1'] = dict(element_rel_max=max(r['rigid']['element_max'] / r['rigid']['force_scale'] for r in NEW.values()), node_max=max(r['rigid']['node_norm'] for r in NEW.values()))
out['O2_antis_max'] = max(max(r['antis'].values()) for r in NEW.values() if r['antis'])
out['O2_antis0_max'] = max(r['antis0'] or 0 for r in NEW.values())
out['shear_scale'] = {}
# canonical table
tab = {}
for k, rs in sorted(fam.items(), key=lambda x: x[1][0]['V']):
    M = np.diag(np.array(rs[0]['M'])); m = float(M.mean()); h = float(M[2] - .5 * (M[0] + M[1]))
    olds = [OLD.get(f"{k}_s{r['seed']}.json") for r in rs]
    e = dict(V=rs[0]['V'], m=m, h=h, n=len(rs), a=rs[0]['a'], p_c=sem([r['p_c'] for r in rs]), lw=sem([r['lw_min'] for r in rs]), gamma=sem([r['gamma'][0] for r in rs]), kmin=sem([min(v[0] for v in r['kpts'].values()) for r in rs]),
             gamma_pr=sem([r['gamma_pr'][0] for r in rs]), lam_iso=sem([r.get('lam_iso_root') for r in rs]), zener=sem([r['zener'] for r in rs]), Fp_over_N=sem([r['Fp_over_N'] for r in rs]), M_over_NL=sem([r['M_over_NL'] for r in rs]), T_over_NL=sem([r['T_over_NL'] for r in rs]),
             mech=[r['mech'] for r in rs], inertia_max=max(r['inertia_below'] for r in rs), lw_lowest_in=int(sum(1 for r in rs if r['lw_min'] is not None and r['lw_min'] <= r['p_c'] + 1e-12)),
             old_p_c=sem([o['p_c'] for o in olds if o]), old_gamma=sem([o['gamma'][0] for o in olds if o]), old_kmin=sem([min(v[0] for v in o['kpts'].values()) for o in olds if o]),
             repro_zener_vs_old=max(abs(o['zener'] - r['zener']) for o, r in zip(olds, rs) if o) if any(olds) else None, low_pr=[dict(seed=r['seed'], lam=r['gamma'][0], pr=r['gamma_pr'][0], margin=(r['gamma'][0] - r['lw_min']) if r['lw_min'] else None) for r in rs if r['gamma_pr'][0] < .1])
    e['delta_p_c'] = e['p_c'][0] - e['old_p_c'][0]; e['gap_gamma'] = e['gamma'][0] - e['p_c'][0]; e['gap_halfk'] = e['kmin'][0] - e['p_c'][0]
    # k PR
    e['kpt_mean_PR'] = {kk: float(np.mean([r['kpts'][kk][1] for r in rs])) for kk in rs[0]['kpts']}
    # tensor harmonics
    for l in ('0.0', '0.5', '0.85', '0.9', '0.92', 'root'):
        vals = []
        for r in rs:
            T = r.get('tensors', {}).get(l)
            if T is None:
                continue
            tau, nu, sp = fn(T)
            vals.append(dict(t0=float(W @ tau), tK4=coef(tau, K4), tY20=coef(tau, Y20), nK4=coef(nu, K4), nY20=coef(nu, Y20), split=float(np.sqrt(W @ sp ** 2)), tauC=float(np.std(tau))))
        if vals:
            e[l] = {q: sem([v[q] for v in vals]) for q in vals[0]}
    tab[k] = e
out['canonical'] = tab
slopes = [(tab[k]['0.92']['t0'][0] - tab[k]['0.9']['t0'][0]) / .02 for k in ('cubic_5-3', 'cubic_8-5') if k in tab and '0.92' in tab[k] and '0.9' in tab[k]]
tp = float(np.mean(slopes)) if slopes else None
out['dtau0_dlam'] = tp
cub = [k for k in ('cubic_2-1', 'cubic_3-2', 'cubic_5-3', 'cubic_8-5') if k in tab]
out['G_series'] = {k: {l: tab[k][l]['tK4'][0] / tab[k]['m'] for l in ('0.0', '0.5', '0.85', '0.9', '0.92') if l in tab[k]} for k in cub}
out['G_series_sem'] = {k: {l: (tab[k][l]['tK4'][1] or 0) / abs(tab[k]['m']) for l in ('0.0', '0.92') if l in tab[k]} for k in cub}
out['split_over_absm_092'] = {k: tab[k]['0.92']['split'][0] / abs(tab[k]['m']) for k in cub if '0.92' in tab[k]}
out['iso_minus_pc'] = {k: (tab[k]['lam_iso'][0] - tab[k]['p_c'][0]) for k in cub}
out['iso_minus_pc_over_absm'] = {k: (tab[k]['lam_iso'][0] - tab[k]['p_c'][0]) / abs(tab[k]['m']) for k in cub}
pairs = [('tetra_2-1x3-2', 'tetra_3-2x2-1'), ('tetra_5-3x3-2', 'tetra_3-2x5-3'), ('tetra_8-5x5-3', 'tetra_5-3x8-5')]
out['pairs'] = {}
for a, b in pairs:
    if a in tab and b in tab:
        ha, hb = tab[a]['h'], tab[b]['h']; row = dict(h=[ha, hb], V=[tab[a]['V'], tab[b]['V']], p_c=[tab[a]['p_c'], tab[b]['p_c']])
        for l in ('0.0', '0.5', '0.85', '0.9', '0.92'):
            if l in tab[a] and l in tab[b]:
                ya, yb = tab[a][l]['tY20'][0], tab[b][l]['tY20'][0]; sa, sb = tab[a][l]['tY20'][1] or 0, tab[b][l]['tY20'][1] or 0
                row[l] = dict(kH=(ya - yb) / (ha - hb), se=float(np.hypot(sa, sb) / abs(ha - hb)), c=(ya + yb) / (ha ** 2 + hb ** 2), kH_n=(tab[a][l]['nY20'][0] - tab[b][l]['nY20'][0]) / (ha - hb))
        out['pairs'][a + '|' + b] = row
# G regress with h^2
rows = [dict(y=tab[k]['0.92']['tK4'][0], m=tab[k]['m'], h2=tab[k]['h'] ** 2) for k in tab if '0.92' in tab[k] and tab[k]['V'] >= 136]
if len(rows) >= 4:
    X = np.array([[r['m'], r['h2']] for r in rows]); y = np.array([r['y'] for r in rows]); out['tK4_092_on_m_h2'] = dict(beta=np.linalg.lstsq(X, y, rcond=None)[0].tolist(), rms=float(np.sqrt(np.mean((X @ np.linalg.lstsq(X, y, rcond=None)[0] - y) ** 2))), n=len(rows))
# extrapolation
if 'cubic_3-2' in tab and 'cubic_8-5' in tab:
    a, b = tab['cubic_3-2'], tab['cubic_8-5']
    for q in ('p_c', 'lam_iso'):
        s = (a[q][0] - b[q][0]) / (a['m'] - b['m']); out[f'extrap_{q}'] = [float(b[q][0] - s * b['m']), float(s)]
# disorder
dr = []
for b, r in NEW.items():
    if not r['flips'] or r['sub'] != 2:
        continue
    pq = r['tag'].split('_')[1]; can = tab.get('cubic_' + pq)
    c0 = NEW.get(f"cubic_{pq}_s{r['seed']}.json"); o = OLD.get(b.replace('_mc', '_mc'));
    ofile = OLD.get(b)
    dr.append(dict(file=b, pq=pq, V=r['V'], seed=r['seed'], flips=r['flips'], fv=r['flips'] / r['V'], p_c=r['p_c'], lw=r['lw_min'], gamma=r['gamma'][0], pr=r['gamma_pr'][0], nodes=r['gamma_pr'][0] * 4 * r['V'], mech=r['mech'],
                   old_p_c=ofile['p_c'] if ofile else None, old_pr=ofile['gamma_pr'] if ofile else None, rel=(r['p_c'] / c0['p_c'] - 1) if c0 else None, old_rel=None, pv=r['perp_var'], moved=r['moved_fraction'], ex=r['window_excess_max'], exc=r['excess_centered_max'],
                   top_ex_pct=r.get('top_ex_pct'), mode_ex_pct=r.get('mode_ex_pct'), Fp_over_N=r['Fp_over_N'], M_over_NL=r['M_over_NL'], T_over_NL=r['T_over_NL'], regen_ok=(abs(ofile['window_excess_max'] - r['window_excess_max']) < 1e-9 if ofile else None)))
out['disorder_rows'] = dr
lad = {}
for x in dr:
    lad.setdefault((x['pq'], x['flips']), []).append(x)
out['ladder'] = {}
for k, v in sorted(lad.items()):
    pc = [x['p_c'] for x in v]; rel = [x['rel'] for x in v if x['rel'] is not None]; op = [x['old_p_c'] for x in v if x['old_p_c']]
    out['ladder'][f'{k[0]}_f{k[1]}'] = dict(n=len(v), fv=v[0]['fv'], p_c=sem(pc), rel=sem(rel) if rel else None, old_p_c=sem(op) if op else None, delta=(float(np.mean([x['p_c'] - x['old_p_c'] for x in v if x['old_p_c']])) if op else None), delta_max=(float(np.max([abs(x['p_c'] - x['old_p_c']) for x in v if x['old_p_c']])) if op else None),
                                         nodes=float(np.mean([x['nodes'] for x in v])), local=sum(x['mech'] != 'long-wave' for x in v), pv=float(np.mean([x['pv'] for x in v])), raises=sum((x['rel'] or 0) > 0 for x in v), Fp_over_N=float(np.mean([x['Fp_over_N'] for x in v])), M_over_NL=float(np.mean([x['M_over_NL'] for x in v])))
rng = np.random.default_rng(0)


def within(xs, key):
    y = np.array([x['p_c'] for x in xs]); z = np.array([x[key] for x in xs]); lv = np.array([x['flips'] for x in xs])
    ls = [l for l in set(lv) if (lv == l).sum() >= 3]

    def stat(zz):
        v = [spearmanr(y[lv == l], zz[lv == l])[0] for l in ls if np.ptp(zz[lv == l]) > 0]
        return float(np.mean(v)) if v else np.nan
    s0 = stat(z); n = 0; NN = 4000
    for _ in range(NN):
        zz = z.copy()
        for l in ls:
            i = np.where(lv == l)[0]; zz[i] = z[rng.permutation(i)]
        if stat(zz) <= s0:
            n += 1
    return dict(rho=float(spearmanr(y, z)[0]), within=s0, p=(n + 1) / (NN + 1), n=len(xs), levels=len(ls))


rk = {}
for pq in ('3-2', '5-3', '8-5'):
    xs = [x for x in dr if x['pq'] == pq]
    if len(xs) >= 4:
        rk[pq] = {k: within(xs, k) for k in ('pv', 'ex', 'exc', 'moved')}
xs = [x for x in dr if x['pq'] in ('3-2', '5-3', '8-5')]
if xs:
    rk['pooled'] = dict(n=len(xs), pv=float(spearmanr([x['p_c'] for x in xs], [x['pv'] for x in xs])[0]), ex=float(spearmanr([x['p_c'] for x in xs], [x['ex'] for x in xs])[0]))
out['ranking'] = rk
f1 = [x for x in dr if x['flips'] == 1 and x['pq'] == '3-2' and x['rel'] is not None]
if f1:
    out['single_flip_3-2'] = dict(n=len(f1), raises=sum(x['rel'] > 0 for x in f1), mean=float(np.mean([x['rel'] for x in f1])), min=min(x['rel'] for x in f1), max=max(x['rel'] for x in f1))
# sub 4
s4 = {}
for b, r in NEW.items():
    if r['sub'] == 4:
        ref = NEW.get(b.replace('_sub4', ''))
        s4[b] = dict(p_c_sub4=r['p_c'], p_c_sub2=ref['p_c'] if ref else None, gamma_sub4=r['gamma'][0], gamma_sub2=ref['gamma'][0] if ref else None, lw_sub4=r['lw_min'], lw_sub2=ref['lw_min'] if ref else None, antis=max(r['antis'].values()) if r['antis'] else None, rigid=r['rigid'])
out['sub4'] = s4
# gates
g = {}
g['O1'] = dict(value=out['O1'], passed=bool(out['O1']['element_rel_max'] <= 1e-11 and out['O1']['node_max'] <= 1e-10))
g['O2'] = dict(value=out['O2_antis_max'], passed=bool(out['O2_antis_max'] <= 1e-9))
if 'cubic_8-5' in tab:
    v = tab['cubic_8-5']['p_c'][0]; g['P1'] = dict(value=v, passed=bool(.9245 <= v <= .9265))
    v = out['extrap_p_c'][0]; g['P2'] = dict(value=v, passed=bool(abs(v - .9247) <= .002))
    rs = fam['cubic_8-5']; g['P3'] = dict(value=[[r['lw_min'], r['gamma'][0], min(x[0] for x in r['kpts'].values())] for r in rs], passed=all(r['lw_min'] is not None and r['lw_min'] < r['gamma'][0] and r['lw_min'] < min(x[0] for x in r['kpts'].values()) for r in rs))
top = {'3-2': 30000, '5-3': 100000, '8-5': 400000}; refp = {'3-2': -33, '5-3': -41, '8-5': -49}
if all((p, f) in lad for p, f in top.items()):
    vals = {p: out['ladder'][f'{p}_f{f}'] for p, f in top.items()}
    g['P4'] = dict(value={p: [v['rel'][0] * 100, v['nodes'], v['local'], v['n']] for p, v in vals.items()}, max_nodes=max(x['nodes'] for x in dr if top.get(x['pq']) == x['flips']),
                   passed=all(abs(vals[p]['rel'][0] * 100 - refp[p]) <= 5 and vals[p]['local'] == vals[p]['n'] for p in top) and max(x['nodes'] for x in dr if top.get(x['pq']) == x['flips']) <= 20)
s = out['G_series']
if len(cub) == 4 and all('0.92' in s[k] for k in cub):
    vv = [s[k]['0.92'] for k in cub]; g['P5'] = dict(value=vv, passed=bool(len(set(np.sign(vv))) == 1 and max(np.abs(vv)) / max(min(np.abs(vv)), 1e-300) < 2))
if len(out['pairs']) == 3:
    vv = [v['0.92']['kH'] for v in out['pairs'].values()]; g['P6'] = dict(value=vv, passed=bool(len(set(np.sign(vv))) == 1 and max(np.abs(vv)) / max(min(np.abs(vv)), 1e-300) < 2))
if '3-2' in rk and '5-3' in rk:
    g['P7'] = dict(value={p: [rk[p]['ex']['within'], rk[p]['pv']['within']] for p in ('3-2', '5-3')}, passed=all(rk[p]['ex']['within'] < rk[p]['pv']['within'] for p in ('3-2', '5-3')))
out['gates'] = g
json.dump(out, open(os.path.join(here, 'icosa_results.json'), 'w'), indent=1, default=float)
print('n', len(NEW), 'O1', out['O1'], 'O2', out['O2_antis_max'], out['O2_antis0_max'], 'dtau', tp)
for k, e in tab.items():
    print(k.ljust(16), 'V%6d n%d m%+.4f h%+.4f pc %.5f±%.5f (old %.5f d%+.5f) lw%d/%d G %.4f (old %.4f) k %.4f(old %.4f) iso %s PR %.2f F %.3f M %.3f T %.3f' % (e['V'], e['n'], e['m'], e['h'], e['p_c'][0], e['p_c'][1] or 0, e['old_p_c'][0], e['delta_p_c'], e['lw_lowest_in'], e['n'], e['gamma'][0], e['old_gamma'][0], e['kmin'][0], e['old_kmin'][0], None if e['lam_iso'][0] is None else round(e['lam_iso'][0], 5), e['gamma_pr'][0], e['Fp_over_N'][0], e['M_over_NL'][0], e['T_over_NL'][0]), 'lowPR', e['low_pr'])
print('G', json.dumps(out['G_series'])); print('split/|m|', out['split_over_absm_092']); print('iso-pc', out['iso_minus_pc_over_absm'])
print(json.dumps({k: {l: v[l] for l in ('0.0', '0.85', '0.92') if l in v} for k, v in out['pairs'].items()}))
print(out.get('tK4_092_on_m_h2'), out.get('extrap_p_c'), out.get('extrap_lam_iso'))
for k, v in out['ladder'].items():
    print(k, {a: (round(b, 5) if isinstance(b, float) else b) for a, b in v.items() if a not in ('p_c', 'rel', 'old_p_c')}, 'pc', np.round(v['p_c'][:2], 4), 'rel', None if not v['rel'] else np.round(v['rel'][0], 4), 'old', None if not v['old_p_c'] else np.round(v['old_p_c'][0], 4))
print(json.dumps(rk)); print(out.get('single_flip_3-2')); print(out['sub4']); print(json.dumps({k: v.get('passed') for k, v in g.items()}), json.dumps({k: v.get('value') for k, v in g.items()}))

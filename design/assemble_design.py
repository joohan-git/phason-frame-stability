"""Collect design tasks -> design_results.json (all numerics P; ledge of perfect lattices C). Re-runnable.
python assemble.py [OUTDIR ...] (default: out_part1 and out_part2)."""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames', 'penrose', 'phason_paths', 'robustness', 'heldout_test', 'isotropic_loss')]  # shared modules
import glob, json, math, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from design_core import boot_ci, harmonics_general
OUTS = sys.argv[1:] or [str(HERE / 'out_part1'), str(HERE / 'out_part2')]
D = []
for o in OUTS:
    for fn in sorted(glob.glob(o + '/plan_*.json')):
        d = json.load(open(fn)); d['src'] = Path(fn).name; D.append(d)
by = defaultdict(list)
for d in D:
    by[d['task']['kind']].append(d)
CONV = math.pi ** 2 / 12
RES = dict(label_key='P = numerical point estimate or bootstrap interval; C = exact algebra', units=dict(lam='pi^2 EI/l^3 per unit stress', modhat='EA/l (EA = 1); M/E_s = modhat * t/l, t/l = sqrt(3) r', S='(sigma/E_s)/(t/l)^3 = lam pi^2/12', B='sigma_c/E_s = B rho^3, B = lam pi^2/(12 ledge^3)'))
RES['gates'] = json.loads((HERE / 'design_checks.json').read_text()) if (HERE / 'design_checks.json').exists() else {}
RES['exact_member_check'] = json.loads((HERE / 'exact_lattice.json').read_text()) if (HERE / 'exact_lattice.json').exists() else {}
ACT = lambda d: d['task']


def slope(x, y):
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


# ------------------------------------------------------------ A references
A = dict(references={}, tilings={})
for d in by['ref']:
    T = ACT(d); name = T['lattice']; out = []
    for x in d['rows']:
        tl = math.sqrt(3) * x['r']; m = x['moduli']; rho = tl * x['ledge']
        out.append(dict(sub=x['sub'], r=x['r'], rho=rho, ledge=x['ledge'], lam_c=x['lam_c'], lam_lw=x['lam_lw'], lam_bz=x['lam_bz'], k_bz=x['k_bz'], S=x['S'],
                        sigma_c_over_Es=x['S'] * tl ** 3, B=x['lam_c'] * CONV / x['ledge'] ** 3, K_over_Es=m['K'] * tl, G_iso_over_Es=m['T0'] * tl,
                        Gmean_over_Es=m['Gmean'] * tl, Gmin_over_Es=m['Gmin'] * tl, Emean_over_Es=m['Emean'] * tl, Emin_over_Es=m['Emin'] * tl, Emax_over_Es=m['Emax'] * tl,
                        K=m['K'], T0=m['T0'], absR0=m['absR0'], absR1=m['absR1'], Gmean=m['Gmean'], Emean=m['Emean'], demand=x['demand']))
    s2 = [o for o in out if o['sub'] == 2]; r = np.array([o['rho'] for o in s2])
    A['references'][name] = dict(rows=out, exp_K=slope(r[:6], [o['K_over_Es'] for o in s2][:6]), exp_G=slope(r[:6], [o['Gmean_over_Es'] for o in s2][:6]), exp_E=slope(r[:6], [o['Emean_over_Es'] for o in s2][:6]))
pcs = defaultdict(list)
for d in by['pc']:
    T = ACT(d); key = (T['name'], 'random' if T.get('flips') == -1 else 'ordered')
    pcs[key].append(dict(r=T.get('r', 0.03), fseed=T.get('fseed'), V=d['info']['V'], ledge=d['info']['ledge'], lam_c=d['crit']['lam_c'], mode=d['crit']['mode'], lw_min=d['crit']['lw_min'], extra=d['crit']['extra'], c0=d['harm']['c0'], absh4=d['harm']['absh4'], absh2=d['harm']['absh2'], K=d['moduli']['K'], T0=d['moduli']['T0'], Gmean=d['moduli']['Gmean'], Emean=d['moduli']['Emean'], absR0=d['moduli']['absR0'], absR1=d['moduli']['absR1'], demand=d['demand'], sec=d['sec']))
mods = defaultdict(list)
for d in by['moduli']:
    T = ACT(d); key = (T['name'], 'random' if T.get('flips') == -1 else 'ordered')
    for x in d['rows']:
        m = x['moduli']
        mods[key].append(dict(r=x['r'], fseed=T.get('fseed'), V=x['V'], ledge=x['ledge'], K=m['K'], T0=m['T0'], Gmean=m['Gmean'], Gmin=m['Gmin'], Emean=m['Emean'], Emin=m['Emin'], Emax=m['Emax'], absR0=m['absR0'], absR1=m['absR1'], demand=x['demand']))
for key, rows in mods.items():
    rs = sorted({x['r'] for x in rows}); summ = []
    for r in rs:
        rr = [x for x in rows if x['r'] == r]; tl = math.sqrt(3) * r
        f = lambda k: float(np.mean([x[k] for x in rr]))
        summ.append(dict(r=r, rho=tl * rr[0]['ledge'], ledge=rr[0]['ledge'], n=len(rr), K=f('K'), T0=f('T0'), Gmean=f('Gmean'), Emean=f('Emean'), absR0=f('absR0'), absR1=f('absR1'),
                         K_over_Es=f('K') * tl, G_over_Es=f('Gmean') * tl, E_over_Es=f('Emean') * tl, ax_mean=float(np.mean([x['demand']['ax_mean'] for x in rr])), ax_max=float(np.mean([x['demand']['ax_max'] for x in rr])), total_max=float(np.mean([x['demand']['total_max'] for x in rr])),
                         total_max_range=[float(min(x['demand']['total_max'] for x in rr)), float(max(x['demand']['total_max'] for x in rr))]))
    rho = np.array([s['rho'] for s in summ])
    A['tilings'][f'{key[0]}|{key[1]}'] = dict(moduli_vs_r=summ, exp_K=slope(rho[:6], [s['K_over_Es'] for s in summ][:6]), exp_G=slope(rho[:6], [s['G_over_Es'] for s in summ][:6]), exp_E=slope(rho[:6], [s['E_over_Es'] for s in summ][:6]))
for key, rows in pcs.items():
    k = f'{key[0]}|{key[1]}'
    A['tilings'].setdefault(k, {})
    rs = sorted({x['r'] for x in rows}); summ = []
    for r in rs:
        rr = [x for x in rows if x['r'] == r]
        lam = [x['lam_c'] for x in rr]; l = rr[0]['ledge']
        summ.append(dict(r=r, n=len(rr), V=rr[0]['V'], ledge=l, lam_c_mean=float(np.mean(lam)), lam_c_min=float(min(lam)), lam_c_max=float(max(lam)), B=float(np.mean(lam)) * CONV / l ** 3, rho=math.sqrt(3) * r * l, sigma_c_over_Es=float(np.mean(lam)) * CONV * (math.sqrt(3) * r) ** 3,
                         lam=[x['lam_c'] for x in rr], seeds=[x['fseed'] for x in rr], total_max=float(np.mean([x['demand']['total_max'] for x in rr])), ax_max=float(np.mean([x['demand']['ax_max'] for x in rr])), modes=sorted({x['mode'] for x in rr}), K=float(np.mean([x['K'] for x in rr])), Gmean=float(np.mean([x['Gmean'] for x in rr])), Emean=float(np.mean([x['Emean'] for x in rr]))))
    A['tilings'][k]['lam_c_vs_r'] = summ
RES['A'] = A

# ------------------------------------------------------------ B phason paths
st = json.loads((HERE.parent / 'phason_paths' / 'ab_families.json').read_text())['rows']
sf = {(r['family'], r['p'], r['q']): r for r in st}
so = json.loads((HERE.parent / 'phason_paths' / 'window_offsets.json').read_text())['rows']
sof = {(r['family'], r['offset_seed']): r for r in so}
B = dict(rows=[])
LAME = math.pi ** 2 * 0.03 ** 2 / 4
for d in sorted(by['path'], key=lambda d: (d['task']['family'], d['task']['q'], d['task']['off'])):
    T = ACT(d); m = d['moduli0']
    g = np.array([h['lam'] for h in d['h_tau']]); t0 = np.array([h['tau0'] for h in d['h_tau']]); h4 = np.array([h['h4'] for h in d['h_tau']])
    i = int(np.argmin(abs(g - 0.8))); dtau = float(np.polyfit(g[i - 1:i + 2], t0[i - 1:i + 2], 1)[0])
    if T['off'] == 7:
        s = sf.get((T['family'], T['p'], T['q']))
        stored = dict(absh4=s['h4_abs'], c0=s['harm']['c0'], absh2=s['h2_abs']) if s else None
    else:
        s = sof.get((T['family'], T['off']))
        stored = dict(absh4=abs(complex(*s['h4'])), c0=s['c0'], absh2=abs(complex(*s['h2']))) if s else None
    row = dict(family=T['family'], p=T['p'], q=T['q'], off=T['off'], V=d['info']['V'], ledge=d['info']['ledge'], absa=d['absa'], absb=d['absb'], K=m['K'], K_over_ledge=m['K'] / d['info']['ledge'], T0=m['T0'], T1=m['T1'], R0=m['R0'], absR0=m['absR0'], R1=m['R1'], absR1=m['absR1'], Emin=m['Emin'], Emax=m['Emax'], Gmin=m['Gmin'], Gmax=m['Gmax'],
               tau0_at_0p8=float(t0[i]), dtau_dlam=dtau, h4tau_at_0p8=float(h4[i]), lemma_pred_h4=float(h4[i] / abs(dtau)), tangent_pred_h4=d['pred_harm']['h4'], tangent_pred_c0=d['pred_harm']['c0'], zero_load_R0_in_lam=m['absR0'] / LAME, zero_load_T0_in_lam=m['T0'] / LAME, softening_ratio=abs(dtau) / LAME, stored=stored)
    if stored:
        row['lemma_over_stored'] = row['lemma_pred_h4'] / stored['absh4'] if stored['absh4'] else None
        row['stored_h4_over_zero_load_R0'] = stored['absh4'] / (m['absR0'] / LAME) if m['absR0'] else None
    B['rows'].append(row)
sq = lambda rows, xk, yk: slope([r[xk] for r in rows], [r[yk] for r in rows])
fits = {}
for fam, xk in (('B1', 'absb'), ('B2', 'absb'), ('E2a', 'absa'), ('E2diag', 'absa')):
    rows = [r for r in B['rows'] if r['family'] == fam and r['off'] == 7]
    rs = [r for r in rows if r['q'] >= 3]
    fits[fam] = dict(n=len(rows), amp=[r[xk] for r in rows], R0=[r['absR0'] for r in rows], R1=[r['absR1'] for r in rows], K=[r['K'] for r in rows], ledge=[r['ledge'] for r in rows],
                     exp_R0=sq(rows, xk, 'absR0'), exp_R0_q3up=sq(rs, xk, 'absR0'), exp_R1=sq(rows, xk, 'absR1'), exp_R1_q3up=sq(rs, xk, 'absR1'), R0_over_amp=[r['absR0'] / r[xk] for r in rows], R1_over_amp=[r['absR1'] / r[xk] for r in rows], R0_over_amp2=[r['absR0'] / r[xk] ** 2 for r in rows])
B['fits'] = fits
off = {}
for fam in ('B1', 'E2a'):
    rr = [r for r in B['rows'] if r['family'] == fam and r['q'] == 17]
    z = np.array([complex(*r['R0']) for r in rr]); h = np.array([complex(*sof[(fam, r['off'])]['h4']) for r in rr if (fam, r['off']) in sof])
    off[fam] = dict(n=len(rr), mean_absR0=float(np.abs(z).mean()), abs_meanR0=float(abs(z.mean())), mean_absR1=float(np.mean([r['absR1'] for r in rr])), mean_abs_h4=float(np.abs(h).mean()), abs_mean_h4=float(abs(h.mean())), K_range=[float(min(r['K'] for r in rr)), float(max(r['K'] for r in rr))])
B['offsets_12_17'] = off
RES['B'] = B

# ------------------------------------------------------------ C load paths
C = dict(cells={}, references={})
def summarize_states(rows):
    uni = [r for r in rows if r['kind'] == 'uniaxial']
    th = np.array([r['th'] for r in uni]); y = np.array([r['lam_c'] for r in uni])
    out = dict(uniaxial_theta=th.tolist(), uniaxial_lam=y.tolist(), uni_min=float(y.min()), uni_max=float(y.max()), uni_mean=float(y.mean()), uni_ptp_over_mean=float((y.max() - y.min()) / y.mean()), uni_harm=harmonics_general(th, y, ms=(2, 4, 8, 10)), uni_modes=sorted({r['mode'] for r in uni}))
    hy = [r for r in rows if r['kind'] == 'hydro'][0]
    out['hydro'] = dict(lam=hy['lam_c'], mode=hy['mode'])
    out['biaxial'] = [dict(R=r['R'], th=r['th'], lam=r['lam_c'], mode=r['mode'], phi=r.get('phi_min')) for r in rows if r['kind'] == 'biaxial']
    return out
for d in by['load']:
    T = ACT(d); k = f"{T['name']}|off{T['off']}" + (f"|random{T['fseed']}" if T.get('flips') == -1 else '')
    C['cells'][k] = dict(V=d['info']['V'], name=T['name'], off=T['off'], random=T.get('flips') == -1, sec=d['sec'], stress_resid_max=d['stress_resid_max'], **summarize_states(d['rows']))
    C['cells'][k]['all'] = [dict(kind=r['kind'], th=r['th'], R=r['R'], lam=r['lam_c']) for r in d['rows']]
for d in by['refload']:
    T = ACT(d)
    C['references'][f"{T['lattice']}|sub{T['sub']}"] = dict(ledge=d['ledge'], **summarize_states(d['rows']))
# random/ordered per state
rat = {}
for nm in ('B112/17', 'fib4'):
    o = [v for k, v in C['cells'].items() if v['name'] == nm and not v['random']]
    rd = [v for k, v in C['cells'].items() if v['name'] == nm and v['random']]
    if not o or not rd:
        continue
    o = o[0]; base = np.array([a['lam'] for a in o['all']])
    arr = np.array([[a['lam'] for a in r['all']] for r in rd]) / base
    labs = [dict(kind=a['kind'], th=a['th'], R=a['R']) for a in o['all']]
    i_h = [i for i, a in enumerate(o['all']) if a['kind'] == 'hydro'][0]
    i_u = [i for i, a_ in enumerate(o['all']) if a_['kind'] == 'uniaxial']
    uni_mean_each = arr[:, i_u].mean(1)
    rmin = np.array([min(a_['lam'] for a_ in r['all'] if a_['kind'] == 'uniaxial') for r in rd]) / min(a_['lam'] for a_ in o['all'] if a_['kind'] == 'uniaxial')
    ptp = np.array([r['uni_ptp_over_mean'] for r in rd])
    extra_ = dict(hydro_ci=boot_ci(arr[:, i_h]), uni_mean=float(uni_mean_each.mean()), uni_mean_ci=boot_ci(uni_mean_each), state_ci=[boot_ci(arr[:, j]) for j in range(arr.shape[1])], uni_weakest_ratio=float(rmin.mean()), uni_weakest_ratio_ci=boot_ci(rmin), random_ptp_mean=float(ptp.mean()), random_ptp_range=[float(ptp.min()), float(ptp.max())], uni_state_min=float(arr.mean(0)[i_u].min()), uni_state_max=float(arr.mean(0)[i_u].max()))
    rat[nm] = dict(n_random=len(rd), mean_ratio=arr.mean(0).tolist(), states=labs, min_ratio=float(arr.mean(0).min()), max_ratio=float(arr.mean(0).max()), hydro_ratio=float(arr.mean(0)[i_h]), **extra_)
C['random_over_ordered'] = rat
C['mode_counts'] = {}
for d in by['load']:
    T = ACT(d); key = ('random' if T.get('flips') == -1 else 'ordered') + '|' + T['name']
    c_ = C['mode_counts'].setdefault(key, {})
    for r_ in d['rows']:
        c_[r_['mode']] = c_.get(r_['mode'], 0) + 1
sel = {}
for nm in ('B15/7', 'E2a5/7', 'E2a12/17', 'B112/17', 'B129/41'):
    v = [x for x in C['cells'].values() if x['name'] == nm and not x['random']]
    if not v:
        continue
    sel[nm] = dict(n=len(v), m2=[x['uni_harm']['h2'] for x in v], m4=[x['uni_harm']['h4'] for x in v], m8=[x['uni_harm']['h8'] for x in v], mean_m2=float(np.mean([x['uni_harm']['h2'] for x in v])), mean_m4=float(np.mean([x['uni_harm']['h4'] for x in v])), c0=float(np.mean([x['uni_harm']['c0'] for x in v])))
    th = np.array(v[0]['uniaxial_theta']); ys = np.array([x['uniaxial_lam'] for x in v]).mean(0)
    h = harmonics_general(th, ys, ms=(2, 4, 8, 10)); sel[nm]['phase_avg'] = h
C['uniaxial_harmonics'] = sel
RES['C'] = C

# ------------------------------------------------------------ D imperfection MC
base = {}
for d in by['base']:
    T = ACT(d)
    if T.get('flips') is None:
        base[T['name']] = d
Dd = dict(cells={})
mcs = defaultdict(list)
for d in by['mc']:
    T = ACT(d); mcs[(T['name'], T['mc'], T['sigma'])].append(d)
flips = defaultdict(list)
for d in by['base']:
    T = ACT(d)
    if T.get('flips') is not None:
        flips[(T['name'], T['flips'])].append(d)
names = sorted({k[0] for k in mcs} | {k[0] for k in flips})
for nm in names:
    b = base.get(nm)
    if not b:
        continue
    lam0 = b['lam_c']; h40 = complex(*b['harm']['h4']); h20 = complex(*b['harm']['h2'])
    cell = dict(V=b['info']['V'], lam0=lam0, c0=b['harm']['c0'], absh4=abs(h40), absh2=abs(h20), K=b['K'], Gmean=b['Gmean'], total_max=b['total_max'], levels=[], flips=[])
    for (n, kind, sig), rows in sorted(mcs.items()):
        if n != nm:
            continue
        x = np.array([r['lam_c'] for r in rows]); h4 = np.array([complex(*r['harm']['h4']) for r in rows]); h2 = np.array([complex(*r['harm']['h2']) for r in rows])
        c0 = np.array([r['harm']['c0'] for r in rows])
        drop = 1 - x / lam0
        reh = boot_ci(h4.real); imh = boot_ci(h4.imag)
        noise = float(np.sqrt(np.mean(np.abs(h4 - h4.mean()) ** 2)))
        cell['levels'].append(dict(kind=kind, sigma=sig, n=len(rows), lam_mean=float(x.mean()), lam_sd=float(x.std(ddof=1)) if len(x) > 1 else None, lam_min=float(x.min()), lam_max=float(x.max()), drop_mean_pct=float(drop.mean() * 100), drop_ci_pct=[float(a * 100) for a in boot_ci(drop)], sd_pct=float(x.std(ddof=1) / lam0 * 100),
                                   h4_mean=[float(h4.mean().real), float(h4.mean().imag)], h4_re_ci=reh, h4_im_ci=imh, h4_noise_rms=noise, h2_mean=[float(h2.mean().real), float(h2.mean().imag)], h2_rms=float(np.sqrt(np.mean(np.abs(h2) ** 2))), c0_mean=float(c0.mean()), modes={m: sum(r['mode'] == m for r in rows) for m in {r['mode'] for r in rows}},
                                   K_mean=float(np.mean([r['K'] for r in rows])), G_mean=float(np.mean([r['Gmean'] for r in rows])), total_max_mean=float(np.mean([r['total_max'] for r in rows])), h4_noise_over_h2_rms=noise / float(np.sqrt(np.mean(np.abs(h2) ** 2))), h2_mean_abs=float(abs(h2.mean())), h2_stderr=float(np.sqrt(np.mean(np.abs(h2 - h2.mean()) ** 2) / max(len(h2) - 1, 1))), lam_ci=boot_ci(x)))
    for (n, k), rows in sorted(flips.items(), key=lambda kv: (kv[0][0], 1e9 if kv[0][1] == -1 else kv[0][1])):
        if n != nm:
            continue
        x = np.array([r['lam_c'] for r in rows]); drop = 1 - x / lam0
        cell['flips'].append(dict(lam_ci=boot_ci(x), k=('equilibrated' if k == -1 else k), n=len(rows), lam_mean=float(x.mean()), lam_sd=float(x.std(ddof=1)) if len(x) > 1 else None, drop_mean_pct=float(drop.mean() * 100), drop_ci_pct=[float(a * 100) for a in boot_ci(drop)], modes={m: sum(r['mode'] == m for r in rows) for m in {r['mode'] for r in rows}}, K_mean=float(np.mean([r['K'] for r in rows])), G_mean=float(np.mean([r['Gmean'] for r in rows])), total_max_mean=float(np.mean([r['total_max'] for r in rows])), accepted_mean=float(np.mean([r['info'].get('accepted', 0) for r in rows]))))
    # noise crossover per kind: rms(h4 noise) linear in sigma -> sigma_x where noise = |h4_0|
    cross = {}
    for kind in ('E', 't', 'node'):
        lv = [l for l in cell['levels'] if l['kind'] == kind]
        if len(lv) >= 2:
            s = np.array([l['sigma'] for l in lv]); y = np.array([l['h4_noise_rms'] for l in lv])
            c = float(np.sum(s * y) / np.sum(s * s))
            cross[kind] = dict(noise_per_sigma=c, sigma_x=abs(h40) / c if c else None, noise_exponent=slope(s, y))
    cell['sigma_cross'] = cross
    Dd['cells'][nm] = cell
RES['D'] = Dd

# ------------------------------------------------------------ robust joins and penalty against slenderness
ROBUST = dict(note='robustness/out_eq: equilibrated random B1 12/17, 279069 attempts (200 sweeps), supersedes the robust random rows; robustness/out: ordered thresholds')
_eq = [json.load(open(f)) for f in glob.glob(str(HERE.parent / 'robustness' / 'out_equilibrated' / '*.json'))]
_or = [json.load(open(f)) for f in glob.glob(str(HERE.parent / 'robustness' / 'out' / '*.json'))]
pcmin = lambda x: min(x['pc_lw_min'], x['pc_gamma'], x['pc_hh'])
ROBUST['ordered'] = {nm: {str(x['r']): dict(lam_c=pcmin(x), c0=x['c0']) for x in _or if x['group'] == 'slender' and x['name'] == nm} for nm in ('B15/7', 'B112/17', 'pell29/41')}
ROBUST['random_eq'] = {}
PEN = {}
for r in sorted({x['r'] for x in _eq}):
    v = np.array([pcmin(x) for x in _eq if x['r'] == r]); o_ = ROBUST['ordered']['B112/17'][str(r)]['lam_c']
    ROBUST['random_eq'][str(r)] = dict(n=len(v), lam_mean=float(v.mean()), lam_sd=float(v.std(ddof=1)), lam_ci=boot_ci(v), lam=v.tolist(), ordered=o_, ratio=float(v.mean() / o_), ratio_ci=[float(a_ / o_) for a_ in boot_ci(v)])
PEN['B112/17 robust_eq n=12'] = {k: dict(n=v['n'], ratio=v['ratio'], ratio_ci=v['ratio_ci'], lam=v['lam_mean'], ordered=v['ordered']) for k, v in ROBUST['random_eq'].items()}
for key, ordsrc in (('B112/17', 'B112/17'), ('B15/7', 'B15/7'), ('fib4', None), ('fib5', None)):
    rr = A['tilings'].get(key + '|random', {}).get('lam_c_vs_r', [])
    tab = {}
    for x in rr:
        o_ = ROBUST['ordered'][ordsrc][str(x['r'])]['lam_c'] if ordsrc else [y for y in A['tilings'][key + '|ordered']['lam_c_vs_r'] if y['r'] == x['r']][0]['lam_c_mean']
        v = np.array(x['lam'])
        tab[str(x['r'])] = dict(n=len(v), ratio=float(v.mean() / o_), ratio_ci=[float(a_ / o_) for a_ in boot_ci(v)], lam=float(v.mean()), ordered=float(o_))
    PEN[key + ' design'] = tab
ROBUST['penalty_vs_r'] = PEN
RES['robust'] = ROBUST
# random tilings at r = 0.03 from the largest ensemble (D, n = 24)
RR = {}
for nm, c in Dd['cells'].items():
    f_ = [x for x in c['flips'] if x['k'] == 'equilibrated']
    if not f_:
        continue
    f_ = f_[0]
    led = base[nm]['info']['ledge']
    RR[nm] = dict(V=c['V'], n=f_['n'], lam=f_['lam_mean'], lam_ci=f_['lam_ci'], B=f_['lam_mean'] * CONV / led ** 3, ledge=led, K=f_['K_mean'], K_change_pct=(f_['K_mean'] / c['K'] - 1) * 100, G=f_['G_mean'], G_change_pct=(f_['G_mean'] / c['Gmean'] - 1) * 100, lam_change_pct=-f_['drop_mean_pct'], lam_change_ci=[-f_['drop_ci_pct'][1], -f_['drop_ci_pct'][0]], K0=c['K'], G0=c['Gmean'], lam0=c['lam0'], B0=c['lam0'] * CONV / led ** 3, K_over_Esrho=f_['K_mean'] / led, K0_over_Esrho=c['K'] / led, demand=f_['total_max_mean'], demand0=c['total_max'])
A['random_rad003_D'] = RR
_nx = {nm: Dd['cells'][nm]['sigma_cross']['E']['noise_per_sigma'] for nm in ('B15/7', 'B112/17', 'B129/41') if nm in Dd['cells']}
if len(_nx) == 3:
    Dd['noise_V_exponent_E'] = slope([Dd['cells'][n]['V'] for n in _nx], list(_nx.values()))

# ------------------------------------------------------------ E strength crossover (simple estimate)
Eps = [0.002, 0.005, 0.01, 0.02, 0.05]
E = dict(note='simple estimate: first-fibre elastic yield in the linear prestress, no plasticity, knockdown or node effect', eps_y=Eps, rows={})
def cross_rows(rs, lam_of_r, dem_of_r, ledge, key):
    res = {}
    rs = np.array(rs)
    for ey in Eps:
        f = np.array([lam_of_r(r) * math.pi ** 2 * r ** 2 / 4 * dem_of_r(r) - ey for r in rs])
        rstar = None
        for i in range(len(rs) - 1):
            if f[i] <= 0 < f[i + 1]:
                la, lb = math.log(rs[i]), math.log(rs[i + 1]); ya, yb = math.log(f[i] + ey), math.log(f[i + 1] + ey)
                rstar = math.exp(la + (math.log(ey) - ya) / (yb - ya) * (lb - la)); break
        res[str(ey)] = dict(r_star=rstar, rho_star=None if rstar is None else math.sqrt(3) * rstar * ledge, tl_star=None if rstar is None else math.sqrt(3) * rstar, below_range=bool(f[0] > 0), above_range=bool(f[-1] <= 0))
    return res
for name, a in A['references'].items():
    rr = [o for o in a['rows'] if o['sub'] == 8]
    rs = [o['r'] for o in rr]; l = {o['r']: o['lam_c'] for o in rr}; d1 = {o['r']: o['demand']['total_max'] for o in rr}; d2 = {o['r']: o['demand']['ax_max'] for o in rr}
    E['rows'][name] = dict(lam_source='this code, sub 8, per r', first_fibre=cross_rows(rs, l.get, d1.get, rr[0]['ledge'], 'total'), axial_only=cross_rows(rs, l.get, d2.get, rr[0]['ledge'], 'ax'), demand_at_0p03=d1[0.03], lam_at_0p03=l[0.03])
ROBUSTR = {'B15/7|ordered': 'B15/7', 'B112/17|ordered': 'B112/17', 'B129/41|ordered': 'pell29/41'}
def _row(key, rs, d1, d2, led, lamd, src):
    grid = sorted(lamd); lam = (lambda r: float(np.interp(r, grid, [lamd[x] for x in grid])))
    ff = cross_rows(rs, lam, d1.get, led, 'total'); aa = cross_rows(rs, lam, d2.get, led, 'ax')
    for v in list(ff.values()) + list(aa.values()):
        v['lam_frozen_below_grid'] = bool(v['r_star'] is not None and v['r_star'] < grid[0])
    E['rows'][key] = dict(lam_source=src, lam_r_grid=grid, first_fibre=ff, axial_only=aa, demand_at_0p03=d1.get(0.03), lam_at_0p03=lam(0.03))
for key, a in A['tilings'].items():
    if not ('lam_c_vs_r' in a or key in ROBUSTR or key == 'B112/17|random'):
        continue
    pc = {x['r']: x for x in a.get('lam_c_vs_r', [])}
    if key in ROBUSTR:
        lamd = {float(r): v['lam_c'] for r, v in ROBUST['ordered'][ROBUSTR[key]].items()}; src = 'robust ordered, four r'
    elif key == 'B112/17|random':
        lamd = {float(r): v['lam_mean'] for r, v in ROBUST['random_eq'].items()}; src = 'robust out_eq n = 12, four r'
    elif len(pc) >= 2:
        lamd = {r: x['lam_c_mean'] for r, x in pc.items()}; src = f"design n = {min(x['n'] for x in pc.values())}, four r"
    else:
        continue
    if 'moduli_vs_r' in a:
        mv = a['moduli_vs_r']; rs = [x['r'] for x in mv]; d1 = {x['r']: x['total_max'] for x in mv}; d2 = {x['r']: x['ax_max'] for x in mv}; led = mv[0]['ledge']
        src += f", demand n = {mv[0]['n']}, nine r"
    else:
        rs = sorted(pc); d1 = {r: pc[r]['total_max'] for r in rs}; d2 = {r: pc[r]['ax_max'] for r in rs}; led = pc[rs[0]]['ledge']
        src += f", demand n = {pc[rs[0]]['n']}, four r"
    _row(key, rs, d1, d2, led, lamd, src)
# sensitivity: random 12/17 with the n = 8 demand on four r
_pc = {x['r']: x for x in A['tilings']['B112/17|random']['lam_c_vs_r']}
if len(_pc) >= 2:
    _row('B112/17|random, demand n = 8', sorted(_pc), {r: _pc[r]['total_max'] for r in _pc}, {r: _pc[r]['ax_max'] for r in _pc}, _pc[0.03]['ledge'], {float(r): v['lam_mean'] for r, v in ROBUST['random_eq'].items()}, 'robust out_eq n = 12, demand design n = 8, four r')
RES['E'] = E

# ------------------------------------------------------------ exploratory (not preregistered): bulk modulus as tracker
from scipy.stats import spearmanr
X = dict(note='exploratory, not preregistered. Resolved by the registered test: K is a proxy (by-product of the same non-affine hydrostatic response), not a cause; the transverse contraction excludes K, and K adds nothing beyond the forbidden-patch fraction.')
fx = HERE / 'fixed_size_moduli.json'
if fx.exists():
    rows = json.loads(fx.read_text())['rows']; X['fixedV'] = {}
    for c, rr in rows.items():
        kl = np.array([r['K'] / r['ledge'] for r in rr])
        g = lambda k: np.array([r[k] for r in rr])
        fit = np.polyfit(g('new2'), kl, 1)
        X['fixedV'][c] = dict(n=len(rr), V=g('V').tolist(), delta=g('delta').tolist(), new2=g('new2').tolist(), K=g('K').tolist(), K_over_ledge=kl.tolist(), deficit=g('deficit').tolist(), c0_mean=g('c0_mean').tolist(), total_max=g('total_max').tolist(),
                              rho_K_new2=float(spearmanr(kl, g('new2'))[0]), rho_K_delta=float(spearmanr(kl, g('delta'))[0]), rho_K_deficit=float(spearmanr(kl, g('deficit'))[0]), rho_delta_deficit=float(spearmanr(g('delta'), g('deficit'))[0]), rho_new2_deficit=float(spearmanr(g('new2'), g('deficit'))[0]), fit_K_over_ledge_vs_new2=fit.tolist(), pearson_K_deficit=float(np.corrcoef(kl, g('deficit'))[0, 1]))
fl_rows = defaultdict(list)
for d in by['base']:
    T = ACT(d)
    fl_rows[T['name']].append((T.get('flips'), d['K'], d['lam_c'], d['total_max']))
X['flip_ladder_per_sample'] = {}
for nm, rr in fl_rows.items():
    K = np.array([r[1] for r in rr]); L = np.array([r[2] for r in rr])
    eq = [r for r in rr if r[0] == -1]
    X['flip_ladder_per_sample'][nm] = dict(n=len(rr), rho_K_lam=float(spearmanr(K, L)[0]), n_eq=len(eq), rho_K_lam_within_equilibrated=float(spearmanr([r[1] for r in eq], [r[2] for r in eq])[0]) if len(eq) > 2 else None, rho_demand_lam=float(spearmanr([r[3] for r in rr], L)[0]))
for fam, xk in (('B2', 'absb'), ('E2a', 'absa'), ('E2diag', 'absa'), ('B1', 'absb')):
    rr = [r for r in B['rows'] if r['family'] == fam and r['off'] == 7 and r['q'] >= 7]
    K0 = 0.5144846 / 2.4142136
    X.setdefault('K_deficit_paths', {})[fam] = dict(q=[r['q'] for r in rr], amp=[r[xk] for r in rr], K_over_ledge_deficit=[K0 - r['K_over_ledge'] for r in rr], per_amp=[(K0 - r['K_over_ledge']) / r[xk] for r in rr])
RES['exploratory'] = X
# ---- independent checks and bookkeeping
IC = {}
_ex = RES['exact_member_check'].get('rows', {})
IC['reference_exact_member'] = {}
for n_, a_ in A['references'].items():
    x_ = [o for o in a_['rows'] if o['sub'] == 8 and o['r'] == 0.005][0]
    e_ = _ex[n_]; ev = min(e_['lam_finite_k'], e_['lam_near_longwave'])
    IC['reference_exact_member'][n_] = dict(exact=ev, frame_sub8=x_['lam_c'], rel=abs(ev / x_['lam_c'] - 1))
IC['triangular_rotation_closed_form'] = RES['exact_member_check'].get('triangular_rotation_closed_form')
heldout = HERE.parent / 'heldout_test' / 'heldout_results.json'
if heldout.exists():
    IC['heldout_cross_check'] = json.loads(heldout.read_text()).get('cross_check')
_f = {str(f_['k']): f_ for f_ in Dd['cells']['B112/17']['flips']}
IC['independent_flip_ladder_12_17'] = dict(independent_100_pct=2.48, independent_1000_pct=25.29, source='independent flip ladder', design_100=_f['100']['drop_mean_pct'], design_100_ci=_f['100']['drop_ci_pct'], design_1000=_f['1000']['drop_mean_pct'], design_1000_ci=_f['1000']['drop_ci_pct'])
RES['independent_checks'] = IC
_rb = [r for r in B['rows'] if r['stored'] and r['off'] == 7]
B['lemma_summary'] = dict(n=len(_rb), root_max_rel=max(abs(r['tangent_pred_h4'] / r['stored']['absh4'] - 1) for r in _rb), root_max_rel_q7=max(abs(r['tangent_pred_h4'] / r['stored']['absh4'] - 1) for r in _rb if r['q'] >= 7), onepoint_within_5pct=sum(abs(r['lemma_over_stored'] - 1) <= 0.05 for r in _rb), onepoint_outliers=[(r['family'], r['p'], r['q'], r['lemma_over_stored']) for r in _rb if abs(r['lemma_over_stored'] - 1) > 0.05], h4_over_R0_B1=[r['stored_h4_over_zero_load_R0'] for r in _rb if r['family'] == 'B1'], h4_over_amp_B1=[r['stored']['absh4'] / r['absb'] for r in _rb if r['family'] == 'B1'])
fr = HERE / 'refinement.json'
if fr.exists():
    RES['C']['fine_phi_check'] = json.loads(fr.read_text())

RES['timing'] = {}
tm = defaultdict(list)
for d in D:
    T = d['task']; tm[f"{T['kind']}|{T.get('name', T.get('lattice'))}"].append(d['sec'])
RES['timing'] = {k: dict(n=len(v), mean=float(np.mean(v)), max=float(np.max(v)), total=float(np.sum(v))) for k, v in sorted(tm.items())}
RES['n_tasks_done'] = len(D)
(HERE / 'design_results.json').write_text(json.dumps(RES, indent=1), encoding='utf-8')
print('tasks', len(D), 'written')

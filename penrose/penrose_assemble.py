"""Assemble penrose_results.json from penrose_cusp_*.json, penrose_offsets_*.json, penrose_checks.json, penrose_disorder.json.
Every number is P (point estimate, finite periodic cells, sampled k)."""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames',)]  # shared modules
import json, math, re
from pathlib import Path
import numpy as np
from scipy import stats

H = Path(__file__).parent
J = lambda f: json.loads((H / f).read_text(encoding='utf-8'))


def base(name):
    return re.sub(r'_(offset\d|off\d)$', '', name)


# ---------------------------------------------------------------- cusp: pool all offsets per cell
cells = {}
for f in ['penrose_cusp_aline.json', 'penrose_cusp_bline.json', 'penrose_cusp_vline.json', 'penrose_cusp_controls.json',
          'penrose_offsets_1.json', 'penrose_offsets_2.json']:
    if not (H / f).exists():
        continue
    for r in J(f)['rows']:
        if 'dqhalf' in r['name']:
            continue
        cells.setdefault(base(r['name']), []).append(dict(src=f, h4=complex(*r['harm']['h4']), h2=complex(*r['harm']['h2']),
                                                          c0=r['harm']['c0'], pcmin=r['pc_min_all'], lwmin=r['pc_lw_min_sampled'],
                                                          a=r['a'][0], b=r['b'][0], v=r['v'][0], V=r['V'],
                                                          kmin=min(r['pc_k'].values())))
tab = {}
for nm, rs in cells.items():
    h4 = np.array([x['h4'] for x in rs]); c0 = np.array([x['c0'] for x in rs]); pm = np.array([x['pcmin'] for x in rs])
    r0 = rs[0]
    tab[nm] = dict(V=r0['V'], a=r0['a'], b=r0['b'], v=r0['v'], n_offsets=len(rs),
                   h4_re_mean=float(h4.real.mean()), h4_im_mean=float(h4.imag.mean()),
                   h4_re_sd=float(h4.real.std(ddof=1)) if len(rs) > 1 else None,
                   abs_h4_mean=float(np.abs(h4).mean()), abs_h2_max=float(max(abs(x['h2']) for x in rs)),
                   c0_mean=float(c0.mean()), c0_sd=float(c0.std(ddof=1)) if len(rs) > 1 else None,
                   pc_min_mean=float(pm.mean()), finite_k_min=float(min(x['kmin'] for x in rs)),
                   kappa_a_eff=float(h4.real.mean() / r0['a']) if abs(r0['a']) > 1e-12 else None,
                   sources=sorted({x['src'] for x in rs}))
out = dict(status='P', units='p_c/(EA lambda_E) = p_c/(pi^2 EI/l^3), stress protocol, 2 el./edge, r=0.03',
           convention='h4 from lam(phi)=c0+Re[h2 e^{-2i phi}]+Re[h4 e^{-4i phi}], 8 directions, Richardson DQ->0',
           cells=tab)

# matched-|a| pairs (a-line a_k vs Fibonacci fib_(k-1): same a, b ratio -18)
pairs = []
for k in range(2, 7):
    A_, B_ = tab.get(f'a_{k}'), tab.get(f'b_fib{k-1}')
    if not A_ or not B_:
        continue
    db = B_['b'] - A_['b']; dh = B_['h4_re_mean'] - A_['h4_re_mean']
    se = math.sqrt(((A_['h4_re_sd'] or 0) ** 2 / A_['n_offsets']) + ((B_['h4_re_sd'] or 0) ** 2 / B_['n_offsets']))
    pairs.append(dict(pair=f'a_{k} vs b_fib{k-1}', a=A_['a'], b_aline=A_['b'], b_bline=B_['b'],
                      h4_aline=A_['h4_re_mean'], h4_bline=B_['h4_re_mean'], n_off=(A_['n_offsets'], B_['n_offsets']),
                      kappa_a_aline=A_['kappa_a_eff'], kappa_a_bline=B_['kappa_a_eff'],
                      kappa_b_est=dh / db, kappa_b_se=se / abs(db), ratio_kb_over_ka=(dh / db) / A_['kappa_a_eff']))
out['matched_pairs'] = pairs

# joint regression Re h4 = ka a + kb b + mu b^2 over in-plane cells with V >= 150 (offset means, weights 1/|a|)
inpl = [t for n, t in tab.items() if (n.startswith('a_') or n.startswith('b_fib')) and t['V'] >= 150]
X = np.array([[t['a'], t['b'], t['b'] ** 2] for t in inpl]); y = np.array([t['h4_re_mean'] for t in inpl])
w = 1 / np.abs(X[:, 0]); coef, *_ = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)
out['joint_fit_h4'] = dict(model='Re h4 = ka a + kb b + mu b^2, weights 1/|a|, V>=150', n=len(inpl),
                           ka=float(coef[0]), kb=float(coef[1]), mu=float(coef[2]))
# local exponent along the a-line, offset means
al = sorted([(abs(t['a']), abs(t['h4_re_mean']), n) for n, t in tab.items() if n.startswith('a_')], reverse=True)
out['aline_local_alpha'] = [dict(between=f'{al[i][2]}-{al[i+1][2]}', alpha=math.log(al[i][1] / al[i + 1][1]) / math.log(al[i][0] / al[i + 1][0]))
                            for i in range(len(al) - 1)]
lg = np.log([x[0] for x in al]); lh = np.log([x[1] for x in al])
out['aline_powerfit_abs_h4_vs_abs_a'] = dict(alpha=float(np.polyfit(lg, lh, 1)[0]), alpha_last3=float(np.polyfit(lg[-3:], lh[-3:], 1)[0]))
# total drop on each line: lam_QC - pc_min = c |a|^alpha; lam_QC = c0 of the largest in-plane cell
lamqc = max(t['c0_mean'] for n, t in tab.items() if n.startswith('b_fib') or n.startswith('a_'))
out['lambda_QC_est'] = dict(value=lamqc, rule='max offset-mean c0 over in-plane cells (largest Fibonacci cells)')
fits = {}
for pre in ('a_', 'b_fib'):
    pts = [(abs(t['a']), lamqc - t['pc_min_mean']) for n, t in tab.items() if n.startswith(pre) and lamqc - t['pc_min_mean'] > 0 and t['V'] >= 50]
    if len(pts) >= 3:
        x, yy = np.log([p[0] for p in pts]), np.log([p[1] for p in pts])
        sl, ic = np.polyfit(x, yy, 1)
        fits[pre.rstrip('_')] = dict(alpha_vs_abs_a=float(sl), c=float(math.exp(ic)), n=len(pts))
out['total_drop_fits'] = fits
# h2 on the b-line (direct Lemma-1 test: b has the m for h = 2)
out['bline_h2_over_b'] = {n: t['abs_h2_max'] / abs(t['b']) for n, t in tab.items() if n.startswith('b_fib')}
out['bline_h4_over_b'] = {n: t['abs_h4_mean'] / abs(t['b']) for n, t in tab.items() if n.startswith('b_fib')}
# v-line: residual after the a contribution
ka = out['joint_fit_h4']['ka']
out['vline'] = {n: dict(V=t['V'], v=t['v'], a=t['a'], c0=t['c0_mean'], kappa_v_est=(t['h4_re_mean'] - ka * t['a']) / t['v'])
                for n, t in tab.items() if n.startswith('v_')}

# ---------------------------------------------------------------- checks
if (H / 'penrose_checks.json').exists():
    ch = J('penrose_checks.json')
    out['dq_ladder'] = [dict(name=r['name'], c0=r['harm']['c0'], abs_h4=r['abs_h4'], lwmin=r['pc_lw_min_sampled']) for r in ch['dq_ladder']]
    out['gpt_scan'] = [dict(name=r['name'], s=r['s'], V=r['V'], c0=r['harm']['c0'], abs_h4=r['abs_h4'], lwmin=r['pc_lw_min_sampled'],
                            finite_k=r['pc_k']) for r in ch['gpt_scan']]
cc = J('penrose_cusp_controls.json')['rows']
dqh = [r for r in cc if 'dqhalf' in r['name']]
if dqh:
    out['dq_half_fib6'] = dict(c0=dqh[0]['harm']['c0'], h4=dqh[0]['harm']['h4'], lwmin=dqh[0]['pc_lw_min_sampled'])

# ---------------------------------------------------------------- disorder
if (H / 'penrose_disorder.json').exists():
    d = J('penrose_disorder.json'); rows = d['rows']
    can = rows[0]['pc']
    dis = [r for r in rows[1:]]
    pc = np.array([r['pc'] for r in dis]); ex = np.array([r['mean_excess'] for r in dis]); mv = np.array([r['moved_fraction'] for r in dis])
    def sp(x, y):
        rho, p = stats.spearmanr(x, y); return dict(rho=float(rho), p=float(p), n=len(x))
    # partial Spearman (rank residuals)
    rp, re_, rm = stats.rankdata(pc), stats.rankdata(ex), stats.rankdata(mv)
    res = lambda u, z: u - np.polyval(np.polyfit(z, u, 1), z)
    pr = stats.pearsonr(res(rp, rm), res(re_, rm))
    pr2 = stats.pearsonr(res(rp, re_), res(rm, re_))
    out['disorder'] = dict(system=d['system'], canonical_pc=can,
                           rows=[dict(name=r['name'], moved=r['moved_fraction'], mean_excess=r['mean_excess'], max_excess=r['max_excess'],
                                      pc=r['pc'], drop_pct=100 * (can - r['pc']) / can, pc_gamma=r['pc_gamma'], pc_lw_min=r['pc_lw_min'],
                                      mode_PR=r['mode_PR'], mode_weight_outside=r['mode_weight_outside'], frac_outside=r['frac_outside_W'])
                                 for r in dis],
                           spearman_pc_vs_mean_excess=sp(pc, ex), spearman_pc_vs_moved=sp(pc, mv),
                           partial_rank_pc_excess_given_moved=dict(r=float(pr[0]), p=float(pr[1])),
                           partial_rank_pc_moved_given_excess=dict(r=float(pr2[0]), p=float(pr2[1])))
    mx = np.array([r['max_excess'] for r in dis]); rx = stats.rankdata(mx)
    out['disorder']['spearman_pc_vs_max_excess'] = sp(pc, mx)
    out['disorder']['partial_rank_pc_maxexcess_given_moved'] = dict(zip(('r', 'p'), map(float, stats.pearsonr(res(rp, rm), res(rx, rm)))))
    out['disorder']['partial_rank_pc_moved_given_maxexcess'] = dict(zip(('r', 'p'), map(float, stats.pearsonr(res(rp, rx), res(rm, rx)))))
    out['disorder']['spearman_within_protocol'] = {}
    for tag in ('uniform', 'disk'):
        idx = [i for i, r in enumerate(dis) if r['name'].startswith(tag)]
        out['disorder']['spearman_within_protocol'][tag] = dict(mean_excess=sp(pc[idx], ex[idx]), max_excess=sp(pc[idx], mx[idx]), moved=sp(pc[idx], mv[idx]))
    small = [r for r in dis if r.get('k') in (1, 3, 10)]
    out['disorder']['small_k_sign'] = [dict(name=r['name'], dpc=r['pc'] - can) for r in small]
if (H / 'penrose_isotropic.json').exists():
    iso = J('penrose_isotropic.json')
    out['isotropic_deficit_lines'] = {k: iso[k] for k in ('fit_free', 'fit_linear', 'fit_quadratic', 'points')}
    out['vertex_star_count_note'] = 'inconclusive: reference set taken from an approximant (fib6), all cells share it'
if (H / 'penrose_size.json').exists():
    lq = 0.8107
    out['isotropic_deficit_fixed_V'] = dict(lam_QC_assumed=lq, rows=[dict(name=r['name'], V=r['V'], a=r['a'][0], b=r['b'][0],
        delta=math.hypot(r['a'][0], r['b'][0]), c0=r['harm']['c0'], deficit_over_delta=(lq - r['harm']['c0']) / math.hypot(r['a'][0], r['b'][0]),
        abs_h4_over_abs_a=r['abs_h4'] / abs(r['a'][0]), pc_lw_min=r['pc_lw_min_sampled']) for r in J('penrose_size.json')['rows']])
tot = 0.0
for f in H.glob('penrose_*.json'):
    try:
        dd = J(f.name)
    except Exception:
        continue
    for key in ('rows', 'dq_ladder', 'gpt_scan'):
        for r in dd.get(key, []) if isinstance(dd, dict) else []:
            tot += r.get('seconds', 0)
out['cpu_hours_mechanics'] = tot / 3600
(H / 'penrose_results.json').write_text(json.dumps(out, indent=1), encoding='utf-8')
print(json.dumps({k: out[k] for k in ['matched_pairs', 'joint_fit_h4', 'aline_local_alpha', 'aline_powerfit_abs_h4_vs_abs_a', 'lambda_QC_est', 'total_drop_fits']}, indent=1))

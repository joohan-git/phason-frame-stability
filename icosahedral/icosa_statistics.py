"""r4 assembly: python icosa_statistics.py [DIR ...] (default runs_objective and runs_objective_large). writes icosa_statistics.json. Single-code."""
import os, sys, json, glob
here = os.path.dirname(os.path.abspath(__file__))
import numpy as np
from scipy.stats import spearmanr, mannwhitneyu
dirs = sys.argv[1:] or [os.path.join(here, 'runs_objective'), os.path.join(here, 'runs_objective_large')]
R = {}
for d in dirs:
    for f in glob.glob(os.path.join(d, '*.json')):
        r = json.load(open(f)); b = os.path.basename(f)
        if b in R and abs(R[b]['p_c'] - r['p_c']) > 1e-8:
            raise SystemExit(f'duplicate disagrees: {b} {R[b]["p_c"]} {r["p_c"]}')
        R[b] = r
HI = {'3-2': 30000, '5-3': 100000, '8-5': 400000}
rng = np.random.default_rng(20261108)
nodes = lambda r: r['gamma_pr'][0] * r['V'] * (1 + 3 * (r['sub'] - 1))


def get(pq, flips=0, sub=2, mc=False):
    return [r for b, r in R.items() if r['tag'].split('_')[1] == pq and r['flips'] == flips and r['sub'] == sub and r['kind'] == 'cubic' and (mc or r.get('mcseed') is None)]


out = dict(single_code=True, n_files=len(R), dirs=[os.path.basename(d) for d in dirs])
allr = list(R.values())
out['O1'] = dict(element_rel_max=max(r['rigid']['element_max'] / r['rigid']['force_scale'] for r in allr), node_max=max(r['rigid']['node_norm'] for r in allr))
out['O2'] = max(max(list(r['antis'].values()) + [r['antis0'] or 0, r.get('antis_load') or 0]) for r in allr)
out['O2_loaded_measured'] = sum(1 for r in allr if r['antis'] or r.get('antis_load') is not None); out['O2_not_measured_under_load'] = [r['tag'] + '_s%d' % r['seed'] for r in allr if not r['antis'] and r.get('antis_load') is None]
ctl = dict(O1=bool(out['O1']['element_rel_max'] <= 1e-11 and out['O1']['node_max'] <= 1e-9), O2=bool(out['O2'] <= 1e-9))
canon = {pq: float(np.mean([r['p_c'] for r in get(pq)])) for pq in HI}
out['canonical_mean_sub2'] = canon
fam = {}
for pq, fl in HI.items():
    rs = sorted(get(pq, fl), key=lambda r: r['seed'])
    pc = np.array([r['p_c'] for r in rs]); can = {r['seed']: r['p_c'] for r in get(pq)}
    pen = pc / canon[pq] - 1
    boot = [rng.choice(pc, len(pc)).mean() for _ in range(10000)]
    nd = [nodes(r) for r in rs]
    fam[pq] = dict(V=rs[0]['V'], n=len(rs), seeds=[r['seed'] for r in rs], p_c=pc.tolist(), mean=float(pc.mean()), sem=float(pc.std(ddof=1) / len(pc) ** .5), sd=float(pc.std(ddof=1)), ci95=np.percentile(boot, [2.5, 97.5]).tolist(), min=float(pc.min()), max=float(pc.max()),
                   penalty_mean=float(pen.mean()), penalty_sem=float(pen.std(ddof=1) / len(pc) ** .5), penalty_seed_matched=float(np.mean([r['p_c'] / can[r['seed']] - 1 for r in rs if r['seed'] in can])), n_seed_matched=sum(r['seed'] in can for r in rs),
                   nodes_median=float(np.median(nd)), nodes_max=float(np.max(nd)), all_gamma_local=all(r['mech'] != 'long-wave' and r['lw_min'] is None for r in rs), not_local=[r['seed'] for r in rs if r['mech'] == 'long-wave' or r['lw_min'] is not None],
                   perp_var=float(np.mean([r['perp_var'] for r in rs])), moved=float(np.mean([r['moved_fraction'] for r in rs])), peak_GB=max([r.get('peak_GB') or 0 for r in rs]), t_mean=float(np.mean([r['t'] for r in rs])))
out['high_flip'] = fam
# Q1
pairs = [('3-2', '5-3'), ('5-3', '8-5')]
ps = [float(mannwhitneyu(fam[a]['p_c'], fam[b]['p_c'], alternative='greater').pvalue) for a, b in pairs]
order = np.argsort(ps); holm = [None, None]; run = 0
for i, j in enumerate(order):
    run = max(run, min(1, ps[j] * (2 - i))); holm[j] = float(run)
lnV = np.log([fam[k]['V'] for k in HI]); slopes = []
for _ in range(10000):
    y = [rng.choice(fam[k]['p_c'], fam[k]['n']).mean() for k in HI]; slopes.append(np.polyfit(lnV, y, 1)[0])
out['Q1'] = dict(p=ps, p_holm=holm, passed_pairs=[pairs[i] for i in range(2) if holm[i] < .05], slope_per_lnV=float(np.polyfit(lnV, [fam[k]['mean'] for k in HI], 1)[0]), slope_ci95=np.percentile(slopes, [2.5, 97.5]).tolist(), passed=all(h < .05 for h in holm) and fam['3-2']['mean'] > fam['5-3']['mean'] > fam['8-5']['mean'])
rngs = {'3-2': (-40, -29), '5-3': (-47, -37), '8-5': (-54, -44)}
out['Q2'] = {k: dict(value=fam[k]['penalty_mean'] * 100, range=rngs[k], passed=bool(rngs[k][0] <= fam[k]['penalty_mean'] * 100 <= rngs[k][1])) for k in HI}
out['Q3'] = {k: dict(all_gamma_local=fam[k]['all_gamma_local'], not_local=fam[k]['not_local'], nodes_median=fam[k]['nodes_median'], nodes_max=fam[k]['nodes_max'], passed=bool(fam[k]['all_gamma_local'] and fam[k]['nodes_median'] <= 15 and fam[k]['nodes_max'] <= 40)) for k in HI}
# Q4
PRED = dict(centred='excess_centered_max', variance='perp_var', moved='moved_fraction')
NP = 20000


def rows(pq):
    return sorted(get(pq, HI[pq]), key=lambda r: r['seed'])


def holm3(p):
    keys = sorted(p, key=p.get); o = {}; run = 0
    for i, k in enumerate(keys):
        run = max(run, min(1, p[k] * (len(keys) - i))); o[k] = float(run)
    return o


Q4 = {}
for pq in HI:
    rs = rows(pq); y = np.array([r['p_c'] for r in rs]); e = {}; pv = {}
    for nm, key in PRED.items():
        z = np.array([r[key] for r in rs]); rho = float(spearmanr(y, z)[0]); rk = np.argsort(np.argsort(y))
        cnt = sum(spearmanr(rk, rng.permutation(z))[0] <= rho for _ in range(NP)) if len(rs) >= 4 else NP
        e[nm] = dict(rho=rho, p=(cnt + 1) / (NP + 1)); pv[nm] = e[nm]['p']
    for k, v in holm3(pv).items():
        e[k]['p_holm'] = v
    Q4[pq] = dict(n=len(rs), **e)
strat = {}; pv = {}
for nm, key in PRED.items():
    def stat(perm=False):
        vals = []
        for pq in HI:
            rs = rows(pq); y = [r['p_c'] for r in rs]; z = np.array([r[key] for r in rs])
            vals.append(spearmanr(y, rng.permutation(z) if perm else z)[0])
        return float(np.mean(vals))
    s0 = stat(); cnt = sum(stat(True) <= s0 for _ in range(NP // 4))
    strat[nm] = dict(mean_rho=s0, p=(cnt + 1) / (NP // 4 + 1)); pv[nm] = strat[nm]['p']
for k, v in holm3(pv).items():
    strat[k]['p_holm'] = v
Q4['stratified'] = strat
Q4['Q4a_passed'] = bool(Q4['8-5']['centred']['rho'] <= -.5 and Q4['8-5']['centred']['p_holm'] < .05)
Q4['Q4b_passed'] = bool(strat['centred']['mean_rho'] <= -.3 and strat['centred']['p_holm'] < .05)
# rho difference, centred minus variance, 8/5
rs = rows('8-5'); dif = []
for _ in range(5000):
    i = rng.integers(0, len(rs), len(rs))
    y = [rs[j]['p_c'] for j in i]; a = [rs[j]['excess_centered_max'] for j in i]; b = [rs[j]['perp_var'] for j in i]
    dif.append(spearmanr(y, a)[0] - spearmanr(y, b)[0])
Q4['rho_diff_centred_minus_variance_8-5_ci95'] = np.nanpercentile(dif, [2.5, 97.5]).tolist()
out['Q4'] = Q4
# Q5
M = {}
for pq, seeds in (('5-3', (1, 2, 3)), ('8-5', (1,))):
    tab = []
    for s in seeds:
        g = lambda fl, sub: next((r for r in get(pq, fl, sub) if r['seed'] == s), None)
        c2, c4, h2, h4 = g(0, 2), g(0, 4), g(HI[pq], 2), g(HI[pq], 4)
        if not all((c2, c4, h2, h4)):
            continue
        tab.append(dict(seed=s, can2=c2['p_c'], can4=c4['p_c'], can_shift=c4['p_c'] / c2['p_c'] - 1, can_inf=c4['p_c'] + (c4['p_c'] - c2['p_c']) / 3, lw_lowest_sub4=bool(c4['lw_min'] is not None and c4['lw_min'] <= c4['p_c'] + 1e-12), gamma4=c4['gamma'][0], hi2=h2['p_c'], hi4=h4['p_c'], hi_shift=h4['p_c'] / h2['p_c'] - 1, hi_inf=h4['p_c'] + (h4['p_c'] - h2['p_c']) / 3,
                        pen2=h2['p_c'] / c2['p_c'] - 1, pen4=h4['p_c'] / c4['p_c'] - 1, nodes4=nodes(h4), peak_GB=[c4.get('peak_GB'), h4.get('peak_GB')], t=[c4['t'], h4['t']]))
    M[pq] = tab
allt = [x for v in M.values() for x in v]
out['Q5'] = dict(table=M, n=len(allt), Q5a=all(x['lw_lowest_sub4'] for x in allt), Q5b=all(-.007 <= x['can_shift'] <= -.001 for x in allt), Q5c=all(-.025 <= x['hi_shift'] <= -.002 for x in allt), Q5d=all(abs(x['pen4'] - x['pen2']) <= .02 for x in allt), complete=len(allt) == 4)
out['controls'] = ctl
json.dump(out, open(os.path.join(here, 'icosa_statistics.json'), 'w'), indent=1, default=float)
print('controls', ctl, out['O1'], out['O2'])
for k, v in fam.items():
    print(k, 'V', v['V'], 'n', v['n'], 'mean %.4f sem %.4f min %.4f penalty %.1f (matched %.1f, n%d) nodes med %.1f max %.1f local %s' % (v['mean'], v['sem'], v['min'], v['penalty_mean'] * 100, v['penalty_seed_matched'] * 100, v['n_seed_matched'], v['nodes_median'], v['nodes_max'], v['all_gamma_local']))
print('Q1', out['Q1']); print('Q2', out['Q2']); print('Q3', out['Q3']); print('Q4', json.dumps(Q4)); print('Q5', json.dumps(out['Q5']))

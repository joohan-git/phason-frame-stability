"""Second assembly stage: corrected slenderness table (assemble_defects.py mixed random tilings into the ordered 12/17 rows at r != 0.03; the
selector is corrected there too), B3 intervals and density correction, flip comparators, design scatter moduli, scorecard. Updates defects_results.json.
No registered rule is changed here."""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames', 'design', 'nonlinear', 'phason_paths')]  # shared modules
import json, math
from collections import defaultdict
from pathlib import Path
import numpy as np
from defects_util import boot_ci
HERE = Path(__file__).resolve().parent; M = HERE.parent
OUT = HERE / 'out'
R = json.loads((HERE / 'defects_results.json').read_text())
design = json.loads((M / 'design' / 'design_results.json').read_text())
NL = [json.loads(p.read_text()) for p in sorted(OUT.glob('plan_nonlinear_*.json'))]
# --- B3 per realization, with intervals; density-corrected under the cubic law with p in units of EI/l^3 (f from the task)
oref = [x for x in NL if x['task']['kind'] == 'bow' and x['task']['name'] == 'B1_12_17' and x['task']['sub'] == 4 and x['task']['w0'] == 1e-4]
rand = {n: [x for x in NL if x['task']['kind'] == 'bow' and x['task']['name'] == n and x['task']['sub'] == 4 and x['task']['w0'] == 1e-4] for n in ('R22', 'R101')}
meas = ('p_lin', 'pe_05', 'pe_10', 'p_05', 'p_cap', 'K0')
o = {m: float(np.mean([x[m] for x in oref])) for m in meas}
R['B3_table'] = dict(ordered=o, flips={n: {m: float(np.mean([x[m] for x in v])) for m in meas} for n, v in rand.items()}, levels={})
R['B3_table']['flips_loss'] = {m: float(np.mean([100 * (1 - R['B3_table']['flips'][n][m] / o[m]) for n in rand])) for m in meas}
R['B3_table']['flips_loss_each'] = {m: {n: 100 * (1 - R['B3_table']['flips'][n][m] / o[m]) for n in rand} for m in meas}
for lv, E_ in (('k1', 2786), ('0.01', 2786), ('0.05', 2786)):
    xs = [x for x in NL if x['task']['kind'] == 'nlmiss' and str(x['task']['level']) == lv]
    row = dict(n=len(xs), n_limit=sum(x.get('p_max') is not None for x in xs), k=xs[0]['k'])
    f = xs[0]['k'] / E_
    row['f'] = f
    for m in meas:
        v = [100 * (1 - x[m] / o[m]) for x in xs if x.get(m) is not None]
        row[m + '_raw'] = float(np.mean(v)); row[m + '_raw_range'] = [float(min(v)), float(max(v))]; row[m + '_raw_ci'] = boot_ci(v)
        if m != 'K0':
            d = [100 * (1 - x[m] / o[m] / (1 - f) ** 3) for x in xs if x.get(m) is not None]
            row[m + '_D'] = float(np.mean(d))
        else:
            row['K0_D'] = float(np.mean([100 * (1 - x[m] / o[m] / (1 - f)) for x in xs]))
    R['B3_table']['levels'][lv] = row
# --- design comparators
c = design['D']['cells']['B112/17']
K0 = c['K']; G0 = c['Gmean']
def lvl(kind, s):
    return [l for l in c['levels'] if l['kind'] == kind and l['sigma'] == s][0]
tab = {}
for nm, kind, s in (('modulus_10', 'E', 0.1), ('thickness_10', 't', 0.1)):
    l = lvl(kind, s)
    tab[nm] = dict(loss=l['drop_mean_pct'], ci=l['drop_ci_pct'], keys=[k for k in l.keys()][:30], K=l.get('K_mean'), G=l.get('G_mean'), total_max=l.get('total_max_mean'))
f = [x for x in c['flips'] if x['k'] == 'equilibrated'][0]
tab['flips_eq'] = dict(loss=f['drop_mean_pct'], ci=f['drop_ci_pct'], K=f['K_mean'], G=f['G_mean'], total_max=f['total_max_mean'])
f = [x for x in c['flips'] if x['k'] == 100][0]
tab['flips_100'] = dict(loss=f['drop_mean_pct'], ci=f['drop_ci_pct'], K=f['K_mean'], G=f['G_mean'], total_max=f['total_max_mean'])
for v in tab.values():
    if v.get('K'):
        v['K_loss'] = 100 * (1 - v['K'] / K0)
    if v.get('G'):
        v['G_loss'] = 100 * (1 - v['G'] / G0)
R['design_comparators'] = dict(tab, K0=K0, G0=G0, total0=c['total_max'], eq_29_41=[x for x in design['D']['cells']['B129/41']['flips'] if x['k'] == 'equilibrated'][0]['drop_mean_pct'],
                            eq_5_7=[x for x in design['D']['cells']['B15/7']['flips'] if x['k'] == 'equilibrated'][0]['drop_mean_pct'], eq_fib5=[x for x in design['D']['cells']['fib5']['flips'] if x['k'] == 'equilibrated'][0]['drop_mean_pct'])
# --- f where D reaches L_F/2 (registered sentence of outcome 2), linear interpolation, 12/17
A = R['A']['B112/17']['levels']; LF = R['decision_missing']['LF']
a, b = A['0.02'], A['0.05']
R['decision_missing']['f_at_D_half_LF'] = a['f'] + (LF / 2 - a['D']) / (b['D'] - a['D']) * (b['f'] - a['f'])
R['decision_missing']['rule_text'] = 'Narrow: not Survives because D(5 %) >= L_F/2; D(2 %) < L_F/2; no D(f <= 5 %) reaches 41.42'
# --- per-cell application of the same rule, descriptive
R['decision_by_cell'] = {}
for name in ('B15/7', 'B112/17', 'fib5'):
    L = R['A'][name]['levels']; e = R['ladder'][name]['eq']; lo = R['ladder'][name]['eq_ci'][0]
    if L['0.02']['D'] >= e / 2 or any(L[k]['D'] >= lo for k in ('k1', '0.005', '0.01', '0.02', '0.05')):
        o_ = 'Comparable'
    elif L['0.02']['D_ci'][1] < e / 3 and L['0.05']['D'] < e / 2:
        o_ = 'Survives'
    else:
        o_ = 'Narrow'
    R['decision_by_cell'][name] = dict(outcome=o_, eq=e, D2=L['0.02']['D'], D5=L['0.05']['D'])
s = R['size']['levels']; e = R['design_comparators']['eq_29_41']
R['size_vs_flips'] = dict(eq_29_41=e, D1_ratio=s['0.01']['D'] / e, D5_ratio=s['0.05']['D'] / e, D1_ratio_12_17=A['0.01']['D'] / LF, D5_ratio_12_17=A['0.05']['D'] / LF)
p = HERE / 'bow_diagnostic.json'
if p.exists():
    R['bow_diagnostic_exploratory'] = json.loads(p.read_text())
c8 = design['robust']['penalty_vs_r'] if 'penalty_vs_r' in design['robust'] else None
R['flips_vs_r_robust'] = c8
R['scorecard'] = dict(A1='fail', A2='partial', A3='fail', A4='fail', A5='fail', A6='pass', A7='fail', A8='partial', B1='pass', B2='pass', B3='partial', B4='fail', C1='fail')
R['gate_summary'] = dict(passed='G0a G0b G0c G1 (square, hexagonal, triangular at 12x12) G2 G3 G4 G5a G5b G5c G6', missed='G7 and G8 at sub 16, cause polygon discretization, see checks_nonlinear_extra.json')
(HERE / 'defects_results.json').write_text(json.dumps(R, indent=1, default=float), encoding='utf-8')
print(json.dumps(R['B3_table'], indent=0)[:300])
print(json.dumps(c8, default=float)[:600])
print(json.dumps({k: v for k, v in R['design_comparators'].items()}, default=float)[:1800])
print(R['decision_missing']); print(R['decision_by_cell']); print(R['size_vs_flips'])
for name, c_ in R['C'].items():
    for r, x in c_.items():
        print(name, r, round(x['lam0'], 4), {lv: (round(s_['raw'], 2), [round(q, 2) for q in s_['raw_ci']], 'D', round(s_['D'], 2), 'K', round(s_['Kl'], 1), s_['n']) for lv, s_ in x['levels'].items()})

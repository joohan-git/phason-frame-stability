"""Assemble defects_results.json from the task files. All numbers P except where the key says _C."""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames', 'design', 'nonlinear', 'phason_paths')]  # shared modules
import json, math, re
from collections import defaultdict
from pathlib import Path
import numpy as np
from defects_util import boot_ci
HERE = Path(__file__).resolve().parent; M = HERE.parent
OUT = HERE / 'out'
design = json.loads((M / 'design' / 'design_results.json').read_text())
FL = [json.loads(p.read_text()) for p in sorted(OUT.glob('plan_linear_*.json'))]
NL = [json.loads(p.read_text()) for p in sorted(OUT.glob('plan_nonlinear_*.json'))]
T_ = math.sqrt(3) * 0.03


def bkey(t):
    return (t['name'], t.get('fseed'), round(t.get('r', 0.03), 8), t.get('sub', 2))


BASE = {bkey(x['task']): x for x in FL if x['task']['kind'] == 'base' and x['task'].get('group') != 'thin'}
BASE_THIN = {bkey(x['task']): x for x in FL if x['task']['kind'] == 'base' and x['task'].get('group') == 'thin'}
for k, v in BASE_THIN.items():
    BASE.setdefault(k, v)


def base_for(t):
    k = bkey(t)
    if k in BASE:
        return BASE[k]
    return BASE[(t['name'], t.get('fseed'), round(t.get('r', 0.03), 8), 2)] if t.get('sub', 2) == 2 else None


def summarize(rows):
    """rows: miss task results with the same cell (fseed may vary), level, r, sub."""
    out = {}
    cl = defaultdict(list)
    for x in rows:
        b = base_for(x['task']); f = x['f_actual']
        rec = dict(raw=100 * (1 - x['lam_c'] / b['lam_c']), D=100 * (1 - x['lam_c'] / b['lam_c'] / (1 - f) ** 3),
                   Kl=100 * (1 - x['K'] / b['K']), Gl=100 * (1 - x['Gmean'] / b['Gmean']),
                   Kc=100 * (1 - x['K'] / b['K'] / (1 - f)), Gc=100 * (1 - x['Gmean'] / b['Gmean'] / (1 - f) ** 3), lam=x['lam_c'])
        if 'lam_direct' in x:
            rec['Ddir'] = 100 * (1 - x['lam_direct'] / b['lam_c'] / (1 - f) ** 3)
        cl[x['task'].get('fseed')].append(rec)
    keys = ['raw', 'D', 'Ddir', 'Kl', 'Gl', 'Kc', 'Gc', 'lam']
    clust = list(cl.keys()) != [None]
    for k in keys:
        v = [[r[k] for r in cl[c] if k in r] for c in cl]
        v = [a for a in v if a]
        if not v:
            continue
        flat = np.array([a for s in v for a in s]) if not clust else np.array([np.mean(s) for s in v])
        out[k] = float(flat.mean()); out[k + '_ci'] = boot_ci(flat)
        if k in ('raw', 'D'):
            pool = np.array([a for s in v for a in s]); out[k + '_sd'] = float(pool.std(ddof=1)) if len(pool) > 1 else 0.0; out[k + '_range'] = [float(pool.min()), float(pool.max())]
    out.update(n=len(rows), n_clusters=len(cl), k=int(np.mean([x['k'] for x in rows])), f=float(np.mean([x['f_actual'] for x in rows])),
               modes={m: sum(x['mode'] == m for x in rows) for m in set(x['mode'] for x in rows)}, part_med=float(np.median([x['part'] for x in rows])), part_max=float(np.max([x['part'] for x in rows])),
               near_w=float(np.mean([x['near_w'] for x in rows])), near_null=float(np.mean([x['near_null'] for x in rows])), peak_dist_med=float(np.median([x['peak_dist'] for x in rows])),
               frac_peak_within2=float(np.mean([x['peak_dist'] <= 2 for x in rows])), spread_med=float(np.median([x['spread'] for x in rows])),
               ndeg2_new=float(np.mean([x['n_deg2_new'] for x in rows])), ndeg1=float(np.mean([x['n_deg1'] for x in rows])), nskip=float(np.mean([x['nskip'] for x in rows])),
               spans_all=bool(all(x['spans'] for x in rows)), ax_max=float(np.mean([x['ax_max'] for x in rows])), total_max=float(np.mean([x['total_max'] for x in rows])))
    return out


miss = [x for x in FL if x['task']['kind'] == 'miss']
R = dict(label_key='P = numerical estimate or bootstrap 95 % interval; C = exact algebra', units='lam in pi^2 EI/l^3; losses in percent', A={}, Arand={}, size={}, mesh={}, C={}, reference={})


def lvname(l):
    return str(l)


def group(sel):
    g = defaultdict(list)
    for x in sel:
        g[lvname(x['task']['level'])].append(x)
    return {k: summarize(v) for k, v in g.items()}


for name in ('B112/17', 'B15/7', 'fib5', 'square38', 'hex26', 'tri18'):
    sel = [x for x in miss if x['task']['group'] == 'A' and x['task']['name'] == name]
    b = BASE[(name, None, 0.03, 2)]
    R['A'][name] = dict(V=b['V'], E=b['E'], lam0=b['lam_c'], mode0=b['mode'], K0=b['K'], G0=b['Gmean'], ax0=b['ax_max'], total0=b['total_max'], ledge0=b['ledge'], levels=group(sel))
sel = [x for x in miss if x['task']['group'] == 'Arand']
R['Arand'] = dict(lam0_mean=float(np.mean([BASE[('B112/17', s, 0.03, 2)]['lam_c'] for s in range(100, 108)])), K0_mean=float(np.mean([BASE[('B112/17', s, 0.03, 2)]['K'] for s in range(100, 108)])), levels=group(sel))
o0 = BASE[('B112/17', None, 0.03, 2)]['lam_c']
for lv, s in R['Arand']['levels'].items():
    s['lam_over_ordered'] = s['lam'] / o0; s['loss_vs_ordered'] = 100 * (1 - s['lam'] / o0); s['lam_ci_over_ordered'] = [c / o0 for c in s['lam_ci']]
sel = [x for x in miss if x['task']['group'] == 'size']
if sel:
    b = BASE[('B129/41', None, 0.03, 2)]
    R['size'] = dict(V=b['V'], lam0=b['lam_c'], levels=group(sel))
sel = [x for x in miss if x['task']['group'] == 'mesh']
if sel:
    b = BASE[('B112/17', None, 0.03, 4)]
    R['mesh'] = dict(lam0=b['lam_c'], levels=group(sel))
# --- C: r = 0.03 from the same first realizations of A
for name, n in (('B112/17', 12), ('fib5', 6), ('hex26', 6)):
    R['C'][name] = {}
    for r in (0.015, 0.03, 0.06, 0.10):
        if r == 0.03:
            sel = [x for x in miss if x['task']['group'] == 'A' and x['task']['name'] == name and x['task']['i'] < n and x['task']['level'] in ('k1', 0.01, 0.05)]
        else:
            sel = [x for x in miss if x['task']['group'] == 'C' and x['task']['name'] == name and not x['task'].get('fseed') and abs(x['task']['r'] - r) < 1e-9]
        R['C'][name][str(r)] = dict(lam0=BASE[(name, None, round(r, 8), 2)]['lam_c'], levels=group(sel))
R['C']['random12/17'] = {}
for r in (0.015, 0.03, 0.06, 0.10):
    if r == 0.03:
        sel = [x for x in miss if x['task']['group'] == 'Arand' and x['task']['fseed'] < 104 and x['task']['i'] == 0 and x['task']['level'] in ('k1', 0.01, 0.05)]
    else:
        sel = [x for x in miss if x['task']['group'] == 'C' and x['task'].get('fseed') and abs(x['task']['r'] - r) < 1e-9]
    R['C']['random12/17'][str(r)] = dict(lam0=float(np.mean([BASE[('B112/17', s, round(r, 8), 2)]['lam_c'] for s in range(100, 104)])), levels=group(sel))
# --- uniform thinning
R['thin'] = {}
for name in ('B112/17', 'hex26'):
    b = BASE[(name, None, 0.03, 2)]['lam_c']; R['thin'][name] = {}
    for f_ in (0.005, 0.01, 0.02, 0.05, 0.10):
        l = BASE[(name, None, round(0.03 * (1 - f_), 8), 2)]['lam_c']; l2 = BASE[(name, None, round(0.03 / (1 - f_), 8), 2)]['lam_c']
        R['thin'][name][str(f_)] = dict(lam_at_thin=l, strength_loss_pct=100 * (1 - l / b * (1 - f_) ** 3), cubic_only_C=100 * (1 - (1 - f_) ** 3), lam_at_thick_ratio=l2 / b)
# --- single-strut census 5/7
rows = [r for x in FL if x['task']['kind'] == 'census' for r in x['rows']]
if rows:
    l0 = BASE[('B15/7', None, 0.03, 2)]['lam_c']; K0 = BASE[('B15/7', None, 0.03, 2)]['K']
    ok = [r for r in rows if not r.get('inadmissible')]
    by = defaultdict(list)
    for r in ok:
        by[tuple(sorted(r['ends']))].append(100 * (1 - r['lam_c'] / l0))
    L = np.array([100 * (1 - r['lam_c'] / l0) for r in ok])
    R['census57'] = dict(n=len(ok), inadmissible=len(rows) - len(ok), mean=float(L.mean()), min=float(L.min()), max=float(L.max()), q10=float(np.percentile(L, 10)), q50=float(np.median(L)), q90=float(np.percentile(L, 90)),
                         frac_negative=float((L < 0).mean()), K_loss_mean=float(np.mean([100 * (1 - r['K'] / K0) for r in ok])),
                         by_end_degrees={str(k): dict(n=len(v), mean=float(np.mean(v)), min=float(np.min(v)), max=float(np.max(v))) for k, v in sorted(by.items())}, part_median=float(np.median([r['part'] for r in ok])))
# single-strut degree dependence on 12/17
s = [x for x in miss if x['task']['group'] == 'A' and x['task']['name'] == 'B112/17' and x['task']['level'] == 'k1']
by = defaultdict(list)
for x in s:
    by[tuple(sorted(x['rem_ends'][0]))].append(100 * (1 - x['lam_c'] / o0))
R['k1_degrees_1217'] = {str(k): dict(n=len(v), mean=float(np.mean(v)), min=float(np.min(v)), max=float(np.max(v))) for k, v in sorted(by.items())}
# --- flip ladder
R['ladder'] = {}
for name in ('B112/17', 'B15/7', 'fib5'):
    fl = design['D']['cells'][name]['flips']; V = design['D']['cells'][name]['V']
    ks = [(f['k'], f['drop_mean_pct']) for f in fl if f['k'] != 'equilibrated']; eq = [f['drop_mean_pct'] for f in fl if f['k'] == 'equilibrated'][0]
    eqci = [f['drop_ci_pct'] for f in fl if f['k'] == 'equilibrated'][0]
    keq = [f for f in fl if f['k'] == 'equilibrated'][0]
    row = dict(flips={str(a): b for a, b in ks}, eq=eq, eq_ci=eqci, K_eq=keq.get('K_mean'), levels={})
    mono = sorted(ks)
    for lv, s_ in R['A'][name]['levels'].items():
        for meas in ('raw', 'D'):
            v = s_[meas]
            if v >= eq:
                eqv = 'at or above equilibrated'
            elif v >= mono[-1][1]:
                eqv = 'between 1000 flips and equilibrated'
            elif v <= mono[0][1]:
                eqv = 'below 10 flips'
            else:
                for (k1, a), (k2, b) in zip(mono, mono[1:]):
                    if a <= v <= b and b > a:
                        eqv = round(math.exp(math.log(k1) + (v - a) / (b - a) * (math.log(k2) - math.log(k1))))
                        break
                else:
                    eqv = None
            row['levels'].setdefault(lv, {})[meas + '_flip_equiv'] = eqv
        row['levels'][lv]['ratio_raw_to_eq'] = s_['raw'] / eq; row['levels'][lv]['ratio_D_to_eq'] = s_['D'] / eq
    fs = sorted(((s_['f'], s_['raw'], s_['D']) for s_ in R['A'][name]['levels'].values()))
    for j, nm in ((1, 'f_star_raw'), (2, 'f_star_D')):
        fstar = None
        for a, b in zip(fs, fs[1:]):
            if a[j] < eq <= b[j]:
                fstar = a[0] + (eq - a[j]) / (b[j] - a[j]) * (b[0] - a[0]); break
        row[nm] = fstar
    R['ladder'][name] = row
# --- stored design comparators
c = design['D']['cells']['B112/17']
R['reference']['design_12_17'] = dict(lam0=c['lam0'], K0=c['K'], G0=c['Gmean'], E10=[l for l in c['levels'] if l['kind'] == 'E' and l['sigma'] == 0.1][0]['drop_mean_pct'], E10_ci=[l for l in c['levels'] if l['kind'] == 'E' and l['sigma'] == 0.1][0]['drop_ci_pct'],
                                   design=[l for l in c['levels'] if l['kind'] == 't' and l['sigma'] == 0.1][0]['drop_mean_pct'], design_ci=[l for l in c['levels'] if l['kind'] == 't' and l['sigma'] == 0.1][0]['drop_ci_pct'],
                                   eq=[f for f in c['flips'] if f['k'] == 'equilibrated'][0]['drop_mean_pct'], eq_ci=[f for f in c['flips'] if f['k'] == 'equilibrated'][0]['drop_ci_pct'])
# --- decision, missing struts
A = R['A']['B112/17']['levels']; LF = R['reference']['design_12_17']['eq']; lfl = R['reference']['design_12_17']['eq_ci'][0]
d2 = A['0.02']['D']; d2u = A['0.02']['D_ci'][1]; d5 = A['0.05']['D']
if d2 >= LF / 2 or any(A[str(k)]['D'] >= lfl for k in ('k1', 0.005, 0.01, 0.02, 0.05)):
    dec = 'Comparable'
elif d2u < LF / 3 and d5 < LF / 2:
    dec = 'Survives'
else:
    dec = 'Narrow'
R['decision_missing'] = dict(outcome=dec, LF=LF, LF_ci_low=lfl, D2=d2, D2_upper=d2u, D5=d5, single=A['k1']['raw'], single_nonperturbative=A['k1']['raw'] >= LF / 2)
# --- nonlinear
bow = [x for x in NL if x['task']['kind'] == 'bow']
g = defaultdict(dict)
for x in bow:
    t = x['task']; g[(t['name'], t['sub'], t['shape'], t['w0'])][t['i']] = x
R['B'] = {}
MEAS = ['pe_05', 'p_05', 'pe_10', 'p_lin', 'K0', 'G0', 'Ksec25', 'Ksec50', 'p_cap']
for (name, sub, shape, w0), d in sorted(g.items(), key=lambda a: (a[0][0], a[0][1], a[0][2], a[0][3])):
    ref = g.get((name, sub, shape, 1e-4), {})
    row = dict(n=len(d), w0=w0, e_C=w0 / T_, chord_bow_rms=float(np.mean([x['chord_bow_rms'] for x in d.values()])), n_limit=sum(x.get('p_max') is not None for x in d.values()), n_first_neg=sum(x.get('p_first_neg') is not None for x in d.values()),
               errors=sum('error' in x for x in d.values()), arc_excess=float(np.mean([x['bow_arc_excess'] for x in d.values()])), sec=float(np.mean([x['sec'] for x in d.values()])), ends=sorted(set(str(x.get('end')) for x in d.values())), psi_end=float(np.mean([x.get('psi_max_end') or 0 for x in d.values()])))
    lim = [x['p_max'] for x in d.values() if x.get('p_max') is not None]; row['p_limit_mean'] = float(np.mean(lim)) if lim else None; row['psi_at_limit'] = float(np.mean([x['psi_at_pmax'] for x in d.values() if x.get('p_max') is not None])) if lim else None
    fn = [x['p_first_neg'] for x in d.values() if x.get('p_first_neg') is not None]; row['p_first_neg_mean'] = float(np.mean(fn)) if fn else None
    for m in MEAS:
        v = {i: x[m] for i, x in d.items() if x.get(m) is not None}
        if not v:
            continue
        row[m] = float(np.mean(list(v.values()))); row[m + '_ci'] = boot_ci(list(v.values()))
        pairs = [100 * (1 - v[i] / ref[i][m]) for i in v if i in ref and ref[i].get(m) is not None]
        if pairs:
            row[m + '_loss'] = float(np.mean(pairs)); row[m + '_loss_ci'] = boot_ci(pairs)
    if w0 > 1e-4 and 'K0' in row and ref:
        cs = [(ref[i]['K0'] / d[i]['K0'] - 1) / (w0 / T_) ** 2 for i in d if i in ref]; row['c_fit'] = float(np.mean(cs)); row['c_fit_ci'] = boot_ci(cs)
    R['B'][f'{name}|sub{sub}|{shape}|{w0}'] = row
# flips in the same nonlinear measure
R['B_flips'] = {}
for sub in (2, 4):
    oref = g.get(('B1_12_17', sub, 'half', 1e-4))
    if not oref:
        continue
    for name in ('R22', 'R101'):
        rr = g.get((name, sub, 'half', 1e-4))
        if not rr:
            continue
        row = {}
        for m in ('pe_05', 'p_05', 'pe_10', 'p_lin', 'K0', 'Ksec50'):
            a = np.mean([x[m] for x in oref.values() if x.get(m) is not None]); b = np.mean([x[m] for x in rr.values() if x.get(m) is not None])
            row[m + '_ordered'] = float(a); row[m + '_random'] = float(b); row[m + '_loss'] = float(100 * (1 - b / a))
        lim = [x['p_max'] for x in rr.values() if x.get('p_max') is not None]
        row['random_first_limit'] = float(np.mean(lim)) if lim else None
        R['B_flips'][f'{name}|sub{sub}'] = row
# nonlinear missing
nm = [x for x in NL if x['task']['kind'] == 'nlmiss']
R['B3'] = {}
if nm and (('B1_12_17', 4, 'half', 1e-4) in g):
    oref = g[('B1_12_17', 4, 'half', 1e-4)]
    o = {m: float(np.mean([x[m] for x in oref.values() if x.get(m) is not None])) for m in ('pe_05', 'p_05', 'pe_10', 'p_lin', 'K0', 'p_cap')}
    by = defaultdict(list)
    for x in nm:
        by[str(x['task']['level'])].append(x)
    for lv, xs in by.items():
        row = dict(n=len(xs), k=xs[0].get('k'), ref_ordered=o, n_limit=sum(x.get('p_max') is not None for x in xs), n_first_neg=sum(x.get('p_first_neg') is not None for x in xs), ends=sorted(set(str(x.get('end')) for x in xs)))
        for m in ('pe_05', 'p_05', 'pe_10', 'p_lin', 'K0', 'p_cap'):
            v = [x[m] for x in xs if x.get(m) is not None]
            if v:
                row[m] = float(np.mean(v)); row[m + '_loss'] = float(100 * (1 - np.mean(v) / o[m])); row[m + '_n'] = len(v)
        lim = [x['p_max'] for x in xs if x.get('p_max') is not None]; row['p_limit_mean'] = float(np.mean(lim)) if lim else None
        row['psi_end'] = float(np.mean([x.get('psi_max_end') or 0 for x in xs]))
        R['B3'][lv] = row
# kconv
kc = [x for x in NL if x['task']['kind'] == 'kconv']
R['kconv'] = {}
for x in kc:
    t = x['task']; k = f"{t['name']}|sub{t['sub']}"
    for w0 in ('0.001', '0.003', '0.01', '0.03'):
        R['kconv'].setdefault(k, {}).setdefault(w0, []).append(dict(ratio=x[w0]['K'] / x['K_straight'], G=x[w0]['G'] / x['G_straight'], c=(x['K_straight'] / x[w0]['K'] - 1) / (float(w0) / T_) ** 2))
for k, v in R['kconv'].items():
    for w0, rows in v.items():
        v[w0] = dict(K_ratio=float(np.mean([a['ratio'] for a in rows])), G_ratio=float(np.mean([a['G'] for a in rows])), c=float(np.mean([a['c'] for a in rows])), n=len(rows))
# --- decision, bow
if R['B_flips'].get('R22|sub4') and 'B1_12_17|sub4|half|0.01' in R['B']:
    fl = np.mean([R['B_flips'][k]['pe_05_loss'] for k in R['B_flips'] if 'sub4' in k])
    l1 = R['B']['B1_12_17|sub4|half|0.01'].get('pe_05_loss'); l3 = R['B']['B1_12_17|sub4|half|0.03'].get('pe_05_loss')
    out = 'Fails' if l1 >= fl / 3 else ('Survives' if l3 < fl / 2 else 'Conditional')
    R['decision_bow'] = dict(outcome=out, flip_loss_pe05=float(fl), loss_1pct=l1, loss_3pct=l3)
R['timing'] = dict(lin_task_hours=sum(x['sec'] for x in FL) / 3600, nl_task_hours=sum(x['sec'] for x in NL) / 3600, n_lin=len(FL), n_nl=len(NL), max_sec_lin=max(x['sec'] for x in FL), max_sec_nl=max([x['sec'] for x in NL] or [0]))
for gf in ('checks_linear', 'checks_linear_extra', 'checks_nonlinear', 'checks_nonlinear_extra'):
    p = HERE / f'{gf}.json'
    R.setdefault('gates', {})[gf] = json.loads(p.read_text()) if p.exists() else None
(HERE / 'defects_results.json').write_text(json.dumps(R, indent=1, default=float), encoding='utf-8')
print('written', len(FL), len(NL), R.get('decision_missing'), R.get('decision_bow'))

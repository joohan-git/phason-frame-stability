"""intervention registered analysis (registered plan sections 4, 5, 6, 7), written before any outcome. python analyse_intervention.py OUTDIR. Writes intervention_results.json. All numbers P."""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames', 'penrose', 'heldout_test', 'design', 'nonlinear', 'defects', 'phason_paths', 'independent_code', 'forbidden_patches')]  # shared modules
import glob, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr, binomtest
HERE = Path(__file__).resolve().parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / 'out'
NB = 10000
LQC = dict(ab=0.79984, pen=0.8106)


def boot_mean(x, seed=16):
    x = np.asarray(x, float); r = np.random.default_rng(seed)
    v = x[r.integers(0, len(x), (NB, len(x)))].mean(1)
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def boot_diff(a, b, seed=16):
    a = np.asarray(a, float); b = np.asarray(b, float); r = np.random.default_rng(seed)
    v = a[r.integers(0, len(a), (NB, len(a)))].mean(1) - b[r.integers(0, len(b), (NB, len(b)))].mean(1)
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def summ(rows, keys=('lam_c', 'phi2', 'var', 'moved', 'x', 'K', 'Gmean', 'total_max', 'ax_max', 'part', 'mode_on_forbidden', 'c0', 'accepted', 'mc_sec', 'mech_sec', 'sec')):
    out = dict(n=len(rows))
    for k in keys:
        v = [r[k] for r in rows if r.get(k) is not None]
        if v:
            out[k] = dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, ci=boot_mean(v), min=float(np.min(v)), max=float(np.max(v)))
    return out


def compare(lo, hi, lam0, coord='var'):
    """Registered M1 (on the matched coordinate), M2, effect and class for lo (penalized) against hi (comparator). The variance gap is always reported."""
    vl, vh = np.mean([r[coord] for r in lo]), np.mean([r[coord] for r in hi])
    gap = float(abs(np.mean([r['var'] for r in lo]) - np.mean([r['var'] for r in hi])) / np.mean([r['var'] for r in hi]))
    pl, ph = np.mean([r['phi2'] for r in lo]), np.mean([r['phi2'] for r in hi])
    m1 = abs(vl - vh) / vh; m2 = ph - pl
    a = np.array([r['lam_c'] for r in lo]); b = np.array([r['lam_c'] for r in hi])
    d = float((a.mean() - b.mean()) / lam0); ci = [c / lam0 for c in boot_diff(a, b)]
    match = bool(m1 <= 0.03 and m2 >= 0.20)
    if ci[0] > 0 and d >= 0.05:
        cl = 'Raised'
    elif ci[0] > 0:
        cl = 'Small positive'
    elif ci[1] < 0:
        cl = 'Lowered'
    elif ci[0] > -0.05 and ci[1] < 0.05:
        cl = 'No effect'
    else:
        cl = 'Inconclusive'
    return dict(coord=coord, var_gap=gap, x=[float(np.mean([r['x'] for r in lo])), float(np.mean([r['x'] for r in hi]))], M1=float(m1), M2=float(m2), matched=match, var=[float(vl), float(vh)], phi2=[float(pl), float(ph)], lam=[float(a.mean()), float(b.mean())], lam_ratio=[float(a.mean() / lam0), float(b.mean() / lam0)], delta=d, ci=ci, cls=cl if match else 'not matched', cls_if_matched=cl)


def main():
    R = {}
    for f in glob.glob(str(OUT / 'plan_tilings_*.json')) + glob.glob(str(OUT / 'plan_paths_*.json')):
        d = json.load(open(f)); d.pop('lifts', None); R[f] = d
    A = defaultdict(lambda: defaultdict(list)); refs = {}; B = defaultdict(list); NL = defaultdict(list); XC = []
    for d in R.values():
        t = d['task']
        if t['kind'] == 'til':
            A[t['cname']][t['ens']].append(d)
        elif t['kind'] == 'ref':
            refs[t['cname']] = d
        elif t['kind'] == 'cell':
            B[(t['cls'], str(t['L1']), str(t['L2']))].append(d)
        elif t['kind'] == 'nl':
            NL[(t['cname'], t['ens'])].append(d)
        elif t['kind'] == 'xc':
            XC.append(d)
    res = dict(label='P', refs={k: {x: v.get(x) for x in ('V', 'lam_c', 'K', 'total_max', 'part', 'phi2', 'var', 'c0', 'sec')} for k, v in refs.items()}, arm_A={}, arm_B={}, nonlinear={}, second_code={})
    # ---------- arm A
    for c, E in A.items():
        lam0 = refs[c]['lam_c']; X = dict(ensembles={e: summ(r) for e, r in E.items()})
        X['primary'] = compare(E['L'], E['U'], lam0)
        X['H_vs_U'] = compare(E['H'], E['U'], lam0) if 'H' in E else None
        X['H_leverage'] = float(np.mean([r['phi2'] for r in E['H']]) - np.mean([r['phi2'] for r in E['U']])) if 'H' in E else None
        X['monotone_HUL'] = bool(np.mean([r['lam_c'] for r in E['H']]) <= np.mean([r['lam_c'] for r in E['U']]) <= np.mean([r['lam_c'] for r in E['L']])) if 'H' in E else None
        if 'UR' in E:
            X['sens_L_vs_UR'] = compare(E['L'], E['UR'], lam0)
        for tag, lo, hi, co in (('low', 'lowL', 'lowUR', 'var'), ('flips', 'flipL', 'flipU', 'accepted'), ('moved', 'movL', 'movU', 'moved')):
            if lo in E and hi in E:
                X[tag] = compare(E[lo], E[hi], lam0, co)
        L = E['L']
        lc = [r['lam_c'] for r in L]
        X['within_L'] = {k: dict(rho=float(spearmanr(lc, [r[k] for r in L]).correlation), p=float(spearmanr(lc, [r[k] for r in L]).pvalue)) for k in ('phi2', 'Nf', 'x', 'var')}
        for nm in ('L', 'U'):
            enr = [r['mode_on_forbidden'] / r['phi2'] for r in E[nm] if r['phi2'] > 0]
            X[f'mechanism_{nm}'] = dict(enrichment=float(np.mean(enr)), ci=boot_mean(enr), peak_on_forbidden=float(np.mean([r['peak_on_forbidden'] for r in E[nm]])), phi2=float(np.mean([r['phi2'] for r in E[nm]])), top1pct=float(np.mean([r['top1pct_on_forbidden'] for r in E[nm]])))
        pool = [r for e in E.values() for r in e if r.get('lam_c') is not None]
        y = np.array([r['lam_c'] for r in pool]); Z = np.array([[r['phi2'], r['var'], r['x']] for r in pool], float)
        Zs = (Z - Z.mean(0)) / Z.std(0); ys = (y - y.mean()) / y.std()
        beta = np.linalg.lstsq(np.c_[np.ones(len(ys)), Zs], ys, rcond=None)[0][1:]
        rng = np.random.default_rng(16); bs = []
        for _ in range(2000):
            i = rng.integers(0, len(ys), len(ys))
            bs.append(np.linalg.lstsq(np.c_[np.ones(len(i)), Zs[i]], ys[i], rcond=None)[0][1:])
        bs = np.array(bs)
        X['pooled_ols'] = dict(names=['phi2', 'var', 'x'], beta=beta.tolist(), lo=np.percentile(bs, 2.5, 0).tolist(), hi=np.percentile(bs, 97.5, 0).tolist(), n=len(ys))
        res['arm_A'][c] = X
    # ---------- arm B
    pairs = json.loads((HERE / 'pairs.json').read_text())

    def cellv(cls, c):
        rows = B[(cls, str(c['L1']), str(c['L2']))]
        return np.array([r['c0'] for r in rows]), np.array([r['phi2'] for r in rows]), rows
    for key, plist in pairs.items():
        cls, st = key.split('_'); D = []; rows = []
        for p in plist:
            a, pa, ra = cellv(cls, p['lo']); b, pb, rb = cellv(cls, p['hi'])
            if len(a) < 3 or len(b) < 3:
                continue
            d = float(a.mean() - b.mean()); D.append(d)
            rows.append(dict(bin=p['bin'], D=d, dn_cov=p['dn'], dn_out=float(pb.mean() - pa.mean()), sd_lo=float(a.std(ddof=1)), sd_hi=float(b.std(ddof=1)), V=[ra[0]['V'], rb[0]['V']], delta=[p['lo']['delta'], p['hi']['delta']], c0=[float(a.mean()), float(b.mean())], lam=[float(np.mean([r['lam_c'] for r in ra])), float(np.mean([r['lam_c'] for r in rb]))]))
        if not D:
            continue
        D = np.array(D); ci = boot_mean(D); sd = float(np.median([x for r in rows for x in (r['sd_lo'], r['sd_hi'])]))
        v = 'B-raised' if ci[0] > 0 else ('B-lowered' if ci[1] < 0 else 'B-null')
        res['arm_B'][key] = dict(n=len(D), mean_D=float(D.mean()), ci=ci, pct_ordered=float(100 * D.mean() / LQC[cls]), ci_pct=[100 * c / LQC[cls] for c in ci], median_offset_sd=sd, ratio_to_offset=float(abs(D.mean()) / sd) if sd > 0 else None,
                                 n_positive=int((D > 0).sum()), sign_p=float(binomtest(int((D > 0).sum()), len(D)).pvalue), slope_per_dn=float(np.mean([r['D'] / r['dn_out'] for r in rows if r['dn_out'] > 0])), mean_dn_out=float(np.mean([r['dn_out'] for r in rows])), verdict=v if len(D) >= 8 else 'no verdict, fewer than 8 pairs', order_1pct=bool(abs(D.mean()) / LQC[cls] < 0.02), rows=rows)
    # ---------- nonlinear
    for (c, e), rows in NL.items():
        keep = {k: [r.get(k) for r in rows] for k in ('p_max', 'p_05', 'p_first_neg', 'p_lin', 'part_lin', 'end', 'nsteps', 'p_end', 'sec')}
        res['nonlinear'][f'{c}_{e}'] = keep
    for c in ('ab1217', 'fib5'):
        a = [r['p_max'] for r in NL.get((c, 'L'), []) if r.get('p_max') is not None]; b = [r['p_max'] for r in NL.get((c, 'U'), []) if r.get('p_max') is not None]
        if a and b:
            res['nonlinear'][f'{c}_fold_L_minus_U'] = dict(n=[len(a), len(b)], n_total=[len(NL[(c, 'L')]), len(NL[(c, 'U')])], mean=[float(np.mean(a)), float(np.mean(b))], ci=boot_diff(a, b), rises=bool(boot_diff(a, b)[0] > 0))
    # ---------- second code
    idx = {}
    for d in R.values():
        t = d['task']
        if t['kind'] == 'til':
            idx[(t['cname'], t['ens'], t['i'])] = d
        elif t['kind'] == 'cell' and t['off'] == 0:
            idx[(t['cls'], str(t['L1']), str(t['L2']))] = d
    rows = []
    for d in XC:
        t = d['task']
        m = idx.get((t['cname'], t['ens'], t['i'])) if 'L1' not in t else idx.get((t['cls'], str(t['L1']), str(t['L2'])))
        if m is None:
            continue
        rows.append(dict(c=t['cname'], e=t['ens'], i=t['i'], xc=d['lam_min'], main=m['lam_c'], rel=abs(d['lam_min'] - m['lam_c']) / m['lam_c']))
    res['second_code'] = dict(rows=rows, max_rel=float(max(r['rel'] for r in rows)) if rows else None, passed=bool(rows and max(r['rel'] for r in rows) <= 0.005))
    # the same tilings rerun with a lower shift and more eigenvalues (plan_second_code)
    XC2 = [json.load(open(f)) for f in sorted(glob.glob(str(OUT / 'plan_second_code_*.json')))]
    if XC2:
        rel2 = [abs(d['lam_min'] - idx[(d['task']['cname'], d['task']['ens'], d['task']['i'])]['lam_c']) / idx[(d['task']['cname'], d['task']['ens'], d['task']['i'])]['lam_c'] for d in XC2]
        res['second_code']['diagnostic_rerun'] = dict(lam_guess=XC2[0]['task']['lam_guess'], nev=XC2[0]['task']['nev'], max_rel=float(max(rel2)), n=len(XC2))
    # ---------- verdict
    cl = {c: X['primary']['cls'] for c, X in res['arm_A'].items()}
    m = [c for c, v in cl.items() if v != 'not matched']
    if not m:
        v = 'A-V'
    elif any(cl[c] == 'Lowered' for c in m):
        v = 'A-IV'
    elif len(m) == 3 and all(cl[c] == 'Raised' for c in m):
        v = 'A-I'
    elif any(cl[c] == 'Raised' for c in m):
        v = 'A-II'
    else:
        v = 'A-III'
    res['verdict_A'] = dict(verdict=v, classes=cl)
    res['verdict_B'] = {k: res['arm_B'][k]['verdict'] for k in res['arm_B'] if k.endswith('_S')}
    res['tasks_done'] = len(R)
    (HERE / 'intervention_results.json').write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(dict(verdict_A=res['verdict_A'], verdict_B=res['verdict_B'], tasks=len(R)), indent=1))


if __name__ == '__main__':
    main()

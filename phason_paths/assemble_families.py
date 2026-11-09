"""R3 analysis -> families_results.json. Every number P unless marked C (exact symmetry statement)."""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames', 'slope_survey')]  # shared modules
import json, math, itertools
from pathlib import Path
import numpy as np
H = Path(__file__).parent
fam = json.loads((H / 'ab_families.json').read_text())['rows']
out = dict(note='All numerical values P (finite cells, linear axial-force tangent, sampled directions). '
                'independent values copied from section 5 (4 elements, independent offset/generator).', items={})

# ---- families
independent = {  # (V, amp, pressure) 4 elements
    'B1': [(239, 0.00505063, 0.79408323), (1393, 0.000866552, 0.79643229), (8119, 0.000148677, 0.79668802)],
    'B2': [(816, 0.00209204, 0.79532677), (4756, 0.000358937, 0.79653674), (27720, 0.0000615839, 0.79670242)],
    'E2a': [(338, 0.00505063, 0.79498839), (1970, 0.000866552, 0.79658519), (11482, 0.000148677, 0.79670653)],
    'E2diag': [(1154, 0.00209204, 0.79625095), (6726, 0.000358937, 0.79666490), (39202, 0.0000615839, 0.79672696)]}
F = {}
for r in fam:
    F.setdefault(r['family'], []).append(r)
for k in F:
    F[k].sort(key=lambda r: -r['amp'])


def pw(x, y):
    a, c = np.polyfit(np.log(x), np.log(y), 1)
    return float(a), float(math.exp(c))


fits = {}
for k, rows in F.items():
    amp = np.array([r['amp'] for r in rows]); pc = np.array([r['pc_min_sampled'] for r in rows])
    h4 = np.array([r['h4_abs'] for r in rows]); h2 = np.array([r['h2_abs'] for r in rows])
    c0 = np.array([r['harm']['c0'] for r in rows])
    d = dict(V=[r['V'] for r in rows], amp=amp.tolist(), pc=pc.tolist(), c0=c0.tolist(), h4=h4.tolist(), h2=h2.tolist())
    # lambda-infinity-free observable: splitting |h4| (B families: first order; E2: second order expected)
    nested = {}
    n = len(rows)
    for i, j in itertools.combinations(range(n), 2):
        if j == i + 1 and h4[i] > 0 and h4[j] > 0:
            nested[f'{rows[i]["p"]}/{rows[i]["q"]}-{rows[j]["p"]}/{rows[j]["q"]}'] = math.log(h4[i] / h4[j]) / math.log(amp[i] / amp[j])
    d['alpha_h4_pairs'] = nested
    d['alpha_h4_all'] = pw(amp, h4)[0]
    d['alpha_h4_last3'] = pw(amp[-3:], h4[-3:])[0] if n >= 3 else None
    fits[k] = d

# shared-limit fit of the total drop: lambda_inf common to all four families (they converge to the same QC)
allrows = [(k, r['amp'], r['pc_min_sampled']) for k, rows in F.items() for r in rows if r['amp'] < 0.03]


def joint(lam):
    res = {}
    for k in F:
        pts = [(a, p) for kk, a, p in allrows if kk == k]
        x = np.array([a for a, _ in pts]); y = lam - np.array([p for _, p in pts])
        if np.any(y <= 0):
            return None
        res[k] = pw(x, y)
    return res


c0max = max(r['harm']['c0'] for r in fam if r['amp'] < 1e-3)
pcmax = max(r['pc_min_sampled'] for r in fam)
band = {}
for lam in np.linspace(pcmax + 2e-6, c0max + 4e-5, 9):
    j = joint(lam)
    if j:
        band[f'{lam:.6f}'] = {k: v[0] for k, v in j.items()}
fits['total_drop_alpha_vs_assumed_lambda_inf'] = band
fits['lambda_inf_range_used'] = [float(pcmax), float(c0max + 4e-5)]
# independent numbers through the same lens: total-drop exponents of independent points for the same lambda band, shifted by the
# constant mesh offset (sub4 - sub2) measured on B1 5/7: identical tilings? (different offsets/generators)
cb = {}
for lam4 in np.linspace(0.796727, 0.79690, 6):
    cb[f'{lam4:.6f}'] = {k: pw(np.array([a for _, a, _ in v]), lam4 - np.array([p for _, _, p in v]))[0]
                         for k, v in independent.items() if all(lam4 > p for _, _, p in v)}
fits['independent_points_alpha_vs_assumed_lambda_inf_sub4'] = cb
out['items']['families'] = fits
off = json.loads((H / 'window_offsets.json').read_text())
out['items']['offset_averaging_12_17'] = dict(summary=off['summary'], rows=off['rows'], source='window_offsets.py', status='P')

# ---- flips
try:
    fl = json.loads((H / 'flip_ladder.json').read_text())['res']
    s = fl['single_flips_5_7']
    dd = np.array([r['delta_sub4'] for r in s])
    out['items']['single_flips_5_7'] = dict(
        canonical_sub4=fl['canonical_5_7']['4']['pc'] if '4' in fl['canonical_5_7'] else fl['canonical_5_7'][4]['pc'],
        canonical_sub8=fl['canonical_5_7']['8']['pc'] if '8' in fl['canonical_5_7'] else None,
        n_legal=len(s), n_raise=int((dd > 0).sum()), n_lower=int((dd < 0).sum()), mean=float(dd.mean()),
        max=float(dd.max()), min=float(dd.min()),
        sub8_checks=[dict(vertex=r['vertex'], d4=r['delta_sub4'], d8=r.get('delta_sub8')) for r in s if 'delta_sub8' in r])
    for key in ('ladder_5_7_sub4', 'ladder_12_17_sub2'):
        if key not in fl:
            continue
        L = fl[key]; tab = {}
        for r in L:
            tab.setdefault(r['k'], []).append(r)
        out['items'][f'{key}'] = {str(k): dict(n=len(v), moved_mean=float(np.mean([r['moved'] for r in v])),
                                                     delta_mean=float(np.mean([r['delta'] for r in v])),
                                                     delta_sem=float(np.std([r['delta'] for r in v], ddof=1) / math.sqrt(len(v))),
                                                     n_raise=int(sum(r['delta'] > 0 for r in v)),
                                                     deltas=[r['delta'] for r in v]) for k, v in sorted(tab.items())}
except FileNotFoundError:
    pass
(H / 'families_results.json').write_text(json.dumps(out, indent=1, default=float), encoding='utf-8')
print(json.dumps(out, indent=1, default=float)[:6000])

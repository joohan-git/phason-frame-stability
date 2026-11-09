"""intervention arm B: exact pair lists from covariate pools only (heldout pools + intervention draw intervention_pool_*.json). No frame quantity is read. Writes pairs.json. Rule in registered plan section 5."""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames', 'penrose', 'heldout_test', 'design', 'nonlinear', 'defects', 'phason_paths', 'independent_code', 'forbidden_patches')]  # shared modules
import json, sys
from collections import Counter
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; M = HERE.parent
P16 = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE
EDGES = np.geomspace(0.004, 0.065, 7)
CAL_D, CAL_V, MIN_DN, NPER = 0.05, 0.05, 0.05, 4


def to_f(L):
    return [L[0], -L[3], L[1], -L[4], L[2]]


def inv(c):
    """Point-group invariant identity, as registered plan, section 2."""
    return (c['V'], round(c['delta'], 9)) + tuple(round(c['sectors'][k], 6) for k in sorted(c['sectors']) if k != 'delta2')


def sect_ok(cls, a, b):
    s, t = a['sectors'], b['sectors']
    if cls == 'ab':
        return abs(s['B1'] - t['B1']) <= 0.10 and abs(s['B2'] - t['B2']) <= 0.10
    return abs(s['E4diag'] - t['E4diag']) <= 0.02 and abs(s['E2'] - t['E2']) <= 0.10


def pairs(cls, cells, sector):
    out = []
    for b in range(6):
        cb = sorted([c for c in cells if EDGES[b] <= c['delta'] < EDGES[b + 1]], key=lambda c: c['delta'])
        cand = []
        for i, a in enumerate(cb):
            for c in cb[i + 1:]:
                if (c['delta'] - a['delta']) / (0.5 * (a['delta'] + c['delta'])) > CAL_D:
                    break
                if abs(a['V'] - c['V']) / (0.5 * (a['V'] + c['V'])) > CAL_V:
                    continue
                dn = abs(a['new2'] - c['new2'])
                if dn < MIN_DN or (sector and not sect_ok(cls, a, c)):
                    continue
                lo, hi = (a, c) if a['new2'] < c['new2'] else (c, a)
                cand.append((dn, lo, hi))
        cand.sort(key=lambda x: (-x[0], str(x[1]['L1']), str(x[2]['L1'])))
        used = set(); n = 0
        for dn, lo, hi in cand:
            if inv(lo) in used or inv(hi) in used:
                continue
            used |= {inv(lo), inv(hi)}; n += 1
            out.append(dict(cls=cls, bin=b, dn=dn, lo=lo, hi=hi))
            if n == NPER:
                break
    return out


res = {}
t1 = json.load(open(M / 'isotropic_loss' / 'isotropic_results.json'))
for cls in ('ab', 'pen'):
    pool = json.load(open(M / 'heldout_test' / f'pool_{cls}.json'))
    sel = json.load(open(M / 'heldout_test' / f'selection_{cls}.json'))['selected']
    if cls == 'pen':
        for c in pool:
            c['L1'] = to_f(c['L1']); c['L2'] = to_f(c['L2'])
    p16 = json.load(open(P16 / f'intervention_pool_{cls}.json'))['cells']
    excl = {inv(c) for c in sel}
    stored = t1['fixedV'][cls]['cells']

    def keep(c):
        return inv(c) not in excl and not any(s['V'] == c['V'] and abs(s['delta'] - c['delta']) < 1e-6 for s in stored)
    seen = {}
    for c in p16 + pool:
        if keep(c):
            seen.setdefault(inv(c), c)
    cells = [dict(L1=c['L1'], L2=c['L2'], V=c['V'], delta=c['delta'], new2=c['new2'], sectors=c['sectors']) for c in seen.values()]
    for tag, sec in (('P', False), ('S', True)):
        r = pairs(cls, cells, sec)
        res[f'{cls}_{tag}'] = r
        print(cls, tag, 'cells', len(cells), 'pairs', len(r), sorted(Counter(x['bin'] for x in r).items()), 'dn', np.round([x['dn'] for x in r], 3).tolist())
(HERE / 'pairs.json').write_text(json.dumps(res, indent=0))

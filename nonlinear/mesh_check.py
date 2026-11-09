"""Gate (d): sub 2 vs 4. Uses edge-chord amplitude (element chords depend on the mesh; this was found during the gate and is recorded in)."""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames', 'phason_paths', 'objectivity', 'disorder')]  # shared modules
import json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; OUT = HERE / 'out'
res = {}


def curve(rows, key):
    a = np.array([r[key] for r in rows]); p = np.array([r['p'] for r in rows])
    return a, p


def p_at(rows, key, t):
    a, p = curve(rows, key)
    k = np.where(a >= t)[0]
    return None if not len(k) or k[0] == 0 else float(np.interp(t, a[:k[0] + 1], p[:k[0] + 1]))


for name, key_amp in (('square', 'rms_edge'), ('B1_5_7', 'rms_edge'), ('E2a_5_7', 'max_edge'), ('R22', 'max_edge')):
    d2 = json.loads((OUT / f'perfect_{name}_s2.json').read_text()); d4 = json.loads((OUT / f'perfect_{name}_s4.json').read_text())
    n2 = json.loads((OUT / f'natural_{name}_s2.json').read_text()); n4 = json.loads((OUT / f'natural_{name}_s4.json').read_text())
    m2 = d2['modes'][0]; m4 = d4['modes'][0]
    e = dict(lin2=d2['lin'][0], lin4=d4['lin'][0], lin_change_pct=100 * (d4['lin'][0] / d2['lin'][0] - 1), pc2=m2['pc'], pc4=m4['pc'], pc_change_pct=100 * (m4['pc'] / m2['pc'] - 1))
    # branch p at fixed edge amplitude, both signs where available
    tab = {}
    for sign in ('1', '-1'):
        b2 = m2['branches'].get(sign) or {}; b4 = m4['branches'].get(sign) or {}
        if 'rows' in b2 and 'rows' in b4:
            for t in (0.01, 0.02, 0.05, 0.1, 0.15):
                x2, x4 = p_at(b2['rows'], key_amp, t), p_at(b4['rows'], key_amp, t)
                if x2 and x4:
                    tab[f'{sign}@{t}'] = dict(p2=x2, p4=x4, ratio2=x2 / m2['pc'], ratio4=x4 / m4['pc'], p_change_pct=100 * (x4 / x2 - 1), excess_ratio=(x4 / m4['pc'] - 1) / max(x2 / m2['pc'] - 1, 1e-12))
    e['branch_p_at_fixed_amp'] = tab
    ntab = {}
    for t in (0.02, 0.05, 0.1, 0.2):
        x2, x4 = p_at(n2['rows'], 'max_edge', t), p_at(n4['rows'], 'max_edge', t)
        if x2 and x4:
            ntab[str(t)] = dict(p2=x2, p4=x4, change_pct=100 * (x4 / x2 - 1))
    e['natural_p_at_fixed_max_edge'] = ntab
    # first limit on the natural path
    def lim(rows):
        p = np.array([r['p'] for r in rows]); dp = np.diff(p); i = np.where(dp < 0)[0]
        return None if not len(i) else float(p[i[0]])
    e['natural_first_limit'] = dict(s2=lim(n2['rows']), s4=lim(n4['rows']))
    res[name] = e
    print(name, 'lin %.3f%% pc %.3f%%' % (e['lin_change_pct'], e['pc_change_pct']), {k: round(v['p_change_pct'], 3) for k, v in tab.items()}, {k: round(v['change_pct'], 3) for k, v in ntab.items()}, e['natural_first_limit'])
(HERE / 'mesh_check.json').write_text(json.dumps(res, indent=1))

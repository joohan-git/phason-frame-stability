"""robust: robustness of the 2D results. A mesh (sub 2, 4, 8 at fixed geometry); B slenderness r (0.015-0.10)."""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames', 'phason_paths', 'penrose', 'isotropic_loss')]  # shared modules
import os, sys, json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; M = HERE.parent
import ab_frame as C
src = ''
ns = {'np': np}; ns.update(vars(__import__('ab_families'))); fl = ns['family_lifts']
t1 = json.loads((R / 'data' / 'isotropic_results.json').read_text())['fixedV']
def L(x): return [list(map(int, x[0])), list(map(int, x[1]))]
AB_C = [t1['ab']['cells'][i] for i in (0, 3, 5, 7, 9)]
PN_C = [t1['pen']['cells'][i] for i in (0, 3, 5, 7, 9)]
T = []
def add(group, cls, name, L1, L2, off, sub, r, **kw):
    T.append(dict(group=group, cls=cls, name=name, L1=L1, L2=L2, off=off, sub=sub, r=r, **kw))
fam = lambda p, q, f: L(fl(p, q, f))
# ---- A mesh
for sub in (2, 4):
    for c in AB_C: add('mesh', 'ab', f"cell{c['V']}", c['L1'], c['L2'], 7, sub, 0.03, delta=c['delta'])
    add('mesh', 'ab', 'pell29/41', *fam(29, 41, 'B1'), 7, sub, 0.03)
    for c in PN_C: add('mesh', 'pen', f"cell{c['V']}", c['L1'], c['L2'], None, sub, 0.03, delta=c['delta'])
for f_, p, q in (('B1', 5, 7), ('B1', 12, 17)):
    for sub in (2, 4, 8): add('mesh', 'ab', f'{f_}{p}/{q}', *fam(p, q, f_), 7, sub, 0.03)
for c in AB_C[:2]: add('mesh', 'ab', f"cell{c['V']}", c['L1'], c['L2'], 7, 8, 0.03, delta=c['delta'])
# ---- B slenderness
for r in (0.015, 0.03, 0.06, 0.10):
    for p, q in ((5, 7), (12, 17)):
        for f_ in ('B1', 'B2'): add('slender', 'ab', f'{f_}{p}/{q}', *fam(p, q, f_), 7, 2, r)
        for s in (7, 11, 23): add('slender', 'ab', f'E2a{p}/{q}', *fam(p, q, 'E2a'), s, 2, r)
    add('slender', 'ab', 'pell29/41', *fam(29, 41, 'B1'), 7, 2, r)
    for c in AB_C: add('slender', 'ab', f"cell{c['V']}", c['L1'], c['L2'], 7, 2, r, delta=c['delta'])
    for seed in (1, 2, 3, 4): add('slender', 'ab', f'random12/17', *fam(12, 17, 'B1'), 7, 2, r, flips=30000, fseed=seed)
print(len(T)); (HERE / 'plan.json').write_text(json.dumps(T), encoding='utf-8')

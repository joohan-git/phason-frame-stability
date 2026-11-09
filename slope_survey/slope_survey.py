"""Random survey of small rational slopes in the closed positive Grassmannian Gr+(2,4),
plus a long-wave (DQ) convergence check. Uses slope_thresholds.py (same folder)."""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames',)]  # shared modules
import itertools, json, math, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import slope_thresholds as S

rng = np.random.default_rng(261108)
seen = {}
for trial in range(20000):
    L1 = rng.integers(-1, 4, 4); L2 = rng.integers(-1, 4, 4)
    m = S.minors(L1, L2)
    v = np.array(list(m.values()))
    if np.all(v <= 0):
        v = -v; L1, L2 = L2, L1
    if np.any(v < 0) or v.sum() == 0 or v.sum() > 30:
        continue
    g = np.gcd.reduce(v)
    key = tuple(v // g)          # projective Pluecker point = slope
    if key in seen:
        continue
    if np.count_nonzero(v) < 3:  # skip pure / 2-type degenerate (covered elsewhere)
        continue
    seen[key] = (L1.tolist(), L2.tolist())
    if len(seen) >= 60:
        break

rows = []
for key, (L1, L2) in seen.items():
    try:
        r = S.evaluate('rand_' + ''.join(map(str, key)), L1, L2, nk=4)
        rows.append(r)
    except AssertionError as e:
        rows.append(dict(name=str(key), error=str(e)))

conv = {}
for name, L1, L2 in [s for s in S.SLOPES if s[0] in ('square_13', 'AB_1/1', 'AB_2/3', 'asym_C')]:
    conv[name] = {}
    for dq in (0.008, 0.004, 0.002, 0.001):
        S.DQ = dq
        r = S.evaluate(name, L1, L2, protocols=('strain',))
        conv[name][dq] = r['results']['strain']['lam_longwave_over_lamE']
S.DQ = 0.004
Path(__file__).with_name('slope_survey.json').write_text(json.dumps(dict(rows=rows, dq_convergence=conv), indent=1, default=str), encoding='utf-8')

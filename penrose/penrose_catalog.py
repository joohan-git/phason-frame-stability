"""Catalog of mirror-symmetric Penrose rational approximants L1=(X,B,C,-C,-B), L2=(0,E,F,F,E)
(rectangular cells; phason components Re a (E4), Re b (E2), Re v (E4, diagonal tilt) only).
-> penrose_catalog.json : valid cells with V <= VMAX that are >= 95% pure along a, b or v. (exact rational geometry)"""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames',)]  # shared modules
import json, math, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import penrose_frame as C
VMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 16000
t = C.TAU
rows = []
for B in range(1, 90):
    for C_ in range(max(1, round(B / t**2) - 2), round(B / t**2) + 3):
        for X in range(max(1, round(2 * B / t) - 2), round(2 * B / t) + 3):
            L1 = [X, B, C_, -C_, -B]
            for E in range(1, 90):
                for F in range(max(1, round(t * E) - 2), round(t * E) + 3):
                    L2 = [0, E, F, F, E]
                    m = C.minors(L1, L2); vals = np.array(list(m.values()))
                    if np.any(vals <= 0) or vals.sum() > VMAX:
                        continue
                    a, b, v = C.phason_parts(C.phason_matrix(L1, L2))
                    rows.append(dict(L1=L1, L2=L2, V=int(vals.sum()), a=a.real, b=b.real, v=v.real,
                                     im=max(abs(a.imag), abs(b.imag), abs(v.imag)),
                                     aspect=float(np.linalg.norm(C.P @ np.array(L1)) / np.linalg.norm(C.P @ np.array(L2)))))
print(len(rows), max(r['im'] for r in rows))
ntot = len(rows)
def pur(r, k):
    d = math.sqrt(r['a'] ** 2 + r['b'] ** 2 + r['v'] ** 2)
    return abs(r[k]) / d if d > 0 else 0
rows = [r for r in rows if max(pur(r, 'a'), pur(r, 'b'), pur(r, 'v')) >= 0.95]
Path(__file__).with_name('penrose_catalog.json').write_text(json.dumps(dict(status='exact geometry', n_valid_cells=ntot, kept='purity >= 0.95 along a, b or v', rows=rows)), encoding='utf-8')

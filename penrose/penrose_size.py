"""Is the isotropic c0 deficit a phason (|delta|) effect or a finite-size (1/V) effect?
Along the a- and Fibonacci lines |delta|*V = 4.35 is constant, so they cannot tell. Here: in-plane
mirror cells (X = 2(B-C), v = 0) at fixed V ~ 1000-1700 with |delta| spread over a decade.
-> penrose_size.json (P)"""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames',)]  # shared modules
import os
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='-1')
import json, math, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import penrose_frame as C
from penrose_cusp import evaluate
t = C.TAU; cand = []
for B in range(2, 40):
    for C_ in range(1, B):
        X = 2 * (B - C_); L1 = [X, B, C_, -C_, -B]
        for E in range(1, 40):
            for F in range(1, 2 * E + 2):
                L2 = [0, E, F, F, E]; m = np.array(list(C.minors(L1, L2).values()))
                if np.any(m <= 0) or not (900 <= m.sum() <= 1700):
                    continue
                a, b, v = C.phason_parts(C.phason_matrix(L1, L2)); d = math.hypot(abs(a), abs(b))
                asp = np.linalg.norm(C.P @ np.array(L1)) / np.linalg.norm(C.P @ np.array(L2))
                if 0.5 < asp < 2.0 and d < 0.08:
                    cand.append((d, int(m.sum()), L1, L2, a.real, b.real))
cand.sort()
targets = [0.003, 0.006, 0.012, 0.025, 0.05]
pick = []
for tg in targets:
    c = min(cand, key=lambda x: abs(math.log(x[0] / tg)))
    if c not in pick:
        pick.append(c)
rows = []
for d, V, L1, L2, a, b in pick:
    r = evaluate(f'fixedV_{V}_d{d:.4f}', L1, L2, None, (0.002, 0.001), False); rows.append(r)
    Path(__file__).with_name('penrose_size.json').write_text(json.dumps(dict(status='P', rows=rows), indent=1), encoding='utf-8')

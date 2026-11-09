"""Controls: (1) long-wave DQ ladder on a_4 (V=1042) and b_fib4 (V=644);
(2) generalized-Penrose offset scan: Fibonacci n=4,5 with the diagonal offset
s = sum(-1)^k c_k - 1/2 in {0 (Penrose), 0.1, 0.2, 0.3, 0.4, 0.5}.  -> penrose_checks.json (P)"""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames',)]  # shared modules
import os
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='-1')
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import penrose_frame as C
from penrose_cusp import evaluate, A, B
out = Path(__file__).with_name('penrose_checks.json'); res = dict(status='P', dq_ladder=[], gpt_scan=[])
for nm, L in (('a_4', A[4]), ('b_fib4', B[3])):
    for dqs in ((0.004, 0.002), (0.002, 0.001), (0.001, 0.0005), (0.0005, 0.00025)):
        r = evaluate(f'{nm}_dq{dqs[0]}', *L, None, dqs, False); res['dq_ladder'].append(r)
        out.write_text(json.dumps(res, indent=1), encoding='utf-8')
sv = np.array([1, -1, 1, -1, 1])
for n in (4, 5):
    for s in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
        o = C.OFFSET.copy(); o += (0.5 + s - sv @ o) / 5 * sv
        r = evaluate(f'fib{n}_s{s}', *C.fib_approx(n), o, (0.002, 0.001), False); r['s'] = s
        res['gpt_scan'].append(r); out.write_text(json.dumps(res, indent=1), encoding='utf-8')

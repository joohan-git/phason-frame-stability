"""Offset (hull-position) averaging for the matched-|a| pairs: a-line cell a_k and Fibonacci
b-line cell fib_(k-1) have identical a but b differing by a factor -18 (b/a = -tau^-3 vs +tau^3).
4 random in-plane offsets per cell (diagonal offset kept at the Penrose value).
python penrose_offsets.py GROUP  (GROUP = 1: a_6, a_3, fib2, fib3; 2: fib6, a_5, fib5, fib4, a_4, a_2)  -> penrose_offsets_GROUP.json (P)"""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames',)]  # shared modules
import os
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='-1')
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import penrose_frame as C
from penrose_cusp import evaluate, A, B
g = sys.argv[1]
cells = {'1': [('a_6', A[6]), ('a_3', A[3]), ('b_fib2', B[1]), ('b_fib3', B[2])],
         '2': [('b_fib6', B[5]), ('a_5', A[5]), ('b_fib5', B[4]), ('b_fib4', B[3]), ('a_4', A[4]), ('a_2', A[2])]}[g]
sv = np.array([1, -1, 1, -1, 1]); rows = []
out = Path(__file__).with_name(f'penrose_offsets_{g}.json')
for nm, L in cells:
    rng = np.random.default_rng(101)
    for s in range(4):
        o = rng.uniform(-.5, .5, 5); o += (0.5 - sv @ o) / 5 * sv
        rows.append(evaluate(f'{nm}_off{s}', *L, o, (0.002, 0.001), False))
        out.write_text(json.dumps(dict(status='P', rows=rows), indent=1), encoding='utf-8')

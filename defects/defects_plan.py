"""plans. plan_linear.json (A, C), plan_nonlinear.json (B, B3). Task lists are deterministic."""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames', 'design', 'nonlinear', 'phason_paths')]  # shared modules
import json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; M = HERE.parent
ns = {'np': np}
src = ''
ns.update(vars(__import__('ab_families')))
fl = ns['family_lifts']


def fam(p, q, f):
    L = fl(p, q, f)
    return [list(map(int, L[0])), list(map(int, L[1]))]


def fib(n):
    F = [1, 1]
    while len(F) < n + 3:
        F.append(F[-1] + F[-2])
    return [[2 * F[n], F[n + 1], F[n - 1], -F[n - 1], -F[n + 1]], [0, F[n], F[n + 1], F[n + 1], F[n]]]


def ab(name, p, q, **kw):
    L = fam(p, q, 'B1'); return dict(cls='ab', name=name, L1=L[0], L2=L[1], off=7, **kw)


def pen(name, n):
    L = fib(n); return dict(cls='pen', name=name, L1=L[0], L2=L[1], off=7)


def ref(name, lat, N):
    return dict(cls='ref', name=name, lattice=lat, N1=N, N2=N)


C = {'B112/17': ab('B112/17', 12, 17), 'B15/7': ab('B15/7', 5, 7), 'B129/41': ab('B129/41', 29, 41), 'fib5': pen('fib5', 5),
     'square38': ref('square38', 'square', 38), 'hex26': ref('hex26', 'hexagonal', 26), 'tri18': ref('tri18', 'triangular', 18)}
LV = ['k1', 0.005, 0.01, 0.02, 0.05]
DIRECT = {'B112/17', 'fib5', 'square38', 'hex26'}
L = []


def base(c, **kw):
    L.append(dict(kind='base', group=kw.pop('group', 'A'), **c, **kw))


def miss(c, level, i, **kw):
    L.append(dict(kind='miss', group=kw.pop('group', 'A'), level=level, i=i, **c, **kw))


# --- A main
for name, n in (('B112/17', 48), ('B15/7', 24), ('fib5', 24), ('square38', 24), ('hex26', 24), ('tri18', 24)):
    c = C[name]
    base(c)
    for lv in LV:
        for i in range(n):
            miss(c, lv, i, direct=(lv != 'k1' and name in DIRECT and i < 12))
for name in ('B112/17', 'hex26'):
    for i in range(12):
        miss(C[name], 0.10, i, direct=True)
# --- uniform thinning on the intact cell
for name in ('B112/17', 'hex26'):
    for f_ in (0.005, 0.01, 0.02, 0.05, 0.10):
        base(C[name], r=0.03 * (1 - f_), group='thin')
        base(C[name], r=0.03 / (1 - f_), group='thin')
# --- A random
for s in range(100, 108):
    c = dict(C['B112/17'], flips=-1, fseed=s)
    base(c, group='Arand')
    for lv in LV:
        for j in range(3):
            miss(c, lv, j, group='Arand')
# --- size and mesh
c = C['B129/41']
base(c, group='size')
for lv in ('k1', 0.01, 0.05):
    for i in range(12):
        miss(c, lv, i, group='size')
c = C['B112/17']
base(c, group='mesh', sub=4)
for lv in ('k1', 0.01, 0.05):
    for i in range(8):
        miss(c, lv, i, group='mesh', sub=4)
# --- census 5/7 (all edges, chunks of 40)
E57 = 2 * json.loads((M / 'design' / 'design_results.json').read_text())['D']['cells']['B15/7']['V']
for a in range(0, E57, 40):
    L.append(dict(kind='census', group='census', edges=list(range(a, min(a + 40, E57))), **C['B15/7']))
# --- C slenderness
for r in (0.015, 0.06, 0.10):
    for name, n in (('B112/17', 12), ('fib5', 6), ('hex26', 6)):
        base(C[name], r=r, group='C')
        for lv in ('k1', 0.01, 0.05):
            for i in range(n):
                miss(C[name], lv, i, r=r, group='C')
    for s in range(100, 104):
        c = dict(C['B112/17'], flips=-1, fseed=s)
        base(c, r=r, group='C')
        for lv in ('k1', 0.01, 0.05):
            miss(c, lv, 0, r=r, group='C')
(HERE / 'plan_linear.json').write_text(json.dumps(L), encoding='utf-8')

# --- nonlinear plan
N = []
AMPS = [1e-4, 1e-3, 3e-3, 1e-2, 3e-2]
for name, sub, n, amps in (('B1_12_17', 4, 8, AMPS), ('B1_5_7', 4, 8, AMPS), ('R22', 4, 4, AMPS), ('R101', 4, 4, AMPS),
                           ('B1_12_17', 2, 4, AMPS), ('R22', 2, 4, AMPS), ('B1_5_7', 8, 4, [1e-4, 1e-2, 3e-2])):
    for w0 in amps:
        for i in range(n):
            N.append(dict(kind='bow', name=name, sub=sub, w0=w0, i=i, shape='half'))
for w0 in (1e-4, 1e-2, 3e-2):
    for i in range(4):
        N.append(dict(kind='bow', name='B1_5_7', sub=8, w0=w0, i=i, shape='full'))
for lv in ('k1', 0.01, 0.05):
    for i in range(4):
        N.append(dict(kind='nlmiss', name='B1_12_17', sub=4, level=lv, i=i, w0=1e-4, cellname='B112/17'))
for name in ('B1_12_17', 'R22'):
    for sub in (2, 4, 8, 16):
        for i in range(2):
            N.append(dict(kind='kconv', name=name, sub=sub, i=i, shape='half'))
(HERE / 'plan_nonlinear.json').write_text(json.dumps(N), encoding='utf-8')
print(len(L), len(N))
import collections
print(collections.Counter((t['group'], t.get('name'), t['kind']) for t in L))

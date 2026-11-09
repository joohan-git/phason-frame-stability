"""plan. plan_part1.json: the first part of the plan. plan_part2.json: the second part of the plan.
Keys identical in both; no task in robustness/plan.json is repeated (mesh and r sweep of lam_c on its AB cells)."""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames', 'penrose', 'phason_paths', 'robustness', 'heldout_test', 'isotropic_loss')]  # shared modules
import json, math, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; M = HERE.parent; R = HERE.parents[2] / 'the article'
src = ''
ns = {'np': np}
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


def cell(cls, name, L, off=7, **kw):
    return dict(cls=cls, name=name, L1=L[0], L2=L[1], off=off, **kw)


RALL = [0.005, 0.01, 0.015, 0.02, 0.03, 0.045, 0.06, 0.08, 0.10]
RT8 = [0.015, 0.03, 0.06, 0.10]
REFS = ['square', 'rhomb45', 'triangular', 'kagome', 'hexagonal', 'rhombille']
EQ = -1  # flips = EQ -> equilibrated
PART1, PART2 = [], []


def add(lst, group, kind, **kw):
    lst.append(dict(group=group, kind=kind, **kw))


# ---- A. references: full r sweep at sub 2 and 8
for name in REFS:
    add(PART1, 'A', 'ref', lattice=name, rs=RALL, subs=[2, 8])
# ---- A/E. moduli and strut demand across r (no eigenproblem, not in robust)
ordered = [cell('ab', f'B1{p}/{q}', fam(p, q, 'B1')) for p, q in ((1, 1), (2, 3), (5, 7), (12, 17), (29, 41))]
ordered += [cell('pen', f'fib{n}', fib(n)) for n in (2, 3, 4, 5, 6)]
for c in ordered:
    add(PART1, 'A', 'moduli', rs=RALL, **c)
for s in (1, 2, 3, 4):
    add(PART1, 'A', 'moduli', rs=RALL, flips=EQ, fseed=s, **cell('ab', 'B112/17', fam(12, 17, 'B1')))
    add(PART1, 'A', 'moduli', rs=RALL, flips=EQ, fseed=s, **cell('pen', 'fib4', fib(4)))
# ---- A. lam_c where robust does not supply it
for r in RT8:   # Penrose has no slenderness sweep in robust; B1 2/3 is not in robust
    for n in (3, 4, 5):
        add(PART1, 'A', 'pc', r=r, **cell('pen', f'fib{n}', fib(n)))
    add(PART1, 'A', 'pc', r=r, **cell('ab', 'B12/3', fam(2, 3, 'B1')))
    for s in (1, 2, 3, 4):
        add(PART1, 'A', 'pc', r=r, flips=EQ, fseed=s, **cell('pen', 'fib4', fib(4)))
    for s in (1, 2):
        add(PART1, 'A', 'pc', r=r, flips=EQ, fseed=s, **cell('ab', 'B15/7', fam(5, 7, 'B1')))
    add(PART2, 'A', 'pc', r=r, **cell('pen', 'fib6', fib(6)))
    for s in (1, 2, 3, 4, 5, 6):
        add(PART2, 'A', 'pc', r=r, flips=EQ, fseed=s, **cell('pen', 'fib5', fib(5)))
for s in (1, 2, 3, 4):
    add(PART1, 'A', 'pc', r=0.03, flips=EQ, fseed=s, **cell('ab', 'B112/17', fam(12, 17, 'B1')))
for r in RT8:
    for s in (1, 2, 3, 4, 5, 6, 7, 8):
        if r == 0.03 and s <= 4:
            continue
        add(PART2, 'A', 'pc', r=r, flips=EQ, fseed=s, **cell('ab', 'B112/17', fam(12, 17, 'B1')))
add(PART1, 'A', 'pc', r=0.03, **cell('ab', 'B112/17', fam(12, 17, 'B1')))
add(PART1, 'A', 'pc', r=0.03, **cell('ab', 'B15/7', fam(5, 7, 'B1')))
# ---- B. phason paths (zero-load C and loaded tangent)
PATH = [(1, 1), (2, 3), (5, 7), (12, 17), (29, 41)]
for f_ in ('B1', 'B2', 'E2a', 'E2diag'):
    for p, q in PATH:
        if q == 41 and f_ in ('B2', 'E2diag'):
            continue
        add(PART1, 'B', 'path', family=f_, p=p, q=q, **cell('ab', f'{f_}{p}/{q}', fam(p, q, f_)))
for f_ in ('B1', 'E2a'):
    for off in (11, 23, 37, 51, 68):
        add(PART1, 'B', 'path', family=f_, p=12, q=17, **cell('ab', f'{f_}12/17', fam(12, 17, f_), off=off))
# ---- C. load paths
for c in (cell('ab', 'B15/7', fam(5, 7, 'B1')), cell('ab', 'B112/17', fam(12, 17, 'B1')), cell('pen', 'fib4', fib(4))):
    add(PART1, 'C', 'load', **c)
for off in (7, 11, 23):
    add(PART1, 'C', 'load', **cell('ab', 'E2a5/7', fam(5, 7, 'E2a'), off=off))
for s in (1, 2):
    add(PART1, 'C', 'load', flips=EQ, fseed=s, **cell('ab', 'B112/17', fam(12, 17, 'B1')))
    add(PART1, 'C', 'load', flips=EQ, fseed=s, **cell('pen', 'fib4', fib(4)))
for name in REFS:
    add(PART1, 'C', 'refload', lattice=name, sub=2)
    add(PART1, 'C', 'refload', lattice=name, sub=8)
for s in range(3, 9):
    add(PART2, 'C', 'load', flips=EQ, fseed=s, **cell('ab', 'B112/17', fam(12, 17, 'B1')))
    add(PART2, 'C', 'load', flips=EQ, fseed=s, **cell('pen', 'fib4', fib(4)))
add(PART2, 'C', 'load', **cell('ab', 'B129/41', fam(29, 41, 'B1')))
add(PART2, 'C', 'load', **cell('pen', 'fib5', fib(5)))
for off in (37, 51, 68):
    add(PART2, 'C', 'load', **cell('ab', 'E2a5/7', fam(5, 7, 'E2a'), off=off))
for off in (7, 11, 23):
    add(PART2, 'C', 'load', **cell('ab', 'E2a12/17', fam(12, 17, 'E2a'), off=off))
# ---- D. imperfection MC
LEVELS = {'E': [0.01, 0.02, 0.05, 0.10], 't': [0.01, 0.02, 0.05, 0.10], 'node': [0.005, 0.01, 0.02, 0.05]}
FLIPK = [10, 30, 100, 300, 1000, EQ]
CD = [cell('ab', 'B15/7', fam(5, 7, 'B1')), cell('ab', 'B112/17', fam(12, 17, 'B1')), cell('ab', 'E2a12/17', fam(12, 17, 'E2a')), cell('pen', 'fib4', fib(4))]
for c in CD:
    add(PART1, 'D', 'base', **c)
    for kind, lv in LEVELS.items():
        for s in lv:
            for i in range(8):
                add(PART1, 'D', 'mc', mc=kind, sigma=s, i=i, **c)
    for k in FLIPK:
        for i in range(8):
            add(PART1, 'D', 'base', flips=k, fseed=100 + i, **c)
CH = CD + [cell('ab', 'B129/41', fam(29, 41, 'B1')), cell('pen', 'fib5', fib(5))]
for c in CH:
    add(PART2, 'D', 'base', **c) if c['name'] in ('B129/41', 'fib5') else None
    for kind, lv in LEVELS.items():
        for s in lv:
            for i in range(8, 48) if c['name'] not in ('B129/41', 'fib5') else range(48):
                add(PART2, 'D', 'mc', mc=kind, sigma=s, i=i, **c)
    for k in FLIPK:
        for i in range(8, 24) if c['name'] not in ('B129/41', 'fib5') else range(24):
            add(PART2, 'D', 'base', flips=k, fseed=100 + i, **c)
(HERE / 'plan_part1.json').write_text(json.dumps(PART1), encoding='utf-8')
(HERE / 'plan_part2.json').write_text(json.dumps(PART2), encoding='utf-8')
import collections
print(len(PART1), collections.Counter((t['group'], t['kind']) for t in PART1))
print(len(PART2), collections.Counter((t['group'], t['kind'], t.get('name')) for t in PART2))

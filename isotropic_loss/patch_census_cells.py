"""radius-2 patch census (vertex star + signed directions to neighbours with THEIR stars).
Fraction of radius-1 and radius-2 patches absent from the QC atlas, for (i) the census cells of
patch_census.py and (ii) the mechanically evaluated fixed-V cells of fixed_size.py.
-> patch_census_cells.json (P)"""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames', 'phason_paths', 'penrose', 'symmetry')]  # shared modules
import os
os.environ.update(OMP_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='-1')
import json, sys, glob, itertools
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
HERE = Path(__file__).parent; ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'phason_paths')); sys.path.insert(0, str(ROOT / 'penrose')); sys.path.insert(0, str(HERE))
import ab_frame as AB  # noqa
import penrose_frame as PN   # noqa


def patches_periodic(lifts, L1, L2, M, nd):
    red = M.Reducer(L1, L2); look = {tuple(r): i for i, r in enumerate(lifts)}
    E = np.eye(nd, dtype=np.int64); nb = []; st = []
    for r in lifts:
        s = []; n = []
        for k in range(nd):
            for sg in (1, -1):
                j = look.get(tuple(red(r + sg * E[k])[0]))
                if j is not None:
                    s.append(sg * (k + 1)); n.append((sg * (k + 1), j))
        st.append(tuple(sorted(s))); nb.append(n)
    p2 = [(st[i], tuple(sorted((d, st[j]) for d, j in nb[i]))) for i in range(len(lifts))]
    return st, p2


def qc_atlas(cls, R=75.0):
    if cls == 'ab':
        nd, Pm, Qi = 4, AB.P, AB.Q; c = np.array([0.1173, 0.2711, -0.0618, 0.3307])
    else:
        nd, Pm, Qi = 5, PN.P, PN.QI; c = PN.OFFSET.copy(); alt = np.array([1, -1, 1, -1, 1]); c = c + (0.5 - (alt @ c) % 1.0) / 5 * alt
    cube = np.array(list(itertools.product([-.5, .5], repeat=nd)))
    eq = ConvexHull(cube @ Qi.T).equations; cq = Qi @ c
    A = eq[:, :-1] @ Qi; b0 = eq[:, -1] - eq[:, :-1] @ cq
    ins = lambda p: bool(np.all(A @ p + b0 < -1e-9))
    E = np.eye(nd, dtype=np.int64)
    start = next(np.array(v, np.int64) for v in itertools.product(range(-2, 3), repeat=nd) if ins(np.array(v, float)))
    S = {tuple(start)}; stack = [start]
    while stack:
        p = stack.pop()
        for k in range(nd):
            for sg in (1, -1):
                q = p + sg * E[k]; t = tuple(q)
                if t not in S and np.linalg.norm(Pm @ q) <= R + 3 and ins(q.astype(float)):
                    S.add(t); stack.append(q)
    def star(t):
        p = np.array(t)
        return tuple(sorted(sg * (k + 1) for k in range(nd) for sg in (1, -1) if tuple(p + sg * E[k]) in S))
    a1, a2 = {}, {}
    for t in S:
        p = np.array(t)
        if np.linalg.norm(Pm @ p) > R - 3:
            continue
        s = star(t); a1[s] = a1.get(s, 0) + 1
        nbs = tuple(sorted((sg * (k + 1), star(tuple(p + sg * E[k]))) for k in range(nd) for sg in (1, -1) if tuple(p + sg * E[k]) in S))
        key = (s, nbs); a2[key] = a2.get(key, 0) + 1
    return a1, a2


def main():
    out = {}
    for cls in ('ab', 'pen'):
        M = AB if cls == 'ab' else PN; nd = 4 if cls == 'ab' else 5
        a1, a2 = qc_atlas(cls)
        cells = [dict(src='census', **{k: r[k] for k in ('V', 'delta', 'abs_a', 'abs_b', 'L1', 'L2')})
                 for r in json.loads((HERE / 'patch_census.json').read_text())[cls]['rows']]
        for f in glob.glob(str(HERE / f'fixed_size_{cls}_*.json')):
            for r in json.loads(Path(f).read_text())['rows']:
                cells.append(dict(src='fixedV', V=r['V'], delta=r['delta'], L1=r['L1'], L2=r['L2'], c0_mean=r['c0_mean'],
                                  c0_sd=r['c0_sd'], frac_m4=r['frac_m4']))
        for c in cells:
            tl = M.canonical(c['L1'], c['L2'])
            st, p2 = patches_periodic(tl['lifts'], tl['L1'], tl['L2'], M, nd)
            c['new1'] = sum(s not in a1 for s in st) / len(st)
            c['new2'] = sum(p not in a2 for p in p2) / len(p2)
        out[cls] = dict(atlas1=len(a1), atlas2=len(a2), cells=cells)
        D = np.array([c['delta'] for c in cells]); n2 = np.array([c['new2'] for c in cells]); n1 = np.array([c['new1'] for c in cells])
        print(cls, 'atlas', len(a1), len(a2), 'new1/d', np.round(np.median(n1 / D), 3), 'new2/d median', np.round(np.median(n2 / D), 3),
              'new2/d small-d', np.round(np.median((n2 / D)[D < 0.02]), 3) if (D < 0.02).any() else None, flush=True)
    (HERE / 'patch_census_cells.json').write_text(json.dumps(dict(status='P', **out), indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()

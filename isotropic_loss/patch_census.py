"""(geometry only, no mechanics): are the C_N-INVARIANT patch statistics of canonical rational cuts
analytic at the QC slope?  Symmetry forbids a linear analytic change of any invariant statistic (no trivial
irrep in the phason strain), so a change ~|delta| is a non-analytic geometric effect.
Statistic: vertex-degree frequencies f_k (invariant under the point group) and the fraction of vertex
stars (radius 1, signed edge set up to nothing) absent from the QC atlas.
QC reference: large irrational cut-and-project patch (interior vertices only).
usage: python patch_census.py -> patch_census.json (P)"""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames', 'phason_paths', 'penrose', 'symmetry')]  # shared modules
import os
os.environ.update(OMP_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='-1')
import json, math, sys, itertools
from functools import reduce
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull

HERE = Path(__file__).parent; ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'phason_paths')); sys.path.insert(0, str(ROOT / 'penrose'))
import ab_frame as AB  # noqa
import penrose_frame as PN   # noqa


def stars_of(lifts, L1, L2, M, nd):
    red = M.Reducer(L1, L2); look = {tuple(r) for r in lifts}
    E = np.eye(nd, dtype=np.int64); out = []
    for r in lifts:
        s = []
        for k in range(nd):
            for sg in (1, -1):
                if tuple(red(r + sg * E[k])[0]) in look:
                    s.append(sg * (k + 1))
        out.append(tuple(sorted(s)))
    return out


def qc_patch(cls, R=None):
    """Irrational canonical patch (BFS over accepted Z^n points, physical radius R); stars of interior vertices."""
    if cls == 'ab':
        nd, Pm, Qi = 4, AB.P, AB.Q
        c = np.array([0.1173, 0.2711, -0.0618, 0.3307]); R = R or 75.0
    else:
        nd, Pm, Qi = 5, PN.P, PN.QI
        c = PN.OFFSET.copy(); alt = np.array([1, -1, 1, -1, 1]); c = c + (0.5 - (alt @ c) % 1.0) / 5 * alt; R = R or 75.0
    cube = np.array(list(itertools.product([-.5, .5], repeat=nd)))
    eq = ConvexHull(cube @ Qi.T).equations
    cq = Qi @ c
    A = eq[:, :-1] @ Qi; bconst = eq[:, -1] - eq[:, :-1] @ cq
    def inside(p):
        return bool(np.all(A @ p + bconst < -1e-9))
    E = np.eye(nd, dtype=np.int64)
    start = next(np.array(v, np.int64) for v in itertools.product(range(-2, 3), repeat=nd) if inside(np.array(v, float)))
    S = {tuple(start)}; stack = [start]
    while stack:
        p = stack.pop()
        for k in range(nd):
            for sg in (1, -1):
                q = p + sg * E[k]; t = tuple(q)
                if t in S or np.linalg.norm(Pm @ q) > R + 2:
                    continue
                if inside(q.astype(float)):
                    S.add(t); stack.append(q)
    stars = []
    for t in S:
        p = np.array(t)
        if np.linalg.norm(Pm @ p) > R - 2:
            continue
        s = [sg * (k + 1) for k in range(nd) for sg in (1, -1) if tuple(p + sg * E[k]) in S]
        stars.append(tuple(sorted(s)))
    return stars


def degree_hist(stars, nd):
    h = np.zeros(2 * nd + 1)
    for s in stars:
        h[len(s)] += 1
    return h / len(stars)


def sample_cells(cls, n, seed):
    M = AB if cls == 'ab' else PN; nd = 4 if cls == 'ab' else 5; scale = 0.5 if cls == 'ab' else 0.4
    rng = np.random.default_rng(seed); cells = []; seen = set()
    while len(cells) < n:
        r1, r2 = rng.uniform(3, 30, 2); t1 = rng.uniform(0, 2 * math.pi); t2 = t1 + rng.uniform(0.35, 0.65) * math.pi
        L1 = np.rint(M.P.T @ (r1 * np.array([math.cos(t1), math.sin(t1)])) * scale + rng.normal(0, .35, nd)).astype(np.int64)
        L2 = np.rint(M.P.T @ (r2 * np.array([math.cos(t2), math.sin(t2)])) * scale + rng.normal(0, .35, nd)).astype(np.int64)
        if cls == 'pen':
            alt = np.array([1, -1, 1, -1, 1])
            for L in (L1, L2):
                s = int(alt @ L); k = rng.integers(0, 5); L[k] -= s * alt[k]
        m = np.array(list(M.minors(L1, L2).values()))
        if np.all(m <= 0):
            L1, L2 = L2, L1; m = -m
        if not np.all(m > 0) or reduce(math.gcd, m.tolist()) != 1 or not (150 <= m.sum() <= 2500):
            continue
        Mt = M.phason_matrix(L1, L2)
        if cls == 'ab':
            a, b = M.harmonic_parts(Mt)
        else:
            a, b, v = M.phason_parts(Mt)
        d = math.hypot(abs(a), abs(b))
        key = (round(d, 8), int(m.sum()))
        if d > 0.08 or key in seen:
            continue
        seen.add(key); cells.append((L1, L2, d, int(m.sum()), abs(a), abs(b)))
    return cells


def main():
    res = {}
    for cls in ('ab', 'pen'):
        M = AB if cls == 'ab' else PN; nd = 4 if cls == 'ab' else 5
        qs = qc_patch(cls); atlas = set(qs); fq = degree_hist(qs, nd)
        rows = []
        for L1, L2, d, V, aa, bb in sample_cells(cls, 120, 3):
            tl = M.canonical(L1, L2)
            st = stars_of(tl['lifts'], tl['L1'], tl['L2'], M, nd)
            fh = degree_hist(st, nd)
            new = sum(1 for s in st if s not in atlas) / len(st)
            rows.append(dict(V=V, delta=d, abs_a=aa, abs_b=bb, deg_freq=fh.tolist(), new_star_frac=new,
                             L1=tl['L1'].tolist(), L2=tl['L2'].tolist()))
        res[cls] = dict(qc_n=len(qs), qc_atlas_size=len(atlas), qc_deg_freq=fq.tolist(), rows=rows)
        # fit |Delta f_k| = A|d| + B d^2 per degree (zero intercept)
        D = np.array([r['delta'] for r in rows]); F = np.array([r['deg_freq'] for r in rows]) - fq
        fits = {}
        for k in range(2 * nd + 1):
            if fq[k] == 0 and np.all(F[:, k] == 0):
                continue
            A = np.c_[D, D ** 2]; c, *_ = np.linalg.lstsq(A, F[:, k], rcond=None)
            c2, *_ = np.linalg.lstsq(np.c_[D ** 2], F[:, k], rcond=None)
            fits[k] = dict(lin=float(c[0]), quad=float(c[1]), rms_linquad=float(np.sqrt(np.mean((A @ c - F[:, k]) ** 2))),
                           rms_quad_only=float(np.sqrt(np.mean((D ** 2 * c2[0] - F[:, k]) ** 2))), qc=float(fq[k]))
        res[cls]['deg_fits'] = fits
        print(cls, len(qs), len(atlas), np.round(fq, 4), json.dumps(fits)[:600],
              'max new', max(r['new_star_frac'] for r in rows), flush=True)
    (HERE / 'patch_census.json').write_text(json.dumps(dict(status='P', **res), indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()

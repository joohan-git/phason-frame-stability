"""part on the unmodified design frames. Cell spec dict, removal through edges=, localization."""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames', 'design', 'nonlinear', 'phason_paths')]  # shared modules
import os
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[v] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import math, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; M = HERE.parent
sys.path.insert(0, str(HERE))
import design_core as X
import design_run as TR
from defects_util import *


def fam(p, q, f):
    ns = {'np': np}
    src = ''
    ns.update(vars(__import__('ab_families')))
    L = ns['family_lifts'](p, q, f)
    return [list(map(int, L[0])), list(map(int, L[1]))]


def fib(n):
    F = [1, 1]
    while len(F) < n + 3:
        F.append(F[-1] + F[-2])
    return [[2 * F[n], F[n + 1], F[n - 1], -F[n - 1], -F[n + 1]], [0, F[n], F[n + 1], F[n + 1], F[n]]]


def ref_module(lattice, r):
    nodes, a1, a2, E = X.ref_def(lattice)
    vecs = []
    for e in E:
        v = tuple(np.round(e[2], 12))
        if v not in vecs:
            vecs.append(v)
    mod = X.load('pen')
    mod.P = np.array(vecs, float).T; mod.NDIM = len(vecs)
    X.set_r(mod, r)
    return mod, nodes, a1, a2, E, vecs


def ref_super(lattice, N1, N2, r):
    """N1 x N2 supercell of a design reference lattice on a star-replaced penrose_frame copy."""
    mod, nodes, a1, a2, E, vecs = ref_module(lattice, r)
    nb = len(nodes)
    idx = lambda a, b, i: (a % N1 + N1 * (b % N2)) * nb + i
    X_ = np.zeros((N1 * N2 * nb, 2)); ed = []
    for b in range(N2):
        for a in range(N1):
            for i in range(nb):
                X_[idx(a, b, i)] = nodes[i] + a * a1 + b * a2
            for (i, j, v, w1, w2) in E:
                ed.append((idx(a, b, i), idx(a + w1, b + w2, j), vecs.index(tuple(np.round(v, 12))), (a + w1) // N1, (b + w2) // N2))
    edges = np.array(ed, np.int64)
    Pinv = np.linalg.pinv(mod.P)
    lifts = (Pinv @ X_.T).T
    return mod, lifts, Pinv @ (N1 * a1), Pinv @ (N2 * a2), edges


def cell_data(T, r):
    """(mod, lifts, L1, L2, edges) for a task spec; r only sets the module."""
    if T['cls'] == 'ref':
        return ref_super(T['lattice'], T['N1'], T['N2'], r)
    mod, tl, lifts, info = TR.build(T, r=r)
    edges = np.asarray(mod.edges_from_lifts(lifts, tl['L1'], tl['L2']), np.int64)
    return mod, lifts, tl['L1'], tl['L2'], edges


def cell_key(T):
    return f"{T['name']}|{T.get('fseed')}|{T['i']}"


def frame(mod, lifts, L1, L2, edges, sub):
    XF = X.make_xframe(mod)
    return XF(lifts, L1, L2, sub=sub, edges=np.asarray(edges, np.int64))


def crit_vector(f, c):
    if c['mode'] == 'longwave':
        p = c['phis'][int(np.argmin(c['lw']))]
        k = 0.001 * (f.cell @ np.array([math.cos(p), math.sin(p)]))
    else:
        k = np.array(eval(c['mode']), float)
    return f.lam_N(k, vectors=True)


def min_image_dist(f, x, y):
    """Minimum-image distance from points x (n,2) to one point y."""
    inv = np.linalg.inv(f.cell)
    d = (x - y) @ inv
    d -= np.rint(d)
    best = np.full(len(x), np.inf)
    for a in (-1, 0, 1):
        for b in (-1, 0, 1):
            v = (d + (a, b)) @ f.cell
            best = np.minimum(best, np.hypot(v[:, 0], v[:, 1]))
    return best


def localization(f, mod, c, removed_mid):
    lam, v = crit_vector(f, c)
    w = np.abs(v[0::3]) ** 2 + np.abs(v[1::3]) ** 2 + np.abs(v[2::3]) ** 2
    part = float(w.sum() ** 2 / (len(w) * (w ** 2).sum()))
    out = dict(part=part, lam_vec=float(lam))
    if len(removed_mid):
        D = np.full(len(w), np.inf)
        for m in removed_mid:
            D = np.minimum(D, min_image_dist(f, f.X, m))
        out.update(near_w=float(w[D <= 2].sum() / w.sum()), near_null=float((D <= 2).mean()), peak_dist=float(D[int(np.argmax(w))]))
    return out


def sample(f, mod, loc_mid=None):
    f.set_stress(-np.eye(2)); c = f.crit(nphi=8)
    m = X.moduli(f.C3); d = f.demand(-np.eye(2))
    out = dict(lam_c=c['lam_c'], lw_min=c['lw_min'], lw=c['lw'], extra=c['extra'], mode=c['mode'], spread=float((max(c['lw']) - min(c['lw'])) / min(c['lw'])),
               K=m['K'], Gmean=m['Gmean'], Emean=m['Emean'], absR0=m['absR0'], absR1=m['absR1'], ax_max=d['ax_max'], total_max=d['total_max'],
               ledge=len(f.ei) / f.sub / abs(np.linalg.det(f.cell)))
    if loc_mid is not None:
        out.update(localization(f, mod, c, loc_mid))
    return out


def removed_midpoints(mod, lifts, edges, rem):
    Xv = lifts @ mod.P.T
    return np.array([Xv[edges[e, 0]] + 0.5 * mod.P[:, edges[e, 2]] for e in rem]).reshape(-1, 2)

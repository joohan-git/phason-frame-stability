"""intervention arm B extension pool: a seeded random draw from the same complete compact enumeration as heldout (keep each index-N sublattice of Z^2 with probability KEEP), reservoir per log-delta bin,
covariates by the intervention census on the main-code canonical cut at the three outcome offsets (held-out seeds). No frame quantity. python build_pool.py cls keep per_bin"""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames', 'penrose', 'heldout_test', 'design', 'nonlinear', 'defects', 'phason_paths', 'independent_code', 'forbidden_patches')]  # shared modules
import json, sys, time
from multiprocessing import Pool
import numpy as np
from intervention_core import *
import cell_tiling as TT
import enumerate_cells as EN

DMIN, DMAX, NB = 0.004, 0.065, 6
EDGES = np.geomspace(DMIN, DMAX, NB + 1)


def enumerate_cells(cls, keep, per_bin, seed, Vmin=850, Vmax=1300, box=5, aspect=2.0):
    rng = np.random.default_rng(seed)
    star = TT.Star(cls); gr = EN.Graph(star); E = star.E; I = star.I
    J = np.zeros((2, 2))
    for c, i in enumerate(gr.base):
        m = np.zeros(2); m[c] = 1
        L = np.zeros(star.d); L[gr.base] = m; L[gr.other] = gr.ideal(m)
        J[:, c] = E @ L
    apv = 2 * (np.sqrt(2) - 1) if cls == 'ab' else (((1 + 5 ** 0.5) / 2) * np.sin(2 * np.pi / 5) + np.sin(np.pi / 5)) / ((1 + 5 ** 0.5) / 2 + 1)
    detJ = abs(np.linalg.det(J))
    Nmin = int(np.floor(Vmin * apv / detJ * 0.9)); Nmax = int(np.ceil(Vmax * apv / detJ * 1.1))
    offs = np.array(list(itertools.product(range(-box, box + 1), repeat=star.d - 2)))
    counts = np.zeros(NB, int); res = [[] for _ in range(NB)]
    for N in range(Nmin, Nmax + 1):
        for (r1, r2) in EN.sublattices_z2(N):
            if rng.random() > keep:
                continue
            m1, m2 = EN.gauss_reduce(r1, r2)
            if (m2 @ m2) > (aspect * 1.3) ** 2 * (m1 @ m1):
                continue
            cand = []
            for m in (m1, m2):
                lifts = np.rint(gr.ideal(m)).astype(np.int64)[None, :] + offs
                Ls = np.zeros((len(lifts), star.d), dtype=np.int64); Ls[:, gr.base] = m; Ls[:, gr.other] = lifts
                ratio = np.linalg.norm(Ls @ I.T, axis=1) / np.linalg.norm(Ls @ E.T, axis=1)
                cand.append(Ls[ratio <= np.sqrt(2) * DMAX * 1.0001])
            if len(cand[0]) == 0 or len(cand[1]) == 0:
                continue
            A, B = cand
            ia, ib = np.meshgrid(np.arange(len(A)), np.arange(len(B)), indexing='ij')
            LA = A[ia.ravel()]; LB = B[ib.ravel()]
            EL = np.stack([LA @ E.T, LB @ E.T], axis=2); IL = np.stack([LA @ I.T, LB @ I.T], axis=2)
            det = EL[:, 0, 0] * EL[:, 1, 1] - EL[:, 0, 1] * EL[:, 1, 0]
            ok = np.abs(det) > 1e-9
            inv = np.zeros_like(EL)
            inv[ok, 0, 0] = EL[ok, 1, 1] / det[ok]; inv[ok, 1, 1] = EL[ok, 0, 0] / det[ok]; inv[ok, 0, 1] = -EL[ok, 0, 1] / det[ok]; inv[ok, 1, 0] = -EL[ok, 1, 0] / det[ok]
            S = np.einsum('nij,njk->nik', IL, inv)
            delta = np.sqrt((S ** 2).sum(axis=(1, 2)) / 2); Vest = np.abs(det) / apv
            sel = ok & (delta >= DMIN) & (delta <= DMAX) & (Vest >= Vmin - 20) & (Vest <= Vmax + 20)
            for idx in np.nonzero(sel)[0]:
                L1, L2 = LA[idx], LB[idx]
                Lm = np.array([L1, L2]).T
                if not TT.is_primitive(Lm):
                    continue
                t1, t2 = EN.gauss_reduce_f(E @ L1, E @ L2)
                if np.linalg.norm(t2) > aspect * np.linalg.norm(t1):
                    continue
                b = min(int(np.searchsorted(EDGES, delta[idx], side='right') - 1), NB - 1)
                counts[b] += 1
                rec = (L1.tolist(), L2.tolist())
                if len(res[b]) < per_bin:
                    res[b].append(rec)
                else:
                    j = rng.integers(0, counts[b])
                    if j < per_bin:
                        res[b][j] = rec
    return res, counts.tolist()


def offsets_canonical(cls, L1, L2, off):
    if cls == 'ab':
        return AB.canonical(L1, L2, seed=[7, 11, 13][off])
    rng = np.random.default_rng(100 + off); o = rng.uniform(-.5, .5, 5); alt = np.array([1, -1, 1, -1, 1]); o += (0.5 - alt @ o) / 5 * alt
    return PN.canonical(L1, L2, offset=o)


_ATL = {}


def cov(T):
    cls, L1e, L2e = T
    star = TT.Star(cls)
    L1 = L1e if cls == 'ab' else list(e_to_f(L1e)); L2 = L2e if cls == 'ab' else list(e_to_f(L2e))
    if cls not in _ATL:
        _ATL[cls] = load_atlas(cls)
    try:
        _, dl, sec = star.phason(np.array([L1e, L2e]).T)
        phis = []; Vs = []
        for off in range(3):
            tl = offsets_canonical(cls, L1, L2, off)
            s = Sampler(cls, tl['L1'], tl['L2'], tl['lifts'], tl['perp'], tl['lifts'], atlas=_ATL[cls])
            phis.append(float(s.phi2)); Vs.append(s.V)
    except AssertionError:
        return None
    return dict(L1=L1, L2=L2, delta=float(dl), sectors={k: float(v) for k, v in sec.items()}, V=Vs[0], new2=float(np.mean(phis)), new2_all=phis)


if __name__ == '__main__':
    cls, keep, per_bin = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
    t0 = time.time()
    res, counts = enumerate_cells(cls, keep, per_bin, 16016)
    print(cls, 'counts in the draw', counts, 'enum sec', time.time() - t0, flush=True)
    T = [(cls, a, b) for r in res for a, b in r]
    out = []
    with Pool(6) as p:
        for i, c in enumerate(p.imap_unordered(cov, T, chunksize=8)):
            if c is not None:
                out.append(c)
            if i % 500 == 0:
                print(i, len(T), time.time() - t0, flush=True)
    json.dump(dict(counts=counts, keep=keep, cells=out), open(sys.argv[4] if len(sys.argv) > 4 else f'intervention_pool_{cls}.json', 'w'))
    print('done', len(out), time.time() - t0)

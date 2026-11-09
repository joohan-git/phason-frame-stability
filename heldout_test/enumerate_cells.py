"""enumerate_cells.py -- complete enumeration of rational planes (periodic approximant
cells) close to the quasicrystal plane, as graphs over two lattice
coordinates: sublattice Lambda' of Z^2 (index N, Gauss-reduced basis) plus
integer lifts of the two basis vectors within +-3 of the ideal lift.
Covariates only (V, delta, sectors, new1, new2); no frame quantities."""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames',)]  # shared modules
if __name__ == '__main__': __import__('os').chdir(__import__('os').path.dirname(__import__('os').path.abspath(__file__)))  # file names below are relative to this folder
import itertools
import json
import os
import sys
import time
from math import gcd

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cell_tiling import Star, Cell, is_primitive, make_gamma, build_atlas, forbidden_fraction  # noqa


def row_hnf(rows):
    """Canonical (row-style Hermite normal form) basis of the lattice spanned by
    the integer row vectors. Returns a tuple."""
    A = [list(map(int, r)) for r in rows]
    m, d = len(A), len(A[0])
    r = 0
    for c in range(d):
        piv = [i for i in range(r, m) if A[i][c] != 0]
        if not piv:
            continue
        while True:
            piv = [i for i in range(r, m) if A[i][c] != 0]
            if len(piv) <= 1:
                break
            i0 = min(piv, key=lambda i: abs(A[i][c]))
            for i in piv:
                if i != i0:
                    q = A[i][c] // A[i0][c]
                    A[i] = [x - q * y for x, y in zip(A[i], A[i0])]
        i0 = [i for i in range(r, m) if A[i][c] != 0][0]
        A[r], A[i0] = A[i0], A[r]
        if A[r][c] < 0:
            A[r] = [-x for x in A[r]]
        for i in range(r):
            q = A[i][c] // A[r][c]
            A[i] = [x - q * y for x, y in zip(A[i], A[r])]
        r += 1
        if r == m:
            break
    return tuple(tuple(row) for row in A)


def lattice_key(star, L1, L2):
    """HNF key minimized over the point group (lattice identity up to symmetry)."""
    L = np.array([L1, L2], dtype=np.int64)
    best = None
    for g in star.group:
        k = row_hnf((g @ L.T).T)
        if best is None or k < best:
            best = k
    return best


def sublattices_z2(N):
    """HNF bases of all index-N sublattices of Z^2: rows (a, b), (0, d), ad = N, 0 <= b < a."""
    out = []
    for a in range(1, N + 1):
        if N % a:
            continue
        d = N // a
        for b in range(d):
            out.append(((a, b), (0, d)))
    return out


def gauss_reduce(u, v):
    u = np.array(u, dtype=np.int64)
    v = np.array(v, dtype=np.int64)
    while True:
        if u @ u > v @ v:
            u, v = v, u
        q = int(np.rint((u @ v) / (u @ u)))
        v2 = v - q * u
        if v2 @ v2 >= v @ v:
            return u, v
        v = v2


def gauss_reduce_f(u, v):
    u = np.array(u, float)
    v = np.array(v, float)
    while True:
        if u @ u > v @ v:
            u, v = v, u
        q = np.rint((u @ v) / (u @ u))
        v2 = v - q * u
        if v2 @ v2 >= v @ v - 1e-9:
            return u, v
        v = v2


class Graph:
    """E_par as a graph over two base coordinates."""

    def __init__(self, star):
        self.star = star
        d = star.d
        if star.cls == "ab":
            self.base = [1, 3]
        else:
            self.base = [0, 1]
        self.other = [i for i in range(d) if i not in self.base]
        I = star.I  # (d-2) x d ; P_perp n = 0  <=>  I n = 0
        Ib = I[:, self.base]
        Io = I[:, self.other]
        # I_o n_o = - I_b n_b  -> n_o = -Io^{-1} Ib n_b
        self.lift = -np.linalg.solve(Io, Ib)  # (d-2) x 2

    def ideal(self, m):
        return self.lift @ np.asarray(m, float)

    def vector(self, m, lifts):
        d = self.star.d
        L = np.zeros(d, dtype=np.int64)
        L[self.base] = m
        L[self.other] = lifts
        return L


def enumerate_pool(cls, Vmin, Vmax, dmin, dmax, box=5, aspect=2.0, log=None,
                   per_bin=200, nbins=6, seed=2026, pure_sector=None, exclude_keys=()):
    """Complete enumeration with seeded reservoir sampling per log-delta bin.
    Cells with pure_sector fraction > 0.99 are all kept (separately). Returns
    (sample dict, pure dict, counts per bin, number of excluded-key hits)."""
    rng = np.random.default_rng(seed)
    edges = np.geomspace(dmin, dmax, nbins + 1)
    counts = np.zeros(nbins, int)
    reservoir = [dict() for _ in range(nbins)]
    pure = {}
    excl_hits = 0
    star = Star(cls)
    gr = Graph(star)
    E = star.E
    I = star.I
    # area per unit cell of the base lattice on E_par
    J = np.zeros((2, 2))
    for c, i in enumerate(gr.base):
        m = np.zeros(2)
        m[c] = 1
        L = np.zeros(star.d)
        L[gr.base] = m
        L[gr.other] = gr.ideal(m)
        J[:, c] = E @ L
    apv = 2 * (np.sqrt(2) - 1) if cls == "ab" else (((1 + 5**0.5) / 2) * np.sin(2 * np.pi / 5) + np.sin(np.pi / 5)) / ((1 + 5**0.5) / 2 + 1)
    detJ = abs(np.linalg.det(J))
    Nmin = int(np.floor(Vmin * apv / detJ * 0.9))
    Nmax = int(np.ceil(Vmax * apv / detJ * 1.1))
    print(f"[{cls}] detJ={detJ:.4f} area/vertex={apv:.4f} index range {Nmin}..{Nmax}", flush=True)
    offs = np.array(list(itertools.product(range(-box, box + 1), repeat=star.d - 2)))
    t0 = time.time()
    nlat = 0
    for N in range(Nmin, Nmax + 1):
        for (r1, r2) in sublattices_z2(N):
            nlat += 1
            m1, m2 = gauss_reduce(r1, r2)
            if (m2 @ m2) > (aspect * 1.3) ** 2 * (m1 @ m1):
                continue  # pre-filter: compact base lattice (exact physical filter below)
            cand = []
            for m in (m1, m2):
                ideal = gr.ideal(m)
                lifts = np.rint(ideal).astype(np.int64)[None, :] + offs
                Ls = np.zeros((len(lifts), star.d), dtype=np.int64)
                Ls[:, gr.base] = m
                Ls[:, gr.other] = lifts
                par = Ls @ E.T
                per = Ls @ I.T
                ratio = np.linalg.norm(per, axis=1) / np.linalg.norm(par, axis=1)
                keep = ratio <= np.sqrt(2) * dmax * 1.0001
                cand.append(Ls[keep])
            if len(cand[0]) == 0 or len(cand[1]) == 0:
                continue
            A = cand[0]
            B = cand[1]
            # all pairs
            ia, ib = np.meshgrid(np.arange(len(A)), np.arange(len(B)), indexing="ij")
            LA = A[ia.ravel()]
            LB = B[ib.ravel()]
            EL = np.stack([LA @ E.T, LB @ E.T], axis=2)  # n x 2 x 2 (columns)
            IL = np.stack([LA @ I.T, LB @ I.T], axis=2)  # n x (d-2) x 2
            det = EL[:, 0, 0] * EL[:, 1, 1] - EL[:, 0, 1] * EL[:, 1, 0]
            ok = np.abs(det) > 1e-9
            inv = np.zeros_like(EL)
            inv[ok, 0, 0] = EL[ok, 1, 1] / det[ok]
            inv[ok, 1, 1] = EL[ok, 0, 0] / det[ok]
            inv[ok, 0, 1] = -EL[ok, 0, 1] / det[ok]
            inv[ok, 1, 0] = -EL[ok, 1, 0] / det[ok]
            S = np.einsum("nij,njk->nik", IL, inv)
            delta = np.sqrt((S**2).sum(axis=(1, 2)) / 2)
            Vest = np.abs(det) / apv
            sel = ok & (delta >= dmin) & (delta <= dmax) & (Vest >= Vmin - 20) & (Vest <= Vmax + 20)
            for idx in np.nonzero(sel)[0]:
                L1, L2 = LA[idx], LB[idx]
                Lm = np.array([L1, L2]).T
                if not is_primitive(Lm):
                    continue
                t1, t2 = gauss_reduce_f(E @ L1, E @ L2)
                if np.linalg.norm(t2) > aspect * np.linalg.norm(t1):
                    continue
                key = lattice_key(star, L1, L2)
                if key in exclude_keys:
                    excl_hits += 1
                    continue
                Smat, dl, sec = star.phason(Lm)
                rec = {"L1": L1.tolist(), "L2": L2.tolist(), "delta": float(dl),
                       "Vest": float(Vest[idx]), "sectors": {k: float(v) for k, v in sec.items()}, "N": N}
                if pure_sector is not None and sec[pure_sector] > 0.99:
                    pure[key] = rec
                    continue
                b = min(int(np.searchsorted(edges, dl, side="right") - 1), nbins - 1)
                counts[b] += 1
                if len(reservoir[b]) < per_bin:
                    reservoir[b][key] = rec
                else:
                    j = rng.integers(0, counts[b])
                    if j < per_bin:
                        kold = list(reservoir[b].keys())[j]
                        del reservoir[b][kold]
                        reservoir[b][key] = rec
        if log and (N % 10 == 0):
            print(f"  N={N} lattices={nlat} counts={counts.tolist()} pure={len(pure)} excl={excl_hits} t={time.time()-t0:.0f}s", flush=True)
    sample = {}
    for r in reservoir:
        sample.update(r)
    return sample, pure, counts.tolist(), excl_hits


def covariates(cls, pool, seeds_offsets=(0, 1, 2), log=None):
    star = Star(cls)
    atl, _ = build_atlas(star, 110, make_gamma(star, 999), reduce=False)
    out = []
    for ci, (key, c) in enumerate(sorted(pool.items(), key=lambda kv: kv[1]["delta"])):
        rec = dict(c)
        rec["pool_index"] = ci
        Vs, n1, n2 = [], [], []
        for oi in seeds_offsets:
            seed = 1300 + 7 * (10000 + ci) + oi  # covariate offsets use a distinct index space
            cell = Cell(star, c["L1"], c["L2"], make_gamma(star, seed))
            ff = forbidden_fraction(cell, atl)
            Vs.append(cell.V)
            n1.append(ff[1])
            n2.append(ff[2])
        rec["V"] = int(Vs[0])
        rec["V_all"] = Vs
        rec["new1"] = float(np.mean(n1))
        rec["new2"] = float(np.mean(n2))
        rec["new2_all"] = n2
        out.append(rec)
        if log and ci % 20 == 0:
            print(f"  cov {ci}/{len(pool)} V={Vs} delta={c['delta']:.5f} new2={rec['new2']:.4f}", flush=True)
    return out


if __name__ == "__main__":
    cls = sys.argv[1]
    Vmin, Vmax = int(sys.argv[2]), int(sys.argv[3])
    per_bin = int(sys.argv[4]) if len(sys.argv) > 4 else 200
    star = Star(cls)
    excl = set()
    if cls == "ab":
        t1 = json.load(open("../isotropic_loss/isotropic_results.json"))
        for c in t1["fixedV"]["ab"]["cells"]:
            excl.add(lattice_key(star, c["L1"], c["L2"]))
    sample, pure, counts, excl_hits = enumerate_pool(cls, Vmin, Vmax, 0.004, 0.065, log=True, per_bin=per_bin,
                                                     pure_sector="B1" if cls == "ab" else None, exclude_keys=excl)
    print("sample", len(sample), "pure", len(pure), "counts per bin", counts, "fixedV hits", excl_hits, flush=True)
    json.dump({"sample": list(sample.values()), "pure": list(pure.values()), "counts": counts,
               "fixedV_hits": excl_hits, "V_window": [Vmin, Vmax]}, open(f"pool_{cls}_raw.json", "w"), indent=0)
    pool = dict(sample)
    pool.update(pure)
    cov = covariates(cls, pool, log=True)
    cov = [c for c in cov if Vmin <= c["V"] <= Vmax]
    json.dump(cov, open(f"pool_{cls}.json", "w"), indent=1)
    print("pool with covariates (V in window)", len(cov))

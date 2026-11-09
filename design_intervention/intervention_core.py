"""intervention sampler and census on (lifts, L1, L2), both classes. Parents ab_frame, penrose_frame, cell_tiling are imported, never edited.

Weight of a tiling T on the flip component: W(T) = exp(-beta N_f(T) - (kappa V / 2) (c(T) - c*)^2), N_f = number of vertices whose
radius-two patch is absent from the quasicrystal atlas, c = internal variance (cmode 1) or moved fraction (cmode 2), no restraint for cmode 0.
Proposal: same uniform vertex draw and same legality tests as ab_frame._mc / penrose_frame._mc, symmetric. Metropolis on dU keeps detailed balance w.r.t. W.
No random number is drawn unless dU > 0 (beta = kappa = 0 draws exactly the unbiased stream).
A patch is a 64-bit mask over the at most 60 lift differences within two unit steps. Membership is exact (C given the atlas)."""
from __future__ import annotations
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames', 'penrose', 'heldout_test', 'design', 'nonlinear', 'defects', 'phason_paths', 'independent_code', 'forbidden_patches')]  # shared modules
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
if os.name == 'nt':
    pass
import itertools, json, math, sys
from pathlib import Path
import numpy as np
import numba

HERE = Path(__file__).resolve().parent; M = HERE.parent; ROOT = M
import ab_frame as AB
import penrose_frame as PN
MODS = {'ab': AB, 'pen': PN}
KB = 11; OFF = 1 << (KB - 1)


def tables(d):
    """bit1[i], bit2[i, j] for unit steps i = 2k + (s < 0). -1 when the two steps cancel."""
    unit = np.zeros((2 * d, d), np.int64)
    for k in range(d):
        unit[2 * k, k] = 1; unit[2 * k + 1, k] = -1
    bit = {}
    for i in range(2 * d):
        bit[tuple(int(x) for x in unit[i])] = len(bit)
    bit1 = np.array([bit[tuple(int(x) for x in unit[i])] for i in range(2 * d)], np.int64)
    bit2 = -np.ones((2 * d, 2 * d), np.int64)
    for i in range(2 * d):
        for j in range(2 * d):
            t = tuple(int(x) for x in unit[i] + unit[j])
            if not any(t):
                continue
            if t not in bit:
                bit[t] = len(bit)
            bit2[i, j] = bit[t]
    assert len(bit) <= 62
    return unit, bit1, bit2, bit


def e_to_f(de):
    """heldout Penrose star e_j (2 pi j / 5) to penrose_frame basis f = (e0, -e3, e1, -e4, e2)."""
    return (de[0], -de[3], de[1], -de[4], de[2])


def build_atlas_masks(cls, R=110):
    """Masks of the heldout raw-key atlas (radius 110, orientation-specific), in the basis of ab_frame / penrose_frame."""
    import cell_tiling as TT
    star = TT.Star(cls)
    atl, _ = TT.build_atlas(star, R, TT.make_gamma(star, 999), radii=(2,), reduce=False)
    _, _, _, bit = tables(star.d)
    out = set()
    for key in atl[2]:
        m = 0
        for de in key:
            df = tuple(de) if cls == 'ab' else e_to_f(de)
            m |= 1 << bit[tuple(int(x) for x in df)]
        out.add(m)
    return sorted(out)


def load_atlas(cls):
    f = HERE / f'atlas_{cls}.json'
    if not f.exists():
        f.write_text(json.dumps(build_atlas_masks(cls)))
    return np.array(json.loads(f.read_text()), np.int64)


@numba.njit(cache=True)
def _key(r, d):
    k = 0
    for c in range(d):
        k = (k << KB) | (r[c] + OFF)
    return k


@numba.njit(cache=True)
def _reduce(n, L1, L2, adj, det, out, d):
    r1 = 0; r2 = 0
    for c in range(d):
        r1 += n[c] * L1[c]; r2 += n[c] * L2[c]
    a = (adj[0] * r1 + adj[1] * r2) // det
    b = (adj[2] * r1 + adj[3] * r2) // det
    for c in range(d):
        out[c] = n[c] - a * L1[c] - b * L2[c]


@numba.njit(cache=True)
def _hfind(keys, k):
    mask = keys.shape[0] - 1
    h = ((k * 0x9E3779B97F4A7C15) >> 20) & mask
    while True:
        kk = keys[h]
        if kk == k:
            return h
        if kk == -1:
            return -1
        h = (h + 1) & mask


@numba.njit(cache=True)
def _hins(keys, vals, k, v):
    mask = keys.shape[0] - 1
    h = ((k * 0x9E3779B97F4A7C15) >> 20) & mask
    while keys[h] != -1 and keys[h] != -2:
        h = (h + 1) & mask
    keys[h] = k; vals[h] = v


@numba.njit(cache=True)
def _idx(keys, vals, n, L1, L2, adj, det, tmp, d):
    _reduce(n, L1, L2, adj, det, tmp, d)
    h = _hfind(keys, _key(tmp, d))
    return -1 if h < 0 else vals[h]


@numba.njit(cache=True)
def _sig(r, keys, vals, L1, L2, adj, det, unit, bit1, bit2, d, a1, a2, tmp):
    m = 0
    nd = 2 * d
    for i in range(nd):
        for c in range(d):
            a1[c] = r[c] + unit[i, c]
        if _idx(keys, vals, a1, L1, L2, adj, det, tmp, d) < 0:
            continue
        m |= (1 << bit1[i])
        for j in range(nd):
            if bit2[i, j] < 0:
                continue
            for c in range(d):
                a2[c] = a1[c] + unit[j, c]
            if _idx(keys, vals, a2, L1, L2, adj, det, tmp, d) >= 0:
                m |= (1 << bit2[i, j])
    return m


@numba.njit(cache=True)
def _census(lifts, keys, vals, L1, L2, adj, det, unit, bit1, bit2, atlas, flags):
    d = lifts.shape[1]
    tmp = np.zeros(d, np.int64); a1 = np.zeros(d, np.int64); a2 = np.zeros(d, np.int64)
    nf = 0
    for v in range(lifts.shape[0]):
        m = _sig(lifts[v], keys, vals, L1, L2, adj, det, unit, bit1, bit2, d, a1, a2, tmp)
        i = np.searchsorted(atlas, m)
        f = 0 if (i < atlas.shape[0] and atlas[i] == m) else 1
        flags[v] = f; nf += f
    return nf


@numba.njit(cache=True)
def _mc(lifts, keys, vals, ckeys, flags, L1, L2, adj, det, Px, Py, perp, nattempt, seed, unit, bit1, bit2, atlas, beta, kappa, cstar, cmode, maxacc, st, stamp, stop_var, allow):
    """st[0] N_f, st[1] sum |y|^2, st[2:2+di] sum y, st[2+di] moved count. Returns accepted, legal proposals, Metropolis rejections.
    stop_var > 0: stop at the first accepted state with variance >= stop_var (first passage). stamp: int64 work array, size V."""
    np.random.seed(seed)
    d = lifts.shape[1]; V = lifts.shape[0]; di = perp.shape[0]
    tmp = np.zeros(d, np.int64); t2 = np.zeros(d, np.int64); a1 = np.zeros(d, np.int64); a2 = np.zeros(d, np.int64)
    r = np.zeros(d, np.int64); new = np.zeros(d, np.int64)
    ds = np.zeros((3, d), np.int64)
    aff = np.zeros(2048, np.int64); newf = np.zeros(2048, np.int64)
    census = (beta != 0.0)
    acc = 0; legal = 0; rej = 0
    tick = int(stamp.max()) + 1
    for it in range(nattempt):
        v = np.random.randint(V)
        if allow[v] == 0:
            continue
        for c in range(d):
            r[c] = lifts[v, c]
        nd = 0
        for k in range(d):
            for s in (1, -1):
                for c in range(d):
                    tmp[c] = r[c]
                tmp[k] += s
                if _idx(keys, vals, tmp, L1, L2, adj, det, t2, d) >= 0:
                    if nd < 3:
                        for c in range(d):
                            ds[nd, c] = 0
                        ds[nd, k] = s
                    nd += 1
        if nd != 3:
            continue
        cr = np.zeros(3)
        for a in range(3):
            b = (a + 1) % 3
            ax = 0.0; ay = 0.0; bx = 0.0; by = 0.0
            for c in range(d):
                ax += Px[c] * ds[a, c]; ay += Py[c] * ds[a, c]
                bx += Px[c] * ds[b, c]; by += Py[c] * ds[b, c]
            cr[a] = ax * by - ay * bx
        if not ((cr[0] > 1e-9 and cr[1] > 1e-9 and cr[2] > 1e-9) or (cr[0] < -1e-9 and cr[1] < -1e-9 and cr[2] < -1e-9)):
            continue
        ok = True
        for a in range(3):
            for b in range(a + 1, 3):
                for c in range(d):
                    tmp[c] = r[c] + ds[a, c] + ds[b, c]
                if _idx(keys, vals, tmp, L1, L2, adj, det, t2, d) < 0:
                    ok = False
        if not ok:
            continue
        for c in range(d):
            tmp[c] = r[c] + ds[0, c] + ds[1, c] + ds[2, c]
        _reduce(tmp, L1, L2, adj, det, new, d)
        if _hfind(keys, _key(new, d)) >= 0:
            continue
        cnt = 0
        for k in range(d):
            for s in (1, -1):
                for c in range(d):
                    tmp[c] = new[c]
                tmp[k] += s
                if _idx(keys, vals, tmp, L1, L2, adj, det, t2, d) >= 0:
                    cnt += 1
        if cnt != 3:
            continue
        legal += 1
        # ---- legal flip. Internal coordinate change, O(1).
        dy2 = 0.0; dsum = np.zeros(di)
        for q in range(di):
            yo = 0.0; yn = 0.0
            for c in range(d):
                yo += perp[q, c] * r[c]; yn += perp[q, c] * new[c]
            dsum[q] = yn - yo; dy2 += yn * yn - yo * yo
        s2n = st[1] + dy2
        vold = st[1] / V
        vnew = s2n / V
        for q in range(di):
            vold -= (st[2 + q] / V) ** 2
            vnew -= ((st[2 + q] + dsum[q]) / V) ** 2
        dm = 0
        if _hfind(ckeys, _key(r, d)) < 0:
            dm -= 1
        if _hfind(ckeys, _key(new, d)) < 0:
            dm += 1
        dU = 0.0
        if kappa > 0.0:
            if cmode == 1:
                dU += 0.5 * kappa * V * ((vnew - cstar) ** 2 - (vold - cstar) ** 2)
            elif cmode == 2:
                mo = st[2 + di] / V; mn = (st[2 + di] + dm) / V
                dU += 0.5 * kappa * V * ((mn - cstar) ** 2 - (mo - cstar) ** 2)
        na = 0; dE = 0
        h0 = _hfind(keys, _key(r, d))
        if census:
            # affected roots: v, its old corners and new corners, and all their neighbours
            tick += 1
            aff[0] = v; na = 1; stamp[v] = tick
            for a in range(6):
                for c in range(d):
                    if a < 3:
                        a1[c] = r[c] + ds[a, c]
                    else:
                        a1[c] = new[c] - ds[a - 3, c]
                w = _idx(keys, vals, a1, L1, L2, adj, det, t2, d)
                if w < 0:
                    continue
                if stamp[w] != tick:
                    stamp[w] = tick; aff[na] = w; na += 1
                for c in range(d):
                    a2[c] = lifts[w, c]
                for j in range(2 * d):
                    for c in range(d):
                        tmp[c] = a2[c] + unit[j, c]
                    w2 = _idx(keys, vals, tmp, L1, L2, adj, det, t2, d)
                    if w2 >= 0 and stamp[w2] != tick:
                        stamp[w2] = tick; aff[na] = w2; na += 1
            eold = 0
            for a in range(na):
                eold += flags[aff[a]]
            # tentative apply
            keys[h0] = -2
            _hins(keys, vals, _key(new, d), v)
            for c in range(d):
                lifts[v, c] = new[c]
            enew = 0
            for a in range(na):
                m = _sig(lifts[aff[a]], keys, vals, L1, L2, adj, det, unit, bit1, bit2, d, a1, a2, t2)
                i = np.searchsorted(atlas, m)
                f = 0 if (i < atlas.shape[0] and atlas[i] == m) else 1
                newf[a] = f; enew += f
            dE = enew - eold
            dU += beta * dE
        if dU > 0.0 and np.random.random() > math.exp(-dU):
            rej += 1
            if census:
                h1 = _hfind(keys, _key(new, d))
                keys[h1] = -2
                _hins(keys, vals, _key(r, d), v)
                for c in range(d):
                    lifts[v, c] = r[c]
            continue
        if census:
            for a in range(na):
                flags[aff[a]] = newf[a]
            st[0] += dE
        else:
            keys[h0] = -2
            _hins(keys, vals, _key(new, d), v)
            for c in range(d):
                lifts[v, c] = new[c]
        st[1] = s2n
        for q in range(di):
            st[2 + q] += dsum[q]
        st[2 + di] += dm
        acc += 1
        if maxacc > 0 and acc >= maxacc:
            break
        if stop_var > 0.0 and vnew >= stop_var:
            break
    return acc, legal, rej


class Sampler:
    """Persistent state on one torus tiling. ref_lifts = the canonical lifts, reference of the moved fraction."""

    def __init__(self, cls, L1, L2, lifts, perp, ref_lifts, atlas=None):
        self.cls = cls; self.C = MODS[cls]; self.L1 = np.array(L1, np.int64); self.L2 = np.array(L2, np.int64)
        self.lifts = np.array(lifts, np.int64).copy(); self.perp = np.array(perp, float); self.V = len(self.lifts); self.d = self.lifts.shape[1]
        self.ref = np.array(ref_lifts, np.int64)
        self.atlas = load_atlas(cls) if atlas is None else atlas
        self.unit, self.bit1, self.bit2, _ = tables(self.d)
        red = self.C.Reducer(self.L1, self.L2)
        self.adj = np.array(red.adj, np.int64); self.det = np.int64(red.det)
        size = 1 << int(math.ceil(math.log2(8 * self.V)))
        self.keys = -np.ones(size, np.int64); self.vals = np.zeros(size, np.int64)
        self.ckeys = -np.ones(size, np.int64); cv = np.zeros(size, np.int64)
        for i, r in enumerate(self.lifts):
            _hins(self.keys, self.vals, _key(r, self.d), i)
        for i, r in enumerate(self.ref):
            _hins(self.ckeys, cv, _key(r, self.d), i)
        self.flags = np.zeros(self.V, np.int64); self.stamp = np.zeros(self.V, np.int64)
        self.allow = np.ones(self.V, np.int64)   # gate use only: restrict flips to a vertex-index set (symmetric, indices are kept by a flip)
        self.st = np.zeros(3 + self.perp.shape[0])
        self.refresh()
        self.n_acc = 0; self.n_legal = 0; self.n_rej = 0; self.n_att = 0

    def census(self):
        return int(_census(self.lifts, self.keys, self.vals, self.L1, self.L2, self.adj, self.det, self.unit, self.bit1, self.bit2, self.atlas, self.flags))

    def refresh(self):
        y = self.lifts @ self.perp.T
        self.st[0] = self.census(); self.st[1] = float((y ** 2).sum()); self.st[2:2 + y.shape[1]] = y.sum(0)
        reft = {tuple(r) for r in self.ref.tolist()}
        self.st[2 + y.shape[1]] = sum(tuple(r) not in reft for r in self.lifts.tolist())

    @property
    def phi2(self):
        return self.st[0] / self.V

    @property
    def var(self):
        y = self.lifts @ self.perp.T
        return float(np.mean(np.sum((y - y.mean(0)) ** 2, 1)))

    @property
    def moved(self):
        return self.st[-1] / self.V

    def run(self, nattempt, seed, beta=0.0, kappa=0.0, cstar=0.0, cmode=0, maxacc=0, stop_var=0.0):
        C = self.C
        a, l, rj = _mc(self.lifts, self.keys, self.vals, self.ckeys, self.flags, self.L1, self.L2, self.adj, self.det, C.P[0].copy(), C.P[1].copy(), self.perp,
                       int(nattempt), int(seed), self.unit, self.bit1, self.bit2, self.atlas, float(beta), float(kappa), float(cstar), int(cmode), int(maxacc), self.st, self.stamp, float(stop_var), self.allow)
        self.n_acc += a; self.n_legal += l; self.n_rej += rj; self.n_att += int(nattempt)
        return a, l, rj

    def tracked_ok(self):
        """Tracked N_f and flags against a full recount (only meaningful when beta != 0 throughout)."""
        nf = self.st[0]; flags = self.flags.copy()
        nnew = self.census()
        return nf == nnew and np.array_equal(flags, self.flags)


def cell(cls, name, off=None):
    """Canonical dict for a named cell. AB name = (p, q), Pell B1 family, seed off (default 7, as design/defects). Penrose name = n (fib_approx), default offset."""
    C = MODS[cls]
    if cls == 'ab':
        L = C.ab_approx(*name)
        return C.canonical(L[0], L[1], seed=7 if off is None else off)
    L = C.fib_approx(name)
    return C.canonical(L[0], L[1], offset=off)

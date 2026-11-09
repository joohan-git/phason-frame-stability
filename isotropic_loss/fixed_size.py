"""is the isotropic c0 deficit a |delta| effect in BOTH classes (AB C8, Penrose C10)?
Along approximant sequences |delta|*V is constant, so |delta| and 1/V are confounded. Here: PRIMITIVE
cells (gcd of minors = 1, i.e. not supercells) at fixed V in a window, |delta| spread over a decade,
several window offsets each.  Reuses main/ab_frame.py (AB) and penrose/penrose_frame.py (Penrose) unchanged.

usage: python fixed_size.py catalog            -> fixed_size_catalog.json
       python fixed_size.py run <cls> <i0> <i1> -> fixed_size_<cls>_<i0>.json   (cls = ab | pen)
All numbers P.  CPU only.
"""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames', 'phason_paths', 'penrose', 'symmetry')]  # shared modules
import os
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='-1')
import json, math, sys, time
from functools import reduce
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'phason_paths'))
sys.path.insert(0, str(ROOT / 'penrose'))
import ab_frame as AB  # noqa: E402
import penrose_frame as PN   # noqa: E402

VMIN, VMAX = 900, 1800


def gcd_list(xs):
    return reduce(math.gcd, [abs(int(x)) for x in xs])


def catalog(seed=1, ntry=400000):
    rng = np.random.default_rng(seed)
    out = {}
    for cls in ('ab', 'pen'):
        M = AB if cls == 'ab' else PN
        nd = 4 if cls == 'ab' else 5
        scale = 0.5 if cls == 'ab' else 0.4
        seen = {}
        for _ in range(ntry):
            r1 = rng.uniform(8, 32); r2 = rng.uniform(8, 32)
            t1 = rng.uniform(0, 2 * math.pi); t2 = t1 + rng.uniform(0.35 * math.pi, 0.65 * math.pi)
            x1 = r1 * np.array([math.cos(t1), math.sin(t1)]); x2 = r2 * np.array([math.cos(t2), math.sin(t2)])
            L1 = np.rint(M.P.T @ x1 * scale + rng.normal(0, 0.35, nd)).astype(np.int64)
            L2 = np.rint(M.P.T @ x2 * scale + rng.normal(0, 0.35, nd)).astype(np.int64)
            if cls == 'pen':
                alt = np.array([1, -1, 1, -1, 1])
                # enforce D.L = 0 (stay in the Penrose LI layering, v = 0) by a minimal correction
                for L in (L1, L2):
                    s = int(alt @ L)
                    k = rng.integers(0, 5)
                    L[k] -= s * alt[k]
            m = np.array(list(M.minors(L1, L2).values()))
            if np.all(m <= 0):
                L1, L2 = L2, L1; m = -m
            if not np.all(m > 0):
                continue
            V = int(m.sum())
            if not (VMIN <= V <= VMAX) or gcd_list(m) != 1:
                continue
            Mt = M.phason_matrix(L1, L2)
            if cls == 'ab':
                a, b = M.harmonic_parts(Mt); v = 0j
            else:
                a, b, v = M.phason_parts(Mt)
                if abs(v) > 1e-12:
                    continue
            d = math.sqrt(abs(a) ** 2 + abs(b) ** 2)
            if d > 0.08:
                continue
            key = tuple(sorted(m.tolist())) + (round(d, 7),)
            if key in seen:
                continue
            seen[key] = dict(L1=L1.tolist(), L2=L2.tolist(), V=V, delta=d, a=[a.real, a.imag], b=[b.real, b.imag],
                             frac_m4=(abs(b) ** 2 if cls == 'ab' else abs(a) ** 2) / d ** 2)
        cand = sorted(seen.values(), key=lambda r: r['delta'])
        # pick ~10 cells: log-spread delta, alternate m4-rich / m2-rich
        targets = np.geomspace(max(cand[0]['delta'], 0.003), 0.06, 10)
        pick = []
        for i, tg in enumerate(targets):
            want_m4 = (i % 2 == 0)
            pool = [c for c in cand if (c['frac_m4'] > 0.5) == want_m4 and c not in pick] or [c for c in cand if c not in pick]
            pick.append(min(pool, key=lambda c: abs(math.log(c['delta'] / tg))))
        out[cls] = dict(n_candidates=len(cand), pick=pick)
        print(cls, len(cand), [(p['V'], round(p['delta'], 4), round(p['frac_m4'], 2)) for p in pick], flush=True)
    (HERE / 'fixed_size_catalog.json').write_text(json.dumps(out, indent=1), encoding='utf-8')


def pen_offset(rng):
    o = rng.uniform(-.5, .5, 5)
    alt = np.array([1, -1, 1, -1, 1])
    o += (0.5 - alt @ o) / 5 * alt
    return o


def evaluate(cls, L1, L2, off):
    t0 = time.time()
    if cls == 'ab':
        tl = AB.canonical(L1, L2, seed=off)
        assert AB.check_tiling(tl['lifts'], tl['L1'], tl['L2']) == 0
        f = AB.Frame(tl['lifts'], tl['L1'], tl['L2'], sub=2)
        f.set_load('stress'); phis, ext, tab = f.longwave(nphi=8); h = AB.harmonics(phis, ext)
    else:
        tl = PN.canonical(L1, L2, offset=off)
        assert PN.check_tiling(tl['lifts'], tl['L1'], tl['L2']) == 0
        f = PN.Frame(tl['lifts'], tl['L1'], tl['L2'], sub=2)
        f.set_load('stress'); phis, ext, tab = f.longwave(nphi=8); h = PN.harmonics(phis, ext)
    kg = float(f.lam((0, 0))); kh = float(f.lam((0.5, 0.5)))
    return dict(V=len(tl['lifts']), c0=h['c0'], h4=h['h4'], h2=h['h2'], pc_lw_min=float(ext.min()),
                pc_gamma=kg, pc_halfhalf=kh, pc_phi=ext.tolist(), seconds=time.time() - t0)


def run(cls, i0, i1, noff=3):
    cat = json.loads((HERE / 'fixed_size_catalog.json').read_text(encoding='utf-8'))[cls]['pick']
    out = HERE / f'fixed_size_{cls}_{i0}.json'
    rows = []
    for i in range(i0, min(i1, len(cat))):
        c = cat[i]
        rng = np.random.default_rng(100 + i)
        offs = [7, 11, 13][:noff] if cls == 'ab' else [pen_offset(rng) for _ in range(noff)]
        res = []
        for o in offs:
            r = evaluate(cls, c['L1'], c['L2'], o)
            r['offset'] = o if cls == 'ab' else list(map(float, o))
            res.append(r)
            print(json.dumps(dict(cls=cls, i=i, V=r['V'], delta=c['delta'], c0=r['c0'], gam=r['pc_gamma'], s=round(r['seconds'], 1))), flush=True)
        c0s = np.array([r['c0'] for r in res])
        rows.append(dict(c, idx=i, runs=res, c0_mean=float(c0s.mean()), c0_sd=float(c0s.std(ddof=1)) if len(c0s) > 1 else None,
                         h4_mean=float(np.mean([math.hypot(*r['h4']) for r in res]))))
        out.write_text(json.dumps(dict(status='P', cls=cls, rows=rows), indent=1), encoding='utf-8')


if __name__ == '__main__':
    if sys.argv[1] == 'catalog':
        catalog()
    else:
        run(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))

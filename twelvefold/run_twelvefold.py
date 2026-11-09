"""D12 (Niizeki-Gaehler-type canonical A2+A2 tiling) swap test registered before the runs.

Families (same L1 for both pair types, so amplitudes and side lengths are matched):
  L1 = u^m * w,  L2 = zeta^3 L1  ->  'C4' cell, pure conformal strain alpha = a + i b   (P1 pair, m = 4)
                 L2 = zeta^2 L1  ->  'C6' cell, pure anticonformal  beta  = c + i d   (P2 pair, m = 6)
  family A: w*u^m = (1+zeta)^n, n = 2..6 (|delta| = (2-sqrt3)^(n/1)... = 0.268^n; arg alpha = 60n deg,
            arg beta = 90n deg, i.e. beta alternates pure c (B1) and pure d (B2)).
  family G: w = 2 + zeta, m = 1, 2 (generic phases: arg alpha = 13.9 deg, arg beta = 33.7 deg).
Mechanics: twelvefold_core.Frame (rigid joints, r = 0.03, stress protocol), nphi = 12 over [0, pi), Richardson DQ = 0.002, 0.001;
harmonics m = 0, 2, 4, 6.  6 generic offsets per cell (seeds 1..6).  sub = 2; sub = 4 at selected points.

PRE-DECLARED STATISTICS (written before any production number exists, 2026-11-08):
  primary   H_hull = |< h4 >_offsets| / |delta|      (offset / hull average of the complex h4)
  secondary H_abs  = < |h4| >_offsets / |delta|
  P1 (alpha, C4 cells) PASSES if the primary ratio does not go to 0: log-log slope of |<h4>| vs |alpha|
     over the family-A points n = 3..6 lies in [0.8, 1.2] (linear), and H_hull(n=6) > 0.3 H_hull(n=3).
  P2 (beta, C6 cells) PASSES if |<h4>|/|beta| -> 0: either slope >= 1.5, or |<h4>| below the numerical floor
     (1e-3 x the P1 |<h4>| at the same |delta|) at every size.  FAILS if the ratio tends to a nonzero constant
     (slope in [0.8, 1.2] with H_hull(C6) > 0.1 H_hull(C4) at the same size).
  P3 PASSES iff P1 and P2 both pass (AB: first-order pair = (c,d); here first-order pair must be (a,b)).
  Expected by symmetry for C6 cells: lambda(phi) 60-deg periodic at symmetric offsets, h4 = 0 exactly; the
  non-trivial content is the generic-offset residual.
Output: runs.jsonl (one row per job, resumable).
"""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames', 'phason_paths')]  # shared modules
import json, math, os, sys, time
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import twelvefold_core as D

OUT = Path(__file__).with_name('runs.jsonl')
U1 = D.GEN + np.eye(4, dtype=np.int64)  # 1 + zeta


def cell(fam, n, kind):
    if fam == 'A':
        L1 = np.linalg.matrix_power(U1, n) @ np.array([1, 0, 0, 0], np.int64)
    else:  # G
        L1 = np.linalg.matrix_power(D.UNIT, n) @ np.array([2, 1, 0, 0], np.int64)
    L2 = np.linalg.matrix_power(D.GEN, 3 if kind == 'C4' else 2) @ L1
    return L1, L2


def offset_of(seed):
    rng = np.random.default_rng(1000 + seed)
    return rng.uniform(-1.5, 1.5, 4)


def job(spec):
    fam, n, kind, seed, sub = spec
    t0 = time.time()
    L1, L2 = cell(fam, n, kind)
    tl = D.tiling(L1, L2, offset=offset_of(seed))
    v = D.validate(tl)
    f = D.Frame(tl, sub=sub); f.set_load()
    phis, ext, tab = f.longwave(nphi=12)
    h = D.harmonics(phis, ext)
    al, be = D.strain_parts(tl['M'])
    lg = f.lam((0, 0))
    return dict(fam=fam, n=n, kind=kind, seed=seed, sub=sub, L1=L1.tolist(), L2=L2.tolist(), V=v['V'], E=v['E'],
                tiles=v['kinds'], euler=v['euler'], window_margin=v['window_margin'], facet_margin=v['facet_margin'],
                alpha=[al.real, al.imag], beta=[be.real, be.imag], pressure_check=f.pressure, resid=f.resid,
                pc_phi=ext.tolist(), pc_phi_dq002=tab[0.002].tolist(), pc_phi_dq001=tab[0.001].tolist(),
                lam_gamma=lg, pc_min=float(ext.min()), harm=h, seconds=time.time() - t0)


def plan():
    P = []
    for n in (2, 3, 4, 5, 6):
        for kind in ('C4', 'C6'):
            for s in range(1, 7):
                P.append(('A', n, kind, s, 2))
    for n in (1, 2):
        for kind in ('C4', 'C6'):
            for s in range(1, 7):
                P.append(('G', n, kind, s, 2))
    for n in (3, 4, 5):
        for kind in ('C4', 'C6'):
            for s in (1, 2):
                P.append(('A', n, kind, s, 4))
    return P


if __name__ == '__main__':
    from multiprocessing import Pool
    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            r = json.loads(line); done.add((r['fam'], r['n'], r['kind'], r['seed'], r['sub']))
    todo = [p for p in plan() if p not in done]
    # big jobs first for load balance
    todo.sort(key=lambda p: -(p[1] * (2 if p[0] == 'G' else 1) + (1 if p[4] == 4 else 0)))
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    with Pool(nproc) as pool:
        for r in pool.imap_unordered(job, todo):
            with OUT.open('a') as fh:
                fh.write(json.dumps(r) + '\n')
            hh = r['harm']
            print(r['fam'], r['n'], r['kind'], r['seed'], r['sub'], r['V'], 'pcmin %.6f c0 %.6f |h4| %.3e |h6| %.1e  %.0fs' % (
                r['pc_min'], hh['c0'], math.hypot(*hh['h4']), math.hypot(*hh['h6']), r['seconds']), flush=True)

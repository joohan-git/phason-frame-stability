"""Same-slope hexagon-flip disorder on the Penrose Fibonacci approximant n=6 (V=4414,
|delta|=9.9e-4), stress protocol, 2 el./edge.  p_c = min(4 long-wave dirs (Richardson), Gamma, (1/2,1/2)).
Configs: canonical; uniform flips with k accepted moves (k = 1..3000) and equilibrated, seeds 1-3;
disk-confined flips (clusters), radii 4/7/10, seeds 1-2.  -> penrose_disorder.json (P)"""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames',)]  # shared modules
import os
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='-1')
import json, math, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import penrose_frame as C
N_FIB = int(sys.argv[1]) if len(sys.argv) > 1 else 6
OUT = Path(__file__).with_name('penrose_disorder.json' if N_FIB == 6 else f'penrose_disorder_fib{N_FIB}.json')


def mech(lifts, tl):
    t0 = time.time()
    L1, L2 = tl['L1'], tl['L2']
    f = C.Frame(lifts, L1, L2, sub=2); f.set_load('stress')
    phis, ext, _ = f.longwave(nphi=4)
    lg, vg = f.lam((0, 0), vectors=True)
    lhh = f.lam((0.5, 0.5))
    w = np.sum(np.abs(vg.reshape(-1, 3)[:, :2]) ** 2, 1); w /= w.sum()
    V = len(lifts)
    wv = w[:V].copy()  # vertex nodes; add edge-midpoint nodes to their first vertex
    edges = C.edges_from_lifts(lifts, L1, L2)
    np.add.at(wv, edges[:, 0], w[V:V + len(edges)])
    ex = C.window_excess(lifts, tl)
    out = ex > 1e-9
    return dict(pc_lw4=ext.tolist(), pc_lw_min=float(ext.min()), pc_gamma=float(lg), pc_hh=float(lhh),
                pc=float(min(ext.min(), lg, lhh)), mode_PR=float(1 / (len(w) * np.sum(w ** 2))),
                frac_outside_W=float(out.mean()), mean_excess=float(np.mean(np.maximum(ex, 0))),
                max_excess=float(ex.max()), mode_weight_outside=float(wv[out].sum()),
                mode_weighted_excess=float(np.sum(wv * np.maximum(ex, 0))), seconds=time.time() - t0)


def main():
    tl = C.canonical(*C.fib_approx(N_FIB)); L1, L2 = tl['L1'], tl['L2']
    ref = {tuple(r) for r in tl['lifts']}
    cell = np.array([C.P @ L1, C.P @ L2]); ctr = cell.sum(0) / 2
    rows = []

    def add(name, lifts, **kw):
        assert C.check_tiling(lifts, L1, L2) == 0
        r = dict(name=name, moved_fraction=float(np.mean([tuple(x) not in ref for x in lifts])), **kw, **mech(lifts, tl))
        rows.append(r)
        print(json.dumps({k: r[k] for k in ['name', 'moved_fraction', 'mean_excess', 'pc', 'pc_lw_min', 'pc_gamma', 'pc_hh', 'mode_PR', 'seconds']}), flush=True)
        OUT.write_text(json.dumps(dict(system=f'Penrose Fibonacci approximant n={N_FIB}, V={len(tl["lifts"])}, same slope (period lifts fixed)',
                                       units='p_c/(EA lambda_E), stress protocol, 2 el./edge', status='P', rows=rows), indent=1), encoding='utf-8')

    add('canonical', tl['lifts'], k=0, seed=None)
    V = len(tl['lifts'])
    for seed in (1, 2, 3):
        for k in (1, 3, 10, 30, 100, 300, 1000, 3000, -1):
            if k > 0:
                l, acc = C.flip_mc(tl['lifts'], L1, L2, 200 * k + 20000, seed * 1000 + k, maxacc=k)
            else:
                l, acc = C.flip_mc(tl['lifts'], L1, L2, 3000 * V, seed * 1000 + 7)
            add(f'uniform_k{k if k > 0 else "eq"}_s{seed}', l, k=k if k > 0 else 'eq', seed=seed, accepted=acc)
    for R in (4.0, 7.0, 10.0):
        for seed in (1, 2):
            l, acc = C.flip_mc(tl['lifts'], L1, L2, 2_000_000, 50 + seed + int(R), center=ctr, rmove=R)
            add(f'disk_R{R:g}_s{seed}', l, k='disk', seed=seed, accepted=acc, R=R)


if __name__ == '__main__':
    main()

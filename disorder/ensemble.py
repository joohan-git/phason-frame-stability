"""random-tiling drift lambda_c(L) versus model-set approximants.

One configuration per call. A configuration is an AB Pell approximant p/q, optionally an m x m
supercell, either canonical (model-set control) or flip-MC randomised at the same period
(uniform measure on the flip-connected component of the canonical tiling on the torus, i.e.
random tilings at mean slope = approximant slope -> s_QC).

Mechanics: ab_frame.Frame, stress protocol (unit hydrostatic stress), 2D Euler-Bernoulli beams,
`--sub` elements/edge; thresholds p_c/(EA lambda_E):
  long-wave: nphi directions, Richardson DQ -> 0 from DQ = 0.002, 0.001;
  finite k : Gamma of the (super)cell (all folded k), (1/2,0), (0,1/2), (1/2,1/2).
Sparse LU (SuperLU COLAMD) + ARPACK on K^{-1} H.

Usage (examples)
  python ensemble.py --p 12 --q 17 --mode canonical
  python ensemble.py --p 12 --q 17 --m 2 --mode random --sweeps 200 --seed 3
  python ensemble.py --array-index <i> --plan plan.json
Writes <outdir>/drift_<tag>.json (numbers status P until cross-checked).
Requires numpy, scipy; numba strongly recommended for the MC (pure-Python fallback is ~100x slower).
Set OMP_NUM_THREADS / OPENBLAS_NUM_THREADS to the cores requested (1-2 is optimal for SuperLU).
"""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames',)]  # shared modules
import argparse, json, math, os, sys, time
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ab_frame as C

C_L = 0.1  # sweeps per (cell length)^2; at V = 1393 (L = 34) the internal-space variance plateaus within ~60 sweeps


def peak_mem_mb():
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except Exception:
        try:
            import psutil
            mi = psutil.Process().memory_info()
            return getattr(mi, 'peak_wset', mi.rss) / 2 ** 20
        except Exception:
            return None


def default_plan():
    """Sizes ~1e3..1e5 vertices; Pell approximants and supercells (off-inflation sizes)."""
    sizes = [(12, 17, 1), (12, 17, 2), (29, 41, 1), (12, 17, 3), (29, 41, 2), (70, 99, 1), (29, 41, 3)]
    plan = []
    for p, q, m in sizes:
        plan.append(dict(p=p, q=q, m=m, mode='canonical', seed=0))
        for seed in range(1, 9):
            plan.append(dict(p=p, q=q, m=m, mode='random', seed=seed))
    return plan


def run(p, q, m=1, mode='random', seed=1, sweeps=200, sub=2, nphi=8, outdir='.'):
    T0 = time.time()
    L1, L2 = C.ab_approx(p, q, m)
    tl = C.canonical(L1, L2)
    lifts = tl['lifts']; V = len(lifts)
    t_gen = time.time() - T0
    info = dict(p=p, q=q, m=m, mode=mode, seed=seed, V=V, sub=sub, nphi=nphi,
                cell_length=float(np.linalg.norm(C.P @ np.array(tl['L1'], float))))
    if mode == 'random':
        # Equilibration: chunks of sweeps; target sweeps = max(sweeps, c_L * cell_length^2)
        # (diffusive phason relaxation), plus a plateau test on the internal-space variance.
        t1 = time.time()
        target = max(float(sweeps), C_L * info['cell_length'] ** 2)
        chunk = max(1.0, target / 10)
        trace = []; done = 0.0; acc = 0; k = 0
        while True:
            lifts, a = C.flip_mc(lifts, tl['L1'], tl['L2'], int(chunk * V), seed * 100003 + k)
            acc += a; done += chunk; k += 1
            pv = float(np.mean(np.sum((lifts @ tl['perp'].T - (lifts @ tl['perp'].T).mean(0)) ** 2, 1)))
            trace.append([done, pv])
            if done >= target and len(trace) >= 9:
                v = np.array([x[1] for x in trace])
                a_, b_ = v[-6:-3].mean(), v[-3:].mean()
                if abs(b_ - a_) / max(a_, 1e-12) < 0.10:   # equilibrium fluctuations are ~ +-15% per chunk at V ~ 1e3
                    break
            if done >= 3 * target:
                break
        info.update(sweeps_done=done, sweeps_target=target, equil_trace=trace, accepted=acc,
                    equilibrated=bool(done < 3 * target), mc_seconds=time.time() - t1, numba=C.HAVE_NUMBA)
        info['invalid_vertices'] = int(C.check_tiling(lifts, tl['L1'], tl['L2'])) if V <= 60000 else 'skipped'
    st = C.phason_stats(lifts, tl['L1'], tl['L2'], tl['perp'], tl['lifts'])
    info.update(st)
    t2 = time.time()
    f = C.Frame(lifts, tl['L1'], tl['L2'], sub=sub)
    f.set_load('stress')
    info['prestress_seconds'] = time.time() - t2; info['dof'] = f.n
    t3 = time.time()
    phis, ext, tab = f.longwave(nphi=nphi)
    info['eig_seconds_per_solve'] = (time.time() - t3) / (2 * nphi)
    info.update(pc_lw_phi=ext.tolist(), pc_lw_min=float(ext.min()), harm=C.harmonics(phis, ext))
    for lab, k in [('gamma', (0, 0)), ('k_half_0', (0.5, 0)), ('k_0_half', (0, 0.5)), ('k_half_half', (0.5, 0.5))]:
        info[f'pc_{lab}'] = float(f.lam(k))
    info['pc_min_all'] = float(min(info['pc_lw_min'], info['pc_gamma'], info['pc_k_half_0'],
                                   info['pc_k_0_half'], info['pc_k_half_half']))
    info.update(total_seconds=time.time() - T0, generate_seconds=t_gen, peak_mem_MB=peak_mem_mb(),
                status='P', units='p_c/(EA lambda_E), stress protocol')
    tag = f"{p}-{q}_sc{m}_{mode}_s{seed}"
    Path(outdir).mkdir(parents=True, exist_ok=True)
    (Path(outdir) / f'drift_{tag}.json').write_text(json.dumps(info, indent=1), encoding='utf-8')
    print(json.dumps({k: info[k] for k in ['V', 'mode', 'seed', 'pc_lw_min', 'pc_gamma', 'pc_min_all',
                                           'total_seconds', 'peak_mem_MB']}), flush=True)
    return info


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--p', type=int, default=12); ap.add_argument('--q', type=int, default=17)
    ap.add_argument('--m', type=int, default=1); ap.add_argument('--mode', default='random')
    ap.add_argument('--seed', type=int, default=1); ap.add_argument('--sweeps', type=float, default=200)
    ap.add_argument('--sub', type=int, default=2); ap.add_argument('--nphi', type=int, default=8)
    ap.add_argument('--outdir', default='drift_out')
    ap.add_argument('--array-index', type=int, default=None); ap.add_argument('--plan', default=None)
    ap.add_argument('--write-plan', default=None)
    a = ap.parse_args()
    if a.write_plan:
        Path(a.write_plan).write_text(json.dumps(default_plan(), indent=1)); sys.exit()
    if a.array_index is not None:
        plan = json.loads(Path(a.plan).read_text()) if a.plan else default_plan()
        job = plan[a.array_index]
        run(job['p'], job['q'], job['m'], job['mode'], job['seed'], a.sweeps, a.sub, a.nphi, a.outdir)
    else:
        run(a.p, a.q, a.m, a.mode, a.seed, a.sweeps, a.sub, a.nphi, a.outdir)

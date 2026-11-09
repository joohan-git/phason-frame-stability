"""critical-mode structure. For near-QC approximants of AB and Penrose, the NEV lowest critical
multipliers (not only the lowest) at long wave (per direction phi, Richardson DQ->0 branch by branch),
at Gamma and at (1/2,1/2).  Question: is the critical eigenvalue degenerate at fixed phi (a multiplet
that a phason could split), or only 'degenerate' across directions (the S^1 of an isotropic medium)?
usage: python critical_modes.py <cls> <name>  -> critical_modes_<cls>_<name>.json   (P)
"""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames', 'phason_paths', 'penrose', 'symmetry')]  # shared modules
import os
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='-1')
import json, math, sys, time
from pathlib import Path
import numpy as np
import scipy.sparse.linalg as spla

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'phason_paths')); sys.path.insert(0, str(ROOT / 'penrose'))
import ab_frame as AB  # noqa
import penrose_frame as PN   # noqa

CELLS = {
    ('ab', 'B1_2_3'): ([3, 2, 0, -2], [0, 2, 3, 2]),
    ('ab', 'B1_5_7'): ([7, 5, 0, -5], [0, 5, 7, 5]),
    ('ab', 'B1_12_17'): ([17, 12, 0, -12], [0, 12, 17, 12]),
    ('ab', 'E2a_5_7'): ([7, 5, 0, -5], [0, 7, 10, 7]),
    ('pen', 'fib3'): tuple(PN.fib_approx(3)),
    ('pen', 'fib4'): tuple(PN.fib_approx(4)),
    ('pen', 'fib5'): tuple(PN.fib_approx(5)),
    ('pen', 'mix1042'): ([10, 8, 3, -3, -8], [0, 8, 13, 13, 8]),
}
NEV = 6


def topk(f, k, nev=NEV):
    k = np.asarray(k, float)
    K = f._assemble(np.ones(len(f.ei)), f.Ke, k).tocsc()
    H = -f._assemble(f.N, f.Ge, k).tocsc()
    gamma = np.allclose(k, np.rint(k))
    fr = f.free_gamma if gamma else np.arange(f.n)
    K = K[fr][:, fr]; H = H[fr][:, fr]
    lu = spla.splu(K.tocsc(), permc_spec='COLAMD')
    op = spla.LinearOperator(K.shape, matvec=lambda x: lu.solve(H @ x), dtype=complex)
    vals = spla.eigs(op, k=nev, which='LR', tol=1e-10, maxiter=50000, return_eigenvectors=False)
    mu = np.sort(vals.real)[::-1]
    lam = np.array([1 / m / f_LAM(f) if m > 0 else np.inf for m in mu])
    return lam


def f_LAM(f):
    return AB.LAM_E  # identical lambda_E in both cores (r = 0.03, EA = 1)


def main(cls, name):
    t0 = time.time()
    L1, L2 = CELLS[(cls, name)]
    M = AB if cls == 'ab' else PN
    tl = M.canonical(L1, L2) if cls == 'pen' else M.canonical(L1, L2)
    f = M.Frame(tl['lifts'], tl['L1'], tl['L2'], sub=2); f.set_load('stress')
    phis = np.arange(8) * np.pi / 8
    dqs = (0.002, 0.001)
    tab = {dq: np.array([topk(f, dq * (f.cell @ np.array([math.cos(p), math.sin(p)]))) for p in phis]) for dq in dqs}
    r = (dqs[0] / dqs[1]) ** 2
    ext = (r * tab[dqs[1]] - tab[dqs[0]]) / (r - 1)
    gam = topk(f, (0, 0)); hh = topk(f, (0.5, 0.5))
    if cls == 'ab':
        a, b = AB.harmonic_parts(AB.phason_matrix(tl['L1'], tl['L2'])); v = 0j
    else:
        a, b, v = PN.phason_parts(PN.phason_matrix(tl['L1'], tl['L2']))
    lw1 = ext[:, 0]; lw2 = ext[:, 1]
    out = dict(status='P', cls=cls, name=name, V=len(tl['lifts']), a=[a.real, a.imag], b=[b.real, b.imag],
               delta=float(math.sqrt(abs(a) ** 2 + abs(b) ** 2 + abs(v) ** 2)),
               phis_deg=np.degrees(phis).tolist(), longwave_branches=ext.tolist(),
               dq_small_branches=tab[dqs[1]].tolist(), gamma=gam.tolist(), halfhalf=hh.tolist(),
               lw1_minus_c0_spread=float(lw1.max() - lw1.min()),
               gap_lw2_lw1_min=float((lw2 - lw1).min()), gap_lw2_lw1_mean=float((lw2 - lw1).mean()),
               gap_gamma_lw=float(gam[0] - lw1.mean()), gap_halfhalf_lw=float(hh[0] - lw1.mean()),
               c0=float(lw1.mean()), seconds=time.time() - t0)
    (HERE / f'critical_modes_{cls}_{name}.json').write_text(json.dumps(out, indent=1), encoding='utf-8')
    print(json.dumps({k: out[k] for k in ('cls', 'name', 'V', 'delta', 'c0', 'gap_lw2_lw1_min', 'gap_gamma_lw', 'gap_halfhalf_lw', 'seconds')}), flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])

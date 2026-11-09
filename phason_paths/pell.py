"""B1 cusp along the Pell sequence to larger cells: p_c at phi = 0 and 45 deg (Richardson DQ->0),
c0 = mean, h4 = half-difference (pure-B1 slopes have h2 = 0 by symmetry). -> pell.json (P)"""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames', 'slope_survey')]  # shared modules
import json, math, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import ab_frame as C
rows = []
for p, q in [(12, 17), (29, 41), (70, 99)]:
    t0 = time.time()
    tl = C.canonical(*C.ab_approx(p, q))
    f = C.Frame(tl['lifts'], tl['L1'], tl['L2'], sub=2); f.set_load('stress')
    vals = []
    for phi in (0.0, math.pi / 4):
        u = np.array([math.cos(phi), math.sin(phi)])
        l2 = f.lam(0.002 * (f.cell @ u)); l1 = f.lam(0.001 * (f.cell @ u))
        vals.append((4 * l1 - l2) / 3)
    b = C.harmonic_parts(C.phason_matrix(tl['L1'], tl['L2']))[1].real
    rows.append(dict(pq=f'{p}/{q}', V=len(tl['lifts']), dof=f.n, b=b, pc_phi0=vals[0], pc_phi45=vals[1],
                     c0=(vals[0] + vals[1]) / 2, h4=(vals[0] - vals[1]) / 2, kappa4=(vals[0] - vals[1]) / 2 / b,
                     pc_min=min(vals), seconds=time.time() - t0))
    print(json.dumps(rows[-1]), flush=True)
    Path(__file__).with_name('pell.json').write_text(json.dumps(dict(status='P', units='p_c/(EA lambda_E), stress protocol, sub=2', rows=rows), indent=1), encoding='utf-8')

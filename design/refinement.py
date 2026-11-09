"""C check: is the 16-direction long-wave grid enough for uniaxial lam(theta)? Direct lam on a 1.40625 deg phi grid (128 dirs) on B1 5/7, and Penrose fib3,
for 9 load angles. Output refinement.json (P)."""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames', 'penrose', 'phason_paths', 'robustness', 'heldout_test', 'isotropic_loss')]  # shared modules
import os
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[v] = '1'
import json, math, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import design_core as X
src = ''
ns = {'np': np}; ns.update(vars(__import__('ab_families'))); fl = ns['family_lifts']
out = {}
for cls, name, L in (('ab', 'B15/7', fl(5, 7, 'B1')), ('pen', 'fib3', None)):
    mod = X.load(cls)
    if cls == 'pen':
        L = mod.fib_approx(3)
        tl = mod.canonical(L[0], L[1])
    else:
        tl = mod.canonical(L[0], L[1], seed=7)
    f = X.make_xframe(mod)(tl['lifts'], tl['L1'], tl['L2'], sub=2)
    phis = np.arange(64) * np.pi / 64
    rows = []
    for th in np.arange(0, 49, 5.625 if cls == 'ab' else 9.0) * np.pi / 180:
        f.set_stress(X.state_tensor('uniaxial', float(th)))
        ext = f.lw_phi(phis)
        i = int(np.argmin(ext))
        # coarse 16-direction grid restricted to multiples of pi/16
        rows.append(dict(theta_deg=float(np.degrees(th)), lam_fine=float(ext.min()), phi_min_deg=float(np.degrees(phis[i])), lam_coarse16=float(ext[::4].min()), extra={str(k): float(f.lam_N(k)) for k in ((0, 0), (.5, .5))}))
        print(name, rows[-1], flush=True)
    out[name] = rows
(HERE / 'refinement.json').write_text(json.dumps(dict(label='P', rows=out), indent=1))

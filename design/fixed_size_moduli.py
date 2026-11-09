"""Exploratory (not preregistered): zero-load K and T0 on the fixed-V cells of data/isotropic_results.json, against the stored c0 and forbidden-patch measures.
Question raised by the B path data: does the bulk-modulus deficit track the isotropic pressure loss? Output fixed_size_moduli.json (P)."""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames', 'penrose', 'phason_paths', 'robustness', 'heldout_test', 'isotropic_loss')]  # shared modules
import os
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[v] = '1'
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import design_core as X
t1 = json.loads((HERE.parent / 'isotropic_loss' / 'isotropic_results.json').read_text())['fixedV']
print({k: type(v).__name__ for k, v in t1['ab']['cells'][0].items()})
out = {}
for cls in ('ab', 'pen'):
    rows = []
    mod = X.load(cls)
    for c in t1[cls]['cells']:
        tl = mod.canonical(c['L1'], c['L2'], seed=7) if cls == 'ab' else mod.canonical(c['L1'], c['L2'])
        f = X.make_xframe(mod)(tl['lifts'], tl['L1'], tl['L2'], sub=2); f.compute_C3()
        m = X.moduli(f.C3); d = f.demand(-np.eye(2))
        row = {k: c[k] for k in c if not isinstance(c[k], (list, dict))}
        row.update(K=m['K'], T0=m['T0'], absR0=m['absR0'], absR1=m['absR1'], total_max=d['total_max'], ax_max=d['ax_max'], ledge=2 * len(tl['lifts']) / abs(np.linalg.det(f.cell)))
        rows.append(row)
    out[cls] = rows
    print(cls, [(r['V'], round(r['delta'], 4), round(r['K'], 4), round(r['total_max'], 2)) for r in rows])
(HERE / 'fixed_size_moduli.json').write_text(json.dumps(dict(label='P', note='exploratory', rows=out), indent=1))

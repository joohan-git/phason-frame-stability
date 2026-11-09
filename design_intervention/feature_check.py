"""Check of the intervention gradient feature against stored T2 values (independent/law_test_raw.json, snapshots). Covariate only. Writes feature_check.json."""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames', 'penrose', 'heldout_test', 'design', 'nonlinear', 'defects', 'phason_paths', 'independent_code', 'forbidden_patches')]  # shared modules
import json, os
os.environ.setdefault('OMP_NUM_THREADS', '2'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
from pathlib import Path
import numpy as np
from intervention_run import *
C2 = M / 'independent_code'
d = json.load(open(C2 / 'law_test_raw.json'))
rows = {r['id']: r for r in d['rows']}
ids = sorted(p.stem for p in (C2 / 'law_snapshots').glob('*.npz') if p.stem in rows)
rng = np.random.default_rng(16)
ab = [i for i in ids if i.startswith('ab')]; pen = [i for i in ids if i.startswith('pen')]
pick = sorted(rng.choice(ab, min(8, len(ab)), replace=False).tolist() + rng.choice(pen, min(6, len(pen)), replace=False).tolist())
print(len(ids), len(ab), len(pen), flush=True)
out = []
for i in pick:
    z = np.load(C2 / 'law_snapshots' / f'{i}.npz')
    cls = 'ab' if i.startswith('ab') else 'pen'
    L = z['L']; L1, L2 = L[0].astype(np.int64), L[1].astype(np.int64)
    q_, _ = np.linalg.qr(np.c_[np.array([L1, L2], float).T, np.eye(L.shape[1])])
    tl = dict(L1=L1, L2=L2, perp=q_[:, 2:L.shape[1]].T)
    f = rows[i]['features']
    g = grad_feature(cls, z['n'].astype(np.int64), tl)
    x_theirs_ref = max(0.0, 4.0 * (g - f['gradient_reference']) / f['sigma'])
    r = dict(id=i, protocol=rows[i]['protocol'], severity=rows[i]['severity'], N=int(len(z['n'])), g_intervention=g, g_t2=f['gradient_max'], rel_g=abs(g - f['gradient_max']) / max(f['gradient_max'], 1e-12), x_intervention=x_theirs_ref, x_t2=f['x']['max_gradient'], abs_x=abs(x_theirs_ref - f['x']['max_gradient']))
    out.append(r); print(r, flush=True)
# reference: intervention default offset against the prediction-law reference of the same class
refs = {}
for k, v in d['references'].items():
    cls = 'ab' if k.startswith('ab') else 'pen'
    q_, _ = np.linalg.qr(np.c_[np.array(v['L'], float).T, np.eye(np.array(v['L']).shape[1])])
    refs[k] = dict(N=v['N'], gradient_reference=v['features']['gradient_max'], sigma=v['features']['sigma'])
for name, cls, cn in (('ab1217', 'ab', (12, 17)), ('fib5', 'pen', 5)):
    tl = cell(cls, cn); r = canon_ref(cls, tl); refs[f'intervention_{name}'] = dict(N=len(tl['lifts']), g0=r['g0'], s0=r['s0'])
print(refs, flush=True)
(HERE / 'feature_check.json').write_text(json.dumps(dict(rows=out, refs=refs, max_rel_g=max(r['rel_g'] for r in out), max_abs_x=max(r['abs_x'] for r in out)), indent=1))
print('max rel g', max(r['rel_g'] for r in out), 'max abs x', max(r['abs_x'] for r in out))

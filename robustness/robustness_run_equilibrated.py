import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames', 'phason_paths', 'penrose', 'isotropic_loss')]  # shared modules
import os, sys, json, math, time
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[v] = '2'
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import ab_frame as AB, penrose_frame as PN
i = int(sys.argv[1]); T = json.loads((HERE / 'plan_equilibrated.json').read_text())[i]; out = HERE / 'out_equilibrated' / f'case{i:03d}.json'
if out.exists(): sys.exit(0)
C = AB if T['cls'] == 'ab' else PN
C.R_STRUT = T['r']; C.EI = T['r'] ** 2 / 4; C.LAM_E = math.pi ** 2 * C.EI / C.EA
t0 = time.time()
if T['cls'] == 'ab':
    tl = C.canonical(T['L1'], T['L2'], seed=T['off'])
else:
    rng = np.random.default_rng(100); o = rng.uniform(-.5, .5, 5); alt = np.array([1, -1, 1, -1, 1]); o += (0.5 - alt @ o) / 5 * alt
    tl = C.canonical(T['L1'], T['L2'], offset=o)
lifts = tl['lifts']
if 'flips' in T:
    lifts = C.flip_mc(lifts, tl['L1'], tl['L2'], T['flips'], T['fseed'])[0]
f = C.Frame(lifts, tl['L1'], tl['L2'], sub=T['sub']); f.set_load('stress')
ph, ext, _ = f.longwave(nphi=8); h = C.harmonics(ph, ext)
rec = dict(T, V=len(lifts), c0=h['c0'], h4=h['h4'], h2=h['h2'], pc_lw_min=float(ext.min()), pc_gamma=float(f.lam((0, 0))), pc_hh=float(f.lam((.5, .5))), pc_phi=ext.tolist(), sec=time.time() - t0)
if T['cls'] == 'ab': a, b = C.harmonic_parts(C.phason_matrix(tl['L1'], tl['L2'])); rec['a'], rec['b'] = float(abs(a)), float(abs(b))
out.write_text(json.dumps(rec)); print(i, rec['name'], rec['sub'], rec['r'], rec['c0'], rec['sec'])

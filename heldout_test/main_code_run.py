
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames',)]  # shared modules
import os, sys, json, time
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[v] = '2'
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import ab_frame as AB, penrose_frame as PN
i = int(sys.argv[1]); T = json.loads((HERE / 'main_code_plan.json').read_text())[i]; out = HERE / 'main_code_out' / f'case{i:03d}.json'
if out.exists(): sys.exit(0)
t0 = time.time()
if T['cls'] == 'ab':
    C = AB; tl = C.canonical(T['L1'], T['L2'], seed=[7, 11, 13][T['off']])
else:
    C = PN; rng = np.random.default_rng(100 + T['off']); o = rng.uniform(-.5, .5, 5); alt = np.array([1, -1, 1, -1, 1]); o += (0.5 - alt @ o) / 5 * alt
    tl = C.canonical(T['L1'], T['L2'], offset=o)
assert C.check_tiling(tl['lifts'], tl['L1'], tl['L2']) == 0
f = C.Frame(tl['lifts'], tl['L1'], tl['L2'], sub=2); f.set_load('stress')
ph, ext, _ = f.longwave(nphi=8); h = C.harmonics(ph, ext)
out.write_text(json.dumps(dict(T, V=len(tl['lifts']), c0=h['c0'], h4=h['h4'], h2=h['h2'], pc_lw_min=float(ext.min()), sec=time.time() - t0)))
print(i, T['name'], len(tl['lifts']), h['c0'], time.time() - t0)

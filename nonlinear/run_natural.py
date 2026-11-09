"""Natural (connected) equilibrium path from p = 0 by arclength, no seed. usage: run_natural.py NAME [sub] [pstop] [psi_stop]"""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames', 'phason_paths', 'objectivity', 'disorder')]  # shared modules
import sys, json, time, math
import numpy as np
from structures import *
name = sys.argv[1]; sub = int(sys.argv[2]) if len(sys.argv) > 2 else 2
pstop = float(sys.argv[3]) if len(sys.argv) > 3 else 0.95; psistop = float(sys.argv[4]) if len(sys.argv) > 4 else 0.8
mesh = get_mesh(name, sub); model = Model(mesh); t0 = time.time()
lam, vec, st = model.linear_critical(macro=True)
z0 = np.zeros(mesh.nz); h = st['z'] * mesh.lamE
scale = math.sqrt(mesh.nz)
def stop(r_, rows):
    pk = max(x['p'] for x in rows if 'p' in x)
    return r_['p'] > pstop or r_['max_chord'] > psistop or (r_['p'] < 0.02 and len(rows) > 5) or r_['p'] < 0.7 * pk
wp = default_wp(mesh)
rows, states = follow(model, z0, 0.0, (st['z'], 1.0), 0.02 * scale, 800, stop=stop, wp=wp, dsmax=0.05 * scale, dsmin=1e-7 * scale, k_eig=3, guard_neg=0.001 * scale)
good = [r_ for r_ in rows if 'p' in r_]
for r_ in good[:: max(1, len(good) // 40)]:
    print('%.5f A %.5f psi_rms %.4f psi_max %.4f tau %.3f edev %.5f neg %d mu %.5f ds %.3g' % (r_['p'], r_['A_rel'], r_['rms_chord'], r_['max_chord'], r_['max_tau'], r_['edev'], r_['neg'], r_['mu'][0], r_['ds']), flush=True)
print('end', rows[-1].get('status', 'maxsteps'), len(good), time.time() - t0)
(HERE / 'out' / f'natural_{name}_s{sub}.json').write_text(json.dumps(dict(name=name, sub=sub, lin=lam.tolist(), rows=good, end=rows[-1].get('status', 'maxsteps'), sec=time.time() - t0), default=float))

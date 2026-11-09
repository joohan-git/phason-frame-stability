"""Registry of structures."""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames', 'phason_paths', 'objectivity', 'disorder')]  # shared modules
import json, numpy as np
from corotational_frame import *
import ab_frame as C
ns = {'np': np}
_src = ''
ns.update(vars(__import__('ab_families'))); family_lifts = ns['family_lifts']

def equilibrate(tl, seed, sweeps=200.0):
    """Protocol of disorder/random_mode.py."""
    lifts = tl['lifts']; V = len(lifts)
    Lc = float(np.linalg.norm(C.P @ np.array(tl['L1'], float)))
    target = max(float(sweeps), 0.1 * Lc ** 2)
    chunk = max(1.0, target / 10)
    trace = []; done = 0.0; k = 0
    while True:
        lifts, a = C.flip_mc(lifts, tl['L1'], tl['L2'], int(chunk * V), seed * 100003 + k)
        done += chunk; k += 1
        y = lifts @ tl['perp'].T
        trace.append([done, float(np.mean(np.sum((y - y.mean(0)) ** 2, 1)))])
        if done >= target and len(trace) >= 9:
            v = np.array([x[1] for x in trace])
            if abs(v[-3:].mean() - v[-6:-3].mean()) / max(v[-6:-3].mean(), 1e-12) < 0.10:
                break
        if done >= 3 * target:
            break
    return lifts

def get_mesh(name, sub=2, r=0.03):
    """name: 'square', 'B1_5_7', ..., 'R22' (Fig. 4), 'R101'."""
    if name == 'square':
        return square_mesh(sub, r)
    if name.startswith('squareN'):
        k = int(name[7:])
        return supercell(square_mesh(sub, r), [[k, 0], [0, k]], name)
    if name == 'R22':
        d = json.loads((ROUGH / 'random_mode.json').read_text())
        return from_lifts(np.array(d['lifts'], np.int64), d['L1'], d['L2'], sub=sub, r=r, name=name)
    if name[0] == 'R':
        seed = int(name[1:])
        tl = C.canonical(*family_lifts(12, 17, 'B1'))
        lifts = equilibrate(tl, seed)
        assert C.check_tiling(lifts, tl['L1'], tl['L2']) == 0
        return from_lifts(lifts, tl['L1'], tl['L2'], sub=sub, r=r, name=name)
    fam, p, q = name.split('_')
    tl = C.canonical(*family_lifts(int(p), int(q), fam))
    return from_lifts(tl['lifts'], tl['L1'], tl['L2'], sub=sub, r=r, name=name)

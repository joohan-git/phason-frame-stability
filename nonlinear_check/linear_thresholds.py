"""Classical linear thresholds (undeformed geometry, N only) for all cases/meshes, to compare definitions."""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames', 'phason_paths', 'disorder')]  # shared modules
import sys, json
from pathlib import Path
import numpy as np
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE)); import hermite_frame as nb
ROUGH = HERE.resolve().parent / 'disorder'
import ab_frame as C
from ab_families import family_lifts
out = {}
def build(case, sub):
    if case == 'square':
        return nb.square_lattice(1, 1, sub)
    if case in ('B1', 'E2a'):
        L = family_lifts(5, 7, case); tl = C.canonical(L[0], L[1], seed=7); lifts, L1, L2 = tl['lifts'], tl['L1'], tl['L2']
    else:
        dat = json.loads((ROUGH / 'random_mode.json').read_text(encoding='utf-8'))
        lifts = np.array(dat['lifts'], np.int64); L1 = np.array(dat['L1'], np.int64); L2 = np.array(dat['L2'], np.int64)
    return nb.build_periodic(lifts, L1, L2, C.edges_from_lifts(lifts, L1, L2), sub)
for case, subs in (('square', (2, 4, 8)), ('B1', (2, 4, 8)), ('E2a', (2, 4, 8)), ('random', (2, 4))):
    for sub in subs:
        fr = build(case, sub)
        cl = nb.linear_buckling_classical(fr, k=4); cl = np.sort(cl[cl > 0])
        se, _ = nb.linear_buckling(fr, k=4); se = np.sort(se[se > 0])
        out[f"{case}_sub{sub}"] = dict(classical=[float(x) for x in cl[:3]], secant=[float(x) for x in se[:3]])
        print(case, sub, 'classical', np.round(cl[:3], 6).tolist(), 'secant', np.round(se[:3], 6).tolist(), flush=True)
(HERE / 'linear_thresholds.json').write_text(json.dumps(out, indent=1), encoding='utf-8')

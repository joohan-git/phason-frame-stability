"""R3: offset (phase) averaging of the 4-theta splitting, B1 vs E2a at p/q = 12/17 (stress protocol, sub=2).
The D8 rule applies to a symmetry-covariant family; a single rational offset need not carry the stabilizer.
Average the complex harmonic h4 over 6 window offsets. -> window_offsets.json (P)"""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames', 'slope_survey')]  # shared modules
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import ab_frame as C
from ab_families import family_lifts
rows = []
for fam in ('B1', 'E2a'):
    L = family_lifts(12, 17, fam)
    for seed in (7, 11, 23, 37, 51, 68):
        tl = C.canonical(L[0], L[1], seed=seed)
        f = C.Frame(tl['lifts'], tl['L1'], tl['L2'], sub=2); f.set_load('stress')
        phis, ext, _ = f.longwave(nphi=8)
        h = C.harmonics(phis, ext)
        rows.append(dict(family=fam, offset_seed=seed, V=len(tl['lifts']), pc_min=float(ext.min()), c0=h['c0'], h4=h['h4'], h2=h['h2']))
        print(json.dumps(rows[-1]), flush=True)
summ = {}
for fam in ('B1', 'E2a'):
    hs = np.array([complex(*r['h4']) for r in rows if r['family'] == fam])
    summ[fam] = dict(mean_abs_h4=float(np.abs(hs).mean()), abs_mean_h4=float(abs(hs.mean())),
                     mean_c0=float(np.mean([r['c0'] for r in rows if r['family'] == fam])),
                     pc_range=[float(min(r['pc_min'] for r in rows if r['family'] == fam)), float(max(r['pc_min'] for r in rows if r['family'] == fam))])
Path(__file__).with_name('window_offsets.json').write_text(json.dumps(dict(status='P', rows=rows, summary=summ), indent=1), encoding='utf-8')
print(json.dumps(summ, indent=1))

"""Fig. 2c data (rough): p_c(phi) petals at 12/17 for the B1 (Pell) and E2a phason families, six window offsets each.
Same construction as main/window_offsets.py (offset seeds 7, 11, 23, 37, 51, 68; stress protocol;
sub = 2; Richardson DQ 0.002/0.001), but sampled at nphi = 24 directions in [0, pi) instead of 8, so the petal is drawn
from sampled values, not from the fitted harmonics. family_lifts is read from main/ab_families.py (not modified).
Writes petals.json.
Usage: python -X utf8 petals.py [nphi] [seeds...]  (B1 at the first two seeds, E2a at all; ~4 min per cell)"""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames', 'slope_survey')]  # shared modules
import os, sys, json, time, math
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ab_frame as C
src = ''
ns = {'np': np}
ns.update(vars(__import__('ab_families')))   # ROT and family_lifts only (static definitions)
family_lifts = ns['family_lifts']

nphi = int(sys.argv[1]) if len(sys.argv) > 1 else 24
seeds = [int(s) for s in sys.argv[2:]] or [7, 11, 23, 37, 51, 68]
out = HERE / 'petals.json'
rows = json.loads(out.read_text())['rows'] if out.exists() else []
done = {(r['family'], r['offset_seed'], r['nphi']) for r in rows}
PLAN = [('B1', s) for s in seeds[:2]] + [('E2a', s) for s in seeds]   # B1 is offset-invariant (window_offsets): 2 offsets suffice
for fam, seed in PLAN:
    if True:
        if (fam, seed, nphi) in done:
            continue
        t0 = time.time()
        L = family_lifts(12, 17, fam)
        tl = C.canonical(L[0], L[1], seed=seed)
        f = C.Frame(tl['lifts'], tl['L1'], tl['L2'], sub=2); f.set_load('stress')
        phis, ext, _ = f.longwave(nphi=nphi)
        h = C.harmonics(phis, ext)
        a, b = C.harmonic_parts(C.phason_matrix(tl['L1'], tl['L2']))
        rows.append(dict(family=fam, p=12, q=17, offset_seed=seed, V=len(tl['lifts']), sub=2, nphi=nphi,
                         abs_a=float(abs(a)), abs_b=float(abs(b)), phi=phis.tolist(), pc_phi=ext.tolist(),
                         c0=h['c0'], h4=h['h4'], h2=h['h2'], fit_rms=h['fit_rms'], seconds=time.time() - t0, status='P'))
        print(fam, seed, len(tl['lifts']), 'c0 %.7f |h4| %.3e  %.1fs' % (h['c0'], abs(complex(*h['h4'])), time.time() - t0), flush=True)
        out.write_text(json.dumps(dict(script='petals.py', status='P',
                                       units='p_c / (pi^2 EI / l^3), stress protocol, 2 elements/strut, r = 0.03 l',
                                       rows=rows), indent=1), encoding='utf-8')

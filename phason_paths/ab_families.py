"""independent's four phason families (B1, B2, E2a, E2diag; lifts copied from independent/m1_geometry.py
family_lifts, re-typed here, no import) under the STRESS protocol with the main code (ab_frame),
main canonical-window generator and main offset. 8 directions, Richardson DQ->0, sub=2.
Outputs ab_families.json (all P)."""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames', 'slope_survey')]  # shared modules
import json, math, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import ab_frame as C

ROT = np.array([[0, 0, 0, -1], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.int64)


def family_lifts(p, q, fam):
    if fam in ('B1', 'B2'):
        L = np.array([[q, p, 0, -p], [0, p, q, p]], dtype=np.int64)
    else:
        L = np.array([[q, p, 0, -p], [0, q, 2 * p, q]], dtype=np.int64)
    if fam in ('B2', 'E2diag'):
        L = L @ (np.eye(4, dtype=np.int64) + ROT).T
    return L


def run(fam, p, q, nphi=8, sub=2):
    t0 = time.time()
    L = family_lifts(p, q, fam)
    tl = C.canonical(L[0], L[1])
    f = C.Frame(tl['lifts'], tl['L1'], tl['L2'], sub=sub); f.set_load('stress')
    phis, ext, tab = f.longwave(nphi=nphi)
    M = C.phason_matrix(tl['L1'], tl['L2']); a, b = C.harmonic_parts(M)
    h = C.harmonics(phis, ext)
    row = dict(family=fam, p=p, q=q, V=len(tl['lifts']), sub=sub, nphi=nphi,
               a=[a.real, a.imag], b=[b.real, b.imag], amp=float(math.hypot(abs(a), abs(b))),
               pc_phi=ext.tolist(), pc_min_sampled=float(ext.min()), harm=h,
               h4_abs=float(abs(complex(*h['h4']))), h2_abs=float(abs(complex(*h['h2']))), seconds=time.time() - t0)
    print(json.dumps({k: row[k] for k in ['family', 'p', 'q', 'V', 'amp', 'pc_min_sampled', 'h4_abs', 'h2_abs', 'seconds']}), flush=True)
    return row


if __name__ == '__main__':
    out = Path(__file__).with_name('ab_families.json')
    rows = []
    plan = [(fam, p, q) for fam in ('B1', 'B2', 'E2a', 'E2diag') for p, q in ((1, 1), (2, 3), (5, 7), (12, 17))]
    plan += [('B1', 29, 41), ('E2a', 29, 41)]
    for fam, p, q in plan:
        rows.append(run(fam, p, q))
        out.write_text(json.dumps(dict(status='P', units='p_c/(EA lambda_E), stress protocol, sub=2, main offset',
                                       rows=rows), indent=1), encoding='utf-8')

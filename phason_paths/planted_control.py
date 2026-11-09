"""Control for the planted test: UNBIASED flips confined to the same region (rmove = 11 around the
cell centre, as for planted R=4.5). Separates 'planting' from 'randomising the annulus'. -> planted_control.json (P)"""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames', 'slope_survey')]  # shared modules
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import ab_frame as C
from flips import mech
t = C.canonical(*C.ab_approx(12, 17)); L1, L2 = t['L1'], t['L2']
cell = np.array([C.P @ L1, C.P @ L2]); c = cell.sum(0) / 2
rows = []
for seed, n in [(11, 2_000_000), (12, 2_000_000), (13, 200_000)]:
    l, acc = C.flip_mc(t['lifts'], L1, L2, n, seed, center=c, rmove=11.0)
    assert C.check_tiling(l, L1, L2) == 0
    st = C.phason_stats(l, L1, L2, t['perp'], t['lifts'])
    m = mech(l, L1, L2, center=c, radius=11.0)
    rows.append(dict(seed=seed, attempts=n, accepted=acc, **st, **m))
    print(json.dumps({k: rows[-1][k] for k in ['seed', 'accepted', 'moved_fraction', 'pc_lw_min', 'pc_gamma', 'gamma_mode_weight_in_disk']}), flush=True)
Path(__file__).with_name('planted_control.json').write_text(json.dumps(dict(
    system='AB 12/17, unbiased flips confined to |x - centre| < 11 (same region as planted R=4.5)', status='P',
    units='p_c/(EA lambda_E), stress protocol', rows=rows), indent=1), encoding='utf-8')

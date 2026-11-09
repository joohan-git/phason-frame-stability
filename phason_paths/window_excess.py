"""Do disorder-induced localized critical modes sit on vertices the canonical window forbids?
For each config: Gamma critical mode weight on vertices with internal coordinate outside the
canonical window W (excess > 0), vs the fraction of such vertices.  -> window_excess.json (P)"""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames', 'slope_survey')]  # shared modules
import json, sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, str(Path(__file__).parent))
import ab_frame as C
t = C.canonical(*C.ab_approx(12, 17)); L1, L2 = t['L1'], t['L2']
cell = np.array([C.P @ L1, C.P @ L2]); box = np.array([cell[0, 0], cell[1, 1]]); ctr = cell.sum(0) / 2
cfg = {}
for k in (100, 300, 1000):
    for s in (1, 2):
        cfg[f'flip_k{k}_s{s}'] = C.flip_mc(t['lifts'], L1, L2, 50 * k + 10000, s * 1000 + k, maxacc=k)[0]
cfg['flip_eq_s1'] = C.flip_mc(t['lifts'], L1, L2, 4_000_000, 1007)[0]
cfg['region_control_s11'] = C.flip_mc(t['lifts'], L1, L2, 2_000_000, 11, center=ctr, rmove=11.0)[0]
l = t['lifts']
for b, n in [(0.5, 3e6), (1.0, 3e6), (2.0, 3e6), (4.0, 4e6), (8.0, 3e6)]:
    l, _ = C.flip_mc(l, L1, L2, int(n), int(10 * b + 7 * 4.5), bias=b, center=ctr, radius=4.5, rmove=11.0)
cfg['plant_R4.5'] = l
rows = {}
for name, lifts in cfg.items():
    f = C.Frame(lifts, L1, L2, sub=2); f.set_load('stress')
    lam, v = f.lam((0, 0), vectors=True)
    w = np.sum(np.abs(v.reshape(-1, 3)[:, :2]) ** 2, 1)
    Xv = np.mod(lifts @ C.P.T, box); Xv[Xv >= box] = 0
    Xn = np.mod(f.X, box); Xn[Xn >= box] = 0
    wv = np.zeros(len(lifts)); np.add.at(wv, cKDTree(Xv, boxsize=box).query(Xn)[1], w); wv /= wv.sum()
    ex = C.window_excess(lifts, t)
    out = ex > 1e-9
    # weight within 2 edge lengths of an outside vertex
    tree = cKDTree(Xv, boxsize=box)
    near = np.zeros(len(lifts), bool)
    if out.any():
        for nb in tree.query_ball_point(Xv[out], 2.0):
            near[nb] = True
    rows[name] = dict(pc_gamma=float(lam), frac_vertices_outside_W=float(out.mean()),
                      mode_weight_on_outside=float(wv[out].sum()),
                      frac_vertices_within2_of_outside=float(near.mean()), mode_weight_within2_of_outside=float(wv[near].sum()),
                      max_excess=float(ex.max()), weighted_mean_excess=float(np.sum(wv * np.maximum(ex, 0))),
                      mean_excess=float(np.mean(np.maximum(ex, 0))))
    print(name, json.dumps(rows[name]), flush=True)
Path(__file__).with_name('window_excess.json').write_text(json.dumps(dict(status='P', rows=rows,
    note='canonical window W = projection of unit 4-cube along the 12/17 slope plane, same offset as the canonical cut'), indent=1), encoding='utf-8')

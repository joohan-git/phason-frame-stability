"""R2 follow-up: where do the disorder-induced critical modes live, and are they mesh-converged?
For flip configs (12/17, seeds/k as in flips.py) and the planted R=4.5 config:
Gamma critical mode -> participation ratio, top-weight vertex star, local phason strain there,
radial weight profile (planted); sub = 2 vs 4 thresholds.  Output mode_followup.json (status P)."""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames', 'slope_survey')]  # shared modules
import json, math, sys, time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, str(Path(__file__).parent))
import ab_frame as C

HERE = Path(__file__).parent
t = C.canonical(*C.ab_approx(12, 17)); L1, L2 = t['L1'], t['L2']
cell = np.array([C.P @ L1, C.P @ L2]); box = np.array([cell[0, 0], cell[1, 1]]); ctr = cell.sum(0) / 2


def configs():
    out = {}
    for k in (300, 1000):
        out[f'flip_k{k}_s1'] = C.flip_mc(t['lifts'], L1, L2, 50 * k + 10000, 1000 + k, maxacc=k)[0]
    out['flip_eq_s1'] = C.flip_mc(t['lifts'], L1, L2, 4_000_000, 1007)[0]
    l = t['lifts']
    for b, n in [(0.5, 3e6), (1.0, 3e6), (2.0, 3e6), (4.0, 4e6), (8.0, 3e6)]:
        l, _ = C.flip_mc(l, L1, L2, int(n), int(10 * b + 7 * 4.5), bias=b, center=ctr, radius=4.5, rmove=11.0)
    out['plant_R4.5'] = l
    return out


def local_strain(lifts, R=4.0):
    X = np.mod(lifts @ C.P.T, box); X[X >= box] = 0
    y = lifts @ t['perp'].T
    tree = cKDTree(X, boxsize=box)
    G = np.zeros(len(X))
    for i, nb in enumerate(tree.query_ball_point(X, R)):
        d = X[nb] - X[i]; d -= box * np.round(d / box)
        coef, *_ = np.linalg.lstsq(np.c_[d, np.ones(len(nb))], y[nb], rcond=None)
        G[i] = np.sqrt(np.sum(coef[:2] ** 2))
    return X, G


rows = {}
for name, lifts in configs().items():
    t0 = time.time()
    f = C.Frame(lifts, L1, L2, sub=2); f.set_load('stress')
    lam, v = f.lam((0, 0), vectors=True)
    w = np.sum(np.abs(v.reshape(-1, 3)[:, :2]) ** 2, 1)
    V = len(lifts)
    # fold mid-element nodes onto their parent vertex position for spatial statistics
    Xn = np.mod(f.X, box); Xn[Xn >= box] = 0
    pr = float(w.sum() ** 2 / (len(w) * np.sum(w ** 2)))
    Xv, G = local_strain(lifts)
    tree = cKDTree(Xv, boxsize=box)
    wv = np.zeros(V); np.add.at(wv, tree.query(Xn)[1], w)
    top = np.argsort(wv)[::-1][:20]
    edges = C.edges_from_lifts(lifts, L1, L2)
    deg = np.bincount(edges[:, :2].ravel(), minlength=V)
    corr = float(np.corrcoef(wv, G)[0, 1])
    row = dict(pc_gamma=float(lam), participation_ratio=pr,
               top20_weight_fraction=float(wv[top].sum() / wv.sum()),
               top20_degree_hist={str(d): int(np.sum(deg[top] == d)) for d in np.unique(deg[top])},
               all_degree_hist={str(d): int(np.sum(deg == d)) for d in np.unique(deg)},
               top20_local_phason_strain_mean=float(G[top].mean()), local_phason_strain_mean=float(G.mean()),
               corr_weight_vs_local_strain=corr)
    if name.startswith('plant'):
        dr = Xv - ctr; dr -= box * np.round(dr / box); r = np.linalg.norm(dr, axis=1)
        prof = [float(wv[(r >= a) & (r < a + 2)].sum() / wv.sum()) for a in range(0, 18, 2)]
        row['radial_weight_profile_2'] = prof
        row['local_strain_profile_2'] = [float(G[(r >= a) & (r < a + 2)].mean()) for a in range(0, 18, 2)]
    f4 = C.Frame(lifts, L1, L2, sub=4); f4.set_load('stress')
    row['pc_gamma_sub4'] = float(f4.lam((0, 0)))
    row['seconds'] = time.time() - t0
    rows[name] = row
    print(name, json.dumps(row), flush=True)
(HERE / 'mode_followup.json').write_text(json.dumps(dict(status='P', units='p_c/(EA lambda_E), stress protocol', rows=rows), indent=1), encoding='utf-8')

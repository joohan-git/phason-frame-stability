"""geometry checks -> geometry_check.json.
(C) exact QC values from the A2+A2 Voronoi/Delaunay structure (derivation in):
    window = regular 12-gon, edge 1/sqrt3, area 2+sqrt3; covolume det B = 3;
    vertex density (2+sqrt3)/3; per vertex: triangles 4 sqrt3 - 6, squares 2 - sqrt3, 30-rhombi 2 - sqrt3;
    area fractions triangles 1/2, squares 1/3, rhombi 1/6.
(P) counted values on large approximants, window symmetry of the C4 / C6 cells, vertex configurations."""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames', 'phason_paths')]  # shared modules
import json, math
from pathlib import Path
import numpy as np
import twelvefold_core as D
from scipy.spatial import ConvexHull

s3 = math.sqrt(3)
exact = dict(window_area=2 + s3, vertex_density=(2 + s3) / 3, tri_per_vertex=4 * s3 - 6, sq_per_vertex=2 - s3,
             rh_per_vertex=2 - s3, area_frac=dict(tri=0.5, sq=1 / 3, rh=1 / 6))
out = dict(status='P (counts) / C (exact)', exact=exact)
# QC window
Wq = D.VOR[:, 2:]
h = ConvexHull(Wq)
pts = Wq[h.vertices]
rad = np.linalg.norm(pts, axis=1); ang = np.degrees(np.arctan2(pts[:, 1], pts[:, 0])) % 360
out['qc_window'] = dict(n_vertices=len(h.vertices), area=h.volume, circumradius=[float(rad.min()), float(rad.max())],
                        vertex_angles_deg=sorted(np.round(ang, 6).tolist()))


def wsym(tl, rotdeg):
    M = tl['M']
    W = D.VOR[:, 2:] - D.VOR[:, :2] @ M.T
    c = math.cos(math.radians(rotdeg)); s = math.sin(math.radians(rotdeg))
    R = np.array([[c, -s], [s, c]])
    hh = ConvexHull(W); P0 = W[hh.vertices]
    P1 = P0 @ R.T
    d = max(np.min(np.linalg.norm(P0[:, None] - P1[None], axis=2), axis=1))
    return float(d)


U1 = D.GEN + np.eye(4, dtype=np.int64)
cells = []
for n in (6, 7):
    for kind in ('C4', 'C6'):
        L1 = np.linalg.matrix_power(U1, n) @ np.array([1, 0, 0, 0])
        L2 = np.linalg.matrix_power(D.GEN, 3 if kind == 'C4' else 2) @ L1
        tl = D.tiling(L1, L2, offset=np.array([0.31, -0.22, 0.47, 0.13]))
        v = D.validate(tl)
        k = v['kinds']; V = v['V']
        area = v['area']
        row = dict(n=n, kind=kind, V=V, density=V / area, tri_pv=k.get('triangle', 0) / V, sq_pv=k.get('square', 0) / V,
                   rh_pv=k.get('rhomb30', 0) / V, other=[x for x in k if x.startswith('other')], euler=v['euler'],
                   face_area_sum_minus_area=v['face_area_sum'] - area,
                   area_frac=dict(tri=k.get('triangle', 0) * s3 / 4 / area, sq=k.get('square', 0) / area, rh=k.get('rhomb30', 0) * 0.5 / area),
                   window_sym_err={'rot90': wsym(tl, 90), 'rot60': wsym(tl, 60), 'rot30': wsym(tl, 30)},
                   vertex_configs_gaps_deg=v['vconf'])
        cells.append(row)
        print(json.dumps({x: row[x] for x in ('n', 'kind', 'V', 'density', 'tri_pv', 'sq_pv', 'rh_pv', 'window_sym_err')}))
out['cells'] = cells
Path(__file__).with_name('geometry_check.json').write_text(json.dumps(out, indent=1), encoding='utf-8')
print(json.dumps(out['qc_window']))
print(json.dumps(cells[-1]['vertex_configs_gaps_deg'], indent=0))

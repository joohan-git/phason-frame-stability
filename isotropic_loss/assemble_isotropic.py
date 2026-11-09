"""Assemble isotropic_results.json from the step outputs (all mechanics numbers P; group items C)."""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames', 'phason_paths', 'penrose', 'symmetry')]  # shared modules
import json, glob, math
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
HERE = Path(__file__).parent; ROOT = HERE.parent
LAM = {'ab': 0.79984, 'pen': 0.8106}  # AB: midpoint of [0.79981, 0.79987] (final.md §0); Penrose:
R = dict(status='mixed: group theory C, mechanics/geometry P', lambda_QC_assumed=LAM)

# 1. AB sequence data (ab_families.json) re-read: c0 deficit per |delta|
seq = []
for r in json.loads((ROOT / 'phason_paths' / 'ab_families.json').read_text())['rows']:
    if r['V'] < 40:
        continue
    d = LAM['ab'] - r['harm']['c0']
    seq.append(dict(family=r['family'], pq=f"{r['p']}/{r['q']}", V=r['V'], delta=r['amp'], c0=r['harm']['c0'],
                    deficit=d, deficit_over_delta=d / r['amp'], delta_times_V=r['amp'] * r['V'],
                    deficit_band=[0.79981 - r['harm']['c0'], 0.79987 - r['harm']['c0']]))
R['ab_sequences_c0'] = seq

# 2. fixed-V primitive cells (fixed_size) + radius-2 census (patch_census_cells)
c2 = json.loads((HERE / 'patch_census_cells.json').read_text())
fx = {}
for cls in ('ab', 'pen'):
    cells = sorted([c for c in c2[cls]['cells'] if c['src'] == 'fixedV'], key=lambda c: c['delta'])
    D = np.array([c['delta'] for c in cells]); c0 = np.array([c['c0_mean'] for c in cells]); de = LAM[cls] - c0
    n2 = np.array([c['new2'] for c in cells]); n1 = np.array([c['new1'] for c in cells])
    fits = {}
    for nm, X in (('linear', np.c_[D]), ('lin+quad', np.c_[D, D ** 2]), ('quadratic', np.c_[D ** 2])):
        co, *_ = np.linalg.lstsq(X, de, rcond=None)
        fits[nm] = dict(coef=co.tolist(), rms=float(np.sqrt(np.mean((X @ co - de) ** 2))))
    X = np.c_[np.ones_like(D), -D]; co, *_ = np.linalg.lstsq(X, c0, rcond=None)
    fits['free_lambda_linear'] = dict(lambda_QC=float(co[0]), slope=float(co[1]), rms=float(np.sqrt(np.mean((X @ co - c0) ** 2))))
    X = np.c_[np.ones_like(D), -D ** 2]; co, *_ = np.linalg.lstsq(X, c0, rcond=None)
    fits['free_lambda_quadratic'] = dict(lambda_QC=float(co[0]), coef=float(co[1]), rms=float(np.sqrt(np.mean((X @ co - c0) ** 2))))
    fx[cls] = dict(V_range=[int(min(c['V'] for c in cells)), int(max(c['V'] for c in cells))], n_cells=len(cells), n_offsets=3,
                   cells=[dict(V=c['V'], delta=c['delta'], frac_m4=c['frac_m4'], c0_mean=c['c0_mean'], c0_sd=c['c0_sd'],
                               deficit=float(LAM[cls] - c['c0_mean']), deficit_over_delta=float((LAM[cls] - c['c0_mean']) / c['delta']),
                               new1=c['new1'], new2=c['new2'], deficit_per_new2=float((LAM[cls] - c['c0_mean']) / c['new2']) if c['new2'] else None,
                               L1=c['L1'], L2=c['L2']) for c in cells],
                   fits=fits, spearman_deficit_delta=float(spearmanr(de, D)[0]), spearman_deficit_new2=float(spearmanr(de, n2)[0]),
                   deficit_over_delta_range=[float((de / D).min()), float((de / D).max())],
                   deficit_per_new2_range=[float((de / n2).min()), float((de / n2).max())])
R['fixedV'] = fx

# 3. patch census over the wider cell sample
cen = {}
for cls in ('ab', 'pen'):
    cells = [c for c in c2[cls]['cells'] if c['src'] == 'census']
    D = np.array([c['delta'] for c in cells]); n1 = np.array([c['new1'] for c in cells]); n2 = np.array([c['new2'] for c in cells])
    cen[cls] = dict(n_cells=len(cells), delta_range=[float(D.min()), float(D.max())], atlas_r1=c2[cls]['atlas1'], atlas_r2=c2[cls]['atlas2'],
                    new1_over_delta_median=float(np.median(n1 / D)), new1_max=float(n1.max()),
                    new2_over_delta_median=float(np.median(n2 / D)),
                    new2_over_delta_by_bin={f'{lo}-{hi}': float(np.median((n2 / D)[(D >= lo) & (D < hi)])) for lo, hi in ((0.01, 0.03), (0.03, 0.05), (0.05, 0.08)) if ((D >= lo) & (D < hi)).any()})
deg = json.loads((HERE / 'patch_census.json').read_text())
for cls in ('ab', 'pen'):
    cen[cls]['degree_freq_fits'] = deg[cls]['deg_fits']; cen[cls]['qc_degree_freq'] = deg[cls]['qc_deg_freq']
R['patch_census'] = cen

# 4. mode structure
R['modes'] = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(HERE / 'critical_modes_*.json')))]
for m in R['modes']:
    m.pop('dq_small_branches', None)
# 5. groups
R['groups'] = json.loads((HERE / '../symmetry/point_groups.json').read_text())
R['sequence_census'] = json.loads((HERE / 'patch_census_sequences.json').read_text())
R['penrose_results_fixedV_note'] = 'p3_sizevsdelta fixedV_1592 = 2x2 supercell of a_3 (gcd of minors 2): 4 independent points, not 5'
R['cpu_hours_estimate'] = 'about 1.2 CPU-h total (fixedV 60 evaluations ~0.6, modes ~0.1, census ~0.5)'
(HERE / 'isotropic_results.json').write_text(json.dumps(R, indent=1), encoding='utf-8')
for cls in ('ab', 'pen'):
    print(cls, json.dumps({k: fx[cls][k] for k in ('fits', 'spearman_deficit_delta', 'spearman_deficit_new2', 'deficit_over_delta_range', 'deficit_per_new2_range')}))
    print(cls, json.dumps({k: v for k, v in cen[cls].items() if k not in ('degree_freq_fits', 'qc_degree_freq')}))
for s in seq:
    print(s['family'], s['pq'], s['V'], round(s['delta'], 5), round(s['deficit'], 6), round(s['deficit_over_delta'], 3), [round(x, 6) for x in s['deficit_band']])
for m in R['modes']:
    print(m['cls'], m['name'], m['V'], round(m['delta'], 5), 'lw1 spread', round(m['lw1_minus_c0_spread'], 5), 'gap2', round(m['gap_lw2_lw1_min'], 4), 'gapG', round(m['gap_gamma_lw'], 4))

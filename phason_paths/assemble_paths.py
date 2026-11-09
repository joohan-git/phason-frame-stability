"""Collect every r2 number into paths_results.json with status (C/P) and source script."""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames', 'slope_survey')]  # shared modules
import json, math
from pathlib import Path
import numpy as np
H = Path(__file__).parent
L = lambda f: json.loads((H / f).read_text(encoding='utf-8'))
out = dict(note='All numbers are P (point estimates on finite periodic cells, linear beam-column tangent, sampled k). '
                'No interval-certified (C) number exists yet. p_c in units EA*lambda_E = pi^2 EI/l^3 (stress protocol).',
           items=[])


def add(key, value, src, status='P', note=''):
    out['items'].append(dict(key=key, value=value, status=status, source=src, note=note))


# cross-validation with independent r1 (independent implementations)
cu = L('cusp.json')
rows = {r['name']: r for r in cu['rows']}
independent = {'B1_2/3': 7.78413, 'B1_1/2': 7.31409}
r1 = json.loads((H.parent / 'slope_survey' / 'slope_thresholds.json').read_text())['rows']
survey_13 = next(r for r in r1 if r['name'] == 'AB_1/3')['results']['stress']['pc_over_EAlamE']
add('xval_AB_1/3_EIunits', dict(main_survey_DQ0004=survey_13 * math.pi ** 2, independent=6.97703, rel_diff=survey_13 * math.pi ** 2 / 6.97703 - 1), 'slope_thresholds.py (r1, DQ=0.004, no Richardson) vs independent', note='r1 value carries the +0.3-0.5% DQ bias')  # independent r1 table (1/1 is the 8-fold image of 1/2), 2 elements, units EI/l^3
for k, v in independent.items():
    mine = rows[k]['pc_min_sampled'] * math.pi ** 2
    add(f'xval_{k}_EIunits', dict(main=mine, independent=v, rel_diff=mine / v - 1), 'cusp.py vs independent/slope_beam_check.py',
        note='independent code, different EI (2.25e-4 vs 1e-4); P, cross-validated')
for c in cu['element_convergence']:
    add(f"element_conv_{c['pq']}_sub{c['sub']}", c['pc_min'], 'cusp.py')
for n, r in rows.items():
    h = r['harm']
    add(f'cusp_{n}', dict(V=r['V'], a=r['a'], b=r['b'], delta=r['delta'], pc_min=r['pc_min_sampled'], c0=h['c0'], h2=h['h2'],
                          h4=h['h4'], pc_gamma=r['pc_gamma'], pc_k_half_half=r['pc_k_half_half']), 'cusp.py')
try:
    for r in L('pell.json')['rows']:
        add(f"pell_{r['pq']}", r, 'pell.py')
except FileNotFoundError:
    pass
# Pell cusp fit: total drop vs |b|, lambda_QC = c0(29/41)
try:
    pell = {r['pq']: r for r in L('pell.json')['rows']}
    lqc = pell[max(pell, key=lambda k: pell[k]['V'])]['c0']
    pts = [(abs(rows[n]['b'][0]), rows[n]['pc_min_sampled']) for n in ['B1_1/2', 'B1_2/3', 'B1_4/5', 'B1_7/10', 'B1_12/17']]
    pts += [(abs(r['b']), r['pc_min']) for k, r in pell.items() if k not in ('12/17',)]
    bx = np.array([a for a, _ in pts]); dy = lqc - np.array([b for _, b in pts])
    al, lc = np.polyfit(np.log(bx), np.log(dy), 1)
    add('pell_cusp_fit', dict(lambda_QC=lqc, alpha=al, prefactor=math.exp(lc), points=[[float(a), float(b)] for a, b in zip(bx, dy)]),
        'assemble_paths.py on cusp.json + pell.json', note='lambda_QC - p_c_min = prefactor*|b|^alpha; lambda_QC = c0 of largest Pell cell')
except Exception as e:
    add('pell_cusp_fit', str(e), 'assemble_paths.py')
# kappa4 summary
k4 = {n: r['harm']['h4'][0] / r['b'][0] for n, r in rows.items() if n.startswith('B1')}
k4.update({n: r['harm']['h4'][1] / r['b'][1] for n, r in rows.items() if n.startswith('B2')})
k2 = {n: abs(complex(*r['harm']['h2'])) / max(abs(complex(*r['a'])), 1e-12) for n, r in rows.items() if n[:4] in ('diag', 'E2i_', 'E2r_')}
add('kappa4_by_slope', k4, 'cusp.py', note='h4/b along B1 (real) and B2 (imag)')
add('kappa2_ratio_by_slope', k2, 'cusp.py', note='|h2|/|a| on E2-direction slopes')
# flips
fl = L('flips.json')['rows']
can = fl[0]['pc_lw_min']
add('flips_canonical_12_17', can, 'flips.py flips')
tab = {}
for r in fl[1:]:
    tab.setdefault(str(r['k']), []).append(dict(seed=r['seed'], moved=r['moved_fraction'], pc=min(r['pc_lw_min'], r['pc_gamma']),
                                                 drop=1 - min(r['pc_lw_min'], r['pc_gamma']) / can,
                                                 phason_ms=r['local_phason_strain_ms_R4'], perp_var=r['perp_var']))
add('flips_by_k', tab, 'flips.py flips')
x = np.array([d['moved'] for v in tab.values() for d in v]); y = np.array([d['drop'] for v in tab.values() for d in v])
m = (x > 0.01) & (x < 0.4) & (y > 1e-3)
beta, lnA = np.polyfit(np.log(x[m]), np.log(y[m]), 1)
add('flips_drop_vs_moved_powerlaw', dict(beta=beta, A=math.exp(lnA), n=int(m.sum())), 'assemble_paths.py on flips.json',
    note='fit drop = A * moved_fraction^beta for 0.01 < moved < 0.4')
fo = L('mode_followup.json')['rows']
add('localized_modes', {k: dict(pc=v['pc_gamma'], PR=v['participation_ratio'], pc_sub4=v['pc_gamma_sub4']) for k, v in fo.items()}, 'mode_followup.py')
add('planted', [dict(R_target=r['R_target'], R_core=r['R_core'], pc=r['pc_min_all'], mode_in_core=r['gamma_mode_weight_in_disk']) for r in L('planted.json')['rows']], 'flips.py plant')
add('planted_control_region_unbiased', [dict(seed=r['seed'], pc=r['pc_min_all'], moved=r['moved_fraction'], mode_in_region=r['gamma_mode_weight_in_disk']) for r in L('planted_control.json')['rows']], 'planted_control.py')
wi = L('window_excess.json')['rows']
from scipy.stats import spearmanr
ex = [v['mean_excess'] for v in wi.values()]; pc = [v['pc_gamma'] for v in wi.values()]
rho = spearmanr(ex, pc)
mv = [v['frac_vertices_outside_W'] for v in wi.values()]
rho_m = spearmanr(mv, pc)
add('moved_fraction_vs_pc', dict(spearman=rho_m.correlation, p=rho_m.pvalue, n=len(mv)), 'window_excess.py',
    note='comparison: plain 3D disorder measure (fraction of moved vertices = fraction outside W)')
add('window_excess_vs_pc', dict(spearman=rho.correlation, p=rho.pvalue, n=len(ex), points={k: [v['mean_excess'], v['pc_gamma']] for k, v in wi.items()}),
    'window_excess.py', note='mean internal-space distance outside the canonical window vs Gamma p_c')
for f in sorted((H / 'drift_check').glob('drift_*.json')):
    d = json.loads(f.read_text())
    add(f'ensemble_test_{f.stem}', {k: d.get(k) for k in ['V', 'dof', 'mode', 'seed', 'pc_min_all', 'pc_lw_min', 'pc_gamma', 'moved_fraction', 'sweeps_done',
                                                     'equilibrated', 'mc_seconds', 'eig_seconds_per_solve', 'total_seconds', 'peak_mem_MB']}, 'ensemble.py')
(H / 'paths_results.json').write_text(json.dumps(out, indent=1, default=float), encoding='utf-8')
print(json.dumps({i['key']: i['value'] for i in out['items'] if i['key'].startswith(('xval', 'kappa', 'flips_drop', 'window', 'pell'))}, indent=1, default=float))

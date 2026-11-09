"""Write slope_table.json: the deduplicated slope-space survey used by BOTH the text and Fig. 1c.
Dedup and the above-QC rule copy si_tables/make_values.py exactly (key = (pc_stress 4 d.p., sq, lam_strain 4 d.p.);
above QC = not an AB_ approximant row and pc_stress > p_QC (1 + 2 bias), bias = max AB_ pc_stress / p_QC - 1).
Adds the harmonic phason parts (|a| = E2, |b| = B1+B2) of each tiling for plotting. Approximant paths (main r3/r2,
Richardson pc_min_sampled) are stored separately under 'approximant_paths' and are NOT part of the survey count.
Usage: python -X utf8 slope_table.py"""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames',)]  # shared modules
import json
from pathlib import Path
import ab_frame as C
HERE = Path(__file__).resolve().parent
DATA = HERE
PATHS = HERE.parent / 'phason_paths'
it = {x['key']: x['value'] for x in json.loads((PATHS / 'paths_results.json').read_text(encoding='utf-8'))['items']}
tab = json.loads((DATA / 'summary_table.json').read_text(encoding='utf-8'))
qc = it['pell_cusp_fit']['lambda_QC']
bias = max(r['pc_stress'] for r in tab if r['name'].startswith('AB_')) / qc - 1
ns = vars(__import__('slope_thresholds'))
defs = {n: (a, b) for n, a, b in ns['SLOPES']}
for r in json.loads((HERE / 'slope_survey.json').read_text())['rows']:
    if 'L1' in r:
        defs[r['name']] = (r['L1'], r['L2'])
seen, rows = {}, []
for r in tab:
    k = (round(r['pc_stress'], 4), r['sq'], round(r['lam_strain'], 4))
    if k in seen:
        seen[k]['duplicates'].append(r['name']); continue
    a, b = C.harmonic_parts(C.phason_matrix(*defs[r['name']]))
    row = dict(name=r['name'], V=r['V'], sq=r['sq'], pc_stress=r['pc_stress'], lam_strain=r['lam_strain'],
               abs_a=abs(a), abs_b=abs(b), a=[a.real, a.imag], b=[b.real, b.imag],
               is_AB_approximant=r['name'].startswith('AB_'),
               above_QC=(not r['name'].startswith('AB_')) and r['pc_stress'] > qc * (1 + 2 * bias), duplicates=[])
    seen[k] = row; rows.append(row)
paths = []
for fn, fam_of in (('ab_families.json', lambda r: r['family']), ('cusp.json', lambda r: 'cusp_' + r['name'].split('_')[0])):
    for r in json.loads((PATHS / fn).read_text())['rows']:
        a = complex(*r['a']); b = complex(*r['b'])
        paths.append(dict(name=r.get('name', f"{r.get('family')}_{r.get('p')}/{r.get('q')}"), family=fam_of(r), V=r['V'],
                          abs_a=abs(a), abs_b=abs(b), pc=r['pc_min_sampled'], source=fn))
out = dict(script='slope_table.py', status='P', p_QC=qc, survey_bias=bias, margin='2 x bias',
           n_survey_rows=len(tab), n_unique=len(rows), n_above_QC=sum(r['above_QC'] for r in rows),
           above_QC=[r['name'] for r in rows if r['above_QC']],
           rule='dedup key (round(pc_stress,4), sq, round(lam_strain,4)); identical to the counts quoted in the text',
           survey=rows, approximant_paths=paths)
(DATA / 'slope_table.json').write_text(json.dumps(out, indent=1), encoding='utf-8')
print(len(tab), '->', len(rows), 'above', out['n_above_QC'], 'bias %.4f' % bias, out['above_QC'])

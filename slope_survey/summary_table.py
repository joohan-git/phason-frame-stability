import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames',)]  # shared modules
import json
from pathlib import Path
d = Path(__file__).parent
rows = json.load(open(d / 'slope_thresholds.json'))['rows'] + json.load(open(d / 'slope_survey.json'))['rows']
rows = [r for r in rows if 'error' not in r]
out = []
for r in rows:
    s, t = r['results']['strain'], r['results']['stress']
    out.append(dict(name=r['name'], V=r['V'], sq=round(r['square_fraction'], 4),
                    lam_strain=round(s['lam_over_lamE'], 4), Nmean=round(s['N_over_affine']['mean'], 3),
                    Nmax=round(s['N_over_affine']['max'], 3),
                    pc_strain=round(s['pc_over_EAlamE'], 4), pc_stress=round(t['pc_over_EAlamE'], 4),
                    finite_k_lower=s['lam_grid_over_lamE'] < s['lam_longwave_over_lamE']))
out.sort(key=lambda x: x['lam_strain'])
for o in out:
    print(o)
print('n slopes', len(out), 'finite-k critical:', sum(o['finite_k_lower'] for o in out))
print('min lam_strain', out[0]['name'], out[0]['lam_strain'])
print('min pc_strain', min(out, key=lambda o: o['pc_strain']))
print('min pc_stress', min(out, key=lambda o: o['pc_stress']))
print('max lam_strain among stretching-bulk (Nmean>0.7)', max((o for o in out if o['Nmean'] > 0.7), key=lambda o: o['lam_strain']))
print('max pc_stress', max(out, key=lambda o: o['pc_stress']))
print('dq', json.load(open(d / 'slope_survey.json'))['dq_convergence'])
json.dump(out, open(d / 'summary_table.json', 'w'), indent=1)

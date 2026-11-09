"""Compare the complete (objective) geometric stiffness with the axial-only operator on the same icosahedral tilings.
Reads the axial-only runs (runs/, runs_axial_large/), their symmetrized estimate (axial_objective_estimate.json) and the
complete-operator sections of icosa_results.json, and writes the sections axial_vs_objective_cubic, _tetra and _high_flip
into icosa_results.json. Run after assemble_icosa.py:

    python icosahedral/compare_operators.py
"""
import json
from pathlib import Path
import numpy as np
from load_axial_runs import load

HERE = Path(__file__).resolve().parent
CUBIC = ('cubic_1-1', 'cubic_2-1', 'cubic_3-2', 'cubic_5-3', 'cubic_8-5')
TETRA = ('tetra_2-1x3-2', 'tetra_3-2x2-1', 'tetra_3-2x5-3', 'tetra_5-3x3-2', 'tetra_5-3x8-5', 'tetra_8-5x5-3')
HIGH = {'3-2': 30000, '5-3': 100000, '8-5': 400000}   # highest flip level per size


def main():
    R, _ = load()
    ob = json.loads((HERE / 'axial_objective_estimate.json').read_text(encoding='utf-8'))
    path = HERE / 'icosa_results.json'
    out = json.loads(path.read_text(encoding='utf-8'))
    can = out['canonical']
    cubic = []
    for k in CUBIC:
        rr = [r for b, r in R.items() if b.startswith(k + '_s')]
        pc = [r['p_c'] for r in rr]
        G = [r['gamma'][0] for r in rr]
        km = [min(v[0] for v in r['kpts'].values()) for r in rr]
        e, c = ob[k], can[k]
        p2 = [float(np.mean(pc)), float(np.std(pc, ddof=1) / len(pc) ** .5)]
        cubic.append(dict(cell=k, V=rr[0]['V'], m=e['m'], n=len(rr), p_c_r2=p2, p_c_r3=c['p_c'], delta=c['delta_p_c'],
                          rel_delta=c['delta_p_c'] / p2[0], p_obj_r2=e['sym_min'][0], lam_iso_r2=e['lam_iso'][0],
                          lam_iso_r3=c['lam_iso'][0], gap_gamma_r2=float(np.mean(G) - np.mean(pc)), gap_gamma_r3=c['gap_gamma'],
                          gap_halfk_r2=float(np.mean(km) - np.mean(pc)), gap_halfk_r3=c['gap_halfk'], gamma_PR=c['gamma_pr'][0]))
    tetra = {k: dict(p_c_r2=ob[k]['lw_stored'] + [ob[k]['n']], p_c_r3=can[k]['p_c'], delta=can[k]['delta_p_c']) for k in TETRA}
    canon = {(b.split('_')[1], r['seed']): r for b, r in R.items() if r['kind'] == 'cubic' and 'flips_accepted' not in r}
    high = {}
    for pq, fl in HIGH.items():
        L = out['ladder'][f'{pq}_f{fl}']
        rr = [r for b, r in R.items() if b.startswith(f'cubic_{pq}_f{fl}_s')]
        rel = [r['p_c'] / canon[(pq, r['seed'])]['p_c'] - 1 for r in rr]
        high[pq] = dict(rel_r3=L['rel'], rel_r2=float(np.mean(rel)), p_c_r3=L['p_c'], p_c_r2=float(np.mean([r['p_c'] for r in rr])),
                        nodes_r3=L['nodes'], nodes_r2=float(np.mean([r['gamma_pr'] for r in rr])) * 4 * rr[0]['V'])
    out['axial_vs_objective_cubic'] = cubic
    out['axial_vs_objective_tetra'] = tetra
    out['axial_vs_objective_high_flip'] = high
    path.write_text(json.dumps(out, indent=1, default=float), encoding='utf-8')
    print(json.dumps(high, indent=1))


if __name__ == '__main__':
    main()

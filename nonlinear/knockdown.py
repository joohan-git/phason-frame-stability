import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames', 'phason_paths', 'objectivity', 'disorder')]  # shared modules
if __name__ == '__main__': __import__('os').chdir(__import__('os').path.dirname(__import__('os').path.abspath(__file__)))  # file names below are relative to this folder
import json, numpy as np
R = json.load(open('nonlinear_results.json'))
order = ['squareN12','B1_5_7','E2a_5_7','B2_5_7','B1_12_17','E2a_12_17','R22','R101','R102','R103','R104','R105']
lines = []
def cell(e, k):
    if k not in e: return '-'
    v = e[k]; return '%.4f' % v['mean'] + (' [%.4f, %.4f]' % (v['lo'], v['hi']) if v['n'] > 2 else '')
for shape in ('nodes', 'mode'):
    lines.append(f'\n**{shape}** (' + ('8 realizations, bootstrap 95 % CI on the mean' if shape == 'nodes' else 'both signs, mean of 2') + ')\n')
    lines.append('| frame | xi | p_05 | p_cap | limit points / n | first negative / n | p_lin |\n|---|---|---|---|---|---|---|')
    for nm in order:
        d = R['imperfection'][nm][shape]
        for xi in ('0.0001', '0.001', '0.01', '0.03'):
            e = d[xi]
            lines.append(f"| {nm} | {xi} | {cell(e,'p_05')} | {cell(e,'p_cap')} | {e.get('p_max',{}).get('n',0)}/{e['n']} | {e.get('p_first_neg',{}).get('n',0)}/{e['n']} | {e['p_lin']['mean']:.4f} |")
s = '\n'.join(lines)
kd = {}
for nm in order:
    o = R['imperfection'][nm]
    kd[nm] = {sh: {xi: dict(p05=100 * (o[sh][xi]['p_05']['mean'] / o[sh]['0.0001']['p_05']['mean'] - 1), pcap=100 * (o[sh][xi]['p_cap']['mean'] / o[sh]['0.0001']['p_cap']['mean'] - 1)) for xi in ('0.01', '0.03')} for sh in ('nodes', 'mode')}
print(json.dumps({k: {a: {b: {c: round(x, 2) for c, x in y.items()} for b, y in z.items()} for a, z in v.items()} for k, v in kd.items()}))
sw = {nm: {sh: {k: R['imperfection'][nm]['southwell_exploratory'][sh][k] for k in ('C', 'A', 'xi_intrinsic', 'max_rel_resid')} for sh in ('mode',)} for nm in order}
print(json.dumps(sw))
R['relative_knockdown_pct_vs_xi_1e-4'] = kd
json.dump(R, open('nonlinear_results.json', 'w'), indent=1, default=float)
P = json.load(open('out/perfect_B1_12_17_s2.json'))['modes'][1]['branches']['1']['rows']
k = int(np.argmax([r['p'] for r in P])); print('12/17 B1 channel limit', {a: round(P[k][a], 4) for a in ('p', 'rms_chord', 'max_chord', 'rms_edge', 'max_edge', 'edev', 'A_rel', 'max_tau')})
P = json.load(open('out/perfect_B1_12_17_s2.json'))['modes'][0]['branches']['1']['rows']; print('12/17 e3 + first rows', [(round(r['p'], 4), round(r['rms_edge'], 4), r['neg']) for r in P[:12:2]])
for t, i in (('B1_5_7_s2_S1-1160', 2), ('B1_5_7_s2_S1-180', 2)):
    m = json.load(open(f'out/perfect_{t}.json'))['modes'][i]['branches']
    print(t, {s: (m[s]['end'], len(m[s]['rows'])) for s in m}, m['1']['rows'][-1].get('status'))
print('natural B1_5_7 mode', json.load(open('out/perfect_B1_5_7_s2.json'))['modes'][0]['mu'])

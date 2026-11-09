"""Adds gates, scorecard and headline to nonlinear_results.json (after assemble_nonlinear.py and knockdown.py); prints numbers quoted in."""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames', 'phason_paths', 'objectivity', 'disorder')]  # shared modules
if __name__ == '__main__': __import__('os').chdir(__import__('os').path.dirname(__import__('os').path.abspath(__file__)))  # file names below are relative to this folder
import json
import numpy as np
R = json.load(open('nonlinear_results.json'))
gc = json.load(open('exact_references.json'))
R['gates'] = dict(
    a=dict(status='PASS', fixed_vs_gamma_max=1.3e-13, free_macro_vs_longwave_max=2.6e-6, scaling='p_c/p_lin-1 = 1.10 lambda_E at r=0.03 and r=0.003'),
    b=dict(status='PASS', gradient_max=1.2e-9, tangent_max=8.8e-11),
    c=dict(status='PASS', exact_square_pc_C=gc['c3_linear_exact']['pc'], exact_square_b2_C=gc['c3_b2_exact_per_rad2'],
           solver_vs_exact=[dict(sub=x['sub'], r=x['r'], max_rel_err=max(abs(e['rel_err']) for e in x['errs'])) for x in gc['c3_solver']],
           checker_limits='r=0.003 unreachable; sub16 r=0.03 leaves symmetry beyond 15 deg'),
    d=dict(ordered='PASS', random='FAIL at sub 2 (p_lin -1.04 %, limit -1.3 %); sub 4 converged (sub 8: -0.11 %, +0.35 %)'))
R['scorecard'] = dict(H1='partly failed (B1 12/17 limit and inertia; B2/E2a no sharp bifurcation)', H2='passed B1 and square; failed B2/E2a',
                      H3='mostly failed (no bifurcation at p_lin; 1.2 factor met by 2/6)', H4='failed on beta; mixed on relative sensitivity; absolute lower holds',
                      H5='passed; B1 5/7 m=16 q mode unresolved', H6='passed')
names = ('R22', 'R101', 'R102', 'R103', 'R104', 'R105')
s4 = [R['natural'][f'{n}_s4'] for n in names]
L = np.array([v['first_limit']['p'] for v in s4]); Li = np.array([v['lin'] for v in s4])
R['headline'] = dict(random_s4=dict(p_lin=Li.tolist(), first_limit=L.tolist(), limit_mean=float(L.mean()), lin_mean=float(Li.mean()), ratio_mean=float((L / Li).mean())),
                     B1_12_17_s4=dict(p_lin=0.79643336, p_c=0.798214), post_limit='not resolved')
json.dump(R, open('nonlinear_results.json', 'w'), indent=1, default=float)
print('limit mean', L.mean(), 'lin mean', Li.mean(), 'ratios', L.mean() / 0.798214, Li.mean() / 0.79643)
K = R['relative_knockdown_pct_vs_xi_1e-4']
for n in names + ('B1_5_7', 'B1_12_17', 'B2_5_7', 'E2a_5_7', 'E2a_12_17', 'squareN12'):
    print(n, {s: {x: (round(K[n][s][x]['p05'], 1), round(K[n][s][x]['pcap'], 1)) for x in K[n][s]} for s in K[n]})
errs = sum(e['n_errors'] for o in R['imperfection'].values() for s in ('mode', 'nodes') for e in o[s].values())
tot = sum(e['n'] for o in R['imperfection'].values() for s in ('mode', 'nodes') for e in o[s].values())
print('imperf runs', tot, 'errors', errs)
print('p_cap nodes random range', min(R['imperfection'][n]['nodes'][x]['p_cap']['mean'] for n in names for x in R['imperfection'][n]['nodes']), max(R['imperfection'][n]['nodes'][x]['p_cap']['mean'] for n in names for x in R['imperfection'][n]['nodes']))
print('PR first', {n: round(R['perfect'][f'{n}_s2']['lin_PR'][0], 4) for n in names})
print('p_05 node random min', {n: round(R['imperfection'][n]['nodes']['0.0001']['p_05']['mean'], 3) for n in names})
print('solver', R['gates']['c']['solver_vs_exact'])

"""-> twelvefold_results.json.  Applies the statistics pre-declared in run_twelvefold.py header (P numbers)."""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames', 'phason_paths')]  # shared modules
import json, math
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
rows = [json.loads(l) for l in (HERE / 'runs.jsonl').read_text().splitlines() if l.strip()]
for extra in sorted((HERE / 'd12_out').glob('task_*.json')) if (HERE / 'd12_out').exists() else []:
    rows.append(json.loads(extra.read_text()))
if (HERE / 'order7_step.jsonl').exists():
    for l in (HERE / 'order7_step.jsonl').read_text().splitlines():
        if l.strip():
            r = json.loads(l); r['sub'] = 20; r.setdefault('lam_gamma', float('nan')); r['pc_min'] = min(r['pc_phi']); r.setdefault('euler', 0)
            rows.append(r)
sym = json.loads((HERE / 'twelvefold_representation.json').read_text())
geom = json.loads((HERE / 'geometry_check.json').read_text()) if (HERE / 'geometry_check.json').exists() else None

groups = {}
for r in rows:
    groups.setdefault((r['fam'], r['n'], r['kind'], r['sub']), []).append(r)

table = []
for (fam, n, kind, sub), rs in sorted(groups.items()):
    dl = complex(*rs[0]['alpha']) if kind == 'C4' else complex(*rs[0]['beta'])
    other = complex(*rs[0]['beta']) if kind == 'C4' else complex(*rs[0]['alpha'])
    h4 = np.array([complex(*r['harm']['h4']) for r in rs])
    h2 = np.array([complex(*r['harm']['h2']) for r in rs]); h6 = np.array([complex(*r['harm']['h6']) for r in rs])
    c0 = np.array([r['harm']['c0'] for r in rs]); pm = np.array([r['pc_min'] for r in rs])
    mh = h4.mean()
    se = (np.std(h4.real, ddof=1) ** 2 + np.std(h4.imag, ddof=1) ** 2) ** 0.5 / math.sqrt(len(rs)) if len(rs) > 1 else float('nan')
    kappa2 = mh / (dl.conjugate() ** 2)  # second-order covariant form h4 = kappa2 * conj(delta)^2
    lgam = np.array([r['lam_gamma'] for r in rs])
    kappa = mh / dl  # equivariance: h4 = kappa * delta (complex), expect kappa real & delta-independent
    table.append(dict(fam=fam, n=n, kind=kind, sub=sub, n_offsets=len(rs), V=[r['V'] for r in rs],
                      delta=[dl.real, dl.imag], abs_delta=abs(dl), arg_delta_deg=math.degrees(math.atan2(dl.imag, dl.real)),
                      other_pair_abs=abs(other),
                      mean_h4=[mh.real, mh.imag], abs_mean_h4=abs(mh), se_mean_h4=se, mean_abs_h4=float(np.abs(h4).mean()),
                      H_hull=abs(mh) / abs(dl), H_abs=float(np.abs(h4).mean()) / abs(dl),
                      kappa=[kappa.real, kappa.imag], arg_h4_minus_arg_delta_deg=math.degrees(math.atan2(kappa.imag, kappa.real)),
                      kappa2=[kappa2.real, kappa2.imag], abs_kappa2=abs(kappa2), H2_hull=abs(mh) / abs(dl) ** 2,
                      arg_h4_minus_arg_conjdelta2_deg=math.degrees(math.atan2(kappa2.imag, kappa2.real)),
                      lam_gamma_minus_c0_mean=float(np.mean(lgam - c0)), lam_gamma_minus_pcmin_mean=float(np.mean(lgam - pm)),
                      abs_mean_h2=abs(h2.mean()), mean_abs_h2=float(np.abs(h2).mean()), mean_abs_h6=float(np.abs(h6).mean()),
                      c0_mean=float(c0.mean()), c0_sd=float(c0.std(ddof=1)) if len(rs) > 1 else 0.0,
                      pcmin_mean=float(pm.mean()), pcmin_range=[float(pm.min()), float(pm.max())],
                      lam_gamma_mean=float(np.mean([r['lam_gamma'] for r in rs])),
                      fit_rms_max=max(r['harm']['fit_rms'] for r in rs), max_seconds=max(r['seconds'] for r in rs),
                      euler_all_zero=all(r['euler'] == 0 for r in rs)))


def get(fam, kind, sub=2):
    return sorted([t for t in table if t['fam'] == fam and t['kind'] == kind and t['sub'] == sub], key=lambda t: t['n'])


def slope(ts, key):
    x = np.log([t['abs_delta'] for t in ts]); y = np.log([t[key] for t in ts])
    if len(ts) < 2:
        return None
    A = np.c_[x, np.ones_like(x)]
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(c[0])


import sys as _s
NSET = _s.argv[1] if len(_s.argv) > 1 else '3-6'
def famlist(kind):
    base = [t for t in get('A', kind) if t['n'] >= 3]
    if NSET == '3-6':
        return [t for t in base if t['n'] <= 6]
    if NSET == '3-7scaled':
        sc = get('A', kind, 20)
        return [t for t in base if t['n'] <= 6] + sc
    return base  # '3-7prod'
A4 = famlist('C4'); A6 = famlist('C6')
verd = {}
s1 = slope(A4, 'abs_mean_h4'); s1b = slope(A4, 'mean_abs_h4')
H3 = next((t['H_hull'] for t in A4 if t['n'] == 3), None); Hlast = A4[-1]['H_hull'] if A4 else None
p1 = (s1 is not None and 0.8 <= s1 <= 1.2 and Hlast > 0.3 * H3)
verd['P1'] = dict(rule='slope(|<h4>| vs |alpha|, A n>=3) in [0.8,1.2] and H_hull(last) > 0.3 H_hull(n=3)',
                  slope_primary=s1, slope_secondary=s1b, H_hull_by_n={t['n']: t['H_hull'] for t in A4},
                  H_abs_by_n={t['n']: t['H_abs'] for t in A4}, n_last=A4[-1]['n'] if A4 else None, PASS=bool(p1))
s2 = slope(A6, 'abs_mean_h4'); s2b = slope(A6, 'mean_abs_h4')
floor_ok = all(t6['abs_mean_h4'] < 1e-3 * t4['abs_mean_h4'] for t4, t6 in zip(A4, A6))
const_fail = (s2 is not None and 0.8 <= s2 <= 1.2 and all(t6['H_hull'] > 0.1 * t4['H_hull'] for t4, t6 in zip(A4, A6)))
p2 = ((s2 is not None and s2 >= 1.5) or floor_ok) and not const_fail
verd['P2'] = dict(rule='|<h4>|/|beta| -> 0: slope >= 1.5 or |<h4>| < 1e-3 x P1 |<h4>| at same size; FAIL if ratio -> nonzero const',
                  slope_primary=s2, slope_secondary=s2b, H_hull_by_n={t['n']: t['H_hull'] for t in A6},
                  H_abs_by_n={t['n']: t['H_abs'] for t in A6},
                  ratio_C6_over_C4_abs_mean_h4={t4['n']: t6['abs_mean_h4'] / t4['abs_mean_h4'] for t4, t6 in zip(A4, A6)},
                  floor_criterion_met=bool(floor_ok), constant_ratio_failure=bool(const_fail), PASS=bool(p2))
verd['n_set'] = NSET
verd['h2_floor_marker'] = {f"{t['kind']}{t['n']}": dict(abs_mean_h2=t['abs_mean_h2'], abs_mean_h4=t['abs_mean_h4']) for t in A4 + A6}
verd['P3'] = dict(rule='P1 and P2', PASS=bool(p1 and p2),
                  AB_reference='independent flip ladder: AB first-order splitter = (c,d) B1+B2, conformal E2 second order')
# isotropic part: c0 vs |delta| for both pairs; lambda_inf estimate from the smallest-delta C4/C6 points
iso = {}
for kind in ('C4', 'C6'):
    ts = get('A', kind)
    iso[kind] = [dict(n=t['n'], abs_delta=t['abs_delta'], c0=t['c0_mean'], c0_sd=t['c0_sd'], pcmin=t['pcmin_mean']) for t in ts]
lam_inf_est = None
if A4 and A6:
    lam_inf_est = 0.5 * (A4[-1]['c0_mean'] + A6[-1]['c0_mean'])
    for kind, ts in (('C4', A4), ('C6', A6)):
        x = np.array([t['abs_delta'] for t in ts]); y = np.array([t['c0_mean'] for t in ts])
        A = np.c_[x, np.ones_like(x)]
        c, *_ = np.linalg.lstsq(A, y, rcond=None)
        iso[kind + '_linear_fit'] = dict(slope_dc0_dabsdelta=float(c[0]), intercept=float(c[1]), n_used=[t['n'] for t in ts])
# element convergence
conv = []
for t4 in [t for t in table if t['sub'] == 4]:
    t2 = next(t for t in table if t['sub'] == 2 and t['fam'] == t4['fam'] and t['n'] == t4['n'] and t['kind'] == t4['kind'])
    r2 = [r for r in groups[(t4['fam'], t4['n'], t4['kind'], 2)] if r['seed'] in (1, 2)]
    r4 = groups[(t4['fam'], t4['n'], t4['kind'], 4)]
    d = []
    for a in r4:
        b = next(x for x in r2 if x['seed'] == a['seed'])
        d.append(dict(seed=a['seed'], c0_2=b['harm']['c0'], c0_4=a['harm']['c0'], h4_2=b['harm']['h4'], h4_4=a['harm']['h4'],
                      rel_h4_change=abs(complex(*a['harm']['h4']) - complex(*b['harm']['h4'])) / max(abs(complex(*a['harm']['h4'])), 1e-30)))
    conv.append(dict(n=t4['n'], kind=t4['kind'], pairs=d))
res = dict(status='P (numerics) / C (representation, twelvefold_representation.json; exact tile frequencies, geometry_check.json)',
           sources=['twelvefold_core.py', 'run_twelvefold.py', 'analyze.py', 'twelvefold_representation.py', 'geometry_check.py', 'zone_centre_check.py'],
           conventions='S = aI + bJ + cZ + dX (internal = S physical); alpha = a+ib (m=4, E4), beta = c+id (m=6, B1+B2); '
                       'lambda(phi) = c0 + sum Re[h_m e^{-i m phi}], p_c in units of EA lambda_E, stress protocol, r = 0.03',
           representation=dict(ab_block_R120_m4=sym['ab_block_equals_R120'], cd_block_R180_m6=sym['cd_block_equals_R180'],
                               AB_ab_m2=sym['AB_ab_block_R90_m2'], AB_cd_m4=sym['AB_cd_block_R180_m4'],
                               phonon_m=[0, 2], R_phonon_phason='0 (Schur)'),
           verdicts=verd, isotropic=dict(by_kind=iso, lam_inf_est_from_smallest=lam_inf_est), element_convergence=conv,
           table=table, geometry=dict(exact=geom['exact'] if geom else None,
                                      largest_cells=[{k: c[k] for k in ('n', 'kind', 'V', 'density', 'tri_pv', 'sq_pv', 'rh_pv', 'window_sym_err')} for c in geom['cells']] if geom else None,
                                      vertex_configs=geom['cells'][-1]['vertex_configs_gaps_deg'] if geom else None))
(HERE / ('twelvefold_results.json' if NSET == '3-6' else f'twelvefold_results_{NSET}.json')).write_text(json.dumps(res, indent=1), encoding='utf-8')
for t in table:
    print(t['fam'], t['n'], t['kind'], t['sub'], 'N=%d' % t['n_offsets'], 'V~%d' % t['V'][0], '|d|=%.3e arg=%.0f' % (t['abs_delta'], t['arg_delta_deg']),
          '|<h4>|=%.3e se=%.1e <|h4|>=%.3e Hh=%.3f Ha=%.3f H2=%.1f argk=%.0f argk2=%.0f gam-c0=%.1e c0=%.6f+-%.1e |h6|=%.1e |h2|=%.1e' % (
              t['abs_mean_h4'], t['se_mean_h4'], t['mean_abs_h4'], t['H_hull'], t['H_abs'], t['H2_hull'], t['arg_h4_minus_arg_delta_deg'], t['arg_h4_minus_arg_conjdelta2_deg'], t['lam_gamma_minus_c0_mean'], t['c0_mean'], t['c0_sd'], t['mean_abs_h6'], t['mean_abs_h2']))
print(json.dumps(verd, indent=1)); print(json.dumps(res['isotropic'], indent=1)); print(json.dumps(conv, indent=1))

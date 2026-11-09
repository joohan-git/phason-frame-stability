"""Fit the bifurcated branch with the nonlinear form p/p_c - 1 = b1 x + b2 x^2, x in [0.0015, 0.02],
for x = raw rms edge chord rotation and x = rms relative to the fundamental state; also the
small-amplitude limit b2 = lim (p/p_c - 1)/x_rel^2. usage: python fit_b2.py <tag> [<tag> ...]"""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames', 'phason_paths', 'disorder')]  # shared modules
import sys, json
from pathlib import Path
import numpy as np
HERE = Path(__file__).parent
for tag in sys.argv[1:]:
    r = json.loads((HERE / f"{tag}.json").read_text(encoding='utf-8'))
    p_c = r['first_critical']['p_c']
    print(f"== {tag}: p_c = {p_c:.6f}, p_lin(secant) = {r['p_lin']:.6f}, p_lin(classical) = {r.get('p_lin_classical', float('nan')):.6f}")
    for sgn, b in r['branches'].items():
        P = np.array([s['p'] for s in b['path']]); y = P / p_c - 1
        for key in ('rms', 'rms_rel'):
            x = np.array([s[key] for s in b['path']])
            for lo, hi in ((0.0015, 0.02), (0.0015, 0.01), (0.0, 0.03)):
                m = (x >= lo) & (x <= hi)
                if m.sum() < 3:
                    continue
                A = np.c_[x[m], x[m] ** 2]
                coef, *_ = np.linalg.lstsq(A, y[m], rcond=None)
                A2 = np.c_[x[m] ** 2]
                c2, *_ = np.linalg.lstsq(A2, y[m], rcond=None)
                print(f"  {sgn} x={key:8s} window [{lo},{hi}] n={m.sum():2d}: b1={coef[0]:+.3e} b2={coef[1]:.4f} | pure-quadratic b2={c2[0]:.4f}")
        xr = np.array([s['rms_rel'] for s in b['path']])
        ratio = y / xr ** 2
        n = min(4, len(xr)); A = np.c_[np.ones(n), xr[:n] ** 2]
        coef, *_ = np.linalg.lstsq(A, ratio[:n], rcond=None)
        print(f"  {sgn} small-amplitude limit (p/p_c-1)/x_rel^2 -> {coef[0]:.4f}; first ratios {np.round(ratio[:4], 4).tolist()} at x_rel {np.round(xr[:4], 4).tolist()}")
        em = [s.get('max_elem_chord') for s in b['path']]
        if em[0] is not None:
            for tgt in (0.35,):
                for a_, b_ in zip(b['path'][:-1], b['path'][1:]):
                    if a_['max_elem_chord'] < tgt <= b_['max_elem_chord']:
                        t = (tgt - a_['max_elem_chord']) / (b_['max_elem_chord'] - a_['max_elem_chord'])
                        print(f"  {sgn} p at max ELEMENT chord {tgt}: {a_['p'] + t * (b_['p'] - a_['p']):.5f} (edge max there {a_['maxrot'] + t * (b_['maxrot'] - a_['maxrot']):.4f})")
        print(f"  {sgn} p at edge max 0.35: {b.get('p_at_maxrot_035')}, p_end {b['p_end']:.5f}, lam_min on branch {b['lam_min_on_branch']:.2e}")

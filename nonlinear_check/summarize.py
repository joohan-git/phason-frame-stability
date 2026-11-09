"""Summarise nonlinear_check results (reads <case>_sub<sub>.json written by run_cases.py)."""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames', 'phason_paths', 'disorder')]  # shared modules
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).parent


def b2_small(branch, key='rms_rel', nmax=4):
    """Small-amplitude limit of (p-p_c)/x^2: fit ratio = b2 + c x^2 on the first nmax branch points."""
    pts = branch['path'][:nmax]
    p_c = None
    return pts


def summarise(tag):
    f = HERE / f"{tag}.json"
    if not f.exists():
        return None
    r = json.loads(f.read_text(encoding='utf-8'))
    out = dict(tag=tag, V=r['V'], ndof=r['ndof'], p_lin=r['p_lin'], p_lin_next=r['p_lin_next'], event=r['fund_event'],
               lam_min_fund=r['lam_min_along_fund'])
    fc = r.get('first_critical')
    if fc:
        out['first_critical'] = {k: fc[k] for k in fc if k in ('p_c', 'kind', 'rms_fund', 'maxrot_fund', 'cos_mode_load', 'p_bracket', 'p_max_path', 'rms_at', 'maxrot_at', 'lam_at')}
    out['knee_0.05'] = r.get('knee_0.05')
    out['p_at_0.05_interp'] = r.get('p_at_maxrot_0.05_fund_interp')
    if 'branches' in r:
        p_c = fc['p_c']
        bs = {}
        for sgn, b in r['branches'].items():
            xs = np.array([s['rms_rel'] for s in b['path']]); ys = np.array([s['p'] for s in b['path']]) - p_c
            ratio = ys / xs ** 2
            # Richardson on the first points: ratio(x) = b2 + c x^2
            n = min(4, len(xs))
            A = np.c_[np.ones(n), xs[:n] ** 2]
            coef, *_ = np.linalg.lstsq(A, ratio[:n], rcond=None)
            xs_raw = np.array([s['rms'] for s in b['path']])
            bs[sgn] = dict(b2_small_rel=float(coef[0]), ratios_rel=[(float(x), float(q)) for x, q in zip(xs[:6], ratio[:6])],
                           ratios_raw=[(float(x), float(q)) for x, q in zip(xs_raw[:6], (ys / xs_raw ** 2)[:6])],
                           lstsq_fit_rel=b.get('fit_rel'), limit_point=b.get('limit_point'), p_at_maxrot_035=b.get('p_at_maxrot_035'),
                           p_end=b['p_end'], maxrot_end=b['reached_maxrot'], lam_min_on_branch=b['lam_min_on_branch'],
                           monotone_p=bool(np.all(np.diff(ys) > 0)))
        out['branches'] = bs
    out['seconds'] = r['seconds']
    return out


if __name__ == '__main__':
    tags = sys.argv[1:] or [p.stem for p in sorted(HERE.glob('*_sub*.json'))]
    for t in tags:
        s = summarise(t)
        print(json.dumps(s, indent=1, default=float))

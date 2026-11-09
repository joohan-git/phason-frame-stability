"""Assemble nonlinear_results.json from out/. All numbers P unless labelled C."""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames', 'phason_paths', 'objectivity', 'disorder')]  # shared modules
import json, glob, math, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
OUT = HERE / 'out'


def fit_branch(rows_pos, rows_neg, key, amax, amin):
    a, y = [], []
    for sg, rows in ((1, rows_pos), (-1, rows_neg)):
        for r in rows or []:
            if key in r and amin <= r[key] <= amax:
                a.append(sg * r[key]); y.append(r['p'])
    a = np.array(a); y = np.array(y)
    return a, y


def branch_summary(b, pc):
    if not b or 'rows' not in b:
        return dict(failed=True, why=(b or {}).get('error'))
    rows = b['rows']; p = np.array([r['p'] for r in rows]); neg = np.array([r['neg'] for r in rows])
    psi = np.array([r['max_chord'] for r in rows]); rms = np.array([r['rms_chord'] for r in rows])
    out = dict(steps=len(rows), end=b['end'], returned=b['returned_to_fundamental'], reseed=b['reseed_factor'], p_start=float(p[0]), p_end=float(p[-1]), p_peak=float(p.max()),
               psi_max_end=float(psi[-1]), psi_rms_end=float(rms[-1]), edev_end=rows[-1]['edev'], A_rel_end=rows[-1]['A_rel'], neg0=int(neg[0]), neg_end=int(neg[-1]), neg_peak=int(neg.max()))
    dp = np.diff(p); idx = np.where(dp < 0)[0]
    out['first_drop'] = None if not len(idx) else dict(p=float(p[idx[0]]), psi_max=float(psi[idx[0]]), psi_rms=float(rms[idx[0]]), neg=int(neg[idx[0]]))
    j = np.where(neg > neg[0])[0]
    out['first_neg_rise'] = None if not len(j) else dict(p=float(p[j[0]]), psi_max=float(psi[j[0]]), psi_rms=float(rms[j[0]]), edev=rows[j[0]]['edev'], neg=int(neg[j[0]]))
    for t in (0.02, 0.05, 0.1):
        k = np.where(rms >= t)[0]
        out[f'p_at_rms_{t}'] = None if not len(k) else float(np.interp(t, rms[:k[0] + 1], p[:k[0] + 1]))
    if 'rms_edge' in rows[0]:
        re_ = np.array([r['rms_edge'] for r in rows])
        for t in (0.02, 0.05, 0.1):
            k = np.where(re_ >= t)[0]
            out[f'p_at_rms_edge_{t}'] = None if not len(k) else float(np.interp(t, re_[:k[0] + 1], p[:k[0] + 1]))
    return out


def fit_mode(mode, localized):
    pc = mode['pc']; bp = mode['branches'].get('1') or {}; bn = mode['branches'].get('-1') or {}
    key, amax, amin = ('max_chord', 0.05, 0.003) if localized else ('rms_chord', 0.02, 0.0015)
    a, y = fit_branch(bp.get('rows'), bn.get('rows'), key, amax, amin)
    res = dict(key=key, window=[amin, amax], npts=len(a))
    if len(a) >= 3 and (a > 0).any() and (a < 0).any():
        A = np.c_[a, a * a]
        c, *_ = np.linalg.lstsq(A, y / pc - 1, rcond=None)
        res.update(b1=float(c[0]), b2=float(c[1]), fit_rms=float(np.sqrt(np.mean((A @ c - (y / pc - 1)) ** 2))))
    # mesh-independent amplitude (edge chord), added for gate (d)
    ke, amaxe, amine = ('max_edge', 0.05, 0.003) if localized else ('rms_edge', 0.02, 0.0015)
    a, y = fit_branch(bp.get('rows'), bn.get('rows'), ke, amaxe, amine)
    res['edge'] = dict(key=ke, npts=len(a))
    if len(a) >= 3 and (a > 0).any() and (a < 0).any():
        A = np.c_[a, a * a]
        c, *_ = np.linalg.lstsq(A, y / pc - 1, rcond=None)
        res['edge'].update(b1=float(c[0]), b2=float(c[1]), fit_rms=float(np.sqrt(np.mean((A @ c - (y / pc - 1)) ** 2))))
    return res


def natural_summary(d):
    rows = d['rows']; p = np.array([r['p'] for r in rows]); psi = np.array([r['max_chord'] for r in rows]); rms = np.array([r['rms_chord'] for r in rows])
    A = np.array([r['A_rel'] for r in rows]); neg = np.array([r['neg'] for r in rows]); mu = np.array([r['mu'][0] if r.get('mu') else np.nan for r in rows])
    tau = np.array([r['max_tau'] for r in rows])
    out = dict(steps=len(rows), end=d['end'], p_end=float(p[-1]), psi_max_end=float(psi[-1]), lin=d['lin'][0], sec=d['sec'])
    dp = np.diff(p); idx = np.where((dp[:-1] > 0) & (dp[1:] <= 0))[0] + 1
    first = np.where(dp <= 0)[0]
    if len(first):
        k = first[0]
        out['first_limit'] = dict(p=float(p[k]), psi_max=float(psi[k]), psi_rms=float(rms[k]), tau=float(tau[k]), A_rel=float(A[k]), edev=rows[k]['edev'], p_over_lin=float(p[k] / d['lin'][0]))
    else:
        out['first_limit'] = None
    out['local_maxima'] = [float(p[i]) for i in idx][:30]
    # peaks on stable segments only
    out['stable_peak_p'] = float(p[neg == 0].max()) if (neg == 0).any() else None
    i0 = int(np.nanargmin(np.abs(mu))) if np.isfinite(mu).any() else None
    j = np.where(neg > 0)[0]
    out['first_neg'] = None if not len(j) else dict(p=float(p[j[0]]), psi_max=float(psi[j[0]]), psi_rms=float(rms[j[0]]))
    lin = d['lin'][0]
    k = np.where(p >= lin)[0]
    out['state_at_p_lin'] = None if not len(k) else dict(psi_max=float(psi[k[0]]), psi_rms=float(rms[k[0]]), mu=float(mu[k[0]]), neg=int(neg[k[0]]), A_rel=float(A[k[0]]))
    # min mu before first limit
    lim = first[0] + 1 if len(first) else len(p)
    if np.isfinite(mu[:lim]).any():
        i = int(np.nanargmin(mu[:lim])); out['mu_min_before_limit'] = dict(mu=float(mu[i]), p=float(p[i]), psi_max=float(psi[i]))
    for t in (0.05, 0.1, 0.2, 0.35):
        k = np.where(psi >= t)[0]
        out[f'p_at_psimax_{t}'] = None if not len(k) else float(np.interp(t, psi[:k[0] + 1], p[:k[0] + 1]))
    if 'max_edge' in rows[0]:
        pe = np.array([r['max_edge'] for r in rows])
        for t in (0.05, 0.1, 0.2):
            k = np.where(pe >= t)[0]
            out[f'p_at_edge_{t}'] = None if not len(k) else float(np.interp(t, pe[:k[0] + 1], p[:k[0] + 1]))
    # tangent bulk modulus relative to initial: dp / d(-A_rel)
    if len(p) > 5:
        K0 = p[1] / max(-A[1], 1e-30); out['K_secant_initial'] = float(K0)
        if len(first):
            k = first[0]; out['K_secant_at_limit_over_K0'] = float((p[k] / -A[k]) / K0)
    return out


def main():
    res = dict(status='P unless labelled C', units='p in pi^2 EI / ell^3; p true hydrostatic pressure; r = 0.03', perfect={}, natural={})
    for f in sorted(OUT.glob('perfect_*.json')):
        d = json.loads(f.read_text())
        loc = d['name'].startswith('R')
        ms = []
        for m in d['modes']:
            ms.append(dict(imode=m['imode'], lin=m['lin'], pc=m['pc'], shift_per_lamE=m['shift_per_lamE'], mu=m['mu'], kind=m.get('kind'), zero_found=m['zero_found'], neg_at_pc=m['neg'], macro=m['macro'], PR=m['PR'],
                           fit=fit_mode(m, loc), plus=branch_summary(m['branches'].get('1'), m['pc']), minus=branch_summary(m['branches'].get('-1'), m['pc'])))
        res['perfect'][d['tag']] = dict(nn=d['nn'], lin=d['lin'][:6], lin_macro=d.get('lin_macro'), lin_PR=d.get('lin_PR'), modes=ms, seconds=d.get('seconds'))
    for f in sorted(OUT.glob('natural_*.json')):
        d = json.loads(f.read_text())
        res['natural'][f.stem.replace('natural_', '')] = natural_summary(d)
    res['imperfection'] = imperf()
    extra = HERE / 'gates_d.json'
    (HERE / 'nonlinear_results.json').write_text(json.dumps(res, indent=1, default=float))
    return res


def imperf():
    rows = []
    for f in OUT.glob('imp_*.jsonl'):
        rows += [json.loads(l) for l in f.read_text().splitlines()]
    out = {}
    rng = np.random.default_rng(0)
    names = sorted({r['name'] for r in rows})
    for nm in names:
        o = {}
        for shape in ('mode', 'nodes'):
            for xi in sorted({r['xi'] for r in rows if r['name'] == nm}):
                rr = [r for r in rows if r['name'] == nm and r['shape'] == shape and r['xi'] == xi and 'error' not in r]
                for r in rr:
                    r['p_cap'] = max(r['p_max'] or 0.0, r.get('p_end') or 0.0)   # peak p up to the stop (p_end alone can follow a descending path)
                if not rr:
                    continue
                e = dict(n=len(rr), n_errors=sum(1 for r in rows if r['name'] == nm and r['shape'] == shape and r['xi'] == xi and 'error' in r))
                for key in ('p_max', 'p_soft', 'p_05', 'p_10', 'pe_05', 'pe_10', 'p_first_neg', 'p_lin', 'p_end', 'p_cap', 'mu_min'):
                    v = np.array([r[key] for r in rr if r.get(key) is not None], float)
                    if len(v):
                        boot = [np.mean(rng.choice(v, len(v))) for _ in range(2000)] if len(v) > 2 else [v.mean()]
                        e[key] = dict(mean=float(v.mean()), lo=float(np.percentile(boot, 2.5)), hi=float(np.percentile(boot, 97.5)), n=len(v), min=float(v.min()), max=float(v.max()))
                o.setdefault(shape, {})[str(xi)] = e
        o['koiter'] = koiter(o, nm)
        o['southwell_exploratory'] = southwell(o)
        out[nm] = o
    return out


def southwell(o, pref_key='p_lin'):
    """EXPLORATORY (not preregistered). Linear amplification psi = g xi p/(p_c - p) gives p_c/p_x - 1 = (g/psi*) xi at psi = psi*.
    Fit (p_ref/p_05 - 1) = A + C xi on the mode-shape series; A absorbs an intrinsic imperfection (A = C xi_intrinsic)."""
    res = {}
    for shape in ('mode', 'nodes'):
        d = o.get(shape, {})
        xs = sorted(float(x) for x in d if 'p_05' in d[x])
        if len(xs) < 4:
            continue
        pref = np.mean([d[str(x)]['p_lin']['mean'] for x in xs[:3]])
        y = np.array([pref / d[str(x)]['p_05']['mean'] - 1 for x in xs]); xx = np.array(xs)
        A = np.c_[np.ones_like(xx), xx]
        c, *_ = np.linalg.lstsq(A / np.maximum(y[:, None], 1e-3), y / np.maximum(y, 1e-3), rcond=None)
        res[shape] = dict(p_ref=float(pref), A=float(c[0]), C=float(c[1]), xi_intrinsic=float(c[0] / c[1]) if c[1] else None, max_rel_resid=float(np.abs((A @ c - y) / np.maximum(np.abs(y), 1e-3)).max()), y=dict(zip(map(str, xs), y.tolist())))
    return res


def koiter(o, nm):
    """Prereg rule: 1 - p_x/p_ref = C xi^beta on >= 4 amplitudes over >= 1.5 decades with log-log residual rms < 0.1. Otherwise not reported."""
    res = {}
    for shape in ('mode', 'nodes'):
        for key in ('p_max', 'p_soft', 'p_05'):
            d = o.get(shape, {})
            xs, ys, refs = [], [], []
            for xi, e in d.items():
                if key in e and 'p_lin' in e:
                    xs.append(float(xi)); ys.append(e[key]['mean'])
            if len(xs) < 4:
                res[f'{shape}_{key}'] = dict(status='fewer than 4 amplitudes', n=len(xs)); continue
            xs = np.array(xs); ys = np.array(ys); idx = np.argsort(xs); xs = xs[idx]; ys = ys[idx]
            ref = ys[0]    # reference: the smallest-amplitude value of the same measure
            kd = 1 - ys[1:] / ref; xx = xs[1:]
            ok = kd > 0
            if ok.sum() < 4 or np.log10(xx[ok].max() / xx[ok].min()) < 1.5:
                res[f'{shape}_{key}'] = dict(status='knock-down not positive on enough amplitudes', kd=dict(zip(map(str, xx.tolist()), kd.tolist()))); continue
            c = np.polyfit(np.log(xx[ok]), np.log(kd[ok]), 1)
            rms = float(np.sqrt(np.mean((np.polyval(c, np.log(xx[ok])) - np.log(kd[ok])) ** 2)))
            res[f'{shape}_{key}'] = dict(status='fit passes' if rms < 0.1 else 'fit rejected (rms)', beta=float(c[0]), C=float(np.exp(c[1])), rms=rms, ref=float(ref), kd=dict(zip(map(str, xx.tolist()), kd.tolist())))
    return res


if __name__ == '__main__':
    main()
    print('ok')

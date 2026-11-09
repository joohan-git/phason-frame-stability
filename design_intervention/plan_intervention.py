"""intervention plans. Deterministic. python plan_intervention.py cal | prod CAL.json. Exact ensembles and rules: registered plan. plan_cal.json holds the covariates of the unbiased ensembles, no frame calculation."""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames', 'penrose', 'heldout_test', 'design', 'nonlinear', 'defects', 'phason_paths', 'independent_code', 'forbidden_patches')]  # shared modules
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
CELLS = {
    'ab1217': dict(cls='ab', cell=[12, 17], sub=2, sweeps=1500, beta=0.6, V=1393),
    'ab2941': dict(cls='ab', cell=[29, 41], sub=2, sweeps=3000, beta=0.6, V=8119),
    'fib5': dict(cls='pen', cell=5, sub=4, sweeps=3000, beta=0.8, V=1686),
}
USW = {'ab1217': 1500, 'ab2941': 4000, 'fib5': 3000}
KAPPA = 30.0


def base(cname, tag, i, **kw):
    C = CELLS[cname]
    return dict(kind='til', cname=cname, cls=C['cls'], cell=C['cell'], sub=C['sub'], tag=f'{cname}_{tag}', ens=tag, i=i, **kw)


def ens(cname, tag, n, **kw):
    return [base(cname, tag, i, **kw) for i in range(n)]


def plan_cal():
    L = []
    for c in CELLS:
        L += ens(c, 'U', 24, protocol='free', beta=0.0, sweeps=USW[c], mech=False)
    return L


def plan_tilings(cal):
    """cal: dict cname -> mean variance of the U ensemble."""
    L = [dict(kind='ref', cname=c, cls=C['cls'], cell=C['cell'], sub=C['sub'], tag=f'{c}_ref', ens='ref') for c, C in CELLS.items()]
    for c, C in CELLS.items():
        v = cal[c]; sw = C['sweeps']; b = C['beta']
        L += ens(c, 'U', 24, protocol='free', beta=0.0, sweeps=USW[c])
        L += ens(c, 'L', 24, protocol='free', beta=b, kappa=KAPPA, cstar=v, cmode=1, sweeps=sw)
        if c != 'ab2941':
            L += ens(c, 'H', 12, protocol='free', beta=-1.0, kappa=KAPPA, cstar=v, cmode=1, sweeps=sw)
            L += ens(c, 'UR', 12, protocol='free', beta=0.0, kappa=KAPPA, cstar=v, cmode=1, sweeps=sw)
    # secondary: lower variance level, c* set per arm from the pilots so that the arms' mean variances coincide
    for c, lv in (('ab1217', dict(UR=0.478, L=0.505)), ('fib5', dict(UR=0.691, L=0.703))):
        C = CELLS[c]
        L += ens(c, 'lowUR', 12, protocol='free', beta=0.0, kappa=KAPPA, cstar=lv['UR'], cmode=1, sweeps=C['sweeps'])
        L += ens(c, 'lowL', 12, protocol='free', beta=C['beta'], kappa=KAPPA, cstar=lv['L'], cmode=1, sweeps=C['sweeps'])
    # secondary: equal accepted flips V/2 from the canonical start; equal moved fraction 0.30
    for c in ('ab1217', 'fib5'):
        C = CELLS[c]; A = C['V'] // 2
        L += ens(c, 'flipU', 12, protocol='flips', beta=0.0, maxacc=A, cap_sweeps=C['sweeps'])
        L += ens(c, 'flipL', 12, protocol='flips', beta=C['beta'], maxacc=A, cap_sweeps=C['sweeps'])
        L += ens(c, 'movU', 12, protocol='free', beta=0.0, kappa=KAPPA, cstar=0.30, cmode=2, sweeps=C['sweeps'])
        L += ens(c, 'movL', 12, protocol='free', beta=C['beta'], kappa=KAPPA, cstar=0.30, cmode=2, sweeps=C['sweeps'])
    # arm B
    pairs = json.loads((HERE / 'pairs.json').read_text())
    seen = {}
    for key, rows in pairs.items():
        for ip, r in enumerate(rows):
            for side in ('lo', 'hi'):
                c = r[side]; k = (r['cls'], str(c['L1']), str(c['L2']))
                if k in seen:
                    continue
                seen[k] = 1
                for off in range(3):
                    L.append(dict(kind='cell', cls=r['cls'], L1=c['L1'], L2=c['L2'], off=off, sub=2, tag=f"B_{r['cls']}_{len(seen):03d}", V_cov=c['V'], delta=c['delta']))
    return L


def plan_paths(P):
    """Nonlinear paths and second code on outputs of plan_tilings (registered plan, sections 6, 7). Indices are fixed by plan_tilings."""
    L = []

    def find(c, e, i):
        return next(k for k, t in enumerate(P) if t.get('cname') == c and t.get('ens') == e and t.get('i') == i)
    for c, n, sub in (('ab1217', 6, 2), ('fib5', 3, 2)):
        for e in ('L', 'U'):
            for i in range(n):
                k = find(c, e, i)
                L.append(dict(kind='nl', cname=c, ens=e, i=i, cls=P[k]['cls'], cell=P[k]['cell'], sub=sub, tag=f'nl_{c}_{e}', lifts_from=f'plan_tilings_{k:05d}.json'))
    for c in ('ab1217', 'fib5'):
        for e in ('L', 'U'):
            for i in range(4):
                k = find(c, e, i)
                L.append(dict(kind='xc', cname=c, ens=e, i=i, cls=P[k]['cls'], cell=P[k]['cell'], sub=P[k]['sub'], tag=f'xc_{c}_{e}', lifts_from=f'plan_tilings_{k:05d}.json'))
    pairs = json.loads((HERE / 'pairs.json').read_text())
    for cls in ('ab', 'pen'):
        r = max(pairs[f'{cls}_S'], key=lambda x: x['dn'])
        for side in ('lo', 'hi'):
            c = r[side]
            L.append(dict(kind='nl', cname=f'B_{cls}', ens=side, i=0, cls=cls, L1=c['L1'], L2=c['L2'], off=0, sub=2, tag=f'nl_B_{cls}_{side}'))
            L.append(dict(kind='xc', cname=f'B_{cls}', ens=side, i=0, cls=cls, L1=c['L1'], L2=c['L2'], off=0, sub=2, tag=f'xc_B_{cls}_{side}'))
    return L


if __name__ == '__main__':
    if sys.argv[1] == 'cal':
        (HERE / 'plan_cal.json').write_text(json.dumps(plan_cal()))
    else:
        cal = json.loads(Path(sys.argv[2]).read_text())
        P = plan_tilings(cal)
        (HERE / 'plan_tilings.json').write_text(json.dumps(P))
        (HERE / 'plan_paths.json').write_text(json.dumps(plan_paths(P)))
        from collections import Counter
        print(len(P), Counter((t.get('cname', 'B'), t.get('ens', t['kind'])) for t in P))

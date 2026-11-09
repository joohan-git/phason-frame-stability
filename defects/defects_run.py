"""task runner. python defects_run.py PLAN INDEX OUTDIR. Kinds: base, miss, census. One thread."""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames', 'design', 'nonlinear', 'phason_paths')]  # shared modules
import json, sys, time
from pathlib import Path
from defects_core import *


def run_base(T):
    r = T.get('r', 0.03); sub = T.get('sub', 2)
    mod, lifts, L1, L2, edges = cell_data(T, r)
    f = frame(mod, lifts, L1, L2, edges, sub)
    return dict(V=len(lifts), E=len(edges), **sample(f, mod))


def run_miss(T):
    r = T.get('r', 0.03); sub = T.get('sub', 2)
    mod, lifts, L1, L2, edges = cell_data(T, r)
    V, E = len(lifts), len(edges)
    k = level_k(T['level'], E)
    order, nskip = removal_order(V, edges, cell_key(T), k)
    rem = order[:k]
    keep = np.ones(E, bool); keep[rem] = False
    ek = edges[keep]
    d0 = degrees(V, edges); d1 = degrees(V, ek)
    f = frame(mod, lifts, L1, L2, ek, sub)
    out = dict(V=V, E=E, k=len(rem), f_actual=len(rem) / E, nskip=nskip, n_deg1=int((d1 == 1).sum()), n_deg2_new=int(((d1 == 2) & (d0 > 2)).sum()), n_deg2=int((d1 == 2).sum()),
               spans=bool(spans(V, ek)), rem_ends=[[int(d0[edges[e, 0]]), int(d0[edges[e, 1]])] for e in rem[:4]],
               **sample(f, mod, removed_midpoints(mod, lifts, edges, rem)))
    if T.get('direct'):
        rp = r / (1 - len(rem) / E)
        mod2, lifts2, L12, L22, edges2 = cell_data(T, rp)
        f2 = frame(mod2, lifts2, L12, L22, edges2[keep], sub); f2.set_stress(-np.eye(2))
        c2 = f2.crit(nphi=8)
        out['lam_direct'] = c2['lam_c']; out['r_direct'] = rp
    return out


def run_census(T):
    mod, lifts, L1, L2, edges = cell_data(T, 0.03)
    V, E = len(lifts), len(edges); d0 = degrees(V, edges); rows = []
    for e in T['edges']:
        keep = np.ones(E, bool); keep[e] = False
        ek = edges[keep]
        if not spans(V, ek):
            rows.append(dict(e=e, inadmissible=True)); continue
        f = frame(mod, lifts, L1, L2, ek, 2)
        s = sample(f, mod, removed_midpoints(mod, lifts, edges, [e]))
        rows.append(dict(e=e, dirk=int(edges[e, 2]), ends=[int(d0[edges[e, 0]]), int(d0[edges[e, 1]])], lam_c=s['lam_c'], mode=s['mode'], K=s['K'], Gmean=s['Gmean'],
                         part=s['part'], near_w=s['near_w'], near_null=s['near_null'], ax_max=s['ax_max']))
    return dict(rows=rows, V=V, E=E)


if __name__ == '__main__':
    plan, i, outd = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
    T = json.loads(Path(plan).read_text())[i]
    out = outd / f"{Path(plan).stem}_{i:05d}.json"
    if out.exists():
        sys.exit(0)
    t0 = time.time()
    res = dict(base=run_base, miss=run_miss, census=run_census)[T['kind']](T)
    res.update(task=T, sec=time.time() - t0, label='P')
    tmp = out.with_suffix('.tmp')
    tmp.write_text(json.dumps(res), encoding='utf-8'); tmp.replace(out)
    print(i, T['kind'], T.get('name'), T.get('level'), T.get('i'), f"{time.time() - t0:.1f}s", flush=True)

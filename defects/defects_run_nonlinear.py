"""task runner. python defects_run_nonlinear.py PLAN INDEX OUTDIR. Kinds: bow, nlmiss. One thread."""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames', 'design', 'nonlinear', 'phason_paths')]  # shared modules
import json, sys, time
from pathlib import Path
from defects_nonlinear import *
import ab_frame as C


def run_bow(T):
    mesh = get_mesh(T['name'], T['sub'])
    lam, _, _ = Model(mesh).linear_critical(macro=True, nev=2)
    pc = float(lam[0])
    sg = edge_signs(mesh, f"{T['name']}|{T['i']}")      # same signs at every amplitude and, per edge index, at every sub
    dU = bow_shape(mesh, sg, T['shape'])
    res = run_path(mesh, dU, T['w0'], pc)
    res.update(p_lin_straight=pc)
    return res


def run_nlmiss(T):
    from structures import family_lifts
    tl = C.canonical(*family_lifts(12, 17, 'B1'))
    lifts = tl['lifts']; edges = np.asarray(C.edges_from_lifts(lifts, tl['L1'], tl['L2']), np.int64)
    V, E = len(lifts), len(edges)
    k = level_k(T['level'], E)
    order, nskip = removal_order(V, edges, f"{T['cellname']}|None|{T['i']}", k)
    keep = np.ones(E, bool); keep[order[:k]] = False
    mesh = from_edges(lifts @ C.P.T, edges[keep], C.P, C.P @ np.array(tl['L1'], float), C.P @ np.array(tl['L2'], float), T['sub'], 0.03, 'miss')
    lam, _, _ = Model(mesh).linear_critical(macro=True, nev=2)
    pc = float(lam[0])
    sg = edge_signs(mesh, f"nlmiss|{T['i']}")
    res = run_path(mesh, bow_shape(mesh, sg, 'half'), T['w0'], pc)
    res.update(p_lin_straight=pc, k=len(order[:k]), nskip=nskip)
    return res


def run_kconv(T):
    mesh = get_mesh(T['name'], T['sub'])
    K0, G0, _ = schur_moduli(mesh)
    dU = bow_shape(mesh, edge_signs(mesh, f"{T['name']}|{T['i']}"), T['shape'])
    out = dict(K_straight=K0, G_straight=G0)
    for w0 in (1e-3, 3e-3, 1e-2, 3e-2):
        K, G, _ = schur_moduli(mesh.with_imperfection(w0 * dU, None))
        out[str(w0)] = dict(K=K, G=G)
    return out


if __name__ == '__main__':
    plan, i, outd = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
    T = json.loads(Path(plan).read_text())[i]
    out = outd / f"{Path(plan).stem}_{i:05d}.json"
    if out.exists():
        sys.exit(0)
    t0 = time.time()
    res = dict(bow=run_bow, nlmiss=run_nlmiss, kconv=run_kconv)[T['kind']](T)
    res.update(task=T, sec=time.time() - t0, label='P')
    tmp = out.with_suffix('.tmp')
    tmp.write_text(json.dumps(res, default=float), encoding='utf-8'); tmp.replace(out)
    print(i, T['kind'], T['name'], T.get('sub'), T.get('w0'), T.get('level'), T.get('i'), f"{time.time() - t0:.1f}s", flush=True)

"""Imperfection sensitivity. For one structure and a task list (shape, xi, realization, sign): stress-free imperfect reference,
natural path from p = 0 by arclength, stop criteria registered before the runs.
usage: run_imperfect.py NAME sub taskfile_index nshards   (tasks are generated deterministically; results append to out/imp_NAME_sSUB_shardK.jsonl)
"""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames', 'phason_paths', 'objectivity', 'disorder')]  # shared modules
import sys, json, time, math
import numpy as np
from structures import *

AMPS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
NREAL = 8


def chord_rms(mesh, dU, E0):
    """rms chord rotation of an imperfection, to set the amplitude."""
    d = mesh.d0 @ (np.eye(2) + E0).T + dU[mesh.ej] - dU[mesh.ei]
    c0, s0 = np.cos(mesh.b0), np.sin(mesh.b0)
    psi = np.arctan2(c0 * d[:, 1] - s0 * d[:, 0], c0 * d[:, 0] + s0 * d[:, 1])
    return float(np.sqrt(np.mean(psi ** 2)))


def mode_shape(mesh, phi):
    """(dU per node, E0) from a mode vector, unit rms chord rotation."""
    q = np.zeros(mesh.n); q[2:] = phi[:mesh.nu]
    dU = np.c_[q[0::3], q[1::3]]
    eh = phi[mesh.nu:] / mesh.s
    E0 = np.array([[eh[0], eh[2]], [eh[2], eh[1]]])
    a = 1e-4 / max(np.abs(dU).max(), np.abs(E0).max(), 1e-30)   # small so chord_rms is linear
    nm = chord_rms(mesh, a * dU, a * E0) / a
    return dU / nm, E0 / nm


def random_shape(mesh, nV, sub, seed):
    """iid Gaussian on vertices (first nV nodes); member nodes interpolated along the edge chain."""
    rng = np.random.default_rng(seed)
    dU = np.zeros((mesh.nn, 2))
    ends = np.unique(np.r_[mesh.ei[::sub], mesh.ej[sub - 1::sub]])
    assert len(ends) == nV, (len(ends), nV)
    dU[ends] = rng.normal(size=(nV, 2))
    # chains: elements come in groups of `sub`, from vertex i through member nodes to vertex j
    for e in range(mesh.ne // sub):
        chain = np.arange(e * sub, (e + 1) * sub)
        a = dU[mesh.ei[chain[0]]]; b = dU[mesh.ej[chain[-1]]]
        for h, c in enumerate(chain[:-1]):
            s_ = (h + 1) / sub
            dU[mesh.ej[c]] = (1 - s_) * a + s_ * b
    E0 = np.zeros((2, 2))
    a = 1e-4
    nm = chord_rms(mesh, a * dU, a * E0) / a
    return dU / nm, E0


def analyse(rows, pc_ref):
    good = [r_ for r_ in rows if 'p' in r_]
    p = np.array([r_['p'] for r_ in good]); psi = np.array([r_['max_chord'] for r_ in good])
    psirms = np.array([r_['rms_chord'] for r_ in good])
    mu = np.array([r_['mu'][0] if r_.get('mu') else np.nan for r_ in good]); neg = np.array([r_['neg'] for r_ in good])
    out = dict(nsteps=len(good), p_end=float(p[-1]) if len(p) else None, psi_max_end=float(psi[-1]) if len(p) else None, psi_rms_end=float(psirms[-1]) if len(p) else None)
    dp = np.diff(p)
    idx = np.where(dp < 0)[0]
    out['p_max'] = float(p[idx[0]]) if len(idx) else None
    out['psi_at_pmax'] = float(psi[idx[0]]) if len(idx) else None
    if np.isfinite(mu).any():
        i = int(np.nanargmin(mu)); out['p_soft'] = float(p[i]); out['mu_min'] = float(mu[i]); out['mu_end'] = float(mu[-1])
    for thr, key in ((0.05, 'p_05'), (0.1, 'p_10'), (0.2, 'p_20')):
        j = np.where(psi >= thr)[0]
        if len(j):
            k = j[0]
            if k > 0:
                out[key] = float(p[k - 1] + (thr - psi[k - 1]) / (psi[k] - psi[k - 1]) * (p[k] - p[k - 1]))
            else:
                out[key] = float(p[0])
    pe = np.array([r_['max_edge'] for r_ in good])
    for thr, key in ((0.05, 'pe_05'), (0.1, 'pe_10')):
        j = np.where(pe >= thr)[0]
        if len(j):
            k = j[0]
            out[key] = float(p[k - 1] + (thr - pe[k - 1]) / (pe[k] - pe[k - 1]) * (p[k] - p[k - 1])) if k > 0 else float(p[0])
    j = np.where(neg > 0)[0]
    out['p_first_neg'] = float(p[j[0]]) if len(j) else None
    return out


def run_one(mesh, dU, E0, xi, pc_ref, psistop=0.35, keep=False):
    imp = mesh.with_imperfection(xi * dU, xi * E0)
    model = Model(imp)
    lam, vec, st = model.linear_critical(macro=True, nev=2)
    scale = math.sqrt(imp.nz)
    p_stop = 1.5 * pc_ref

    def stop(r_, rows):
        return r_['p'] > p_stop or r_['max_chord'] > psistop or (r_['p'] < 0.02 and len(rows) > 5)
    wp = default_wp(imp)
    rows, _ = follow(model, np.zeros(imp.nz), 0.0, (st['z'], 1.0), 0.02 * scale, 500, stop=stop, wp=wp, dsmax=0.02 * scale, dsmin=1e-7 * scale, k_eig=2, guard_neg=0.001 * scale)
    res = analyse(rows, pc_ref); res['p_lin'] = float(lam[0]); res['end'] = rows[-1].get('status', 'maxsteps')
    if keep:
        res['rows'] = [r_ for r_ in rows if 'p' in r_]
    return res


def main():
    name = sys.argv[1]; sub = int(sys.argv[2]); ishard = int(sys.argv[3]) if len(sys.argv) > 3 else 0; nshard = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    only = sys.argv[5] if len(sys.argv) > 5 else None
    mesh = get_mesh(name, sub)
    model = Model(mesh)
    nV = mesh.nn - mesh.ne // sub * (sub - 1)
    t0 = time.time()
    lam, vec, st = model.linear_critical(macro=True, nev=3)
    pc_ref = float(lam[0])
    du, e0 = mode_shape(mesh, vec[:, 0])
    tasks = []
    for xi in AMPS:
        for sign in (1, -1):
            tasks.append(('mode', xi, sign, 0))
        for k in range(NREAL):
            tasks.append(('nodes', xi, 1, k))
    if only:
        tasks = [t for t in tasks if t[0] == only]
    out = HERE / 'out' / f'imp_{name}_s{sub}_{ishard}of{nshard}.jsonl'
    done = set()
    if out.exists():
        for l in out.read_text().splitlines():
            d = json.loads(l); done.add((d['shape'], d['xi'], d['sign'], d['k']))
    for it, (shape, xi, sign, k) in enumerate(tasks):
        if it % nshard != ishard or (shape, xi, sign, k) in done:
            continue
        t1 = time.time()
        if shape == 'mode':
            dU, E0 = du * sign, e0 * sign
        else:
            dU, E0 = random_shape(mesh, nV, sub, 1000 + k)
        try:
            res = run_one(mesh, dU, E0, xi, pc_ref)
        except Exception as ex:
            res = dict(error=repr(ex))
        res.update(name=name, sub=sub, shape=shape, xi=xi, sign=sign, k=k, sec=time.time() - t1)
        with out.open('a') as o:
            o.write(json.dumps(res, default=float) + '\n')
        print(shape, xi, sign, k, {a: (round(b, 5) if isinstance(b, float) else b) for a, b in res.items() if a in ('p_max', 'p_soft', 'p_05', 'p_10', 'pe_05', 'p_first_neg', 'p_lin', 'nsteps', 'end', 'sec', 'error')}, flush=True)
    print('done', name, time.time() - t0, flush=True)


if __name__ == '__main__':
    main()

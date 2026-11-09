"""task runner.  python design_run.py PLAN.json INDEX OUTDIR.  One thread, GPU hidden. All numbers P.
Kinds: ref, moduli, pc, path, load, refload, base, mc (registered before the runs)."""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames', 'penrose', 'phason_paths', 'robustness', 'heldout_test', 'isotropic_loss')]  # shared modules
import os
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[v] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import hashlib, json, math, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import design_core as X

TH_UNI = np.arange(16) * np.pi / 16
PHI16 = np.arange(16) * np.pi / 16


def states():
    S = [dict(kind='uniaxial', th=float(t), R=0.0) for t in TH_UNI]
    S.append(dict(kind='hydro', th=0.0, R=1.0))
    for R in (0.75, 0.5, 0.25, -0.5, -1.0):
        for t in (0.0, math.pi / 8, math.pi / 4):
            S.append(dict(kind='biaxial', th=float(t), R=R))
    return S


def build(T, r=None):
    mod = X.load(T['cls'])
    X.set_r(mod, T.get('r', 0.03) if r is None else r)
    if T['cls'] == 'ab':
        tl = mod.canonical(T['L1'], T['L2'], seed=T.get('off', 7))
    else:
        tl = mod.canonical(T['L1'], T['L2'])
    lifts = tl['lifts']
    info = dict(V=len(lifts))
    k = T.get('flips')
    if k is not None:
        V = len(lifts)
        if k == -1:
            att = 200 * V if T['cls'] == 'ab' else 3000 * V
            lifts, acc = X.flipper(T['cls'])(lifts, tl['L1'], tl['L2'], att, 1000 + T['fseed'])
        else:
            lifts, acc = X.flipper(T['cls'])(lifts, tl['L1'], tl['L2'], 200 * k + 50000, 1000 * T['fseed'] + k, maxacc=k)
        info['accepted'] = int(acc)
    area = abs(np.linalg.det(np.array([mod.P @ np.array(tl['L1'], float), mod.P @ np.array(tl['L2'], float)])))
    info['ledge'] = 2 * len(lifts) / float(area)
    return mod, tl, lifts, info


def hfit(mod, phis, ext):
    h = mod.harmonics(np.asarray(phis), np.asarray(ext))
    return dict(c0=h['c0'], h2=h['h2'], h4=h['h4'], fit_rms=h['fit_rms'], absh2=float(abs(complex(*h['h2']))), absh4=float(abs(complex(*h['h4']))))


def run_ref(T):
    conv = math.pi ** 2 / 12; rows = []
    for sub in T['subs']:
        for r in T['rs']:
            f = X.make_ref(T['lattice'], r, sub); f.set_stress(-np.eye(2))
            bz = X.bz_scan(f, 24); ext = f.lw_phi(np.arange(12) * np.pi / 12)
            d = f.demand(-np.eye(2)); m = X.moduli(f.C3)
            lam = min(bz[0], float(ext.min()))
            rows.append(dict(sub=sub, r=r, lam_c=lam, lam_bz=bz[0], k_bz=bz[1], lam_lw=float(ext.min()), S=lam * conv, moduli=m, demand=d,
                             ledge=f.ledge, gamma=float(f.lam_N((0, 0)))))
    return dict(rows=rows)


def run_moduli(T):
    rows = []
    for r in T['rs']:
        mod, tl, lifts, info = build(T, r=r)
        XF = X.make_xframe(mod); f = XF(lifts, tl['L1'], tl['L2'], sub=T.get('sub', 2)); f.compute_C3()
        d = f.demand(-np.eye(2))
        rows.append(dict(r=r, moduli=X.moduli(f.C3), demand=d, **info))
    return dict(rows=rows)


def run_pc(T):
    mod, tl, lifts, info = build(T)
    XF = X.make_xframe(mod); f = XF(lifts, tl['L1'], tl['L2'], sub=T.get('sub', 2)); f.set_stress(-np.eye(2))
    c = f.crit(nphi=8)
    return dict(info=info, crit=c, harm=hfit(mod, c['phis'], c['lw']), moduli=X.moduli(f.C3), demand=f.demand(-np.eye(2)))


def run_path(T):
    mod, tl, lifts, info = build(T)
    XF = X.make_xframe(mod); f = XF(lifts, tl['L1'], tl['L2'], sub=2); f.set_stress(-np.eye(2))
    M = mod.phason_matrix(tl['L1'], tl['L2']); a, b = mod.harmonic_parts(M)
    zero = X.moduli(f.C3)
    # zero-load acoustic estimate
    grid = np.linspace(0.0, 1.04 * 0.82, 8)
    grid = np.concatenate([grid, [0.76, 0.78, 0.80, 0.82, 0.84]]); grid = np.unique(np.round(grid, 6))
    Ls = [f.tangent(l) for l in grid]
    phis = np.arange(16) * np.pi / 16
    pred, tau = [], []
    for p_ in phis:
        lr, ev = X.acoustic_lambda(grid, Ls, p_); pred.append(lr); tau.append(ev)
    pred = np.array(pred); tau = np.array(tau)  # tau[phi, lam]
    # tau(theta, lam) harmonics at each lam
    h_tau = []
    for j in range(len(grid)):
        h = X.harmonics_general(phis, tau[:, j], ms=(2, 4))
        h_tau.append(dict(lam=float(grid[j]), tau0=h['c0'], h2=h['h2'], h4=h['h4']))
    hp = X.harmonics_general(phis[np.isfinite(pred)], pred[np.isfinite(pred)], ms=(2, 4))
    # zero-load transverse threshold from C(0) tensor incl. geometric term at lam=0
    return dict(info=info, a=[a.real, a.imag], b=[b.real, b.imag], absa=abs(a), absb=abs(b), moduli0=zero, grid=grid.tolist(),
                h_tau=h_tau, pred_lw=pred.tolist(), pred_harm=hp, L_at={str(float(g)): L.tolist() for g, L in zip(grid, Ls) if abs(g - 0.80) < 1e-9})


def run_load(T):
    mod, tl, lifts, info = build(T)
    XF = X.make_xframe(mod); f = XF(lifts, tl['L1'], tl['L2'], sub=2)
    return _load(f, info)


def _load(f, info):
    ST = states(); Ns = []; rs = []
    for s in ST:
        f.set_stress(X.state_tensor('biaxial', s['th'], s['R'])); Ns.append(f.N.copy()); rs.append(f.stress_resid)
    rows = f.crit_multi(Ns, PHI16)
    for s, row in zip(ST, rows):
        row.update(s); row['lw'] = [round(x, 6) for x in row['lw']]
    return dict(info=info, rows=rows, stress_resid_max=float(max(rs)))


def run_refload(T):
    f = X.make_ref(T['lattice'], 0.03, T['sub']); ST = states(); rows = []
    for s in ST:
        f.set_stress(X.state_tensor('biaxial', s['th'], s['R']))
        bz = X.bz_scan(f, 24); ext = f.lw_phi(PHI16)
        lam = min(bz[0], float(ext.min()))
        rows.append(dict(s, lam_c=lam, lam_bz=bz[0], k_bz=bz[1], lam_lw=float(ext.min()), phi_min=float(PHI16[int(np.argmin(ext))]),
                         mode='longwave' if ext.min() <= bz[0] else str(bz[1])))
    return dict(rows=rows, ledge=f.ledge)


def _sample_crit(f, mod):
    f.set_stress(-np.eye(2)); c = f.crit(nphi=8)
    m = X.moduli(f.C3); d = f.demand(-np.eye(2))
    return dict(lam_c=c['lam_c'], lw_min=c['lw_min'], extra=c['extra'], mode=c['mode'], harm=hfit(mod, c['phis'], c['lw']),
                K=m['K'], Gmean=m['Gmean'], Emean=m['Emean'], absR0=m['absR0'], absR1=m['absR1'], T0=m['T0'], ax_max=d['ax_max'], total_max=d['total_max'])


def run_base(T):
    mod, tl, lifts, info = build(T)
    XF = X.make_xframe(mod); f = XF(lifts, tl['L1'], tl['L2'], sub=2)
    return dict(info=info, **_sample_crit(f, mod))


def run_mc(T):
    mod, tl, lifts, info = build(T)
    edges = mod.edges_from_lifts(lifts, tl['L1'], tl['L2'])
    key = hashlib.sha256(f"{T['name']}|{T.get('off')}|{T['mc']}|{T['i']}".encode()).digest()   # same xi at every sigma (common random numbers)
    rng = np.random.default_rng(int.from_bytes(key[:8], 'little'))
    kw = {}
    if T['mc'] in ('E', 't'):
        xi = rng.standard_normal(len(edges))
        while np.any(np.abs(xi) > 3):
            b = np.abs(xi) > 3; xi[b] = rng.standard_normal(int(b.sum()))
        g = 1 + T['sigma'] * xi
        kw = dict(g_EA=g, g_EI=g if T['mc'] == 'E' else g ** 3)
    else:
        xi = rng.standard_normal((len(lifts), 2))
        while np.any(np.abs(xi) > 3):
            b = np.abs(xi) > 3; xi[b] = rng.standard_normal(int(b.sum()))
        kw = dict(dX=T['sigma'] * xi)
    SF = X.make_scatter_frame(mod); f = SF(lifts, tl['L1'], tl['L2'], sub=2, edges=edges, **kw)
    return dict(info=info, **_sample_crit(f, mod))


if __name__ == '__main__':
    plan, i, outd = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
    T = json.loads(Path(plan).read_text())[i]
    out = outd / f"{Path(plan).stem}_{i:05d}.json"
    if out.exists():
        sys.exit(0)
    t0 = time.time()
    res = dict(ref=run_ref, moduli=run_moduli, pc=run_pc, path=run_path, load=run_load, refload=run_refload, base=run_base, mc=run_mc)[T['kind']](T)
    res.update(task=T, sec=time.time() - t0, label='P')
    tmp = out.with_suffix('.tmp')
    tmp.write_text(json.dumps(res), encoding='utf-8'); tmp.replace(out)
    print(i, T['kind'], T.get('name', T.get('lattice')), f"{time.time() - t0:.1f}s", flush=True)

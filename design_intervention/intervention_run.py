"""intervention task runner. python intervention_run.py PLAN INDEX OUTDIR. Kinds: til (arm A), cell (arm B), nl (nonlinear path), xc (second code). One thread, GPU hidden. All numbers P."""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames', 'penrose', 'heldout_test', 'design', 'nonlinear', 'defects', 'phason_paths', 'independent_code', 'forbidden_patches')]  # shared modules
if __name__ == '__main__': __import__('os').chdir(__import__('os').path.dirname(__import__('os').path.abspath(__file__)))  # file names below are relative to this folder
import os
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[v] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import json, math, sys, time, hashlib
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from intervention_core import *
import design_core as X


def h32(*a):
    return int.from_bytes(hashlib.sha256('|'.join(map(str, a)).encode()).digest()[:4], 'little') & 0x7fffffff


# ------------------------------------------------------------------ covariates
def grad_feature(cls, lifts, tl, ref=None, R=4.0):
    """Methods: max_i ||G_i||_F, G_i = local internal-height gradient, weighted least squares with residual weights exp(-2 r^2 / R^2)."""
    C = MODS[cls]
    x = lifts @ C.P.T; h = lifts @ tl['perp'].T
    cellm = np.array([C.P @ np.array(tl['L1'], float), C.P @ np.array(tl['L2'], float)])
    sh = np.array(list(itertools.product([-1, 0, 1], repeat=2))) @ cellm
    xx = np.concatenate([x + s for s in sh]); hh = np.concatenate([h + np.zeros_like(h) for _ in sh])
    tree = cKDTree(xx); gmax = 0.0
    for c in x:
        ids = tree.query_ball_point(c, R); dx = xx[ids] - c
        W = np.exp(-np.sum(dx * dx, 1) / R ** 2)
        A = np.c_[np.ones(len(ids)), dx] * W[:, None]
        G = np.linalg.lstsq(A, hh[ids] * W[:, None], rcond=None)[0][1:]
        gmax = max(gmax, float(np.linalg.norm(G)))
    return gmax


def canon_ref(cls, tl):
    y = tl['lifts'] @ tl['perp'].T
    return dict(g0=grad_feature(cls, tl['lifts'], tl), s0=float(np.sqrt(np.mean(np.sum((y - y.mean(0)) ** 2, 1)))))


# ------------------------------------------------------------------ mechanics
def measure(cls, tl, lifts, sub, flags=None, nphi=8, vec=True):
    """ensemble.run protocol on XFrame: long-wave Richardson (0.002, 0.001) at nphi, Gamma, (1/2,0), (0,1/2), (1/2,1/2). K, G, peak strut demand, critical-mode localization."""
    mod = X.load(cls)
    XF = X.make_xframe(mod)
    f = XF(lifts, tl['L1'], tl['L2'], sub=sub)
    f.set_stress(-np.eye(2))
    extra = ((0, 0), (0.5, 0), (0, 0.5), (0.5, 0.5))
    c = f.crit(nphi=nphi, extra=extra)
    m = X.moduli(f.C3); d = f.demand(-np.eye(2)); f.set_stress(-np.eye(2))
    h = mod.harmonics(np.array(c['phis']), np.array(c['lw']))
    out = dict(lam_c=c['lam_c'], lw_min=c['lw_min'], lw=c['lw'], extra=c['extra'], mode=c['mode'], c0=h['c0'], h4=h['h4'], h2=h['h2'], K=m['K'], Gmean=m['Gmean'], Emean=m['Emean'], ax_max=d['ax_max'], total_max=d['total_max'], bend_max=d['bend_max'], dof=int(f.n))
    if vec:
        if c['mode'] == 'longwave':
            p = c['phis'][int(np.argmin(c['lw']))]
            k = 0.001 * (f.cell @ np.array([math.cos(p), math.sin(p)]))
        else:
            k = np.array(eval(c['mode']), float)
        lam, v = f.lam_N(k, vectors=True)
        w = np.abs(v[0::3]) ** 2 + np.abs(v[1::3]) ** 2 + np.abs(v[2::3]) ** 2
        out['part'] = float(w.sum() ** 2 / (len(w) * (w ** 2).sum()))
        out['lam_vec'] = float(lam)
        if flags is not None:
            # node weight to the nearest vertex: vertex nodes first; an interior node belongs to the nearer end of its edge
            V = len(lifts); edges = mod.edges_from_lifts(lifts, tl['L1'], tl['L2'])
            wv = w[:V].copy()
            for e, (i, j, *_r) in enumerate(edges):
                for hh_ in range(sub - 1):
                    n = V + e * (sub - 1) + hh_
                    t = (hh_ + 1) / sub
                    if t < 0.5:
                        wv[i] += w[n]
                    elif t > 0.5:
                        wv[j] += w[n]
                    else:
                        wv[i] += 0.5 * w[n]; wv[j] += 0.5 * w[n]
            wv /= wv.sum()
            out['mode_on_forbidden'] = float(wv[flags > 0].sum())
            pk = int(np.argmax(wv)); out['peak_on_forbidden'] = int(flags[pk] > 0)
            top = np.argsort(-wv)[:max(1, V // 100)]
            out['top1pct_on_forbidden'] = float(np.mean(flags[top] > 0))
    return out


# ------------------------------------------------------------------ arm A tilings
def make_tiling(T):
    """Returns tl, Sampler at its final state, chain info. Protocols: 'restrained' or 'free' (burn at weight W from the canonical start) and 'quench' (unbiased first passage to c* or to maxacc accepted flips)."""
    cls = T['cls']; name = tuple(T['cell']) if cls == 'ab' else T['cell']
    tl = cell(cls, name)
    s = Sampler(cls, tl['L1'], tl['L2'], tl['lifts'], tl['perp'], tl['lifts'])
    seed = h32(T['tag'], T['i'])
    tr = []
    t0 = time.time()
    if T['protocol'] == 'flips':
        # equal accepted flips from the canonical start, biased or not; attempts capped, a chain that does not reach maxacc is reported
        s.run(T['cap_sweeps'] * s.V, seed, beta=T.get('beta', 0.0), maxacc=T['maxacc'])
        s.refresh()
    elif T['protocol'] == 'quench':
        s.run(10 ** 9, seed, stop_var=T['cstar'])
        s.refresh()
    else:
        beta, kappa, cstar, cmode = T['beta'], T.get('kappa', 0.0), T.get('cstar', 0.0), T.get('cmode', 0)
        if T.get('start') == 'random':
            s.run(T['rand_sweeps'] * s.V, seed + 1); s.refresh()
        nch = T.get('nch', 20)
        for k in range(nch):
            s.run(int(T['sweeps'] / nch * s.V), seed + 10 + k, beta=beta, kappa=kappa, cstar=cstar, cmode=cmode)
            if beta == 0.0:
                s.refresh()
            tr.append([round(float(s.phi2), 4), round(s.var, 4), round(float(s.moved), 4)])
    s.refresh()
    ok = int(s.C.check_tiling(s.lifts, tl['L1'], tl['L2']))
    assert ok == 0
    info = dict(V=s.V, phi2=float(s.phi2), Nf=int(s.st[0]), var=s.var, moved=float(s.moved), accepted=s.n_acc, legal=s.n_legal, rejected=s.n_rej, attempts=s.n_att, trace=tr, mc_sec=time.time() - t0)
    return tl, s, info


def run_til(T):
    tl, s, info = make_tiling(T)
    r = canon_ref(T['cls'], tl)
    g = grad_feature(T['cls'], s.lifts, tl)
    info.update(gmax=g, x=max(0.0, 4.0 * (g - r['g0']) / r['s0']))
    t1 = time.time()
    if T.get('mech', True):
        info.update(measure(T['cls'], tl, s.lifts, T['sub'], flags=s.flags.copy()))
    info['mech_sec'] = time.time() - t1
    info['lifts'] = s.lifts.tolist()
    return info


# ------------------------------------------------------------------ arm B cells
def offsets_canonical(cls, L1, L2, off):
    if cls == 'ab':
        return AB.canonical(L1, L2, seed=[7, 11, 13][off])
    rng = np.random.default_rng(100 + off); o = rng.uniform(-.5, .5, 5); alt = np.array([1, -1, 1, -1, 1]); o += (0.5 - alt @ o) / 5 * alt
    return PN.canonical(L1, L2, offset=o)


def run_cell(T):
    cls = T['cls']
    tl = offsets_canonical(cls, T['L1'], T['L2'], T['off'])
    assert MODS[cls].check_tiling(tl['lifts'], tl['L1'], tl['L2']) == 0
    s = Sampler(cls, tl['L1'], tl['L2'], tl['lifts'], tl['perp'], tl['lifts'])
    out = dict(V=s.V, phi2=float(s.phi2), var=s.var)
    out.update(measure(cls, tl, tl['lifts'], T.get('sub', 2), flags=None, vec=False))
    return out


def run_ref(T):
    """Canonical reference of a cell at the default offset: lambda_ord of registered plan, section 2."""
    cls = T['cls']; name = tuple(T['cell']) if cls == 'ab' else T['cell']
    tl = cell(cls, name)
    s = Sampler(cls, tl['L1'], tl['L2'], tl['lifts'], tl['perp'], tl['lifts'])
    out = dict(V=s.V, phi2=float(s.phi2), var=s.var)
    r = canon_ref(cls, tl); out.update(g0=r['g0'], s0=r['s0'])
    out.update(measure(cls, tl, tl['lifts'], T['sub'], flags=s.flags.copy()))
    return out


def tiling_of(T):
    cls = T['cls']
    if 'L1' in T:
        tl = offsets_canonical(cls, T['L1'], T['L2'], T['off'])
        return cls, tl, tl['lifts']
    name = tuple(T['cell']) if cls == 'ab' else T['cell']
    tl = cell(cls, name)
    lifts = np.array(T['lifts'], np.int64) if 'lifts' in T else np.array(json.loads((Path(T['outdir']) / T['lifts_from']).read_text())['lifts'], np.int64)
    return cls, tl, lifts


# ------------------------------------------------------------------ nonlinear
def run_nl(T):
    import corotational_frame as NL
    import run_imperfect as RI
    cls, tl, lifts = tiling_of(T)
    C = MODS[cls]
    edges = C.edges_from_lifts(lifts, tl['L1'], tl['L2'])
    mesh = NL.from_edges(lifts @ C.P.T, edges, C.P, C.P @ np.array(tl['L1'], float), C.P @ np.array(tl['L2'], float), T['sub'], 0.03, T['tag'])
    model = NL.Model(mesh)
    lam, vec, st = model.linear_critical(macro=True, nev=2)
    scale = math.sqrt(mesh.nz); pc = float(lam[0])
    pstop, psistop = 1.5 * pc, 0.6

    def stop(r_, rows):
        return r_['p'] > pstop or r_['max_chord'] > psistop or (r_['p'] < 0.02 and len(rows) > 5)
    rows, _ = NL.follow(model, np.zeros(mesh.nz), 0.0, (st['z'], 1.0), 0.02 * scale, T.get('nsteps', 500), stop=stop, wp=NL.default_wp(mesh), dsmax=0.02 * scale, dsmin=1e-7 * scale, k_eig=2, guard_neg=0.001 * scale)
    res = RI.analyse(rows, pc)
    res.update(p_lin=pc, end=rows[-1].get('status', 'maxsteps'), part_lin=NL.participation(model, vec[:, 0]))
    good = [r for r in rows if 'p' in r]
    res['path'] = [[round(r['p'], 5), round(r['max_chord'], 4), round(r['max_edge'], 4), r['neg']] for r in good][::4]
    return res


# ------------------------------------------------------------------ second code on the same lifts
def run_xc(T):
    import cell_frame as F
    import cell_tiling as TT
    F.set_section('circ')
    cls, tl, lifts = tiling_of(T)
    C = MODS[cls]

    class S_:
        E = C.P; d = lifts.shape[1]

    class C_:
        star = S_; V = len(lifts); x = lifts @ C.P.T; n = lifts
        T = np.array([C.P @ np.array(tl['L1'], float), C.P @ np.array(tl['L2'], float)]).T
        area = abs(np.linalg.det(T))
        edges = [(int(i), int(j), int(a), int(b), int(k)) for i, j, k, a, b in C.edges_from_lifts(lifts, tl['L1'], tl['L2'])]
    fr = F.Frame(C_, nelem=T['sub'])
    h = fr.homogenize()
    ths = np.arange(8) * np.pi / 8
    lw = fr.longwave_exact(ths, p_hint=0.5)
    # finite-k localized branch: a small Bloch vector needs no translation pinning; a localized mode is flat in k (main code: long-wave and Gamma agree to 1e-6 on random tilings)
    lg, _ = fr.critical_factor(np.array([0.01, 0.0]), lam_guess=T.get('lam_guess', 0.4), nev=T.get('nev', 4))
    return dict(K=h['K'], virial=h['virial_trace'], lw_min=lw['lw_min'], c0=lw['c0'], lam_k=float(lg[0]), lam_min=float(min(lw['lw_min'], lg[0])))


if __name__ == '__main__':
    plan, i, outd = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
    T = json.loads(Path(plan).read_text())[i]
    out = outd / f"{Path(plan).stem}_{i:05d}.json"
    if out.exists():
        sys.exit(0)
    t0 = time.time()
    T['outdir'] = str(outd)
    res = dict(til=run_til, cell=run_cell, nl=run_nl, xc=run_xc, ref=run_ref)[T['kind']](T)
    res.update(task={k: v for k, v in T.items() if k != 'lifts'}, sec=time.time() - t0, label='P')
    tmp = out.with_suffix('.tmp')
    tmp.write_text(json.dumps(res, default=float), encoding='utf-8'); tmp.replace(out)
    print(i, T['kind'], T.get('tag'), T.get('i'), f"{time.time() - t0:.1f}s", flush=True)

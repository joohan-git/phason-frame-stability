"""r3 production: every configuration with the complete objective operator, sequential, one process.
Resumable: skips existing runs_objective/*.json. Usage: python run_icosa.py [stage ...]  stages: canon tetra dis sub4"""
import os, sys, json, time
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
from icosa_objective import *
from scipy.optimize import brentq
OUT = os.environ.get('ICOSA_OUT', os.path.join(here, 'runs_objective')); os.makedirs(OUT, exist_ok=True)
FIX = [0.0, 0.5, 0.85, 0.90, 0.92]
SPH = np.vstack([fib_sphere(200), np.array(dirs13())])
KALL = ((0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5), (0.5, 0.5, 0.5))
KDIS = ((0.5, 0, 0), (0.5, 0.5, 0.5))


def tau0(Lam):
    n = fib_sphere(200)
    A = np.einsum('jlab,ej,el->eab', Lam, n, n)
    nu = np.einsum('ea,eab,eb->e', n, A, n)
    return float(np.mean((np.trace(A, axis1=1, axis2=2) - nu) / 2))


def run(kind, pars, seed, flips=0, mcseed=None, sub=2):
    if kind == 'cubic':
        L = cubic_periods(*pars); tag = f'cubic_{pars[0]}-{pars[1]}'
    else:
        L = tetra_periods(pars[:2], pars[2:]); tag = f'tetra_{pars[0]}-{pars[1]}x{pars[2]}-{pars[3]}'
    if flips:
        tag += f'_f{flips}' + ('' if mcseed is None else f'_mc{mcseed}')
    if sub != 2:
        tag += f'_sub{sub}'
    fn = os.path.join(OUT, f'{tag}_s{seed}.json')
    if os.path.exists(fn):
        return
    t0 = time.time()
    tl = canonical(L, seed=seed); lifts = tl['lifts']
    res = dict(tag=tag, kind=kind, seed=seed, flips=flips, mcseed=mcseed, sub=sub, V=len(lifts), M=phason_matrix(L).tolist(), a=[float(np.linalg.norm(c)) for c in (P @ L).T])
    if flips:
        lifts, acc = flip_mc(lifts, L, nattempt=200 * max(flips, 100), seed=(seed + 1000 if mcseed is None else mcseed), maxacc=flips)
        res['flips_accepted'] = acc
        ref = {tuple(r) for r in tl['lifts']}
        res['moved_fraction'] = float(np.mean([tuple(r) not in ref for r in lifts]))
        ex = window_excess(lifts, tl); res['window_excess_max'] = float(ex.max())
        y = lifts @ tl['perp'].T; y0 = tl['lifts'] @ tl['perp'].T
        yc = y - y.mean(0) + y0.mean(0); eq = tl['window_eq']
        exc = np.max((yc - tl['cperp']) @ eq[:, :3].T + eq[:, 3], axis=1)
        res['excess_centered_max'] = float(exc.max())
    res['perp_var'] = perp_var(lifts, tl)
    f = Frame3D.from_tiling(lifts, L, sub=sub); prep_load(f); f.set_load('stress')
    res['zener'] = float(np.mean(np.diag(f.C6)[3:]) / (np.mean(np.diag(f.C6)[:3]) - np.mean([f.C6[0, 1], f.C6[0, 2], f.C6[1, 2]])))
    om = ObjModel(f); res['rigid'] = om.rigid_test(); res['Fp_over_N'] = om.Fp_over_N; res['M_over_NL'] = om.Mrms_over_NL; res['T_over_NL'] = om.Trms_over_NL
    H = HomogObj(f, om)
    lg, vg = frame_gamma_obj(f, om, 4)
    res['gamma'] = lg; res['gamma_pr'] = [participation(vg[:, i]) for i in range(len(lg))]
    lamW = lg[0] * (1 - 1e-7)
    res['inertia_below'] = H.tensor(lamW * 0.999)[1]
    res['kpts'] = {}
    for k in (KDIS if flips >= 1000 else KALL):
        lk, vk = frame_halfk_obj(f, om, k, 2)
        res['kpts'][str(k)] = [lk[0], participation(vk[:, 0])]
    # long wave: bracket over the sphere on a load grid, then exact refinement
    grid = sorted(set([l for l in (0.5, 0.7, 0.85, 0.9, 0.92) if l < lamW] + [lamW * x for x in (0.9, 0.96, 0.98, 0.99, 0.995, 0.999)]))
    lw = None; ldir = None; prev = None
    for l in grid:
        ee = np.linalg.eigvalsh(np.einsum('jlab,ej,el->eab', H.tensor(l)[0], SPH, SPH))[:, 0]
        if ee.min() <= 0:
            i = int(np.argmin(ee))
            guess = (prev + l) / 2 if prev else l
            try:
                lw, ldir = refine_min(H, [SPH[i]], guess, lamW, tol=1e-7)
            except Exception as err:
                res['lw_error'] = repr(err)
            if lw is None:
                lw = float(l)
                res['lw_coarse'] = True
            break
        prev = l
    res['lw_min'] = lw; res['lw_dir'] = ldir
    cands = [lg[0]] + [v[0] for v in res['kpts'].values()] + ([lw] if lw else [])
    res['p_c'] = float(min(cands))
    res['mech'] = 'long-wave' if (lw is not None and lw <= min(cands) + 1e-12 and lg[0] - lw > 1e-3) else 'Gamma/local'
    # stored tensors
    keep = {}
    for l in FIX:
        if l < lamW:
            keep[str(l)] = H.tensor(l)[0]
    if not flips:
        g = lambda l: tau0(H.tensor(l)[0])
        top = lamW * (1 - 1e-6); root = None
        lo = min(0.90, 0.98 * top)
        if g(lo) > 0 and g(top) < 0:
            root = float(brentq(g, lo, top, xtol=2e-5))
        elif g(lo) <= 0 and g(.5) > 0:
            root = float(brentq(g, .5, lo, xtol=2e-5))
        res['lam_iso_root'] = root
        if root:
            keep['root'] = H.tensor(root)[0]
        res['tensors'] = {k: v.tolist() for k, v in keep.items()}
    res['antis'] = {k: antis_ratio(v) for k, v in keep.items() if k != '0.0'}
    res['antis0'] = antis_ratio(keep['0.0']) if '0.0' in keep else None
    res['antis_load'] = antis_ratio(H.tensor(lamW * 0.9)[0])  # always below lamW, so measured even when lamW < 0.5
    res['odd'] = {k: odd_part(v) for k, v in keep.items()}
    if flips and sub == 2:
        u = vg[:, 0].reshape(-1, 6)[:, :3]; w = np.sum(u ** 2, 1)
        V = len(lifts); edges = edges_from_lifts(lifts, L)
        vw = w[:V].copy(); np.add.at(vw, edges[:, 0], .5 * w[V:]); np.add.at(vw, edges[:, 1], .5 * w[V:]); vw /= vw.sum()
        c = int(np.argmax(vw))
        res['top_ex_pct'] = float(np.mean(exc < exc[c])); res['mode_ex_pct'] = float(np.mean(exc < vw @ exc)); res['top_weight'] = float(vw[c])
    res['nfact'] = len(H.cache); res['t'] = time.time() - t0
    try:
        import psutil
        mi = psutil.Process().memory_info()
        res['peak_GB'] = getattr(mi, 'peak_wset', mi.rss) / 1e9
    except Exception:
        pass
    try:
        import resource
        res['peak_GB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    except Exception:
        pass
    json.dump(res, open(fn, 'w'))
    print(tag, seed, 'V', res['V'], 'pc %.5f lw %s G %.5f PR %.4f %s rig %.1e/%.1e antis %.1e t %.0f' % (res['p_c'], None if lw is None else round(lw, 5), lg[0], res['gamma_pr'][0], res['mech'], res['rigid']['element_max'], res['rigid']['node_norm'], max(res['antis'].values() or [0]), res['t']), flush=True)


if __name__ == '__main__':
    stages = sys.argv[1:] or ['canon', 'tetra', 'dis', 'sub4']
    if 'canon' in stages:
        for pq, n in (((1, 1), 8), ((2, 1), 8), ((3, 2), 8), ((5, 3), 8)):
            for s in range(1, n + 1):
                run('cubic', pq, s)
    if 'tetra' in stages:
        for pars in ((2, 1, 3, 2), (3, 2, 2, 1), (3, 2, 5, 3), (5, 3, 3, 2), (5, 3, 8, 5)):
            for s in range(1, 5):
                run('tetra', pars, s)
    if 'dis' in stages:
        for s in range(1, 9):
            run('cubic', (3, 2), s, flips=1)
        for s in range(1, 6):
            run('cubic', (3, 2), s, flips=1, mcseed=s + 77)
        for fl in (10, 100, 1000, 30000):
            for s in (1, 2, 3):
                run('cubic', (3, 2), s, flips=fl)
        for fl in (10, 100, 1000, 10000, 100000):
            for s in range(1, 5):
                run('cubic', (5, 3), s, flips=fl)
    if 'sub4' in stages:
        run('cubic', (3, 2), 1, sub=4)
        run('cubic', (3, 2), 1, flips=30000, sub=4)
    if 'large' in stages:
        for s in range(1, 5):
            run('cubic', (8, 5), s)
        for s in range(1, 5):
            run('tetra', (8, 5, 5, 3), s)
        for s in range(1, 5):
            run('cubic', (8, 5), s, flips=400000)
    print('STAGEDONE', stages, flush=True)

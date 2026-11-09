"""Step 2: equilibrium paths under true hydrostatic pressure for
  B1 5/7 (seed 7 canonical), E2a 5/7 (seed 7), stored random 12/17 tiling (random_mode.json).
usage: python run_cases.py <case> <sub>      case in {B1, E2a, random, square}
Outputs: <case>_sub<sub>.json and .log in this folder.
"""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames', 'phason_paths', 'disorder')]  # shared modules
import sys, os, json, math, time
from pathlib import Path
import numpy as np
import scipy.sparse.linalg as spla
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import hermite_frame as nb
ROUGH = HERE.resolve().parent / 'disorder'
import ab_frame as C   # canonical / edges_from_lifts only
from ab_families import family_lifts

case = sys.argv[1]; sub = int(sys.argv[2])
tag = f"{case}_sub{sub}" + os.environ.get("TAG", "")
logf = open(HERE / f"{tag}.log", 'w', encoding='utf-8')


def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); logf.write(s + '\n'); logf.flush()


t0 = time.time()
if case == 'square':
    fr = nb.square_lattice(2, 2, sub)
else:
    if case in ('B1', 'E2a'):
        L = family_lifts(5, 7, case); tl = C.canonical(L[0], L[1], seed=7)
        lifts, L1, L2 = tl['lifts'], tl['L1'], tl['L2']
    elif case == 'random':
        dat = json.loads((ROUGH / 'random_mode.json').read_text(encoding='utf-8'))
        lifts = np.array(dat['lifts'], np.int64); L1 = np.array(dat['L1'], np.int64); L2 = np.array(dat['L2'], np.int64)
    else:
        raise SystemExit(case)
    edges = C.edges_from_lifts(lifts, L1, L2)
    fr = nb.build_periodic(lifts, L1, L2, edges, sub)
log(f"[{tag}] V={fr.V} edges={len(fr.edge_i)} nodes={fr.nn} dofs={fr.ndof} A0={fr.A0:.4f} cell={fr.cell.tolist()}")
DENSE = 1200
fr_free = fr.free

# ---------------------------------------------------------------- linear threshold
pcs, d_b = nb.linear_buckling(fr, k=6)
pcs_pos = np.sort(pcs[pcs > 0])
p_lin = float(pcs_pos[0])
log(f"linear (secant-tangent) thresholds: {np.round(pcs_pos[:4], 6).tolist()}   p_lin = {p_lin:.6f}")
pcs_cl = nb.linear_buckling_classical(fr, k=6); pcs_cl = np.sort(pcs_cl[pcs_cl > 0])
log(f"linear (classical, undeformed geometry, N only) thresholds: {np.round(pcs_cl[:4], 6).tolist()}   p_lin_classical = {pcs_cl[0]:.6f}")
res = dict(case=case, sub=sub, V=int(fr.V), ndof=int(fr.ndof), p_lin=p_lin, p_lin_next=[float(v) for v in pcs_pos[1:4]],
           p_lin_classical=float(pcs_cl[0]), p_lin_classical_next=[float(v) for v in pcs_cl[1:4]],
           units='p in pi^2 EI/ell^3; chord rotation in rad', model='corotational Hermite beam, consistent bending shortening, true pressure potential p*A0*detF')


# ---------------------------------------------------------------- monitor
def diag(d, p, k=3, lu=None):
    Rv, K = fr.residual(d, p)
    Kf = fr.Kff(K)
    if lu is None:
        lu = spla.splu(Kf)
    w, v = nb.lowest_eigs(Kf, k=k, lu=lu, dense_limit=DENSE)
    th = nb.chord_rotations(fr, d)
    q = fr.dR_dp(d)[fr_free]
    phi = v[:, 0]
    cosq = float(phi @ q / (np.linalg.norm(phi) * np.linalg.norm(q)))
    st = dict(p=float(p), lam=[float(x) for x in w], rms=float(np.sqrt(np.mean(th ** 2))), maxrot=float(np.abs(th).max()),
                E=[float(x) for x in fr.strain(d)], detF=float(fr.detF(d)), detsign=int(nb.det_sign(lu)), cos_mode_load=cosq)
    st.update(nb.element_measures(fr, d))
    # other code's edge measure: edge chord rotation minus the mean joint rotation (all nodes / vertices only), no fundamental subtraction
    thall = d[2:3 * fr.nn:3]; thm_all = float(thall.mean()); thm_v = float(thall[:fr.V].mean())
    st.update(rms_edge_thm=float(np.sqrt(np.mean((th - thm_all) ** 2))), rms_edge_thmv=float(np.sqrt(np.mean((th - thm_v) ** 2))), thm_all=thm_all)
    return st, phi, th


path = []


def monitor(d, p, Dp, dl, n):
    st, phi, th = diag(d, p)
    st.update(step=n, Dp=float(Dp), dl=float(dl))
    log(f"  step {n:3d} p={p:.6f} Dp={Dp:+.2e} lam={st['lam'][0]:+.3e},{st['lam'][1]:+.3e} rms={st['rms']:.3e} max={st['maxrot']:.3e} "
        f"E=({st['E'][0]:+.2e},{st['E'][1]:+.2e},{st['E'][2]:+.2e}) detsign={st['detsign']} cos={st['cos_mode_load']:+.2f}")
    st['_d'] = d.copy(); st['_phi'] = phi
    path.append(st)
    return st


MAXROT = 0.35
ROT_KNEE = 0.05


def stop_fund(st, states):
    if st['lam'][0] < 0:
        return True
    if st['maxrot'] > MAXROT:
        return True
    if len(states) > 1 and np.sign(st['Dp']) != np.sign(states[-2]['Dp']):
        return True
    if st['p'] > 2.0 * p_lin:
        return True
    return False


# ---------------------------------------------------------------- fundamental path (arc-length from p = 0)
d = np.zeros(fr.ndof); p = 0.0
al = nb.ArcLength(fr, tol=1e-9, maxit=10, target_it=4)
dpdir, _ = al.tangent(d, p)
al.psi = float(np.linalg.norm(dpdir))
dl0 = 0.05 * math.sqrt(2) * al.psi          # first step ~ Dp = 0.05
dl_max = 0.08 * math.sqrt(2) * al.psi       # cap ~ Dp <= 0.08 on the fundamental path
log(f"arc-length: psi={al.psi:.4e} dl0={dl0:.4e}")
st0, _, _ = diag(d, 0.0)
log(f"  p=0: lam={st0['lam']}")
DROT = 0.01   # max increase of the max chord rotation per step (step is redone with dl/2 otherwise)


def accept(d_old, d_new):
    a = np.abs(nb.chord_rotations(fr, d_old)).max(); b = np.abs(nb.chord_rotations(fr, d_new)).max()
    return (b - a) <= DROT


states, d, p, prev, dl = al.run(d, p, dl0, 600, monitor, stop_fund, dl_max=dl_max, accept=accept)
last = states[-1]
event = None
if last['lam'][0] < 0:
    event = 'instability'
elif len(states) > 1 and np.sign(last['Dp']) != np.sign(states[-2]['Dp']):
    event = 'limit'
elif last['maxrot'] > MAXROT:
    event = 'maxrot'
else:
    event = 'p>2plin'
log(f"fundamental path ended by: {event} at p={last['p']:.6f}")

# refine the first critical point
crit = None
if event == 'instability' and len(states) >= 2:
    sa, sb = states[-2], states[-1]
    # classify: limit point if the mode has a first-order area change (cos_mode_load not small) and Dp ~ 0
    cosb = abs(sb['cos_mode_load'])
    # locate lam=0 by secant Newton at fixed p (stays on the current branch) from sa
    pa, pb = sa['p'], sb['p']; la_, lb = sa['lam'][0], sb['lam'][0]
    da = sa['_d']
    ok_all = True
    for it in range(40):
        pm = pa + (pb - pa) * (-la_) / (lb - la_)
        pm = min(max(pm, min(pa, pb) + 1e-12), max(pa, pb) - 1e-12)
        dm, ok, _ = nb.newton(fr, da, pm, tol=1e-10)
        if not ok:
            ok_all = False; log("  refine: newton failed at fixed p (limit point?)"); break
        stm, phim, thm = diag(dm, pm)
        lm = stm['lam'][0]
        log(f"  refine it {it}: p={pm:.8f} lam0={lm:+.3e} rms={stm['rms']:.3e} max={stm['maxrot']:.3e} cos={stm['cos_mode_load']:+.3f}")
        if abs(lm) < 1e-13 or abs(pb - pa) < 1e-9:
            break
        if np.sign(lm) == np.sign(la_):
            pa, la_, da = pm, lm, dm
        else:
            pb, lb = pm, lm
    crit = dict(p_c=float(pm), lam=stm['lam'], rms_fund=stm['rms'], maxrot_fund=stm['maxrot'], E_fund=stm['E'], cos_mode_load=stm['cos_mode_load'],
                refine_ok=ok_all, kind='bifurcation' if abs(stm['cos_mode_load']) < 0.05 else 'limit-like')
    log(f"first critical point: {crit}")
    res['first_critical'] = crit
    d_c, p_c, phi_c, th_fund_c = dm, pm, phim, thm
elif event == 'limit':
    # continue past the turning point to see whether p keeps falling (true limit) or the path wiggles / jumped
    post, _, _, _, _ = al.run(d, p, dl, 15, monitor, lambda st, ss: st['lam'][0] < 0 or st['maxrot'] > MAXROT, dl_max=dl_max, prev=prev, accept=accept)
    res['post_limit_path'] = [{k: v for k, v in s.items() if not k.startswith('_')} for s in post]
    # quadratic fit of p(s) through last 3 states for the limit point
    s3 = states[-3:]
    ps = np.array([s['p'] for s in s3]); rm = np.array([s['rms'] for s in s3]); mx = np.array([s['maxrot'] for s in s3])
    crit = dict(kind='limit', p_bracket=[float(ps.min()), float(ps.max())], p_max_path=float(max(s['p'] for s in states)),
                rms_at=[float(x) for x in rm], maxrot_at=[float(x) for x in mx], lam_at=[s['lam'][0] for s in s3])
    res['first_critical'] = crit
    log(f"limit point: {crit}")

# pressure at which max chord rotation reaches ROT_KNEE (interpolated on the path)
def p_at_maxrot(states, target):
    for a, b in zip(states[:-1], states[1:]):
        if a['maxrot'] < target <= b['maxrot']:
            t = (target - a['maxrot']) / (b['maxrot'] - a['maxrot'])
            return float(a['p'] + t * (b['p'] - a['p'])), float(a['rms'] + t * (b['rms'] - a['rms']))
    return None, None


res['fund_path'] = [{k: v for k, v in s.items() if not k.startswith('_')} for s in states]
res['fund_event'] = event
pk, rk = p_at_maxrot(states, ROT_KNEE)
res['p_at_maxrot_0.05_fund_interp'] = pk; res['rms_at_maxrot_0.05_fund_interp'] = rk
# refine the knee point: re-step from the bracketing state with a secant on the arc-length increment
if pk is not None:
    ia = max(i for i in range(len(states) - 1) if states[i]['maxrot'] < ROT_KNEE <= states[i + 1]['maxrot'])
    sa, sb = states[ia], states[ia + 1]
    prev_a = (states[ia]['_d'][fr_free] - states[ia - 1]['_d'][fr_free], sa['p'] - states[ia - 1]['p']) if ia > 0 else None
    dl_b = sb['dl'] if sb['dl'] > 0 else dl0
    t_lo, m_lo, t_hi, m_hi = 0.0, sa['maxrot'], 1.0, sb['maxrot']
    best = None
    for it in range(25):
        t = t_lo + (t_hi - t_lo) * (ROT_KNEE - m_lo) / (m_hi - m_lo)
        t = min(max(t, t_lo + 0.05 * (t_hi - t_lo)), t_hi - 0.05 * (t_hi - t_lo))
        dn, pn, Dd, Dp, itn, _, ok = al.step(sa['_d'], sa['p'], t * dl_b, prev_a)
        if not ok:
            log('  knee refine: step failed'); break
        thn = nb.chord_rotations(fr, dn); mr = float(np.abs(thn).max())
        best = dict(p=float(pn), maxrot=mr, rms=float(np.sqrt(np.mean(thn ** 2))), E=[float(x) for x in fr.strain(dn)])
        log(f"  knee refine it {it}: t={t:.4f} p={pn:.7f} maxrot={mr:.6f} rms={best['rms']:.5e}")
        if abs(mr - ROT_KNEE) < 1e-5:
            break
        if mr < ROT_KNEE:
            t_lo, m_lo = t, mr
        else:
            t_hi, m_hi = t, mr
    res['knee_0.05'] = best
    if best is not None:
        w, _ = nb.lowest_eigs(fr.Kff(fr.residual(dn, pn)[1]), k=2, dense_limit=DENSE)
        res['knee_0.05']['lam'] = [float(x) for x in w]
        log(f"knee point (maxrot=0.05): {res['knee_0.05']}")
lam_min_path = min(s['lam'][0] for s in states)
res['lam_min_along_fund'] = lam_min_path
log(f"p at maxrot=0.05 on fundamental path: {pk}  (rms {rk});  min lam along path {lam_min_path:+.3e}")

# ---------------------------------------------------------------- branch switching at a bifurcation
if crit is not None and crit.get('kind') == 'bifurcation' and not os.environ.get('NOBRANCH'):
    branch_runs = {}
    for sgn in (+1, -1):
        bpath = []

        def bmon(d, p, Dp, dl, n, _b=bpath):
            st, phi, th = diag(d, p)
            thr = th - th_fund_c
            st.update(step=n, Dp=float(Dp), dl=float(dl), rms_rel=float(np.sqrt(np.mean(thr ** 2))), maxrot_rel=float(np.abs(thr).max()))
            log(f"  branch{sgn:+d} {n:3d} p={p:.7f} dp={p-p_c:+.3e} Dp={Dp:+.2e} lam={st['lam'][0]:+.3e},{st['lam'][1]:+.3e} rms={st['rms']:.4e} rms_rel={st['rms_rel']:.4e} max={st['maxrot']:.4e} E=({st['E'][0]:+.2e},{st['E'][1]:+.2e},{st['E'][2]:+.2e})")
            _b.append(st)
            return st

        def bstop(st, states):
            return st['maxrot'] > MAXROT or len(states) > 300

        # inject mode: scale so that the linearised rms chord rotation = a0
        dphi = np.zeros(fr.ndof); dphi[fr_free] = phi_c
        th_lin = nb.chord_rotations(fr, d_c + 1e-4 * dphi) - th_fund_c
        scale = 1e-4 / np.sqrt(np.mean(th_lin ** 2))
        a0 = float(os.environ.get('A0', '0.004'))
        dd0 = sgn * a0 * scale * dphi
        # first corrector with the eigenvector as predictor (normal plane through the injected increment)
        dn = d_c + dd0; pn = p_c
        Dd0 = dd0[fr_free]; Dp0 = 0.0
        for it in range(20):
            Rv, K = fr.residual(dn, pn); Rf = Rv[fr_free]
            if np.abs(Rf).max() < 1e-9:
                break
            lu = spla.splu(fr.Kff(K))
            d1 = lu.solve(-Rf); d2 = lu.solve(-fr.dR_dp(dn)[fr_free])
            dp = -(Dd0 @ d1) / (Dd0 @ d2 + al.psi ** 2 * Dp0)
            dn = dn.copy(); dn[fr_free] += d1 + dp * d2; pn += dp
        log(f"  branch{sgn:+d} injected a0={a0}: p={pn:.7f} (dp={pn-p_c:+.3e}) it={it}")
        st, phi, th = diag(dn, pn); thr = th - th_fund_c
        st.update(step=-1, Dp=float(pn - p_c), dl=0.0, rms_rel=float(np.sqrt(np.mean(thr ** 2))), maxrot_rel=float(np.abs(thr).max()))
        bpath.append(st)
        prev_b = (dn[fr_free] - d_c[fr_free], pn - p_c)
        dlb = np.linalg.norm(prev_b[0]) * 1.0
        alb = nb.ArcLength(fr, psi=al.psi, tol=1e-9, maxit=10, target_it=4)
        try:
            bstates, dB, pB, _, _ = alb.run(dn, pn, dlb, 600, bmon, bstop, prev=prev_b, dl_max=float(os.environ.get('BR_DLMAX', '4')) * dlb)
        except RuntimeError as e:
            log(f"  branch{sgn:+d} stopped: {e}")
        # b2 fit: p - p_c = b2 x^2 + b4 x^4, x = rms_rel ; use points with rms_rel < 0.06
        xs = np.array([s['rms_rel'] for s in bpath]); ys = np.array([s['p'] - p_c for s in bpath])
        m = xs < 0.06
        fit = None
        if m.sum() >= 3:
            Amat = np.c_[xs[m] ** 2, xs[m] ** 4]
            coef, *_ = np.linalg.lstsq(Amat, ys[m], rcond=None)
            fit = dict(b2=float(coef[0]), b4=float(coef[1]), npts=int(m.sum()), xmax=float(xs[m].max()))
        xs_raw = np.array([s['rms'] for s in bpath])
        m2 = xs_raw < 0.06
        fit_raw = None
        if m2.sum() >= 3:
            coef, *_ = np.linalg.lstsq(np.c_[xs_raw[m2] ** 2, xs_raw[m2] ** 4], ys[m2], rcond=None)
            fit_raw = dict(b2=float(coef[0]), b4=float(coef[1]), npts=int(m2.sum()))
        ratios = [(float(s['rms_rel']), float((s['p'] - p_c) / s['rms_rel'] ** 2)) for s in bpath[:6]]
        ps = np.array([s['p'] for s in bpath])
        ilim = None
        for i in range(1, len(bpath)):
            if bpath[i]['p'] < bpath[i - 1]['p']:
                ilim = i - 1; break
        lim = None
        if ilim is not None:
            lim = dict(p_max=float(ps[ilim]), rms=bpath[ilim]['rms'], maxrot=bpath[ilim]['maxrot'], step=ilim)
        branch_runs[f"{sgn:+d}"] = dict(fit_rel=fit, fit_raw=fit_raw, ratios_small=ratios, limit_point=lim,
                                        reached_maxrot=float(bpath[-1]['maxrot']), p_end=float(bpath[-1]['p']),
                                        p_at_maxrot_035=p_at_maxrot(bpath, MAXROT)[0],
                                        lam_min_on_branch=float(min(s['lam'][0] for s in bpath)),
                                        path=[{k: v for k, v in s.items() if not k.startswith('_')} for s in bpath])
        log(f"  branch{sgn:+d}: fit_rel={fit} fit_raw={fit_raw} limit={lim} p_end={bpath[-1]['p']:.6f} maxrot_end={bpath[-1]['maxrot']:.4f}")
    res['branches'] = branch_runs

res['seconds'] = time.time() - t0
(HERE / f"{tag}.json").write_text(json.dumps(res, indent=1, default=float), encoding='utf-8')
log(f"done in {res['seconds']:.1f} s")

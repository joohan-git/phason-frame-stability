"""intervention mechanical gates on stored controls only (no intervention tiling). Writes frame_checks.json. python frame_checks.py [g1 g2 g3 ...]"""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames', 'penrose', 'heldout_test', 'design', 'nonlinear', 'defects', 'phason_paths', 'independent_code', 'forbidden_patches')]  # shared modules
import json, sys, time
import numpy as np
from intervention_run import *

OUTF = HERE / 'frame_checks.json'


def rel(a, b):
    return abs(a - b) / abs(b)


def g1():
    """Stored B1 12/17 off 7 at sub 2 through the intervention measure against design lam0 0.7995225588873206 and drift_check."""
    tl = cell('ab', (12, 17))
    s = Sampler('ab', tl['L1'], tl['L2'], tl['lifts'], tl['perp'], tl['lifts'])
    t0 = time.time(); m = measure('ab', tl, tl['lifts'], 2, flags=s.flags.copy())
    return dict(lam_c=m['lam_c'], target=0.7995225588873206, rel=rel(m['lam_c'], 0.7995225588873206), K=m['K'], K_target=0.51448461, K_rel=rel(m['K'], 0.51448461), part=m['part'], phi2=float(s.phi2), sec=time.time() - t0, extra=m['extra'], total_max=m['total_max'])


def g2():
    """The equilibrated tiling of drift_12-17_sc1_random_s1 through the same chunks of the intervention sampler at beta = 0: lifts, p_c, gamma, variance, against the stored JSON."""
    d = json.load(open(ROOT / 'disorder' / 'ab_ensemble' / 'drift_12-17_sc1_random_s1.json')) if (ROOT / 'disorder' / 'ab_ensemble' / 'drift_12-17_sc1_random_s1.json').exists() else json.load(open(M / 'phason_paths' / 'drift_check' / 'drift_12-17_sc1_random_s1.json'))
    tl = cell('ab', (12, 17))
    s = Sampler('ab', tl['L1'], tl['L2'], tl['lifts'], tl['perp'], tl['lifts'])
    chunk = float(d['equil_trace'][0][0]); nch = len(d['equil_trace'])   # chunk size of the stored run, from its own trace
    for k in range(nch):
        s.run(int(chunk * s.V), 1 * 100003 + k)
    s.refresh()
    t0 = time.time(); m = measure('ab', tl, s.lifts, 2, flags=s.flags.copy())
    return dict(lam=m['lam_c'], stored=d['pc_min_all'], rel=rel(m['lam_c'], d['pc_min_all']), var=s.var, var_stored=d['perp_var'], moved=float(s.moved), moved_stored=d['moved_fraction'], gamma=m['extra']['(0, 0)'], gamma_stored=d['pc_gamma'], phi2=float(s.phi2), part=m['part'], mode_on_forb=m['mode_on_forbidden'], sec=time.time() - t0, accepted=s.n_acc, acc_stored=d['accepted'])


def g3():
    """Penrose fib5 at sub 2 against the stored defects gate target; sub 4 reported."""
    st = json.load(open(M / 'defects' / 'checks_linear.json'))
    tl = cell('pen', 5)
    s = Sampler('pen', tl['L1'], tl['L2'], tl['lifts'], tl['perp'], tl['lifts'])
    t0 = time.time(); m2 = measure('pen', tl, tl['lifts'], 2, flags=s.flags.copy()); t2 = time.time() - t0
    t0 = time.time(); m4 = measure('pen', tl, tl['lifts'], 4, flags=s.flags.copy()); t4 = time.time() - t0
    return dict(V=s.V, phi2=float(s.phi2), lam2=m2['lam_c'], lam4=m4['lam_c'], sec2=t2, sec4=t4, K=m2['K'], part4=m4['part'], stored=str(st)[:700])


def g4():
    """Zero-penalty arm of the intervention sampler against the unbiased parent ensemble, thresholds: 16 tilings, same attempts, different seeds; mean p_c and phi2 against drift ensemble 12/17 values and parent flip_mc tilings."""
    tl = cell('ab', (12, 17)); rows = []
    for i in range(16):
        T = dict(cls='ab', cell=[12, 17], protocol='free', beta=0.0, sweeps=1000, tag='g4', i=i)
        tl, s, info = make_tiling(T)
        m = measure('ab', tl, s.lifts, 2, flags=s.flags.copy(), vec=False)
        lp, _ = AB.flip_mc(tl['lifts'], tl['L1'], tl['L2'], 1000 * s.V, h32('g4par', i))
        sp = Sampler('ab', tl['L1'], tl['L2'], lp, tl['perp'], tl['lifts'], atlas=s.atlas)
        mp = measure('ab', tl, lp, 2, flags=sp.flags.copy(), vec=False)
        rows.append(dict(lam=m['lam_c'], phi2=float(s.phi2), var=s.var, lam_parent=mp['lam_c'], phi2_parent=float(sp.phi2), var_parent=sp.var, sec_mech=None))
        print('g4', i, rows[-1], flush=True)
    a = np.array([[r['lam'], r['phi2'], r['var'], r['lam_parent'], r['phi2_parent'], r['var_parent']] for r in rows])
    return dict(rows=rows, mean=a.mean(0).tolist(), sem=(a.std(0, ddof=1) / 4).tolist())


def g5():
    """Second code on the same lifts: the canonical 12/17, one equilibrated 12/17 tiling, canonical fib5."""
    out = {}
    tl = cell('ab', (12, 17))
    T = dict(cls='ab', cell=[12, 17], lifts=tl['lifts'].tolist(), sub=2)
    t0 = time.time(); out['canon_ab'] = dict(run_xc(T), sec=time.time() - t0)
    s = Sampler('ab', tl['L1'], tl['L2'], tl['lifts'], tl['perp'], tl['lifts']); s.run(1000 * s.V, 123); s.refresh()
    T = dict(cls='ab', cell=[12, 17], lifts=s.lifts.tolist(), sub=2)
    t0 = time.time(); x = run_xc(T); m = measure('ab', tl, s.lifts, 2, vec=False); out['eq_ab'] = dict(x, main_lw=m['lw_min'], main_gamma=m['extra']['(0, 0)'], main_lam=m['lam_c'], main_K=m['K'], sec=time.time() - t0)
    tl = cell('pen', 5); T = dict(cls='pen', cell=5, lifts=tl['lifts'].tolist(), sub=2)
    t0 = time.time(); x = run_xc(T); m = measure('pen', tl, tl['lifts'], 2, vec=False); out['canon_pen'] = dict(x, main_lw=m['lw_min'], main_gamma=m['extra']['(0, 0)'], main_lam=m['lam_c'], main_K=m['K'], sec=time.time() - t0)
    return out


def g6():
    """Nonlinear solver on the stored R22 of Fig. 4: linear critical of corotational_frame against the frame code on the same tiling; path timing and first fold."""
    d = json.loads((ROOT / 'disorder' / 'random_mode.json').read_text())
    T = dict(cls='ab', cell=[12, 17], lifts=d['lifts'], sub=2, tag='R22')
    t0 = time.time(); r = run_nl(T)
    r['sec'] = time.time() - t0; r.pop('path')
    tl = cell('ab', (12, 17))
    tlr = dict(tl); tlr['L1'] = d['L1']; tlr['L2'] = d['L2']
    m = measure('ab', tlr, np.array(d['lifts'], np.int64), 2, vec=False)
    return dict(r, frame_lam=m['lam_c'], frame_lw=m['lw_min'], rel=rel(r['p_lin'], m['lam_c']))


if __name__ == '__main__':
    OUT = json.loads(OUTF.read_text()) if OUTF.exists() else {}
    for w in sys.argv[1:]:
        t0 = time.time()
        OUT[w] = globals()[w]()
        print(w, f'{time.time() - t0:.0f}s', json.dumps(OUT[w], default=float)[:1500], flush=True)
        OUTF.write_text(json.dumps(OUT, indent=1, default=float))

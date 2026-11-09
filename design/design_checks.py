"""gates G0a-G6 (registered before the runs). Writes gates.json. Single thread."""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames', 'penrose', 'phason_paths', 'robustness', 'heldout_test', 'isotropic_loss')]  # shared modules
import os
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[v] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import json, math, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import design_core as X
out = HERE / 'design_checks.json'
R = json.loads(out.read_text()) if out.exists() else {}


def save():
    out.write_text(json.dumps(R, indent=1), encoding='utf-8')


def rec(k, val, ok, **kw):
    R[k] = dict(value=val, passed=bool(ok), **kw)
    print(k, val, 'PASS' if ok else 'FAIL', str(kw)[:500], flush=True)
    save()


src = ''
ns = {'np': np}
ns.update(vars(__import__('ab_families')))
fl = ns['family_lifts']
AB = X.load('ab'); PN = X.load('pen')
which = sys.argv[1].split(',') if len(sys.argv) > 1 else ['G0a', 'G0b', 'G1', 'G2', 'G3', 'G4', 'G5', 'G6']
if 'G0a' in which:
    t0 = time.time(); L = fl(5, 7, 'E2a'); tl = AB.canonical(L[0], L[1], seed=7)
    f = AB.Frame(tl['lifts'], tl['L1'], tl['L2'], sub=2); f.set_load('stress')
    ph, ext, _ = f.longwave(nphi=8); h = AB.harmonics(ph, ext)
    rec('G0a', h['c0'], abs(h['c0'] - 0.798890) < 2e-6, target=0.798890, V=len(tl['lifts']), sec=time.time() - t0)
if 'G0b' in which:
    t0 = time.time(); L = PN.fib_approx(6); tl = PN.canonical(L[0], L[1])
    XF = X.make_xframe(PN); f = XF(tl['lifts'], tl['L1'], tl['L2'], sub=2); f.set_load('stress'); c = f.crit(nphi=4)
    rec('G0b', c['lam_c'], abs(c['lam_c'] - 0.8102991399946218) < 2e-5, target=0.8102991399946218, V=len(tl['lifts']), sec=time.time() - t0)
if 'G1' in which:
    errs = {}
    sets = [('square', AB, np.array([[0, 0, 0, 0]]), [1, 0, 0, 0], [0, 0, 1, 0])]
    L = fl(2, 3, 'B1'); tl = AB.canonical(L[0], L[1], seed=7); sets.append(('B1_2_3', AB, tl['lifts'], tl['L1'], tl['L2']))
    L = PN.fib_approx(2); tp = PN.canonical(L[0], L[1]); sets.append(('pen2', PN, tp['lifts'], tp['L1'], tp['L2']))
    for name, mod, lifts, L1, L2 in sets:
        lifts = np.array(lifts, np.int64)
        f0 = mod.Frame(lifts, L1, L2, sub=2); f0.set_load('stress')
        edges = mod.edges_from_lifts(lifts, L1, L2)
        gm = X.load('pen'); gm.P = mod.P.copy(); gm.NDIM = mod.P.shape[1]; X.set_r(gm, 0.03)
        XF = X.make_xframe(gm); f1 = XF(lifts, L1, L2, sub=2, edges=edges); f1.set_stress(-np.eye(2))
        vals = []
        for k in [(0, 0), (.5, .5), (.25, .1), (0.002 * f0.cell[0, 0], 0.002 * f0.cell[1, 0])]:
            a = f0.lam(k); b = f1.lam_N(k); vals.append(abs(a - b) / a)
        errs[name] = max(vals)
    fr = X.make_ref('square', 0.03, 2); fr.set_stress(-np.eye(2))
    f0 = AB.Frame(np.array([[0, 0, 0, 0]], np.int64), [1, 0, 0, 0], [0, 0, 1, 0], sub=2); f0.set_load('stress')
    errs['square_ref_def'] = max(abs(f0.lam(k) - fr.lam_N(k)) / f0.lam(k) for k in [(0, 0), (.5, .5), (.25, .1), (0.002, 0.0)])
    fr = X.make_ref('rhomb45', 0.03, 2); fr.set_stress(-np.eye(2))
    f0 = AB.Frame(np.array([[0, 0, 0, 0]], np.int64), [1, 0, 0, 0], [0, 1, 0, 0], sub=2); f0.set_load('stress')
    errs['rhomb_ref_def'] = max(abs(f0.lam(k) - fr.lam_N(k)) / f0.lam(k) for k in [(0, 0), (.5, .5), (.25, .1), (0.002, 0.0)])
    rec('G1', max(errs.values()), max(errs.values()) <= 1e-8, detail=errs)
if 'G2' in which:
    errs = {}
    XF = X.make_xframe(AB)
    for fam, p, q in (('B1', 2, 3), ('E2a', 2, 3)):
        L = fl(p, q, fam); tl = AB.canonical(L[0], L[1], seed=7)
        f0 = XF(tl['lifts'], tl['L1'], tl['L2'], sub=2); f0.set_load('stress')
        for k in [(0.002, 0.001), (0, 0), (.5, .5)]:
            a = f0.lam(k); f0.dense_max = 0; b = f0.lam_N(k); f0.dense_max = 600
            errs[f'{fam}{k}'] = abs(a - b) / a
    rec('G2', max(errs.values()), max(errs.values()) <= 1e-8, detail=errs)
if 'G3' in which:
    SF = X.make_scatter_frame(AB); XF = X.make_xframe(AB); L = fl(2, 3, 'B1'); tl = AB.canonical(L[0], L[1], seed=7)
    kk = [(0, 0), (.5, .5), (.3, .1)]
    f0 = XF(tl['lifts'], tl['L1'], tl['L2'], sub=2); f0.set_load('stress'); base = {k: f0.lam_N(k) for k in kk}
    fs = SF(tl['lifts'], tl['L1'], tl['L2'], sub=2); fs.set_stress(-np.eye(2))
    e0 = max(abs(fs.lam_N(k) - base[k]) / base[k] for k in kk)
    nE = len(AB.edges_from_lifts(tl['lifts'], tl['L1'], tl['L2']))
    g = 1.37; fs = SF(tl['lifts'], tl['L1'], tl['L2'], sub=2, g_EA=np.full(nE, g), g_EI=np.full(nE, g)); fs.set_stress(-np.eye(2))
    e1 = max(abs(fs.lam_N(k) / g - base[k]) / base[k] for k in kk)
    d = 0.083
    fs = SF(tl['lifts'], tl['L1'], tl['L2'], sub=2, cellF=(1 + d) * np.eye(2)); fs.set_stress(-np.eye(2))
    lhs = {k: fs.lam_N(k) * (1 + d) ** 3 for k in kk}
    AB2 = X.load('ab'); X.set_r(AB2, 0.03 / (1 + d)); XF2 = X.make_xframe(AB2)
    f2 = XF2(tl['lifts'], tl['L1'], tl['L2'], sub=2); f2.set_load('stress')
    e2 = max(abs(lhs[k] - f2.lam_N(k)) / lhs[k] for k in kk)
    th = 0.37; Q = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    fs = SF(tl['lifts'], tl['L1'], tl['L2'], sub=2, cellF=Q); fs.set_stress(-np.eye(2))
    e3 = max(abs(fs.lam_N(k) - base[k]) / base[k] for k in kk)
    rng = np.random.default_rng(1); dX = rng.normal(0, 0.02, (len(tl['lifts']), 2))
    fs = SF(tl['lifts'], tl['L1'], tl['L2'], sub=2, dX=dX); xb = fs.xa + fs.ev * fs.Le[:, None]
    e4 = 0.0
    for e in range(0, len(fs.ei), 17):
        ra = fs.xa[e]; rb = xb[e]; mode = np.array([-ra[1], ra[0], 1, -rb[1], rb[0], 1.])
        e4 = max(e4, np.linalg.norm(fs.Ke[e] @ mode) / np.linalg.norm(fs.Ke[e]))
    rec('G3', max(e0, e1, e2, e3), max(e0, e1, e2, e3) <= 1e-7 and e4 <= 1e-9, zero=e0, uniform_g=e1, scale=e2, rotation=e3, rigid_null=e4)
if 'G4' in which:
    XF = X.make_xframe(AB); L = fl(2, 3, 'E2a'); tl = AB.canonical(L[0], L[1], seed=7)
    f = XF(tl['lifts'], tl['L1'], tl['L2'], sub=2)
    res = []
    for S in (X.state_tensor('uniaxial', 0.4), X.state_tensor('biaxial', 1.1, 0.5), X.state_tensor('biaxial', 0.2, -1.0)):
        f.set_stress(S); res.append(f.stress_resid)
    f0 = AB.Frame(tl['lifts'], tl['L1'], tl['L2'], sub=2); f0.set_load('stress'); f.set_stress(-np.eye(2))
    res.append(float(np.max(np.abs(f.N - f0.N))))
    rec('G4', max(res), max(res) <= 1e-9, detail=res)
if 'G5' in which:
    XF = X.make_xframe(AB); det = {}
    for fam, p, q in (('B1', 2, 3), ('B1', 5, 7)):
        L = fl(p, q, fam); tl = AB.canonical(L[0], L[1], seed=7)
        f = XF(tl['lifts'], tl['L1'], tl['L2'], sub=2); f.set_stress(-np.eye(2))
        phis = np.arange(8) * np.pi / 8; ext = f.lw_phi(phis)
        g = np.linspace(0, 1.1 * ext.max(), 12); Ls = [f.tangent(l) for l in g]
        pred = np.array([X.acoustic_lambda(g, Ls, p_)[0] for p_ in phis])
        det[f'{fam}{p}/{q}'] = dict(direct=ext.tolist(), tangent=pred.tolist(), maxrel=float(np.nanmax(np.abs(pred - ext) / ext)))
    v = max(d['maxrel'] for d in det.values()); rec('G5', v, v <= 0.002, detail=det)
if 'G6' in which:
    conv = math.pi ** 2 / 12; det = {}   # S/E/(t/L)^3 = lam * pi^2/12
    tgt = [('square', 'hydro', 0.453), ('square', 'pureshear_x', 0.493), ('square', 'pureshear_y', 0.493),
           ('triangular', 'hydro', 2.395), ('triangular', 'pureshear_x', 2.099), ('triangular', 'pureshear_y', 3.213),
           ('hexagonal', 'hydro', 0.175)]
    for name, st, val in tgt:
        S = {'hydro': -np.eye(2), 'pureshear_x': np.diag([-1., 1.]), 'pureshear_y': np.diag([1., -1.])}[st]
        row = {}
        for r, sub in ((0.03, 2), (0.03, 8), (0.005, 8)):
            f = X.make_ref(name, r, sub); f.set_stress(S); b = X.bz_scan(f, 24)
            ph = np.arange(12) * np.pi / 12; ext = f.lw_phi(ph)
            row[f'r{r}_sub{sub}'] = dict(bz=b[0] * conv, kbz=b[1], lw=float(ext.min()) * conv, S=min(b[0], float(ext.min())) * conv)
        row['target'] = val; row['err_pct_rad005_sub8'] = abs(row['r0.005_sub8']['S'] - val) / val * 100
        det[f'{name}_{st}'] = row; print(name, st, row, flush=True)
    v = max(d['err_pct_rad005_sub8'] for d in det.values()); rec('G6', v, v <= 1.0, detail=det)

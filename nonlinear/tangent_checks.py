"""Gate (a): linear operator vs ObjFrame. Gate (b): FD check of residual and tangent. Writes tangent_checks.json."""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames', 'phason_paths', 'objectivity', 'disorder')]  # shared modules
import json, math, sys, time
import numpy as np
from corotational_frame import *
import ab_frame as C
from objective_frame import ObjFrame
ns = {'np': np}
src = ''
ns.update(vars(__import__('ab_families'))); family_lifts = ns['family_lifts']
out = {}

def fd_check(model, seed=0, amp=0.02):
    m = model.m; rng = np.random.default_rng(seed)
    z = amp * rng.normal(size=m.nz) * 0.1; z[m.nu:] *= 0.5
    p = 0.7 * m.lamE
    R = model.residual(z, p)
    Kuu, B, Cc = model.tangent(z, p); T = Tangent(Kuu, B, Cc, m)
    res = {}
    v = rng.normal(size=m.nz); v /= np.linalg.norm(v)
    errs = []
    for h in (1e-4, 1e-5, 1e-6):
        dE = (model.energy(z + h * v, p) - model.energy(z - h * v, p)) / (2 * h)
        errs.append(abs(dE - R @ v) / max(abs(R @ v), 1e-12))
    res['grad_rel_err'] = min(errs)
    errs = []
    for h in (1e-4, 1e-5, 1e-6, 1e-7):
        dR = (model.residual(z + h * v, p) - model.residual(z - h * v, p)) / (2 * h)
        errs.append(np.linalg.norm(dR - T.mv(v)) / np.linalg.norm(dR))
    res['tan_rel_err'] = min(errs)
    # area gradient (dR/dp)
    res['dRdp_err'] = float(np.linalg.norm((model.residual(z, p + 1e-6) - R) / 1e-6 - model.area_grad(z)))
    # symmetry
    u = rng.normal(size=m.nz)
    res['sym_err'] = abs(u @ T.mv(v) - v @ T.mv(u)) / abs(u @ T.mv(v))
    # inertia vs dense
    if m.nz < 3000:
        dense = np.block([[Kuu.toarray(), B], [B.T, Cc]])
        ev = np.linalg.eigvalsh(dense)
        res['inertia_lu'] = T.inertia(); res['inertia_dense'] = int((ev < 0).sum())
        vs, _ = T.lowest(k=3); res['lowest_err'] = float(np.abs(np.sort(ev[np.argsort(np.abs(ev))[:3]]) - vs).max())
        x = rng.normal(size=m.nz); res['solve_err'] = float(np.linalg.norm(dense @ T.solve(x) - x) / np.linalg.norm(x))
    return res

def rigid_rotation(model):
    """Element level: E(R d, t + a, t + a) = E(d, t, t)."""
    m = model.m; rng = np.random.default_rng(3)
    d = m.d0 * (1 + 0.01 * rng.normal(size=(m.ne, 1))); t1 = 0.03 * rng.normal(size=m.ne); t2 = 0.03 * rng.normal(size=m.ne)
    a = 0.7; R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    E0 = element_gh(d, t1, t2, m.L, m.b0, m.EA, m.EI, hess=False)[0]
    E1 = element_gh(d @ R.T, t1 + a, t2 + a, m.L, m.b0, m.EA, m.EI, hess=False)[0]
    return float(np.abs(E1 - E0).max() / E0.max())

tl57 = C.canonical(*family_lifts(5, 7, 'B1'))
# FD: imperfect and perfect
m57 = from_lifts(tl57['lifts'], tl57['L1'], tl57['L2'], sub=2, name='B1_5/7')
rng = np.random.default_rng(1)
imp = m57.with_imperfection(0.02 * rng.normal(size=(m57.nn, 2)), np.array([[.003, .001], [.001, -.002]]))
out['b_FD'] = dict(perfect=fd_check(Model(m57)), imperfect=fd_check(Model(imp)), square=fd_check(Model(square_mesh(4)), seed=4))
out['b_rotation_invariance_rel'] = rigid_rotation(Model(imp))
print(json.dumps(out['b_FD'], indent=0)); print('rot', out['b_rotation_invariance_rel'], flush=True)

# gate (a)
rows = []
def compare(name, lifts, L1, L2, sub, sq=False, nphi=8):
    t0 = time.time()
    if sq:
        F = ObjFrame(np.zeros((1, 4), np.int64), [1, 0, 0, 0], [0, 0, 1, 0], sub=sub)
        mesh = square_mesh(sub)
    else:
        F = ObjFrame(lifts, L1, L2, sub=sub)
        mesh = from_lifts(lifts, L1, L2, sub=sub, name=name)
    F.set_load('stress')
    model = Model(mesh)
    lam_m, vec, st = model.linear_critical(macro=True)
    lam_f, _, _ = model.linear_critical(macro=False)
    # same axial forces?
    dN = float(np.abs(np.sort(st['N']) - np.sort(F.N)).max())
    gamma = F.lam((0, 0))
    ph, ext, _ = F.longwave(nphi=nphi); hh = C.harmonics(ph, ext)
    # refine the minimum over phi on the fine harmonic model; report both
    r = dict(name=name, sub=sub, V=int(mesh.nn), dN=dN, nl_lin_macro=float(lam_m[0]), nl_lin_macro2=float(lam_m[1]), nl_lin_fixed=float(lam_f[0]),
             obj_gamma=float(gamma), obj_lw_min_sampled=float(ext.min()), obj_lw_min_model=hh['min_model'], obj_phi_deg=hh['phi_min_model_deg'], obj_c0=hh['c0'], obj_h4=hh['h4'],
             err_macro_vs_lw_model=float(lam_m[0] / hh['min_model'] - 1), err_fixed_vs_gamma=float(lam_f[0] / gamma - 1),
             macro_mode=macro_content(model, vec[:, 0]), macro_mode2=macro_content(model, vec[:, 1]), sec=time.time() - t0)
    rows.append(r)
    print(json.dumps({k: r[k] for k in ('name', 'sub', 'dN', 'nl_lin_macro', 'nl_lin_macro2', 'obj_lw_min_model', 'obj_lw_min_sampled', 'err_macro_vs_lw_model', 'nl_lin_fixed', 'obj_gamma', 'err_fixed_vs_gamma', 'sec')}), flush=True)
compare('square', None, None, None, 2, sq=True)
compare('square', None, None, None, 4, sq=True)
compare('B1_5/7', tl57['lifts'], tl57['L1'], tl57['L2'], 2)
tl = C.canonical(*family_lifts(12, 17, 'B1'))
compare('B1_12/17', tl['lifts'], tl['L1'], tl['L2'], 2)
tl = C.canonical(*family_lifts(5, 7, 'B2'))
compare('B2_5/7', tl['lifts'], tl['L1'], tl['L2'], 2)
tl = C.canonical(*family_lifts(5, 7, 'E2a'))
compare('E2a_5/7', tl['lifts'], tl['L1'], tl['L2'], 2)
fig4 = json.loads((ROUGH / 'random_mode.json').read_text())
lf = np.array(fig4['lifts'], np.int64)
compare('random_fig4', lf, fig4['L1'], fig4['L2'], 2)
out['a_rows'] = rows
# supercell check against Bloch, fixed macro: S = diag(2,1): min(gamma, k=(1/2,0))
F = ObjFrame(tl57['lifts'], tl57['L1'], tl57['L2'], sub=2); F.set_load('stress')
sc = supercell(m57, [[2, 0], [0, 1]])
lam, _, _ = Model(sc).linear_critical(macro=False)
ref = min(F.lam((0, 0)), F.lam((0.5, 0)))
out['a_supercell'] = dict(sc_fixed=float(lam[0]), bloch=float(ref), err=float(lam[0] / ref - 1), V=sc.nn)
sc3 = supercell(m57, [[3, 0], [0, 1]])
lam3, _, _ = Model(sc3).linear_critical(macro=False)
ref3 = min(F.lam((0, 0)), F.lam((1 / 3, 0)))
out['a_supercell3'] = dict(sc_fixed=float(lam3[0]), bloch=float(ref3), err=float(lam3[0] / ref3 - 1), V=sc3.nn)
print(out['a_supercell'], out['a_supercell3'])
(HERE / 'tangent_checks.json').write_text(json.dumps(out, indent=1))

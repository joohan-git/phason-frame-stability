"""r3 validation of the objective operator. Writes objective_validation.json.
 V1 B^T Kn B = local elastic element.
 V2 rigid rotation: element residual and node residual (random prestressed element; real frames).
 V3 planar reduction equals the 2D corotational term (M1+M2)(r z + z r)/L^2 of objectivity/objective_frame.py.
 V4 Hessian symmetric and E_V-like translation block.
 V5 Greenhill: clamped-clamped rod under end twist, T_cr L/EI = 8.9868 (2 x 4.4934), convergence with elements.
 V6 odd and full major antisymmetry of Lambda vanish; simple-cubic long-wave unchanged (0.5518 sub 2 vs analytic 0.5506).
 V7 Bloch eigenvalue (dense, q -> 0 Richardson) equals the Schur-tensor root with the objective operator, 1/1 and 2/1.
 V8 axial_only = True reproduces stored r2 numbers."""
import os, sys, json, math
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
from icosa_objective import *
from load_axial_runs import load
import scipy.linalg as la
out = {}
L = 0.5
B, Kn, D = natural_model(L)
kl, gl = beam12(L)
out['V1_BKB_minus_kl'] = float(np.max(np.abs(B.T @ Kn @ B - kl)) / np.max(np.abs(kl)))
out['V4_hessian_asym'] = float(max(np.max(np.abs(D[a] - D[a].T)) for a in range(6)))
# V2 single random element in random orientation
rng = np.random.default_rng(1)
e = rng.normal(size=3); e /= np.linalg.norm(e)
T = rot12(e); d = rng.normal(size=12)
ql = T @ d
Sn = Kn @ (B @ ql); Fe = T.T @ kl @ ql
Gl = sum(Sn[a] * D[a] for a in range(1, 6))
GT = Sn[0] * (T.T @ gl @ T) + T.T @ Gl @ T
x1 = rng.normal(size=3); x2 = x1 + L * e
res = []
for w in np.eye(3):
    r = np.r_[np.cross(w, x1), w, np.cross(w, x2), w]
    exp = np.r_[np.cross(w, Fe[:3]), .5 * np.cross(w, Fe[3:6]), np.cross(w, Fe[6:9]), .5 * np.cross(w, Fe[9:12])]
    res.append(float(np.linalg.norm(GT @ r - exp) / np.linalg.norm(exp)))
out['V2_element_relative_residual'] = max(res)
axial = Sn[0] * (T.T @ gl @ T)
out['V2_axial_only_relative_residual'] = max(float(np.linalg.norm(axial @ np.r_[np.cross(w, x1), w, np.cross(w, x2), w] - np.r_[np.cross(w, Fe[:3]), .5 * np.cross(w, Fe[3:6]), np.cross(w, Fe[6:9]), .5 * np.cross(w, Fe[9:12])]) / np.linalg.norm(np.r_[np.cross(w, Fe[:3]), .5 * np.cross(w, Fe[3:6]), np.cross(w, Fe[6:9]), .5 * np.cross(w, Fe[9:12])])) for w in np.eye(3))
# V3 planar reduction: local x-y plane; dofs (u,v,tz) at both ends
idx = [0, 1, 5, 6, 7, 11]
M1, M2 = 0.7, -0.2
# S for phi1z, phi2z chosen to give end moments M1, M2 in the natural sense
G3 = (M1 * D[3] + M2 * D[5])[np.ix_(idx, idx)]
c, s = 1.0, 0.0
rr = np.array([-c, -s, 0, c, s, 0.]); z = np.array([s, -c, 0, -s, c, 0.])
vm = (np.outer(rr, z) + np.outer(z, rr)) / L ** 2
out['V3_planar_vs_2D_term'] = [float(np.max(np.abs(G3 - (M1 + M2) * vm))), float(np.max(np.abs(G3 + (M1 + M2) * vm))), float(np.max(np.abs(vm)))]
# V5 Greenhill
gh = {}
for nel in (4, 8, 16, 32, 64):
    Le = 1.0 / nel
    Bl, Knl, Dl = natural_model(Le); kloc, _ = beam12(Le)
    # unit torque: twist coordinate force = 1, all elements
    Gel = Dl[1]
    n = 6 * (nel + 1)
    K = np.zeros((n, n)); G = np.zeros((n, n))
    for i in range(nel):
        id_ = list(range(6 * i, 6 * i + 12))
        K[np.ix_(id_, id_)] += kloc; G[np.ix_(id_, id_)] += Gel
    fr = list(range(6, 6 * nel))
    mu = la.eigh(G[np.ix_(fr, fr)], K[np.ix_(fr, fr)], eigvals_only=True)
    mm = np.max(np.abs(mu))
    gh[str(nel)] = float(1 / mm / EI)   # T_cr L / EI, L = 1
out['V5_Greenhill_TL_over_EI'] = gh; out['V5_exact'] = 2 * 4.493409457909064
# V6 simple cubic
sc = {}
for sub in (2, 4):
    f = sc_frame(sub); prep_load(f); f.set_load('stress')
    om = ObjModel(f); H = HomogObj(f, om)
    g = lambda l: np.linalg.eigvalsh(Homog.amat(H.tensor(l)[0], [1, 0, 0]))[0]
    from scipy.optimize import brentq
    sc[str(sub)] = float(brentq(g, .4, .7, xtol=1e-9))
out['V6_SC_longwave'] = sc
# frames
fr_res = {}
for p, q in ((1, 1), (2, 1), (3, 2)):
    Lm = cubic_periods(p, q); tl = canonical(Lm, seed=1)
    f = Frame3D.from_tiling(tl['lifts'], Lm, sub=2); prep_load(f); f.set_load('stress')
    om = ObjModel(f); oa = ObjModel(f, axial_only=True)
    rt = om.rigid_test(); ra = oa.rigid_test()
    H = HomogObj(f, om); Ha = HomogObj(f, oa)
    row = dict(rigid=rt, rigid_axial=ra, Fp_over_N=om.Fp_over_N, M_over_NL=om.Mrms_over_NL, T_over_NL=om.Trms_over_NL)
    for l in (0.0, 0.5, 0.85):
        T_ = H.tensor(l)[0]; Ta = Ha.tensor(l)[0]
        row[f'odd_{l}'] = [odd_part(T_), odd_part(Ta)]; row[f'antis_{l}'] = [antis_ratio(T_), antis_ratio(Ta)]
    # V8 regression against stored
    R, _ = load(); st = R[f'cubic_{p}-{q}_s1.json']
    lg_a, _ = frame_gamma_obj(f, oa)
    row['V8_gamma_axial_vs_stored'] = [lg_a[0], st['gamma'][0]]
    lg, vg = frame_gamma_obj(f, om)
    row['gamma_objective'] = lg[0]
    fr_res[f'{p}/{q}'] = row
    print(p, q, 'rigid', rt, 'axial', ra['element_max'], ra['node_norm'], 'odd.85', row['odd_0.85'], 'G', lg[0], lg_a[0], st['gamma'][0], flush=True)
out['frames'] = fr_res
# V7 dense Bloch vs tensor
v7 = {}
for p, q in ((1, 1), (2, 1)):
    Lm = cubic_periods(p, q); tl = canonical(Lm, seed=1)
    f = Frame3D.from_tiling(tl['lifts'], Lm, sub=2); prep_load(f); f.set_load('stress')
    om = ObjModel(f); H = HomogObj(f, om)
    n = np.array([1., 0, 0]); lg, _ = frame_gamma_obj(f, om); top = lg[0] * (1 - 1e-6)
    g = lambda l: np.linalg.eigvalsh(Homog.amat(H.tensor(l)[0], n))[0]
    from scipy.optimize import brentq
    root = float(brentq(g, .5, top, xtol=1e-9))
    vals = []
    for dq in (0.002, 0.001):
        k = dq * (f.cell @ n)
        Kc = f._assemble(np.ones(len(f.ei)), f.Ke, k).toarray()
        ph = np.exp(2j * np.pi * (f.wrap @ k))
        vv = om.GT.reshape(len(f.ei), 144).astype(complex)
        phase = np.where(f.blk == 1, ph[:, None], np.where(f.blk == -1, np.conj(ph)[:, None], 1.0))
        G = sp.csr_matrix(((vv * phase).ravel(), (f.rows, f.cols)), shape=(f.n, f.n)); G = ((G + G.conj().T) * .5).toarray()
        mu = la.eigh(-G, Kc, eigvals_only=True, subset_by_index=(f.n - 3, f.n - 1))
        vals.append(float(1 / mu.max() / LAM_E))
    rich = (4 * vals[1] - vals[0]) / 3
    v7[f'{p}/{q}'] = dict(tensor_root=root, bloch_richardson=rich, diff=abs(root - rich))
    print('V7', p, q, root, rich, flush=True)
out['V7'] = v7
json.dump(out, open(os.path.join(here, 'objective_validation.json'), 'w'), indent=1, default=float)
for k in ('V1_BKB_minus_kl', 'V4_hessian_asym', 'V2_element_relative_residual', 'V2_axial_only_relative_residual', 'V3_planar_vs_2D_term', 'V5_Greenhill_TL_over_EI', 'V5_exact', 'V6_SC_longwave'):
    print(k, out[k])

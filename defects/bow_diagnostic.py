"""Exploratory diagnostic, not registered: is the early 0.05 rad edge rotation of B3 and of random tilings plain linear bending around a defect, or buckling?
Prediction from the linear solution at the reference state: p_lin05 = 0.05 / (max edge rotation per unit p). Also: linear and nonlinear codes on identical removal sets.
Writes bow_diagnostic.json. Sub 4, four realizations, seconds per case."""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames', 'design', 'nonlinear', 'phason_paths')]  # shared modules
import json
from defects_nonlinear import *
import ab_frame as C
from structures import family_lifts
OUT = Path(__file__).resolve().parent / 'out'
NL = [json.loads(p.read_text()) for p in OUT.glob('plan_nonlinear_*.json')]
LIN = [json.loads(p.read_text()) for p in OUT.glob('plan_linear_*.json')]
tl = C.canonical(*family_lifts(12, 17, 'B1'))
lifts = tl['lifts']; edges = np.asarray(C.edges_from_lifts(lifts, tl['L1'], tl['L2']), np.int64); V, E = len(lifts), len(edges)
deg0 = degrees(V, edges)


def linear_rot(mesh, keep_edges=None):
    mo = Model(mesh); st = mo.linear_state(); z = st['z'] * 1e-6
    d, t1, t2 = mo.gvec(z)
    q, _ = mo.unpack(z); th = q[2::3]; thm = th.mean()
    sub = mesh.sub; ned = mesh.ne // sub
    D = d.reshape(ned, sub, 2).sum(1); D0 = mesh.d0.reshape(ned, sub, 2).sum(1)
    pe = np.arctan2(D0[:, 0] * D[:, 1] - D0[:, 1] * D[:, 0], D0[:, 0] * D[:, 0] + D0[:, 1] * D[:, 1]) - thm
    per_p = np.abs(pe) / 1e-6 * mesh.lamE     # rotation per unit p in lamE units
    return per_p


out = dict(cases=[])
for lv, k in (('k1', 1), ('0.01', 28), ('0.05', 139)):
    for i in range(4):
        order, _ = removal_order(V, edges, f'B112/17|None|{i}', k)
        keep = np.ones(E, bool); keep[order] = False
        ek = edges[keep]
        mesh = from_edges(lifts @ C.P.T, ek, C.P, C.P @ np.array(tl['L1'], float), C.P @ np.array(tl['L2'], float), 4, 0.03, 'miss')
        r_ = linear_rot(mesh); j = int(np.argmax(r_))
        d1 = degrees(V, ek)
        nl = [x for x in NL if x['task']['kind'] == 'nlmiss' and str(x['task']['level']) == lv and x['task']['i'] == i][0]
        lin = [x for x in LIN if x['task']['kind'] == 'miss' and x['task'].get('group') == 'mesh' and str(x['task']['level']) == lv and x['task']['i'] == i and x['task'].get('sub') == 4][0]
        lamN = float(Model(mesh).linear_critical(macro=True, nev=2)[0][0])
        out['cases'].append(dict(kind='missing', level=lv, i=i, pe05_nl=nl['pe_05'], pe05_linear_pred=0.05 / r_[j], ends_deg_now=[int(d1[ek[int(j), 0]]), int(d1[ek[int(j), 1]])], lam_designcode=lin['lam_c'], lam_nonlinearcode=lamN, n_edges_over_0p03=int((r_ * 0.3 > 0.03).sum())))
        print(out['cases'][-1], flush=True)
for name in ('R22', 'R101', 'B1_12_17'):
    mesh = get_mesh(name, 4); r_ = linear_rot(mesh); j = int(np.argmax(r_))
    nl = [x for x in NL if x['task']['kind'] == 'bow' and x['task']['name'] == name and x['task']['sub'] == 4 and x['task']['w0'] == 1e-4]
    out['cases'].append(dict(kind='intact', name=name, pe05_nl=float(np.mean([x['pe_05'] for x in nl])), pe05_linear_pred=0.05 / r_[j]))
    print(out['cases'][-1], flush=True)
(HERE_defects / 'bow_diagnostic.json').write_text(json.dumps(out, indent=1, default=float), encoding='utf-8')

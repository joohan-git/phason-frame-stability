"""Fig. 4 (and Fig. 1d) data: one equilibrated random AB tiling at the 12/17 Pell period, its critical
buckling mode, local internal-space gradient and window excess.

Recipe = ensemble.run (main/, copied logic, unchanged): canonical 12/17 cut -> flip MC in chunks of
target/10 sweeps with seed*100003+k, target = max(200, 0.1 L^2), plateau test on internal variance;
Frame(sub=2), stress protocol, Gamma-point critical mode (for random tilings pc_gamma = pc_lw_min to 1e-5,
see data/drift). The recomputed pc_gamma is checked against data/drift/drift_12-17_sc1_random_s<seed>.json.

Usage: python -X utf8 random_mode.py [seed]   -> random_mode.json
"""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames',)]  # shared modules
import os, sys, json, math, time
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ab_frame as C   # verbatim copy of main/ab_frame.py
from scipy.spatial import cKDTree

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 22
P_, Q_ = 12, 17
DRIFT = HERE / 'ab_ensemble'


def equilibrate(tl, seed, sweeps=200.0):
    lifts = tl['lifts']; V = len(lifts)
    Lc = float(np.linalg.norm(C.P @ np.array(tl['L1'], float)))
    target = max(float(sweeps), 0.1 * Lc ** 2)
    chunk = max(1.0, target / 10)
    trace = []; done = 0.0; k = 0
    while True:
        lifts, a = C.flip_mc(lifts, tl['L1'], tl['L2'], int(chunk * V), seed * 100003 + k)
        done += chunk; k += 1
        y = lifts @ tl['perp'].T
        pv = float(np.mean(np.sum((y - y.mean(0)) ** 2, 1)))
        trace.append([done, pv])
        if done >= target and len(trace) >= 9:
            v = np.array([x[1] for x in trace])
            if abs(v[-3:].mean() - v[-6:-3].mean()) / max(v[-6:-3].mean(), 1e-12) < 0.10:
                break
        if done >= 3 * target:
            break
    return lifts, trace, done


def faces(lifts, L1, L2):
    """Rhombic faces {r, r+e_k, r+e_l, r+e_k+e_l} (k<l), each listed once by its corner r. Returns
    unreduced corner lifts (4 x 4 ints) so the face is geometrically intact across the cell boundary."""
    red = C.Reducer(L1, L2)
    look = {tuple(r) for r in lifts}
    E = np.eye(4, dtype=np.int64)
    out = []
    for r in lifts:
        for k in range(4):
            for l in range(k + 1, 4):
                c = [r, r + E[k], r + E[k] + E[l], r + E[l]]
                if all(tuple(red(x)[0]) in look for x in c[1:]):
                    out.append((np.array(c).tolist(), k, l))
    return out


def local_gradient(lifts, L1, L2, perp, R=4.0):
    """Per-vertex |grad y|^2 of the internal coordinate y(x) (least-squares plane within radius R) —
    same estimator as ab_frame.phason_stats (orthogonal-cell branch), kept per vertex."""
    X = lifts @ C.P.T
    cell = np.array([C.P @ np.array(L1, float), C.P @ np.array(L2, float)])
    box = np.array([cell[0, 0], cell[1, 1]])
    y = lifts @ perp.T
    Xm = np.mod(X, box); Xm[Xm >= box] = 0.0
    tree = cKDTree(Xm, boxsize=box)
    G2 = np.zeros(len(X))
    for i, nb in enumerate(tree.query_ball_point(Xm, R)):
        d = Xm[nb] - Xm[i]; d -= box * np.round(d / box)
        A = np.c_[d, np.ones(len(nb))]
        coef, *_ = np.linalg.lstsq(A, y[nb], rcond=None)
        G2[i] = np.sum(coef[:2] ** 2)
    return G2


def main():
    T0 = time.time()
    L1, L2 = C.ab_approx(P_, Q_, 1)
    tl = C.canonical(L1, L2)
    can = tl['lifts'].copy()
    lifts, trace, done = equilibrate(tl, SEED)
    bad = C.check_tiling(lifts, tl['L1'], tl['L2'])
    st = C.phason_stats(lifts, tl['L1'], tl['L2'], tl['perp'], can)
    f = C.Frame(lifts, tl['L1'], tl['L2'], sub=2)
    f.set_load('stress')
    pc, vec = f.lam((0, 0), vectors=True)
    # canonical reference (same Frame settings), Gamma and the published long-wave minimum
    fc = C.Frame(can, tl['L1'], tl['L2'], sub=2); fc.set_load('stress')
    pc_can_gamma = float(fc.lam((0, 0)))
    N_can = fc.N.copy()
    ref = json.loads((DRIFT / f'drift_12-17_sc1_random_s{SEED}.json').read_text())
    refc = json.loads((DRIFT / 'drift_12-17_sc1_canonical_s0.json').read_text())
    # real mode: rotate the global phase so the vector is (nearly) real
    ph = np.angle(np.sum(vec ** 2)) / 2
    u = np.real(vec * np.exp(-1j * ph))
    u /= np.max(np.hypot(u[0::3], u[1::3]))       # max node translation = 1
    ux, uy, th = u[0::3], u[1::3], u[2::3]
    w = ux ** 2 + uy ** 2
    PR = float(np.sum(w) ** 2 / (len(w) * np.sum(w ** 2)))
    V = len(lifts)
    wv = w[:V]
    G2 = local_gradient(lifts, tl['L1'], tl['L2'], tl['perp'])
    G2c = local_gradient(can, tl['L1'], tl['L2'], tl['perp'])
    ex = C.window_excess(lifts, tl)
    exc = C.window_excess(can, tl)
    # mode-weighted centre (circular mean; biased by tails, use mode_peak_X for the cluster)
    X = lifts @ C.P.T
    Lc = float(f.cell[0, 0])
    ang = 2 * np.pi * X / Lc
    cen = [float((np.angle(np.sum(wv * np.exp(1j * ang[:, d]))) % (2 * np.pi)) * Lc / (2 * np.pi)) for d in range(2)]
    out = dict(
        status='P',
        units='p_c/(EA lambda_E), stress protocol, sub=2, Gamma point',
        p=P_, q=Q_, m=1, seed=SEED, V=V, sweeps_done=done, equil_trace=trace, invalid_vertices=int(bad),
        moved_fraction=st['moved_fraction'], perp_var=st['perp_var'],
        local_phason_strain_ms_R4=st.get('local_phason_strain_ms_R4'),
        pc_gamma_recomputed=float(pc), pc_gamma_ensemble=ref['pc_gamma'], pc_min_all_ensemble=ref['pc_min_all'],
        pc_gamma_rel_diff=float(pc / ref['pc_gamma'] - 1),
        pc_canonical_gamma_recomputed=pc_can_gamma, pc_canonical_min_all_ensemble=refc['pc_min_all'],
        participation_ratio=PR, mode_centre=cen, cell_length=Lc,
        L1=list(map(int, tl['L1'])), L2=list(map(int, tl['L2'])),
        perp=tl['perp'].tolist(), cperp=tl['cperp'].tolist(), window_eq=tl['window_eq'].tolist(),
        lifts=lifts.tolist(), lifts_canonical=can.tolist(),
        faces=faces(lifts, tl['L1'], tl['L2']), faces_canonical=faces(can, tl['L1'], tl['L2']),
        local_grad2=G2.tolist(), local_grad2_canonical=G2c.tolist(),
        window_excess=ex.tolist(), window_excess_canonical=exc.tolist(),
        node_X=f.X.tolist(), elem_i=f.ei.tolist(), elem_j=f.ej.tolist(), elem_wrap=f.wrap.tolist(),
        elem_dir=f.dirk.tolist(), sub=f.sub, axial_N=f.N.tolist(),
        axial_N_canonical=N_can.tolist(), axial_N_min=float(f.N.min()), axial_N_canonical_min=float(N_can.min()),
        mode_peak_node=int(np.argmax(ux ** 2 + uy ** 2)), mode_peak_X=f.X[int(np.argmax(ux ** 2 + uy ** 2))].tolist(),
        mode_ux=ux.tolist(), mode_uy=uy.tolist(), mode_theta=th.tolist(), mode_weight_vertex=wv.tolist(),
        seconds=time.time() - T0)
    (HERE / 'random_mode.json').write_text(json.dumps(out), encoding='utf-8')
    print(json.dumps({k: out[k] for k in ['seed', 'V', 'invalid_vertices', 'moved_fraction', 'pc_gamma_recomputed', 'pc_gamma_ensemble',
                                          'pc_gamma_rel_diff', 'pc_canonical_gamma_recomputed', 'participation_ratio',
                                          'mode_centre', 'seconds']}, indent=1))


if __name__ == '__main__':
    main()

"""R2-C(b) same-slope phason disorder and R2-C(c) planted square patch (K08 H2 shielding),
both in the AB 12/17 approximant (V = 1393), stress protocol (p_c / (EA lambda_E)).
Run: python flips.py flips   -> flips.json
     python flips.py plant   -> planted.json"""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames', 'slope_survey')]  # shared modules
import json, math, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import ab_frame as C

HERE = Path(__file__).parent
P_, Q_ = 12, 17


def mech(lifts, L1, L2, nphi=8, center=None, radius=None):
    t0 = time.time()
    f = C.Frame(lifts, L1, L2, sub=2); f.set_load('stress')
    phis, ext, tab = f.longwave(nphi=nphi)
    out = dict(pc_lw_min=float(ext.min()), pc_lw_phi=ext.tolist(), harm=C.harmonics(phis, ext),
               N_mean=float(-f.N.mean()), N_max=float(-f.N.min()), N_min=float(-f.N.max()))
    lg, vg = f.lam((0, 0), vectors=True)
    out['pc_gamma'] = float(lg); out['pc_k_half_half'] = float(f.lam((0.5, 0.5)))
    out['pc_k_half_0'] = float(f.lam((0.5, 0.0)))
    if center is not None:
        X = f.X - np.asarray(center)
        ci = np.linalg.inv(f.cell); fr = X @ ci; fr -= np.floor(fr + 0.5); X = fr @ f.cell
        inside = np.sum(X ** 2, 1) < radius ** 2
        amp = np.abs(vg.reshape(-1, 3)[:, :2]) ** 2
        w = amp.sum(1)
        out['gamma_mode_weight_in_disk'] = float(w[inside].sum() / w.sum())
        out['disk_area_fraction'] = float(math.pi * radius ** 2 / abs(np.linalg.det(f.cell)))
    out['pc_min_all'] = float(min(out['pc_lw_min'], out['pc_gamma'], out['pc_k_half_half'], out['pc_k_half_0']))
    out['seconds'] = time.time() - t0
    return out


def flips():
    t = C.canonical(*C.ab_approx(P_, Q_)); L1, L2 = t['L1'], t['L2']
    base = C.phason_stats(t['lifts'], L1, L2, t['perp'], t['lifts'])
    rows = [dict(k=0, seed=None, **base, **mech(t['lifts'], L1, L2))]
    print(json.dumps({x: rows[-1][x] for x in ['k', 'pc_lw_min', 'pc_gamma', 'seconds']}), flush=True)
    for seed in (1, 2, 3):
        for k in (10, 30, 100, 300, 1000, 3000, -1):
            if k > 0:
                l, acc = C.flip_mc(t['lifts'], L1, L2, 50 * k + 10000, seed * 1000 + k, maxacc=k)
            else:
                l, acc = C.flip_mc(t['lifts'], L1, L2, 4_000_000, seed * 1000 + 7)
            assert C.check_tiling(l, L1, L2) == 0
            st = C.phason_stats(l, L1, L2, t['perp'], t['lifts'])
            rows.append(dict(k=k if k > 0 else 'equilibrated(4e6 attempts)', accepted=acc, seed=seed, **st, **mech(l, L1, L2)))
            print(json.dumps({x: rows[-1][x] for x in ['k', 'seed', 'moved_fraction', 'local_phason_strain_ms_R4', 'pc_lw_min', 'pc_gamma', 'seconds']}), flush=True)
    (HERE / 'flips.json').write_text(json.dumps(dict(
        system=f'AB {P_}/{Q_} approximant, V=1393, periodic, same slope (period lifts fixed)',
        units='p_c/(EA lambda_E), stress protocol, 2 elements/edge, long-wave Richardson DQ->0, 8 directions',
        status='P', rows=rows), indent=1), encoding='utf-8')


def plant():
    t = C.canonical(*C.ab_approx(P_, Q_)); L1, L2 = t['L1'], t['L2']
    cell = np.array([C.P @ L1, C.P @ L2]); c = cell.sum(0) / 2
    rows = []
    sched = [(0.5, 3e6), (1.0, 3e6), (2.0, 3e6), (4.0, 4e6), (8.0, 3e6)]
    for R, rm in [(3.0, 9.0), (4.5, 11.0), (6.0, 13.0), (8.0, 17.0)]:
        l = t['lifts']
        for b, n in sched:
            l, _ = C.flip_mc(l, L1, L2, int(n), int(10 * b + 7 * R), bias=b, center=c, radius=R, rmove=rm)
        assert C.check_tiling(l, L1, L2) == 0
        nin, nsq = C.disk_square_fraction(l, L1, L2, c, R)
        # radius of the fully square core
        core = 0.0
        for rr in np.arange(1.0, R + 0.01, 0.5):
            a, b_ = C.disk_square_fraction(l, L1, L2, c, rr)
            if a and a == b_:
                core = rr
        st = C.phason_stats(l, L1, L2, t['perp'], t['lifts'])
        m = mech(l, L1, L2, center=c, radius=max(core, 1.0))
        rows.append(dict(R_target=R, rmove=rm, disk_vertices=nin, disk_square_vertices=nsq, R_core=core, **st, **m))
        print(json.dumps({x: rows[-1][x] for x in ['R_target', 'R_core', 'disk_square_vertices', 'moved_fraction', 'pc_lw_min', 'pc_gamma', 'pc_k_half_half', 'gamma_mode_weight_in_disk']}), flush=True)
    (HERE / 'planted.json').write_text(json.dumps(dict(
        system=f'AB {P_}/{Q_} approximant with a biased-flip planted pure-square (e1,e3) disk; flips limited to rmove',
        reference=dict(pure_square_pc=0.5518, canonical_12_17_pc='see flips.json row k=0'),
        units='p_c/(EA lambda_E), stress protocol', status='P', rows=rows), indent=1), encoding='utf-8')


if __name__ == '__main__':
    {'flips': flips, 'plant': plant}[sys.argv[1]]()

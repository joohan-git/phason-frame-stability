"""single-flip raise and the small-disorder regime (stress protocol, p_c/(EA lambda_E)).
 (i)  B1 5/7 canonical (V=239, main offset): EVERY legal single flip, sub=4 (+ sub=8 for extremes).
 (ii) B1 5/7: k = 1,2,3,5,10,20,30 accepted flips x 6 seeds, sub=4.
 (iii) B1 12/17 (V=1393): k = 1,3,10,30 x 5 seeds, sub=2.
p_c = min(long-wave 8 directions Richardson, Gamma). Output flip_ladder.json (all P)."""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames', 'slope_survey')]  # shared modules
import json, math, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import ab_frame as C

OUT = Path(__file__).with_name('flip_ladder.json')
E4 = np.eye(4, dtype=np.int64)


def pc(lifts, L1, L2, sub, nphi=8):
    f = C.Frame(lifts, L1, L2, sub=sub); f.set_load('stress')
    phis, ext, _ = f.longwave(nphi=nphi)
    g = f.lam((0, 0))
    return dict(pc=float(min(ext.min(), g)), pc_lw=float(ext.min()), pc_gamma=float(g),
                c0=C.harmonics(phis, ext)['c0'], h4=C.harmonics(phis, ext)['h4'])


def legal_flips(lifts, L1, L2):
    red = C.Reducer(L1, L2)
    look = {tuple(r): i for i, r in enumerate(lifts)}
    out = []
    for v, r in enumerate(lifts):
        ds = [s * E4[k] for k in range(4) for s in (1, -1) if tuple(red(r + s * E4[k])[0]) in look]
        if len(ds) != 3:
            continue
        ph = [C.P @ d for d in ds]
        cr = [ph[a][0] * ph[(a + 1) % 3][1] - ph[a][1] * ph[(a + 1) % 3][0] for a in range(3)]
        if not (all(c > 1e-9 for c in cr) or all(c < -1e-9 for c in cr)):
            continue
        if any(tuple(red(r + ds[a] + ds[b])[0]) not in look for a in range(3) for b in range(a + 1, 3)):
            continue
        new = red(r + ds[0] + ds[1] + ds[2])[0]
        if tuple(new) in look:
            continue
        nb = sum(tuple(red(new + s * E4[k])[0]) in look for k in range(4) for s in (1, -1))
        if nb != 3:
            continue
        l2 = lifts.copy(); l2[v] = new
        out.append((v, l2))
    return out


def main():
    res = {}
    t = C.canonical(*C.ab_approx(5, 7)); L1, L2 = t['L1'], t['L2']
    base = {s: pc(t['lifts'], L1, L2, s) for s in (4, 8)}
    res['canonical_5_7'] = base
    print('canonical', base, flush=True)
    singles = []
    for v, l2 in legal_flips(t['lifts'], L1, L2):
        assert C.check_tiling(l2, L1, L2) == 0
        r = pc(l2, L1, L2, 4); r['vertex'] = int(v); r['delta_sub4'] = r['pc'] - base[4]['pc']
        singles.append(r)
        print('single', v, r['delta_sub4'], flush=True)
    order = sorted(range(len(singles)), key=lambda i: singles[i]['delta_sub4'])
    for i in order[:3] + order[-3:]:
        v = singles[i]['vertex']
        l2 = t['lifts'].copy(); l2 = dict(legal_flips(t['lifts'], L1, L2))[v]
        r8 = pc(l2, L1, L2, 8)
        singles[i]['pc_sub8'] = r8['pc']; singles[i]['delta_sub8'] = r8['pc'] - base[8]['pc']
    res['single_flips_5_7'] = singles
    OUT.write_text(json.dumps(dict(status='P', res=res), indent=1), encoding='utf-8')
    ladder = []
    for k in (1, 2, 3, 5, 10, 20, 30):
        for seed in range(1, 7):
            l, acc = C.flip_mc(t['lifts'], L1, L2, 200 * k + 20000, 7000 + 100 * seed + k, maxacc=k)
            st = C.phason_stats(l, L1, L2, t['perp'], t['lifts'], radii=())
            r = pc(l, L1, L2, 4); r.update(k=k, seed=seed, accepted=acc, moved=st['moved_fraction'], delta=r['pc'] - base[4]['pc'])
            ladder.append(r)
            print('ladder57', k, seed, r['moved'], r['delta'], flush=True)
    res['ladder_5_7_sub4'] = ladder
    OUT.write_text(json.dumps(dict(status='P', res=res), indent=1), encoding='utf-8')
    t = C.canonical(*C.ab_approx(12, 17)); L1, L2 = t['L1'], t['L2']
    b12 = pc(t['lifts'], L1, L2, 2)
    res['canonical_12_17_sub2'] = b12
    ladder = []
    for k in (1, 3, 10, 30):
        for seed in range(1, 6):
            l, acc = C.flip_mc(t['lifts'], L1, L2, 200 * k + 20000, 9000 + 100 * seed + k, maxacc=k)
            st = C.phason_stats(l, L1, L2, t['perp'], t['lifts'], radii=())
            r = pc(l, L1, L2, 2); r.update(k=k, seed=seed, accepted=acc, moved=st['moved_fraction'], delta=r['pc'] - b12['pc'])
            ladder.append(r)
            print('ladder1217', k, seed, r['moved'], r['delta'], flush=True)
    res['ladder_12_17_sub2'] = ladder
    OUT.write_text(json.dumps(dict(status='P', res=res), indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()

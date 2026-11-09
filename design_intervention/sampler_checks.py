"""intervention sampler and census gates. No frame calculation. Writes sampler_checks.json. All P except exact membership counts (C given the atlas)."""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames', 'penrose', 'heldout_test', 'design', 'nonlinear', 'defects', 'phason_paths', 'independent_code', 'forbidden_patches')]  # shared modules
import json, sys, time, collections
import numpy as np
from intervention_core import *
import cell_tiling as TT
OUT = {}


def sampler(cls, name, off=None):
    tl = cell(cls, name, off)
    return tl, Sampler(cls, tl['L1'], tl['L2'], tl['lifts'], tl['perp'], tl['lifts'])


# G_C1: atlas against atlas (independent, independent); keys are offsets in the atlas star convention
def g_atlas():
    d = json.load(open(M / 'forbidden_patches' / 'patch_atlas.json', encoding='utf8'))
    res = {}
    for cls, lab in (('ab', 'AB'), ('pen', 'Penrose')):
        s = d['symmetries'][lab]
        res[cls] = dict(keys=list(s.keys()), n_intervention=int(len(load_atlas(cls))))
        for k, v in s.items():
            if isinstance(v, (int, float)):
                res[cls][k] = v
    return res


# G_C2: census on canonical cuts against heldout forbidden_fraction, same lifts through heldout Cell (shared tilings)
def g_heldout():
    rows = []
    for cls in ('ab', 'pen'):
        star = TT.Star(cls)
        atl, _ = TT.build_atlas(star, 110, TT.make_gamma(star, 999), radii=(2,), reduce=False)
        sel = json.load(open(M / 'heldout_test' / f'selection_{cls}.json'))['selected']
        for c in sel[:: max(1, len(sel) // 7)]:
            cl = TT.Cell(star, c['L1'], c['L2'], TT.make_gamma(star, 4242))
            ff = TT.forbidden_fraction(cl, atl)[2]
            # the same tiling on the lift lookup; heldout basis, so masks need the heldout convention for Penrose: map lifts to penrose_frame basis
            if cls == 'ab':
                lifts = cl.n.copy(); L1, L2 = np.array(c['L1']), np.array(c['L2'])
            else:
                mp = lambda n: np.array([n[..., 0], -n[..., 3], n[..., 1], -n[..., 4], n[..., 2]]).T if n.ndim == 2 else np.array([n[0], -n[3], n[1], -n[4], n[2]])
                lifts = mp(cl.n); L1, L2 = mp(np.array(c['L1'])), mp(np.array(c['L2']))
            red = MODS[cls].Reducer(L1, L2)
            lifts = np.array([red(r)[0] for r in lifts], np.int64)
            q_, _ = np.linalg.qr(np.c_[np.array([L1, L2], float).T, np.eye(star.d)])
            s = Sampler(cls, L1, L2, lifts, q_[:, 2:star.d].T, lifts)
            rows.append(dict(cls=cls, name=c['role'], V=int(cl.V), heldout=float(ff), intervention=float(s.phi2), diff=abs(float(ff) - float(s.phi2))))
    return rows


# G_C3: R22 of Fig. 4 and the ensemble stream
def g_fig4():
    d = json.loads((ROOT / 'disorder' / 'random_mode.json').read_text())
    tl = cell('ab', (12, 17))
    s = Sampler('ab', d['L1'], d['L2'], np.array(d['lifts'], np.int64), tl['perp'], tl['lifts'])
    return dict(phi2=float(s.phi2), keys=[k for k in d.keys() if 'phi' in k or 'forb' in k], stored={k: d[k] for k in d if ('phi' in k or 'forb' in k) and not isinstance(d[k], (list, dict))}, var=s.var, moved=s.moved)


# G_S1: beta = 0 is bitwise the parent stream (ensemble chunking), census running or not
def g_bitwise():
    out = {}
    for cls, name, chunk in (('ab', (12, 17), 500), ('pen', 5, 500)):
        tl, s = sampler(cls, name)
        C = MODS[cls]; lp = tl['lifts'].copy(); same = True; accs = []
        for k in range(4):
            lp, a = C.flip_mc(lp, tl['L1'], tl['L2'], chunk * s.V, 100003 + k)
            a2, _, _ = s.run(chunk * s.V, 100003 + k, beta=0.0)
            accs.append((a, a2)); same = same and np.array_equal(lp, s.lifts)
        s.refresh()
        out[cls] = dict(bitwise=bool(same), acc=accs)
    return out


# G_S2: tracked N_f and flags match a full recount after a long biased run; all moves legal
def g_tracked():
    out = {}
    for cls, name in (('ab', (12, 17)), ('pen', 5)):
        C = MODS[cls]
        for beta in (0.5, -0.5, 2.0):
            tl, s = sampler(cls, name); t0 = time.time()
            s.run(50 * s.V, 11, beta=0.0); s.refresh()
            a, l, r = s.run(200 * s.V, 12, beta=beta)
            out[f'{cls}_{beta}'] = dict(ok=bool(s.tracked_ok()), bad=int(C.check_tiling(s.lifts, tl['L1'], tl['L2'])), phi2=float(s.phi2), var=s.var, acc=a, legal=l, rej=r, sec=time.time() - t0)
    return out


# G_S3: exact detailed balance on a real cell with flips restricted to a small vertex-index set: the whole component is enumerated, the exact Boltzmann law is compared with the chain.
def comp_masked(s, cap=20000):
    """States = lifts of the allowed indices in index order. Same legality as the sampler, by brute force on the tuple."""
    C = s.C; red = C.Reducer(s.L1, s.L2); E = np.eye(s.d, dtype=np.int64); idx = np.nonzero(s.allow)[0]
    base = s.lifts.copy(); start = tuple(tuple(int(x) for x in base[i]) for i in idx)
    fixed = {tuple(int(x) for x in base[i]) for i in range(s.V) if not s.allow[i]}
    seen = {start: 0}; todo = [start]
    while todo:
        st = todo.pop(); look = fixed | set(st)
        for a_, r in enumerate(st):
            r_ = np.array(r, np.int64); ds = []
            for k in range(s.d):
                for sg in (1, -1):
                    if tuple(int(x) for x in red(r_ + sg * E[k])[0]) in look:
                        ds.append(sg * E[k])
            if len(ds) != 3:
                continue
            pts = [C.P @ d for d in ds]
            cr = [pts[a][0] * pts[(a + 1) % 3][1] - pts[a][1] * pts[(a + 1) % 3][0] for a in range(3)]
            if not (all(x > 1e-9 for x in cr) or all(x < -1e-9 for x in cr)):
                continue
            if any(tuple(int(x) for x in red(r_ + ds[a] + ds[b])[0]) not in look for a in range(3) for b in range(a + 1, 3)):
                continue
            new = tuple(int(x) for x in red(r_ + ds[0] + ds[1] + ds[2])[0])
            if new in look:
                continue
            if sum(tuple(int(x) for x in red(np.array(new) + sg * E[k])[0]) in look for k in range(s.d) for sg in (1, -1)) != 3:
                continue
            nx = st[:a_] + (new,) + st[a_ + 1:]
            if nx not in seen:
                seen[nx] = len(seen); todo.append(nx)
                if len(seen) > cap:
                    return None
    return seen


def g_exact():
    """Partly flipped real cell, flips restricted to a disk, disk chosen for the widest N_f range among components of 50 to 3000 states. Restraint at the production kappa."""
    out = {}
    for cls, name, nfl in (('ab', (5, 7), 25), ('pen', 3, 25)):
        tl, s0 = sampler(cls, name)
        s0.run(100 * s0.V, 99, maxacc=nfl); s0.refresh()
        X = s0.lifts @ s0.C.P.T; start = s0.lifts.copy()
        best = None
        for ctr in range(0, s0.V, max(1, s0.V // 24)):
            for R in (1.2, 1.5, 1.8, 2.1):
                s0.allow[:] = (np.linalg.norm(X - X[ctr], axis=1) <= R).astype(np.int64)
                c = comp_masked(s0, cap=3000)
                if c is None:
                    break
                if len(c) < 50:
                    continue
                st = list(c.keys()); idx = np.nonzero(s0.allow)[0]; E = []
                for t in st:
                    l = start.copy(); l[idx] = np.array(t, np.int64)
                    E.append(Sampler(cls, tl['L1'], tl['L2'], l, tl['perp'], tl['lifts'], atlas=s0.atlas).st[0])
                rg = max(E) - min(E)
                if best is None or rg > best[0]:
                    best = (rg, R, s0.allow.copy(), c, ctr)
        if best is None:
            out[f'{cls}{name}'] = dict(component='no usable disk'); continue
        rg, R, allow, comp, ctr = best
        states = list(comp.keys()); idx = np.nonzero(allow)[0]
        Es, Vs, Ms = [], [], []
        for st in states:
            l = start.copy(); l[idx] = np.array(st, np.int64)
            s2 = Sampler(cls, tl['L1'], tl['L2'], l, tl['perp'], tl['lifts'], atlas=s0.atlas)
            Es.append(s2.st[0]); Vs.append(s2.var); Ms.append(s2.moved)
        Es = np.array(Es); Vs = np.array(Vs); Ms = np.array(Ms)
        for beta, kappa, cmode, frac in ((0.5, 0.0, 0, 0), (-0.5, 0.0, 0, 0), (2.0, 0.0, 0, 0), (0.0, 0.0, 0, 0), (0.5, 30.0, 1, 0.3), (0.5, 30.0, 2, 0.3)):
            Cs = Vs if cmode == 1 else Ms
            cstar = float(Cs.min() + frac * (Cs.max() - Cs.min()))
            W = np.exp(-beta * Es - (kappa * s0.V / 2 * (Cs - cstar) ** 2 if kappa > 0 else 0)); W /= W.sum()
            s = Sampler(cls, tl['L1'], tl['L2'], start, tl['perp'], tl['lifts'], atlas=s0.atlas); s.allow = allow.copy()
            hist = np.zeros(len(states)); n = 40000
            for i in range(n):
                s.run(10 * s.V, 5000 + i, beta=beta, kappa=kappa, cstar=cstar, cmode=cmode)
                hist[comp[tuple(tuple(int(x) for x in s.lifts[j]) for j in idx)]] += 1
            hist /= n
            ok = W * n >= 5
            chi = float((((hist - W) ** 2 / W)[ok]).sum() * n); dof = int(ok.sum() - 1)
            k = f'{cls}{name}_R{R}_b{beta}_k{kappa}_m{cmode}'
            tv = 0.5 * float(np.abs(hist - W).sum()); tvn = 0.5 * float(np.sum(np.sqrt(2 * W * (1 - W) / (np.pi * n))))
            out[k] = dict(n_states=len(states), n_allowed=int(allow.sum()), center=int(ctr), Nf_range=[float(Es.min()), float(Es.max())], E_exact=float(W @ Es), E_chain=float(hist @ Es), TV=tv, TV_null=tvn, chi2=chi, dof=dof, chi2_per_dof=chi / max(dof, 1), tracked=bool(beta == 0 or s.tracked_ok()))
            print(k, out[k], flush=True)
    return out


def g_exact3():
    """Only the failed case of g_exact2 (Penrose disk, beta 0.2), thinning 200 V instead of 20 V, n = 4000. Tests whether the failure is autocorrelation."""
    cls, name, ctr, R = 'pen', 3, 60, 1.2
    tl, s0 = sampler(cls, name)
    s0.run(100 * s0.V, 99, maxacc=25); s0.refresh()
    X = s0.lifts @ s0.C.P.T; start = s0.lifts.copy()
    s0.allow[:] = (np.linalg.norm(X - X[ctr], axis=1) <= R).astype(np.int64)
    allow = s0.allow.copy(); comp = comp_masked(s0, cap=3000)
    states = list(comp.keys()); idx = np.nonzero(allow)[0]
    Es = []
    for st in states:
        l = start.copy(); l[idx] = np.array(st, np.int64)
        Es.append(Sampler(cls, tl['L1'], tl['L2'], l, tl['perp'], tl['lifts'], atlas=s0.atlas).st[0])
    Es = np.array(Es); beta = 0.2
    W = np.exp(-beta * Es); W /= W.sum()
    s = Sampler(cls, tl['L1'], tl['L2'], start, tl['perp'], tl['lifts'], atlas=s0.atlas); s.allow = allow.copy()
    hist = np.zeros(len(states)); n = 4000
    for i in range(n):
        s.run(200 * s.V, 70000 + i, beta=beta)
        hist[comp[tuple(tuple(int(x) for x in s.lifts[j]) for j in idx)]] += 1
    hist /= n
    ok = W * n >= 5
    chi = float((((hist - W) ** 2 / W)[ok]).sum() * n); dof = int(ok.sum() - 1)
    tv = 0.5 * float(np.abs(hist - W).sum()); tvn = 0.5 * float(np.sum(np.sqrt(2 * W * (1 - W) / (np.pi * n))))
    out = dict(E_exact=float(W @ Es), E_chain=float(hist @ Es), TV=tv, TV_null=tvn, TV_ratio=tv / tvn, chi2_per_dof=chi / max(dof, 1), dof=dof, passed=bool(chi / max(dof, 1) <= 2 and tv <= 1.5 * tvn))
    print(out, flush=True)
    return out


def g_edges():
    """Detailed balance edge by edge, without mixing. On the two gate disks, every state S and allowed vertex v: a forced flip gives S'.
    C part: S' is in the enumerated component; the reverse flip at v returns S; flip count per state equals the brute-force neighbour count; tracked N_f, variance and moved equal a recount of S'.
    P part: on 300 random edges, acceptance per legal proposal at beta 0.2, kappa 30 (variance) against min(1, W'/W), 400 trials per direction."""
    out = {}
    rng = np.random.default_rng(16)
    for cls, name, ctr, R in (('ab', (5, 7), 27, 2.1), ('pen', 3, 60, 1.2)):
        tl, s0 = sampler(cls, name)
        s0.run(100 * s0.V, 99, maxacc=25); s0.refresh()
        X = s0.lifts @ s0.C.P.T; start = s0.lifts.copy()
        s0.allow[:] = (np.linalg.norm(X - X[ctr], axis=1) <= R).astype(np.int64)
        allow = s0.allow.copy(); comp = comp_masked(s0, cap=3000)
        states = list(comp.keys()); idx = np.nonzero(allow)[0]
        E_, V_, M_ = {}, {}, {}

        def fresh(st):
            l = start.copy(); l[idx] = np.array(st, np.int64)
            return Sampler(cls, tl['L1'], tl['L2'], l, tl['perp'], tl['lifts'], atlas=s0.atlas)
        for st in states:
            q = fresh(st); E_[st] = q.st[0]; V_[st] = q.var; M_[st] = q.moved
        nedge = 0; bad_comp = 0; bad_rev = 0; bad_track = 0; edges = []
        for st in states:
            for a_, v in enumerate(idx):
                q = fresh(st); q.allow[:] = 0; q.allow[v] = 1
                acc, _, _ = q.run(40 * q.V, 11, beta=1e-9, maxacc=1)
                if acc == 0:
                    continue
                nedge += 1
                st2 = tuple(tuple(int(x) for x in q.lifts[j]) for j in idx)
                if st2 not in comp:
                    bad_comp += 1; continue
                vt = q.st[1] / q.V - float(np.sum((q.st[2:2 + q.perp.shape[0]] / q.V) ** 2))
                if q.st[0] != E_[st2] or abs(vt - V_[st2]) > 1e-9 or abs(q.moved - M_[st2]) > 1e-12:
                    bad_track += 1
                r = fresh(st2); r.allow[:] = 0; r.allow[v] = 1
                acc2, _, _ = r.run(40 * r.V, 12, beta=1e-9, maxacc=1)
                if acc2 == 0 or tuple(tuple(int(x) for x in r.lifts[j]) for j in idx) != st:
                    bad_rev += 1
                edges.append((st, st2, int(v)))
        res = dict(n_states=len(states), n_edges=nedge, bad_component=bad_comp, bad_reverse=bad_rev, bad_tracked=bad_track)
        beta, kappa = 0.2, 30.0; cstar = float(np.mean(list(V_.values()))); z = []
        for e in rng.choice(len(edges), size=min(300, len(edges)), replace=False):
            st, st2, v = edges[e]
            dU = beta * (E_[st2] - E_[st]) + 0.5 * kappa * s0.V * ((V_[st2] - cstar) ** 2 - (V_[st] - cstar) ** 2)
            p = min(1.0, float(np.exp(-dU)))
            if p < 0.02:
                continue
            q = fresh(st); q.allow[:] = 0; q.allow[v] = 1
            ntr = 0; nacc = 0
            for t in range(400):
                l = q.lifts.copy()
                acc, leg, rej = q.run(40 * q.V, 90000 + int(e) * 500 + t, beta=beta, kappa=kappa, cstar=cstar, cmode=1, maxacc=1)
                ntr += leg; nacc += acc
                if acc:
                    q = fresh(st); q.allow[:] = 0; q.allow[v] = 1
            ph = nacc / max(ntr, 1)
            z.append((ph - p) / max(np.sqrt(p * (1 - p) / max(ntr, 1)), 1e-3))
        z = np.array(z)
        res.update(n_accept_edges=len(z), z_rms=float(np.sqrt(np.mean(z ** 2))), z_max=float(np.abs(z).max()), frac_over_4=float(np.mean(np.abs(z) > 4)), passed=bool(bad_comp == 0 and bad_rev == 0 and bad_track == 0 and np.mean(np.abs(z) > 4) <= 0.01))
        out[f'{cls}{name}'] = res
        print(res, flush=True)
    return out


def g_exact2():
    """GS3 amended (registered plan, section 9): the disks found by g_exact, at beta small enough that the chain crosses the N_f barriers of the disk within the run. Same code path; beta enters only as beta * dE."""
    out = {}
    for cls, name, ctr, R in (('ab', (5, 7), 27, 2.1), ('pen', 3, 60, 1.2)):
        tl, s0 = sampler(cls, name)
        s0.run(100 * s0.V, 99, maxacc=25); s0.refresh()
        X = s0.lifts @ s0.C.P.T; start = s0.lifts.copy()
        s0.allow[:] = (np.linalg.norm(X - X[ctr], axis=1) <= R).astype(np.int64)
        allow = s0.allow.copy(); comp = comp_masked(s0, cap=3000)
        states = list(comp.keys()); idx = np.nonzero(allow)[0]
        Es, Vs, Ms = [], [], []
        for st in states:
            l = start.copy(); l[idx] = np.array(st, np.int64)
            s2 = Sampler(cls, tl['L1'], tl['L2'], l, tl['perp'], tl['lifts'], atlas=s0.atlas)
            Es.append(s2.st[0]); Vs.append(s2.var); Ms.append(s2.moved)
        Es = np.array(Es); Vs = np.array(Vs); Ms = np.array(Ms)
        for beta, kappa, cmode, frac in ((0.1, 0.0, 0, 0), (0.2, 0.0, 0, 0), (-0.1, 0.0, 0, 0), (-0.2, 0.0, 0, 0), (0.1, 30.0, 1, 0.3), (0.1, 30.0, 2, 0.3)):
            Cs = Vs if cmode == 1 else Ms
            cstar = float(Cs.min() + frac * (Cs.max() - Cs.min()))
            W = np.exp(-beta * Es - (kappa * s0.V / 2 * (Cs - cstar) ** 2 if kappa > 0 else 0)); W /= W.sum()
            s = Sampler(cls, tl['L1'], tl['L2'], start, tl['perp'], tl['lifts'], atlas=s0.atlas); s.allow = allow.copy()
            hist = np.zeros(len(states)); n = 20000
            for i in range(n):
                s.run(20 * s.V, 5000 + i, beta=beta, kappa=kappa, cstar=cstar, cmode=cmode)
                hist[comp[tuple(tuple(int(x) for x in s.lifts[j]) for j in idx)]] += 1
            hist /= n
            ok = W * n >= 5
            chi = float((((hist - W) ** 2 / W)[ok]).sum() * n); dof = int(ok.sum() - 1)
            tv = 0.5 * float(np.abs(hist - W).sum()); tvn = 0.5 * float(np.sum(np.sqrt(2 * W * (1 - W) / (np.pi * n))))
            k = f'{cls}{name}_R{R}_b{beta}_k{kappa}_m{cmode}'
            out[k] = dict(n_states=len(states), Nf_range=[float(Es.min()), float(Es.max())], E_exact=float(W @ Es), E_chain=float(hist @ Es), TV=tv, TV_null=tvn, TV_ratio=tv / tvn, chi2_per_dof=chi / max(dof, 1), dof=dof, passed=bool(chi / max(dof, 1) <= 2 and tv <= 1.5 * tvn), tracked=bool(s.tracked_ok()))
            print(k, out[k], flush=True)
    return out


if __name__ == '__main__':
    which = sys.argv[1:] or ['atlas', 'heldout', 'fig4', 'bitwise', 'tracked', 'exact']
    f = HERE / 'sampler_checks.json'
    if f.exists():
        OUT = json.loads(f.read_text())
    for w in which:
        t0 = time.time()
        OUT[w] = dict(atlas=g_atlas, heldout=g_heldout, fig4=g_fig4, bitwise=g_bitwise, tracked=g_tracked, exact=g_exact, exact2=g_exact2, exact3=g_exact3, edges=g_edges)[w]()
        print(w, time.time() - t0, json.dumps(OUT[w], default=float)[:2500], flush=True)
        f.write_text(json.dumps(OUT, indent=1, default=float))

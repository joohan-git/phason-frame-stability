"""Perfect structures: fundamental path, critical modes, both signs of the secondary branch.
usage: run_perfect.py NAME [sub] [nmodes] [S11 S12 S21 S22]
One branch sign may run back down the fundamental path when the bifurcation is unfolded; then it is reseeded at a larger amplitude,
which reaches the disconnected branch."""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames', 'phason_paths', 'objectivity', 'disorder')]  # shared modules
import sys, json, time, math
import numpy as np
from structures import *
PSI_STOP = 0.35


def stopper(pc):
    def stop(r_, rows):
        return r_['max_chord'] > PSI_STOP or r_['p'] < 0.9 * pc or r_['p'] > 3 * pc
    return stop


def branches_for(model, cr, nsteps=250):
    mesh = model.m; phi = cr['mode']; pc = cr['p']; PR = participation(model, phi)
    out = {}
    for sign in (1, -1):
        got = None
        for fac in (1, 5, 25):
            eps = fac * 0.002 * math.sqrt(mesh.nz * max(PR, 1e-4))
            t1 = time.time()
            try:
                rows, _ = run_branch(model, cr['z'], pc * mesh.lamE, phi, sign, eps, nsteps, stopper(pc), ds_factor=0.5 / fac, k_eig=3)
            except Stop as ex:
                got = dict(error=str(ex), eps=eps); continue
            good = [r_ for r_ in rows if 'p' in r_]
            ps = np.array([r_['p'] for r_ in good])
            returned = bool(ps[-1] < 0.9 * pc)
            got = dict(rows=good, sec=time.time() - t1, end=rows[-1].get('status', 'maxsteps'), eps=eps, reseed_factor=fac, returned_to_fundamental=returned)
            if not returned:
                break
        out[str(sign)] = got
        if 'rows' in got:
            g = got['rows']; ps = np.array([r_['p'] for r_ in g]); neg = [r_.get('neg') for r_ in g]
            print('  branch', sign, 'fac', got['reseed_factor'], 'returned', got['returned_to_fundamental'], 'steps', len(g), got['end'], 'p0 %.5f pmax %.5f pend %.5f' % (ps[0], ps.max(), ps[-1]),
                  'psi_rms %.4f psi_max %.4f edev %.4f' % (g[-1]['rms_chord'], g[-1]['max_chord'], g[-1]['edev']), 'neg', sorted(set(neg)), flush=True)
        else:
            print('  branch', sign, 'FAILED', got, flush=True)
    return out


def main():
    name = sys.argv[1]; sub = int(sys.argv[2]) if len(sys.argv) > 2 else 2; nmodes = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    S = [[int(sys.argv[4]), int(sys.argv[5])], [int(sys.argv[6]), int(sys.argv[7])]] if len(sys.argv) > 7 else None
    tag = name + f'_s{sub}' + ('' if S is None else '_S%d%d%d%d' % tuple(np.ravel(S)))
    mesh = get_mesh(name, sub)
    if S is not None:
        mesh = supercell(mesh, S, tag)
    model = Model(mesh); t0 = time.time()
    lam, vec, st = model.linear_critical(macro=True, nev=max(6, nmodes + 3))
    print(tag, 'nodes', mesh.nn, 'lin', lam[:5], flush=True)
    rec = dict(tag=tag, name=name, sub=sub, S=S, nn=mesh.nn, nz=mesh.nz, lin=lam.tolist(), lin_macro=[macro_content(model, vec[:, i]) for i in range(min(4, vec.shape[1]))],
               lin_PR=[participation(model, vec[:, i]) for i in range(min(4, vec.shape[1]))], area0=mesh.A0, modes=[])
    if name.startswith('R'):
        crs = [find_critical(model, lam[0], frac=0.985, step_frac=0.002)]
    else:
        crs = channels_extrapolated(model, lam[:nmodes], vec[:, :nmodes])
    for imode in range(len(crs)):
        cr = crs[imode]
        mc = macro_content(model, cr['mode']); PR = participation(model, cr['mode'])
        info = dict(imode=imode, lin=float(lam[imode]), pc=cr['p'], shift=cr['p'] / float(lam[imode]) - 1, shift_per_lamE=(cr['p'] / float(lam[imode]) - 1) / mesh.lamE,
                    mu=cr['mu'], vals=cr['vals'].tolist(), kind=cr['kind'], zero_found=cr['zero_found'], neg=cr['neg'], macro=mc, PR=PR, overlap=cr.get('overlap'), trace=cr['trace'][-8:], sec_trace=time.time() - t0)
        print(' mode', imode, 'pc %.6f shift/lamE %.3f mu %.2e kind %s neg %s' % (cr['p'], info['shift_per_lamE'], cr['mu'], cr['kind'], cr['neg']), 'macro', {k: round(v, 5) for k, v in mc.items()}, 'PR %.4f' % PR, 'its', len(cr['trace']), flush=True)
        info['branches'] = branches_for(model, cr)
        rec['modes'].append(info)
        (HERE / 'out' / f'perfect_{tag}.json').write_text(json.dumps(rec, default=float))
    rec['seconds'] = time.time() - t0
    (HERE / 'out' / f'perfect_{tag}.json').write_text(json.dumps(rec, default=float))
    print('done', tag, time.time() - t0, flush=True)


if __name__ == '__main__':
    main()

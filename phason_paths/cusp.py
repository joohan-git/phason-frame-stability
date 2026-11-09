"""R2-C(a): cusp certification near s_QC from several phason directions (stress protocol, p_c).
Outputs cusp.json. Run: python cusp.py [--quick]"""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames', 'slope_survey')]  # shared modules
import json, math, sys, time
from fractions import Fraction
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import ab_frame as C

OUT = Path(__file__).with_name('cusp.json')
t = C.T


def lift_dir(x, M):
    return (C.P.T @ x + C.Q.T @ (M @ x)) / 2


def valid(L1, L2, vmax):
    m = C.minors(L1, L2); v = np.array(list(m.values()))
    if np.all(v <= 0):
        v = -v
    return np.all(v >= 0) and 0 < v.sum() <= vmax and np.count_nonzero(v) == 6


def generic(Mt, vmax, thetas=(0.0, math.pi / 8, math.pi / 16, 3 * math.pi / 16)):
    best = []
    for th in thetas:
        u = np.array([math.cos(th), math.sin(th)]); w = np.array([-u[1], u[0]])
        for lam in np.arange(4, 40, 0.125):
            L1 = np.rint(lam * lift_dir(u, Mt)).astype(int); L2 = np.rint(lam * lift_dir(w, Mt)).astype(int)
            if not valid(L1, L2, vmax):
                continue
            M = C.phason_matrix(L1, L2)
            best.append((np.linalg.norm(M - Mt), L1.tolist(), L2.tolist()))
    best.sort(key=lambda x: x[0])
    return best[0] if best else None


def diag_family(d, den=12):
    """L1=(a,b,0,-b), L2=(0,c,d,c): diagonal phason strain (E2-real + B1 components)."""
    r1 = Fraction(t * (1 - d) / (1 + d)).limit_denominator(den)
    r2 = Fraction(t * (1 + d) / (1 - d)).limit_denominator(den)
    return [r1.denominator, r1.numerator, 0, -r1.numerator], [0, r2.numerator, r2.denominator, r2.numerator]


def evaluate(name, L1, L2, sub=2, nphi=12, extra_k=True):
    t0 = time.time()
    tl = C.canonical(L1, L2)
    f = C.Frame(tl['lifts'], tl['L1'], tl['L2'], sub=sub)
    f.set_load('stress')
    phis, ext, tab = f.longwave(nphi=nphi)
    M = C.phason_matrix(tl['L1'], tl['L2']); a, b = C.harmonic_parts(M)
    row = dict(name=name, L1=tl['L1'].tolist(), L2=tl['L2'].tolist(), V=len(tl['lifts']), sub=sub,
               M=M.tolist(), a=[a.real, a.imag], b=[b.real, b.imag], delta=float(math.hypot(abs(a), abs(b))),
               pc_phi_rich=ext.tolist(), pc_phi_dq001=tab[0.001].tolist(), phis_deg=np.degrees(phis).tolist(),
               pc_min_sampled=float(ext.min()), harm=C.harmonics(phis, ext), prestress_resid=f.resid,
               square_fraction=(tl['minors']['13'] + tl['minors']['24']) / len(tl['lifts']))
    if extra_k:
        row['pc_gamma'] = float(f.lam((0, 0))); row['pc_k_half_half'] = float(f.lam((0.5, 0.5)))
    row['seconds'] = time.time() - t0
    print(json.dumps({k: row[k] for k in ['name', 'V', 'delta', 'pc_min_sampled', 'seconds']}), flush=True)
    return row


def main(quick=False):
    rows = []
    # (i) B1 line (AB p/q family), both sides of 1/sqrt2
    for p, q in ([(1, 2), (2, 3), (5, 7)] if quick else [(1, 2), (2, 3), (4, 5), (7, 10), (5, 7), (12, 17), (17, 24)]):
        rows.append(evaluate(f'B1_{p}/{q}', *C.ab_approx(p, q)))
    if not quick:
        # (ii) diagonal family: E2-real admixture
        seen = set()
        for d in (-0.2, -0.12, -0.06, 0.06, 0.12, 0.2):
            L1, L2 = diag_family(d)
            key = (tuple(L1), tuple(L2))
            if key in seen or not valid(np.array(L1), np.array(L2), 1500):
                continue
            seen.add(key)
            rows.append(evaluate(f'diag_{d:+.2f}', L1, L2))
        # (iii) generic rounding toward B2 and E2-imag (and E2-real) directions
        B = {'B2': np.array([[0, 1.], [1, 0]]), 'E2i': np.array([[0, -1.], [1, 0]]), 'E2r': np.eye(2)}
        for lab, Bm in B.items():
            for d in (0.06, 0.12, 0.2):
                g = generic(d * Bm, 900)
                if g is None:
                    continue
                rows.append(evaluate(f'{lab}_{d:.2f}', g[1], g[2]))
                rows[-1]['target'] = (d * Bm).tolist(); rows[-1]['target_error'] = g[0]
    # element convergence on the B1 line (phi = 0, 45 deg only)
    conv = []
    for p, q in [(1, 2), (2, 3), (5, 7)]:
        for sub in (2, 4, 8):
            tl = C.canonical(*C.ab_approx(p, q))
            f = C.Frame(tl['lifts'], tl['L1'], tl['L2'], sub=sub); f.set_load('stress')
            vals = []
            for phi in (0.0, math.pi / 4):
                u = np.array([math.cos(phi), math.sin(phi)])
                l2 = f.lam(0.002 * (f.cell @ u)); l1 = f.lam(0.001 * (f.cell @ u))
                vals.append((4 * l1 - l2) / 3)
            conv.append(dict(pq=f'{p}/{q}', sub=sub, pc_phi0=vals[0], pc_phi45=vals[1], pc_min=min(vals)))
            print(conv[-1], flush=True)
    OUT.write_text(json.dumps(dict(rows=rows, element_convergence=conv,
                                   units='p_c / (EA lambda_E) = p_c / (pi^2 EI / l^3); stress protocol; Richardson DQ->0',
                                   status='P (finite cells, sampled directions, linear FE)'), indent=1), encoding='utf-8')


if __name__ == '__main__':
    main('--quick' in sys.argv)

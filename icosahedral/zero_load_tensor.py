"""r2: loaded acoustic tensor of each canonical configuration at fixed multipliers, plus the root of
the isotropic transverse part. Usage: python zero_load_tensor.py FAMILY[,FAMILY...]  where FAMILY = c:P:Q:NOFF or
t:P:Q:PZ:QZ:NOFF. Writes runs_zero_load/<tag>_s<seed>.json (Lambda per lambda, reproduction check)."""
import os, sys, json, time
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
from icosa_fast import *
from load_axial_runs import load
from scipy.optimize import brentq
FIX = [0.0, 0.5, 0.85, 0.90, 0.92]
R, _ = load()
# exact quadrature on the sphere (degree <= 8): Gauss-Legendre x uniform phi
_z, _w = np.polynomial.legendre.leggauss(8)
_ph = np.arange(18) * 2 * np.pi / 18
NODES = np.array([[np.sqrt(1 - z * z) * np.cos(p), np.sqrt(1 - z * z) * np.sin(p), z] for z in _z for p in _ph])
WTS = np.array([w / 18 / 2 for w in _w for p in _ph])


def tau0(Lam):
    n = NODES
    A = np.einsum('jlab,ej,el->eab', Lam, n, n)
    tr = np.trace(A, axis1=1, axis2=2) - np.einsum('ea,eab,eb->e', n, A, n)
    return float(WTS @ (tr / 2))


for fam in sys.argv[1].split(','):
    p = fam.split(':')
    if p[0] == 'c':
        P_, Q_, N_ = map(int, p[1:]); L = cubic_periods(P_, Q_); tag = f'cubic_{P_}-{Q_}'
    else:
        a, b, c, d, N_ = map(int, p[1:]); L = tetra_periods((a, b), (c, d)); tag = f'tetra_{a}-{b}x{c}-{d}'
    for s in range(1, N_ + 1):
        fn = os.path.join(here, 'runs_zero_load', f'{tag}_s{s}.json')
        if os.path.exists(fn):
            continue
        t0 = time.time()
        tl = canonical(L, seed=s)
        f = Frame3D.from_tiling(tl['lifts'], L, sub=2); prep_load(f); f.set_load('stress')
        st = R[f'{tag}_s{s}.json']
        repro = float(np.max(np.abs(np.array(st['C6_over_EI']) - f.C6 / EI)))
        H = Homog(f)
        out = dict(tag=tag, seed=s, V=st['V'], M=st['M'], a=[float(np.linalg.norm(c)) for c in f.cell], repro_C6=repro,
                   lamW=st['lamW'], lw_min=st['lw_min'], tensors={})
        for l in FIX:
            if l < st['lamW']:
                out['tensors'][str(l)] = H.tensor(l)[0].tolist()
        top = st['lamW'] * (1 - 1e-6)
        lo = min(0.90, 0.98 * top)
        g = lambda l: tau0(H.tensor(l)[0])
        root = None
        if g(lo) > 0 and g(top) < 0:
            root = float(brentq(g, lo, top, xtol=2e-5))
        elif g(lo) <= 0:
            lo2 = 0.5
            if g(lo2) > 0:
                root = float(brentq(g, lo2, lo, xtol=2e-5))
        out['lam_iso_root'] = root
        if root is not None:
            out['tensors']['root'] = H.tensor(root)[0].tolist()
        out['nfact'] = len(H.cache); out['t'] = time.time() - t0
        json.dump(out, open(fn, 'w'))
        print(tag, s, 'repro', repro, 'root', root, 'nfact', out['nfact'], 't', round(out['t']), flush=True)

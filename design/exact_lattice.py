"""Independent check of the reference lattices: exact beam-column member stiffness (stability functions, no FE
discretization, none of the Frame code). Hydrostatic stress, all struts carry the same force by symmetry
(N = 2 p A / (total length), virial, exact). For each Bloch k the smallest multiplier with K(lam,k) singular is found by
inertia count + bisection. Output exact_lattice.json (P). Geometry reused from design_core.ref_def only."""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames', 'penrose', 'phason_paths', 'robustness', 'heldout_test', 'isotropic_loss')]  # shared modules
import os
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='-1')
import json, math, sys
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
sys.path.insert(0, str(Path(__file__).resolve().parent))
from design_core import ref_def

EA = 1.0; r = 0.005; EI = r ** 2 / 4; L = 1.0


def member(N, c, s):
    """12-term 6x6 global stiffness of a unit compressed member (N > 0 compression), exact."""
    phi = math.sqrt(N * L ** 2 / EI)
    D = 2 - 2 * math.cos(phi) - phi * math.sin(phi)
    S = phi * (math.sin(phi) - phi * math.cos(phi)) / D      # s
    SC = phi * (phi - math.sin(phi)) / D                     # s c
    a = 2 * (S + SC) - phi ** 2                              # transverse stiffness coefficient
    b = S + SC
    k = np.zeros((6, 6))
    k[0, 0] = k[3, 3] = EA / L; k[0, 3] = k[3, 0] = -EA / L
    k[1, 1] = k[4, 4] = a * EI / L ** 3; k[1, 4] = k[4, 1] = -a * EI / L ** 3
    k[1, 2] = k[2, 1] = k[1, 5] = k[5, 1] = b * EI / L ** 2
    k[4, 2] = k[2, 4] = k[4, 5] = k[5, 4] = -b * EI / L ** 2
    k[2, 2] = k[5, 5] = S * EI / L; k[2, 5] = k[5, 2] = SC * EI / L
    T = np.zeros((6, 6))
    for o in (0, 3):
        T[o, o] = c; T[o, o + 1] = s; T[o + 1, o] = -s; T[o + 1, o + 1] = c; T[o + 2, o + 2] = 1
    return T.T @ k @ T


def K_bloch(E, nn, N, kv):
    K = np.zeros((3 * nn, 3 * nn), complex)
    for i, j, v, w1, w2 in E:
        c, s = v[0], v[1]
        ke = member(N, c, s)
        ph = np.exp(2j * np.pi * (w1 * kv[0] + w2 * kv[1]))
        Tn = np.zeros((6, 3 * nn), complex)
        Tn[:3, 3 * i:3 * i + 3] = np.eye(3)
        Tn[3:, 3 * j:3 * j + 3] += ph * np.eye(3)
        K += Tn.conj().T @ ke @ Tn
    return 0.5 * (K + K.conj().T)


def nneg(E, nn, N, kv):
    w = np.linalg.eigvalsh(K_bloch(E, nn, N, kv))
    # scale: one zero-translation at Gamma is expected; count clearly negative only
    return int(np.sum(w < -1e-10 * EI)), w


def first_lam(E, nn, fac, kv, lam_hi):
    unit = math.pi ** 2 * EI / L ** 3   # p in units pi^2 EI / L^3
    grid = np.linspace(0.02, lam_hi, 140)
    n0 = nneg(E, nn, fac * grid[0] * unit, kv)[0]
    for a, b in zip(grid[:-1], grid[1:]):
        # stay away from the pole of the stability functions at phi = 2 pi: inertia still counts correctly
        if nneg(E, nn, fac * b * unit, kv)[0] > n0:
            lo, hi = a, b
            for _ in range(45):
                m = .5 * (lo + hi)
                if nneg(E, nn, fac * m * unit, kv)[0] > n0:
                    hi = m
                else:
                    lo = m
            return .5 * (lo + hi)
    return math.inf


out = {}
for name, hi in (('square', 1.5), ('rhomb45', 1.5), ('triangular', 3.5), ('kagome', 1.2), ('hexagonal', 0.6), ('rhombille', 1.5)):
    nodes, a1, a2, E = ref_def(name)
    A = abs(a1[0] * a2[1] - a1[1] * a2[0])
    fac = 2 * A / len(E)   # N = fac * p, virial (all struts equal by symmetry; unit length)
    n = 12; best = (math.inf, None)
    for i in range(n):
        for j in range(n):
            if (i, j) == (0, 0):
                continue
            kv = (i / n, j / n)
            v = first_lam(E, len(nodes), fac, kv, min(hi, best[0] + 1e-6) if best[0] < math.inf else hi)
            if v < best[0]:
                best = (v, kv)
    # long-wave limit from a small k
    lw = min(first_lam(E, len(nodes), fac, (1e-3 * math.cos(t), 1e-3 * math.sin(t)), hi) for t in np.arange(6) * np.pi / 6)
    out[name] = dict(fac_N_per_p=fac, lam_finite_k=best[0], k=best[1], lam_near_longwave=lw)
    print(name, out[name], flush=True)
    Path(__file__).with_name('exact_lattice.json').write_text(json.dumps(dict(label='P', r=r, method='exact member stiffness', rows=out), indent=1))
# rotation-only closed forms for the triangular grid: 6(sin f - f cos f) + Sigma (f - sin f) = 0, Sigma = sum over neighbours of e^{ik.a}
tri = {}
for lab, Sig in (('K', -3.0), ('M', -2.0)):
    g = lambda f: 6 * (math.sin(f) - f * math.cos(f)) + Sig * (f - math.sin(f))
    f = brentq(g, 3.0, 5.5)
    tri[lab] = dict(phi=f, S=f ** 2 * math.sqrt(3) / 12, lam=f ** 2 * math.sqrt(3) / math.pi ** 2)
print(tri)
d = json.loads(Path(__file__).with_name('exact_lattice.json').read_text()); d['triangular_rotation_closed_form'] = tri
Path(__file__).with_name('exact_lattice.json').write_text(json.dumps(d, indent=1))

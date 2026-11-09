"""part on the unmodified nonlinear solver: bow shapes, moduli from the macro Schur block, path measures."""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames', 'design', 'nonlinear', 'phason_paths')]  # shared modules
import os
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[v] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import math, sys
from pathlib import Path
import numpy as np
HERE_defects = Path(__file__).resolve().parent; M_ = HERE_defects.parent
sys.path.insert(0, str(HERE_defects))
import run_imperfect as RI          # brings structures, corotational_frame (nonlinear, unmodified)
from run_imperfect import *
import design_core as X10
from defects_util import *


def n_edges_vertices(mesh):
    nE = mesh.ne // mesh.sub
    return nE, mesh.nn - nE * (mesh.sub - 1)


def bow_shape(mesh, signs=None, shape='half'):
    """Unit-amplitude dU: interior nodes of edge e on w(x) = s_e sin(pi x) [half] or sin(2 pi x) [full], along the left normal of the edge.
    Interior node ids follow corotational_frame.from_edges: V + e (sub-1) + h."""
    sub = mesh.sub; nE, V = n_edges_vertices(mesh)
    s = np.ones(nE) if signs is None else np.asarray(signs, float)
    D = mesh.d0.reshape(nE, sub, 2).sum(1)
    nv = np.c_[-D[:, 1], D[:, 0]] / np.linalg.norm(D, axis=1)[:, None]
    dU = np.zeros((mesh.nn, 2))
    for e in range(nE):
        for h in range(sub - 1):
            x = (h + 1) / sub
            w = math.sin(math.pi * x) if shape == 'half' else math.sin(2 * math.pi * x)
            dU[V + e * (sub - 1) + h] = s[e] * w * nv[e]
    return dU


def edge_signs(mesh, key):
    nE, _ = n_edges_vertices(mesh)
    return rng_of(key).choice([-1.0, 1.0], size=nE)


def schur_moduli(mesh):
    """K, G at p = 0 of the reference geometry. Energy density 1/2 e^T S e with e = (e11, e22, e12); Mandel uses sqrt2 e12."""
    st = Model(mesh).linear_state()
    D = np.diag([1, 1, 1 / math.sqrt(2)])
    m = X10.moduli(D @ st['S'] @ D)
    return m['K'], m['Gmean'], m


def secant(rows, mesh, p):
    """K from the area change on the path at pressure p (units of lamE)."""
    P = np.array([r['p'] for r in rows]); A = np.array([r['A_rel'] for r in rows])
    i = np.where(P >= p)[0]
    if not len(i) or i[0] == 0:
        return None
    j = i[0]
    a = A[j - 1] + (p - P[j - 1]) / (P[j] - P[j - 1]) * (A[j] - A[j - 1])
    return float(p * mesh.lamE / (-a))


def run_path(mesh, dU, w0, pc_ref, psistop=0.6):
    """run_one with zero macro part, plus K, G, secants and the limit data."""
    E0 = np.zeros((2, 2))
    res = RI.run_one(mesh, dU, E0, w0, pc_ref, psistop=psistop, keep=True)
    imp = mesh.with_imperfection(w0 * dU, None)
    K, G, m = schur_moduli(imp)
    res.update(K0=K, G0=G)
    rows = res.pop('rows')
    res['Ksec25'] = secant(rows, imp, 0.25 * res['p_lin']); res['Ksec50'] = secant(rows, imp, 0.5 * res['p_lin'])
    res['chord_bow_rms'] = RI.chord_rms(mesh, w0 * dU, E0)
    P = np.array([r['p'] for r in rows]);
    res['p_cap'] = float(P.max()) if len(P) else None
    res['psi_cap'] = float(rows[int(P.argmax())]['max_chord']) if len(P) else None
    res['neg_end'] = rows[-1]['neg'] if rows else None
    res['tau_end'] = rows[-1]['max_tau'] if rows else None
    res['path'] = [[round(r['p'], 5), round(r['max_chord'], 4), round(r['max_edge'], 4), round(r['A_rel'], 6), r['neg']] for r in rows][::4]
    res['bow_arc_excess'] = float(np.sum(imp.L) / np.sum(mesh.L) - 1)
    return res

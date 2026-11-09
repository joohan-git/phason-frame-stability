"""I_h representation theory (numerical character decomposition, exact up to rounding).
Writes icosa_representation.json."""
import json, itertools, math, sys
import numpy as np
sys.path.insert(0, __file__.rsplit('\\', 1)[0] if '\\' in __file__ else '.')
from icosa_core import *

grp = icosahedral_group()
out = dict(order=len(grp))
chP = [np.trace(R) for g, R, Rp in grp]
chQ = [np.trace(Rp) for g, R, Rp in grp]
ch6 = [np.trace(g) for g, R, Rp in grp]
out['V_par'] = decompose(chP, grp)
out['V_perp'] = decompose(chQ, grp)
out['Z6'] = decompose(ch6, grp)
# phason strain Hom(V_par, V_perp): M -> Rp M R^T
out['phason'] = decompose([np.trace(Rp) * np.trace(R) for g, R, Rp in grp], grp)


def sym2_mat(R):
    """Representation of R on Sym^2(R^3) (orthonormal basis, 6-dim)."""
    B = []
    for i in range(3):
        b = np.zeros((3, 3)); b[i, i] = 1; B.append(b)
    for i, j in ((1, 2), (0, 2), (0, 1)):
        b = np.zeros((3, 3)); b[i, j] = b[j, i] = 1 / math.sqrt(2); B.append(b)
    M = np.zeros((6, 6))
    for a, Ba in enumerate(B):
        RB = R @ Ba @ R.T
        for b, Bb in enumerate(B):
            M[b, a] = np.sum(Bb * RB)
    return M


def sym2_char(M):
    return 0.5 * (np.trace(M) ** 2 + np.trace(M @ M))


S2 = [sym2_mat(R) for g, R, Rp in grp]
out['Sym2_strain'] = decompose([np.trace(m) for m in S2], grp)
out['elastic_C_ijkl'] = decompose([sym2_char(m) for m in S2], grp)
# fully symmetric rank-4 (longitudinal acoustic functional n_i n_j n_k n_l C_ijkl) = l=0+2+4
def symk_char(R, k):
    # character of Sym^k(R^3) via eigenvalues
    ev = np.linalg.eigvals(R)
    tot = 0
    for c in itertools.combinations_with_replacement(range(3), k):
        tot += np.prod([ev[i] for i in c])
    return float(np.real(tot))
out['Sym4_fully'] = decompose([symk_char(R, 4) for g, R, Rp in grp], grp)
# SO(3) spherical harmonics restricted to I_h (character of rank-l on rotation angle)
def harm_char(R, l):
    d = np.linalg.det(R); r = R * d
    c = (np.trace(r) - 1) / 2; c = max(-1, min(1, c)); th = math.acos(c)
    chi = (2 * l + 1) if th < 1e-9 else math.sin((l + .5) * th) / math.sin(th / 2)
    return chi * (1 if (d > 0 or l % 2 == 0) else -1)
out['Y_l'] = {l: decompose([harm_char(R, l) for g, R, Rp in grp], grp) for l in range(0, 13)}
out['first_anisotropic_invariant_l'] = [l for l in range(1, 13) if out['Y_l'][l].get('Ag', 0) > 0][:3]
# equivariant couplings phason -> elastic
ph, el = out['phason'], out['elastic_C_ijkl']
out['n_linear_couplings_phason_to_C'] = sum(ph[k] * el.get(k, 0) for k in ph)
out['couplings_by_irrep'] = {k: ph[k] * el.get(k, 0) for k in ph}
# T_h-invariant subspace of phason (cubic approximant symmetry): elements fixing the cubic axes
Th = [(g, R, Rp) for g, R, Rp in grp if np.allclose(np.abs(R), np.round(np.abs(R)), atol=1e-9)]
out['T_h_order'] = len(Th)
projs = phason_projectors(grp)
avgTh = sum(np.kron(Rp, R) for g, R, Rp in Th) / len(Th)
out['T_h_invariant_dim_phason'] = int(round(np.trace(avgTh)))
out['T_h_invariant_dim_in_G'] = int(round(np.trace(avgTh @ projs['Gg'])))
out['T_h_invariant_dim_in_H'] = int(round(np.trace(avgTh @ projs['Hg'])))
# D_2h (tetragonal approximant symmetry here: diagonal R only)
D2 = [(g, R, Rp) for g, R, Rp in grp if np.allclose(R, np.diag(np.diag(R)), atol=1e-9)]
avgD = sum(np.kron(Rp, R) for g, R, Rp in D2) / len(D2)
out['D_2h_order'] = len(D2)
out['D_2h_invariant_dim_in_G'] = int(round(np.trace(avgD @ projs['Gg'])))
out['D_2h_invariant_dim_in_H'] = int(round(np.trace(avgD @ projs['Hg'])))
# transverse-trace lemma check: does the l=2 part of a generic C survive t n t n contraction in 3D?
rng = np.random.default_rng(0)
D = rng.normal(size=(3, 3)); D = D + D.T; D -= np.trace(D) / 3 * np.eye(3)
d = np.eye(3)
C2 = (np.einsum('ij,kl->ijkl', d, D) + np.einsum('ij,kl->ijkl', D, d))      # l=2 part, type 1
C2b = (np.einsum('ik,jl->ijkl', d, D) + np.einsum('jl,ik->ijkl', d, D) + np.einsum('il,jk->ijkl', d, D) + np.einsum('jk,il->ijkl', d, D))
vals = []
for _ in range(5):
    n = rng.normal(size=3); n /= np.linalg.norm(n)
    Pt = np.eye(3) - np.outer(n, n)
    vals.append([float(np.einsum('ijkl,ik,j,l', C2, Pt, n, n)), float(np.einsum('ijkl,ik,j,l', C2b, Pt, n, n))])
out['transverse_trace_of_l2_parts_3D(nonzero=>no 3D lemma)'] = vals
# same in 2D for contrast
D2d = np.array([[1., 0.3], [0.3, -1.]]); d2 = np.eye(2)
C2b2 = (np.einsum('ik,jl->ijkl', d2, D2d) + np.einsum('jl,ik->ijkl', d2, D2d) + np.einsum('il,jk->ijkl', d2, D2d) + np.einsum('jk,il->ijkl', d2, D2d))
C22 = np.einsum('ij,kl->ijkl', d2, D2d) + np.einsum('ij,kl->ijkl', D2d, d2)
v2 = []
for th in (0.3, 1.1):
    n = np.array([math.cos(th), math.sin(th)]); t = np.array([-n[1], n[0]])
    v2.append([float(np.einsum('ijkl,i,j,k,l', C22, t, n, t, n)), float(np.einsum('ijkl,i,j,k,l', C2b2, t, n, t, n))])
out['transverse_of_m2_parts_2D(zero=>2D lemma)'] = v2
# phason G/H content of the approximant sequences
appr = {}
for p, q in ((1, 1), (2, 1), (3, 2), (5, 3), (8, 5), (13, 8)):
    L = cubic_periods(p, q); M = phason_matrix(L).ravel()
    appr[f'{p}/{q}'] = dict(V=check_minor_signs(L)[1], valid=check_minor_signs(L)[0],
                            G=float(np.linalg.norm(projs['Gg'] @ M)), H=float(np.linalg.norm(projs['Hg'] @ M)),
                            M_diag=float(phason_matrix(L)[0, 0]), M_offdiag_max=float(np.max(np.abs(phason_matrix(L) - np.diag(np.diag(phason_matrix(L)))))),
                            a_cell=float((P @ L)[0, 0]), primitive_half=bool((p + q) % 2 == 0 and (p - q) % 2 == 0))
out['cubic_approximants'] = appr
tet = {}
for xy, z in (((2, 1), (3, 2)), ((3, 2), (2, 1)), ((3, 2), (5, 3)), ((5, 3), (3, 2))):
    L = tetra_periods(xy, z); M = phason_matrix(L).ravel()
    ok, nv = check_minor_signs(L)
    tet[f'{xy[0]}/{xy[1]}x{z[0]}/{z[1]}'] = dict(V=nv, valid=ok, G=float(np.linalg.norm(projs['Gg'] @ M)),
                                              H=float(np.linalg.norm(projs['Hg'] @ M)), M=phason_matrix(L).round(6).tolist())
out['tetragonal_approximants'] = tet
here = __file__.rsplit('\\', 1)[0] if '\\' in __file__ else '.'
json.dump(out, open(here + '\\icosa_representation.json', 'w'), indent=1, default=str)
print(json.dumps(out, indent=1, default=str))

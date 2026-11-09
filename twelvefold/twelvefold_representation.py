"""D12 representation of the uniform phason strain
S = a I + b J + c Z + d X  (internal = S physical), I = id, J = rot90, Z = diag(1,-1), X = [[0,1],[1,0]].
Generator g: physical rotation 30 deg, internal rotation 150 deg; S -> R(150) S R(30)^-1.
Mirror s: physical and internal complex conjugation (y -> -y in both), S -> Zs S Zs.
Also: phonon strain harmonics; fourth-rank tensor harmonics; Z[zeta12] lattice facts used by twelvefold_core.py.
Writes twelvefold_representation.json."""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames', 'phason_paths')]  # shared modules
import json
from pathlib import Path
import sympy as sp

a, b, c, d = sp.symbols('a b c d', real=True)
I2 = sp.eye(2); J = sp.Matrix([[0, -1], [1, 0]]); Z = sp.Matrix([[1, 0], [0, -1]]); X = sp.Matrix([[0, 1], [1, 0]])
R = lambda deg: sp.Matrix([[sp.cos(sp.rad(deg)), -sp.sin(sp.rad(deg))], [sp.sin(sp.rad(deg)), sp.cos(sp.rad(deg))]])
S = a * I2 + b * J + c * Z + d * X


def coords(M):
    # M = a I + b J + c Z + d X  ->  (a,b,c,d)
    return [sp.simplify((M[0, 0] + M[1, 1]) / 2), sp.simplify((M[1, 0] - M[0, 1]) / 2),
            sp.simplify((M[0, 0] - M[1, 1]) / 2), sp.simplify((M[1, 0] + M[0, 1]) / 2)]


gS = R(150) * S * R(30).inv()
G = sp.Matrix([[sp.diff(e, v) for v in (a, b, c, d)] for e in coords(gS)])
G = sp.simplify(G)
mir = Z * S * Z
Mm = sp.Matrix([[sp.diff(e, v) for v in (a, b, c, d)] for e in coords(mir)])
out = {}
out['g_on_abcd'] = str(G)
blk_ab = G[:2, :2]; blk_cd = G[2:, 2:]
out['ab_block_equals_R120'] = bool(sp.simplify(blk_ab - R(120)) == sp.zeros(2))
out['cd_block_equals_R180'] = bool(sp.simplify(blk_cd - R(180)) == sp.zeros(2))
out['offdiag_zero'] = bool(sp.simplify(G[:2, 2:]) == sp.zeros(2) and sp.simplify(G[2:, :2]) == sp.zeros(2))
out['mirror_on_abcd'] = str(Mm)
# harmonic index m: block = R(30 m) mod 360
out['m_ab'] = 4  # 120 = 4*30
out['m_cd'] = 6  # 180 = 6*30
out['check_m'] = dict(ab=bool(sp.simplify(blk_ab - R(4 * 30)) == sp.zeros(2)), cd=bool(sp.simplify(blk_cd - R(6 * 30)) == sp.zeros(2)))
out['D12_irreps'] = '(a,b): E4 (2D, g->R120, mirror diag(1,-1));  (c,d): g -> -1 on both, mirror: c even, d odd -> B1 (+) B2'
# same bookkeeping for AB (45 deg phys, 135 deg int) for the swap statement
gA = R(135) * S * R(45).inv()
GA = sp.simplify(sp.Matrix([[sp.diff(e, v) for v in (a, b, c, d)] for e in coords(gA)]))
out['AB_ab_block_R90_m2'] = bool(sp.simplify(GA[:2, :2] - R(90)) == sp.zeros(2))
out['AB_cd_block_R180_m4'] = bool(sp.simplify(GA[2:, 2:] - R(180)) == sp.zeros(2))
# phonon strain: symmetric e -> R e R^T under physical rotation only
e11, e12, e22 = sp.symbols('e11 e12 e22', real=True)
E = sp.Matrix([[e11, e12], [e12, e22]])
gE = R(30) * E * R(30).T
tr = sp.simplify(gE.trace() - E.trace())
dev = sp.Matrix([sp.simplify((gE[0, 0] - gE[1, 1]) / 2), gE[0, 1]])
# express dev rotation: should be R(60) = m=2
Dm = sp.simplify(sp.Matrix([[sp.diff(dev[i].subs({e11: sp.Symbol('p') + sp.Symbol('q'), e22: sp.Symbol('q') - sp.Symbol('p')}), v) for v in (sp.Symbol('p'), e12)] for i in range(2)]))
out['phonon_trace_invariant_m0'] = bool(tr == 0)
out['phonon_deviator_block'] = str(Dm)
out['phonon_deviator_is_R60_m2'] = bool(sp.simplify(Dm - R(60)) == sp.zeros(2))
out['linear_phonon_phason_R_allowed'] = 'no: phonon m in {0,2}, phason m in {4,6}; D12 irreps distinct (m mod 12, m ~ -m) -> R = 0 by Schur'
out['tangent_stiffness_4th_rank_m'] = [0, 2, 4]
out['first_order_h4_source'] = '(a,b) only (m=4); (c,d) m=6 not in {0,2,4}'
out['all_orders_cd'] = 'products of m=6 give m in {0,6,12=0} mod 12: pure (c,d) never carries m=4 in a covariant (hull-averaged) family'
# lattice facts
z = sp.exp(sp.I * sp.pi / 6)
v = lambda k: sp.Matrix([sp.cos(sp.pi * k / 6), sp.sin(sp.pi * k / 6), sp.cos(5 * sp.pi * k / 6), sp.sin(5 * sp.pi * k / 6)])
Gram = sp.Matrix(4, 4, lambda i, j: sp.simplify(v(i).dot(v(j))))
out['gram_v0_v3'] = str(Gram); out['det_gram'] = int(Gram.det()); out['det_B'] = str(sp.sqrt(Gram.det()))
out['v4_eq_v2_minus_v0'] = bool(sp.simplify(v(4) - v(2) + v(0)) == sp.zeros(4, 1))
out['v5_eq_v3_minus_v1'] = bool(sp.simplify(v(5) - v(3) + v(1)) == sp.zeros(4, 1))
out['lattice'] = 'A2 (+) A2: {v0,v2,v4} and {v1,v3,v5} orthogonal root systems (Gram block structure)'
Path(__file__).with_name('twelvefold_representation.json').write_text(json.dumps(dict(status='C', **out), indent=1), encoding='utf-8')
print(json.dumps(out, indent=1))

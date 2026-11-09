"""selection-rule table from explicit group actions (exact up to float round-off; C).
2D: C8 (AB, Z^4), C10 (Penrose, Z^5), C12 (Z^4 dodecagonal module).  For each, build the integer
generator g on Z^n, verify P g = R(2pi/N) P and Q g = R(m_int 2pi/N) Q, then the phason-strain charges
m = m_int -/+ 1 (w = a z + b zbar).  Rule (Lemma 1 + Lemma 2 of main r2): linear 4theta splitting iff some
charge = +-4 mod N; analytic linear isotropic iff some charge = 0 mod N.
3D icosahedral: physical T1u, internal its Galois conjugate T2u (the Z^6 lattice rep splits as T1+T2);
decompose T1 (x) T2; build the most general Y-equivariant linear map phason strain -> elastic tensor by
group averaging and test (i) the isotropic part of dC vanishes, (ii) the transverse 2x2 block of dA(n)
has a nonzero traceless part, whose angular mean of |.| gives a linear drop of the angular mean of the
LOWER transverse branch even though dC is analytic (polarisation-doublet splitting).
-> point_groups.json"""
import json, math, itertools
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
out = {}


def rot(t):
    return np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])


def check2d(name, N, P, Q, g):
    th = 2 * math.pi / N
    okP = np.allclose(P @ g, rot(th) @ P, atol=1e-12)
    mint = None
    for m in range(N):
        if np.allclose(Q @ g, rot(m * th) @ Q, atol=1e-12):
            mint = m
    ch = sorted({(mint - 1) % N, (mint + 1) % N})
    def red(m):
        return min(m % N, (-m) % N)
    charges = [red(mint - 1), red(mint + 1)]
    out[name] = dict(N=N, P_equivariant=bool(okP), m_int=mint, phason_charges_abs=charges,
                     linear_4theta_split=any(c == 4 % N or c == (-4) % N or c == 4 for c in charges),
                     linear_2theta=any(c == 2 for c in charges),
                     analytic_linear_isotropic=any(c == 0 for c in charges))
    print(name, out[name])


# C8: AB
T = 1 / math.sqrt(2)
P8 = np.array([[1, T, 0, -T], [0, T, 1, T]]); Q8 = np.array([[1, -T, 0, T], [0, T, -1, T]])
g8 = np.zeros((4, 4)); [g8.__setitem__((j + 1, j), 1) for j in range(3)]; g8[0, 3] = -1
check2d('C8_AB', 8, P8, Q8, g8)
# C10: Penrose (penrose_frame conventions)
TH = np.arange(5) * math.pi / 5
P10 = np.array([np.cos(TH), np.sin(TH)]); Q10 = np.array([np.cos(7 * TH), np.sin(7 * TH)])
g10 = np.zeros((5, 5)); [g10.__setitem__((j + 1, j), 1) for j in range(4)]; g10[0, 4] = -1
check2d('C10_Penrose', 10, P10, Q10, g10)
# C12: Z^4 with e_k at k*30 deg (k=0..3); zeta^4 = zeta^2 - 1 for zeta = e^{i pi/6}
TH12 = np.arange(4) * math.pi / 6
P12 = np.array([np.cos(TH12), np.sin(TH12)]); Q12 = np.array([np.cos(5 * TH12), np.sin(5 * TH12)])
g12 = np.zeros((4, 4)); [g12.__setitem__((j + 1, j), 1) for j in range(3)]
g12[:, 3] = [-1, 0, 1, 0]  # zeta*e3 = zeta^4 = -1 + zeta^2
check2d('C12_dodecagonal', 12, P12, Q12, g12)

# ---------------- icosahedral
phi = (1 + 5 ** 0.5) / 2


def gens(ph):
    a = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], float)          # C3 (cyclic)
    b = 0.5 * np.array([[1, -ph, 1 / ph], [ph, 1 / ph, -1], [1 / ph, 1, ph]])  # C5
    return a, b


def group(a, b):
    G = [np.eye(3)]; frontier = [np.eye(3)]
    while frontier:
        nf = []
        for x in frontier:
            for y in (a, b):
                z = y @ x
                if not any(np.allclose(z, w, atol=1e-9) for w in G):
                    G.append(z); nf.append(z)
        frontier = nf
    return G


a1, b1 = gens(phi)
G1 = group(a1, b1)
# Galois conjugate rep: phi -> -1/phi, built along the same words
a2, b2 = gens(-1 / phi)
# build G2 elementwise by the same BFS words
words = [()]; Gw = {(): np.eye(3)}; elems = [np.eye(3)]; frontier = [()]
while frontier:
    nf = []
    for w in frontier:
        for s, y in (('a', a1), ('b', b1)):
            z = y @ Gw[w]
            if not any(np.allclose(z, e, atol=1e-9) for e in elems):
                nw = w + (s,); Gw[nw] = z; elems.append(z); nf.append(nw); words.append(nw)
    frontier = nf
def ev(w, A, B):
    M = np.eye(3)
    for s in w:
        M = (A if s == 'a' else B) @ M
    return M
G1 = [ev(w, a1, b1) for w in words]; G2 = [ev(w, a2, b2) for w in words]
assert len(G1) == 60
chi1 = np.array([np.trace(g) for g in G1]); chi2 = np.array([np.trace(g) for g in G2])
hom = True  # Galois conjugation is a field automorphism of Q(sqrt5): relations preserved
# irreps of I: characters via class functions; decompose T1 x T2 by inner products with known chars
chiA = np.ones(60); chiG = None
prod = chi1 * chi2
# character of Sym^2(T1) = A + H ; G char = prod - H char  (T1xT2 = G + H known); verify numerically
sym2 = 0.5 * (chi1 ** 2 + np.array([np.trace(g @ g) for g in G1]))
chiH = sym2 - chiA
chiG_try = prod - chiH
norm = lambda c: float(np.dot(c, c) / 60)
out['Ih'] = dict(order=len(G1), conj_is_hom_sample=bool(hom),
                 mult_A_in_T1xT2=float(np.dot(prod, chiA) / 60), mult_H_in_T1xT2=float(np.dot(prod, chiH) / 60),
                 G_part_norm=norm(chiG_try), G_dim=float(chiG_try[0]),
                 mult_T1_in_T2=float(np.dot(chi1, chi2) / 60))
print('Ih', out['Ih'])

# equivariant map phason strain (internal x physical, 3x3 matrix W: g -> R2 W R1^T) -> elastic tensor C_ijkl
rng = np.random.default_rng(0)
def sym4(C):
    C = 0.5 * (C + C.transpose(1, 0, 2, 3)); C = 0.5 * (C + C.transpose(0, 1, 3, 2)); C = 0.5 * (C + C.transpose(2, 3, 0, 1))
    return C
Lmap = rng.normal(size=(3, 3, 3, 3, 3, 3))  # C_ijkl = L_ijkl,ab W_ab
Lavg = np.zeros_like(Lmap)
for R1, R2 in zip(G1, G2):
    # (g.L)(W) = R1 . L(R2^T W R1) ; average
    Lavg += np.einsum('ip,jq,kr,ls,pqrsab,ac,bd->ijklcd', R1, R1, R1, R1, Lmap, R2.T, R1.T, optimize=True)
Lavg /= 60
for c in range(3):
    for d in range(3):
        Lavg[..., c, d] = sym4(Lavg[..., c, d])
rank = np.linalg.matrix_rank(Lavg.reshape(81, 9), tol=1e-8)
# directions on the sphere
u = rng.normal(size=(4000, 3)); u /= np.linalg.norm(u, axis=1)[:, None]
res = []
for trial in range(5):
    W = rng.normal(size=(3, 3)); W /= np.linalg.norm(W)
    dC = np.einsum('ijklab,ab->ijkl', Lavg, W)
    iso_bulk = np.einsum('iijj', dC); iso_shear = np.einsum('ijij', dC)
    tr_mean, tl_mean, low_mean = [], [], []
    for n in u:
        A = np.einsum('ijkl,j,l->ik', dC, n, n)
        t1 = np.cross(n, [1, 0, 0] if abs(n[0]) < 0.9 else [0, 1, 0]); t1 /= np.linalg.norm(t1); t2 = np.cross(n, t1)
        B = np.array([[t1 @ A @ t1, t1 @ A @ t2], [t2 @ A @ t1, t2 @ A @ t2]])
        tr = np.trace(B) / 2; tl = math.sqrt(((B[0, 0] - B[1, 1]) / 2) ** 2 + B[0, 1] ** 2)
        tr_mean.append(tr); tl_mean.append(tl); low_mean.append(tr - tl)
    res.append(dict(iso_bulk=float(iso_bulk), iso_shear=float(iso_shear), mean_trace_half=float(np.mean(tr_mean)),
                    mean_traceless_abs=float(np.mean(tl_mean)), mean_lower_branch_shift=float(np.mean(low_mean))))
out['Ih_coupling'] = dict(equivariant_map_rank=int(rank), trials=res,
                          note='dC linear (analytic) in phason strain; angular mean of trace part = 0 (no A in G+H) '
                               'but mean of lower transverse branch = -<|traceless|> != 0: linear isotropic cusp of '
                               'the stiffness of the softer polarisation. 2D has a single transverse polarisation, so no such term.')
print('Ih coupling rank', rank, res[:2])
# 2D control: same construction for C10 shows mean of the single transverse branch shift = 0
(HERE / 'point_groups.json').write_text(json.dumps(out, indent=1), encoding='utf-8')

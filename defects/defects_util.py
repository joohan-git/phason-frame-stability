"""without solver imports: edge removal with connectivity guards (registered before the runs)."""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames', 'design', 'nonlinear', 'phason_paths')]  # shared modules
import hashlib, math
import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csg


def rng_of(key):
    return np.random.default_rng(int.from_bytes(hashlib.sha256(str(key).encode()).digest()[:8], 'little'))


def degrees(V, edges):
    d = np.zeros(V, int)
    np.add.at(d, edges[:, 0], 1)
    np.add.at(d, edges[:, 1], 1)
    return d


def connected(V, edges):
    A = sp.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(V, V))
    return csg.connected_components(A, directed=False, return_labels=False) == 1


def wrap_generators(V, edges):
    """Cycle wrap vectors of the quotient graph. Convention: X_i + vec = X_j + w1 a1 + w2 a2, so crossing i -> j adds (w1, w2)
    to the cell of the image. Returns the integer vectors of non-tree edges, or None if not connected."""
    adj = [[] for _ in range(V)]
    for e, (i, j, k, w1, w2) in enumerate(edges):
        adj[i].append((j, w1, w2, e)); adj[j].append((i, -w1, -w2, e))
    pot = np.zeros((V, 2), np.int64); seen = np.zeros(V, bool); seen[0] = True; tree = np.zeros(len(edges), bool)
    st = [0]
    while st:
        a = st.pop()
        for b, w1, w2, e in adj[a]:
            if not seen[b]:
                seen[b] = True; tree[e] = True; pot[b] = pot[a] + (w1, w2); st.append(b)
    if not seen.all():
        return None
    nt = ~tree
    return pot[edges[nt, 0]] + edges[nt, 3:5] - pot[edges[nt, 1]]


def spans(V, edges):
    """True when the frame is connected on the torus and its cycles generate the full period lattice Z^2 (gcd of 2x2 minors = 1)."""
    W = wrap_generators(V, edges)
    if W is None or len(W) < 2:
        return False
    W = W[np.any(W != 0, axis=1)]
    if len(W) < 2:
        return False
    W = np.unique(W, axis=0)
    g = 0
    for a in range(len(W)):
        for b in range(a + 1, len(W)):
            g = math.gcd(g, int(W[a, 0] * W[b, 1] - W[a, 1] * W[b, 0]))
            if g == 1:
                return True
    return g == 1


def removal_order(V, edges, key, kmax):
    """Nested removal: walk a random permutation; skip an edge that would isolate a vertex, disconnect the torus or lose a period.
    Returns (order, nskip); order = removed edge indices, kmax long. Independent of r and sub."""
    edges = np.asarray(edges, np.int64); E = len(edges)
    perm = rng_of(key).permutation(E)
    keep = np.ones(E, bool); deg = degrees(V, edges); order = []; nskip = 0
    for e in perm:
        if len(order) >= kmax:
            break
        i, j = edges[e, 0], edges[e, 1]
        if deg[i] <= 1 or deg[j] <= 1:
            nskip += 1; continue
        keep[e] = False
        if not spans(V, edges[keep]):
            keep[e] = True; nskip += 1; continue
        deg[i] -= 1; deg[j] -= 1
        order.append(int(e))
    return order, nskip


def level_k(level, E):
    return 1 if level == 'k1' else max(1, int(round(float(level) * E)))


def boot_ci(x, nb=4000, seed=0):
    x = np.asarray(x, float)
    if len(x) < 2:
        return [float(x.mean())] * 2
    rng = np.random.default_rng(seed)
    v = x[rng.integers(0, len(x), (nb, len(x)))].mean(1)
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

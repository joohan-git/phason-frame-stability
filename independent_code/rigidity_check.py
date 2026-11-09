"""Small, reproducible counterexample checks; no bulk-limit claims.

Run: python rigidity_check.py
Writes ab_rigidity_results.json next to this script. All springs are unstressed,
unit stiffness, with rest lengths equal to the geometry, and unit node masses.
Only in-plane translations are counted. NumPy + SciPy are required.
"""
from __future__ import annotations

import itertools
import json
import math
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from scipy.linalg import svdvals
from scipy.spatial import ConvexHull

SEED = 261108
UNIT = np.eye(4, dtype=int)


def projections(t=1 / np.sqrt(2)):
    s = 1 / (2 * t)
    physical = np.array([[1, t, 0, -t], [0, t, 1, t]])
    internal = np.array([[1, -s, 0, s], [0, s, -1, s]])
    cube = np.array(list(itertools.product([-0.5, 0.5], repeat=4)))
    corners = cube @ internal.T
    hull = ConvexHull(corners)
    return physical, internal, hull.equations


def inside_window(n, internal, equations, shift):
    y = n @ internal.T - shift
    return np.all(y @ equations[:, :2].T + equations[:, 2] < -1e-10, axis=1)


def candidates(bound):
    return np.array(list(itertools.product(range(-bound, bound + 1), repeat=4)), dtype=int)


def open_patch(radius, shift=(0.117, 0.271)):
    physical, internal, equations = projections()
    bound = math.ceil(radius / 2 + 2)
    n = candidates(bound)
    x = n @ physical.T
    keep = (np.linalg.norm(x, axis=1) < radius) & inside_window(n, internal, equations, np.array(shift))
    n = n[keep]
    lookup = {tuple(v): i for i, v in enumerate(n)}
    edges = []
    for i, v in enumerate(n):
        for step in UNIT:
            j = lookup.get(tuple(v + step))
            if j is not None:
                edges.append((i, j, 0, 0))
    return n, n @ physical.T, edges


def approximant(p, q, shift=(0.117, 0.271)):
    """Rationalized cut slope t=p/q; fixed unit physical edge vectors.

    The two period lifts generate the internal-projection kernel for the odd q
    used here. The physical projection is kept irrational, so approximants use
    exactly the same four unit edge vectors as AB. This is a strained cut with
    nonorthogonal physical/internal coordinate maps. A generic offset avoids
    window-boundary ambiguity.
    """
    _, internal, equations = projections(p / q)
    physical, _, _ = projections()
    lifts = np.array([[q, p, 0, -p], [0, p, q, p]])
    cell = lifts @ physical.T
    length = cell[0, 0]
    assert np.max(np.abs(lifts @ internal.T)) < 1e-12
    n = candidates(max(p, math.ceil(q / 2)) + 3)
    x = n @ physical.T
    keep = np.all((x >= -length / 2 - 1e-9) & (x < length / 2 - 1e-9), axis=1)
    keep &= inside_window(n, internal, equations, np.array(shift))
    n = n[keep]
    lookup = {tuple(v): i for i, v in enumerate(n)}
    edges = []
    for i, v in enumerate(n):
        for step in UNIT:
            raw = v + step
            xraw = physical @ raw
            wrap = np.floor((xraw + length / 2 + 1e-9) / length).astype(int)
            representative = raw - wrap @ lifts
            j = lookup.get(tuple(representative))
            if j is not None:
                edges.append((i, j, int(wrap[0]), int(wrap[1])))
    return n, n @ physical.T, edges, cell


def faces(x, edges):
    """Planar half-edge walk, used only for finite open patches."""
    adj = [set() for _ in x]
    for i, j, _, _ in edges:
        adj[i].add(j)
        adj[j].add(i)
    cyclic = [sorted(a, key=lambda j: math.atan2(*(x[j] - x[i])[::-1])) for i, a in enumerate(adj)]
    visited = set()
    bounded = []
    for i, a in enumerate(adj):
        for j in a:
            if (i, j) in visited:
                continue
            polygon = []
            u, v = i, j
            while (u, v) not in visited:
                visited.add((u, v))
                polygon.append(u)
                ordered = cyclic[v]
                w = ordered[(ordered.index(u) - 1) % len(ordered)]
                u, v = v, w
            z = x[polygon]
            area = np.sum(z[:, 0] * np.roll(z[:, 1], -1) - z[:, 1] * np.roll(z[:, 0], -1)) / 2
            if area > 1e-8:
                bounded.append(polygon)
    return bounded


def tile_flips(n_initial, edges_initial, target, seed=SEED):
    """Local three-rhombus flips, fixed boundary and N,E.

    This is a finite phason-disordered control, NOT a certified equilibrium
    random-tiling draw, and does not preserve the full degree histogram.
    """
    rng = np.random.default_rng(seed)
    n = n_initial.copy()
    adj = [set() for _ in n]
    for i, j, _, _ in edges_initial:
        adj[i].add(j)
        adj[j].add(i)
    lookup = {tuple(v): i for i, v in enumerate(n)}
    physical, _, _ = projections()
    accepted = 0
    attempts = 0
    while accepted < target and attempts < target * 100:
        attempts += 1
        i = int(rng.integers(len(n)))
        if len(adj[i]) != 3:
            continue
        neighbors = sorted(adj[i])
        delta = (n[neighbors] - n[i]) @ physical.T
        angles = np.sort(np.arctan2(delta[:, 1], delta[:, 0]))
        if np.max(np.diff(np.r_[angles, angles[0] + 2 * np.pi])) >= np.pi - 1e-8:
            continue
        replacement = n[neighbors].sum(axis=0) - 2 * n[i]
        if tuple(replacement) in lookup:
            continue
        opposite = []
        legal = True
        for j, k in itertools.combinations(neighbors, 2):
            o = lookup.get(tuple(n[j] + n[k] - n[i]))
            if o is None or o not in adj[j] or o not in adj[k]:
                legal = False
                break
            opposite.append(o)
        if not legal:
            continue
        # The new point must have precisely the three intended unit-step neighbours.
        expected = set(opposite)
        actual = set()
        for step in np.r_[UNIT, -UNIT]:
            j = lookup.get(tuple(replacement + step))
            if j is not None and j != i:
                actual.add(j)
        if actual != expected:
            continue
        del lookup[tuple(n[i])]
        for j in neighbors:
            adj[j].remove(i)
        n[i] = replacement
        lookup[tuple(replacement)] = i
        adj[i] = expected
        for j in opposite:
            adj[j].add(i)
        accepted += 1
    edges = [(i, j, 0, 0) for i, a in enumerate(adj) for j in a if i < j]
    assert len(edges) == len(edges_initial)
    return n, n @ physical.T, edges, accepted, attempts


def metrics(name, n, x, edges, cell=None, wavevector=(0.0, 0.0), jitter=0.0):
    rng = np.random.default_rng(SEED)
    x = x + jitter * rng.normal(size=x.shape)
    v, e = len(x), len(edges)
    r = np.zeros((e, 2 * v), dtype=complex if cell is not None else float)
    lengths = []
    degree = np.zeros(v, dtype=int)
    for row, (i, j, a, b) in enumerate(edges):
        wrap = np.array([a, b])
        displacement = x[j] - x[i] + (wrap @ cell if cell is not None else 0)
        length = np.linalg.norm(displacement)
        direction = displacement / length
        phase = np.exp(2j * np.pi * np.dot(wavevector, wrap)) if cell is not None else 1
        r[row, 2 * i:2 * i + 2] -= direction
        r[row, 2 * j:2 * j + 2] += phase * direction
        lengths.append(length)
        degree[i] += 1
        degree[j] += 1
    singular = svdvals(r, check_finite=False)
    ranks = {str(tol): int(np.sum(singular > tol * singular[0])) for tol in [1e-8, 1e-9, 1e-10]}
    assert len(set(ranks.values())) == 1, ranks
    rank = ranks['1e-09']
    zero_modes, stress = 2 * v - rank, e - rank
    assert zero_modes - stress == 2 * v - e
    result = dict(name=name, vertices=v, edges=e, index=2 * v - e, mean_coordination=2 * e / v,
                  rank=rank, zero_modes_including_rigid=zero_modes, self_stress=stress,
                  zero_modes_per_vertex=zero_modes / v, self_stress_per_vertex=stress / v,
                  index_per_vertex=(2 * v - e) / v, ranks_by_relative_tolerance=ranks,
                  degree_histogram={str(k): int(np.sum(degree == k)) for k in np.unique(degree)},
                  edge_length_range=[float(min(lengths)), float(max(lengths))],
                  smallest_positive_singular=float(singular[rank - 1]),
                  last_singular_values=singular[-12:].tolist(),
                  normalized_D_moments=[float(np.sum(singular ** (2 * k)) / (2 * v)) for k in [1, 2, 3]],
                  boundary='periodic fixed cell' if cell is not None else 'free',
                  physical_period_vectors=cell.tolist() if cell is not None else None,
                  wavevector_fractional=list(wavevector) if cell is not None else None,
                  jitter_amplitude=jitter)
    if wavevector == (0.0, 0.0):
        translations = np.tile(np.eye(2), (v, 1))
        result['translation_residual'] = float(np.max(np.abs(r @ translations)))
        assert result['translation_residual'] < 1e-12
    if cell is None:
        rotation = x[:, [1, 0]].copy()
        rotation[:, 0] *= -1
        result['rotation_residual'] = float(np.max(np.abs(r @ rotation.ravel())))
        assert result['rotation_residual'] < 1e-12
        f = faces(x, edges)
        result['bounded_face_histogram'] = {str(k): sum(len(p) == k for p in f) for k in sorted(set(map(len, f)))}
        assert all(len(p) == 4 for p in f)
        assert v - e + len(f) == 1  # connected disk, no holes
    print(json.dumps({k: result[k] for k in ['name', 'vertices', 'edges', 'rank', 'zero_modes_including_rigid', 'self_stress']}), flush=True)
    return result


def main():
    rows = []
    for radius in [4, 6, 8]:
        n, x, edges = open_patch(radius)
        rows.append(metrics(f'AB_free_R{radius}', n, x, edges))
        rows.append(metrics(f'AB_same_graph_jitter_R{radius}', n, x, edges, jitter=0.015))
        if radius == 8:
            nf, xf, ef, accepted, attempts = tile_flips(n, edges, 4 * len(n))
            row = metrics('AB_local_flip_control_R8', nf, xf, ef)
            row['accepted_flips'] = accepted
            row['attempted_flips'] = attempts
            row['changed_vertex_lifts'] = int(np.sum(np.any(nf != n, axis=1)))
            physical, internal, eq = projections()
            row['vertices_outside_original_window'] = int(np.sum(~inside_window(nf, internal, eq, np.array([0.117, 0.271]))))
            rows.append(row)
    for shift in [(0.381, 0.193), (-0.227, 0.409)]:
        n, x, edges = open_patch(6, shift)
        row = metrics(f'AB_free_R6_shift_{shift}', n, x, edges)
        row['window_shift'] = list(shift)
        rows.append(row)
    for p, q in [(2, 3), (5, 7)]:
        n, x, edges, cell = approximant(p, q)
        assert len(edges) == 2 * len(n)
        for k in [(0.0, 0.0), (0.173, 0.319)]:
            rows.append(metrics(f'AB_approximant_t{p}_{q}_k{k}', n, x, edges, cell, k))
            rows.append(metrics(f'AB_approximant_jitter_t{p}_{q}_k{k}', n, x, edges, cell, k, jitter=0.015))
    output = dict(seed=SEED,
                  interpretation='Finite unstressed central-force checks, not certified bulk asymptotics or equilibrium random tilings.',
                  cut_and_project={'physical_columns': projections()[0].T.tolist(),
                                   'internal_columns': projections()[1].T.tolist(),
                                   'window': 'projection of [-1/2,1/2]^4',
                                   'generic_window_shift': [0.117, 0.271],
                                   'connectivity': 'integer lifts differ by a signed coordinate unit vector'},
                  rows=rows)
    Path(__file__).with_name('ab_rigidity_results.json').write_text(json.dumps(output, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()

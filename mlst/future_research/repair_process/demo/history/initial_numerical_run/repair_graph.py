"""Fixed synthetic grid and a proper Gaussian precision, independent of materials."""
from collections import deque
import numpy as np


def grid_fixture(size=13, hole_low=3, hole_high=10, ridge=0.3, edge_weight=1.0):
    """Return a declared relation graph and synthetic observations.

    Values at missing positions are NaN, not zero. The mean is a synthetic
    spatial function, not a biological field or a measured material response.
    All horizontal/vertical edges have the same declared weight.
    """
    axis = np.linspace(-1.0, 1.0, size)
    xx, yy = np.meshgrid(axis, axis)
    mean = (0.5 * np.sin(np.pi * xx) + 0.25 * yy).ravel()
    known_field = mean + (0.35 * np.cos(np.pi * yy / 2) + 0.2 * xx).ravel()
    observed = np.ones((size, size), dtype=bool)
    observed[hole_low:hole_high, hole_low:hole_high] = False
    observed = observed.ravel()
    values = np.where(observed, known_field, np.nan)
    n = size * size
    q = np.eye(n, dtype=float) * ridge
    adjacency = [[] for _ in range(n)]
    edges = []
    for row in range(size):
        for col in range(size):
            i = row * size + col
            for rr, cc in ((row + 1, col), (row, col + 1)):
                if rr >= size or cc >= size:
                    continue
                j = rr * size + cc
                q[i, i] += edge_weight
                q[j, j] += edge_weight
                q[i, j] -= edge_weight
                q[j, i] -= edge_weight
                adjacency[i].append(j)
                adjacency[j].append(i)
                edges.append([i, j, edge_weight])
    return {
        'size': size, 'coordinates': np.column_stack([xx.ravel(), yy.ravel()]),
        'mean': mean, 'observed': observed, 'values': values, 'precision': q,
        'adjacency': adjacency, 'edges': edges,
    }


def free_problem(fixture):
    """Condition one fixed Gaussian joint law on the genuine observed nodes."""
    observed = fixture['observed']
    free = np.flatnonzero(~observed)
    obs = np.flatnonzero(observed)
    q = fixture['precision']
    a = q[np.ix_(free, free)]
    # For centered free state z = y_H - mean_H, E(z|O)=A^{-1}h.
    h = -q[np.ix_(free, obs)] @ (fixture['values'][obs] - fixture['mean'][obs])
    return {'free': free, 'A': a, 'h': h,
            'mean_centered': np.linalg.solve(a, h),
            'covariance': np.linalg.solve(a, np.eye(len(free)))}


def activation_depth(fixture):
    """Mask/topology-only growth front; it never depends on generated values."""
    n = len(fixture['observed'])
    depth = np.full(n, -1, dtype=int)
    queue = deque(np.flatnonzero(fixture['observed']).tolist())
    depth[fixture['observed']] = 0
    while queue:
        i = queue.popleft()
        for j in fixture['adjacency'][i]:
            if depth[j] < 0:
                depth[j] = depth[i] + 1
                queue.append(j)
    if np.any(depth < 0):
        raise ValueError('The declared demonstration graph has unreachable nodes')
    return depth

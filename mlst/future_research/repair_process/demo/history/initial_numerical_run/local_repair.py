"""Local mean relaxation and a correct Gaussian Gibbs comparison.

This is not a new algorithm: the first mode is Gauss-Seidel and the second
uses exact Gaussian full conditionals. Their difference is instructional.
"""
import numpy as np


def coordinate_mean(z, a, h, i):
    return (h[i] - a[i] @ z + a[i, i] * z[i]) / a[i, i]


def mean_relaxation(fixture, problem, depth, sweeps, snapshot_steps):
    """Activate a front then update only activated free states locally.

    Dormant free states have an internal prior-mean placeholder but are
    displayed as missing. The front is a visualization/computation schedule,
    not extra evidence. Before convergence its state is not a posterior mean.
    """
    z = np.zeros(len(problem['free']))
    unknown_depth = depth[problem['free']]
    snapshots = []
    max_errors = []
    energies = []
    a, h = problem['A'], problem['h']
    for sweep in range(sweeps + 1):
        if sweep > 0:
            for i in range(len(z)):
                if unknown_depth[i] <= sweep:
                    z[i] = coordinate_mean(z, a, h, i)
        all_values = fixture['values'].copy()
        all_values[problem['free']] = fixture['mean'][problem['free']] + z
        active = fixture['observed'] | (depth <= sweep)
        if sweep in snapshot_steps:
            snapshots.append({'sweep': sweep, 'values': all_values.copy(),
                              'active': active.copy(),
                              'activated_missing_nodes': int(np.sum(active & ~fixture['observed']))})
        max_errors.append(float(np.max(np.abs(z - problem['mean_centered']))))
        energies.append(float(0.5 * z @ a @ z - h @ z))
        # Real observations are never altered. Internal generated placeholders
        # never change fixture['observed'] or count as new measurements.
        if not np.array_equal(all_values[fixture['observed']], fixture['values'][fixture['observed']]):
            raise AssertionError('An observed value changed')
    return z, snapshots, max_errors, energies


def gibbs_samples(problem, initial, seed, burn_in, retained, thinning):
    """Fixed systematic scan, no state-adaptive selection or precision updates."""
    rng = np.random.default_rng(seed)
    z = initial.copy()
    a, h = problem['A'], problem['h']
    sigma = 1.0 / np.sqrt(np.diag(a))
    out = np.empty((retained, len(z)))
    steps = burn_in + retained * thinning
    saved = 0
    for sweep in range(steps):
        noise = rng.standard_normal(len(z))
        for i in range(len(z)):
            z[i] = coordinate_mean(z, a, h, i) + sigma[i] * noise[i]
        if sweep >= burn_in and (sweep - burn_in + 1) % thinning == 0:
            out[saved] = z
            saved += 1
    assert saved == retained
    return out


def collapse_of_deterministic_ensemble(problem, members, sweeps, seed):
    """Independent initial states all converge to the same conditional mean."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((members, len(problem['free'])))
    a, h = problem['A'], problem['h']
    for _ in range(sweeps):
        for i in range(z.shape[1]):
            z[:, i] = (h[i] - z @ a[i] + a[i, i] * z[:, i]) / a[i, i]
    return z

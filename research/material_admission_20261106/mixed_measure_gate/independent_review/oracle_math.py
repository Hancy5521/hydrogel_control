"""Independent finite-h mixed-measure oracle, not a material model or SUT wrapper."""
from fractions import Fraction
import math

import numpy as np
from scipy.stats import multivariate_normal

FIELDS = ('A', 'G', 'Y')
PRIOR = (.1, .2, .3, .4)
P0 = (.1, .3, .6, .8)
RHO = (-.6, -.2, .25, .65)


def branch_density(h, observed, spike):
    """Observed submatrix PDF; spike AY law is NOT slab G=0 conditional."""
    rho = RHO[h]
    if spike:
        order = ('A', 'Y')
        mu = np.asarray([.2 * h, -.1 * h])
        covariance = np.asarray([[1., rho], [rho, 1.]])
    else:
        order = FIELDS
        mu = np.asarray([.2 * h, .3 - .1 * h, -.1 * h])
        covariance = np.asarray([[rho ** abs(i - j) for j in range(3)] for i in range(3)])
    indices = [j for j, name in enumerate(order) if name in observed]
    if not indices:
        return 1.
    values = np.asarray([observed[order[j]] for j in indices])
    return float(multivariate_normal.pdf(values, mean=mu[indices], cov=covariance[np.ix_(indices, indices)]))


def row_likelihood(h, observed):
    if not observed:
        return 1.
    if 'G' not in observed:
        return P0[h] * branch_density(h, observed, True) + (1 - P0[h]) * branch_density(h, observed, False)
    if observed['G'] == 0:
        return P0[h] * branch_density(h, observed, True)
    return (1 - P0[h]) * branch_density(h, observed, False)


def cumulative_log_likelihood(observed_rows):
    return [math.fsum(math.log(row_likelihood(h, values)) for values in observed_rows.values()) for h in range(4)]


def posterior(observed_rows):
    loglik = cumulative_log_likelihood(observed_rows)
    logjoint = [math.log(w) + ell for w, ell in zip(PRIOR, loglik)]
    peak = max(logjoint)
    normalizer = math.fsum(math.exp(x - peak) for x in logjoint)
    weights = [math.exp(x - peak) / normalizer for x in logjoint]
    return {'cumulative_log_likelihood': loglik, 'weights': weights,
            'fresh_row_unconditional_Y_mean': math.fsum(w * (-.1 * h) for h, w in enumerate(weights))}


def naive_all_continuous_row_likelihood(h, observed):
    """Intentional wrong model used only to distinguish delta-mass from density."""
    return branch_density(h, observed, False)


def prior_as_exact_fractions():
    return [Fraction(1, 10), Fraction(2, 10), Fraction(3, 10), Fraction(4, 10)]

"""Independent synthetic audit draft. NO SUT execution before root approval.

Reference calculations below are authored independently of the producer. The
source-pinned adapters and audit entry point are completed after its API freezes.
This script never needs or reads the two downloaded material data tables.
"""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def gaussian_cdf(z):
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def gaussian_pdf(z):
    return math.exp(-z * z / 2.0) / math.sqrt(2.0 * math.pi)


def direct_dense_condition(mean, covariance, observed, values):
    """Independent one-shot Schur complement, exact noiseless observations.

    Whole conditioned covariance is PSD, not generally SPD. Only a fresh
    unobserved bundle must have positive definite predictive covariance.
    """
    import numpy as np
    m = np.asarray(mean, dtype=float)
    k = np.asarray(covariance, dtype=float)
    idx = np.asarray(observed, dtype=int)
    y = np.asarray(values, dtype=float)
    if len(idx) == 0:
        return m.copy(), k.copy()
    if len(set(map(int, idx))) != len(idx):
        raise ValueError('Independent oracle rejects repeated observation indices')
    # Deliberately use one dense inverse, unlike the producer's sequential path.
    inverse = np.linalg.inv(k[np.ix_(idx, idx)])
    cross = k[:, idx]
    updated_mean = m + cross @ inverse @ (y - m[idx])
    updated_covariance = k - cross @ inverse @ cross.T
    updated_covariance = (updated_covariance + updated_covariance.T) / 2.0
    # The exact conditional law is a point mass on the observed coordinates.
    updated_mean[idx] = y
    updated_covariance[idx, :] = 0.0
    updated_covariance[:, idx] = 0.0
    return updated_mean, updated_covariance


def direct_weighted_trace_drop(covariance, observed, targets, weights):
    import numpy as np
    n = len(covariance)
    _, after = direct_dense_condition(np.zeros(n), covariance, observed,
                                      np.zeros(len(observed)))
    before = np.asarray(covariance, dtype=float)
    idx = np.asarray(targets, dtype=int)
    w = np.asarray(weights, dtype=float)
    return float(np.dot(w, np.diag(before)[idx] - np.diag(after)[idx]))


def exact_expected_max_affine_normal(intercepts, slopes):
    """Integrate the finite upper envelope of a_i+b_i*Z, Z~N(0,1).

    Every active interval is found from pairwise linear inequalities. No GH
    rule, producer envelope implementation, random samples or model are used.
    """
    lines = list(dict.fromkeys((float(a), float(b))
                              for a, b in zip(intercepts, slopes)))
    if not lines or not all(math.isfinite(a) and math.isfinite(b) for a, b in lines):
        raise ValueError('Finite aligned nonempty lines required')
    total = 0.0
    for a, b in lines:
        low, high = -math.inf, math.inf
        feasible = True
        for aa, bb in lines:
            gap = b - bb
            if gap == 0.0:
                if a < aa:
                    feasible = False
                    break
            elif gap > 0.0:
                low = max(low, (aa - a) / gap)
            else:
                high = min(high, (aa - a) / gap)
        if not feasible or low >= high:
            continue
        mass = gaussian_cdf(high) - gaussian_cdf(low)
        first_moment = gaussian_pdf(low) - gaussian_pdf(high)
        total += a * mass + b * first_moment
    return total


def independent_tensor_gh(intercepts, coefficients, order=5):
    """Independent rebuild of the stated finite GH approximation.

    Matching this value tests implementation, not accuracy of the quadrature as
    an approximation to the continuous Gaussian expectation.
    """
    import itertools
    import numpy as np
    a = np.asarray(intercepts, dtype=float)
    b = np.asarray(coefficients, dtype=float)
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    total = 0.0
    for indices in itertools.product(range(order), repeat=b.shape[1]):
        z = math.sqrt(2.0) * nodes[list(indices)]
        weight = float(np.prod(weights[list(indices)])) / math.pi ** (b.shape[1] / 2.0)
        total += weight * float(np.max(a + b @ z))
    return total


def independent_two_dimensional_expectation(intercepts, coefficients):
    """Nested independent integration for a two-dimensional affine maximum.

    The inner one-dimensional upper envelope is integrated analytically; the
    outer integral is adaptive scipy.quad. Only an approximation diagnostic.
    """
    import numpy as np
    from scipy.integrate import quad
    a = np.asarray(intercepts, dtype=float)
    b = np.asarray(coefficients, dtype=float)
    if b.shape != (len(a), 2):
        raise ValueError('Two coefficients per decision line required')
    value, estimated_error = quad(
        lambda z: gaussian_pdf(z) * exact_expected_max_affine_normal(a + b[:, 0] * z,
                                                                    b[:, 1]),
        -math.inf, math.inf, epsabs=2e-9, epsrel=2e-9, limit=150)
    return value, estimated_error


if __name__ == '__main__':
    raise SystemExit('DRAFT: wait for producer API, source pin and root-approved plan; no SUT executed.')

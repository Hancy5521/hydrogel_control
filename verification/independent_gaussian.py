"""Independent frozen-parameter formulas; never import production model code.

No fitting, truth labels, material cache, or scores.  Inputs are public saved
parameters and an explicitly synthetic six-anchor JSON object.
"""
import numpy as np
from scipy.special import ndtri


def mean_and_kernel(model, left, right=None, report=False):
    """Physical log-response units; report noise only for same-variable square K."""
    ref = model['reference']
    norm = ref['normalization']
    left = np.asarray(left, dtype=float)
    right = left if right is None else np.asarray(right, dtype=float)
    u = (left - norm['t_center']) / norm['t_scale']
    v = (right - norm['t_center']) / norm['t_scale']
    level, slope, smooth, length, discrepancy = ref['hyper_values']
    distance = np.sqrt(3.) * np.abs(u[:, None] - v[None, :]) / length
    kernel = norm['y_scale'] ** 2 * (
        level ** 2 + slope ** 2 * np.outer(u, v)
        + smooth ** 2 * (1. + distance) * np.exp(-distance))
    if report:
        if not np.array_equal(left, right):
            raise ValueError('Independent oracle report identities must match')
        kernel += norm['y_scale'] ** 2 * discrepancy ** 2 * np.eye(len(left))
    beta = ref['beta_standardized']
    mean = norm['y_center'] + norm['y_scale'] * (beta[0] + beta[1] * u)
    return mean, kernel


def curvature_rows(q):
    """Construct differences of adjacent slopes, then average fixed windows.

    Deliberately uses a first-difference matrix composition instead of the
    production implementation's three explicit second-difference coefficients.
    """
    q = np.asarray(q, dtype=float)
    h = np.diff(q)
    slopes = np.zeros((len(q) - 1, len(q)))
    for j, width in enumerate(h):
        slopes[j, j:j + 2] = [-1. / width, 1. / width]
    curvature = np.diff(slopes, axis=0) * (2. / (q[2:] - q[:-2]))[:, None]
    windows = []
    for fraction in (1. / 3., 2. / 3.):
        local = (q[1:-1] - (q[0] + fraction * np.ptp(q))) / (np.ptp(q) / 3.)
        weight = np.where(np.abs(local) < 1., (1. - local * local) ** 3, 0.)
        weight *= (q[2:] - q[:-2]) / 2.
        windows.append(weight / weight.sum())
    return np.asarray(windows) @ curvature


def joint(model, coordinates):
    """Return m0, K0, C, corrected m, corrected K without production helpers."""
    q = np.asarray(model['q'], dtype=float)
    m0, k0 = mean_and_kernel(model, coordinates, report=True)
    _, kq = mean_and_kernel(model, q)
    _, cross = mean_and_kernel(model, coordinates, q)
    f = curvature_rows(q)
    feature_covariance = f @ kq @ f.T
    feature_covariance = (feature_covariance + feature_covariance.T) / 2.
    eigenvalues, eigenvectors = np.linalg.eigh(feature_covariance)
    if np.min(eigenvalues) <= 0.:
        raise ValueError('Reference features are not positive definite')
    symmetric_inverse_root = eigenvectors @ np.diag(eigenvalues ** -.5) @ eigenvectors.T
    c = cross @ f.T @ symmetric_inverse_root
    theta = np.asarray(model['theta'], dtype=float)
    a = theta[:2]
    first, off_diagonal, last = np.exp(theta[2]), theta[3], np.exp(theta[4])
    s = np.array([[first * first, first * off_diagonal],
                  [first * off_diagonal, off_diagonal ** 2 + last ** 2]])
    mean = m0 + c @ a
    covariance = k0 - c @ c.T + c @ s @ c.T
    covariance = (covariance + covariance.T) / 2.
    return m0, k0, c, mean, covariance


def condition(mean, covariance, observed, values, target):
    """Explicit Gaussian block elimination; no production conditioning imports."""
    observed, target = np.asarray(observed, dtype=int), np.asarray(target, dtype=int)
    oo = covariance[np.ix_(observed, observed)]
    ho = covariance[np.ix_(target, observed)]
    mu = mean[target] + ho @ np.linalg.solve(oo, np.asarray(values) - mean[observed])
    variance = covariance[np.ix_(target, target)] - ho @ np.linalg.solve(oo, ho.T)
    return mu, (variance + variance.T) / 2.


def complete(model, input_payload):
    """Return oracle conditional law, including point masses for stored anchors."""
    anchor = np.asarray(input_payload['observed_gamma'], dtype=float)
    eta = np.asarray(input_payload['observed_eta'], dtype=float)
    target = np.asarray(input_payload['target_gamma'], dtype=float)
    known = {float(x): j for j, x in enumerate(anchor)}
    mapping = [known.get(float(x)) for x in target]
    missing = [j for j, v in enumerate(mapping) if v is None]
    coords = np.log(np.r_[anchor, target[missing]])
    m0, k0, c, m, k = joint(model, coords)
    mu, var = condition(m, k, np.arange(6), np.log(eta), np.arange(6, len(coords)))
    whole_mean = np.empty(len(target))
    whole_covariance = np.zeros((len(target), len(target)))
    whole_mean[missing] = mu
    whole_covariance[np.ix_(missing, missing)] = var
    for j, v in enumerate(mapping):
        if v is not None:
            whole_mean[j] = np.log(eta[v])
    quantiles = np.exp(whole_mean[None, :] + ndtri(np.array([.05, .5, .95]))[:, None]
                       * np.sqrt(np.diag(whole_covariance))[None, :])
    for j, v in enumerate(mapping):
        if v is not None:
            quantiles[:, j] = eta[v]
    return {'mean': whole_mean, 'covariance': whole_covariance,
            'quantiles': quantiles, 'unknown': missing, 'mapping': mapping,
            'unknown_mean': mu, 'unknown_covariance': var,
            'm0': m0, 'k0': k0, 'cross': c, 'joint_mean': m, 'joint_covariance': k}

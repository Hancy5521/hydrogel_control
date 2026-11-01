"""Deterministic Gaussian metrics and bounded-tilt marginal quantiles.

Metrics consume frozen laws. They never choose or fit model parameters.
"""
from itertools import product
import numpy as np
from scipy.special import ndtr, ndtri, logsumexp
from scipy.optimize import brentq


def gaussian_metrics(mean, covariance, truth):
    mean, covariance, truth = map(np.asarray, (mean, covariance, truth))
    sd = np.sqrt(np.diag(covariance))
    if (truth.shape != mean.shape or not np.isfinite(sd).all() or (sd <= 0).any()):
        raise ValueError('Nonempty finite truths and positive marginal variances required')
    z = (truth - mean) / sd
    crps = sd * (z * (2 * ndtr(z) - 1) + 2 * np.exp(-z * z / 2) / np.sqrt(2 * np.pi) - 1 / np.sqrt(np.pi))
    d = np.linalg.solve(np.linalg.cholesky(covariance), truth - mean)
    logpdf = -.5 * (len(mean) * np.log(2 * np.pi) + np.linalg.slogdet(covariance)[1] + d @ d)
    half = ndtri(.95) * sd
    lower, upper = mean - half, mean + half
    return {'crps': float(crps.mean()), 'per_target_crps': crps.tolist(),
        'mean_mse': float(np.mean((mean - truth) ** 2)), 'joint_log_density': float(logpdf),
        'coverage90': float(np.mean((truth >= lower) & (truth <= upper))),
        'width90': float(np.mean(upper - lower)), 'mean': mean.tolist(),
        'lower90': lower.tolist(), 'upper90': upper.tolist()}


def tilt_quantiles(query, potential, theta, order=128):
    factor = query.latent_support.factor
    rank = factor.shape[1]
    if rank:
        x, w = np.polynomial.hermite.hermgauss(order)
        grid = np.array(list(product(range(order), repeat=rank)), dtype=int)
        offsets = np.sqrt(2) * x[grid] @ factor.T
        lw = np.log(w[grid] / np.sqrt(np.pi)).sum(1)
    else:
        offsets = np.zeros((1, potential.rank))
        lw = np.zeros(1)
    energy, _ = potential.evaluate(theta, query.latent_mean + offsets)
    lw = lw - energy
    weights = np.exp(lw - logsumexp(lw))
    means = query.mean + offsets @ query.decoder_gain.T
    sd = np.sqrt(np.sum(query.residual_support.factor ** 2, axis=1))
    quantiles = np.empty((2, len(query.mean)))
    root_errors = []
    for j in range(len(query.mean)):
        if sd[j] <= 0:
            order_idx = np.argsort(means[:, j])
            cdf = np.cumsum(weights[order_idx])
            quantiles[:, j] = means[order_idx[np.searchsorted(cdf, [.05, .95])], j]
            continue
        lo, hi = float(means[:, j].min() - 10 * sd[j]), float(means[:, j].max() + 10 * sd[j])
        def cdf(z):
            return float(ndtr((z - means[:, j]) / sd[j]) @ weights)
        for k, prob in enumerate([.05, .95]):
            result = brentq(lambda z: cdf(z) - prob, lo, hi, xtol=1e-12)
            quantiles[k, j] = result
            root_errors.append(abs(cdf(result) - prob))
    return quantiles[0], quantiles[1], float(max(root_errors, default=0))

"""Exact finite-hypothesis CRPS risk identity with one-dimensional quadrature.

The identity is a classical Bayes/energy-distance specialization. No generator,
model fit, observation mutation, or material source I/O is performed here.
"""
import math
import warnings
import numpy as np
from scipy import integrate, special
from hypothesis_model import checked, observation_message


def gaussian_pair_distances(means, standard_deviations):
    means, sd = np.asarray(means, dtype=float), np.asarray(standard_deviations, dtype=float)
    checked([1/3] * 3, means, sd)
    difference = means[:, None] - means[None, :]
    width = np.sqrt(sd[:, None] ** 2 + sd[None, :] ** 2)
    z = difference / width
    # Includes nonzero diagonal E|Y_h-Y'_h|, which must not be set to zero.
    return width * math.sqrt(2 / math.pi) * np.exp(-.5 * z ** 2) + difference * (2 * special.ndtr(z) - 1)


def risk_value(weights, query_means, query_sd, target_means, target_sd):
    weights, query_means, query_sd = checked(weights, query_means, query_sd)
    distance = gaussian_pair_distances(target_means, target_sd)
    calls = 0

    def integrand(z):
        nonlocal calls
        calls += 1
        posterior, log_density = observation_message(weights, query_means, query_sd, z)
        density = math.exp(log_density)
        difference = posterior - weights
        return density * np.concatenate((np.outer(difference, difference).ravel(),
                                         np.outer(posterior, posterior).ravel(), posterior, [1.]))

    with warnings.catch_warnings():
        warnings.simplefilter('error', integrate.IntegrationWarning)
        moment, error, info = integrate.quad_vec(integrand, -14., 14., epsabs=2e-12,
                                                epsrel=2e-11, limit=200, full_output=True)
    if not info.success or not np.isfinite(moment).all():
        raise ArithmeticError('finite original-density quadrature required')
    covariance, second = moment[:9].reshape(3, 3), moment[9:18].reshape(3, 3)
    value = -.5 * float(np.sum(covariance * distance))
    # Keep raw numerical nulls, including negative roundoff. Never clamp success.
    return {'value_C_D': value, 'C': covariance.tolist(), 'D': distance.tolist(),
            'posterior_second_moment': second.tolist(), 'expected_posterior_truncated': moment[18:21].tolist(),
            'query_density_mass_truncated': float(moment[21]),
            'quadrature_diagnostics': {'query_domain': [-14., 14.],
                'query_density_renormalized': False, 'quad_vec_error_estimate': float(error),
                'integrand_calls': calls, 'success': bool(info.success),
                'target_integrated_analytically_over_entire_real_line': True,
                'value_error_diagnostic_from_moment_norm': float(.5 * np.linalg.norm(distance) * error),
                'not_a_directed_rounding_certificate': True}}


"""Fixed Gaussian product hypotheses and genuine observed-label Bayes messages.

Synthetic mechanism only. Given H the query and target residuals are independent.
This condition does not follow from zero Student-t correlation.
"""
import numpy as np
from scipy import special


def checked(weights, means, standard_deviations):
    weights, means, sd = (np.asarray(x, dtype=float) for x in (weights, means, standard_deviations))
    if (weights.shape != (3,) or means.shape != (3,) or sd.shape != (3,)
            or not np.isfinite(weights).all() or not np.isfinite(means).all()
            or not np.isfinite(sd).all() or (weights <= 0).any() or (sd <= 0).any()
            or abs(float(weights.sum()) - 1) > 2e-14):
        raise ValueError('three positive normalized hypotheses and finite Gaussian laws required')
    return weights, means, sd


def observation_message(weights, means, standard_deviations, observed_value):
    weights, means, sd = checked(weights, means, standard_deviations)
    z = float(observed_value)
    if not np.isfinite(z):
        raise ValueError('finite actual or explicitly hypothetical scalar required')
    log_likelihood = -.5 * ((z - means) / sd) ** 2 - np.log(sd) - .5 * np.log(2 * np.pi)
    log_unnormalized = np.log(weights) + log_likelihood
    log_density = float(special.logsumexp(log_unnormalized))
    posterior = np.exp(log_unnormalized - log_density)
    return posterior, log_density


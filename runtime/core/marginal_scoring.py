"""Deterministic marginal CRPS for a bounded tilt of a Gaussian decoder.

Only rank-one/two latent potentials are supported. Gauss-Hermite integrates the
conditional latent Gaussian; Gauss-Legendre integrates each mixture marginal's
squared CDF, split at the truth. There is no fitting, sampling, or added jitter.
The truth vector is aligned with query.targets, not a vector of addresses.
"""
from itertools import product
import numpy as np
from scipy.special import ndtr, logsumexp
from latent_reference import ConditionalQuery, real
from energy_correction import BoundedPotential

MAX_LATENT_NODES = 16384
TAIL_STANDARD_DEVIATIONS = 10.0
_SQRT_2PI = np.sqrt(2.0 * np.pi)
_INV_SQRT_PI = 1.0 / np.sqrt(np.pi)


def _order(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, int) or not 2 <= value <= 256:
        raise ValueError(name + " must be a Python integer in 2..256")
    return value


def _gaussian_crps(mean, covariance, truth):
    variance = np.diag(covariance)
    if np.any(variance < 0):
        raise ValueError("Negative reference marginal variance; no jitter fallback")
    result = np.abs(truth - mean)
    positive = variance > 0
    sd = np.sqrt(variance[positive])
    z = (truth[positive] - mean[positive]) / sd
    # Same operation order as the registered reference Gaussian CRPS expression.
    result[positive] = sd * (z * (2 * ndtr(z) - 1)
        + 2 * np.exp(-z * z / 2) / _SQRT_2PI - _INV_SQRT_PI)
    return result


def _atomic_crps(centered_means, weights, centered_truth):
    order = np.argsort(centered_means, kind="stable")
    m, w = centered_means[order], weights[order]
    before_w = np.r_[0.0, np.cumsum(w)[:-1]]
    before_wm = np.r_[0.0, np.cumsum(w * m)[:-1]]
    half_pair_absolute = np.sum(w * (m * before_w - before_wm))
    result = float(weights @ np.abs(centered_means - centered_truth) - half_pair_absolute)
    tolerance = 128 * np.finfo(float).eps * max(
        1.0, float(np.max(np.abs(centered_means), initial=0.0)), abs(float(centered_truth)))
    if result < -tolerance:
        raise FloatingPointError("Negative atomic CRPS beyond roundoff")
    return max(0.0, result), result < 0.0


def _cdf_integral(component_means, weights, residual_sd, truth, reference_mean,
                  reference_sd, legendre_nodes, legendre_weights):
    base_left = reference_mean - TAIL_STANDARD_DEVIATIONS * reference_sd
    base_right = reference_mean + TAIL_STANDARD_DEVIATIONS * reference_sd
    left, right = min(base_left, truth), max(base_right, truth)
    if not np.isfinite([left, right]).all():
        raise FloatingPointError("Nonfinite CRPS integration domain")
    # The original window endpoints and midpoint keep the central transition
    # in panels no wider than 10 reference SD even for a far-outside truth.
    # Every interval is still split at the truth.
    edges = np.unique([left, base_left, reference_mean, truth, base_right, right])
    total, count = 0.0, 0
    for a, b in zip(edges[:-1], edges[1:]):
        if a == b:
            continue
        half, middle = b / 2 - a / 2, a / 2 + b / 2
        x = middle + half * legendre_nodes
        standardized = (x[:, None] - component_means[None, :]) / residual_sd
        if b <= truth:
            probability = ndtr(standardized) @ weights
        elif a >= truth:
            # Evaluate the survival CDF directly, avoiding 1-F cancellation.
            probability = ndtr(-standardized) @ weights
        else:
            raise AssertionError("CDF integration interval crossed the truth")
        total += half * (legendre_weights @ (probability * probability))
        count += len(legendre_nodes)

    # Since F^2 <= F, each omitted tail is bounded by a mixture stop-loss
    # expectation. The expressions below are conservative cancellation-free
    # upper bounds on those expectations, valid for this finite GH mixture.
    z_left = (left - component_means) / residual_sd
    z_right = (right - component_means) / residual_sd
    phi_left = np.exp(-z_left * z_left / 2) / _SQRT_2PI
    phi_right = np.exp(-z_right * z_right / 2) / _SQRT_2PI
    left_bound = float(weights @ (np.maximum(left - component_means, 0.0) + residual_sd * phi_left))
    right_bound = float(weights @ (np.maximum(component_means - right, 0.0) + residual_sd * phi_right))
    return float(total), left_bound + right_bound, [float(left), float(right)], count


def score(query, potential, theta, targets, latent_order=32, cdf_order=64):
    """Return CRPS, tilted joint mean/covariance, and numerical metadata.

    targets are finite truth values in the order of query.targets.
    Conditional decoder covariance is the qualified query residual, not the
    prior residual. Tails of the finite GH mixture are analytically bounded;
    latent quadrature and CDF quadrature errors require order refinement.
    """
    if not isinstance(query, ConditionalQuery) or not isinstance(potential, BoundedPotential):
        raise ValueError("Qualified ConditionalQuery and BoundedPotential required")
    if potential.rank not in (1, 2):
        raise ValueError("Marginal scorer supports latent potential rank at most two")
    latent_order, cdf_order = _order(latent_order, "latent_order"), _order(cdf_order, "cdf_order")
    theta, truth = real(theta, "theta", 1), real(targets, "target truth values", 1)
    mean0, covariance0 = real(query.mean, "reference mean", 1), real(query.covariance, "reference covariance", 2)
    dimension = len(mean0)
    if truth.shape != mean0.shape or covariance0.shape != (dimension, dimension):
        raise ValueError("Truth/query dimensions mismatch")
    if len(theta) != potential.parameter_count or len(query.latent_mean) != potential.rank:
        raise ValueError("Potential/query coefficient dimensions mismatch")
    if dimension == 0:
        raise ValueError("At least one missing target is required")
    variance0 = np.diag(covariance0)
    if np.any(variance0 < 0):
        raise ValueError("Negative reference marginal variance; no jitter fallback")
    factor = query.latent_support.factor
    residual_factor = query.residual_support.factor
    conditional_rank = factor.shape[1]
    if factor.shape[0] != potential.rank or query.decoder_gain.shape != (dimension, potential.rank):
        raise ValueError("Invalid latent factor/decoder dimensions")
    if residual_factor.shape[0] != dimension:
        raise ValueError("Invalid conditional residual dimensions")

    metadata = {
        "method": "GH latent mixture; split Gauss-Legendre marginal CDF-square integral",
        "latent_order": latent_order, "cdf_order": cdf_order,
        "latent_support_rank": conditional_rank,
        "latent_nodes": 0, "max_latent_nodes": MAX_LATENT_NODES,
        "numerical_jitter": 0.0,
        "residual_covariance": "post-observation conditional decoder residual",
        "truth_order": "query.targets",
        "tail_standard_deviations": TAIL_STANDARD_DEVIATIONS,
        "latent_quadrature_error_certified": False,
        "cdf_quadrature_error_certified": False,
        "refinement_required_for_nonGaussian_case": True,
    }
    analytic_reason = ("kappa_zero" if potential.kappa == 0.0 else
                       "theta_zero" if np.count_nonzero(theta) == 0 else
                       "latent_rank_zero" if conditional_rank == 0 else None)
    if analytic_reason is not None:
        log_a = 0.0
        if analytic_reason == "latent_rank_zero":
            energy, _ = potential.evaluate(theta, query.latent_mean[None, :])
            log_a = -float(energy[0])
        metadata.update(
            method="analytic Gaussian CRPS", analytic_reason=analytic_reason,
            log_normalizer=log_a, cdf_nodes_per_target=[0] * dimension,
            finite_mixture_tail_bound_per_target=[0.0] * dimension,
            finite_mixture_tail_bound_max=0.0,
            integration_domain_per_target=[None] * dimension,
            refinement_required_for_nonGaussian_case=False,
        )
        return {"per_target_crps": _gaussian_crps(mean0, covariance0, truth),
                "mean": mean0.copy(), "cov": covariance0.copy(),
                "numerical_metadata": metadata}

    nodes = latent_order ** conditional_rank
    if nodes > MAX_LATENT_NODES:
        raise ValueError("Latent quadrature node budget exceeded; no hidden rule change")
    x, w = np.polynomial.hermite.hermgauss(latent_order)
    grid = np.array(list(product(range(latent_order), repeat=conditional_rank)), dtype=int)
    offsets = np.sqrt(2.0) * x[grid] @ factor.T
    points = query.latent_mean + offsets
    log_weights = np.log(w[grid] / np.sqrt(np.pi)).sum(axis=1)
    log_weights -= logsumexp(log_weights)
    energy, _ = potential.evaluate(theta, points)
    log_a = float(logsumexp(log_weights - energy))
    weights = np.exp(log_weights - energy - log_a)
    weights /= weights.sum()

    decoder_offsets = offsets @ query.decoder_gain.T
    mean_offset = weights @ decoder_offsets
    mean = mean0 + mean_offset
    centered = decoder_offsets - mean_offset
    residual_covariance = residual_factor @ residual_factor.T
    covariance = residual_covariance + centered.T @ (weights[:, None] * centered)
    covariance = (covariance + covariance.T) / 2
    residual_sd = np.sqrt(np.diag(residual_covariance))
    reference_sd = np.sqrt(variance0)
    legendre_nodes, legendre_weights = np.polynomial.legendre.leggauss(cdf_order)
    crps, tails, domains, counts, methods = [], [], [], [], []
    atomic_roundoff_clips = 0
    for j in range(dimension):
        if residual_sd[j] == 0.0:
            value, clipped = _atomic_crps(decoder_offsets[:, j], weights, truth[j] - mean0[j])
            crps.append(value); tails.append(0.0); domains.append(None); counts.append(0)
            methods.append("exact finite atomic-mixture CRPS")
            atomic_roundoff_clips += int(clipped)
        else:
            value, tail, domain, count = _cdf_integral(
                mean0[j] + decoder_offsets[:, j], weights, residual_sd[j],
                truth[j], mean0[j], reference_sd[j], legendre_nodes, legendre_weights)
            crps.append(value); tails.append(tail); domains.append(domain); counts.append(count)
            methods.append("split Gauss-Legendre CDF-square integral")
    crps = np.array(crps)
    if not np.isfinite(crps).all() or not np.isfinite(mean).all() or not np.isfinite(covariance).all():
        raise FloatingPointError("Nonfinite marginal score/moments")
    metadata.update(
        latent_nodes=nodes, log_normalizer=log_a,
        effective_mixture_components=float(1.0 / (weights @ weights)),
        cdf_nodes_per_target=counts,
        finite_mixture_tail_bound_per_target=tails,
        finite_mixture_tail_bound_max=float(max(tails)),
        integration_domain_per_target=domains,
        per_target_integration_method=methods,
        atomic_roundoff_clips=atomic_roundoff_clips,
    )
    return {"per_target_crps": crps, "mean": mean, "cov": covariance,
            "numerical_metadata": metadata}


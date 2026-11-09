"""One-query effects for a fixed conditional Gaussian curve reference.

Classical Bayesian design adapted to weighted marginal CRPS on a fixed scoring
panel. No material values, fitting, sampling policy, or filesystem access.
The query coordinate is outside the scoring panel; known-cell zero loss cannot
make the scoring panel easier. Fixed Gaussian utilities do not depend on Y.
"""
from dataclasses import dataclass
from math import isfinite, pi, sqrt
import numpy as np


def real_array(value, ndim, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or any(isinstance(x, (bool, np.bool_)) for x in np.asarray(value, dtype=object).flat):
        raise ValueError(name + " must contain real nonboolean numbers")
    out = np.array(raw, dtype=float, copy=True)
    if out.ndim != ndim or not np.isfinite(out).all():
        raise ValueError(name + " has invalid shape or values")
    return out


def integer_indices(value, size):
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.dtype.kind not in "iu" or not len(raw):
        raise ValueError("Nonempty integer target indices required")
    out = np.array(raw, dtype=int, copy=True)
    if len(set(out.tolist())) != len(out) or (out < 0).any() or (out >= size).any():
        raise ValueError("Unique in-range target indices required")
    return out


@dataclass(frozen=True)
class QueryEffect:
    query_index: int
    innovation_variance: float
    response: np.ndarray
    variance_after: np.ndarray
    expected_gain_by_target: np.ndarray
    normalized_target_weights: np.ndarray
    reference_bayes_crps: float
    posterior_bayes_crps: float
    weighted_expected_gain: float


def query_effect(covariance, target_indices, query_index, *, noise_variance=0.0, target_weights=None):
    """Return model-expected CRPS gain and d posterior_mean_T / d observed_Yq.

    K describes the intended scored coordinates. noise_variance is extra,
    independent noise on the newly queried coordinate in the SAME value scale
    (e.g. log viscosity variance). It is not a standard deviation. The scoring
    target law and its noise semantics must be fixed by the calling protocol.
    Targets/query must be unobserved latent coordinates with SPD joint sub-K.
    Other already-clamped rows may have zero variance and are not factored.
    """
    k = real_array(covariance, 2, "covariance")
    if k.shape[0] != k.shape[1] or not len(k):
        raise ValueError("Nonempty square covariance required")
    if not np.allclose(k, k.T, rtol=1e-12, atol=1e-12):
        raise ValueError("Symmetric covariance required")
    targets = integer_indices(target_indices, len(k))
    if isinstance(query_index, (bool, np.bool_)) or not isinstance(query_index, (int, np.integer)):
        raise ValueError("Integer query index required")
    query_index = int(query_index)
    if query_index < 0 or query_index >= len(k) or query_index in targets:
        raise ValueError("Query must be in range and outside fixed target panel")
    if isinstance(noise_variance, (bool, np.bool_)) or not isinstance(noise_variance, (int, float, np.integer, np.floating)):
        raise ValueError("Nonnegative real noise variance required")
    noise = float(noise_variance)
    if not isfinite(noise) or noise < 0:
        raise ValueError("Nonnegative finite noise variance required")
    sub = np.r_[targets, query_index]
    np.linalg.cholesky(k[np.ix_(sub, sub)])
    weights = (np.ones(len(targets)) if target_weights is None else
               real_array(target_weights, 1, "target_weights"))
    if weights.shape != (len(targets),) or (weights <= 0).any():
        raise ValueError("One positive weight per target required")
    total = float(np.sum(weights))
    if not isfinite(total) or total <= 0:
        raise ValueError("Finite positive weight sum required")
    weights = weights / total
    denominator = float(k[query_index, query_index] + noise)
    if not isfinite(denominator) or denominator <= 0:
        raise ArithmeticError("Finite positive innovation variance required")
    cross = k[targets, query_index]
    variance = np.diag(k)[targets]
    after = variance - (cross / sqrt(denominator)) ** 2
    if not np.isfinite(after).all() or (after <= 0).any():
        raise ArithmeticError("Positive finite Schur variances required; no jitter/clipping")
    per_target = (np.sqrt(variance) - np.sqrt(after)) / sqrt(pi)
    before_risk = float(weights @ np.sqrt(variance) / sqrt(pi))
    after_risk = float(weights @ np.sqrt(after) / sqrt(pi))
    response = cross / denominator
    for array in (response, after, per_target, weights):
        array.setflags(write=False)
    return QueryEffect(query_index, denominator, response, after, per_target,
                       weights, before_risk, after_risk, float(weights @ per_target))


def rank_queries(covariance, target_indices, candidate_indices, *, noise_variances=None,
                 costs=None, target_weights=None):
    """Rank fixed candidates by expected CRPS gain per declared positive cost."""
    candidates = integer_indices(candidate_indices, len(covariance))
    noise = (np.zeros(len(candidates)) if noise_variances is None else
             real_array(noise_variances, 1, "noise_variances"))
    cost = (np.ones(len(candidates)) if costs is None else real_array(costs, 1, "costs"))
    if noise.shape != candidates.shape or cost.shape != candidates.shape or (cost <= 0).any():
        raise ValueError("Aligned noise and positive costs required")
    rows = []
    for i, q in enumerate(candidates):
        effect = query_effect(covariance, target_indices, q, noise_variance=noise[i], target_weights=target_weights)
        ratio = effect.weighted_expected_gain / float(cost[i])
        if not isfinite(ratio):
            raise ArithmeticError("Finite gain per cost required")
        rows.append({"query_index": int(q), "cost": float(cost[i]),
                     "weighted_expected_gain": effect.weighted_expected_gain,
                     "gain_per_cost": ratio,
                     "effect": effect})
    return sorted(rows, key=lambda row: (-row["gain_per_cost"], row["query_index"]))

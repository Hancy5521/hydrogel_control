"""Decision-specific positive-moment readout; full posterior branches retained.

Analytic tail bounds concern ideal arithmetic. This code does not implement
general IEEE interval arithmetic or certify material-model correctness.
"""
import math
import numpy as np


class NumericalRangeUnsupported(ArithmeticError):
    pass


def validate(weights, rhos, scales):
    w, rho, scale = (np.asarray(v, dtype=float) for v in (weights, rhos, scales))
    if (w.ndim != 2 or rho.ndim != 2 or scale.shape != (len(rho),)
            or w.shape[1] != rho.shape[1] or w.shape[1] < 2):
        raise ValueError('Branch-by-hypothesis weights, proxy-by-hypothesis rhos and proxy scales required')
    if (not all(np.isfinite(v).all() for v in (w, rho, scale)) or (w < 0).any()
            or not np.allclose(w.sum(axis=1), 1., atol=1e-13, rtol=0.)
            or (np.abs(rho) >= 1).any() or (scale <= 0).any()):
        raise ValueError('Finite simplex weights, open correlations and positive scales required')
    return w, rho, scale


def coefficients(degree):
    if type(degree) is not int or degree < 1:
        raise ValueError('Positive integer degree required')
    values, coefficient = [], .5
    for n in range(1, degree + 1):
        values.append(coefficient)
        coefficient *= (2 * n - 1) / (2 * (n + 1))
    return np.asarray(values)


def analytic_tail_bound(r, scale, degree):
    """Sharper coefficient-mass/geometric ideal-arithmetic bound, float display."""
    if (not math.isfinite(r) or not 0 <= r < 1 or not math.isfinite(scale) or scale <= 0
            or type(degree) is not int or degree < 0):
        raise ValueError('Open r bound, positive scale and nonnegative degree required')
    if r == 0:
        return 0.
    beta, next_a = 1., .5
    for n in range(1, degree + 1):
        beta *= (2 * n - 1) / (2 * n)
        next_a *= (2 * n - 1) / (2 * (n + 1))
    constant = min(beta, next_a / (1 - r * r))
    log_bound = math.log(scale / math.sqrt(math.pi)) + math.log(constant) + (2 * degree + 2) * math.log(r)
    if log_bound < math.log(np.nextafter(0., 1.)):
        raise NumericalRangeUnsupported('Positive tail display underflows; no false zero certificate')
    value = math.exp(log_bound)
    if value == 0 or not math.isfinite(value):
        raise NumericalRangeUnsupported('Tail display is outside supported float range')
    return value


def degree_for(r, scale, hypothesis_count, epsilon, cap=64):
    if not math.isfinite(epsilon) or epsilon <= 0 or type(cap) is not int or cap < 1:
        raise ValueError('Positive tolerance and degree cap required')
    if r == 0:
        return 0, 0., 'exact_zero_relation'
    if not math.isfinite(r) or not 0 < r < 1 or not math.isfinite(scale) or scale <= 0:
        raise ValueError('Open finite correlation cap and positive scale required')
    beta, next_a = 1., .5
    log_r, log_prefactor = math.log(r), math.log(scale / math.sqrt(math.pi))
    for degree in range(1, cap + 1):
        beta *= (2 * degree - 1) / (2 * degree)
        next_a *= (2 * degree - 1) / (2 * (degree + 1))
        log_tail = log_prefactor + math.log(min(beta, next_a / (1 - r * r))) + (2 * degree + 2) * log_r
        if log_tail < math.log(np.nextafter(0., 1.)):
            raise NumericalRangeUnsupported('Positive degree-selection tail underflows; no false certificate')
        tail = math.exp(log_tail)
        if tail <= epsilon:
            if degree >= hypothesis_count:
                return None, None, 'degree_not_smaller_than_H_use_exact'
            return degree, tail, 'moment_readout'
    return None, None, 'tail_not_certified_within_cap_use_exact'


def exact_kernel(rho, scale):
    products = np.outer(rho, rho)
    if np.any((rho[:, None] != 0) & (rho[None, :] != 0) & (products == 0)):
        raise NumericalRangeUnsupported('Nonzero correlation product underflows')
    result = (scale / math.sqrt(math.pi)) * products / (1 + np.sqrt(1 - products))
    if not np.isfinite(result).all() or np.any((products != 0) & (result == 0)):
        raise NumericalRangeUnsupported('Full kernel entry is outside numeric range')
    return result


def exact_streamed(weights, rhos, scales):
    w, rho, scale = validate(weights, rhos, scales)
    scores = np.empty((len(w), len(rho)))
    peak = 0
    for b in range(len(rho)):
        kernel = exact_kernel(rho[b], scale[b])
        intermediate = w @ kernel
        scores[:, b] = np.sum(intermediate * w, axis=1)
        peak = max(peak, kernel.nbytes + intermediate.nbytes)
    if not np.isfinite(scores).all():
        raise ArithmeticError('Nonfinite exact score')
    return scores, {'algorithm_workspace_numeric_bytes': int(peak + scores.nbytes),
                    'kernel_builds': len(rho), 'full_kernel_bank_stored': False,
                    'posterior_outer_products_stored': False}


def exact_cached(weights, rhos, scales):
    w, rho, scale = validate(weights, rhos, scales)
    bank = np.stack([exact_kernel(r, s) for r, s in zip(rho, scale)])
    scores = np.empty((len(w), len(rho)))
    peak_intermediate = 0
    for b, kernel in enumerate(bank):
        intermediate = w @ kernel
        scores[:, b] = np.sum(intermediate * w, axis=1)
        peak_intermediate = max(peak_intermediate, intermediate.nbytes)
    if not np.isfinite(scores).all():
        raise ArithmeticError('Nonfinite cached exact score')
    return scores, {'algorithm_workspace_numeric_bytes': int(bank.nbytes + scores.nbytes + peak_intermediate),
                    'kernel_builds': len(rho), 'full_kernel_bank_stored': True,
                    'posterior_outer_products_stored': False}


def moment_streamed(weights, rhos, scales, epsilon=1e-6, cap=64):
    w, rho, scale = validate(weights, rhos, scales)
    scores = np.empty((len(w), len(rho)))
    tails = np.zeros(len(rho))
    degrees, reasons = [], []
    workspace, fallbacks = 0, 0
    positive_weights = w[w > 0]
    minimum_weight = float(positive_weights.min())
    for b, profile in enumerate(rho):
        degree, tail, reason = degree_for(float(np.abs(profile).max()), float(scale[b]), w.shape[1], epsilon, cap)
        degrees.append(degree)
        reasons.append(reason)
        if degree is None:
            kernel = exact_kernel(profile, scale[b])
            intermediate = w @ kernel
            scores[:, b] = np.sum(intermediate * w, axis=1)
            workspace = max(workspace, kernel.nbytes + intermediate.nbytes)
            tails[b] = 0.  # No truncation; this is NOT a zero floating-roundoff bound.
            fallbacks += 1
            continue
        if degree == 0:
            scores[:, b] = 0.
            continue
        powers = np.empty((len(profile), degree))
        previous = np.ones(len(profile))
        for n in range(degree):
            current = previous * profile
            if np.any((previous != 0) & (profile != 0) & (current == 0)):
                raise NumericalRangeUnsupported('Nonzero feature power underflows')
            powers[:, n] = current
            previous = current
        nonzero_powers = np.abs(powers[powers != 0])
        if len(nonzero_powers) and minimum_weight * float(nonzero_powers.min()) == 0:
            raise NumericalRangeUnsupported('Positive posterior-feature term may underflow; conservative refusal')
        moments = w @ powers
        squared = moments * moments
        if np.any((moments != 0) & (squared == 0)):
            raise NumericalRangeUnsupported('Nonzero moment square underflows')
        coefficient = coefficients(degree) * (scale[b] / math.sqrt(math.pi))
        terms = squared * coefficient[None, :]
        if np.any((squared != 0) & (terms == 0)):
            raise NumericalRangeUnsupported('Positive CRPS term underflows')
        scores[:, b] = np.sum(terms, axis=1)
        tails[b] = tail
        workspace = max(workspace, powers.nbytes + moments.nbytes + squared.nbytes + terms.nbytes + coefficient.nbytes)
    if not np.isfinite(scores).all() or (scores < 0).any():
        raise ArithmeticError('Positive-moment readout failed')
    return scores, {'degrees': degrees, 'fallback_reasons': reasons, 'fallback_proxies': fallbacks,
                    'ideal_arithmetic_truncation_tail_by_proxy': tails,
                    'tail_excludes_roundoff_and_model_error': True,
                    'algorithm_workspace_numeric_bytes': int(workspace + scores.nbytes + tails.nbytes),
                    'posterior_compressed_for_future_updates': False}


def adaptive_value(scores, branch_probabilities):
    values = np.asarray(scores, dtype=float)
    p = np.asarray(branch_probabilities, dtype=float)
    if values.ndim != 2 or p.ndim != 2 or p.shape[1] != 2 or len(values) != p.size:
        raise ValueError('Two outcome branches per first action required')
    if not np.isfinite(values).all() or not np.isfinite(p).all() or (p <= 0).any() or not np.allclose(p.sum(axis=1), 1., atol=1e-13, rtol=0.):
        raise ValueError('Finite scores and positive complete branch probabilities required')
    best = np.max(values, axis=1).reshape(p.shape)
    products = best * p
    if np.any((best != 0) & (products == 0)):
        raise NumericalRangeUnsupported('Nonzero final expected branch gain underflows')
    return np.sum(products, axis=1)


def h2_symmetric_stable(r, scale=1.):
    if not math.isfinite(r) or not 0 < r < 1 or not math.isfinite(scale) or scale <= 0:
        raise ValueError('Open positive r and scale required')
    fourth = (r * r) ** 2
    if fourth == 0:
        raise NumericalRangeUnsupported('Positive shape gain is outside supported float range')
    result = scale * fourth / (math.sqrt(math.pi) * (1 + math.sqrt(1 - fourth))
                 * (2 + math.sqrt(1 - r * r) + math.sqrt(1 + r * r)))
    if result == 0 or not math.isfinite(result):
        raise NumericalRangeUnsupported('Final positive shape gain underflows')
    return result

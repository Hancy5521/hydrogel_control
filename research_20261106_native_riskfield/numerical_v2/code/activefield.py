"""Finite-df native CRPS value field; deterministic quadrature and explicit error accounts.

Risk is model-based. Analytic rho/tail bounds do not certify finite quadrature,
CDF library error or IEEE arithmetic. No material data or active campaign runs here.
"""
import math
import numpy as np
from scipy import special, stats

LEVELS = (32, 64, 128, 256)
LV = 32.
LZ = 64.


def inverse_yj(value):
    value = np.asarray(value, dtype=float)
    result = np.empty_like(value)
    mask = value >= 0
    result[mask] = value[mask] * (1 + value[mask] / 4)
    result[~mask] = -np.expm1((2 / 3) * np.log1p(-1.5 * value[~mask]))
    return result


def derivative(value, coordinate):
    value = np.asarray(value, dtype=float)
    if coordinate == 'affine':
        return np.ones_like(value)
    if coordinate != 'YJ_half':
        raise ValueError('fixed YJ_half or affine loss required')
    result = np.empty_like(value)
    mask = value >= 0
    result[mask] = 1 + .5 * value[mask]
    result[~mask] = np.exp(-np.log1p(-1.5 * value[~mask]) / 3)
    return result


def legendre_interval(count, lower, upper):
    nodes, weights = np.polynomial.legendre.leggauss(count)
    return (lower + upper) / 2 + (upper - lower) * nodes / 2, weights * (upper - lower) / 2


def student_kernel(v, nu, rho, count):
    """Original truncated t law in theta coordinates, not a normalized surrogate."""
    lower_angle, upper_angle = math.atan(-LZ / math.sqrt(nu)), math.atan(LZ / math.sqrt(nu))
    angles, angle_weights = legendre_interval(count, lower_angle, upper_angle)
    cosine, sine = np.cos(angles), np.sin(angles)
    # t_nu(sqrt(nu)tan(theta))*dz = cos(theta)^(nu-1)/B(.5,nu/2)dtheta.
    density = np.exp((nu - 1) * np.log(cosine) - special.betaln(.5, nu / 2))
    weights = angle_weights * density
    # cos(theta)>0 throughout the fixed atan limits, so this is the exact conditional argument.
    argument = math.sqrt((nu + 1) / (nu * (1 - rho ** 2))) * (
        v[:, None] * cosine[None, :] - rho * math.sqrt(nu) * sine[None, :])
    difference = np.empty_like(argument)
    right = v >= 0
    difference[right] = special.stdtr(nu + 1, -argument[right]) - special.stdtr(nu, -v[right, None])
    difference[~right] = special.stdtr(nu + 1, argument[~right]) - special.stdtr(nu, v[~right, None])
    kernel = (difference * difference) @ weights
    if not np.isfinite(kernel).all() or (kernel < 0).any():
        raise ArithmeticError('nonnegative finite Student kernel required')
    probability = float(special.stdtr(nu, LZ) - special.stdtr(nu, -LZ))
    return kernel, {'query_theta_interval': [lower_angle, upper_angle],
        'query_probability_mass_integrated': float(weights.sum()),
        'analytic_query_probability_mass': probability,
        'query_mass_quadrature_abs_difference': abs(float(weights.sum()) - probability),
        'renormalized_query_weights': False, 'query_nodes': count,
        'coordinate': 'z=sqrt(nu)*tan(theta)',
        'weighted_original_density': 'cos(theta)^(nu-1)/Beta(1/2,nu/2)'}


def gaussian_kernel(v, rho, count):
    """OwenT one-dimensional readout, positive Plackett-angle fallback for cancellation."""
    if rho == 0:
        return np.zeros_like(v), {'exact_zero_rho_independence': True, 'fallback_nodes': 0}
    c = rho ** 2
    first = special.ndtr(v) * special.ndtr(-v)
    second = 2 * special.owens_t(v, math.sqrt((1 - c) / (1 + c)))
    raw = first - second
    tolerance = 64 * np.finfo(float).eps * (np.abs(first) + np.abs(second))
    fallback = (raw < 0) | (raw <= tolerance)
    kernel = raw.copy()
    angle_underflows = 0
    if fallback.any():
        angles, weights = legendre_interval(count, 0., math.asin(c))
        exponent = -v[fallback, None] ** 2 / (1 + np.sin(angles[None, :]))
        integrand = np.exp(exponent)
        angle_underflows = int(((integrand == 0) & np.isfinite(exponent)).sum())
        kernel[fallback] = integrand @ weights / (2 * math.pi)
    if not np.isfinite(kernel).all() or (kernel < 0).any():
        raise ArithmeticError('positive-angle fallback must yield nonnegative finite Gaussian kernel')
    return kernel, {'exact_zero_rho_independence': False, 'raw_Owen_negative_nodes': int((raw < 0).sum()),
        'raw_Owen_min': float(raw.min()), 'fallback_nodes': int(fallback.sum()),
        'positive_angle_exp_underflow_nodes': angle_underflows,
        'fallback_formula': 'K=1/(2pi) integral_0^asin(rho²) exp[-v²/(1+sin(theta))]dtheta',
        'hard_negative_clamp_used': False}


def native_value(fixture, count):
    family, mu, scale, s0, rho = (fixture[name] for name in ('family', 'mu', 'scale', 's0', 'rho'))
    if not all(math.isfinite(value) for value in (mu, scale, s0, rho)) or min(scale, s0) <= 0 or abs(rho) >= 1:
        raise ValueError('finite nondegenerate fixture required')
    if family == 'normal' and rho == 0:
        return {'value': 0., 'nodes_per_segment': count, 'segments': [],
                'exact_model_zero_independence': True, 'tail': {'total_bound_ideal_arithmetic': 0.}}
    split = sorted(set([-LV, LV, 0.] + ([-mu / scale] if -LV < -mu / scale < LV else [])))
    pieces = []
    for lower, upper in zip(split[:-1], split[1:]):
        v, weights = legendre_interval(count, lower, upper)
        kernel, diagnostics = (student_kernel(v, fixture['nu'], rho, count) if family == 'student_t'
                               else gaussian_kernel(v, rho, count))
        value = s0 * scale * float(weights @ (kernel * derivative(mu + scale * v, fixture['loss_coordinate'])))
        if value < 0 or not math.isfinite(value):
            raise ArithmeticError('nonnegative native value required')
        pieces.append({'lower_v': lower, 'upper_v': upper, 'value': value,
                       'target_nodes': count, 'kernel_diagnostics': diagnostics})
    return {'value': math.fsum(piece['value'] for piece in pieces), 'nodes_per_segment': count,
            'segments': pieces, 'exact_model_zero_independence': False, 'tail': tail_bounds(fixture)}


def tail_bounds(fixture):
    mu, scale, s0 = (fixture[name] for name in ('mu', 'scale', 's0'))
    affine = fixture['loss_coordinate'] == 'affine'
    a, b = (1., 0.) if affine else (1 + .5 * abs(mu), .5 * scale)
    if fixture['family'] == 'student_t':
        nu = fixture['nu']
        density, sf = float(stats.t.pdf(LV, nu)), float(special.stdtr(nu, -LV))
        positive0 = (nu + LV ** 2) / (nu - 1) * density
        positive1 = .5 * (nu * sf + LV * (nu + LV ** 2) * density) / (nu - 2)
        exact0 = positive0 - LV * sf
        exact1 = positive1 - .5 * LV ** 2 * sf
        outer = 2 * s0 * scale * (a * positive0 + b * positive1)
        u = np.asarray([mu - scale * LV, mu + scale * LV])
        native = s0 * (u if affine else inverse_yj(u))
        inner = 2 * float(special.stdtr(nu, -LZ)) * float(native[1] - native[0])
        return {'outer_v_cut': LV, 'query_z_cut': LZ,
            'I0_closed_difference_display': exact0, 'I1_closed_difference_display': exact1,
            'I0_safe_ignored_negative_term_bound': positive0, 'I1_safe_ignored_negative_term_bound': positive1,
            'outer_tail_bound_ideal_arithmetic': outer, 'query_tail_bound_ideal_arithmetic': inner,
            'total_bound_ideal_arithmetic': outer + inner,
            'not_IEEE_directed_or_CDF_error_certificate': True}
    density, sf = float(stats.norm.pdf(LV)), float(special.ndtr(-LV))
    outer = 2 * s0 * scale * (a * density + b * .5 * (LV * density + sf))
    return {'outer_v_cut': LV, 'outer_tail_bound_ideal_arithmetic': outer,
        'query_tail_bound_ideal_arithmetic': 0.,
        'total_bound_ideal_arithmetic': outer,
        'Gaussian_query_integrated_by_Owen_or_positive_angle_no_query_truncation': True,
        'not_IEEE_directed_or_CDF_error_certificate': True}


def risk_constant_student(nu):
    """Bayes CRPS risk of unit-scale t_nu, not CRPS at its median."""
    return 2 * math.sqrt(nu) / (nu - 1) * math.exp(special.betaln(.5, nu - .5)
                                                  - 2 * special.betaln(.5, nu / 2))


def affine_student_value(nu, scale, rho, s0):
    factor = math.sqrt(nu / (nu + 1)) * math.exp(special.betaln(.5, (nu - 1) / 2)
                                                  - special.betaln(.5, nu / 2))
    return s0 * scale * (risk_constant_student(nu)
                        - risk_constant_student(nu + 1) * factor * math.sqrt(1 - rho ** 2))


def affine_matched_gaussian_value(nu, scale, rho, s0):
    return s0 * scale * math.sqrt(nu / (nu - 2)) / math.sqrt(math.pi) * (1 - math.sqrt(1 - rho ** 2))


def rho_coupling_bound(fixture, other_rho):
    nu, scale, mu, s0, rho = (fixture[name] for name in ('nu', 'scale', 'mu', 's0', 'rho'))
    second_norm = math.sqrt(mu ** 2 + scale ** 2 * nu / (nu - 2))
    chord = math.hypot(rho - other_rho, math.sqrt(1 - rho ** 2) - math.sqrt(1 - other_rho ** 2))
    constant = s0 * (1 + second_norm) * scale * math.sqrt(nu / (nu - 2))
    return {'target_L2_norm': second_norm, 'constant': constant, 'chord': chord,
            'angle_distance': abs(math.asin(rho) - math.asin(other_rho)),
            'bound_ideal_arithmetic': constant * chord,
            'scope': 'conditional risk W1 first, then E_Z; not unconditional identical-marginal W1 or finite-quad bound'}

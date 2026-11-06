"""Native scoring of a three-Student mixture in one shared YJ coordinate.

CRPS_mix = sum(w_h CRPS_h) - integral Var_H(F_h)dy. Component scores use
the frozen, independently audited scalar scorer; quantiles/PIT use the mixture.
Numerical error estimates and ideal-arithmetic tails remain separate diagnostics.
This module has no material source I/O or fitted state updates.
"""
import math
import warnings
import numpy as np
from scipy import integrate, optimize, special, stats


def validate(law):
    if (law.get('family') != 'mixture_student_t' or law.get('coordinate') != 'u=YJ_half(native/s0)'
            or law.get('lambda') != .5 or len(law.get('components', [])) != 3):
        raise ValueError('fixed three-Student shared inverse-YJ law required')
    logweights = np.asarray(law['log_weights'], dtype=float)
    if logweights.shape != (3,) or not np.isfinite(logweights).all():
        raise ValueError('three finite log probabilities required')
    if abs(float(special.logsumexp(logweights))) > 1e-10:
        raise ValueError('normalized log probabilities required; no weight floors')
    weights = np.exp(logweights - special.logsumexp(logweights))
    s0 = float(law['native_first_scale_s0'])
    if not math.isfinite(s0) or s0 <= 0:
        raise ValueError('positive finite fixed native scale required')
    for component in law['components']:
        if (not all(math.isfinite(component[k]) for k in ('location', 'scale', 'df'))
                or component['scale'] <= 0 or component['df'] <= 2):
            raise ValueError('finite nondegenerate first-moment-safe Student components required')
    return weights, s0


def yj_scalar(native, s0):
    value = native / s0
    if value >= 0:
        return 2 * value / (math.sqrt(1 + value) + 1)
    return -math.expm1(1.5 * math.log1p(-value)) / 1.5


def inverse_yj_scalar(u, s0):
    value = u * (1 + u / 4) if u >= 0 else -math.expm1((2 / 3) * math.log1p(-1.5 * u))
    return s0 * value


def cdf_components(law, u, survival=False):
    return np.asarray([special.stdtr(c['df'], (-(u - c['location']) if survival else u - c['location']) / c['scale'])
                       for c in law['components']])


def quantile_native(law, probability):
    weights, s0 = validate(law)
    if not 0 < probability < 1:
        raise ValueError('interior predictive quantile required')
    brackets = [c['location'] + c['scale'] * float(stats.t.ppf(probability, c['df']))
                for c in law['components']]
    lower, upper = min(brackets), max(brackets)
    if lower == upper:
        u = lower
    else:
        survival = probability > .5
        target = 1 - probability if survival else probability
        def difference(value):
            return float(weights @ cdf_components(law, value, survival)) - target
        # A mixture quantile lies between the component quantiles. This remains
        # true for zero display weights; no weighted average of quantiles is used.
        low_value, high_value = difference(lower), difference(upper)
        if low_value == 0:
            u = lower
        elif high_value == 0:
            u = upper
        elif low_value * high_value > 0:
            # An endpoint may round just past the theoretical bracket. Expand
            # deterministically, never change probabilities or predictive scales.
            width = max(upper - lower, max(c['scale'] for c in law['components']))
            for _ in range(8):
                lower -= width; upper += width; width *= 2
                if difference(lower) * difference(upper) <= 0: break
            else: raise ArithmeticError('mixture quantile bracket failure')
            u = float(optimize.brentq(difference, lower, upper, xtol=1e-12, rtol=8*np.finfo(float).eps))
        else:
            u = float(optimize.brentq(difference, lower, upper, xtol=1e-12, rtol=8*np.finfo(float).eps))
    return inverse_yj_scalar(u, s0)


def cdf_disagreement_integral(law):
    weights, s0 = validate(law)
    lower = min(c['location'] - 32*c['scale'] for c in law['components'])
    upper = max(c['location'] + 32*c['scale'] for c in law['components'])
    split = sorted(set([lower, upper] + [v for v in [0.] + [c['location'] for c in law['components']]
                                        if lower < v < upper]))
    values, errors = [], []
    def integrand(u):
        components = cdf_components(law, u)
        if float(weights @ components) >= .5:
            components = cdf_components(law, u, survival=True)
        mixture = float(weights @ components)
        variance = float(weights @ ((components - mixture)**2))
        derivative = 1 + u/2 if u >= 0 else math.exp(-math.log1p(-1.5*u)/3)
        return s0 * derivative * variance
    with warnings.catch_warnings():
        warnings.simplefilter('error', integrate.IntegrationWarning)
        for a, b in zip(split[:-1], split[1:]):
            value, error = integrate.quad(integrand, a, b, epsabs=2e-9, epsrel=2e-10, limit=250)
            values.append(value); errors.append(error)
    tails = []
    # Var_H(F_h) <= F_mix(1-F_mix) <= the relevant one-sided mixture tail.
    # The union endpoints extend beyond each component's own +/-32 scale cut.
    for w, component in zip(weights, law['components']):
        nu, scale, mu = component['df'], component['scale'], component['location']
        pdf, sf = float(stats.t.pdf(32., nu)), float(special.stdtr(nu, -32.))
        i0 = (nu + 32**2)/(nu-1)*pdf
        i1 = .5*(nu*sf + 32*(nu + 32**2)*pdf)/(nu-2)
        tails.append(float(w)*2*s0*scale*((1+.5*abs(mu))*i0 + .5*scale*i1))
    value = math.fsum(values)
    if not math.isfinite(value) or value < 0:
        raise ArithmeticError('nonnegative finite predictive CDF disagreement required')
    return value, {'finite_U_window': [lower, upper], 'adaptive_quad_error_estimate': math.fsum(errors),
                   'outer_tail_bound_ideal_arithmetic': math.fsum(tails),
                   'not_a_directed_rounding_or_CDF_error_certificate': True}


def score_law(native_target, law, scalar_score):
    """Use actual truth only in the evaluator after global forecast freeze.

    scalar_score(y, component_law) is the immutable parent single-law scorer.
    It is deliberately injected, so this pure module cannot find a dataset.
    """
    weights, s0 = validate(law)
    y = float(native_target)
    if not math.isfinite(y): raise ValueError('finite actual evaluator outcome required')
    component_scores = []
    for component in law['components']:
        scalar = {'coordinate': law['coordinate'], 'family': 'student_t', 'location': component['location'],
                  'scale': component['scale'], 'df': component['df'], 'native_first_scale_s0': s0,
                  'lambda': .5, 'unit': law.get('unit')}
        component_scores.append(scalar_score(y, scalar))
    disagreement, diagnostics = cdf_disagreement_integral(law)
    weighted = math.fsum(float(w)*r['CRPS_native'] for w, r in zip(weights, component_scores))
    crps = weighted - disagreement
    if not math.isfinite(crps) or crps < 0:
        raise ArithmeticError('mixture CRPS must be finite/nonnegative; retain failure rather than clip')
    u = yj_scalar(y, s0)
    cdf = float(weights @ cdf_components(law, u))
    sf = float(weights @ cdf_components(law, u, survival=True))
    median = quantile_native(law, .5)
    intervals = {}
    for level in (.8, .9, .95):
        a, b = quantile_native(law, (1-level)/2), quantile_native(law, (1+level)/2)
        intervals[str(level)] = {'lower': a, 'upper': b, 'width': b-a, 'covered': bool(a <= y <= b)}
    value = y/s0
    jacobian = -math.log(s0) + (-.5*math.log1p(value) if value >= 0 else .5*math.log1p(-value))
    logdensity = []
    for component in law['components']:
        nu, scale = component['df'], component['scale']
        z = (u-component['location'])/scale
        logdensity.append(special.gammaln((nu+1)/2)-special.gammaln(nu/2)-.5*math.log(nu*math.pi)
                          -math.log(scale)-(nu+1)/2*math.log1p(z*z/nu))
    normalized_logs = np.asarray(law['log_weights']) - special.logsumexp(law['log_weights'])
    logdensity = float(special.logsumexp(normalized_logs + np.asarray(logdensity))) + jacobian
    return {'CRPS_native': crps, 'squared_error_native': (median-y)**2,
            'predictive_median_native': median, 'intervals': intervals,
            'PIT_left': cdf, 'PIT_right': cdf, 'PIT_midpoint': cdf, 'PIT_kind': 'continuous_mixture',
            'log_density_native': logdensity, 'negative_log_density_native': -logdensity,
            'native_CDF_plus_SF_error': abs(cdf+sf-1),
            'mixture_CRPS_decomposition': {'weighted_component_CRPS': weighted,
                'subtract_CDF_disagreement': disagreement, 'disagreement_numerics': diagnostics,
                'component_scores': component_scores,
                'mixture_quantiles_are_roots_not_averaged': True}}

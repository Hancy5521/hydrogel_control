"""Independent native-coordinate quadrature for fixed finite-df risk fixtures."""
import math
from scipy import special
from scipy.integrate import quad


def yj(v):
    return 2 * v / (math.sqrt(1 + v) + 1) if v >= 0 else -math.expm1(1.5 * math.log1p(-v)) / 1.5


def inv_yj(u):
    return u * (1 + u / 4) if u >= 0 else -math.expm1((2 / 3) * math.log1p(-1.5 * u))


def log_forward_derivative(v):
    return -.5 * math.log1p(v) if v >= 0 else .5 * math.log1p(-v)


def stdcdf(x, family, nu):
    return float(special.ndtr(x)) if family == 'gaussian' else float(special.stdtr(nu, x))


def stdpdf(x, family, nu):
    if family == 'gaussian':
        return math.exp(-.5 * x * x) / math.sqrt(2 * math.pi)
    logc = math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2) - .5 * math.log(nu * math.pi)
    return math.exp(logc - .5 * (nu + 1) * math.log1p(x * x / nu))


def affine_t_value(nu, rho, s, s0):
    logC = math.log(2) + .5 * math.log(nu) - math.log(nu - 1) + float(special.betaln(.5, nu - .5)) - 2 * float(special.betaln(.5, nu / 2))
    other = nu + 1
    logNext = math.log(2) + .5 * math.log(other) - math.log(other - 1) + float(special.betaln(.5, other - .5)) - 2 * float(special.betaln(.5, other / 2))
    logA = .5 * (math.log(nu) - math.log(nu + 1)) + math.lgamma((nu + 1) / 2) + math.lgamma((nu - 1) / 2) - 2 * math.lgamma(nu / 2)
    return s0 * s * (math.exp(logC) - math.sqrt(1 - rho * rho) * math.exp(logNext + logA))


def gaussian_positive_kernel(v, rho):
    if rho == 0:
        return 0.
    angle = math.asin(rho * rho)
    value, error = quad(lambda t: math.exp(-v * v / (1 + math.sin(t))) / (2 * math.pi),
                        0., angle, epsabs=2e-13, epsrel=2e-11, limit=200)
    return value


def gaussian_owen_kernel(v, rho):
    if rho == 0:
        return 0.
    return float(special.ndtr(v) * special.ndtr(-v) - 2 * special.owens_t(v, math.sqrt((1 - rho * rho) / (1 + rho * rho))))


def native_double_quad(fixture, Lv=32., Lz=64.):
    """Finite native target x and direct standardized-query z original-density quad.

    This changes only the inner reference coordinate from the failed native-q
    implementation. It is independent of producer theta/GL and uses no inverse
    Jacobian in the target coordinate. No truncated law is renormalized.
    """
    family, nu, rho = fixture['family'], fixture['nu'], fixture['rho']
    mu, s, s0 = fixture['mu_target'], fixture['scale_target'], fixture['s0']
    if family == 'gaussian' and rho == 0:
        return {'value': 0., 'method': 'exact independent Gaussian CDF',
                'outer_calls': 0, 'inner_calls': 0, 'quad_error_estimate': 0.}
    if fixture['target_map'] == 'affine':
        return {'value': affine_t_value(nu, rho, s, s0), 'method': 'closed-form affine Student-t Bayes-risk difference, full support',
                'outer_calls': 0, 'inner_calls': 0, 'quad_error_estimate': 0.}
    if family not in ('gaussian', 'student_t') or not abs(rho) < 1 or min(s, s0) <= 0:
        raise ValueError('Finite positive-scale nondegenerate fixture required')
    ylo, yhi = s0 * inv_yj(mu - s * Lv), s0 * inv_yj(mu + s * Lv)
    ycuts = [ylo] + sorted({v for v in (0., s0 * inv_yj(mu)) if ylo < v < yhi}) + [yhi]
    zcuts = [-Lz] + [v for v in (-16., -8., -4., -2., 0., 2., 4., 8., 16.) if -Lz < v < Lz] + [Lz]
    inner_calls, outer_calls, max_inner_error = 0, 0, 0.
    def outer(y):
        nonlocal inner_calls, outer_calls, max_inner_error
        outer_calls += 1
        v = (yj(y / s0) - mu) / s
        use_sf = v >= 0
        marginal = stdcdf(-v if use_sf else v, family, nu)
        def inner(z):
            nonlocal inner_calls
            inner_calls += 1
            den = math.sqrt(1 - rho * rho)
            if family == 'student_t':
                den *= math.sqrt((nu + z * z) / (nu + 1))
                condnu = nu + 1
            else:
                condnu = nu
            arg = (v - rho * z) / den
            prob = stdcdf(-arg if use_sf else arg, family, condnu)
            return (prob - marginal) ** 2 * stdpdf(z, family, nu)
        vals, errs = [], []
        for lower, upper in zip(zcuts[:-1], zcuts[1:]):
            value, error = quad(inner, lower, upper, epsabs=2e-10, epsrel=2e-7, limit=300)
            vals.append(value); errs.append(error)
        max_inner_error = max(max_inner_error, math.fsum(errs))
        return math.fsum(vals)
    values, errors = [], []
    for lower, upper in zip(ycuts[:-1], ycuts[1:]):
        value, error = quad(outer, lower, upper, epsabs=2e-8, epsrel=2e-7, limit=300)
        values.append(value); errors.append(error)
    return {'value': math.fsum(values), 'method': 'finite target-native-x and direct standardized-query-z original-density adaptive quad',
            'outer_calls': outer_calls, 'inner_calls': inner_calls,
            'query_domain': [-Lz, Lz], 'target_standardized_domain': [-Lv, Lv], 'query_weights_renormalized': False,
            'quad_error_estimate': math.fsum(errors), 'max_sampled_inner_quad_error_estimate': max_inner_error,
            'native_target_window_width': yhi - ylo,
            'inner_error_sample_based_width_diagnostic_not_rigorous_bound': (yhi - ylo) * max_inner_error,
            'not_a_global_numerical_or_directed_rounding_certificate': True}


def independent_tail_integrals(nu, L):
    i0, e0 = quad(lambda v: stdcdf(-v, 'student_t', nu), L, math.inf, epsabs=2e-12, epsrel=2e-10, limit=300)
    i1, e1 = quad(lambda v: v * stdcdf(-v, 'student_t', nu), L, math.inf, epsabs=2e-12, epsrel=2e-10, limit=300)
    return {'I0': i0, 'I1': i1, 'error_estimates': [e0, e1]}

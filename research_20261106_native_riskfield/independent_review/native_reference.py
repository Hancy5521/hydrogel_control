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
    """Integrate in true target/query native x coordinates, no inverse Jacobian in target.

    The query density includes its forward-YJ Jacobian. Probability weights are
    the original law and are never renormalized after truncation.
    """
    family, nu, rho = fixture['family'], fixture['nu'], fixture['rho']
    mu, s, s0 = fixture['mu_target'], fixture['scale_target'], fixture['s0']
    mq, sq = fixture['mu_query'], fixture['scale_query']
    if family == 'gaussian' and rho == 0:
        return {'value': 0., 'method': 'exact independent Gaussian conditional CDF, any monotone target map',
                'outer_calls': 0, 'inner_calls': 0, 'quad_error_estimate': 0.}
    if fixture['target_map'] == 'affine':
        return {'value': affine_t_value(nu, rho, s, s0), 'method': 'closed-form affine Student-t Bayes-risk difference, full support',
                'outer_calls': 0, 'inner_calls': 0, 'quad_error_estimate': 0.}
    if family not in ('gaussian', 'student_t') or not abs(rho) < 1 or min(s, s0, sq) <= 0:
        raise ValueError('Supported finite positive scale nondegenerate fixture required')
    qlo, qhi = s0 * inv_yj(mq - sq * Lz), s0 * inv_yj(mq + sq * Lz)
    ylo, yhi = s0 * inv_yj(mu - s * Lv), s0 * inv_yj(mu + s * Lv)
    qcuts = [qlo] + sorted({v for v in (0., s0 * inv_yj(mq)) if qlo < v < qhi}) + [qhi]
    ycuts = [ylo] + sorted({v for v in (0., s0 * inv_yj(mu)) if ylo < v < yhi}) + [yhi]
    inner_calls, outer_calls, max_inner_error = 0, 0, 0.
    def outer(y):
        nonlocal inner_calls, outer_calls, max_inner_error
        outer_calls += 1
        v = (yj(y / s0) - mu) / s
        use_sf = v >= 0
        marginal = stdcdf(-v if use_sf else v, family, nu)
        def inner(q):
            nonlocal inner_calls
            inner_calls += 1
            z = (yj(q / s0) - mq) / sq
            den = math.sqrt(1 - rho * rho)
            if family == 'student_t':
                den *= math.sqrt((nu + z * z) / (nu + 1))
                condnu = nu + 1
            else:
                condnu = nu
            arg = (v - rho * z) / den
            prob = stdcdf(-arg if use_sf else arg, family, condnu)
            native_query_density = stdpdf(z, family, nu) * math.exp(log_forward_derivative(q / s0)) / (s0 * sq)
            return (prob - marginal) ** 2 * native_query_density
        vals, errs = [], []
        for lower, upper in zip(qcuts[:-1], qcuts[1:]):
            value, error = quad(inner, lower, upper, epsabs=2e-10, epsrel=2e-7, limit=300)
            vals.append(value); errs.append(error)
        max_inner_error = max(max_inner_error, math.fsum(errs))
        return math.fsum(vals)
    values, errors = [], []
    for lower, upper in zip(ycuts[:-1], ycuts[1:]):
        value, error = quad(outer, lower, upper, epsabs=2e-8, epsrel=2e-7, limit=300)
        values.append(value); errors.append(error)
    return {'value': math.fsum(values), 'method': 'target-native-x/query-native-x adaptive quad with forwardYJ and original query density',
            'outer_calls': outer_calls, 'inner_calls': inner_calls,
            'quad_error_estimate': math.fsum(errors), 'max_sampled_inner_quad_error_estimate': max_inner_error,
            'native_target_window_width': yhi - ylo,
            'inner_error_sample_based_width_diagnostic_not_rigorous_bound': (yhi - ylo) * max_inner_error,
            'not_a_global_numerical_or_directed_rounding_certificate': True}


def independent_tail_integrals(nu, L):
    i0, e0 = quad(lambda v: stdcdf(-v, 'student_t', nu), L, math.inf, epsabs=2e-12, epsrel=2e-10, limit=300)
    i1, e1 = quad(lambda v: v * stdcdf(-v, 'student_t', nu), L, math.inf, epsabs=2e-12, epsrel=2e-10, limit=300)
    return {'I0': i0, 'I1': i1, 'error_estimates': [e0, e1]}

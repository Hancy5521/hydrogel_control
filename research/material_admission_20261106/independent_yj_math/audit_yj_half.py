"""Fixed synthetic YJ-half mathematics; no producer/material/random access."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
import sys
import traceback
import warnings

NORMAL = NormalDist()
CONFIGS = ({'s0': 2.5, 'a': .4, 'b': .8, 'loc': .5, 'scale': .6},
           {'s0': .7, 'a': -.9, 'b': 1.3, 'loc': -.5, 'scale': 1.1})
DFS = (None, 5.0, 69.0, 71.0)
TARGETS = (-3.2, 2.4)
PROBABILITIES = (1e-6, .025, .5, .975, 1 - 1e-6)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text('utf-8-sig'))


def require(condition, text):
    if not condition:
        raise AssertionError(text)


def check(actual, expected, atol=2e-10, rtol=2e-10):
    require(math.isfinite(actual) and math.isfinite(expected), 'Finite check values required')
    error = abs(actual - expected)
    require(error <= atol + rtol * abs(expected),
            f'actual={actual}, expected={expected}, error={error}, tolerance={atol + rtol * abs(expected)}')
    return error


def transform(v):
    # Algebraically the fixed Yeo-Johnson lambda=1/2 formula; stable near0.
    if v >= 0:
        return 2 * v / (math.sqrt(1 + v) + 1)
    return -math.expm1(1.5 * math.log1p(-v)) / 1.5


def inverse(u):
    if u >= 0:
        return u + u * u / 4
    return -math.expm1((2 / 3) * math.log1p(-1.5 * u))


def derivative(v):
    return (1 + v) ** -.5 if v >= 0 else (1 - v) ** .5


def inverse_derivative(u):
    return 1 + u / 2 if u >= 0 else math.exp(-math.log1p(-1.5 * u) / 3)


def native(z, c):
    return c['s0'] * inverse(c['a'] + c['b'] * z)


def latent(x, c):
    return (transform(x / c['s0']) - c['a']) / c['b']


def jacobian(z, c):
    return c['s0'] * c['b'] * inverse_derivative(c['a'] + c['b'] * z)


def standardized(z, c):
    return (z - c['loc']) / c['scale']


def cdf(z, c, df):
    q = standardized(z, c)
    return .5 * math.erfc(-q / math.sqrt(2)) if df is None else float(special.stdtr(df, q))


def sf(z, c, df):
    q = standardized(z, c)
    return .5 * math.erfc(q / math.sqrt(2)) if df is None else float(special.stdtr(df, -q))


def logpdf(z, c, df):
    q = standardized(z, c)
    if df is None:
        return -.5 * q * q - .5 * math.log(2 * math.pi) - math.log(c['scale'])
    return (math.lgamma((df + 1) / 2) - math.lgamma(df / 2)
            - .5 * math.log(df * math.pi) - math.log(c['scale'])
            - (df + 1) / 2 * math.log1p(q * q / df))


def quantile(p, c, df):
    q = NORMAL.inv_cdf(p) if df is None else float(special.stdtrit(df, p))
    return c['loc'] + c['scale'] * q


def log_native_pdf(x, c, df):
    v = x / c['s0']
    logj = -.5 * math.log1p(v) if v >= 0 else .5 * math.log1p(-v)
    return logpdf(latent(x, c), c, df) + logj - math.log(c['s0']) - math.log(c['b'])


def integrate(fn, lower, upper, cuts=()):
    points = [lower] + sorted({x for x in cuts if lower < x < upper}) + [upper]
    values, errors = [], []
    for lo, hi in zip(points[:-1], points[1:]):
        value, error = quad(fn, lo, hi, epsabs=2e-9, epsrel=2e-9, limit=400)
        require(math.isfinite(value) and math.isfinite(error), 'Finite quadrature required')
        values.append(value)
        errors.append(error)
    return math.fsum(values), math.fsum(errors)


def transformations():
    points = (-1e6, -3.5, -1e-8, -1e-12, 0., 1e-12, 1e-8, 2.5, 1e6)
    rows = []
    for v in points:
        u = transform(v)
        scipy_u = float(stats.yeojohnson([v], lmbda=.5)[0])
        e1 = check(inverse(u), v, 2e-10, 2e-11)
        e2 = check(u, scipy_u, 2e-10, 2e-11)
        e3 = check(derivative(v) * inverse_derivative(u), 1., 2e-11, 2e-11)
        require((v == 0 and u == 0) or (v * u > 0), 'Sign/zero preservation failed')
        rows.append({'v': v, 'u': u, 'roundtrip_error': e1,
                     'official_SciPy_formula_error': e2, 'inverse_derivative_product_error': e3})
    derivative_rows = []
    for v in (-4., -.1, 0., .2, 3.):
        step = 1e-5 * (1 + abs(v))
        numerical = (transform(v + step) - transform(v - step)) / (2 * step)
        error = check(numerical, derivative(v), 5e-8, 5e-8)
        gi = (inverse(v + step) - inverse(v - step)) / (2 * step)
        inverse_error = check(gi, inverse_derivative(v), 5e-8, 5e-8)
        derivative_rows.append({'coordinate': v, 'step': step, 'forward_error': error,
                                'inverse_error': inverse_error})
    return {'roundtrip_coordinates': len(rows), 'rows': rows,
            'finite_difference_coordinates': len(derivative_rows), 'derivatives': derivative_rows}


def probabilities_and_density():
    quantiles, norms, derivative_rows = [], [], []
    for config_index, c in enumerate(CONFIGS):
        for df in DFS:
            for p in PROBABILITIES:
                z = quantile(p, c, df)
                x = native(z, c)
                e = check(cdf(latent(x, c), c, df), p, 5e-10, 5e-10)
                quantiles.append({'configuration': config_index, 'df': df, 'p': p,
                                  'native_quantile': x, 'CDF_inverse_error': e})
            norm, error = integrate(lambda x: math.exp(log_native_pdf(x, c, df)),
                                     -math.inf, math.inf, (0.,))
            check(norm, 1., 5e-8, 5e-8)
            norms.append({'configuration': config_index, 'df': df,
                          'native_density_integral': norm, 'reported_quad_error': error})
            for x in TARGETS:
                step = 1e-5 * (1 + abs(x))
                numerical = (cdf(latent(x + step, c), c, df)
                             - cdf(latent(x - step, c), c, df)) / (2 * step)
                error = check(numerical, math.exp(log_native_pdf(x, c, df)), 2e-8, 2e-7)
                derivative_rows.append({'configuration': config_index, 'df': df,
                                        'native_x': x, 'CDF_derivative_density_error': error})
                for level in (.8, .9, .95):
                    zl = quantile((1 - level) / 2, c, df)
                    zu = quantile((1 + level) / 2, c, df)
                    require((zl <= latent(x, c) <= zu)
                            == (native(zl, c) <= x <= native(zu, c)),
                            'Monotone interval event changed')
    return {'quantile_coordinates': len(quantiles), 'quantiles': quantiles,
            'density_normalizations': len(norms), 'normalizations': norms,
            'CDF_density_derivative_coordinates': len(derivative_rows), 'density_derivatives': derivative_rows,
            'synthetic_interval_event_checks': 48,
            'PIT_coverage_interpretation': 'Same frozen probability law: monotone coordinate change preserves PIT and interval events, not a calibration repair.'}


def crps_two_coordinates():
    rows = []
    for config_index, c in enumerate(CONFIGS):
        z0 = -c['a'] / c['b']
        for df in DFS:
            for y in TARGETS:
                zy = latent(y, c)
                nl, nle = integrate(lambda x: cdf(latent(x, c), c, df) ** 2,
                                     -math.inf, y, (0.,))
                nr, nre = integrate(lambda x: sf(latent(x, c), c, df) ** 2,
                                     y, math.inf, (0.,))
                ll, lle = integrate(lambda z: cdf(z, c, df) ** 2 * jacobian(z, c),
                                     -math.inf, zy, (z0,))
                lr, lre = integrate(lambda z: sf(z, c, df) ** 2 * jacobian(z, c),
                                     zy, math.inf, (z0,))
                native_value, latent_weighted = nl + nr, ll + lr
                error = check(native_value, latent_weighted, 2e-7, 2e-8)
                require(min(native_value, latent_weighted) >= 0, 'Nonnegative proper score required')
                rows.append({'configuration': config_index, 'df': df, 'native_target': y,
                             'native_x_CDF_integral': native_value,
                             'latent_z_weighted_CDF_integral': latent_weighted,
                             'absolute_error': error, 'native_reported_quad_error': nle + nre,
                             'latent_reported_quad_error': lle + lre})
    return {'fixed_cases': len(rows), 'rows': rows,
            'scope': 'Two independent coordinate quadratures of same exact CDF, not independent models or material measurements. Error estimates are not outward-rounded certificates.'}


def moments_median_atoms():
    existence = []
    for df in (1., 2., 4., 5., 69., 71.):
        existence.append({'df': df, 'native_first_absolute_moment_by_tail_theory': df > 2,
                           'native_second_moment_by_tail_theory': df > 4})
    require(existence[1]['native_first_absolute_moment_by_tail_theory'] is False
            and existence[2]['native_second_moment_by_tail_theory'] is False,
            'Strict moment thresholds required')
    rows = []
    for config_index, c in enumerate(CONFIGS):
        z0 = -c['a'] / c['b']
        for df in DFS:
            mean, me = integrate(lambda z: native(z, c) * math.exp(logpdf(z, c, df)),
                                  -math.inf, math.inf, (z0,))
            absolute, ae = integrate(lambda z: abs(native(z, c)) * math.exp(logpdf(z, c, df)),
                                      -math.inf, math.inf, (z0,))
            second, se = integrate(lambda z: native(z, c) ** 2 * math.exp(logpdf(z, c, df)),
                                    -math.inf, math.inf, (z0,))
            median = native(c['loc'], c)
            require(mean > median + 1e-5 and absolute >= abs(mean) - 5e-8
                    and second > mean * mean, 'Finite moment/Jensen/median control failed')
            check(cdf(latent(median, c), c, df), .5, 2e-10, 2e-10)
            rows.append({'configuration': config_index, 'df': df, 'native_mean_density_integral': mean,
                         'native_median_inverse_latent_location': median,
                         'strict_Jensen_mean_minus_median': mean - median,
                         'native_absolute_first_moment_integral': absolute,
                         'native_second_moment_integral': second,
                         'reported_quad_errors': [me, ae, se]})
    values, masses = (-1., 0., 2.), (.2, .5, .3)
    pushforward_zero_mass = math.fsum(p for x, p in zip(values, masses) if transform(x) == 0.)
    require(pushforward_zero_mass == .5, 'Bijection must preserve point mass at0')
    return {'tail_moment_thresholds': existence, 'finite_moment_laws_integrated': len(rows),
            'moments': rows, 'point_mass_at_zero_before': .5, 'point_mass_at_zero_after': pushforward_zero_mass,
            'important_limit': 'Moment existence proved by asymptotic polynomial tails, not by finite quadrature. Integrals only crosscheck finite df5/69/71 and Normal cases. Monotone YJ does not remove atoms; inverse location is a median, not a raw mean.'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', required=True)
    ap.add_argument('--approval', required=True)
    ap.add_argument('--output-dir', required=True)
    args = ap.parse_args()
    plan, approval = read(args.plan), read(args.approval)
    source = sha(__file__)
    require(plan['script_sha256'] == source, 'Frozen audit source changed')
    require(approval.get('approved') is True and approval['scope'] == 'INDEPENDENT_YJ_HALF_MATH_ONCE'
            and approval['plan_sha256'] == sha(args.plan)
            and approval['script_sha256'] == source, 'Exact source-and-plan approval required')
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    global special, stats, quad
    import scipy
    from scipy import special, stats
    from scipy.integrate import quad, IntegrationWarning
    warnings.simplefilter('error', IntegrationWarning)
    results = []
    for name, function in (('roundtrip_derivative_and_primary_formula', transformations),
                           ('CDF_quantiles_intervals_and_density_Jacobian', probabilities_and_density),
                           ('native_CRPS_two_coordinate_integrals', crps_two_coordinates),
                           ('finite_moments_median_and_atom_preservation', moments_median_atoms)):
        try:
            results.append({'name': name, 'status': 'PASS', 'details': function()})
        except Exception as error:
            results.append({'name': name, 'status': 'FAIL', 'type': type(error).__name__,
                            'message': str(error), 'traceback': traceback.format_exc()})
    payload = {'status': 'PASS' if all(x['status'] == 'PASS' for x in results) else 'FAIL',
               'start_utc': started, 'end_utc': datetime.now(timezone.utc).isoformat(),
               'script_sha256': source, 'plan_sha256': sha(args.plan), 'approval_sha256': sha(args.approval),
               'scipy': scipy.__version__, 'Python': sys.version, 'families': results,
               'native_material_reads': 0, 'producer_imports_or_executions': 0,
               'random_samples': 0, 'scope': 'Fixed hypothetical mathematical audit only; no new material model or cohort evidence, no calibration guarantee or numerical interval proof.'}
    target = out / 'INDEPENDENT_YJ_HALF_MATH_AUDIT.json'
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'status': payload['status'], 'result': str(target), 'sha256': sha(target),
                      'families': [{'name': x['name'], 'status': x['status']} for x in results]}))
    return 0 if payload['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())

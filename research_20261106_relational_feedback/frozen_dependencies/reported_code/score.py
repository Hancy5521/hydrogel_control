"""Freeze-gated native-unit scoring of new reported-feedback development data.

The YJ numerical integration helpers retain the prior audited formula and bounds
(phase2 score source0160ad1..., selected functions only), with local puretransform.
Empirical scores use positive stepCDF sums; independent audit uses another route.
"""
import argparse
from fractions import Fraction
import json
import math
from pathlib import Path
import warnings
import numpy as np
from scipy import integrate, special, stats
import data_scope as task
import joint_model as model

def standard_distribution(law):
    if law['family'] == 'normal' and law['df'] is None:
        return stats.norm()
    if law['family'] == 'student_t' and math.isfinite(law['df']) and law['df'] > 2:
        return stats.t(df=law['df'])
    raise ValueError('Nondegenerate Gaussian or t with native first-moment-safe df>2 required')

def tail_logs(law, cut):
    """Ideal-arithmetic analytic tail bounds, displayed in log domain; not IEEE intervals."""
    mu, scale, s0 = law['location'], law['scale'], law['native_first_scale_s0']
    if not all(math.isfinite(x) for x in (mu, scale, s0, cut)) or min(scale, s0, cut) <= 0 or cut < abs(mu) / scale:
        raise ValueError('Tail cut must extend beyond inverse branch switch')
    a, b = 1 + .5 * abs(mu), .5 * scale
    log_pre, log_l = math.log(s0) + math.log(scale), math.log(cut)
    if law['family'] == 'normal':
        common = log_pre - math.log(4 * math.pi) - cut * cut
        left = common - 3 * log_l
        right = common + float(np.logaddexp(math.log(a) - 3 * log_l, math.log(b) - 2 * log_l))
    else:
        nu = law['df']
        if not math.isfinite(nu) or nu <= 2:
            raise ValueError('Native t first absolute moment must exist')
        log_c = special.gammaln((nu + 1) / 2) - .5 * math.log(nu * math.pi) - special.gammaln(nu / 2)
        log_C = log_c + .5 * (nu - 1) * math.log(nu)
        common = log_pre + 2 * log_C
        left = common + (1 - 2 * nu) * log_l - math.log(2 * nu - 1)
        right = common + float(np.logaddexp(math.log(a) + (1 - 2 * nu) * log_l - math.log(2 * nu - 1),
                    math.log(b) + (2 - 2 * nu) * log_l - math.log(2 * nu - 2)))
    if not all(math.isfinite(x) for x in (left, right)):
        raise ArithmeticError('Tail log bound is not finite')
    return float(left), float(right)

def native_crps_integral(y, law, settings=None):
    cfg = task.configuration() if settings is None else settings
    if law['coordinate'] != 'u=YJ_half(native/s0)' or law['lambda'] != .5:
        raise ValueError('Only fixed YJ-half law uses the transformed integral')
    mu, scale, s0 = law['location'], law['scale'], law['native_first_scale_s0']
    if not all(math.isfinite(v) for v in (y, mu, scale, s0)) or scale <= 0 or s0 <= 0:
        raise ValueError('Finite target/law and positive scales required')
    distribution = standard_distribution(law)
    target_t = (float(model.yj(y / s0)) - mu) / scale
    branch_t = -mu / scale
    if not math.isfinite(target_t) or not math.isfinite(branch_t):
        raise ArithmeticError('Nonfinite transformed target/switch')
    cut = max(cfg['tail_initial_standard_cut'], abs(target_t) + 1, abs(branch_t) + 1)
    budget = cfg['score_absolute_tolerance_native'] * cfg['tail_each_absolute_budget_fraction']
    log_budget = math.log(budget)
    refinements = 0
    while True:
        if cut > cfg['tail_max_standard_cut']:
            raise ArithmeticError('Required tail cut exceeds preregistered numerical range')
        left_log, right_log = tail_logs(law, cut)
        if max(left_log, right_log) <= log_budget:
            break
        refinements += 1
        if refinements > cfg['tail_max_doublings']:
            raise ArithmeticError('Tail envelope failed within fixed refinement cap')
        cut *= 2
    points = sorted(set([-cut, cut, target_t, branch_t] + [v for v in cfg['fixed_integral_subdivision_points'] if -cut < v < cut]))
    pieces = list(zip(points[:-1], points[1:]))
    abs_piece = cfg['score_absolute_tolerance_native'] / (4 * len(pieces))
    rel_piece = cfg['score_relative_tolerance'] / (4 * len(pieces))
    total, total_error, details = 0., 0., []
    for lower, upper in pieces:
        use_cdf = (lower + upper) / 2 <= target_t
        def integrand(t):
            probability = float(distribution.cdf(t) if use_cdf else distribution.sf(t))
            u = mu + scale * t
            gprime = 1 + .5 * u if u >= 0 else math.exp(-math.log1p(-1.5 * u) / 3)
            value = probability * probability * s0 * scale * gprime
            if not math.isfinite(value) or value < 0:
                raise ArithmeticError('Invalid nonnegative native CRPS integrand')
            return value
        with warnings.catch_warnings():
            warnings.simplefilter('error', integrate.IntegrationWarning)
            value, error = integrate.quad(integrand, lower, upper, epsabs=abs_piece, epsrel=rel_piece,
                        limit=cfg['quad_subinterval_limit'])
        if not math.isfinite(value) or not math.isfinite(error) or value < 0 or error < 0:
            raise ArithmeticError('Quadrature did not yield finite nonnegative score/error')
        total += value
        total_error += error
        details.append({'lower_standard_t': lower, 'upper_standard_t': upper,
                'probability_function': 'CDF' if use_cdf else 'SF', 'integral_native': value, 'quaderr_estimate_native': error})
    tail_sum_log = float(np.logaddexp(left_log, right_log))
    tail_display = math.exp(tail_sum_log)
    tolerance = cfg['score_absolute_tolerance_native'] + cfg['score_relative_tolerance'] * abs(total)
    if total_error + tail_display > tolerance or not math.isfinite(total):
        raise ArithmeticError('Combined quadrature estimate and analytic-tail display exceed fixed score tolerance')
    return total, {'method': 'native CDF_SF integral in standard latent coordinate with inverse Jacobian',
            'quadrature_error_estimate_native': total_error, 'analytic_tail_total_log_bound_ideal_arithmetic': tail_sum_log,
            'analytic_tail_total_display_native': tail_display,
            'tail_display_underflow_not_zero_mathematical_tail': tail_display == 0,
            'tail_each_log_bound': [left_log, right_log], 'tail_cut_standard_t': cut,
            'tail_doublings': refinements, 'actual_quadrature_segments': details,
            'score_absolute_plus_relative_tolerance_native': tolerance,
            'score_return_is_truncated_integral_with_declared_tail_error': True,
            'not_a_directed_IEEE_or_CDF_library_error_certificate': True}

def empirical_score(y, law):
    masses = {}
    if len(law['support_native']) != len(law['weight_fractions']):
        raise ValueError('empirical support and masses disagree')
    for support, value in zip(law['support_native'], law['weight_fractions']):
        mass = Fraction(value)
        if not math.isfinite(support) or mass < 0:
            raise ValueError('finite support and nonnegative exact masses required')
        masses[support] = masses.get(support, Fraction(0)) + mass
    if sum(masses.values(), Fraction(0)) != 1:
        raise ValueError('exact empirical simplex required')
    points = sorted(masses)
    cuts = sorted(set(points + [float(y)]))
    cumulative, crps = Fraction(0), 0.
    for lower, upper in zip(cuts[:-1], cuts[1:]):
        cumulative += masses.get(lower, Fraction(0))
        error = cumulative - int(lower >= y)
        crps += float(error * error) * (upper - lower)
    def quantile(probability):
        cumulative = Fraction(0)
        for value in points:
            cumulative += masses[value]
            if cumulative >= probability:
                return value
        raise AssertionError('complete positive empirical mass required')
    left = sum((masses[x] for x in points if x < y), Fraction(0))
    right = sum((masses[x] for x in points if x <= y), Fraction(0))
    median = quantile(Fraction(1, 2))
    intervals = {}
    for level in ('0.8', '0.9', '0.95'):
        coverage = Fraction(level)
        lower, upper = quantile((1 - coverage) / 2), quantile((1 + coverage) / 2)
        intervals[level] = {'lower': lower, 'upper': upper, 'width': upper - lower,
                            'covered': bool(lower <= y <= upper)}
    return {'CRPS_native': crps, 'predictive_median_native': median, 'squared_error_native': (y - median) ** 2,
        'PIT_left': float(left), 'PIT_right': float(right), 'PIT_midpoint': float((left + right) / 2),
        'PIT_kind': 'discreteinterval, midpoint not continuous uniform PIT',
        'intervals': intervals, 'log_density_native': None,
        'score_numerics': {'method': 'positive empirical stepCDF intervalsum with exact rational masses'}}


def score_law(y, law):
    if type(y) not in (int, float) or not math.isfinite(y):
        raise ValueError('finite original reported target required')
    if law['family'] == 'weighted_empirical':
        result = empirical_score(y, law)
    elif law['coordinate'] == 'native' and law['family'] == 'normal' and law['df'] is None:
        mean, sigma = law['location'], law['scale']
        z = (y - mean) / sigma
        pit = float(stats.norm.cdf(z))
        crps = sigma * (z * (2 * pit - 1) + 2 * float(stats.norm.pdf(z)) - 1 / math.sqrt(math.pi))
        intervals = {}
        for level in (.8, .9, .95):
            lower, upper = stats.norm.ppf([(1 - level) / 2, (1 + level) / 2], loc=mean, scale=sigma)
            intervals[str(level)] = {'lower': float(lower), 'upper': float(upper), 'width': float(upper - lower),
                                      'covered': bool(lower <= y <= upper)}
        result = {'CRPS_native': float(crps), 'predictive_median_native': mean,
            'squared_error_native': (y - mean) ** 2, 'PIT_left': pit, 'PIT_right': pit, 'PIT_midpoint': pit,
            'PIT_kind': 'continuous Gaussian', 'intervals': intervals,
            'log_density_native': float(stats.norm.logpdf(y, loc=mean, scale=sigma)),
            'score_numerics': {'method': 'closed GaussianCRPS'}}
    else:
        standard = standard_distribution(law)
        mean, sigma, s0 = law['location'], law['scale'], law['native_first_scale_s0']
        z = (float(model.yj(y / s0)) - mean) / sigma
        pit = float(standard.cdf(z))
        median = s0 * float(model.inverse_yj(mean))
        crps, numerics = native_crps_integral(y, law)
        intervals = {}
        for level in (.8, .9, .95):
            lower, upper = s0 * model.inverse_yj(mean + sigma * standard.ppf([(1 - level) / 2, (1 + level) / 2]))
            if not all(math.isfinite(value) for value in (lower, upper)) or upper <= lower:
                raise ArithmeticError('finite positive-width transformed t interval required')
            intervals[str(level)] = {'lower': float(lower), 'upper': float(upper), 'width': float(upper - lower),
                                      'covered': bool(lower <= y <= upper)}
        jac = float(model.log_jacobian(y, s0, 1.))
        result = {'CRPS_native': crps, 'predictive_median_native': median,
            'squared_error_native': (y - median) ** 2, 'PIT_left': pit, 'PIT_right': pit, 'PIT_midpoint': pit,
            'PIT_kind': 'continuous transported Studentt', 'intervals': intervals,
            'log_density_native': float(standard.logpdf(z) - math.log(sigma) + jac), 'score_numerics': numerics}
    if (not math.isfinite(result['CRPS_native']) or result['CRPS_native'] < 0
            or not math.isfinite(result['squared_error_native'])
            or not 0 <= result['PIT_left'] <= result['PIT_right'] <= 1):
        raise ArithmeticError('invalid proper distribution score')
    return result


def ecdf_deviation(values):
    points = sorted(values)
    count = len(points)
    return max(max((i + 1) / count - value, value - i / count) for i, value in enumerate(points))


def summary(rows):
    result = {}
    for method in task.METHODS:
        result[method] = {}
        for count in task.CHECKPOINTS:
            selected = [row['score'] for row in rows if row['method'] == method and row['feedback_rows'] == count]
            if len(selected) != 128:
                raise ValueError('all128fixedterminalgroups required per method/budget')
            result[method][str(count)] = {
                'terminal_formula_groups': 128, 'new_native_cell_budget': 3 * count,
                'mean_CRPS_native': float(np.mean([value['CRPS_native'] for value in selected])),
                'RMSE_predictive_median_native': float(math.sqrt(np.mean([value['squared_error_native'] for value in selected]))),
                'coverage': {level: float(np.mean([value['intervals'][level]['covered'] for value in selected]))
                             for level in ('0.8', '0.9', '0.95')},
                'mean_interval_width_native': {level: float(np.mean([value['intervals'][level]['width'] for value in selected]))
                                              for level in ('0.8', '0.9', '0.95')},
                'PIT_ECDF_deviation_descriptive': (None if method == task.METHODS[-1]
                    else ecdf_deviation([value['PIT_midpoint'] for value in selected])),
                'mean_PIT_left': float(np.mean([value['PIT_left'] for value in selected])),
                'mean_PIT_right': float(np.mean([value['PIT_right'] for value in selected])),
                'PIT_discrete_midpoint_not_uniform': method == task.METHODS[-1]}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True); parser.add_argument('--approval', required=True)
    parser.add_argument('--forecast-dir', required=True); parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    plan, roles = task.validate(args.plan, args.approval, 'score_after_freeze')
    forecasts, frozen = task.verify_forecast(args.forecast_dir, args.plan, roles)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=False)
    task.write(output / 'TERMINAL_TARGET_REVEAL_START.json', {'actual_utc': task.utc(), 'source_sha256': task.sources(),
        'plan_sha256': task.sha(args.plan), 'approval_sha256': task.sha(args.approval),
        'forecast_freeze_sha256': task.sha(Path(args.forecast_dir) / 'FORECAST_FREEZE.json'),
        'all2560primary_and32secondary_payloads_verified_before_terminalY_extraction': True})
    try:
        evaluator = task.TrustedDisclosure(plan['inputs']['raw_source']['file'], roles)
        truth = evaluator.terminal_truth_after_freeze()
        task.write(output / 'TERMINAL_TRUTH.json', {'role': 'new_terminal_development', 'truth': truth})
        targets = {item['identity']['component_id']: item['native_target'] for item in truth}
        records, cache, laws = [], {}, {}
        for item in forecasts:
            key = (targets[item['component_id']], json.dumps(item['law'], sort_keys=True, separators=(',', ':'), allow_nan=False))
            if key not in cache:
                cache[key] = score_law(key[0], item['law'])
            records.append({'component_id': item['component_id'], 'jid': item['jid'],
                'method': item['method'], 'feedback_rows': item['feedback_rows'],
                'new_native_cell_budget': item['new_native_cell_budget'],
                'native_target': key[0], 'score': cache[key]})
            laws[(item['component_id'], item['method'], item['feedback_rows'])] = item['law']
        for group in targets:
            stale = laws[(group, 'hurdle_stale', 0)]
            if stale != laws[(group, 'hurdle_updated', 0)] or any(laws[(group, 'hurdle_stale', n)] != stale for n in task.CHECKPOINTS):
                raise AssertionError('same warm hurdle and stale exact alias controls failed')
        summarized = summary(records)
        lookup = {(item['component_id'], item['method'], item['feedback_rows']): item['score']['CRPS_native'] for item in records}
        paired = []
        for group in targets:
            paired.append({'component_id': group,
                'updated_minus_stale_CRPS_native': {str(n): lookup[(group, 'hurdle_updated', n)] - lookup[(group, 'hurdle_stale', 0)]
                                                   for n in task.CHECKPOINTS},
                'final_updated_minus_learned_references_CRPS_native': {method: lookup[(group, 'hurdle_updated', 32)] - lookup[(group, method, 32)]
                                                                      for method in task.METHODS[2:]}})
        candidate = summarized['hurdle_updated']['32']['mean_CRPS_native']
        stale = summarized['hurdle_stale']['0']['mean_CRPS_native']
        references = {method: summarized[method]['32']['mean_CRPS_native'] for method in task.METHODS[2:]}
        best = min(references.values())
        primary = {'endpoint': 'final96cell same128terminal paired nativeCRPS updated minus stale warm',
            'updated_final_mean_CRPS_native': candidate, 'stale_warm_mean_CRPS_native': stale,
            'signed_mean_difference_native': float(np.mean([item['updated_minus_stale_CRPS_native']['32'] for item in paired])),
            'relative_gain_fraction_vs_stale': (stale - candidate) / stale if stale else None,
            'learning_gain_observed_descriptive': candidate < stale,
            'learned_reference_final_CRPS_native': references,
            'candidate_to_best_learned_reference_ratio': candidate / best if best else None,
            'within2percent_of_best_learned_reference_descriptive': candidate <= 1.02 * best,
            'not_formal_noninferiority_independent_external_confirmation_or_agent_effectiveness': True}
        secondary = []
        directory = Path(args.forecast_dir)
        prequential_index = task.read(directory / 'PREQUENTIAL_INDEX.json')['secondary_process_laws']
        for item in prequential_index:
            prediction = task.read(directory / item['forecast_file'])
            update = task.read(directory / item['outcome_update_file'])
            secondary.append({'pool_index': item['pool_index'], 'component_id': prediction['identity']['component_id'],
                'frozen_forecast_sha256': item['forecast_sha256'], 'outcome_update_sha256': item['outcome_update_sha256'],
                'prediction_utc': prediction['actual_utc'], 'reveal_utc': update['true_target_disclosure']['actual_utc'],
                'native_target': update['true_target_disclosure']['native_target'],
                'score': score_law(update['true_target_disclosure']['native_target'], prediction['law']),
                'secondary_process_not_part_of_primary_endpoint': True})
        if len(secondary) != 32 or task.sources() != plan['source_sha256']:
            raise ValueError('source or process count changed')
        task.write(output / 'RESULTS.json', {'status': 'COMPLETE_REPORTED_EXACT_FEEDBACK_DEVELOPMENT_PILOT',
            'actual_end_utc': task.utc(), 'source_sha256': task.sources(), 'plan_sha256': task.sha(args.plan),
            'forecast_freeze_sha256': task.sha(directory / 'FORECAST_FREEZE.json'),
            'formula_groups': {'warm': 64, 'fixed_feedback_pool': 32, 'terminal_development': 128},
            'actual_baseline_fits': 8, 'primary_logical_forecasts': 2560, 'secondary_process_forecasts': 32,
            'summary': summarized, 'primary': primary, 'per_terminal_paired_differences': paired,
            'per_primary_forecast_scores': records, 'secondary_prequential_scores': secondary,
            'unique_primary_law_outcome_score_cache_entries': len(cache),
            'scope': 'reported computational source, fixed disclosure schedule, new development cohort; no physical atom provenance inferred',
            'new_wet_DFT_or_adaptive_agent_calls': 0})
        print(json.dumps(primary, ensure_ascii=True))
    except Exception as problem:
        task.write(output / 'FAILED_STOP.json', {'actual_utc': task.utc(), 'type': type(problem).__name__,
            'message': str(problem), 'source_sha256': task.sources(), 'retain_failure_no_retune': True})
        raise


if __name__ == '__main__':
    main()

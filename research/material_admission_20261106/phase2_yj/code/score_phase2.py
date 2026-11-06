"""Native-unit phase2 scores; transformed laws integrate CDF/SF with Jacobian."""
import argparse
import json
import math
from pathlib import Path
import warnings

import numpy as np
from scipy import integrate, special, stats

import run_phase2 as task
import yj_model as adapter


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
    target_t = (float(adapter.transform(y / s0)) - mu) / scale
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


def score_law(y, law):
    _, _, base_score = adapter.parents()
    if law['coordinate'] == 'native':
        result = base_score.score_law(y, law)
        result['point_forecast'] = law['location']
        result['point_forecast_definition'] = 'native_predictive_median'
        result['score_numerics'] = {'method': 'unchanged parent closed positive-loss CRPS', 'quadrature_not_used': True}
        distribution = stats.norm(loc=law['location'], scale=law['scale']) if law['family'] == 'normal' else stats.t(df=law['df'], loc=law['location'], scale=law['scale'])
        result['log_density_native'] = float(distribution.logpdf(y))
        result['log_abs_native_to_u_Jacobian_at_target'] = 0.
        return result
    standard = standard_distribution(law)
    mu, scale, s0 = law['location'], law['scale'], law['native_first_scale_s0']
    t_y = (float(adapter.transform(y / s0)) - mu) / scale
    median = s0 * float(adapter.inverse(mu))
    pit = float(standard.cdf(t_y))
    crps, numerics = native_crps_integral(y, law)
    intervals = {}
    for coverage in (.8, .9, .95):
        q = standard.ppf([(1 - coverage) / 2, (1 + coverage) / 2])
        lower, upper = s0 * adapter.inverse(mu + scale * q)
        if not all(math.isfinite(v) for v in (lower, upper)) or not upper > lower:
            raise ArithmeticError('Invalid transported native quantiles')
        intervals[str(coverage)] = {'lower': float(lower), 'upper': float(upper),
                'width': float(upper - lower), 'covered': bool(lower <= y <= upper)}
    log_jac = float(adapter.log_native_to_standardized_jacobian(y, s0, 1.))
    log_density = float(standard.logpdf(t_y) - math.log(scale) + log_jac)
    squared_error = (y - median) ** 2
    if not all(math.isfinite(v) for v in (pit, median, log_density, squared_error)) or not 0 <= pit <= 1:
        raise ArithmeticError('Invalid native transported probability/median/error')
    return {'CRPS_native': crps, 'squared_error_native': squared_error, 'PIT': pit,
            'intervals': intervals, 'point_forecast': median, 'point_forecast_definition': 'native_predictive_median',
            'log_density_native': log_density, 'log_abs_native_to_u_Jacobian_at_target': log_jac,
            'score_numerics': numerics}


def summarize(rows):
    _, _, base_score = adapter.parents()
    summary = {}
    for kind in adapter.MODELS:
        summary[kind] = {}
        for mask in adapter.MASK_NAMES:
            selected = [r['score'] for r in rows if r['model'] == kind and r['mask'] == mask]
            if len(selected) != 192:
                raise ValueError('All192 groups per law role/mask required')
            values = {'formula_groups': 192,
                'equal_group_mean_CRPS_native': float(np.mean([s['CRPS_native'] for s in selected])),
                'RMSE_of_predictive_median_native': float(math.sqrt(np.mean([s['squared_error_native'] for s in selected]))),
                'PIT_empirical_CDF_max_deviation': base_score.empirical_cdf_deviation([s['PIT'] for s in selected]),
                'coverage': {}, 'mean_interval_width_native': {}}
            for level in ('0.8', '0.9', '0.95'):
                values['coverage'][level] = float(np.mean([s['intervals'][level]['covered'] for s in selected]))
                values['mean_interval_width_native'][level] = float(np.mean([s['intervals'][level]['width'] for s in selected]))
            summary[kind][mask] = values
    return summary


def admission_gates(summary):
    cfg = task.configuration()['gates']
    dual = summary['yj_joint_t']['both']
    loss = dual['equal_group_mean_CRPS_native']
    none = summary['yj_joint_t']['none']['equal_group_mean_CRPS_native']
    references = {kind: summary[kind]['both']['equal_group_mean_CRPS_native'] for kind in cfg['references']}
    best = min(references.values())
    criteria = {'conditional_gain_at_least2percent': loss <= cfg['YJdual_to_own_none_CRPS_ratio_max'] * none,
                'within2percent_all_same_information_strong_references': loss <= cfg['YJdual_to_best_same_information_reference_CRPS_ratio_max'] * best,
                'dual90coverage_at_least0p80': dual['coverage']['0.9'] >= cfg['YJdual_90_coverage_min'],
                'dual_PIT_ECDF_deviation_at_most0p15': dual['PIT_empirical_CDF_max_deviation'] <= cfg['YJdual_PIT_ECDF_max_deviation_max']}
    return {'criteria': criteria, 'same_information_reference_losses': references,
            'candidate_to_own_none_ratio': loss / none if none else None,
            'candidate_to_strongest_reference_ratio': loss / best if best else None,
            'status': 'PASS_FOR_SEPARATE_FEEDBACK_TASK_DESIGN_ONLY' if all(criteria.values()) else 'NO_GO_STOP_FULL_CAMPAIGN_NO_TUNING',
            'not_independent_confirmation_calibration_or_novel_algorithm_evidence': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--approval', required=True)
    parser.add_argument('--forecast-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    plan, roles = task.validate(args.plan, args.approval, 'score_after_freeze')
    forecasts, _ = task.verify_freeze(args.forecast_dir, args.plan, roles)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    task.write(output / 'TARGET_REVEAL_START.json', {'actual_utc': task.utc(), 'source_sha256': task.source_hashes(),
        'parent_source_sha256': adapter.PARENT_SHA, 'plan_sha256': task.sha(args.plan),
        'approval_sha256': task.sha(args.approval),
        'forecast_freeze_sha256': task.sha(Path(args.forecast_dir) / 'FORECAST_FREEZE.json'),
        'all3840_laws_verified_before_target_extraction': True, 'full_gzip_physical_reparse_disclosed': True})
    try:
        _, base_io, _ = adapter.parents()
        truth = base_io.reveal_diagnostic_truth(plan['inputs']['raw_source']['file'], roles)
        task.write(output / 'DIAGNOSTIC_TRUTH.json', {'role': 'phase2_development_only', 'truth': truth})
        by_group = {r['identity']['component_id']: r['native_target'] for r in truth}
        laws, rows, score_cache = {}, [], {}
        for f in forecasts:
            y = by_group[f['component_id']]
            laws[(f['component_id'], f['model'], f['mask'])] = f['law']
            cache_key = (y, json.dumps(f['law'], sort_keys=True, separators=(',', ':'), allow_nan=False))
            if cache_key not in score_cache:
                score_cache[cache_key] = score_law(y, f['law'])
            rows.append({'component_id': f['component_id'], 'jid': f['jid'], 'model': f['model'], 'mask': f['mask'],
                'native_target': y, 'score': score_cache[cache_key]})
        for group in by_group:
            marginal = laws[(group, 'yj_joint_t', 'none')]
            if any(laws[(group, 'yj_target_marginal', mask)] != marginal for mask in adapter.MASK_NAMES):
                raise AssertionError('Transported target marginal identity control failed')
        summary = summarize(rows)
        loss = {(r['component_id'], r['model'], r['mask']): r['score']['CRPS_native'] for r in rows}
        paired = [{'component_id': group,
             'YJ_conditional_gain_vs_own_none': {mask: loss[(group, 'yj_joint_t', 'none')] - loss[(group, 'yj_joint_t', mask)] for mask in adapter.MASK_NAMES},
             'dual_YJ_minus_same_information_references': {kind: loss[(group, 'yj_joint_t', 'both')] - loss[(group, kind, 'both')] for kind in task.configuration()['gates']['references']}}
             for group in by_group]
        if task.source_hashes() != plan['source_sha256'] or adapter.parent_hashes() != adapter.PARENT_SHA:
            raise ValueError('Bound parent or new source changed during scoring')
        task.write(output / 'RESULTS.json', {'status': 'COMPLETE_PHASE2_DEVELOPMENT_MODEL_ADMISSION',
            'actual_end_utc': task.utc(), 'source_sha256': task.source_hashes(), 'parent_source_sha256': adapter.PARENT_SHA,
            'plan_sha256': task.sha(args.plan), 'forecast_freeze_sha256': task.sha(Path(args.forecast_dir) / 'FORECAST_FREEZE.json'),
            'formula_groups_fit': 64, 'formula_groups_diagnostic_development': 192, 'independent_confirmation_groups': 0,
            'actual_NIW_fits': 2, 'forecast_law_roles': 5, 'summary': summary, 'admission': admission_gates(summary),
            'per_forecast_scores': rows, 'per_group_paired_differences': paired,
            'deterministic_identical_law_outcome_score_cache_entries': len(score_cache),
            'identical_readout_cache_is_not_new_material_sample': True,
            'native_unit': 'source bulk GPa attribution, aggregation qualified',
            'all_negative_results_preserved_phase1_NO_GO_unchanged': True,
            'no_new_wet_DFT_allocation_agent_calls': 0})
        print(json.dumps({'status': 'COMPLETE_PHASE2_DEVELOPMENT_MODEL_ADMISSION', 'admission': admission_gates(summary)}))
    except Exception as error:
        task.write(output / 'FAILED_STOP.json', {'actual_utc': task.utc(), 'type': type(error).__name__,
            'message': str(error), 'retain_failure_no_tuning': True})
        raise


if __name__ == '__main__':
    main()

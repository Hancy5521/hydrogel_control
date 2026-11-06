"""Closed-form native-unit scores; target reveal only after complete forecast freeze."""
import argparse
import math
from pathlib import Path

import numpy as np
from scipy import special, stats

import admission_io as io
import niw_predictive as model


def crps_normal(y, location, scale):
    if not all(math.isfinite(v) for v in (y, location, scale)) or scale <= 0:
        raise ValueError('Finite outcome/location and positive normal SD required')
    z = abs((y - location) / scale)
    value = scale * (z * (2 * special.ndtr(z) - 1) + 2 * math.exp(-.5 * z * z) / math.sqrt(2 * math.pi) - 1 / math.sqrt(math.pi))
    if not math.isfinite(value) or value < 0:
        raise ArithmeticError('Normal CRPS outside supported numerical range')
    return value


def crps_student_t(y, location, scale, df):
    """scoringRules author article p26; location/scale, not variance, df>1."""
    if not all(math.isfinite(v) for v in (y, location, scale, df)) or scale <= 0 or df <= 1:
        raise ValueError('Finite t parameters, positive scale and df>1 required')
    z = abs((y - location) / scale)
    beta_ratio = math.exp(special.betaln(.5, df - .5) - 2 * special.betaln(.5, .5 * df))
    constant = 2 * math.sqrt(df) / (df - 1) * beta_ratio
    value = scale * (z * (2 * stats.t.cdf(z, df) - 1)
            + 2 * stats.t.pdf(z, df) * (df + z * z) / (df - 1) - constant)
    if not math.isfinite(value) or value < 0:
        raise ArithmeticError('Student-t CRPS outside supported numerical range')
    return value


def score_law(y, law):
    mu, scale, df = law['location'], law['scale'], law['df']
    if law['family'] == 'normal':
        distribution = stats.norm(loc=mu, scale=scale)
        crps = crps_normal(y, mu, scale)
    elif law['family'] == 'student_t':
        distribution = stats.t(df=df, loc=mu, scale=scale)
        crps = crps_student_t(y, mu, scale, df)
    else:
        raise ValueError('Unknown forecast family')
    pit = float(distribution.cdf(y))
    intervals = {}
    for coverage in (.8, .9, .95):
        lower, upper = distribution.ppf([(1 - coverage) / 2, (1 + coverage) / 2])
        if not all(math.isfinite(v) for v in (lower, upper, pit)) or not 0 <= pit <= 1 or not upper > lower:
            raise ArithmeticError('Invalid PIT/central interval')
        intervals[str(coverage)] = {'lower': float(lower), 'upper': float(upper),
                'width': float(upper - lower), 'covered': bool(lower <= y <= upper)}
    return {'CRPS_native': crps, 'squared_error_native': float((y - mu) ** 2),
            'PIT': pit, 'intervals': intervals}


def empirical_cdf_deviation(values):
    u = np.sort(np.asarray(values, dtype=float))
    if len(u) != 192 or not np.isfinite(u).all() or (u < 0).any() or (u > 1).any():
        raise ValueError('All192 development PIT values required')
    upper = np.arange(1, len(u) + 1) / len(u) - u
    lower = u - np.arange(len(u)) / len(u)
    return float(max(upper.max(), lower.max()))


def summarize(rows):
    summary = {}
    for kind in model.MODELS:
        summary[kind] = {}
        for mask in model.MASK_NAMES:
            selected = [r for r in rows if r['model'] == kind and r['mask'] == mask]
            if len(selected) != 192:
                raise ValueError('Complete192-group model/mask scores required')
            result = {'formula_groups': 192, 'equal_group_mean_CRPS_native': float(np.mean([r['score']['CRPS_native'] for r in selected])),
                'RMSE_native': float(math.sqrt(np.mean([r['score']['squared_error_native'] for r in selected]))),
                'PIT_empirical_CDF_max_deviation': empirical_cdf_deviation([r['score']['PIT'] for r in selected]),
                'PIT_is_not_strictly_IID_under_shared_learned_Bayes_parameters': True,
                'coverage': {}, 'mean_interval_width_native': {}}
            for level in ('0.8', '0.9', '0.95'):
                result['coverage'][level] = float(np.mean([r['score']['intervals'][level]['covered'] for r in selected]))
                result['mean_interval_width_native'][level] = float(np.mean([r['score']['intervals'][level]['width'] for r in selected]))
            summary[kind][mask] = result
    return summary


def admission_gates(summary):
    dual = summary['niw_joint_t']['both']
    none = summary['niw_joint_t']['none']['equal_group_mean_CRPS_native']
    gauss = summary['gaussian_momentmatched']['both']['equal_group_mean_CRPS_native']
    loss = dual['equal_group_mean_CRPS_native']
    criteria = {'at_least_2percent_CRPS_gain_vs_own_no_proxy': loss <= .98 * none,
                'within_2percent_conditional_Gaussian_CRPS': loss <= 1.02 * gauss,
                'dual_90_coverage_at_least_0p80': dual['coverage']['0.9'] >= .8,
                'dual_PIT_ECDF_max_deviation_at_most_0p15': dual['PIT_empirical_CDF_max_deviation'] <= .15}
    return {'criteria': criteria, 'status': 'PASS_FOR_SEPARATE_FEEDBACK_TASK_DESIGN_ONLY' if all(criteria.values()) else 'NO_GO_STOP_FULL_CAMPAIGN_NO_RETUNE',
            'CRPS_dual_to_none_ratio': loss / none if none else None,
            'CRPS_dual_to_Gaussian_ratio': loss / gauss if gauss else None,
            'thresholds_are_descriptive_admission_not_hypothesis_tests_or_calibration_guarantees': True,
            'no_allocation_or_material_novelty_claim': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--approval', required=True)
    parser.add_argument('--forecast-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    plan, roles = io.validate(args.plan, args.approval, 'score_after_freeze')
    forecasts, receipt = io.verify_forecast_freeze(args.forecast_dir, args.plan, roles)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    io.write(output / 'TARGET_REVEAL_START.json', {'actual_utc': io.utc(), 'source_sha256': io.sources(),
        'plan_sha256': io.sha(args.plan), 'approval_sha256': io.sha(args.approval),
        'forecast_freeze_sha256': io.sha(Path(args.forecast_dir) / 'FORECAST_FREEZE.json'),
        'all2304_forecasts_verified_before_native_target_extraction': True,
        'physical_full_gzip_reparse_disclosed': True})
    try:
        truth = io.reveal_diagnostic_truth(plan['inputs']['raw_source']['file'], roles)
        io.write(output / 'DIAGNOSTIC_TRUTH.json', {'role': 'consumed_development_only', 'truth': truth})
        by_group = {r['identity']['component_id']: r['native_target'] for r in truth}
        scores = []
        laws = {}
        for forecast in forecasts:
            key = (forecast['component_id'], forecast['model'], forecast['mask'])
            laws[key] = forecast['law']
            scores.append({'component_id': forecast['component_id'], 'jid': forecast['jid'],
                'model': forecast['model'], 'mask': forecast['mask'],
                'native_target': by_group[forecast['component_id']],
                'score': score_law(by_group[forecast['component_id']], forecast['law'])})
        for group in by_group:
            marginal = laws[(group, 'niw_joint_t', 'none')]
            if any(laws[(group, 'independent_target_marginal', mask)] != marginal for mask in model.MASK_NAMES):
                raise AssertionError('Independent target marginal identity control failed')
        summary = summarize(scores)
        score_by_key = {(r['component_id'], r['model'], r['mask']): r['score']['CRPS_native'] for r in scores}
        paired = [{'component_id': group, 'conditional_gain_vs_none_by_mask': {
            mask: score_by_key[(group, 'niw_joint_t', 'none')] - score_by_key[(group, 'niw_joint_t', mask)]
            for mask in model.MASK_NAMES}} for group in by_group]
        if plan['source_sha256'] != io.sources():
            raise ValueError('Source changed during scoring')
        io.write(output / 'RESULTS.json', {'actual_end_utc': io.utc(), 'status': 'COMPLETE_DEVELOPMENT_MODEL_ADMISSION',
            'plan_sha256': io.sha(args.plan), 'source_sha256': io.sources(),
            'forecast_freeze_sha256': io.sha(Path(args.forecast_dir) / 'FORECAST_FREEZE.json'),
            'formula_groups_fit': 64, 'formula_groups_diagnostic_development': 192,
            'independent_confirmation_groups': 0, 'summary': summary, 'admission': admission_gates(summary),
            'per_forecast_scores': scores, 'per_group_paired_transfer': paired,
            'native_unit': 'source bulk field GPa attribution; aggregation remains qualified',
            'diagnostics_not_feedback_for_same_cohort_model_selection': True,
            'new_wet_DFT_allocation_or_agent_calls': 0})
        print(summary['niw_joint_t']['both'])
        print(admission_gates(summary))
    except Exception as error:
        io.write(output / 'FAILED_STOP.json', {'actual_utc': io.utc(), 'type': type(error).__name__,
            'message': str(error), 'retain_failure_no_retune': True})
        raise


if __name__ == '__main__':
    main()

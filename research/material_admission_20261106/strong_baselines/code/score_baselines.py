"""Exact empirical-law or parent Gaussian scores, only after all1536law freeze."""
import argparse
from fractions import Fraction
import importlib
import math
from pathlib import Path
import sys

import numpy as np

import baseline_models as models
import run_baselines as task

_PARENT_SCORER = None


def parent_scorer():
    global _PARENT_SCORER
    if _PARENT_SCORER is None:
        if task.parent_hashes() != task.PARENT_SHA:
            raise ValueError('Original independently audited Gaussian scorer changed')
        directory = task.parent_directory()
        for name in ('niw_predictive', 'admission_io', 'score_admission'):
            if name in sys.modules and Path(sys.modules[name].__file__).resolve() != (directory / (name + '.py')).resolve():
                raise ValueError('Conflicting parent module path')
        sys.path.insert(0, str(directory))
        _PARENT_SCORER = importlib.import_module('score_admission')
    return _PARENT_SCORER


def empirical_nodes(law):
    support = law['support_native']
    mass = [Fraction(w) for w in law['weight_fractions']]
    if len(support) != len(mass) or not support or sum(mass, Fraction(0)) != 1 or any(w < 0 for w in mass):
        raise ValueError('Finite empirical support with exact probability simplex required')
    nodes = {}
    for value, weight in zip(support, mass):
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError('Native support must be finite')
        nodes[float(value)] = nodes.get(float(value), Fraction(0)) + weight
    return sorted(nodes), nodes


def empirical_quantile(points, masses, p):
    p = Fraction(p)
    if not 0 < p < 1:
        raise ValueError('Empirical quantile probability must be in open unit interval')
    cumulative = Fraction(0)
    for value in points:
        cumulative += masses[value]
        if cumulative >= p:
            return value
    raise AssertionError('Complete empirical distribution required')


def score_empirical(y, law):
    if type(y) not in (int, float) or not math.isfinite(y):
        raise ValueError('Finite original target required')
    points, masses = empirical_nodes(law)
    # Positive piecewise CDF integral; no random samples or subtractive energy cancellation.
    cuts = sorted(set(points + [float(y)]))
    cumulative, crps = Fraction(0), 0.
    for lower, upper in zip(cuts[:-1], cuts[1:]):
        cumulative += masses.get(lower, Fraction(0))
        difference = cumulative - int(lower >= y)
        crps += float(difference * difference) * (upper - lower)
    left = sum((masses[x] for x in points if x < y), Fraction(0))
    right = sum((masses[x] for x in points if x <= y), Fraction(0))
    midpoint = (left + right) / 2
    median = empirical_quantile(points, masses, Fraction(1, 2))
    intervals = {}
    for coverage in ('0.8', '0.9', '0.95'):
        level = Fraction(coverage)
        lower = empirical_quantile(points, masses, (1 - level) / 2)
        upper = empirical_quantile(points, masses, (1 + level) / 2)
        intervals[coverage] = {'lower': lower, 'upper': upper, 'width': upper - lower,
                'covered': bool(lower <= y <= upper)}
    squared_error = (y - median) ** 2
    if not math.isfinite(crps) or crps < 0 or not math.isfinite(squared_error):
        raise ArithmeticError('Native empirical score exceeds numerical range')
    return {'CRPS_native': crps, 'squared_error_native': squared_error,
            'predictive_median': median, 'intervals': intervals,
            'PIT_left': float(left), 'PIT_right': float(right), 'PIT_midpoint': float(midpoint),
            'PIT_exact_fractions': {'left': str(left), 'right': str(right), 'midpoint': str(midpoint)},
            'PIT_kind': 'discrete interval; midpoint is not continuous-uniform PIT',
            'score_formula': 'exact finite empirical-CDF interval sum, equivalent to energy identity; native arithmetic float',
            'sampled_distribution_or_randomized_PIT': False}


def score_law(y, law):
    if law['family'] == 'weighted_empirical':
        return score_empirical(y, law)
    if law['family'] != 'normal':
        raise ValueError('Only two preregistered forecast-law families allowed')
    result = parent_scorer().score_law(y, law)
    result['predictive_median'] = law['location']
    result['PIT_left'] = result['PIT']
    result['PIT_right'] = result['PIT']
    result['PIT_midpoint'] = result['PIT']
    result['PIT_kind'] = 'continuous normal'
    return result


def summarize(rows):
    result = {}
    for method in models.METHODS:
        result[method] = {}
        for mask in models.MASK_NAMES:
            selected = [r['score'] for r in rows if r['method'] == method and r['mask'] == mask]
            if len(selected) != 192:
                raise ValueError('Every fixed method/mask must retain192formula groups')
            current = {'formula_groups': 192,
                'equal_group_mean_CRPS_native': float(np.mean([s['CRPS_native'] for s in selected])),
                'RMSE_of_predictive_median_native': float(math.sqrt(np.mean([s['squared_error_native'] for s in selected]))),
                'mean_PIT_left': float(np.mean([s['PIT_left'] for s in selected])),
                'mean_PIT_right': float(np.mean([s['PIT_right'] for s in selected])),
                'mean_PIT_midpoint': float(np.mean([s['PIT_midpoint'] for s in selected])),
                'coverage': {}, 'mean_interval_width_native': {}}
            for level in ('0.8', '0.9', '0.95'):
                current['coverage'][level] = float(np.mean([s['intervals'][level]['covered'] for s in selected]))
                current['mean_interval_width_native'][level] = float(np.mean([s['intervals'][level]['width'] for s in selected]))
            if method == 'gpr_fixed_recipe':
                current['continuous_PIT_ECDF_deviation_descriptive'] = parent_scorer().empirical_cdf_deviation([s['PIT_midpoint'] for s in selected])
            else:
                current['continuous_PIT_ECDF_deviation_descriptive'] = None
                current['PIT_caution'] = 'Discrete left/right intervals; do not apply continuous-uniform admission gate to midpoint'
            result[method][mask] = current
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--approval', required=True)
    parser.add_argument('--forecast-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    plan, roles = task.validate(args.plan, args.approval, 'score_after_freeze')
    forecasts = task.verify_freeze(args.forecast_dir, args.plan, roles)
    # Neither file is opened/hashed by forecast. These reads occur after1536laws are verified.
    task.validate_score_only_inputs(plan)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    task.write(output / 'SAVED_TRUTH_SCORE_START.json', {'actual_utc': task.utc(),
        'source_sha256': task.source_hashes(), 'parent_source_sha256': task.PARENT_SHA,
        'plan_sha256': task.sha(args.plan), 'approval_sha256': task.sha(args.approval),
        'forecast_freeze_sha256': task.sha(Path(args.forecast_dir) / 'FORECAST_FREEZE.json'),
        'all1536_baseline_laws_verified_before_saved_truth_numeric_read': True,
        'no_main_model_refit_or_rescore': True})
    try:
        truth = task.read(plan['inputs']['diagnostic_truth']['file'])['truth']
        if len(truth) != 192:
            raise ValueError('Exactly192saved development targets required')
        by_group = {}
        for row, meta in zip(truth, roles['representatives']['diagnostic']):
            if row['identity'] != meta or type(row['native_target']) not in (int, float) or not math.isfinite(row['native_target']):
                raise ValueError('Saved native target identity mismatch')
            by_group[meta['component_id']] = row['native_target']
        scores = []
        for f in forecasts:
            y = by_group[f['component_id']]
            scores.append({'component_id': f['component_id'], 'jid': f['jid'], 'method': f['method'],
                'mask': f['mask'], 'native_target': y, 'score': score_law(y, f['law'])})
        summary = summarize(scores)
        # Trusted JSON parser sees the saved object; only its existing summary is used.
        candidate_object = task.read(plan['inputs']['candidate_results']['file'])
        candidate_summary = candidate_object['summary']['yj_joint_t']['both']
        if candidate_summary['formula_groups'] != 192:
            raise ValueError('Candidate summary is not the same development cohort size')
        candidate_loss = candidate_summary['equal_group_mean_CRPS_native']
        del candidate_object
        reference_losses = {method: summary[method]['both']['equal_group_mean_CRPS_native'] for method in models.METHODS}
        best = min(reference_losses.values())
        comparison = {'candidate_fixed_frozen_YJ_dual_CRPS': candidate_loss,
            'added_baseline_dual_CRPS': reference_losses,
            'candidate_to_best_added_baseline_ratio': candidate_loss / best if best else None,
            'candidate_minus_best_added_baseline_native': candidate_loss - best,
            'read_existing_candidate_summary_only_no_model_or_score_recalculation': True,
            'does_not_reopen_or_change_original_phase2_four_gates': True,
            'posthoc_development_check_not_new_independent_confirmation_or_all_SOTA_guarantee': True}
        if task.source_hashes() != plan['source_sha256'] or task.parent_hashes() != task.PARENT_SHA:
            raise ValueError('Bound source changed during baseline scoring')
        task.write(output / 'RESULTS.json', {'status': 'COMPLETE_POSTHOC_FIXED_LEARNING_BASELINE_CHECK',
            'actual_end_utc': task.utc(), 'source_sha256': task.source_hashes(), 'parent_source_sha256': task.PARENT_SHA,
            'plan_sha256': task.sha(args.plan), 'forecast_freeze_sha256': task.sha(Path(args.forecast_dir) / 'FORECAST_FREEZE.json'),
            'formula_groups_fit': 64, 'formula_groups_consumed_diagnostic_development': 192,
            'new_independent_confirmation_groups': 0, 'actual_baseline_fits': 6,
            'summary': summary, 'candidate_comparison': comparison, 'per_forecast_scores': scores,
            'unit': 'source bulk GPa attribution; aggregation remains qualified',
            'GP_optimizer_warnings_and_forest_replay_state': 'frozen MODEL_STATES.json',
            'original_main_model_or_four_gates_modified': False})
        print(comparison)
    except Exception as error:
        task.write(output / 'FAILED_STOP.json', {'actual_utc': task.utc(), 'type': type(error).__name__,
            'message': str(error), 'retain_failure_no_recipe_search': True})
        raise


if __name__ == '__main__':
    main()

"""Original Gaussian scores and separate finite-grid prediction-set diagnostics.

No curve interpolation, label synthesis, predictive fitting or density rescaling.
The primary score averages native log-rate CRPS within each curve/mask, masks
within original curve, curves within independent unit, then equal-weight units.
"""
from collections import defaultdict
import math
from statistics import NormalDist

import numpy as np
from calibration import alpha_fraction, make_band, real


def target_weights(target_gamma):
    """Normalized trapezoid on [first hidden, last hidden], not anchor-to-end."""
    g = [real(v, 'target gamma') for v in target_gamma]
    if len(g) < 2 or any(v <= 0 for v in g) or any(b <= a for a, b in zip(g, g[1:])):
        raise ValueError('At least two positive strictly increasing native targets required')
    x = [math.log(v) for v in g]
    widths = [b - a for a, b in zip(x, x[1:])]
    w = [widths[0] / 2] + [(a + b) / 2 for a, b in zip(widths, widths[1:])] + [widths[-1] / 2]
    span = x[-1] - x[0]
    return [v / span for v in w]


def gaussian_scores(mean, covariance, truth, weights, alpha):
    """Scores of the original frozen density; bands never enter this function."""
    m, c, y, w = map(lambda v: np.asarray(v, dtype=float), (mean, covariance, truth, weights))
    n = len(m)
    if m.shape != (n,) or y.shape != (n,) or c.shape != (n, n) or w.shape != (n,):
        raise ValueError('Aligned vector and full covariance shapes required')
    if n == 0 or not all(np.isfinite(v).all() for v in (m, c, y, w)):
        raise ValueError('Finite original law, truth and weights required')
    if not np.allclose(c, c.T, atol=1e-12, rtol=1e-12) or (w < 0).any() or not math.isclose(float(w.sum()), 1., abs_tol=1e-12):
        raise ValueError('Symmetric covariance and normalized nonnegative weights required')
    chol = np.linalg.cholesky(c)
    sd = np.sqrt(np.diag(c))
    errors = y - m
    z = errors / sd
    phi = np.asarray([math.exp(-float(v) ** 2 / 2) / math.sqrt(2 * math.pi) for v in z])
    cdf = np.asarray([0.5 * math.erfc(-float(v) / math.sqrt(2)) for v in z])
    crps = sd * (z * (2 * cdf - 1) + 2 * phi - 1 / math.sqrt(math.pi))
    nll = .5 * math.log(2 * math.pi) + np.log(sd) + .5 * z ** 2
    radius = NormalDist().inv_cdf(1 - float(alpha_fraction(alpha)) / 2)
    covered = np.abs(z) <= radius
    solve = np.linalg.solve(chol, errors)
    joint_log_density = -.5 * (n * math.log(2 * math.pi) + 2 * np.log(np.diag(chol)).sum() + solve @ solve)
    values = {'log_rate_CRPS': float(w @ crps), 'log_rate_MSE': float(w @ errors ** 2),
              'log_rate_marginal_NLL': float(w @ nll), 'joint_log_density': float(joint_log_density),
              'weighted_point_coverage': float(w @ covered), 'whole_finite_grid_covered': bool(covered.all()),
              'mean_log_width': float(w @ (2 * radius * sd))}
    if any(not math.isfinite(v) for v in values.values() if type(v) is float):
        raise OverflowError('Original density metric overflow; retain failure rather than omit record')
    return values


def band_diagnostics(mean, sd, truth, weights, threshold):
    """Coverage/width/interval-score diagnostics of a set, never CRPS or NLL."""
    band = make_band(mean, sd, threshold)
    w = np.asarray(weights, dtype=float)
    y = np.asarray(truth, dtype=float)
    if band.unbounded:
        return {'weighted_point_coverage': 1., 'whole_finite_grid_covered': True,
                'mean_log_width': None, 'interval_score_same_alpha': None,
                'unbounded': True, 'coverage_is_vacuous': True}
    low, high = np.asarray(band.lower_log), np.asarray(band.upper_log)
    covered = (low <= y) & (y <= high)
    score = high - low + 2 / float(alpha_fraction(threshold.alpha)) * (np.maximum(low - y, 0) + np.maximum(y - high, 0))
    width, interval_score = float(w @ (high - low)), float(w @ score)
    if not math.isfinite(width) or not math.isfinite(interval_score):
        raise OverflowError('Band diagnostic overflow; retain failure')
    return {'weighted_point_coverage': float(w @ covered), 'whole_finite_grid_covered': bool(covered.all()),
            'mean_log_width': width, 'interval_score_same_alpha': interval_score,
            'unbounded': False, 'coverage_is_vacuous': False}


def _equal_curve_average(rows, section, key):
    """Repeated masks do not become extra independent curves or units."""
    curves = defaultdict(list)
    for row in rows:
        curves[row['source_record_key']].append(row[section][key])
    if any(value is None for values in curves.values() for value in values):
        return None
    return float(np.mean([np.mean(values) for values in curves.values()]))


def evaluate_complete_cohort(predictions, truth, protocol, thresholds):
    enrolled = {r['record_id']: r for r in protocol['cohort']['records']}
    methods = [m['id'] for m in protocol['config']['methods']]
    alpha = protocol['config']['alpha']
    result = {'data_mode': protocol['config']['data_mode'], 'is_new_material_evidence': False,
              'automatic_independence_certification': False,
              'estimand': 'Equal independent units; equal original curves within unit; equal registered masks within curve; native target-only log-rate trapezoid.',
              'density_and_set_are_separate': True, 'methods': {}}
    for mid in methods:
        by_unit, record_metrics = defaultdict(list), []
        for record in predictions['records']:
            rid = record['record_id']
            meta, law, values = enrolled[rid], record['methods'][mid], truth[rid]
            w = target_weights(record['target_gamma'])
            sd = np.sqrt(np.diag(np.asarray(law['covariance_log_eta'], dtype=float))).tolist()
            entry = {'record_id': rid, 'unit_id': meta['unit_id'], 'source_record_key': meta['source_record_key'],
                     'n_targets': len(values), 'target_weights': w,
                     'target_only_log_rate_span': math.log(record['target_gamma'][-1] / record['target_gamma'][0]),
                     'omitted_log_gap_from_highest_anchor': math.log(record['target_gamma'][0] / record['observed_gamma'][-1]),
                     'raw_gaussian': gaussian_scores(law['mean_log_eta'], law['covariance_log_eta'], values, w, alpha),
                     'conformal_set': band_diagnostics(law['mean_log_eta'], sd, values, w, thresholds[mid])}
            record_metrics.append(entry)
            by_unit[meta['unit_id']].append(entry)
        unit_metrics = []
        for uid, rows in sorted(by_unit.items()):
            raw = {key: _equal_curve_average(rows, 'raw_gaussian', key) for key in
                   ('log_rate_CRPS', 'log_rate_MSE', 'log_rate_marginal_NLL', 'joint_log_density', 'weighted_point_coverage', 'mean_log_width')}
            raw['whole_unit_finite_grid_covered'] = all(r['raw_gaussian']['whole_finite_grid_covered'] for r in rows)
            sets = {key: _equal_curve_average(rows, 'conformal_set', key) for key in
                    ('weighted_point_coverage', 'mean_log_width', 'interval_score_same_alpha')}
            sets['whole_unit_finite_grid_covered'] = all(r['conformal_set']['whole_finite_grid_covered'] for r in rows)
            sets['unbounded'] = any(r['conformal_set']['unbounded'] for r in rows)
            unit_metrics.append({'unit_id': uid, 'original_curves': len({r['source_record_key'] for r in rows}),
                                 'registered_curve_masks': len(rows), 'raw_gaussian': raw, 'conformal_set': sets})
        if not unit_metrics:
            raise ValueError('A nonempty complete confirmation cohort is required')
        aggregate = {}
        for section in ('raw_gaussian', 'conformal_set'):
            aggregate[section] = {}
            for key in unit_metrics[0][section]:
                vals = [r[section][key] for r in unit_metrics]
                aggregate[section][key] = None if any(v is None for v in vals) else float(np.mean(vals))
        aggregate['raw_gaussian']['log_rate_RMSE'] = math.sqrt(aggregate['raw_gaussian']['log_rate_MSE'])
        aggregate['n_independent_units'] = len(unit_metrics)
        aggregate['n_registered_curve_masks'] = len(record_metrics)
        aggregate['n_units_with_unbounded_sets'] = sum(r['conformal_set']['unbounded'] for r in unit_metrics)
        result['methods'][mid] = {'aggregate': aggregate, 'per_unit': unit_metrics, 'per_record': record_metrics,
                                  'threshold': thresholds[mid].to_json()}
    primary = protocol['config']['primary_method_id']
    candidate = {r['unit_id']: r['raw_gaussian']['log_rate_CRPS'] for r in result['methods'][primary]['per_unit']}
    result['paired_raw_CRPS_candidate_minus_comparator'] = {}
    for mid in methods:
        if mid != primary:
            diffs = {r['unit_id']: candidate[r['unit_id']] - r['raw_gaussian']['log_rate_CRPS'] for r in result['methods'][mid]['per_unit']}
            result['paired_raw_CRPS_candidate_minus_comparator'][mid] = {'per_unit': diffs, 'equal_unit_mean': float(np.mean(list(diffs.values()))),
                                                                         'negative_favors': primary, 'inferential_interval': None}
    return result

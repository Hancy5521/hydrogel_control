"""Independent saved baseline score audit; no SUT/sklearn fit, predict, or scoring imports."""
import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import traceback
import warnings

from scipy.integrate import quad
from scipy.special import ndtri

METHODS = ('gpr_fixed_recipe', 'forest_all_original_leaf_cdf')
MASKS = ('none', 'e_form', 'gap_opt', 'both')
LEVELS = ('0.8', '0.9', '0.95')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON key: ' + key)
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('Nonfinite JSON token: ' + value)
    return json.loads(Path(path).read_text('utf-8-sig'), object_pairs_hook=unique, parse_constant=invalid)


def require(ok, message):
    if not ok:
        raise AssertionError(message)


class Checks:
    def __init__(self):
        self.counts, self.errors = {}, {}
    def number(self, name, actual, expected, atol=2e-8, rtol=2e-10):
        require(math.isfinite(actual) and math.isfinite(expected), name + ' nonfinite')
        error = abs(actual - expected)
        require(error <= atol + rtol * abs(expected), f'{name}: {actual} versus {expected}, error={error}')
        self.counts[name] = self.counts.get(name, 0) + 1
        self.errors[name] = max(self.errors.get(name, 0), error)
    def exact(self, name, actual, expected):
        require(type(actual) == type(expected) and actual == expected, name + ' exact mismatch')
        self.counts[name] = self.counts.get(name, 0) + 1


def phi_cdf(x):
    return .5 * math.erfc(-x / math.sqrt(2))


def phi_sf(x):
    return .5 * math.erfc(x / math.sqrt(2))


def gaussian(y, law):
    m, s = law['location'], law['scale']
    require(law['family'] == 'normal' and law['df'] is None and math.isfinite(m) and math.isfinite(s) and s > 0,
            'Finite normal law required')
    z = (y - m) / s
    # Independent CDF-integral identity; producer uses the analytic normal CRPS.
    lower, e1 = quad(lambda u: phi_cdf(u) ** 2, -math.inf, z, epsabs=1e-11, epsrel=1e-11, limit=300)
    upper, e2 = quad(lambda u: phi_sf(u) ** 2, z, math.inf, epsabs=1e-11, epsrel=1e-11, limit=300)
    pit = phi_cdf(z)
    intervals = {}
    for level in LEVELS:
        a = (1 - float(level)) / 2
        lo, hi = m + s * float(ndtri(a)), m + s * float(ndtri(1 - a))
        intervals[level] = {'lower': lo, 'upper': hi, 'width': hi - lo, 'covered': lo <= y <= hi}
    return {'CRPS_native': s * (lower + upper), 'squared_error_native': (y - m) ** 2,
            'predictive_median': m, 'PIT': pit, 'PIT_left': pit, 'PIT_right': pit,
            'PIT_midpoint': pit, 'intervals': intervals, 'quad_error_scaled': s * (e1 + e2)}


def empirical(y, law):
    require(law['family'] == 'weighted_empirical', 'Finite empirical law required')
    x = law['support_native']
    w = [Fraction(a) for a in law['weight_fractions']]
    require(len(x) == len(w) == 64 and all(math.isfinite(a) for a in x)
            and all(a >= 0 for a in w) and sum(w, Fraction(0)) == 1, 'Invalid original support or probability simplex')
    active = [(a, b) for a, b in zip(x, w) if b > 0]
    # Independent energy identity, with the 1/2 double expectation written
    # as unordered source-index pairs. Producer uses step-CDF squared intervals.
    first = math.fsum(float(b) * abs(a - y) for a, b in active)
    spread = math.fsum(float(wi * wj) * abs(xi - xj)
                      for i, (xi, wi) in enumerate(active) for xj, wj in active[:i])
    crps = first - spread
    left = sum((b for a, b in active if a < y), Fraction(0))
    right = sum((b for a, b in active if a <= y), Fraction(0))
    mid = (left + right) / 2
    ordered = sorted(active)
    def quantile(p):
        cumulative = Fraction(0)
        for value, mass in ordered:
            cumulative += mass
            if cumulative >= p:
                return value
        raise AssertionError('Quantile not bracketed')
    median = quantile(Fraction(1, 2))
    intervals = {}
    for level in LEVELS:
        p = Fraction(level)
        lo, hi = quantile((1 - p) / 2), quantile((1 + p) / 2)
        intervals[level] = {'lower': lo, 'upper': hi, 'width': hi - lo, 'covered': lo <= y <= hi}
    return {'CRPS_native': crps, 'squared_error_native': (y - median) ** 2,
            'predictive_median': median, 'PIT_left': float(left), 'PIT_right': float(right), 'PIT_midpoint': float(mid),
            'PIT_exact_fractions': {'left': str(left), 'right': str(right), 'midpoint': str(mid)}, 'intervals': intervals}


def summary(rows, method):
    out = {'formula_groups': 192}
    require(len(rows) == 192, '192formula groups required per method/mask')
    mean = lambda seq: math.fsum(seq) / 192
    out['equal_group_mean_CRPS_native'] = mean([r['CRPS_native'] for r in rows])
    out['RMSE_of_predictive_median_native'] = math.sqrt(mean([r['squared_error_native'] for r in rows]))
    for name in ('PIT_left', 'PIT_right', 'PIT_midpoint'):
        out['mean_' + name] = mean([r[name] for r in rows])
    out['coverage'] = {level: sum(r['intervals'][level]['covered'] for r in rows) / 192 for level in LEVELS}
    out['mean_interval_width_native'] = {level: mean([r['intervals'][level]['width'] for r in rows]) for level in LEVELS}
    if method == METHODS[0]:
        u = sorted(r['PIT_midpoint'] for r in rows)
        out['continuous_PIT_ECDF_deviation_descriptive'] = max(max((i + 1) / 192 - x, x - i / 192) for i, x in enumerate(u))
    else:
        out['continuous_PIT_ECDF_deviation_descriptive'] = None
    return out


def verify_summary(saved, own, checks):
    checks.exact('summary_group_count', saved['formula_groups'], 192)
    for key in ('equal_group_mean_CRPS_native', 'RMSE_of_predictive_median_native', 'mean_PIT_left', 'mean_PIT_right', 'mean_PIT_midpoint'):
        checks.number('summary_' + key, saved[key], own[key])
    for name in ('coverage', 'mean_interval_width_native'):
        for level in LEVELS:
            checks.number('summary_' + name, saved[name][level], own[name][level])
    key = 'continuous_PIT_ECDF_deviation_descriptive'
    if own[key] is None:
        checks.exact('discrete_PIT_no_continuous_uniform_gate', saved[key], None)
    else:
        checks.number('continuous_PIT_ECDF_D', saved[key], own[key], atol=2e-12, rtol=2e-12)


def audit(plan, checks):
    loaded = {}
    for name, pin in plan['inputs'].items():
        require(sha(pin['file']) == pin['sha256'], 'Pinned input changed: ' + name)
        loaded[name] = read(pin['file'])
    baseplan, freeze = loaded['baseline_plan'], loaded['forecast_freeze']
    require(freeze['status'] == 'COMPLETE_ALL1536_BASELINE_LAWS_FROZEN'
            and freeze['plan_sha256'] == plan['inputs']['baseline_plan']['sha256'], 'Complete baseline freeze required')
    require(loaded['snapshot_audit']['status'] == 'PASS', 'Saved model-state independent audit must pass first')
    for name, digest in freeze['files'].items():
        require(sha(Path(plan['forecast_directory']) / name) == digest, 'Frozen forecast payload changed')
    for name, digest in baseplan['source_sha256'].items():
        require(sha(Path(plan['baseline_code_directory']) / name) == digest, 'Original model/scoring source changed')
    start, result = loaded['score_start'], loaded['baseline_result']
    require(start['all1536_baseline_laws_verified_before_saved_truth_numeric_read'] is True
            and start['forecast_freeze_sha256'] == plan['inputs']['forecast_freeze']['sha256']
            and start['plan_sha256'] == plan['inputs']['baseline_plan']['sha256']
            and result['forecast_freeze_sha256'] == plan['inputs']['forecast_freeze']['sha256']
            and result['plan_sha256'] == plan['inputs']['baseline_plan']['sha256'], 'Saved-score provenance mismatch')
    require(datetime.fromisoformat(freeze['actual_utc']) <= datetime.fromisoformat(start['actual_utc'])
            <= datetime.fromisoformat(result['actual_end_utc']), 'Score occurred before complete forecast freeze')
    require(result['status'] == 'COMPLETE_POSTHOC_FIXED_LEARNING_BASELINE_CHECK'
            and result['formula_groups_fit'] == 64 and result['formula_groups_consumed_diagnostic_development'] == 192
            and result['new_independent_confirmation_groups'] == 0 and result['actual_baseline_fits'] == 6
            and result['original_main_model_or_four_gates_modified'] is False, 'Baseline scientific scope changed')
    roles = loaded['roles']['representatives']['diagnostic']
    truth = loaded['truth']['truth']
    require(len(roles) == len(truth) == 192, 'Complete192truth and identity role rows required')
    target = {}
    for row, identity in zip(truth, roles):
        require(row['identity'] == identity and math.isfinite(row['native_target']), 'Truth provenance mismatch')
        target[identity['component_id']] = row['native_target']
    require(len(target) == 192, 'Duplicate truth group')
    forecasts = loaded['forecasts']['forecasts']
    inventory = {(r['component_id'], m, k) for r in roles for m in METHODS for k in MASKS}
    laws = {(r['component_id'], r['method'], r['mask']): r for r in forecasts}
    scores = {(r['component_id'], r['method'], r['mask']): r for r in result['per_forecast_scores']}
    require(len(forecasts) == len(laws) == len(result['per_forecast_scores']) == len(scores) == len(inventory) == 1536
            and set(laws) == set(scores) == inventory, 'All1536unique forecast-score identities required')
    ownrows = {(m, k): [] for m in METHODS for k in MASKS}
    maxquad = 0.
    for identity in roles:
        group, jid = identity['component_id'], identity['jid']
        y = target[group]
        for method in METHODS:
            for mask in MASKS:
                f, saved = laws[(group, method, mask)], scores[(group, method, mask)]
                checks.exact('saved_forecast_jid', f['jid'], jid)
                checks.exact('saved_score_jid', saved['jid'], jid)
                checks.number('saved_native_target', saved['native_target'], y, atol=0, rtol=0)
                own = gaussian(y, f['law']) if method == METHODS[0] else empirical(y, f['law'])
                value = saved['score']
                for field in ('CRPS_native', 'squared_error_native', 'predictive_median', 'PIT_left', 'PIT_right', 'PIT_midpoint'):
                    checks.number(('GP_CDF_integral_' if method == METHODS[0] else 'RF_energy_') + field, value[field], own[field])
                if method == METHODS[0]:
                    maxquad = max(maxquad, own['quad_error_scaled'])
                    checks.number('GP_original_PIT', value['PIT'], own['PIT'], atol=2e-12, rtol=2e-12)
                    checks.exact('GP_PIT_kind', value['PIT_kind'], 'continuous normal')
                else:
                    checks.exact('RF_exact_PIT_fractions', value['PIT_exact_fractions'], own['PIT_exact_fractions'])
                    checks.exact('RF_no_sampling', value['sampled_distribution_or_randomized_PIT'], False)
                    checks.exact('RF_PIT_kind', value['PIT_kind'], 'discrete interval; midpoint is not continuous-uniform PIT')
                for level in LEVELS:
                    for field in ('lower', 'upper', 'width'):
                        checks.number('native_interval_' + field, value['intervals'][level][field], own['intervals'][level][field])
                    checks.exact('interval_coverage_event', value['intervals'][level]['covered'], own['intervals'][level]['covered'])
                ownrows[(method, mask)].append(own)
    summaries = {}
    for method in METHODS:
        summaries[method] = {}
        for mask in MASKS:
            v = summary(ownrows[(method, mask)], method)
            verify_summary(result['summary'][method][mask], v, checks)
            summaries[method][mask] = v
    c = result['candidate_comparison']
    frozen = loaded['candidate_result']['summary']['yj_joint_t']['both']
    require(frozen['formula_groups'] == 192, 'Same candidate development cohort required')
    candidate = frozen['equal_group_mean_CRPS_native']
    added = {m: summaries[m]['both']['equal_group_mean_CRPS_native'] for m in METHODS}
    best = min(added.values())
    checks.number('frozen_candidate_mean_only', c['candidate_fixed_frozen_YJ_dual_CRPS'], candidate, atol=0, rtol=0)
    for method in METHODS:
        checks.number('added_baseline_both_mean', c['added_baseline_dual_CRPS'][method], added[method])
    checks.number('candidate_to_best_ratio', c['candidate_to_best_added_baseline_ratio'], candidate / best)
    checks.number('candidate_minus_best', c['candidate_minus_best_added_baseline_native'], candidate - best)
    for key in ('read_existing_candidate_summary_only_no_model_or_score_recalculation',
                'does_not_reopen_or_change_original_phase2_four_gates',
                'posthoc_development_check_not_new_independent_confirmation_or_all_SOTA_guarantee'):
        checks.exact(key, c[key], True)
    bound = loaded['root_scope']['descriptive_competitiveness_screen']['candidate_to_best_new_baseline_CRPS_ratio_max_for_competitive_description']
    return {'status': 'PASS', 'all_saved_forecast_scores_checked': 1536, 'normal_native_CDF_integrals': 768,
            'empirical_native_energy_scores': 768, 'interval_events_checked': 4608, 'method_mask_summaries': 8,
            'max_Gaussian_CDF_quadrature_error_estimate_native': maxquad, 'recomputed_summary': summaries,
            'frozen_candidate_mean_native': candidate, 'candidate_to_best_added_baseline_ratio': candidate / best,
            'root_fixed_descriptive_threshold': bound, 'within_root_fixed_descriptive_threshold': candidate / best <= bound,
            'original_admission_four_gates_changed': False, 'new_material_groups_or_true_measurements': 0,
            'scope': 'Only saved forecast distribution scores and saved reference mean; no original model rescoring, fit, bootstrap, optimizer, SUT imports, causal policy or calibration claim.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--approval', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    plan, approval = read(args.plan), read(args.approval)
    source = sha(__file__)
    require(plan['script_sha256'] == source, 'Independent score audit source changed')
    require(approval.get('approved') is True and approval['scope'] == 'INDEPENDENT_FIXED_BASELINE_SCORES_ONCE'
            and approval['plan_sha256'] == sha(args.plan) and approval['script_sha256'] == source,
            'Exact root score audit approval required')
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    start = datetime.now(timezone.utc).isoformat()
    checks = Checks()
    caught = []
    try:
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter('always')
            result = audit(plan, checks)
            caught = [{'category': type(w.message).__name__, 'message': str(w.message)} for w in records]
    except Exception as error:
        result = {'status': 'FAIL', 'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()}
    result.update({'start_utc': start, 'end_utc': datetime.now(timezone.utc).isoformat(), 'script_sha256': source,
                   'plan_sha256': sha(args.plan), 'approval_sha256': sha(args.approval),
                   'counts': checks.counts, 'max_absolute_errors': checks.errors, 'warnings': caught,
                   'producer_sklearn_imports_or_training_replays': 0,
                   'raw_source_reads': 0, 'new_material_labels': 0, 'random_samples': 0})
    target = output / 'INDEPENDENT_FIXED_BASELINE_SCORE_AUDIT.json'
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'status': result['status'], 'file': str(target), 'sha256': sha(target)}))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())

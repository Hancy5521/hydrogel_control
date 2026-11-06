"""All3840 native-coordinate CDF/SF score checks; no producer import/replay."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from statistics import NormalDist
import sys
import time
import traceback
import warnings

sys.dont_write_bytecode = True
NORMAL = NormalDist()
MODELS = ('raw_joint_t', 'raw_momentmatched_Gaussian', 'yj_joint_t',
          'yj_latent_momentmatched_Gaussian', 'yj_target_marginal')
MASKS = ('none', 'e_form', 'gap_opt', 'both')
LEVELS = (.8, .9, .95)


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
    return json.loads(Path(path).read_text('utf-8-sig'), object_pairs_hook=unique,
                      parse_constant=invalid)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


class Checks:
    def __init__(self):
        self.counts, self.max_errors = {}, {}

    def check(self, name, actual, expected, atol=2e-9, rtol=2e-11, bound=None):
        require(type(actual) in (int, float) and math.isfinite(actual)
                and math.isfinite(expected), name + ' nonfinite comparison')
        error = abs(actual - expected)
        limit = atol + rtol * abs(expected) if bound is None else bound
        require(error <= limit, f'{name}: actual={actual}, expected={expected}, error={error}, bound={limit}')
        self.counts[name] = self.counts.get(name, 0) + 1
        self.max_errors[name] = max(self.max_errors.get(name, 0.), error)
        return error


def crps_tolerance(value):
    return 5e-7 + 2e-8 * abs(value)


def load_own_yj(plan):
    entry = plan['independent_YJ_helper']
    require(sha(entry['file']) == entry['sha256'], 'Own scalarYJ helper changed')
    spec = importlib.util.spec_from_file_location('independent_scalar_yj_for_native_audit', entry['file'])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def functions(law):
    if law['family'] == 'normal':
        require(law['df'] is None, 'Normal df must be null')
        cdf = lambda t: .5 * math.erfc(-t / math.sqrt(2))
        sf = lambda t: .5 * math.erfc(t / math.sqrt(2))
        ppf = lambda p: NORMAL.inv_cdf(p)
        logpdf = lambda t: -.5 * t * t - .5 * math.log(2 * math.pi)
    else:
        df = law['df']
        require(law['family'] == 'student_t' and math.isfinite(df) and df > 4,
                'Frozen Studentt df must allow first and second native moments')
        cdf = lambda t: float(special.stdtr(df, t))
        sf = lambda t: float(special.stdtr(df, -t))
        ppf = lambda p: float(special.stdtrit(df, p))
        constant = math.lgamma((df + 1) / 2) - math.lgamma(df / 2) - .5 * math.log(df * math.pi)
        logpdf = lambda t: constant - .5 * (df + 1) * math.log1p(t * t / df)
    return cdf, sf, ppf, logpdf


def integrate(fn, lower, upper, cuts=()):
    points = [lower] + sorted({x for x in cuts if lower < x < upper}) + [upper]
    values, errors = [], []
    for lo, hi in zip(points[:-1], points[1:]):
        value, error = quad(fn, lo, hi, epsabs=2e-10, epsrel=2e-10, limit=400)
        require(math.isfinite(value) and math.isfinite(error) and value >= 0 and error >= 0,
                'NativeCDF quadrature must be finite/nonnegative')
        values.append(value)
        errors.append(error)
    return math.fsum(values), math.fsum(errors)


def native_cdf_score(y, law, own):
    """Integrate in native affine coordinate. Never use inverse gprime/Jacobian."""
    mu, scale = law['location'], law['scale']
    require(all(math.isfinite(v) for v in (y, mu, scale)) and scale > 0, 'Finite law/outcome required')
    cdf, sf, ppf, logpdf = functions(law)
    if law['coordinate'] == 'native':
        # Native x=mu+scale*v: only a constant affine changeofvariable.
        target = (y - mu) / scale
        outer = scale
        cdf_native_coordinate = cdf
        sf_native_coordinate = sf
        cuts = (-mu / scale,)  # Native x=0, as well as the target split.
        method = 'native-x CDF integral via affine x=mu+scale*v'
    else:
        require(law['coordinate'] == 'u=YJ_half(native/s0)' and law['lambda'] == .5,
                'Only the frozen monotone YJ-half law is supported')
        outer = law['native_first_scale_s0']
        require(math.isfinite(outer) and outer > 0, 'Positive native first scale required')
        target = y / outer
        # Native x=s0*v. F_X(s0*v)=F_base((T(v)-muU)/sigmaU).
        # No inverse derivative and no producer weighted-z integrator appear.
        cdf_native_coordinate = lambda v: cdf((own.transform(v) - mu) / scale)
        sf_native_coordinate = lambda v: sf((own.transform(v) - mu) / scale)
        cuts = (0.,)
        method = 'native-x CDF integral via affine x=s0*v; forward T inside CDF only'
    left, le = integrate(lambda v: cdf_native_coordinate(v) ** 2, -math.inf, target, cuts)
    right, re = integrate(lambda v: sf_native_coordinate(v) ** 2, target, math.inf, cuts)
    return outer * (left + right), outer * (le + re), method


def independent_probability_score(y, law, own):
    mu, scale = law['location'], law['scale']
    cdf, sf, ppf, logpdf = functions(law)
    if law['coordinate'] == 'native':
        t = (y - mu) / scale
        median = mu
        jac = 0.
        quantile = lambda p: mu + scale * ppf(p)
    else:
        s0 = law['native_first_scale_s0']
        v = y / s0
        t = (own.transform(v) - mu) / scale
        median = s0 * own.inverse(mu)
        jac = (-.5 * math.log1p(v) if v >= 0 else .5 * math.log1p(-v)) - math.log(s0)
        quantile = lambda p: s0 * own.inverse(mu + scale * ppf(p))
    intervals = {}
    for level in LEVELS:
        lower, upper = quantile((1 - level) / 2), quantile((1 + level) / 2)
        require(math.isfinite(lower) and math.isfinite(upper) and upper > lower, 'Finite ordered quantiles required')
        intervals[str(level)] = {'lower': lower, 'upper': upper, 'width': upper - lower,
                                 'covered': lower <= y <= upper}
    return {'point_forecast': median, 'squared_error_native': (y - median) ** 2,
            'PIT': cdf(t), 'log_abs_native_to_u_Jacobian_at_target': jac,
            'log_density_native': logpdf(t) - math.log(scale) + jac, 'intervals': intervals}


def summary_all(independent):
    summary = {}
    for kind in MODELS:
        summary[kind] = {}
        for mask in MASKS:
            rows = [v for k, v in independent.items() if k[1:] == (kind, mask)]
            require(len(rows) == 192, 'All192 group scores required')
            pit = sorted(x['PIT'] for x in rows)
            deviation = max(max((i + 1) / 192 - u, u - i / 192) for i, u in enumerate(pit))
            summary[kind][mask] = {
                'equal_group_mean_CRPS_native': math.fsum(x['CRPS_native'] for x in rows) / 192,
                'RMSE_of_predictive_median_native': math.sqrt(math.fsum(x['squared_error_native'] for x in rows) / 192),
                'PIT_empirical_CDF_max_deviation': deviation,
                'coverage': {str(level): sum(x['intervals'][str(level)]['covered'] for x in rows) / 192 for level in LEVELS},
                'mean_interval_width_native': {str(level): math.fsum(x['intervals'][str(level)]['width'] for x in rows) / 192 for level in LEVELS}}
    return summary


def audit(plan, checks):
    own = load_own_yj(plan)
    base = Path(plan['phase2_directory'])
    original_path = base / 'EXECUTION_PLAN_REGISTERED.json'
    require(sha(original_path) == plan['phase2_plan_sha256'], 'Actual scientificplan changed')
    original = read(original_path)
    for name, pin in plan['producer_source_sha256'].items():
        require(sha(base / 'code' / name) == pin, 'Phase2 source changed')
    for name, pin in plan['parent_source_sha256'].items():
        require(sha(base.parent / 'admission' / 'code' / name) == pin, 'Parent source changed')
    auditfile = Path(plan['saved_forecast_audit']['file'])
    require(sha(auditfile) == plan['saved_forecast_audit']['sha256']
            and read(auditfile)['status'] == 'PASS', 'Independent3840 forecast PASS required')
    rolefile = Path(original['inputs']['roles']['file'])
    require(sha(rolefile) == plan['role_sha256'], 'Role file changed')
    roles = read(rolefile)['representatives']['diagnostic']
    require(len(roles) == 192, '192diagnostic metadata roles required')
    directory, scoredir = base / 'forecast_once', base / 'score_once'
    require(sha(directory / 'FORECAST_FREEZE.json') == plan['forecast_freeze_sha256'], 'Forecast freeze changed')
    freeze = read(directory / 'FORECAST_FREEZE.json')
    require(freeze['forecast_count'] == 3840 and freeze['diagnostic_target_model_access'] == 0
            and freeze['source_sha256'] == plan['producer_source_sha256'], 'Invalid3840 forecast freeze')
    for name, pin in plan['forecast_artifact_sha256'].items():
        require(sha(directory / name) == pin == freeze['files'][name], 'Frozen forecast artifact changed')
    for name, pin in plan['score_artifact_sha256'].items():
        require(sha(scoredir / name) == pin, 'Frozen score artifact changed')
    require(sha(plan['pretarget_extent']['file']) == plan['pretarget_extent']['sha256'], 'Pretarget full-inventory extent changed')
    extent = read(plan['pretarget_extent']['file'])
    reveal = read(scoredir / 'TARGET_REVEAL_START.json')
    require(extent['all_forecast_score_coordinates'] == 3840
            and extent['comparison_abs'] == 5e-7 and extent['comparison_rel'] == 2e-8
            and datetime.fromisoformat(extent['actual_registered_utc']) <= datetime.fromisoformat(reveal['actual_utc'])
            and datetime.fromisoformat(freeze['actual_utc']) <= datetime.fromisoformat(reveal['actual_utc']),
            'Full3840 scope or truth timing invalid')
    require(reveal['forecast_freeze_sha256'] == plan['forecast_freeze_sha256'], 'Reveal freeze binding changed')
    forecasts = read(directory / 'FORECASTS.json')['forecasts']
    keyed = {(x['component_id'], x['model'], x['mask']): x for x in forecasts}
    wanted = {(x['component_id'], m, k) for x in roles for m in MODELS for k in MASKS}
    require(len(forecasts) == len(keyed) == 3840 and set(keyed) == wanted, 'Forecast inventory omissions/duplicates')
    truthrows = read(scoredir / 'DIAGNOSTIC_TRUTH.json')['truth']
    require(len(truthrows) == 192 and all(x['identity'] == m for x, m in zip(truthrows, roles)), 'Truth identity binding invalid')
    truth = {x['identity']['component_id']: x['native_target'] for x in truthrows}
    saved = read(scoredir / 'RESULTS.json')
    require(datetime.fromisoformat(saved['actual_end_utc']) >= datetime.fromisoformat(reveal['actual_utc']),
            'Score completion precedes target reveal')
    require(saved['plan_sha256'] == plan['phase2_plan_sha256']
            and saved['source_sha256'] == plan['producer_source_sha256']
            and saved['parent_source_sha256'] == plan['parent_source_sha256']
            and saved['forecast_freeze_sha256'] == plan['forecast_freeze_sha256'], 'Score source/plan bindings changed')
    savedrows = saved['per_forecast_scores']
    stored = {(x['component_id'], x['model'], x['mask']): x for x in savedrows}
    require(len(savedrows) == len(stored) == 3840 and set(stored) == wanted, 'Score inventory omissions/duplicates')
    require(saved['actual_NIW_fits'] == 2 and saved['forecast_law_roles'] == 5
            and saved['independent_confirmation_groups'] == 0, 'Training/role/evidence count misstatement')
    fixed = plan['fixed40_spots']
    require(fixed['diagnostic_role_indices'] == [0, 191]
            and fixed['component_ids'] == [roles[i]['component_id'] for i in (0, 191)], 'Previously fixed40spot metadata changed')
    spotgroups = set(fixed['component_ids'])
    independent, cache, errors, spots = {}, {}, [], []
    for number, f in enumerate(forecasts, 1):
        key = f['component_id'], f['model'], f['mask']
        y, law = truth[key[0]], f['law']
        row = stored[key]
        require(row['jid'] == f['jid'] and row['native_target'] == y, 'Score JID/native truth mismatch')
        cachekey = json.dumps([y, law], sort_keys=True, separators=(',', ':'), allow_nan=False)
        if cachekey not in cache:
            integral, error, method = native_cdf_score(y, law, own)
            cache[cachekey] = (len(cache), integral, error, method)
            if len(cache) % 128 == 0:
                print(json.dumps({'audit_progress_unique_integrals': len(cache), 'coordinates_seen': number,
                                  'actual_utc': datetime.now(timezone.utc).isoformat()}), flush=True)
        cache_index, integral, error, method = cache[cachekey]
        value = independent_probability_score(y, law, own)
        value['CRPS_native'] = integral
        independent[key] = value
        s = row['score']
        crps_error = checks.check('all3840_native_CDF_CRPS', s['CRPS_native'], integral, 5e-7, 2e-8)
        require(s['point_forecast_definition'] == 'native_predictive_median', 'Median point role mislabeled')
        for name in ('point_forecast', 'squared_error_native', 'PIT', 'log_density_native',
                     'log_abs_native_to_u_Jacobian_at_target'):
            checks.check(name, s[name], value[name])
        for level in ('0.8', '0.9', '0.95'):
            for name in ('lower', 'upper', 'width'):
                checks.check('interval_' + name, s['intervals'][level][name], value['intervals'][level][name])
            require(s['intervals'][level]['covered'] == value['intervals'][level]['covered'], 'Coverage Boolean mismatch')
        item = {'component_id': key[0], 'model': key[1], 'mask': key[2],
                'independent_native_CDF_CRPS': integral, 'producer_score_absolute_error': crps_error,
                'unique_integral_index': cache_index, 'reference_quad_error_estimate': error,
                'coordinate_method': method}
        errors.append(item)
        if key[0] in spotgroups:
            spots.append(item)
    require(len(independent) == 3840 and len(spots) == 40, 'Full3840 and fixed40coverage required')
    for group in truth:
        marginal = keyed[(group, 'yj_joint_t', 'none')]['law']
        require(all(keyed[(group, 'yj_target_marginal', k)]['law'] == marginal for k in MASKS), 'Marginal forecast identity failed')
    summary = summary_all(independent)
    for kind in MODELS:
        for mask in MASKS:
            actual, expect = saved['summary'][kind][mask], summary[kind][mask]
            require(actual['formula_groups'] == 192, 'Formula-group summary denominator changed')
            checks.check('summary_native_CRPS', actual['equal_group_mean_CRPS_native'], expect['equal_group_mean_CRPS_native'], 5e-7, 2e-8)
            for name in ('RMSE_of_predictive_median_native', 'PIT_empirical_CDF_max_deviation'):
                checks.check('summary_' + name, actual[name], expect[name])
            for level in ('0.8', '0.9', '0.95'):
                require(actual['coverage'][level] == expect['coverage'][level], 'Summary coverage count mismatch')
                checks.check('summary_interval_width', actual['mean_interval_width_native'][level], expect['mean_interval_width_native'][level])
    refs = ('raw_joint_t', 'raw_momentmatched_Gaussian', 'yj_latent_momentmatched_Gaussian')
    d, n = summary['yj_joint_t']['both'], summary['yj_joint_t']['none']
    loss, none = d['equal_group_mean_CRPS_native'], n['equal_group_mean_CRPS_native']
    reflosses = {m: summary[m]['both']['equal_group_mean_CRPS_native'] for m in refs}
    best = min(reflosses.values())
    criteria = {'conditional_gain_at_least2percent': loss <= .98 * none,
                'within2percent_all_same_information_strong_references': loss <= 1.02 * best,
                'dual90coverage_at_least0p80': d['coverage']['0.9'] >= .8,
                'dual_PIT_ECDF_deviation_at_most0p15': d['PIT_empirical_CDF_max_deviation'] <= .15}
    require(saved['admission']['criteria'] == criteria, 'Four-gate Boolean mismatch')
    status = 'PASS_FOR_SEPARATE_FEEDBACK_TASK_DESIGN_ONLY' if all(criteria.values()) else 'NO_GO_STOP_FULL_CAMPAIGN_NO_TUNING'
    require(saved['admission']['status'] == status, 'Four-gate decision mismatch')
    for m in refs:
        checks.check('gate_reference_CRPS', saved['admission']['same_information_reference_losses'][m], reflosses[m], 5e-7, 2e-8)
    checks.check('gate_ratio_none', saved['admission']['candidate_to_own_none_ratio'], loss / none, 5e-7, 2e-8)
    checks.check('gate_ratio_best', saved['admission']['candidate_to_strongest_reference_ratio'], loss / best, 5e-7, 2e-8)
    pairedrows = saved['per_group_paired_differences']
    paired = {x['component_id']: x for x in pairedrows}
    require(len(pairedrows) == len(paired) == 192 and set(paired) == set(truth), 'Paired group inventory changed')
    for group in truth:
        base_score = independent[(group, 'yj_joint_t', 'none')]['CRPS_native']
        dual_score = independent[(group, 'yj_joint_t', 'both')]['CRPS_native']
        for mask in MASKS:
            c = independent[(group, 'yj_joint_t', mask)]['CRPS_native']
            checks.check('paired_gain', paired[group]['YJ_conditional_gain_vs_own_none'][mask], base_score - c,
                         bound=crps_tolerance(base_score) + crps_tolerance(c))
        for m in refs:
            c = independent[(group, m, 'both')]['CRPS_native']
            checks.check('paired_reference_difference', paired[group]['dual_YJ_minus_same_information_references'][m], dual_score - c,
                         bound=crps_tolerance(dual_score) + crps_tolerance(c))
    return {'status': 'PASS', 'all_native_integral_coordinates': 3840,
            'unique_native_CDF_integrals_executed': len(cache), 'exact_law_outcome_cache_hits': 3840 - len(cache),
            'fixed40_spots': spots, 'all3840_integral_coordinate_results': errors,
            'independent_summaries': summary, 'admission_criteria': criteria, 'admission_status': status,
            'actual_formula_groups': 192, 'interval_coverage_booleans': 11520,
            'summary_model_masks': 20, 'paired_gain_and_reference_coordinates': 1344,
            'scope': 'Every3840 coordinate compared to independent native-xCDF/SFintegral; only exactlaw+outcome caching. No inverse-Jacobian CRPS integrator, SUT import/fit/score replay, new measurement or independent material confirmation.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--approval', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    plan, approval = read(args.plan), read(args.approval)
    source = sha(__file__)
    require(plan['script_sha256'] == source, 'Frozen independent audit source changed')
    require(approval.get('approved') is True and approval['scope'] == 'INDEPENDENT_PHASE2_NATIVE_SCORE_ALL_COORDINATES_ONCE'
            and approval['plan_sha256'] == sha(args.plan) and approval['script_sha256'] == source,
            'Exact root approval of all3840 nativeCDF audit required')
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    start = datetime.now(timezone.utc).isoformat()
    global special, quad
    import scipy
    from scipy import special
    from scipy.integrate import quad, IntegrationWarning
    warnings.simplefilter('error', IntegrationWarning)
    checks = Checks()
    try:
        result = audit(plan, checks)
    except Exception as error:
        result = {'status': 'FAIL', 'type': type(error).__name__, 'message': str(error),
                  'traceback': traceback.format_exc()}
    result.update({'start_utc': start, 'end_utc': datetime.now(timezone.utc).isoformat(),
                   'script_sha256': source, 'plan_sha256': sha(args.plan), 'approval_sha256': sha(args.approval),
                   'numeric_comparison_counts': checks.counts, 'max_absolute_errors': checks.max_errors,
                   'scipy': scipy.__version__, 'Python': sys.version,
                   'producer_imports_or_calls': 0, 'raw_gzip_or_unselected_response_reads': 0,
                   'original_statistical_fit_or_score_replays': 0, 'random_samples': 0})
    path = output / 'INDEPENDENT_PHASE2_ALL_NATIVE_SCORE_AUDIT.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'status': result['status'], 'result': str(path), 'sha256': sha(path),
                      'unique_native_integrals': result.get('unique_native_CDF_integrals_executed')}))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())

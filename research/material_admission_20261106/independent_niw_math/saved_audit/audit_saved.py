"""Two approved stages over saved artifacts; no SUT import or native gzip read."""
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

MODELS = ('niw_joint_t', 'gaussian_momentmatched', 'independent_target_marginal')
MASKS = {'none': (), 'e_form': (0,), 'gap_opt': (1,), 'both': (0, 1)}
LEVELS = (0.8, 0.9, 0.95)
NORMAL = NormalDist()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    def unique(pairs):
        result = {}
        for k, v in pairs:
            if k in result:
                raise ValueError('Duplicate JSON key: ' + k)
            result[k] = v
        return result
    def invalid(x):
        raise ValueError('Nonfinite JSON token: ' + x)
    return json.loads(Path(path).read_text('utf-8-sig'),
                      object_pairs_hook=unique, parse_constant=invalid)


def require(value, message):
    if not value:
        raise AssertionError(message)


def flat(value):
    if isinstance(value, list):
        return [x for row in value for x in flat(row)]
    return [value]


class Checks:
    def __init__(self):
        self.counts = {}
        self.max_errors = {}

    def numeric(self, name, actual, expected, atol=2e-9, rtol=2e-11):
        a, e = flat(actual), flat(expected)
        require(len(a) == len(e), name + ' shape mismatch')
        for x, y in zip(a, e):
            require(type(x) in (int, float) and math.isfinite(x)
                    and math.isfinite(y), name + ' nonfinite numeric value')
            error = abs(x - y)
            require(error <= atol + rtol * abs(y),
                    f'{name}: actual={x}, expected={y}, absolute_error={error}')
            self.counts[name] = self.counts.get(name, 0) + 1
            self.max_errors[name] = max(self.max_errors.get(name, 0.0), error)


def build_reference(rows):
    # Independent scalar fsum sufficient statistics; no producer or NumPy fit.
    n = len(rows)
    require(n == 64 and all(len(x) == 3 for x in rows), '64x3 fit required')
    center = [math.fsum(x[j] for x in rows) / n for j in range(3)]
    sd = [math.sqrt(math.fsum((x[j] - center[j]) ** 2 for x in rows) / (n - 1))
          for j in range(3)]
    require(all(math.isfinite(v) and v > 0 for v in sd), 'Positive fit-only SD required')
    z = [[(x[j] - center[j]) / sd[j] for j in range(3)] for x in rows]
    zbar = [math.fsum(x[j] for x in z) / n for j in range(3)]
    lam = [[(3.0 if i == j else 0.0)
            + math.fsum((x[i] - zbar[i]) * (x[j] - zbar[j]) for x in z)
            + (64.0 / 65.0) * zbar[i] * zbar[j]
            for j in range(3)] for i in range(3)]
    location = [64.0 * x / 65.0 for x in zbar]
    shape = [[66.0 * x / (65.0 * 69.0) for x in row] for row in lam]
    covariance = [[x * 69.0 / 67.0 for x in row] for row in shape]
    return {'center': center, 'sd': sd, 'lambda': lam, 'location': location,
            'shape': shape, 'covariance': covariance, 'df': 69.0}


def solve_small(matrix, vector):
    if len(vector) == 1:
        require(matrix[0][0] > 0, 'Positive scalar covariance required')
        return [vector[0] / matrix[0][0]]
    require(len(vector) == 2, 'Only frozen0/1/2proxy scopes permitted')
    a, b = matrix[0]
    c, d = matrix[1]
    det = a * d - b * c
    require(det > 0, 'SPD two-proxy block required')
    return [(d * vector[0] - b * vector[1]) / det,
            (a * vector[1] - c * vector[0]) / det]


def reference_law(ref, kind, mask, proxy):
    observed = () if kind == 'independent_target_marginal' else MASKS[mask]
    matrix = ref['covariance'] if kind == 'gaussian_momentmatched' else ref['shape']
    loc, variance, df = ref['location'][2], matrix[2][2], ref['df']
    if observed:
        block = [[matrix[i][j] for j in observed] for i in observed]
        cross = [matrix[2][i] for i in observed]
        residual = [(proxy[i] - ref['center'][i]) / ref['sd'][i]
                    - ref['location'][i] for i in observed]
        solved = solve_small(block, residual)
        loc += math.fsum(a * b for a, b in zip(cross, solved))
        projection = solve_small(block, cross)
        variance -= math.fsum(a * b for a, b in zip(cross, projection))
        if kind != 'gaussian_momentmatched':
            delta = math.fsum(a * b for a, b in zip(residual, solved))
            variance *= (df + delta) / (df + len(observed))
            df += len(observed)
    require(variance > 0, 'Positive conditional scale required')
    return {'family': 'normal' if kind == 'gaussian_momentmatched' else 'student_t',
            'location': ref['center'][2] + ref['sd'][2] * loc,
            'scale': ref['sd'][2] * math.sqrt(variance),
            'df': None if kind == 'gaussian_momentmatched' else df}


def common(plan, checks):
    base = Path(plan['admission_directory'])
    admission = read(base / 'EXECUTION_PLAN_REGISTERED.json')
    require(sha(base / 'EXECUTION_PLAN_REGISTERED.json') == plan['admission_plan_sha256'], 'Admission plan changed')
    for name, pin in plan['producer_source_sha256'].items():
        require(sha(base / 'code' / name) == pin, 'Producer source changed')
    roles_path = Path(admission['inputs']['roles']['file'])
    require(sha(roles_path) == plan['role_sha256'], 'Role source changed')
    roles = read(roles_path)
    require(len(roles['representatives']['fit']) == 64
            and len(roles['representatives']['diagnostic']) == 192, 'Exact roles required')
    directory = base / 'forecast_once'
    freeze = read(directory / 'FORECAST_FREEZE.json')
    require(sha(directory / 'FORECAST_FREEZE.json') == plan['forecast_freeze_sha256'], 'Forecast freeze changed')
    require(freeze['status'] == 'COMPLETE_ALL_FORECASTS_FROZEN'
            and freeze['forecast_count'] == 2304
            and freeze['diagnostic_target_model_access'] == 0, 'Incomplete or target-exposed forecast freeze')
    require(freeze['source_sha256'] == plan['producer_source_sha256']
            and freeze['plan_sha256'] == plan['admission_plan_sha256'], 'Freeze source/plan mismatch')
    require(set(freeze['files']) == set(plan['forecast_artifact_sha256']), 'Exact five artifacts required')
    for name, pin in plan['forecast_artifact_sha256'].items():
        require(sha(directory / name) == pin == freeze['files'][name], 'Frozen artifact mismatch: ' + name)
    start = read(directory / 'EXECUTION_START.json')
    timestamps = [admission['registered_actual_utc'], start['actual_utc'], freeze['actual_utc']]
    dates = [datetime.fromisoformat(x) for x in timestamps]
    require(all(x.tzinfo is not None for x in dates)
            and dates[0] <= dates[1] <= dates[2], 'Plan/start/freeze chronology invalid')
    forecasts = read(directory / 'FORECASTS.json')['forecasts']
    keyed = {(x['component_id'], x['model'], x['mask']): x for x in forecasts}
    wanted = {(x['component_id'], m, k) for x in roles['representatives']['diagnostic']
              for m in MODELS for k in MASKS}
    require(len(forecasts) == len(keyed) == 2304 and set(keyed) == wanted,
            'Complete unique2304 identity/model/mask keys required')
    expected_jid = {x['component_id']: x['jid'] for x in roles['representatives']['diagnostic']}
    require(all(x['jid'] == expected_jid[x['component_id']] for x in forecasts), 'JID binding mismatch')
    return roles, forecasts, keyed, directory


def forecast_stage(plan, checks):
    roles, forecasts, keyed, directory = common(plan, checks)
    visible = read(directory / 'VISIBLE_INPUTS.json')
    require(set(visible) == {'fit', 'diagnostic'}, 'Unexpected visible role')
    for role, value_key in (('fit', 'native_values'), ('diagnostic', 'native_proxies')):
        items = visible[role]
        expected = roles['representatives'][role]
        require(len(items) == len(expected), 'Visible role count changed')
        for row, meta in zip(items, expected):
            require(set(row) == {'identity', value_key} and row['identity'] == meta,
                    'Visible schema/identity or unexpected target field')
            values = row[value_key]
            require(len(values) == (3 if role == 'fit' else 2)
                    and all(type(x) in (int, float) and math.isfinite(x) for x in values), 'Invalid visible values')
    ref = build_reference([x['native_values'] for x in visible['fit']])
    saved = read(directory / 'MODEL.json')
    checks.numeric('fit_center', saved['affine_center'], ref['center'])
    checks.numeric('fit_sample_SD', saved['affine_scale_sample_SD'], ref['sd'])
    checks.numeric('posterior_location', saved['posterior']['location'], ref['location'], 2e-12)
    checks.numeric('predictive_location', saved['predictive']['location'], ref['location'], 2e-12)
    checks.numeric('posterior_Lambda', saved['posterior']['Lambda'], ref['lambda'], 2e-11)
    checks.numeric('predictive_shape', saved['predictive']['scale_matrix_not_covariance'], ref['shape'], 2e-12)
    checks.numeric('predictive_covariance', saved['predictive']['covariance'], ref['covariance'], 2e-12)
    require(saved['posterior']['kappa'] == 65 and saved['posterior']['nu'] == 71
            and saved['predictive']['df'] == 69 and saved['diagnostic_rows_used_in_fit'] == 0,
            'NIW count/df/diagnostic fit scope mismatch')
    require(saved['fit_rows'] == 64 and saved['dimension'] == 3
            and saved['prior']['m0'] == [0.0, 0.0, 0.0]
            and saved['prior']['kappa0'] == 1 and saved['prior']['nu0'] == 7,
            'Empirical-reference prior or fit dimension changed')
    checks.numeric('prior_Psi', saved['prior']['Lambda0'],
                   [[3.0 if i == j else 0.0 for j in range(3)] for i in range(3)], 0, 0)
    for row in visible['diagnostic']:
        group = row['identity']['component_id']
        for kind in MODELS:
            for mask in MASKS:
                expected = reference_law(ref, kind, mask, row['native_proxies'])
                actual = keyed[(group, kind, mask)]['law']
                require(actual['family'] == expected['family'] and actual['df'] == expected['df'], 'Forecast family/df mismatch')
                checks.numeric('forecast_native_location', actual['location'], expected['location'])
                checks.numeric('forecast_native_scale', actual['scale'], expected['scale'])
        marginal = keyed[(group, 'niw_joint_t', 'none')]['law']
        require(all(keyed[(group, 'independent_target_marginal', k)]['law'] == marginal for k in MASKS),
                'Marginal identity control failed')
    return {'checked_all_forecasts': 2304, 'formula_groups': 192,
            'independent_fit_reconstruction': 'scalar fsum and explicit1x1/2x2 solves, no SUT',
            'forbidden_data_opened': [], 'diagnostic_target_reads': 0,
            'native_gzip_reads': 0, 'scope': 'Saved forecasts only; no target scoring or material-effectiveness conclusion.'}


def distribution_functions(law):
    mu, sd = law['location'], law['scale']
    if law['family'] == 'normal':
        cdf = lambda z: 0.5 * math.erfc(-z / math.sqrt(2))
        sf = lambda z: 0.5 * math.erfc(z / math.sqrt(2))
        ppf = lambda p: NORMAL.inv_cdf(p)
    else:
        df = law['df']
        cdf = lambda z: float(special.stdtr(df, z))
        sf = lambda z: float(special.stdtr(df, -z))
        ppf = lambda p: float(special.stdtrit(df, p))
    return mu, sd, cdf, sf, ppf


def independent_score(y, law):
    mu, sd, cdf, sf, ppf = distribution_functions(law)
    z = abs((y - mu) / sd)
    if law['family'] == 'normal':
        density = math.exp(-z * z / 2) / math.sqrt(2 * math.pi)
        crps = sd * (z * (2 * cdf(z) - 1) + 2 * density - 1 / math.sqrt(math.pi))
    else:
        df = law['df']
        density = math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2)
                          - .5 * math.log(df * math.pi)
                          - (df + 1) / 2 * math.log1p(z * z / df))
        lb1 = math.lgamma(.5) + math.lgamma(df - .5) - math.lgamma(df)
        lb2 = math.lgamma(.5) + math.lgamma(df / 2) - math.lgamma((df + 1) / 2)
        constant = 2 * math.sqrt(df) / (df - 1) * math.exp(lb1 - 2 * lb2)
        crps = sd * (z * (2 * cdf(z) - 1) + 2 * density * (df + z * z) / (df - 1) - constant)
    intervals = {}
    for level in LEVELS:
        lo = mu + sd * ppf((1 - level) / 2)
        hi = mu + sd * ppf((1 + level) / 2)
        intervals[str(level)] = {'lower': lo, 'upper': hi, 'width': hi - lo,
                                 'covered': lo <= y <= hi}
    return {'CRPS_native': crps, 'squared_error_native': (y - mu) ** 2,
            'PIT': cdf((y - mu) / sd), 'intervals': intervals}


def cdf_crps_quad(y, law):
    mu, sd, cdf, sf, ppf = distribution_functions(law)
    z = (y - mu) / sd
    left, le = quad(lambda x: cdf(x) ** 2, -math.inf, z,
                    epsabs=2e-10, epsrel=2e-10, limit=400)
    right, re = quad(lambda x: sf(x) ** 2, z, math.inf,
                     epsabs=2e-10, epsrel=2e-10, limit=400)
    return sd * (left + right), sd * (le + re)


def summarize(independent):
    result = {}
    for kind in MODELS:
        result[kind] = {}
        for mask in MASKS:
            rows = [r for key, r in independent.items() if key[1:] == (kind, mask)]
            require(len(rows) == 192, 'Complete192 score values required')
            u = sorted(x['PIT'] for x in rows)
            deviation = max(max((i + 1) / 192 - x, x - i / 192) for i, x in enumerate(u))
            result[kind][mask] = {
                'formula_groups': 192,
                'equal_group_mean_CRPS_native': math.fsum(x['CRPS_native'] for x in rows) / 192,
                'RMSE_native': math.sqrt(math.fsum(x['squared_error_native'] for x in rows) / 192),
                'PIT_empirical_CDF_max_deviation': deviation,
                'coverage': {str(v): sum(x['intervals'][str(v)]['covered'] for x in rows) / 192 for v in LEVELS},
                'mean_interval_width_native': {str(v): math.fsum(x['intervals'][str(v)]['width'] for x in rows) / 192 for v in LEVELS}}
    return result


def score_stage(plan, approval, checks):
    roles, forecasts, keyed, directory = common(plan, checks)
    prior_audit = Path(approval['forecast_audit_result']['file'])
    require(sha(prior_audit) == approval['forecast_audit_result']['sha256']
            and read(prior_audit)['status'] == 'PASS', 'Approved forecast audit PASS required')
    scoredir = Path(plan['admission_directory']) / 'score_once'
    bindings = approval['score_artifact_sha256']
    require(set(bindings) == {'DIAGNOSTIC_TRUTH.json', 'RESULTS.json', 'TARGET_REVEAL_START.json'}, 'Exact3score artifacts required')
    for name, pin in bindings.items():
        require(sha(scoredir / name) == pin, 'Frozen score input changed: ' + name)
    reveal = read(scoredir / 'TARGET_REVEAL_START.json')
    freeze = read(directory / 'FORECAST_FREEZE.json')
    require(datetime.fromisoformat(reveal['actual_utc']) >= datetime.fromisoformat(freeze['actual_utc'])
            and reveal['forecast_freeze_sha256'] == plan['forecast_freeze_sha256'], 'Truth-before-freeze violation')
    truthrows = read(scoredir / 'DIAGNOSTIC_TRUTH.json')['truth']
    expected = roles['representatives']['diagnostic']
    require(len(truthrows) == 192 and all(x['identity'] == m for x, m in zip(truthrows, expected)), 'Truth role/identity mismatch')
    truth = {x['identity']['component_id']: x['native_target'] for x in truthrows}
    saved = read(scoredir / 'RESULTS.json')
    require(saved['plan_sha256'] == plan['admission_plan_sha256']
            and saved['source_sha256'] == plan['producer_source_sha256']
            and saved['forecast_freeze_sha256'] == plan['forecast_freeze_sha256'], 'Score bindings invalid')
    savedrows = saved['per_forecast_scores']
    stored = {(x['component_id'], x['model'], x['mask']): x for x in savedrows}
    require(len(savedrows) == len(stored) == 2304 and set(stored) == set(keyed), 'Score coordinate omissions/duplicates')
    independent = {}
    for key, forecast in keyed.items():
        y = truth[key[0]]
        require(stored[key]['jid'] == forecast['jid'] and stored[key]['native_target'] == y, 'Score truth/JID mismatch')
        value = independent_score(y, forecast['law'])
        independent[key] = value
        s = stored[key]['score']
        for name in ('CRPS_native', 'squared_error_native', 'PIT'):
            checks.numeric('score_' + name, s[name], value[name])
        for level in ('0.8', '0.9', '0.95'):
            for name in ('lower', 'upper', 'width'):
                checks.numeric('interval_' + name, s['intervals'][level][name], value['intervals'][level][name])
            require(s['intervals'][level]['covered'] == value['intervals'][level]['covered'], 'Coverage Boolean mismatch')
    summary = summarize(independent)
    for kind in MODELS:
        for mask in MASKS:
            s, e = saved['summary'][kind][mask], summary[kind][mask]
            require(s['formula_groups'] == 192, 'Group count mismatch')
            for name in ('equal_group_mean_CRPS_native', 'RMSE_native', 'PIT_empirical_CDF_max_deviation'):
                checks.numeric('summary_' + name, s[name], e[name])
            for v in ('0.8', '0.9', '0.95'):
                require(s['coverage'][v] == e['coverage'][v], 'Summary coverage count mismatch')
                checks.numeric('summary_mean_width', s['mean_interval_width_native'][v], e['mean_interval_width_native'][v])
    dual, none, gaussian = [summary[k][m] for k, m in
        [('niw_joint_t', 'both'), ('niw_joint_t', 'none'), ('gaussian_momentmatched', 'both')]]
    ratio0 = dual['equal_group_mean_CRPS_native'] / none['equal_group_mean_CRPS_native']
    ratioG = dual['equal_group_mean_CRPS_native'] / gaussian['equal_group_mean_CRPS_native']
    criteria = {'at_least_2percent_CRPS_gain_vs_own_no_proxy': dual['equal_group_mean_CRPS_native'] <= .98 * none['equal_group_mean_CRPS_native'],
                'within_2percent_conditional_Gaussian_CRPS': dual['equal_group_mean_CRPS_native'] <= 1.02 * gaussian['equal_group_mean_CRPS_native'],
                'dual_90_coverage_at_least_0p80': dual['coverage']['0.9'] >= .8,
                'dual_PIT_ECDF_max_deviation_at_most_0p15': dual['PIT_empirical_CDF_max_deviation'] <= .15}
    require(saved['admission']['criteria'] == criteria, 'Admission criteria mismatch')
    expected_status = 'PASS_FOR_SEPARATE_FEEDBACK_TASK_DESIGN_ONLY' if all(criteria.values()) else 'NO_GO_STOP_FULL_CAMPAIGN_NO_RETUNE'
    require(saved['admission']['status'] == expected_status, 'Admission decision mismatch')
    checks.numeric('admission_ratio_none', saved['admission']['CRPS_dual_to_none_ratio'], ratio0)
    checks.numeric('admission_ratio_Gaussian', saved['admission']['CRPS_dual_to_Gaussian_ratio'], ratioG)
    paired = {x['component_id']: x for x in saved['per_group_paired_transfer']}
    require(len(paired) == len(saved['per_group_paired_transfer']) == 192 and set(paired) == set(truth), 'Paired transfer identity mismatch')
    for group in truth:
        for mask in MASKS:
            gain = independent[(group, 'niw_joint_t', 'none')]['CRPS_native'] - independent[(group, 'niw_joint_t', mask)]['CRPS_native']
            checks.numeric('paired_gain', paired[group]['conditional_gain_vs_none_by_mask'][mask], gain)
    spots = []
    for role_index in (0, 191):
        group = expected[role_index]['component_id']
        for kind in MODELS:
            for mask in MASKS:
                key = group, kind, mask
                integral, estimate = cdf_crps_quad(truth[group], keyed[key]['law'])
                closed = independent[key]['CRPS_native']
                checks.numeric('24_native_unit_CDF_quad_spots', integral, closed, 5e-8, 2e-8)
                spots.append({'diagnostic_role_index': role_index, 'component_id': group, 'model': kind,
                              'mask': mask, 'native_unit_CDF_quad': integral, 'independent_closed': closed,
                              'reported_quad_error_not_interval_bound': estimate})
    return {'checked_independent_closed_scores': 2304, 'checked_interval_coverage_booleans': 6912,
            'summary_model_masks': 12, 'paired_gain_coordinates': 768,
            'CDF_quad_spots': spots, 'admission_criteria': criteria, 'admission_status': expected_status,
            'scope': 'All saved closed scores checked independently; only fixed24 CDF-integral coordinates, not2304integrals. No original fit/prediction/timing rerun or new material result.'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=('forecast', 'score'), required=True)
    ap.add_argument('--plan', required=True)
    ap.add_argument('--approval', required=True)
    ap.add_argument('--output-dir', required=True)
    args = ap.parse_args()
    plan, approval = read(args.plan), read(args.approval)
    selfsha = sha(__file__)
    require(plan['script_sha256'] == selfsha, 'Audit source changed')
    require(approval.get('approved') is True and approval['plan_sha256'] == sha(args.plan)
            and approval['script_sha256'] == selfsha
            and approval['scope'] == 'INDEPENDENT_SAVED_NIW_' + args.phase.upper() + '_ONCE', 'Exact phase approval required')
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    start = datetime.now(timezone.utc).isoformat()
    global quad, special
    if args.phase == 'score':
        from scipy import special
        from scipy.integrate import quad, IntegrationWarning
        warnings.simplefilter('error', IntegrationWarning)
    checks = Checks()
    try:
        details = forecast_stage(plan, checks) if args.phase == 'forecast' else score_stage(plan, approval, checks)
        result = {'status': 'PASS', 'details': details}
    except Exception as error:
        result = {'status': 'FAIL', 'error_type': type(error).__name__,
                  'message': str(error), 'traceback': traceback.format_exc()}
    result.update({'phase': args.phase, 'start_utc': start, 'end_utc': datetime.now(timezone.utc).isoformat(),
                   'script_sha256': selfsha, 'plan_sha256': sha(args.plan), 'approval_sha256': sha(args.approval),
                   'checked_numeric_counts': checks.counts, 'max_absolute_errors': checks.max_errors,
                   'producer_imports_or_calls': 0, 'native_gzip_reads': 0, 'random_samples': 0,
                   'new_model_fits_or_original_timing_runs': 0, 'Python': sys.version})
    target = out / ('INDEPENDENT_SAVED_NIW_' + args.phase.upper() + '_AUDIT.json')
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'status': result['status'], 'phase': args.phase, 'result': str(target), 'sha256': sha(target)}))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())

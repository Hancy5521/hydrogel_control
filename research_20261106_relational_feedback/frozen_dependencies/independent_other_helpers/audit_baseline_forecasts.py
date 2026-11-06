"""Saved fixed-GP/forest forecast arithmetic only; never import sklearn or SUT."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import struct
import traceback

METHODS = ('gpr_fixed_recipe', 'forest_all_original_leaf_cdf')
MASKS = {'none': (), 'e_form': (0,), 'gap_opt': (1,), 'both': (0, 1)}
FILES = ('EXECUTION_START.json', 'INPUT_SCOPE_RECEIPT.json', 'LIBRARY_VERSIONS.json',
         'MODEL_STATES.json', 'FORECASTS.json')


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


def require(value, message):
    if not value:
        raise AssertionError(message)


class Checks:
    def __init__(self):
        self.counts, self.max_errors = {}, {}

    def numeric(self, name, actual, expected, atol=2e-9, rtol=2e-11):
        require(math.isfinite(actual) and math.isfinite(expected), name + ' nonfinite comparison')
        error = abs(actual - expected)
        require(error <= atol + rtol * abs(expected),
                f'{name}: actual={actual}, expected={expected}, error={error}')
        self.counts[name] = self.counts.get(name, 0) + 1
        self.max_errors[name] = max(self.max_errors.get(name, 0.), error)
        return error


def norm_inf(matrix):
    return max(math.fsum(abs(x) for x in row) for row in matrix)


def scalar_rbf(a, b, constant, length):
    distance = math.fsum(((x - y) / ell) ** 2 for x, y, ell in zip(a, b, length))
    return constant * math.exp(-.5 * distance)


def forward_solve(lower, vector):
    result = []
    for i, row in enumerate(lower):
        require(math.isfinite(row[i]) and row[i] > 0, 'Positive Cholesky diagonal required')
        result.append((vector[i] - math.fsum(row[j] * result[j] for j in range(i))) / row[i])
    return result


def same_matrix(actual, expected, checks, name, exact=False):
    require(len(actual) == len(expected) and all(len(a) == len(e) for a, e in zip(actual, expected)), name + ' matrix shape')
    for a, e in zip(actual, expected):
        for x, y in zip(a, e):
            if exact:
                require(x == y, name + ' exact matrix mismatch')
            else:
                checks.numeric(name, x, y)


def float32(value):
    return struct.unpack('f', struct.pack('f', value))[0]


def audit_gp(state, mask, xfit, xdiag, yfit, laws, checks, criteria):
    require(state['feature_dtype'] == 'float64', 'GPR dtype changed')
    same_matrix(state['X_train_standardized_float64'], xfit, checks, 'GP_train_features')
    length, constant = state['RBF_length_scale'], state['constant_value']
    noise, alpha = state['white_noise_level'], state['training_alpha_nugget']
    require(len(length) == len(mask) and all(.01 * (1 - 1e-12) <= x <= 100 * (1 + 1e-12) for x in length)
            and .01 * (1 - 1e-12) <= constant <= 100 * (1 + 1e-12)
            and 1e-6 * (1 - 1e-12) <= noise <= 10 * (1 + 1e-12) and alpha == 1e-8,
            'Learned GP hyperparameters outside fixed recipe')
    wanted_theta = [math.log(constant)] + [math.log(x) for x in length] + [math.log(noise)]
    require(len(state['kernel_theta_log']) == len(wanted_theta), 'Kernel theta dimension changed')
    for actual, expected in zip(state['kernel_theta_log'], wanted_theta):
        checks.numeric('GP_log_theta', actual, expected)
    mean = math.fsum(yfit) / 64
    popsd = math.sqrt(math.fsum((y - mean) ** 2 for y in yfit) / 64)
    checks.numeric('GP_target_mean', state['y_train_mean'], mean)
    checks.numeric('GP_target_population_SD', state['y_train_std_population'], popsd)
    # Use the stored normalization after checking it independently. This avoids
    # replacing the saved fitted state with a new training/optimization run.
    ymean, ystd = state['y_train_mean'], state['y_train_std_population']
    normalized = [(y - ymean) / ystd for y in yfit]
    require(len(state['y_train_normalized']) == 64, 'Normalized fit target count')
    for actual, expected in zip(state['y_train_normalized'], normalized):
        checks.numeric('GP_normalized_fit_target', actual, expected)
    train = state['X_train_standardized_float64']
    kernel = [[scalar_rbf(train[i], train[j], constant, length)
               + (noise + alpha if i == j else 0.) for j in range(64)] for i in range(64)]
    lower, dual = state['L_training_cholesky'], state['alpha_dual_coefficients']
    require(len(lower) == len(dual) == 64 and all(len(row) == 64 for row in lower), 'GP state matrix shape')
    require(all(math.isfinite(x) for row in lower for x in row)
            and all(math.isfinite(x) for x in dual), 'GP saved factor/dual finite')
    require(all(lower[i][i] > 0 for i in range(64))
            and all(lower[i][j] == 0 for i in range(64) for j in range(i + 1, 64)), 'Lower triangular factor required')
    product = [[math.fsum(lower[i][k] * lower[j][k] for k in range(min(i, j) + 1))
                for j in range(64)] for i in range(64)]
    residual = [[kernel[i][j] - product[i][j] for j in range(64)] for i in range(64)]
    transpose = [[lower[j][i] for j in range(64)] for i in range(64)]
    factor_abs = norm_inf(residual)
    factor_backward = factor_abs / (norm_inf(kernel) + norm_inf(lower) * norm_inf(transpose))
    require(factor_backward <= criteria['GP']['K_equals_LLt_normwise_backward_error_max'], 'GP factor backward residual too large')
    response_residual = [math.fsum(kernel[i][j] * dual[j] for j in range(64)) - state['y_train_normalized'][i] for i in range(64)]
    dual_abs = max(abs(x) for x in response_residual)
    denominator = norm_inf(kernel) * max(abs(x) for x in dual) + max(abs(x) for x in normalized)
    dual_backward = dual_abs / denominator
    require(dual_backward <= criteria['GP']['Kalpha_equals_ynormalized_normwise_backward_error_max'], 'GP dual backward residual too large')
    for query, law in zip(xdiag, laws):
        cross = [scalar_rbf(query, row, constant, length) for row in train]
        predmean = ymean + ystd * math.fsum(k * a for k, a in zip(cross, dual))
        v = forward_solve(lower, cross)
        predvar = constant + noise - math.fsum(x * x for x in v)
        require(math.isfinite(predvar) and predvar > 0, 'Positive predictive variance with test White noise required')
        predstd = ystd * math.sqrt(predvar)
        require(law['family'] == 'normal' and law['df'] is None
                and law['predictive_WhiteKernel_noise_included'] is True, 'Incorrect GP forecast role')
        checks.numeric('GP_native_location', law['location'], predmean,
                       criteria['GP']['native_predictive_mean_std_abs'], criteria['GP']['native_predictive_mean_std_rel'])
        checks.numeric('GP_native_std', law['scale'], predstd,
                       criteria['GP']['native_predictive_mean_std_abs'], criteria['GP']['native_predictive_mean_std_rel'])
    return {'mask': list(mask), 'kernel_entries_rebuilt': 4096, 'native_laws_rebuilt': 192,
            'factor_residual_normInf': factor_abs, 'factor_normwise_backward_error': factor_backward,
            'dual_residual_normInf': dual_abs, 'dual_normwise_backward_error': dual_backward,
            'conservative_spectral_condition_upper_bound': (64 * constant + noise + alpha) / (noise + alpha),
            'not_a_global_hyperparameter_optimum_or_forward_error_guarantee': True,
            'fit_warnings_retained': state['fit_warnings_all_categories'],
            'predict_warnings_retained': state['predict_warnings_all_categories']}


def tree_leaf(tree, row):
    left, right, feature, threshold = (tree[k] for k in ('children_left', 'children_right', 'feature', 'threshold'))
    n = len(left)
    require(len(right) == len(feature) == len(threshold) == n and n > 0, 'Valid tree arrays required')
    node, steps = 0, 0
    while left[node] != -1:
        require(0 <= feature[node] < len(row) and 0 <= left[node] < n and 0 <= right[node] < n,
                'Invalid tree child or feature')
        node = left[node] if row[feature[node]] <= threshold[node] else right[node]
        steps += 1
        require(steps <= n, 'Cyclic tree state')
    require(right[node] == -1, 'Leaf sentinels inconsistent')
    return node


def audit_forest(state, xfit, xdiag, support, laws):
    require(state['tree_input_dtype'] == 'float32'
            and state['leaf_weights_include_OOB_original_rows'] is True
            and state['bootstrap_multiplicity_in_CDF_weights'] is False, 'Explicit forest CDF variant changed')
    train = [[float32(x) for x in row] for row in xfit]
    query = [[float32(x) for x in row] for row in xdiag]
    require(train == state['X_train_standardized_then_float32']
            and query == state['X_diagnostic_standardized_then_float32'], 'RF exact float32 input mismatch')
    trees = state['trees']
    require(len(trees) == 256 and [x['tree_id'] for x in trees] == list(range(256)), 'Exactly256 ordered trees required')
    weights = [[Fraction(0) for j in range(64)] for i in range(192)]
    for tree in trees:
        train_leaves = [tree_leaf(tree, row) for row in train]
        query_leaves = [tree_leaf(tree, row) for row in query]
        require(train_leaves == tree['original_train_leaf_ids']
                and query_leaves == tree['diagnostic_leaf_ids'], 'RF exact saved leaf path mismatch')
        members = defaultdict(list)
        for index, leaf in enumerate(train_leaves):
            members[leaf].append(index)
        for i, leaf in enumerate(query_leaves):
            require(leaf in members, 'Query leaf lacks original training support')
            add = Fraction(1, 256 * len(members[leaf]))
            for j in members[leaf]:
                weights[i][j] += add
    for wanted, law in zip(weights, laws):
        require(sum(wanted, Fraction(0)) == 1 and law['family'] == 'weighted_empirical'
                and law['support_native'] == support, 'Invalid forest native support or simplex')
        require([Fraction(x) for x in law['weight_fractions']] == wanted
                and law['weights_float_display'] == [float(x) for x in wanted], 'Exact RF leaf CDF mass mismatch')
    return {'trees_replayed': 256, 'exact_training_paths': 64 * 256,
            'exact_diagnostic_paths': 192 * 256, 'CDF_laws_rebuilt': 192,
            'exact_Fraction_mass_positions': 192 * 64,
            'fit_warnings_retained': state['fit_warnings_all_categories'],
            'scope': 'Replay frozen tree partitions and all-original-row CDF only, not rerun bootstrap training or officialQRF.'}


def audit(plan, checks):
    criteria_path = Path(plan['criteria']['file'])
    require(sha(criteria_path) == plan['criteria']['sha256'], 'Pre-state criteria changed')
    criteria = read(criteria_path)
    base = Path(plan['baseline_directory'])
    actual = base / 'EXECUTION_PLAN_REGISTERED.json'
    require(sha(actual) == plan['baseline_plan_sha256'], 'Actual baseline plan changed')
    original = read(actual)
    for name, pin in plan['source_sha256'].items():
        require(sha(base / 'code' / name) == pin, 'Baseline source changed')
    roles_path = Path(original['inputs']['phase2_roles']['file'])
    require(sha(roles_path) == plan['role_sha256'], 'Metadata roles changed')
    roles = read(roles_path)['representatives']
    visible_path = Path(original['inputs']['visible_inputs']['file'])
    require(sha(visible_path) == plan['visible_inputs_sha256'], 'Visible-input snapshot changed')
    visible = read(visible_path)
    require(set(visible) == {'fit', 'diagnostic'}, 'Only permitted visible inputs required')
    for name, key, n in (('fit', 'native_values', 64), ('diagnostic', 'native_proxies', 192)):
        require(len(visible[name]) == len(roles[name]) == n, 'Fixed role lengths required')
        for row, meta in zip(visible[name], roles[name]):
            require(row['identity'] == meta and set(row) == {'identity', key}, 'Visible metadata or target permission mismatch')
    fit = [x['native_values'] for x in visible['fit']]
    proxies = [x['native_proxies'] for x in visible['diagnostic']]
    support = [x[2] for x in fit]
    folder = base / 'forecast_once'
    require(sha(folder / 'FORECAST_FREEZE.json') == plan['forecast_freeze_sha256'], 'Baseline freeze changed')
    frozen = read(folder / 'FORECAST_FREEZE.json')
    require(frozen['status'] == 'COMPLETE_ALL1536_BASELINE_LAWS_FROZEN'
            and frozen['source_sha256'] == plan['source_sha256']
            and frozen['plan_sha256'] == plan['baseline_plan_sha256'], 'Invalid baseline forecast freeze')
    require(set(frozen['files']) == set(FILES) == set(plan['forecast_artifact_sha256']), 'Exact5baseline artifacts required')
    for name, pin in plan['forecast_artifact_sha256'].items():
        require(sha(folder / name) == pin == frozen['files'][name], 'Fitted snapshot artifact changed')
    start = read(folder / 'EXECUTION_START.json')
    times = [datetime.fromisoformat(x) for x in
             (criteria['actual_registered_utc'], original['registered_actual_utc'], start['actual_utc'], frozen['actual_utc'])]
    require(all(x.tzinfo is not None for x in times)
            and max(times[0], times[1]) <= times[2] <= times[3],
            'Pre-state numerical criteria or actual execution chronology changed')
    state = read(folder / 'MODEL_STATES.json')
    require(state['actual_fits'] == 6 and state['none_mask_fits'] == 0 and len(state['fits']) == 6,
            'Six fits and no none-mask fit required')
    fs = state['feature_transform']
    require(fs['fit_feature_values_native'] == [x[:2] for x in fit]
            and fs['diagnostic_feature_values_native'] == proxies
            and fs['fit_targets_native'] == support and fs['diagnostic_targets_visible_to_fit'] == 0,
            'Feature/target fit-only provenance mismatch')
    for j in (0, 1):
        mean = math.fsum(x[j] for x in fit) / 64
        sd = math.sqrt(math.fsum((x[j] - mean) ** 2 for x in fit) / 63)
        checks.numeric('feature_mean', fs['native_feature_center_fit64'][j], mean)
        checks.numeric('feature_sampleSD', fs['native_feature_sample_SD_fit64'][j], sd)
    center, scale = fs['native_feature_center_fit64'], fs['native_feature_sample_SD_fit64']
    xfit = [[(x[j] - center[j]) / scale[j] for j in (0, 1)] for x in fit]
    xdiag = [[(x[j] - center[j]) / scale[j] for j in (0, 1)] for x in proxies]
    forecasts = read(folder / 'FORECASTS.json')['forecasts']
    laws = {(x['component_id'], x['method'], x['mask']): x for x in forecasts}
    wanted = {(x['component_id'], m, k) for x in roles['diagnostic'] for m in METHODS for k in MASKS}
    require(len(forecasts) == len(laws) == 1536 and set(laws) == wanted, '1536complete law inventory required')
    require(all(laws[(x['component_id'], m, k)]['jid'] == x['jid'] for x in roles['diagnostic'] for m in METHODS for k in MASKS),
            'Forecast JID identity mismatch')
    fitted = {(x['method'], x['mask']): x for x in state['fits']}
    require(len(fitted) == 6 and set(fitted) == {(m, k) for m in METHODS for k in MASKS if k != 'none'},
            'Fitted method/mask inventory changed')
    gp_checks, forest_checks = [], []
    for method in METHODS:
        for mask_name, mask in MASKS.items():
            predictions = [laws[(x['component_id'], method, mask_name)]['law'] for x in roles['diagnostic']]
            if not mask:
                for law in predictions:
                    if method == 'gpr_fixed_recipe':
                        mean = math.fsum(support) / 64
                        sd = math.sqrt(math.fsum((x - mean) ** 2 for x in support) / 63)
                        require(law['family'] == 'normal' and law['df'] is None, 'GP fixed target marginal family')
                        checks.numeric('GP_none_location', law['location'], mean)
                        checks.numeric('GP_none_sampleSD', law['scale'], sd)
                    else:
                        require(law['family'] == 'weighted_empirical' and law['support_native'] == support
                                and [Fraction(x) for x in law['weight_fractions']] == [Fraction(1, 64)] * 64
                                and law['weights_float_display'] == [1 / 64] * 64,
                                'Forest uniform64 marginal mismatch')
                continue
            fit_entry = fitted[(method, mask_name)]
            require(fit_entry['feature_indices'] == list(mask), 'Mask feature indices changed')
            a, q = [[x[j] for j in mask] for x in xfit], [[x[j] for j in mask] for x in xdiag]
            if method == 'gpr_fixed_recipe':
                gp_checks.append(audit_gp(fit_entry['state'], mask, a, q, support, predictions, checks, criteria))
            else:
                forest_checks.append(dict(mask=mask_name, **audit_forest(fit_entry['state'], a, q, support, predictions)))
    return {'status': 'PASS', 'all_saved_forecast_coordinates': 1536, 'GP_learned_laws': 576,
            'forest_learned_CDF_laws': 576, 'none_mask_laws': 384, 'GP_state_checks': gp_checks,
            'forest_state_checks': forest_checks, 'RF_tree_paths_exact': 196608,
            'RF_learned_Fraction_mass_positions_exact': 36864,
            'RF_none_uniform_Fraction_mass_positions_exact': 12288,
            'posthoc_development_not_confirmation': True,
            'scope': 'Saved kernel/factor/dual and tree-state arithmetic only; no sklearn/SUT fit/predict, optimizer rerun, material outcome scoring or calibration guarantee.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--approval', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    plan, approval = read(args.plan), read(args.approval)
    source = sha(__file__)
    require(plan['script_sha256'] == source, 'Audit source changed')
    require(approval.get('approved') is True and approval['scope'] == 'INDEPENDENT_FIXED_BASELINE_SNAPSHOT_ONCE'
            and approval['plan_sha256'] == sha(args.plan) and approval['script_sha256'] == source,
            'Exact root snapshot audit approval required')
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    start = datetime.now(timezone.utc).isoformat()
    checks = Checks()
    try:
        result = audit(plan, checks)
    except Exception as error:
        result = {'status': 'FAIL', 'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()}
    result.update({'start_utc': start, 'end_utc': datetime.now(timezone.utc).isoformat(),
                   'script_sha256': source, 'plan_sha256': sha(args.plan), 'approval_sha256': sha(args.approval),
                   'numeric_comparison_counts': checks.counts, 'max_absolute_errors': checks.max_errors,
                   'producer_or_sklearn_imports': 0, 'original_training_prediction_or_timing_replays': 0,
                   'native_gzip_or_diagnostic_target_reads': 0, 'random_samples': 0})
    target = output / 'INDEPENDENT_FIXED_BASELINE_SNAPSHOT_AUDIT.json'
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'status': result['status'], 'result': str(target), 'sha256': sha(target)}))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())

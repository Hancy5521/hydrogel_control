"""Post-hoc fixed baselines on already-consumed development; freeze before scores."""
import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True

import baseline_models as models

CODE = ('baseline_models.py', 'run_baselines.py', 'score_baselines.py')
PARENT_SHA = {
    'admission_io.py': '90011d05e1071b02422622249006cf669bd88e2f7792f8da87f87e0589188466',
    'niw_predictive.py': 'a8d74e5398f2bb606efb195d3395324f014ba3f9b4f801a67b022c1cc6687818',
    'score_admission.py': 'ef7e29f8dab89a0a644011b8c188fcbeaa178095bbd823d082be8887e0aaccc0'}
PHASE2_FREEZE_SHA = 'dcd88d11401f7f8953e9b921bf3ab2a157740c937084b106e77c61732e806faf'
CANDIDATE_RESULT_SHA = '4bf07550b9b4840967b6cca13d4a81c190616e7a4bdb47813126793bde2b5711'
FREEZE_FILES = ('EXECUTION_START.json', 'INPUT_SCOPE_RECEIPT.json', 'LIBRARY_VERSIONS.json', 'MODEL_STATES.json', 'FORECASTS.json')
SCORE_ONLY_INPUTS = ('diagnostic_truth', 'candidate_results')


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    def invalid(value):
        raise ValueError('Nonstandard JSON constant: ' + value)
    def unique(pairs):
        result = {}
        for k, v in pairs:
            if k in result:
                raise ValueError('Duplicate JSON key: ' + k)
            result[k] = v
        return result
    return json.loads(Path(path).read_text('utf-8-sig'), parse_constant=invalid, object_pairs_hook=unique)


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def source_hashes():
    folder = Path(__file__).resolve().parent
    return {name: sha(folder / name) for name in CODE}


def parent_directory():
    return Path(__file__).resolve().parents[2] / 'admission' / 'code'


def parent_hashes():
    return {name: sha(parent_directory() / name) for name in PARENT_SHA}


def configuration():
    return {'version': 'posthoc_fixed_learning_baselines_v1', 'fit_rows': 64, 'diagnostic_development_rows': 192,
            'methods': list(models.METHODS), 'masks': [list(x) for x in models.MASKS],
            'mask_names': list(models.MASK_NAMES), 'nonempty_mask_fits_per_method': 3,
            'actual_total_fits': 6, 'none_mask_fits': 0, 'forecast_laws': 1536,
            'native_features': ['e_form', 'gap opt'], 'native_target': 'bulk modulus',
            'feature_transform': 'fit64 means and sampleSD in float64, no diagnostic values in fit',
            'GPR': {'normalize_y': True, 'alpha': 1e-8, 'optimizer': 'fmin_l_bfgs_b',
                    'n_restarts_optimizer': 0, 'random_state': 20261106,
                    'kernel': 'Constant(1,[.01,100])*RBF(ones(k),[.01,100])+White(.01,[1e-6,10])',
                    'feature_dtype': 'float64', 'test_noise': 'learned WhiteKernel included, alpha only training diagonal',
                    'none': 'fixed fit-target mean/sampleSD Gaussian, no GP fit'},
            'forest': {'n_estimators': 256, 'min_samples_leaf': 3, 'max_features': 1,
                       'bootstrap': True, 'criterion': 'squared_error', 'random_state': 20261106,
                       'n_jobs': 2, 'max_depth': None, 'tree_input_dtype': 'float32',
                       'CDF_weight_rule': 'average1/n_leaf_all_original; includesOOB, no bootstrap multiplicity',
                       'mass_arithmetic': 'exact Fraction over64original source target positions',
                       'none': 'uniform64empirical native target marginal, no fit',
                       'scope': 'explicit QRF-inspired all-original-leaf variant, not an official QRF package run'},
            'libraries': {'scikit-learn': '1.6.1', 'numpy': '2.1.3', 'scipy': '1.15.3', 'joblib': '1.4.2'},
            'coverage_levels': [.8, .9, .95], 'RMSE_point_forecast': 'predictive_median',
            'empirical_quantile': 'inf{x:F(x)>=p}, exact rational mass, source-value ties kept',
            'empirical_PIT': 'left/right/midpoint separately; no continuous uniformity gate',
            'candidate_comparison': 'read frozen phase2summary only, candidate/best-new both-mask mean nativeCRPS ratio',
            'posthoc_added_baselines_on_consumed_development_not_confirmation': True,
            'original_model_fit_score_or_four_gates_changed': False,
            'hyperparameter_sweep_library_upgrade_new_network_or_agent': False}


def validate(plan_file, approval_file, phase):
    plan, approval = read(plan_file), read(approval_file)
    keys = {'kind', 'status', 'registered_actual_utc', 'source_sha256', 'parent_source_sha256',
            'configuration', 'inputs', 'independent_score_audit_receipt'}
    if (set(plan) != keys or plan['kind'] != 'posthoc_fixed_learning_strong_baseline_check'
            or plan['status'] != 'REGISTERED' or not isinstance(plan['registered_actual_utc'], str)
            or not plan['registered_actual_utc'] or plan['source_sha256'] != source_hashes()
            or plan['parent_source_sha256'] != PARENT_SHA or parent_hashes() != PARENT_SHA
            or plan['configuration'] != configuration()):
        raise ValueError('Exact registered fixed baseline source/recipe required')
    if (approval.get('approved') is not True or approval.get('scope') != 'POSTHOC_FIXED_STRONG_BASELINES'
            or approval.get('plan_sha256') != sha(plan_file) or phase not in approval.get('allowed_phases', [])):
        raise ValueError('Root approval of exact plan and phase required')
    inputs = plan['inputs']
    if set(inputs) != {'visible_inputs', 'phase2_freeze', 'phase2_roles', 'phase2_registered_plan',
                      'diagnostic_truth', 'candidate_results', 'root_scope'}:
        raise ValueError('Pinned original frozen-input provenance and score-only references required')
    for name, item in inputs.items():
        if name not in SCORE_ONLY_INPUTS and sha(item['file']) != item['sha256']:
            raise ValueError('Baseline model-visible source/provenance changed')
    if inputs['phase2_freeze']['sha256'] != PHASE2_FREEZE_SHA or inputs['candidate_results']['sha256'] != CANDIDATE_RESULT_SHA:
        raise ValueError('Do not replace the already-frozen main-model baseline or candidate result')
    old_freeze = read(inputs['phase2_freeze']['file'])
    if (old_freeze['status'] != 'COMPLETE_ALL3840_FORECASTS_FROZEN'
            or old_freeze['files']['VISIBLE_INPUTS.json'] != inputs['visible_inputs']['sha256']
            or old_freeze['plan_sha256'] != inputs['phase2_registered_plan']['sha256']):
        raise ValueError('VISIBLE_INPUTS is not bound to the completed main forecast freeze')
    receipt = plan['independent_score_audit_receipt']
    if sha(receipt['file']) != receipt['sha256']:
        raise ValueError('Independent empirical/Gaussian score audit receipt changed')
    roles = read(inputs['phase2_roles']['file'])
    representatives = roles['representatives']
    if len(representatives['fit']) != 64 or len(representatives['diagnostic']) != 192:
        raise ValueError('Fixed original64/192roles required')
    allrows = representatives['fit'] + representatives['diagnostic']
    if len({r['component_id'] for r in allrows}) != 256:
        raise ValueError('Full formula groups overlap')
    return plan, roles


def model_visible_arrays(plan, roles):
    visible = read(plan['inputs']['visible_inputs']['file'])
    if set(visible) != {'fit', 'diagnostic'} or len(visible['fit']) != 64 or len(visible['diagnostic']) != 192:
        raise ValueError('Only original frozen fit/proxy snapshot allowed')
    for name in ('fit', 'diagnostic'):
        for record, meta in zip(visible[name], roles['representatives'][name]):
            if record['identity'] != meta or set(record) != {'identity', 'native_values' if name == 'fit' else 'native_proxies'}:
                raise ValueError('Visible row metadata or allowed response scope changed')
    fit = [r['native_values'] for r in visible['fit']]
    proxies = [r['native_proxies'] for r in visible['diagnostic']]
    return fit, proxies


def verify_freeze(directory, plan_file, roles):
    directory = Path(directory)
    receipt = read(directory / 'FORECAST_FREEZE.json')
    if (receipt['status'] != 'COMPLETE_ALL1536_BASELINE_LAWS_FROZEN' or receipt['source_sha256'] != source_hashes()
            or receipt['parent_source_sha256'] != PARENT_SHA or receipt['plan_sha256'] != sha(plan_file)
            or set(receipt['files']) != set(FREEZE_FILES)):
        raise ValueError('Complete fixed baseline freeze required before reading saved truth')
    for name, digest in receipt['files'].items():
        if sha(directory / name) != digest:
            raise ValueError('Frozen baseline artifact changed')
    forecasts = read(directory / 'FORECASTS.json')['forecasts']
    expected = {(r['component_id'], method, mask) for r in roles['representatives']['diagnostic']
                for method in models.METHODS for mask in models.MASK_NAMES}
    actual = [(r['component_id'], r['method'], r['mask']) for r in forecasts]
    if len(actual) != 1536 or len(set(actual)) != 1536 or set(actual) != expected:
        raise ValueError('Missing, duplicate or unexpected baseline forecast')
    state = read(directory / 'MODEL_STATES.json')
    if state['actual_fits'] != 6 or state['none_mask_fits'] != 0 or len(state['fits']) != 6:
        raise ValueError('Exactlysix fixed fits and no marginal fit required')
    native_support = state['feature_transform']['fit_targets_native']
    jid = {r['component_id']: r['jid'] for r in roles['representatives']['diagnostic']}
    for forecast in forecasts:
        law = forecast['law']
        if forecast['jid'] != jid[forecast['component_id']]:
            raise ValueError('Forecast JID misbound')
        if forecast['method'] == 'gpr_fixed_recipe':
            if law['family'] != 'normal' or law['df'] is not None or not all(type(law[k]) in (int, float) and math.isfinite(law[k]) for k in ('location', 'scale')) or law['scale'] <= 0:
                raise ValueError('Invalid raw-normal baseline law')
        else:
            weights = [Fraction(x) for x in law['weight_fractions']]
            if (law['family'] != 'weighted_empirical' or law['support_native'] != native_support
                    or len(weights) != 64 or any(w < 0 for w in weights) or sum(weights, Fraction(0)) != 1
                    or law['weights_float_display'] != [float(w) for w in weights]):
                raise ValueError('Invalid exact all-original empirical CDF law')
    return forecasts


def validate_score_only_inputs(plan):
    for name in SCORE_ONLY_INPUTS:
        item = plan['inputs'][name]
        if sha(item['file']) != item['sha256']:
            raise ValueError('Sealed saved truth or frozen candidate result changed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--approval', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    plan, roles = validate(args.plan, args.approval, 'forecast')
    versions = {name: importlib.metadata.version(name) for name in configuration()['libraries']}
    if versions != configuration()['libraries']:
        raise RuntimeError('Do not upgrade/change the pinned baseline runtime')
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'EXECUTION_START.json', {'actual_utc': utc(), 'source_sha256': source_hashes(),
            'parent_source_sha256': PARENT_SHA, 'plan_sha256': sha(args.plan), 'approval_sha256': sha(args.approval),
            'posthoc_baselines_on_consumed_development': True, 'main_model_or_gates_modified': False})
    try:
        write(output / 'LIBRARY_VERSIONS.json', {'packages': versions, 'python': sys.version, 'no_upgrade': True})
        fit, proxies = model_visible_arrays(plan, roles)
        write(output / 'INPUT_SCOPE_RECEIPT.json', {'actual_utc': utc(),
            'source': 'Already-frozen phase2 VISIBLE_INPUTS, no raw gzip or new group',
            'visible_inputs_sha256': plan['inputs']['visible_inputs']['sha256'],
            'fit_native_cells': 192, 'diagnostic_proxy_cells': 384,
            'diagnostic_truth_or_candidate_result_file_opened_by_forecast_process': False})
        laws, state = models.fit_and_forecast(fit, proxies)
        write(output / 'MODEL_STATES.json', state)
        forecasts = []
        for i, row in enumerate(roles['representatives']['diagnostic']):
            for method in models.METHODS:
                for mask in models.MASK_NAMES:
                    forecasts.append({'component_id': row['component_id'], 'jid': row['jid'],
                        'method': method, 'mask': mask, 'law': laws[method][mask][i]})
        if len(forecasts) != 1536:
            raise AssertionError('All1536laws must be frozen before any score')
        write(output / 'FORECASTS.json', {'posthoc_development_only': True, 'forecasts': forecasts})
        if source_hashes() != plan['source_sha256'] or parent_hashes() != PARENT_SHA:
            raise ValueError('Bound baseline or parent score source changed during fitting')
        write(output / 'FORECAST_FREEZE.json', {'status': 'COMPLETE_ALL1536_BASELINE_LAWS_FROZEN',
            'actual_utc': utc(), 'source_sha256': source_hashes(), 'parent_source_sha256': PARENT_SHA,
            'plan_sha256': sha(args.plan), 'files': {n: sha(output / n) for n in FREEZE_FILES},
            'forecast_count': 1536, 'actual_fits': 6, 'none_fits': 0,
            'no_main_model_refit_or_score_or_new_material_sample': True})
        print(json.dumps({'status': 'COMPLETE_ALL1536_BASELINE_LAWS_FROZEN', 'actual_fits': 6}))
    except Exception as error:
        write(output / 'FAILED_STOP.json', {'actual_utc': utc(), 'type': type(error).__name__,
            'message': str(error), 'retain_failure_no_recipe_search': True})
        raise


if __name__ == '__main__':
    main()

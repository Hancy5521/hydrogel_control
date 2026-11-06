"""Read-only phase2 forecast arithmetic audit; no producer or native truth IO."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import traceback

sys.dont_write_bytecode = True
MODELS = ('raw_joint_t', 'raw_momentmatched_Gaussian', 'yj_joint_t',
          'yj_latent_momentmatched_Gaussian', 'yj_target_marginal')
MASKS = ('none', 'e_form', 'gap_opt', 'both')
FIVE_FILES = ('EXECUTION_START.json', 'VISIBLE_INPUTS.json', 'EXTRACTION_RECEIPT.json',
              'FIT_REFERENCE.json', 'FORECASTS.json')


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


def require(value, message):
    if not value:
        raise AssertionError(message)


def helper(path, pin, name):
    require(sha(path) == pin, 'Independent reference helper changed')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compare_fitted(saved, ref, checks, name):
    for stored_key, key in (('affine_center', 'center'),
                            ('affine_scale_sample_SD', 'sd')):
        checks.numeric(name + '_' + key, saved[stored_key], ref[key])
    checks.numeric(name + '_posterior_location', saved['posterior']['location'], ref['location'], 2e-12)
    checks.numeric(name + '_predictive_location', saved['predictive']['location'], ref['location'], 2e-12)
    checks.numeric(name + '_posterior_Psi', saved['posterior']['Lambda'], ref['lambda'], 2e-11)
    checks.numeric(name + '_predictive_shape', saved['predictive']['scale_matrix_not_covariance'], ref['shape'], 2e-12)
    checks.numeric(name + '_predictive_covariance', saved['predictive']['covariance'], ref['covariance'], 2e-12)
    require(saved['fit_rows'] == 64 and saved['dimension'] == 3
            and saved['posterior']['kappa'] == 65 and saved['posterior']['nu'] == 71
            and saved['predictive']['df'] == 69 and saved['diagnostic_rows_used_in_fit'] == 0,
            'Fixed complete-row NIW reference counts/df changed')
    require(saved['prior']['m0'] == [0., 0., 0.]
            and saved['prior']['kappa0'] == 1 and saved['prior']['nu0'] == 7,
            'NIW prior changed')
    checks.numeric(name + '_prior_Psi', saved['prior']['Lambda0'],
                   [[3. if i == j else 0. for j in range(3)] for i in range(3)], 0, 0)


def audit(plan):
    own_ref = helper(plan['independent_NIW_helper']['file'],
                     plan['independent_NIW_helper']['sha256'], 'audit_independent_NIW_reference')
    own_yj = helper(plan['independent_YJ_helper']['file'],
                    plan['independent_YJ_helper']['sha256'], 'audit_independent_YJ_reference')
    checks = own_ref.Checks()
    base = Path(plan['phase2_directory'])
    registered = base / 'EXECUTION_PLAN_REGISTERED.json'
    require(sha(registered) == plan['phase2_plan_sha256'], 'Phase2 actual plan changed')
    original = read(registered)
    for name, pin in plan['producer_source_sha256'].items():
        require(sha(base / 'code' / name) == pin, 'Phase2 producer source changed')
    for name, pin in plan['parent_source_sha256'].items():
        require(sha(base.parent / 'admission' / 'code' / name) == pin, 'Parent source changed')
    require(original['source_sha256'] == plan['producer_source_sha256']
            and original['parent_source_sha256'] == plan['parent_source_sha256'], 'Actual plan source binding changed')
    role_file = Path(original['inputs']['roles']['file'])
    require(sha(role_file) == plan['role_sha256'], 'Role file changed')
    roles = read(role_file)
    require(len(roles['representatives']['fit']) == 64
            and len(roles['representatives']['diagnostic']) == 192, '64/192 roles required')
    directory = base / 'forecast_once'
    freeze_file = directory / 'FORECAST_FREEZE.json'
    require(sha(freeze_file) == plan['forecast_freeze_sha256'], 'Forecast freeze changed')
    frozen = read(freeze_file)
    require(frozen['status'] == 'COMPLETE_ALL3840_FORECASTS_FROZEN'
            and frozen['forecast_count'] == 3840 and frozen['diagnostic_target_model_access'] == 0
            and frozen['source_sha256'] == plan['producer_source_sha256']
            and frozen['parent_source_sha256'] == plan['parent_source_sha256']
            and frozen['plan_sha256'] == plan['phase2_plan_sha256'], 'Incomplete/invalid forecast freeze')
    require(set(frozen['files']) == set(FIVE_FILES) == set(plan['forecast_artifact_sha256']), 'Exact5forecast artifacts required')
    for name, pin in plan['forecast_artifact_sha256'].items():
        require(sha(directory / name) == pin == frozen['files'][name], 'Frozen snapshot byte mismatch: ' + name)
    start = read(directory / 'EXECUTION_START.json')
    approval_file = Path(plan['actual_forecast_approval']['file'])
    require(sha(approval_file) == plan['actual_forecast_approval']['sha256']
            == start['approval_sha256'], 'Actual successful forecast approval binding changed')
    forecast_approval = read(approval_file)
    require(forecast_approval['approved'] is True
            and forecast_approval['scope'] == 'PHASE2_FIXED_YJ_HALF_ADMISSION'
            and forecast_approval['plan_sha256'] == plan['phase2_plan_sha256'],
            'Incorrect actual forecast approval scope/plan')
    times = [datetime.fromisoformat(x) for x in
             (original['registered_actual_utc'], start['actual_utc'], frozen['actual_utc'])]
    require(all(x.tzinfo is not None for x in times) and times[0] <= times[1] <= times[2], 'Invalid plan/start/freeze chronology')
    visible = read(directory / 'VISIBLE_INPUTS.json')
    require(set(visible) == {'fit', 'diagnostic'}, 'No extra visible roles permitted')
    for role, key, count in (('fit', 'native_values', 3), ('diagnostic', 'native_proxies', 2)):
        require(len(visible[role]) == len(roles['representatives'][role]), 'Visible role count mismatch')
        for row, meta in zip(visible[role], roles['representatives'][role]):
            require(set(row) == {'identity', key} and row['identity'] == meta,
                    'Invalid visible identity/schema or diagnostic target field')
            require(len(row[key]) == count and all(type(x) in (int, float) and math.isfinite(x) for x in row[key]),
                    'Finite permitted visible cells required')
    fit = [r['native_values'] for r in visible['fit']]
    raw_ref = own_ref.build_reference(fit)
    s0 = raw_ref['sd']
    ufit = [[own_yj.transform(x[j] / s0[j]) for j in range(3)] for x in fit]
    u_ref = own_ref.build_reference(ufit)
    saved = read(directory / 'FIT_REFERENCE.json')
    require(saved['lambda'] == .5 and saved['actual_NIW_fits'] == 2
            and saved['fit_rows_per_reference'] == 64
            and saved['all_three_coordinates_transformed'] is True
            and saved['new_fit_parameters_selected_or_optimized'] == 0,
            'Fixed YJ-half two-pipeline scope mismatch')
    checks.numeric('native_first_scale_s0', saved['native_first_scale_s0'], s0)
    compare_fitted(saved['raw'], raw_ref, checks, 'raw')
    compare_fitted(saved['yj'], u_ref, checks, 'YJ')
    forecasts = read(directory / 'FORECASTS.json')['forecasts']
    keyed = {(r['component_id'], r['model'], r['mask']): r for r in forecasts}
    expected = {(r['component_id'], kind, mask) for r in roles['representatives']['diagnostic']
                 for kind in MODELS for mask in MASKS}
    require(len(forecasts) == len(keyed) == 3840 and set(keyed) == expected,
            'Missing/duplicate/extra phase2 forecast coordinates')
    base_names = {'raw_joint_t': 'niw_joint_t', 'raw_momentmatched_Gaussian': 'gaussian_momentmatched',
                  'yj_joint_t': 'niw_joint_t', 'yj_latent_momentmatched_Gaussian': 'gaussian_momentmatched',
                  'yj_target_marginal': 'independent_target_marginal'}
    for row in visible['diagnostic']:
        group, jid = row['identity']['component_id'], row['identity']['jid']
        proxies = row['native_proxies']
        uproxies = [own_yj.transform(proxies[j] / s0[j]) for j in (0, 1)]
        for kind in MODELS:
            israw = kind.startswith('raw_')
            for mask in MASKS:
                entry = keyed[(group, kind, mask)]
                require(entry['jid'] == jid, 'Wrong forecast JID')
                law = entry['law']
                reference = own_ref.reference_law(raw_ref if israw else u_ref,
                    base_names[kind], mask, proxies if israw else uproxies)
                require(law['family'] == reference['family'] and law['df'] == reference['df'], 'Forecast family/df mismatch')
                coordinate = 'native' if israw else 'u=YJ_half(native/s0)'
                require(law['coordinate'] == coordinate, 'Native/latent law coordinate confusion')
                checks.numeric('raw_native_location' if israw else 'YJ_u_location', law['location'], reference['location'])
                checks.numeric('raw_native_scale' if israw else 'YJ_u_scale', law['scale'], reference['scale'])
                if not israw:
                    require(law['lambda'] == .5 and law['latent_moment_matching_not_native_moment_matching'] is True,
                            'Nonlinear law scope mismatch')
                    checks.numeric('YJ_law_s0', law['native_first_scale_s0'], s0[2])
        marginal = keyed[(group, 'yj_joint_t', 'none')]['law']
        require(all(keyed[(group, 'yj_target_marginal', k)]['law'] == marginal for k in MASKS), 'YJ marginal identity control failed')
    return {'status': 'PASS', 'checked_laws': 3840, 'checked_native_raw_laws': 1536,
            'checked_transformed_u_laws': 2304, 'independent_reference_pipelines': 2,
            'diagnostic_formula_groups': 192, 'target_reads': 0, 'native_gzip_reads': 0,
            'per_numeric_counts': checks.counts, 'max_absolute_errors': checks.max_errors,
            'scope': 'Snapshot forecast arithmetic and transform/model byte bindings only. Not source algorithm replay, independent material confirmation, score or calibration result.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--approval', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    plan, approval = read(args.plan), read(args.approval)
    source = sha(__file__)
    require(plan['script_sha256'] == source, 'Audit source changed')
    require(approval.get('approved') is True and approval['scope'] == 'INDEPENDENT_PHASE2_SAVED_FORECAST_ONCE'
            and approval['plan_sha256'] == sha(args.plan) and approval['script_sha256'] == source,
            'Exact root phase2 forecast audit approval required')
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    start = datetime.now(timezone.utc).isoformat()
    try:
        result = audit(plan)
    except Exception as error:
        result = {'status': 'FAIL', 'type': type(error).__name__, 'message': str(error),
                  'traceback': traceback.format_exc()}
    result.update({'start_utc': start, 'end_utc': datetime.now(timezone.utc).isoformat(),
                   'script_sha256': source, 'plan_sha256': sha(args.plan), 'approval_sha256': sha(args.approval),
                   'producer_imports_or_calls': 0, 'native_source_or_diagnostic_target_reads': 0,
                   'random_samples': 0, 'original_fit_or_timing_replays': 0,
                   'independent_helper_source_pins': {'NIW': plan['independent_NIW_helper'], 'YJ': plan['independent_YJ_helper']}})
    path = output / 'INDEPENDENT_PHASE2_FORECAST_AUDIT.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'status': result['status'], 'result': str(path), 'sha256': sha(path)}))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())

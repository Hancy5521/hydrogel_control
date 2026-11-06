"""Phase2 source-pinned forecast freeze; no target scoring in this process."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True

import yj_model as adapter

CODE = ('yj_model.py', 'run_phase2.py', 'score_phase2.py')
RAW_SHA = '3d0fc1315b74cf84071f16aa25bb93434dafbc83867beb75872064ac245bd747'
PHASE1_ROLES_SHA = '0bc81d0400a4076c854af11e654ca3ca9bcbceb249f2ee6c26d356c53ceae9c0'
FROZEN_FILES = ('EXECUTION_START.json', 'VISIBLE_INPUTS.json', 'EXTRACTION_RECEIPT.json', 'FIT_REFERENCE.json', 'FORECASTS.json')


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    def invalid(value):
        raise ValueError('Nonstandard JSON constant: ' + value)
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError('Duplicate JSON key: ' + key)
            value[key] = item
        return value
    return json.loads(Path(path).read_text('utf-8-sig'), parse_constant=invalid, object_pairs_hook=unique)


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def source_hashes():
    folder = Path(__file__).resolve().parent
    return {name: sha(folder / name) for name in CODE}


def configuration():
    return {'version': 'phase2_fixed_YJ_half_reference_v1', 'dataset': 'jarvis_dft_3d',
            'new_fit_groups': 64, 'new_diagnostic_development_groups': 192,
            'old40_and_all_phase1_256_whole_components_excluded': True,
            'models': list(adapter.MODELS), 'masks': [list(x) for x in adapter.MASKS],
            'mask_names': list(adapter.MASK_NAMES), 'forecast_count': 3840, 'actual_NIW_fits': 2,
            'target_fields': ['e_form', 'gap opt', 'bulk modulus'], 'target_index': 2,
            'lambda': .5, 'same_fixed_lambda_all3fields': True,
            'transform': 'v=native/s0_fitSD; u=YJ_half(v); z=(u-fit_mean_u)/fit_SD_u',
            'prior': {'p': 3, 'm0': [0, 0, 0], 'kappa0': 1, 'nu0': 7, 'Lambda0_diagonal': [3, 3, 3]},
            'empirical_reference_not_native_scale_pure_Bayes': True,
            'own_row_proxy_conditioning_only': True, 'all_diag_groups_are_development': True,
            'raw_scores': 'unchanged parent closed CRPS',
            'yj_scores': 'deterministic native CDF_SF integral with nonconstant inverse Jacobian',
            'score_absolute_tolerance_native': 1e-8, 'score_relative_tolerance': 1e-8,
            'quad_subinterval_limit': 200, 'tail_initial_standard_cut': 16.,
            'tail_max_standard_cut': 1e6, 'tail_max_doublings': 16,
            'tail_each_absolute_budget_fraction': .125, 'tail_bounds_are_ideal_arithmetic_not_IEEE_intervals': True,
            'fixed_integral_subdivision_points': [-16., -8., -4., -1., 0., 1., 4., 8., 16.],
            'coverage_levels': [.8, .9, .95], 'RMSE_point_forecast': 'native_predictive_median_for_every_model',
            'gates': {'YJdual_to_own_none_CRPS_ratio_max': .98,
                      'YJdual_to_best_same_information_reference_CRPS_ratio_max': 1.02,
                      'references': ['raw_joint_t', 'raw_momentmatched_Gaussian', 'yj_latent_momentmatched_Gaussian'],
                      'YJdual_90_coverage_min': .8, 'YJdual_PIT_ECDF_max_deviation_max': .15},
            'PIT_threshold_is_descriptive_not_exact_KS_or_calibration': True,
            'no_lambda_prior_role_selection_or_threshold_tuning': True,
            'stage1_NO_GO_retained_and_not_rescored': True,
            'transported_latent_Gaussian_is_not_native_momentmatched_Gaussian': True,
            'readout_transport_or_star_moment_certificate_not_inherited': True,
            'new_wet_DFT_or_allocation_agent_calls': 0}


def validate(plan_file, approval_file, phase):
    plan, approval = read(plan_file), read(approval_file)
    expected = {'kind', 'status', 'registered_actual_utc', 'source_sha256', 'parent_source_sha256',
                'inputs', 'stable_sources_sha256', 'configuration', 'independent_math_receipts'}
    if (set(plan) != expected or plan['kind'] != 'phase2_fixed_YJ_half_material_admission'
            or plan['status'] != 'REGISTERED' or not isinstance(plan['registered_actual_utc'], str)
            or not plan['registered_actual_utc'] or plan['source_sha256'] != source_hashes()
            or plan['parent_source_sha256'] != adapter.PARENT_SHA or adapter.parent_hashes() != adapter.PARENT_SHA
            or plan['configuration'] != configuration()):
        raise ValueError('Exact registered phase2 source and configuration required')
    if (approval.get('approved') is not True or approval.get('scope') != 'PHASE2_FIXED_YJ_HALF_ADMISSION'
            or approval.get('plan_sha256') != sha(plan_file) or phase not in approval.get('allowed_phases', [])):
        raise ValueError('Root approval of this exact plan and process phase required')
    if set(plan['inputs']) != {'raw_source', 'roles', 'phase1_roles', 'contract', 'root_scope'}:
        raise ValueError('All phase2 roles, whole-component exclusions and scope must be bound')
    for name, item in plan['inputs'].items():
        if sha(item['file']) != item['sha256']:
            raise ValueError('Bound phase2 input changed: ' + name)
    if plan['inputs']['raw_source']['sha256'] != RAW_SHA or plan['inputs']['phase1_roles']['sha256'] != PHASE1_ROLES_SHA:
        raise ValueError('Original source and fully consumed phase1 roles must remain pinned')
    if len(plan['stable_sources_sha256']) != 16:
        raise ValueError('All sixteen stable source/provenance inputs required')
    for file, digest in plan['stable_sources_sha256'].items():
        if sha(file) != digest:
            raise ValueError('Stable source/provenance file changed')
    if set(plan['independent_math_receipts']) != {'niw_t', 'YJ_native_probability_and_score'}:
        raise ValueError('Both already-approved mathematical receipts must be pinned')
    for item in plan['independent_math_receipts'].values():
        if sha(item['file']) != item['sha256']:
            raise ValueError('Independent mathematical receipt changed')
    _, base_io, _ = adapter.parents()
    contract, roles = read(plan['inputs']['contract']['file']), read(plan['inputs']['roles']['file'])
    if contract['dataset']['raw_asset_sha256_from_prior_manifest_not_reread'] != RAW_SHA:
        raise ValueError('Contract source does not match')
    base_io.validate_roles(roles, contract)
    previous = read(plan['inputs']['phase1_roles']['file'])
    phase1 = {r['component_id'] for role in ('fit', 'diagnostic') for r in previous['representatives'][role]}
    current = {r['component_id'] for role in ('fit', 'diagnostic') for r in roles['representatives'][role]}
    if len(phase1) != 256 or len(current) != 256 or current & phase1:
        raise ValueError('Phase1 whole components must not enter new phase2 cohort')
    return plan, roles


def verify_freeze(directory, plan_file, roles):
    directory = Path(directory)
    receipt = read(directory / 'FORECAST_FREEZE.json')
    if (receipt['status'] != 'COMPLETE_ALL3840_FORECASTS_FROZEN' or receipt['plan_sha256'] != sha(plan_file)
            or receipt['source_sha256'] != source_hashes() or receipt['parent_source_sha256'] != adapter.PARENT_SHA
            or set(receipt['files']) != set(FROZEN_FILES)):
        raise ValueError('Complete3840 forecast freeze required before target extraction')
    for name, digest in receipt['files'].items():
        if sha(directory / name) != digest:
            raise ValueError('Frozen phase2 artifact changed')
    forecasts = read(directory / 'FORECASTS.json')['forecasts']
    expected = {(r['component_id'], kind, mask) for r in roles['representatives']['diagnostic']
                for kind in adapter.MODELS for mask in adapter.MASK_NAMES}
    keys = [(r['component_id'], r['model'], r['mask']) for r in forecasts]
    if len(keys) != 3840 or len(set(keys)) != 3840 or set(keys) != expected:
        raise ValueError('Missing, duplicated or unexpected forecasts')
    jid = {r['component_id']: r['jid'] for r in roles['representatives']['diagnostic']}
    for f in forecasts:
        law, kind = f['law'], f['model']
        if f['jid'] != jid[f['component_id']] or not all(type(law[k]) in (int, float) and math.isfinite(law[k]) for k in ('location', 'scale')) or law['scale'] <= 0:
            raise ValueError('Invalid frozen identity/location/scale')
        gaussian = kind in ('raw_momentmatched_Gaussian', 'yj_latent_momentmatched_Gaussian')
        k = 0 if kind == 'yj_target_marginal' else len(adapter.MASKS[adapter.MASK_NAMES.index(f['mask'])])
        if (law['family'] != ('normal' if gaussian else 'student_t')
                or law['df'] != (None if gaussian else 69 + k)
                or law['coordinate'] != ('native' if kind.startswith('raw_') else 'u=YJ_half(native/s0)')):
            raise ValueError('Incorrect coordinate, family or conditional df')
        if not kind.startswith('raw_') and (law['lambda'] != .5 or not math.isfinite(law['native_first_scale_s0']) or law['native_first_scale_s0'] <= 0):
            raise ValueError('Incorrect fixed YJ transform')
    return forecasts, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--approval', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    plan, roles = validate(args.plan, args.approval, 'forecast')
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'EXECUTION_START.json', {'actual_utc': utc(), 'source_sha256': source_hashes(),
        'parent_source_sha256': adapter.PARENT_SHA, 'plan_sha256': sha(args.plan),
        'approval_sha256': sha(args.approval), 'configuration': configuration(),
        'new_cohort_development_only': True, 'phase1_NO_GO_preserved': True})
    try:
        _, base_io, _ = adapter.parents()
        visible, extraction = base_io.extract_visible(plan['inputs']['raw_source']['file'], roles)
        extraction['phase2_model_visible_scope'] = 'same64x3fit and192x2proxy arrays; two mathematical reference fits are not two datasets'
        write(output / 'VISIBLE_INPUTS.json', visible)
        write(output / 'EXTRACTION_RECEIPT.json', extraction)
        fitted = adapter.fit_reference([r['native_values'] for r in visible['fit']])
        write(output / 'FIT_REFERENCE.json', fitted)
        forecasts = []
        for row in visible['diagnostic']:
            for kind in adapter.MODELS:
                for mask_name, mask in zip(adapter.MASK_NAMES, adapter.MASKS):
                    proxies = [row['native_proxies'][j] for j in mask]
                    forecasts.append({'component_id': row['identity']['component_id'], 'jid': row['identity']['jid'],
                        'model': kind, 'mask': mask_name, 'law': adapter.predict_law(fitted, kind, mask, proxies)})
        if len(forecasts) != 3840:
            raise AssertionError('All3840 forecast laws required')
        write(output / 'FORECASTS.json', {'diagnostic_role': 'development_only', 'forecasts': forecasts})
        if plan['source_sha256'] != source_hashes() or adapter.parent_hashes() != adapter.PARENT_SHA:
            raise ValueError('New or parent source changed during execution')
        write(output / 'FORECAST_FREEZE.json', {'status': 'COMPLETE_ALL3840_FORECASTS_FROZEN',
            'actual_utc': utc(), 'source_sha256': source_hashes(), 'parent_source_sha256': adapter.PARENT_SHA,
            'plan_sha256': sha(args.plan), 'files': {name: sha(output / name) for name in FROZEN_FILES},
            'forecast_count': 3840, 'diagnostic_target_model_access': 0,
            'physical_whole_source_parse_disclosed': True})
        print(json.dumps({'status': 'COMPLETE_ALL3840_FORECASTS_FROZEN', 'forecast_count': 3840}))
    except Exception as error:
        write(output / 'FAILED_STOP.json', {'actual_utc': utc(), 'type': type(error).__name__,
            'message': str(error), 'retain_failure_no_retune': True})
        raise


if __name__ == '__main__':
    main()

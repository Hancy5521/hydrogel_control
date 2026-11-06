"""Trusted full-source parser, restricted model inputs and exclusive forecast freeze.

Parsing the full gzip physically reads all source rows. Unselected properties are
not exposed, summarized or used. This is an audited access contract, not encryption.
"""
import argparse
from datetime import datetime, timezone
from fractions import Fraction
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True

import niw_predictive as model

CODE_FILES = ('admission_io.py', 'niw_predictive.py', 'score_admission.py')
RAW_SHA = '3d0fc1315b74cf84071f16aa25bb93434dafbc83867beb75872064ac245bd747'
FIELDS = ('e_form', 'gap opt', 'bulk modulus')
FREEZE_FILES = ('VISIBLE_INPUTS.json', 'MODEL.json', 'FORECASTS.json', 'EXTRACTION_RECEIPT.json', 'EXECUTION_START.json')


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key: ' + key)
        result[key] = value
    return result


def read(path):
    def invalid(value):
        raise ValueError('Nonstandard JSON constant: ' + value)
    return json.loads(Path(path).read_text('utf-8-sig'), parse_constant=invalid, object_pairs_hook=unique)


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def sources():
    folder = Path(__file__).resolve().parent
    return {name: sha(folder / name) for name in CODE_FILES}


def configuration():
    return {'version': 'full_joint_niw_model_admission_v1', 'dataset': 'jarvis_dft_3d',
            'fit_groups': 64, 'diagnostic_development_groups': 192, 'one_row_per_formula_group': True,
            'target_fields': list(FIELDS), 'target_index': 2,
            'models': list(model.MODELS), 'masks': [list(v) for v in model.MASKS],
            'mask_names': list(model.MASK_NAMES), 'forecast_count': 2304,
            'prior': {'p': 3, 'm0': [0, 0, 0], 'kappa0': 1, 'nu0': 7, 'Lambda0_diagonal': [3, 3, 3]},
            'transform': 'fit64_only_affine_sample_SD_empirical_reference',
            'gaussian_definition': 'same NIW predictive location and moment-matched full predictive covariance',
            'independent_role': 'same target t marginal, identity/association ablation, not a third independent algorithm',
            'diag_proxy_values_accumulated_into_warm_posterior': False,
            'Student_t_CRPS': 'closed_beta_formula_scoringRules_author_article_p26',
            'coverage_levels': [.8, .9, .95], 'primary_mask': 'both',
            'gates': {'dual_vs_niw_none_CRPS_ratio_max': .98, 'dual_vs_gaussian_CRPS_ratio_max': 1.02,
                      'dual_90_coverage_min': .8, 'dual_PIT_empirical_CDF_max_deviation_max': .15},
            'PIT_threshold_is_descriptive_not_KS_p_or_calibration_guarantee': True,
            'model_is_empirical_reference_not_native_prior_pure_Bayes': True,
            'all_diagnostic_groups_consumed_as_development': True,
            'allocation_agent_or_pseudo_label_training': False,
            'full_gzip_physically_parsed_in_trusted_reader': True,
            'unselected_response_summaries_or_model_use': False}


def validate(plan_file, approval_file, phase):
    plan, approval = read(plan_file), read(approval_file)
    expected = {'kind', 'status', 'registered_actual_utc', 'source_sha256', 'inputs', 'configuration', 'independent_math_receipt'}
    if (set(plan) != expected or plan['kind'] != 'full_joint_warm_only_material_admission'
            or plan['status'] != 'REGISTERED' or not isinstance(plan['registered_actual_utc'], str)
            or not plan['registered_actual_utc'] or plan['source_sha256'] != sources()
            or plan['configuration'] != configuration()):
        raise ValueError('Exact registered source-and-configuration plan required')
    if (approval.get('approved') is not True or approval.get('scope') != 'WARM_ONLY_NIW_MATERIAL_ADMISSION'
            or approval.get('plan_sha256') != sha(plan_file)
            or phase not in approval.get('allowed_phases', [])):
        raise ValueError('Root approval of exact plan and phase required')
    if set(plan['inputs']) != {'raw_source', 'roles', 'contract', 'root_scope'}:
        raise ValueError('Exactly source, metadata roles, task contract and pre-label root scope must be pinned')
    for name, item in plan['inputs'].items():
        if not isinstance(item['file'], str) or sha(item['file']) != item['sha256']:
            raise ValueError('Input changed: ' + name)
    if plan['inputs']['raw_source']['sha256'] != RAW_SHA:
        raise ValueError('Only the already-pinned JARVIS source is admitted')
    receipt = plan['independent_math_receipt']
    if sha(receipt['file']) != receipt['sha256']:
        raise ValueError('Independent mathematical receipt changed')
    contract = read(plan['inputs']['contract']['file'])
    if (contract['dataset']['id'] != 'jarvis_dft_3d'
            or contract['dataset']['raw_asset_sha256_from_prior_manifest_not_reread'] != RAW_SHA):
        raise ValueError('Contract does not identify this source')
    roles = read(plan['inputs']['roles']['file'])
    validate_roles(roles, contract)
    return plan, roles


def validate_roles(roles, contract):
    representatives = roles['representatives']
    if set(representatives) != {'fit', 'diagnostic'}:
        raise ValueError('Only fit and diagnostic roles allowed')
    groups, raw_rows, jids = [], [], []
    excluded = set(contract['eligibility_and_consumption']['old_consumed_component_ids'])
    for role, count in (('fit', 64), ('diagnostic', 192)):
        rows = representatives[role]
        if len(rows) != count:
            raise ValueError('Fixed64/192 groups required')
        for row in rows:
            if (row['role'] != role or row['component_id'] in excluded
                    or type(row['raw_row_zero_based']) is not int or row['raw_row_zero_based'] < 0
                    or not isinstance(row['jid'], str) or not row['jid']):
                raise ValueError('Invalid or consumed representative identity')
            groups.append(row['component_id'])
            raw_rows.append(row['raw_row_zero_based'])
            jids.append(row['jid'])
    if len(set(groups)) != 256 or len(set(raw_rows)) != 256 or len(set(jids)) != 256:
        raise ValueError('Formula/row/JID roles overlap or repeat')


def normalized_composition(value):
    if not isinstance(value, dict):
        raise ValueError('Pinned source composition dictionary required')
    amounts = {key: Fraction(str(v)) for key, v in value.items() if not key.startswith('@')}
    if any(v < 0 for v in amounts.values()):
        raise ValueError('Negative composition count')
    amounts = {k: v for k, v in amounts.items() if v > 0}
    total = sum(amounts.values(), Fraction(0))
    if total <= 0:
        raise ValueError('Empty composition')
    return '|'.join(k + ':' + str(amounts[k] / total) for k in sorted(amounts))


def parse_trusted_source(path):
    if sha(path) != RAW_SHA:
        raise ValueError('Source hash changed')
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        payload = json.load(stream, object_pairs_hook=unique)
    columns = payload['columns']
    if len(set(columns)) != len(columns) or len(payload['index']) != len(payload['data']):
        raise ValueError('Raw source schema mismatch')
    return payload, {name: i for i, name in enumerate(columns)}


def checked_source_row(payload, columns, meta):
    position = meta['raw_row_zero_based']
    if position >= len(payload['data']) or payload['index'][position] != meta['artifact_native_index']:
        raise ValueError('Artifact row/index does not match registered metadata')
    raw = payload['data'][position]
    if (raw[columns['jid']] != meta['jid'] or raw[columns['mpid']] != meta['mpid']
            or normalized_composition(raw[columns['composition']]) != meta['normalized_composition']):
        raise ValueError('Source JID/MPID/composition does not match metadata')
    return raw


def native_value(raw, columns, index):
    value = raw[columns[FIELDS[index]]]
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('Registered native source cell is not finite')
    return float(value)


def extract_visible(raw_path, roles):
    payload, columns = parse_trusted_source(raw_path)
    fit, diagnostic = [], []
    for meta in roles['representatives']['fit']:
        raw = checked_source_row(payload, columns, meta)
        fit.append({'identity': meta, 'native_values': [native_value(raw, columns, j) for j in range(3)]})
    for meta in roles['representatives']['diagnostic']:
        raw = checked_source_row(payload, columns, meta)
        diagnostic.append({'identity': meta, 'native_proxies': [native_value(raw, columns, j) for j in (0, 1)]})
    del payload
    return {'fit': fit, 'diagnostic': diagnostic}, {
        'actual_utc': utc(), 'physical_scope': 'Complete compressed JSON source and all payload rows parsed by trusted reader',
        'model_visible_fit_cells': 192, 'model_visible_diagnostic_proxy_cells': 384,
        'diagnostic_target_cells_explicitly_extracted_for_model': 0,
        'unselected_rows_used_for_property_statistics_or_model': False,
        'unselected_rows_discarded_without_response_summaries': True,
        'truth_extraction_entrypoint': 'score_admission only after complete forecast freeze',
        'access_contract_not_cryptographic_blinding': True}


def reveal_diagnostic_truth(raw_path, roles):
    """Evaluator-only caller must validate complete forecast freeze before entry."""
    payload, columns = parse_trusted_source(raw_path)
    truth = []
    for meta in roles['representatives']['diagnostic']:
        raw = checked_source_row(payload, columns, meta)
        truth.append({'identity': meta, 'native_target': native_value(raw, columns, 2)})
    del payload
    return truth


def expected_keys(roles):
    return {(r['component_id'], m, mask) for r in roles['representatives']['diagnostic']
            for m in model.MODELS for mask in model.MASK_NAMES}


def verify_forecast_freeze(directory, plan_file, roles):
    directory = Path(directory)
    receipt = read(directory / 'FORECAST_FREEZE.json')
    if (receipt['status'] != 'COMPLETE_ALL_FORECASTS_FROZEN' or receipt['plan_sha256'] != sha(plan_file)
            or receipt['source_sha256'] != sources() or set(receipt['files']) != set(FREEZE_FILES)):
        raise ValueError('Complete registered forecast freeze required before target reveal')
    for name, digest in receipt['files'].items():
        if sha(directory / name) != digest:
            raise ValueError('Frozen forecast artifact changed: ' + name)
    forecasts = read(directory / 'FORECASTS.json')['forecasts']
    keys = [(x['component_id'], x['model'], x['mask']) for x in forecasts]
    if len(keys) != 2304 or len(set(keys)) != 2304 or set(keys) != expected_keys(roles):
        raise ValueError('Missing, repeated or unexpected model/mask/group forecast')
    expected_jid = {r['component_id']: r['jid'] for r in roles['representatives']['diagnostic']}
    for f in forecasts:
        law = f['law']
        if f['jid'] != expected_jid[f['component_id']] or law['scale'] <= 0 or not all(
                type(law[k]) in (int, float) and math.isfinite(law[k]) for k in ('location', 'scale')):
            raise ValueError('Invalid frozen forecast identity or law')
        if f['model'] == 'gaussian_momentmatched':
            if law['family'] != 'normal' or law['df'] is not None:
                raise ValueError('Incorrect Gaussian law')
        else:
            k = len(model.MASKS[model.MASK_NAMES.index(f['mask'])]) if f['model'] == 'niw_joint_t' else 0
            if law['family'] != 'student_t' or law['df'] != 69 + k:
                raise ValueError('Incorrect Student-t conditional df')
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
    write(output / 'EXECUTION_START.json', {'actual_utc': utc(), 'plan_sha256': sha(args.plan),
        'approval_sha256': sha(args.approval), 'source_sha256': sources(),
        'configuration': configuration(), 'diagnostic_targets_used_by_model': 0})
    try:
        visible, receipt = extract_visible(plan['inputs']['raw_source']['file'], roles)
        write(output / 'VISIBLE_INPUTS.json', visible)
        write(output / 'EXTRACTION_RECEIPT.json', receipt)
        fitted = model.fit_niw([r['native_values'] for r in visible['fit']])
        write(output / 'MODEL.json', fitted)
        forecasts = []
        for row in visible['diagnostic']:
            for kind in model.MODELS:
                for name, mask in zip(model.MASK_NAMES, model.MASKS):
                    proxies = [row['native_proxies'][j] for j in mask]
                    forecasts.append({'component_id': row['identity']['component_id'], 'jid': row['identity']['jid'],
                        'model': kind, 'mask': name, 'law': model.predict_law(fitted, kind, mask, proxies)})
        if len(forecasts) != 2304:
            raise AssertionError('All model/mask forecasts required')
        write(output / 'FORECASTS.json', {'diagnostic_role': 'development_only', 'forecasts': forecasts})
        if plan['source_sha256'] != sources():
            raise ValueError('Source changed during execution')
        write(output / 'FORECAST_FREEZE.json', {'status': 'COMPLETE_ALL_FORECASTS_FROZEN', 'actual_utc': utc(),
            'plan_sha256': sha(args.plan), 'source_sha256': sources(),
            'files': {name: sha(output / name) for name in FREEZE_FILES},
            'forecast_count': 2304, 'diagnostic_target_model_access': 0,
            'physical_full_source_parse_disclosed': True})
        print(json.dumps({'status': 'COMPLETE_ALL_FORECASTS_FROZEN', 'forecast_count': 2304}))
    except Exception as error:
        write(output / 'FAILED_STOP.json', {'actual_utc': utc(), 'type': type(error).__name__,
            'message': str(error), 'retain_failure_no_retune': True})
        raise


if __name__ == '__main__':
    main()

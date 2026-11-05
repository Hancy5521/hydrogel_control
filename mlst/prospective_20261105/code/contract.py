"""Metadata, grouping and byte-identity guards for a prospective study.

These checks enforce a declared ledger. They cannot prove physical independence,
exchangeability, blindness, rights or truthful human metadata.
"""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
from calibration import alpha_fraction, real

SHA = re.compile(r'^[0-9a-f]{64}$')
ROLES = {'consumed', 'calibration', 'confirmation'}
GOALS = {'new_formulation', 'new_preparation_known_formula'}
SCHEMA = 'prospective_unit_calibration_v1'


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_json(text):
    def invalid(value):
        raise ValueError('Nonstandard JSON constant is forbidden: ' + value)
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON key is forbidden: ' + key)
            result[key] = value
        return result
    return json.loads(text, parse_constant=invalid, object_pairs_hook=unique)


def read(path):
    return parse_json(Path(path).read_text(encoding='utf-8-sig'))


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, allow_nan=False)


def nonempty(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(name + ' must be a nonempty string')
    return value


def source_bindings():
    root = Path(__file__).resolve().parent
    return {name: sha(root / name) for name in
            ('contract.py', 'calibration.py', 'evaluation.py', 'cli.py')}


def planned_continuation(record, support):
    """Consume only metadata; preserve the existing six-anchor plan contract."""
    gamma = [real(v, 'native_gamma') for v in record['native_gamma']]
    if len(gamma) < 8 or any(v <= 0 for v in gamma):
        raise ValueError('Positive native coordinates with six anchors and at least two targets required')
    if any(b <= a for a, b in zip(gamma, gamma[1:])):
        raise ValueError('Native scan must already be strictly increasing; no sorting or deduplication')
    if gamma[0] < support[0] or gamma[-1] > support[1]:
        raise ValueError('Native coordinates outside frozen support; no clipping')
    observed = record['observed_indices']
    if (not isinstance(observed, list) or len(observed) != 6
            or any(type(v) is not int for v in observed)
            or observed != sorted(set(observed))
            or observed[0] < 0 or observed[-1] >= len(gamma)):
        raise ValueError('Six distinct increasing native anchor indices required')
    targets = [i for i in range(len(gamma)) if gamma[i] > gamma[observed[-1]]]
    if len(targets) < 2:
        raise ValueError('Primary integrated continuation endpoint needs at least two targets')
    declared_targets = record['target_indices']
    if (not isinstance(declared_targets, list) or any(type(v) is not int for v in declared_targets)
            or declared_targets != targets):
        raise ValueError('Targets must be every declared native point strictly above highest anchor')
    return gamma, observed, targets


def check_design(config, cohort):
    """Validate a complete registration design, without reading measurements."""
    config_keys = {'schema', 'data_mode', 'protocol_id', 'scientific_goal', 'alpha', 'response_scale',
                   'primary_endpoint', 'quadrature_domain', 'single_target_rule', 'prediction_failure_rule',
                   'anchor_rule', 'anchor_policy_reference', 'global_support_gamma_s_inverse',
                   'primary_method_id', 'methods', 'material_state', 'temperature_kelvin',
                   'scan_protocol_reference', 'source_admission_reference', 'independence_assumption_evidence',
                   'design_reviewed_by', 'design_review_reference', 'n_calibration_units', 'n_confirmation_units',
                   'sample_size_basis', 'practical_error_tolerance', 'practical_error_tolerance_unit', 'consumed_registry'}
    if not isinstance(config, dict) or set(config) != config_keys:
        raise ValueError('Exact metadata-only design schema required; unknown/missing fields forbidden')
    if not isinstance(cohort, dict) or set(cohort) != {'schema', 'data_mode', 'records'}:
        raise ValueError('Exact metadata-only cohort envelope required')
    if config.get('schema') != SCHEMA or cohort.get('schema') != SCHEMA:
        raise ValueError('Known prospective schema required')
    mode = config.get('data_mode')
    if mode not in ('synthetic', 'real') or cohort.get('data_mode') != mode:
        raise ValueError('Matching explicit real/synthetic modes required')
    goal = config.get('scientific_goal')
    if goal not in GOALS:
        raise ValueError('Choose new-formulation or new-preparation goal; do not mix their claims')
    nonempty(config.get('protocol_id'), 'protocol_id')
    alpha_fraction(config['alpha'])
    if config.get('response_scale') != 'ln(eta/(1 Pa s))':
        raise ValueError('Explicit natural-log viscosity scale required')
    if config.get('primary_endpoint') != 'unit_equal_mean_target_log_trapezoid_CRPS':
        raise ValueError('Fixed primary endpoint identifier required')
    if config.get('quadrature_domain') != 'first_hidden_native_to_last_hidden_native':
        raise ValueError('Explicit target-only integration domain required')
    if config.get('single_target_rule') != 'reject_primary_endpoint':
        raise ValueError('No arbitrary integral weight for a single target')
    if config.get('prediction_failure_rule') != 'retain_receipt_and_stop_no_subset':
        raise ValueError('Atomic failure policy is required')
    nonempty(config.get('anchor_rule'), 'anchor_rule')
    nonempty(config.get('anchor_policy_reference'), 'anchor_policy_reference')
    support = config['global_support_gamma_s_inverse']
    if len(support) != 2 or not 0 < real(support[0], 'support') < real(support[1], 'support'):
        raise ValueError('Positive increasing support interval required')
    if (support != [0.012787740677595104, 754.5143432617188]):
        raise ValueError('This extension retains the currently frozen hydrogel support')
    methods = config.get('methods')
    if not isinstance(methods, list) or not methods:
        raise ValueError('Declare frozen methods before calibration')
    method_ids = []
    for method in methods:
        if not isinstance(method, dict) or set(method) != {'id', 'model_sha256', 'runtime_bindings_sha256'}:
            raise ValueError('Exact frozen-method identity schema required')
        method_ids.append(nonempty(method.get('id'), 'method id'))
        for field in ('model_sha256', 'runtime_bindings_sha256'):
            if not isinstance(method.get(field), str) or not SHA.fullmatch(method[field]):
                raise ValueError('Exact model/runtime hashes required')
    if len(set(method_ids)) != len(method_ids) or config.get('primary_method_id') not in method_ids:
        raise ValueError('Unique methods and an explicit primary method required')
    for field in ('material_state', 'scan_protocol_reference', 'source_admission_reference',
                  'independence_assumption_evidence', 'design_reviewed_by', 'design_review_reference',
                  'sample_size_basis', 'practical_error_tolerance_unit'):
        nonempty(config.get(field), field)
    real(config.get('temperature_kelvin'), 'temperature_kelvin')
    if config['temperature_kelvin'] <= 0:
        raise ValueError('Positive absolute temperature required')
    if real(config.get('practical_error_tolerance'), 'practical_error_tolerance') <= 0:
        raise ValueError('A positive author-chosen practical tolerance is required')
    registry = config.get('consumed_registry')
    registry_keys = {'review_complete', 'review_reference', 'record_keys', 'preparation_batch_ids',
                     'formulation_ids', 'identity_aliases'}
    if (not isinstance(registry, dict) or set(registry) != registry_keys
            or type(registry['review_complete']) is not bool):
        raise ValueError('Exact consumed-registry metadata schema required')
    if not isinstance(registry.get('review_reference'), str) or not registry['review_reference'].strip():
        raise ValueError('Reviewed historical consumed-record registry required')
    for field in ('record_keys', 'preparation_batch_ids', 'formulation_ids', 'identity_aliases'):
        entries = registry.get(field)
        if not isinstance(entries, list) or any(not isinstance(v, str) or not v for v in entries):
            raise ValueError('Consumed inventory fields must be explicit string lists: ' + field)
    forbidden_records = set(registry['record_keys'])
    forbidden_batches = set(registry['preparation_batch_ids'])
    consumed_formulations = set(registry['formulation_ids'])
    forbidden_aliases = set(registry['identity_aliases'])
    if mode == 'real' and registry.get('review_complete') is not True:
        raise ValueError('The current consumed-record inventory needs author review')
    records = cohort.get('records')
    if not isinstance(records, list) or not records:
        raise ValueError('A pre-enrolled cohort is required')
    record_ids, source_keys, linked = set(), set(), {}
    unit_roles, unit_formulations = {}, {}
    for record in records:
        allowed = {'record_id', 'unit_id', 'role', 'formulation_id', 'preparation_batch_id',
                   'source_id', 'source_version', 'source_record_key', 'provenance_reference',
                   'material_state', 'temperature_kelvin', 'identity_aliases', 'is_synthetic',
                   'targets_previously_inspected', 'native_gamma', 'observed_indices', 'target_indices'}
        if not isinstance(record, dict) or set(record) - allowed:
            raise ValueError('Cohort ledger contains unknown fields; no target measurements permitted')
        rid = nonempty(record.get('record_id'), 'record_id')
        uid = nonempty(record.get('unit_id'), 'unit_id')
        role = record.get('role')
        if role not in ROLES or rid in record_ids:
            raise ValueError('Unique record IDs and known roles required')
        record_ids.add(rid)
        if uid in unit_roles and unit_roles[uid] != role:
            raise ValueError('One independent unit cannot cross data roles')
        unit_roles[uid] = role
        for field in ('formulation_id', 'preparation_batch_id', 'source_id',
                      'source_version', 'source_record_key', 'provenance_reference',
                      'material_state'):
            nonempty(record.get(field), field)
        fid, batch, key = record['formulation_id'], record['preparation_batch_id'], record['source_record_key']
        source_keys.add(key)
        aliases = record.get('identity_aliases')
        if not isinstance(aliases, list) or not aliases or any(not isinstance(v, str) or not v for v in aliases):
            raise ValueError('Explicit reviewed identity aliases are required')
        if record.get('is_synthetic') is not (mode == 'synthetic'):
            raise ValueError('Synthetic/real identity cannot be silently changed')
        if mode == 'synthetic' and (not rid.startswith('SYNTHETIC_') or not uid.startswith('SYNTHETIC_')):
            raise ValueError('Synthetic IDs must be visibly labelled')
        if record['material_state'] != config['material_state'] or record.get('temperature_kelvin') != config['temperature_kelvin']:
            raise ValueError('Enrolled material state/temperature differs from declared domain')
        # A batch or alias links physical units even if source names differ.
        # Multiple preregistered masks on a curve may repeat its source key, but
        # remain one physical unit and cannot cross calibration/confirmation.
        tokens = [('batch', batch), ('source_record', key)] + [('alias', a) for a in aliases]
        if goal == 'new_formulation':
            tokens.append(('formulation', fid))
        for token in tokens:
            if token in linked and linked[token] != uid:
                raise ValueError('Linked physical records must share an independent-unit ID')
            linked[token] = uid
        if goal == 'new_formulation' and uid in unit_formulations and unit_formulations[uid] != fid:
            raise ValueError('A new-formulation unit contains exactly one formulation')
        unit_formulations[uid] = fid
        if role != 'consumed':
            if record.get('targets_previously_inspected') is not False:
                raise ValueError('Only explicitly unconsumed future target records are eligible')
            if key in forbidden_records or batch in forbidden_batches or any(a in forbidden_aliases for a in aliases):
                raise ValueError('Previously consumed physical record/batch/alias is not new')
            if goal == 'new_formulation' and fid in consumed_formulations:
                raise ValueError('A known formulation cannot support a new-formulation claim')
            planned_continuation(record, support)
    counts = {role: sum(v == role for v in unit_roles.values()) for role in ROLES}
    for role in ('calibration', 'confirmation'):
        required = config.get('n_' + role + '_units')
        if type(required) is not int or required < 1 or counts[role] != required:
            raise ValueError('Fill and match positive independent-unit sample counts: ' + role)
    return {'data_mode': mode, 'declared_unit_counts': counts, 'records': len(records),
            'declared_ledger_consistent': True, 'physical_independence_certified': False,
            'exchangeability_certified': False, 'blindness_certified': False}


def check_predictions(payload, protocol, role, protocol_hash):
    """Accept complete frozen laws, not a selected successful subset or truth."""
    expected_keys = {'schema', 'data_mode', 'protocol_sha256', 'records'}
    if not isinstance(payload, dict) or set(payload) != expected_keys or payload['schema'] != SCHEMA:
        raise ValueError('Prediction envelope contains unknown fields or labels')
    if payload['data_mode'] != protocol['config']['data_mode']:
        raise ValueError('Prediction mode differs from protocol')
    if payload['protocol_sha256'] != protocol_hash:
        raise ValueError('Prediction protocol byte identity changed')
    enrolled = {r['record_id']: r for r in protocol['cohort']['records'] if r['role'] == role}
    methods = {m['id']: m for m in protocol['config']['methods']}
    rows = payload['records']
    if not isinstance(rows, list) or len(rows) != len(enrolled) or {r.get('record_id') for r in rows} != set(enrolled):
        raise ValueError('All pre-enrolled records, once each, are required')
    for row in rows:
        if set(row) != {'record_id', 'observed_gamma', 'target_gamma', 'anchor_input_sha256', 'prediction_source_sha256', 'methods'}:
            raise ValueError('Exact anchor-only forecast fields required; no truth or extra fields')
        for field in ('anchor_input_sha256', 'prediction_source_sha256'):
            if not isinstance(row[field], str) or not SHA.fullmatch(row[field]):
                raise ValueError('Actual anchor-input/source byte hashes required')
        record = enrolled[row['record_id']]
        gamma, observed, targets = planned_continuation(record, protocol['config']['global_support_gamma_s_inverse'])
        if row['observed_gamma'] != [gamma[i] for i in observed] or row['target_gamma'] != [gamma[i] for i in targets]:
            raise ValueError('Prediction coordinates differ from preregistered native plan')
        if set(row['methods']) != set(methods):
            raise ValueError('All fixed methods must be frozen on every record')
        n = len(targets)
        for mid, law in row['methods'].items():
            if set(law) != {'model_sha256', 'runtime_bindings_sha256', 'mean_log_eta', 'covariance_log_eta'}:
                raise ValueError('Exact Gaussian law and method identities required')
            if any(law[k] != methods[mid][k] for k in ('model_sha256', 'runtime_bindings_sha256')):
                raise ValueError('Model/runtime changed after registration')
            m, cov = law['mean_log_eta'], law['covariance_log_eta']
            if len(m) != n or len(cov) != n or any(len(r) != n for r in cov):
                raise ValueError('Complete target-aligned Gaussian covariance required')
            for v in m:
                real(v, 'Gaussian mean')
            for rr in cov:
                for v in rr:
                    real(v, 'Gaussian covariance')
            mm, cc = np.array(m, dtype=float), np.array(cov, dtype=float)
            if not np.isfinite(mm).all() or not np.isfinite(cc).all() or not np.allclose(cc, cc.T, atol=1e-12, rtol=1e-12):
                raise ValueError('Finite symmetric Gaussian laws required')
            np.linalg.cholesky(cc)  # No clipping, jitter or successful-subset selection.
    return rows


def check_truth(payload, predictions, protocol, role, protocol_hash):
    if (not isinstance(payload, dict)
            or set(payload) != {'schema', 'data_mode', 'protocol_sha256', 'records'} or payload['schema'] != SCHEMA):
        raise ValueError('Exact revealed-truth envelope required')
    if payload['data_mode'] != protocol['config']['data_mode']:
        raise ValueError('Truth mode differs from protocol')
    if payload['protocol_sha256'] != protocol_hash:
        raise ValueError('Truth protocol byte identity changed')
    expected = {r['record_id']: r for r in predictions['records']}
    rows = payload['records']
    if not isinstance(rows, list) or len(rows) != len(expected) or {r.get('record_id') for r in rows} != set(expected):
        raise ValueError('Reveal every pre-enrolled frozen record once')
    result, original_curves = {}, {}
    enrolled = {r['record_id']: r for r in protocol['cohort']['records'] if r['role'] == role}
    for row in rows:
        if set(row) != {'record_id', 'target_gamma', 'y_log_eta', 'original_measurement_sha256'}:
            raise ValueError('Native measured truth and byte provenance required')
        rid = row['record_id']
        if row['target_gamma'] != expected[rid]['target_gamma'] or len(row['y_log_eta']) != len(row['target_gamma']):
            raise ValueError('Revealed values do not align with frozen target coordinates')
        if not isinstance(row['original_measurement_sha256'], str) or not SHA.fullmatch(row['original_measurement_sha256']):
            raise ValueError('Original measurement hash required')
        values = [real(y, 'revealed log viscosity') for y in row['y_log_eta']]
        source_key = enrolled[rid]['source_record_key']
        if source_key not in original_curves:
            original_curves[source_key] = {'measurement_sha256': row['original_measurement_sha256'], 'values': {}}
        original = original_curves[source_key]
        if original['measurement_sha256'] != row['original_measurement_sha256']:
            raise ValueError('Repeated masks of one original curve must bind the same measurement bytes')
        for gamma, value in zip(row['target_gamma'], values):
            if gamma in original['values'] and original['values'][gamma] != value:
                raise ValueError('Overlapping targets of one original curve must have exactly equal truth')
            original['values'][gamma] = value
        result[rid] = values
    return result

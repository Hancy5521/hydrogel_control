"""Explicit roles, source-specific units, and true-report disclosure.

The trusted JSON decoder reads a whole public file. Only registered fields and
authorized groups leave the reader; this is a software disclosure protocol, not
physical data hiding, fresh DFT calculation, or experimental-cost evidence.
"""
import gzip
import hashlib
import importlib.util
import json
import math
from copy import deepcopy
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

SCOPE = 'DIELECTRIC_TRANSPORT_V1'
FILES = ('task_io.py', 'field_admission.py', 'run_dielectric.py',
         'score_dielectric.py', 'empirical_native.py', 'simplex_transport.py', 'calibration_views.py')
FIELDS = ('band_gap', 'poly_electronic', 'poly_total')
METHODS = ('RF_identity', 'RF_beta', 'RF_global4', 'RF_anchor4', 'RF_auto4', 'GP10', 'electronic_delta')
BATCHES = ('validation1', 'validation2', 'validation3')
ROLE_SIZES = {'fit': 256, 'validation1': 64, 'validation2': 64, 'validation3': 64}
UNIT = 'dimensionless relative static permittivity; poly_total = mean dielectric-tensor eigenvalues'
FEATURE_IDS = ['band_gap_eV', 'poly_electronic_dimensionless', 'O_present', 'f_O', 'f_H',
               'f_C_plus_f_N', 'f_F_Cl_Br_I', 'f_S_Se_Te', 'element_count', 'composition_entropy']


def utc(): return datetime.now(timezone.utc).isoformat()


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            value.update(block)
    return value.hexdigest()


def unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result: raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), object_pairs_hook=unique,
                      parse_constant=lambda text: (_ for _ in ()).throw(ValueError('nonfinite JSON: ' + text)))


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')


def sources():
    return {name: sha(Path(__file__).resolve().parent / name) for name in FILES}


def validate(planfile, approvalfile, phase):
    plan, approval = read(planfile), read(approvalfile)
    if (plan.get('status') != 'REGISTERED' or plan.get('scope') != SCOPE
            or approval.get('approved') is not True or approval.get('scope') != SCOPE
            or approval.get('plan_sha256') != sha(planfile)
            or phase not in approval.get('allowed_phases', [])):
        raise ValueError('exact registered root phase approval required')
    if sources() != plan['source_sha256']:
        raise ValueError('registered scientific source changed')
    for section in ('inputs', 'dependencies'):
        for key, pin in plan[section].items():
            if sha(pin['file']) != pin['sha256']:
                raise ValueError('registered dependency changed: ' + key)
    if read(plan['inputs']['identity_independent_review']['file']).get('status') != 'PASS_INDEPENDENT_IDENTITY_ONLY':
        raise ValueError('independent full-identity audit must have passed')
    if plan['role_sizes'] != ROLE_SIZES or plan['methods'] != list(METHODS):
        raise ValueError('fixed 256 + three64 roles/seven methods required')
    if plan['predicate'] != 'finite real band_gap>0, poly_electronic>0, poly_total>0; representative only; no magnitude or decomposition filter':
        raise ValueError('fixed eligibility predicate required')
    return plan


def verify_payloads(folder, pins):
    folder = Path(folder).resolve()
    for name, expected in pins.items():
        path = (folder / name).resolve()
        path.relative_to(folder)
        if sha(path) != expected: raise ValueError('frozen payload changed: ' + name)


def load_roles(plan):
    pin = plan['inputs']['admission_complete']
    folder = Path(pin['file']).resolve().parent
    complete = read(pin['file'])
    if (complete['status'] != 'ADMISSION_COMPLETE_NO_FIT_NO_PROPERTY_VALUES_EMITTED'
            or complete['source_sha256'] != sources()):
        raise ValueError('completed registered field admission required')
    verify_payloads(folder, complete['files_sha256'])
    roles = read(folder / 'ROLES.json')
    if roles['role_sizes'] != ROLE_SIZES:
        raise ValueError('registered disjoint role sizes required')
    if any(len(roles['representatives'][role]) != size for role, size in ROLE_SIZES.items()):
        raise ValueError('all exact registered role lengths required')
    ids = [r['component_id'] for role in ROLE_SIZES for r in roles['representatives'][role]]
    if len(ids) != 448 or len(set(ids)) != 448:
        raise ValueError('448 unique whole components required')
    if set(ids) & set(roles['remaining_unselected_component_ids']):
        raise ValueError('selected groups overlap untouched remainder')
    return roles


def table(rawfile):
    with gzip.open(rawfile, 'rt', encoding='utf-8') as stream:
        # Standard decoder permits unused-property NaN. Registered three fields
        # are independently checked; no generic filter over unregistered fields.
        payload = json.load(stream, object_pairs_hook=unique)
    if not isinstance(payload, dict) or not all(k in payload for k in ('columns', 'index', 'data')):
        raise ValueError('unknown public dataframe schema')
    columns, rows, indices = payload['columns'], payload['data'], payload['index']
    if (len(columns) != len(set(columns)) or not all(k in columns for k in FIELDS + ('material_id',))
            or len(rows) != 1056 or len(rows) != len(indices)
            or any(len(row) != len(columns) for row in rows)):
        raise ValueError('exact1056 rows and explicit unique source fields required')
    return payload, {name: columns.index(name) for name in FIELDS + ('material_id',)}


def values_at(payload, columns, identity):
    index = identity['raw_row_zero_based']
    if (payload['index'][index] != identity['artifact_native_index']
            or payload['data'][index][columns['material_id']] != identity['material_id']):
        raise ValueError('source artifact index/material ID mismatch')
    return [payload['data'][index][columns[field]] for field in FIELDS]


def admitted(values):
    return (len(values) == 3 and all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in values))


def descriptor_pair(identity):
    from calibration_views import composition_descriptors
    bounded = composition_descriptors({e: float(Fraction(v)) for e, v in identity['composition_amount_fractions'].items()})
    raw = bounded[:6] + [bounded[6] * 118, bounded[7] * math.log(118)]
    return raw, bounded


def import_reference(plan):
    pin = plan['dependencies']['reference_baselines']
    path = Path(pin['file']).resolve()
    spec = importlib.util.spec_from_file_location('_dielectric_verified_reference_baselines', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if sha(path) != pin['sha256']: raise ValueError('baseline source changed')
    return module


def fit_reference(reference, kind, X, y, query, transform):
    """Keep numeric recipe; overwrite legacy physical labels before any emission."""
    function = reference.fit_forest if kind == 'RF10' else reference.fit_gp
    laws, state = function(X, y, query, transform)
    for law in laws: law['unit'] = UNIT
    state['target_field'] = 'poly_total'
    state['target_unit'] = UNIT
    state['source_specific_metadata_adapter'] = 'legacy bulk-law unit replaced before emitting artifact; numeric recipe unchanged'
    state['feature_ids'] = list(FEATURE_IDS)
    return laws, state


class Disclosure:
    def __init__(self, rawfile, roles):
        self.payload, self.columns = table(rawfile)
        self.roles = roles
        self.started = False
        self.next = 0
        self.events = []

    def initial(self):
        if self.started: raise ValueError('initial disclosure already occurred')
        self.started = True
        fit, visible = [], {}
        for role in ROLE_SIZES:
            rows = []
            for identity in self.roles['representatives'][role]:
                values = values_at(self.payload, self.columns, identity)
                if not admitted(values): raise ValueError('registered field eligibility changed')
                if role == 'fit':
                    fit.append({'identity': identity, 'native_values': values})
                else:
                    rows.append({'identity': identity, 'native_proxies': values[:2]})
            if role != 'fit': visible[role] = rows
        self.events.append({'actual_utc': utc(), 'event': 'initial', 'fit_Y_disclosed': 256,
                            'future_context_cells_disclosed': 384, 'future_Y_disclosed': 0})
        return fit, visible

    def reveal(self, role, freeze_file, expected_source, plan_sha):
        if not self.started or self.next >= 3 or BATCHES[self.next] != role:
            raise ValueError('ordered, single-use future disclosure required')
        freeze = read(freeze_file)
        if (freeze['status'] != 'ALL448_BATCH_LAWS_FROZEN_BEFORE64Y' or freeze['role'] != role
                or freeze['source_sha256'] != expected_source or freeze['plan_sha256'] != plan_sha):
            raise ValueError('exact current forecast freeze required before truth')
        verify_payloads(Path(freeze_file).parent, freeze['files_sha256'])
        rows = [{'identity': identity, 'native_target': values_at(self.payload, self.columns, identity)[2]}
                for identity in self.roles['representatives'][role]]
        self.next += 1
        event = {'actual_utc': utc(), 'role': role, 'true_Y_disclosed': len(rows),
                 'forecast_freeze_sha256': sha(freeze_file)}
        self.events.append(event)
        return {'actual_utc': event['actual_utc'], 'rows': rows,
                'forecast_freeze_sha256': event['forecast_freeze_sha256'],
                'no_synthetic_or_predicted_labels': True}

    def ledger(self):
        return {'events': self.events, 'initial_real_report_cells': 768,
                'future_real_report_cells': 576, 'new_computed_source_components': 448,
                'new_wet_measurements': 0, 'new_DFT_calculations': 0,
                'full_JSON_in_trusted_reader': True, 'unused_remainder_property_values_emitted': 0}

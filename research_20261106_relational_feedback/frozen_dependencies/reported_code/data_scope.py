"""Fixed protocol, metadata identity and trusted staged disclosure; no model fitting."""
from datetime import datetime, timezone
from fractions import Fraction
import gzip
import hashlib
import json
import math
from pathlib import Path

CODE_FILES = ('data_scope.py', 'joint_model.py', 'run_forecast.py', 'score.py')
RAW_SHA = '3d0fc1315b74cf84071f16aa25bb93434dafbc83867beb75872064ac245bd747'
FIELDS = ('e_form', 'gap opt', 'bulk modulus')
METHODS = ('hurdle_updated', 'hurdle_stale', 'pooled_yj_updated', 'gpr_fixed_recipe', 'forest_all_original_leaf_cdf')
CHECKPOINTS = (0, 8, 16, 32)
SCOPE = 'REPORTED_PROPERTY_EXACT_FEEDBACK_PILOT'
BASE_FILES = ('EXECUTION_START.json', 'LIBRARY_VERSIONS.json', 'VISIBLE_INITIAL_INPUTS.json',
              'QUERY_LEDGER.json', 'MODEL_STATES.json', 'FORECASTS.json', 'PREQUENTIAL_INDEX.json')


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def read(path):
    def invalid(value):
        raise ValueError('nonstandard JSON constant: ' + value)
    return json.loads(Path(path).read_text('utf-8-sig'), object_pairs_hook=unique, parse_constant=invalid)


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def sources():
    folder = Path(__file__).resolve().parent
    return {name: sha(folder / name) for name in CODE_FILES}


def configuration():
    return {'version': 'reported_exact_complete_feedback_v1', 'dataset': 'jarvis_dft_3d',
        'warm_groups': 64, 'feedback_pool_groups': 32, 'terminal_development_groups': 128,
        'old552_whole_components_excluded': True, 'methods': list(METHODS),
        'checkpoint_complete_rows': list(CHECKPOINTS), 'checkpoint_new_cell_budgets': [0, 24, 48, 96],
        'primary_terminal_laws': 2560, 'secondary_prequential_laws': 32,
        'reported_atom_rule': 'native G==0, signed zero identical; no physical zero or censoring inference',
        'prior_Beta': [1, 1], 'prior_NIW': 'each dimensiond: mean0,kappa1,nu=d+4,Psi3I',
        'global_transform': 'all3fixedwarm YJhalf(native/warmSD), then fixedwarmu mean/sampleSD',
        'baseline_X_transform': 'fixedwarm native proxy mean/sampleSD; not refitted at checkpoint',
        'terminal_proxies_update_global_posterior': False,
        'own_terminal_proxies': ['e_form', 'gap opt'], 'terminal_target': 'bulk modulus',
        'complete_feedback_schedule': 'metadatafixedpoolorder; proxies2cells then frozenYlaw then trueY1cell; globalupdateonlycompletepanel',
        'cell_fee': 'one per new native cell; warm192 and terminalproxy256 common separately; not actual DFT compute cost',
        'baseline_fits': 8, 'GPR_recipe': 'Constant(1,[.01,100])*RBF(ones2,[.01,100])+White(.01,[1e-6,10]);alpha1e-8;normalize_yTrue;optimizerL-BFGS-B;restarts0;seed20261106',
        'forest_recipe': '256trees;minleaf3;max_features integer1;bootstrapTrue;seed20261106;n_jobs2',
        'forest_CDF': 'all original currenttrain rows includingOOB; no bootstrapmultiplicity; exactFractionleafweights',
        'libraries': {'numpy': '2.1.3', 'scipy': '1.15.3', 'scikit-learn': '1.6.1', 'joblib': '1.4.2'},
        'score_absolute_tolerance_native': 1e-9, 'score_relative_tolerance': 1e-9,
        'tail_each_absolute_budget_fraction': .125, 'tail_initial_standard_cut': 16.,
        'tail_max_standard_cut': 1048576., 'tail_max_doublings': 20,
        'fixed_integral_subdivision_points': [-8., -4., -2., -1., 0., 1., 2., 4., 8.],
        'quad_subinterval_limit': 300, 'coverage_levels': [.8, .9, .95],
        'point_forecast': 'native predictive median',
        'primary': 'final96cell signed paired nativeCRPS hurdle_updated minus samewarm stale; all128retained',
        'descriptive_competitiveness_ratio_max': 1.02,
        'no_required2percent_learning_gain_or_formal_noninferiority': True,
        'adaptive_planner_LLM_pseudolabels_or_new_wet_DFT': False}


def validate_roles(roles):
    representatives = roles['representatives']
    if set(representatives) != {'warm', 'pool', 'terminal'}:
        raise ValueError('exactwarm/pool/terminalroles required')
    excluded = set(roles['all_prior_excluded_component_ids'])
    if len(excluded) != 552:
        raise ValueError('all552 prior wholecomponents must remain excluded')
    groups, rows, identifiers = [], [], []
    for role, count in (('warm', 64), ('pool', 32), ('terminal', 128)):
        if len(representatives[role]) != count:
            raise ValueError('fixed64/32/128 role counts required')
        for item in representatives[role]:
            if (item['role'] != role or item['component_id'] in excluded
                    or type(item['raw_row_zero_based']) is not int or item['raw_row_zero_based'] < 0):
                raise ValueError('invalid or consumed group locator')
            groups.append(item['component_id']); rows.append(item['raw_row_zero_based']); identifiers.append(item['jid'])
    if len(set(groups)) != 224 or len(set(rows)) != 224 or len(set(identifiers)) != 224:
        raise ValueError('overlapping or repeated role identity')
    if roles['pool_sequence_component_ids'] != [item['component_id'] for item in representatives['pool']]:
        raise ValueError('registered metadata-only pool sequence changed')


def validate(plan_path, approval_path, phase):
    plan, approval = read(plan_path), read(approval_path)
    if (set(plan) != {'kind', 'status', 'registered_actual_utc', 'source_sha256', 'configuration', 'inputs', 'independent_math_receipt'}
            or plan['kind'] != 'reported_property_exact_feedback' or plan['status'] != 'REGISTERED'
            or not isinstance(plan['registered_actual_utc'], str) or not plan['registered_actual_utc']
            or plan['source_sha256'] != sources() or plan['configuration'] != configuration()):
        raise ValueError('exact registered source and scientific recipe required')
    if (approval.get('approved') is not True or approval.get('scope') != SCOPE
            or approval.get('plan_sha256') != sha(plan_path) or phase not in approval.get('allowed_phases', [])):
        raise ValueError('root approval of exact plan and phase required')
    if set(plan['inputs']) != {'raw_source', 'roles', 'root_scope'}:
        raise ValueError('exactpinnedsource/roles/root_scope required')
    for name, item in plan['inputs'].items():
        if sha(item['file']) != item['sha256']:
            raise ValueError('input changed: ' + name)
    if plan['inputs']['raw_source']['sha256'] != RAW_SHA:
        raise ValueError('only frozen public raw source admitted')
    receipt = plan['independent_math_receipt']
    if sha(receipt['file']) != receipt['sha256'] or not str(read(receipt['file'])['status']).startswith('PASS'):
        raise ValueError('actual independent mathematical audit PASS receipt required')
    roles = read(plan['inputs']['roles']['file']); validate_roles(roles)
    return plan, roles


def normalized_composition(value):
    values = {key: Fraction(str(number)) for key, number in value.items() if not key.startswith('@')}
    if any(number < 0 for number in values.values()):
        raise ValueError('negative composition amount')
    values = {key: number for key, number in values.items() if number > 0}
    total = sum(values.values(), Fraction(0))
    if total <= 0:
        raise ValueError('empty composition')
    return '|'.join(key + ':' + str(values[key] / total) for key in sorted(values))


class TrustedDisclosure:
    """Physically parses full public source; exposes only role/per-step authorized slices.

    This is an audited access contract, not encryption or physical row-level blinding.
    """
    def __init__(self, raw_path, roles):
        if sha(raw_path) != RAW_SHA:
            raise ValueError('public source changed')
        with gzip.open(raw_path, 'rt', encoding='utf-8') as stream:
            self._payload = json.load(stream, object_pairs_hook=unique)
        self._columns = {name: i for i, name in enumerate(self._payload['columns'])}
        if len(self._columns) != len(self._payload['columns']) or len(self._payload['index']) != len(self._payload['data']):
            raise ValueError('source schema changed')
        self._roles = roles
        self._next = 0
        self._proxy_pending = False
        self._events = []
        self._cells = {}

    def _row(self, metadata):
        index = metadata['raw_row_zero_based']
        if index >= len(self._payload['data']) or self._payload['index'][index] != metadata['artifact_native_index']:
            raise ValueError('native artifact row/index identity mismatch')
        row = self._payload['data'][index]
        if (row[self._columns['jid']] != metadata['jid'] or row[self._columns['mpid']] != metadata['mpid']
                or normalized_composition(row[self._columns['composition']]) != metadata['normalized_composition']):
            raise ValueError('JID/MPID/composition identity mismatch')
        return row

    def _value(self, row, field):
        value = row[self._columns[FIELDS[field]]]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError('nonfinite selected native source value')
        return float(value)

    def initial_inputs(self):
        warm = [{'identity': meta, 'native_values': [self._value(self._row(meta), j) for j in range(3)]}
                for meta in self._roles['representatives']['warm']]
        terminal = [{'identity': meta, 'native_proxies': [self._value(self._row(meta), j) for j in (0, 1)]}
                    for meta in self._roles['representatives']['terminal']]
        return {'warm': warm, 'pool_metadata_only': self._roles['representatives']['pool'], 'terminal': terminal}

    def pool_proxies(self, index):
        if type(index) is not int or index != self._next or not 0 <= index < 32 or self._proxy_pending:
            raise ValueError('next registered pool row proxies must be disclosed once before target')
        metadata = self._roles['representatives']['pool'][index]
        row = self._row(metadata)
        values = [self._value(row, j) for j in (0, 1)]
        for field, value in zip(FIELDS[:2], values):
            key = (metadata['component_id'], field)
            if key in self._cells:
                raise ValueError('duplicate physical cell in fixed acquisition sequence')
            self._cells[key] = value
        self._proxy_pending = True
        record = {'actual_utc': utc(), 'action': 'reveal_registered_pool_proxies', 'pool_index': index,
            'identity': metadata, 'native_proxies': values, 'incremental_cost': 2, 'total_new_cell_cost': len(self._cells),
            'true_origin_raw_sha256': RAW_SHA, 'target_extracted_or_returned': False}
        self._events.append(record)
        return record

    def pool_target(self, index, forecast_path, expected_sources):
        if index != self._next or not self._proxy_pending:
            raise ValueError('target disclosure requires prior same-row proxy request')
        metadata = self._roles['representatives']['pool'][index]
        frozen = read(forecast_path)
        if (frozen['status'] != 'PREQUENTIAL_LAW_FROZEN_BEFORE_OWN_TARGET'
                or frozen['identity'] != metadata or frozen['pool_index'] != index
                or frozen['source_sha256'] != expected_sources or frozen['native_target_visible'] is not False
                or frozen['law']['family'] != 'student_t' or frozen['law']['scale'] <= 0):
            raise ValueError('actualsame-row pre-target law freeze required')
        freeze_pin = sha(forecast_path)
        value = self._value(self._row(metadata), 2)
        key = (metadata['component_id'], FIELDS[2])
        if key in self._cells:
            raise ValueError('duplicate target physical cell')
        self._cells[key] = value
        self._events.append({'actual_utc': utc(), 'action': 'reveal_registered_pool_target_after_law_freeze',
            'pool_index': index, 'identity': metadata, 'native_target': value, 'incremental_cost': 1,
            'total_new_cell_cost': len(self._cells), 'prequential_forecast_file': str(forecast_path),
            'prequential_forecast_sha256': freeze_pin, 'true_origin_raw_sha256': RAW_SHA,
            'model_generated_label': False, 'freeze_actual_utc': frozen['actual_utc']})
        self._next += 1; self._proxy_pending = False
        return self._events[-1]

    def ledger(self):
        return {'events': self._events, 'unique_new_native_cells': len(self._cells),
            'completed_pool_rows': self._next, 'common_warm_cells': 192, 'common_terminal_proxy_cells': 256,
            'terminal_target_cells_extracted_for_model': 0,
            'physical_access': 'full gzipparsed by trustedreader; selected terminalY not extracted or exposed',
            'cell_count_not_actual_DFT_cost': True}

    def terminal_truth_after_freeze(self):
        return [{'identity': meta, 'native_target': self._value(self._row(meta), 2)}
                for meta in self._roles['representatives']['terminal']]


def verify_forecast(directory, plan_path, roles):
    directory = Path(directory)
    receipt = read(directory / 'FORECAST_FREEZE.json')
    expected_files = set(BASE_FILES)
    expected_files.update('prequential/%04d_FORECAST.json' % i for i in range(32))
    expected_files.update('prequential/%04d_OUTCOME_UPDATE.json' % i for i in range(32))
    if (receipt['status'] != 'ALL2560_TERMINAL_AND32_PREQUENTIAL_LAWS_FROZEN'
            or receipt['source_sha256'] != sources() or receipt['plan_sha256'] != sha(plan_path)
            or set(receipt['files']) != expected_files):
        raise ValueError('complete registered forecast freeze required before terminal target access')
    for name, pin in receipt['files'].items():
        if sha(directory / name) != pin:
            raise ValueError('frozen scientific payload changed')
    forecasts = read(directory / 'FORECASTS.json')['forecasts']
    expected = {(meta['component_id'], method, count) for meta in roles['representatives']['terminal']
                for method in METHODS for count in CHECKPOINTS}
    actual = [(item['component_id'], item['method'], item['feedback_rows']) for item in forecasts]
    if len(actual) != 2560 or len(set(actual)) != 2560 or set(actual) != expected:
        raise ValueError('missing/duplicate or extraneous primary forecasts')
    jids = {meta['component_id']: meta['jid'] for meta in roles['representatives']['terminal']}
    for item in forecasts:
        law = item['law']
        if item['jid'] != jids[item['component_id']] or item['new_native_cell_budget'] != 3 * item['feedback_rows']:
            raise ValueError('forecast identity or budget changed')
        if item['method'] == METHODS[-1]:
            weights = [Fraction(value) for value in law['weight_fractions']]
            if (law['family'] != 'weighted_empirical' or len(weights) != 64 + item['feedback_rows']
                    or len(law['support_native']) != len(weights) or any(x < 0 for x in weights)
                    or sum(weights, Fraction(0)) != 1):
                raise ValueError('invalid exact finite empirical distribution')
        elif not all(type(law[key]) in (int, float) and math.isfinite(law[key]) for key in ('location', 'scale')) or law['scale'] <= 0:
            raise ValueError('invalid scalar predictive parameters')
    return forecasts, receipt

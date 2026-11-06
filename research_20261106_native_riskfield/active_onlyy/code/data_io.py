"""Strict metadata and per-policy only-Y disclosure; no numerical models here."""
from datetime import datetime, timezone
from fractions import Fraction
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

FILES = ('posterior_update.py', 'risk_readout.py', 'reference_models.py', 'data_io.py', 'run_active.py', 'score.py')
POLICIES = ('native_student', 'native_gaussian', 'latent_ivr', 'random', 'gp_variance', 'warm_openloop')
BUDGETS = (0, 4, 8)
REF_POLICIES = ('native_student', 'random')
RAW_SHA = '3d0fc1315b74cf84071f16aa25bb93434dafbc83867beb75872064ac245bd747'
FIELDS = ('e_form', 'gap opt', 'bulk modulus')
SCOPE = 'ACTIVE_ONLYY_FIXED_NIG_PILOT'
BASE_FILES = ('EXECUTION_START.json', 'LIBRARY_VERSIONS.json', 'VISIBLE_INPUTS.json', 'MODEL_STATES.json',
              'QUERY_LEDGER.json', 'TERMINAL_FORECASTS.json', 'REFERENCE_FORECASTS.json', 'PREQUENTIAL_INDEX.json', 'FIT_INDEX.json')


def utc(): return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result: raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def read(path):
    def invalid(value): raise ValueError('nonstandard JSON constant: ' + value)
    return json.loads(Path(path).read_text('utf-8-sig'), object_pairs_hook=unique, parse_constant=invalid)


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')


def sources(): return {name: sha(Path(__file__).resolve().parent / name) for name in FILES}


def configuration():
    return {'version': 'active_onlyY_pooled_NIG_v1', 'policies': list(POLICIES), 'budgets': list(BUDGETS),
        'warm': 64, 'pool': 24, 'terminal_development': 64, 'queries_per_policy': 8,
        'common_native_cell_cost': 368, 'perquery_Y_cost': 1, 'forced_budget_no_early_STOP': True,
        'model': 'singlepooledp2 conditionalNIG; G0 validknowncontext, no branchorBeta',
        'warm_coordinates': 'fixedall3YJhalf/nativeSD thenfixedwarmu mean/sampleSD',
        'owncontexts_update_global_NIG': False,
        'primary_prediction': 'actualYupdatedNIG for all6policy trajectories',
        'warm_openloop_acquisition': 'warmmean;currentVdesignonly;alpha=warmalpha+i/2;b=(alpha-1)*warmb/(warmalpha-1);predictiveactualNIGtrueYupdated',
        'Gaussian_acquisition': 'latentmean/covariance matched then same nativeYJCRPS; not nativemomentmatch',
        'latent_IVR': 'mean_t (b/(alpha-1))*(phi_qVphi_t)^2/(1+phi_qVphi_q), latentvarianceobjective',
        'random_seed': 20261106, 'tie_rule': 'first inoriginalrolepoolorder',
        'GP_variance': 'currentGP predictivevarianceincludingWhite onremainingpool;fixedoriginalkernelrecipe;reoptimizeactualYonly',
        'GP_RF_X_scales': 'fixedwarmnativeproxymean/sampleSD', 'GP_normalize_y': 'currentactualtrainlabels',
        'reference_prediction_policies': list(REF_POLICIES), 'reference_prediction_budgets': list(BUDGETS),
        'main_terminal_laws': 1152, 'reference_terminal_laws': 960, 'secondary_prequential_laws': 48,
        'actual_sklearn_fits': 18, 'actual_GP_fits': 13, 'actual_RF_fits': 5,
        'shared_warm_GP_RF_fits': 2, 'reference_fit_total_including_shared': 10, 'extra_GP_policy_fits': 8, 'GP_policy_own_prediction_laws': 192,
        'libraries': {'numpy': '2.1.3', 'scipy': '1.15.3', 'scikit-learn': '1.6.1', 'joblib': '1.4.2'},
        'native_score': 'unchangedfrozenreported-feedbackscore_law; nativeYJJac/empiricalCDF',
        'all152_components_consumed_development': True, 'counterfactual_policy_reuse_not_materialN': True,
        'new_noise_model_flow_LLM_or_pseudolabels': False}


def validate_roles(roles):
    representatives = roles['representatives']; excluded = set(roles['all_prior_excluded_component_ids'])
    if set(representatives) != {'warm', 'pool', 'terminal'} or len(excluded) != 776:
        raise ValueError('fixed3roles andall776excludedwholecomponents required')
    groups, rows, jids = [], [], []
    for name, count in (('warm', 64), ('pool', 24), ('terminal', 64)):
        if len(representatives[name]) != count: raise ValueError('fixed64/24/64 rolecounts required')
        for item in representatives[name]:
            if item['role'] != name or item['component_id'] in excluded or type(item['raw_row_zero_based']) is not int:
                raise ValueError('invalid/consumedroleidentity')
            groups.append(item['component_id']); rows.append(item['raw_row_zero_based']); jids.append(item['jid'])
    if len(set(groups)) != 152 or len(set(rows)) != 152 or len(set(jids)) != 152:
        raise ValueError('repeated/overlappingphysicalgrouporrow')
    if roles['pool_sequence_component_ids'] != [r['component_id'] for r in representatives['pool']]:
        raise ValueError('originalmetadataordertie rulechanged')


def validate(plan_file, approval_file, phase):
    plan, approval = read(plan_file), read(approval_file)
    if (set(plan) != {'kind', 'status', 'registered_actual_utc', 'source_sha256', 'configuration', 'inputs', 'external_sources', 'math_receipts'}
            or plan['kind'] != 'active_onlyY_pooled_conditional_NIG' or plan['status'] != 'REGISTERED'
            or not plan['registered_actual_utc'] or plan['source_sha256'] != sources()
            or plan['configuration'] != configuration()): raise ValueError('exactregisteredsource/configuration required')
    if (approval.get('approved') is not True or approval.get('scope') != SCOPE
            or approval.get('plan_sha256') != sha(plan_file) or phase not in approval.get('allowed_phases', [])):
        raise ValueError('rootapprovalofexactplan/phase required')
    if set(plan['inputs']) != {'raw_source', 'roles', 'root_scope'}: raise ValueError('threepinned scientificinputs required')
    for item in list(plan['inputs'].values()) + list(plan['external_sources'].values()) + list(plan['math_receipts'].values()):
        if sha(item['file']) != item['sha256']: raise ValueError('scientificinput/source/mathreceipt changed')
    if plan['inputs']['raw_source']['sha256'] != RAW_SHA: raise ValueError('originalpublicsource required')
    roles = read(plan['inputs']['roles']['file']); validate_roles(roles)
    return plan, roles


def load_module(name, reference):
    path = Path(reference['file']).resolve()
    if sha(path) != reference['sha256']: raise ValueError('frozen externalmodule changed')
    if name in sys.modules:
        if Path(sys.modules[name].__file__).resolve() != path: raise ValueError('conflictingfrozenmodule namespace')
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path); module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module; spec.loader.exec_module(module)
    return module


def warm_module(plan): return load_module('joint_model', plan['external_sources']['warm_joint_model'])


def score_module(plan):
    warm_module(plan); load_module('data_scope', plan['external_sources']['legacy_score_scope'])
    return load_module('_frozen_reported_score', plan['external_sources']['legacy_score'])


def composition(value):
    values = {key: Fraction(str(number)) for key, number in value.items() if not key.startswith('@')}
    if any(value < 0 for value in values.values()): raise ValueError('negativecomposition')
    values = {key: number for key, number in values.items() if number > 0}; total = sum(values.values(), Fraction(0))
    if total <= 0: raise ValueError('emptycomposition')
    return '|'.join(key + ':' + str(values[key] / total) for key in sorted(values))


class Disclosure:
    """Full source physically parsed; independent perpolicy observed-cell ledgers."""
    def __init__(self, raw_file, roles):
        if sha(raw_file) != RAW_SHA: raise ValueError('rawsourcechanged')
        with gzip.open(raw_file, 'rt', encoding='utf-8') as stream: self._payload = json.load(stream, object_pairs_hook=unique)
        self._columns = {name: i for i, name in enumerate(self._payload['columns'])}
        if len(self._columns) != len(self._payload['columns']): raise ValueError('duplicatecolumns')
        self.roles = roles; self._seen = {name: set() for name in POLICIES}; self._events = []

    def _row(self, item):
        i = item['raw_row_zero_based']; row = self._payload['data'][i]
        if (self._payload['index'][i] != item['artifact_native_index'] or row[self._columns['jid']] != item['jid']
                or row[self._columns['mpid']] != item['mpid'] or composition(row[self._columns['composition']]) != item['normalized_composition']):
            raise ValueError('nativeidentity mismatch')
        return row

    def _values(self, item, indices):
        row = self._row(item); values = [row[self._columns[FIELDS[j]]] for j in indices]
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
            raise ValueError('nonfiniteauthorizedcell')
        return [float(value) for value in values]

    def initial(self):
        return {'warm': [{'identity': r, 'native_values': self._values(r, (0, 1, 2))} for r in self.roles['representatives']['warm']],
            'pool': [{'identity': r, 'native_proxies': self._values(r, (0, 1))} for r in self.roles['representatives']['pool']],
            'terminal': [{'identity': r, 'native_proxies': self._values(r, (0, 1))} for r in self.roles['representatives']['terminal']]}

    def query_Y(self, policy, index, frozen_file, expected_sources, expected_plan_sha256):
        if policy not in POLICIES or type(index) is not int or not 0 <= index < 24:
            raise ValueError('onlyregisteredpoolY actions allowed')
        if index in self._seen[policy] or len(self._seen[policy]) >= 8:
            raise ValueError('repeatedcellorbudgetexhaustion')
        metadata = self.roles['representatives']['pool'][index]; frozen = read(frozen_file)
        if (frozen['status'] != 'ACTION_AND_PREQUENTIAL_LAW_FROZEN_BEFORE_QUERY_Y'
                or frozen['policy'] != policy or frozen['pool_index'] != index or frozen['identity'] != metadata
                or frozen['budget_before_query'] != len(self._seen[policy]) or frozen['source_sha256'] != expected_sources
                or frozen['plan_sha256'] != expected_plan_sha256 or frozen['native_query_target_visible'] is not False):
            raise ValueError('correctsamepolicy/actionlawfreeze requiredbeforetrueY')
        pin = sha(frozen_file); value = self._values(metadata, (2,))[0]
        self._seen[policy].add(index)
        record = {'actual_utc': utc(), 'policy': policy, 'pool_index': index, 'identity': metadata,
            'native_target': value, 'canonical_cell': [metadata['component_id'], FIELDS[2]], 'incremental_cost': 1,
            'policy_total_query_cost': len(self._seen[policy]), 'prequery_file_sha256': pin,
            'prequery_actual_utc': frozen['actual_utc'], 'true_origin_raw_sha256': RAW_SHA, 'synthetic_label': False}
        self._events.append(record); return record

    def ledger(self):
        return {'events': self._events, 'common_setup_native_cells': 368,
            'cost_per_policy': {name: len(values) for name, values in self._seen.items()},
            'union_pool_components_with_Y_extracted': len(set(event['identity']['component_id'] for event in self._events)),
            'counterfactual_reuse_not_new_materialN': True, 'terminalY_model_access': 0,
            'physical_scope': 'fullgziptrustedparse; onlyrole/perpolicyauthorizednumericcells exposed, notcryptographicblindness'}


def verify_freeze(directory, plan_file, roles):
    directory = Path(directory); freeze = read(directory / 'FORECAST_FREEZE.json')
    if (freeze['status'] != 'ALL2112_TERMINAL_AND48_PREQUENTIAL_LAWS_FROZEN'
            or freeze['plan_sha256'] != sha(plan_file) or freeze['source_sha256'] != sources()
            or not set(BASE_FILES).issubset(freeze['files'])): raise ValueError('completeexactforecastfreeze required')
    for name, pin in freeze['files'].items():
        path = (directory / name).resolve(); path.relative_to(directory.resolve())
        if sha(path) != pin: raise ValueError('frozenpayloadchanged')
    primary = read(directory / 'TERMINAL_FORECASTS.json')['forecasts']
    references = read(directory / 'REFERENCE_FORECASTS.json')['forecasts']
    expected = {(r['component_id'], p, n) for r in roles['representatives']['terminal'] for p in POLICIES for n in BUDGETS}
    keys = [(r['component_id'], r['policy'], r['budget']) for r in primary]
    if len(keys) != 1152 or len(set(keys)) != 1152 or set(keys) != expected: raise ValueError('incompleteprimaryforecasts')
    expected_ref = {(r['component_id'], p, n, m) for r in roles['representatives']['terminal'] for p in REF_POLICIES for n in BUDGETS for m in ('GP', 'RF')}
    expected_ref.update((r['component_id'], 'gp_variance', n, 'GP') for r in roles['representatives']['terminal'] for n in BUDGETS)
    keys = [(r['component_id'], r['policy'], r['budget'], r['method']) for r in references]
    if len(keys) != 960 or len(set(keys)) != 960 or set(keys) != expected_ref: raise ValueError('incompletereferenceforecasts')
    preq = read(directory / 'PREQUENTIAL_INDEX.json')['records']; fits = read(directory / 'FIT_INDEX.json')['records']
    if len(preq) != 48 or len(fits) != 18: raise ValueError('exact48prequentialand18fits required')
    for item in preq:
        for name, key in ((item['forecast_file'], 'forecast_sha256'), (item['outcome_file'], 'outcome_sha256')):
            if freeze['files'].get(name) != item[key]: raise ValueError('prequentialnotboundbyfreeze')
    for item in fits:
        if freeze['files'].get(item['file']) != item['sha256']: raise ValueError('learnedstate notfrozen')
    if len(freeze['files']) != len(BASE_FILES) + 2 * 48 + 18: raise ValueError('missing/extraneouspayload')
    return primary, references, preq, freeze


def reveal_terminal_truth(plan, roles, forecast_directory, plan_file):
    verify_freeze(forecast_directory, plan_file, roles)
    evaluator = Disclosure(plan['inputs']['raw_source']['file'], roles)
    return [{'identity': item, 'native_target': evaluator._values(item, (2,))[0]} for item in roles['representatives']['terminal']]

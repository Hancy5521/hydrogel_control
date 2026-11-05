"""Pinned metadata, declared roles and a restricted offline bundle oracle.

Only original, finite source values can be revealed. There are no pseudo-label
promotion or physical-independence claims. Python privacy is an audit boundary,
not a security sandbox against a malicious caller.
"""
from collections import defaultdict
from datetime import datetime, timezone
from fractions import Fraction
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

SYMBOLS = ('H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn '
           'Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd '
           'Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac '
           'Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og').split()
SOURCE_HASHES = {
    'ucsb_thermoelectrics': '7d0c9c74fd52995b876bd251a7732b5498314702441fa463197fb492be68ecc0',
    'jarvis_dft_3d': '3d0fc1315b74cf84071f16aa25bb93434dafbc83867beb75872064ac245bd747',
}
TARGETS = {
    'ucsb_thermoelectrics': ['rho (ohm.cm)', 'S [muV/K]', 'kappa [W/mK]'],
    'jarvis_dft_3d': ['e_form', 'gap opt', 'bulk modulus'],
}
BUNDLES = ((0, 1), (2,))
ROLES = ('warm', 'repair', 'frontier', 'test')
ROLE_COUNTS = (8, 8, 16, 8)
SEEDS = (2026110501, 2026110502, 2026110503, 2026110504)
CHECKPOINTS = (0, 8, 16, 24)
POLICIES = ('random', 'repair_only_IVR', 'expansion_only_KG', 'fixed_cycle', 'fixed_random_modules',
            'joint_lookahead_all', 'hypercycle_predictive', 'hypercycle_no_cross')
DETERMINISTIC = frozenset(('repair_only_IVR', 'expansion_only_KG', 'fixed_cycle', 'joint_lookahead_all'))
ACTUAL_CAMPAIGNS = 40
LOGICAL_EPISODES = 64
CODE_FILES = ('benchmark_io.py', 'joint_gaussian.py', 'policies.py',
              'run_campaign.py', 'evaluate_campaign.py')
INPUT_FILES = ('ELIGIBLE_METADATA.json', 'PILOT_ELIGIBILITY.json', 'DATA_MANIFEST.json',
               'SOURCE_GATE.json', 'SOURCE_GATE_PILOT_ADDENDUM.json',
               'ucsb_thermoelectrics.json.gz', 'jarvis_dft_3d.json.gz')


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    def reject_constant(value):
        raise ValueError('Nonstandard JSON constant: ' + value)
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON key: ' + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text('utf-8-sig'), parse_constant=reject_constant,
                      object_pairs_hook=unique)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def source_bindings():
    here = Path(__file__).resolve().parent
    return {name: sha(here / name) for name in CODE_FILES}


def expected_schedule():
    """Exact actual campaign identities and transparent deterministic aliases."""
    actual, logical = [], []
    for dataset in SOURCE_HASHES:
        for policy in POLICIES:
            for seed in SEEDS[:1] if policy in DETERMINISTIC else SEEDS:
                actual.append({'name': dataset + '__' + policy + '__' + str(seed),
                               'dataset': dataset, 'policy': policy, 'seed': seed})
            for seed in SEEDS:
                used_seed = SEEDS[0] if policy in DETERMINISTIC else seed
                logical.append({'dataset': dataset, 'policy': policy, 'seed': seed,
                    'campaign': dataset + '__' + policy + '__' + str(used_seed),
                    'is_alias': policy in DETERMINISTIC and seed != used_seed})
    return actual, logical


def validate_bindings(plan):
    expected_keys = {'kind', 'registered_actual_utc', 'data_directory', 'source_sha256',
        'input_sha256', 'independent_role_plan_file', 'independent_role_plan_sha256',
        'datasets', 'policies', 'seeds', 'checkpoints', 'new_bundle_budget', 'bundle_schema',
        'each_bundle_cost', 'cost_is_offline_assumption_not_measured_lab_or_DFT_cost',
        'role_counts_groups', 'rows_per_group_cap', 'actual_campaign_count',
        'logical_episode_count', 'deterministic_alias_policy', 'primary_endpoint',
        'relation_learning', 'no_novelty_or_topconference_success_certification',
        'no_protected_old_MLST_input', 'expected_stop', 'runtime_metadata'}
    if not isinstance(plan, dict) or set(plan) != expected_keys:
        raise ValueError('Exact metadata protocol fields required; no unknown response arrays')
    if (plan['kind'] != 'source_pinned_classical_catalytic_feasibility_pilot'
            or set(plan['input_sha256']) != set(INPUT_FILES)
            or plan['policies'] != list(POLICIES) or plan['seeds'] != list(SEEDS)
            or plan['checkpoints'] != list(CHECKPOINTS)
            or plan['new_bundle_budget'] != 24 or plan['each_bundle_cost'] != 1
            or plan['bundle_schema'] != [list(x) for x in BUNDLES]
            or plan['rows_per_group_cap'] != 2
            or plan['role_counts_groups'] != dict(zip(ROLES, ROLE_COUNTS))
            or plan['actual_campaign_count'] != ACTUAL_CAMPAIGNS or plan['logical_episode_count'] != LOGICAL_EPISODES
            or plan['primary_endpoint'] != 'equal_test_metadata_group_mean_standardized_Gaussian_CRPS'
            or plan['cost_is_offline_assumption_not_measured_lab_or_DFT_cost'] is not True
            or plan['no_novelty_or_topconference_success_certification'] is not True
            or plan['no_protected_old_MLST_input'] is not True):
        raise ValueError('Fixed bounded protocol declaration changed')
    if plan['source_sha256'] != source_bindings():
        raise ValueError('Pilot source bytes changed after registration')
    directory = Path(plan['data_directory'])
    for name, digest in plan['input_sha256'].items():
        if sha(directory / name) != digest:
            raise ValueError('Input changed after registration: ' + name)
    if sha(plan['independent_role_plan_file']) != plan['independent_role_plan_sha256']:
        raise ValueError('Independent metadata role plan changed')
    independent = read(plan['independent_role_plan_file'])
    expected_design = select_design(read(directory / 'ELIGIBLE_METADATA.json'))
    if plan['datasets'] != expected_design:
        raise ValueError('Protocol changed the deterministic metadata selection')
    for dataset, design in expected_design.items():
        other = independent['datasets'][dataset]
        if other['rows'] != design['rows'] or other['target_fields'] != design['target_fields']:
            raise ValueError('Independent role/response plan does not agree')


def select_design(metadata):
    """Roles and row caps depend only on supplied full-source identity metadata."""
    datasets = {}
    for dataset in SOURCE_HASHES:
        grouped = defaultdict(list)
        for row in metadata['datasets'][dataset]:
            grouped[row['component_id']].append(row)
        ordered = sorted(grouped, key=lambda g: (hashlib.sha256(
            ('hypercycle_pilot_v1|' + g).encode()).hexdigest(), g))
        if len(ordered) < sum(ROLE_COUNTS):
            raise ValueError('Fewer than 40 eligible conservative metadata groups')
        selected, start = [], 0
        for role, count in zip(ROLES, ROLE_COUNTS):
            for group in ordered[start:start + count]:
                rows = sorted(grouped[group], key=lambda r: (hashlib.sha256(
                    ('hypercycle_pilot_v1|row|' + dataset + '|' + str(r['raw_row_zero_based'])).encode()
                ).hexdigest(), r['raw_row_zero_based']))[:2]
                selected.extend(dict(row, role=role) for row in rows)
            start += count
        ids = [r['raw_row_zero_based'] for r in selected]
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate original rows in selected roles')
        datasets[dataset] = {'target_fields': TARGETS[dataset], 'rows': selected,
                             'frontier_utility_sign': -1 if dataset == 'ucsb_thermoelectrics' else 1,
                             'selected_group_count': 40,
                             'unused_eligible_group_count': len(ordered) - 40,
                             'unused_eligible_group_ids': ordered[40:],
                             'row_count': len(selected), 'identity_is_metadata_only': True}
    return datasets


def feature_matrix(dataset, rows):
    """Atom fractions, and known T for UCSB. No response/derived/source features."""
    positions = {s: i for i, s in enumerate(SYMBOLS)}
    if len(positions) != 118:
        raise ValueError('Expected the declared 118-element feature dictionary')
    matrix = np.zeros((len(rows), 119 if dataset == 'ucsb_thermoelectrics' else 118))
    for i, row in enumerate(rows):
        amounts = {}
        for token in row['normalized_composition'].split('|'):
            symbol, value = token.split(':', 1)
            if symbol not in positions or symbol in amounts:
                raise ValueError('Invalid canonical composition symbol')
            amounts[symbol] = float(Fraction(value))
        if (not amounts or any(v < 0 or not np.isfinite(v) for v in amounts.values())
                or not np.isclose(sum(amounts.values()), 1., rtol=0., atol=1e-12)):
            raise ValueError('Invalid atom fractions')
        for symbol, value in amounts.items():
            matrix[i, positions[symbol]] = value
        if dataset == 'ucsb_thermoelectrics':
            temp = row['temperature_K']
            if type(temp) not in (int, float) or not np.isfinite(temp):
                raise ValueError('Finite known temperature required')
            matrix[i, -1] = temp / 1000.
    # Transductive metadata scaling: all methods may see the same unlabeled X.
    center, scale = matrix.mean(axis=0), matrix.std(axis=0, ddof=0)
    zero = scale == 0
    scale[zero] = 1.
    active = int((~zero).sum())
    distance_scale = float(np.sqrt(active)) if active else 1.
    return (matrix - center) / scale / distance_scale, {'metadata_only_center': center.tolist(),
                                     'metadata_only_scale': scale.tolist(),
                                     'constant_metadata_columns': np.flatnonzero(zero).tolist(),
                                     'nonconstant_metadata_dimension': active,
                                     'RMS_feature_distance_scale': distance_scale}


class BundleOracle:
    """Stateful offline native-label lookup with explicit role/cost receipts."""
    def __init__(self, dataset, data_directory, design):
        if dataset not in SOURCE_HASHES or design['target_fields'] != TARGETS[dataset]:
            raise ValueError('Undeclared dataset/response fields')
        path = Path(data_directory) / (dataset + '.json.gz')
        if sha(path) != SOURCE_HASHES[dataset]:
            raise ValueError('Original public table hash changed')
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            payload = json.load(stream)
        column = {name: i for i, name in enumerate(payload['columns'])}
        # Keep only declared native responses. Never expose PF, zT, sigma, etc.
        self._responses = np.full((len(design['rows']), 3), np.nan)
        for i, row in enumerate(design['rows']):
            if row['role'] != 'test':
                self._responses[i] = [payload['data'][row['raw_row_zero_based']][column[name]]
                                      for name in TARGETS[dataset]]
        pool = [i for i, row in enumerate(design['rows']) if row['role'] != 'test']
        if self._responses.shape != (len(design['rows']), 3) or not np.isfinite(self._responses[pool]).all():
            raise ValueError('Every pre-eligible original response must be finite')
        self.rows = design['rows']
        self.observed = np.zeros(self._responses.shape, dtype=bool)
        self.receipts = []
        self.campaign_cost = 0
        self.initial_cost = 0

    def fork(self):
        """Independent simulated episode state, same pinned role-filtered source."""
        other = object.__new__(BundleOracle)
        other._responses = self._responses.copy()
        other.rows = [dict(row) for row in self.rows]
        other.observed = np.zeros_like(self.observed)
        other.receipts = []
        other.campaign_cost = 0
        other.initial_cost = 0
        return other

    def query(self, row, bundle, *, initial=False):
        if type(row) is not int or not 0 <= row < len(self.rows) or type(bundle) is not int or bundle not in (0, 1):
            raise ValueError('Declared row and bundle required')
        role = self.rows[row]['role']
        fields = BUNDLES[bundle]
        if role == 'test':
            raise ValueError('Test labels cannot be queried by a campaign')
        if initial:
            if role != 'warm' and not (role == 'repair' and bundle == 0):
                raise ValueError('Only warm/all and repair/first-two initialization is allowed')
        elif role not in ('repair', 'frontier'):
            raise ValueError('Only the declared repair/frontier pool can consume action budget')
        if self.observed[row, list(fields)].any():
            raise ValueError('A known or partly known bundle cannot be charged/revealed twice')
        if not initial and self.campaign_cost >= 24:
            raise ValueError('Campaign bundle budget exhausted')
        kind = 'R' if self.observed[row].any() else 'E'
        values = self._responses[row, list(fields)].copy()
        self.observed[row, list(fields)] = True
        self.initial_cost += int(initial)
        self.campaign_cost += int(not initial)
        record = {'sequence': len(self.receipts), 'row_position': row,
                  'original_raw_row': self.rows[row]['raw_row_zero_based'],
                  'component_id': self.rows[row]['component_id'], 'role': role,
                  'bundle_id': bundle, 'property_indices': list(fields),
                  'flat_indices': [3 * row + p for p in fields], 'native_values': values.tolist(),
                  'cost': 1, 'phase': 'initial' if initial else 'campaign',
                  'dynamic_kind_before_reveal': kind,
                  'value_role': 'original_public_label_lookup_not_new_measurement'}
        self.receipts.append(record)
        return record

    def available_actions(self):
        actions = []
        for row, meta in enumerate(self.rows):
            if meta['role'] not in ('repair', 'frontier'):
                continue
            kind = 'R' if self.observed[row].any() else 'E'
            for bundle, fields in enumerate(BUNDLES):
                if not self.observed[row, list(fields)].any():
                    actions.append({'row': row, 'bundle': bundle, 'kind': kind,
                                    'indices': [3 * row + p for p in fields], 'cost': 1})
        return actions


def evaluator_truth(dataset, data_directory, design, *, forecasts_verified):
    if forecasts_verified is not True:
        raise ValueError('Verify the complete forecast freeze before evaluator label access')
    # Separate evaluator entry; this is never called from run_campaign.py.
    path = Path(data_directory) / (dataset + '.json.gz')
    if dataset not in SOURCE_HASHES or sha(path) != SOURCE_HASHES[dataset]:
        raise ValueError('Original evaluation source changed')
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        payload = json.load(stream)
    column = {name: i for i, name in enumerate(payload['columns'])}
    values = np.asarray([[payload['data'][row['raw_row_zero_based']][column[name]]
                         for name in TARGETS[dataset]] for row in design['rows']], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('Registered complete-case evaluator truth must remain finite')
    return values

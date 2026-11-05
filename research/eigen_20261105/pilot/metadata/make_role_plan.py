"""Independent metadata-only role selection; no producer imports or labels.

Read exactly the eligible metadata and source gate. An immutable role plan is
written before any real fit. The offline utility signs were chosen by root
before fitting: lower thermal conductivity / higher source bulk-modulus field.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent.parent / 'data'
INPUTS = ('ELIGIBLE_METADATA.json', 'SOURCE_GATE.json')
DATASETS = ('ucsb_thermoelectrics', 'jarvis_dft_3d')
ROLES = ('warm', 'repair', 'frontier', 'test')
COUNTS = (8, 8, 16, 8)
PREFIX = 'hypercycle_pilot_v1|'
TARGETS = {
    'ucsb_thermoelectrics': ['rho (ohm.cm)', 'S [muV/K]', 'kappa [W/mK]'],
    'jarvis_dft_3d': ['e_form', 'gap opt', 'bulk modulus'],
}
TARGET_UNITS = {
    'ucsb_thermoelectrics': ['ohm.cm', 'microvolt/kelvin', 'watt/(metre kelvin)'],
    'jarvis_dft_3d': ['electronvolt/atom', 'electronvolt', 'GPa (source field; aggregation interpretation qualified in SOURCE_GATE_PILOT_ADDENDUM)'],
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha_text(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def main():
    metadata = json.loads((DATA / INPUTS[0]).read_text('utf-8'))
    gate = json.loads((DATA / INPUTS[1]).read_text('utf-8'))
    assert gate['decision'] == 'TWO_TABLES_ADMITTED_FOR_A_PREREGISTERED_OFFLINE_PILOT_ONLY'
    known = {row['id'] for row in gate['datasets']}
    assert set(DATASETS) == known
    plan = {
        'artifact_kind': 'Independent pre-fit metadata role/row plan, not a fitted or evaluated result',
        'actual_registered_utc': datetime.now(timezone.utc).isoformat(),
        'input_sha256': {name: digest(DATA / name) for name in INPUTS},
        'independent_selector_script_sha256': digest(__file__),
        'independent_selector_script': 'metadata/make_role_plan.py',
        'read_scope': 'Only ELIGIBLE_METADATA.json and SOURCE_GATE.json; no gzip responses, model imports, forecast files or scores.',
        'declared_group_role_counts': dict(zip(ROLES, COUNTS)),
        'global_group_sort': "(SHA256('hypercycle_pilot_v1|' + component_id), component_id)",
        'row_sort_within_group': "(SHA256('hypercycle_pilot_v1|row|' + dataset + '|' + str(raw_row_zero_based)), raw_row_zero_based)",
        'max_rows_per_group': 2,
        'group_identity_origin': 'Full original-source connectivity, retained from upstream curator metadata before eligibility selection. No identity recomputation on eligible rows.',
        'identity_limit': 'Conservative source/composition components do not certify independent preparations or physical units.',
        'datasets': {},
        'producer_execution': False,
        'model_fit_calls': 0,
        'target_value_or_prediction_score_access': False,
        'disjointness_contract': {
            'no_component_crosses_role': True,
            'no_original_row_duplicated': True,
            'selected_and_unused_components_disjoint': True,
            'no_normalized_composition_crosses_roles': True,
            'UCSB_source_citation_does_not_cross_roles': True,
            'all_selected_rows_are_exact_eligible_metadata_copies_plus_role': True,
        },
        'common_feature_scope': {
            'allowed': 'Atom fractions from normalized composition; additionally UCSB known temperature_K/1000.',
            'excluded_from_feature_descriptors': ['response values', 'sigma [S/cm]', 'PF [W/mK^2]', 'zT', 'source_citation', 'component_id', 'source/row IDs', 'role'],
            'scope_limit': 'Identity/source/role fields remain in this ledger for grouping/provenance only, not predictive features. No response or derived output is stored in this file.',
        },
    }
    for dataset in DATASETS:
        grouped = defaultdict(list)
        for row in metadata['datasets'][dataset]:
            grouped[row['component_id']].append(row)
        order = sorted(grouped, key=lambda component: (sha_text(PREFIX + component), component))
        assert len(order) >= sum(COUNTS)
        selected, per_role, per_role_groups = [], {}, {}
        begin = 0
        for role, n_groups in zip(ROLES, COUNTS):
            group_ids = order[begin:begin + n_groups]
            per_role_groups[role] = group_ids
            local_rows = []
            for component in group_ids:
                row_order = sorted(grouped[component], key=lambda row: (
                    sha_text(PREFIX + 'row|' + dataset + '|' + str(row['raw_row_zero_based'])), row['raw_row_zero_based']))
                for row in row_order[:2]:
                    local_rows.append(dict(row, role=role))
            selected.extend(local_rows)
            per_role[role] = {'groups': len(group_ids), 'rows': len(local_rows)}
            begin += n_groups
        assert begin == 40
        selected_ids = [row['raw_row_zero_based'] for row in selected]
        assert len(set(selected_ids)) == len(selected_ids)
        component_roles = defaultdict(set)
        formula_roles = defaultdict(set)
        source_roles = defaultdict(set)
        for row in selected:
            component_roles[row['component_id']].add(row['role'])
            formula_roles[row['normalized_composition']].add(row['role'])
            if dataset == 'ucsb_thermoelectrics':
                source_roles[row['source_citation']].add(row['role'])
        assert all(len(roles) == 1 for roles in component_roles.values())
        assert all(len(roles) == 1 for roles in formula_roles.values())
        assert all(len(roles) == 1 for roles in source_roles.values())
        unused = order[40:]
        assert set(unused).isdisjoint(component_roles)
        assert len(component_roles) == 40
        assert all(n <= 2 for n in Counter(row['component_id'] for row in selected).values())
        feature_count = 119 if dataset == 'ucsb_thermoelectrics' else 118
        plan['datasets'][dataset] = {
            'target_fields': TARGETS[dataset],
            'target_units': TARGET_UNITS[dataset],
            'rows': selected,
            'selected_group_count': 40,
            'unused_eligible_group_count': len(unused),
            'unused_eligible_group_ids': unused,
            'unused_eligible_rows_in_selected_groups': sum(len(grouped[c]) for c in component_roles) - len(selected),
            'row_count': len(selected),
            'identity_is_metadata_only': True,
            'role_counts': per_role,
            'role_group_ids': per_role_groups,
            'selected_metadata_role_sha256': sha_text(json.dumps(selected, sort_keys=True, separators=(',', ':'), ensure_ascii=False)),
            'predictive_feature_count': feature_count,
            'feature_scope': '118 declared element fractions' + (' + known temperature_K/1000' if dataset == 'ucsb_thermoelectrics' else ''),
            'frontier_utility': {
                'target_index': 2,
                'source_field': TARGETS[dataset][2],
                'sign': -1 if dataset == 'ucsb_thermoelectrics' else 1,
                'definition': 'Finite-pool extreme original-target retrieval; not thermoelectric efficiency/zT or certified real-world material discovery.',
                'chosen_by_root_before_fit': True,
            },
            'source_provenance': {
                'source_gate': 'data/SOURCE_GATE.json',
                'source_gate_sha256': plan['input_sha256']['SOURCE_GATE.json'],
                'original_data_kind': 'experimental source literature aggregation' if dataset == 'ucsb_thermoelectrics' else 'DFT computational properties, not wet measurement',
                'original_property_values_not_loaded': True,
            },
        }
    destination = HERE.parent / 'DATASET_ROLE_PLAN.json'
    with destination.open('x', encoding='utf-8') as stream:
        json.dump(plan, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({'output': str(destination), 'sha256': digest(destination),
                      'role_counts': {name: row['role_counts'] for name, row in plan['datasets'].items()},
                      'no_model_or_response_access': True}))


if __name__ == '__main__':
    main()

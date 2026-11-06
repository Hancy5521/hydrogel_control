"""Registered152-group active-onlyY metadata selector; no native/model access."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import traceback

IDENTITY = {'raw_row_zero_based', 'artifact_native_index', 'component_id',
            'normalized_composition', 'jid', 'mpid', 'composition_amounts'}
INPUTS = {'source_contract', 'eligible_metadata', 'legacy_roles', 'phase1_roles', 'phase2_roles', 'completed_feedback_roles'}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(p):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    return json.loads(Path(p).read_text('utf-8-sig'), object_pairs_hook=unique, parse_constant=invalid)


def write(p, value):
    with Path(p).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def utc():
    return datetime.now(timezone.utc).isoformat()


def configuration():
    return {'version': 'native_riskfield_active_onlyY_v1', 'dataset': 'jarvis_dft_3d',
        'hash_prefix': 'native_riskfield_active_onlyY_v1|',
        'order': 'SHA256(prefix+component_id)hexascending,thencomponent_idascending',
        'original_eligible_components': 8842, 'original_eligible_rows': 11020,
        'legacy_excluded': 40, 'phase1_excluded': 256, 'phase2_excluded': 256,
        'completed_feedback_excluded': 224, 'all_prior_excluded': 776,
        'remaining_before_selection': 8066, 'selected_development_groups': 152,
        'warm': 64, 'pool': 24, 'terminal': 64, 'remaining_after_selection': 7914,
        'role_order': 'first64warm,next24pool,next64terminal',
        'representative': 'minimumeligibleintegerraw_row_zero_basedperfullsourcecomponent',
        'pool_sequence': 'fixedmetadatahashselectionorder;policyactionsmaylaterquerywithinpoolonlyunderseparatescienceprotocol',
        'whole_components_and_all_siblings_reserved': True,
        'all152groups_development_not_final_confirmation': True,
        'planned_common_setup_cells': {'warm': 192, 'pool_dual_proxies': 48, 'terminal_dual_proxies': 128, 'total': 368},
        'future_perpolicy_onlyY_queries': 8,
        'counterfactual_reuse_same24pool_does_not_add_materialN': True,
        'metadata_roles_authorize_native_reads_fits_or_queries': False}


def consumed(parent, count, expected_exclusions, exclusion_key):
    selected = parent['new_development_consumed_component_ids']
    require(len(selected) == len(set(selected)) == count, 'Parent consumed count changed')
    require(set(parent[exclusion_key]) == expected_exclusions, 'Parent full exclusions changed')
    reps = [r for block in parent['representatives'].values() for r in block]
    require(len(reps) == count and {r['component_id'] for r in reps} == set(selected), 'Parent representatives differ')
    return set(selected)


def select(plan):
    q = {key: read(item['file']) for key, item in plan['inputs'].items()}
    contract = q['source_contract']
    eligible_file = plan['inputs']['eligible_metadata']['file']
    require(contract['input_sha256'].get(eligible_file) == plan['inputs']['eligible_metadata']['sha256'], 'Original registry metadata required')
    metadata = q['eligible_metadata']['datasets']['jarvis_dft_3d']
    groups, positions, jids = defaultdict(list), set(), set()
    for row in metadata:
        require(set(row) == IDENTITY, 'Metadata may not contain native property values')
        idx = row['raw_row_zero_based']
        require(type(idx) is int and idx >= 0 and idx not in positions and row['jid'] not in jids,
                'Invalid or duplicate original row/JID')
        require(isinstance(row['jid'], str) and bool(row['jid']) and isinstance(row['component_id'], str) and bool(row['component_id']),
                'Canonical row/group identity required')
        positions.add(idx); jids.add(row['jid']); groups[row['component_id']].append(row)
    require(len(metadata) == 11020 and len(groups) == 8842, 'Original metadata inventory changed')
    legacy = q['legacy_roles']['datasets']['jarvis_dft_3d']
    old = {r['component_id'] for r in legacy['rows']}
    require(len(old) == 40 and old == set(contract['eligibility_and_consumption']['old_consumed_component_ids']), 'Legacy40identity mismatch')
    require(set(groups) == old | set(legacy['unused_eligible_group_ids']) and not old & set(legacy['unused_eligible_group_ids']), 'Legacy fullgroup ledger mismatch')
    first = consumed(q['phase1_roles'], 256, old, 'old_excluded_component_ids')
    second = consumed(q['phase2_roles'], 256, old | first, 'old_excluded_component_ids')
    feedback = consumed(q['completed_feedback_roles'], 224, old | first | second, 'all_prior_excluded_component_ids')
    sets = [old, first, second, feedback]
    require(all(not sets[i] & sets[j] for i in range(4) for j in range(i)), 'Previously consumed cohorts overlap')
    excluded = set.union(*sets)
    unused = set(groups) - excluded
    require(len(excluded) == 776 and len(unused) == 8066
            and unused == set(q['completed_feedback_roles']['remaining_unselected_component_ids_in_registered_hash_order']),
            'Expected8066unusedafter776wholegroups')
    prefix = configuration()['hash_prefix']
    ordered = sorted(unused, key=lambda c: (hashlib.sha256((prefix + c).encode()).hexdigest(), c))
    selected, remaining = ordered[:152], ordered[152:]
    representatives = {'warm': [], 'pool': [], 'terminal': []}
    selected_groups = []
    for rank, comp in enumerate(selected):
        role = 'warm' if rank < 64 else ('pool' if rank < 88 else 'terminal')
        siblings = sorted(groups[comp], key=lambda r: r['raw_row_zero_based'])
        representative = dict(siblings[0], role=role)
        representatives[role].append(representative)
        selected_groups.append({'component_id': comp, 'rank_zero_based': rank, 'role': role,
            'representative': representative, 'eligible_siblings': siblings, 'eligible_sibling_count': len(siblings),
            'whole_source_component_consumed_for_future_confirmation': True,
            'all_original_source_siblings_enumerated': False,
            'ineligible_sibling_scope': 'Not listed by eligible-only metadata; all original fullcomponent siblings remain reserved to the same role.'})
    require([len(representatives[k]) for k in ('warm', 'pool', 'terminal')] == [64, 24, 64]
            and len(remaining) == 7914, 'Declared152role counts changed')
    return {'artifact_kind': 'METADATA_ONLY_NATIVE_RISKFIELD_ONLYY_ROLE_SELECTION', 'actual_utc': utc(),
        'configuration': configuration(), 'representatives': representatives, 'selected_groups': selected_groups,
        'pool_sequence_component_ids': [r['component_id'] for r in representatives['pool']],
        'pool_sequence_jids': [r['jid'] for r in representatives['pool']],
        'legacy40_excluded_component_ids': sorted(old), 'phase1_development256_excluded_component_ids': sorted(first),
        'phase2_development256_excluded_component_ids': sorted(second), 'completed_feedback224_excluded_component_ids': sorted(feedback),
        'all_prior_excluded_component_ids': sorted(excluded), 'new_development_consumed_component_ids': selected,
        'remaining_unselected_component_ids_in_registered_hash_order': remaining,
        'complete_unused_hash_order_sha256': hashlib.sha256(json.dumps(ordered, separators=(',', ':')).encode()).hexdigest(),
        'group_count': {'legacy_excluded': 40, 'phase1_excluded': 256, 'phase2_excluded': 256,
                        'completed_feedback_excluded': 224, 'all_prior_excluded': 776,
                        'warm': 64, 'pool': 24, 'terminal': 64, 'selected_development': 152, 'remaining_unselected': 7914},
        'raw_asset_reference_not_read': plan['raw_asset_reference_not_read'],
        'response_exposure': {'warm_property_values_read': 0, 'pool_property_values_read': 0,
                              'terminal_property_values_read': 0, 'raw_gzip_opened_or_hashed': False},
        'interpretation': '152wholecomponents andallknownorunlisted siblings reserved development, not finalconfirmation. Contexts/8Y-policy/counterfactual roles are futureplanned only; thisselection authorizesno numeric extraction, prediction, query or score.'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan', required=True); ap.add_argument('--approval', required=True); ap.add_argument('--output-dir', required=True)
    a = ap.parse_args(); p, approval = read(a.plan), read(a.approval)
    require(p['kind'] == 'NATIVE_RISKFIELD_ONLYY_METADATA_ROLE_SELECTION' and p['status'] == 'REGISTERED'
            and p['configuration'] == configuration() and p['source_sha256'] == sha(__file__), 'Exact registered source/recipe required')
    registered = datetime.fromisoformat(p['registered_actual_utc'])
    require(registered.tzinfo is not None and registered.utcoffset().total_seconds() == 0
            and registered <= datetime.now(timezone.utc), 'Actual registeredUTC before selection required')
    require(approval.get('approved') is True and approval['scope'] == 'METADATA_ONLY_NATIVE_RISKFIELD_ONLYY_SELECTION'
            and approval['source_sha256'] == sha(__file__) and approval['plan_sha256'] == sha(a.plan), 'Exact root metadata approval required')
    require(set(p['inputs']) == INPUTS, 'Only six pinned metadata inputs required')
    for key, item in p['inputs'].items():
        require(Path(item['file']).suffix.lower() == '.json' and sha(item['file']) == item['sha256'], 'Pinned metadata changed: ' + key)
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=False)
    write(out / 'METADATA_SELECTION_START.json', {'actual_utc': utc(), 'selector_source_sha256': sha(__file__),
        'plan_sha256': sha(a.plan), 'approval_sha256': sha(a.approval), 'inputs': p['inputs'], 'native_reads_and_models': 0})
    try:
        roles = select(p)
        roles.update({'registered_plan_sha256': sha(a.plan), 'selector_source_sha256': sha(__file__), 'inputs': p['inputs']})
        write(out / 'ROLE_SELECTION.json', roles)
        for key, item in p['inputs'].items():
            require(sha(item['file']) == item['sha256'], 'Metadata changed during selection: ' + key)
        require(sha(__file__) == p['source_sha256'], 'Selector changed during selection')
        result = {'status': 'COMPLETE_METADATA_ONLY152ROLE_SELECTION', 'actual_utc': utc(),
            'role_file_sha256': sha(out / 'ROLE_SELECTION.json'), 'selector_source_sha256': sha(__file__),
            'plan_sha256': sha(a.plan), 'group_count': roles['group_count'],
            'known_eligible_sibling_rows_reserved': sum(x['eligible_sibling_count'] for x in roles['selected_groups']),
            'raw_property_decodes_fits_queries_or_scores': 0}
        write(out / 'METADATA_SELECTION_COMPLETE.json', result); print(json.dumps(result))
    except Exception:
        write(out / 'FAILED_STOP.json', {'status': 'FAIL_METADATA_SELECTION_PRESERVED', 'actual_utc': utc(),
            'traceback': traceback.format_exc(), 'raw_property_reads_or_models': 0})
        raise


if __name__ == '__main__':
    main()

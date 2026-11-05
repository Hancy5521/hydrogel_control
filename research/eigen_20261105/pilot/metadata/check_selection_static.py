"""Static source/metadata comparison only; never import or execute the SUT."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PILOT = HERE.parent
ROLE_PLAN_SHA = 'ea9120cf04bc78c8292be8ebba3eab6f9b1b0f830574b3e1a803386105451de8'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    role_path = PILOT / 'DATASET_ROLE_PLAN.json'
    assert sha(role_path) == ROLE_PLAN_SHA
    plan = json.loads(role_path.read_text('utf-8'))
    source_path = PILOT / 'code' / 'benchmark_io.py'
    text = source_path.read_text('utf-8')
    tree = ast.parse(text)
    literals = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ('ROLES', 'ROLE_COUNTS', 'TARGETS', 'BUNDLES'):
                literals[node.targets[0].id] = ast.literal_eval(node.value)
    assert literals['ROLES'] == ('warm', 'repair', 'frontier', 'test')
    assert literals['ROLE_COUNTS'] == (8, 8, 16, 8)
    assert literals['BUNDLES'] == ((0, 1), (2,))
    for name, row in plan['datasets'].items():
        assert literals['TARGETS'][name] == row['target_fields']
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    select_text = ast.get_source_segment(text, functions['select_design'])
    assert "('hypercycle_pilot_v1|' + g).encode()" in select_text
    assert "('hypercycle_pilot_v1|row|' + dataset + '|' + str(r['raw_row_zero_based'])).encode()" in select_text
    assert 'for role, count in zip(ROLES, ROLE_COUNTS)' in select_text
    assert 'ordered[start:start + count]' in select_text
    assert ').hexdigest(), g)' in select_text
    assert ").hexdigest(), r['raw_row_zero_based']))[:2]" in select_text
    assert 'selected.extend(dict(row, role=role) for row in rows)' in select_text
    assert 'start += count' in select_text
    assert "'unused_eligible_group_ids': ordered[40:]" in select_text
    assert "'frontier_utility_sign': -1 if dataset == 'ucsb_thermoelectrics' else 1" in select_text
    feature_node = functions['feature_matrix']
    row_keys = set()
    for node in ast.walk(feature_node):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == 'row':
            assert isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str)
            row_keys.add(node.slice.value)
    assert row_keys == {'normalized_composition', 'temperature_K'}
    assert all(plan['datasets'][name]['frontier_utility']['sign'] == sign for name, sign in
               [('ucsb_thermoelectrics', -1), ('jarvis_dft_3d', 1)])
    report = {
        'actual_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'PASS_STATIC_SOURCE_INSPECTION_NOT_SUT_EXECUTION',
        'checker_script_sha256': sha(__file__),
        'role_plan_sha256': sha(role_path),
        'independent_selector_script_sha256': sha(HERE / 'make_role_plan.py'),
        'producer_source_snapshot': 'pilot/code/benchmark_io.py',
        'producer_source_sha256_at_static_read': sha(source_path),
        'producer_select_design_AST_sha256': hashlib.sha256(ast.dump(functions['select_design'], include_attributes=False).encode()).hexdigest(),
        'static_comparison': [
            'Same40 full-source component IDs are selected by SHA256prefix and component-ID tie break.',
            'Same8/8/16/8 role blocks, role ordering and at-most2 original-row hash selection.',
            'Same original-row integer tie break, copy-plus-role semantics, target fields and unused group-ID suffix.',
            'Source utility_sign -kappa/+bulk matches independent pre-fit plan; no zT objective.',
            'Feature descriptor directly reads only normalized_composition and temperature_K. Source/role/response/derived fields remain excluded.',
        ],
        'feature_scaling_scope_observed': 'Producer feature_matrix uses shared transductive centering/scaling on selected unlabeled metadata X, including test X. This is not label leakage, but does not establish inductive unseen-metadata generalization.',
        'independent_plan_role_rows': {name: row['role_counts'] for name, row in plan['datasets'].items()},
        'pinned_metadata_inputs': plan['input_sha256'],
        'role_plan_rewritten_or_reselected': False,
        'producer_imported_or_executed': False,
        'raw_response_arrays_opened': False,
        'model_fitting_or_prediction_scoring': False,
        'scope_limit': 'Static equivalence inspection and independent metadata plan only; no runtime equality claim or empirical efficacy. Later producer edits require their own final source binding.',
    }
    with (HERE / 'SELECT_DESIGN_STATIC_REVIEW.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({'status': report['status'], 'source_sha256': report['producer_source_sha256_at_static_read'],
                      'role_plan_sha256': report['role_plan_sha256'], 'review_receipt': str(HERE / 'SELECT_DESIGN_STATIC_REVIEW.json')}))


if __name__ == '__main__':
    main()

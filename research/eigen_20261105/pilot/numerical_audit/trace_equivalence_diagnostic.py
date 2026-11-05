"""Posthoc action-trace comparison, not a new model run or efficacy test."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PILOT = HERE.parent
SCORE_PLAN_SHA = 'd557053f58ae70eb527ac9202f6eb3cdf8656651dba4dff6cb959aad2b62f12d'
SCORE_PASS = HERE / 'score_execution_20261105T1114299269991Z' / 'INDEPENDENT_NUMERICAL_RESULT.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text('utf-8-sig'))


def dump(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def action_trace(actions):
    return [(row['oracle']['original_raw_row'], row['oracle']['bundle_id']) for row in actions['campaign']]


def main():
    assert sha(HERE / 'SCORE_AUDIT_PLAN.json') == SCORE_PLAN_SHA
    approval = read(HERE / 'ROOT_SCORE_AUDIT_APPROVAL.json')
    assert approval['approved'] is True and approval['phase'] == 'scores'
    assert approval['audit_plan_sha256'] == SCORE_PLAN_SHA
    passed = read(SCORE_PASS)
    assert passed['status'] == 'PASS' and passed['phase'] == 'scores'
    plan = read(HERE / 'SCORE_AUDIT_PLAN.json')
    root = Path(plan['campaign_directory'])
    assert sha(root / 'ALL_FORECASTS_FROZEN.json') == plan['forecast_freeze_sha256']
    freeze = read(root / 'ALL_FORECASTS_FROZEN.json')
    frozen = {row['file']: row['sha256'] for row in freeze['files']}
    assert sha(plan['score_result_file']) == plan['score_result_sha256']
    results = read(plan['score_result_file'])
    execution = {'actual_registered_utc': datetime.now(timezone.utc).isoformat(),
                 'scope': 'Posthoc eight paired action-identity diagnostics after numerical audit; not an independent experiment.',
                 'prior_information': 'Root reported negative primary candidate comparisons and suspected on/off trace equality before this diagnostic. No selection settings, source code, scores or policies are changed.',
                 'script_sha256': sha(__file__), 'score_plan_sha256': SCORE_PLAN_SHA,
                 'passed_score_audit_sha256': sha(SCORE_PASS),
                 'forecast_freeze_sha256': plan['forecast_freeze_sha256'],
                 'test_truth_arrays_read': False, 'SUT_or_policy_execution': False,
                 'pairs': 'Two datasets times four matched seeds: predictive hypercycle versus no-cross ablation.'}
    dump(HERE / 'TRACE_DIAGNOSTIC_PLAN.json', execution)
    pairs = []
    for dataset in ('ucsb_thermoelectrics', 'jarvis_dft_3d'):
        for seed in (2026110501, 2026110502, 2026110503, 2026110504):
            names = [dataset + '__' + policy + '__' + str(seed)
                     for policy in ('hypercycle_predictive', 'hypercycle_no_cross')]
            for name in names:
                assert sha(root / name / 'ACTIONS.json') == frozen[name + '/ACTIONS.json']
            actions = [read(root / name / 'ACTIONS.json') for name in names]
            traces = [action_trace(value) for value in actions]
            assert len(traces[0]) == len(traces[1]) == 24
            differences = [i for i, (a, b) in enumerate(zip(*traces)) if a != b]
            equal_prefix = {}
            score_difference = {}
            checkpoint_file_bytes_equal = {}
            for step in (0, 8, 16, 24):
                equal_prefix[str(step)] = traces[0][:step] == traces[1][:step]
                scored = [next(x for x in results['actual_campaigns'][name]['scores']
                               if x['checkpoint_new_bundle_cost'] == step) for name in names]
                score_difference[str(step)] = scored[0]['equal_group_standardized_CRPS'] - scored[1]['equal_group_standardized_CRPS']
                files = [root / name / ('forecast_' + str(step).zfill(2) + '.npz') for name in names]
                checkpoint_file_bytes_equal[str(step)] = sha(files[0]) == sha(files[1])
            pairs.append({'dataset': dataset, 'seed': seed, 'whole24action_trace_equal': not differences,
                          'equal_positions': 24 - len(differences), 'different_zero_based_steps': differences,
                          'action_prefix_equal_at_checkpoint': equal_prefix,
                          'frozen_checkpoint_npz_bytes_equal': checkpoint_file_bytes_equal,
                          'frozen_primary_CRPS_predictive_minus_no_cross': score_difference})
    assert sha(plan['score_result_file']) == plan['score_result_sha256']
    assert sha(root / 'ALL_FORECASTS_FROZEN.json') == plan['forecast_freeze_sha256']
    report = {'actual_end_utc': datetime.now(timezone.utc).isoformat(), 'status': 'READ_ONLY_POSTHOC_DIAGNOSTIC_COMPLETE',
              'plan_sha256': sha(HERE / 'TRACE_DIAGNOSTIC_PLAN.json'),
              'pairs': pairs, 'pair_count': len(pairs),
              'entire_action_traces_equal': sum(x['whole24action_trace_equal'] for x in pairs),
              'scientific_limit': 'Equal action sequences in this finite pilot show no distinct decisions for those seeds. They neither prove general equivalence nor learn causal catalytic transfer.',
              'original_source_or_results_modified': False, 'test_truth_arrays_read': False,
              'model_fit_or_policy_reruns': 0}
    dump(HERE / 'TRACE_EQUIVALENCE_POSTHOC.json', report)
    print(json.dumps({'pairs': len(pairs), 'identical_entire_traces': report['entire_action_traces_equal'],
                      'result_sha256': sha(HERE / 'TRACE_EQUIVALENCE_POSTHOC.json')}))


if __name__ == '__main__':
    main()

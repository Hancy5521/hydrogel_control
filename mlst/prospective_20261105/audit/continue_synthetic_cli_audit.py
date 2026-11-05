"""Source-pinned continuation after one disclosed audit-field-name error.

Only the three preregistered remaining CLI calls are allowed. Original failed
results, source bytes, past phase receipts and the 12 passing families remain.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys


def utc():
    return datetime.now(timezone.utc).isoformat()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    plan_path = Path(args.plan)
    plan = load(plan_path)
    source = Path(plan['source_directory'])
    audit = plan_path.parent
    execution = Path(plan['existing_execution_directory'])
    run = execution / 'SYNTHETIC_CLI' / 'RUN'
    for path, expected in plan['existing_file_sha256'].items():
        assert sha(path) == expected, ('Existing audit state changed', path)
    assert {str(p): sha(p) for p in run.iterdir() if p.is_file()} == plan['existing_run_file_sha256']
    assert not (run / 'ACTIVE_PHASE_LOCK.json').exists()
    assert not (run / 'CONFIRMATION_EVALUATION.json').exists()
    assert sha(__file__) == plan['continuation_script_sha256']
    hashes = {name: sha(source / name) for name in plan['source_sha256']}
    assert hashes == plan['source_sha256']
    first = load(execution / 'INDEPENDENT_SYNTHETIC_RESULTS.json')
    assert first['pass_count'] == 12 and first['fail_count'] == 1
    failures = [r for r in first['results'] if r['status'] == 'FAIL']
    assert failures[0]['name'] == 'synthetic_nine_cal_three_confirm_three_methods_CLI'
    assert failures[0]['error_type'] == 'KeyError' and failures[0]['message'] == "'scores_by_independent_unit'"
    events = []
    start = {'actual_utc': utc(), 'continuation_plan_sha256': sha(plan_path),
             'source_sha256': hashes, 'mode': 'synthetic',
             'original_failure_preserved': True, 'new_material_evidence': False}
    write(audit / 'CONTINUATION_EXECUTION_START.json', start)
    try:
        config = load(execution / 'SYNTHETIC_CLI' / 'DESIGN.json')
        cohort = load(execution / 'SYNTHETIC_CLI' / 'COHORT.json')
        calibrated = load(run / 'CALIBRATION_FROZEN.json')
        units = {r['unit_id'] for r in cohort['records'] if r['role'] == 'calibration'}
        assert len(units) == 9 and len(calibrated['thresholds']) == 3
        for mid, value in calibrated['thresholds'].items():
            assert value['n_calibration_units'] == 9 and value['order_index_one_based'] == 9
            assert set(calibrated['scores_by_declared_unit'][mid]) == units
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
        for command_spec in plan['remaining_commands']:
            command = [sys.executable, str(source / 'cli.py'), *command_spec['arguments']]
            result = subprocess.run(command, cwd=str(source), env=env, text=True,
                                    encoding='utf-8', capture_output=True, timeout=30)
            event = {'actual_utc': utc(), 'command': command, 'returncode': result.returncode,
                     'stdout': result.stdout, 'stderr': result.stderr}
            events.append(event)
            with (audit / 'CONTINUATION_COMMANDS.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + '\n')
            assert result.returncode == command_spec['expected_returncode'], event
            if command_spec['arguments'][0] == 'evaluate' and result.returncode == 0:
                first_result_hash = sha(run / 'CONFIRMATION_EVALUATION.json')
        result_path = run / 'CONFIRMATION_EVALUATION.json'
        result = load(result_path)
        assert result['data_mode'] == 'synthetic' and result['is_new_material_evidence'] is False
        assert result['automatic_material_evidence_certified'] is False
        assert result['scientific_evidence_status'] == 'SYNTHETIC_MATH_ONLY'
        assert result['new_fits'] == 0 and result['density_and_set_are_separate'] is True
        assert len(result['methods']) == 3
        for mid, value in result['methods'].items():
            assert value['aggregate']['n_declared_units'] == 3
            assert value['aggregate']['n_registered_curve_masks'] == 3
            assert value['threshold'] == calibrated['thresholds'][mid]
            assert value['aggregate']['n_units_with_unbounded_sets'] == 0
            assert len(value['per_unit']) == 3 and len(value['per_record']) == 3
            for row in value['per_record']:
                assert math.isclose(row['omitted_log_gap_from_highest_anchor'], math.log(8 / 6), abs_tol=1e-13)
                assert math.isclose(row['target_only_log_rate_span'], math.log(32 / 8), abs_tol=1e-13)
                assert row['target_weights'] == [.25, .5, .25]
        assert sha(result_path) == first_result_hash
        failure = load(sorted(run.glob('FAILURE_*.json'))[-1])
        assert failure['phase'] == 'evaluate'
        assert not any(failure[k] for k in ('truth_open_attempt_started', 'truth_bytes_loaded', 'truth_payload_parsed'))
        assert not (run / 'REVEAL_ATTEMPT_FAILED_STOP.json').exists()
        # Check original pinned artifacts, allowing only new completion files.
        for path, expected in plan['existing_file_sha256'].items():
            assert sha(path) == expected, ('Original failed result/input changed', path)
        for path, expected in plan['existing_run_file_sha256'].items():
            assert sha(path) == expected, ('Prior phase/receipt was changed', path)
        final_hashes = {name: sha(source / name) for name in plan['source_sha256']}
        assert final_hashes == hashes
        qualification = {
            'artifact_kind': 'combined_independent_synthetic_software_qualification',
            'actual_end_utc': utc(), 'status': '13_BOUNDED_TEST_FAMILIES_QUALIFIED_ACROSS_TWO_PRESERVED_RECEIPTS',
            'first_result_sha256': sha(execution / 'INDEPENDENT_SYNTHETIC_RESULTS.json'),
            'first_result_original_status': '12_PASS_1_AUDIT_HARNESS_KEYERROR_RETAINED',
            'continuation_plan_sha256': sha(plan_path),
            'continued_family': failures[0]['name'], 'continued_family_status': 'PASS',
            'first_pass_families_reused_without_rerun': 12,
            'total_test_families_qualified': 13,
            'actual_additional_CLI_commands': len(events),
            'source_sha256': final_hashes, 'source_unchanged': True,
            'exact_order_cases_from_first_receipt': 155,
            'finite_exchangeability_heldout_assignments_from_first_receipt': 2475,
            'independent_CDF_CRPS_cases_from_first_receipt': 15,
            'max_CRPS_CDF_absolute_error_from_first_receipt': 8.881784197001252e-16,
            'artificial_confirmation_result_sha256': first_result_hash,
            'future_corrected_full_replay_script_sha256': sha(audit / 'independent_synthetic_audit.py'),
            'new_material_measurements_or_fits': 0,
            'scope': 'Artificial numerical/formal-contract/CLI qualification only. No real material calibration, independent batch/formulation validation, experimental savings or submission-readiness conclusion.'}
        write(audit / 'COMBINED_SYNTHETIC_QUALIFICATION.json', qualification)
        template = load(audit / 'SYNTHETIC_AUDIT_PLAN.json')
        template.update({'artifact_kind': 'post-audit portable full synthetic replay template; not original preregistration',
                         'template_created_actual_utc': utc(),
                         'execution_approval_status': 'TEMPLATE_ONLY_REGISTER_NEW_ACTUAL_UTC_PLAN_BEFORE_FUTURE_SUT_EXECUTION',
                         'source_directory_relative_to_template': '../code',
                         'audit_script': str(audit / 'independent_synthetic_audit.py'),
                         'audit_script_sha256': sha(audit / 'independent_synthetic_audit.py'),
                         'source_sha256': final_hashes,
                         'requires_before_replay': 'Resolve portable paths; write a NEW actual UTC/source-pinned plan and unique output directory before executing full synthetic replay. Preserve this template and original failed receipts.',
                         'qualification_receipt_sha256': sha(audit / 'COMBINED_SYNTHETIC_QUALIFICATION.json')})
        template.pop('actual_registered_utc', None)
        template.pop('actual_registered_asia_shanghai', None)
        template['prior_activity'] = 'Template assembled after qualification; it is not evidence the corrected harness was the originally registered harness.'
        write(audit / 'QUALIFIED_REPLAY_TEMPLATE.json', template)
        print(json.dumps({'status': qualification['status'], 'actual_additional_CLI_commands': len(events),
                          'source_unchanged': True}, indent=2))
        return 0
    except Exception as error:
        write(audit / 'CONTINUATION_FAILED.json', {'actual_utc': utc(), 'error_type': type(error).__name__,
                                                 'message': str(error), 'actual_command_count': len(events),
                                                 'original_failures_preserved': True, 'new_material_evidence': False})
        raise


if __name__ == '__main__':
    raise SystemExit(main())

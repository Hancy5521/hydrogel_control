"""Prepare-only by default; explicit --run replays consumed development, never confirmation."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
COMMON_FREEZE = ('EXECUTION_START.json', 'VISIBLE_INPUTS.json', 'EXTRACTION_RECEIPT.json', 'FORECASTS.json')
TASKS = {
    'phase1': dict(template='admission/EXECUTION_PLAN_TEMPLATE.json', code='admission/code',
        forecast='admission_io.py', score='score_admission.py', scope='WARM_ONLY_NIW_MATERIAL_ADMISSION',
        count=2304, status='COMPLETE_ALL_FORECASTS_FROZEN', law_key='model',
        freeze_files=COMMON_FREEZE + ('MODEL.json',)),
    'phase2': dict(template='phase2_yj/EXECUTION_PLAN_TEMPLATE.json', code='phase2_yj/code',
        forecast='run_phase2.py', score='score_phase2.py', scope='PHASE2_FIXED_YJ_HALF_ADMISSION',
        count=3840, status='COMPLETE_ALL3840_FORECASTS_FROZEN', law_key='model',
        freeze_files=COMMON_FREEZE + ('FIT_REFERENCE.json',)),
    'baseline': dict(template='strong_baselines/EXECUTION_PLAN_REGISTERED.json', code='strong_baselines/code',
        forecast='run_baselines.py', score='score_baselines.py', scope='POSTHOC_FIXED_STRONG_BASELINES',
        count=1536, status='COMPLETE_ALL1536_BASELINE_LAWS_FROZEN', law_key='method',
        freeze_files=('EXECUTION_START.json', 'INPUT_SCOPE_RECEIPT.json', 'LIBRARY_VERSIONS.json',
                      'MODEL_STATES.json', 'FORECASTS.json')),
}


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text('utf-8-sig'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def inside(path):
    resolved = Path(path).resolve()
    resolved.relative_to(ROOT)
    return resolved


def validate(plan, task):
    for name, pin in plan['source_sha256'].items():
        if sha(ROOT / task['code'] / name) != pin:
            raise ValueError('Scientific source changed: ' + name)
    for name, pin in plan.get('parent_source_sha256', {}).items():
        if sha(ROOT / 'admission/code' / name) != pin:
            raise ValueError('Parent source changed: ' + name)
    # Byte hashing does not parse properties, fit a model, or calculate scores.
    for item in plan['inputs'].values():
        if sha(inside(item['file'])) != item['sha256']:
            raise ValueError('Bound local input changed')
    receipts = dict(plan.get('independent_math_receipts', {}))
    for key in ('independent_math_receipt', 'independent_score_audit_receipt'):
        if plan.get(key) is not None:
            receipts[key] = plan[key]
    for item in receipts.values():
        if sha(inside(item['file'])) != item['sha256']:
            raise ValueError('Independent math receipt changed')
    for filename, pin in plan.get('stable_sources_sha256', {}).items():
        if sha(inside(filename)) != pin:
            raise ValueError('Stable source changed')


def local_plan(phase):
    task = TASKS[phase]
    plan = read(ROOT / task['template'])
    plan['status'] = 'REGISTERED'
    plan['registered_actual_utc'] = datetime.now(timezone.utc).isoformat()
    if phase == 'baseline':
        # Replay original64/192 snapshot; no new phase2 refit is required.
        mapping = {
            'visible_inputs': 'phase2_yj/forecast_once/VISIBLE_INPUTS.json',
            'phase2_freeze': 'phase2_yj/forecast_once/FORECAST_FREEZE.json',
            'phase2_roles': 'phase2_yj/task_contract/selection_once/ADMISSION_ROLE_SELECTION.json',
            'phase2_registered_plan': 'phase2_yj/EXECUTION_PLAN_REGISTERED.json',
            'diagnostic_truth': 'phase2_yj/score_once/DIAGNOSTIC_TRUTH.json',
            'candidate_results': 'phase2_yj/score_once/RESULTS.json',
            'root_scope': 'strong_baselines/ROOT_FIXED_BASELINE_SCOPE_BEFORE_FITS.json',
        }
        plan['independent_score_audit_receipt']['file'] = str(
            ROOT / 'independent_discrete_score_math_once/INDEPENDENT_DISCRETE_GAUSSIAN_SCORE_MATH.json')
    else:
        mapping = {
            'raw_source': 'trusted_data/jarvis_dft_3d.json.gz',
            'roles': ('task_contract/selection_once/ADMISSION_ROLE_SELECTION.json' if phase == 'phase1'
                      else 'phase2_yj/task_contract/selection_once/ADMISSION_ROLE_SELECTION.json'),
            'phase1_roles': 'task_contract/selection_once/ADMISSION_ROLE_SELECTION.json',
            'contract': 'task_contract/MATERIAL_TASK_CONTRACT_V1.json',
            'root_scope': ('root_review/ROOT_ADMISSION_SCOPE_BEFORE_LABELS.json' if phase == 'phase1'
                           else 'phase2_yj/ROOT_FIXED_POWER_SCOPE_BEFORE_NEW_LABELS.json'),
        }
        niw_math = str(ROOT / 'independent_niw_math/execution_once/INDEPENDENT_NIW_T_CRPS_AUDIT.json')
        if phase == 'phase1':
            plan['independent_math_receipt']['file'] = niw_math
        else:
            plan['independent_math_receipts']['niw_t']['file'] = niw_math
            plan['independent_math_receipts']['YJ_native_probability_and_score']['file'] = str(
                ROOT / 'independent_yj_math/execution_once/INDEPENDENT_YJ_HALF_MATH_AUDIT.json')
            path_map = read(ROOT / 'RESOURCE_PATH_MAP.json')['original_to_packaged_path']
            plan['stable_sources_sha256'] = {
                str(ROOT / path_map[name]): pin for name, pin in plan['stable_sources_sha256'].items()}
    for name, item in plan['inputs'].items():
        item['file'] = str(ROOT / mapping[name])
    validate(plan, task)
    return plan


def validate_freeze(folder, protocol, task):
    frozen = read(folder / 'FORECAST_FREEZE.json')
    if frozen['status'] != task['status'] or frozen['plan_sha256'] != sha(protocol):
        raise ValueError('Complete matching replay freeze required')
    if set(frozen['files']) != set(task['freeze_files']):
        raise ValueError('Incomplete freeze')
    for name, pin in frozen['files'].items():
        if sha(folder / name) != pin:
            raise ValueError('Frozen forecast changed')
    laws = read(folder / 'FORECASTS.json')['forecasts']
    actual = {(row['component_id'], row[task['law_key']], row['mask']) for row in laws}
    if len(laws) != task['count'] or len(actual) != task['count']:
        raise ValueError('All unique laws required')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=[a + '-' + b for a in TASKS for b in ('forecast', 'score')],
                        default='phase2-forecast')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--forecast-reproduction', type=Path)
    flags = parser.add_mutually_exclusive_group()
    flags.add_argument('--prepare-only', action='store_true')
    flags.add_argument('--run', action='store_true')
    args = parser.parse_args()
    phase, operation = args.mode.split('-')
    task = TASKS[phase]
    output = args.output_dir.resolve()
    if output == ROOT or output.is_relative_to(ROOT) or output.exists():
        raise ValueError('New absent directory outside sealed package required')
    forecast = None
    plan_bytes = None
    if operation == 'forecast':
        if args.forecast_reproduction:
            raise ValueError('Forecast cannot consume prior replay')
        plan = local_plan(phase)
        allowed = 'forecast'
    else:
        if not args.forecast_reproduction:
            raise ValueError('Score needs completed same-phase forecast replay')
        prior = args.forecast_reproduction.resolve()
        origin = read(prior / 'REPRODUCTION_ORIGIN.json')
        if origin['mode'] != phase + '-forecast' or origin['prepared_only'] is not False:
            raise ValueError('Same-phase executed replay required')
        original_protocol = prior / 'REPRODUCTION_PROTOCOL.json'
        plan = read(original_protocol)
        validate(plan, task)
        forecast = prior / 'execution'
        validate_freeze(forecast, original_protocol, task)
        plan_bytes = original_protocol.read_bytes()
        allowed = 'score_after_freeze'
    output.mkdir(parents=True, exist_ok=False)
    protocol = output / 'REPRODUCTION_PROTOCOL.json'
    approval = output / 'REPRODUCTION_APPROVAL.json'
    if plan_bytes is None:
        write(protocol, plan)
    else:
        # Scoring binds forecast protocol bytes, not a fresh UTC rewrite.
        protocol.write_bytes(plan_bytes)
    now = datetime.now(timezone.utc).isoformat()
    write(approval, dict(actual_utc=now, approved=True, scope=task['scope'], allowed_phases=[allowed],
        plan_sha256=sha(protocol), kind='MANUAL_REPLAY_RECORD_NOT_ORIGINAL_RESEARCH_APPROVAL',
        independent_confirmation=False, new_wet_or_DFT_runs=0))
    command = [sys.executable, '-X', 'utf8', '-B', str(ROOT / task['code'] / task[operation]),
               '--plan', str(protocol), '--approval', str(approval)]
    if forecast is not None:
        command += ['--forecast-dir', str(forecast)]
    command += ['--output-dir', str(output / 'execution')]
    write(output / 'REPRODUCTION_ORIGIN.json', dict(actual_utc=now, mode=args.mode,
        prepared_only=not args.run, scope='Replay of consumed development; not independent confirmation',
        source_package=str(ROOT), command=command, protocol_sha256=sha(protocol),
        original_scientific_artifacts_modified=False,
        baseline_input='original frozen phase2 snapshot; no prerequisite phase2 replay',
        original_baseline_fits=6 if phase == 'baseline' else None))
    if args.run:
        result = subprocess.run(command, check=False)
        write(output / 'REPRODUCTION_EXIT.json',
              dict(actual_utc=datetime.now(timezone.utc).isoformat(), returncode=result.returncode))
        if result.returncode:
            raise SystemExit(result.returncode)
    print(json.dumps(dict(status='RUN_FINISHED' if args.run else 'PREPARED_ONLY_NO_MODEL_OR_PROPERTY_DECODE',
                          mode=args.mode, output=str(output)), ensure_ascii=True))


if __name__ == '__main__':
    main()

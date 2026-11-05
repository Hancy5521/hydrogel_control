"""Explicit local reproduction. Default is metadata-only; no material fitting.

Full mode repeats the already-consumed offline benchmark. It is not a fresh
confirmation cohort, and its new receipts never replace the original records.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--mode', choices=('metadata', 'full'), default='metadata')
    args = parser.parse_args()
    output, data = args.output_dir.resolve(), args.data_dir.resolve()
    frozen = json.loads((ROOT / 'pilot/PREFIT_SOURCE_RECEIPT.json').read_text('utf-8'))
    for record in frozen['sources']:
        if sha(ROOT / 'pilot/code' / record['file']) != record['sha256']:
            raise ValueError('Original model/controller source changed')
    for name, digest in frozen['source_input_sha256'].items():
        if sha(data / name) != digest:
            raise ValueError('Original benchmark data/metadata changed: ' + name)
    if sha(ROOT / 'pilot/DATASET_ROLE_PLAN.json') != frozen['independent_role_plan_sha256']:
        raise ValueError('Original independent role plan changed')
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'REPRODUCTION_START.json', {'actual_utc': datetime.now(timezone.utc).isoformat(),
        'mode': args.mode, 'kind': 'explicit local reproduction, not the original preregistration',
        'source_receipt_sha256': sha(ROOT / 'pilot/PREFIT_SOURCE_RECEIPT.json'),
        'not_new_material_validation': True})
    code = ROOT / 'pilot/code'
    plan = output / 'PROTOCOL.json'
    commands = []

    def execute(script, *arguments):
        command = [sys.executable, str(code / script), *map(str, arguments)]
        commands.append({'actual_utc': datetime.now(timezone.utc).isoformat(), 'command': command})
        with (output / 'COMMANDS.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(commands[-1], ensure_ascii=False) + '\n')
        subprocess.run(command, cwd=ROOT, check=True)

    try:
        execute('run_campaign.py', 'prepare', '--data-dir', data,
                '--role-plan', ROOT / 'pilot/DATASET_ROLE_PLAN.json', '--output', plan)
        if args.mode == 'metadata':
            write(output / 'REPRODUCTION_COMPLETE.json', {'status': 'METADATA_ONLY_PASS',
                'plan_sha256': sha(plan), 'material_fit_or_score_calls': 0,
                'note': 'Full mode requires an explicit --mode full invocation.'})
            return
        approval = output / 'EXPLICIT_REPLAY_FIT_RECORD.json'
        write(approval, {'actual_utc': datetime.now(timezone.utc).isoformat(), 'approved': True,
            'scope': 'BOUNDED_REAL_SOURCE_OFFLINE_PILOT', 'plan_sha256': sha(plan),
            'reviewer': 'explicit user-invoked reproduction, not original root approval',
            'not_a_new_independent_confirmation': True})
        campaigns = output / 'campaigns'
        execute('run_campaign.py', 'run', '--plan', plan, '--approval', approval, '--output', campaigns)
        score_record = output / 'EXPLICIT_REPLAY_SCORE_RECORD.json'
        write(score_record, {'actual_utc': datetime.now(timezone.utc).isoformat(), 'approved': True,
            'scope': 'SCORE_FROZEN_OFFLINE_PILOT', 'plan_sha256': sha(plan),
            'freeze_sha256': sha(campaigns / 'ALL_FORECASTS_FROZEN.json'),
            'reviewer': 'explicit user-invoked reproduction, not original root approval'})
        execute('evaluate_campaign.py', '--plan', plan, '--campaign-dir', campaigns,
                '--approval', score_record, '--output-dir', output / 'evaluation')
        write(output / 'REPRODUCTION_COMPLETE.json', {'actual_utc': datetime.now(timezone.utc).isoformat(),
            'status': 'FULL_OFFLINE_REPRODUCTION_COMPLETE_NOT_NEW_VALIDATION',
            'result_sha256': sha(output / 'evaluation/RESULTS.json'), 'new_measurements': 0})
    except Exception as error:
        write(output / 'REPRODUCTION_FAILED.json', {'actual_utc': datetime.now(timezone.utc).isoformat(),
            'error_type': type(error).__name__, 'message': str(error), 'automatic_retry': False})
        raise


if __name__ == '__main__':
    main()

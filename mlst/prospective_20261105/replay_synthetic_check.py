"""Replay the bounded artificial-number audit after relocating this companion.

This never calls a predictor, reads a material outcome, or changes the original
audit plan. A new source-pinned local plan precedes each explicitly invoked run.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    root = Path(__file__).resolve().parent
    original = root / 'audit' / 'QUALIFIED_REPLAY_TEMPLATE.json'
    plan = json.loads(original.read_text(encoding='utf-8-sig'))
    source = root / 'code'
    for name, digest in plan['source_sha256'].items():
        if sha(source / name) != digest:
            raise ValueError('Archived source changed: ' + name)
    script = root / 'audit' / 'independent_synthetic_audit.py'
    if sha(script) != plan['audit_script_sha256']:
        raise ValueError('Independent audit script changed')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    plan.update({
        'source_directory': str(source), 'audit_script': str(script),
        'actual_registered_utc': datetime.now(timezone.utc).isoformat(),
        'reviewer': 'explicit local CLI replay, not the original independent reviewer',
        'execution_approval_status': 'EXPLICITLY_INVOKED_SYNTHETIC_REPLAY_ONLY',
        'parent_qualified_replay_template_sha256': sha(original),
        'original_failed_plan_and_continuation_kept_unchanged': True,
        'actual_runtime': {
            'python_executable': sys.executable, 'python_version': sys.version,
            'numpy_package_metadata': importlib.metadata.version('numpy'),
            'scipy_package_metadata': importlib.metadata.version('scipy'),
        },
        'purpose': 'Relocated software requalification; not another material validation or new scientific result.',
    })
    local_plan = args.output_dir / 'REPLAY_PLAN.json'
    with local_plan.open('x', encoding='utf-8') as stream:
        json.dump(plan, stream, ensure_ascii=False, indent=2, allow_nan=False)
    command = [sys.executable, str(script), '--plan', str(local_plan),
               '--output', str(args.output_dir / 'execution')]
    with (args.output_dir / 'REPLAY_COMMAND.json').open('x', encoding='utf-8') as stream:
        json.dump({'actual_utc': datetime.now(timezone.utc).isoformat(),
                   'command': command, 'replay_plan_sha256': sha(local_plan),
                   'new_material_evidence': False}, stream, indent=2)
    result = subprocess.run(command, cwd=str(root), check=False)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()

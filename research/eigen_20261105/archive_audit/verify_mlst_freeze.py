"""Independent byte/clone audit of the sealed MLST record; no scientific replay."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
ARCHIVE = Path('H:/PolymerReview/MLST_frozen_20261105T093229Z')
CLONE = Path('H:/mlst_frozen_clone_20261105T093229Z')
MANIFEST_SHA = '4bced649fef0a25005610e284ad9b9bdb34de691e1027bc67b24063f45c3638a'
EXPECTED_HEAD = '05a40915be149857b2bb9cf65144d53fa9b24f37'


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def save(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def main():
    plan = {
        'actual_registered_utc': utc(), 'reviewer': '/root/top_venue_fit',
        'scope': 'Rehash all18archive payloads; clone complete bundle offline; run only read-only public manifest checker; pin archived model/paper. No fitting, forecasts, target scoring or synthetic replay.',
        'manifest_sha256': MANIFEST_SHA, 'checker_sha256': sha(__file__),
        'archive': str(ARCHIVE), 'new_clone_directory': str(CLONE),
        'expected_head': EXPECTED_HEAD, 'root_explicit_authorization': True,
        'CRC_policy': 'Use existing exact-archive delivery receipts for oldZIP CRC; do not decompress/reexecute old science.',
    }
    save(ROOT / 'MLST_FREEZE_AUDIT_PLAN.json', plan)
    result = {'actual_start_utc': utc(), 'plan_sha256': sha(ROOT / 'MLST_FREEZE_AUDIT_PLAN.json'),
              'status': 'STARTED', 'checks': {}, 'commands': [],
              'new_material_or_synthetic_executions': 0}
    try:
        assert sha(ARCHIVE / 'MANIFEST.json') == MANIFEST_SHA
        manifest = json.loads((ARCHIVE / 'MANIFEST.json').read_text('utf-8'))
        expected_files = set()
        verified = []
        for row in manifest['files']:
            name = row['path']
            parts = PurePosixPath(name)
            assert not parts.is_absolute() and '..' not in parts.parts and ':' not in name and chr(92) not in name
            path = ARCHIVE.joinpath(*parts.parts)
            assert path.is_file() and path.stat().st_size == row['bytes']
            actual = sha(path)
            assert actual == row['sha256'], name
            expected_files.add(name)
            verified.append({'path': name, 'bytes': path.stat().st_size, 'sha256': actual})
        actual_files = {p.relative_to(ARCHIVE).as_posix() for p in ARCHIVE.rglob('*') if p.is_file()}
        assert len(verified) == 18
        assert actual_files == expected_files | {'MANIFEST.json'}
        result['checks']['exact18_payloads_plus_manifest'] = verified
        assert not CLONE.exists(), 'Use a new clone directory; never overwrite an earlier run'
        commands = [
            ['git', 'clone', '--branch', 'main', str(ARCHIVE / 'MLST_all_history.bundle'), str(CLONE)],
            ['git', '-C', str(CLONE), 'rev-parse', 'HEAD'],
            [sys.executable, str(CLONE / 'verify_checkpoint.py')],
        ]
        for index, command in enumerate(commands):
            start = utc()
            completed = subprocess.run(command, cwd='D:/codex-root', capture_output=True,
                                       text=True, encoding='utf-8', timeout=90)
            row = {'actual_utc': start, 'command': command, 'returncode': completed.returncode,
                   'stdout': completed.stdout, 'stderr': completed.stderr}
            result['commands'].append(row)
            assert completed.returncode == 0, row
            if index == 1:
                assert completed.stdout.strip() == EXPECTED_HEAD
            if index == 2:
                values = json.loads(completed.stdout)
                assert values['status'] == 'PASS' and values['files'] == 712
                assert values['frozen_runtime_model_bindings'] == 16
                result['checks']['clone_public_manifest'] = values
        checkpoint = json.loads((CLONE / 'CHECKPOINT.json').read_text('utf-8'))
        protections = {
            'models/SHARED_MODEL.json': 'd72fc262fbae0e6fecc36568f0638166fe95e991bd5f7df100ab1434097d82d1',
            checkpoint['paper_main_path']: 'be4930e6f1d27a2959e6061002974d770e425f1c664c6bc01617ff5a4c1a3cfb',
            checkpoint['paper_supplement_path']: 'e04edca0bb960e7b3e47182f80131b1222a906b324a4aa79a489ca60b31095d7',
        }
        for name, digest in protections.items():
            assert sha(CLONE / name) == digest, name
        result['checks']['clone_model_main17_SI27_bytes'] = protections
        assert sha(ARCHIVE / 'MANIFEST.json') == MANIFEST_SHA
        result['status'] = 'PASS'
        result['scope_limit'] = 'Offline archive byte integrity and public-checkpoint clone only; not a new numerical/material efficacy, independent calibration, author-rights or submission-readiness certification. Remote Git status is separately checked by root.'
    except Exception as error:
        result['status'] = 'FAIL'
        result['error'] = {'type': type(error).__name__, 'message': str(error)}
    result['actual_end_utc'] = utc()
    save(ROOT / 'MLST_FREEZE_AUDIT.json', result)
    print(json.dumps({'status': result['status'], 'result_sha256': sha(ROOT / 'MLST_FREEZE_AUDIT.json'),
                      'payloads': len(result['checks'].get('exact18_payloads_plus_manifest', [])),
                      'material_or_synthetic_executions': 0}, indent=2))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())

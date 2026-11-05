"""Select the reviewed preparation increment for local and Git archives.

Does not execute scientific code, create a DOI, alter a model or publish data.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parent
LOCAL = Path('H:/PolymerReview/MLST_prospective_20261105')
PUBLIC = Path('D:/codex-root/hydrogel_control/mlst/prospective_20261105')
RAW_WEB_EVIDENCE = {
    'sources/datacite_dataset_v4.json',
    'sources/mendeley_dataset_v4_html.html',
    'sources/mendeley_dataset_v4_html_embedded_10.txt',
    'sources/mendeley_dataset_v4_html_embedded_4.txt',
    'sources/RECOVERY_QUERY_RESULT.txt',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def new_json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def entry(path, root):
    return {'path': path.relative_to(root).as_posix(), 'bytes': path.stat().st_size,
            'sha256': sha(path)}


def main():
    status = json.loads((ROOT / 'STAGE_STATUS.json').read_text('utf-8'))
    assert status['independent_synthetic_audit_status'] == 'PASS'
    assert not LOCAL.exists() and not PUBLIC.exists()
    payload = sorted(p for p in ROOT.rglob('*') if p.is_file()
                     and '__pycache__' not in p.parts and p.suffix not in ('.pyc', '.pyo'))
    assert not any(p.suffix.lower() in ('.npz', '.npy', '.zip', '.pdf', '.docx') for p in payload)
    for path in payload:
        name = path.relative_to(ROOT).as_posix()
        for destination in (LOCAL, PUBLIC):
            if destination == PUBLIC and name in RAW_WEB_EVIDENCE:
                continue
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            assert sha(path) == sha(target)
    selection = {
        'actual_utc': datetime.now(timezone.utc).isoformat(),
        'kind': 'Reviewed future-confirmation helper increment, not material data release',
        'local_source_files': len(payload),
        'public_omitted_raw_web_receipts': sorted(RAW_WEB_EVIDENCE),
        'public_selection_reason': 'Raw fetched page/query content remains in local evidence; public copy keeps source facts, URLs, hashes and access receipts rather than republication of fetched pages.',
        'artificial_audit_fixtures_are_not_material_measurements': True,
        'historical_nonsecret_execution_paths': 'Original receipts may name historical H:/ or D:/ artifact/runtime paths. These are not required current paths. Use replay_synthetic_check.py, which resolves the relocated source and writes a new plan.',
        'protected_original_manuscript_and_model_unchanged': True,
        'not_embedded': 'Original610MB full archive, original measurements, Python distribution. PRIOR_ARCHIVES.json records exact old delivery pins.',
    }
    for destination in (LOCAL, PUBLIC):
        new_json(destination / 'PUBLIC_SELECTION.json', selection)
    manifest = {
        'actual_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'All local increment files except this manifest; no model or material original archive embedded.',
        'files': [entry(p, LOCAL) for p in sorted(LOCAL.rglob('*')) if p.is_file()],
    }
    new_json(LOCAL / 'PACKAGE_MANIFEST.json', manifest)
    # Public repository's own MANIFEST.json will bind its reviewed subset.
    archive = LOCAL.with_suffix('.zip')
    assert not archive.exists()
    with ZipFile(archive, 'x', compression=ZIP_DEFLATED, compresslevel=6) as zipped:
        for path in sorted(LOCAL.rglob('*')):
            if path.is_file():
                zipped.write(path, LOCAL.name + '/' + path.relative_to(LOCAL).as_posix())
    receipt = {
        'actual_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'PREPARED_AWAITING_INDEPENDENT_DELIVERY_AUDIT',
        'directory': str(LOCAL), 'zip': str(archive), 'zip_bytes': archive.stat().st_size,
        'zip_sha256': sha(archive), 'payload_files': len(manifest['files']),
        'manifest_sha256': sha(LOCAL / 'PACKAGE_MANIFEST.json'),
        'public_increment_directory': str(PUBLIC), 'public_increment_files': len([p for p in PUBLIC.rglob('*') if p.is_file()]),
        'source_stage_directory': str(ROOT), 'builder_sha256': sha(Path(__file__)),
        'new_material_evidence': False, 'old_full_package_replaced_or_relabelled': False,
    }
    new_json(LOCAL.with_name(LOCAL.name + '_PREPARED.json'), receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

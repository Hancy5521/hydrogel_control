"""Exact local checkpoint and reviewed public subset; never run a model."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parent
LOCAL = Path('H:/PolymerReview/Eigen_repair_expansion_20261105')
PUBLIC = Path('H:/hypercycle_git_20261105T093229Z/research/eigen_20261105')
CORE_METADATA = {'ELIGIBLE_METADATA.json', 'PILOT_ELIGIBILITY.json', 'DATA_MANIFEST.json',
                 'SOURCE_GATE.json', 'SOURCE_GATE_PILOT_ADDENDUM.json', 'SOURCE_FACTS_ZH.txt'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def put_json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def local_admitted(path):
    rel = path.relative_to(ROOT)
    return ('__pycache__' not in rel.parts and '.git' not in rel.parts
            and 'local_reading_only' not in rel.parts and path.suffix.lower() not in ('.pdf', '.pyc', '.pyo'))


def public_admitted(path):
    rel = path.relative_to(ROOT)
    name = rel.as_posix()
    if rel.parts[0] in ('logs_snapshot', 'history') or path.suffix.lower() in ('.gz', '.npz'):
        return False
    if rel.parts[0] == 'data':
        return len(rel.parts) == 2 and rel.name in CORE_METADATA
    if name.startswith('pilot/campaigns/'):
        return name == 'pilot/campaigns/ALL_FORECASTS_FROZEN.json'
    if name.startswith('pilot/audit/execution_') and path.suffix == '.json':
        # Keep the original independent audit result/start, not its artificial
        # 281-file corruption fixtures; the audit source reconstructs them.
        return rel.name in ('INDEPENDENT_SYNTHETIC_AUDIT.json', 'EXECUTION_START.json') and len(rel.parts) == 4
    return True


def main():
    assert not LOCAL.exists() and not PUBLIC.exists()
    assert sha(ROOT / 'pilot/evaluation/RESULTS.json') == '091c0819de30a7adf7945560105bb734f41e3b1a86fc5af34be764294e00ec83'
    assert sha(ROOT / 'pilot/numerical_audit/INDEPENDENT_AUDIT_SUMMARY.json') == 'f7cb8985d917bc5413413c8782711b62edeb7996ebf2388ea3547509f801560e'
    assert sha(ROOT / 'exact_control/audit/FINAL_AUDIT_RECEIPT.json') == 'defba9f017020738e316225def3a81d2a05db6307f07437a1f966b467edb772f'
    source = sorted(p for p in ROOT.rglob('*') if p.is_file() and local_admitted(p))
    public_files, omitted = [], []
    for path in source:
        name = path.relative_to(ROOT)
        target = LOCAL / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        if sha(path) != sha(target):
            raise ValueError('Local copy differs: ' + name.as_posix())
        if public_admitted(path):
            target = PUBLIC / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            assert sha(path) == sha(target)
            public_files.append(name.as_posix())
        else:
            omitted.append(name.as_posix())
    selection = {
        'actual_utc': datetime.now(timezone.utc).isoformat(),
        'kind': 'Complete current new-direction local checkpoint plus a reviewed public subset',
        'source_files_in_local_package': len(source), 'source_files_in_public_subset': len(public_files),
        'public_files': public_files, 'omitted_from_public': omitted,
        'not_in_local_package': 'Downloaded full papers/local_reading_only, Python runtime, old620MBMLST archive. URLs/read scope, requirements and exact old archive receipts remain included.',
        'public_reproduction_scope': 'Source/metadata/derived scores and audit evidence. Original public gzip data can be retrieved with fetch_pinned_data.py; full campaign ACTIONS/NPZ are local-only. Explicit full-mode reproduction rebuilds them, not an independent experiment.',
        'original_scientific_core_model_results_modified': False,
        'new_wet_or_DFT_measurements': 0,
        'top_conference_or_novel_algorithm_certified': False,
    }
    put_json(LOCAL / 'PACKAGING_SELECTION.json', selection)
    put_json(PUBLIC / 'PUBLIC_SELECTION.json', selection)
    files = []
    for p in sorted(LOCAL.rglob('*')):
        if p.is_file():
            files.append({'path': p.relative_to(LOCAL).as_posix(), 'bytes': p.stat().st_size, 'sha256': sha(p)})
    put_json(LOCAL / 'PACKAGE_MANIFEST.json', {'actual_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'All local checkpoint payloads except this manifest', 'files': files})
    archive = LOCAL.with_suffix('.zip')
    with ZipFile(archive, 'x', ZIP_DEFLATED, compresslevel=6) as zipped:
        for p in sorted(LOCAL.rglob('*')):
            if p.is_file():
                zipped.write(p, LOCAL.name + '/' + p.relative_to(LOCAL).as_posix())
    receipt = {'actual_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'PREPARED_AWAITING_INDEPENDENT_DELIVERY',
        'directory': str(LOCAL), 'entry': str(LOCAL / 'INDEX.html'), 'zip': str(archive),
        'zip_bytes': archive.stat().st_size, 'zip_sha256': sha(archive),
        'payload_files': len(files), 'zip_members': len(files) + 1,
        'manifest_sha256': sha(LOCAL / 'PACKAGE_MANIFEST.json'),
        'exact_topfolder': LOCAL.name + '/', 'public_subset': str(PUBLIC),
        'public_source_files_plus_selection': len(public_files) + 1,
        'builder_sha256': sha(Path(__file__)), 'new_measurements': 0}
    put_json(LOCAL.with_name(LOCAL.name + '_PREPARED.json'), receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

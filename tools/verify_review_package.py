"""Read-only SHA256 verification of a local hydrogel review resource package.

Python 3.10+; no model execution, raw-number parsing, downloads or installation.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


def digest(path):
    # Streaming also works on Python3.10, which has no hashlib.file_digest.
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def member(root, name):
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or chr(92) in name or ':' in name:
        raise ValueError('Unsafe resource member')
    resolved = root.joinpath(*p.parts).resolve()
    resolved.relative_to(root)
    return resolved


def verify(root, require_history=False):
    root = Path(root).resolve()
    manifest = json.loads((root / 'MANIFEST.json').read_text(encoding='utf-8-sig'))
    count = 0
    for entry in manifest['files']:
        path = member(root, entry['path'])
        if path.stat().st_size != entry['bytes'] or digest(path) != entry['sha256']:
            raise ValueError('Changed resource: ' + entry['path'])
        count += 1
    history = json.loads((root / 'COMPLETE_HISTORY_INDEX.json').read_text(encoding='utf-8-sig'))['archives']
    present = 0
    for entry in history:
        path = member(root, entry['path'])
        if not path.exists():
            if require_history:
                raise FileNotFoundError('Required original archive: ' + entry['path'])
            continue
        if path.stat().st_size != entry['bytes'] or digest(path) != entry['sha256']:
            raise ValueError('Changed historical archive: ' + entry['path'])
        present += 1
    return {'status': 'PASS', 'current_payload_files': count,
            'historical_archives_present': present, 'historical_archives_declared': len(history),
            'all_history_required': require_history, 'material_model_execution': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--require-history', action='store_true')
    args = parser.parse_args()
    print(json.dumps(verify(args.package_root, args.require_history)))


if __name__ == '__main__':
    main()

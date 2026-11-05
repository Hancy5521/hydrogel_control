"""Verify sealed resource bytes using the standard library; execute no model."""
from pathlib import Path, PurePosixPath
import hashlib
import json

ROOT=Path(__file__).resolve().parent


def main():
    manifest=json.loads((ROOT/'PACKAGE_MANIFEST.json').read_text('utf-8'))
    entries=manifest['files']
    names=set()
    for item in entries:
        relative=PurePosixPath(item['path'])
        if relative.is_absolute() or '..' in relative.parts or ':' in str(relative) or '\\' in str(relative):
            raise ValueError('Unsafe relative path')
        if item['path'] in names:
            raise ValueError('Duplicate manifest entry')
        names.add(item['path'])
        p=ROOT.joinpath(*relative.parts).resolve();p.relative_to(ROOT)
        raw=p.read_bytes()
        if len(raw)!=item['bytes'] or hashlib.sha256(raw).hexdigest()!=item['sha256']:
            raise ValueError('Changed resource: '+item['path'])
    print(json.dumps({'status':'PASS','verified_payload_files':len(entries),
                      'model_or_experiment_runs':0,'new_material_measurements':0}))


if __name__=='__main__':
    main()

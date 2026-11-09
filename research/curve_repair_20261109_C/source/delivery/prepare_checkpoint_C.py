"""Prepare a new, unsealed local resource folder. Never mutate A/B/MLST.

The complete frozen MLST release is retained separately from current research,
so the old paper/results cannot be mistaken for new generalization evidence.
No upload, fit, scoring, recursive deletion or overwrite of an existing output.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import zipfile


B_ZIP=Path('H:/PolymerReview/Curve_Repair_20261109_B.zip')
B_SHA='7db538b014ead374c5e7b640bc32db954642ae394172cb2de4ff77b5ebec7031'
LEGACY=Path('H:/PolymerReview/MLST_controls_20261104T154143Z_r2')


def sha(raw):return hashlib.sha256(raw).hexdigest()


def write(path,data):
    with path.open('x',encoding='utf-8') as f:json.dump(data,f,ensure_ascii=False,indent=2)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve()
    if not out.is_relative_to(Path('H:/PolymerReview').resolve()):
        raise ValueError('Explicit local delivery directory must remain within H:/PolymerReview')
    raw=B_ZIP.read_bytes()
    if sha(raw)!=B_SHA:raise ValueError('Immutable B ZIP bytes changed')
    out.mkdir(parents=True,exist_ok=False)
    write(out/'BUILD_STATE.json',{'actual_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
         'status':'BUILDING_NOT_SEALED_NOT_DELIVERED','old_B_zip_sha256':B_SHA,
         'new_science_ready':False,'fitting_or_upload':False})
    stage=out/'stage';stage.mkdir()
    import io
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        if z.testzip() is not None:raise ValueError('B CRC failure')
        for item in z.infolist():
            pure=PurePosixPath(item.filename)
            if pure.is_absolute() or '..' in pure.parts or '\\' in item.filename:
                raise ValueError('Unsafe original member path')
            target=(stage/Path(*pure.parts)).resolve()
            if not target.is_relative_to(stage.resolve()):raise ValueError('ZIP member escaped output')
            target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as f:f.write(z.read(item))
    # The immutable release manifest enumerates exact payload bytes, avoiding
    # a broad recursive inventory of unrelated drives/workspaces.
    manifest=json.loads((LEGACY/'MANIFEST.json').read_text('utf-8'))
    legacy_out=out/'legacy_mlst';legacy_out.mkdir()
    count=0;total=0
    for item in manifest['files']:
        pure=PurePosixPath(item['path'])
        if pure.is_absolute() or '..' in pure.parts:raise ValueError('Unsafe legacy manifest path')
        source=LEGACY/Path(*pure.parts);data=source.read_bytes()
        if len(data)!=item['bytes'] or sha(data)!=item['sha256']:
            raise ValueError('Legacy immutable payload changed: '+item['path'])
        target=legacy_out/Path(*pure.parts)
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as f:f.write(data)
        count+=1;total+=len(data)
    shutil.copyfile(LEGACY/'MANIFEST.json',legacy_out/'MANIFEST.json')
    write(out/'PREPARATION_RECEIPT.json',{'actual_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
         'state':'UNSEALED_BASE_RESOURCE_PREPARED','B_members':281,'legacy_payloads':count,
         'legacy_payload_bytes':total,'B_original_unchanged':True,'MLST_original_unchanged':True,
         'scope':'Old complete release and B resources; current C scientific additions not yet copied or sealed'})
    print(json.dumps({'output':str(out),'status':'UNSEALED_PREPARED','legacy_files':count,'legacy_bytes':total}))


if __name__=='__main__':main()

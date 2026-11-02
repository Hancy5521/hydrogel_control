"""Prepare or run the fixed external-data reconstruction as a reproducibility replay."""
import argparse,hashlib,json,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path, PurePosixPath
import re

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,value):
    with p.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)

ORIGIN_COMMIT = '4e689d3815695e1317496b465ef8eb898c634044'
ORIGIN_PROTOCOL_SHA256 = '348d507321a41b60edf97f1da757f62f0a68c0d926b89696bfa4eeba0e50f4bd'
EXPECTED_INVENTORY_SHA256 = 'd10a310f22ca4df551d67b486af463ac6b569a985848fcbafc0c46671befd29c'

def _unique_inventory_object(pairs):
    result={}
    for key,value in pairs:
        if key in result:raise ValueError('Duplicate inventory JSON field: '+key)
        result[key]=value
    return result

def validate_frozen_components(repo,study):
    """Require the complete original20 components before rebuilding any data.

    Canonical paths come from a separately pinned inventory. Existing-file
    globs are used only to reject unexpected Python files, never to infer
    which original components are required. This function executes no source,
    model, data reconstruction, forecast, score, optimizer or output write.
    """
    roots={'repo':Path(repo).resolve(),'study':Path(study).resolve()}
    if not all(root.is_dir() for root in roots.values()):
        raise ValueError('Repository and study roots must exist')
    inventory_path=roots['study']/'reproduction/FROZEN_REPLAY_BINDINGS.json'
    raw=inventory_path.read_bytes()
    inventory=json.loads(raw.decode('utf-8'),object_pairs_hook=_unique_inventory_object)
    if not isinstance(inventory,dict) or type(inventory.get('schema_version')) is not int or inventory['schema_version']!=1:
        raise ValueError('Invalid replay inventory schema')
    if inventory.get('origin_commit')!=ORIGIN_COMMIT or inventory.get('origin_execution_protocol_sha256')!=ORIGIN_PROTOCOL_SHA256:
        raise ValueError('Replay inventory origin differs from original execution')
    components=inventory.get('components')
    if not isinstance(components,list) or len(components)!=20:
        raise ValueError('Complete20-component replay inventory required')
    seen=set();counts={'repo':0,'study':0}
    for component in components:
        if not isinstance(component,dict) or set(component)!={'origin','path','sha256'}:
            raise ValueError('Invalid component inventory fields')
        origin=component['origin'];relative=component['path'];digest=component['sha256']
        if not isinstance(origin,str) or origin not in roots:
            raise ValueError('Invalid inventory component origin')
        if not isinstance(relative,str) or not relative or chr(92) in relative or ':' in relative:
            raise ValueError('Unsafe inventory component path')
        path=PurePosixPath(relative)
        if path.is_absolute() or '..' in path.parts or str(path)!=relative:
            raise ValueError('Unsafe inventory component path')
        identity=(origin,relative)
        if identity in seen:raise ValueError('Duplicate inventory component path')
        seen.add(identity);counts[origin]+=1
        if not isinstance(digest,str) or re.fullmatch('[0-9a-f]{64}',digest) is None:
            raise ValueError('Invalid component SHA256')
    if counts!={'repo':16,'study':4}:
        raise ValueError('Original16repo/4study component inventory required')
    inventory_sha=hashlib.sha256(raw).hexdigest()
    if inventory_sha!=EXPECTED_INVENTORY_SHA256:
        raise ValueError('Replay inventory bytes differ from the pinned original20 manifest')
    for component in components:
        root=roots[component['origin']]
        path=root.joinpath(*PurePosixPath(component['path']).parts)
        try:path.resolve().relative_to(root)
        except ValueError as exc:raise ValueError('Component escapes declared root') from exc
        if not path.is_file():raise ValueError('Missing frozen component: '+component['path'])
        if sha(path)!=component['sha256']:
            raise ValueError('Changed frozen component: '+component['path'])
    # Compare full relative paths, including directory layout and origin.
    expected_python={(c['origin'],c['path']) for c in components if c['path'].endswith('.py')}
    actual_python={('repo',p.relative_to(roots['repo']).as_posix())
        for p in (roots['repo']/'runtime').rglob('*.py')}
    actual_python|={('study',p.relative_to(roots['study']).as_posix())
        for p in (roots['study']/'program').rglob('*.py')}
    if actual_python!=expected_python:
        raise ValueError('Unexpected or misplaced runtime/program Python components')
    return {'status':'PASS','inventory_sha256':inventory_sha,'validated_components':20,
        'repo_components':16,'study_components':4,'origin_commit':ORIGIN_COMMIT,
        'original_execution_protocol_sha256':ORIGIN_PROTOCOL_SHA256}


RECONSTRUCTION_BINDINGS = {
    'reproduction/SOURCE_CELL_MAP.json': '8a7f766edafaaf813ba7dc58fb65daa760df42729b347dcc70793c8f1801ca81',
    'reproduction/rebuild_inputs.py': '1a90fb052804f4d5329dcb8692f0321cc992395a1a5c3072a4342ecc5a4df4ba',
}

def validate_reconstruction_assets(study):
    """Reject version drift of the exact original map and reader before use.

    These two reader assets supplement, rather than replace or alter, the
    original20 scientific runtime/model/program bindings. No source workbook,
    numeric response, reconstruction, model or output is touched here.
    """
    root=Path(study).resolve()
    for relative,digest in RECONSTRUCTION_BINDINGS.items():
        asset=root.joinpath(*PurePosixPath(relative).parts)
        try:asset.resolve().relative_to(root)
        except ValueError as exc:raise ValueError('Reconstruction asset escapes study root') from exc
        if not asset.is_file():raise ValueError('Missing fixed reconstruction asset: '+relative)
        if sha(asset)!=digest:raise ValueError('Changed fixed reconstruction asset: '+relative)
    return {'status':'PASS','validated_reconstruction_assets':2,
        'origin_commit':ORIGIN_COMMIT,'bindings':dict(RECONSTRUCTION_BINDINGS)}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,required=True);p.add_argument('--raw-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args();repo=a.repo.resolve();folder=Path(__file__).resolve().parent;study=folder.parent
    frozen_validation=validate_frozen_components(repo,study)
    reconstruction_validation=validate_reconstruction_assets(study)
    map_path=folder/'SOURCE_CELL_MAP.json';mapping=json.loads(map_path.read_text(encoding='utf-8'),object_pairs_hook=_unique_inventory_object)
    from rebuild_inputs import rebuild
    curves,verified=rebuild(mapping,a.raw_dir)
    if a.output_dir.exists():raise FileExistsError('New empty output path required')
    a.output_dir.mkdir(parents=True)
    admitted=a.output_dir/'ADMITTED_CURVES.json';save(admitted,{'curves':curves,'purpose':'Exact source-vector reconstruction; not a new confirmation'})
    model=repo/'models/SHARED_MODEL.json';control=repo/'models/BASELINES_MODEL.json'
    inventory_path=folder/'FROZEN_REPLAY_BINDINGS.json'
    files=[map_path,admitted,model,control,Path(__file__).resolve(),folder/'rebuild_inputs.py',inventory_path]+list((study/'program').glob('*.py'))+list((repo/'runtime').rglob('*.py'))
    protocol={'actual_utc':datetime.now(timezone.utc).isoformat(),'purpose':'Reproducibility replay of the fixed 46-curve source-inspected assessment; not new independent evidence',
        'repo':str(repo),'admitted_curves':str(admitted.resolve()),'model':str(model),'baselines':str(control),
        'prediction_dir':str((a.output_dir/'frozen').resolve()),'bindings':{str(f.resolve()):sha(f) for f in files},
        'source_inspected_not_blind':True,'frozen_model_retraining':False,'source_files_verified':verified,
        'frozen_replay_inventory_sha256':frozen_validation['inventory_sha256'],
        'frozen_component_validation':frozen_validation,
        'fixed_reconstruction_asset_validation':reconstruction_validation,
        'original_execution_protocol_sha256':mapping['frozen_bindings']['frozen_protocol_sha256']}
    path=a.output_dir/'PROTOCOL.json';save(path,protocol)
    if not a.prepare_only:
        runner=study/'program/run_external.py'
        for phase,dest in [('freeze','frozen'),('score','scored')]:
            subprocess.run([sys.executable,str(runner),phase,'--protocol',str(path),'--output',str(a.output_dir/dest)],check=True)
    print(json.dumps({'status':'PREPARED_ONLY' if a.prepare_only else 'REPLAY_COMPLETED','curves':len(curves),'source_vectors_hash_verified':len(curves),'not_new_independent_validation':True}))
if __name__=='__main__':main()

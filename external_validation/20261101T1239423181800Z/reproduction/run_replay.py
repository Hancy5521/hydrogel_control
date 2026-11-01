"""Prepare or run the fixed external-data reconstruction as a reproducibility replay."""
import argparse,hashlib,json,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
from rebuild_inputs import rebuild

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,value):
    with p.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,required=True);p.add_argument('--raw-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args();repo=a.repo.resolve();folder=Path(__file__).resolve().parent;study=folder.parent
    map_path=folder/'SOURCE_CELL_MAP.json';mapping=json.loads(map_path.read_text())
    original=json.loads((study/'EXECUTION_PROTOCOL_PUBLIC.json').read_text(encoding='utf-8'))
    expected={}
    for item in original['original_execution_binding_hashes']:
        expected.setdefault(item['binding_role'],set()).add(item['sha256'])
    frozen_files=[repo/'models/SHARED_MODEL.json',repo/'models/BASELINES_MODEL.json']+list((repo/'runtime').rglob('*.py'))+list((study/'program').glob('*.py'))
    for file in frozen_files:
        if expected.get(file.name)!={sha(file)}:
            raise ValueError('Not the original frozen runtime/model/program bytes: '+file.name)
    curves,verified=rebuild(mapping,a.raw_dir)
    if a.output_dir.exists():raise FileExistsError('New empty output path required')
    a.output_dir.mkdir(parents=True)
    admitted=a.output_dir/'ADMITTED_CURVES.json';save(admitted,{'curves':curves,'purpose':'Exact source-vector reconstruction; not a new confirmation'})
    model=repo/'models/SHARED_MODEL.json';control=repo/'models/BASELINES_MODEL.json'
    files=[map_path,admitted,model,control,Path(__file__).resolve(),folder/'rebuild_inputs.py']+list((study/'program').glob('*.py'))+list((repo/'runtime').rglob('*.py'))
    protocol={'actual_utc':datetime.now(timezone.utc).isoformat(),'purpose':'Reproducibility replay of the fixed 46-curve source-inspected assessment; not new independent evidence',
        'repo':str(repo),'admitted_curves':str(admitted.resolve()),'model':str(model),'baselines':str(control),
        'prediction_dir':str((a.output_dir/'frozen').resolve()),'bindings':{str(f.resolve()):sha(f) for f in files},
        'source_inspected_not_blind':True,'frozen_model_retraining':False,'source_files_verified':verified,
        'original_execution_protocol_sha256':mapping['frozen_bindings']['frozen_protocol_sha256']}
    path=a.output_dir/'PROTOCOL.json';save(path,protocol)
    if not a.prepare_only:
        runner=study/'program/run_external.py'
        for phase,dest in [('freeze','frozen'),('score','scored')]:
            subprocess.run([sys.executable,str(runner),phase,'--protocol',str(path),'--output',str(a.output_dir/dest)],check=True)
    print(json.dumps({'status':'PREPARED_ONLY' if a.prepare_only else 'REPLAY_COMPLETED','curves':len(curves),'source_vectors_hash_verified':len(curves),'not_new_independent_validation':True}))
if __name__=='__main__':main()

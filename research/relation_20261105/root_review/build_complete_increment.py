"""Seal a complete current increment; omit downloaded papers and runtime caches."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import shutil
from zipfile import ZipFile,ZIP_DEFLATED

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT.parent/'hypercycle_agents_20261105T093229687251Z'
PROJECT=Path('D:/codex-root/polymer_completion')
DEST=Path('H:/PolymerReview/Relation_Moment_20261105T124422Z')


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p,obj):
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    assert not DEST.exists() and not DEST.with_suffix('.zip').exists(), 'Never overwrite an archive'
    native_pins={
       'math_design/relation_core.py':'37d35e51701961beffb57da89c03cbe5305f39a7087375d78397f2e6183dfa91',
       'math_design/run_synthetic.py':'024cb2755cc9cc7a37c1b1242216bb1db1e2e56a1d74fa9f9f6173cc44a2f60f',
       'math_design/moment_followup/moment_core.py':'fa2e63d672a23cc490786ca21f03161122e7670d9c9d8457f6a123a710025de8',
       'math_design/moment_followup/run_moment_control.py':'d55f39d9cff4931cb3bd63a7669b804934764a755fc23e5c064bd1a189823a62',
       'math_design/moment_followup/posthoc_scaling/run_scaling_extension.py':'3762ce367677ab5b1443eba090b79e850ca6c049607f57ed032c69f93da7dfd7'}
    for name,pin in native_pins.items():assert sha(ROOT/name)==pin,name
    for name,pin in {
      'math_design/execution_once/RESULTS.json':'77204bcc7da621852782900f8ef985c11d222337773051c513126220ec102ebe',
      'math_design/moment_followup/execution_once/RESULTS.json':'cc48fe6581545569760d544da6d5a1342ce36a2ab0855a1fb3fad80787c37893',
      'math_design/moment_followup/posthoc_scaling/execution_once/RESULTS.json':'1ece313e34b5b4586fa7bedf0d1fbea079f14b17fcac0057395f4cbb18fdd927'}.items():assert sha(ROOT/name)==pin,name
    meta=json.loads((ROOT/'data_feasibility/UNUSED_IDENTITY_FEASIBILITY_FINAL.json').read_text('utf-8'))
    ref=ROOT/'reference_inputs';ref.mkdir(exist_ok=False)
    bindings=[]
    for old_rel,pin in meta['input_sha256'].items():
        source=OLD/old_rel;assert sha(source)==pin
        target=ref/source.name;assert not target.exists();shutil.copy2(source,target)
        bindings.append({'original_relative_path':old_rel,'packaged_path':target.relative_to(ROOT).as_posix(),
                         'sha256':pin,'source_stage':str(OLD),'native_property_values':False})
    write(ref/'INPUT_BINDINGS.json',{'actual_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'Exact public-source metadata and previous consumed-role input read this round; no property gzip', 'files':bindings})
    shutil.copy2(OLD/'DATA_ATTRIBUTION.txt',ref/'LEGACY_DATA_ATTRIBUTION.txt')
    write(ROOT/'ARCHIVE_REFERENCES.json',{'current_increment':'relation complementarity and moment decision readout',
        'previous_complete_new_direction_zip':{'path':'H:/PolymerReview/Eigen_repair_expansion_20261105.zip',
             'bytes':47377915,'sha256':'d8caed0f804a26535c2d4b3d6c42a99db80d522dd91e25afc6612455f6f6056b','embedded':False},
        'MLST_separate_archive':{'path':'H:/PolymerReview/MLST_frozen_20261105T093229Z','embedded':False,
             'remote_archive_commit':'05a40915be149857b2bb9cf65144d53fa9b24f37'},
        'raw_source_gzip_inputs_used_in_current_numerical_experiments':False,
        'mathematical_source_and_current_result_pins':native_pins,
        'papers':'Downloaded local-reading-only primary papers excluded; original URLs/version/read depths retained',
        'private_measurements_or_authentication_secrets':False})
    logs=ROOT/'logs_snapshot';logs.mkdir(exist_ok=False)
    run=PROJECT/'logs/research/20261105T124422210519Z'
    for name in ('RUN.json','EVENTS.jsonl'):shutil.copy2(run/name,logs/name)
    write(logs/'SNAPSHOT_SCOPE.json',{'actual_utc':datetime.now(timezone.utc).isoformat(),
        'meaning':'Actual current run snapshot before immutable packaging; final closure stays in project log and external delivery receipt',
        'not_a_retroactive_run_or_goal_completion':True})

    omitted=[];copied=[]
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file():continue
        rel=p.relative_to(ROOT)
        if any(part in ('__pycache__','mpl_cache','local_reading_only') for part in rel.parts) or p.suffix.lower() in ('.pyc','.pyo','.pdf'):
            omitted.append(rel.as_posix());continue
        dest=DEST/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
        copied.append(rel.as_posix())
    write(DEST/'RESOURCE_SCOPE.json',{'actual_utc':datetime.now(timezone.utc).isoformat(),
        'kind':'Complete current new-stage increment, not a top-conference-ready manuscript',
        'numerical_data_scope':'All current synthetic source/inputs/protocols/results/audits including each16-case scaling receipt; previous real-source metadata included, original numeric gzip/oldMLST remain external',
        'downloaded_papers_and_runtime_caches_omitted':omitted,
        'new_material_fits_predictions_target_scores':0,
        'old_frozen_source_bytes_preserved':True})
    entries=[]
    for p in sorted(DEST.rglob('*')):
        if p.is_file():entries.append({'path':p.relative_to(DEST).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)})
    write(DEST/'PACKAGE_MANIFEST.json',{'actual_utc':datetime.now(timezone.utc).isoformat(),
        'checkpoint':'relation-moment-20261105T124422Z','self_excluded':True,'files':entries})
    zip_path=DEST.with_suffix('.zip')
    with ZipFile(zip_path,'x',compression=ZIP_DEFLATED,compresslevel=9) as z:
        for p in sorted(DEST.rglob('*')):
            if p.is_file():z.write(p,DEST.name+'/'+p.relative_to(DEST).as_posix())
    receipt={'actual_utc':datetime.now(timezone.utc).isoformat(),'status':'PREPARED_AWAITING_INDEPENDENT_DELIVERY',
        'folder':str(DEST),'entry':str(DEST/'INDEX.html'),'zip':str(zip_path),'zip_bytes':zip_path.stat().st_size,
        'zip_sha256':sha(zip_path),'payload_files':len(entries),'zip_members':len(entries)+1,
        'manifest_sha256':sha(DEST/'PACKAGE_MANIFEST.json'),'new_material_execution':False,
        'original_scientific_source_pins':native_pins}
    target=DEST.parent/(DEST.name+'_PREPARED.json');write(target,receipt)
    print(json.dumps(receipt))


if __name__=='__main__':main()

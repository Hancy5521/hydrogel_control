"""Build only after root's final science/audit packaging gate; never runs a model."""
from pathlib import Path
from datetime import datetime,timezone
from zipfile import ZipFile,ZIP_DEFLATED
import argparse,hashlib,html,json,shutil

STAGE=Path(__file__).resolve().parent.parent
PACKAGING=Path(__file__).resolve().parent
DEFAULT=Path(r'H:\PolymerReview\Material_Feedback_20261105T145420Z')
RAW_SHA='3d0fc1315b74cf84071f16aa25bb93434dafbc83867beb75872064ac245bd747'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text('utf-8'))
def write(p,x):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def copy(source,target,pin=None):
    source,target=Path(source),Path(target)
    if pin is not None and sha(source)!=pin:raise ValueError('Pinned source changed: '+str(source))
    target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    if sha(source)!=sha(target):raise ValueError('Copy mismatch')
def omitted(path):
    parts=path.relative_to(STAGE).parts
    return any(p in ('__pycache__','local_reading_only','mpl_cache','.git') for p in parts) or path.suffix.lower() in ('.pyc','.pyo','.pyd','.dll','.exe','.so','.whl','.pdf','.docx','.zip','.bundle') or path.name.startswith('~$')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--release-gate',type=Path,required=True);p.add_argument('--target',type=Path,default=DEFAULT);args=p.parse_args()
    gate=read(args.release_gate)
    if gate.get('status')!='READY_FOR_LOCAL_SCIENCE_PACKAGE' or gate.get('science_and_independent_audits_frozen') is not True:raise ValueError('Root final science/audit packaging gate required; do not seal while audits are pending')
    required=gate['required_stage_file_sha256']
    for rel,pin in required.items():
        q=(STAGE/rel).resolve();q.relative_to(STAGE)
        if sha(q)!=pin:raise ValueError('Final science binding changed: '+rel)
    for key in ('forecast_freeze','score_results','independent_saved_audit','report'):
        rel=gate['required_artifact_roles'][key]
        if rel not in required:raise ValueError('Final artifact role is not pinned: '+key)
    target=args.target.resolve();archive=target.with_suffix('.zip')
    if target.exists() or archive.exists():raise ValueError('New immutable destination required; preserve old archives')
    contract=read(STAGE/'task_contract/MATERIAL_TASK_CONTRACT_V1.json');selection_plan=read(STAGE/'task_contract/SELECTION_PLAN_REGISTERED.json')
    raw=Path(selection_plan['raw_asset_reference_not_read']['file'])
    if raw.stat().st_size!=30669285 or sha(raw)!=RAW_SHA:raise ValueError('Original public native source changed')
    target.mkdir(parents=True)
    copied=[];excluded=[]
    # This snapshots only after the final gate. Failed/old-source artifacts are preserved.
    for source in sorted(STAGE.rglob('*')):
        if not source.is_file():continue
        rel=source.relative_to(STAGE)
        if omitted(source):excluded.append(rel.as_posix());continue
        copy(source,target/rel);copied.append(rel.as_posix())
    for name in ('verify_resources.py','reproduce.py','RESOURCE_GUIDE_ZH.txt'):copy(PACKAGING/name,target/name)
    stable=selection_plan['stable_input_sha256'];references=[]
    for source,pin in stable.items():
        src=Path(source);rel=Path('reference_inputs')/src.name
        if (target/rel).exists():raise ValueError('Reference basename collision; explicitly map before sealing')
        copy(src,target/rel,pin);references.append({'source':source,'packaged_path':rel.as_posix(),'sha256':pin,'role':'stable scientific source metadata/code provenance'})
    write(target/'reference_inputs/INPUT_BINDINGS.json',{'actual_utc':datetime.now(timezone.utc).isoformat(),'stable_input_count':len(references),'files':references,'historical_admin_readtime_provenance_not_current_copies':selection_plan['administrative_read_time_provenance_sha256'],'no_historical_admin_bytes_reconstructed':True})
    copy(raw,target/'trusted_data/jarvis_dft_3d.json.gz',RAW_SHA)
    (target/'DATA_ATTRIBUTION.txt').write_text('JARVIS processed Matminer jarvis_dft_3d: DOI10.6084/m9.figshare.7261832.v1, published MIT. Originalsource DOI10.6084/m9.figshare.6815699.v2, CC BY4. Retain both upstream attribution and supplied source-license notices. Source bulk units/aggregation remain qualified; this is existing DFT data, not wet/newDFT measurements.\n',encoding='utf-8')
    refs={'current_scope':'selfcontained current NIW admission increment, including original public native source and local numerical artifacts','raw_source':{'path':'trusted_data/jarvis_dft_3d.json.gz','bytes':30669285,'sha256':RAW_SHA,'producer_default_input':False,'trusted_reader_physical_full_parse':True},'historical_external_not_embedded':[
        {'path':'H:/PolymerReview/Relation_Moment_20261105T124422Z.zip','sha256':'2b9825b8d023ba8d070c95d0b182e8536d639e8662782dd292ad69a66fa208fe'},
        {'path':'H:/PolymerReview/Eigen_repair_expansion_20261105.zip','sha256':'d8caed0f804a26535c2d4b3d6c42a99db80d522dd91e25afc6612455f6f6056b'},
        {'path':'H:/PolymerReview/MLST_frozen_20261105T093229Z','remote_archive_commit':'05a40915be149857b2bb9cf65144d53fa9b24f37'}]}
    write(target/'ARCHIVE_REFERENCES.json',refs)
    write(target/'RESOURCE_SCOPE.json',{'actual_utc':datetime.now(timezone.utc).isoformat(),'stage_files_copied':copied,'omitted_downloaded_papers_compiled_cache_archives':excluded,'local_only_numerical_payloads':['trusted_data','admission/forecast_once','admission/score_once'],'publicGit_scope':'separate explicit root allowlist; no default rawgzip/fit/diagY/numericalpayload export','cohort':'256developmentgroups,64fit/192diagnostic;0independentconfirmation','laws':'2304probabilityforecasts,not2304materials','release_gate_file_sha256':sha(args.release_gate),'science_admission_success_not_implied_by_packaging':True})
    report=gate['required_artifact_roles']['report']
    title='材料反馈：真实DFT模型准入检查点'
    text='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>'+title+'</title><body style="font:19px/1.9 sans-serif;max-width:980px;margin:40px auto"><h1>'+title+'</h1><p>'+html.escape(gate['summary_zh'])+'</p><ol><li><a href="'+html.escape(report,quote=True)+'">完整中文报告</a></li><li><a href="RESOURCE_GUIDE_ZH.txt">程序、算法与复现说明</a></li><li><a href="admission/README.txt">三模块原始执行说明</a></li><li><a href="task_contract/CONTRACT_BRIEF_ZH.txt">任务契约与来源</a></li></ol><p>原公共gzip、13稳定来源、实际源码/协议、真实预测/评分及独立审计均在本地资源内。它保存256个开发组，不是独立确认或已完成闭环智能体。旧包独立保留；默认只准备复现，不自动运行。</p></body></html>'
    (target/'INDEX.html').write_text(text,encoding='utf-8')
    if not(target/'requirements.txt').exists():(target/'requirements.txt').write_text('numpy==2.1.3\nscipy==1.15.3\n# Python3.10+; actual original environment Python3.13.5. Matplotlib only for optional report figures.\n',encoding='utf-8')
    manifest=[]
    for file in sorted(target.rglob('*')):
        if file.is_file():manifest.append({'path':file.relative_to(target).as_posix(),'bytes':file.stat().st_size,'sha256':sha(file)})
    write(target/'PACKAGE_MANIFEST.json',{'actual_utc':datetime.now(timezone.utc).isoformat(),'fixed_topfolder':target.name,'self_excluded':True,'files':manifest})
    with ZipFile(archive,'x',ZIP_DEFLATED,compresslevel=6) as z:
        for file in sorted(target.rglob('*')):
            if file.is_file():z.write(file,target.name+'/'+file.relative_to(target).as_posix())
    # Files were copied, not executed; root/auditor subsequently verify delivery.
    for rel,pin in required.items():
        if sha(STAGE/rel)!=pin or sha(target/rel)!=pin:raise ValueError('Scientific bytes changed during package build')
    receipt={'actual_utc':datetime.now(timezone.utc).isoformat(),'status':'PREPARED_AWAITING_INDEPENDENT_DELIVERY_AUDIT','directory':str(target),'zip':str(archive),'zip_sha256':sha(archive),'zip_bytes':archive.stat().st_size,'manifest_sha256':sha(target/'PACKAGE_MANIFEST.json'),'payload_files':len(manifest),'members':len(manifest)+1,'root_release_gate_sha256':sha(args.release_gate),'builder_sha256':sha(__file__),'model_or_score_runs':0,'original_public_native_asset_embedded':True}
    write(target.with_name(target.name+'_PREPARED.json'),receipt);print(json.dumps(receipt,ensure_ascii=True))

if __name__=='__main__':main()

"""Build only after root's final science/audit packaging gate; never runs a model."""
from pathlib import Path
from datetime import datetime,timezone
from zipfile import ZipFile,ZIP_DEFLATED
import argparse,hashlib,html,json,shutil

STAGE=Path(__file__).resolve().parent.parent
PACKAGING=Path(__file__).resolve().parent
DEFAULT=Path(r'H:\PolymerReview\Material_Feedback_20261106T020921Z')
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
    roles=('phase1_forecast_freeze','phase1_score_results','phase1_independent_saved_audit',
           'phase2_forecast_freeze','phase2_score_results','phase2_independent_saved_audit',
           'strong_baseline_terminal_record','strong_baseline_independent_audit','report')
    for key in roles:
        rel=gate['required_artifact_roles'][key]
        if rel not in required:raise ValueError('Final artifact role is not pinned: '+key)
    log_rows=gate['run_log_snapshots']
    expected_runs={'20261105T145420019666Z','20261106T020921917619Z'}
    if len(log_rows)!=2 or {r['run_id'] for r in log_rows}!=expected_runs:
        raise ValueError('Both original interrupted and actual resumed run snapshots required')
    for row in log_rows:
        base='run_logs/'+row['run_id']+'/'
        pins=row['files_sha256']
        if not {base+'RUN.json',base+'EVENTS.jsonl'}.issubset(pins):
            raise ValueError('Real start and append-only event snapshots required')
        for rel,pin in pins.items():
            if not rel.startswith(base) or required.get(rel)!=pin:
                raise ValueError('Run snapshot is not separately bound in root final gate')
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
    for name in ('verify_resources.py','reproduce.py','RESOURCE_GUIDE_ZH.txt','DETAILED_ALGORITHM_AND_CODE_ZH.txt'):copy(PACKAGING/name,target/name)
    stable=selection_plan['stable_input_sha256'];references=[];path_map={}
    for source,pin in stable.items():
        src=Path(source);rel=Path('reference_inputs')/src.name
        if (target/rel).exists():raise ValueError('Reference basename collision; explicitly map before sealing')
        copy(src,target/rel,pin);references.append({'source':source,'packaged_path':rel.as_posix(),'sha256':pin,'role':'stable scientific source metadata/code provenance'});path_map[source]=rel.as_posix()
    write(target/'reference_inputs/INPUT_BINDINGS.json',{'actual_utc':datetime.now(timezone.utc).isoformat(),'stable_input_count':len(references),'files':references,'historical_admin_readtime_provenance_not_current_copies':selection_plan['administrative_read_time_provenance_sha256'],'no_historical_admin_bytes_reconstructed':True})
    phase2=read(STAGE/'phase2_yj/EXECUTION_PLAN_TEMPLATE.json');all_bindings=[]
    if len(stable)!=13 or len(phase2['stable_sources_sha256'])!=16:raise ValueError('Expected thirteen original plus three existing stage-local provenance bindings')
    for source,pin in phase2['stable_sources_sha256'].items():
        if source not in path_map:
            src=Path(source).resolve();rel=src.relative_to(STAGE).as_posix()
            if sha(src)!=pin or sha(target/rel)!=pin:raise ValueError('Stage-local stable source changed')
            path_map[source]=rel
        if sha(target/path_map[source])!=pin:raise ValueError('Offline source binding changed')
        all_bindings.append({'original_source':source,'packaged_path':path_map[source],'sha256':pin})
    write(target/'RESOURCE_PATH_MAP.json',{'actual_utc':datetime.now(timezone.utc).isoformat(),'original_to_packaged_path':path_map,'phase2_sixteen_stable_bindings':all_bindings,'original_external_sources_copied_once':13,'stage_internal_bindings_reused_without_duplicate_copy':3})
    copy(raw,target/'trusted_data/jarvis_dft_3d.json.gz',RAW_SHA)
    (target/'DATA_ATTRIBUTION.txt').write_text('JARVIS processed Matminer jarvis_dft_3d: DOI10.6084/m9.figshare.7261832.v1, published MIT. Originalsource DOI10.6084/m9.figshare.6815699.v2, CC BY4. Retain both upstream attribution and supplied source-license notices. Source bulk units/aggregation remain qualified; this is existing DFT data, not wet/newDFT measurements.\n',encoding='utf-8')
    refs={'current_scope':'selfcontained two-phase NIW/fixed-YJ development admission and fixed posthoc learning references, with original public native source','raw_source':{'path':'trusted_data/jarvis_dft_3d.json.gz','bytes':30669285,'sha256':RAW_SHA,'producer_default_input':False,'trusted_reader_physical_full_parse':True},'historical_external_not_embedded':[
        {'path':'H:/PolymerReview/Relation_Moment_20261105T124422Z.zip','sha256':'2b9825b8d023ba8d070c95d0b182e8536d639e8662782dd292ad69a66fa208fe'},
        {'path':'H:/PolymerReview/Eigen_repair_expansion_20261105.zip','sha256':'d8caed0f804a26535c2d4b3d6c42a99db80d522dd91e25afc6612455f6f6056b'},
        {'path':'H:/PolymerReview/MLST_frozen_20261105T093229Z','remote_archive_commit':'05a40915be149857b2bb9cf65144d53fa9b24f37'}]}
    write(target/'ARCHIVE_REFERENCES.json',refs)
    write(target/'RESOURCE_SCOPE.json',{'actual_utc':datetime.now(timezone.utc).isoformat(),'stage_files_copied':copied,'omitted_downloaded_papers_compiled_cache_archives':excluded,'local_only_numerical_payloads':['trusted_data','admission/forecast_once','admission/score_once','phase2_yj/forecast_once','phase2_yj/score_once','strong_baselines/forecast_once','strong_baselines/score_once'],'publicGit_scope':'separate explicit root allowlist; no default rawgzip/fit/diagY/numericalpayload export','cohort':'two disjoint256development-group cohorts, each64fit/192diagnostic;512representative materials total,0independentconfirmation','laws':'phase1=2304,phase2=3840,fixedposthocreferences=1536 probability laws on samephase2;not material sample counts','phase1_NO_GO_retained':True,'phase2_admission_not_all_domain_strongbaseline_or_agent_success':True,'run_log_snapshots':log_rows,'run_snapshot_boundary':'Real copied bytes at root gate; no fabricated historical close. Any later actual run close/delivery receipt remains external to sealed package.','release_gate_file_sha256':sha(args.release_gate),'science_admission_success_not_implied_by_packaging':True})
    report=gate['required_artifact_roles']['report']
    optional_links=''
    if 'mixed_measure_addendum' in gate['required_artifact_roles']:
        addendum=gate['required_artifact_roles']['mixed_measure_addendum']
        if addendum not in required:raise ValueError('Mixed-measure addendum must be separately root-pinned')
        optional_links+='<li><a href="'+html.escape(addendum,quote=True)+'">新增混合测度与台账数学门：结果及限定</a></li>'
    baseline_guide='report/BASELINE_ALGORITHM_AND_CODE_ZH.txt'
    if (target/baseline_guide).is_file():
        optional_links+='<li><a href="'+baseline_guide+'">GP与森林新增模块的详细算法和代码说明</a></li>'
    title='材料反馈：真实DFT模型准入检查点'
    text='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>'+title+'</title><body style="font:19px/1.9 sans-serif;max-width:980px;margin:40px auto"><h1>'+title+'</h1><p>'+html.escape(gate['summary_zh'])+'</p><ol><li><a href="'+html.escape(report,quote=True)+'">完整中文报告</a></li><li><a href="RESOURCE_GUIDE_ZH.txt">资源与六模式复现入口</a></li><li><a href="DETAILED_ALGORITHM_AND_CODE_ZH.txt">算法、推导与程序资源说明</a></li><li><a href="root_review/DETAILED_ALGORITHM_AND_CODE_ZH.txt">核心算法与各模块详细讲解</a></li><li><a href="strong_baselines/README.txt">固定GP与森林条件分布对照</a></li>'+optional_links+'<li><a href="task_contract/CONTRACT_BRIEF_ZH.txt">任务契约与来源</a></li><li><a href="RESOURCE_SCOPE.json">资料边界与两轮真实日志快照</a></li></ol><p>原公共gzip、13外部稳定来源及全部16绑定、两阶段实际源码/协议、真实预测/评分、失败与独立审计均在本地资源内。两阶段共有512个开发材料代表，不是独立确认或已完成闭环智能体。旧包独立保留；默认只准备复现，不自动运行。</p></body></html>'
    (target/'INDEX.html').write_text(text,encoding='utf-8')
    if not(target/'requirements.txt').exists():(target/'requirements.txt').write_text('numpy==2.1.3\nscipy==1.15.3\nscikit-learn==1.6.1\njoblib==1.4.2\nmatplotlib==3.10.0\n# Python3.10+; original environment Python3.13.5. No runtime binaries embedded.\n',encoding='utf-8')
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

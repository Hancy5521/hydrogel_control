"""Prepare by default; explicit --run repeats this consumed development admission only."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,subprocess,sys

ROOT=Path(__file__).resolve().parent

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text('utf-8'))
def write(p,x):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def validate_bound_inputs(plan):
    for n,h in plan['source_sha256'].items():
        if sha(ROOT/'admission/code'/n)!=h:raise ValueError('Scientific source changed: '+n)
    for item in plan['inputs'].values():
        Path(item['file']).resolve().relative_to(ROOT)
        if sha(item['file'])!=item['sha256']:raise ValueError('Bound reproduction input changed')
    e=plan['independent_math_receipt']
    Path(e['file']).resolve().relative_to(ROOT)
    if sha(e['file'])!=e['sha256']:raise ValueError('Independent math receipt changed')
def validate_saved_freeze(folder,plan_file):
    f=read(folder/'FORECAST_FREEZE.json')
    if f['status']!='COMPLETE_ALL_FORECASTS_FROZEN' or f['plan_sha256']!=sha(plan_file):raise ValueError('Complete matching reproduction forecast freeze required')
    if set(f['files'])!={'VISIBLE_INPUTS.json','MODEL.json','FORECASTS.json','EXTRACTION_RECEIPT.json','EXECUTION_START.json'}:raise ValueError('Incomplete frozen artifact set')
    for n,h in f['files'].items():
        if sha(folder/n)!=h:raise ValueError('Frozen forecast file changed')
    forecasts=read(folder/'FORECASTS.json')['forecasts']
    if len(forecasts)!=2304 or len({(r['component_id'],r['model'],r['mask']) for r in forecasts})!=2304:raise ValueError('All2304unique forecasts required')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=['forecast','score'],default='forecast');p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--forecast-reproduction',type=Path)
    m=p.add_mutually_exclusive_group();m.add_argument('--prepare-only',action='store_true');m.add_argument('--run',action='store_true');args=p.parse_args()
    out=args.output_dir.resolve()
    if out==ROOT or out.is_relative_to(ROOT):raise ValueError('Use a new output directory outside the sealed package')
    if out.exists():raise ValueError('Output already exists; preserve previous reproduction')
    now=datetime.now(timezone.utc).isoformat();source=ROOT/'admission/code'
    if args.phase=='forecast':
        if args.forecast_reproduction:raise ValueError('Forecast phase does not consume another forecast')
        plan=read(ROOT/'admission/EXECUTION_PLAN_TEMPLATE.json');plan['status']='REGISTERED';plan['registered_actual_utc']=now
        mapping={'raw_source':ROOT/'trusted_data/jarvis_dft_3d.json.gz','roles':ROOT/'task_contract/selection_once/ADMISSION_ROLE_SELECTION.json','contract':ROOT/'task_contract/MATERIAL_TASK_CONTRACT_V1.json','root_scope':ROOT/'root_review/ROOT_ADMISSION_SCOPE_BEFORE_LABELS.json'}
        for name,path in mapping.items():plan['inputs'][name]['file']=str(path)
        plan['independent_math_receipt']['file']=str(ROOT/'independent_niw_math/execution_once/INDEPENDENT_NIW_T_CRPS_AUDIT.json')
        validate_bound_inputs(plan);plan_bytes=None;forecast=None;script=source/'admission_io.py';allowed='forecast'
    else:
        if not args.forecast_reproduction:raise ValueError('Score requires an actually completed external forecast reproduction')
        prior=args.forecast_reproduction.resolve();prior_plan=prior/'REPRODUCTION_PROTOCOL.json';plan=read(prior_plan);validate_bound_inputs(plan)
        forecast=prior/'execution';validate_saved_freeze(forecast,prior_plan)
        # Keep exact forecast-plan bytes/hash; a fresh timestamp here would break freeze linkage.
        plan_bytes=prior_plan.read_bytes();script=source/'score_admission.py';allowed='score_after_freeze'
    out.mkdir(parents=True,exist_ok=False);plan_path=out/'REPRODUCTION_PROTOCOL.json';approval_path=out/'REPRODUCTION_APPROVAL.json'
    if plan_bytes is None:write(plan_path,plan)
    else:plan_path.write_bytes(plan_bytes)
    approval={'actual_utc':now,'approved':True,'scope':'WARM_ONLY_NIW_MATERIAL_ADMISSION','allowed_phases':[allowed],'plan_sha256':sha(plan_path),'kind':'MANUAL_REPRODUCTION_PHASE_RECORD_NOT_ORIGINAL_RESEARCH_APPROVAL','independent_confirmation':False,'new_wet_or_DFT_runs':0}
    write(approval_path,approval)
    command=[sys.executable,str(script),'--plan',str(plan_path),'--approval',str(approval_path)]
    if forecast is not None:command+=['--forecast-dir',str(forecast)]
    command+=['--output-dir',str(out/'execution')]
    write(out/'REPRODUCTION_ORIGIN.json',{'actual_utc':now,'phase':args.phase,'prepared_only':not args.run,'resolved_command':command,'source_template_sha256':sha(ROOT/'admission/EXECUTION_PLAN_TEMPLATE.json'),'forecast_plan_reused_exactly_for_score':args.phase=='score','repeat_of_consumed_256_development_groups':True,'not_independent_confirmation':True,'original_frozen_protocols_modified':False,'byte_checks_may_read_entire_gzip_but_no_decode_in_prepare':True})
    if not args.run:
        print(json.dumps({'status':'PREPARED_ONLY','phase':args.phase,'plan_sha256':sha(plan_path),'native_property_decoding':0,'model_or_score_runs':0}));return
    result=subprocess.run(command,check=False)
    write(out/'REPRODUCTION_FINISHED.json',{'actual_utc':datetime.now(timezone.utc).isoformat(),'returncode':result.returncode,'phase':args.phase,'independent_confirmation':False})
    raise SystemExit(result.returncode)

if __name__=='__main__':main()

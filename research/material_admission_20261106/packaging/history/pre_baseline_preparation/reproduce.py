"""Default metadata preparation; --run explicitly repeats consumed development only."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, subprocess, sys
ROOT=Path(__file__).resolve().parent
TASKS={
 'phase1':dict(template='admission/EXECUTION_PLAN_TEMPLATE.json',code='admission/code',forecast='admission_io.py',score='score_admission.py',scope='WARM_ONLY_NIW_MATERIAL_ADMISSION',count=2304,status='COMPLETE_ALL_FORECASTS_FROZEN',reference='MODEL.json'),
 'phase2':dict(template='phase2_yj/EXECUTION_PLAN_TEMPLATE.json',code='phase2_yj/code',forecast='run_phase2.py',score='score_phase2.py',scope='PHASE2_FIXED_YJ_HALF_ADMISSION',count=3840,status='COMPLETE_ALL3840_FORECASTS_FROZEN',reference='FIT_REFERENCE.json')}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def write(p,x):
 with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def inside(p):
 q=Path(p).resolve();q.relative_to(ROOT);return q
def validate(plan,t):
 for n,h in plan['source_sha256'].items():
  if sha(ROOT/t['code']/n)!=h:raise ValueError('Scientific source changed: '+n)
 for n,h in plan.get('parent_source_sha256',{}).items():
  if sha(ROOT/'admission/code'/n)!=h:raise ValueError('Parent source changed')
 for item in plan['inputs'].values():
  if sha(inside(item['file']))!=item['sha256']:raise ValueError('Bound local input changed')
 receipts=plan.get('independent_math_receipts',{'single':plan.get('independent_math_receipt')})
 for item in receipts.values():
  if item is not None and sha(inside(item['file']))!=item['sha256']:raise ValueError('Independent math receipt changed')
 for n,h in plan.get('stable_sources_sha256',{}).items():
  if sha(inside(n))!=h:raise ValueError('Stable source changed')
def local_plan(phase):
 t=TASKS[phase];p=read(ROOT/t['template']);p['status']='REGISTERED';p['registered_actual_utc']=datetime.now(timezone.utc).isoformat()
 mapping={'raw_source':'trusted_data/jarvis_dft_3d.json.gz','roles':('task_contract/selection_once/ADMISSION_ROLE_SELECTION.json' if phase=='phase1' else 'phase2_yj/task_contract/selection_once/ADMISSION_ROLE_SELECTION.json'),'phase1_roles':'task_contract/selection_once/ADMISSION_ROLE_SELECTION.json','contract':'task_contract/MATERIAL_TASK_CONTRACT_V1.json','root_scope':('root_review/ROOT_ADMISSION_SCOPE_BEFORE_LABELS.json' if phase=='phase1' else 'phase2_yj/ROOT_FIXED_POWER_SCOPE_BEFORE_NEW_LABELS.json')}
 for n,item in p['inputs'].items():item['file']=str(ROOT/mapping[n])
 if phase=='phase1':p['independent_math_receipt']['file']=str(ROOT/'independent_niw_math/execution_once/INDEPENDENT_NIW_T_CRPS_AUDIT.json')
 else:
  p['independent_math_receipts']['niw_t']['file']=str(ROOT/'independent_niw_math/execution_once/INDEPENDENT_NIW_T_CRPS_AUDIT.json')
  p['independent_math_receipts']['YJ_native_probability_and_score']['file']=str(ROOT/'independent_yj_math/execution_once/INDEPENDENT_YJ_HALF_MATH_AUDIT.json')
  mapping=read(ROOT/'RESOURCE_PATH_MAP.json')['original_to_packaged_path'];p['stable_sources_sha256']={str(ROOT/mapping[n]):h for n,h in p['stable_sources_sha256'].items()}
 validate(p,t);return p
def validate_freeze(folder,protocol,t):
 frozen=read(folder/'FORECAST_FREEZE.json')
 if frozen['status']!=t['status'] or frozen['plan_sha256']!=sha(protocol):raise ValueError('Complete matching replay freeze required')
 if set(frozen['files'])!={'EXECUTION_START.json','VISIBLE_INPUTS.json','EXTRACTION_RECEIPT.json',t['reference'],'FORECASTS.json'}:raise ValueError('Incomplete freeze')
 for n,h in frozen['files'].items():
  if sha(folder/n)!=h:raise ValueError('Frozen forecast changed')
 laws=read(folder/'FORECASTS.json')['forecasts']
 if len(laws)!=t['count'] or len({(x['component_id'],x['model'],x['mask']) for x in laws})!=t['count']:raise ValueError('All unique laws required')
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=[a+'-'+b for a in TASKS for b in ('forecast','score')],default='phase2-forecast');p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--forecast-reproduction',type=Path)
 m=p.add_mutually_exclusive_group();m.add_argument('--prepare-only',action='store_true');m.add_argument('--run',action='store_true');a=p.parse_args();phase,op=a.mode.split('-');t=TASKS[phase];out=a.output_dir.resolve()
 if out==ROOT or out.is_relative_to(ROOT) or out.exists():raise ValueError('New absent directory outside sealed package required')
 now=datetime.now(timezone.utc).isoformat();forecast=None;plan_bytes=None
 if op=='forecast':
  if a.forecast_reproduction:raise ValueError('Forecast cannot consume prior replay')
  plan=local_plan(phase);allowed='forecast'
 else:
  if not a.forecast_reproduction:raise ValueError('Score needs completed same-phase forecast replay')
  prior=a.forecast_reproduction.resolve();origin=read(prior/'REPRODUCTION_ORIGIN.json')
  if origin['mode']!=phase+'-forecast' or origin['prepared_only'] is not False:raise ValueError('Same-phase executed replay required')
  protocol=prior/'REPRODUCTION_PROTOCOL.json';plan=read(protocol);validate(plan,t);forecast=prior/'execution';validate_freeze(forecast,protocol,t);plan_bytes=protocol.read_bytes();allowed='score_after_freeze'
 out.mkdir(parents=True,exist_ok=False);protocol=out/'REPRODUCTION_PROTOCOL.json';approval=out/'REPRODUCTION_APPROVAL.json'
 if plan_bytes is None:write(protocol,plan)
 else:protocol.write_bytes(plan_bytes)
 write(approval,dict(actual_utc=now,approved=True,scope=t['scope'],allowed_phases=[allowed],plan_sha256=sha(protocol),kind='MANUAL_REPLAY_RECORD_NOT_ORIGINAL_RESEARCH_APPROVAL',independent_confirmation=False,new_wet_or_DFT_runs=0))
 cmd=[sys.executable,str(ROOT/t['code']/t[op]),'--plan',str(protocol),'--approval',str(approval)]
 if forecast is not None:cmd+=['--forecast-dir',str(forecast)]
 cmd+=['--output-dir',str(out/'execution')]
 write(out/'REPRODUCTION_ORIGIN.json',dict(actual_utc=now,mode=a.mode,prepared_only=not a.run,scope='Replay of consumed development; not independent confirmation',source_package=str(ROOT),command=cmd,protocol_sha256=sha(protocol),original_scientific_artifacts_modified=False,strong_baseline_replay='separate frozen CLI, not automatically run'))
 if a.run:
  r=subprocess.run(cmd,check=False);write(out/'REPRODUCTION_EXIT.json',dict(actual_utc=datetime.now(timezone.utc).isoformat(),returncode=r.returncode))
  if r.returncode:raise SystemExit(r.returncode)
 print(json.dumps(dict(status='RUN_FINISHED' if a.run else 'PREPARED_ONLY_NO_MODEL_OR_PROPERTY_DECODE',mode=a.mode,output=str(out)),ensure_ascii=True))
if __name__=='__main__':main()

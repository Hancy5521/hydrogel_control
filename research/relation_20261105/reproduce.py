"""Replay an existing synthetic control in a NEW external directory.

This creates actual-time reproduction records, never independent confirmation.
Use --prepare-only to inspect resolved paths and source pins without executing.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
TASKS={
 'core':('math_design/run_synthetic.py','math_design/SYNTHETIC_PROTOCOL_TEMPLATE.json','RELATION_POSTERIOR_SYNTHETIC_CONTROL'),
 'moment':('math_design/moment_followup/run_moment_control.py','math_design/moment_followup/EXECUTION_PLAN_TEMPLATE.json','MOMENT_READOUT_SYNTHETIC_CONTROL'),
 'scaling':('math_design/moment_followup/posthoc_scaling/run_scaling_extension.py','math_design/moment_followup/posthoc_scaling/EXECUTION_PLAN_TEMPLATE.json','POSTHOC_MOMENT_SCALING_EXTENSION'),
 'rational_math':('independent_moment_math/math_audit.py','independent_moment_math/MOMENT_MATH_AUDIT_PLAN.json','INDEPENDENT_RATIONAL_MOMENT_MATH_AUDIT'),
 'cdf_math':('independent_math/independent_complementarity_audit.py','independent_math/EXECUTION_PLAN_V2.json','INDEPENDENT_TOY_RELATION_CRPS_MATH_ONLY'),
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p,value):
    with p.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',choices=tuple(TASKS),required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args();out=args.output_dir.resolve()
    if out==ROOT or out.is_relative_to(ROOT):
        raise ValueError('Choose an output directory outside the sealed resource folder')
    script_rel,template_rel,scope=TASKS[args.task]
    script=ROOT/script_rel;template=ROOT/template_rel
    plan=json.loads(template.read_text('utf-8'));stamp=datetime.now(timezone.utc).isoformat()
    if 'source_sha256' in plan and isinstance(plan['source_sha256'],dict):
        for name,pin in plan['source_sha256'].items():
            if sha(script.parent/name)!=pin:raise ValueError('Source changed: '+name)
    if args.task=='core':
        plan['prior_independent_identity_receipt']['file']=str(ROOT/'independent_math/execution_v2_once/INDEPENDENT_MATH_RESULT.json')
    elif args.task=='moment':
        plan['prior_audit_receipts']['centered_CRPS_identity']['file']=str(ROOT/'independent_math/execution_v2_once/INDEPENDENT_MATH_RESULT.json')
        plan['prior_audit_receipts']['exact_moment_interval']['file']=str(ROOT/'independent_moment_math/execution_20261105T133636538641Z/INDEPENDENT_MOMENT_MATH_AUDIT.json')
    elif args.task=='scaling':
        plan['prior_result']['file']=str(ROOT/'math_design/moment_followup/execution_once/RESULTS.json')
    elif args.task=='rational_math':
        plan['audit_script_path']=str(script)
        if sha(script)!=plan['audit_script_sha256']:raise ValueError('Independent source changed')
        plan['planned_output_directory']=str(out/'execution')
    elif args.task=='cdf_math':
        plan['source_file']=str(script)
        if sha(script)!=plan['source_sha256']:raise ValueError('Independent source changed')
    if 'status' in plan:plan['status']='REGISTERED'
    if 'registered_actual_utc' in plan:plan['registered_actual_utc']=stamp
    if 'actual_registered_utc' in plan:plan['actual_registered_utc']=stamp
    if 'actual_registered_asia_shanghai' in plan:
        from zoneinfo import ZoneInfo
        plan['actual_registered_asia_shanghai']=datetime.fromisoformat(stamp).astimezone(ZoneInfo('Asia/Shanghai')).isoformat()
    if 'registered_asia_shanghai' in plan:
        from zoneinfo import ZoneInfo
        plan['registered_asia_shanghai']=datetime.fromisoformat(stamp).astimezone(ZoneInfo('Asia/Shanghai')).isoformat()
    out.mkdir(parents=True,exist_ok=False)
    plan_path=out/'REPRODUCTION_PROTOCOL.json';approval_path=out/'REPRODUCTION_APPROVAL.json'
    command=[sys.executable,str(script),'--plan',str(plan_path),'--approval',str(approval_path),'--output-dir',str(out/'execution')]
    if 'unique_command' in plan:plan['unique_command']=subprocess.list2cmdline(command)
    write(plan_path,plan)
    approval={'actual_utc':stamp,'approved':True,'scope':scope,'approved_scope':scope,
              'plan_sha256':sha(plan_path),'source_sha256':sha(script),
              'kind':'MANUALLY_INVOKED_REPRODUCTION_RECORD_NOT_ORIGINAL_RESEARCH_APPROVAL',
              'independent_confirmation':False,'new_material_measurements':0}
    write(approval_path,approval)
    write(out/'REPRODUCTION_ORIGIN.json',{'actual_utc':stamp,'task':args.task,'original_template':template_rel,
        'original_template_sha256':sha(template),'source_sha256':sha(script),'prepared_only':args.prepare_only,
        'meaning':'Repeat already-consumed synthetic computation; no new material confirmation',
        'resolved_command':command,'original_frozen_protocols_modified':False})
    if args.prepare_only:
        print(json.dumps({'status':'PREPARED_ONLY','task':args.task,'plan_sha256':sha(plan_path),
                          'experiment_runs':0,'new_material_measurements':0}));return
    result=subprocess.run(command,check=False)
    write(out/'REPRODUCTION_FINISHED.json',{'actual_utc':datetime.now(timezone.utc).isoformat(),
        'returncode':result.returncode,'independent_confirmation':False,'new_material_measurements':0})
    raise SystemExit(result.returncode)


if __name__=='__main__':
    main()

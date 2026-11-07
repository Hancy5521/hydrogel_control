"""Run the six fixed forecast/verification phases once; never score or retry.

Interrupted runs retain child PIDs and exact completed-phase receipts. Resume
by inspecting those receipts, not by invoking this driver on an existing root.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def utc():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,indent=2);f.write('\n')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--approval',type=Path,required=True)
    ap.add_argument('--roles',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True)
    a=ap.parse_args();stage=Path(__file__).resolve().parent
    plan=json.loads(a.plan.read_text('utf8'));approval=json.loads(a.approval.read_text('utf8'))
    if (plan['status']!='REGISTERED' or approval['approved'] is not True
        or approval['scope']!=plan['scope'] or approval['plan_sha256']!=sha(a.plan)
        or not {'produce_round_forecasts','verify_saved_round'}.issubset(approval['allowed_phases'])
        or sha(__file__)!=plan['dependencies']['round_driver']['sha256']):
        raise ValueError('exact source-frozen execution and two-phase approval required')
    for name,h in plan['source_sha256'].items():
        if sha(stage/'code'/name)!=h:raise ValueError('changed scientific source')
    roles=json.loads(a.roles.read_text('utf8'))
    if roles['plan_sha256']!=sha(a.plan):raise ValueError('roles from another execution')
    a.output_dir.mkdir(parents=True,exist_ok=False)
    write(a.output_dir/'DRIVER_START.json',{'actual_utc':utc(),'plan_sha256':sha(a.plan),
          'approval_sha256':sha(a.approval),'roles_sha256':sha(a.roles),'driver_sha256':sha(__file__),
          'no_fullscore_or_automatic_retry':True})
    events=a.output_dir/'DRIVER_EVENTS.jsonl'
    for number in (1,2,3):
        current=a.output_dir/('round'+str(number));current.mkdir()
        common=['--plan',str(a.plan.resolve()),'--approval',str(a.approval.resolve()),
                '--roles',str(a.roles.resolve()),'--round',str(number)]
        forecast=[sys.executable,'-X','utf8','-B',str(stage/'code/forecast_stage.py')]+common+['--output-dir',str((current/'forecast').resolve())]
        if number>1:forecast+=['--previous-round-dir',str((a.output_dir/('round'+str(number-1))).resolve())]
        verify=[sys.executable,'-X','utf8','-B',str(stage/'code/verification_stage.py')]+common+[
                '--forecast-dir',str((current/'forecast').resolve()),'--output-dir',str((current/'verification').resolve())]
        for phase,command in (('forecast',forecast),('verification',verify)):
            log=current/(phase+'_stdout_stderr.txt')
            with log.open('xb') as output:
                child=subprocess.Popen(command,cwd=stage,stdout=output,stderr=subprocess.STDOUT)
                event={'actual_utc':utc(),'round':number,'phase':phase,'status':'started','pid':child.pid,'command':command}
                with events.open('a',encoding='utf8') as f:f.write(json.dumps(event)+'\n')
                returncode=child.wait()
            event={'actual_utc':utc(),'round':number,'phase':phase,'status':'succeeded' if returncode==0 else 'failed',
                   'returncode':returncode,'log_sha256':sha(log)}
            with events.open('a',encoding='utf8') as f:f.write(json.dumps(event)+'\n')
            print(event,flush=True)
            if returncode:
                write(a.output_dir/'DRIVER_STOP.json',event)
                raise SystemExit(returncode)
    write(a.output_dir/'DRIVER_COMPLETE.json',{'actual_utc':utc(),'plan_sha256':sha(a.plan),
          'status':'ALL_THREE_FIXED_ROUNDS_CLOSED_AWAITING_INDEPENDENT_STATE_AUDIT',
          'full_score_authorized':False,'scientific_outcome_success_not_asserted':True})


if __name__=='__main__':main()

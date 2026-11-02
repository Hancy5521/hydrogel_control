"""Descriptive material-family arithmetic on frozen score scalars only.

No model/data reconstruction or target scorer is imported. Python standard library.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse,hashlib,json,math

EXPECTED='555f39ca076351b063511940f7b2a2a011e47ec9a55b0ce968c905e5f9b8c8d0'
DOI='10.17638/datacat.liverpool.ac.uk/1296'
METHODS=('curvature_joint_gaussian','reference_gp','conditional_gaussian','pchip_log','powerlaw_6anchors','powerlaw_last2')
FIELDS=('crps_log','mse_log','mae_log','coverage90','width90_log','interval_score90_log','whole_curve_covered90')
def average(records):
    result={k:math.fsum(float(r[k]) for r in records)/len(records) for k in FIELDS if all(k in r for r in records)}
    result['root_weighted_mse_log']=math.sqrt(result['mse_log'])
    return result
def analyze(path):
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=EXPECTED:raise ValueError('This diagnostic requires the exact frozen source scores')
    s=json.loads(raw)['sources'][DOI]['methods'];out={};influence=[]
    for name in METHODS:
        groups=s[name]['groups'];families={g['group_id']:('GG' if '_GG_' in g['group_id'] else 'peK') for g in groups}
        if len(groups)!=5 or list(families.values()).count('GG')!=3 or list(families.values()).count('peK')!=2:raise ValueError('Unexpected frozen group inventory')
        by={f:average([g for g in groups if families[g['group_id']]==f]) for f in ('GG','peK')}
        out[name]={'primary_five_groups':average(groups),'families':by,'equal_two_families':average(list(by.values()))}
        for omitted in sorted(families):influence.append({'method':name,'omitted_saved_group':omitted,**average([g for g in groups if g['group_id']!=omitted])})
    comps=[];c=out['curvature_joint_gaussian']
    for baseline in ('reference_gp','conditional_gaussian'):
        b=out[baseline];a=c['families']['GG']['crps_log']-b['families']['GG']['crps_log'];v=c['families']['peK']['crps_log']-b['families']['peK']['crps_log']
        comps.append({'baseline':baseline,'delta_GG':a,'delta_peK':v,'delta_primary_w_0p6':.6*a+.4*v,'delta_equal_family_w_0p5':.5*(a+v),'break_even_GG_weight':-v/(a-v)})
    return {'actual_utc':datetime.now(timezone.utc).isoformat(),'scope':'Descriptive saved-score arithmetic; not new prediction, target scoring, training, validation or independent batch evidence','input_sha256':EXPECTED,'methods':out,'comparisons':comps,'leave_one_saved_group_out':influence,'fit_calls':0,'forecast_calls':0,'target_score_calls':0,'primary_endpoint_replaced':False}
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError('Choose a new output path')
    result=analyze(args.repo_root/'external_validation/20261101T1239423181800Z/SCORES.json')
    with args.output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({'status':'SAVED_SCALAR_ARITHMETIC_ONLY','methods':6,'groups':5,'families':2,'model_or_scoring_calls':0}))
if __name__=='__main__':main()

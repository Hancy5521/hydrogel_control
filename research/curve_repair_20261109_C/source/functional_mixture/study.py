"""Registered development-only derivation from saved eighteen consumed contexts.

No raw data reader, no fitting, no source or hyperparameter selection. The old
mean-only pilot remains unchanged. No access to other161 temperature curves.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from pure import build_laws,score,METHODS


def utc():return dt.datetime.now(dt.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,obj):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,allow_nan=False,indent=2)


def authorize(path,phase):
    r=read(path)
    if r['schema']!='functional-mixture-saved18-development-v1' or phase not in r['authorized_phases']:
        raise PermissionError('ActualUTC saved-development registration required')
    stamp=dt.datetime.fromisoformat(r['actual_utc'])
    if stamp.tzinfo is None or stamp>dt.datetime.now(dt.timezone.utc):raise PermissionError('Actual nonfuture UTC required')
    if (r['methods']!=list(METHODS) or r['mixture_weight']!=.5 or r['student_df']!=4 or
            r['new_raw_G_accesses']!=0 or r['new_fits']!=0 or r['role']!='POST_REVEAL_DEVELOPMENT_DIAGNOSTIC'):
        raise PermissionError('No search, newraw or confirmation claim authorized')
    for p,digest in r['source_pins'].items():
        if sha(p)!=digest:raise ValueError('Pinned source/saved artifact changed: '+p)
    for name in ('toy_gate','independent_gate'):
        if sha(r[name+'_path'])!=r[name+'_sha256'] or read(r[name+'_path'])['status']!='PASS':
            raise PermissionError('Independent numerical gates required')
    post=read(r['parent_post_audit'])
    if (post['status']!='PASS' or post['phase']!='post' or
            post['predictions_sha256']!=sha(r['parent_predictions']) or
            post['targets_sha256']!=sha(r['parent_targets']) or post['scores_sha256']!=sha(r['parent_scores'])):
        raise PermissionError('Only originally audited consumed context artifacts admitted')
    return r


def predict(out,reg,path):
    old=read(reg['parent_predictions']);forecasts=[]
    for f in old['forecasts']:
        base={k:f[k] for k in ('context','condition_id','state_block','mask')}
        if not f['method_availability']['half_maxwell_mean_gaussian']['available']:
            forecasts.append({**base,'status':'FAIL','reason':'ParentGP_or_weightedTeacher_unavailable','laws':{}});continue
        mg=np.asarray(f['gp_mean']);mm=np.asarray(f['maxwell_log_location']);c=np.asarray(f['full_residual_covariance'])
        laws=build_laws(mg,mm,c)
        if not np.array_equal(laws[METHODS[0]]['mean'],f['laws']['half_maxwell_mean_gaussian']['mean']):
            raise ValueError('Common point location differs from original half-mean')
        forecasts.append({**base,'status':'OK','mG':mg.tolist(),'mM':mm.tolist(),'C':c.tolist(),
            'laws':laws,'common_point_log_location':(.5*(mg+mm)).tolist(),
            'component_residual_crosschannel_covariance':'original block-diagonal C retained',
            'total_predictive_crosschannel_covariance':'may be nonzero via sharedbranch .25ddprime',
            'student_radial_variable':'one chi_square4 for all14 dimensions, not independent7D products'})
    if len(forecasts)!=18 or len({f['context'] for f in forecasts})!=18:raise ValueError('All18 original contexts required')
    write(out/'PREDICTIONS.json',{'schema':'functional-mixture-saved-predictions-v1','actual_utc':utc(),
        'registration_sha256':sha(path),'parent_predictions_sha256':sha(reg['parent_predictions']),
        'forecasts':forecasts,'methods':list(METHODS),'contexts':18,
        'law_count':sum(len(f['laws']) for f in forecasts),'new_fits':0,'new_raw_G_accesses':0,
        'role':'POST_REVEAL_DEVELOPMENT_DIAGNOSTIC','other161_temperature_curves_accessed':False})
    write(out/'PREDICTION_COMPLETE.json',{'actual_utc':utc(),'predictions_sha256':sha(out/'PREDICTIONS.json')})


def evaluate(out,reg,path,pre_audit):
    pre=read(pre_audit);p=read(out/'PREDICTIONS.json')
    if (pre['status']!='PASS' or pre['predictions_sha256']!=sha(out/'PREDICTIONS.json') or
            read(out/'PREDICTION_COMPLETE.json')['predictions_sha256']!=sha(out/'PREDICTIONS.json')):
        raise PermissionError('Saved derived laws must freeze and pass independent law audit')
    targets={r['context']:r for r in read(reg['parent_targets'])['contexts']}
    oldrows={(r['context'],r['method']):r for r in read(reg['parent_scores'])['rows']}
    rows=[];failures=[];identities=[];bounds=[]
    for f in p['forecasts']:
        if f['status']!='OK':failures.append({'context':f['context'],'reason':f['reason']});continue
        target=targets[f['context']]
        if not target['target_admissible']:
            failures.append({'context':f['context'],'reason':'Parent_target_log_unavailable'});continue
        values=np.array(target['values_Pa']);y=np.log(values.reshape(14))
        base={k:f[k] for k in ('context','condition_id','state_block','mask')}
        old_rmse=oldrows[(f['context'],'half_maxwell_mean_gaussian')]['log_rmse']
        for method in METHODS:
            result=score(f['laws'][method],y)
            if abs(result['log_rmse']-old_rmse)>2e-12:raise ArithmeticError('Point RMSE identity violated')
            rows.append({**base,'method':method,**result})
            identities.append({'context':f['context'],'method':method,'log_rmse':result['log_rmse'],
                'original_half_log_rmse':old_rmse,'difference':result['log_rmse']-old_rmse,
                'scope':'structural identity, not a new point-prediction gain'})
            if method in METHODS[:2]:
                endpoint='full_anchor_gaussian' if method==METHODS[0] else 'full_anchor_student4'
                endpoint_nll=oldrows[(f['context'],endpoint)]['joint_nll_per_dim']
                gap=result['joint_nll_per_dim']-endpoint_nll
                if gap>math.log(2)/14+2e-10:raise ArithmeticError('Pointwise mixture logscore bound violated')
                bounds.append({'context':f['context'],'method':method,'same_tail_GP_endpoint':endpoint,
                    'endpoint_joint_nll_per_dim':endpoint_nll,'actual_gap':gap,'bound_log2_per14':math.log(2)/14,
                    'bound_applies_to_mixture_only':True,'holds':True})
    macro=[];comparisons=[]
    for mask in sorted({f['mask'] for f in p['forecasts']}):
        for method in METHODS:
            subset=[r for r in rows if r['mask']==mask and r['method']==method]
            macro.append({'mask':mask,'method':method,'planned_state_conditions':6,'scored_state_conditions':len(subset),
                'conditional_subset':len(subset)!=6,
                **{k:float(np.mean([r[k] for r in subset])) if subset else None
                   for k in ('joint_nll_per_dim','log_crps','log_rmse','coverage90','log_width90')}})
            references=('full_anchor_gaussian','full_anchor_student4','half_maxwell_mean_gaussian')
            for reference in references:
                paired=[(r,oldrows[(r['context'],reference)]) for r in subset if (r['context'],reference) in oldrows]
                comparisons.append({'mask':mask,'method':method,'reference':reference,'paired_state_conditions':len(paired),
                    'planned_state_conditions':6,'conditional_subset':len(paired)!=6,
                    **{'mean_delta_'+k:float(np.mean([a[k]-b[k] for a,b in paired])) if paired else None
                       for k in ('joint_nll_per_dim','log_crps','log_rmse','coverage90','log_width90')}})
    write(out/'SCORES.json',{'schema':'functional-mixture-saved-scores-v1','actual_utc':utc(),
        'registration_sha256':sha(path),'predictions_sha256':sha(out/'PREDICTIONS.json'),
        'parent_targets_sha256':sha(reg['parent_targets']),'parent_scores_sha256':sha(reg['parent_scores']),
        'pre_audit_sha256':sha(pre_audit),'rows':rows,'macro_by_mask':macro,'paired_vs_original':comparisons,
        'RMSE_identities':identities,'mixture_pointwise_bounds':bounds,'failures':failures,
        'contexts_denominator':18,'role':'POST_REVEAL_DEVELOPMENT_DIAGNOSTIC','new_raw_G_accesses':0,'new_fits':0,
        'other161_temperature_curves_accessed':False,'unseen60C_confirmation_executed':False,
        'independent_material_confirmation':False})
    write(out/'SCORE_COMPLETE.json',{'actual_utc':utc(),'scores_sha256':sha(out/'SCORES.json')})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['predict','score'])
    parser.add_argument('--registration',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--pre-audit');args=parser.parse_args();reg=authorize(args.registration,args.phase)
    out=Path(args.output)
    if args.phase=='predict':out.mkdir(parents=True,exist_ok=False);predict(out,reg,args.registration)
    else:evaluate(out,reg,args.registration,args.pre_audit)


if __name__=='__main__':main()

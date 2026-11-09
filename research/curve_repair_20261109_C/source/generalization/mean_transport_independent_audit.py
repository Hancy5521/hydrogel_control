"""Independent high-precision audit of one fixed-covariance mean segment.

SOURCE ONLY until root registers/executes the consumed-development oracle.
No production import, raw CSV/final20 reader, model refit or tau grid search.
Targets here were already consumed by preceding studies; no confirmation.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import datetime as dt
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import mpmath as mp
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def utc():return dt.datetime.now(dt.timezone.utc).isoformat()
def read(p):
    def unique(pairs):
        out={}
        for k,v in pairs:
            if k in out:raise ValueError('Duplicate JSON key: '+k)
            out[k]=v
        return out
    return json.loads(Path(p).read_text(encoding='utf-8-sig'),object_pairs_hook=unique,
        parse_constant=lambda value:(_ for _ in ()).throw(ValueError(value)))
def write_new(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

class Check:
    def __init__(self):self.items={}
    def close(self,name,actual,expected,atol=3e-7,rtol=3e-8):
        a,b=np.asarray(actual,float),np.asarray(expected,float)
        if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():raise AssertionError(name+':shape/nonfinite')
        err=float(np.max(np.abs(a-b),initial=0))
        row=self.items.setdefault(name,{'elements':0,'max_absolute_error':0.,'atol':atol,'rtol':rtol})
        row['elements']+=int(a.size);row['max_absolute_error']=max(row['max_absolute_error'],err)
        if not np.allclose(a,b,atol=atol,rtol=rtol):raise AssertionError(name+':error='+str(err))

def indices(rows,key):
    out={}
    for row in rows:
        if key(row) in out:raise AssertionError('Duplicate saved identity')
        out[key(row)]=row
    return out

def independent_contexts(reg):
    old,exp,xg=(reg['inputs'][k] for k in ['old_dev','expansion','xanthan'])
    g=read(old['gp_predictions']);s=read(old['student_predictions']);yt=read(old['targets'])
    if s['main_predictions_sha256']!=sha(old['gp_predictions']) or yt['predictions_sha256']!=sha(old['gp_predictions']):
        raise AssertionError('Old consumed source-chain mismatch')
    gs=indices(g['forecasts'],lambda r:(r['member'],r['method_id']))
    ss=indices(s['forecasts'],lambda r:(r['member'],r['method_id']))
    result=[]
    for truth in yt['curves']:
        member=truth['member'];gp=gs[member,'GP_nearest_target6'];st=ss[member,'power_law__curv_nearest_target6']
        if (gp['anchor_values']!=st['anchor_values'] or gp['anchor_indices']!=st['anchor_indices']
            or gp['anchor_indices']!=[0,1,13,14,15,16]
            or gp['target_indices']!=st['target_indices'] or gp['target_indices']!=truth['target_indices']
            or gp['recipe']!=st['recipe'] or gp['recipe']!=truth['recipe']):
            raise AssertionError('Same six known readings/panel/group required')
        result.append({'context':'resin_dev|'+member,'source':'resin','group':gp['recipe'],'condition':gp['recipe'],
            'window':'resin_native_nominal1_to50','mask':'near_tail','mean':gp['mean'],'physical':st['location'],
            'covariance':gp['covariance'],'values':truth['values']})
    ep,ey=read(exp['predictions']),read(exp['targets'])
    if ey['predictions_sha256']!=sha(exp['predictions']):raise AssertionError('Consumed expansion chain changed')
    es=indices(ep['forecasts'],lambda r:(r['member'],r['method_id']))
    for truth in ey['curves']:
        member=truth['member'];gp=es[member,'gaussian_frozen_gp'];st=es[member,'physical_full_student']
        if (gp['role']!='future_expansion' or truth['role']!='future_expansion' or st['role']!='future_expansion'
            or gp['anchor_values']!=st['anchor_values'] or gp['anchor_indices']!=st['anchor_indices']
            or gp['anchor_indices']!=[0,1,13,14,15,16]
            or gp['target_indices']!=st['target_indices'] or gp['target_indices']!=truth['target_indices']
            or gp['recipe']!=st['recipe'] or gp['recipe']!=truth['recipe']):
            raise AssertionError('Only original consumed expansion group accepted, final20forbidden')
        result.append({'context':'resin_expansion|'+member,'source':'resin','group':gp['recipe'],'condition':gp['recipe'],
            'window':'resin_native_nominal1_to50','mask':'near_tail','mean':gp['mean'],'physical':st['location'],
            'covariance':gp['covariance'],'values':truth['values']})
    xp,xy=read(xg['predictions']),read(xg['targets'])
    if xy['predictions_sha256']!=sha(xg['predictions']):raise AssertionError('Consumed XG chain changed')
    tx=indices(xy['contexts'],lambda r:r['context'])
    if len(tx)!=66 or len(xp['forecasts'])!=66:raise AssertionError('Exact66consumedXG contexts required')
    for row in xp['forecasts']:
        target=tx[row['context']];gp=row['laws']['frozen_gaussian_gp'];st=row['laws']['physical_full_student']
        if row['window'] not in ['overlap_resin_1_to_50','full_reference_support'] or row['mask'] not in ['near_tail','prefix_tail','interior_hole']:
            raise AssertionError('XG task identity changed')
        result.append({'context':'xanthan|'+row['context'],'source':'xanthan',
            'group':format(float(row['salinity_M']),'.17g'),'condition':row['preparation_group'],
            'window':row['window'],'mask':row['mask'],'mean':gp['mean'],'physical':st['location'],
            'covariance':gp['covariance'],'values':target['values']})
    if len(result)!=96 or len({r['context'] for r in result})!=96:raise AssertionError('Exact96distinct consumed contexts required')
    return result

def weights(rows):
    """Independent exact rational source/task/group/condition/run weights."""
    tree=defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:defaultdict(list))))
    for row in rows:tree[row['source']][row['window'],row['mask']][row['group']][row['condition']].append(row['context'])
    if set(tree)!={'resin','xanthan'}:raise AssertionError('Both original sources required')
    out={}
    for source,tasks in tree.items():
        if len(tasks)!=(1 if source=='resin' else 6):raise AssertionError('Wrong task count')
        for task,groups in tasks.items():
            if len(groups)!=(6 if source=='resin' else 7):raise AssertionError('Wrong source group count')
            for group,conditions in groups.items():
                for condition,members in conditions.items():
                    value=Fraction(1,2*len(tasks)*len(groups)*len(conditions)*len(members))
                    for key in members:
                        if key in out:raise AssertionError('Duplicate weighted context')
                        out[key]=value
    if sum(out.values())!=1:raise AssertionError('Exact balanced risk weights must sum1')
    return out

def mp_coefficients(row):
    with mp.workdps(80):
        m=mp.matrix([mp.mpf(float(v)) for v in row['mean']]);y=mp.matrix([mp.mpf(float(v)) for v in row['values']])
        delta_array=np.asarray(row['physical'])-np.asarray(row['mean'])
        delta=mp.matrix([mp.mpf(float(v)) for v in delta_array])
        k=mp.matrix([[mp.mpf(float(v)) for v in r] for r in row['covariance']])
        d=len(m)
        if d!=8 or k.rows!=8 or k.cols!=8:raise AssertionError('Complete8D law required')
        residual=m-y
        a=(delta.T*mp.lu_solve(k,delta))[0]/d
        b=(delta.T*mp.lu_solve(k,residual))[0]/d
        c=(d*mp.log(2*mp.pi)+mp.log(mp.det(k))+(residual.T*mp.lu_solve(k,residual))[0])/(2*d)
        return {'a':a,'b':b,'c':c,'delta':delta_array,'matrix':k,'residual':residual,'dimension':d}

def optimum(a,b,c):
    with mp.workdps(80):
        if a<0:raise AssertionError('Negative independent curvature')
        if a==0:
            if b!=0:raise AssertionError('Zero direction requires zero slope')
            raw=None;tau=mp.mpf(0)
        else:raw=-b/a;tau=max(mp.mpf(0),min(mp.mpf(1),raw))
        value=c+b*tau+a*tau*tau/2
        return {'tau_unbounded':None if raw is None else float(raw),'tau_star_0_to_1':float(tau),
            'nll_tau0':float(c),'nll_tau1':float(c+b+a/2),'nll_tau_star':float(value),
            'delta_nll_tau_star_minus_tau0':float(value-c),'gradient_at_tau0':float(b),'gradient_at_tau1':float(a+b)}

def compare_opt(check,actual,expected,name):
    for key,value in expected.items():
        if value is None:
            if actual[key] is not None:raise AssertionError('Expected null unbounded optimum')
        else:check.close(name+'_'+key,actual[key],value)

def audit(reg,capacity,check):
    rows=independent_contexts(reg);w=weights(rows)
    saved=indices(capacity['context_results'],lambda r:r['context'])
    if len(saved)!=96 or set(saved)!={r['context'] for r in rows}:raise AssertionError('Context population changed')
    numerical={};nll_checks=[]
    for row in rows:
        name=row['context'];actual=saved[name];calc=mp_coefficients(row);numerical[name]=calc
        for key in ['source','group','condition','window','mask']:
            if actual[key]!=row[key]:raise AssertionError('Source/task/group identity changed')
        check.close('context_delta',actual['delta'],calc['delta'],atol=0,rtol=0)
        check.close('exact_rational_weight',actual['balanced_global_weight'],float(w[name]),atol=2e-16,rtol=2e-16)
        for key in ['a','b','c']:check.close('mp80_context_'+key,actual[key],float(calc[key]))
        fit=optimum(calc['a'],calc['b'],calc['c']);compare_opt(check,actual,fit,'context_optimum')
        # Direct full8D likelihoods and KL at endpoints, a non-selected midpoint
        # and the context-oracle optimum. No production evaluator is called.
        with mp.workdps(80):
            k=calc['matrix'];d=calc['dimension']
            vector=mp.matrix([mp.mpf(float(v)) for v in calc['delta']])
            for tau in [mp.mpf(0),mp.mpf('0.5'),mp.mpf(1),mp.mpf(fit['tau_star_0_to_1'])]:
                rr=calc['residual']+tau*vector
                direct=(d*mp.log(2*mp.pi)+mp.log(mp.det(k))+(rr.T*mp.lu_solve(k,rr))[0])/(2*d)
                polynomial=calc['c']+tau*calc['b']+tau*tau*calc['a']/2
                check.close('full_residual_NLL_vs_quadratic',float(direct),float(polynomial),atol=2e-11,rtol=2e-11)
                # Equal-covariance log-density ratio is affine in Y. Its exact
                # expectation at q_tau therefore equals its value at m_tau.
                displacement=tau*vector
                direct_KL=(displacement.T*mp.lu_solve(k,displacement))[0]/2
                check.close('full_Gaussian_KL_vs_scalar_capacity',float(direct_KL),float(d*tau*tau*calc['a']/2),atol=2e-11,rtol=2e-11)
            nll_checks.append({'context':name,'mp_a':float(calc['a']),'mp_b':float(calc['b']),
                'mp_c':float(calc['c']),'tau_star':fit['tau_star_0_to_1'],
                'KL_tauhalf_raw8D':float(d*mp.mpf('0.25')*calc['a']/2)})
    def aggregate(selected):
        total=sum(w[row['context']] for row in selected)
        with mp.workdps(80):
            coefficients={key:mp.fsum(mp.mpf(w[row['context']].numerator)/w[row['context']].denominator
                *numerical[row['context']][key] for row in selected)/(mp.mpf(total.numerator)/total.denominator)
                for key in ['a','b','c']}
        return coefficients,total
    source_results={}
    summaries=[('global',capacity['global_summary'],rows)]
    for item in capacity['source_summaries']:
        chosen=[r for r in rows if r['source']==item['scope']]
        summaries.append(('source',item,chosen))
    for item in capacity['task_summaries']:
        source,window,mask=item['scope'].split('|')
        summaries.append(('task',item,[r for r in rows if (r['source'],r['window'],r['mask'])==(source,window,mask)]))
    for item in capacity['group_summaries']:
        source,group=item['scope'].split('|')
        summaries.append(('group',item,[r for r in rows if (r['source'],r['group'])==(source,group)]))
    if len(summaries)!=23 or len(capacity['group_summaries'])!=13 or len(capacity['task_summaries'])!=7:raise AssertionError('Expected1global2source7task13group aggregates')
    independently=[]
    for kind,item,chosen in summaries:
        if not chosen or item['contexts']!=len(chosen):raise AssertionError('Summary context count mismatch')
        coeff,total=aggregate(chosen)
        for key in ['a','b','c']:check.close('mp80_'+kind+'_'+key,item[key],float(coeff[key]))
        check.close('summary_weight_mass',item['source_weight_in_global'],float(total),atol=2e-15,rtol=2e-15)
        for row in chosen:check.close('normalized_hierarchical_weight',item['normalized_context_weights'][row['context']],float(w[row['context']]/total),atol=2e-15,rtol=2e-15)
        fit=optimum(**coeff);compare_opt(check,item,fit,kind+'_optimum')
        if kind=='source':source_results[item['scope']]=(coeff,fit)
        independently.append({'kind':kind,'scope':item['scope'],**{k:float(v) for k,v in coeff.items()},**fit})
    for cross in capacity['cross_source_tau_diagnostics']:
        train,test=cross['tau_fit_scope'],cross['evaluation_scope']
        coef,fit=source_results[test];tau=source_results[train][1]['tau_star_0_to_1']
        value=float(coef['c']+tau*coef['b']+tau*tau*coef['a']/2)
        check.close('source_fitted_tau_application',cross['tau'],tau)
        check.close('source_transfer_NLL',cross['joint_nll_per_dimension'],value)
        check.close('source_transfer_delta',cross['delta_vs_unchanged_GP'],value-float(coef['c']))
    if len(capacity['cross_source_tau_diagnostics'])!=4:raise AssertionError('Both self andoffdiagonal source applications required')
    ledger=capacity['ledger']
    if (ledger['existing_contexts']!=96 or ledger['saved_contextual_target_terms_read']!=768
        or ledger['analytic_scalar_minimizations']!=119 or ledger['new_final20_values']!=0
        or ledger['new_covariance_fits']!=0 or ledger['new_OLS_fits']!=0):
        raise AssertionError('Consumed-data/fit/final-protection ledger mismatch')
    return {'independent_aggregates':independently,'mp_contexts':nll_checks,'comparisons':check.items,
        'contexts':96,'aggregate_oracles':23,'context_oracles':96,'analytic_scalar_minimizations_checked':119,
        'original_group_count':13,'source_count':2,'target_terms_reused':768,
        'weighting':'exact rational equal2sources/task/group/preparation-condition/run',
        'scope':'capacityoracle/cross-source diagnostic on consumeddevelopment; not independent confirmation',
        'raw_CSV_opened':False,'final20_read':0,'production_imported':False,
        'new_OLS_or_covariance_fits':0,'complete_residual_preserved':True,
        'KL_note':'KL(qtau||q0)=tau²*delta C^-1delta/2; likelihood-ratio scalar is sufficient only inside this fixed-direction family'}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--registration',type=Path,required=True)
    parser.add_argument('--capacity',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--audit-registration',type=Path)
    args=parser.parse_args()
    reg=read(args.registration)
    if reg['schema']!='mean-transport-capacity-registration-v1' or reg['target_scope']!='ALREADY_CONSUMED_DEVELOPMENT_ONLY':
        raise PermissionError('Explicit root consumed-capacity authorization required')
    stamp=dt.datetime.fromisoformat(reg['actual_utc'].replace('Z','+00:00'))
    if stamp.utcoffset() is None or stamp>dt.datetime.now(dt.timezone.utc):raise PermissionError('Actual nonfuture UTC required')
    for path,pin in reg['input_pins'].items():
        if sha(path)!=pin:raise PermissionError('Root input bytes changed: '+path)
    audit_reg=args.audit_registration or args.output.with_name('INDEPENDENT_AUDIT_REGISTERED.json')
    write_new(audit_reg,{'actual_utc':utc(),'auditor_source_sha256':sha(__file__),
        'root_registration_sha256':sha(args.registration),'capacity_sha256':sha(args.capacity),
        'scope':'independent80digit fullGaussian solves/exact rationalweights; consumed targets only,0raw/finalY'})
    checks=Check()
    try:
        cap=read(args.capacity)
        if cap['registration_sha256']!=sha(args.registration) or cap['role']!='CONSUMED_DEVELOPMENT_CAPACITY_ORACLE':raise AssertionError('Capacity role/input binding differs')
        result=audit(reg,cap,checks)
        report={'status':'PASS','actual_utc':utc(),'auditor_source_sha256':sha(__file__),
            'root_registration_sha256':sha(args.registration),'audited_capacity_sha256':sha(args.capacity),
            'audit_registration_sha256':sha(audit_reg),**result}
    except Exception as error:
        write_new(args.output,{'status':'FAIL','actual_utc':utc(),'error_type':type(error).__name__,
            'error':str(error),'comparisons':checks.items,'raw_CSV_opened':False,'final20_read':0});raise
    write_new(args.output,report)
    print('PASS independent fullGaussian mean-segment capacity;96contexts119bounded scalar oracles')

if __name__=='__main__':main()

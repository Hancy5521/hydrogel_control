"""Independent fixed-df4, same-full-moments Student strong-control audit.

Uses only independent consumed-data adapters/rational weights, never production
Student score or capacity coefficient/minimizer functions. No raw/final20,
refitting, covariance adjustment or nu search; no fresh confirmation.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr
from scipy.stats import multivariate_t,t

ADAPTER_SHA='cee2202e8094dc46a07f6b2deb619ed359ef8e26c76011a56a0f17290e05bcf7'
if hashlib.sha256(Path(__file__).with_name('mean_transport_independent_audit.py').read_bytes()).hexdigest()!=ADAPTER_SHA:
    raise ImportError('Independent saved-data adapter bytes changed')
from mean_transport_independent_audit import independent_contexts,weights

METRICS=('joint_nll_per_dimension','crps','log_rmse','coverage90','width90')
KINDS=('gaussian','moment_matched_student')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def utc():return dt.datetime.now(dt.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write_new(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

class Checks:
    def __init__(self):self.rows={}
    def close(self,name,a,b,atol=3e-8,rtol=3e-8):
        a,b=np.asarray(a,float),np.asarray(b,float)
        if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():raise AssertionError(name+':shape/nonfinite')
        error=float(np.max(np.abs(a-b),initial=0))
        record=self.rows.setdefault(name,{'elements':0,'max_absolute_error':0.,'atol':atol,'rtol':rtol})
        record['elements']+=int(a.size);record['max_absolute_error']=max(record['max_absolute_error'],error)
        if not np.allclose(a,b,atol=atol,rtol=rtol):raise AssertionError(name+':error='+str(error))

def expected_scores(mean,cov,y,constant):
    m,k,y=np.asarray(mean),np.asarray(cov),np.asarray(y)
    shape=.5*k;s=np.sqrt(np.diag(shape));sd=np.sqrt(np.diag(k));error=y-m
    z=error/s
    ct=s*(z*(2*t.cdf(z,4)-1)+2*(4+z*z)*t.pdf(z,4)/3-constant)
    z0=error/sd
    cg=sd*(z0*(2*ndtr(z0)-1)+2*np.exp(-z0*z0/2)/math.sqrt(2*math.pi)-1/math.sqrt(math.pi))
    sign,det=np.linalg.slogdet(k)
    if sign!=1:raise AssertionError('Complete Gaussian C must be SPD')
    ng=(8*math.log(2*math.pi)+det+error@np.linalg.solve(k,error))/16
    nt=-float(multivariate_t.logpdf(y,loc=m,shape=shape,df=4))/8
    result={}
    for kind,crps,nll,scale,q in [('gaussian',cg,ng,sd,1.6448536269514722),
        ('moment_matched_student',ct,nt,s,float(t.ppf(.95,4)))]:
        inside=np.abs(error)<=q*scale
        result[kind]={'joint_nll_per_dimension':float(nll),'crps':float(np.mean(crps)),
            'log_rmse':float(np.sqrt(np.mean(error**2))),'coverage90':float(np.mean(inside)),
            'width90':float(np.mean(2*q*scale)),'point_crps':crps,'inside':inside,'width':2*q*scale}
    return result

def direct_CDF_CRPS(z):
    a=abs(float(z))
    constant=quad(lambda u:t.sf(u,4)**2,0,np.inf,epsabs=2e-12,epsrel=2e-12,limit=180)[0]
    # Positive finite segments avoid losing the neighbourhood of zero when
    # an observed standardized error is very large. No probability clipping.
    end=[0.];value=1.
    while value<a:end.append(value);value*=2
    if a>0:end.append(a)
    middle=sum(quad(lambda u:t.cdf(u,4)**2,lo,hi,epsabs=2e-11,epsrel=2e-11,limit=180)[0]
               for lo,hi in zip(end[:-1],end[1:]))
    tail=quad(lambda u:t.sf(u,4)**2,a,np.inf,epsabs=2e-12,epsrel=2e-12,limit=180)[0]
    return constant+middle+tail

def audit(reg,saved,check):
    contexts=independent_contexts(reg);w=weights(contexts)
    actual={row['context']:row for row in saved['context_results']}
    points={(p['context'],p['law_id'],p['target_offset']):p for p in saved['point_terms']}
    if len(actual)!=96 or len(saved['context_results'])!=96 or len(points)!=1536 or len(saved['point_terms'])!=1536:
        raise AssertionError('Exact96contexts/192laws/1536point terms required')
    constant=quad(lambda u:2*t.cdf(u,4)*t.sf(u,4),0,np.inf,epsabs=2e-12,epsrel=2e-12,limit=180)[0]
    expected={};context_map={r['context']:r for r in contexts}
    for context in contexts:
        key=context['context'];row=actual[key];law=row['law']
        for field in ['source','group','condition','window','mask']:
            if row[field]!=context[field]:raise AssertionError('Source/task/group alias changed')
        m,k=np.asarray(context['mean']),np.asarray(context['covariance'])
        if (law['df']!=4 or law['mean_identity_exact'] is not True or law['full_covariance_identity_exact'] is not True
            or law['df_is_preset_not_fitted'] is not True):raise AssertionError('Preset same-moment semantics changed')
        check.close('exact_same_mean',law['location'],m,atol=0,rtol=0)
        check.close('exact_same_full_predictive_covariance',law['predictive_covariance'],k,atol=0,rtol=0)
        check.close('shape_is_half_complete_covariance',law['scale_matrix'],.5*k,atol=0,rtol=0)
        check.close('full_Student_covariance_is_twice_shape',2*np.asarray(law['scale_matrix']),k,atol=0,rtol=0)
        check.close('independent_risk_weight',row['balanced_global_weight'],float(w[key]),atol=2e-16,rtol=2e-16)
        metrics=expected_scores(m,k,context['values'],constant);expected[key]=metrics
        for kind in KINDS:
            for metric in METRICS:check.close('independent_'+kind+'_'+metric,row[kind][metric],metrics[kind][metric])
            for index in range(8):
                point=points[key,kind,index]
                y=context['values'][index];err=y-m[index]
                for name,value in [('truth',y),('location',m[index]),('squared_error',err*err),
                    ('crps',metrics[kind]['point_crps'][index]),('width90',metrics[kind]['width'][index])]:
                    check.close('independent_point_'+name,point[name],value)
                if point['coverage90']!=bool(metrics[kind]['inside'][index]):raise AssertionError('Point coverage mismatch')
                if kind=='gaussian':check.close('Gaussian_predictive_SD',point['predictive_sd'],math.sqrt(k[index,index]))
                else:
                    if point['df']!=4:raise AssertionError('Point df changed')
                    check.close('Student_scale_not_SD',point['t_scale'],math.sqrt(k[index,index]/2))
                    check.close('same_predictive_SD',point['t_sd'],math.sqrt(k[index,index]))
                    original=points[key,'gaussian',index]
                    if point['squared_error']!=original['squared_error']:raise AssertionError('Same mean must imply exact same squared error')
        if row['moment_matched_student']['log_rmse']!=row['gaussian']['log_rmse']:
            raise AssertionError('RMSE identity is structural, not an empirical gain')
        if metrics['moment_matched_student']['coverage90']>metrics['gaussian']['coverage90']:
            raise AssertionError('Same-moment t4 central90 interval must be narrower than Gaussian90 interval')
        for metric in METRICS:check.close('per_context_delta_'+metric,row['delta_student_minus_gaussian'][metric],metrics['moment_matched_student'][metric]-metrics['gaussian'][metric])
    summaries=[saved['global_summary']]+saved['source_summaries']+saved['task_summaries']+saved['group_summaries']
    if len(summaries)!=23:raise AssertionError('Expected1global2sources7tasks13groups')
    independent=[]
    for row in summaries:
        scope=row['scope']
        if scope=='equal_sources':selected=contexts
        elif scope in ['resin','xanthan']:selected=[r for r in contexts if r['source']==scope]
        else:
            parts=scope.split('|')
            if len(parts)==2:selected=[r for r in contexts if (r['source'],r['group'])==tuple(parts)]
            elif len(parts)==3:selected=[r for r in contexts if (r['source'],r['window'],r['mask'])==tuple(parts)]
            else:raise AssertionError('Unregistered summary scope')
        total=sum(w[r['context']] for r in selected)
        if len(selected)!=row['contexts'] or total<=0:raise AssertionError('Summary population changed')
        check.close('summary_weight',row['weight_in_global'],float(total),atol=2e-15,rtol=2e-15)
        computed={kind:{metric:math.fsum(float(w[r['context']]/total)*expected[r['context']][kind][metric] for r in selected)
            for metric in METRICS} for kind in KINDS}
        for kind in KINDS:
            for metric,value in computed[kind].items():check.close('summary_'+kind+'_'+metric,row[kind][metric],value)
        delta={metric:computed[KINDS[1]][metric]-computed[KINDS[0]][metric] for metric in METRICS}
        for metric,value in delta.items():check.close('summary_delta_'+metric,row['delta_student_minus_gaussian'][metric],value)
        independent.append({'scope':scope,**computed,'delta':delta})
    direct=[]
    tasks=sorted({(r['source'],r['window'],r['mask']) for r in contexts})
    for task in tasks:
        selected=sorted([r for r in contexts if (r['source'],r['window'],r['mask'])==task],key=lambda r:r['context'])
        for pick,offset in zip([0,len(selected)//3,2*len(selected)//3,len(selected)-1],[0,2,5,7]):
            context=selected[pick];key=context['context'];s=math.sqrt(context['covariance'][offset][offset]/2)
            z=(context['values'][offset]-context['mean'][offset])/s
            value=s*direct_CDF_CRPS(z)
            check.close('direct_Student_CDF_integral_CRPS',points[key,KINDS[1],offset]['crps'],value,atol=3e-8,rtol=3e-8)
            direct.append({'context':key,'target_offset':offset,'standardized_error':float(z),'directCDF_CRPS':value})
    ledger=saved['ledger']
    if (ledger['existing_contexts']!=96 or ledger['law_evaluations']!=192 or ledger['point_score_terms']!=1536
        or ledger['df_candidates']!=1 or ledger['df_search'] or ledger['new_OLS_or_kernel_or_scale_fits']!=0
        or ledger['new_scalar_minimizations']!=0 or ledger['final20_accessed']):raise AssertionError('Fixed-control role/search/fit ledger changed')
    return {'contexts':96,'complete_laws':192,'point_terms':1536,'independent_summaries':independent,
        'direct_CDF_integrals':direct,'Student_Gini_constant_CDF_integral':constant,
        'central90_Student_to_Gaussian_width_ratio':float(t.ppf(.95,4)/math.sqrt(2)/1.6448536269514722),
        'mean_and_full_covariance_identity':True,'RMSE_identity_structural':True,
        'scope':'classic fixednu4 strong control; consumeddevelopment only, no confirmation',
        'production_Student_or_capacity_numeric_functions_called':False,
        'new_fits':0,'new_raw_or_final20_values':0,'comparisons':check.rows}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--registration',type=Path,required=True)
    parser.add_argument('--results',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();reg=read(args.registration)
    if reg['schema']!='moment-matched-student-registration-v1' or reg['target_scope']!='ALREADY_CONSUMED_DEVELOPMENT_ONLY':
        raise PermissionError('Explicit root fixednu4 consumed-control registration required')
    stamp=dt.datetime.fromisoformat(reg['actual_utc'].replace('Z','+00:00'))
    if stamp.utcoffset() is None or stamp>dt.datetime.now(dt.timezone.utc):raise PermissionError('Actual nonfuture UTC required')
    for path,pin in reg['input_pins'].items():
        if sha(path)!=pin:raise PermissionError('Consumed input bytes changed')
    audit_reg=args.output.with_name('INDEPENDENT_AUDIT_REGISTERED.json')
    write_new(audit_reg,{'actual_utc':utc(),'source_sha256':sha(__file__),'independent_adapter_sha256':ADAPTER_SHA,
        'root_registration_sha256':sha(args.registration),'results_sha256':sha(args.results),
        'scope':'independentStudentPDF/CDF/no productioncalls/all96 consumedcontexts/0nu search'})
    check=Checks()
    try:
        saved=read(args.results)
        if saved['registration_sha256']!=sha(args.registration) or saved['df']!=4:raise AssertionError('Result source/df binding changed')
        report={'status':'PASS','actual_utc':utc(),'auditor_source_sha256':sha(__file__),
            'audited_results_sha256':sha(args.results),'audit_registration_sha256':sha(audit_reg),**audit(reg,saved,check)}
    except Exception as error:
        write_new(args.output,{'status':'FAIL','actual_utc':utc(),'error_type':type(error).__name__,'error':str(error),
            'comparisons':check.rows,'new_raw_or_final20_values':0});raise
    write_new(args.output,report);print('PASS independent same-moment Student4 control;96contexts1536points;0fit/nusearch')

if __name__=='__main__':main()

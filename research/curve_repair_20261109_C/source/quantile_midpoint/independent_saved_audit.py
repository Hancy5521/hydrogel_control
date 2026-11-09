"""Independent saved-forecast audit of one fixed midpoint development study.

No production imports/fits/tau search/raw reader. Previously consumed240targets
are explicitly development reuse; historical B results and final20 are intact.
Midpoint CRPS uses normal-domain energy quadrature; density-mixture CRPS uses
cross-distribution energy, not production's CDF integral.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path,PureWindowsPath
import sys
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import gammaln,logsumexp,ndtr,ndtri,ndtri_exp
from scipy.stats import multivariate_normal,norm,t

GATE_HELPER_SHA="a3f0a0754e8ab66c68f92cb13098cb6a2bdd5993d791cea2daf1da437b676f5f"
helper=Path(__file__).with_name("independent_gate_v2.py")
if hashlib.sha256(helper.read_bytes()).hexdigest()!=GATE_HELPER_SHA:
    raise ImportError("Qualified independent normal-domain oracle changed")
from independent_gate_v2 import independent_h,independent_inverse,mp_h_and_jacobian,normal_domain_crps,t4_algebraic

METHODS=("gaussian_gp","physical_full_student","student_gp_copula","half_quantile_transport","half_density_mixture")
METRICS=("joint_nll_per_dim","mean_crps","log_rmse","coverage90","width90")

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):return json.loads(Path(path).read_text(encoding="utf-8-sig"))
def write_new(path,payload):
    with Path(path).open("x",encoding="utf-8") as f:
        json.dump(payload,f,ensure_ascii=False,indent=2,allow_nan=False);f.write("\n")
def utc():return dt.datetime.now(dt.timezone.utc).isoformat()

class Checks:
    def __init__(self):self.groups={}
    def close(self,label,a,b,atol=3e-8,rtol=3e-8):
        a,b=np.asarray(a,float),np.asarray(b,float)
        if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():raise AssertionError(label+":shape/nonfinite")
        error=float(np.max(np.abs(a-b),initial=0))
        row=self.groups.setdefault(label,{"elements":0,"max_absolute_error":0.,"atol":atol,"rtol":rtol})
        row["elements"]+=int(a.size);row["max_absolute_error"]=max(row["max_absolute_error"],error)
        if not np.allclose(a,b,atol=atol,rtol=rtol):raise AssertionError(label+":error="+str(error))

def t_joint_logpdf(y,m,shape):
    y,m,k=np.asarray(y),np.asarray(m),np.asarray(shape)
    sign,det=np.linalg.slogdet(k)
    if sign!=1:raise AssertionError("Expected Student shape SPD")
    d=len(y);e=y-m;quadform=float(e@np.linalg.solve(k,e))
    return float(gammaln((4+d)/2)-gammaln(2)-d/2*math.log(4*math.pi)-det/2-(4+d)/2*math.log1p(quadform/4))

def student_crps(y,m,s,c):
    z=(y-m)/s
    absolute=s*(z*(2*t.cdf(z,4)-1)+2*(4+z*z)*t.pdf(z,4)/3)
    return absolute-s*c

def gaussian_crps(y,m,s):
    z=(y-m)/s
    return s*(z*(2*ndtr(z)-1)+2*norm.pdf(z)-1/math.sqrt(math.pi))

def student_copula_logpdf(y,m,s,r):
    zt=(y-m)/s
    latent=np.empty_like(zt);left=zt<=0
    latent[left]=ndtri_exp(t.logcdf(zt[left],4))
    latent[~left]=-ndtri_exp(t.logsf(zt[~left],4))
    return float(multivariate_normal.logpdf(latent,mean=np.zeros(len(y)),cov=r)
                 -np.sum(norm.logpdf(latent))+np.sum(t.logpdf(zt,4)-np.log(s)))

def cross_absolute(mg,sg,ms,ss):
    def integrand(z):
        student=ms+ss*float(t4_algebraic(z))
        difference=student-mg
        standardized=difference/sg
        conditional=2*sg*norm.pdf(standardized)+difference*(2*ndtr(standardized)-1)
        return conditional*norm.pdf(z)
    return quad(integrand,-12,12,epsabs=2e-11,epsrel=2e-11,limit=200)[0]

def mixture_quantiles(mg,sg,ms,ss):
    result=np.empty((3,len(mg)))
    for k,p in enumerate([.05,.5,.95]):
        for j in range(len(mg)):
            lo=min(mg[j]+sg[j]*ndtri(p),ms[j]+ss[j]*t.ppf(p,4))-1e-8
            hi=max(mg[j]+sg[j]*ndtri(p),ms[j]+ss[j]*t.ppf(p,4))+1e-8
            result[k,j]=brentq(lambda y:.5*(ndtr((y-mg[j])/sg[j])+t.cdf((y-ms[j])/ss[j],4))-p,
                                lo,hi,xtol=4e-14,rtol=1e-14)
    return result

def all_expected(f,y,c,checks):
    g,s,co,q=(f[key] for key in ['gaussian','student','student_gp_copula','midpoint'])
    mg,cg,ms,cs=np.asarray(g['mean']),np.asarray(g['covariance']),np.asarray(s['location']),np.asarray(s['scale_matrix'])
    sg,ss=np.sqrt(np.diag(cg)),np.sqrt(np.diag(cs));r=np.asarray(co['correlation'])
    checks.close('same_full_GP_latent_R',r,cg/np.outer(sg,sg),atol=3e-12,rtol=3e-12)
    for name,value in [('gaussian_location',mg),('gaussian_sd',sg),('student_location',ms),('student_scale',ss),('correlation',r)]:
        checks.close('copied_midpoint_'+name,q[name],value,atol=0,rtol=0)
    if q['tau']!=.5 or q['df']!=4 or q['score_method']!='analytic_crps_gini_identity_v2':raise AssertionError('Only fixed half analytic-v2 law permitted')
    lg=float(multivariate_normal.logpdf(y,mean=mg,cov=cg));ls=t_joint_logpdf(y,ms,cs)
    lc=student_copula_logpdf(y,ms,ss,r)
    z=np.array([independent_inverse(yj,*par) for yj,par in zip(y,zip(mg,sg,ms,ss))])
    logjac=np.array([mp_h_and_jacobian(zj,*par)[2] for zj,par in zip(z,zip(mg,sg,ms,ss))])
    lq=float(multivariate_normal.logpdf(z,mean=np.zeros(8),cov=r)-logjac.sum())
    cg_score=gaussian_crps(y,mg,sg);cs_score=student_crps(y,ms,ss,c)
    cq=np.array([normal_domain_crps(yj,*par,c,12)[0] for yj,par in zip(y,zip(mg,sg,ms,ss))])
    cross=np.array([cross_absolute(*par) for par in zip(mg,sg,ms,ss)])
    cm=.5*(cg_score+cs_score)+.25*(sg/math.sqrt(math.pi)+ss*c-cross)
    qg=mg[None,:]+sg[None,:]*ndtri([.05,.5,.95])[:,None]
    qs=ms[None,:]+ss[None,:]*t.ppf([.05,.5,.95],4)[:,None]
    qq=.5*(qg+qs);qm=mixture_quantiles(mg,sg,ms,ss)
    checks.close('copied_midpoint_quantiles',f['midpoint_quantiles05_50_95'],qq,atol=3e-10,rtol=3e-10)
    if f['density_mixture_weights']!=[.5,.5] or f['density_mixture_parents']!=['gaussian_gp','student_gp_copula']:
        raise AssertionError('Mixture must use same endpoints and fixed equal weights')
    entries=[(lg,cg_score,mg,qg),(ls,cs_score,ms,qs),(lc,cs_score,ms,qs),
        (lq,cq,(mg+ms)/2,qq),(float(logsumexp([lg,lc]))-math.log(2),cm,(mg+ms)/2,qm)]
    results={}
    for method,(density,crps,mean,quantiles) in zip(METHODS,entries):
        results[method]={'joint_nll_per_dim':-density/8,'mean_crps':float(np.mean(crps)),
            'pointwise_crps':crps,'log_rmse':float(np.sqrt(np.mean((y-mean)**2))),
            'coverage90':float(np.mean((y>=quantiles[0])&(y<=quantiles[2]))),
            'width90':float(np.mean(quantiles[2]-quantiles[0]))}
    return results

def audit(stage,checks):
    root=stage/'quantile_midpoint';out=root/'development_once'
    reg=load(root/'DEVELOPMENT_REGISTRATION.json')
    if reg['schema']!='fixed-midpoint-reused-development-v1' or reg['tau']!=.5 or reg['fresh_confirmation'] or reg['final_runs_accessed']!=0:
        raise PermissionError('Explicit reused-development fixed-half scope required')
    # Root's original absolute paths are resolved by their stage-relative
    # landmark solely for byte-identical relocated reproduction.
    landmarks={'quantile_midpoint','resin_dev_once','student_dev_once','transport_dev_once','expansion_once'}
    for old,digest in reg['pins'].items():
        parts=PureWindowsPath(old).parts
        pos=next((i for i,part in enumerate(parts) if part in landmarks),None)
        if pos is None or sha(stage.joinpath(*parts[pos:]))!=digest:
            raise PermissionError('Registered immutable source/input bytes changed: '+old)
    pred,scores=load(out/'PREDICTIONS.json'),load(out/'SCORES.json')
    regsha=sha(root/'DEVELOPMENT_REGISTRATION.json')
    if (pred['registration_sha256']!=regsha or scores['registration_sha256']!=regsha
        or scores['predictions_sha256']!=sha(out/'PREDICTIONS.json') or pred['tau_candidates']!=[.5]
        or pred['final_runs_accessed']!=0 or pred['new_fits']!=0):
        raise AssertionError('Saved study byte chain or fixed candidate changed')
    parents_original=load(stage/'resin_dev_once/predict/PREDICTIONS.json')['forecasts']
    parents_student=load(stage/'student_dev_once/predict/PREDICTIONS.json')['forecasts']
    parents_copula=load(stage/'transport_dev_once/predict/PREDICTIONS.json')['forecasts']
    parents_expansion=load(stage/'expansion_once/predict/PREDICTIONS.json')['forecasts']
    def index(rows,method):return {r['member']:r for r in rows if r['method_id']==method}
    groups=[('original_development',index(parents_original,'GP_nearest_target6'),index(parents_student,'power_law__GP_nearest_target6'),index(parents_copula,'frozen_gp_posterior')),
        ('former_expansion_now_development_for_new_method',index(parents_expansion,'gaussian_frozen_gp'),index(parents_expansion,'physical_full_student'),index(parents_expansion,'copula_frozen_gp_posterior'))]
    actual_parents={}
    for label,gs,ss,cs in groups:
        if len(gs)!=15 or set(gs)!=set(ss) or set(gs)!=set(cs):raise AssertionError('Original two groups of15must match')
        for member in gs:
            if member in actual_parents:raise AssertionError('Parent cohorts overlap')
            actual_parents[member]=(label,gs[member],ss[member],cs[member])
    truths={}
    for path in ['resin_dev_once/reveal/TARGETS.json','expansion_once/reveal/TARGETS.json']:
        for row in load(stage/path)['curves']:
            if row['member'] in truths:raise AssertionError('Repeated target run alias')
            truths[row['member']]=row
    if len(pred['forecasts'])!=30 or len(truths)!=30 or set(truths)!=set(actual_parents):raise AssertionError('Exact30curves240oldtargets required')
    actual={(r['member'],r['method']):r for r in scores['rows']}
    if len(scores['rows'])!=150 or set(actual)!={(m,method) for m in truths for method in METHODS}:raise AssertionError('Exact150unique score rows required')
    constant=quad(lambda a:2*t.cdf(a,4)*t.sf(a,4),0,np.inf,epsabs=2e-12,epsrel=2e-12,limit=180)[0]
    expected={}
    for f in pred['forecasts']:
        member=f['member'];label,g,s,co=actual_parents[member]
        if (f['reuse_partition']!=label or f['recipe']!=g['recipe'] or f['run_id']!=g['run_id']
            or f['anchor_indices']!=[0,1,13,14,15,16] or f['target_indices']!=list(range(17,25))):
            raise AssertionError('Saved grouping or native mask changed')
        if f['gaussian']!={'mean':g['mean'],'covariance':g['covariance']} or f['student']!={'location':s['location'],'scale_matrix':s['scale_matrix'],'df':s['df']}:
            raise AssertionError('Endpoint parameters not copied from frozen parent')
        if any(f['student_gp_copula'][key]!=co[key] for key in ['location','scale','correlation','df']):raise AssertionError('Student copula parent altered')
        truth=truths[member]
        if truth['recipe']!=f['recipe'] or truth['target_indices']!=f['target_indices']:raise AssertionError('Consumed target identity changed')
        computed=all_expected(f,np.asarray(truth['values']),constant,checks)
        for method,result in computed.items():
            key=member,method;expected[key]=result
            for metric in (*METRICS,'pointwise_crps'):
                checks.close('independent_'+metric,actual[key][metric],result[metric],atol=3e-8,rtol=3e-8)
        checks.close('midpoint_mixture_equal_mean_RMSE',actual[member,'half_quantile_transport']['log_rmse'],actual[member,'half_density_mixture']['log_rmse'],atol=0,rtol=0)
    recipes=sorted({row['recipe'] for row in pred['forecasts']})
    recipe_actual={(r['recipe'],r['method']):r for r in scores['by_recipe']}
    macro_actual={r['method']:r for r in scores['macro']}
    if len(recipes)!=6 or len(recipe_actual)!=30 or len(macro_actual)!=5:raise AssertionError('Expected six recipes/five methods')
    recipe_expected={};macro_expected={}
    for method in METHODS:
        for recipe in recipes:
            chosen=[r for (member,m),r in expected.items() if m==method and truths[member]['recipe']==recipe]
            if len(chosen)!=5:raise AssertionError('Exactly5originalruns per recipe')
            recipe_expected[recipe,method]={k:float(np.mean([r[k] for r in chosen])) for k in METRICS}
            for metric,value in recipe_expected[recipe,method].items():checks.close('recipe_'+metric,recipe_actual[recipe,method][metric],value)
        macro_expected[method]={k:float(np.mean([recipe_expected[recipe,method][k] for recipe in recipes])) for k in METRICS}
        for metric,value in macro_expected[method].items():checks.close('macro_'+metric,macro_actual[method][metric],value)
    by_partition={}
    for label,_,_,_ in groups:
        names={truths[member]['recipe'] for member,parent in actual_parents.items() if parent[0]==label}
        by_partition[label]={method:{k:float(np.mean([recipe_expected[recipe,method][k] for recipe in names])) for k in METRICS} for method in METHODS}
    return {'audited_predictions_sha256':sha(out/'PREDICTIONS.json'),'audited_scores_sha256':sha(out/'SCORES.json'),
        'registration_sha256':regsha,'curves':30,'methods':5,'score_rows':150,'pointwise_CRPS_compared':1200,
        'groups':6,'independent_macro':macro_expected,'independent_macro_by_reuse_partition':by_partition,
        'comparisons':checks.groups,'new_fits':0,'raw_CSV_opened':False,'final20_target_access':0,
        'targets_scope':'240already_consumed values, development reuse only; historical B results unchanged',
        'expected_imports_production':False,'midpoint_expected':'normal-domain energy, cubic t4 inverse,mp160Jacobian',
        'density_mixture_expected':'cross-distribution absolute energy, not CDF integration',
        'fresh_confirmation':False}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',type=Path,default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();root=args.stage/'quantile_midpoint'
    output=args.output or root/'development_once/INDEPENDENT_AUDIT.json'
    write_new(root/'INDEPENDENT_SAVED_AUDIT_REGISTERED.json',{'actual_utc':utc(),'source_sha256':sha(__file__),
        'oracle_source_sha256':GATE_HELPER_SHA,'root_development_registration_sha256':sha(root/'DEVELOPMENT_REGISTRATION.json'),
        'predictions_sha256':sha(root/'development_once/PREDICTIONS.json'),'scores_sha256':sha(root/'development_once/SCORES.json'),
        'scope':'root-authorized independent saved240target audit; no fit/new source/final20/tau search'})
    checks=Checks()
    try:report={'status':'PASS','actual_utc':utc(),'auditor_source_sha256':sha(__file__),**audit(args.stage,checks)}
    except Exception as error:
        write_new(output,{'status':'FAIL','actual_utc':utc(),'error_type':type(error).__name__,'error':str(error),
            'comparisons':checks.groups,'raw_CSV_opened':False,'final20_target_access':0});raise
    write_new(output,report)
    print('PASS independent saved midpoint study:150rows,6reused-development recipes;no new fit/confirmation')

if __name__=='__main__':main()

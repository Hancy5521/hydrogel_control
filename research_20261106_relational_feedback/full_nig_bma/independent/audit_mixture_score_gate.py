"""Independent four native mixture scores and saved-law quantile/PIT/density unit audit."""
import argparse
from datetime import datetime,timezone
import hashlib,json,math
from pathlib import Path
import traceback,warnings
from scipy.integrate import IntegrationWarning
import scoring_math_reference as ref


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def require(x,m):
    if not x:raise AssertionError(m)


class Checks:
    def __init__(self):self.counts={};self.errors={}
    def exact(self,k,a,b):
        require(type(a)==type(b) and a==b,k+': exact mismatch');self.counts[k]=self.counts.get(k,0)+1
    def number(self,k,a,b,atol=2e-10,rtol=0.):
        require(math.isfinite(a) and math.isfinite(b),k+': nonfinite');error=abs(a-b)
        require(error<=atol+rtol*abs(b),'%s: %.17g vs %.17g error %.17g'%(k,a,b,error))
        self.counts[k]=self.counts.get(k,0)+1;self.errors[k]=max(self.errors.get(k,0),error)


def audit(plan,c):
    for name,digest in plan['independent_sources'].items():require(sha(Path(__file__).parent/name)==digest,'independent source changed')
    loaded={}
    for name,pin in plan['inputs'].items():
        require(sha(pin['file'])==pin['sha256'],'pinned input changed:'+name);loaded[name]=read(pin['file'])
    proposal=loaded['fixed_proposal'];result=loaded['producer_result']
    c.exact('result_plan',result['plan_sha256'],plan['inputs']['producer_plan']['sha256'])
    for name,pin in loaded['producer_plan']['inputs'].items():
        require(sha(pin['file'])==pin['sha256'],'producer dependency changed:'+name)
    require(loaded['old_fullgate_independent']['status']=='PASS','old fullgate actually passed')
    records={row['id']:row for row in result['law_records']}
    require(len(records)==len(result['law_records'])==2 and set(records)=={'initial','after_positive'},'two oldsavedlaws only')
    checked=[]
    for name,probe in [('initial',None),('after_positive',1.)]:
        saved=records[name];law=ref.law_from_fixed_proposal(proposal,probe);actual=saved['law']
        for key in ('family','coordinate','lambda'):c.exact('law_'+key,actual[key],law[key])
        c.number('law_s0',actual['native_first_scale_s0'],law['native_first_scale_s0'],0,0)
        for a,b in zip(actual['log_weights'],law['log_weights']):c.number('law_logweight',a,b,2e-12,0.)
        require(len(actual['components'])==3,'three fixedcomponents')
        for a,b in zip(actual['components'],law['components']):
            c.exact('component_order',a['id'],b['id'])
            for key in ('location','scale','df'):c.number('component_'+key,a[key],b[key],2e-10,2e-10)
        quantiles={float(row['probability']):row['native_quantile'] for row in saved['quantiles']}
        require(len(quantiles)==len(saved['quantiles'])==7 and set(quantiles)==set(ref.PROBABILITIES),'sevenfixedquantiles')
        independent_quantiles={}
        for p in ref.PROBABILITIES:
            own=ref.quantile(law,p);independent_quantiles[p]=own
            c.number('native_quantile',quantiles[p],own,2e-7,2e-9)
            c.number('quantile_native_CDF_root',ref.cdf(law,quantiles[p]),p,2e-10,0.)
        scores={float(row['native_target']):row['score'] for row in saved['scores']}
        require(len(scores)==len(saved['scores'])==2 and set(scores)=={-1.,1.},'nativehypotheticaltargets±1only')
        for y in (-1.,1.):
            score=scores[y];crps,error=ref.direct_crps(law,y);pit=ref.cdf(law,y);logdensity=ref.log_density(law,y)
            c.number('native_mixture_CRPS_direct_CDF',score['CRPS_native'],crps,2e-4,2e-5)
            for key in ('PIT_left','PIT_right','PIT_midpoint'):c.number('mixture_'+key,score[key],pit,2e-10,0.)
            c.number('mixture_native_logdensity',score['log_density_native'],logdensity,2e-9,2e-11)
            c.number('mixture_native_NLL',score['negative_log_density_native'],-logdensity,2e-9,2e-11)
            median=independent_quantiles[.5]
            c.number('mixture_native_median',score['predictive_median_native'],median,2e-7,2e-9)
            c.number('mixture_native_squared_error',score['squared_error_native'],(y-median)**2,2e-7,2e-9)
            for level in (.8,.9,.95):
                a=(1-level)/2;b=(1+level)/2
                lo=independent_quantiles[min(ref.PROBABILITIES,key=lambda p:abs(p-a))]
                hi=independent_quantiles[min(ref.PROBABILITIES,key=lambda p:abs(p-b))]
                interval=score['intervals'][str(level)]
                c.number('interval_lower',interval['lower'],lo,2e-7,2e-9)
                c.number('interval_upper',interval['upper'],hi,2e-7,2e-9)
                c.number('interval_width',interval['width'],hi-lo,2e-7,2e-9)
                c.exact('interval_event',interval['covered'],lo<=y<=hi)
            checked.append({'law':name,'native_target':y,'independent_native_CRPS':crps,
                            'native_CDF_quad_error_estimate':error,'independent_PIT':pit,
                            'independent_native_logdensity':logdensity})
    c.exact('no_new_material_or_VOI',result['new_material_measurements_fits_or_joint_VOI_reintegrals'],0)
    return {'status':'PASS','four_native_score_checks':checked,'native_quantile_roots':14,
            'mixture_PIT_events':4,'density_and_NLL_events':4,'coverage_events':12,
            'independent_route':'nativeYentireCDFsquare/survivalCDFsquare; nativeYroot; stats.t.logpdf+commonJac',
            'component_score_averages_or_disagreement_subtractions_in_reference':False,
            'new_models_material_labels_fit_or_fullVOI_2D':0,
            'scope':'necessary unit check of newly added mixture scoring family using two oldfixedlaws, not new scientific evidence'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--approval',required=True);p.add_argument('--output-dir',required=True)
    a=p.parse_args();plan=read(a.plan);approval=read(a.approval)
    require(approval.get('approved') is True and approval['scope']=='INDEPENDENT_FOUR_NATIVE_MIXTURE_SCORES_ONCE'
            and approval['plan_sha256']==sha(a.plan) and approval['script_sha256']==sha(__file__),'exact root independentapproval')
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=False);start=datetime.now(timezone.utc).isoformat();c=Checks()
    try:
        with warnings.catch_warnings():warnings.simplefilter('error',IntegrationWarning);result=audit(plan,c)
    except Exception as problem:result={'status':'FAIL','type':type(problem).__name__,'message':str(problem),'traceback':traceback.format_exc()}
    result.update({'actual_start_utc':start,'actual_end_utc':datetime.now(timezone.utc).isoformat(),'script_sha256':sha(__file__),
                   'plan_sha256':sha(a.plan),'approval_sha256':sha(a.approval),'comparison_counts':c.counts,'max_absolute_errors':c.errors})
    f=out/'INDEPENDENT_FOUR_MIXTURE_SCORES_AUDIT.json'
    with f.open('x',encoding='utf-8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({'status':result['status'],'file':str(f),'sha256':sha(f)}))
    return 0 if result['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())

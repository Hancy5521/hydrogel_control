"""Independent all2592 saved reported-feedback scores, no producer imports or replay."""
import argparse
from datetime import datetime,timezone
from fractions import Fraction
import hashlib,importlib.util,json,math
from pathlib import Path
from types import SimpleNamespace
import traceback,warnings
from scipy import special
from scipy.integrate import quad,IntegrationWarning
import saved_state_reference as ref

METHODS=('hurdle_updated','hurdle_stale','pooled_yj_updated','gpr_fixed_recipe','forest_all_original_leaf_cdf')
COUNTS=(0,8,16,32)
LEVELS=('0.8','0.9','0.95')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def require(x,m):
 if not x:raise AssertionError(m)

def inverse_yj(u):
 return u*(1+u/4)if u>=0 else -math.expm1((2/3)*math.log1p(-1.5*u))


class Checks:
 def __init__(self):self.counts={};self.errors={}
 def number(self,n,a,b,atol=2e-6,rtol=2e-8):
  require(math.isfinite(a)and math.isfinite(b),n+' nonfinite');e=abs(a-b);require(e<=atol+rtol*abs(b),f'{n}: {a} vs {b},error={e}')
  self.counts[n]=self.counts.get(n,0)+1;self.errors[n]=max(self.errors.get(n,0),e)
 def exact(self,n,a,b):
  require(type(a)==type(b)and a==b,n+' exact mismatch');self.counts[n]=self.counts.get(n,0)+1


def empirical(y,law):
 x=law['support_native'];w=[Fraction(t)for t in law['weight_fractions']];require(len(x)==len(w)and sum(w,Fraction(0))==1 and all(v>=0 for v in w),'finite empirical simplex')
 active=[(a,b)for a,b in zip(x,w)if b>0]
 first=math.fsum(float(b)*abs(a-y)for a,b in active)
 spread=math.fsum(float(wi*wj)*abs(xi-xj)for i,(xi,wi)in enumerate(active)for xj,wj in active[:i])
 left=sum((b for a,b in active if a<y),Fraction(0));right=sum((b for a,b in active if a<=y),Fraction(0))
 ordered=sorted(active)
 def quantile(p):
  mass=Fraction(0)
  for a,b in ordered:
   mass+=b
   if mass>=p:return a
  raise AssertionError('Empiricalquantileunbracketed')
 median=quantile(Fraction(1,2));intervals={}
 for level in LEVELS:
  p=Fraction(level);lo=quantile((1-p)/2);hi=quantile((1+p)/2)
  intervals[level]={'lower':lo,'upper':hi,'width':hi-lo,'covered':lo<=y<=hi}
 return {'CRPS_native':first-spread,'predictive_median_native':median,'squared_error_native':(y-median)**2,
         'PIT_left':float(left),'PIT_right':float(right),'PIT_midpoint':float((left+right)/2),
         'intervals':intervals,'log_density_native':None}


def evaluate(y,law,N,own):
 if law['family']=='weighted_empirical':return empirical(y,law)
 crps,error,method=N.native_cdf_score(y,law,own)
 p=N.independent_probability_score(y,law,own)
 return {'CRPS_native':crps,'predictive_median_native':p['point_forecast'],'squared_error_native':p['squared_error_native'],
         'PIT_left':p['PIT'],'PIT_right':p['PIT'],'PIT_midpoint':p['PIT'],'intervals':p['intervals'],
         'log_density_native':p['log_density_native'],'quad_error_estimate_native':error,'independent_route':method}


def score_compare(saved,own,c):
 for k in ('CRPS_native','squared_error_native'):c.number(k,saved[k],own[k])
 c.number('predictive_median_native',saved['predictive_median_native'],own['predictive_median_native'],2e-7,2e-9)
 for k in ('PIT_left','PIT_right','PIT_midpoint'):c.number(k,saved[k],own[k],2e-12,2e-12)
 if own['log_density_native']is None:c.exact('discrete_no_density',saved['log_density_native'],None)
 else:c.number('native_log_density',saved['log_density_native'],own['log_density_native'],2e-9,2e-11)
 for level in LEVELS:
  for k in ('lower','upper','width'):c.number('native_interval_'+k,saved['intervals'][level][k],own['intervals'][level][k],2e-7,2e-9)
  c.exact('exact_interval_event',saved['intervals'][level]['covered'],own['intervals'][level]['covered'])


def summarize(rows,method,count):
 require(len(rows)==128,'128groupspermethodbudget')
 mean=lambda seq:math.fsum(seq)/128
 pit=sorted(r['PIT_midpoint']for r in rows)
 return {'terminal_formula_groups':128,'new_native_cell_budget':3*count,
         'mean_CRPS_native':mean([r['CRPS_native']for r in rows]),
         'RMSE_predictive_median_native':math.sqrt(mean([r['squared_error_native']for r in rows])),
         'coverage':{l:sum(r['intervals'][l]['covered']for r in rows)/128 for l in LEVELS},
         'mean_interval_width_native':{l:mean([r['intervals'][l]['width']for r in rows])for l in LEVELS},
         'PIT_ECDF_deviation_descriptive':None if method==METHODS[-1]else max(max((i+1)/128-v,v-i/128)for i,v in enumerate(pit)),
         'mean_PIT_left':mean([r['PIT_left']for r in rows]),'mean_PIT_right':mean([r['PIT_right']for r in rows]),
         'PIT_discrete_midpoint_not_uniform':method==METHODS[-1]}


def check_summary(saved,own,c):
 for k in ('terminal_formula_groups','new_native_cell_budget','PIT_discrete_midpoint_not_uniform'):c.exact('summary_'+k,saved[k],own[k])
 for k in ('mean_CRPS_native','RMSE_predictive_median_native','mean_PIT_left','mean_PIT_right'):c.number('summary_'+k,saved[k],own[k])
 k='PIT_ECDF_deviation_descriptive'
 if own[k]is None:c.exact('discrete_no_PITuniformgate',saved[k],None)
 else:c.number('summary_PIT_ECDF_D',saved[k],own[k],2e-12,2e-12)
 for level in LEVELS:
  c.number('summary_exact_coverage',saved['coverage'][level],own['coverage'][level],0,0)
  c.number('summary_interval_width',saved['mean_interval_width_native'][level],own['mean_interval_width_native'][level])


def audit(plan,c):
 here=Path(__file__).parent
 for n,digest in plan['independent_source_sha256'].items():require(sha(here/n)==digest,'Independent sourcechanged')
 entry=plan['old_independent_native_CDF_helpers'];require(sha(entry['file'])==entry['sha256'],'OldindependentCDFhelperschanged')
 spec=importlib.util.spec_from_file_location('_independent_old_native_cdf_helpers',entry['file']);N=importlib.util.module_from_spec(spec);spec.loader.exec_module(N)
 # Original helper module defers its numerical backend imports to its oldmain.
 # Bind exactly those libraries without executing thatoldmain or changing its file.
 N.special=special;N.quad=quad
 own=SimpleNamespace(transform=ref.forward_yj,inverse=inverse_yj)
 loaded={}
 for name,pin in plan['inputs'].items():require(sha(pin['file'])==pin['sha256'],'Pinnedinputchanged:'+name);loaded[name]=read(pin['file'])
 protocol=loaded['science_plan'];freeze=loaded['forecast_freeze'];result=loaded['score_result'];start=loaded['score_start'];roles=loaded['roles']['representatives']
 require(loaded['independent_state_audit']['status']=='PASS','Independent savedstateauditsmustpassfirst')
 for n,digest in protocol['source_sha256'].items():require(sha(Path(plan['producer_code_directory'])/n)==digest,'Science source changed')
 folder=Path(plan['forecast_directory'])
 for name,digest in freeze['files'].items():require(sha(folder/name)==digest,'Forecast payloadchanged')
 require(start['all2560primary_and32secondary_payloads_verified_before_terminalY_extraction']is True
         and start['forecast_freeze_sha256']==result['forecast_freeze_sha256']==plan['inputs']['forecast_freeze']['sha256']
         and result['plan_sha256']==plan['inputs']['science_plan']['sha256'],'Scorefreeze/protocolbinding')
 require(datetime.fromisoformat(freeze['actual_utc'])<=datetime.fromisoformat(start['actual_utc'])<=datetime.fromisoformat(result['actual_end_utc']),'Targetscoringprecedesfreeze')
 truth=loaded['terminal_truth']['truth'];require(len(truth)==len(roles['terminal'])==128,'128fixedterminaltargets')
 target={}
 for row,meta in zip(truth,roles['terminal']):c.exact('truth_identity',row['identity'],meta);require(math.isfinite(row['native_target']),'Nonfinitetarget');target[meta['component_id']]=row['native_target']
 require(len(target)==128,'Terminalgroupsduplicate')
 forecasts=read(folder/'FORECASTS.json')['forecasts'];laws={(r['component_id'],r['method'],r['feedback_rows']):r for r in forecasts}
 records=result['per_primary_forecast_scores'];saved={(r['component_id'],r['method'],r['feedback_rows']):r for r in records}
 expected={(r['component_id'],m,k)for r in roles['terminal']for m in METHODS for k in COUNTS}
 require(len(forecasts)==len(laws)==len(records)==len(saved)==len(expected)==2560 and set(laws)==set(saved)==expected,'All2560primarycoordinates')
 values={};cache={};maxquad=0.;families={'nativeCDF':0,'empiricalEnergy':0}
 for meta in roles['terminal']:
  g=meta['component_id'];y=target[g]
  for m in METHODS:
   for k in COUNTS:
    key=(g,m,k);forecast=laws[key];r=saved[key];law=forecast['law'];ck=(y,json.dumps(law,sort_keys=True,separators=(',',':'),allow_nan=False))
    c.exact('score_jid',r['jid'],meta['jid']);c.exact('score_budget',r['new_native_cell_budget'],3*k);c.number('score_native_target',r['native_target'],y,0,0)
    if ck not in cache:
     cache[ck]=evaluate(y,law,N,own);families['empiricalEnergy'if law['family']=='weighted_empirical'else'nativeCDF']+=1
    v=cache[ck];values[key]=v;maxquad=max(maxquad,v.get('quad_error_estimate_native',0.));score_compare(r['score'],v,c)
 summary={}
 for m in METHODS:
  summary[m]={}
  for k in COUNTS:
   ownsummary=summarize([values[(r['component_id'],m,k)]for r in roles['terminal']],m,k)
   check_summary(result['summary'][m][str(k)],ownsummary,c);summary[m][str(k)]=ownsummary
 paired=result['per_terminal_paired_differences'];bygroup={r['component_id']:r for r in paired}
 require(len(paired)==len(bygroup)==128 and set(bygroup)==set(target),'Allpairedterminalgroups')
 for group in target:
  stale=values[(group,'hurdle_stale',0)]['CRPS_native'];last=values[(group,'hurdle_updated',32)]['CRPS_native']
  for k in COUNTS:c.number('paired_updated_minus_stale',bygroup[group]['updated_minus_stale_CRPS_native'][str(k)],values[(group,'hurdle_updated',k)]['CRPS_native']-stale,4e-6,4e-8)
  for m in METHODS[2:]:c.number('paired_updated_minus_reference',bygroup[group]['final_updated_minus_learned_references_CRPS_native'][m],last-values[(group,m,32)]['CRPS_native'],4e-6,4e-8)
 candidate=summary['hurdle_updated']['32']['mean_CRPS_native'];stale=summary['hurdle_stale']['0']['mean_CRPS_native'];refs={m:summary[m]['32']['mean_CRPS_native']for m in METHODS[2:]};best=min(refs.values());primary=result['primary']
 c.number('primary_updated',primary['updated_final_mean_CRPS_native'],candidate);c.number('primary_stale',primary['stale_warm_mean_CRPS_native'],stale)
 c.number('primary_mean_paired_difference',primary['signed_mean_difference_native'],math.fsum(values[(g,'hurdle_updated',32)]['CRPS_native']-values[(g,'hurdle_stale',0)]['CRPS_native']for g in target)/128,4e-6,4e-8)
 c.number('primary_relative_gain',primary['relative_gain_fraction_vs_stale'],(stale-candidate)/stale)
 c.exact('primary_learning_descriptive',primary['learning_gain_observed_descriptive'],candidate<stale)
 for m,v in refs.items():c.number('learned_reference_final',primary['learned_reference_final_CRPS_native'][m],v)
 c.number('primary_best_reference_ratio',primary['candidate_to_best_learned_reference_ratio'],candidate/best)
 c.exact('primary_descriptive_within2percent',primary['within2percent_of_best_learned_reference_descriptive'],candidate<=1.02*best)
 secondary=result['secondary_prequential_scores'];index=read(folder/'PREQUENTIAL_INDEX.json')['secondary_process_laws'];require(len(secondary)==len(index)==32,'32secondaryprocessforecasts')
 for j,(r,item)in enumerate(zip(secondary,index)):
  p=read(folder/item['forecast_file']);u=read(folder/item['outcome_update_file']);y=u['true_target_disclosure']['native_target']
  c.exact('secondary_index',r['pool_index'],j);c.exact('secondary_group',r['component_id'],p['identity']['component_id']);c.exact('secondary_primary_separate',r['secondary_process_not_part_of_primary_endpoint'],True)
  c.exact('secondary_forecast_sha',r['frozen_forecast_sha256'],item['forecast_sha256']);c.exact('secondary_outcome_sha',r['outcome_update_sha256'],item['outcome_update_sha256']);c.number('secondary_native_target',r['native_target'],y,0,0)
  require(datetime.fromisoformat(r['prediction_utc'])<=datetime.fromisoformat(r['reveal_utc']),'Secondaryforecastbeforetarget')
  ck=(y,json.dumps(p['law'],sort_keys=True,separators=(',',':'),allow_nan=False))
  if ck not in cache:cache[ck]=evaluate(y,p['law'],N,own);families['nativeCDF']+=1
  v=cache[ck];score_compare(r['score'],v,c);maxquad=max(maxquad,v.get('quad_error_estimate_native',0.))
 require(result['status']=='COMPLETE_REPORTED_EXACT_FEEDBACK_DEVELOPMENT_PILOT'and result['actual_baseline_fits']==8 and result['primary_logical_forecasts']==2560 and result['secondary_process_forecasts']==32,'Originalsci scope')
 return {'status':'PASS','primary_terminal_score_coordinates':2560,'secondary_prequential_score_coordinates':32,'all_score_coordinates':2592,'all_interval_coverage_events':7776,'all_method_budget_summaries':20,'paired_terminal_difference_coordinates':896,'unique_law_outcome_reference_scores':len(cache),'unique_reference_score_routes':families,'max_native_CDF_quad_error_estimate':maxquad,'independent_summary':summary,'primary_updated_minus_stale':candidate-stale,'primary_relative_gain':(stale-candidate)/stale,'best_reference_ratio':candidate/best,'learning_observed_descriptive':candidate<stale,'within2percent_best_reference_descriptive':candidate<=1.02*best,'SUT_imports_fits_or_original_score_replays':0,'new_material_data_acquisitions':0,'scope':'Every primary/secondary frozenlaw checked through independent nativeCDF or energy; consumed development reportedsource not independent external confirmation/agent evidence'}


def main():
 p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--approval',required=True);p.add_argument('--output-dir',required=True);a=p.parse_args();plan=read(a.plan);approval=read(a.approval);source=sha(__file__)
 require(approval.get('approved')is True and approval['scope']=='INDEPENDENT_REPORTED_FEEDBACK_ALL_SCORES_ONCE'and approval['plan_sha256']==sha(a.plan)and approval['script_sha256']==source,'Exact rootallscoreauditapproval')
 out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=False);start=datetime.now(timezone.utc).isoformat();c=Checks()
 try:
  with warnings.catch_warnings():warnings.simplefilter('error',IntegrationWarning);result=audit(plan,c)
 except Exception as e:result={'status':'FAIL','type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()}
 result.update({'start_utc':start,'end_utc':datetime.now(timezone.utc).isoformat(),'script_sha256':source,'plan_sha256':sha(a.plan),'approval_sha256':sha(a.approval),'comparison_counts':c.counts,'max_absolute_errors':c.errors,'original_training_or_score_replays':0,'raw_gzip_or_unselected_native_reads':0})
 f=out/'INDEPENDENT_REPORTED_FEEDBACK_ALL_SCORE_AUDIT.json'
 with f.open('x',encoding='utf-8')as o:json.dump(result,o,ensure_ascii=False,indent=2,allow_nan=False)
 print(json.dumps({'status':result['status'],'file':str(f),'sha256':sha(f)}));return 0 if result['status']=='PASS'else 1
if __name__=='__main__':raise SystemExit(main())

"""Read-only audit of the unique theta-V2 result against frozen independent references.

Never rerun producer quadrature or native references. Every fixture and tolerance
is inherited from the original fixed protocol and V1 comparisons remain preserved.
"""
import argparse
from datetime import datetime,timezone
import hashlib,json,math
from pathlib import Path
import traceback


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def require(x,m):
 if not x:raise AssertionError(m)


def audit(plan):
 data={}
 for name,pin in plan['inputs'].items():require(sha(pin['file'])==pin['sha256'],'Pinnedinputchanged:'+name);data[name]=read(pin['file'])
 old=data['V1_result'];new=data['V2_result'];protocol=data['V2_plan'];ref=data['independent_reference_audit'];review=data['V2_static_review']
 require(ref['status']=='PASS'and len(ref['independent_reference_results'])==12,'Complete successful independent numericreference required')
 require(review['status']=='PASS_STATIC_V2_TRIG_QUERY_COORDINATE_REPAIR','Independentsourcestaticgate')
 for name,digest in protocol['source_sha256'].items():require(sha(Path(plan['V2_code_directory'])/name)==digest,'V2sourcechanged')
 require(new['plan_sha256']==data['V2_complete']['plan_sha256']==plan['inputs']['V2_plan']['sha256'],'V2planbinding')
 require(data['V2_complete']['files']['RESULTS.json']==plan['inputs']['V2_result']['sha256'],'ActualV2outputbinding')
 before,after=old['evaluations'],new['evaluations'];refs=ref['independent_reference_results'];require(len(before)==len(after)==len(refs)==12,'All12fixedcases')
 summary=[];values=[];eta=[];allok=True;allconv=True;alltail=True
 for a,b,r in zip(before,after,refs):
  require(a['fixture']==b['fixture'],'No model/fixturechangeinnumericalrepair')
  require(b['fixture']['id']==r['id']and r.get('reference_evaluation_status')=='COMPLETE'and r['value']is not None,'Fixedavailableindependentreference')
  require([x['nodes_per_segment']for x in b['levels']]==[32,64,128,256],'Fixed4levelschedule')
  final=b['levels'][-1]['value'];difference=abs(final-b['levels'][-2]['value']);tail=b['levels'][-1]['tail']['total_bound_ideal_arithmetic']
  require(b['final_refinement_abs_diagnostic']==difference and b['convergence_diagnostic_pass']==(difference<=2e-5)and b['analytic_tail_budget_pass']==(tail<=1e-4),'Unchangednumericcriteriasummary')
  error=abs(final-r['value']);limit=2e-4+2e-5*abs(r['value']);ok=math.isfinite(final)and error<=limit
  allok=allok and ok;allconv=allconv and b['convergence_diagnostic_pass'];alltail=alltail and b['analytic_tail_budget_pass']
  numerical_ref_error=r.get('quad_error_estimate',0.)+r.get('inner_error_sample_based_width_diagnostic_not_rigorous_bound',0.)
  summary.append({'id':r['id'],'V1_value':a['levels'][-1]['value'],'V1_refinement_pass':a['convergence_diagnostic_pass'],'V2_value':final,'V2_refinement_difference':difference,'V2_refinement_pass':b['convergence_diagnostic_pass'],'reference_value':r['value'],'V2_reference_error':error,'unchanged_reference_agreement_limit':limit,'reference_agreement_pass':ok,'analytic_tail':tail,'reference_error_estimate_diagnostic':numerical_ref_error})
  values.append(final);eta.append(difference+tail+error+numerical_ref_error)
  for level in b['levels']:
   for piece in level['segments']:
    d=piece['kernel_diagnostics']
    if b['fixture']['family']=='student_t':
     require(d['renormalized_query_weights']is False and d['coordinate']=='z=sqrt(nu)*tan(theta)','Correctunnormalizedthetaquerycoordinate')
     require(abs(d['query_probability_mass_integrated']-d['analytic_query_probability_mass'])<=2e-10,'Originaltruncatedquerymass quadrature')
 require(old['NIG_state']==new['NIG_state']and old['conditional_update_probes']==new['conditional_update_probes'],'SameexactNIGmodel/hypotheticalconditionalchecks')
 rankings={}
 for name,indices in [('Studentt_native',[0,1,2,3]),('matchedGaussian_native',[8,9,10,11])]:
  qplus=sum(values[i]for i in indices[:2])/2;qminus=sum(values[i]for i in indices[2:])/2;delta=qplus-qminus;e=sum(eta[i]for i in indices)/2
  ranking=new['summary']['rankings'][name]
  require(ranking['qplus_mean_value']==qplus and ranking['qminus_mean_value']==qminus and ranking['signed_qplus_minus_qminus']==delta,'SavedV2rankings')
  rankings[name]={'qplus':qplus,'qminus':qminus,'signed_difference':delta,'retained_reference_refinement_tail_estimate_sum':e,'numerically_resolved_not_IEEE_certificate':abs(delta)>e and all(after[i]['convergence_diagnostic_pass']for i in indices),'higher_query':'qplus'if delta>0 else'qminus'if delta<0 else'tie'}
 require(new['summary']['all_refinement_diagnostics_pass']==allconv and new['summary']['all_analytic_tail_budgets_pass']==alltail,'Allgateflagsconsistent')
 return {'status':'PASS'if allok and allconv and alltail else'FAIL_FIXED_NUMERICAL_CRITERIA','fixed12_comparisons':summary,'all_reference_agreements':allok,'all_refinement_diagnostics_pass':allconv,'all_analytic_tail_budgets_pass':alltail,'rankings':rankings,'V1_refinement_failure_preserved':True,'old_independent_reference_FAILURE_preserved':True,'reference_integrals_rerun':0,'producer_quadrature_rerun':0,'scope':'SavedV2versusonefrozenindependentreferencetable,notnewmaterialevidence/newalgorithm/nativeGaussianpolicyreversal or speedupguarantee'}


def main():
 p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--approval',required=True);p.add_argument('--output-dir',required=True);a=p.parse_args();plan=read(a.plan);approval=read(a.approval);source=sha(__file__)
 require(plan['script_sha256']==source and approval.get('approved')is True and approval['scope']=='INDEPENDENT_RISKFIELD_THETA_V2_SAVED_TABLE_ONCE'and approval['plan_sha256']==sha(a.plan)and approval['script_sha256']==source,'ExactrootV2savedtableauditapproval')
 out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=False);start=datetime.now(timezone.utc).isoformat()
 try:r=audit(plan)
 except Exception as e:r={'status':'FAIL','type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()}
 r.update({'start_utc':start,'end_utc':datetime.now(timezone.utc).isoformat(),'script_sha256':source,'plan_sha256':sha(a.plan),'approval_sha256':sha(a.approval),'SUT_imports_or_new_numeric_integrals_or_materials':0})
 f=out/'INDEPENDENT_THETA_V2_SAVED_REFERENCE_AUDIT.json'
 with f.open('x',encoding='utf-8')as o:json.dump(r,o,ensure_ascii=False,indent=2,allow_nan=False)
 print(json.dumps({'status':r['status'],'file':str(f),'sha256':sha(f)}));return 0 if r['status']=='PASS'else 1
if __name__=='__main__':raise SystemExit(main())

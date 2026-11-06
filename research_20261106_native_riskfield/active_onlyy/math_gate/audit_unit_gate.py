"""Independent saved helper-unit audit; no risk_readout/SUT imports or 2D reintegration."""
import argparse
from datetime import datetime,timezone
import hashlib,json,math
from pathlib import Path
import traceback,warnings
from scipy import special
from scipy.integrate import quad,IntegrationWarning


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def require(x,m):
 if not x:raise AssertionError(m)


class Check:
 def __init__(self):self.counts={};self.errors={}
 def num(self,n,a,b,atol=2e-10,rtol=2e-10):
  require(math.isfinite(a)and math.isfinite(b),n+' nonfinite');e=abs(a-b);require(e<=atol+rtol*abs(b),f'{n}: {a} vs {b} err{e}')
  self.counts[n]=self.counts.get(n,0)+1;self.errors[n]=max(self.errors.get(n,0),e)


def k0(nu,rho):
 def C(v):return 2*math.sqrt(v)/(v-1)*math.exp(float(special.betaln(.5,v-.5)-2*special.betaln(.5,v/2)))
 A=math.sqrt(nu/(nu+1))*math.exp(math.lgamma((nu+1)/2)+math.lgamma((nu-1)/2)-2*math.lgamma(nu/2))
 return C(nu)-C(nu+1)*A*math.sqrt(1-rho*rho)


def independent_K(nu,rho,v):
 const=math.lgamma((nu+1)/2)-math.lgamma(nu/2)-.5*math.log(nu*math.pi)
 marginal=float(special.stdtr(nu,-v))
 def fun(z):
  arg=(v-rho*z)/math.sqrt((nu+z*z)/(nu+1)*(1-rho*rho))
  prob=float(special.stdtr(nu+1,-arg));pdf=math.exp(const-(nu+1)/2*math.log1p(z*z/nu))
  return (prob-marginal)**2*pdf
 cuts=[-64.,-16.,-8.,-4.,-2.,0.,2.,4.,8.,16.,64.]
 vals=[quad(fun,a,b,epsabs=2e-11,epsrel=2e-9,limit=300)[0]for a,b in zip(cuts[:-1],cuts[1:])]
 return math.fsum(vals)


def audit(plan,c):
 objects={}
 for n,pin in plan['inputs'].items():require(sha(pin['file'])==pin['sha256'],'Pinnedinputchanged:'+n);objects[n]=read(pin['file'])
 protocol=objects['unit_plan'];result=objects['unit_result'];old=objects['old_fixed12_result'];oldref=objects['old_independent_reference']
 require(result['status']=='COMPLETE_BOUNDED_NATIVE_READOUT_UNIT_GATE'and result['plan_sha256']==plan['inputs']['unit_plan']['sha256'],'Actualcompleteunitresultbinding')
 require(oldref['status']=='PASS','Frozen12independentreferencesnotcomplete')
 for n,pin in protocol['inputs'].items():require(sha(pin['file'])==pin['sha256'],'Unitproducersource/pinnedinputchanged')
 require(result['source']['core']==protocol['inputs']['core_source']['sha256'],'Actualhelperhash')
 refs={r['id']:r for r in oldref['independent_reference_results']};require(len(refs)==12,'All12oldreferencevalues')
 require(len(result['reused_fixed12'])==12,'All12unitreferencecoordinates')
 for row,orig in zip(result['reused_fixed12'],old['evaluations']):
  require(row['id']==orig['fixture']['id'],'Fixtureorderidentity')
  c.num('helper_vs_frozen_native_reference',row['value'],refs[row['id']]['value'],2e-4,2e-5)
  if orig['fixture']['loss_coordinate']!='affine':require([r['nodes']for r in row['levels']]==[32,64,128,256],'Fixed4levels')
 unique=result['new_fixed8_scalars'];expected={(nu,mu,r)for nu in(6.,70.)for mu in(-1.,1.)for r in(0.,.6)};require(len(unique)==8 and {(x['nu'],x['mu'],x['rho'])for x in unique}==expected,'Fixed8unique scalar inputs')
 cache={};kerneldetails=[];decomp=[]
 for row in unique:
  nu,mu,rho=row['nu'],row['mu'],row['rho'];require(row['scale']==1. and row['s0']==2.,'Fixedscaless1s02')
  require(row['kernel_probe_v']==[0.,.5,2.,8.]and len(row['kernel_values'])==4,'Fixed4vkernelprobes')
  for v,saved in zip(row['kernel_probe_v'],row['kernel_values']):
   key=(nu,rho,v)
   if key not in cache:cache[key]=independent_K(*key)
   c.num('kernel_vs_one_dimensional_z_pdf_quad',saved,cache[key],2e-8,2e-6)
   kerneldetails.append({'nu':nu,'mu':mu,'rho':rho,'v':v,'saved':saved,'reference':cache[key]})
  levels=row['levels'];require([r['nodes']for r in levels]==[32,64,128,256],'Newscalarfourlevels')
  require(all(math.isfinite(r['value'])and r['value']>=0 and r['tail']>=0 for r in levels),'Nonnegativefinitevalues/tails')
  last=levels[-1]['value'];change=abs(last-levels[-2]['value']);require(change<=2e-5,'Newfixedscalarrefinementdiagnosticfailed')
  if mu>=0:
   base=2*(1+mu/2)*k0(nu,rho)
   # Direct positive tail integral for I1-aI0, no subtractive upper-bound ambiguity.
   actual,err=quad(lambda z:z*float(special.stdtr(nu,-(mu+z))),0.,math.inf,epsabs=2e-11,epsrel=2e-9,limit=300)
   upper=actual # .5*s0*s^2=1 for fixeds1s02
   require(last>=base-2e-10,'Positivedecompositionbelowexactaffinebase')
   require(last-base<=upper+2e-5,'Positivecorrectionviolatessafeh/SFupperbound')
   decomp.append({'nu':nu,'rho':rho,'exact_affine_positive_base':base,'computed_negativeU_correction':last-base,'independent_positive_tail_correction_upper':upper,'quad_error_estimate':err,'source_full_tail_account':levels[-1]['tail']})
 for u,h in zip(result['negative_correction_probes']['u'],result['negative_correction_probes']['h']):
  expectedh=math.expm1(-math.log1p(-1.5*u)/3)-.5*u if u<0 else 0.
  c.num('negative_h_exact_formula',h,expectedh,2e-14,2e-10)
  require(h>=0 and (u>=0 or h<=-.5*u+2e-14),'h positivity/linearbound')
 api=result['nontrivial_response_affine_public_API'];apiresults={}
 for family,prefix in [('student_t',''),('normal','_matchedGaussian_native')]:
  reference=[(refs['qplus_tplus'+prefix]['value']+refs['qplus_tminus'+prefix]['value'])/2,(refs['qminus_tplus'+prefix]['value']+refs['qminus_tminus'+prefix]['value'])/2]
  values=api[family]['values'];require(len(values)==2,'Twoavailablequerychoices')
  for a,b in zip(values,reference):c.num('public_API_fixedaffine_and_mean_target_reference',a,b,2e-4,2e-5)
  d=api[family]['diagnostics'];require(d['target_count']==d['query_count']==2 and d['mean_over_all_targets_not_sum']is True and d['full_NIG_posterior_unchanged']is True,'APImeannotsumandposteriorboundary')
  apiresults[family]={'values':values,'frozen_mean2_reference':reference,'levels':d['levels_evaluated']}
 return {'status':'PASS','old12_reference_coordinates':12,'new_unique_scalars':8,'kernel_coordinates':32,'unique_one_dimensional_kernel_integrals':len(cache),'negative_correction_bounds_checked':decomp,'kernel_comparisons':kerneldetails,'public_API_nontrivial_response_affine':apiresults,'original2D_reference_integrals_rerun':0,'SUT_imports_materialfits_or_newreported_values':0,'scope':'helperunit/nativepositivebranchmath only;allowsknown-contextconditionalNIG genuineactivepilot,not materialperformanceornewalgorithmclaim'}


def main():
 p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--approval',required=True);p.add_argument('--output-dir',required=True);a=p.parse_args();plan=read(a.plan);approval=read(a.approval);source=sha(__file__)
 require(plan['script_sha256']==source and approval.get('approved')is True and approval['scope']=='INDEPENDENT_NATIVE_READOUT_UNIT_ONCE'and approval['plan_sha256']==sha(a.plan)and approval['script_sha256']==source,'Exactrootindependentunitapproval')
 out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=False);start=datetime.now(timezone.utc).isoformat();c=Check()
 try:
  with warnings.catch_warnings():warnings.simplefilter('error',IntegrationWarning);r=audit(plan,c)
 except Exception as e:r={'status':'FAIL','type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()}
 r.update({'start_utc':start,'end_utc':datetime.now(timezone.utc).isoformat(),'script_sha256':source,'plan_sha256':sha(a.plan),'approval_sha256':sha(a.approval),'comparison_counts':c.counts,'max_absolute_errors':c.errors})
 f=out/'INDEPENDENT_READOUT_UNIT_AUDIT.json'
 with f.open('x',encoding='utf-8')as o:json.dump(r,o,ensure_ascii=False,indent=2,allow_nan=False)
 print(json.dumps({'status':r['status'],'file':str(f),'sha256':sha(f)}));return 0 if r['status']=='PASS'else 1
if __name__=='__main__':raise SystemExit(main())

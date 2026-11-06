"""Independent fixed12 native riskfield audit: no producer imports or reruns."""
import argparse
from datetime import datetime,timezone
import hashlib,json,math
from pathlib import Path
import traceback,warnings
from scipy import special
from scipy.integrate import quad,IntegrationWarning
import native_reference as ref


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def require(x,m):
 if not x:raise AssertionError(m)


class Checks:
 def __init__(self):self.counts={};self.errors={}
 def exact(self,n,a,b):
  require(type(a)==type(b)and a==b,n+' exactmismatch');self.counts[n]=self.counts.get(n,0)+1
 def number(self,n,a,b,atol=2e-10,rtol=2e-10):
  require(math.isfinite(a)and math.isfinite(b),n+' nonfinite');e=abs(a-b);require(e<=atol+rtol*abs(b),f'{n}: {a} vs {b}, error={e}')
  self.counts[n]=self.counts.get(n,0)+1;self.errors[n]=max(self.errors.get(n,0),e)


def expected_fixtures():
 out=[]
 for qname,q in [('qplus',1.),('qminus',-1.1)]:
  for tname,t in [('tplus',1.),('tminus',-1.)]:
   qd=1.2+q*q;td=1.2+t*t
   out.append({'id':qname+'_'+tname,'family':'student_t','nu':6.,'rho':(.2+q*t)/math.sqrt(qd*td),'mu_target':t,'scale_target':math.sqrt(.5*td),'s0':2.,'mu_query':q,'scale_query':math.sqrt(.5*qd),'target_map':'YJ_half','query_context':q,'target_context':t})
 a=dict(out[0]);a['id']='case1_rho_sign_mirror';a['rho']=-a['rho'];out.append(a)
 z={'id':'Studentt_rho0_native','family':'student_t','nu':6.,'rho':0.,'mu_target':1.,'scale_target':math.sqrt(1.1),'s0':2.,'mu_query':1.,'scale_query':math.sqrt(1.1),'target_map':'YJ_half'}
 out.append(z);out.append(dict(z,id='Gaussian_rho0_native',family='gaussian',nu=None));out.append(dict(z,id='Studentt_rho0_affine_target',target_map='affine'))
 for original in out[:4]:
  out.append(dict(original,id=original['id']+'_matchedGaussian_native',family='gaussian',nu=None,scale_target=original['scale_target']*math.sqrt(1.5),scale_query=original['scale_query']*math.sqrt(1.5)))
 return out


def compare_fixture(saved,own,c):
 c.exact('fixture_id',saved['id'],own['id']);c.exact('family',saved['family'],'normal'if own['family']=='gaussian'else'student_t');c.exact('loss_coordinate',saved['loss_coordinate'],own['target_map'])
 for key,other in [('mu','mu_target'),('scale','scale_target'),('rho','rho'),('s0','s0')]:c.number('fixture_'+key,saved[key],own[other])
 if own['nu']is None:c.exact('normal_nu_null',saved['nu'],None)
 else:c.number('fixture_nu',saved['nu'],own['nu'],0,0)


def rank(p,m):
 return {'qplus_mean_value':p,'qminus_mean_value':m,'signed_qplus_minus_qminus':p-m,'higher_value_query':'qplus'if p>m else'qminus'if p<m else'tie'}


def affine_gaussian(nu,s,r,s0):return s0*s*math.sqrt(nu/(nu-2))/math.sqrt(math.pi)*(1-math.sqrt(1-r*r))


def audit(plan,c):
 for n,digest in plan['independent_source_sha256'].items():require(sha(Path(__file__).parent/n)==digest,'Ownsourcechanged')
 loaded={}
 for n,pin in plan['inputs'].items():require(sha(pin['file'])==pin['sha256'],'Pinnedinputchanged:'+n);loaded[n]=read(pin['file'])
 protocol=loaded['producer_plan'];result=loaded['producer_result'];start=loaded['producer_start'];complete=loaded['producer_complete']
 for n,digest in protocol['source_sha256'].items():require(sha(Path(plan['producer_code_directory'])/n)==digest,'Producersourcechanged')
 require(result['status']==complete['status']=='COMPLETE_FIXED12_NATIVE_RISKFIELD_MECHANISM_GATE'and result['source_sha256']==protocol['source_sha256'],'Actualcompletebound result')
 require(result['plan_sha256']==start['plan_sha256']==complete['plan_sha256']==plan['inputs']['producer_plan']['sha256'],'Planbinding')
 require(complete['files']['RESULTS.json']==plan['inputs']['producer_result']['sha256']and complete['files']['EXECUTION_START.json']==plan['inputs']['producer_start']['sha256'],'Outputbindings')
 require(datetime.fromisoformat(protocol['registered_actual_utc'])<=datetime.fromisoformat(start['actual_utc'])<=datetime.fromisoformat(result['actual_end_utc'])<=datetime.fromisoformat(complete['actual_utc']),'ActualUTCchronology')
 c.exact('configuration',result['configuration'],protocol['configuration']);cfg=protocol['configuration']
 c.exact('fixed_levels',cfg['levels'],[32,64,128,256]);c.number('fixed_Lv',cfg['outer_Lv'],32.,0,0);c.number('fixed_Lz',cfg['query_Lz'],64.,0,0)
 state=result['NIG_state'];c.exact('NIG_mean',state['mean'],[0.,1.]);c.exact('NIG_V',state['V'],[[.2,0.],[0.,1.]]);c.number('NIG_alpha',state['alpha'],3.,0,0);c.number('NIG_b',state['b'],1.5,0,0)
 ownfixtures=expected_fixtures();evaluations=result['evaluations'];require(len(evaluations)==len(ownfixtures)==12,'Fixed12evaluationinventory')
 references=[];allfinal=[];errs=[];allconv=[];alltails=[];tailreference=ref.independent_tail_integrals(6.,32.)
 gaussian_kernel_probes=[]
 for saved,f in zip(evaluations,ownfixtures):
  compare_fixture(saved['fixture'],f,c);levels=saved['levels'];c.exact('all4levelcounts',[r['nodes_per_segment']for r in levels],[32,64,128,256])
  for level in levels:
   require(level['value']>=0 and math.isfinite(level['value']),'Finitepositivefieldvalue')
   for piece in level['segments']:
    require(piece['value']>=0 and piece['target_nodes']==level['nodes_per_segment'],'Allsegmentsretained')
    diag=piece['kernel_diagnostics']
    if f['family']=='student_t':
     c.exact('Z_weights_unnormalized',diag['renormalized_query_weights'],False)
     expectedmass=float(special.stdtr(6.,64.)-special.stdtr(6.,-64.));c.number('Z_mass',diag['query_probability_mass_integrated'],expectedmass,2e-12,2e-12)
    elif not diag.get('exact_zero_rho_independence',False):c.exact('no_negative_hard_clamp',diag['hard_negative_clamp_used'],False)
  final=levels[-1];difference=abs(final['value']-levels[-2]['value']);c.number('refinement_actual_difference',saved['final_refinement_abs_diagnostic'],difference,0,0)
  c.exact('refinement_flag',saved['convergence_diagnostic_pass'],difference<=2e-5)
  tail=final['tail'];b=tail['total_bound_ideal_arithmetic'];c.exact('tail_flag',saved['analytic_tail_budget_pass'],b<=1e-4)
  if f['family']=='student_t':
   nu=6.;mu=f['mu_target'];s=f['scale_target'];s0=f['s0'];L=32.;density=ref.stdpdf(L,'student_t',nu);sf=ref.stdcdf(-L,'student_t',nu)
   p0=(nu+L*L)/(nu-1)*density;p1=.5*(nu*sf+L*(nu+L*L)*density)/(nu-2)
   c.number('I0_formula_vs_actualtailquad',tail['I0_closed_difference_display'],tailreference['I0'],2e-11,2e-8)
   c.number('I1_formula_vs_actualtailquad',tail['I1_closed_difference_display'],tailreference['I1'],2e-11,2e-8)
   c.number('I0_safe_positive_bound',tail['I0_safe_ignored_negative_term_bound'],p0);c.number('I1_safe_positive_bound',tail['I1_safe_ignored_negative_term_bound'],p1)
   require(p0>=tailreference['I0']-2e-11 and p1>=tailreference['I1']-2e-11,'Safeconservativetailingredientbound')
   a,bb=(1.,0.)if f['target_map']=='affine'else(1+.5*abs(mu),.5*s)
   outer=2*s0*s*(a*p0+bb*p1)
   width=s0*((mu+s*32)-(mu-s*32))if f['target_map']=='affine'else s0*(ref.inv_yj(mu+s*32)-ref.inv_yj(mu-s*32))
   inner=2*ref.stdcdf(-64.,'student_t',nu)*width
   c.number('outer_analytic_tail',tail['outer_tail_bound_ideal_arithmetic'],outer);c.number('inner_analytic_tail',tail['query_tail_bound_ideal_arithmetic'],inner);c.number('total_analytic_tail',b,outer+inner)
  reference=ref.native_double_quad(f,32.,64.);references.append(dict(id=f['id'],**reference));value=reference['value']
  agreement_error=abs(final['value']-value);agreement_limit=2e-4+2e-5*abs(value)
  references[-1].update({'producer_final_value':final['value'],'agreement_abs_error':agreement_error,'agreement_limit':agreement_limit,'agreement_pass':agreement_error<=agreement_limit})
  c.counts['native_reference_coordinates_collected']=c.counts.get('native_reference_coordinates_collected',0)+1
  c.errors['native_independent_reference_abs_error']=max(c.errors.get('native_independent_reference_abs_error',0.),agreement_error)
  allfinal.append(final['value']);errs.append(difference+b);allconv.append(saved['convergence_diagnostic_pass']);alltails.append(saved['analytic_tail_budget_pass'])
  if f['family']=='gaussian'and f['rho']!=0:
   for v in(0.,.5,2.,8.):
    positive=ref.gaussian_positive_kernel(v,f['rho']);owen=ref.gaussian_owen_kernel(v,f['rho']);c.number('Owen_positive_Plackett_identity',owen,positive,2e-12,2e-9)
    gaussian_kernel_probes.append({'fixture_id':f['id'],'v':v,'Owen':owen,'positive_angle_quad':positive})
 probes=result['conditional_update_probes'];require(len(probes)==12,'Fixed12conditionalprobes')
 lookup={f['id']:f for f in ownfixtures[:4]}
 for r in probes:
  f=lookup[r['fixture_id']];z=r['standardized_hypothetical_z'];require(z in(-1.,0.,1.),'Fixedhypotheticalz')
  q,t=f['query_context'],f['target_context'];yq=f['mu_query']+f['scale_query']*z;c.number('hypothetical_query_coordinate',r['hypothetical_modeled_query'],yq)
  det=6+5*q*q;aa=(yq-q)/det;beta=(6+5*q*yq)/det;bnew=1.5+.5*(yq-q)**2/(1.2+q*q)
  v00=(1+q*q)/det;v01=-q/det;v11=6/det
  expected={'mu':aa+beta*t,'scale':math.sqrt(bnew/3.5*(1+v00+2*t*v01+t*t*v11)),'nu':7.}
  for field in expected:
   c.number('NIG_full_precision_update',r['NIG_updated_target'][field],expected[field]);c.number('bivariate_t_conditional',r['conditional_bivariate_t_target'][field],expected[field])
  c.exact('hypothetical_not_material_label',r['not_a_real_measurement_or_persisted_material_training_label'],True)
 require({(r['fixture_id'],r['standardized_hypothetical_z'])for r in probes}=={(f['id'],z)for f in ownfixtures[:4]for z in(-1.,0.,1.)},'Completeconditionalprobeinventory')
 byid={f['id']:i for i,f in enumerate(ownfixtures)}
 rho_probes=result['rho_coupling_probes'];require(len(rho_probes)==4,'Fourboundprobes')
 z2,zerr=quad(lambda z:z*z*ref.stdpdf(z,'student_t',6.),-math.inf,math.inf,epsabs=2e-11,epsrel=2e-11,limit=300)
 for r in rho_probes:
  i,j=byid[r['left_fixture']],byid[r['right_fixture']];f,g=ownfixtures[i],ownfixtures[j]
  require(all(f[k]==g[k]for k in('mu_target','scale_target','s0','nu','target_map')),'Couplingmarginalsmustmatch')
  rho,rhop=f['rho'],g['rho'];nu=f['nu'];s=f['scale_target'];mu=f['mu_target'];s0=f['s0'];M=math.sqrt(mu*mu+s*s*nu/(nu-2));constant=s0*(1+M)*s*math.sqrt(nu/(nu-2));chord=math.hypot(rho-rhop,math.sqrt(1-rho*rho)-math.sqrt(1-rhop*rhop))
  b=constant*chord;coupling=r['analytic_coupling'];c.number('rho_coupling_constant',coupling['constant'],constant);c.number('rho_chord',coupling['chord'],chord);c.number('rho_coupling_bound',coupling['bound_ideal_arithmetic'],b)
  l2quad=s*math.sqrt((rho-rhop)**2*z2+(math.sqrt(1-rho*rho)-math.sqrt(1-rhop*rhop))**2*(nu+z2)/(nu+1)*(nu+1)/(nu-1))
  c.number('coupling_L2_moment_quadrature',l2quad,s*math.sqrt(nu/(nu-2))*chord)
  observed=abs(allfinal[i]-allfinal[j]);c.number('rho_observed_value_difference',r['observed_value_difference'],observed,0,0);c.exact('rho_bound_reported_flag',r['within_analytic_plus_predeclared_agreement_tolerance'],observed<=b+2e-4)
 summary=result['summary'];c.exact('fixed12_count',summary['fixed_evaluations'],12);c.exact('conditional12_count',summary['NIG_conditional_probes'],12);c.exact('all_convergence_flag',summary['all_refinement_diagnostics_pass'],all(allconv));c.exact('all_tail_flag',summary['all_analytic_tail_budgets_pass'],all(alltails))
 c.number('signed_rho_symmetry',summary['rho_sign_symmetry_abs_difference'],abs(allfinal[0]-allfinal[4]),0,0)
 c.number('independent_signedrho_symmetry',references[0]['value'],references[4]['value'],2e-4,2e-5)
 c.exact('t_zero_rho_positive_flag',summary['finite_Student_rho0_positive_value_observed'],allfinal[5]>0);require(references[5]['value']>0,'FiniteStudenttzero-rhoexpectedpositive')
 c.number('Gaussianzero_exact0',allfinal[6],0.,0,0);c.number('Gaussianzero_summary0',summary['Gaussian_rho0_exact_value'],0.,0,0)
 affine=ref.affine_t_value(6.,0.,math.sqrt(1.1),2.);c.number('affine_closed_coefficient',summary['affine_rho0_closed_value'],affine);c.number('affine_GL_closed_reported_difference',summary['affine_rho0_GL_vs_closed_abs_difference'],abs(allfinal[7]-affine))
 rankings={}
 for name,indices in [('Studentt_native',[0,1,2,3]),('matchedGaussian_native',[8,9,10,11])]:
  a,b,d,e=[allfinal[i]for i in indices];ranking=rank((a+b)/2,(d+e)/2);saved=summary['rankings'][name]
  for k,v in ranking.items():c.exact('ranking_'+k,saved[k],v)if isinstance(v,str)else c.number('ranking_'+k,saved[k],v,0,0)
  eta=sum(errs[i]for i in indices)/2;c.number('ranking_error_proxies',saved['summed_numerical_proxy_and_tail'],eta)
  c.exact('ranking_declared_resolution',saved['resolved_on_declared_proxies_only'],abs(ranking['signed_qplus_minus_qminus'])>eta)
  rv=[references[i]['value']for i in indices];ranking['independent_native_reference_signed_difference']=(rv[0]+rv[1]-rv[2]-rv[3])/2
  ranking['producer_proxy_and_tail_sum']=eta;ranking['convergence_diagnostics_all_pass']=all(allconv[i]for i in indices)
  ranking['reference_agreement_and_estimate_envelope']=sum(abs(allfinal[i]-references[i]['value'])+references[i].get('quad_error_estimate',0.)+references[i].get('inner_error_sample_based_width_diagnostic_not_rigorous_bound',0.)for i in indices)/2+eta
  ranking['numerically_resolved_in_independent_checks_not_IEEE_certificate']=abs(ranking['signed_qplus_minus_qminus'])>ranking['reference_agreement_and_estimate_envelope']and ranking['convergence_diagnostics_all_pass']
  rankings[name]=ranking
 for name,method in [('Studentt_affine',lambda f:ref.affine_t_value(f['nu'],f['rho'],f['scale_target'],f['s0'])),('matchedGaussian_affine',lambda f:affine_gaussian(f['nu'],f['scale_target'],f['rho'],f['s0']))]:
  vs=[method(f)for f in ownfixtures[:4]];r=rank(sum(vs[:2])/2,sum(vs[2:])/2);saved=summary['rankings'][name]
  for k,v in r.items():c.exact('affine_rank_'+k,saved[k],v)if isinstance(v,str)else c.number('affine_rank_'+k,saved[k],v)
  rankings[name]=r
 c.exact('affineStudent_vs_Gaussian_query_rank',rankings['Studentt_affine']['higher_value_query'],rankings['matchedGaussian_affine']['higher_value_query'])
 return {'status':'PASS'if all(r['agreement_pass']for r in references)else'FAIL_NUMERICAL_REFERENCE_AGREEMENT','fixed_mechanism8_plus_sameobjective_Gaussian4':12,'independent_reference_results':references,'independent_Gaussian_kernel_probes':gaussian_kernel_probes,'independent_tail_integral_probes':tailreference,'independent_coupling_Z2_integral':z2,'all_native_reference_agreements':all(r['agreement_pass']for r in references),'failed_reference_fixture_ids':[r['id']for r in references if not r['agreement_pass']],'producer_numerical_gate_fully_passed':all(allconv)and all(alltails), 'audit_PASS_does_not_override_producer_refinement_failure':True,'all_final_refinement_diagnostics_pass':all(allconv),'all_analytic_tail_budgets_pass':all(alltails),'rankings':rankings,'crossbranch_independent_information_value':0.,'crossbranch_zero_is_theoretical_only_not_a_runtime_multibranch_API_test':True,'source_training_material_or_producer_replay_calls':0,'scope':'knownNIG artificialmechanism andnumericalreadout,not newalgorithm/speedup/actualactive materialsuccess;unresolvedornegative rankresultretained'}


def main():
 p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--approval',required=True);p.add_argument('--output-dir',required=True);a=p.parse_args();plan=read(a.plan);approval=read(a.approval);source=sha(__file__)
 require(approval.get('approved')is True and approval['scope']=='INDEPENDENT_NATIVE_RISKFIELD_FIXED12_ONCE'and approval['plan_sha256']==sha(a.plan)and approval['script_sha256']==source,'Exactrootindependentfixed12approval')
 out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=False);start=datetime.now(timezone.utc).isoformat();c=Checks()
 try:
  with warnings.catch_warnings():warnings.simplefilter('error',IntegrationWarning);r=audit(plan,c)
 except Exception as e:r={'status':'FAIL','type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()}
 r.update({'start_utc':start,'end_utc':datetime.now(timezone.utc).isoformat(),'script_sha256':source,'plan_sha256':sha(a.plan),'approval_sha256':sha(a.approval),'counts':c.counts,'max_absolute_errors':c.errors,'random_samples_new_materials_or_SUT_imports':0})
 f=out/'INDEPENDENT_NATIVE_RISKFIELD_FIXED12_AUDIT.json'
 with f.open('x',encoding='utf-8')as o:json.dump(r,o,ensure_ascii=False,indent=2,allow_nan=False)
 print(json.dumps({'status':r['status'],'file':str(f),'sha256':sha(f)}));return 0 if r['status']=='PASS'else 1
if __name__=='__main__':raise SystemExit(main())

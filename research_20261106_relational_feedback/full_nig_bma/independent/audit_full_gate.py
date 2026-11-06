"""Independent single full-NIG-BMA fixture and exact label/CDF probes; no SUT imports."""
import argparse
from datetime import datetime,timezone
import hashlib,json,math
from pathlib import Path
import traceback,warnings
import numpy as np
from scipy import special
from scipy.integrate import IntegrationWarning
import full_native_reference as ref


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def require(x,m):
    if not x:raise AssertionError(m)


class Checks:
    def __init__(self):self.counts={};self.errors={}
    def exact(self,key,a,b):
        require(type(a)==type(b) and a==b,key+': exact mismatch');self.counts[key]=self.counts.get(key,0)+1
    def number(self,key,a,b,atol=2e-10,rtol=2e-10):
        require(math.isfinite(a) and math.isfinite(b),key+': nonfinite')
        error=abs(a-b);require(error<=atol+rtol*abs(b),'%s: %.17g vs %.17g, error %.17g'%(key,a,b,error))
        self.counts[key]=self.counts.get(key,0)+1;self.errors[key]=max(self.errors.get(key,0),error)
    def array(self,key,a,b,atol=2e-10,rtol=2e-10):
        require(len(a)==len(b),key+': shape')
        for x,y in zip(a,b):
            if isinstance(y,list):self.array(key,x,y,atol,rtol)
            else:self.number(key,x,y,atol,rtol)


def audit(plan,c):
    for name,digest in plan['independent_sources'].items():require(sha(Path(__file__).parent/name)==digest,'independent source changed')
    loaded={}
    for name,pin in plan['inputs'].items():
        require(sha(pin['file'])==pin['sha256'],'pinned input changed:'+name);loaded[name]=read(pin['file'])
    proposal,result,protocol=loaded['proposal'],loaded['producer_result'],loaded['producer_plan']
    c.exact('exact_fixed_fixture',result['fixed_fixture'],proposal)
    c.exact('result_plan_binding',result['plan_sha256'],plan['inputs']['producer_plan']['sha256'])
    c.exact('result_proposal_binding',result['proposal_sha256'],plan['inputs']['proposal']['sha256'])
    c.exact('result_sources',result['source_sha256'],protocol['source_sha256'])
    for name,digest in protocol['source_sha256'].items():require(sha(Path(plan['producer_code_directory'])/name)==digest,'producer source changed')
    c.exact('no_material',result['material_reads_fits_or_queries'],0)
    ensemble=result['initial_ensemble'];states=proposal['models'];weights=proposal['weights']
    c.array('initial_weights',np.exp(ensemble['log_weights']).tolist(),weights,2e-12,0.)
    for saved,expected in zip(ensemble['models'],states):
        c.exact('model_order',saved['id'],expected['id'])
        for key in ('mean','V'):c.array('initial_'+key,saved['state'][key],expected[key])
        for key in ('alpha','b'):c.number('initial_'+key,saved['state'][key],expected[key],0,0)
    pairs=[ref.pair(state,proposal['query_context'],proposal['target_context']) for state in states]
    levels=result['evaluations_by_level'];require(len(levels)==4,'four fixedlevels')
    c.exact('level_nodes',[v['nodes_per_segment'] for v in levels],[32,64,128,256])
    for level in levels:
        c.exact('signed_between_not_clipped',level['signed_between_clipped'],False)
        for saved,expected in zip(level['pair_laws'],pairs):
            for own,key in [('q_mu','mu_query'),('t_mu','mu_target'),('q_scale','scale_query'),('t_scale','scale_target'),('rho','rho'),('nu','nu')]:
                c.number('jointbivt_'+key,saved[key],expected[own])
        c.number('raw_signedbetween',level['native_signed_between'],level['native_B_before']-level['native_E_B_after'])
        c.number('raw_within_sum',level['native_weighted_within_total'],math.fsum(level['native_weighted_within_components']))
        c.number('raw_closure',level['decomposition_closure'],level['native_full_VOI']-level['native_weighted_within_total']-level['native_signed_between'])
    probe_results=[];probes=result['synthetic_fixed_label_probes']
    require(len(probes)==2,'two fixedprobes from originalprior')
    for saved,value in zip(probes,proposal['fixed_observed_modeled_query_Y']):
        c.number('fixed_query_probe',saved['artificial_modeled_query_Y'],value,0,0)
        post,logdensity=ref.weights_at(weights,pairs,value)
        c.array('Bayes_H_weights',np.exp(saved['updated_log_weights']).tolist(),post.tolist(),2e-12,0.)
        c.array('bivt_H_weights',saved['bivariate_query_posterior_weights'],post.tolist(),2e-12,0.)
        expected_logs=[ref.t_logpdf(value,p['q_mu'],p['q_scale'],p['nu']) for p in pairs]
        c.array('old_query_logdensity_before_update',saved['evidence']['preupdate_logpredictive'],expected_logs)
        c.exact('density_evidence_before_update',saved['evidence']['evidence_before_NIG_update'],True)
        c.exact('common_nativeJac_cancels',saved['evidence']['native_shared_response_Jacobian_cancels'],True)
        c.exact('no_H_probability_floor',saved['evidence']['probability_floor'],False)
        conditional=[]
        for state,pair,row in zip(states,pairs,saved['updated_models']):
            updated=ref.dense_update(state,proposal['query_context'],value)
            for key in ('mean','V'):c.array('dense_NIG_update_'+key,row['updated_NIG'][key],updated[key])
            for key in ('alpha','b'):c.number('dense_NIG_update_'+key,row['updated_NIG'][key],updated[key])
            c.exact('one_actual_label_update',row['updated_NIG']['labeled_updates'],1)
            target=ref.marginal(updated,proposal['target_context']);direct=ref.conditional(pair,value)
            for key in ('mu','scale','nu'):
                c.number('dense_target_law_'+key,row['updated_NIG_target_law'][key],target[key])
                c.number('joint_bivt_condition_'+key,row['bivariate_conditional_target_law'][key],direct[key])
                c.number('independent_update_equals_jointcondition_'+key,target[key],direct[key])
            conditional.append(direct)
        points=proposal['target_modeled_CDF_points'];expected=[]
        for target in points:
            expected.append(math.fsum(float(w)*float(special.stdtr(v['nu'],(target-v['mu'])/v['scale'])) for w,v in zip(post,conditional)))
        c.array('full_mixture_CDF_after_denseupdate',saved['mixture_CDF_from_updated_NIG'],expected,2e-11,0.)
        c.array('full_mixture_CDF_after_bivt',saved['mixture_CDF_from_bivariate_conditionals'],expected,2e-11,0.)
        probe_results.append({'modeled_query_Y':value,'posterior_H_weights':post.tolist(),
                              'independent_mixture_CDF_at_fixed_points':expected,'native_query_Y_artificial':ref.to_native(value,proposal['response_transform'])})
    own=ref.native_reference(proposal)
    last=levels[-1]
    comparisons={'native_full_VOI':'full_native_VOI','native_weighted_within_total':'weighted_within_native_VOI',
                 'native_B_before':'Bbefore_native','native_E_B_after':'expected_Bafter_native','native_signed_between':'signed_between_native'}
    for saved_key,own_key in comparisons.items():c.number('native_reference_'+saved_key,last[saved_key],own[own_key],2e-4,2e-5)
    component=[r['weighted_within_H_native'] for r in own['by_generating_H']]
    c.array('native_reference_within_components',last['native_weighted_within_components'],component,2e-4,2e-5)
    c.number('initial_same_targetCDF_Bbefore',last['native_B_before'],0.,2e-10,0.)
    quantities=('native_full_VOI','native_weighted_within_total','native_B_before','native_E_B_after','native_signed_between')
    differences=[abs(levels[-1][key]-levels[-2][key]) for key in quantities]
    differences.extend(abs(a-b) for a,b in zip(levels[-1]['native_weighted_within_components'],levels[-2]['native_weighted_within_components']))
    c.number('saved_refinement_proxy',result['max_refinement_proxy'],max(differences),2e-12,2e-12)
    producer_refinement=max(differences)<=2e-5
    tail=last['tail'];producer_tail=tail['total_value_tail_bound_ideal_arithmetic']<=1e-4
    require(tail['finite_query_decomposition_closure_allowance']>=0,'positive closurecut account')
    closure_error=own['outer_quad_error_sum']+own['max_inner_quad_error_estimate']
    require(abs(own['finite_closure_raw'])<=own['ideal_tails']['finite_closure_ideal_bound_for_equal_initial_targetCDF']+3*closure_error+2e-8,'independent finitequeryclosure beyond idealcut+numericdiagnostic')
    uncertainty=3*closure_error+2e-4+2e-5*abs(own['signed_between_native'])
    negative_resolved=own['signed_between_native'] < -uncertainty
    require(negative_resolved,'fixedsignedbetween negativevalue unresolved beyond declared numeric uncertainty')
    require(own['full_native_VOI']>0 and own['weighted_within_native_VOI']>own['full_native_VOI'],'withinH must not be removed or negativebetween clipped')
    return {'status':'PASS','one_fixed_mixture_native_risk_reference':own,'label_and_CDF_probes':probe_results,
            'negative_signedbetween_resolved':negative_resolved,'producer_refinement_pass':producer_refinement,
            'producer_ideal_tail_pass':producer_tail,'producer_status':result['status'],
            'producer_all_recorded_diagnostics_pass':all(result['checks'].values()),
            'algorithm_gate_ready':producer_refinement and producer_tail and all(result['checks'].values()),
            'singleH_reduction':'classical bivt/conditionalNIG formula; previous fixed12 receipts reused, no old2D calculations',
            'new_material_reads_fits_or_SUT_imports':0,'scientific_scope':'onefixed fullNIG-BMA arithmetic gate; no material benefit, newBayesprinciple or topconference readiness'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--approval',required=True);p.add_argument('--output-dir',required=True)
    a=p.parse_args();plan=read(a.plan);approval=read(a.approval)
    require(approval.get('approved') is True and approval['scope']=='INDEPENDENT_ONE_FULL_NIG_BMA_NATIVE_REFERENCE_ONCE'
            and approval['plan_sha256']==sha(a.plan) and approval['script_sha256']==sha(__file__),'exact root audit approval')
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=False);start=datetime.now(timezone.utc).isoformat();c=Checks()
    try:
        with warnings.catch_warnings():warnings.simplefilter('error',IntegrationWarning);result=audit(plan,c)
    except Exception as problem:result={'status':'FAIL','type':type(problem).__name__,'message':str(problem),'traceback':traceback.format_exc()}
    result.update({'actual_start_utc':start,'actual_end_utc':datetime.now(timezone.utc).isoformat(),'script_sha256':sha(__file__),
                   'plan_sha256':sha(a.plan),'approval_sha256':sha(a.approval),'comparison_counts':c.counts,'max_absolute_errors':c.errors})
    f=out/'INDEPENDENT_FULL_NIG_BMA_GATE_AUDIT.json'
    with f.open('x',encoding='utf-8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({'status':result['status'],'file':str(f),'sha256':sha(f)}))
    return 0 if result['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())

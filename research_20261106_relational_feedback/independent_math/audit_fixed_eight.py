"""Independent fixed8 identifiable-relation mechanism audit; no SUT import or material data."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import traceback
import warnings
import numpy as np
from scipy.integrate import IntegrationWarning
import native_cdf_reference as reference


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def require(x,m):
    if not x:raise AssertionError(m)


class Checks:
    def __init__(self):self.counts={};self.errors={}
    def number(self,key,a,b,atol=2e-8,rtol=2e-7):
        require(math.isfinite(a) and math.isfinite(b),key+': nonfinite')
        error=abs(a-b);require(error<=atol+rtol*abs(b),'%s: %.17g vs %.17g, err %.17g'%(key,a,b,error))
        self.counts[key]=self.counts.get(key,0)+1;self.errors[key]=max(self.errors.get(key,0),error)
    def exact(self,key,a,b):
        require(type(a)==type(b) and a==b,key+': exact mismatch');self.counts[key]=self.counts.get(key,0)+1
    def array(self,key,a,b,atol=2e-12,rtol=0.):
        require(len(a)==len(b),key+': dimensions')
        for x,y in zip(a,b):
            if isinstance(y,list):self.array(key,x,y,atol,rtol)
            else:self.number(key,x,y,atol,rtol)


def audit(plan,c):
    for name,digest in plan['independent_sources'].items():require(sha(Path(__file__).parent/name)==digest,'independent source changed')
    for name,pin in plan['inputs'].items():require(sha(pin['file'])==pin['sha256'],'pinned input changed:'+name)
    result=read(plan['inputs']['producer_result']['file'])
    rows={r['id']:r for r in result['evaluations']}
    cases=reference.fixtures()
    require(len(rows)==len(result['evaluations'])==8 and set(rows)=={c['id'] for c in cases},'exact fixed8case identities')
    values={};references=[];covariance_errors={};null_diagnostics=[]
    # Attempt all cases and preserve any numerical-reference failures instead of filling values.
    failures=[]
    for case in cases:
        saved=rows[case['id']]
        try:
            for name in ('weights','query_means','query_SD','target_means','target_SD'):
                c.array('fixed_'+name,saved[name],case[name])
            own=reference.native_risk_gain(case)
            covariance,cov_error=reference.covariance(case)
            distances=reference.distances(case)
            c.number('native_CDF_vs_saved_C_D',saved['value_C_D'],own['value'])
            c.array('centered_weight_covariance',saved['C'],covariance,2e-10,0.)
            c.array('Gaussian_absolute_distances',saved['D'],distances,2e-12,2e-12)
            for i,row in enumerate(saved['C']):
                c.number('covariance_row_sum',math.fsum(row),0.,2e-10,0.)
                for j,value in enumerate(row):c.number('covariance_symmetry',value,saved['C'][j][i],2e-10,0.)
            minimum_eigenvalue=float(np.linalg.eigvalsh(np.array(saved['C']))[0])
            require(minimum_eigenvalue>=-2e-10,'posterior weight covariance PSD')
            cd=-.5*math.fsum(saved['C'][i][j]*saved['D'][i][j] for i in range(3) for j in range(3))
            c.number('saved_C_D_readout',saved['value_C_D'],cd,2e-12,2e-12)
            require(saved['value_C_D']>=-2e-10 and own['value']>=0,'unclipped finite gain nonnegative tolerance')
            if case['id'].startswith('null_'):
                c.number('null_gain_producer',saved['value_C_D'],0.,2e-10,0.)
                c.number('null_gain_reference',own['value'],0.,2e-10,0.)
                null_diagnostics.append({'id':case['id'],'raw_producer_gain':saved['value_C_D'],
                                         'raw_native_CDF_gain':own['value'],'covariance_trace':math.fsum(saved['C'][i][i] for i in range(3))})
            values[case['id']]=own
            references.append({'id':case['id'],'fixed_case':case,'native_reference':own,'independent_C':covariance,
                               'independent_D':distances,'C_quad_error_estimate':cov_error,
                               'producer_value':saved['value_C_D'],'min_saved_C_eigenvalue':minimum_eigenvalue})
        except Exception as problem:
            failures.append({'id':case['id'],'type':type(problem).__name__,'message':str(problem),'traceback':traceback.format_exc()})
    require(not failures,'one or more independent fixed references failed: '+json.dumps(failures,ensure_ascii=True))
    c.number('prior_mirror_tie',values['prior_qL']['value'],values['prior_qR']['value'],2e-10,0.)
    c.number('opposite_probe_mirror_LR',values['after_negative_qL']['value'],values['after_positive_qR']['value'],2e-10,0.)
    c.number('opposite_probe_mirror_RL',values['after_negative_qR']['value'],values['after_positive_qL']['value'],2e-10,0.)
    ranks={}
    for stem in ('after_negative','after_positive'):
        left,right=values[stem+'_qL'],values[stem+'_qR']
        delta=left['value']-right['value']
        uncertainty=math.fsum(x['summed_error_estimate_diagnostic']+x['ideal_arithmetic_target_tail']+x['ideal_arithmetic_query_tail']
                              +2e-8+2e-7*abs(x['value']) for x in (left,right))
        ranks[stem]={'qL_minus_qR':delta,'sum_numeric_proxies_tail_and_tolerance':uncertainty,
                     'resolved':abs(delta)>uncertainty,'higher_action':'qL' if delta>0 else 'qR'}
    mechanism=all(r['resolved'] for r in ranks.values()) and ranks['after_negative']['higher_action']=='qL' and ranks['after_positive']['higher_action']=='qR'
    require(mechanism,'fixed synthetic outcome-adaptive query mechanism not resolved as predicted; retain failure, no parameter scan')
    return {'status':'PASS','fixed_scalar_values':8,'independent_native_CDF_references':references,
            'mechanism_outcome_dependent_next_query':mechanism,'rankings':ranks,'null_controls':null_diagnostics,
            'reference_domain':{'native_query':[-14,14],'native_target':[-12,12],'query_density_renormalized':False},
            'new_material_labels_fits_or_SUT_imports':0,'all_cases_attempted':8,
            'scientific_scope':'synthetic fixedparameter Gaussian product hypotheses only; classic Bayes/proper-score mechanism, not new algorithm/material evidence'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);parser.add_argument('--approval',required=True);parser.add_argument('--output-dir',required=True)
    args=parser.parse_args();plan=read(args.plan);approval=read(args.approval)
    require(approval.get('approved') is True and approval['scope']=='INDEPENDENT_RELATIONAL_FIXED8_NATIVE_CDF_ONCE'
            and approval['plan_sha256']==sha(args.plan) and approval['script_sha256']==sha(__file__),'exact root independent approval')
    output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=False);start=datetime.now(timezone.utc).isoformat();c=Checks()
    try:
        with warnings.catch_warnings():warnings.simplefilter('error',IntegrationWarning);result=audit(plan,c)
    except Exception as problem:
        result={'status':'FAIL','type':type(problem).__name__,'message':str(problem),'traceback':traceback.format_exc()}
    result.update({'actual_start_utc':start,'actual_end_utc':datetime.now(timezone.utc).isoformat(),'plan_sha256':sha(args.plan),
                   'source_sha256':sha(__file__),'approval_sha256':sha(args.approval),'comparison_counts':c.counts,'max_absolute_errors':c.errors})
    file=output/'INDEPENDENT_RELATIONAL_FIXED8_AUDIT.json'
    with file.open('x',encoding='utf-8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({'status':result['status'],'file':str(file),'sha256':sha(file)}))
    return 0 if result['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())

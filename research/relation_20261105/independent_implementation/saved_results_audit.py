"""Standard-library-only arithmetic audit of frozen synthetic outputs; never imports SUT."""
from pathlib import Path
from fractions import Fraction as Q
import argparse
import datetime
import hashlib
import json
import math
import traceback

def sha(b):
    return hashlib.sha256(b).hexdigest()

def put(path,obj):
    with path.open('x',encoding='utf-8') as f:
        json.dump(obj,f,ensure_ascii=True,indent=2,allow_nan=False);f.write('\n')

def require(x,message):
    if not x:
        raise AssertionError(message)

def decode_fraction(x):
    if isinstance(x,dict):
        q=Q(x['exact_fraction'])
        require(float(q)==x['float_display'],'Fraction display differs from exact source')
        return q
    return Q(x)

def independent_branches(prior,likelihood):
    w=[Q(x) for x in prior]
    require(sum(w)==1 and all(x>=0 for x in w),'Invalid prior')
    ls=[[Q(x) for x in row] for row in likelihood]
    require(all(len(row)==len(w) for row in ls),'Bad finite likelihood dimensions')
    require(all(sum(row[h] for row in ls)==1 for h in range(len(w))),'Likelihood column not normalized')
    result=[]
    for i,row in enumerate(ls):
        joint=[a*b for a,b in zip(w,row)];p=sum(joint)
        if p:
            result.append((i,p,[x/p for x in joint]))
    require(sum(p for _,p,_ in result)==1,'Finite branch probability lost')
    F=[[sum(p*(wp[i]-w[i])*(wp[j]-w[j]) for _,p,wp in result) for j in range(len(w))] for i in range(len(w))]
    require(all(sum(row)==0 for row in F),'Exact posterior covariance row sum failed')
    return w,result,F

def kernel(action):
    r=action['rhos'];scale=action['scale']*action.get('weight',1.0)
    # Independent scalar log1p/expm1 form, rather than producer NumPy rationalization.
    return [[-math.expm1(0.5*math.log1p(-x*y))*scale/math.sqrt(math.pi) for y in r] for x in r]

def quadratic(w,S):
    return math.fsum(float(w[i]*w[j])*S[i][j] for i in range(len(w)) for j in range(len(w)))

def tolerance(x,config):
    return config['absolute']+config['relative']*abs(x)

def close(actual,expected,config,label):
    error=abs(actual-expected)
    require(error<=tolerance(expected,config),label+' exceeds registered arithmetic tolerance')
    return error

def audit_case(row,budget,config,is_principle):
    actions=row['repair_actions'];proxies=row['proxy_actions'];prior=row['prior']
    Ss=[kernel(b) for b in proxies]
    labels=['adaptive'] if is_principle else ['reference','cached','blocked']
    out={'family_or_principle':row.get('case',row.get('family')),'repairs':len(actions),'proxies':len(proxies),
         'max_value_error':0.0,'max_kernel_entry_error':0.0,'max_fixed_pair_error':0.0,'max_choice_regret':0.0,
         'first_actions':[],'source_values_are_from_one_synthetic_run_not_new_experiments':True}
    if is_principle:
        for i,S in enumerate(Ss):
            for h in range(len(S)):
                for k in range(len(S)):
                    out['max_kernel_entry_error']=max(out['max_kernel_entry_error'],close(row['S'][i][h][k],S[h][k],config,'Saved kernel entry'))
    for a,action in enumerate(actions):
        w,branches,F=independent_branches(prior,action['likelihood'])
        legal=[j for j,b in enumerate(proxies) if action['cost']+b['cost']<=budget]
        require(bool(legal),'Compared first action has no legal continuation')
        if is_principle:
            savedF=[[decode_fraction(v) for v in rr] for rr in row['F_exact'][a]]
            require(savedF==F,'Saved exact F differs from independent Bayes reconstruction')
        branch_scores=[]
        fixed=[0.0 for b in proxies]
        for outcome,p,wp in branches:
            scores=[quadratic(wp,S) for S in Ss]
            for j,v in enumerate(scores):
                fixed[j]+=float(p)*v
            best=max(scores[j] for j in legal)
            near=[j for j in legal if best-scores[j]<=tolerance(best,config)]
            branch_scores.append({'outcome':outcome,'probability':p,'posterior':wp,'scores':scores,'best':best,'near_optimal_indices':near})
        adaptive=math.fsum(float(b['probability'])*b['best'] for b in branch_scores)
        best_fixed=max(fixed[j] for j in legal)
        gap=adaptive-best_fixed
        require(gap>=-tolerance(adaptive,config),'Adaptive value below best feasible fixed policy')
        stats={'repair_index':a,'legal_proxy_count':len(legal),'forbidden_proxy_count':len(proxies)-len(legal),
               'branch_count':len(branches),'independent_adaptive_value':adaptive,'best_feasible_fixed_value':best_fixed,
               'adaptive_minus_best_fixed':gap,'strictly_positive_adaptive_branch_value':gap>tolerance(adaptive,config),
               'choices_by_saved_algorithm':{},'branch_argmax_tie_counts':sum(len(b['near_optimal_indices'])>1 for b in branch_scores)}
        for label in labels:
            saved=row[label]
            out['max_value_error']=max(out['max_value_error'],close(saved['values'][a],adaptive,config,label+' adaptive value'))
            choices=saved['choices'][a]
            require(len(choices)==len(branches),'Saved outcome choices lost/added branches')
            require([x['outcome'] for x in choices]==[b['outcome'] for b in branch_scores],'Saved outcome order differs from exact branches')
            chosen=[]
            for c,b in zip(choices,branch_scores):
                j=c['proxy_index'];require(j in legal,'Saved choice violates hard cost budget')
                regret=b['best']-b['scores'][j]
                require(regret<=tolerance(b['best'],config),'Saved branch choice is suboptimal beyond numeric tie tolerance')
                out['max_choice_regret']=max(out['max_choice_regret'],max(0.0,regret));chosen.append(j)
            stats['choices_by_saved_algorithm'][label]={'indices':chosen,'indices_differ_between_outcomes':len(set(chosen))>1,
                'different_indices_certify_positive_adaptive_value':len(set(chosen))>1 and stats['strictly_positive_adaptive_branch_value']}
        if is_principle:
            for j in range(len(proxies)):
                out['max_fixed_pair_error']=max(out['max_fixed_pair_error'],close(row['fixed_values'][a][j],fixed[j],config,'Saved fixed pair value'))
                trace=math.fsum(float(F[h][k])*Ss[j][k][h] for h in range(len(w)) for k in range(len(w)))
                out['max_fixed_pair_error']=max(out['max_fixed_pair_error'],close(row['complementarity'][a][j],trace,config,'Saved complementarity'))
        else:
            for label in ('cached','blocked'):
                for j in range(len(proxies)):
                    out['max_fixed_pair_error']=max(out['max_fixed_pair_error'],close(row[label]['fixed_pair_values'][a][j],fixed[j],config,'Saved fixed pair value'))
        if 'separable_shortcut' in row:
            require(all(b['rhos']==proxies[0]['rhos'] and b.get('weight',1)==proxies[0].get('weight',1) for b in proxies),'Shortcut family is not common-rho/scaled')
            best_scale=max(proxies[j]['scale'] for j in legal)
            jscale=next(j for j in legal if proxies[j]['scale']==best_scale)
            unit=dict(proxies[0]);unit['scale']=1.0
            S0=kernel(unit)
            analytic=best_scale*math.fsum(float(p)*quadratic(wp,S0) for _,p,wp in branches)
            out['max_fixed_pair_error']=max(out['max_fixed_pair_error'],close(row['separable_shortcut']['values'][a],analytic,config,'Recorded analytic separable shortcut'))
            close(analytic,adaptive,config,'Independent separable shortcut vs adaptive')
            require(row['separable_shortcut']['chosen_proxy'][a]==jscale,'Separable shortcut did not select first maximum legal scale')
            stats['independent_separable_shortcut_value']=analytic
        out['first_actions'].append(stats)
    out['strict_adaptive_gain_first_actions']=sum(x['strictly_positive_adaptive_branch_value'] for x in out['first_actions'])
    out['different_saved_branch_indices_first_actions']=sum(any(y['indices_differ_between_outcomes'] for y in x['choices_by_saved_algorithm'].values()) for x in out['first_actions'])
    out['different_indices_but_no_strict_adaptive_gain_first_actions']=sum(any(y['indices_differ_between_outcomes'] for y in x['choices_by_saved_algorithm'].values()) and not x['strictly_positive_adaptive_branch_value'] for x in out['first_actions'])
    out['adaptive_gain_min_max']=[min(x['adaptive_minus_best_fixed'] for x in out['first_actions']),max(x['adaptive_minus_best_fixed'] for x in out['first_actions'])]
    out['timing_reexecuted_or_speed_claimed']=False
    return out

def exact_same_F_mapping(producer,independent):
    names={'two_branch':'posterior_quarters','three_branch':'posterior_endpoints_and_midpoint'}
    out=[]
    for row in producer['exact_results']:
        d=independent['designs'][names[row['id']]]
        require([[decode_fraction(x) for x in r] for r in row['F']]==[[Q(x) for x in r] for r in d['F']],'Same-F saved covariance mismatch')
        require(all(decode_fraction(x)==Q('5/64') for x in row['fixed_gains']),'Fixed pair exact gain mismatch')
        require(decode_fraction(row['adaptive_gain'])==Q(d['best_policy_by_exact_joint_risk']['gain']),'Exact adaptive gain differs from independently enumerated full-joint result')
        require(decode_fraction(row['final_Brier'])==Q(d['best_policy_by_exact_joint_risk']['risk']),'Saved final binary risk mismatch')
        out.append({'id':row['id'],'same_F':d['F'],'fixed_gain':'5/64','fixed_pair_synergy':'1/64','adaptive_gain':str(decode_fraction(row['adaptive_gain'])),'final_binary_risk':str(decode_fraction(row['final_Brier'])),'old_full_joint_enumeration_reused_not_rerun':True})
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--approval',required=True);ap.add_argument('--output-dir',required=True)
    args=ap.parse_args();planp=Path(args.plan);approvalp=Path(args.approval);out=Path(args.output_dir)
    pb=planp.read_bytes();plan=json.loads(pb);approval=json.loads(approvalp.read_text('utf-8'))
    require(approval['plan_sha256']==sha(pb) and approval['source_sha256']==sha(Path(__file__).read_bytes()),'Exact audit plan/source approval required')
    require(approval['scope']=='SAVED_SYNTHETIC_STDLIB_ARITHMETIC_ONLY','Wrong audit scope')
    require(plan['source_sha256']==sha(Path(__file__).read_bytes()),'Audit source changed')
    require(not out.exists(),'Output exists; no automatic repeat audit')
    for path,h in plan['input_sha256'].items():
        require(sha(Path(path).read_bytes())==h,'Frozen input changed: '+path)
    out.mkdir(parents=True)
    start={'actual_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'plan_sha256':sha(pb),'approval_sha256':sha(approvalp.read_bytes()),'source_sha256':sha(Path(__file__).read_bytes()),'SUT_imports_or_runs':0,'timing_runs':0,'material_reads_or_fits_or_scores':0}
    put(out/'EXECUTION_START.json',start)
    try:
        results=json.loads(Path(plan['producer_results']).read_text('utf-8'))
        registered=json.loads(Path(plan['producer_protocol']).read_text('utf-8'))
        oldjoint=json.loads(Path(plan['prior_independent_full_joint_results']).read_text('utf-8'))
        budget=registered['configuration']['hard_budget_integer_ticks'];tol=plan['arithmetic_tolerances']
        require(results['no_real_material_fit_prediction_or_score'] is True and results['no_new_algorithm_or_topconference_evidence_claimed'] is True,'Producer scope flags differ')
        principles=[];sameF=[]
        for row in results['principle_cases']:
            if row['case']=='same_F_different_adaptive_branches':
                sameF=exact_same_F_mapping(row,oldjoint)
            else:
                principles.append(audit_case(row,budget,tol,True))
        benches=[audit_case(row,budget,tol,False) for row in results['bounded_benchmark_cases']]
        require(len(principles)==5 and len(sameF)==2 and len(benches)==6,'Registered case count differs')
        # Principle-specific interpretation checks, using saved values reconstructed above.
        pp={r['family_or_principle']:r for r in principles}
        require(pp['noncollinear_single_proxy_selector']['strict_adaptive_gain_first_actions']==1,'Expected genuine branch adaptation positive control absent')
        require(pp['zero_relation']['first_actions'][0]['independent_adaptive_value']==0,'Zero relation value should be zero')
        for case in ('theta_known','fake_calibration_identity'):
            saved=next(r for r in results['principle_cases'] if r['case']==case)
            require(all(decode_fraction(x)==0 for m in saved['F_exact'] for rr in m for x in rr),'Point-known/uninformative A should have F=0')
            require(all(v==0 for row in saved['complementarity'] for v in row),'F=0 should have no relation complementarity')
        for path,h in plan['input_sha256'].items():
            require(sha(Path(path).read_bytes())==h,'Frozen input changed during audit')
        put(out/'PRINCIPLE_ARITHMETIC.json',{'gaussian_principles':principles,'same_F_independent_joint_result_mapping':sameF})
        put(out/'SIX_BENCHMARK_ARITHMETIC.json',benches)
        allrows=principles+benches
        receipt={'status':'PASS_REGISTERED_SYNTHETIC_SAVED_ARITHMETIC','actual_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'execution':start,
            'max_errors':{k:max(x[k] for x in allrows) for k in ['max_value_error','max_kernel_entry_error','max_fixed_pair_error','max_choice_regret']},
            'principle_count':6,'benchmark_count':6,'benchmark_strict_adaptive_gain_first_actions':sum(x['strict_adaptive_gain_first_actions'] for x in benches),
            'benchmark_different_indices_without_strict_gain_first_actions':sum(x['different_indices_but_no_strict_adaptive_gain_first_actions'] for x in benches),
            'original_source_protocol_results_unchanged':True,'source_supports_no_general_extreme_number_robustness_claim':True,
            'cached_and_blocked_are_strong_classical_baselines_not_new_acceleration':True,'no_timing_rerun':True,
            'arithmetic_replays_are_not_new_measurements_or_material_experiments':True,
            'files_sha256':{p.name:sha(p.read_bytes()) for p in sorted(out.iterdir()) if p.is_file()}}
        put(out/'INDEPENDENT_SAVED_ARITHMETIC_RESULT.json',receipt)
        print(json.dumps({'status':receipt['status'],'max_errors':receipt['max_errors'],'benchmark_strict_adaptive_gain_first_actions':receipt['benchmark_strict_adaptive_gain_first_actions'],'benchmark_different_indices_without_strict_gain_first_actions':receipt['benchmark_different_indices_without_strict_gain_first_actions'],'receipt_sha256':sha((out/'INDEPENDENT_SAVED_ARITHMETIC_RESULT.json').read_bytes())},ensure_ascii=True))
    except Exception:
        put(out/'FAILURE.json',{'status':'FAIL_PRESERVED_NO_RETRY_OR_TOLERANCE_CHANGE','actual_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'execution':start,'traceback':traceback.format_exc()})
        raise

if __name__=='__main__':
    main()

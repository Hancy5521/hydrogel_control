"""Independent all2160 frozen active-onlyY scores, using previously audited CDF/energy math."""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import traceback
import warnings
from scipy import special
from scipy.integrate import quad,IntegrationWarning

POLICIES=('native_student','native_gaussian','latent_ivr','random','gp_variance','warm_openloop')
BUDGETS=(0,4,8)
LEVELS=('0.8','0.9','0.95')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def require(x,m):
    if not x:raise AssertionError(m)


def load(pin,name):
    require(sha(pin['file'])==pin['sha256'],'independent reference dependency changed: '+name)
    spec=importlib.util.spec_from_file_location(name,pin['file'])
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def summary(scores,discrete):
    require(len(scores)==64,'complete64 terminal summary')
    mean=lambda values:math.fsum(values)/64
    pits=sorted(x['PIT_midpoint'] for x in scores)
    return {'formula_groups':64,
            'mean_CRPS_native':mean([x['CRPS_native'] for x in scores]),
            'RMSE_predictive_median_native':math.sqrt(mean([x['squared_error_native'] for x in scores])),
            'coverage':{l:sum(x['intervals'][l]['covered'] for x in scores)/64 for l in LEVELS},
            'mean_interval_width_native':{l:mean([x['intervals'][l]['width'] for x in scores]) for l in LEVELS},
            'PIT_ECDF_deviation_descriptive':None if discrete else max(max((i+1)/64-v,v-i/64) for i,v in enumerate(pits)),
            'mean_PIT_left':mean([x['PIT_left'] for x in scores]),
            'mean_PIT_right':mean([x['PIT_right'] for x in scores])}


def check_summary(saved,own,c):
    c.exact('summary_formula_groups',saved['formula_groups'],64)
    for key in ('mean_CRPS_native','RMSE_predictive_median_native','mean_PIT_left','mean_PIT_right'):
        c.number('summary_'+key,saved[key],own[key])
    if own['PIT_ECDF_deviation_descriptive'] is None:
        c.exact('discrete_no_uniformPIT_claim',saved['PIT_ECDF_deviation_descriptive'],None)
    else:c.number('summary_PIT_ECDF',saved['PIT_ECDF_deviation_descriptive'],own['PIT_ECDF_deviation_descriptive'],2e-12,2e-12)
    for level in LEVELS:
        c.number('summary_exact_coverage',saved['coverage'][level],own['coverage'][level],0,0)
        c.number('summary_interval_width',saved['mean_interval_width_native'][level],own['mean_interval_width_native'][level])


def audit(plan):
    for name,digest in plan['independent_sources'].items():require(sha(Path(__file__).parent/name)==digest,'audit source changed')
    loaded={}
    for name,pin in plan['inputs'].items():
        require(sha(pin['file'])==pin['sha256'],'pinned input changed:'+name);loaded[name]=read(pin['file'])
    ref=load(plan['independent_references']['batch'],'saved_state_reference')
    old=load(plan['independent_references']['allscore_functions'],'_independent_prior_allscore_functions')
    native=load(plan['independent_references']['native_CDF'],'_independent_nativeCDF_reference')
    native.special=special;native.quad=quad
    coordinate=SimpleNamespace(transform=ref.forward_yj,inverse=old.inverse_yj)
    c=old.Checks()
    protocol,freeze,result=loaded['science_plan'],loaded['forecast_freeze'],loaded['score_result']
    require(loaded['state_audit']['status']=='PASS','actual independent saved state audit required')
    for name,digest in protocol['source_sha256'].items():require(sha(Path(plan['producer_code_directory'])/name)==digest,'science source changed')
    directory=Path(plan['forecast_directory'])
    for name,digest in freeze['files'].items():require(sha(directory/name)==digest,'saved forecast payload changed')
    start=loaded['score_start']
    c.exact('all_laws_freeze_before_terminalY',start['all2112terminal_and48prequential_payloads_verified_before_terminalY'],True)
    require(start['forecast_freeze_sha256']==result['forecast_freeze_sha256']==plan['inputs']['forecast_freeze']['sha256']
            and result['plan_sha256']==plan['inputs']['science_plan']['sha256'],'actual score protocol/freeze binding')
    require(datetime.fromisoformat(freeze['actual_utc'])<=datetime.fromisoformat(start['actual_utc'])<=datetime.fromisoformat(result['actual_end_utc']),'complete forecast before terminal reveal')
    roles=loaded['roles']['representatives'];truth=loaded['terminal_truth']['truth']
    require(len(truth)==len(roles['terminal'])==64,'fixed64terminaltargets')
    targets={}
    for item,meta in zip(truth,roles['terminal']):
        c.exact('truth_identity',item['identity'],meta);require(math.isfinite(item['native_target']),'finite terminal target')
        targets[meta['component_id']]=item['native_target']
    require(len(targets)==64,'unique64terminalidentities')
    forecasts=read(directory/'TERMINAL_FORECASTS.json')['forecasts']
    reference_forecasts=read(directory/'REFERENCE_FORECASTS.json')['forecasts']
    laws={(r['component_id'],r['policy'],r['budget']):r for r in forecasts}
    refs={(r['component_id'],r['policy'],r['budget'],r['method']):r for r in reference_forecasts}
    saved={(r['component_id'],r['policy'],r['budget']):r for r in result['primary_NIG_forecast_scores']}
    saved_refs={(r['component_id'],r['policy'],r['budget'],r['method']):r for r in result['strong_reference_forecast_scores']}
    expected={(g,p,b) for g in targets for p in POLICIES for b in BUDGETS}
    expected_ref={(g,p,b,m) for g in targets for p in ('native_student','random') for b in BUDGETS for m in ('GP','RF')}
    expected_ref.update((g,'gp_variance',b,'GP') for g in targets for b in BUDGETS)
    require(len(forecasts)==len(saved)==len(laws)==1152 and set(saved)==set(laws)==expected,'all1152primaryscores')
    require(len(reference_forecasts)==len(saved_refs)==len(refs)==960 and set(saved_refs)==set(refs)==expected_ref,'all960reference scores')
    cache={};maximum_quad=0.;routes={'native_CDF':0,'empirical_energy':0}
    def evaluate(y,law):
        nonlocal maximum_quad
        key=(y,json.dumps(law,sort_keys=True,separators=(',',':'),allow_nan=False))
        if key not in cache:
            cache[key]=old.evaluate(y,law,native,coordinate)
            routes['empirical_energy' if law['family']=='weighted_empirical' else 'native_CDF']+=1
        value=cache[key];maximum_quad=max(maximum_quad,value.get('quad_error_estimate_native',0.))
        return value
    values={};reference_values={}
    for law_map,score_map,output in ((laws,saved,values),(refs,saved_refs,reference_values)):
        for key,forecast in law_map.items():
            record=score_map[key];group=key[0];y=targets[group]
            c.exact('score_jid',record['jid'],forecast['jid']);c.number('native_target',record['native_target'],y,0,0)
            value=evaluate(y,forecast['law']);old.score_compare(record['score'],value,c);output[key]=value
    summaries={};reference_summaries={}
    for policy in POLICIES:
        summaries[policy]={}
        for budget in BUDGETS:
            own=summary([values[(g,policy,budget)] for g in targets],False)
            check_summary(result['summary_NIG_prediction_by_policy'][policy][str(budget)],own,c)
            summaries[policy][str(budget)]=own
    for policy in ('native_student','random','gp_variance'):
        reference_summaries[policy]={}
        for method in (('GP','RF') if policy!='gp_variance' else ('GP',)):
            reference_summaries[policy][method]={}
            for budget in BUDGETS:
                own=summary([reference_values[(g,policy,budget,method)] for g in targets],method=='RF')
                check_summary(result['summary_strong_native_prediction_references'][policy][method][str(budget)],own,c)
                reference_summaries[policy][method][str(budget)]=own
    paired={r['component_id']:r for r in result['per_terminal_paired_differences']}
    require(len(paired)==len(result['per_terminal_paired_differences'])==64 and set(paired)==set(targets),'64pairedterminalrecords')
    for group in targets:
        row=paired[group]
        for policy in POLICIES:
            if policy=='native_student':continue
            for budget in BUDGETS:
                delta=values[(group,'native_student',budget)]['CRPS_native']-values[(group,policy,budget)]['CRPS_native']
                c.number('paired_policy_CRPS_difference',row['native_student_minus_other_NIG_policies'][policy][str(budget)],delta,4e-6,4e-8)
        final=values[(group,'native_student',8)]['CRPS_native']
        c.number('paired_active_GP',row['native_student_final_minus_GP_active_GP_prediction'],final-reference_values[(group,'gp_variance',8,'GP')]['CRPS_native'],4e-6,4e-8)
        c.number('paired_own_final_minus_warm',row['native_student_own_final_minus_warm'],final-values[(group,'native_student',0)]['CRPS_native'],4e-6,4e-8)
    candidate=summaries['native_student']['8']['mean_CRPS_native']
    controls={'NIGpolicy_'+p:summaries[p]['8']['mean_CRPS_native'] for p in POLICIES if p!='native_student'}
    controls.update({p+'_'+m:reference_summaries[p][m]['8']['mean_CRPS_native'] for p in reference_summaries for m in reference_summaries[p]})
    primary=result['primary_comparison']
    c.number('candidate_final_CRPS',primary['candidate_native_student_mean_CRPS_budget8'],candidate)
    c.exact('complete_final_control_names',set(primary['other_final_losses']),set(controls))
    for key,value in controls.items():c.number('final_control_CRPS',primary['other_final_losses'][key],value)
    best=min(controls.values())
    c.number('best_reference_ratio',primary['candidate_to_best_reference_ratio'],candidate/best)
    c.exact('within2percent_descriptive',primary['within2percent_descriptive'],candidate<=1.02*best)
    indices=read(directory/'PREQUENTIAL_INDEX.json')['records']
    secondary=result['secondary_prequential_scores']
    require(len(indices)==len(secondary)==48,'all48secondaryscores')
    for record,item in zip(secondary,indices):
        before=read(directory/item['forecast_file']);after=read(directory/item['outcome_file']);y=after['true_Y_reveal']['native_target']
        for key in ('policy','step','pool_index'):c.exact('secondary_'+key,record[key],item[key])
        c.exact('secondary_group',record['component_id'],before['identity']['component_id'])
        c.exact('secondary_freeze_sha',record['forecast_sha256'],item['forecast_sha256'])
        c.exact('secondary_update_sha',record['outcome_sha256'],item['outcome_sha256'])
        c.exact('secondary_not_confirmation',record['process_only_not_terminal_confirmation'],True)
        c.number('secondary_true_target',record['native_target'],y,0,0)
        require(datetime.fromisoformat(record['prediction_utc'])<=datetime.fromisoformat(record['reveal_utc']),'prequential freeze/reveal')
        old.score_compare(record['score'],evaluate(y,before['law']),c)
    checkpoints=read(directory/'MODEL_STATES.json')['checkpoint_states']
    sequences={p:next(r['selected_pool_indices'] for r in checkpoints if r['policy']==p and r['budget']==8) for p in POLICIES}
    c.exact('result_actual_sequences',result['actual_selected_pool_sequences'],sequences)
    expected_paths=[{'left':a,'right':b,'exact_sequence_equal':sequences[a]==sequences[b],
                     'same_final_set':set(sequences[a])==set(sequences[b])} for a,b in itertools.combinations(POLICIES,2)]
    c.exact('result_path_aliases',result['policy_path_comparisons'],expected_paths)
    c.exact('score_status',result['status'],'COMPLETE_ACTIVE_ONLYY_FROZEN_DEVELOPMENT_COMPARISON')
    c.exact('score_actual_fits',result['actual_sklearn_fits'],18)
    c.exact('score_common_cell_cost',result['common_setup_native_cells'],368)
    c.exact('score_fixed_policy_cost',result['fixed_perpolicy_onlyY_cost'],8)
    return {'status':'PASS','all_score_coordinates':2160,'primary_NIG_score_coordinates':1152,
            'reference_score_coordinates':960,'secondary_prequential_coordinates':48,'all_coverage_events':6480,
            'method_budget_summaries':33,'paired_difference_coordinates':1088,
            'unique_law_outcome_reference_scores':len(cache),'reference_routes':routes,
            'max_native_CDF_quad_error_estimate':maximum_quad,'independent_summary_NIG':summaries,
            'independent_summary_strong_references':reference_summaries,'independent_candidate_final':candidate,
            'independent_best_reference_ratio':candidate/best,'within2percent_descriptive':candidate<=1.02*best,
            'actual_sequences':sequences,'alias_paths':expected_paths,'comparison_counts':c.counts,
            'max_absolute_errors':c.errors,'original_SUT_imports_fit_or_score_replays':0,
            'raw_data_or_new_material_reads':0,'scope':'saved score arithmetic for fixed152 developmentgroups; no independent source confirmation, significance or actual wet/DFT efficiency claim'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);parser.add_argument('--approval',required=True);parser.add_argument('--output-dir',required=True)
    args=parser.parse_args();plan=read(args.plan);approval=read(args.approval)
    require(approval.get('approved') is True and approval['scope']=='INDEPENDENT_ACTIVE_ONLYY_ALL_SCORES_ONCE'
            and approval['plan_sha256']==sha(args.plan) and approval['script_sha256']==sha(__file__),'exact root score audit approval')
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=False);start=datetime.now(timezone.utc).isoformat()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error',IntegrationWarning);result=audit(plan)
    except Exception as problem:result={'status':'FAIL','type':type(problem).__name__,'message':str(problem),'traceback':traceback.format_exc()}
    result.update({'actual_start_utc':start,'actual_end_utc':datetime.now(timezone.utc).isoformat(),
                   'source_sha256':sha(__file__),'plan_sha256':sha(args.plan),'approval_sha256':sha(args.approval)})
    file=out/'INDEPENDENT_ACTIVE_ONLYY_ALL_SCORE_AUDIT.json'
    with file.open('x',encoding='utf-8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({'status':result['status'],'file':str(file),'sha256':sha(file)}))
    return 0 if result['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
